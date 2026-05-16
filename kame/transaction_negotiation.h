/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
// =====================================================================
// transaction_negotiation.h
//
// Negotiation-layer types and primitives that surround the
// negotiate_internal() / negotiate_after_retry_pause() implementations.
// Specifically:
//   * retry_pause()                         — CPU-relax for CAS retry loops
//   * effective_runners() forward decl      — hardware-aware contender helper
//   * ScopedNegotiateLinkage<XN>            — RAII helper for retry loops,
//                                             with the per-Linkage spin
//                                             path / SiteState / digest
//                                             machinery this branch adds
//
// Out-of-class definitions of NegotiationCounter members and the full
// negotiate_internal() body live in transaction_neg_impl.h.  Tuning
// macros live in transaction_definitions.h.  Linkage::m_recent_ops_state,
// NegSite, LivelockProbe, RunnerDigest are declared in transaction.h
// (they're referenced from inline bodies in Snapshot/Linkage).
//
// This header is included from transaction_impl.h after transaction.h,
// so all of Node<XN>'s nested types (Linkage, NegotiationCounter,
// Snapshot<XN>, Transaction<XN>, TidBitset, NegSite, LivelockProbe) are
// already visible.
// =====================================================================
#ifndef TRANSACTION_NEGOTIATION_H
#define TRANSACTION_NEGOTIATION_H

#include "transaction.h"
#include "transaction_definitions.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

namespace Transactional {

// Retry-proportional randomized CPU relax for tight CAS-retry loops.
// On retry N, issues `spins ∈ [1, min(N, 64)]` pause instructions before
// the next CAS attempt. Randomized count desynchronizes threads that
// reach the same retry depth simultaneously — a fixed pause count would
// leave them re-synchronized, re-creating the lock-step that causes
// CAS-retry livelock.
//
// Pseudo-random count is produced by bit-mixing `retry` with the
// address of the local stack variable (per-thread entropy) via golden-
// ratio hash, reduced to [1, max_spins] by Lemire-style multiply-shift.
// No thread-local state, no RNG call, ~5 ALU ops total.
//
// Purpose: on strong-memory x86 (TSO), coherence ping-pong causes all
// threads to observe each other's identity churn immediately; without
// any relax, no thread gets an uncontested window to win its CAS.
// `pause` (x86) / `yield`-inst (ARM) lowers coherence traffic and gives
// other threads small winner windows. Zero cost on the fast path
// (only invoked when retry > 0). Independent of the jitter-based
// negotiate path (which only fires when a collision marker is set).
static inline void retry_pause(int retry) noexcept {
#if defined(KAME_STM_DISABLE_BACKOFF) && KAME_STM_DISABLE_BACKOFF
    (void)retry;
    return;
#endif
    uint32_t h = (uint32_t)retry * 0x9E3779B1u;
    h ^= (uint32_t)(uintptr_t)&retry;
    int spins = 1 + (int)(((uint64_t)(h >> 16) * (uint32_t)retry) >> 16);
    for(int k = 0; k < spins; ++k) pause4spin();
}

// Forward declaration.  effective_runners() is the hardware_concurrency-
// aware contender-count helper used by negotiate_internal's MIN_RUNNERS
// / MAX_RUNNERS gating; declared here so it is visible to all of the
// detail:: machinery (notify_n_contenders, etc.). Full definition is
// in transaction_neg_impl.h, gated on the MIN_RUNNERS / MAX_RUNNERS
// macros.
#if (KAME_STM_MIN_RUNNERS != 0) || (KAME_STM_MAX_RUNNERS != 0)
inline int effective_runners(int c_obs) noexcept;
#else
inline int effective_runners(int) noexcept {
    int hw = (int)std::thread::hardware_concurrency();
    return hw > 0 ? hw : 1;
}
#endif

} // namespace Transactional

namespace Transactional {

//! RAII helper for CAS-retry / spin loops that branch on Linkage state.
//!
//! Construct at the TOP of each loop iteration, passing the Linkage,
//! the iteration counter, and the desired tagging mode.
//!
//! Ctor always calls
//! `Linkage::negotiate_after_retry_pause(retry, snap, mult_wait)` so
//! the iter starts with a livelock-free yield-to-privileged-Tx gate
//! (self-gates at retry==0 unless fair-mode active).
//!
//! CAS-management policy
//! ---------------------
//! All CASes on the scope's linkage MUST go through `compareAndSet*` /
//! `compareAndSetWithHint*` member functions, not via direct
//! `m_link->compareAndSet()` calls.  Routing the CAS through the scope
//! lets the scope completely manage the success/failure state:
//!
//!   - **success** → automatically marks `m_committed = true`; the
//!     dtor neither tags.
//!   - **failure** → records `m_contention_observed` (forces dtor
//!     tag even at retry==0).
//!
//! Pre-CAS conflict — when the caller detects a conflict caused by
//! ANOTHER thread (e.g. `wrapper->packet() != tr.m_oldpacket`,
//! UnbundledStatus::SUBVALUE_HAS_CHANGED) — should call
//! `confirm_contention()` so the dtor tags this iteration despite
//! the retry==0 fast-path optimization.
//!
//! Tagging modes (when to tag at ctor vs dtor):
//!   - **OnEntry** (default): tag_as_contender immediately at ctor
//!     IF `retry != 0`.  Eager — other threads' negotiate sees our
//!     contender mark for the duration of this iter's body.  Matches
//!     commit / bundle / Node::snapshot retry-loop sites.
//!   - **OnExit**: never tag at ctor.  Tag at dtor when not committed
//!     and `m_should_tag` (retry > 0).  Lazy — leaves `commit()` and
//!     retry==0 fast-path exits untagged.  Matches insert(online) /
//!     eraseSerials sites.
//!
//! Retry gating (m_should_tag = retry != 0):
//!   - `retry >  0`: tag at ctor (OnEntry) or dtor (OnExit).
//!   - `retry == 0`: do NOT tag at ctor or dtor by default
//!     (first-attempt fast path).  EXCEPT when contention is
//!     OBSERVED — `compareAndSet` failure or
//!     `confirm_contention()` — then dtor tags regardless (the
//!     iteration confirmed it is a contender; retry==0 fast-path is
//!     no longer applicable).
//!   - `retry == -1`: tag at ctor unconditionally in OnEntry mode
//!     (always-tag); the value `0` is passed down to
//!     `negotiate_after_retry_pause`, since the function self-gates
//!     on fair-mode anyway.
//!
//! Privilege assertion (KAME_STM_ASSERT_PRIVILEGE):
//!   Both prior variants — the dtor-time "privileged Tx CAS failed"
//!   check and the CAS-time "non-privileged Tx attempted CAS while
//!   another Tx holds privilege" check — were retired as false
//!   positives.  `negotiate_after_retry_pause` is the only
//!   livelock-free yield point: as long as it correctly returns
//!   only when (priv==0 || priv==me), CAS-time and dtor-time
//!   conditions are racy snapshots of the global slot and don't
//!   reflect any negotiate-yield bypass.  Specifically, between
//!   our negotiate's return and our CAS, another Tx may legitimately
//!   livelock-probe-claim the slot — that thread's CAS racing ours
//!   is normal contention, not a fairness violation (our retry will
//!   observe the new privilege and yield).  KAME_STM_ASSERT_PRIVILEGE
//!   is kept as a hook for future invariants that aren't expressible
//!   as a single-snapshot equality.
//!
//! Definition lives here (transaction_impl.h) — only the retry-loop
//! sites in this file use it. Forward-declared in transaction.h, with
//! friend declarations in Node<XN> / Snapshot<XN> for the assert path.
template <class XN>
class ScopedNegotiateLinkage {
    using LinkagePtr = std::shared_ptr<typename Node<XN>::Linkage>;
    using PacketWrapper = typename Node<XN>::PacketWrapper;
    LinkagePtr      m_link;
    Snapshot<XN>   *m_snap;
    //! Tag-ref'd view of m_link's current PacketWrapper.  Acquired in
    //! ctor (DEFER_THRESHOLD — stays TagHeld; no fetch_add unless
    //! moved out as local_shared_ptr).  Used as oldr for the scope's
    //! built-in compareAndSet / compareAndSetWithHint methods.
    //! After a successful CAS the view is consumed (empty); after a
    //! failed CAS the view may also be empty depending on the
    //! weak+scoped contract — caller can re-acquire via compareAndSetRetain().
    scoped_atomic_view<PacketWrapper> m_view;
    float           m_mult_wait;             // retained from ctor for dtor's negotiate
    bool            m_eager;
    bool            m_should_tag;            // retry != 0 — fast-path optimization
    bool            m_committed = false;
    bool            m_contention_observed = false;  // forces dtor tag despite retry==0
    //! True iff the privileged thread (s_privileged_tidstamp holder)
    //! constructed this scope.  Cached at construction; subsequent CAS
    //! operations dispatch to STRONG (spinning) variants when set —
    //! safe because privilege is exclusive and fair_mode blocks all
    //! other threads' CAS, so single-thread strong cannot livelock.
    //! Stale-true case (privilege preempted mid-scope) is bounded —
    //! scope ends quickly, and the strong CAS still terminates: it
    //! returns false on real pointer mismatch.
    bool            m_strong_mode = false;
    //! Cached pointer into NegSite::state_map() for this scope's
    //! call-site.  Populated by NegSite::Scope at ctor entry — the hot
    //! path in _on_cas_success / _on_cas_fail (and the lease check in
    //! negotiate_internal) dereferences this pointer instead of doing
    //! a per-call unordered_map lookup.  Carries both the adaptive
    //! state machine (take_gate/streak/lease) and the cumulative
    //! counters (entries/commits/per-peer breakdowns) — counters are
    //! always-on now, gated only by m_link (skipped when moved-from).
    NegSite::SiteState *m_site_state = nullptr;
    //! Whether THIS scope's ctor-time negotiate fired a gate-return.
    //! Per-scope (not thread_local) so nested scopes don't overwrite
    //! one another's correlation flag.  Written by negotiate_internal
    //! via the thread_local sink NegSite::last_was_gate_return(); consumed
    //! by _on_cas_fail / _on_cas_success of THIS scope only.
    bool            m_was_gate_return = false;
#if KAME_ENABLE_RUNNER_DIGEST
    //! __LINE__ of the ctor call site.  Used to fill the digest's
    //! `site_line_lo` field at each publish point so peers can tell
    //! which scope this thread is currently in.
    int             m_caller_line = 0;

    //! Shift this scope's gate decision into the TLS digest's
    //! `gate_history` ring (LSB = newest).  Called from ctor right
    //! after `_capture_gate_return()` — before `_on_cas_*` clears
    //! `m_was_gate_return`.  Mutates the TLS cache only; the value is
    //! published at scope end via `_publish_digest()`.
    void _shift_gate_history() noexcept {
        auto &d = detail::tls_runner_digest;
        d.f.gate_history = ((unsigned)d.f.gate_history << 1
                           | (m_was_gate_return ? 1u : 0u)) & 0xFFu;
    }

    //! Pack the current scope/site state into the TLS digest cache
    //! and publish it atomically (relaxed) to this thread's Runner
    //! slot.  Called once per scope, at dtor entry — peers see the
    //! scope-end snapshot (outcome reflected in `m_site_state`).
    //!
    //! Bitfield writes are issued against a register-local copy
    //! (`local.raw` loaded from TLS once, written back once after all
    //! fields are set).  This avoids the per-field load-modify-store
    //! against TLS memory that a naive `tls_runner_digest.f.x = …`
    //! sequence would generate for each bitfield assignment.  No
    //! `now_us()` call: tx_start_us is taken directly from the
    //! Snapshot's already-packed stamp; peer computes age locally.
    void _publish_digest() noexcept {
        detail::RunnerDigest local;
        local.raw = detail::tls_runner_digest.raw;
        using NC = typename Node<XN>::NegotiationCounter;
        local.f.tx_start_us = (uint64_t)NC::stamp_us(m_snap->m_started_time)
                              & ((1ULL << 22) - 1);
        local.f.op_kind = (uint64_t)detail::s_current_op_kind & 0x3u;
        // m_site_state non-null invariant (NegSite::Scope set it
        // unconditionally at ctor; only the moved-from path would
        // clear it, and that path is gated by m_link before reaching
        // here from dtor).
        local.f.consec_succs = m_site_state->consec_succs > 255
            ? (uint64_t)255 : (uint64_t)m_site_state->consec_succs;
        local.f.consec_fails = m_site_state->consec_fails > 255
            ? (uint64_t)255 : (uint64_t)m_site_state->consec_fails;
        // take_gate ∈ {-1, 0, 1} → +1 = {0, 1, 2}; 3 reserved.
        int8_t tg = m_site_state->take_gate;
        local.f.take_gate_p1 = (uint64_t)(tg < 0 ? 0 : (tg + 1)) & 0x3u;
        local.f.site_line_lo = (uint64_t)(m_caller_line & 0xFFF);
        detail::tls_runner_digest.raw = local.raw;
        // Skip the atomic store when the *peer-actionable* fields
        // haven't changed since the previous publish from this thread.
        // gate_history and site_line_lo are excluded from the diff —
        // they tick every scope and the CV-sleep judge primarily reads
        // consec_*/take_gate/op_kind/tx_start_us, which only change on
        // CAS events / op_kind transitions / Tx boundaries.  Saves an
        // atomic store on uncontended retry loops without losing
        // peer-relevant signal.
        constexpr uint64_t SITE_LINE_MASK = ((1ULL << 12) - 1) << 50;
        constexpr uint64_t GATE_HIST_MASK = ((1ULL << 8) - 1) << 24;
        constexpr uint64_t ACTIONABLE_MASK = ~(SITE_LINE_MASK | GATE_HIST_MASK);
        static thread_local uint64_t s_last_actionable = 0;
        const uint64_t actionable = local.raw & ACTIONABLE_MASK;
        if(actionable == s_last_actionable)
            return;
        s_last_actionable = actionable;
        if(auto *r = detail::tls_runner_counter_ptr)
            r->digest.store(local.raw, std::memory_order_relaxed);
    }
#endif // KAME_ENABLE_RUNNER_DIGEST
public:
    enum class TagMode { OnEntry, OnExit };

    //! Standard ctor: negotiate + tag-bit WEAK-acquire view of m_link.
    //!
    //! The view acquire is single-shot (`weakly=true`) — on contention
    //! the underlying CAS may lose without retry.  A lost acquire is
    //! treated as a CAS failure: `m_contention_observed` is set (so
    //! the dtor tags this iteration as a contender), and the view is
    //! empty so `bool(*this)` returns false.
    //!
    //! Callers that access the view (operator->, operator*, internal
    //! scope-driven CAS via compareAndSet(desired)) MUST guard with
    //! `if(!scope) continue;` after construction in their retry loop.
    //! Sites that only use external-expected CAS
    //! (compareAndSet(expected, desired)) may proceed unguarded since
    //! the empty view does not affect that path.
    //!
    //! `caller_line` is auto-populated by __builtin_LINE() as a default
    //! argument — it resolves at the call site (not the ctor's
    //! definition line).  Used by KAME_ADAPT_INSTRUMENT profiling to
    //! key NegSite::SiteState per source-line.  The per-site map is
    //! always populated; INSTRUMENT only controls whether merged stats
    //! are auto-dumped at thread/process exit.
    ScopedNegotiateLinkage(LinkagePtr link, Snapshot<XN> &snap, int retry,
                           TagMode mode = TagMode::OnEntry,
                           float mult_wait = 2.0f,
                           int caller_line = __builtin_LINE()) noexcept
        : m_link(std::move(link)), m_snap(&snap),
          m_mult_wait(mult_wait),
          m_eager(mode == TagMode::OnEntry),
          m_should_tag(retry != 0)
#if KAME_ENABLE_RUNNER_DIGEST
        , m_caller_line(caller_line)
#endif
    {
        // NegSite::Scope primes the SiteState pointer so negotiate /
        // CAS hooks below can address per-site state by pointer alone.
        NegSite::Scope _site_scope(caller_line);
        // NegSite::Scope::Scope(line) sets current_state to
        // &state_map()[line] — operator[] always returns a valid
        // reference (inserts if missing).  m_site_state is therefore
        // guaranteed non-null here; later guards use m_link as the
        // freshness sentinel (cleared on move-out).
        m_site_state = NegSite::current_state();
        ++m_site_state->entries;
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
        // Touch the thread_local sentinel so its dtor (which merges
        // this thread's stats into the global aggregator) is wired
        // up for this thread.  Cost: one TLS access on first ctor.
        (void)NegSite::auto_merge_stats();
#endif
        if(retry < 0)
            _negotiate();   // always negotiate, no retry_pause
        else
            _negotiate_after_retry_pause(retry);
        _capture_gate_return();
#if KAME_ENABLE_RUNNER_DIGEST
        _shift_gate_history();   // captures decision before _on_cas_* clears it
#endif
        // Privilege-state-aware threshold for the weak tag-bit acquire:
        //
        //   - **We hold privilege** (TID matches s_privileged_tidstamp):
        //     DEFER_THRESHOLD — sole contender by design, no need to
        //     promote (and pay an extra fetch_add + release_tag_ref_).
        //   - **Otherwise** (no privilege OR someone else holds it):
        //     ADAPTIVE_THRESHOLD = LOCAL_REF_CAPACITY-1 — the thread
        //     that lands at the last free tag slot promotes to Owned,
        //     freeing the slot for the next contender.  This is a
        //     back-pressure escape valve specifically for high-thread-
        //     count systems (>LOCAL_REF_CAPACITY contenders) where
        //     plain DEFER_THRESHOLD would weakly-fail most acquires.
        //
        // Negotiate-bypass note: a non-privileged thread "shouldn't"
        // reach this point under livelock-free design (negotiate parks
        // it in negotiate_sleep), but TOCTOU between negotiate's fast
        // path and our s_privileged_tidstamp load makes a hard
        // refusal here counter-productive (causes spurious view
        // failures and livelock — observed empirically).  We let the
        // acquire proceed; if it loses CAS, the normal weak-fail path
        // handles it.
        using NC = typename Node<XN>::NegotiationCounter;
        bool we_hold_priv = NC::i_am_privileged_now(m_snap->m_started_time);
        // STRONG-mode acquire+CAS for the privileged thread: privilege
        // is exclusive and fair_mode blocks all other threads' CAS on
        // this linkage, so a strong spin has no peer to contend with.
        // Single-thread strong cannot livelock by definition.  All
        // other threads use weak acquire (the existing fast path).
        //
        // Threshold split: privileged uses DEFER (= last slot), all
        // others use ADAPTIVE (= second-to-last slot — drain one slot
        // earlier).  Non-privileged threads pre-emptively drain so the
        // privileged thread is more likely to find tag space without
        // promoting (cheap TagHeld preserved).  Both paths DO promote
        // eventually; "never-promote" mode was removed because it lets
        // peer-thread TagHelds accumulate and block the privileged
        // zero-reset CAS indefinitely (b413a98b livelock).
        m_strong_mode = we_hold_priv;
        m_view = scoped_atomic_view<PacketWrapper>(
            *m_link,
            we_hold_priv
                ? scoped_atomic_view<PacketWrapper>::DEFER_THRESHOLD
                : scoped_atomic_view<PacketWrapper>::ADAPTIVE_THRESHOLD,
            /*weakly=*/!we_hold_priv);
        if(!m_view.acquire_succeeded()) {
            // Weak acquire CAS lost (or local refcnt at capacity) —
            // same treatment as a CAS failure: forces dtor tag despite
            // retry==0 fast-path optimization.  STRONG mode never
            // returns acquire_succeeded()==false (it spins until success).
            _on_cas_fail();
        }
        if(m_eager && m_should_tag)
            m_snap->tag_as_contender(m_link);
    }

    //! Move-in ctor: take ownership of an existing
    //! local_shared_ptr<PacketWrapper> (e.g. one already loaded in the
    //! caller's frame).  Zero atomic ops for the view setup — the
    //! local_shared_ptr's +1 ref is reused as the view's Owned ref.
    //!
    //! By default does **not** negotiate (`with_negotiate=false`):
    //! the move-in pattern signals that the caller already has an
    //! outer ScopedNegotiateLinkage on the same linkage (e.g. bundle's
    //! Phase 1 child_scope wraps subnode->m_link, then bundle_subpacket
    //! constructs an inner subscope on the same linkage).  The outer
    //! scope's negotiate already covers this iteration; double-
    //! negotiating the same linkage is redundant.  This matches the
    //! pre-view-ification baseline where bundle_subpacket had no scope
    //! at all and only the outer Phase 1 child_scope negotiated.
    //! Pass `with_negotiate=true` for sites that genuinely need their
    //! own negotiate (no outer scope on this linkage).
    ScopedNegotiateLinkage(LinkagePtr link, Snapshot<XN> &snap, int retry,
                           local_shared_ptr<PacketWrapper> &&from,
                           TagMode mode = TagMode::OnEntry,
                           float mult_wait = 2.0f,
                           bool with_negotiate = false,
                           int caller_line = __builtin_LINE()) noexcept
        : m_link(std::move(link)), m_snap(&snap),
          m_mult_wait(mult_wait),
          m_eager(mode == TagMode::OnEntry),
          m_should_tag(retry != 0)
#if KAME_ENABLE_RUNNER_DIGEST
        , m_caller_line(caller_line)
#endif
    {
        // NegSite::Scope primes the SiteState pointer so negotiate /
        // CAS hooks below can address per-site state by pointer alone.
        NegSite::Scope _site_scope(caller_line);
        // NegSite::Scope::Scope(line) sets current_state to
        // &state_map()[line] — operator[] always returns a valid
        // reference (inserts if missing).  m_site_state is therefore
        // guaranteed non-null here; later guards use m_link as the
        // freshness sentinel (cleared on move-out).
        m_site_state = NegSite::current_state();
        ++m_site_state->entries;
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
        // Touch the thread_local sentinel so its dtor (which merges
        // this thread's stats into the global aggregator) is wired
        // up for this thread.  Cost: one TLS access on first ctor.
        (void)NegSite::auto_merge_stats();
#endif
        if(with_negotiate) {
            if(retry < 0)
                _negotiate();
            else
                _negotiate_after_retry_pause(retry);
        }
        _capture_gate_return();
#if KAME_ENABLE_RUNNER_DIGEST
        _shift_gate_history();   // captures decision before _on_cas_* clears it
#endif
        m_view = scoped_atomic_view<PacketWrapper>(*m_link, std::move(from));
        m_strong_mode = Node<XN>::NegotiationCounter::i_am_privileged_now(
                            m_snap->m_started_time);
        if(m_eager && m_should_tag)
            m_snap->tag_as_contender(m_link);
    }

    //! Move-in ctor: take ownership of an existing
    //! scoped_atomic_view<PacketWrapper> (e.g. one stored in a CASInfo).
    //! Zero atomic ops — the view is directly moved.
    //!
    //! By default does **not** negotiate (`with_negotiate=false`):
    //! use `with_negotiate=true` for sites that need their own negotiate.
    ScopedNegotiateLinkage(LinkagePtr link, Snapshot<XN> &snap, int retry,
                           scoped_atomic_view<PacketWrapper> &&from,
                           TagMode mode = TagMode::OnEntry,
                           float mult_wait = 2.0f,
                           bool with_negotiate = false,
                           int caller_line = __builtin_LINE()) noexcept
        : m_link(std::move(link)), m_snap(&snap),
          m_mult_wait(mult_wait),
          m_eager(mode == TagMode::OnEntry),
          m_should_tag(retry != 0)
#if KAME_ENABLE_RUNNER_DIGEST
        , m_caller_line(caller_line)
#endif
    {
        // NegSite::Scope primes the SiteState pointer so negotiate /
        // CAS hooks below can address per-site state by pointer alone.
        NegSite::Scope _site_scope(caller_line);
        // NegSite::Scope::Scope(line) sets current_state to
        // &state_map()[line] — operator[] always returns a valid
        // reference (inserts if missing).  m_site_state is therefore
        // guaranteed non-null here; later guards use m_link as the
        // freshness sentinel (cleared on move-out).
        m_site_state = NegSite::current_state();
        ++m_site_state->entries;
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
        // Touch the thread_local sentinel so its dtor (which merges
        // this thread's stats into the global aggregator) is wired
        // up for this thread.  Cost: one TLS access on first ctor.
        (void)NegSite::auto_merge_stats();
#endif
        if(with_negotiate) {
            if(retry < 0)
                _negotiate();
            else
                _negotiate_after_retry_pause(retry);
        }
        _capture_gate_return();
#if KAME_ENABLE_RUNNER_DIGEST
        _shift_gate_history();   // captures decision before _on_cas_* clears it
#endif
        m_view = std::move(from);
        m_strong_mode = Node<XN>::NegotiationCounter::i_am_privileged_now(
                            m_snap->m_started_time);
        if(m_eager && m_should_tag)
            m_snap->tag_as_contender(m_link);
    }

    ScopedNegotiateLinkage(const ScopedNegotiateLinkage &) = delete;
    ScopedNegotiateLinkage &operator=(const ScopedNegotiateLinkage &) = delete;
    ScopedNegotiateLinkage(ScopedNegotiateLinkage &&o) noexcept
        : m_link(std::move(o.m_link)), m_snap(o.m_snap),
          m_view(std::move(o.m_view)),
          m_mult_wait(o.m_mult_wait), m_eager(o.m_eager),
          m_should_tag(o.m_should_tag), m_committed(o.m_committed),
          m_contention_observed(o.m_contention_observed),
          m_strong_mode(o.m_strong_mode),
          m_site_state(o.m_site_state),
          m_was_gate_return(o.m_was_gate_return)
#if KAME_ENABLE_RUNNER_DIGEST
        , m_caller_line(o.m_caller_line)
#endif
    {
        o.m_committed = true;  // prevent dtor effects on moved-from
        o.m_site_state = nullptr;
        o.m_was_gate_return = false;  // moved out — don't double-attribute
    }
    ScopedNegotiateLinkage &operator=(ScopedNegotiateLinkage &&o) noexcept {
        if(this != &o) {
            // Run this->dtor logic first (release current view, tag if needed).
            // Simplified: just mark committed to skip dtor effects.
            m_committed = true;  // suppress this->dtor side effects
            m_view = std::move(o.m_view);  // release old view, take new
            m_link = std::move(o.m_link);
            m_snap = o.m_snap;
            m_mult_wait = o.m_mult_wait;
            m_eager = o.m_eager;
            m_should_tag = o.m_should_tag;
            m_committed = o.m_committed;
            m_contention_observed = o.m_contention_observed;
            m_strong_mode = o.m_strong_mode;
            m_site_state = o.m_site_state;
            o.m_site_state = nullptr;
            m_was_gate_return = o.m_was_gate_return;
            o.m_was_gate_return = false;
#if KAME_ENABLE_RUNNER_DIGEST
            m_caller_line = o.m_caller_line;
#endif
            o.m_committed = true;
        }
        return *this;
    }

    // ---------- View access (forwarding to internal scoped view) ----------

    //! Pointer-style access to the wrapped PacketWrapper.
    PacketWrapper *operator->() const noexcept { return m_view.get(); }
    PacketWrapper &operator*() const noexcept { return *m_view; }
    explicit operator bool() const noexcept { return bool(m_view); }
    bool operator!() const noexcept { return !m_view; }

    //! Identity comparisons against another local_shared_ptr<PacketWrapper>
    //! (avoids materialising a fresh local_shared_ptr from the view).
    bool operator==(const local_shared_ptr<PacketWrapper> &rhs) const noexcept {
        // Public access: get() is public; ref_ptr_ is protected.
        // For intrusive types (PacketWrapper inherits atomic_countable),
        // Ref == T, so rhs.get() == rhs.ref_ptr_().
        return m_view.ref_ptr_() == rhs.get();
    }
    bool operator!=(const local_shared_ptr<PacketWrapper> &rhs) const noexcept {
        return !(*this == rhs);
    }
    friend bool operator==(const local_shared_ptr<PacketWrapper> &lhs,
                           const ScopedNegotiateLinkage &rhs) noexcept { return rhs == lhs; }
    friend bool operator!=(const local_shared_ptr<PacketWrapper> &lhs,
                           const ScopedNegotiateLinkage &rhs) noexcept { return rhs != lhs; }

    //! Identity comparisons against scoped_atomic_view<PacketWrapper>
    //! (e.g. CASInfo::old_wrapper after it's been changed to view type).
    bool operator==(const scoped_atomic_view<PacketWrapper> &rhs) const noexcept {
        return m_view.ref_ptr_() == rhs.ref_ptr_();
    }
    bool operator!=(const scoped_atomic_view<PacketWrapper> &rhs) const noexcept {
        return !(*this == rhs);
    }

    //! Identity comparisons against atomic_shared_ptr / Linkage.
    //! Forwards to scoped_atomic_view::operator==/!=(atomic_shared_ptr).
    //! Pure relaxed load + pointer comparison — no load_shared_.
    bool operator==(const atomic_shared_ptr<PacketWrapper> &rhs) const noexcept {
        return m_view == rhs;
    }
    bool operator!=(const atomic_shared_ptr<PacketWrapper> &rhs) const noexcept {
        return m_view != rhs;
    }

    // consume_view() removed — promotes TagHeld to Owned (fetch_add).
    // Use consume_scoped_view() (zero ops) or view() (const ref) instead.

    //! Move-out the view as a scoped_atomic_view<PacketWrapper>.
    //! ZERO atomic ops regardless of mode — the view is directly moved.
    //! After this, the scope's view is empty.  Use when the caller
    //! wants to store the view for later comparison (e.g. bundle's
    //! subwrappers_org) without paying promote + fetch_add costs.
    scoped_atomic_view<PacketWrapper> consume_scoped_view() noexcept {
        return std::move(m_view);
    }

    // view_copy() removed — promotes TagHeld to Owned (fetch_add + release_tag_ref_).
    // Use pointer comparison + compareAndSetRetain (retain newr) instead.

    //! Bare pointer to the linkage (for code that needs to compare
    //! linkage identity, e.g. unbundle()'s `oldsuperwrapper` chain
    //! tracking).
    const std::shared_ptr<typename Node<XN>::Linkage> &linkage() const noexcept {
        return m_link;
    }

    //! Const-ref access to the internal scoped_atomic_view.
    const scoped_atomic_view<PacketWrapper> &view() const noexcept {
        return m_view;
    }

    //! Access the Snapshot this scope is bound to.
    Snapshot<XN> &snap() const noexcept { return *m_snap; }

    //! Replace the internal view with a value from `from`, taking
    //! ownership of `from`'s +1 refcount (zero atomic ops on the
    //! transfer).  Use after a successful multi-phase CAS where the
    //! caller wants the scope's view to track the new linkage value
    //! without paying a load_shared_.
    void set_view(local_shared_ptr<PacketWrapper> &&from) noexcept {
        m_view.assign_from_local(std::move(from));
    }

    // ---------- CAS using internal view ----------

    //! CAS using the internal view as oldr.  Auto-commits on success.
    //! Dispatches to STRONG (spinning) variant when this scope was
    //! constructed by the privileged thread (m_strong_mode set in
    //! ctor).  STRONG retries internally on spurious weak failures;
    //! returns false only on real pointer mismatch.  Non-privileged
    //! scopes use the WEAK fast path; conservative dtor tag on
    //! spurious failure (m_contention_observed).
    bool compareAndSet(const local_shared_ptr<PacketWrapper> &desired) noexcept {
        bool ok = m_strong_mode
            ? m_link->compareAndSetStrong(m_view, desired)
            : m_link->compareAndSetWeak(m_view, desired);
        if(ok) {
            _on_cas_success();
            return true;
        }
        _on_cas_fail();
        return false;
    }

    //! compareAndSet + tags_successful_cas() priority/lease hint on
    //! success.  `started_time = 0` (default) makes `tags_successful_cas`
    //! use now_us().  Strong/weak dispatch as in compareAndSet.
    bool compareAndSetWithHint(const local_shared_ptr<PacketWrapper> &desired,
                                typename Node<XN>::NegotiationCounter::cnt_t
                                    started_time = 0) noexcept {
        bool ok = m_strong_mode
            ? m_link->compareAndSetStrong(m_view, desired)
            : m_link->compareAndSetWeak(m_view, desired);
        if(ok) {
            m_link->tags_successful_cas(started_time);
            _on_cas_success();
            return true;
        }
        _on_cas_fail();
        return false;
    }

    // External-expected CAS overloads removed — they accept
    // local_shared_ptr which forces callers to promote (view_copy).
    // Use pointer comparison + 1-arg CAS (internal view) instead.

    // ---------- CAS with newr retention (RETAIN_NEWR) ----------

    //! Like compareAndSet but on success the internal view transitions
    //! to Owned(desired) instead of going Empty.  The scope continues
    //! to track the new m_link value — no reload needed for a follow-up
    //! CAS in a later phase.  Entry does fetch_add(2) instead of (1);
    //! failure undo is fetch_sub(2) (same op count).
    //! Strong/weak dispatch as in compareAndSet.
    bool compareAndSetRetain(const local_shared_ptr<PacketWrapper> &desired) noexcept {
        bool ok = m_strong_mode
            ? m_link->compareAndSetStrongRetain(m_view, desired)
            : m_link->compareAndSetWeakRetain(m_view, desired);
        if(ok) {
            _on_cas_success();
            return true;
        }
        _on_cas_fail();
        return false;
    }

    //! compareAndSetRetain + tags_successful_cas hint.
    bool compareAndSetRetainWithHint(const local_shared_ptr<PacketWrapper> &desired,
                                typename Node<XN>::NegotiationCounter::cnt_t
                                    started_time = 0) noexcept {
        bool ok = m_strong_mode
            ? m_link->compareAndSetStrongRetain(m_view, desired)
            : m_link->compareAndSetWeakRetain(m_view, desired);
        if(ok) {
            m_link->tags_successful_cas(started_time);
            _on_cas_success();
            return true;
        }
        _on_cas_fail();
        return false;
    }

    // ---------- CAS with local_unique_ptr desired ----------

    //! Weak CAS using internal view as oldr, with local_unique_ptr<T>
    //! as desired (saves 2 atomic ops vs the local_shared_ptr<T>
    //! version).  desired is in/out: released on success (m_ref takes
    //! ownership), retained on failure.
    bool compareAndSet(local_unique_ptr<PacketWrapper> &desired) noexcept {
        if(m_link->compareAndSetWeak(m_view, desired)) {
            _on_cas_success();
            return true;
        }
        _on_cas_fail();
        return false;
    }

    bool compareAndSetWithHint(local_unique_ptr<PacketWrapper> &desired,
                                typename Node<XN>::NegotiationCounter::cnt_t
                                    started_time = 0) noexcept {
        if(m_link->compareAndSetWeak(m_view, desired)) {
            m_link->tags_successful_cas(started_time);
            _on_cas_success();
            return true;
        }
        _on_cas_fail();
        return false;
    }

    //! Caller-side hook for pre-CAS conflict detection (e.g.
    //! `wrapper->packet() != tr.m_oldpacket`,
    //! UnbundledStatus::SUBVALUE_HAS_CHANGED, walkUpChain DISTURBED).
    //! Forces the dtor to tag this iteration despite the retry==0
    //! fast-path optimization.  Not a CAS failure → does not bump
    //! the gate→cas_fail correlation counter.
    void confirm_contention() noexcept { m_contention_observed = true; }

private:
    //! Snapshot the thread_local gate-return flag set by the
    //! ctor-time negotiate_internal into this scope's per-scope
    //! storage, then clear the thread_local.  Called once per ctor
    //! after the negotiate call returns and before any CAS in the
    //! body.  Per-scope storage avoids nested-scope leakage where a
    //! nested scope's negotiate would overwrite the outer scope's
    //! gate-return flag.
    void _capture_gate_return() noexcept {
        m_was_gate_return = NegSite::last_was_gate_return();
        NegSite::last_was_gate_return() = false;
    }
    //! Common CAS-success path: mark committed and drive the
    //! per-site promotion-to-FORCE_GATE transition.
    //!
    //! Success streak (consec_succs) accumulates in:
    //!   - UNDEFINED, when this scope gated (m_was_gate_return true):
    //!     gating chose to skip the sleep and CAS succeeded — the
    //!     site looks "gate-safe".
    //!   - FORCE_SLEEP, on every success (we slept, then CAS
    //!     succeeded — the contention has eased enough that the
    //!     site can be retried as FORCE_GATE).
    //! Reaching K_SUCC flips to FORCE_GATE with a fresh lease.
    //! Any fail resets the streak (handled in _on_cas_fail).
    void _on_cas_success() noexcept {
        m_committed = true;
        // Adaptive anti-phase tighten: any CAS success means peer's
        // coalesce window is in sync — reset to default sensitivity.
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
        // INSTRUMENT: if this CAS success follows a recent gate-return
        // (m_last_gate_returned still true), record the latency.
        if(m_snap->m_last_gate_returned) {
            const uint32_t now_us =
                (uint32_t)Node<XN>::NegotiationCounter::now_us();
            const uint32_t latency =
                now_us - m_snap->m_gate_return_time_us;
            NegSite::record_gr_in_time(
                m_snap->m_gate_return_my_kind, latency);
        }
#endif
        m_snap->m_last_gate_returned = false;
        m_snap->m_gate_return_tighten = 0;
#if KAME_LEGACY_GATING
        // ===== Legacy gating: success-streak → FORCE_GATE transition ===
        m_site_state->consec_fails = 0;     // any success clears fails
        const int8_t tg = m_site_state->take_gate;
        const bool count_succ =
            (tg == 0 /*FORCE_SLEEP*/)
            || (tg == -1 /*UNDEFINED*/ && m_was_gate_return);
        if(count_succ
           && ++m_site_state->consec_succs
                  >= (uint16_t)NegSite::GATE_K_SUCC) {
            const uint64_t now_us = (uint64_t)
                Node<XN>::NegotiationCounter::now_us();
            m_site_state->take_gate = 1;    // FORCE_GATE
            m_site_state->normal_until_us = now_us
                + (uint64_t)NegSite::NORMAL_LEASE_US;
            m_site_state->consec_succs = 0;
            ++m_site_state->mode_flips_promote;
        }
        // ===== end legacy gating =======================================
#endif
        m_was_gate_return = false;
    }
    //! Common CAS-fail path.  Marks contention observed, accumulates the
    //! INSTRUMENT `gate_then_cas_fail` counter, and (under
    //! KAME_LEGACY_GATING) drives the per-site demotion-to-FORCE_SLEEP
    //! transition when the gate→fail streak is BOTH deep (K_FAIL) AND
    //! time-clustered (FAIL_WINDOW_US).
    void _on_cas_fail() noexcept {
        m_contention_observed = true;
        if(m_was_gate_return)
            ++m_site_state->gate_then_cas_fail;
#if KAME_LEGACY_GATING
        // ===== Legacy gating: fail-streak → FORCE_SLEEP transition =====
        // Failures count in UNDEFINED (when gated) and FORCE_GATE
        // (always gated); they cannot occur in FORCE_SLEEP (no gate, so
        // m_was_gate_return is false).  Any failure also clears the
        // success streak.
        m_site_state->consec_succs = 0;
        if(m_was_gate_return) {
            const uint64_t now_us = (uint64_t)
                Node<XN>::NegotiationCounter::now_us();
            if(m_site_state->consec_fails == 0
               || (now_us - m_site_state->last_fail_us)
                      > (uint64_t)NegSite::GATE_FAIL_WINDOW_US) {
                m_site_state->consec_fails = 1;
                m_site_state->last_fail_us = now_us;
            } else {
                ++m_site_state->consec_fails;
                if(m_site_state->consec_fails
                       >= (uint16_t)NegSite::GATE_K_FAIL) {
                    m_site_state->take_gate = 0;  // FORCE_SLEEP
                    m_site_state->normal_until_us = now_us
                        + (uint64_t)NegSite::NORMAL_LEASE_US;
                    m_site_state->consec_fails = 0;
                    ++m_site_state->mode_flips_g2n;
                }
            }
        }
        // ===== end legacy gating =======================================
#endif
        m_was_gate_return = false;
    }

    //! Adaptive backoff entry point — replaces the former
    //! `Linkage::negotiate()` inline.  Loads m_link's collision marker
    //! and short-circuits the call when no peer Tx has tagged this
    //! Linkage; otherwise delegates to `_negotiate_internal()`.
    void _negotiate() noexcept;

    //! Replaces `Linkage::negotiate_after_retry_pause(retry, snap, mult_wait)`.
    //! `retry==0` fast-paths out unless fair-mode privilege blocks
    //! this Tx; otherwise issues `retry_pause(retry)` then `_negotiate()`.
    void _negotiate_after_retry_pause(int retry) noexcept;

    //! Body of the priority-based adaptive backoff loop.  Moved from
    //! `Linkage::negotiate_internal` into this scope so that per-scope
    //! state (m_snap, m_mult_wait, m_link, m_site_state) is accessed
    //! directly instead of being threaded through arguments.  See the
    //! header comment on the definition for the algorithm details.
    void _negotiate_internal() noexcept;
public:

    //! Manual commit override.  Use when (a) the CAS happened in a
    //! nested scope so this scope's logical work is done, or (b) a
    //! prior phase's CAS already advanced state (e.g. bundle Phase 3
    //! child-CAS failure after Phase 2 succeeded).
    void commit() noexcept { m_committed = true; }

    ~ScopedNegotiateLinkage() noexcept {
        // Re-prime the SiteState slot so any negotiate() invocation
        // below sees the right per-site state.  m_link is the
        // freshness sentinel — cleared on move-out, so the moved-from
        // dtor skips all side effects (the move target handles them).
        // m_site_state itself is guaranteed non-null whenever m_link
        // is, so no additional null checks are needed.
        auto *_save_state = NegSite::current_state();
        if(m_link) {
            NegSite::current_state() = m_site_state;
            if(m_committed) ++m_site_state->commits;
#if KAME_ENABLE_RUNNER_DIGEST
            // Scope-end digest publish — peer-visible snapshot of
            // outcome (consec_succs/fails / take_gate updated by
            // _on_cas_*).  Internal skip-unchanged guard avoids the
            // atomic store when peer-actionable fields are stable.
            _publish_digest();
#endif
        }
        if(!m_committed) {
            // Tag rules:
            //  - OnEntry m_should_tag: ctor already tagged; skip dtor.
            //  - Otherwise: tag iff m_contention_observed (CAS failure
            //    or confirm_contention) OR original OnExit retry > 0
            //    optimization (!m_eager && m_should_tag).
            // tag_as_contender dedups internally, but we skip the call
            // when ctor already tagged to avoid the dedup walk.
            //
            // Tag is performed BEFORE the wait below so that any
            // subsequent notify_n_contenders walking tid_bitset can
            // find us and wake our sleep slot.
            if( !(m_eager && m_should_tag) &&
                (m_contention_observed || (!m_eager && m_should_tag)))
                m_snap->tag_as_contender(m_link);

            // On observed contention, wait before letting the caller
            // re-enter the loop.  But: if our view is in Owned mode
            // (already promoted to global ref via fetch_add), we hold
            // a ref decoupled from m_link's tag bits — neither
            // contender for tag-space (other acquires) nor for the
            // CAS (other compareAndSet's) gates us.  We can fall
            // through immediately; the caller's retry will load a
            // fresh state.  Skip the wait entirely in that case.
            //
            // For TagHeld and Empty views, wait based on privilege
            // state:
            //
            //   - **Someone ELSE holds privilege**: that Tx is making
            //     progress; we won't get the CAS until it commits.
            //     Block on the same CV slot negotiate_internal uses,
            //     so notify_n_contenders (called by the privileged Tx
            //     on commit) wakes us.  1ms timeout bounds latency if
            //     no notification arrives.  Avoids burning CPU yields.
            //   - **No privilege held / we hold privilege**: no specific
            //     thread to wait for; a light std::this_thread::yield()
            //     suffices to break the same-cycle re-entry pattern
            //     without paying the mutex/CV overhead.
            // Lock-free style release loop with privilege-aware
            // back-off.  rcnt_added tracks our cumulative pre-pay so
            // we only fetch_add/fetch_sub the DIFF on each iter
            // (no-op when observed tag count is stable).
            //   - We hold privilege: bare retry (no-op).
            //   - Other holds privilege: CV-wait via negotiate_sleep.
            //   - No privilege: yield.
            // Direct negotiate call mirroring ctor's retry<0 path.
            // m_mult_wait inherited from ctor — per-scope tuning carries
            // through (e.g. bundle's Phase 3 childScope's smaller
            // mult_wait stays smaller in dtor too).
            if(m_contention_observed) {
                uintptr_t rcnt_added = 0;
                while( !m_view.try_release_single_attempt(rcnt_added)) {
                    _negotiate();
                }
            }
        }
        // Restore the SiteState slot saved at the top of dtor.
        NegSite::current_state() = _save_state;
    }
};

} // namespace Transactional

#endif /* TRANSACTION_NEGOTIATION_H */
