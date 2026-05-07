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
#include "transaction.h"
#include <vector>
#include <thread>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <condition_variable>

#define KAME_STM_LIVELOCK_FALLBACK 1

#include <chrono>
#include <cstdio>

// --- Compile-time tuning knobs for the adaptive-negotiate backoff ---
// All are -D overridable at cmake time.

// Half-range of the jittered gate in percent; must be ≥1 (0 causes div-by-zero
// in JITTER_DIV). Sweep (N=128 median/3): JIT=10 avg4=4841k > JIT=25 avg4=4672k.
#ifndef KAME_STM_JITTER_RANGE
#define KAME_STM_JITTER_RANGE 25   // sweep winner (N=128 median/3)
#endif

// Ablation knob: 1 → disable both the jittered gate (gate factor pinned to
// 1.0) and the sleep-chunk ±1ms de-phasing (chunk fixed at 1 ms). For paper
// comparison / on-off measurement.
#ifndef KAME_STM_DISABLE_JITTER
#define KAME_STM_DISABLE_JITTER 0
#endif

// Gate coefficient: gate opens when mult_wait * GATE_MULT * dt * J < dt2.
// Smaller = more permissive (threads break out sooner). Default 0 (closed).
#ifndef KAME_STM_GATE_MULT
#define KAME_STM_GATE_MULT 1.0f
#endif

// Multiplier on the √C lottery threshold. 1 = ~√C bypass per iteration.
#ifndef KAME_STM_LOTTERY_MULT
#define KAME_STM_LOTTERY_MULT 1
#endif

// Cap on threads simultaneously in the CAS-retry loop.
// Positive = excess lottery winners fall through to the sleep path to
// limit simultaneous-CAS bursts. Gate winners (earned priority) are
// never capped.
//  -1 = auto (hardware_concurrency(), fallback to max(C_obs))
//   0 = disabled
//   N > 0 = fixed threshold
#ifndef KAME_STM_MAX_RUNNERS
#define KAME_STM_MAX_RUNNERS 2   // sweep winner: 2 > 10 > -1 > 6 > 8 > 4 > 16
#endif

// Floor on concurrent runners; lottery wins are denied while fewer
// than this many runners are active so the wake pipeline has room.
//  -1 = auto (hardware_concurrency(), fallback to max(C_obs))
//   0 = disabled
//   N > 0 = fixed threshold
#ifndef KAME_STM_MIN_RUNNERS
#define KAME_STM_MIN_RUNNERS -1
#endif

// --- Adaptive-lease tuning (see lease block in negotiate_internal) ---
// Per-Linkage `lease_us` drifts by these schedules. Asymmetric: growth
// scales with C (capped) so heavy contention climbs quickly; shrink is
// a smaller fixed step so a single C=0 call doesn't undo many C>=2
// adjustments — equilibrium biases toward higher lease where Linkages
// see contention.
#ifndef KAME_LEASE_GROW_PER_C_PERCENT
#define KAME_LEASE_GROW_PER_C_PERCENT 30   // additive per unit of (C-1)
#endif
#ifndef KAME_LEASE_GROW_MAX_PERCENT
#define KAME_LEASE_GROW_MAX_PERCENT   80   // cap on per-call growth
#endif
#ifndef KAME_LEASE_SHRINK_PERCENT
#define KAME_LEASE_SHRINK_PERCENT     5    // shrink step when C == 0; sweep winner
#endif
// Quantized lease write: skip the atomic store unless |delta| ≥ this
// (in µs). Once the lease pins at MIN/MAX rail, clamping yields delta=0
// and the store is suppressed — avoids ping-ponging m_priority_state.
#ifndef KAME_LEASE_QWRITE_US
#define KAME_LEASE_QWRITE_US 1
#endif
// dt2 fairness gate: suppress owner-skip when the competing tx has
// been waiting longer than this (µs). Hot-loop commits (dt2 ≈ 2-8 µs)
// stay below the gate and benefit from the skip; long-held tx
// (test_negotiation msecsleep, dt2 ≈ 60-150 µs) bypass it.
#ifndef KAME_DT2_FAIRNESS_US
#define KAME_DT2_FAIRNESS_US 2000
#endif

// STRICT_assert / STRICT_TEST — debug-build-only macros.
// STRICT_assert(expr): behaves like assert(expr) when TRANSACTIONAL_STRICT_assert is defined;
//   compiles to nothing otherwise. Used to verify STM invariants (e.g. packet consistency)
//   that are too expensive for release builds.
// STRICT_TEST(expr): evaluates expr only in strict-assert builds. Typically used to maintain
//   auxiliary state (e.g. s_serial_abandoned) consumed by STRICT_assert checks.
#ifdef TRANSACTIONAL_STRICT_assert
    #undef STRICT_assert
    #define STRICT_assert(expr) assert(expr)
    #define STRICT_TEST(expr) expr
#else
    #define STRICT_assert(expr)
    #define STRICT_TEST(expr)
#endif

namespace Transactional {

// Per-thread runner counter infrastructure. The vector and registration
// helper live in Transactional::detail (see transaction.h for design).
// Definitions are non-templated (only one Node<XN> instantiation in
// practice). transaction_impl.h is included from exactly one TU per
// binary (libkame's xnode.cpp; each tests/transaction_*_test.cpp), so
// these can be plain (non-inline) definitions. macOS two-level
// namespace would otherwise duplicate `inline` storage across
// libkame.dylib and module dylibs (libthamway, libnmr, ...) and
// break the runner-counter / nest-depth singleton invariants.
namespace detail {

// Per-thread nesting / TLS storage. Apple/Linux: declared `extern
// thread_local` in transaction.h, defined here as plain thread_local
// (transaction_impl.h is included from exactly one TU per binary).
// Windows: `__declspec(dllexport) thread_local` is forbidden by MSVC,
// so libkame instead exports `*_ref()` accessor functions; the storage
// lives as a function-local `thread_local` inside each accessor (one
// instance per thread, per program — same DLL hosts the storage).
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
// transaction.h aliases bare names to the accessor calls. Undef here
// so we can write the accessor bodies without recursive expansion.
#  undef s_tx_nest
#  undef s_sleep_nest
#  undef tls_runner_counter_holder
#  undef tls_runner_counter_ptr
DECLSPEC_KAME int& s_tx_nest_ref() noexcept { thread_local int v = 0; return v; }
DECLSPEC_KAME int& s_sleep_nest_ref() noexcept { thread_local int v = 0; return v; }
DECLSPEC_KAME void*& tls_payload_creator_ptr() noexcept { thread_local void* v = nullptr; return v; }
// Lamport serial: int64_t holding (48-bit counter | 16-bit thread ID).
// Initialized to ProcessCounter::id() on first call (matches cnt_t ctor).
// Must be defined AFTER tls_payload_creator_ptr so ProcessCounter::id()
// is already declared (via transaction_signal.h included before this block).
DECLSPEC_KAME int64_t& tls_serial_ref() noexcept {
    thread_local int64_t v = (int64_t)ProcessCounter::id();
    return v;
}
DECLSPEC_KAME std::shared_ptr<RunnerCounterEntry>& tls_runner_counter_holder_ref() noexcept {
    thread_local std::shared_ptr<RunnerCounterEntry> v;
    return v;
}
DECLSPEC_KAME RunnerCounterEntry*& tls_runner_counter_ptr_ref() noexcept {
    thread_local RunnerCounterEntry* v = nullptr;
    return v;
}
// Re-instate the aliases so the rest of transaction_impl.h refers to
// the variables uniformly.
#  define s_tx_nest                  s_tx_nest_ref()
#  define s_sleep_nest               s_sleep_nest_ref()
#  define tls_runner_counter_holder  tls_runner_counter_holder_ref()
#  define tls_runner_counter_ptr     tls_runner_counter_ptr_ref()
#else
thread_local int s_tx_nest = 0;
thread_local int s_sleep_nest = 0;
thread_local std::shared_ptr<RunnerCounterEntry> tls_runner_counter_holder;
thread_local RunnerCounterEntry* tls_runner_counter_ptr = nullptr;
#endif

// `DECLSPEC_KAME` on the definitions too — MSVC is more lenient when
// dllexport appears on both declaration and definition. Without it,
// each module DLL can end up with its own private copy of these
// symbols, defeating the libkame singleton invariant (the macOS DSO
// duplication failure mode that motivated this whole reorg).
DECLSPEC_KAME atomic_shared_ptr<RunnerCounterVec> s_runner_counters{};

DECLSPEC_KAME RunnerCounterEntry& runner_counter_register() {
    auto sp = std::make_shared<RunnerCounterEntry>();
    tls_runner_counter_holder = sp;
    tls_runner_counter_ptr = sp.get();
    // COW publish: append our weak_ptr to the global vector and prune
    // expired entries (threads that have exited) in the same step so
    // the vector self-trims without a separate maintenance path.
    for(local_shared_ptr<RunnerCounterVec> old(s_runner_counters);;) {
        local_shared_ptr<RunnerCounterVec> next;
        next.reset(new RunnerCounterVec);
        if(old) {
            next->reserve(old->size() + 1);
            for(auto &w : *old)
                if( !w.expired())
                    next->push_back(w);
        }
        next->push_back(std::weak_ptr<RunnerCounterEntry>(sp));
        if(s_runner_counters.compareAndSwap(old, next)) break;
    }
    return *sp;
}

DECLSPEC_KAME RunnerCounterEntry& my_runner_counter_impl() {
    auto *p = tls_runner_counter_ptr;
    if(p) return *p;
    return runner_counter_register();
}

DECLSPEC_KAME unsigned int num_threads_running_impl() noexcept {
    local_shared_ptr<RunnerCounterVec> snap(s_runner_counters);
    if( !snap) return 0;
    uint64_t s = 0;
    for(auto &w : *snap)
        if(auto sp = w.lock())
            s += sp->v.load(std::memory_order_relaxed);
    return (unsigned)s;
}

} // namespace detail

// tx_nest / sleep_nest are defined at namespace scope in transaction.h
// (detail::s_tx_nest, detail::s_sleep_nest). Placing them outside the
// class template avoids an Apple clang / arm64 bug where the TLS wrapper
// for a template static `thread_local` member is not emitted in TUs
// that only include transaction.h.

STRICT_TEST(static atomic<int64_t> s_serial_abandoned = -2);

// (popcount_u64 moved to transaction.h so inline member functions there
//  — negotiate() — can call it too.)

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
    // Ablation: disable the backoff layer; retry loops become pure
    // CAS-retry spin. Paired with an early return in negotiate().
    (void)retry;
    return;
#endif
    // Spin count grows linearly with retry depth (uncapped) so at large
    // retry the pause exceeds the inter-core coherence window and some
    // thread gets an uncontested CAS slot.
    uint32_t h = (uint32_t)retry * 0x9E3779B1u;
    h ^= (uint32_t)(uintptr_t)&retry;       // per-thread entropy from stack addr
    int spins = 1 + (int)(((uint64_t)(h >> 16) * (uint32_t)retry) >> 16);
    for(int k = 0; k < spins; ++k) pause4spin();
}

// --- CV-based negotiate sleep helpers ---------------------------------
// Fixed 512-entry array of (mutex, cv, notified) slots at namespace scope.
// A sleeping thread picks slot = ProcessCounter::id() % 512 and cv.waits.
// Notifiers walk tid_bitset and mark + notify_one on each indexed slot.
//
// Why a fixed array rather than per-thread TLS + pointer slots:
//  - No thread-exit race / UAF: slot objects live for process lifetime.
//  - No MSVC + DLL non-trivial thread_local hazard.
//  - No atomic_shared_ptr refcount cost on the notify hot path —
//    mutex+cv is a plain global, notify_one is the only indirection.
//
// Kept under Transactional::detail with C++17 `inline` variables so every
// TU that includes this header shares the SAME array; an anonymous
// namespace would create one array per TU and inter-TU notify would not
// cross over.
//
// Slot collision (id % 512 shared by two live threads): both use the same
// cv, and a notify intended for one can wake the other. The woken thread
// re-checks the `notified` predicate under the lock; a stray wake just
// returns to wait. A genuine collision where the intended receiver
// misses its wake falls back to the wait_for timeout (same fallback as
// the previous pointer-slot design). 512 slots handles practical thread
// pool sizes; raise NEGOTIATE_SLEEP_SLOTS if you regularly run > 512
// workers with persistent ProcessCounter ids.
}   // close Transactional so detail can open with clean scope

namespace Transactional {

// Forward declaration. effective_runners() is the hardware_concurrency-
// aware contender-count helper used by negotiate_internal's MIN_RUNNERS
// / MAX_RUNNERS gating; declared here so it is visible to all of the
// detail:: machinery (notify_n_contenders, etc.). Full definition is
// below the detail namespace, gated on the MIN_RUNNERS / MAX_RUNNERS
// macros.
#if (KAME_STM_MIN_RUNNERS != 0) || (KAME_STM_MAX_RUNNERS != 0)
inline int effective_runners(int c_obs) noexcept;
#else
inline int effective_runners(int) noexcept {
    int hw = (int)std::thread::hardware_concurrency();
    return hw > 0 ? hw : 1;
}
#endif

namespace detail {

    // Livelock observation probe (always on).
    //
    // Signal: MY Transaction-retry rate vs. the LINKAGE's
    // Transaction-commit rate on a rolling ≥ 10 ms window. Emits one
    // stderr line per window:
    //   [ll-probe] tid=... linkage=... my_tx_retry_rate=N/s
    //              tx_commit_rate=M/s ratio=X window_ms=T
    //
    // Transaction-level, NOT CAS-level: "CAS operations succeed but
    // iterate_commit keeps invalidating the whole Transaction" is the
    // pathology we care about. my_tx_retry comes from Snapshot's
    // m_tx_retry_count, tx_commit from Linkage::m_tx_commit_count
    // (bumped in finalizeCommitment).
    //
    // Note: m_tx_retry_count is bumped from TWO sites — Transaction::
    // operator++ (outer iterate_commit retries) AND Node::snapshot()'s
    // retry loop (pure-Snapshot bundle retries). The `tx_retry_window`
    // member below snapshots that field, so the printed `my_tx_retries`
    // / `my_tx_retry_rate` aggregate both Tx-retry and bundle-retry
    // counts. Name is kept for log-format / ABI continuity (see
    // m_tx_retry_count doc-block in transaction.h).
    //
    // Fires only at negotiate_internal entry (slow path); gate hits
    // and lottery wins stay zero-cost. Cost over a probe-less build
    // is one uint32_t per Snapshot, one uint64_t per Linkage, and two
    // unconditional `++` statements.
    struct LivelockProbe {
        const void *linkage_id       = nullptr;
        int64_t     t_window_us      = 0;
        uint32_t    tx_retry_window  = 0;
        uint64_t    tx_commit_window = 0;
    };
    // Non-inline: transaction_impl.h is materialised in exactly one
    // TU per binary (see header-block comment above), so a plain
    // `thread_local` definition gives one slot per program.
    thread_local LivelockProbe tls_livelock_probe;

    inline int64_t ll_now_us() noexcept {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Fires the livelock probe. The caller (negotiate_internal, which
    // has access to Snapshot's protected members via enclosing-class
    // friendship) pre-computes the tag-ownership counts, the
    // retry-threshold (chosen from Priority), and the priority label,
    // then passes plain values here.
    //
    // Verdict:
    //   tags_owned == tags_total > 0                  AND
    //   my_tx_retries >= retry_threshold              AND
    //   tx_age_us > KAME_STM_LIVELOCK_MIN_AGE_US (20 ms default)
    //
    // retry_threshold is priority-dependent:
    //   HIGHEST         → 2  (real-time, must not retry)
    //   NORMAL          → 3
    //   UI_DEFERRABLE   → 5  (deferred UI repaint — retries tolerated)
    //   LOWEST          → 5  (background — tolerates yielding)
    //
    // Two unrelated time scales appear in the printout:
    //   tx_age_us  = age of the current Tx/Snapshot (set by Snapshot or
    //                Transaction ctor; persists across retries — what
    //                the verdict gates on).
    //   window_ms  = inter-tick interval on this (linkage, thread) —
    //                the rate-window for my_tx_retry_rate /
    //                tx_commit_rate. NOT Tx-related; in tight retry
    //                loops it is often sub-ms, which is why the printed
    //                rates can look enormous (small delta divided by a
    //                tiny window).
    //! Returns true iff a tick concluded `verdict=LIVELOCK`. Defined
    //! out-of-class as `Node<XN>::NegotiationCounter::livelock_probe_tx_tick`.

//=============================================================================
    // Fair-mode escape: globally registered "privileged TID" mechanism.
    //
    // When the livelock probe (above) returns verdict=LIVELOCK on a
    // Transaction that is the Greedy CM winner (oldest started_time,
    // all linkage tags claimed) yet making no commit progress, that Tx
    // CAS-claims `s_privileged_tidstamp`. While the slot is non-zero AND
    // != my_tid:
    //
    //   * the jittered gate and √C lottery in negotiate_internal's
    //     main sleep loop are bypassed — strict Greedy CM (older
    //     started_time wins → others sleep) is the sole priority
    //     mechanism, ensuring the privileged Tx makes progress
    //     against deterministic backoff alone.
    //   * notify_n_contenders wakes the privileged TID's sleep slot
    //     first, before walking tid_bitset.
    //
    // Per-Transaction ownership lives on `Snapshot::m_registered_privileged`
    // for nesting safety (see transaction.h). The slot is cleared by:
    //   - finalizeCommitment (success): plain store(0, relaxed)
    //   - ~Transaction (abort): same plain store
    // — only the registering Tx holds the flag, no concurrent writer
    // is possible, so a CAS would be overkill.
    //
    // V0 (legacy non-adaptive) and the V0↔ADAPTIVE mode switch were
    // removed in this revision: empirical M4 / iMac Pro comparison
    // (paper §3.6) showed V0 to be at best on par and sometimes 5×
    // slower than the always-on adaptive-lease path. The fair-mode
    // escape, in contrast, is orthogonal to mode and the retry-path
    // tag_as_contender call sites are now unconditional.
    //=============================================================================

} // namespace detail
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
    Snapshot<XN>   &m_snap;
    //! Tag-ref'd view of m_link's current PacketWrapper.  Acquired in
    //! ctor (DEFER_THRESHOLD — stays TagHeld; no fetch_add unless
    //! moved out as local_shared_ptr).  Used as oldr for the scope's
    //! built-in compareAndSet / compareAndSetWithHint methods.
    //! After a successful CAS the view is consumed (empty); after a
    //! failed CAS the view may also be empty depending on the
    //! weak+scoped contract — caller can re-acquire via reload_view().
    scoped_atomic_view<PacketWrapper> m_view;
    bool            m_eager;
    bool            m_should_tag;            // retry != 0 — fast-path optimization
    bool            m_committed = false;
    bool            m_contention_observed = false;  // forces dtor tag despite retry==0
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
    ScopedNegotiateLinkage(LinkagePtr link, Snapshot<XN> &snap, int retry,
                           TagMode mode = TagMode::OnEntry,
                           float mult_wait = 2.0f) noexcept
        : m_link(std::move(link)), m_snap(snap),
          m_eager(mode == TagMode::OnEntry),
          m_should_tag(retry != 0) {
        if(retry < 0)
            m_link->negotiate(snap, mult_wait);  // always negotiate, no retry_pause
        else
            m_link->negotiate_after_retry_pause(retry, snap, mult_wait);
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
        auto priv = NC::s_privileged_tidstamp.load(std::memory_order_relaxed);
        bool we_hold_priv = (priv != (typename NC::cnt_t)0)
            && (detail::stamp_tid(priv)
                == detail::stamp_tid(m_snap.m_started_time));
        m_view = scoped_atomic_view<PacketWrapper>(
            *m_link,
            we_hold_priv
                ? scoped_atomic_view<PacketWrapper>::DEFER_THRESHOLD
                : scoped_atomic_view<PacketWrapper>::ADAPTIVE_THRESHOLD,
            /*weakly=*/true);
        if(!m_view.acquire_succeeded()) {
            // Weak acquire CAS lost (or local refcnt at capacity) —
            // same treatment as a CAS failure: forces dtor tag despite
            // retry==0 fast-path optimization.
            m_contention_observed = true;
        }
        if(m_eager && m_should_tag)
            m_snap.tag_as_contender(m_link);
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
                           bool with_negotiate = false) noexcept
        : m_link(std::move(link)), m_snap(snap),
          m_eager(mode == TagMode::OnEntry),
          m_should_tag(retry != 0) {
        if(with_negotiate) {
            if(retry < 0)
                m_link->negotiate(snap, mult_wait);
            else
                m_link->negotiate_after_retry_pause(retry, snap, mult_wait);
        }
        m_view = scoped_atomic_view<PacketWrapper>(*m_link, std::move(from));
        if(m_eager && m_should_tag)
            m_snap.tag_as_contender(m_link);
    }

    ScopedNegotiateLinkage(const ScopedNegotiateLinkage &) = delete;
    ScopedNegotiateLinkage &operator=(const ScopedNegotiateLinkage &) = delete;

    // ---------- View access (forwarding to internal scoped view) ----------

    //! Pointer-style access to the wrapped PacketWrapper.
    PacketWrapper *operator->() const noexcept { return m_view.get(); }
    PacketWrapper &operator*() const noexcept { return *m_view; }
    explicit operator bool() const noexcept { return bool(m_view); }
    bool operator!() const noexcept { return !m_view; }

    //! Identity comparisons against another local_shared_ptr<PacketWrapper>
    //! (avoids materialising a fresh local_shared_ptr from the view).
    bool operator==(const local_shared_ptr<PacketWrapper> &rhs) const noexcept {
        return m_view.ref_ptr_() == rhs.ref_ptr_();
    }
    bool operator!=(const local_shared_ptr<PacketWrapper> &rhs) const noexcept {
        return !(*this == rhs);
    }
    friend bool operator==(const local_shared_ptr<PacketWrapper> &lhs,
                           const ScopedNegotiateLinkage &rhs) noexcept { return rhs == lhs; }
    friend bool operator!=(const local_shared_ptr<PacketWrapper> &lhs,
                           const ScopedNegotiateLinkage &rhs) noexcept { return rhs != lhs; }

    //! Move-out the view as a local_shared_ptr<PacketWrapper>.  After
    //! this, the scope's view is empty; the scope can no longer be
    //! used as oldr for compareAndSet*() until reload_view() is called.
    //! Use for sub-routines that need a chain-walking value-typed
    //! wrapper (walkUpChain etc).  Cheap when the view is in Owned
    //! mode (zero atomic ops); two ops if still TagHeld (promote).
    local_shared_ptr<PacketWrapper> consume_view() noexcept {
        return std::move(m_view);
    }

    //! Lvalue-copy the view as a local_shared_ptr<PacketWrapper>.
    //! Scope's view is preserved (still usable as oldr).  Costs one
    //! fetch_add(1) for the new ref, plus 2 ops if the view was in
    //! TagHeld mode (promotes to Owned in-place).  Use when the
    //! scope's CAS will still be needed after extracting a copy
    //! (e.g. unbundle's superwrapper for walkUpChain, while the
    //! scope's internal view is reserved as oldr for the final CAS).
    local_shared_ptr<PacketWrapper> view_copy() noexcept {
        return m_view;  // forwards to scoped_atomic_view::operator local_shared_ptr<T>() &
    }

    //! Bare pointer to the linkage (for code that needs to compare
    //! linkage identity, e.g. unbundle()'s `oldsuperwrapper` chain
    //! tracking).
    const std::shared_ptr<typename Node<XN>::Linkage> &linkage() const noexcept {
        return m_link;
    }

    //! Re-acquire the view from m_link (after a CAS failure or after
    //! consume_view()).
    void reload_view() noexcept {
        m_view = scoped_atomic_view<PacketWrapper>(*m_link);
    }

    //! Replace the internal view with a value from `from`, taking
    //! ownership of `from`'s +1 refcount (zero atomic ops on the
    //! transfer).  Use after a successful multi-phase CAS where the
    //! caller wants the scope's view to track the new linkage value
    //! without paying a load_shared_.
    void set_view(local_shared_ptr<PacketWrapper> &&from) noexcept {
        m_view.assign_from_local(std::move(from));
    }

    // ---------- CAS using internal view ----------

    //! Weak CAS using the internal view as oldr.  Auto-commits on success.
    //! Spurious failures (no actual mismatch) are possible on LL/SC
    //! architectures; the retry loop handles them.  A spurious failure
    //! conservatively records m_contention_observed → dtor tag.
    bool compareAndSet(const local_shared_ptr<PacketWrapper> &desired) noexcept {
        if(m_link->compareAndSetWeak(m_view, desired)) {
            m_committed = true;
            return true;
        }
        m_contention_observed = true;
        return false;
    }

    //! Weak CAS using internal view + tags_successful_cas() priority/
    //! lease hint on success.  `started_time = 0` (default) makes
    //! `tags_successful_cas` use now_us().
    bool compareAndSetWithHint(const local_shared_ptr<PacketWrapper> &desired,
                                typename Node<XN>::NegotiationCounter::cnt_t
                                    started_time = 0) noexcept {
        if(m_link->compareAndSetWeak(m_view, desired)) {
            m_link->tags_successful_cas(started_time);
            m_committed = true;
            return true;
        }
        m_contention_observed = true;
        return false;
    }

    // ---------- CAS with externally-supplied expected (legacy) ----------

    //! Weak CAS on the scope's linkage with caller-supplied `expected`.
    //! Used by the few sites that hold the expected wrapper as a
    //! separate value (e.g. bundle Phase 3 child CAS).
    bool compareAndSet(const local_shared_ptr<PacketWrapper> &expected,
                       const local_shared_ptr<PacketWrapper> &desired) noexcept {
        if(m_link->compareAndSetWeak(expected, desired)) {
            m_committed = true;
            return true;
        }
        m_contention_observed = true;
        return false;
    }

    //! Weak CAS with hint, externally-supplied expected.
    bool compareAndSetWithHint(const local_shared_ptr<PacketWrapper> &expected,
                                const local_shared_ptr<PacketWrapper> &desired,
                                typename Node<XN>::NegotiationCounter::cnt_t
                                    started_time = 0) noexcept {
        if(m_link->compareAndSetWeak(expected, desired)) {
            m_link->tags_successful_cas(started_time);
            m_committed = true;
            return true;
        }
        m_contention_observed = true;
        return false;
    }

    //! Caller-side hook for pre-CAS conflict detection (e.g.
    //! `wrapper->packet() != tr.m_oldpacket`,
    //! UnbundledStatus::SUBVALUE_HAS_CHANGED, walkUpChain DISTURBED).
    //! Forces the dtor to tag this iteration despite the retry==0
    //! fast-path optimization.
    void confirm_contention() noexcept { m_contention_observed = true; }

    //! Manual commit override.  Use when (a) the CAS happened in a
    //! nested scope so this scope's logical work is done, or (b) a
    //! prior phase's CAS already advanced state (e.g. bundle Phase 3
    //! child-CAS failure after Phase 2 succeeded).
    void commit() noexcept { m_committed = true; }

    ~ScopedNegotiateLinkage() noexcept {
        if(!m_committed) {
            // Yield to other threads before tagging.  This gives:
            //  - drainers of m_link's tag bits (load_shared_, successful
            //    CAS) a chance to absorb leftover IOUs before our caller
            //    re-enters the loop and acquires a fresh tag,
            //  - the privileged Tx (if any) a window to make progress
            //    while we're parked in the scheduler queue,
            //  - the contention pattern on m_ref to dissipate by
            //    splitting the natural same-cycle re-entry back into
            //    interleaved iterations.
            // Only fires when contention was actually observed (CAS
            // failure or confirm_contention) — the common no-contention
            // path stays cheap.
            if(m_contention_observed)
                std::this_thread::yield();
            // Tag rules:
            //  - OnEntry m_should_tag: ctor already tagged; skip dtor.
            //  - Otherwise: tag iff m_contention_observed (CAS failure
            //    or confirm_contention) OR original OnExit retry > 0
            //    optimization (!m_eager && m_should_tag).
            // tag_as_contender dedups internally, but we skip the call
            // when ctor already tagged to avoid the dedup walk.
            if( !(m_eager && m_should_tag) &&
                (m_contention_observed || (!m_eager && m_should_tag)))
                m_snap.tag_as_contender(m_link);
        }
    }
};

// =====================================================================
// Out-of-class template member definitions for Node<XN>::NegotiationCounter.
// Declarations live in transaction.h; bodies here pick up the namespace-
// scope `detail::tls_livelock_probe`, `detail::ll_now_us`, and the
// `s_privileged_tidstamp` / `s_sleep_slots` C++17 inline static members
// (defined in the class body in transaction.h).
// =====================================================================

// Floor used for both (a) the LIVELOCK verdict gate (probe fires only
// Floor used for both (a) the LIVELOCK verdict gate (probe fires only
// when tx_age > floor) and (b) the age-preempt threshold in
// try_register_privileged_tidstamp (preemptor must be older than holder
// by floor µs). Values balance "fire quickly when stuck" vs "don't
// thrash on normal contention". The previous 100ms / 20ms / 1ms-broken
// regimes were chasing a *different* bug: `fair_mode_blocks_me`
// originally compared the full packed tid+stamp instead of just the
// TID, so a nested Tx on the privilege-holding thread (e.g. an outer
// iterate_commit's retry path triggering ~Node()->releaseAll()) was
// blocked by its own privilege and self-deadlocked. With that fix in
// place the threshold can be set as low as a few ms without
// triggering the test_dyn churn-deadlock.
#ifndef KAME_STM_PRIV_AGE_NORMAL_US
  // OS-aware default. The lower bound is set by the OS scheduler
  // quantum and condition-variable wait granularity, since the
  // privileged Tx's commit critical path includes one or more
  // OS-scheduled wake/sleep cycles.
  //   Windows: timer tick ≈ 15.6 ms (default), wait_for(1 ms)
  //            actually waits ~16 ms. PRIV_AGE under that round-up
  //            risks preempt thrash in heavy-contention scenarios.
  //   Linux/macOS: 1 ms or finer granularity. The M3 Air 9-point sweep
  //                (2026-04-29) found 100–200 µs as the throughput sweet-
  //                spot for K=10 N=128 (2L +135 % / 3L +55 % vs 750 µs)
  //                with bimodal regime collapse eliminated. 300 µs is a
  //                conservative pick: it captures most of the gain
  //                (2L +82 % / 3L +27 %) while staying out of the
  //                non-monotonic 3L window where 200 µs regresses.
  //                Other arches (x86 Xeon, NUMA AMD EPYC) are unmeasured;
  //                300 µs is the safer default for them too.
  #if defined(_WIN32) || defined(WINDOWS) || defined(__WIN32__)
    #define KAME_STM_PRIV_AGE_NORMAL_US 10'000   // 10 ms — Windows scheduler quantum
  #else
    #define KAME_STM_PRIV_AGE_NORMAL_US 300      // 300 us — Linux/macOS
  #endif
#endif
template <class XN>
int64_t Node<XN>::NegotiationCounter::min_privilege_age_us(Priority pr) noexcept {
    switch (pr) {
    case Priority::LOWEST:        return 30'000;
    case Priority::UI_DEFERRABLE: return 50'000;
    default:                      return KAME_STM_PRIV_AGE_NORMAL_US;  /* HIGHEST / NORMAL */
    }
}

template <class XN>
bool Node<XN>::NegotiationCounter::try_register_privileged_tidstamp(
    Priority pr, cnt_t tidstamp) noexcept
{
    int64_t tx_age_us = detail::ll_now_us() - detail::stamp_us(tidstamp);
    // CAS-loop: claim the slot if empty, OR preempt the current holder
    // if we are older than them by at least min_privilege_age_us(pr).
    // Preemption helps when the existing privileged Tx is itself stuck
    // (failed to release in finite time): a sufficiently older Tx can
    // take over and try to make progress instead of waiting indefinitely.
    cnt_t expected = s_privileged_tidstamp.load(std::memory_order_relaxed);
    const int64_t age_floor = min_privilege_age_us(pr);
    while (true) {
        if (expected != (cnt_t)0) {
            // Slot held. Allow override only if we are older than the
            // current holder by at least age_floor µs.
            int64_t age_diff =
                (int64_t)detail::stamp_us(expected)
                - (int64_t)detail::stamp_us(tidstamp);
            if (age_diff < age_floor)
                return false;
        }
        if (s_privileged_tidstamp.compare_exchange_weak(
                expected, tidstamp,
                std::memory_order_seq_cst,
                std::memory_order_relaxed))
            break;
        // CAS failed; `expected` reloaded — re-evaluate.
    }
    std::fprintf(stderr,
        "[ll-probe] privileged_tid=%u "
        "(claimed by stuck oldest Tx; age=%lld us, prio=%d%s)\n",
        (unsigned)detail::stamp_tid(tidstamp),
        (long long)tx_age_us, (int)pr,
        expected == (cnt_t)0 ? "" : " preempted");
    return true;
}

template <class XN>
void Node<XN>::NegotiationCounter::release_privileged_tidstamp(cnt_t my_tidstamp) noexcept {
    // CAS-based release: only clear the slot if it still holds OUR
    // stamp. Required because age-preempt can cause an older Tx to
    // overwrite our slot while we still hold m_registered_privileged.
    // Plain store(0) would then erase the preemptor's claim, leading
    // to slot/flag desynchronisation — observed as a hang in
    // transaction_dynamic_node_test under heavy churn.
    cnt_t expected = my_tidstamp;
    s_privileged_tidstamp.compare_exchange_strong(
        expected, (cnt_t)0,
        std::memory_order_seq_cst,
        std::memory_order_relaxed);
    // CAS fail = preemptor took our slot; they will release on their
    // own commit/abort.
}

template <class XN>
bool Node<XN>::NegotiationCounter::fair_mode_blocks_me(cnt_t tidstamp) noexcept {
    cnt_t priv = s_privileged_tidstamp.load(std::memory_order_relaxed);
    if(priv == (cnt_t)0) return false;
    // Compare by TID only (upper 16 bits of the packed stamp), NOT by
    // the full timestamp. The privileged Tx and a *nested* Tx on the
    // same thread carry different started_time stamps (e.g., the
    // outer Tx's retry path triggers ~Node()->releaseAll() which
    // starts an inner iterate_commit_if), but the inner Tx is still
    // owned by the privilege-holding thread and must not be blocked.
    // A full-stamp inequality check (priv != tidstamp) self-deadlocks
    // because the inner Tx waits in negotiate_sleep for a privilege
    // it already holds via the outer Tx — see hang in
    // transaction_dynamic_node_test backtrace (~Node->releaseAll on
    // frame #15-16, negotiate_sleep on frame #9).
    return detail::stamp_tid(priv) != detail::stamp_tid(tidstamp);
}

template <class XN>
typename Node<XN>::NegotiationCounter::PriorityProbeInfo
Node<XN>::NegotiationCounter::priority_probe_info(Priority pr) noexcept {
    switch (pr) {
#ifndef KAME_STM_RETRY_THRESH_NORMAL
#define KAME_STM_RETRY_THRESH_NORMAL 4   // sweep winner: 4 > 3 > 5 at AGE=20ms
#endif
        case Priority::HIGHEST:       return { 2, "HIGHEST" };
        case Priority::NORMAL:        return { KAME_STM_RETRY_THRESH_NORMAL, "NORMAL" };
        case Priority::UI_DEFERRABLE: return { 4, "UI_DEFERRABLE" };
        case Priority::LOWEST:        return { 4, "LOWEST" };
        default:                      return { 3, "?" };
    }
}

template <class XN>
bool Node<XN>::NegotiationCounter::livelock_probe_tx_tick(
    const void *linkage,
    uint32_t my_tx_retries,
    uint64_t tx_commit_count,
    int tags_owned,
    int tags_total,
    int64_t tx_age_us,
    Priority prio) noexcept
{
    auto &p = detail::tls_livelock_probe;
    if (p.linkage_id != linkage) {
        p.linkage_id       = linkage;
        p.t_window_us      = detail::ll_now_us();
        p.tx_retry_window  = my_tx_retries;
        p.tx_commit_window = tx_commit_count;
        return false;
    }
    int64_t now_us    = detail::ll_now_us();
    int64_t window_us = now_us - p.t_window_us;

    // m_tx_retry_count restarts at 0 when a new Transaction ctor fires;
    // handle wrap-to-smaller-value by treating delta as the current value.
    uint32_t my_retry_delta = my_tx_retries >= p.tx_retry_window
                            ? my_tx_retries - p.tx_retry_window
                            : my_tx_retries;
    uint64_t cmt_delta      = tx_commit_count - p.tx_commit_window;

    double elapsed_sec     = window_us * 1e-6;
    double my_retry_rate   = my_retry_delta / elapsed_sec;
    double tx_commit_rate  = cmt_delta       / elapsed_sec;
    double ratio           = my_retry_rate /
                             std::max(1.0, tx_commit_rate);

    const auto pinfo = priority_probe_info(prio);

    const char *verdict =
        (tags_total > 0 && tags_owned == tags_total
         && (int)my_tx_retries >= pinfo.retry_threshold
         && tx_age_us > min_privilege_age_us(prio))
            ? "LIVELOCK" : "ok";

    if(window_us > 100'000)
        if(verdict[0] == 'L')
            std::fprintf(stderr,
                "[ll-probe] tid=%u linkage=%p prio=%s threshold=%d "
                "my_tx_retries=%u my_tx_retry_rate=%.0f/s "
                "tx_commit_rate=%.0f/s ratio=%.1f "
                "tags_owned=%d/%d tx_age_us=%lld "
                "verdict=%s window_ms=%lld\n",
                (unsigned)ProcessCounter::id(), linkage,
                pinfo.name, pinfo.retry_threshold,
                (unsigned)my_tx_retries, my_retry_rate, tx_commit_rate,
                ratio, tags_owned, tags_total,
                (long long)(tx_age_us), verdict,
                (long long)(window_us / 1'000));

    bool saw_livelock = (verdict[0] == 'L');

    p.t_window_us      = now_us;
    p.tx_retry_window  = my_tx_retries;
    p.tx_commit_window = tx_commit_count;
    return saw_livelock;
}

template <class XN>
void Node<XN>::NegotiationCounter::negotiate_sleep(int ms_timeout) noexcept {
    int slot = (int)((unsigned)ProcessCounter::id() % NEGOTIATE_SLEEP_SLOTS);
    auto &st = s_sleep_slots[slot];
    std::unique_lock<std::mutex> lock(st.mtx);
    // Reset under the lock so a notify delivered between the previous
    // call's wake and this reset is not silently consumed.
    st.notified = false;
    st.cv.wait_for(lock, std::chrono::milliseconds(ms_timeout),
                   [&]{ return st.notified; });
}

template <class XN>
template <int WORDS>
void Node<XN>::NegotiationCounter::notify_n_contenders(
    const uint64_t (&tid_bitset)[WORDS], int n) noexcept
{
    // Fair-mode escape: if a privileged TID is registered, wake its
    // sleep slot first so the stuck oldest Tx gets a chance to retry
    // ahead of the rest of the bitset.
    uint16_t priv_tid = detail::stamp_tid(
        s_privileged_tidstamp.load(std::memory_order_relaxed));
    int priv_slot = -1;
    if (priv_tid != 0 && n > 0) {
        priv_slot = (int)(((unsigned)priv_tid) % NEGOTIATE_SLEEP_SLOTS);
        auto &st = s_sleep_slots[priv_slot];
        { std::lock_guard<std::mutex> lk(st.mtx); st.notified = true; }
        st.cv.notify_one();
        --n;
    }
    for(int i = 0; i < WORDS && n > 0; ++i) {
        uint64_t word = tid_bitset[i];
        while(word && n > 0) {
            int bit = __builtin_ctzll(word);
            word &= word - 1;
            int slot = (int)(((unsigned)(i * 64 + bit)) % NEGOTIATE_SLEEP_SLOTS);
            if (slot == priv_slot) continue;
            auto &st = s_sleep_slots[slot];
            { std::lock_guard<std::mutex> lk(st.mtx); st.notified = true; }
            st.cv.notify_one();
            --n;
        }
    }
}

template <class XN>
template <int WORDS>
void Node<XN>::NegotiationCounter::try_notify_n_contenders(
    const uint64_t (&tid_bitset)[WORDS], int n) noexcept
{
    for(int i = 0; i < WORDS && n > 0; ++i) {
        uint64_t word = tid_bitset[i];
        while(word && n > 0) {
            int bit = __builtin_ctzll(word);
            word &= word - 1;
            int slot = (int)(((unsigned)(i * 64 + bit)) % NEGOTIATE_SLEEP_SLOTS);
            auto &st = s_sleep_slots[slot];
            std::unique_lock<std::mutex> lk(st.mtx, std::try_to_lock);
            if( !lk.owns_lock()) continue;
            st.notified = true;
            lk.unlock();
            st.cv.notify_one();
            --n;
        }
    }
}

#if (KAME_STM_MIN_RUNNERS != 0) || (KAME_STM_MAX_RUNNERS != 0)
    // Running maximum of observed C (contender count), used as fallback when
    // hardware_concurrency() returns 0.
    alignas(KAME_CACHE_LINE) std::atomic<int> s_max_c_obs{1};

    // Update max C_obs (relaxed: approximate max is fine)
    inline int effective_runners(int c_obs) noexcept {
        int prev = s_max_c_obs.load(std::memory_order_relaxed);
        while(c_obs > prev &&
              !s_max_c_obs.compare_exchange_weak(prev, c_obs,
                  std::memory_order_relaxed, std::memory_order_relaxed))
            {}
        static const int hw = (int)std::thread::hardware_concurrency();
        if(hw > 0) return std::max(1, hw);
        return std::max(1, s_max_c_obs.load(std::memory_order_relaxed));
    }
    // Effective MIN_RUNNERS threshold, computed once (hardware_concurrency is
    // fixed at runtime; s_max_c_obs is updated each call as a side effect).
    inline int effective_min_runners(int c_obs) noexcept {
#if KAME_STM_MIN_RUNNERS > 0
        return KAME_STM_MIN_RUNNERS;
#endif // auto (-1)
        return effective_runners(c_obs) / 1;
    }
    inline int effective_max_runners(int c_obs) noexcept {
#if KAME_STM_MAX_RUNNERS > 0
        return KAME_STM_MAX_RUNNERS;
#endif // auto (-1)
        return effective_runners(c_obs) / 1;
    }
#endif // KAME_STM_MIN_RUNNERS != 0 || KAME_STM_MAX_RUNNERS != 0

// Unified retry-loop backoff: always call retry_pause + negotiate.
// retry==0 → fast-path return UNLESS another Tx currently holds the
// fair-mode privileged slot. The yield is part of the livelock-free
// guarantee: when a stuck Tx claims privilege, all other Txs must
// release their CAS pressure so the privileged commit can succeed.
// retry>0 always runs retry_pause + negotiate.
template <class XN>
void
Node<XN>::Linkage::negotiate_after_retry_pause(
    int retry,
    Snapshot<XN> &snap,
    float mult_wait) noexcept {
    if(retry == 0
        && !NegotiationCounter::fair_mode_blocks_me(snap.m_started_time))
        [[likely]] return;  // fast path; zero-overhead steady state
    retry_pause(retry);
    negotiate(snap, mult_wait);
}

template <class XN>
XThreadLocal<typename Node<XN>::FuncPayloadCreator> Node<XN>::stl_funcPayloadCreator;

template <class XN>
XThreadLocal<typename Node<XN>::SerialGenerator::cnt_t> Node<XN>::SerialGenerator::stl_serial;

atomic<ProcessCounter::cnt_t> ProcessCounter::s_count = ProcessCounter::MAINTHREADID - 1;
XThreadLocal<ProcessCounter> ProcessCounter::stl_processID;

ProcessCounter::ProcessCounter() {
    for(;;) {
        cnt_t oldv = s_count;
        cnt_t newv = oldv + (cnt_t)1u;
        if( !newv) ++newv;
        if(s_count.compare_set_strong(oldv, newv)) {
            //avoids zero.
//            fprintf(stderr, "Assigning a new process ID=%d\n", newv);
            m_var = newv;
            break;
        }
    }
}
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
// Non-inline definition so the body compiles into kame.dll and is imported by
// plugin DLLs — ensuring *stl_processID is always kame.dll's TLS slot.
ProcessCounter::cnt_t ProcessCounter::id() noexcept { return *stl_processID; }
#endif

XThreadLocal<Priority> stl_currentPriority;

Priority getCurrentPriorityMode() {
    return *stl_currentPriority;
}

template <class XN>
void
Node<XN>::Packet::print_() const {
    printf("Packet: ");
    printf("%s@%p, ", typeid(*this).name(), &node());
    printf("BP@%p, ", node().m_link.get());
    if(missing())
        printf("missing, ");
    if(size()) {
        printf("%d subnodes : [ \n", (int)size());
        for(int i = 0; i < size(); i++) {
            if(subpackets()->at(i)) {
                subpackets()->at(i)->print_();
                printf("; ");
            }
            else {
                printf("%s@%p, w/o packet, ", typeid(*this).name(), subnodes()->at(i).get());
            }
        }
        printf("]\n");
    }
    printf(";");
}

template <class XN>
bool
Node<XN>::Packet::checkConsistensy(const local_shared_ptr<Packet> &rootpacket) const {
    try {
        if(size()) {
            if( !(payload()->m_serial - subpackets()->m_serial < 0x7fffffffffffffffLL))
                throw __LINE__;
        }
        for(int i = 0; i < size(); i++) {
            if( !subpackets()->at(i)) {
                if( !rootpacket->missing()) {
                    if( !subnodes()->at(i)->reverseLookup(
                        const_cast<local_shared_ptr<Packet>&>(rootpacket), false, 0, false, 0))
                        throw __LINE__;
                }
            }
            else {
                if(subpackets()->at(i)->size())
                    if( !(subpackets()->m_serial - subpackets()->at(i)->subpackets()->m_serial < 0x7fffffffffffffffLL))
                        throw __LINE__;
                if(subpackets()->at(i)->missing() && (rootpacket.get() != this)) {
                    if( !missing())
                        throw __LINE__;
                }
                if( !subpackets()->at(i)->checkConsistensy(
                    subpackets()->at(i)->missing() ? rootpacket : subpackets()->at(i)))
                    return false;
            }
        }
    }
    catch (int &line) {
        fprintf(stderr, "Line %d, losing consistensy on node %p:\n", line, &node());
        rootpacket->print_();
        throw *this;
    }
    return true;
}

template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const local_shared_ptr<Packet> &x, int64_t bundle_serial) noexcept :
    m_bundledBy(), m_packet(x), m_reverse_index((int)PACKET_STATE::PACKET_HAS_PRIORITY),
    m_bundle_serial(bundle_serial) {
}
template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const shared_ptr<Linkage > &bp, int reverse_index,
    int64_t bundle_serial) noexcept :
    m_bundledBy(bp), m_packet(), m_reverse_index(),
    m_bundle_serial(bundle_serial) {
    setReverseIndex(reverse_index);
}
template <class XN>
Node<XN>::PacketWrapper::PacketWrapper(const PacketWrapper &x, int64_t bundle_serial) noexcept :
    m_bundledBy(x.m_bundledBy), m_packet(x.m_packet),
    m_reverse_index(x.m_reverse_index), m_bundle_serial(bundle_serial) {}

template <class XN>
void
Node<XN>::PacketWrapper::print_() const {
    printf("PacketWrapper: ");
    if( !hasPriority()) {
        printf("referred to BP@%p, ", bundledBy().get());
    }
    printf("serial:%lld, ", (long long)m_bundle_serial);
    if(packet()) {
        packet()->print_();
    }
    else {
        printf("absent, ");
    }
    printf("\n");
}

template <class XN>
void
Node<XN>::print_recoverable_error(const char* reason) {
#ifdef gErrPrint
    try {
    char buf[256] = {};
        snprintf(buf, sizeof(buf), "Out of memory!: %s\nClose unnescessary windows & Store your data immediately.\n", reason);
        gErrPrint(buf);
    }
    catch (const std::bad_alloc &) {
#else
    {
#endif
        fprintf(stderr, "Memory allocation has failed: %s\nTransaction is delaying...\n", reason);
    }
    msecsleep(1000);
}


// Adaptive-lease constants. The active lease window (us) is stored per-Linkage
// as Linkage::m_adapt_lease_us, so contention sites converge independently
// (hot Linkage → longer lease; cold Linkage → short).
// Clamped to [KAME_LEASE_NS_MIN, KAME_LEASE_NS_MAX] (default 1 µs..2 ms).
// Fixed-threshold drift is used instead of proportional-rate variants
// because the C distribution is heavily skewed toward low C; keeping
// C == 1 neutral avoids dragging the lease down during the low-C phase
// that dominates total call count.
#ifndef KAME_LEASE_US_MIN
#define KAME_LEASE_US_MIN  1     // 1 µs
#endif
#ifndef KAME_LEASE_US_MAX
#define KAME_LEASE_US_MAX  20   // 20 µs — uint16_t field; keep ≤65535. Sweep winner.
#endif


// Optional diagnostic counters (opt-in via -DKAME_ADAPT_INSTRUMENT=1).
// Inspect with gdb while a test runs: `thread apply all print <name>`.
// Off by default to keep per-call overhead minimal in production builds.
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
// dt2 of the most recent negotiate() call — used by the adaptive fairness
// gate (always-on, zero cost beyond one thread_local store).
// Non-inline: see runner-counter detail block above for why
// transaction_impl.h drops `inline` on namespace-scope thread_local.
thread_local uint64_t s_adapt_dt2_last_us       = 0;
thread_local int      s_adapt_C_last            = 0;  // popcount(tid_bitset)
thread_local uint32_t s_adapt_last_priority_tid = 0;  // last m_priority_tid seen
thread_local uint32_t s_adapt_bounce_count      = 0;  // # times it changed
thread_local uint64_t s_adapt_negotiate_calls   = 0;  // negotiate() entries
thread_local uint64_t s_adapt_skip_hits         = 0;  // lease-skip fires
thread_local uint32_t s_adapt_skip_per1k        = 0;  // skip_hits/calls × 1000
#endif
//=============================================================================
// negotiate_internal() — priority-based backoff for collision avoidance
//
// Purpose: when two transactions contend on the same Linkage, impose a
//   proportional wait on the lower-priority (or younger) transaction so
//   that the older/higher-priority one can finish first, preventing live-lock.
//
// Priority/proportional semantics (unchanged):
//   - dt  = (this thread's start time) − (contending thread's start time)
//   - dt2 = (wall-clock now) − (contending thread's start time)
//   - mult_wait * 2 * dt < dt2 → contender has run long enough; proceed anyway
//   - HIGHEST bypasses; LOWEST never escapes early
//   - Nominal sleep = max(dt2/10000, prev_ms + 1) [ms], capped at 5 s
//
// Adaptive jitter (Anderson 1990; Herlihy & Shavit 2008 ch.7;
//   Bianchi 2000 IEEE 802.11 √N damping; Brooker AWS 2015 decorrelated):
//   The sleep is drawn uniformly from [ms/√C, ms*√C] (capped at 5 s), where
//   C = popcount(tid_bitset) = number of distinct committer ProcessCounter::id
//   values observed at all linkages touched by the current transaction so far.
//     C=1  → √C=1  → no jitter  (sleep = ms)
//     C=4  → √C=2  → range [ms/2, 2*ms]
//     C=16 → √C=4  → range [ms/4, 4*ms]
//     C=128→ √C=11 → range [ms/11, 11*ms]
//   C=1 (no observed contention) stays deterministic to avoid range inflation
//   in low-contention paths; C>1 fans out proportionally to live contenders
//   to break lock-step retry cycles (the livelock root cause on strong-memory
//   x86, Darwin x86_64, >=32 threads).
//
// Bitset ownership:
//   The caller passes a reference to its per-transaction bitset
//   (Transaction::m_tid_bitset; stack-local for Snapshot-only paths). This
//   avoids TLS and makes nested transactions observe their own scope
//   naturally. No CAS or peek of the linkage is performed inside the loop.
//=============================================================================
template <class XN>
void
Node<XN>::Linkage::negotiate_internal(Snapshot<XN> &snap,
                                      float mult_wait) noexcept {
    auto &started_time = snap.m_started_time;
    auto &tid_bitset = snap.m_tid_bitset;
    // Single now_us() snapshot: livelock-probe window, livelock age,
    // adaptive-lease dt2 and owner-skip lease age all read it. The few
    // µs between these reads in the original code carried no useful
    // information (no observable state changes between them).
    const int64_t now_us_entry = Node<XN>::NegotiationCounter::now_us();
    // Priority is a per-thread/per-Tx invariant for the duration of this
    // call: read it once and reuse for both the livelock-probe block and
    // the adaptive-lease block below.
    const Priority entry_pr = getCurrentPriorityMode();

    // Compute popcount once per call; the live tid_bitset is unchanged
    // until the loop body's first iteration adds new entries.
    int sig_C = 0;
    for(int i = 0; i < TID_BITSET_WORDS; ++i)
        sig_C += popcount_u64(tid_bitset[i]);
    // No pre-loop yield: the m_transaction_started_time load below is
    // the cheap collision-clear check.
    // tx age = wall time since the Snapshot/Transaction ctor stamped
    // m_started_time. The field is set by BOTH Snapshot(const Node&)
    // and Transaction ctors and is not reset by operator++ — so the
    // probe's `tx_age_us` printout is really "Snapshot/Tx age". The
    // `tx_` label is kept for log-format continuity.
    // m_started_time is tid-packed; unpack the µs component before
    // subtracting the raw-µs now_us() value.
    int64_t _ll_age_us = now_us_entry
                     - NegotiationCounter::stamp_us(started_time);
    if (_ll_age_us >= NegotiationCounter::min_privilege_age_us(Priority::HIGHEST)
        && !snap.m_tagged_linkages.empty()) {  // skip when too young or untagged
        // Count tagged linkages whose m_transaction_started_time == ours
        // (= "priority is already mine on every linkage" = primary
        //   livelock precondition per the refined definition).
        int _ll_total = (int)snap.m_tagged_linkages.size();
        int _ll_owned = 0;
        for (auto &_l : snap.m_tagged_linkages) {
            if (_l && _l->m_transaction_started_time.load(
                    std::memory_order_relaxed) == snap.m_started_time)
                ++_ll_owned;
        }
        // `entry_pr` was read once at function entry; the probe maps it
        // to retry-threshold / label internally.
        bool _ll_saw = NegotiationCounter::livelock_probe_tx_tick(
            static_cast<const void*>(this),
            snap.m_tx_retry_count,
            m_tx_commit_count,
            _ll_owned, _ll_total, _ll_age_us,
            entry_pr);
        // Fair-mode escape: when verdict=LIVELOCK fires for this Tx, the
        // global slot is free, and the Tx has aged past the per-priority
        // floor (see NegotiationCounter::min_privilege_age_us), claim it.
        // Subsequent LIVELOCK ticks on the same Tx are no-ops because
        // m_registered_privileged is already set. Cleared in
        // finalizeCommitment / ~Transaction.
        if (_ll_saw && !snap.m_registered_privileged
            && NegotiationCounter::try_register_privileged_tidstamp(
                   entry_pr, snap.m_started_time)) {
            snap.m_registered_privileged = true;
        }
    }

    // Always-on adaptive path: the V0 (legacy) path and the V0↔ADAPTIVE
    // mode switch were removed in favour of the orthogonal fair-mode
    // escape (s_privileged_tidstamp). See top of detail:: in this file.
  { // adaptive-path scope
    // One atomic load of the packed (tid | lease_us | start_us) tuple.
    auto ps = loadPriority();
    if(ps.tid) {
        unsigned tid = (unsigned)ps.tid & (unsigned)(TID_BITSET_WORDS * 64 - 1);
        tid_bitset[tid >> 6] |= 1ULL << (tid & 63);
    }
    typename NegotiationCounter::cnt_t transaction_started_time =
        m_transaction_started_time.load(std::memory_order_relaxed);
    if( !transaction_started_time)
        return; //collision has not been detected.
    // LOWEST and UI_DEFERRABLE explicitly tolerate yielding, so skip
    // the adaptive-lease block entirely (priority-tag CAS, lease
    // tracking, fairness gate, owner-skip). The main sleep loop
    // below is plenty for these priorities.
    // `entry_pr` is computed once at the top of negotiate_internal.
#ifdef KAME_PRIORITY_LEASE
    if(entry_pr != Priority::LOWEST && entry_pr != Priority::UI_DEFERRABLE) {
    // transaction_started_time is tid-packed; unpack before diffing
    // against the raw-µs now_us() clock.
    auto adapt_dt2_last_us =
        (typename NegotiationCounter::cnt_t)
        (now_us_entry
         - Node<XN>::NegotiationCounter::stamp_us(transaction_started_time));

#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
    s_adapt_dt2_last_us = adapt_dt2_last_us;
    s_adapt_C_last = sig_C;
    if(ps.tid && ps.tid != s_adapt_last_priority_tid) {
        ++s_adapt_bounce_count;
        s_adapt_last_priority_tid = ps.tid;
    }
    ++s_adapt_negotiate_calls;
    s_adapt_skip_per1k = s_adapt_negotiate_calls > 0
                             ? (uint32_t)((uint64_t)s_adapt_skip_hits * 1000
                                           / s_adapt_negotiate_calls)
                             : 0;
#endif

    // Adaptive lease tracking (per-Linkage). Drift the lease_us field
    // and write back via storePriority; relaxed races benignly because
    // any value in [MIN,MAX] is a valid lease. Only touch the atomic
    // if the value actually changes. Schedule constants live at file
    // top (KAME_LEASE_*).
    static constexpr uint16_t LEASE_US_MIN =
        (uint16_t)(KAME_LEASE_US_MIN ? KAME_LEASE_US_MIN : 1);
    static constexpr uint16_t LEASE_US_MAX =
        (uint16_t)(KAME_LEASE_US_MAX);
    uint16_t new_lease_us = ps.lease_us;
    if(sig_C >= 2) {
        int grow = (sig_C - 1) * (int)KAME_LEASE_GROW_PER_C_PERCENT;
        if(grow > (int)KAME_LEASE_GROW_MAX_PERCENT)
            grow = (int)KAME_LEASE_GROW_MAX_PERCENT;
        uint32_t next = (uint32_t)ps.lease_us
                        * (uint32_t)(100 + grow) / 100;
        if(next > LEASE_US_MAX) next = LEASE_US_MAX;
        new_lease_us = (uint16_t)next;
    } else if(sig_C == 0) {
        uint32_t next = (uint32_t)ps.lease_us
                        * (uint32_t)(100 - KAME_LEASE_SHRINK_PERCENT) / 100;
        if(next < LEASE_US_MIN) next = LEASE_US_MIN;
        new_lease_us = (uint16_t)next;
    }
    int delta = (int)new_lease_us - (int)ps.lease_us;
    if(delta >= (int)KAME_LEASE_QWRITE_US || delta <= -(int)KAME_LEASE_QWRITE_US) {
        PriorityState drifted = ps;
        drifted.lease_us = new_lease_us;
        storePriority(drifted);
        ps.lease_us = new_lease_us;
    }

    // Adaptive gate: suppress owner-skip when dt2 exceeds
    // KAME_DT2_FAIRNESS_US (long-held competing tx → starvation risk).
    unsigned my_tid = ProcessCounter::id() & 0xFFFFu;
#if KAME_STM_MIN_RUNNERS != 0
    const int min_r_pre = effective_min_runners(1);
    if(NegotiationCounter::numThreadsRunning() < min_r_pre)
#endif
    if(my_tid == ps.tid
        && adapt_dt2_last_us < (uint64_t)KAME_DT2_FAIRNESS_US) {
        // Age in µs via modular 32-bit subtraction (wrap-safe up to ~35 min).
        uint32_t age_us = (uint32_t)now_us_entry - ps.start_us;
        if(age_us < (uint32_t)ps.lease_us) {
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
            ++s_adapt_skip_hits;
#endif
            if(entry_pr == Priority::HIGHEST || entry_pr == Priority::NORMAL)
                return; //skips
        }
    }
    } // end lease-block
#endif

    // Thread-local LCG for sleep-duration jitter randomization.
    // Seed mixes thread ID (unique per thread) with stack address; Murmur finalizer
    // avalanches all bits so threads with adjacent stack addresses (8 MB spacing on
    // macOS) get unrelated seeds — preventing correlated jitter and synchronized wakeups.
    static thread_local uint32_t s_backoff_seed = [&]{
        uint32_t h = (uint32_t)ProcessCounter::id() * 2654435761u
                   ^ (uint32_t)(uintptr_t)&started_time;
        h ^= h >> 16; h *= 0x85ebca6bu;
        h ^= h >> 13; h *= 0xc2b2ae35u;
        h ^= h >> 16;
        return h ? h : 1u;
    }();

    // Live-contention estimate. sig_C is the popcount taken at function
    // entry; tid_bitset accumulates across retries within this Tx, but
    // not within a single negotiate_internal call — re-popcount inside
    // the loop would yield the same value, so we reuse sig_C.
    //
    // Floor at KAME_STM_C_OBS_MIN (default 2): with C_obs=1 the
    // √C lottery threshold becomes ~1.0 (always-fire), causing
    // unnecessary wake-broadcast overhead even when the workload is
    // really just 2 threads alternating. Treating C=1 as C=2 for
    // formula purposes lets the lottery fire at 50% per iteration
    // (= the natural rate for the 2-thread case) without inflating
    // contender count anywhere else.
#ifndef KAME_STM_C_OBS_MIN
#define KAME_STM_C_OBS_MIN 2
#endif
    int C_obs = sig_C < KAME_STM_C_OBS_MIN ? KAME_STM_C_OBS_MIN : sig_C;

    for(int ms = 0;;) {
        if(entry_pr == Priority::HIGHEST)
            break;
        // Single-contender fast path: only this thread is visible in
        // tid_bitset (sig_C=1). The probabilistic √C lottery is
        // meaningless when there is no peer to share the slot with —
        // every iteration would just roll a coin and (eventually)
        // wake/break. Skip the gate and lottery entirely, send one
        // notify (in case any sleeper was waiting on this linkage),
        // and break out so the caller can retry CAS. Greedy CM
        // resolves any concurrent commit by the older Tx; if a real
        // contender appears later, tid_bitset accumulates and the
        // next negotiate call sees sig_C ≥ 2.
        if(sig_C == 1) {
            NegotiationCounter::notify_n_contenders(tid_bitset, 1);
            break;
        }
        // Both stamps are tid-packed; subtract their µs components.
        auto dt = NegotiationCounter::stamp_us(started_time)
                - NegotiationCounter::stamp_us(transaction_started_time);
        if(dt <= 0)
            break; //This thread is the oldest.
        auto transaction_started_time =
            m_transaction_started_time.load(std::memory_order_acquire);
        if( !transaction_started_time)
            break; //collision has not been detected.

        auto dt2 = Node<XN>::NegotiationCounter::now_us()
                 - NegotiationCounter::stamp_us(transaction_started_time);

        // Fair-mode escape: when some other thread holds the privileged-
        // TID slot, suppress the jittered gate and the √C lottery so the
        // privileged Tx alone gets to commit. Strict Greedy CM (older
        // started_time wins → I sleep below) is the only mechanism left
        // to allocate priority while fair-mode is active.
        const bool _fair_blocks = NegotiationCounter::fair_mode_blocks_me(started_time);
        if(entry_pr != Priority::LOWEST && dt > 0 && !_fair_blocks) {
            // Single LCG advance per iteration; bits 16-31 → r_j (jitter),
            // bits 0-15 → r_l (lottery). Independent windows of one PRNG
            // sample are sufficient and save one multiply+add per loop.
            s_backoff_seed = s_backoff_seed * 1103515245u + 12345u;
            uint32_t r_j = (s_backoff_seed >> 16) & 0xFFFFu;
            uint32_t r_l =  s_backoff_seed        & 0xFFFFu;
            // (a) Jittered gate: break early when the waiting time justifies it.
            //     LHS = mult_wait * 2 * dt * J, RHS = dt2.  J ∈ [1-R/100, 1+R/100]
            //     with R = KAME_STM_JITTER_RANGE.  Fixed-point: multiply both sides
            //     by 65536; J encoded as (LO + r_j / DIV) where LO = (100-R)*65536/100.
            enum {
                JITTER_LO  = (100 - KAME_STM_JITTER_RANGE) * 65536 / 100,
                JITTER_DIV = 100 / (2 * KAME_STM_JITTER_RANGE)
            };
#if KAME_STM_DISABLE_JITTER
            // Ablation: gate factor pinned at J = 1.0 (LO mid-point = 65536).
            (void)r_j;
            uint64_t lhs_j = (uint64_t)(mult_wait * KAME_STM_GATE_MULT * (double)dt)
                           * (uint64_t)65536u;
#else
            uint64_t lhs_j = (uint64_t)(mult_wait * KAME_STM_GATE_MULT * (double)dt)
                           * (uint64_t)(JITTER_LO + r_j / JITTER_DIV);
#endif
            uint64_t rhs_j = (uint64_t)dt2 * 65536u;
            if((KAME_STM_GATE_MULT > 0.0f) && (lhs_j < rhs_j)) {
#if KAME_STM_MAX_RUNNERS != 0
                const int max_r = effective_max_runners(C_obs);
                if(NegotiationCounter::numThreadsRunning() < max_r)
#endif
                    break; // gate: earned priority — always proceeds, never capped
            }
#ifndef KAME_STM_DISABLE_LOTTERY
#define KAME_STM_DISABLE_LOTTERY 0
#endif
#if !KAME_STM_DISABLE_LOTTERY
            // (b) C fairness lottery: LOTTERY_MULT*C threads bypass per iteration.
            //     Prevents all threads from being stuck in the gate simultaneously.
#if KAME_STM_MIN_RUNNERS != 0
            const int min_r_lot = effective_min_runners(C_obs);
            if(NegotiationCounter::numThreadsRunning() < min_r_lot) {
#else
            if(C_obs > 1) {
#endif
                uint64_t t64 = (uint64_t)KAME_STM_LOTTERY_MULT * 0x10000u / (uint32_t)C_obs;
                uint32_t threshold = (t64 >= 0xFFFFu) ? 0xFFFFu : (uint32_t)t64;
                if(r_l < threshold) {
                    // Lottery firing at the wake-broadcast point. Default:
                    // blocking lock_guard for reliable wakes. Rebuild with
                    // -DKAME_STM_NOTIFY_TRY_LOCK=1 to select the try_lock
                    // skip variant for ablation / regression measurement.
#if defined(KAME_STM_NOTIFY_TRY_LOCK) && KAME_STM_NOTIFY_TRY_LOCK
                    NegotiationCounter::try_notify_n_contenders(tid_bitset, C_obs);
#else
                    NegotiationCounter::notify_n_contenders(tid_bitset, C_obs);
#endif
                    break;
                }
            }
#endif
        }

        ms = std::max((int)(dt2 * mult_wait / 10000),  ms + 1);
        if(ms > 5000) {
            fprintf(stderr, "Nested transaction?, ");
            fprintf(stderr, "Negotiating, %f sec. requested, limited to 5s.", ms*1e-3);
            fprintf(stderr, "for BP@%p\n", this);
            ms = 5000;
        }

        // Sleep ms in 1-ms CV chunks + random ±1ms de-phasing jitter.
        // Jitter breaks the synchronized-wakeup oscillation that forms when
        // all threads enter and exit negotiate_sleep at the same 1 ms tick.
        //
        // Low-contention shortcut: at numThreadsRunning() ≤ 2 the
        // privileged-TID escape cannot fire (age-spread between
        // 2 contenders stays µs-scale, well below
        // min_privilege_age_us), so the standard 1 ms CV sleep
        // chunk becomes the throughput ceiling. Replace it with
        // std::this_thread::yield() so Greedy CM (older Tx wins)
        // drives a tight CAS-retry alternation. Yield (not bare
        // break) is essential — bare break leaves the same thread
        // hot-spinning the CAS, which loses the alternation; yield
        // gives the OS scheduler the opportunity to swap to the
        // other contender, allowing it to commit cleanly.
        if(NegotiationCounter::numThreadsRunning() <= 2 && ms <= 1) {
            typename NegotiationCounter::ReleaseOneCount onedown;
            std::this_thread::yield();
        }
        else {
            int ms_actual = ms;
            typename NegotiationCounter::ReleaseOneCount onedown;
#if KAME_STM_MIN_RUNNERS != 0
            // Sleep in 1 ms chunks so the MIN_RUNNERS check fires after this
            // thread has registered in s_negotiate_sleepers (i.e. is visible
            // as a sleeper). Each chunk is interruptible by
            // notify_n_contenders, so effective latency is well below 1 ms
            // once a lottery winner fires.
            const int min_r = effective_min_runners(C_obs);
            auto t_end = Node<XN>::NegotiationCounter::now_us()
                         + (int64_t)ms_actual * 1000;
            do {
                // Advance seed for de-phasing; chunk sleep = 1 or 2 ms.
                s_backoff_seed = s_backoff_seed * 1103515245u + 12345u;
                int running = (int)NegotiationCounter::numThreadsRunning();
                if(running < min_r)
                    NegotiationCounter::notify_n_contenders(tid_bitset,
                        std::min(min_r - running, C_obs));
#if KAME_STM_DISABLE_JITTER
                NegotiationCounter::negotiate_sleep(1);
#else
                NegotiationCounter::negotiate_sleep(1 + (int)(s_backoff_seed >> 31));
#endif
            } while(Node<XN>::NegotiationCounter::now_us() < t_end);
#else
            NegotiationCounter::negotiate_sleep(ms_actual);
#endif
        }
    }
  } // end adaptive-path scope
}

template <class XN>
Node<XN>::Node() : m_link(std::make_shared<Linkage>()) {
    local_shared_ptr<Packet> packet(new Packet());
    m_link->reset(new PacketWrapper(packet, SerialGenerator::gen()));
    //Use create() for this hack.
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
    // Read and clear the shared void* slot — see create() for why we use
    // this instead of stl_funcPayloadCreator on Windows.
    auto creator = reinterpret_cast<FuncPayloadCreator>(detail::tls_payload_creator_ptr());
    detail::tls_payload_creator_ptr() = nullptr;
#else
    auto creator = *stl_funcPayloadCreator;
    *stl_funcPayloadCreator = nullptr;
#endif
    packet->m_payload.reset(creator(static_cast<XN&>( *this)));
}
template <class XN>
Node<XN>::~Node() {
    releaseAll();
}
template <class XN>
void
Node<XN>::print_() const {
    local_shared_ptr<PacketWrapper> packet( *m_link);
//	printf("Node:%p, ", this);
//	printf("BP:%p, ", m_link.get());
//	printf(" packet: ");
    packet->print_();
}

template <class XN>
void
Node<XN>::insert(const shared_ptr<XN> &var) {
    iterate_commit_if([this, var](Transaction<XN> &tr)->bool {
        return insert(tr, var);
    });
}
//=============================================================================
// insert() — add a child node to the tree within a transaction
//   (Comments by Claude Opus — based on source code analysis)
//
// When online_after_insertion is true, the child is committed to the live tree
// immediately so that tr[*child] is accessible within the same transaction.
// This requires an intermediate commit of the parent, followed by a CAS on
// the child's Linkage to point it into the parent's packet.
// Without online_after_insertion, the child becomes visible only after the
// enclosing transaction commits.
//=============================================================================
template <class XN>
bool
Node<XN>::insert(Transaction<XN> &tr, const shared_ptr<XN> &var, bool online_after_insertion) {
    local_shared_ptr<Packet> packet = reverseLookup(tr.m_packet, true, tr.m_serial, true);
    packet->subpackets() = packet->size() ? std::make_shared<PacketList>( *packet->subpackets()) : std::make_shared<PacketList>();
    packet->subpackets()->m_serial = tr.m_serial;
    packet->m_missing = true;
    packet->subnodes() = packet->size() ? std::make_shared<NodeList>( *packet->subnodes()) : std::make_shared<NodeList>();
//    if( !packet->subpackets()->size()) {
//        packet->subpackets()->reserve(4);
//        packet->subnodes()->reserve(4);
//    }
    packet->subpackets()->resize(packet->size() + 1);
    assert(std::find(packet->subnodes()->begin(), packet->subnodes()->end(), var) == packet->subnodes()->end());
    packet->subnodes()->resize(packet->subpackets()->size());
    packet->subnodes()->back() = var;

    if(online_after_insertion) {
        bool has_failed = false;
        //Tags serial.
        local_shared_ptr<Packet> newpacket(tr.m_packet);
        tr.m_packet.reset(new Packet( *tr.m_oldpacket));
        if( !tr.m_packet->node().commit(tr)) {
            printf("*\n");
            has_failed = true;
        }
        tr.m_oldpacket = tr.m_packet;
        tr.m_packet = newpacket;
        for(int iter = 0;; ++iter) {
            // RAII OnExit: tags only on scope exit (continue / return)
            // unless commit() is called. iter==0 successful CAS stays
            // untagged (cheap fast path).
            ScopedNegotiateLinkage<XN> scope(var->m_link, tr, iter,
                ScopedNegotiateLinkage<XN>::TagMode::OnExit);
            if( !scope) continue;  // weak acquire lost — treat as CAS failure

            local_shared_ptr<Packet> subpacket_new;
            // Reuse scope's view instead of a fresh load_shared_ on the
            // same var->m_link.  view_copy is one fetch_add (Owned) or
            // 2-3 ops (TagHeld → promote); fresh load is ~2 ops via
            // tag-bit acquire CAS plus fetch_add.  Slightly older
            // value on contention is fine — the CAS at scope.compareAndSetWithHint
            // below will detect any drift and retry.
            local_shared_ptr<PacketWrapper> subwrapper = scope.view_copy();
            BundledStatus status = bundle_subpacket(0, var, subwrapper, subpacket_new,
                tr, tr.m_serial);
            if(status != BundledStatus::SUCCESS) {
                continue;  // RAII tags on scope exit (iter > 0)
            }
            if( !subpacket_new) {
                //Inserted twice inside the package.
                scope.commit();
                break;
            }

            //Marks for writing at subnode.
            tr.m_packet.reset(new Packet( *tr.m_oldpacket));
            if( !tr.m_packet->node().commit(tr)) {
                printf("&\n");
                has_failed = true;
            }
            tr.m_oldpacket = tr.m_packet;
            tr.m_packet = newpacket;

            local_shared_ptr<PacketWrapper> newwrapper(
                new PacketWrapper(m_link, packet->size() - 1, tr.m_serial));
            newwrapper->packet() = subpacket_new;
            packet->subpackets()->back() = subpacket_new;
            if(has_failed)
                return false;
            if( !scope.compareAndSetWithHint(subwrapper, newwrapper)) {
                tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
                return false;  // RAII tags
            }
            // scope auto-committed + tagged successful_cas via the call.
            break;
        }
    }
    tr[ *this].catchEvent(var, packet->size() - 1);
    tr[ *this].listChangeEvent();
    STRICT_assert(tr.m_packet->checkConsistensy(tr.m_packet));
    return true;
//		printf("i");
}
template <class XN>
void
Node<XN>::release(const shared_ptr<XN> &var) {
    iterate_commit_if([this, var](Transaction<XN> &tr)->bool {
        return release(tr, var);
    });
}

// eraseSerials() — recursively reset all serial markers matching the given
// serial back to SERIAL_NULL. Called during release() to prevent a released
// sub-tree's stale serial from causing false collision detection in future
// transactions.
template <class XN>
void
Node<XN>::eraseSerials(local_shared_ptr<Packet> &packet, int64_t serial,
                       Snapshot<XN> &snap) {
    if(packet->size() && packet->subpackets()->m_serial == serial)
        packet->subpackets()->m_serial = SerialGenerator::SERIAL_NULL;
    if(packet->payload()->m_serial == serial)
        packet->payload()->m_serial = SerialGenerator::SERIAL_NULL;

    for(int iter = 0;; ++iter) {
        // RAII OnExit.
        ScopedNegotiateLinkage<XN> scope(packet->node().m_link, snap, iter,
            ScopedNegotiateLinkage<XN>::TagMode::OnExit);
        if( !scope) continue;  // weak acquire lost — treat as CAS failure
        // Use scope's internal m_view directly — no separate
        // scoped_atomic_view on the same linkage needed.  scope-> reads
        // through m_view; scope.compareAndSet(newwrapper) (1-arg) uses
        // m_view as the CAS oldr.
        if(scope->m_bundle_serial != serial) {
            scope.commit();
            break;
        }
        local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper( *scope, SerialGenerator::SERIAL_NULL));
        if(scope.compareAndSet(newwrapper))
            break;
        // RAII tags on continue (iter > 0)
    }
    for(int i = 0; i < packet->size(); ++i) {
        local_shared_ptr<Packet> &subpacket(( *packet->subpackets())[i]);
        if(subpacket)
            eraseSerials(subpacket, serial, snap);
    }
}

template <class XN>
void
Node<XN>::lookupFailure() const {
    fprintf(stderr, "Node not found during a lookup.\n");
    throw NodeNotFoundError("Lookup failure.");
}

template <class XN>
bool
Node<XN>::release(Transaction<XN> &tr, const shared_ptr<XN> &var) {
    local_shared_ptr<Packet> packet = reverseLookup(tr.m_packet, true, tr.m_serial, true);
    assert(packet->size());
    packet->subpackets().reset(new PacketList( *packet->subpackets()));
    packet->subpackets()->m_serial = tr.m_serial;
    packet->subnodes().reset(new NodeList( *packet->subnodes()));
    unsigned int idx = 0;
    int old_idx = -1;
    // Loaded once below into a scope at function tail; pre-CAS read
    // here uses load_shared_ since we need the value across the
    // packet-loop iterations and the final commit() call before the
    // CAS.  Move-into a scope just before the CAS to avoid an extra
    // fetch_add (the ScopedNegotiateLinkage move-in ctor reuses the
    // local_shared_ptr's +1 ref).
    local_shared_ptr<PacketWrapper> nullsubwrapper;
    local_shared_ptr<PacketWrapper> newsubwrapper;
    auto nit = packet->subnodes()->begin();
    for(auto pit = packet->subpackets()->begin(); pit != packet->subpackets()->end();) {
        assert(nit != packet->subnodes()->end());
        if(nit->get() == &*var) {
            if( *pit) {
                nullsubwrapper = *var->m_link;
                if(nullsubwrapper->hasPriority()) {
                    if(nullsubwrapper->packet() != *pit) {
                        tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
                        return false;
                    }
                }
                else {
                    shared_ptr<Linkage> bp(nullsubwrapper->bundledBy());
                    if((bp && (bp != m_link)) ||
                        ( !bp && (nullsubwrapper->packet() != *pit))) {
                        tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
                        return false;
                    }
                }
                newsubwrapper.reset(new PacketWrapper(m_link, idx, SerialGenerator::SERIAL_NULL));
                newsubwrapper->packet() = *pit;
            }
            pit = packet->subpackets()->erase(pit);
            nit = packet->subnodes()->erase(nit);
            old_idx = idx;
        }
        else {
            ++nit;
            ++pit;
            ++idx;
        }
    }
    if(old_idx < 0)
        lookupFailure();

    if( !packet->subpackets()->size()) {
        packet->subpackets().reset();
        packet->m_missing = false;
    }
    else {
//        if(packet->subpackets()->capacity() - packet->subpackets()->size() > 8) {
//            packet->subpackets()->shrink_to_fit();
//            packet->subnodes()->shrink_to_fit();
//        }
    }
    if(tr.m_packet->size()) {
        tr.m_packet->m_missing = true;
    }

    tr[ *this].releaseEvent(var, old_idx);
    tr[ *this].listChangeEvent();

    if( !newsubwrapper) {
        //Packet of the released node is held by the other point inside the tr.m_packet.
        return true;
    }

    eraseSerials(packet, tr.m_serial, tr);

    local_shared_ptr<Packet> newpacket(tr.m_packet);
    tr.m_packet = tr.m_oldpacket;
    tr.m_packet.reset(new Packet( *tr.m_packet));
    if( !tr.m_packet->node().commit(tr)) {
        tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
        tr.m_packet = newpacket;
        return false;
    }
    tr.m_oldpacket = tr.m_packet;
    tr.m_packet = newpacket;

    //Unload the packet of the released node.
    // Move-in the pre-loaded nullsubwrapper as the scope's view (the
    // +1 refcount transfers without a fresh fetch_add).
    ScopedNegotiateLinkage<XN> scope(var->m_link, tr, -1,
        std::move(nullsubwrapper),
        ScopedNegotiateLinkage<XN>::TagMode::OnExit);
    if( !scope.compareAndSetWithHint(newsubwrapper)) {
        tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
        return false; // destructor tags
    }
    // scope auto-committed + tagged successful_cas via the call.
    STRICT_assert(tr.m_packet->checkConsistensy(tr.m_packet));
    return true;
}
template <class XN>
void
Node<XN>::releaseAll() {
    iterate_commit_if([this](Transaction<XN> &tr)->bool {
        while(tr.size()) {
            if( !release(tr, tr.list()->back())) {
                return false;
            }
        }
        return true;
    });
}
template <class XN>
void
Node<XN>::swap(const shared_ptr<XN> &x, const shared_ptr<XN> &y) {
    iterate_commit_if([this, x, y](Transaction<XN> &tr)->bool {
        return swap(tr, x, y);
    });
}
template <class XN>
bool
Node<XN>::swap(Transaction<XN> &tr, const shared_ptr<XN> &x, const shared_ptr<XN> &y) {
    local_shared_ptr<Packet> packet = reverseLookup(tr.m_packet, true, tr.m_serial, true);
    packet->subpackets().reset(packet->size() ? (new PacketList( *packet->subpackets())) : (new PacketList));
    packet->subpackets()->m_serial = tr.m_serial;
    packet->m_missing = true;
    packet->subnodes().reset(packet->size() ? (new NodeList( *packet->subnodes())) : (new NodeList));
    unsigned int idx = 0;
    int x_idx = -1, y_idx = -1;
    for(auto nit = packet->subnodes()->begin(); nit != packet->subnodes()->end(); ++nit) {
        if( *nit == x)
            x_idx = idx;
        if( *nit == y)
            y_idx = idx;
        ++idx;
    }
    if((x_idx < 0) || (y_idx < 0))
        lookupFailure();
    local_shared_ptr<Packet> px = packet->subpackets()->at(x_idx);
    local_shared_ptr<Packet> py = packet->subpackets()->at(y_idx);
    packet->subpackets()->at(x_idx) = py;
    packet->subpackets()->at(y_idx) = px;
    packet->subnodes()->at(x_idx) = y;
    packet->subnodes()->at(y_idx) = x;
    tr[ *this].moveEvent(x_idx, y_idx);
    tr[ *this].listChangeEvent();
    STRICT_assert(tr.m_packet->checkConsistensy(tr.m_packet));
    return true;
}

// reverseLookupWithHint() — fast path for finding a node's packet within a
// transaction's packet tree. Uses the bundledBy back-reference chain stored
// in the node's PacketWrapper to walk directly to the containing slot,
// avoiding a full tree scan. Falls back to nullptr if the hint is stale.
// When copy_branch is true, applies copy-on-write along the path.
template <class XN>
local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::reverseLookupWithHint(shared_ptr<Linkage> &linkage,
    local_shared_ptr<Packet> &superpacket, bool copy_branch, int64_t tr_serial, bool set_missing,
    local_shared_ptr<Packet> *upperpacket, int *index) {
    if( !superpacket->size())
        return nullptr;
    local_shared_ptr<PacketWrapper> wrapper( *linkage);
    if(wrapper->hasPriority())
        return nullptr;
    shared_ptr<Linkage> linkage_upper(wrapper->bundledBy());
    if( !linkage_upper)
        return nullptr;
    local_shared_ptr<Packet> *foundpacket;
    if(linkage_upper == superpacket->node().m_link)
        foundpacket = &superpacket;
    else {
        foundpacket = reverseLookupWithHint(linkage_upper,
            superpacket, copy_branch, tr_serial, set_missing, nullptr, nullptr);
        if( !foundpacket)
            return nullptr;
    }
    int ridx = wrapper->reverseIndex();
    if( !( *foundpacket)->size() || (ridx >= ( *foundpacket)->size()))
        return nullptr;
    if(copy_branch) {
        if(( *foundpacket)->subpackets()->m_serial != tr_serial) {
            foundpacket->reset(new Packet( **foundpacket));
            ( *foundpacket)->subpackets().reset(new PacketList( *( *foundpacket)->subpackets()));
            ( *foundpacket)->m_missing = ( *foundpacket)->m_missing || set_missing;
            ( *foundpacket)->subpackets()->m_serial = tr_serial;
        }
    }
    local_shared_ptr<Packet> &p(( *foundpacket)->subpackets()->at(ridx));
    if( !p || (p->node().m_link != linkage)) {
        return nullptr;
    }
    if(upperpacket) {
        *upperpacket = *foundpacket;
        *index = ridx;
    }
    return &p;
}

// forwardLookup() — fallback tree scan when reverseLookupWithHint() fails.
// Searches the packet tree top-down by matching subnodes[i]->m_link == this.
// More expensive (O(N) in tree size) but always correct.
template <class XN>
inline local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::forwardLookup(local_shared_ptr<Packet> &superpacket,
    bool copy_branch, int64_t tr_serial, bool set_missing,
    local_shared_ptr<Packet> *upperpacket, int *index) const {
    assert(superpacket);
    if( !superpacket->subpackets())
        return nullptr;
    if(copy_branch) {
        if(superpacket->subpackets()->m_serial != tr_serial) {
            superpacket.reset(new Packet( *superpacket));
            superpacket->subpackets().reset(new PacketList( *superpacket->subpackets()));
            superpacket->subpackets()->m_serial = tr_serial;
            superpacket->m_missing = superpacket->m_missing || set_missing;
        }
    }
    for(unsigned int i = 0; i < superpacket->subnodes()->size(); i++) {
        if(( *superpacket->subnodes())[i].get() == this) {
            local_shared_ptr<Packet> &subpacket(( *superpacket->subpackets())[i]);
            if(subpacket) {
                *upperpacket = superpacket;
                *index = i;
                return &subpacket;
            }
        }
    }
    for(unsigned int i = 0; i < superpacket->subnodes()->size(); i++) {
        local_shared_ptr<Packet> &subpacket(( *superpacket->subpackets())[i]);
        if(subpacket) {
            if(local_shared_ptr<Packet> *p =
                forwardLookup(subpacket, copy_branch, tr_serial, set_missing, upperpacket, index)) {
                return p;
            }
        }
    }
    return nullptr;
}

// reverseLookup() — find this node's packet within a transaction packet tree.
// Tries the fast hint-based path first; falls back to forward scan.
// When copy_branch is true, copy-on-write is applied along the found path.
template <class XN>
inline local_shared_ptr<typename Node<XN>::Packet>*
Node<XN>::reverseLookup(local_shared_ptr<Packet> &superpacket,
    bool copy_branch, int64_t tr_serial, bool set_missing, XN **uppernode) {
    local_shared_ptr<Packet> *foundpacket;
    if( &superpacket->node() == this) {
        foundpacket = &superpacket;
    }
    else {
        local_shared_ptr<Packet> upperpacket;
        int index;
        foundpacket = reverseLookupWithHint(m_link, superpacket,
            copy_branch, tr_serial, set_missing, &upperpacket, &index);
        if(foundpacket) {
//				printf("$");
        }
        else {
//				printf("!");
            foundpacket = forwardLookup(superpacket, copy_branch, tr_serial, set_missing,
                &upperpacket, &index);
            if( !foundpacket)
                return 0;
        }
        if(uppernode)
            *uppernode = static_cast<XN*>(&upperpacket->node());
        assert( &( *foundpacket)->node() == this);
    }
    if(copy_branch && (( *foundpacket)->payload()->m_serial != tr_serial)) {
        foundpacket->reset(new Packet( **foundpacket));
    }
//						printf("#");
    return foundpacket;
}

template <class XN>
local_shared_ptr<typename Node<XN>::Packet>&
Node<XN>::reverseLookup(local_shared_ptr<Packet> &superpacket,
    bool copy_branch, int64_t tr_serial, bool set_missing) {
    local_shared_ptr<Packet> *foundpacket = reverseLookup(superpacket, copy_branch, tr_serial, set_missing, 0);
    if( !foundpacket) {
        fprintf(stderr, "Node not found during a lookup.\n");
        throw NodeNotFoundError("Lookup failure.");
    }
    return *foundpacket;
}

template <class XN>
const local_shared_ptr<typename Node<XN>::Packet> &
Node<XN>::reverseLookup(const local_shared_ptr<Packet> &superpacket) const {
    local_shared_ptr<Packet> *foundpacket = const_cast<Node*>(this)->reverseLookup(
        const_cast<local_shared_ptr<Packet> &>(superpacket), false, 0, false, 0);
    if( !foundpacket) {
        fprintf(stderr, "Node not found during a lookup.\n");
        throw NodeNotFoundError("Lookup failure.");
    }
    return *foundpacket;
}

template <class XN>
XN *
Node<XN>::upperNode(Snapshot<XN> &shot) {
    XN *uppernode = 0;
    reverseLookup(shot.m_packet, false, 0, false, &uppernode);
    return uppernode;
}

//=============================================================================
//=============================================================================
// ascendOneLevel() — read bundledBy and prepare to go one level up
//
// Reads the bundledBy pointer from root_wrapper. On success:
//   - Moves root_wrapper into r.child_wrapper for staleness check
//     (zero atomic ops — steals caller's +1 ref).
//   - Loads the parent's wrapper into root_wrapper via *r.parent_linkage.
//   - Copies root_wrapper to parent_wrapper as a snapshot for this level
//     (one fetch_add — used by convertRecursiveStatus to restore
//     root_wrapper on the VOID_PACKET / NODE_MISSING fallback path).
// Returns a WalkUpResult with find_status indicating success/failure.
// Other fields (status, is_root_level, parent_packet) are set later by
// walkUpChainImpl.
//
// child_wrapper is a non-owning identity reference for the staleness
// check at Step D in walkUpChainImpl (only `*child_linkage !=
// r.child_wrapper`, which compares the underlying Ref* pointers).
// Liveness of the saved Ref is guaranteed by the caller's ownership
// chain:
//   - LEVEL 0: caller's outer ScopedNegotiateLinkage view
//     (snapshot: scope.m_view; unbundle: subscope.m_view) holds the
//     original linkage value alive for the duration of walkUpChain.
//   - LEVEL N (recursion): the previous level's r.parent_wrapper
//     (still a value-typed local_shared_ptr) holds the next level's
//     incoming Ref alive across the recursive call.
// This is why we can std::move the save without paying a fetch_add —
// the caller's frame already keeps the Ref alive.
//=============================================================================
template <class XN>
inline typename Node<XN>::WalkUpResult
Node<XN>::ascendOneLevel(
    const shared_ptr<Linkage> &child_linkage,
    local_shared_ptr<PacketWrapper> &root_wrapper) {

    WalkUpResult r;
    r.parent_packet = nullptr;
    assert( !root_wrapper->hasPriority());
    r.parent_linkage = root_wrapper->bundledBy();
    if( !r.parent_linkage) {
        r.find_status = ( *child_linkage == root_wrapper)
            ? SnapshotStatus::NODE_MISSING : SnapshotStatus::DISTURBED;
        return r;
    }
    // Read reverse_index BEFORE moving root_wrapper out (the move
    // empties root_wrapper, so any further root_wrapper-> access would
    // be invalid).
    r.reverse_index = root_wrapper->reverseIndex();
    // Steal caller's +1 ref into r.child_wrapper for the staleness
    // check.  Zero atomic ops — root_wrapper is now empty.
    r.child_wrapper = std::move(root_wrapper);
    // Load parent into the now-empty root_wrapper — operator= calls
    // reset() first which is a no-op on empty, so no fetch_sub on the
    // (already moved-out) old value.
    root_wrapper = *r.parent_linkage;
    r.parent_wrapper = root_wrapper;
    r.find_status = SnapshotStatus::SUCCESS;
    return r;
}

//=============================================================================
// convertRecursiveStatus() — translate recursive call's status for this level
//=============================================================================
template <class XN>
inline typename Node<XN>::SnapshotStatus
Node<XN>::convertRecursiveStatus(
    SnapshotStatus recursive_status,
    WalkUpResult &r,
    local_shared_ptr<PacketWrapper> &root_wrapper,
    local_shared_ptr<Packet> *&parent_packet) {

    switch(recursive_status) {
    case SnapshotStatus::DISTURBED:
    default:
        return recursive_status;
    case SnapshotStatus::VOID_PACKET:
    case SnapshotStatus::NODE_MISSING:
        root_wrapper = r.parent_wrapper;
        parent_packet = &root_wrapper->packet();
        r.is_root_level = true;
        return SnapshotStatus::SUCCESS;
    case SnapshotStatus::NODE_MISSING_AND_COLLIDED:
        root_wrapper = r.parent_wrapper;
        parent_packet = &root_wrapper->packet();
        r.is_root_level = true;
        return SnapshotStatus::COLLIDED;
    case SnapshotStatus::COLLIDED:
    case SnapshotStatus::SUCCESS:
        r.is_root_level = false;
        return recursive_status;
    }
}

//=============================================================================
// findChildSlot() — scan parent's subnodes to find child's sub-packet slot
//
// Starting from reverse_index hint, wraps around to find the child whose
// m_link matches child_linkage. The hint may be stale if swap() reordered
// children.
//
// Returns:
//   current_status  if found (child_subpacket_out is set)
//   VOID_PACKET     if slot exists but sub-packet is null (missing child)
//   NODE_MISSING    if child not found in parent's subnodes
//=============================================================================
template <class XN>
inline typename Node<XN>::SnapshotStatus
Node<XN>::findChildSlot(
    const shared_ptr<Linkage> &child_linkage,
    local_shared_ptr<Packet> *parent_packet,
    local_shared_ptr<Packet> **child_subpacket_out,
    int &reverse_index,
    SnapshotStatus current_status) {

    assert( *parent_packet);
    int size = ( *parent_packet)->size();
    int i = reverse_index;
    for(int cnt = 0;; ++cnt) {
        if(cnt >= size) {
            if(current_status == SnapshotStatus::COLLIDED)
                return SnapshotStatus::NODE_MISSING;
            return SnapshotStatus::NODE_MISSING;
        }
        if(i >= size)
            i = 0;
        if(( *( *parent_packet)->subnodes())[i]->m_link == child_linkage) {
            // Child node found at index i.
            *child_subpacket_out = &( *( *parent_packet)->subpackets())[i];
            reverse_index = i;
            if( !**child_subpacket_out) {
                assert(( *parent_packet)->missing());
                return SnapshotStatus::VOID_PACKET;
            }
            return current_status;  // SUCCESS or COLLIDED — child has a sub-packet.
        }
        ++i;
    }
}

//=============================================================================
// walkUpChainImpl() — common chain-walk: Steps A → B → C → D → E
//
// Shared by walkUpChain() and snapshotForUnbundle().
// Only the recursive call (Step B) differs between callers — passed as Recurser.
// Steps A–E are executed here; the result (with context) is returned via
// Steps:
//   A. ascendOneLevel: read bundledBy, save child_wrapper, return WalkUpResult
//   B. recurse via Recurser if parent is bundled
//   C. convertRecursiveStatus: map recursive result to this-level status
//   D. staleness check: verify child's linkage hasn't changed
//   E. findChildSlot: locate child's sub-packet in parent's packet
//=============================================================================
template <class XN>
template <class Recurser>
inline typename Node<XN>::WalkUpResult
Node<XN>::walkUpChainImpl(const shared_ptr<Linkage> &child_linkage,
    local_shared_ptr<PacketWrapper> &root_wrapper,
    local_shared_ptr<Packet> **child_subpacket_out,
    Recurser &&recurse) {

    // Step A: ascend one level — fills parent_linkage, parent_wrapper, child_wrapper, reverse_index.
    WalkUpResult r = ascendOneLevel(child_linkage, root_wrapper);
    if(r.find_status != SnapshotStatus::SUCCESS)
        return r;

    // Step B: recurse if parent is also bundled.
    SnapshotStatus recursive_status = SnapshotStatus::NODE_MISSING;
    local_shared_ptr<Packet> *parent_packet;
    if( !r.parent_wrapper->hasPriority()) {
        recursive_status = recurse(r.parent_linkage, root_wrapper, &parent_packet);
    }

    // Step C: convert recursive result — sets is_root_level, may update root_wrapper.
    SnapshotStatus status = convertRecursiveStatus(
        recursive_status, r, root_wrapper, parent_packet);
    if(status == SnapshotStatus::DISTURBED) {
        r.find_status = SnapshotStatus::DISTURBED;
        return r;
    }

    // Step D: staleness check.
    if( *child_linkage != r.child_wrapper) {
        r.find_status = SnapshotStatus::DISTURBED;
        return r;
    }

    // Step E: find child's sub-packet slot in parent's packet.
    r.find_status = findChildSlot(child_linkage, parent_packet,
        child_subpacket_out, r.reverse_index, status);
    r.status = status;
    r.parent_packet = parent_packet;
    return r;
}

//=============================================================================
// walkUpChain() — walk up the chain for snapshot/bundle (no CAS construction)
//=============================================================================
template <class XN>
inline typename Node<XN>::SnapshotStatus
Node<XN>::walkUpChain(const shared_ptr<Linkage> &child_linkage,
    local_shared_ptr<PacketWrapper> &root_wrapper,
    local_shared_ptr<Packet> **child_subpacket_out) {

    return walkUpChainImpl(child_linkage, root_wrapper, child_subpacket_out,
        [](const shared_ptr<Linkage> &pl, local_shared_ptr<PacketWrapper> &rw,
           local_shared_ptr<Packet> **pp) {
            return walkUpChain(pl, rw, pp);
        }).find_status;
}

//=============================================================================
// snapshotForUnbundle() — walk up the chain with CAS info construction
//
// Calls walkUpChainImpl for Steps A–E, then performs Step F (CAS preparation)
// using the WalkUpResult context.
//=============================================================================
template <class XN>
inline typename Node<XN>::SnapshotStatus
Node<XN>::snapshotForUnbundle(const shared_ptr<Linkage> &child_linkage,
    local_shared_ptr<PacketWrapper> &root_wrapper,
    local_shared_ptr<Packet> **child_subpacket_out,
    int64_t serial, CASInfoList *cas_infos) {

    auto r = walkUpChainImpl(child_linkage, root_wrapper, child_subpacket_out,
        [serial, cas_infos](const shared_ptr<Linkage> &pl, local_shared_ptr<PacketWrapper> &rw,
           local_shared_ptr<Packet> **pp) {
            return snapshotForUnbundle(pl, rw, pp, serial, cas_infos);
        });

    // --- Post-processing for unbundle (Step F) ---
    if(r.find_status == SnapshotStatus::DISTURBED)
        return SnapshotStatus::DISTURBED;
    if(r.find_status == SnapshotStatus::VOID_PACKET) {
        cas_infos->clear();
        return SnapshotStatus::VOID_PACKET;
    }
    if(r.find_status == SnapshotStatus::NODE_MISSING && !r.parent_packet) {
        return SnapshotStatus::NODE_MISSING;
    }
    SnapshotStatus status = r.status;
    if(r.find_status == SnapshotStatus::NODE_MISSING) {
        if(status == SnapshotStatus::COLLIDED)
            return SnapshotStatus::NODE_MISSING;
        status = SnapshotStatus::NODE_MISSING;
    }

    // Identity check intentionally omitted:
    //   r.parent_wrapper->packet()->node().m_link == r.parent_linkage
    //   ( *r.parent_packet)->node().m_link == r.parent_linkage
    // Both can transiently fail when insert(online_after_insertion=true)
    // or release() publish the child's bundledBy and the parent's
    // linkage in two phases (the child's bundledBy must be published
    // first so reverseLookup works; the parent's linkage update lags).
    // The two are semantically atomic but cannot be made physically so.
    //
    // We do NOT short-circuit to DISTURBED here — that would force
    // retries in cases where the eventual CAS in unbundle() would have
    // succeeded (the transient resolves by CAS time), causing livelock.
    // Instead the cas_infos-driven CAS at unbundle() acts as the
    // natural race detector: if the observed wrapper has changed at
    // CAS time, the CAS fails and unbundle() returns DISTURBED, so the
    // caller retries.

    // CAS preparation
    if(status == SnapshotStatus::COLLIDED)
        return SnapshotStatus::COLLIDED;

    if((serial != SerialGenerator::SERIAL_NULL) &&
        (r.parent_wrapper->m_bundle_serial == serial)) {
        if(status == SnapshotStatus::NODE_MISSING)
            return SnapshotStatus::NODE_MISSING;
        return SnapshotStatus::COLLIDED;
    }

    // Build new wrapper for this ancestor level.
    local_shared_ptr<Packet> *p(r.parent_packet);
    local_shared_ptr<PacketWrapper> newwrapper;
    if(r.is_root_level) {
        newwrapper.reset(
            new PacketWrapper( *r.parent_wrapper, r.parent_wrapper->m_bundle_serial));
    }
    else {
        assert(cas_infos->size());
        newwrapper.reset(
            new PacketWrapper( *p, root_wrapper->m_bundle_serial));
    }
    if(newwrapper) {
        cas_infos->emplace_back(r.parent_linkage, r.parent_wrapper, newwrapper);
        p = &newwrapper->packet();
    }
    int size = ( *r.parent_packet)->size();
    if(size) {
        p->reset(new Packet( **p));
        ( *p)->m_missing = true;
    }

    if((status == SnapshotStatus::NODE_MISSING) && (serial != SerialGenerator::SERIAL_NULL) &&
        (( !r.child_wrapper->hasPriority()) && (r.child_wrapper->m_bundle_serial == serial))) {
        printf("!");
        return SnapshotStatus::NODE_MISSING_AND_COLLIDED;
    }

    return status;
}

//=============================================================================
// snapshot() — obtain an immutable, consistent view of a subtree
//   (Comments by Claude Opus — based on source code analysis)
//
// Three cases depending on the node's current state:
//   1. hasPriority() and not missing: the node owns its packet directly —
//      just return the packet (fast path for leaf or already-bundled nodes).
//   2. !hasPriority(): the node is bundled into a super-node — walk up via
//      walkUpChain() to find the sub-packet, possibly triggering
//      unbundle() if the packet is incomplete (missing sub-packets).
//   3. hasPriority() but missing: sub-packets are stale — call bundle() to
//      absorb all child packets into a consistent parent packet.
//
// The Lamport serial (snapshot.m_serial) is generated from the node's
// m_bundle_serial so that the snapshot serial is always greater than the
// committed state it observes. After bundle(), gen() is called again to
// capture any Lamport advances that occurred during the recursive bundling.
//=============================================================================
template <class XN>
void
Node<XN>::snapshot(Snapshot<XN> &snapshot, bool multi_nodal) const {
    auto &started_time = snapshot.m_started_time;
    auto &tid_bitset = snapshot.m_tid_bitset;
    struct GuardSnapshotRetryCount {
        GuardSnapshotRetryCount(Snapshot<XN> &s) : snapshot(s) {
            m_retry_count_started = snapshot.m_tx_retry_count;
        }
        ~GuardSnapshotRetryCount() {
            snapshot.m_tx_retry_count = m_retry_count_started;
        }
        Snapshot<XN> &snapshot;
        uint32_t m_retry_count_started;
    } guard(snapshot);

    for(int retry = 0;; ++retry) {
        if(retry)
            ++snapshot.m_tx_retry_count;
        // RAII OnEntry: negotiates + tag-bit acquires view of m_link
        // + tags eagerly (retry > 0).
        ScopedNegotiateLinkage<XN> scope(m_link, snapshot, retry,
            ScopedNegotiateLinkage<XN>::TagMode::OnEntry);
        if( !scope) {
            // Weak acquire CAS lost — treat as CAS failure: skip body
            // (m_contention_observed already set in ctor → dtor tags).
            continue;
        }
        snapshot.m_serial = SerialGenerator::gen(scope->m_bundle_serial);
        if(scope->hasPriority()) {
            if( !multi_nodal) {
                snapshot.m_packet = scope->packet();
                scope.commit();
                return;
            }
            if( !scope->packet()->missing()) {
                STRICT_assert(scope->packet()->checkConsistensy(scope->packet()));
                snapshot.m_packet = scope->packet();
                scope.commit();
                return;
            }
        }
        else {
            // Taking a snapshot inside the super packet.  walkUpChain
            // mutates superwrapper across atomic_shared_ptr boundaries
            // (cf. ascendOneLevel) — must be a value-typed
            // local_shared_ptr.  Lvalue-copy from scope's view costs
            // one fetch_add(1) (plus 2 ops if still TagHeld).
            shared_ptr<Linkage > linkage(m_link);
            local_shared_ptr<PacketWrapper> superwrapper = scope.view_copy();
            local_shared_ptr<Packet> *foundpacket;
            auto status = walkUpChain(linkage, superwrapper, &foundpacket);
            switch(status) {
            case SnapshotStatus::SUCCESS: {
                    if( !( *foundpacket)->missing() || !multi_nodal) {
                        snapshot.m_packet = *foundpacket;
                        STRICT_assert(snapshot.m_packet->checkConsistensy(snapshot.m_packet));
                        scope.commit();
                        return;
                    }
                    // The packet is imperfect, and then re-bundling the
                    // subpackets via unbundle.  Pass scope directly —
                    // its view IS the bundled_ref for the final CAS.
                    UnbundledStatus status = unbundle(nullptr, snapshot, scope);
                    switch(status) {
                    case UnbundledStatus::W_NEW_SUBVALUE:
                        // unbundle's final CAS via subscope succeeded —
                        // scope.m_committed already true.  Redundant
                        // commit() is a no-op but kept for clarity.
                        scope.commit();
                        break;
                    case UnbundledStatus::COLLIDED:
                    case UnbundledStatus::SUBVALUE_HAS_CHANGED:
                    default:
                        // unbundle() saw conflict from another thread.
                        scope.confirm_contention();
                        break;
                    }
                    continue;
                }
            case SnapshotStatus::DISTURBED:
            default:
                // walkUpChain disturbed by another thread; pre-CAS contention.
                scope.confirm_contention();
                continue;
            case SnapshotStatus::NODE_MISSING:
            case SnapshotStatus::VOID_PACKET:
                //The packet has been released.
                if( !scope->packet()->missing() || !multi_nodal) {
                    snapshot.m_packet = scope->packet();
                    scope.commit();
                    return;
                }
                break;
            }
        }
        // Fall through to bundle.  Pass scope directly — bundle()
        // consumes scope's view at entry and restores via set_view on
        // success, so on SUCCESS we can read the post-bundle wrapper
        // through scope->packet().
        BundledStatus status = const_cast<Node *>(this)->bundle(
            scope, snapshot, snapshot.m_serial, true);
        switch (status) {
        case BundledStatus::SUCCESS:
            assert( !scope->packet()->missing());
            STRICT_assert(scope->packet()->checkConsistensy(scope->packet()));
            snapshot.m_serial = SerialGenerator::gen(); //Capture Lamport advances from bundle().
            snapshot.m_packet = scope->packet();
            scope.commit();
            return;
        default:
            // bundle() failed (DISTURBED); pre-CAS contention from
            // another thread's interference within bundle's own scopes.
            scope.confirm_contention();
            continue;
        }
    }
}

//=============================================================================
// bundle_subpacket() — prepare one child's packet for inclusion in a bundle
//   (Comments by Claude Opus — based on source code analysis)
//
// Handles the various states a child node can be in:
//   - Already bundled into this parent (linkage == m_link): if the sub-packet
//     is complete, nothing to do (SUCCESS). If missing, recursively bundle.
//   - Bundled into a *different* parent: unbundle it first (with collision
//     detection via bundle_serial), then recursively bundle if still missing.
//   - Has priority (owns its own packet): recursively bundle if missing.
//
// On success, subpacket_new is set to the child's complete packet.
// A null subpacket_new after SUCCESS means the child was already included
// in the current bundle (collision detected, deduplicated).
//=============================================================================
template <class XN>
typename Node<XN>::BundledStatus
Node<XN>::bundle_subpacket(local_shared_ptr<PacketWrapper> *superwrapper,
    const shared_ptr<Node> &subnode,
    local_shared_ptr<PacketWrapper> &subwrapper, local_shared_ptr<Packet> &subpacket_new,
    Snapshot<XN> &snap,
    int64_t bundle_serial) {
    auto &started_time = snap.m_started_time;
    auto &tid_bitset = snap.m_tid_bitset;

    if( !subwrapper->hasPriority()) {
        shared_ptr<Linkage > linkage(subwrapper->bundledBy());
        bool need_for_unbundle = false;
        bool detect_collision = false;
        if(linkage == m_link) {
            if(subpacket_new) {
                if(subpacket_new->missing()) {
                    need_for_unbundle = true;
                }
                else
                    return BundledStatus::SUCCESS;
            }
            else {
                if(subwrapper->packet()) {
                    //Re-inserted.
//					need_for_unbundle = true;
                }
                else
                    return BundledStatus::DISTURBED;
            }
        }
        else {
            need_for_unbundle = true;
            detect_collision = true;
        }
        if(need_for_unbundle) {
            // Move-in subwrapper as the temporary subscope's view (zero
            // atomic ops; reuses subwrapper's +1 ref).  The new
            // unbundle() takes ScopedNegotiateLinkage&, replacing the
            // separate (sublinkage, bundled_ref) parameters.  Tag mode
            // OnEntry retry==-1 matches the previous inner-scope used
            // inside unbundle for its final sublinkage CAS.
            ScopedNegotiateLinkage<XN> subscope(subnode->m_link, snap, -1,
                std::move(subwrapper),
                ScopedNegotiateLinkage<XN>::TagMode::OnEntry);
            local_shared_ptr<PacketWrapper> subwrapper_new;
            UnbundledStatus status = unbundle(detect_collision ? &bundle_serial : nullptr, snap,
                subscope, nullptr, &subwrapper_new, superwrapper);
            switch(status) {
            case UnbundledStatus::W_NEW_SUBVALUE:
                // Final CAS via subscope succeeded → subscope's view
                // is consumed.  Promote the returned newsubwrapper
                // into our local subwrapper for the rest of the
                // function (the bundle() recurse below mutates it).
                subwrapper = subwrapper_new;
                break;
            case UnbundledStatus::COLLIDED:
                //The subpacket has already been included in the snapshot.
                // unbundle returned pre-CAS; subscope's view is still
                // valid.  Restore subwrapper from it (zero ops Owned /
                // 2 ops TagHeld) so the caller's `subwrappers_org[i] =
                // subwrapper` after this returns sees a valid value.
                subwrapper = subscope.consume_view();
                subpacket_new.reset();
                return BundledStatus::SUCCESS;
            case UnbundledStatus::SUBVALUE_HAS_CHANGED:
            default:
                // Caller (bundle Phase 1) does not use subwrapper after
                // a non-SUCCESS return — no need to restore it.
                return BundledStatus::DISTURBED;
            }
        }
    }
    if(subwrapper->packet()->missing()) {
        assert(subwrapper->packet()->size());
        // Move-in subwrapper as the temporary subscope's view (zero
        // atomic ops — reuses subwrapper's +1 ref).  bundle() consumes
        // the view and restores via set_view on SUCCESS.
        ScopedNegotiateLinkage<XN> subscope(subnode->m_link, snap, -1,
            std::move(subwrapper),
            ScopedNegotiateLinkage<XN>::TagMode::OnEntry);
        BundledStatus status = subnode->bundle(subscope, snap, bundle_serial, false);
        // Restore subwrapper from subscope.view (move-out, 0 ops on
        // SUCCESS Owned; empty on DISTURBED).  For DISTURBED the
        // caller (bundle Phase 1) doesn't use subwrapper after.
        subwrapper = subscope.consume_view();
        switch(status) {
        case BundledStatus::SUCCESS:
            break;
        case BundledStatus::DISTURBED:
        default:
            return BundledStatus::DISTURBED;
        }
    }
    subpacket_new = subwrapper->packet();
    return BundledStatus::SUCCESS;
}

//=============================================================================
// bundle() — multi-phase CAS protocol to absorb child packets into a parent
//   (Comments by Claude Opus — based on source code analysis)
//
// Purpose: make a subtree atomically snapshotable by collecting all child
//   node packets into the parent's packet. After bundling, a single
//   atomic_shared_ptr read on the parent's Linkage yields the entire
//   consistent subtree.
//
// Preconditions:
//   - oldsuperwrapper points to a valid PacketWrapper with a non-null,
//     missing packet (i.e. the packet knows it has stale child slots).
//   - bundle_serial is the Lamport serial for this bundling operation.
//
// Protocol phases (inside the retry loop):
//
//   Phase 1 — Collect sub-packets:
//     Create a copy of the parent packet. For each child node, read its
//     current PacketWrapper and call bundle_subpacket() to obtain the
//     child's packet (recursively bundling if the child itself is missing).
//     If a child is bundled into a *different* super-node, unbundle() it
//     first. Advances the Lamport clock past each child's serial.
//
//   Phase 2 — First checkpoint (CAS on parent Linkage):
//     Atomically replace the parent's PacketWrapper with the new one that
//     contains the collected sub-packets (still marked missing). If another
//     thread modified the parent, return DISTURBED and let the caller retry.
//
//   Phase 3 — Second checkpoint (CAS on each child Linkage):
//     Replace each child's PacketWrapper with a bundled_ref — a wrapper
//     that points back to the parent (bundledBy = m_link) and stores the
//     child's reverse index. These back-references allow the child to find
//     its packet inside the parent's bundled packet.
//     If any child was modified between phases 2 and 3, restart from phase 1.
//
//   Phase 4 — Finalize:
//     Create a final PacketWrapper. If no sub-packets are still missing,
//     clear the missing flag (the subtree is now fully consistent).
//     CAS the parent's Linkage one last time.
//
// The is_bundle_root flag (true only for the outermost bundle() call from
// snapshot()) forces missing=false at phase 4, because the root bundler
// will not be bundled further and needs a complete packet.
//=============================================================================
template <class XN>
typename Node<XN>::BundledStatus
Node<XN>::bundle(ScopedNegotiateLinkage<XN> &supscope,
    Snapshot<XN> &snap,
    int64_t bundle_serial, bool is_bundle_root) {
    auto &started_time = snap.m_started_time;
    auto &tid_bitset = snap.m_tid_bitset;

    // Move-out the caller's view as a value-typed local_shared_ptr.
    // Bundle's multi-phase CAS reassigns this across Phase 2/4 success;
    // cannot stay as a scoped view bound to a single atomic.  Owned
    // mode → 0 atomic ops; TagHeld → 2 ops (promote).  At the success
    // exit below we move the final wrapper back into supscope.view via
    // set_view (zero atomic ops).
    local_shared_ptr<PacketWrapper> oldsuperwrapper = supscope.consume_view();

    assert(oldsuperwrapper->packet());
    assert(oldsuperwrapper->packet()->size());
    assert(oldsuperwrapper->packet()->missing());

    Node &supernode(oldsuperwrapper->packet()->node());

    if( !oldsuperwrapper->hasPriority() ||
        (oldsuperwrapper->m_bundle_serial != bundle_serial)) {
        //Tags serial.
        local_shared_ptr<PacketWrapper> superwrapper(
            new PacketWrapper(oldsuperwrapper->packet(), bundle_serial));
        // OnExit retry==-1: negotiates eagerly; tagging on dtor when
        // !m_committed.  Note: a CAS failure here on a privileged Tx
        // would still trip priv-assert, since serial-tag CAS contends
        // on the same supernode.m_link as commit/bundle main loop.
        ScopedNegotiateLinkage<XN> scope(supernode.m_link, snap, -1,
            ScopedNegotiateLinkage<XN>::TagMode::OnExit);
        if( !scope) return BundledStatus::DISTURBED;  // weak acquire lost
        if( !scope.compareAndSet(oldsuperwrapper, superwrapper))
            return BundledStatus::DISTURBED; // dtor tags
        oldsuperwrapper = std::move(superwrapper);
    }

    fast_vector<local_shared_ptr<PacketWrapper>, 16> subwrappers_org(oldsuperwrapper->packet()->subpackets()->size());

    for(int retry = 0;; ++retry) {
        // RAII OnEntry: negotiates supernode.m_link + tags eagerly (retry > 0).
        ScopedNegotiateLinkage<XN> scope(supernode.m_link, snap, retry,
            ScopedNegotiateLinkage<XN>::TagMode::OnEntry);
        if( !scope) continue;  // weak acquire lost — retry
        local_shared_ptr<PacketWrapper> superwrapper(
            new PacketWrapper( *oldsuperwrapper, bundle_serial));
        local_shared_ptr<Packet> &newpacket(
            reverseLookup(superwrapper->packet(), true, SerialGenerator::gen()));
        assert(newpacket->size());
        assert(newpacket->missing());

        STRICT_assert(s_serial_abandoned != newpacket->subpackets()->m_serial);

        //--- Phase 1: collect sub-packets from child nodes ---
        newpacket->subpackets().reset(new PacketList( *newpacket->subpackets()));
        shared_ptr<PacketList> &subpackets(newpacket->subpackets());
        shared_ptr<NodeList> &subnodes(newpacket->subnodes());

        bool missing = false;
        for(unsigned int i = 0; i < subpackets->size(); ++i) {
            shared_ptr<Node> child(( *subnodes)[i]);
            local_shared_ptr<Packet> &subpacket_new(( *subpackets)[i]);
            local_shared_ptr<PacketWrapper> subwrapper;
            for(int child_retry = 0;; ++child_retry) {
                // RAII OnEntry: negotiates child->m_link + tags eagerly
                // on retry > 0.  Covers the read at *child->m_link below
                // and the CAS in bundle_subpacket → unbundle.
                ScopedNegotiateLinkage<XN> child_scope(
                    child->m_link, snap, child_retry,
                    ScopedNegotiateLinkage<XN>::TagMode::OnEntry);
                if( !child_scope) continue;  // weak acquire lost — retry
                // Reuse child_scope's view as subwrapper instead of a
                // fresh load on child->m_link.  view_copy ≤ load_shared_
                // in atomic ops; bundle_subpacket may then reassign
                // subwrapper internally.
                subwrapper = child_scope.view_copy();
                if(subwrapper == subwrappers_org[i]) {
                    child_scope.commit(); //fast path for retry > 0.
                    break;
                }
                SerialGenerator::gen(subwrapper->m_bundle_serial); //Lamport: advance past sub-node serial.
                BundledStatus status = bundle_subpacket( &oldsuperwrapper,
                    child, subwrapper, subpacket_new, snap, bundle_serial);
                switch(status) {
                case BundledStatus::SUCCESS:
                    break;
                case BundledStatus::DISTURBED:
                default:
                    // bundle_subpacket DISTURBED: contention observed on
                    // child->m_link (no scope-CAS attempted; pre-CAS conflict).
                    child_scope.confirm_contention();
                    if(oldsuperwrapper == *supernode.m_link)
                        continue;
                    // Phase 1 early exit: no CAS on supernode.m_link was
                    // attempted yet (Phase 2 not reached).  Outer scope
                    // observed contention indirectly via child failure.
                    scope.confirm_contention();
                    return status;
                }
                subwrappers_org[i] = subwrapper;
                if(subpacket_new) {
                    if(subpacket_new->missing()) {
                        missing = true;
                    }
                    assert(&subpacket_new->node() == child.get());
                }
                else
                    missing = true;
                child_scope.commit();
                break;
            }
        }
        if(is_bundle_root) {
            assert( &supernode == this);
            missing = false;
        }
        newpacket->m_missing = true;

        //--- Phase 2: first checkpoint — CAS the parent PacketWrapper ---
        // CAS via scope: success auto-commits; failure records
        // m_contention_observed (dtor tag) and triggers priv-assert at
        // the CAS site if the negotiate-yield was bypassed.
        if( !scope.compareAndSet(oldsuperwrapper, superwrapper))
            return BundledStatus::DISTURBED;
        oldsuperwrapper = superwrapper;

        //--- Phase 3: second checkpoint — CAS each child's Linkage to point back to parent ---
        //  Each bundled_ref is a PacketWrapper holding a back-reference
        //  (bundledBy → parent's m_link) and the child's reverse index.
        bool changed_during_bundling = false;
        for(unsigned int i = 0; i < subnodes->size(); i++) {
            shared_ptr<Node> child(( *subnodes)[i]);
            local_shared_ptr<PacketWrapper> bundled_ref;
            if(( *subpackets)[i])
                bundled_ref.reset(new PacketWrapper(m_link, i, bundle_serial));
            else
                bundled_ref.reset(new PacketWrapper( *subwrappers_org[i], bundle_serial));

            assert( !bundled_ref->hasPriority());
            //Second checkpoint, the written bundle is valid or not.
            ScopedNegotiateLinkage<XN> childScope(child->m_link, snap, retry,
                ScopedNegotiateLinkage<XN>::TagMode::OnExit,
                2.0f / subnodes->size());
            if( !childScope) {
                // Weak acquire lost — treat as Phase 3 CAS failure.
                changed_during_bundling = true;
                break;
            }
            if( !childScope.compareAndSet(subwrappers_org[i], bundled_ref)) {
                // Phase 3 child-CAS failure.  childScope auto-recorded
                // contention (dtor will tag).  Note: priv-assert WILL fire
                // for a privileged Tx here — Phase 3 CAS failure means
                // another thread raced our bundling, which under privilege
                // should be impossible.
                if((local_shared_ptr<PacketWrapper>( *child->m_link)->m_bundle_serial != bundle_serial)
                 || (oldsuperwrapper != *supernode.m_link)) {
                    // Phase 2 CAS on supernode.m_link already succeeded;
                    // commit the outer scope before returning DISTURBED so
                    // its dtor doesn't re-tag/assert on legitimate
                    // forward progress.
                    scope.commit();
                    return BundledStatus::DISTURBED;
                }
                changed_during_bundling = true;
                break;
            }
            // childScope auto-committed via scope.compareAndSet.
        }
        if(changed_during_bundling) {
            // Phase 2 CAS already succeeded — supernode state advanced.
            // Mark the outer scope committed before re-iterating to
            // suppress the dtor tag/assert (continuation handles Phase 3
            // contention by retrying the entire bundle).
            scope.commit();
            continue;
        }

        //--- Phase 4: finalize — clear missing flag if all sub-packets are present ---
        superwrapper.reset(new PacketWrapper( *superwrapper, bundle_serial));
        if( !missing) {
            local_shared_ptr<Packet> &newpacket(
                reverseLookup(superwrapper->packet(), true, SerialGenerator::gen()));
            newpacket->m_missing = false;
            STRICT_assert(newpacket->checkConsistensy(newpacket));
        }

        // CAS via scope.  On failure, Phase 2 already advanced state, so
        // the outer scope.commit() is needed to suppress the dtor tag
        // (legitimate forward progress).
        if( !scope.compareAndSetWithHint(oldsuperwrapper, superwrapper, started_time)) {
            scope.commit();
            return BundledStatus::DISTURBED;
        }
        oldsuperwrapper = std::move(superwrapper);

        for(unsigned int i = 0; i < subnodes->size(); i++) {
            shared_ptr<Node> child(( *subnodes)[i]);
            //this tagging significantly increased a commiting rate.
            child->m_link->tags_successful_cas(started_time);
        }
        // scope.compareAndSetWithHint above already auto-committed +
        // tagged successful_cas on supernode.m_link.
        break;
    }
    // Restore supscope.view from the final wrapper (move-in, 0 ops).
    supscope.set_view(std::move(oldsuperwrapper));
    return BundledStatus::SUCCESS;
}

//template <class XN>
//void
//Node<XN>::fetchSubpackets(std::deque<local_shared_ptr<PacketWrapper> > &subwrappers,
//	const local_shared_ptr<Packet> &packet) {
//	for(int i = 0; i < packet->size(); ++i) {
//		const local_shared_ptr<Packet> &subpacket(( *packet->subpackets())[i]);
//		subwrappers.push_back( *( *packet->subnodes())[i]->m_link);
//		if(subpacket)
//			fetchSubpackets(subwrappers, subpacket);
//	}
//}
//template <class XN>
//bool
//Node<XN>::commit_at_super(Transaction<XN> &tr) {
//	Node &node(tr.m_packet->node());
//	for(Transaction<XN> tr_super( *this);; ++tr_super) {
//		local_shared_ptr<Packet> *packet
//			= node.reverseLookup(tr_super.m_packet, false, Packet::SERIAL_NULL, false, 0);
//		if( !packet)
//			return false; //Released.
//		if( *packet != tr.m_oldpacket) {
//			if( !tr.isMultiNodal() && (( *packet)->payload() == tr.m_oldpacket->payload())) {
//				//Single-node mode, the payload in the snapshot is unchanged.
//				tr.m_packet->subpackets() = ( *packet)->subpackets();
//				tr.m_packet->m_missing = ( *packet)->missing();
//			}
//			else {
//				return false;
//			}
//		}
//		node.reverseLookup(tr_super.m_packet, true, tr_super.m_serial, tr.m_packet->missing())
//			= tr.m_packet;
//		if(tr_super.commit()) {
//			tr.m_packet = tr_super.m_packet;
//			return true;
//		}
//	}
//}
//=============================================================================
// commit() — atomically publish a transaction's modified packet
//   (Comments by Claude Opus — based on source code analysis)
//
// Two paths depending on the node's current state:
//
//   hasPriority() = true (the node owns its Linkage directly):
//     Fast path — single CAS on m_link to replace the old PacketWrapper
//     with one containing the transaction's new packet.
//     Single-node optimization: if only the payload changed (not children),
//     and another transaction changed the children since our snapshot,
//     we can adopt the new children and still commit — avoiding a spurious
//     conflict when two transactions touch disjoint parts of the same node.
//
//   hasPriority() = false (the node is bundled into a super-node):
//     Must unbundle first to reclaim the Linkage, then CAS the new packet.
//     For multi-nodal transactions, the new packet is passed into unbundle
//     so it can be installed in a single step. For single-node transactions,
//     unbundle restores priority and commit retries via the fast path.
//=============================================================================
template <class XN>
bool
Node<XN>::commit(Transaction<XN> &tr) {
    assert(tr.m_oldpacket != tr.m_packet);
    assert(tr.isMultiNodal() || tr.m_packet->subpackets() == tr.m_oldpacket->subpackets());
    assert(this == &tr.m_packet->node());

    local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper(tr.m_packet, tr.m_serial));
    for(int retry = 0;; ++retry) {
        // RAII OnEntry: negotiates + tag-bit acquires view of m_link
        // + tags eagerly (retry > 0).  scope's internal view is the
        // CAS oldr.
        ScopedNegotiateLinkage<XN> scope(m_link, tr, retry,
            ScopedNegotiateLinkage<XN>::TagMode::OnEntry);
        if( !scope) {
            // Weak acquire CAS lost.  m_contention_observed already
            // set in ctor — dtor tags this iteration.  Skip the body.
            continue;
        }
        if(scope->hasPriority()) {
            //Committing directly to the node.
            if(scope->packet() != tr.m_oldpacket) {
                if( !tr.isMultiNodal() && (scope->packet()->payload() == tr.m_oldpacket->payload())) {
                    //Single-node mode, the payload in the snapshot is unchanged.
                    tr.m_packet->subpackets() = scope->packet()->subpackets();
                    tr.m_packet->m_missing = scope->packet()->missing();
                }
                else {
                    STRICT_TEST(s_serial_abandoned = tr.m_serial);
//					fprintf(stderr, "F");
                    // Conflict detected: another thread committed a new packet
                    // before we even reached the CAS.  Pre-CAS exit; no CAS
                    // attempted via the scope.  confirm_contention() forces
                    // dtor tag so retry==0 fast-path doesn't skip the
                    // contender mark.
                    scope.confirm_contention();
                    return false;
                }
            }
//			STRICT_TEST(std::deque<local_shared_ptr<PacketWrapper> > subwrappers);
//			STRICT_TEST(fetchSubpackets(subwrappers, wrapper->packet()));
            STRICT_assert(tr.m_packet->checkConsistensy(tr.m_packet));

            // CAS via the scope: success → auto-commit + priority hint;
            // failure → m_contention_observed → dtor tag.
            if(scope.compareAndSetWithHint(newwrapper, tr.m_started_time))
                return true;
            continue;
        }
        //Unbundling this node from the super packet.
        // Pass the outer scope directly — its view IS the bundled_ref
        // (current value of m_link == sublinkage).  unbundle uses
        // subscope.compareAndSetWithHint() for the final CAS, no
        // separate inner negotiate.
        UnbundledStatus status = unbundle(nullptr, tr, scope,
            tr.isMultiNodal() ? &tr.m_oldpacket : nullptr, tr.isMultiNodal() ? &newwrapper : nullptr);
        switch(status) {
        case UnbundledStatus::W_NEW_SUBVALUE:
            if(tr.isMultiNodal()) {
                scope.commit();
                return true;
            }
            // unbundle() succeeded in its inner sublinkage CAS on m_link.
            // The outer scope here covers the same m_link; mark it
            // committed so the dtor neither tags (already tagged by the
            // inner CAS path) nor asserts.
            scope.commit();
            continue;  // RAII already tagged at ctor for retry > 0
        case UnbundledStatus::SUBVALUE_HAS_CHANGED: {
                STRICT_TEST(s_serial_abandoned = tr.m_serial);
//				fprintf(stderr, "F");
                // Pre-CAS conflict from another thread.
                scope.confirm_contention();
                return false;
            }
        case UnbundledStatus::DISTURBED:
        default:
            // unbundle() failed before touching m_link (ancestor CAS race,
            // or snapshotForUnbundle disturbed).  Pre-CAS contention.
            scope.confirm_contention();
            continue;
        }
    }
}

//=============================================================================
// unbundle() — extract a sub-node's packet from its super-node's bundle
//   (Comments by Claude Opus — based on source code analysis)
//
// Purpose: the reverse of bundle(). Restores a child node's Linkage to
//   point directly to its own PacketWrapper (hasPriority = true) so that
//   the child can be committed independently.
//
// Parameters:
//   - bundle_serial: if non-null, check for collisions (the sub-packet
//     may already be included in an ongoing snapshot with this serial).
//   - subscope: scope around the child's Linkage; its internal view is
//     the back-reference wrapper (formerly the `bundled_ref`
//     parameter).  The final sublinkage CAS uses subscope's view as
//     oldr — no re-negotiate before the final CAS (caller's outer
//     negotiate covers it).
//   - oldsubpacket: if provided, verify the sub-packet hasn't changed
//     (conflict detection for commit).
//   - newsubwrapper_returned: if provided, the new PacketWrapper for the
//     child is stored here (used by commit to install in one step).
//   - oldsuperwrapper: if provided, track super-node wrapper updates
//     (needed when commit must know the super-node's current state).
//
// Steps:
//   1. Walk up via snapshotForUnbundle() to locate the sub-packet inside
//      the super-node's bundled packet. snapshotForUnbundle() builds a
//      cas_infos list of (linkage, old_wrapper, new_wrapper) for each
//      ancestor that needs its PacketWrapper updated.  Walks on a
//      separate `superwrapper` value-typed wrapper because chain-walk
//      crosses atomic_shared_ptr boundaries (scoped_atomic_view binds
//      to one atomic).
//   2. If oldsubpacket was given and doesn't match, return
//      SUBVALUE_HAS_CHANGED (the transaction must fail).
//   3. Execute the CAS list from cas_infos bottom-up: each ancestor's
//      Linkage gets a new PacketWrapper with a copied packet that marks
//      the child's slot as missing (the child is leaving the bundle).
//   4. CAS the child's Linkage via subscope.compareAndSetWithHint:
//      replace the back-reference (subscope's view) with a new
//      PacketWrapper that holds the extracted sub-packet directly
//      (hasPriority = true). The serial is advanced past the super-
//      node's bundle serial via gen().
//=============================================================================
template <class XN>
typename Node<XN>::UnbundledStatus
Node<XN>::unbundle(const int64_t *bundle_serial, Snapshot<XN> &snap,
    ScopedNegotiateLinkage<XN> &subscope,
    const local_shared_ptr<Packet> *oldsubpacket, local_shared_ptr<PacketWrapper> *newsubwrapper_returned,
    local_shared_ptr<PacketWrapper> *oldsuperwrapper) {
    auto &time_started = snap.m_started_time;
    auto &tid_bitset = snap.m_tid_bitset;

    assert( !subscope->hasPriority());

// Taking a snapshot inside the super packet.
//   superwrapper is a separate value-typed wrapper because
//   snapshotForUnbundle / walkUpChain mutate it across atomic
//   boundaries (cf. ascendOneLevel).  Lvalue copy from subscope's
//   view costs one fetch_add(1) (plus 2 ops if still TagHeld).
    local_shared_ptr<PacketWrapper> superwrapper = subscope.view_copy();
    local_shared_ptr<Packet> *newsubpacket;
    CASInfoList cas_infos;
    SnapshotStatus status = snapshotForUnbundle(subscope.linkage(), superwrapper, &newsubpacket,
        bundle_serial ? *bundle_serial : SerialGenerator::SERIAL_NULL, &cas_infos);
    if(status == SnapshotStatus::DISTURBED)
        return UnbundledStatus::DISTURBED;
    if(status == SnapshotStatus::VOID_PACKET || status == SnapshotStatus::NODE_MISSING) {
        newsubpacket = const_cast<local_shared_ptr<Packet> *>( &subscope->packet());
        assert(newsubpacket);
    }
    if(status == SnapshotStatus::NODE_MISSING_AND_COLLIDED) {
        newsubpacket = const_cast<local_shared_ptr<Packet> *>( &subscope->packet());
        assert(newsubpacket);
        status = SnapshotStatus::COLLIDED;
    }
    // SUCCESS, COLLIDED → fall through

    if(oldsubpacket && ( *newsubpacket != *oldsubpacket))
        return UnbundledStatus::SUBVALUE_HAS_CHANGED;

    for(auto it = cas_infos.begin(); it != cas_infos.end(); ++it) {
        // RAII OnEntry retry==-1: negotiates + tags eagerly.
        // Each cas_info is a different ancestor linkage (≠ subscope's
        // linkage), so a fresh ScopedNegotiateLinkage is needed here.
        ScopedNegotiateLinkage<XN> scope(it->linkage, snap, -1,
            ScopedNegotiateLinkage<XN>::TagMode::OnEntry, 2.0f / cas_infos.size());
        if( !scope) return UnbundledStatus::DISTURBED;  // weak acquire lost
        if( !scope.compareAndSet(it->old_wrapper, it->new_wrapper))
            return UnbundledStatus::DISTURBED;
        // scope.compareAndSet auto-committed on success.  A subsequent
        // return DISTURBED below from the oldsuperwrapper check is
        // legitimate forward progress (linkage already advanced), and
        // m_committed=true silences the dtor's tag/assert.
        if(oldsuperwrapper) {
            if( ( *oldsuperwrapper)->packet()->node().m_link == it->linkage) {
                if( *oldsuperwrapper != it->old_wrapper)
                    return UnbundledStatus::DISTURBED;
//				printf("1\n");
                *oldsuperwrapper = it->new_wrapper;
            }
        }
    }
    if(status == SnapshotStatus::COLLIDED)
        return UnbundledStatus::COLLIDED;

    local_shared_ptr<PacketWrapper> newsubwrapper;
    if(oldsubpacket)
        newsubwrapper = *newsubwrapper_returned;
    else
        newsubwrapper.reset(new PacketWrapper( *newsubpacket, SerialGenerator::gen(superwrapper->m_bundle_serial)));

    // Final sublinkage CAS via the caller-provided subscope.  The
    // outer scope's negotiate (at construction) covers this CAS — no
    // separate inner re-negotiate, saves one negotiate() per unbundle.
    if( !subscope.compareAndSetWithHint(newsubwrapper, time_started))
        return UnbundledStatus::SUBVALUE_HAS_CHANGED;
    // subscope.compareAndSetWithHint auto-committed + tagged success.

    for(auto it = cas_infos.begin(); it != cas_infos.end(); ++it) {
        it->linkage->tags_successful_cas(time_started);
    }

    if(newsubwrapper_returned)
        *newsubwrapper_returned = newsubwrapper;

//	if(oldsuperwrapper) {
//		if( &( *oldsuperwrapper)->packet()->node().m_link != &cas_infos.front().linkage)
//			return UNBUNDLE_DISTURBED;
//		if( *oldsuperwrapper != cas_infos.front().old_wrapper)
//			return UNBUNDLE_DISTURBED;
//		printf("1\n");
//		*oldsuperwrapper = cas_infos.front().new_wrapper;
//	}

    return UnbundledStatus::W_NEW_SUBVALUE;
}

#if defined __WIN32__ || defined WINDOWS || defined _WIN32
    #include <windows.h>
#endif

void setCurrentPriorityMode(Priority pr) {
    *stl_currentPriority = pr;
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
    SetThreadPriority(GetCurrentThread(),
        (pr == Priority::HIGHEST) ? THREAD_PRIORITY_TIME_CRITICAL : THREAD_PRIORITY_NORMAL);
#endif
}

} //namespace Transactional

