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
//   * Transactional::detail livelock probe  — LivelockProbe + TLS instance
//   * ScopedNegotiateLinkage<XN>            — RAII helper for retry loops
//
// Out-of-class definitions of NegotiationCounter members and the full
// negotiate_internal() body live in transaction_neg_impl.h. Tuning
// macros live in transaction_definitions.h.
//
// This header is included from transaction_impl.h after transaction.h,
// so all of Node<XN>'s nested types (Linkage, NegotiationCounter,
// Snapshot<XN>, Transaction<XN>, TidBitset, ...) are already visible.
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

// (popcount_u64 lives in transaction.h so inline member functions there
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

// Forward declaration. effective_runners() is the hardware_concurrency-
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
    inline thread_local LivelockProbe tls_livelock_probe;

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
//! Tagging modes:
//!   - **OnEntry** (default): tag_as_contender immediately at ctor.
//!     Eager — other threads' negotiate sees our contender mark for
//!     the duration of this iter's body. Matches the pre-RAII manual
//!     pattern in commit / bundle / Node::snapshot retry loops.
//!   - **OnExit**: tag_as_contender at dtor when `commit()` was not
//!     called. Lazy — tags only on continue/return/exception paths,
//!     leaving `commit()` paths untagged. Matches the pre-RAII
//!     pattern at insert(online) / eraseSerials.
//!
//! Retry gating:
//!   - `retry >  0`: tag (subject to TagMode).
//!   - `retry == 0`: do NOT tag (first-attempt fast path).
//!   - `retry == -1`: tag unconditionally (always-tag mode); the
//!     value `0` is passed down to `negotiate_after_retry_pause`,
//!     since the function self-gates on fair-mode anyway.
//!
//! Use everywhere a loop body's branch depends on `*linkage` state OR
//! a CAS on the linkage; ensures negotiate-before-decision +
//! tag-on-disturb at every site by construction.
//!
//! Definition lives here (transaction_negotiation.h) — only the
//! retry-loop sites in transaction_impl.h use it. Forward-declared in
//! transaction.h, with friend declarations in Node<XN> / Snapshot<XN>
//! for the assert path.
template <class XN>
class ScopedNegotiateLinkage {
    using LinkagePtr = std::shared_ptr<typename Node<XN>::Linkage>;
    LinkagePtr      m_link;
    Snapshot<XN>   &m_snap;
    bool            m_eager;
    bool            m_should_tag;   // false at retry==0; true at retry>0 or retry==-1
    bool            m_committed = false;
#if KAME_STM_ASSERT_PRIVILEGE
    bool            m_privilege_onentry;
#endif
public:
    enum class TagMode { OnEntry, OnExit };
    ScopedNegotiateLinkage(LinkagePtr link, Snapshot<XN> &snap, int retry,
                           TagMode mode = TagMode::OnEntry,
                           float mult_wait = 2.0f) noexcept
        : m_link(std::move(link)), m_snap(snap),
          m_eager(mode == TagMode::OnEntry),
          m_should_tag(retry != 0) {
#if KAME_STM_ASSERT_PRIVILEGE
        // True iff THIS Tx currently owns the privileged slot.
        m_privilege_onentry =
            (Node<XN>::NegotiationCounter::s_privileged_tidstamp.load(std::memory_order_relaxed)
             == m_snap.m_started_time);
#endif
        const int negotiate_retry = (retry < 0) ? 0 : retry;
        m_link->negotiate_after_retry_pause(negotiate_retry, snap, mult_wait);
        if(m_eager && m_should_tag)
            m_snap.tag_as_contender(m_link);
    }
    ScopedNegotiateLinkage(const ScopedNegotiateLinkage &) = delete;
    ScopedNegotiateLinkage &operator=(const ScopedNegotiateLinkage &) = delete;
    void commit() noexcept { m_committed = true; }
    //! Convenience: when the scope's linkage IS the CAS target and
    //! the CAS just succeeded, this calls
    //! `m_link->tags_successful_cas(started_time)` (updates the
    //! priority slot) and marks the scope committed in one step.
    //! `started_time = 0` (default) tells `tags_successful_cas` to
    //! use `now_us()` instead of unpacking a tid-stamped value.
    void commit_after_cas(typename Node<XN>::NegotiationCounter::cnt_t
                          started_time = 0) noexcept {
        m_link->tags_successful_cas(started_time);
        m_committed = true;
    }
    ~ScopedNegotiateLinkage() noexcept {
        if(!m_committed) {
       #if KAME_STM_ASSERT_PRIVILEGE
            // Fires when we held the privilege at entry AND still hold
            // it at exit without committing — the privileged Tx failed a
            // CAS / loop iteration, violating the livelock-free invariant.
            assert(!(m_privilege_onentry &&
                Node<XN>::NegotiationCounter::s_privileged_tidstamp.load(std::memory_order_relaxed)
                    == m_snap.m_started_time)
                && "privileged Tx CAS/loop failure: ScopedNegotiateLinkage dtor");
       #endif
            if(!m_eager && m_should_tag)
                m_snap.tag_as_contender(m_link);
        }
    }
};

} // namespace Transactional

#endif /* TRANSACTION_NEGOTIATION_H */
