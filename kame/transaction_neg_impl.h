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
// transaction_neg_impl.h
//
// Out-of-class template member definitions for:
//   * Node<XN>::NegotiationCounter (fair-mode escape, livelock probe,
//     CV-based sleep/notify slots).
//   * Node<XN>::Linkage::negotiate_after_retry_pause()
//   * Node<XN>::Linkage::negotiate_internal()       (the big one)
//
// Declarations and class bodies live in transaction.h. Tuning macros
// live in transaction_definitions.h. Surrounding negotiation-layer
// helpers (retry_pause, ScopedNegotiateLinkage, detail:: probe types,
// effective_runners forward decl) live in transaction_negotiation.h.
//
// Included from transaction_impl.h after transaction.h and
// transaction_negotiation.h so all dependent types are visible.
// =====================================================================
#ifndef TRANSACTION_NEG_IMPL_H
#define TRANSACTION_NEG_IMPL_H

#include "transaction.h"
#include "transaction_negotiation.h"
#include "transaction_definitions.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>

namespace Transactional {

template <class XN>
alignas(64) atomic<unsigned int> Node<XN>::NegotiationCounter::s_running{0};
// tx_nest / sleep_nest are defined at namespace scope in transaction.h
// (detail::s_tx_nest, detail::s_sleep_nest). Placing them outside the
// class template avoids an Apple clang / arm64 bug where the TLS wrapper
// for a template static `thread_local` member is not emitted in TUs
// that only include transaction.h.

// =====================================================================
// Out-of-class template member definitions for Node<XN>::NegotiationCounter.
// Declarations live in transaction.h; bodies here pick up the namespace-
// scope `detail::tls_livelock_probe`, `detail::ll_now_us`, and the
// `s_privileged_tidstamp` / `s_sleep_slots` C++17 inline static members
// (defined in the class body in transaction.h).
// =====================================================================

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
    return priv != tidstamp;
}

template <class XN>
typename Node<XN>::NegotiationCounter::PriorityProbeInfo
Node<XN>::NegotiationCounter::priority_probe_info(Priority pr) noexcept {
    switch (pr) {
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
    alignas(64) inline std::atomic<int> s_max_c_obs{1};

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

// Optional diagnostic counters (opt-in via -DKAME_ADAPT_INSTRUMENT=1).
// Inspect with gdb while a test runs: `thread apply all print <name>`.
// Off by default to keep per-call overhead minimal in production builds.
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
// dt2 of the most recent negotiate() call — used by the adaptive fairness
// gate (always-on, zero cost beyond one thread_local store).
inline thread_local uint64_t s_adapt_dt2_last_us       = 0;
inline thread_local int      s_adapt_C_last            = 0;  // popcount(tid_bitset)
inline thread_local uint32_t s_adapt_last_priority_tid = 0;  // last m_priority_tid seen
inline thread_local uint32_t s_adapt_bounce_count      = 0;  // # times it changed
inline thread_local uint64_t s_adapt_negotiate_calls   = 0;  // negotiate() entries
inline thread_local uint64_t s_adapt_skip_hits         = 0;  // lease-skip fires
inline thread_local uint32_t s_adapt_skip_per1k        = 0;  // skip_hits/calls × 1000
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
    // if the value actually changes. Schedule constants live in
    // transaction_definitions.h (KAME_LEASE_*).
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
    int C_obs = sig_C < 1 ? 1 : sig_C;

    for(int ms = 0;;) {
        if(entry_pr == Priority::HIGHEST)
            break;
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
        {
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

} // namespace Transactional

#endif /* TRANSACTION_NEG_IMPL_H */
