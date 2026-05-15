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
//     CV-based sleep/notify slots, priority_probe_info).
//   * Node<XN>::Linkage::negotiate_after_retry_pause()
//   * Node<XN>::Linkage::negotiate_internal()       (the big one,
//     including this branch's spin-for-same-kind shortcut).
// Plus the production-side `effective_runners`/`effective_min_runners`/
// `effective_max_runners` helpers and the KAME_ADAPT_INSTRUMENT
// diagnostic thread_local counters used by negotiate_internal.
//
// Class bodies / declarations live in transaction.h.  Tuning macros
// live in transaction_definitions.h.  Surrounding negotiation-layer
// types (retry_pause, ScopedNegotiateLinkage<XN>, effective_runners
// forward decl) live in transaction_negotiation.h.
//
// Included from transaction_impl.h after transaction.h and
// transaction_negotiation.h so all dependent types are visible.
// =====================================================================
#ifndef TRANSACTION_NEG_IMPL_H
#define TRANSACTION_NEG_IMPL_H

#include "transaction.h"
#include "transaction_definitions.h"
#include "transaction_negotiation.h"

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
struct Node<XN>::WalkUpResult {
    SnapshotStatus find_status;  //!< result of findChildSlot (or early-return status)
    SnapshotStatus status;       //!< status after convertRecursiveStatus (before find)
    bool is_root_level;          //!< true if this parent is the chain root
    shared_ptr<Linkage> parent_linkage;    //!< m_link of the parent node (= bundledBy)
    //! ScopedNeg on parent's linkage (1 CAS, with_negotiate=false).
    //! Provides contention tagging on DISTURBED unwind.
    //! Disengaged on early-return (DISTURBED/NODE_MISSING before acquire).
    std::optional<ScopedNegotiateLinkage<XN>> parent_scope;
    int reverse_index;
    local_shared_ptr<Packet> *parent_packet;  //!< parent's packet containing child slot
};

// =====================================================================
// Out-of-class template member definitions for Node<XN>::NegotiationCounter.
// Declarations live in transaction.h; bodies here pick up the namespace-
// scope `LivelockProbe::state()`, `LivelockProbe::now_us()`, and the
// `s_privileged_tidstamp` / `s_sleep_slots` C++17 inline static members
// (defined in the class body in transaction.h).
// =====================================================================

// KAME_STM_PRIV_PREEMPT_WINDOW_US and KAME_STM_PRIV_AGE_NORMAL_US
// (age-ordered preemption window + per-priority floor) live in
// transaction_definitions.h.

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
    Priority pr, cnt_t tidstamp, int sig_C) noexcept
{
    int64_t now_us = LivelockProbe::now_us();
    int64_t tx_age_us = (int64_t)diff_us_packed(now_us, tidstamp);
    // CAS-loop: claim the slot if empty, OR preempt the current holder
    // using age-ordered preemption (challenger must be older than holder
    // by ≥ PRIV_PREEMPT_WINDOW_US). This serialises privilege by age
    // while suppressing rapid cycling between contemporaneous threads.
    //
    // For the initial claim (empty slot), the age threshold is scaled
    // by max(1, N/4) where N = numThreadsRunning(). This prevents
    // privilege churn at high thread counts: with 128 threads on ~10
    // cores, many threads exceed the base 300µs threshold after just
    // one OS scheduling quantum and all race to claim the empty slot.
    // Scaling by N/4 raises the bar proportionally to system load.
    // At N=128: claim_floor = 300 * 32 = 9600µs ≈ 10ms, giving the
    // privileged thread enough scheduling slices to complete a
    // multi-CAS bundle before another thread can claim the slot.
    cnt_t expected = s_privileged_tidstamp.load(std::memory_order_relaxed);
    const int64_t age_floor = min_privilege_age_us(pr);
    // Scale initial-claim threshold by global thread count / 4.
    int N = (int)numThreadsRunning();
    int scale = N / 4;
    if (scale < 1) scale = 1;
    const int64_t claim_floor = age_floor * scale;
    while (true) {
        if (expected != (cnt_t)0) {
            // Slot held. Preempt only if the challenger (us) is older
            // than the holder by at least PRIV_PREEMPT_WINDOW_US.
            // Age-ordered preemption: older transactions take priority,
            // but a small window prevents rapid cycling between threads
            // of similar age. This replaces the old hard-expiry approach
            // (PRIV_EXPIRE_US) which was unresponsive (50 ms) and could
            // not distinguish OS-preempted holders from merely slow ones.
            int64_t holder_tx_age = (int64_t)diff_us_packed(now_us, expected);
            if (tx_age_us < holder_tx_age + (int64_t)KAME_STM_PRIV_PREEMPT_WINDOW_US)
                return false;  // holder is at least as old; don't preempt
        } else {
            // Empty slot. Require scaled age threshold to reduce churn
            // when many threads are contending.
            if (tx_age_us < claim_floor)
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
        "(claimed by stuck oldest Tx; age=%lld us, prio=%d, N=%d%s)\n",
        (unsigned)stamp_tid(tidstamp),
        (long long)tx_age_us, (int)pr, N,
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
    return stamp_tid(priv) != stamp_tid(tidstamp);
}

template <class XN>
bool Node<XN>::NegotiationCounter::i_am_privileged_now(cnt_t my_tidstamp) noexcept {
    // Compare by TID (upper bits) — the privileged Tx and a *nested* Tx
    // on the same thread carry different started_time stamps but share
    // the same TID. Either Tx is "privileged" for the purpose of choosing
    // STRONG-mode acquire/CAS: no other thread is doing CAS on the same
    // linkage (fair_mode blocks them), so a strong spin has no peer.
    cnt_t priv = s_privileged_tidstamp.load(std::memory_order_relaxed);
    if(priv == (cnt_t)0) return false;
    return stamp_tid(priv) == stamp_tid(my_tidstamp);
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
    int sig_C,
    int64_t tx_age_us,
    Priority prio) noexcept
{
    auto &p = LivelockProbe::state();
    if (p.linkage_id != linkage) {
        p.linkage_id       = linkage;
        p.t_window_us      = LivelockProbe::now_us();
        p.tx_retry_window  = my_tx_retries;
        p.tx_commit_window = tx_commit_count;
        return false;
    }
    int64_t now_us    = LivelockProbe::now_us();
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

    // Dynamic LL-probe retry threshold: each peer contributes ~2
    // expected CAS retries (bidirectional contention), capped at
    // hardware_concurrency() since beyond that count, threads can't all
    // be physically running CAS simultaneously. Floor 3 keeps the
    // early-call (sig_C ≈ 0) path safe before the bitset has accumulated
    // peers. Machine-generic: no per-platform tuning constants — the
    // hardware_concurrency() call adapts to SMT / core count.
    int hw_procs = (int)std::thread::hardware_concurrency();
    if (hw_procs <= 0) hw_procs = 4;
    int retry_thresh_dyn = sig_C * 2;
    if (retry_thresh_dyn < 3) retry_thresh_dyn = 3;
    if (retry_thresh_dyn > hw_procs) retry_thresh_dyn = hw_procs;

    const char *verdict =
        (tags_total > 0 && tags_owned == tags_total
         && (int)my_tx_retries >= retry_thresh_dyn
         && tx_age_us > min_privilege_age_us(prio))
            ? "LIVELOCK" : "ok";

    if(window_us > 100'000)
        if(verdict[0] == 'L')
            std::fprintf(stderr,
                "[ll-probe] tid=%u linkage=%p prio=%s threshold=%d (sig_C=%d) "
                "my_tx_retries=%u my_tx_retry_rate=%.0f/s "
                "tx_commit_rate=%.0f/s ratio=%.1f "
                "tags_owned=%d/%d tx_age_us=%lld "
                "verdict=%s window_ms=%lld\n",
                (unsigned)ProcessCounter::id(), linkage,
                pinfo.name, retry_thresh_dyn, sig_C,
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
void Node<XN>::NegotiationCounter::notify_n_contenders(
    const TidBitset &tid_bitset, int n) noexcept
{
    // Fair-mode escape: if a privileged TID is registered, wake its
    // sleep slot first so the stuck oldest Tx gets a chance to retry
    // ahead of the rest of the bitset.
    uint16_t priv_tid = stamp_tid(
        s_privileged_tidstamp.load(std::memory_order_relaxed));
    int priv_slot = -1;
    if (priv_tid != 0 && n > 0) {
        priv_slot = (int)(((unsigned)priv_tid) % NEGOTIATE_SLEEP_SLOTS);
        auto &st = s_sleep_slots[priv_slot];
        { std::lock_guard<std::mutex> lk(st.mtx); st.notified = true; }
        st.cv.notify_one();
        --n;
    }
    for(int i = 0; i < TidBitset::WORDS && n > 0; ++i) {
        uint64_t word = tid_bitset.word(i);
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
void Node<XN>::NegotiationCounter::try_notify_n_contenders(
    const TidBitset &tid_bitset, int n) noexcept
{
    for(int i = 0; i < TidBitset::WORDS && n > 0; ++i) {
        uint64_t word = tid_bitset.word(i);
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

// KAME_LEASE_US_MIN / KAME_LEASE_US_MAX live in transaction_definitions.h.

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
thread_local uint32_t s_adapt_last_priority_tid = 0;  // last m_priority_state.tid seen
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
    // Single now_us() snapshot: livelock-probe window, livelock age and
    // the per-call-site adaptive NORMAL-lease expiry check below all
    // read it.  The few µs between these reads in the original code
    // carried no useful information (no observable state changes
    // between them).
    const int64_t now_us_entry = Node<XN>::NegotiationCounter::now_us();
    // Priority is a per-thread/per-Tx invariant for the duration of this
    // call: read it once and reuse for both the livelock-probe block and
    // the per-call-site adaptive gate decision below.
    const Priority entry_pr = getCurrentPriorityMode();

    // Compute popcount once per call; the live tid_bitset is unchanged
    // until the loop body's first iteration adds new entries.
    int sig_C = tid_bitset.popcount();
    // No pre-loop yield: the m_transaction_started_time load below is
    // the cheap collision-clear check.
    // tx age = wall time since the Snapshot/Transaction ctor stamped
    // m_started_time. The field is set by BOTH Snapshot(const Node&)
    // and Transaction ctors and is not reset by operator++ — so the
    // probe's `tx_age_us` printout is really "Snapshot/Tx age". The
    // `tx_` label is kept for log-format continuity.
    // m_started_time is a tid+kind+us-packed stamp; diff_us_packed
    // extracts the µs component and applies modular subtraction
    // (wrap-safe at US_BITS = 46).
    int64_t _ll_age_us =
        (int64_t)NegotiationCounter::diff_us_packed(now_us_entry, started_time);
    if (_ll_age_us >= NegotiationCounter::min_privilege_age_us(Priority::HIGHEST)
        && !snap.m_tagged_linkages.empty()) {  // skip when too young or untagged
        // Count tagged linkages whose m_transaction_started_time == ours
        // (= "priority is already mine on every linkage" = primary
        //   livelock precondition per the refined definition).
        // Identity check ignores kind bits — see drop_tags_n_privilege.
        const auto _ll_my_id = NegotiationCounter::strip_kind(
                                    snap.m_started_time);
        int _ll_total = (int)snap.m_tagged_linkages.size();
        int _ll_owned = 0;
        for (auto &_l : snap.m_tagged_linkages) {
            if (_l && NegotiationCounter::strip_kind(
                    _l->m_transaction_started_time.load(
                        std::memory_order_relaxed)) == _ll_my_id)
                ++_ll_owned;
        }
        // `entry_pr` was read once at function entry; the probe maps it
        // to retry-threshold / label internally.
        bool _ll_saw = NegotiationCounter::livelock_probe_tx_tick(
            static_cast<const void*>(this),
            snap.m_tx_retry_count,
            m_tx_commit_count,
            _ll_owned, _ll_total, sig_C, _ll_age_us,
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
            // (No adaptive-state reset here.)  Privileged threads
            // skip the gate-decision block entirely (guarded by
            // `!_fair_blocks && !snap.m_registered_privileged`) and
            // fall through to the adaptive sleep automatically.  Once
            // privilege is released at Tx finalize/dtor, the next
            // Tx starts with default per-site state = GATE — i.e.
            // step-1 behaviour (my_kind != NONE → gate-return) is the
            // initial / post-privilege resting state, and NORMAL
            // engages only via the streak+time path below.
        }
    }

    // Always-on adaptive path: the V0 (legacy) path and the V0↔ADAPTIVE
    // mode switch were removed in favour of the orthogonal fair-mode
    // escape (s_privileged_tidstamp). See top of detail:: in this file.
  { // adaptive-path scope
    // One atomic load of the packed (tid | lease_us | start_us) tuple.
    auto ps = loadPriority();
    if(ps.tid) {
        tid_bitset.observe((unsigned)ps.tid);
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
    // transaction_started_time is tid+kind+us-packed; diff_us_packed
    // extracts the µs and applies modular subtraction (wrap-safe).
    auto adapt_dt2_last_us =
        NegotiationCounter::diff_us_packed(
            now_us_entry, transaction_started_time);

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
    // KAME_STM_C_OBS_MIN lives in transaction_definitions.h.
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
        // Both stamps are tid+kind+us-packed; signed_diff_us_packed
        // returns (my_us - other_us) interpreted as a signed wrap-safe
        // delta.  dt <= 0  ⇒  I am oldest (or equal) → break.
        int64_t dt = NegotiationCounter::signed_diff_us_packed(
            started_time, transaction_started_time);
        if(dt <= 0)
            break; //This thread is the oldest.
        auto transaction_started_time =
            m_transaction_started_time.load(std::memory_order_acquire);
        if( !transaction_started_time)
            break; //collision has not been detected.

        auto dt2 = NegotiationCounter::diff_us_packed(
            Node<XN>::NegotiationCounter::now_us(),
            transaction_started_time);

        // Fair-mode escape: when some other thread holds the privileged-
        // TID slot, suppress the jittered gate and the √C lottery so the
        // privileged Tx alone gets to commit. Strict Greedy CM (older
        // started_time wins → I sleep below) is the only mechanism left
        // to allocate priority while fair-mode is active.
        const bool _fair_blocks = NegotiationCounter::fair_mode_blocks_me(started_time);

        NegSite::last_was_gate_return() = false;
#if KAME_LEGACY_GATING
        // ===== Legacy per-site adaptive gating ============================
        // Deprecation candidate; superseded by the per-Linkage spin-for-
        // same-kind path further down.  Enable with -DKAME_LEGACY_GATING=1
        // for A/B regression.  Tri-state `take_gate` per call-site:
        //
        //   -1 (UNDEFINED)   : initial / post-privilege state — the
        //                      hot-path decides by my_kind alone
        //                      (non-NONE → gate, NONE → sleep).
        //    0 (FORCE_SLEEP) : forced sleep, time-leased.  Set by
        //                      K_FAIL gate→fail streak inside
        //                      FAIL_WINDOW_US (see _on_cas_fail);
        //                      auto-reverts to UNDEFINED at lease
        //                      expiry, or on privilege observation
        //                      (the streak history is stale once
        //                      contention enters the privilege path).
        //    1 (FORCE_GATE)  : forced bypass via the `break` below.
        //
        // Empirically (KAME_ADAPT_INSTRUMENT, N=4-128 × CR=1-20):
        //   - kind-gated (peer == my || peer == MNC) was too conservative
        //   - all-gate crashed my=NONE fairness (stand-alone read/release)
        //   - my != NONE alone was the goldilocks (+9-23 % over kind-gated).
        // ------------------------------------------------------------------
        auto *_adapt = NegSite::current_state();
        if(_fair_blocks || snap.m_registered_privileged) {
            // Privilege observed at this site → reset to UNDEFINED.
            if(_adapt && _adapt->take_gate != -1) {
                _adapt->take_gate = -1;
                _adapt->consec_fails = 0;
                _adapt->consec_succs = 0;
                ++_adapt->mode_flips_n2g;
            }
        } else {
            // Lease expiry: any FORCE state → UNDEFINED.
            if(_adapt && _adapt->take_gate != -1
               && (uint64_t)now_us_entry >= _adapt->normal_until_us) {
                _adapt->take_gate = -1;
                _adapt->consec_fails = 0;
                _adapt->consec_succs = 0;
                ++_adapt->mode_flips_n2g;
            }
            const detail::StampKind my_kind = detail::s_current_op_kind;
            bool take_gate;
            const int8_t tg = _adapt ? _adapt->take_gate : (int8_t)-1;
            if(tg == -1) {
                // UNDEFINED → my_kind decides; cache verdict with a
                // fresh lease so subsequent callers follow it for
                // NegSite::NORMAL_LEASE_US, then re-evaluate.
                take_gate = (my_kind != detail::StampKind::NONE);
                if(_adapt) {
                    _adapt->take_gate = take_gate ? (int8_t)1 : (int8_t)0;
                    _adapt->normal_until_us =
                        (uint64_t)now_us_entry
                        + (uint64_t)NegSite::NORMAL_LEASE_US;
                    _adapt->consec_fails = 0;
                    _adapt->consec_succs = 0;
                    if(take_gate) ++_adapt->mode_flips_promote;
                    else          ++_adapt->mode_flips_g2n;
                }
            } else {
                take_gate = (tg != 0);     // 0 = FORCE_SLEEP, 1 = FORCE_GATE
            }
            if(take_gate)
                NegSite::last_was_gate_return() = true;
            if(_adapt) {
                const detail::StampKind peer_kind = (detail::StampKind)
                    NegotiationCounter::stamp_kind(transaction_started_time);
                if(take_gate)
                    ++_adapt->gate_returns_by_peer[(int)peer_kind & 3];
                else
                    ++_adapt->blocked_by_peer[(int)peer_kind & 3];
            }
            if(take_gate) break;
            // Otherwise fall through to adaptive sleep (FORCE_SLEEP
            // or UNDEFINED-with-my_kind-NONE).
        }
        // ===== end legacy gating ==========================================
#endif // KAME_LEGACY_GATING

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
    // KAME_STM_DISABLE_LOTTERY lives in transaction_definitions.h.
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

        // --- Spin-for-peer-progress shortcut.
        //
        // Two distinct policies, selected at compile time:
        //
        //  (A) KAME_COALESCE_MODE != 0 — kind-match coalesce
        //      Trigger fires iff peer is doing the SAME kind of
        //      operation as us (peer_kind == my_kind, both != NONE).
        //      Rationale: when the slot holder is about to commit a
        //      same-kind op, we bet that stepping in right after them
        //      is cheaper than negotiate_sleep + CV wake.  When the
        //      kinds differ, spin has no payoff — we'd retry the CAS
        //      only to lose to a different kind anyway.
        //
        //      Win condition (K1 strict / K4 blind): peer released
        //          (slot == 0).
        //      Lose condition (K1):  peer_kind changed mid-spin (some
        //          other thread stepped in with a different op).
        //      Loose mode (K2):  also treat kind change as win.
        //      Blind mode (K4):  no in-loop reads — relevant for ARM
        //          where polling the slot triggers holder stlxr fails.
        //
        //  (B) KAME_COALESCE_MODE == 0 — legacy any-change spin
        //      Trigger: this Linkage flipped within the last
        //      KAME_SPIN_RECENT_FLIP_US µs (wall-clock age) AND
        //      ops_since_flip < sig_C × 8.  We spin on
        //      m_transaction_started_time waiting for ANY change
        //      from its initial value (release OR different stamp).
        //      Effectively disabled when KAME_SPIN_RECENT_FLIP_US == 0.
        //
        // Budget = min(flip_period_us_ema, KAME_{SPIN,COALESCE}_MAX_US)
        // (per-Linkage EMA period as proxy for typical scope hold time).
        {
            using L = Linkage;
            const uint64_t fs = m_flip_state.load(std::memory_order_relaxed);
            const uint64_t fs_last_us =
                (fs >> L::FLIP_LAST_US_SHIFT) & L::FLIP_LAST_US_MASK;
            const uint64_t fs_ops_since =
                (fs >> L::FLIP_OPS_SHIFT) & 0xFFu;
            const uint64_t now_lo =
                (uint64_t)NegotiationCounter::now_us() & L::FLIP_LAST_US_MASK;
            const uint64_t age =
                (now_lo - fs_last_us) & L::FLIP_LAST_US_MASK;
            NegSite::SpinOutcome outcome;
#if KAME_COALESCE_MODE != 0
            // ===== (A) Kind-match coalesce ============================
            const auto slot_now =
                m_transaction_started_time.load(std::memory_order_relaxed);
            const uint8_t my_kind  =
                (uint8_t)detail::s_current_op_kind & 0x3u;
            const uint8_t peer_kind =
                NegotiationCounter::stamp_kind(slot_now);
            if(fs == 0) {
                outcome = NegSite::SpinOutcome::SKIPPED_NO_PERIOD;
            } else if(my_kind == 0
                      || peer_kind != my_kind
                      || age > (uint64_t)KAME_COALESCE_RECENT_US
                      || fs_ops_since >= (uint64_t)(sig_C * 8u)) {
                outcome = NegSite::SpinOutcome::SKIPPED_COLD;
            } else if(((fs >> L::FLIP_PERIOD_SHIFT)
                       & L::FLIP_PERIOD_MASK) > 0
                      && ((fs >> L::FLIP_PERIOD_SHIFT)
                          & L::FLIP_PERIOD_MASK)
                         < (uint64_t)(sig_C * KAME_THRASHING_C_MULT)) {
                // Over-thrashing: period (µs) shorter than 2 × thread
                // count means even with a 2-period spin budget we'd
                // still race more flips than we could catch.  Defer
                // to negotiate_sleep.  (Bare sig_C is too conservative
                // for BUBU rapid-alternation patterns where 1 period
                // ≈ N µs is still spin-recoverable.)
                outcome = NegSite::SpinOutcome::SKIPPED_THRASHING;
            } else {
                const uint64_t fs_period =
                    (fs >> L::FLIP_PERIOD_SHIFT) & L::FLIP_PERIOD_MASK;
                // Budget = min(fs_period * BUDGET_PCT / 100, MAX_US).
                // Polled defaults to 100 % (one period; early-exit
                // makes over-shooting cheap), blind defaults to 75 %
                // (no early-exit so cap waste).  Override
                // KAME_COALESCE_BUDGET_PCT to widen (e.g. 200 = 2
                // periods on x86 to raise coalesce hit rate).
                const uint64_t period_cap =
                    (fs_period * (uint64_t)KAME_COALESCE_BUDGET_PCT) / 100u;
                const uint64_t budget =
                    (fs_period > 0
                     && period_cap < (uint64_t)KAME_COALESCE_MAX_US)
                    ? period_cap
                    : (uint64_t)KAME_COALESCE_MAX_US;
                const uint64_t start_us =
                    (uint64_t)NegotiationCounter::now_us();
                const uint64_t deadline = start_us + budget;
                bool won = false;
#if KAME_COALESCE_MODE == 4
                // K4 blind: no in-loop reads; single check at end.
                while((uint64_t)NegotiationCounter::now_us() < deadline) {
                    for(int i = 0; i < 16; ++i) pause4spin();
                }
                {
                    auto t = m_transaction_started_time.load(
                        std::memory_order_relaxed);
                    won = (!t)
                        || (NegotiationCounter::stamp_kind(t) != my_kind);
                }
#else
                // K1 (strict) or K2 (loose) polled.
                do {
                    for(int i = 0; i < 16; ++i) pause4spin();
                    auto t = m_transaction_started_time.load(
                        std::memory_order_relaxed);
                    if(!t) { won = true; break; }     // peer released
                    if(NegotiationCounter::stamp_kind(t) != my_kind) {
#  if KAME_COALESCE_MODE == 2
                        won = true; break;            // K2 loose: also win
#  else
                        won = false; break;           // K1 strict: lose
#  endif
                    }
                } while((uint64_t)NegotiationCounter::now_us() < deadline);
#endif // KAME_COALESCE_MODE
                const uint64_t end_us = (uint64_t)NegotiationCounter::now_us();
                const uint32_t elapsed =
                    (uint32_t)(end_us > start_us ? end_us - start_us : 0);
                outcome = won
                    ? NegSite::SpinOutcome::WON
                    : NegSite::SpinOutcome::TIMEOUT;
                NegSite::record_spin_event(outcome, elapsed);
                if(won) break;   // gate-return: caller retries CAS
            }
#else // KAME_COALESCE_MODE == 0
            // ===== (B) Legacy any-change spin =========================
            if(fs == 0) {
                outcome = NegSite::SpinOutcome::SKIPPED_NO_PERIOD;
            } else if(age > (uint64_t)KAME_SPIN_RECENT_FLIP_US
                      || fs_ops_since >= (uint64_t)(sig_C * 8u)) {
                outcome = NegSite::SpinOutcome::SKIPPED_COLD;
            } else if(((fs >> L::FLIP_PERIOD_SHIFT)
                       & L::FLIP_PERIOD_MASK) > 0
                      && ((fs >> L::FLIP_PERIOD_SHIFT)
                          & L::FLIP_PERIOD_MASK)
                         < (uint64_t)(sig_C * KAME_THRASHING_C_MULT)) {
                // Over-thrashing: see (A) for rationale.
                outcome = NegSite::SpinOutcome::SKIPPED_THRASHING;
            } else {
                const uint64_t fs_period =
                    (fs >> L::FLIP_PERIOD_SHIFT) & L::FLIP_PERIOD_MASK;
                // Legacy path is polled.  budget = min(fs_period *
                // KAME_SPIN_BUDGET_PCT / 100, KAME_SPIN_MAX_US).
                const uint64_t period_cap =
                    (fs_period * (uint64_t)KAME_SPIN_BUDGET_PCT) / 100u;
                const uint64_t budget =
                    (fs_period > 0
                     && period_cap < (uint64_t)KAME_SPIN_MAX_US)
                    ? period_cap
                    : (uint64_t)KAME_SPIN_MAX_US;
                const uint64_t start_us =
                    (uint64_t)NegotiationCounter::now_us();
                const uint64_t deadline = start_us + budget;
                const auto initial_t =
                    m_transaction_started_time.load(std::memory_order_relaxed);
                bool won = false;
                do {
                    for(int i = 0; i < 16; ++i) pause4spin();
                    auto t = m_transaction_started_time.load(
                        std::memory_order_relaxed);
                    if(!t) { won = true; break; }     // Linkage released
                    if(t != initial_t) { won = true; break; }  // peer changed
                } while((uint64_t)NegotiationCounter::now_us() < deadline);
                const uint64_t end_us = (uint64_t)NegotiationCounter::now_us();
                const uint32_t elapsed =
                    (uint32_t)(end_us > start_us ? end_us - start_us : 0);
                outcome = won
                    ? NegSite::SpinOutcome::WON
                    : NegSite::SpinOutcome::TIMEOUT;
                NegSite::record_spin_event(outcome, elapsed);
                if(won) break;   // gate-return: caller retries CAS
            }
#endif // KAME_COALESCE_MODE
            if(outcome != NegSite::SpinOutcome::WON
               && outcome != NegSite::SpinOutcome::TIMEOUT) {
                NegSite::record_spin_event(outcome, 0);
            }
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

} // namespace Transactional

#endif /* TRANSACTION_NEG_IMPL_H */
