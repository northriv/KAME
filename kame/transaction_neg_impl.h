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
bool Node<XN>::NegotiationCounter::i_am_privileged_now(
        cnt_t my_tidstamp,
        const Linkage *link) noexcept {
#if KAME_PER_LINKAGE_PRIVILEGE
    // Per-Linkage: "I am privileged" iff this Linkage's slot carries
    // a Reserved-kind stamp whose (us, tid) identity matches mine.
    if(link == nullptr) return false;
    cnt_t slot = link->m_transaction_started_time.load(std::memory_order_relaxed);
    if( !is_priv_stamp(slot)) return false;
    return strip_kind(slot) == strip_kind(my_tidstamp);
#else
    (void)link;
    cnt_t priv = s_privileged_tidstamp.load(std::memory_order_relaxed);
    if(priv == (cnt_t)0) return false;
    return stamp_tid(priv) == stamp_tid(my_tidstamp);
#endif
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
bool Node<XN>::NegotiationCounter::fair_mode_blocks_me(
        cnt_t tidstamp,
        const Linkage *link) noexcept {
#if KAME_PER_LINKAGE_PRIVILEGE
    // Per-Linkage: check the linkage's own slot for a Reserved-kind
    // stamp held by some other thread.  Nested Txs on the same TID
    // are NOT blocked (strip_kind identity match returns "us").
    if(link == nullptr) return false;
    cnt_t slot = link->m_transaction_started_time.load(std::memory_order_relaxed);
    if( !is_priv_stamp(slot)) return false;
    return strip_kind(slot) != strip_kind(tidstamp);
#else
    (void)link;
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
#endif
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

    // Age condition (`tx_age_us > min_privilege_age_us(prio)`)
    // dropped — claim eligibility now depends on tag-ownership +
    // retry count.  `tx_age_us` is still logged below for diagnostic.
    const char *verdict =
        (tags_total > 0 && tags_owned == tags_total
         && (int)my_tx_retries >= retry_thresh_dyn)
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
void Node<XN>::NegotiationCounter::negotiate_sleep(
    int ms_timeout, uint64_t my_stamp) noexcept
{
    int slot = (int)((unsigned)ProcessCounter::id() % NEGOTIATE_SLEEP_SLOTS);
    auto &st = s_sleep_slots[slot];
    // Snapshot the kind this thread is about to commit; the notifier
    // reads this field under the slot lock to bias wake-up toward the
    // same kind as the linkage's most recent commit (see
    // `notify_n_contenders` preferred_kind argument).
    const uint8_t my_kind = (uint8_t)detail::s_current_op_kind & 0x3u;
    std::unique_lock<std::mutex> lock(st.mtx);
    st.op_kind = my_kind;
    // Publish the tenant stamp under the lock so wakers (also holding
    // the lock) can verify they are notifying the intended thread on a
    // `tid % N_SLOTS` hash collision.
    st.stamp = my_stamp;
    // Reset under the lock so a notify delivered between the previous
    // call's wake and this reset is not silently consumed.
    st.notified = false;
    st.cv.wait_for(lock, std::chrono::milliseconds(ms_timeout),
                   [&]{ return st.notified; });
    // Clear the tenant stamp on exit so the next sleeper's stamp
    // is not preceded by a stale value that could match a target.
    st.stamp = 0;
}

template <class XN>
void Node<XN>::NegotiationCounter::notify_n_contenders(
    const TidBitset &tid_bitset, int n, uint8_t preferred_kind) noexcept
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
    // Two-pass walk when a preferred kind is supplied: pass 1 wakes
    // only kind-matching slots; pass 2 wakes any remaining slots.
    // `woken` tracks which slot indices were already notified by pass
    // 1 so pass 2 doesn't burn the budget redundantly.
    const bool has_pref = (preferred_kind <= 2u);
    uint64_t woken[NEGOTIATE_SLEEP_SLOTS / 64] = {0};
    auto mark_woken = [&](int slot) {
        woken[slot >> 6] |= (uint64_t)1u << (slot & 63);
    };
    auto is_woken = [&](int slot) -> bool {
        return (woken[slot >> 6] >> (slot & 63)) & 1u;
    };
    if(has_pref) {
        for(int i = 0; i < TidBitset::WORDS && n > 0; ++i) {
            uint64_t word = tid_bitset.word(i);
            while(word && n > 0) {
                int bit = __builtin_ctzll(word);
                word &= word - 1;
                int slot = (int)(((unsigned)(i * 64 + bit)) % NEGOTIATE_SLEEP_SLOTS);
                if (slot == priv_slot) continue;
                if (is_woken(slot)) continue;
                auto &st = s_sleep_slots[slot];
                {
                    std::lock_guard<std::mutex> lk(st.mtx);
                    if(st.op_kind != preferred_kind) continue;
                    st.notified = true;
                }
                st.cv.notify_one();
                mark_woken(slot);
                --n;
            }
        }
    }
    for(int i = 0; i < TidBitset::WORDS && n > 0; ++i) {
        uint64_t word = tid_bitset.word(i);
        while(word && n > 0) {
            int bit = __builtin_ctzll(word);
            word &= word - 1;
            int slot = (int)(((unsigned)(i * 64 + bit)) % NEGOTIATE_SLEEP_SLOTS);
            if (slot == priv_slot) continue;
            if (has_pref && is_woken(slot)) continue;
            auto &st = s_sleep_slots[slot];
            { std::lock_guard<std::mutex> lk(st.mtx); st.notified = true; }
            st.cv.notify_one();
            if(has_pref) mark_woken(slot);
            --n;
        }
    }
}

template <class XN>
void Node<XN>::NegotiationCounter::try_notify_n_contenders(
    const TidBitset &tid_bitset, int n, uint8_t preferred_kind) noexcept
{
    const bool has_pref = (preferred_kind <= 2u);
    uint64_t woken[NEGOTIATE_SLEEP_SLOTS / 64] = {0};
    auto mark_woken = [&](int slot) {
        woken[slot >> 6] |= (uint64_t)1u << (slot & 63);
    };
    auto is_woken = [&](int slot) -> bool {
        return (woken[slot >> 6] >> (slot & 63)) & 1u;
    };
    if(has_pref) {
        for(int i = 0; i < TidBitset::WORDS && n > 0; ++i) {
            uint64_t word = tid_bitset.word(i);
            while(word && n > 0) {
                int bit = __builtin_ctzll(word);
                word &= word - 1;
                int slot = (int)(((unsigned)(i * 64 + bit)) % NEGOTIATE_SLEEP_SLOTS);
                if (is_woken(slot)) continue;
                auto &st = s_sleep_slots[slot];
                std::unique_lock<std::mutex> lk(st.mtx, std::try_to_lock);
                if( !lk.owns_lock()) continue;
                if(st.op_kind != preferred_kind) continue;
                st.notified = true;
                lk.unlock();
                st.cv.notify_one();
                mark_woken(slot);
                --n;
            }
        }
    }
    for(int i = 0; i < TidBitset::WORDS && n > 0; ++i) {
        uint64_t word = tid_bitset.word(i);
        while(word && n > 0) {
            int bit = __builtin_ctzll(word);
            word &= word - 1;
            int slot = (int)(((unsigned)(i * 64 + bit)) % NEGOTIATE_SLEEP_SLOTS);
            if (has_pref && is_woken(slot)) continue;
            auto &st = s_sleep_slots[slot];
            std::unique_lock<std::mutex> lk(st.mtx, std::try_to_lock);
            if( !lk.owns_lock()) continue;
            st.notified = true;
            lk.unlock();
            st.cv.notify_one();
            if(has_pref) mark_woken(slot);
            --n;
        }
    }
}

#if (KAME_STM_MIN_RUNNERS != 0) || (KAME_STM_MAX_RUNNERS != 0)
    // Running maximum of observed C (contender count), used as fallback when
    // hardware_concurrency() returns 0.
    alignas(KAME_CACHE_LINE) std::atomic<int> s_max_c_obs{1};

    // Spinners actively busy-polling the per-Linkage privilege state.
    // Inc/dec around the fair-spin block in `_negotiate_internal`.
    alignas(KAME_CACHE_LINE) std::atomic<unsigned> s_fair_spinners{0};

    // Threads currently holding per-Linkage privilege on at least one
    // Linkage.  Unrelated Linkages can be claimed independently, so the
    // count can grow up to `numThreadsRunning()` in principle.  Used
    // to subtract from the fair-spin admission cap: spinners +
    // priv-holders together should not exceed `effective_runners`.
    alignas(KAME_CACHE_LINE) std::atomic<unsigned> s_num_privileged_threads{0};
#endif // (KAME_STM_MIN_RUNNERS != 0) || (KAME_STM_MAX_RUNNERS != 0)

template <class XN>
void Node<XN>::NegotiationCounter::release_priv_count_slot() noexcept {
#if (KAME_STM_MIN_RUNNERS != 0) || (KAME_STM_MAX_RUNNERS != 0)
    s_num_privileged_threads.fetch_sub(1, std::memory_order_relaxed);
#endif
}

#if (KAME_STM_MIN_RUNNERS != 0) || (KAME_STM_MAX_RUNNERS != 0)
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

// Fast-path adaptive-backoff entry point.  Short-circuits when no peer
// Tx has tagged this Linkage; otherwise calls `_negotiate_internal()`.
// The relaxed load on m_transaction_started_time is the same one
// `_negotiate_internal` would do first, so the collision path pays no
// extra.  `is_active_stamp(s)` is just `s != 0` — release zero-stores
// the slot, so any non-zero word means "tagged".
template <class XN>
void
ScopedNegotiateLinkage<XN>::_negotiate() noexcept {
#if defined(KAME_STM_DISABLE_BACKOFF) && KAME_STM_DISABLE_BACKOFF
    return;
#else
    using NC = typename Node<XN>::NegotiationCounter;
    if( !NC::is_active_stamp(
            m_link->m_transaction_started_time.load(std::memory_order_relaxed)))
        [[likely]]
        return;
    _negotiate_internal();
#endif
}

// Unified retry-loop backoff: always call retry_pause + negotiate.
// retry==0 → fast-path return UNLESS another Tx currently holds the
// fair-mode privileged slot. The yield is part of the livelock-free
// guarantee: when a stuck Tx claims privilege, all other Txs must
// release their CAS pressure so the privileged commit can succeed.
// retry>0 always runs retry_pause + negotiate.
template <class XN>
void
ScopedNegotiateLinkage<XN>::_negotiate_after_retry_pause(int retry) noexcept {
    using NC = typename Node<XN>::NegotiationCounter;
    if(retry == 0
        && !NC::fair_mode_blocks_me(m_snap->m_started_time, m_link.get()))
        [[likely]] return;  // fast path; zero-overhead steady state
    retry_pause(retry);
    _negotiate();
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

// Per-Linkage privilege diagnostic counters declared in transaction_impl.h
// (g_neg_claim_attempts / g_neg_claim_successes /
//  g_neg_internal_calls_non_priv / g_neg_internal_calls_priv).
extern std::atomic<uint64_t> g_neg_claim_attempts;
extern std::atomic<uint64_t> g_neg_claim_successes;
extern std::atomic<uint64_t> g_neg_internal_calls_non_priv;
extern std::atomic<uint64_t> g_neg_internal_calls_priv;
thread_local uint64_t s_adapt_skip_hits         = 0;  // lease-skip fires
thread_local uint32_t s_adapt_skip_per1k        = 0;  // skip_hits/calls × 1000
#endif
//=============================================================================
// _neg_apply_lease() — per-Linkage adaptive lease drift + owner-skip
//
// Updates `ps.lease_us` by drifting it up (sig_C >= 2) or down
// (sig_C == 0) using the KAME_LEASE_GROW_* / KAME_LEASE_SHRINK_PERCENT
// schedule, writing back via `storePriority` when the delta crosses
// the quantum (KAME_LEASE_QWRITE_US).
//
// Then, when our TID matches the recorded committer and our Tx age
// is below `ps.lease_us`, fires the owner-skip → caller returns
// early.  This is the soft "this thread just committed; let it chain
// a follow-up" fairness gate.
//
// LOWEST / UI_DEFERRABLE skip the whole block (priority-tag CAS,
// lease tracking, fairness gate, owner-skip).  When
// KAME_PRIORITY_LEASE is not defined, the helper is a no-op
// (returns false).
//=============================================================================
template <class XN>
bool
ScopedNegotiateLinkage<XN>::_neg_apply_lease(
    typename Node<XN>::Linkage::PriorityState &ps,
    typename Node<XN>::NegotiationCounter::cnt_t transaction_started_time,
    int sig_C,
    int64_t now_us_entry,
    Priority entry_pr) noexcept {
#ifdef KAME_PRIORITY_LEASE
    using NegotiationCounter = typename Node<XN>::NegotiationCounter;
    using Linkage = typename Node<XN>::Linkage;
    Linkage *const self = m_link.get();
    if(entry_pr == Priority::LOWEST || entry_pr == Priority::UI_DEFERRABLE)
        return false;
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
        typename Linkage::PriorityState drifted = ps;
        drifted.lease_us = new_lease_us;
        self->storePriority(drifted);
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
                return true;  // owner-skip → caller returns early
        }
    }
    return false;
#else
    (void)ps; (void)transaction_started_time; (void)sig_C;
    (void)now_us_entry; (void)entry_pr;
    return false;
#endif
}

//=============================================================================
// _neg_spin_block() — unified PRE-spin band gate + any-change spin shortcut
//
// band [LOW, HIGH>>tighten] gates whether we attempt the spin at all —
// failed gate routes to CV-sleep instead.  spin-win (= peer's
// m_recent_ops_state changed during our budget) and the speculative
// "no-spin gate-return" are the SAME break path: they only differ in
// the initial-time spent.
//
// Period (= spin budget proxy) = (2 × window_us) / total_count.
// Counts are the per-kind windowed counters in m_recent_ops_state
// (same-kind consecutive events filtered out at record time, so
// count == flip count).
//
//   tighten ramps up on each detected fail (previous break-for-CAS
//   didn't reach a CAS success); CAS success resets tighten=0 (see
//   `_on_cas_success`).  The effective HIGH band narrows right-shift
//   per step.
//
//   Below LOW       → SKIPPED_NO_PERIOD
//   Above HIGH      → SKIPPED_THRASHING (hyper-thrash)
//   Runners cap hit → SKIPPED_THRASHING (CAS-storm risk)
//   In-band & runners ok → spin → WON / TIMEOUT
//
// The entire body is compiled out when KAME_ENABLE_SPIN_BAND_GATE=0
// — see the master-enable comment in transaction_definitions.h.
//=============================================================================
#if KAME_ENABLE_SPIN_BAND_GATE
template <class XN>
bool
ScopedNegotiateLinkage<XN>::_neg_spin_block(int C_obs) noexcept {
    using NegotiationCounter = typename Node<XN>::NegotiationCounter;
    using Linkage = typename Node<XN>::Linkage;
    using L = Linkage;
    Linkage *const self = m_link.get();
    Snapshot<XN> &snap = *m_snap;

    const uint64_t fs = self->m_recent_ops_state.load(std::memory_order_acquire);
    // Decode windowed counts.  The state now carries a single
    // 16-bit merged flip count per window (BUNDLE and UNBUNDLE
    // share the slot — kind-specific filtering happens via
    // `latest_kind` below).  Apply rotation logic at READ time.
    const uint64_t now_us_full =
        (uint64_t)NegotiationCounter::now_us();
    const uint8_t  now_epoch = (uint8_t)((now_us_full / KAME_KIND_WINDOW_US) & 0xFFu);
    const uint8_t  cur_epoch = (uint8_t)((fs >> L::RSO_CUR_EPOCH_SHIFT)
                                          & L::RSO_EPOCH_MASK);
    const uint8_t  delta_ep = (uint8_t)((now_epoch - cur_epoch) & 0xFFu);
    uint64_t eff_count = 0;
    if(fs != 0) {
        if(delta_ep == 0) {
            eff_count = ((fs >> L::RSO_CUR_COUNT_SHIFT)  & L::RSO_COUNT_MASK)
                      + ((fs >> L::RSO_PREV_COUNT_SHIFT) & L::RSO_COUNT_MASK);
        } else if(delta_ep == 1) {
            // cur (= window now-1) → effective prev only.
            eff_count = (fs >> L::RSO_CUR_COUNT_SHIFT) & L::RSO_COUNT_MASK;
        }
        // delta >= 2: all stale, all zeros.
    }

    const uint8_t mk = (uint8_t)detail::s_current_op_kind & 0x3u;
    const bool prev_failed = snap.m_last_gate_returned;
    if(prev_failed) {
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
        // Log the tighten depth right when prev_failed is detected.
        // The post-increment value is the level we were AT when the
        // failure happened.
        NegSite::record_gr_tighten_level(snap.m_gate_return_tighten);
#endif
        if(snap.m_gate_return_tighten < (uint8_t)KAME_GATE_RETURN_MAX_TIGHTEN)
            ++snap.m_gate_return_tighten;
    }
    snap.m_last_gate_returned = false;
    // First-failure grace: the very first observed WON-then-fail
    // doesn't get to narrow the window — CAS naturally retries and
    // the natural retry pause is often enough to clear the race.
    // Only from the 2nd recorded failure onwards do we right-shift,
    // hence `effective = tighten > 0 ? tighten - 1 : 0`.  The
    // counter itself still increments normally so the level
    // histogram remains comparable across runs.
    const uint8_t raw_tighten = snap.m_gate_return_tighten;
    const uint8_t tighten = raw_tighten > 0
                            ? (uint8_t)(raw_tighten - 1)
                            : (uint8_t)0;
    const uint64_t lo = (uint64_t)KAME_KIND_COUNT_THRESHOLD;
    uint64_t hi = (uint64_t)KAME_KIND_COUNT_HIGH >> tighten;
    if(hi < lo) hi = lo;
    // my_count = total flip count.  Per-kind separation was dropped
    // when BUNDLE / UNBUNDLE slots were merged; kind sensitivity is
    // now carried in `latest_kind` and consumed by the kind filter
    // in the spin loop below (peer's UNBUNDLE doesn't yield to my
    // BUNDLE, etc.).
    const uint64_t my_count = eff_count;
    // Storm guard: skip spin attempt when the running-thread count
    // is already at or above the MAX_RUNNERS cap.  If too many threads
    // are simultaneously in the CAS-retry phase, even a successful
    // spin-WON just dumps us into a contended CAS race we are very
    // likely to lose.  Falling through to SKIPPED_THRASHING routes us
    // to CV-sleep instead, where the wake-up pipeline naturally
    // limits concurrent CAS attempts.
    bool runners_ok = true;
#if KAME_STM_MAX_RUNNERS != 0
    runners_ok = NegotiationCounter::numThreadsRunning()
                 < effective_max_runners(C_obs);
#else
    (void)C_obs;
#endif
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
    {
        NegSite::BandOutcome bo =
              (my_count < lo) ? NegSite::BandOutcome::BELOW_LOW
            : (my_count > hi) ? NegSite::BandOutcome::ABOVE_HIGH
                              : NegSite::BandOutcome::IN_BAND;
        NegSite::record_band_event(mk, bo, tighten);
    }
    if(prev_failed) {
        // With per-kind counts merged, the "who was active" diagnostic
        // collapses to "the latest publisher" — read latest_kind.
        const uint8_t active_kind = (uint8_t)((fs >> L::RSO_LATEST_KIND_SHIFT)
                                              & L::RSO_LATEST_KIND_MASK);
        NegSite::record_gr_not_in_time(
            snap.m_gate_return_my_kind, active_kind);
    }
#endif
    if(fs == 0 || my_count == 0 || my_count < lo) {
        NegSite::record_spin_event(
            NegSite::SpinOutcome::SKIPPED_NO_PERIOD, 0);
        return false;
    }
    if(my_count > hi || !runners_ok) {
        NegSite::record_spin_event(
            NegSite::SpinOutcome::SKIPPED_THRASHING, 0);
        return false;
    }
    // Band IN_BAND + runners OK → spin attempt.
    // Period = (2 windows) / total count → spin budget.
    //
    // The budget arithmetic lives in ns because µs-domain integer
    // division underflows to 0 at high `total_count` (e.g. count=300
    // → 2*128/300 = 0 µs).  Using ns gives ~3-decimal-digit headroom
    // before underflow.  The `cnt_t` packed-stamp API (stamp_us /
    // diff_us_packed) is NOT touched — it stays µs-domain.
    const uint64_t total_count = eff_count;
    const uint64_t fs_period_ns = (total_count > 0)
        ? (2u * (uint64_t)KAME_KIND_WINDOW_NS / total_count)
        : (uint64_t)KAME_KIND_WINDOW_NS;
    const uint64_t period_cap_ns =
        (fs_period_ns * (uint64_t)KAME_SPIN_BUDGET_PCT) / 100u;
    const uint64_t budget_ns =
        period_cap_ns < (uint64_t)KAME_SPIN_MAX_NS
        ? period_cap_ns : (uint64_t)KAME_SPIN_MAX_NS;
    const uint64_t start_ns =
        (uint64_t)NegotiationCounter::now_ns();
    const uint64_t deadline_ns = start_ns + budget_ns;
    // Poll m_recent_ops_state (not the slot) for peer progress.  The
    // slot only flags "an older Tx is tagging me" — low diagnostic
    // value.  recent_ops changes only when record_successful_op fires
    // (= a confirmed B/U publish on this Linkage), which is the
    // actual signal we want to ride.
    //
    // Two win predicates depending on the count regime:
    //
    //   (b) High count (fs_period_ns small) → fine-grain "recent"
    //       check using the 22-bit `ro_timestamp` field, encoded in
    //       (KAME_KIND_WINDOW_NS / 65536) ≈ 2 ns units.  Visible
    //       window ≈ 8 ms.  The denominator 65536 matches the
    //       16-bit count saturation (= smallest meaningful fs_period
    //       ≈ 2·WINDOW_NS / 65535 ≈ 4 ns, so unit ≤ 2 ns resolves it).
    //       Works when ro_timelimit_units < MASK/2.
    //
    //   (a) Low count → ro_timelimit_units overflows the visible
    //       window so the modular comparison is unusable.  Fall back
    //       to `ro != initial_ro` (any-change), which is bounded by
    //       the spin budget anyway (changes during spin are recent
    //       by construction).
    //
    // The kind filter (`!is_ro_unbundle || ro_kind == mk`) applies to
    // both — peer's UNBUNDLE on this Linkage doesn't help a BUNDLE
    // retry.  (Multi-nodal commits now stamp BUNDLE too — the former
    // MultiNodalCommit kind was an alias and is now Reserved.)
    const uint64_t initial_ro = fs;
    bool won = false;
    // Floor unit at 1 ns so very short windows
    // (KAME_KIND_WINDOW_NS < 65536) don't trigger div-by-zero.  Same
    // clamp as the writer side in `Linkage::record_successful_op`.
    constexpr uint64_t TS_UNIT_NS_RAW = (uint64_t)KAME_KIND_WINDOW_NS / 65536u;
    constexpr uint64_t TS_UNIT_NS = TS_UNIT_NS_RAW < 1u ? 1u : TS_UNIT_NS_RAW;
    const uint64_t MAX_USABLE_UNITS = L::RSO_LATEST_TIMESTAMP_MASK / 2u;
    // ro_timelimit = (fs_period_ns / 4) shifted by tighten, in ts-units.
    // /4 keeps the "recent" window to 25 % of the inter-flip period —
    // a balance between catching genuine fresh activity (which /8 and
    // /16 increasingly miss) and not over-firing on stale events
    // (which /2 did).  x86 4-core sweep WON / attempts share:
    //   /2  : 29.4 %  /4 : 16.1 %  /8 : 7.3 %  /16 : 4.3 %
    const uint64_t ro_timelimit_raw =
        ((fs_period_ns / 4u) / TS_UNIT_NS) >> tighten;
    const bool use_recency = (ro_timelimit_raw > 0
                              && ro_timelimit_raw < MAX_USABLE_UNITS);
    const uint64_t ro_timelimit = use_recency ? ro_timelimit_raw : 0;
    // Track whether m_recent_ops_state actually changed while we
    // were spinning.  WON with a state CHANGE during spin (= peer
    // wrote DURING our wait) signals an active in-flight CAS and
    // hence a stale view → caller's scope must abort.  WON without
    // any observed change (the LOAD-AND-CHECK we did at function
    // entry already satisfied the recency predicate, no peer wrote
    // since) is a speculative gate-return: peer may already be
    // done, our view is probably still valid, no abort needed.
    bool observed_change_during_spin = false;
    uint64_t end_ns = NegotiationCounter::now_ns();
    for(;;) {
        end_ns = NegotiationCounter::now_ns();
        for(int i = 0; i < 2; ++i) pause4spin();
        auto ro = self->m_recent_ops_state.load(
            std::memory_order_acquire);
        if(ro != initial_ro)
            observed_change_during_spin = true;

        auto ro_kind = (ro >> L::RSO_LATEST_KIND_SHIFT) & L::RSO_LATEST_KIND_MASK;
        auto ro_timestamp = (ro >> L::RSO_LATEST_TIMESTAMP_SHIFT) & L::RSO_LATEST_TIMESTAMP_MASK;
        bool is_ro_unbundle = ro_kind == (uint8_t)detail::StampKind::UNBUNDLE;
        if( !is_ro_unbundle || (ro_kind == mk)) {
            // Kind filter passes — choose predicate by regime.
            bool fired;
            if(use_recency) {
                const uint64_t end_ts =
                    (end_ns / TS_UNIT_NS) & L::RSO_LATEST_TIMESTAMP_MASK;
                fired = (((end_ts - ro_timestamp - ro_timelimit)
                          & L::RSO_LATEST_TIMESTAMP_MASK)
                         > L::RSO_LATEST_TIMESTAMP_MASK / 2);
            } else {
                fired = (ro != initial_ro);
            }
            if(fired) { won = true; break; }
        }
        if(end_ns > deadline_ns)
            break;
    }
    // elapsed reported in µs to keep record_spin_event histogram
    // binning compatible across the macro change.
    const uint32_t elapsed =
        (uint32_t)(end_ns > start_ns ? (end_ns - start_ns) / 1000u : 0);
    NegSite::record_spin_event(
        won ? NegSite::SpinOutcome::WON
            : NegSite::SpinOutcome::TIMEOUT, elapsed);
    if( !won)
        return false;  // TIMEOUT → fall to CV-sleep.
    // Mark as gate-return ONLY if we actually observed a state
    // change while spinning.  No-spin / speculative WON (initial ro
    // already satisfied the recency predicate, no fresh peer write
    // during our wait) does NOT set m_last_gate_returned: the ctor
    // will use the freshly-acquired view and the post-WON CAS has a
    // fair chance.  Spin WON with change (peer wrote during the
    // wait) means the view is racing — set the flag so the ctor's
    // abort-on-WON path drops the view and the retry loop produces
    // a fresh scope.
    if(observed_change_during_spin) {
        snap.m_last_gate_returned = true;
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
        snap.m_gate_return_time_us =
            (uint32_t)NegotiationCounter::now_us();
        snap.m_gate_return_my_kind = mk;
#endif
    }
    return true;
}
#endif // KAME_ENABLE_SPIN_BAND_GATE

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
ScopedNegotiateLinkage<XN>::_negotiate_internal() noexcept {
    using NegotiationCounter = typename Node<XN>::NegotiationCounter;
    using Linkage = typename Node<XN>::Linkage;
    Linkage *const self = m_link.get();
    Snapshot<XN> &snap = *m_snap;
    // Note: TLA+-equivalent semantics (older-always-wins via
    // preempt-ON for Reserved in tag_as_contender) make disjoint
    // privilege coexistence legitimate.  A privilege-holding Tx
    // may correctly enter `_negotiate_internal` on a Linkage where
    // a peer holds Reserved, and may correctly fair-spin / CV-sleep
    // while waiting for the older peer to commit and release.
    // The cross-link entry-time assert was removed because it
    // flagged this legitimate behaviour as a bug.  Disjoint
    // priv-on-different-Linkages is OK; cycles are broken by the
    // age-ordered preempt path (older's `tag_as_contender` on an
    // overlapping Linkage preempts the younger's Reserved, and the
    // younger's preempt-recovery clears its stale
    // `m_registered_privileged`).
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
    if(snap.m_registered_privileged)
        g_neg_internal_calls_priv.fetch_add(1, std::memory_order_relaxed);
    else
        g_neg_internal_calls_non_priv.fetch_add(1, std::memory_order_relaxed);
#endif
    const float mult_wait = m_mult_wait;
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
    // Age threshold removed: claim eligibility now depends on
    // tag-ownership (in the probe) and retry count, not wall-clock age.
    // Rationale — CAS storms manifest in microseconds, well before the
    // old 300 µs age floor would have fired; serializing early via
    // privilege limits the storm window.  `_ll_age_us` is still
    // computed and passed to the probe for diagnostic logging.
    if ( !snap.m_tagged_linkages.empty()) {
        // Count tagged linkages whose m_transaction_started_time == ours
        // (= "priority is already mine on every linkage" = primary
        //   livelock precondition per the refined definition).
        // Identity check ignores kind bits — see drop_tags_n_privilege.
        const auto _ll_my_id = NegotiationCounter::strip_kind(
                                    snap.m_started_time);
        int _ll_total = (int)snap.m_tagged_linkages.size();
        int _ll_owned = 0;
        int _ll_priv_held = 0;  // # linkages still carrying our Reserved
        for (auto &_l : snap.m_tagged_linkages) {
            if (!_l) continue;
            auto cur = _l->m_transaction_started_time.load(
                std::memory_order_relaxed);
            if (NegotiationCounter::strip_kind(cur) == _ll_my_id) {
                ++_ll_owned;
                if (NegotiationCounter::is_priv_stamp(cur))
                    ++_ll_priv_held;
            }
        }
        // Preemption detection: the snapshot's m_registered_privileged
        // flag is set on first successful claim and previously was
        // cleared only in `drop_tags_n_privilege` (Tx scope end).
        // After preemption (tag_as_contender's older-overwrites-younger
        // rule replaces our Reserved on every Linkage), the flag
        // stays stale-true and the claim gate below blocks all
        // re-claim attempts.  Detect "no Reserved still mine" here
        // and clear the flag so the claim path can re-fire.
        if (snap.m_registered_privileged && _ll_priv_held == 0) {
            snap.m_registered_privileged = false;
            NegotiationCounter::release_priv_count_slot();
        }
        // `entry_pr` was read once at function entry; the probe maps it
        // to retry-threshold / label internally.
        bool _ll_saw = NegotiationCounter::livelock_probe_tx_tick(
            static_cast<const void*>(self),
            snap.m_tx_retry_count,
            self->m_tx_commit_count,
            _ll_owned, _ll_total, sig_C, _ll_age_us,
            entry_pr);
        // Fair-mode escape: when verdict=LIVELOCK fires for this Tx
        // and the Tx has aged past the per-priority floor (see
        // NegotiationCounter::min_privilege_age_us), claim privilege.
        //
        // Per-Linkage mode (KAME_PER_LINKAGE_PRIVILEGE=1, default):
        //   walk our `m_tagged_linkages` and CAS the kind field of
        //   each slot we still own (strip_kind match) to Reserved.
        //   The global `s_privileged_tidstamp` slot is NOT touched —
        //   peers detect privilege by reading the per-Linkage stamp
        //   directly via `is_priv_stamp` in fair_mode_blocks_me.
        //   `claimed` = at least one slot upgraded; sets
        //   `m_registered_privileged` so subsequent probe ticks are
        //   no-ops on this Tx.  drop_tags_n_privilege clears the
        //   Reserved stamps via strip_kind, so no explicit release
        //   is needed.
        //
        // Global mode (=0):
        //   CAS-claim the singleton `s_privileged_tidstamp`.  Peers
        //   detect privilege globally via the old code path.
        if (_ll_saw && !snap.m_registered_privileged) {
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
            g_neg_claim_attempts.fetch_add(1, std::memory_order_relaxed);
#endif
            bool claimed = false;
#if KAME_PER_LINKAGE_PRIVILEGE
            (void)entry_pr;
            const auto my_id = NegotiationCounter::strip_kind(snap.m_started_time);
            const auto my_priv = NegotiationCounter::with_kind(
                snap.m_started_time, detail::StampKind::Reserved);
            for (auto &l : snap.m_tagged_linkages) {
                if (!l) continue;
                auto cur = l->m_transaction_started_time.load(
                    std::memory_order_relaxed);
                if (cur != 0
                    && NegotiationCounter::strip_kind(cur) == my_id) {
                    if (l->m_transaction_started_time.compare_exchange_strong(
                            cur, my_priv,
                            std::memory_order_release,
                            std::memory_order_relaxed)) {
                        claimed = true;
                    }
                }
            }
#else
            claimed = NegotiationCounter::try_register_privileged_tidstamp(
                          entry_pr, snap.m_started_time);
#endif
            if (claimed) {
                snap.m_registered_privileged = true;
                // Pair with the decrement in
                // `Snapshot::drop_tags_n_privilege`.
                s_num_privileged_threads.fetch_add(1, std::memory_order_relaxed);
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
                g_neg_claim_successes.fetch_add(1, std::memory_order_relaxed);
#endif
                // Note: we do NOT assert post-claim that any Linkage still
                // carries our Reserved.  A racing older Tx can preempt our
                // Reserved (symmetric window rule in tag_as_contender)
                // between our CAS-upgrade loop and this point.  The
                // claim-success accounting (flag + counter increment)
                // remains paired with the drop_tags_n_privilege decrement
                // regardless of subsequent preemption.
            }
        }
    }

    // Always-on adaptive path: the V0 (legacy) path and the V0↔ADAPTIVE
    // mode switch were removed in favour of the orthogonal fair-mode
    // escape (s_privileged_tidstamp). See top of detail:: in this file.
  { // adaptive-path scope
    // One atomic load of the packed (tid | lease_us | start_us) tuple.
    auto ps = self->loadPriority();
    if(ps.tid) {
        tid_bitset.observe((unsigned)ps.tid);
    }
    typename NegotiationCounter::cnt_t transaction_started_time =
        self->m_transaction_started_time.load(std::memory_order_relaxed);
    if( !transaction_started_time)
        return; //collision has not been detected.
    // LOWEST and UI_DEFERRABLE explicitly tolerate yielding, so the
    // helper internally skips the lease/owner-skip block for those
    // priorities.  Returns true iff the owner-skip fired (we hold the
    // soft lease and our age < lease_us) — caller returns early.
    if(_neg_apply_lease(ps, transaction_started_time, sig_C,
                        now_us_entry, entry_pr))
        return;

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
        // preferred_kind: wake threads whose op_kind matches the
        // *notifier's own* op_kind.  Rationale: we only fire
        // notify_n_contenders at points where the notifier is about
        // to retry CAS itself (sig_C==1 fast path and lottery break)
        // or to refill the running pipeline (MIN_RUNNERS escape).
        // In either case the upcoming commit is the notifier's kind,
        // so peers with matching kind set up a same-kind streak
        // (BB or UU) that passes the spin-block same-kind filter.
        // An earlier variant biased on the linkage's last_commit
        // kind, but that is stale relative to the imminent CAS.
        //
        // Set -DKAME_CV_WAKE_KIND_PREF=0 to disable (ablation knob).
#ifndef KAME_CV_WAKE_KIND_PREF
#define KAME_CV_WAKE_KIND_PREF 1
#endif
        auto preferred_kind_for_wake = []() -> uint8_t {
#if KAME_CV_WAKE_KIND_PREF
            return (uint8_t)detail::s_current_op_kind & 0x3u;
#else
            return (uint8_t)0xFFu;
#endif
        };

        if(sig_C == 1) {
            NegotiationCounter::notify_n_contenders(
                tid_bitset, 1, preferred_kind_for_wake());
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
            self->m_transaction_started_time.load(std::memory_order_acquire);
        if( !NegotiationCounter::is_active_stamp(transaction_started_time))
            break; //collision has not been detected.

        auto dt2 = NegotiationCounter::diff_us_packed(
            Node<XN>::NegotiationCounter::now_us(),
            transaction_started_time);

        // Fair-mode escape: when some other thread holds the privileged-
        // TID slot, suppress the jittered gate and the √C lottery so the
        // privileged Tx alone gets to commit. Strict Greedy CM (older
        // started_time wins → I sleep below) is the only mechanism left
        // to allocate priority while fair-mode is active.
        //
        // Whether some peer's privilege blocks our CAS on this Linkage.
        // The choice between per-Linkage and global privilege happens
        // inside `fair_mode_blocks_me` based on KAME_PER_LINKAGE_PRIVILEGE
        // — see helper definition in this file.  Pre-loaded
        // `transaction_started_time` is *not* reused here because the
        // helper does its own load; the cost is one extra atomic load
        // under per-Linkage mode (negligible vs. the surrounding CV-wait
        // / spin work).
        const bool _fair_blocks =
            NegotiationCounter::fair_mode_blocks_me(started_time, self);

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
                    NegotiationCounter::try_notify_n_contenders(
                        tid_bitset, C_obs, preferred_kind_for_wake());
#else
                    NegotiationCounter::notify_n_contenders(
                        tid_bitset, C_obs, preferred_kind_for_wake());
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
            fprintf(stderr, "for BP@%p\n", (void*)self);
            ms = 5000;
        }

#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
        // ====== PRIV-HOLDER-YIELDING DIAGNOSTIC (opt-in) ======
        // Per user: "プリビレッジは譲り合い不要なはずなので、譲り合い
        // 必要な時点でバグ".  If we are a priv holder about to either
        // fair-spin or CV-sleep, that violates the design invariant —
        // dump the slot state so we can identify the owner of the
        // Reserved tag blocking us.
        if(snap.m_registered_privileged && ms >= 30) {
            // Throttle: at most one dump per Linkage per ~50ms across
            // all threads.  Use the slot's address as the throttle key;
            // since slot is a Linkage-owned atomic, racing prints from
            // multiple threads on the same Linkage CAS each other for
            // the next-print-us value — at most one wins per window.
            static std::atomic<int64_t> s_next_print_us{0};
            int64_t now_us_dump = NegotiationCounter::now_us();
            int64_t exp = s_next_print_us.load(std::memory_order_relaxed);
            if(now_us_dump >= exp
               && s_next_print_us.compare_exchange_strong(
                      exp, now_us_dump + 50000,
                      std::memory_order_relaxed)) {
                // Decode this thread / Tx state
                auto self_slot = self->m_transaction_started_time.load(
                    std::memory_order_relaxed);
                fprintf(stderr,
                    "[PRIV-YIELDING] tid=%u my_stamp=0x%llx kind=%u "
                    "self=%p self.slot=0x%llx slot.tid=%u slot.kind=%u "
                    "slot.age_us=%lld ms=%d retry=%u "
                    "s_num_priv=%u s_fair_spinners=%u tagged.size=%zu\n",
                    (unsigned)ProcessCounter::id(),
                    (unsigned long long)started_time,
                    (unsigned)NegotiationCounter::stamp_kind(started_time),
                    (void*)self,
                    (unsigned long long)self_slot,
                    (unsigned)NegotiationCounter::stamp_tid(self_slot),
                    (unsigned)NegotiationCounter::stamp_kind(self_slot),
                    (long long)NegotiationCounter::diff_us_packed(
                        now_us_dump, self_slot),
                    ms, (unsigned)snap.m_tx_retry_count,
                    (unsigned)s_num_privileged_threads.load(
                        std::memory_order_relaxed),
                    (unsigned)s_fair_spinners.load(
                        std::memory_order_relaxed),
                    snap.m_tagged_linkages.size());
                int idx = 0;
                for(auto &sp : snap.m_tagged_linkages) {
                    if(!sp) { ++idx; continue; }
                    auto slot_val = sp->m_transaction_started_time.load(
                        std::memory_order_relaxed);
                    fprintf(stderr,
                        "  [tagged[%d]] link=%p slot=0x%llx tid=%u kind=%u "
                        "age_us=%lld is_self=%d\n",
                        idx, (void*)sp.get(),
                        (unsigned long long)slot_val,
                        (unsigned)NegotiationCounter::stamp_tid(slot_val),
                        (unsigned)NegotiationCounter::stamp_kind(slot_val),
                        (long long)NegotiationCounter::diff_us_packed(
                            now_us_dump, slot_val),
                        (sp.get() == self) ? 1 : 0);
                    ++idx;
                }
            }
        }
        // ====== end PRIV-YIELDING DIAGNOSTIC ======
#endif

        // Unified PRE-spin band gate + any-change spin shortcut.
        // Spin won → break out of the negotiate loop; otherwise fall
        // through to CV-sleep.  See `_neg_spin_block` definition for
        // the band / tighten / spin-budget rationale.
        // Compiled out entirely when KAME_ENABLE_SPIN_BAND_GATE=0.
#if KAME_ENABLE_SPIN_BAND_GATE
        if(_neg_spin_block(C_obs))
            break;
#else
        (void)C_obs;
#endif

        // Privilege bistability guard: when a peer holds per-Linkage
        // privilege on `self` (`_fair_blocks`) and the spinner pool
        // has capacity, busy-poll until the peer releases instead of
        // going to CV-sleep — saves the ~1 ms CV-wake restart on
        // privilege release.
        //
        // Cap: spinners + currently-privileged threads ≤
        //   effective_runners(C_obs) (≈ hardware concurrency).
        // Priv holders are independent on unrelated Linkages, so the
        // global `s_num_privileged_threads` counter is subtracted
        // from the spinner admission ceiling — otherwise spinners
        // would oversubscribe the cores against the productive priv
        // holders.
        //
        // The decision pays one `numThreadsRunning()`-style cost
        // *once* on the way in (here: two relaxed atomic loads on
        // `s_fair_spinners` / `s_num_privileged_threads` —
        // significantly cheaper than the weak_ptr-sum in
        // `num_threads_running()`).  The inner loop only reads
        // `fair_mode_blocks_me` (one relaxed atomic load on x86 —
        // plain `mov`, acquire ≠ cmpxchg).
        //
        // No iteration bound: an unbounded spin that never
        // terminates means a programming error (peer privilege held
        // forever) — the lack of a fallback exposes such bugs
        // rather than masking them under a timeout.
        //
        // Note: `m_snap->m_registered_privileged` may be true here.
        // Unrelated Linkages can be claimed by us independently, so
        // we may hold priv on Linkage A while waiting on peer's priv
        // on Linkage B (= self).
#if KAME_STM_MIN_RUNNERS != 0
        {
            const int run_cap = effective_runners(C_obs);
            const int n_priv =
                (int)s_num_privileged_threads.load(std::memory_order_relaxed);
            const int spin_cap = run_cap > n_priv ? run_cap - n_priv : 0;
            if(_fair_blocks
               && (int)s_fair_spinners.load(std::memory_order_relaxed)
                  < spin_cap) {
                s_fair_spinners.fetch_add(1, std::memory_order_relaxed);
                // Periodic yield: even with our spinner cap respecting
                // the core count, *external* processes can saturate
                // cores beyond our control.  Yield every ~2^18 PAUSE
                // iterations (~1 ms at typical x86 PAUSE latency) so
                // the OS scheduler has a chance to run any preempted
                // privilege holder / other progress-maker.
                unsigned iter = 0;
                do {
                    pause4spin();
                    if((++iter & 0x3FFFFu) == 0)
                        std::this_thread::yield();
                } while(NegotiationCounter::fair_mode_blocks_me(
                                started_time, self));
                s_fair_spinners.fetch_sub(1, std::memory_order_relaxed);
                continue;
            }
        }
#endif

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
        // A privilege holder *may* legitimately enter CV-sleep here
        // when waiting on an older peer's Reserved on a different
        // Linkage (disjoint-priv coexistence under TLA+-equivalent
        // older-wins semantics).  The older peer is guaranteed to
        // commit and release first, waking us.  If our own Reserved
        // overlapped with the older peer's tagging, our Reserved
        // was already preempted via `tag_as_contender`, and the
        // preempt-recovery scan in `_negotiate_internal` cleared
        // our `m_registered_privileged` flag before reaching here.
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
                        std::min(min_r - running, C_obs),
                        preferred_kind_for_wake());
                // Symmetric wake-older rule (per user) — runs every
                // CV chunk so repeated notifies keep the oldest peer
                // awake until our Reserved is preempted:
                //   - newer non-priv: ALWAYS wake older before sleeping.
                //   - newer priv:     within the burst window from our
                //                     own start, do NOT disturb older.
                //                     Outside the window, wake older (so
                //                     they can preempt our Reserved via
                //                     the matching tag_as_contender
                //                     window — see Snapshot::tag_as_contender).
                // Budget = 1 keeps the per-chunk overhead bounded (one
                // 512-slot try-lock scan).  Independent of MIN_RUNNERS
                // per user: "olderは、MIN_RUNNERSの設定によらず起こす".
                bool _wake_older = true;
                if(snap.m_registered_privileged) {
                    int64_t _my_age_us = (int64_t)NegotiationCounter::diff_us(
                        NegotiationCounter::now_us(),
                        NegotiationCounter::stamp_us(started_time));
                    if(_my_age_us < KAME_STM_PREEMPT_WINDOW_US)
                        _wake_older = false;
                }
                if(_wake_older) {
                    // Targeted wake (per user): we know which Tx
                    // blocked us — the stamp on `self->m_transaction_
                    // started_time`.  No 512-slot scan needed.  The
                    // blocker is presumably committing (not sleeping)
                    // in the normal case, in which case the notify is
                    // harmless; the wake is a safety net for the bug
                    // case where the blocker is somehow stuck in
                    // CV-sleep itself.
                    auto _slot_val =
                        self->m_transaction_started_time.load(
                            std::memory_order_relaxed);
                    // Fresh dt re-check: the outer `dt` (computed once
                    // at function entry) can become stale when the
                    // slot rotates to a YOUNGER stamp during the
                    // CV-sleep cycle.  Without this re-check the loop
                    // sleeps forever on a now-stale "I'm younger"
                    // verdict.  Two gates limit the eager break-out:
                    //   - `m_registered_privileged`: priv holders MUST
                    //     break out (otherwise they hold their
                    //     Reserved indefinitely and block everyone).
                    //   - `ms >= 30`: non-priv threads only escape
                    //     after a substantial wait has accumulated.
                    //     M2 (low contention) rarely hits this; M3
                    //     (CAS-storm prone) reaches it during
                    //     mutual-wait cycles.  Threshold chosen so M2
                    //     fast path (ms ≤ a few ms) is unaffected.
                    if((snap.m_registered_privileged || ms >= 30)
                       && ( !NegotiationCounter::is_active_stamp(_slot_val)
                            || NegotiationCounter::signed_diff_us_packed(
                                   started_time, _slot_val) <= 0))
                        goto _exit_cv_sleep;
                    uint16_t _blocker_tid =
                        NegotiationCounter::stamp_tid(_slot_val);
                    if(_blocker_tid != 0) {
                        int _idx = (int)((unsigned)_blocker_tid
                            % NegotiationCounter::NEGOTIATE_SLEEP_SLOTS);
                        auto &_st = NegotiationCounter::s_sleep_slots[_idx];
                        std::lock_guard<std::mutex> _lk(_st.mtx);
                        // Tenant verification: only wake when the slot's
                        // current tenant matches the blocker (same tid +
                        // same started_us).  Comparison strips the kind
                        // bits because the linkage slot is stamped via
                        // `with_kind(started_time, op_kind)` in
                        // `tag_as_contender` while the sleep slot stores
                        // the bare `started_time` (kind=NONE).
                        if(NegotiationCounter::strip_kind(_st.stamp)
                           == NegotiationCounter::strip_kind(_slot_val)) {
                            _st.notified = true;
                            _st.cv.notify_one();
                        }
                    }
                }
#if KAME_STM_DISABLE_JITTER
                NegotiationCounter::negotiate_sleep(1, started_time);
#else
                NegotiationCounter::negotiate_sleep(
                    1 + (int)(s_backoff_seed >> 31), started_time);
#endif
            } while(Node<XN>::NegotiationCounter::now_us() < t_end);
#else
            NegotiationCounter::negotiate_sleep(ms_actual, started_time);
#endif
        }
    }
_exit_cv_sleep:;
  } // end adaptive-path scope
}

} // namespace Transactional

#endif /* TRANSACTION_NEG_IMPL_H */
