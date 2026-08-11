/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            LICENSE-APACHE-2.0 in this directory)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see LICENSE-GPL-2.0 in this directory).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
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
#if defined(__linux__)
#  include <sched.h>       // sched_getscheduler — the RT fast-priv gate
#elif defined(__APPLE__)
#  include <pthread.h>     // pthread_getschedparam — same gate
#endif
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>    // getenv/atoi — the KAME_STM_RT_FAST_PRIV knob
#include <mutex>
#include <thread>

namespace Transactional {

template <class XN>
struct Node<XN>::WalkUpResult {
    SnapshotStatus find_status;  //!< result of findChildSlot (or early-return status)
    SnapshotStatus status;       //!< status after convertRecursiveStatus (before find)
    bool is_root_level;          //!< true if this parent is the chain root
    local_shared_ptr<Linkage> parent_linkage;    //!< m_link of the parent node (= bundledBy)
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

//! Per-priority claim-age floor.
//!
//! Reachable with a per-level `pr` ONLY from `try_register_privileged_tidstamp`,
//! which is the `#else` arm of `KAME_PER_LINKAGE_PRIVILEGE` — i.e. the
//! non-default global-privilege mode.  In the default per-Linkage mode the claim
//! path opens with a literal `(void)entry_pr;`, and the only calls that survive
//! pass `Priority::SCRIPTING` explicitly (`stamp_is_expired_lowprio`, and one
//! diagnostic print), so the LOWEST / UI_DEFERRABLE / NORMAL branches are dead
//! there.  Kept because the global mode still compiles, not because the
//! graduated ladder is operative by default.
template <class XN>
int64_t Node<XN>::NegotiationCounter::min_privilege_age_us(Priority pr) noexcept {
    switch (pr) {
    case Priority::SCRIPTING:     return 1'000;        // 1 ms — script/MCP/ZMQ
    case Priority::LOWEST:        return 30'000;       // 30 ms — bulk/analysis
    case Priority::UI_DEFERRABLE: return 50'000;       // 50 ms — interactive UI
    default:                      return KAME_STM_PRIV_AGE_NORMAL_US;  /* HIGHEST / NORMAL */
    }
}

template <class XN>
bool Node<XN>::NegotiationCounter::stamp_is_expired_lowprio(cnt_t stamp) noexcept {
    // Single source of truth shared by `try_register_privileged_tidstamp`,
    // `i_am_privileged_now`, and `fair_mode_blocks_me`.  All three must
    // agree on "expired" or per-Linkage Reserved stamps go stuck (the
    // failure mode is analysed in detail in fair_mode_blocks_me below).
    //
    // SCRIPTING's claim floor dominates the threshold because it is the
    // longest-tenured LOW priority — using its floor here uniformly
    // gives LOWEST / UI_DEFERRABLE holders the same generous wall-clock
    // window before eviction, which simplifies reasoning and avoids
    // race windows where two LOW priorities use different cutoffs.
    if( !stamp_is_lowprio(stamp)) return false;
    int64_t now_us  = LivelockProbe::now_us();
    int64_t age     = (int64_t)diff_us_packed(now_us, stamp);
    int64_t max_age = min_privilege_age_us(Priority::SCRIPTING)
                    + (int64_t)KAME_STM_PRIV_MAX_HOLD_US;
    return age > max_age;
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
        // Expiration via shared helper.  Only LOW-priority holders
        // (lowprio bit set; LOWEST / UI_DEFERRABLE / SCRIPTING) can
        // expire; NORMAL / HIGHEST are immune (measurement / driver
        // critical).  Treating an expired holder as "empty slot"
        // lets a stuck older SCRIPTING Tx be evicted by a newer one
        // that would otherwise be blocked by the older-only
        // preemption rule.
        bool holder_expired =
            (expected != (cnt_t)0) && stamp_is_expired_lowprio(expected);
        if (expected != (cnt_t)0 && !holder_expired) {
            // Live holder. Preempt only if the challenger (us) is older
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
            // Empty slot OR expired holder.  Require scaled age
            // threshold to reduce churn when many threads are contending.
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
    // Diagnostic moved to the caller's `if (claimed)` block (see
    // `_negotiate_internal`) so it fires uniformly for both global
    // and per-Linkage privilege modes.  In per-Linkage mode this
    // function is not called at all (the CAS-claim runs inline in
    // the caller), so leaving the print here would make it dead.
    return true;
}

//! Rule 0d (default OFF) is two `if`s -- one here, one in
//! `fair_mode_blocks_me` -- reading the same three facts off the one word the
//! Linkage already holds: whose tag it is (tid), what planted it (kind), and
//! what class that Tx started at (PRIO).  No new state, no helper: a helper
//! would only hide that both sides test one load.
#ifndef KAME_STM_HIGHEST_BUNDLE_BLOCK
#define KAME_STM_HIGHEST_BUNDLE_BLOCK 0
#endif

template <class XN>
bool Node<XN>::NegotiationCounter::i_am_privileged_now(
        cnt_t my_tidstamp,
        const Linkage *link) noexcept {
    // Expiration check delegated to `stamp_is_expired_lowprio`: a
    // LOW-priority priv stamp older than `min_privilege_age_us(SCRIPTING)
    // + PRIV_MAX_HOLD_US` is considered expired (holder lost privilege by
    // timeout).  NORMAL / HIGHEST priv never expires — deliberately (user
    // ruling, reaffirmed 2026-07-30): their privilege is the COMPLETION
    // guarantee.  The revocable tiers get the starvation timeout as their
    // exit; NORMAL has no exit by design, so its shield must outlast any
    // wall clock, and the TLA+ liveness argument assumes exactly that.  An
    // OWNERLESS stamp (the 2026-07-30 T1Mode abort) is not a holder — it
    // was a mid-construction-throw leak, fixed at the source in the
    // Snapshot/Transaction constructors; the HANG watchdog remains the
    // terminal backstop.
#if KAME_PER_LINKAGE_PRIVILEGE
    // Per-Linkage: "mine" iff this Linkage's slot carries a Reserved-
    // kind stamp with matching TID.  Compare by TID alone (NOT
    // `strip_kind`, which keeps the US field) so that a nested inner
    // Tx on the same thread — different `m_started_time` from the
    // outer Tx but same TID — recognises itself as the privilege
    // holder.  Mirrors the global-mode self-deadlock workaround in
    // the else branch below.  (Fix 2026-05-20: was `strip_kind`.)
    if(link == nullptr) return false;
    cnt_t slot = link->m_transaction_started_time.load(std::memory_order_relaxed);
    // Rule 0d, self side — the mirror of the peer test in
    // `fair_mode_blocks_me`, same two words, `==` instead of `!=`.  It has to
    // be here or the rule is all cost and no benefit: peers defer while the
    // holder still takes the weak-acquire / ADAPTIVE-threshold path built for
    // a CAS that is no longer contended.  Note this widens the CAS-fail-twice
    // assertion in `_on_cas_fail` to cover Rule 0d, which is what we want —
    // if it fires, peers are racing a tag they were supposed to defer to.
#if KAME_STM_HIGHEST_BUNDLE_BLOCK
    if(slot && stamp_tid(slot) == stamp_tid(my_tidstamp)
            && is_bundling_kind(slot) && stamp_is_highest(slot))
        return true;
#endif
    if( !is_priv_stamp(slot)) return false;
    if(stamp_tid(slot) != stamp_tid(my_tidstamp)) return false;
    // Expiration: stale priv stamp from a stuck Tx no longer grants
    // privilege.  Peers see the matching update via `fair_mode_blocks_me`,
    // which uses the same `stamp_is_expired_lowprio` predicate to
    // treat the per-Linkage Reserved stamp as unblocking.
    if(stamp_is_expired_lowprio(slot)) return false;
    return true;
#else
    (void)link;
    cnt_t priv = s_privileged_tidstamp.load(std::memory_order_relaxed);
    if(priv == (cnt_t)0) return false;
    if(stamp_tid(priv) != stamp_tid(my_tidstamp)) return false;
    // Global-mode expiration: pairs with the expired-slot detection
    // in `try_register_privileged_tidstamp` so a stuck holder cannot
    // block all other priv-claimants forever.  Critical for SCRIPTING
    // (two stuck SCRIPTING Tx could otherwise deadlock each other
    // under the older-only preemption rule).
    if(stamp_is_expired_lowprio(priv)) return false;
    return true;
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
    // Expiration check delegated to `stamp_is_expired_lowprio`.  A
    // stuck low-priority holder leaves a stale Reserved stamp on some
    // Linkage (per-Linkage mode) or in `s_privileged_tidstamp`
    // (global mode) that the holder's own thread can no longer refresh
    // (`i_am_privileged_now` returns false past the same threshold),
    // yet without this check peers would still see the stamp here and
    // yield to a holder that has already conceded — a frozen Linkage
    // nobody can overwrite.  Treating expired stamps as unblocking
    // lets peers fall through to the ordinary commit CAS, which
    // clobbers the dead stamp naturally.  NORMAL / HIGHEST stamps
    // never carry the lowprio bit, so they are never reported expired.
    // KAME_STM_PRIV_DIAG: one-shot print per expired stamp (TLS-dedup).
    auto report_expired = [](cnt_t stamp) {
#if KAME_STM_PRIV_DIAG
        static thread_local cnt_t s_last_reported = 0;
        if(stamp == s_last_reported) return;
        s_last_reported = stamp;
        std::fprintf(stderr,
            "[priv-timeout] expired lowprio stamp tid=%u age>%lld us\n",
            (unsigned)stamp_tid(stamp),
            (long long)(min_privilege_age_us(Priority::SCRIPTING)
                        + (int64_t)KAME_STM_PRIV_MAX_HOLD_US));
#else
        (void)stamp;
#endif
    };
#if KAME_PER_LINKAGE_PRIVILEGE
    // Per-Linkage: blocked iff the slot's Reserved stamp is held by
    // SOME OTHER thread.  TID-only compare (see `i_am_privileged_now`
    // above for the nested-Tx self-deadlock rationale).
    if(link == nullptr) return false;
    cnt_t slot = link->m_transaction_started_time.load(std::memory_order_relaxed);
    // Rule 0d (KAME_STM_HIGHEST_BUNDLE_BLOCK=1, default OFF): a Tx holding a
    // validated HIGHEST tag on this linkage blocks lower-priority committers
    // on it, exactly as a Reserved stamp does.
    //
    // Rule 0c stopped peers OVERWRITING a HIGHEST tag; it does not stop them
    // COMMITTING, because only Reserved stamps reach the test below — "a
    // plain tag merely shortens the loser's adaptive backoff".  So a peer
    // still replaces the packet under a HIGHEST bundle, the bundle returns
    // DISTURBED, and Node::snapshot() rebuilds.  That rebuild count is the
    // one quantity here with no bound: measured 2 -> 142 as the subtree grew
    // 2 -> 13 linkages, while the outer retry count the 2L tag argument
    // covers stayed flat at 2-3.
    //
    // The class comes off the tag's own PRIO field, so any tag — plain or
    // Reserved — reports it, and there is nothing to cross-validate.  Master
    // writes the same `stamp_is_highest` test inline for Rule 0's strip
    // decision in `tag_as_contender`.
    //
    // The `is_bundling_kind` gate is load-bearing, which is not obvious and
    // was measured only after dropping it looked like a clean simplification.
    // The argument for dropping it: a tag's lifetime is already bounded,
    // since `drop_tags_n_privilege()` clears it from ~Transaction(), so "a
    // validated HIGHEST tag holds this linkage" already means "a HIGHEST Tx
    // is in flight on it".  True, and far too long a window — a whole Tx
    // rather than the microseconds inside one bundle()/unbundle() pass.
    // 8 interleaved 20 s pairs, ungated: acq -24 %, NORMAL -20 %, and the
    // outer retry max — the thing the rule exists to bound — went from 2 to
    // 4-16.  Shield the pass, not the transaction.
    //
    // Exposure, with the gate: a HIGHEST thread preempted mid-bundle blocks
    // lower-priority peers on that linkage for the whole preemption, with no
    // expiry (HIGHEST stamps never carry the lowprio bit).  Same
    // never-expiring exposure as privilege, on a more frequent stamp.
    //
    // CONTAINER A/B, gated, 8 interleaved 20 s pairs at 16 leaves:
    // acq 40,893 -> 38,613 /s (-5.6 %), NORMAL 283,130 -> 272,571 (-3.7 %),
    // UI_DEFERRABLE +2.6 %.  No measured benefit: the outer retry max reads
    // 2-3 vs 3-4, but the OFF arm's own distribution moved by that much
    // BETWEEN batches, so a 20 s max cannot resolve it here.  (An earlier
    // 4-pair read claimed -2.9 % and a retry-max win; both were subset
    // artifacts of an ON arm with 7.5 % spread.  n=8, interleaved, or it is
    // not a number.)
    //
    // So: a measured 4-6 % throughput charge and, in a container, nothing to
    // show for it.  The tail it is meant to cut is not measurable here at all
    // — MAX across these runs is 0.8-15.9 ms of scheduler noise — and this
    // file's conclusions have been reversed by the RT host before (the
    // DISJOINT control came out backwards in the container).  Default OFF;
    // the RT host decides.
#if KAME_STM_HIGHEST_BUNDLE_BLOCK
    if(slot && stamp_tid(slot) != stamp_tid(tidstamp)
            && is_bundling_kind(slot) && stamp_is_highest(slot))
        return true;
#endif
    if( !is_priv_stamp(slot)) return false;
    if(stamp_tid(slot) == stamp_tid(tidstamp)) return false;
    if(stamp_is_expired_lowprio(slot)) { report_expired(slot); return false; }
    return true;
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
    if(stamp_tid(priv) == stamp_tid(tidstamp)) return false;
    if(stamp_is_expired_lowprio(priv)) { report_expired(priv); return false; }
    return true;
#endif
}


template <class XN>
typename Node<XN>::NegotiationCounter::PriorityProbeInfo
Node<XN>::NegotiationCounter::priority_probe_info(Priority pr) noexcept {
    switch (pr) {
        case Priority::HIGHEST:       return { "HIGHEST" };
        case Priority::NORMAL:        return { "NORMAL" };
        case Priority::UI_DEFERRABLE: return { "UI_DEFERRABLE" };
        case Priority::LOWEST:        return { "LOWEST" };
        case Priority::SCRIPTING:     return { "SCRIPTING" };
        default:                      return { "?" };
    }
}

namespace detail {
//! Is the CURRENT thread under an OS realtime policy (SCHED_FIFO/RR)?
//! Cached per thread: one pthread/sched syscall on first use, then a TLS
//! load.  The cache means a thread that elevates itself AFTER its first
//! negotiation keeps reading "not RT" — the conservative direction (the
//! fast path stays off) — so set the OS policy at thread start, as every
//! KAME acquisition thread and this repo's harnesses already do.  Windows:
//! no mapping attempted, always false.
inline bool os_sched_rt() noexcept {
#if defined(__linux__)
    static thread_local int t_rt = -1;
    if(t_rt < 0) {
        const int pol = ::sched_getscheduler(0);
        t_rt = (pol == SCHED_FIFO || pol == SCHED_RR) ? 1 : 0;
    }
    return t_rt == 1;
#elif defined(__APPLE__)
    static thread_local int t_rt = -1;
    if(t_rt < 0) {
        int pol = 0; struct sched_param sp {};
        if(::pthread_getschedparam(::pthread_self(), &pol, &sp) != 0)
            pol = 0;
        t_rt = (pol == SCHED_FIFO || pol == SCHED_RR) ? 1 : 0;
    }
    return t_rt == 1;
#else
    return false;
#endif
}
} // namespace detail

// The negotiation diagnostic counters are defined HERE, above the first
// template that touches them: `detail::neg_diag()` is a NON-dependent name
// inside `livelock_probe_tx_tick`, so two-phase lookup resolves it at the
// point of definition and a later declaration would not be found.
#if KAME_STM_NEG_DIAG
namespace detail {
//! Plain (non-atomic) per-thread counters: only the owning thread writes, and
//! a reader snapshots between commits.  See `neg_diag_snapshot`.
struct NegDiag {
    std::uint64_t rounds;      //!< negotiation wait-loop iterations
    std::uint64_t sleeps;      //!< times we actually entered cell.wait()
    std::uint64_t slept_ns;    //!< wall time inside cell.wait()
    std::uint64_t priv_tries;  //!< privilege claims attempted
    std::uint64_t priv_grants; //!< privilege claims that succeeded
    //! Sleeps entered while this Tx still OWNS the tag on >=1 linkage.  A
    //! multi-linkage Tx (a grand-scope commit tags parent + every child) can
    //! hold some tags and be blocked on another — hold-and-wait.  If that is
    //! where the sleeps are, peers are queued behind a SLEEPING holder and the
    //! wait is set by the sleep, not by anyone's work.
    std::uint64_t sleeps_holding;
    std::uint64_t tags_held_at_sleep;   //!< sum of tags owned at those sleeps
    std::uint64_t sleeps_priv;          //!< ... and while holding privilege
    //! Sum of m_tagged_linkages.size() at each sleep.  Distinguishes "owned 0
    //! of N tags" (displaced) from "had tagged nothing yet" (vacuous).
    std::uint64_t tagged_list_at_sleep;
    //! Sum of the durations we ASKED cell.wait() for.  Compared against
    //! slept_ns (what actually elapsed), this separates "the STM chose to
    //! sleep this long" from "the OS did not run us again for this long":
    //! actual ~= requested means the former, actual >> requested the latter.
    std::uint64_t req_ns;
    //! Distribution of the per-round sleep BUDGET (`ms_actual`).  `ms` grows
    //! every round (`ms = max(dt2*mult/10000, ms+1)`, capped at 5000), so if
    //! the tail is the sum of an escalating backoff this is where it shows.
    std::uint64_t ms_sum;
    std::uint64_t ms_max;
    std::uint64_t entries;   //!< calls into _negotiate_internal
    //! Rounds entered while a peer's privilege blocks us — the rounds in which
    //! `_wb_round` is forced to 0, i.e. the wait budget is CONTRACTUALLY
    //! SUSPENDED.  Any latency past the budget has to live in these, so this
    //! is the field that decides whether an overshoot is the exemption or a
    //! defect in the clamping.
    std::uint64_t rounds_exempt;
    //! Wall time inside the `_fair_blocks` busy-spin (bounded by
    //! KAME_STM_FAIR_SPIN_MAX_US, but its deadline is only re-checked every
    //! 2^18 PAUSEs, so it can overshoot its own cap).
    std::uint64_t spin_ns;
    std::uint64_t spins;        //!< entries into that spin
    //! `slept_ns` split by whether the round was exempt.  budgeted + exempt
    //! == slept_ns; the interesting one is `slept_exempt_ns`.
    std::uint64_t slept_exempt_ns;
    //! Worst single `cell.wait()` OVERSHOOT (actual − requested).  The one
    //! number that separates "the STM chose to wait this long" from "the OS
    //! did not run us again for this long", per wait rather than summed —
    //! a sum cannot tell one 700 us late wake-up from seventy 10 us ones.
    std::uint64_t late_max_ns;
    //! The deadline-tail spin that replaces the last KAME_NEG_SPIN_TAIL_US of
    //! a budget: how often it fired and how long it actually held the core.
    std::uint64_t tail_spins;
    std::uint64_t tail_spin_ns;
    //! The two INNER CAS loops, which nothing else counts.  `attempts` as a
    //! harness measures it is `iterate_commit` re-running the caller's lambda
    //! — the OUTERMOST loop.  Inside one such attempt, commit() and bundle()
    //! each spin their own `for(int retry = 0;; ++retry)` (transaction_impl.h
    //! :2928 and :2571) which retry a CAS without restarting the transaction,
    //! so a commit can be expensive with `attempts` at 2.  That is exactly the
    //! gap the 2026-08 tail investigation ran into: 8,496 slow commits averaged
    //! 2.1 attempts and 100 % of their time was unaccounted for by every field
    //! above.  These two say whether the time is bundle churn or the final CAS.
    std::uint64_t commit_cas_retries;
    std::uint64_t bundle_cas_retries;
    //! And the third, which is the one on the hot path nobody had looked at:
    //! Node<XN>::snapshot()'s own `for(int retry = 0;; ++retry)`
    //! (transaction_impl.h:2212).  Transaction construction takes a snapshot,
    //! iterate_commit rebuilds the Transaction every attempt, and a
    //! multi-nodal snapshot bundles the subtree to get a consistent view — so
    //! this loop can call bundle()/unbundle() repeatedly while bundle itself
    //! never retries, which is exactly the shape observed (bundle_cas = 0.00
    //! over 8,013 slow commits).  Its retries are deliberately hidden from the
    //! livelock probe: GuardSnapshotRetryCount RESTORES m_tx_retry_count on
    //! scope exit, because they are snapshot-internal rather than
    //! transaction-level.  Correct for the probe, invisible for latency.
    std::uint64_t snapshot_retries;
    //! The livelock probe, which is the only door to a privilege claim
    //! (`if(_ll_saw && !registered)` — no priority term, so HIGHEST claims on
    //! the same terms as anyone).  priv strips have been 0 in every run of
    //! transaction_priority_mixed_test, and three AND-ed conditions inside the
    //! probe can each account for that; these counters separate them instead
    //! of leaving it to a reading of the source.  What they found (container,
    //! 1.15 M commits — see that test's header for the full write-up): the
    //! retry threshold `clamp(sig_C*2, 3, hardware_concurrency)` is the
    //! largest blocker at 55 % of ticks, because outer attempts peak at 3 and
    //! so `m_tx_retry_count` peaks at 2 against a floor of 3; the per-linkage
    //! window RESET is 36 % (the state holds ONE linkage_id and a multi-nodal
    //! commit negotiates on several); `tags_owned == tags_total` is 9 %.  A
    //! fourth gate sits OUTSIDE the probe and is bigger than any of them —
    //! `if(!snap.m_tagged_linkages.empty())` — so a transaction that is merely
    //! losing a CAS never ticks at all.  The gate itself converts 100 % of the
    //! verdicts it is given, so a 0 here is upstream of the gate, never in it.
    std::uint64_t ll_ticks;        //!< calls into livelock_probe_tx_tick
    std::uint64_t ll_resets;       //!< ... that returned early, linkage changed
    std::uint64_t ll_no_tags;      //!< ... blocked by tags_owned != tags_total
    std::uint64_t ll_few_retries;  //!< ... blocked by the retry threshold alone
    std::uint64_t ll_verdicts;     //!< ... that returned LIVELOCK
    std::uint64_t ll_rt_fast;      //!< ... of those, via the RT fast path
    //! The retry threshold turned out to be the largest blocker on both hosts,
    //! so these say by HOW MUCH — and guard against an inference that looked
    //! safe and is not.  "Outer attempts peak at 3, so m_tx_retry_count peaks
    //! at 2, below the floor of 3" ignores Node::snapshot()'s own retry loop,
    //! which increments the SAME field live (transaction_impl.h:2214) and only
    //! restores it when GuardSnapshotRetryCount goes out of scope.  A probe
    //! tick taken from inside that loop therefore sees a value the outer
    //! attempt count does not bound.  Measure the margin, do not derive it.
    std::uint64_t ll_retry_max;    //!< max my_tx_retries seen at any tick
    std::uint64_t ll_retry_sum;    //!< ... summed, for a mean
    std::uint64_t ll_thresh_max;   //!< max clamp(sig_C*2, 3, nproc) seen
    //! tags_total (= m_tagged_linkages.size(), "L") at each tick.  There to
    //! test the analytic retry bound Rule 0c makes possible: a HIGHEST Tx
    //! can lose a linkage at most TWICE — once on the retry==0 fast path,
    //! which CASes with no tag planted, and once in the race between that
    //! CAS failing and the scope dtor planting the tag — after which Rule 0c
    //! forbids any lower-priority overwrite.  So retries <= 2L.  Printed
    //! beside ll_retry_max so the two read against each other in one run,
    //! and KAME_MIX_LEAVES sweeps L directly, which turns the bound into a
    //! SLOPE rather than a single coincidence.
    std::uint64_t ll_tags_max;
    std::uint64_t ll_tags_sum;
    //! Wall time inside bundle() / unbundle(), which is the one term the tail
    //! investigation asserted and never multiplied out.  "A failed attempt
    //! re-bundles the subtree and throws it away" has the right shape, but
    //! bounding one bundle pass by the SUCCESSFUL commit phase (1,199 ns,
    //! which contains one) makes 2 entries x bundle+unbundle come to 4.8 us
    //! against a measured 15.6 us per failed attempt — short by 3.2x, and by
    //! 6.5x if only bundle is counted.  bundle_cas_retries is 0.00, so it is
    //! not spinning either.  Nothing here could close that gap because
    //! nothing timed the pass; these do.
    //!
    //! Both functions RECURSE (bundle bundles its children), so the timer runs
    //! only at depth 0 — otherwise a 3-level subtree would report its own time
    //! three times over.  `*_calls` counts outermost passes, `*_calls_all`
    //! every level, and their ratio is the fan-out per pass.
    std::uint64_t bundle_ns, bundle_calls, bundle_calls_all;
    std::uint64_t unbundle_ns, unbundle_calls, unbundle_calls_all;
    int           bundle_depth, unbundle_depth;   //!< not counters
    //! Set by the round loop, read by negotiate_sleep — not a counter.
    std::uint8_t  exempt_round;
};
inline NegDiag &neg_diag() { static thread_local NegDiag d{}; return d; }
//! Times the OUTERMOST call only; see NegDiag::bundle_ns.  Counting at every
//! level and timing at one is deliberate — the fan-out and the cost are
//! different questions and a nested timer answers neither.
struct ScopedPassTimer {
    ScopedPassTimer(std::uint64_t NegDiag::*ns, std::uint64_t NegDiag::*calls,
                    std::uint64_t NegDiag::*all, int NegDiag::*depth) noexcept
        : m_ns(ns), m_calls(calls), m_depth(depth) {
        auto &d = neg_diag();
        ++(d.*all);
        m_outer = (d.*depth == 0);
        ++(d.*depth);
        if(m_outer) m_t0 = std::chrono::steady_clock::now();
    }
    ~ScopedPassTimer() {
        auto &d = neg_diag();
        --(d.*m_depth);
        if( !m_outer) return;
        d.*m_ns += (std::uint64_t)std::chrono::duration_cast<
            std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - m_t0).count();
        ++(d.*m_calls);
    }
    std::uint64_t NegDiag::*m_ns;
    std::uint64_t NegDiag::*m_calls;
    int NegDiag::*m_depth;
    std::chrono::steady_clock::time_point m_t0;
    bool m_outer;
};
}
//! Snapshot this thread's negotiation breakdown (and optionally zero it).
//! Only compiled when KAME_STM_NEG_DIAG=1; callers guard on the same macro.
inline detail::NegDiag neg_diag_snapshot(bool reset) {
    detail::NegDiag out = detail::neg_diag();
    if(reset) detail::neg_diag() = detail::NegDiag{};
    return out;
}
#endif

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
    // ---- RT fast privilege (opt-in) -----------------------------------
    // KAME_STM_RT_FAST_PRIV=N (N >= 1): a transaction that is BOTH
    // STM-HIGHEST and under an OS realtime policy (SCHED_FIFO/RR) claims
    // privilege as soon as it has been forced to retry N times (N=2
    // recommended) — bypassing the organic threshold clamp(sig_C*2,3,nproc)
    // AND the tags_owned == tags_total condition, which the phase
    // instrumentation showed blocking 8.5 ticks per slow commit while the
    // snapshot loop rebuilds under peer fire.  The claim path upgrades only
    // the slots this Tx still owns, so partial ownership is fine; hence
    // tags_owned >= 1, not == tags_total.
    //
    // Evaluated BEFORE the per-linkage window reset below, deliberately:
    // this check uses none of the window state, and a multi-nodal commit
    // resets the window on 26-34 % of ticks — placed after it, the fast
    // path would forfeit a third of its firing opportunities.
    //
    // WITH THIS OFF THERE IS NO BOUND on the rebuild count.  The organic
    // gate's binding condition is tags_owned == tags_total, which is
    // race-dependent rather than a counter, so crossing any retry threshold
    // grants nothing: measured retries reached 10 against a threshold of 4,
    // with 8.5 of 11.8 probe ticks per slow commit blocked by that
    // condition alone.  Every MAX published for this workload is an
    // observed maximum, not a guarantee.
    //
    // MEASURED NULL (settled 2026-08-11, three 300 s arms per side on the
    // RT host): slow commits 62 vs 50 per 900 s (Poisson-overlapping), p50 /
    // p99.9 / p99.99 / p99.999 identical in all six runs, MAX bands
    // 24.3-34.6 vs 22.8-46.6 us — single samples, overlapping.  The trigger
    // fires ~12/s and every grant converts, and the tail does not move in
    // either direction.  (A first 90 s pass had concluded "measurably
    // worse" from one arm of each; that died in the repeat, and the 6x
    // data set buries it.)  Why it is null is visible in the counters: even
    // with the fast path granting, no_tags still blocks 6-9 probe ticks per
    // slow commit — the rebuild storm the grant is supposed to end keeps
    // OVERWRITING the holder's tags, so the grant neither spreads nor
    // sticks.  That overwrite is what Rule 0c (tag_as_contender) removes —
    // and measured, Rule 0c delivers what this trigger could not: slow
    // commits 62 -> 15 per 900 s and organic grants ~30x, with this knob
    // off.  Default stays OFF; the numbers live at the Rule 0c comment.
    //
    // WHY NO EXPIRY VALVE (a decision, 2026-08-10): a preempted holder's
    // Reserved stamp blocks that linkage's contenders until the holder runs
    // again — but that exposure is not new, it is the SAME one every organic
    // NORMAL/HIGHEST privilege grant has carried all along (no expiry above
    // the LOW band; ~25 grants per 90 s since the sysfs fix).  The gate here
    // is exactly the deployment doctrine this library already ships —
    // SCHED_FIFO together with core isolation, where invol = 0 was measured
    // — and a HIGHEST+FIFO thread on its own core completes in ~10 us.
    // FIFO without isolation is documented as catastrophic with or without
    // this feature.  The 5 s negotiation HANG watchdog stays the backstop.
    // Windows never takes this path (os_sched_rt is false there).
    static const int s_rt_fast_retries = []{
        const char *e = std::getenv("KAME_STM_RT_FAST_PRIV");
        if( !e || !*e) return 0;
        const int v = std::atoi(e);
        return (v >= 1) ? v : 0;
    }();
    if(s_rt_fast_retries > 0 && prio == Priority::HIGHEST
            && (int)my_tx_retries >= s_rt_fast_retries
            && tags_owned >= 1
            && detail::os_sched_rt()) {
#if KAME_STM_NEG_DIAG
        {   auto &_d = detail::neg_diag();
            ++_d.ll_ticks;      // still a tick; keep the partition exhaustive
            ++_d.ll_verdicts;   // a verdict is a verdict...
            ++_d.ll_rt_fast;    // ...this one via the fast path
            _d.ll_retry_sum += my_tx_retries;
            if(my_tx_retries > _d.ll_retry_max) _d.ll_retry_max = my_tx_retries;
            _d.ll_tags_sum += (std::uint64_t)tags_total;
            if((std::uint64_t)tags_total > _d.ll_tags_max)
                _d.ll_tags_max = (std::uint64_t)tags_total;
        }
#endif
        // The probe window state is left untouched: it belongs to the
        // organic detector, and the next organic tick handles a linkage
        // change exactly as it would have.
        return true;
    }
    // ---- organic livelock detection ------------------------------------
    auto &p = LivelockProbe::state();
#if KAME_STM_NEG_DIAG
    ++detail::neg_diag().ll_ticks;
    if(p.linkage_id != linkage) ++detail::neg_diag().ll_resets;
#endif
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
    //
    // CACHED, and the cache is not an optimisation nicety.  On Linux/glibc,
    // std::thread::hardware_concurrency() is get_nprocs(), which is an
    // openat+read+close of /sys/devices/system/cpu/online ON EVERY CALL —
    // three syscalls through PTI+IBRS, inside the negotiation of a realtime
    // commit.  Found by Intel PT on the RT host, not by reading: the sparse
    // trace windows of the slow-commit tail were __read_nocancel /
    // __close_nocancel_nostatus / memchr / strtoul clusters, and the
    // arithmetic closed — 5,565 ns of unattributed retry cost per slow
    // commit over 2.83 probe ticks is 1,966 ns per tick, a 3-syscall sysfs
    // read.  A cycles profile could never see it (0.27 % of total time).
    // effective_runners() three hundred lines down already caches the same
    // call with the same `static const` pattern; this site predates it.
    // Hotplug caveat: the value is now process-lifetime.  A stale cap only
    // shifts this heuristic clamp, while the per-call read put syscalls
    // into an RT thread's commit path — strictly worse.
    static const int s_hw_procs = []{
        int h = (int)std::thread::hardware_concurrency();
        return h > 0 ? h : 4;
    }();
    const int hw_procs = s_hw_procs;
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
#if KAME_STM_NEG_DIAG
    {   auto &_d = detail::neg_diag();
        const bool _tags_ok = (tags_total > 0 && tags_owned == tags_total);
        const bool _retry_ok = ((int)my_tx_retries >= retry_thresh_dyn);
        if(saw_livelock)      ++_d.ll_verdicts;
        else if( !_tags_ok)   ++_d.ll_no_tags;
        else if( !_retry_ok)  ++_d.ll_few_retries;
        _d.ll_retry_sum += my_tx_retries;
        if(my_tx_retries > _d.ll_retry_max) _d.ll_retry_max = my_tx_retries;
        if((std::uint64_t)retry_thresh_dyn > _d.ll_thresh_max)
            _d.ll_thresh_max = (std::uint64_t)retry_thresh_dyn;
        _d.ll_tags_sum += (std::uint64_t)tags_total;
        if((std::uint64_t)tags_total > _d.ll_tags_max)
            _d.ll_tags_max = (std::uint64_t)tags_total;
    }
#endif

    p.t_window_us      = now_us;
    p.tx_retry_window  = my_tx_retries;
    p.tx_commit_window = tx_commit_count;
    return saw_livelock;
}


template <class XN>
void Node<XN>::NegotiationCounter::negotiate_sleep(
    int ms_timeout, cnt_t my_stamp, unsigned us_override) noexcept
{
    int slot = (int)((unsigned)ProcessCounter::id() % NEGOTIATE_SLEEP_SLOTS);
    auto &st = s_sleep_slots[slot];
    // Snapshot the kind this thread is about to commit; the notifier
    // reads this field (lock-free) to bias wake-up toward the same kind
    // as the linkage's most recent commit (see `notify_n_contenders`
    // preferred_kind argument).
    const uint8_t my_kind = (uint8_t)*detail::s_current_op_kind & 0x3u;
    // Read the cell generation BEFORE publishing op_kind/stamp.  A
    // wake_one() racing in after this read advances the generation, so
    // the value-compare in cell.wait() returns at once — the
    // lost-wakeup window is closed without a mutex, and the generation
    // subsumes the former `notified` flag + its reset race.
    uint32_t g = st.cell.gen();
    // Publish op_kind/stamp for the waker's lock-free, best-effort
    // tenant/kind targeting.  A racy read only changes WHICH sleeper a
    // waker picks (mis-target → natural timeout; spurious wake →
    // re-check + re-sleep), never correctness.
    st.op_kind.store(my_kind, std::memory_order_relaxed);
    st.stamp.store(my_stamp, std::memory_order_release);
    // Physical chunk length = ms_timeout * KAME_NEG_SLEEP_US_PER_MS µs
    // (default 1000 → the original 1 ms; smaller tightens the re-check /
    // notify cadence now that __ulock makes sub-ms waits cheap).
    unsigned us = us_override ? us_override
        : ((ms_timeout > 0)
           ? (unsigned)ms_timeout * (unsigned)KAME_NEG_SLEEP_US_PER_MS : 0u);
#if KAME_STM_NEG_DIAG
    {
        auto &d = detail::neg_diag();
        ++d.sleeps;
        d.req_ns += (std::uint64_t)us * 1000ull;
        auto t0 = std::chrono::steady_clock::now();
        st.cell.wait(g, us);
        const std::uint64_t _dt = (std::uint64_t)std::chrono::duration_cast<
            std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count();
        d.slept_ns += _dt;
        if(d.exempt_round) d.slept_exempt_ns += _dt;
        const std::uint64_t _want = (std::uint64_t)us * 1000ull;
        if(_dt > _want && _dt - _want > d.late_max_ns)
            d.late_max_ns = _dt - _want;
    }
#else
    st.cell.wait(g, us);
#endif
    // Clear the tenant stamp on exit so the next sleeper's stamp is not
    // preceded by a stale value that could match a target.
    st.stamp.store(0, std::memory_order_relaxed);
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
        s_sleep_slots[priv_slot].cell.wake_one();
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
                int bit = ctz_u64(word);
                word &= word - 1;
                int slot = (int)(((unsigned)(i * 64 + bit)) % NEGOTIATE_SLEEP_SLOTS);
                if (slot == priv_slot) continue;
                if (is_woken(slot)) continue;
                auto &st = s_sleep_slots[slot];
                if(st.op_kind.load(std::memory_order_relaxed) != preferred_kind)
                    continue;
                st.cell.wake_one();
                mark_woken(slot);
                --n;
            }
        }
    }
    for(int i = 0; i < TidBitset::WORDS && n > 0; ++i) {
        uint64_t word = tid_bitset.word(i);
        while(word && n > 0) {
            int bit = ctz_u64(word);
            word &= word - 1;
            int slot = (int)(((unsigned)(i * 64 + bit)) % NEGOTIATE_SLEEP_SLOTS);
            if (slot == priv_slot) continue;
            if (has_pref && is_woken(slot)) continue;
            s_sleep_slots[slot].cell.wake_one();
            if(has_pref) mark_woken(slot);
            --n;
        }
    }
}

// Historically `try_*` differed from notify_n_contenders by taking the
// slot lock with std::try_to_lock and SKIPPING any slot whose lock was
// momentarily held (to keep the notifier off the critical path).  With
// the mutex-less XWaitCell wake_one() that distinction is gone — the
// wake never blocks — so this now reliably delivers every wake the
// kind/bitset selects (an improvement: it no longer drops wakes on lock
// contention).  Kept as a distinct entry point (no privileged-TID first
// pass) for its call sites.
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
                int bit = ctz_u64(word);
                word &= word - 1;
                int slot = (int)(((unsigned)(i * 64 + bit)) % NEGOTIATE_SLEEP_SLOTS);
                if (is_woken(slot)) continue;
                auto &st = s_sleep_slots[slot];
                if(st.op_kind.load(std::memory_order_relaxed) != preferred_kind)
                    continue;
                st.cell.wake_one();
                mark_woken(slot);
                --n;
            }
        }
    }
    for(int i = 0; i < TidBitset::WORDS && n > 0; ++i) {
        uint64_t word = tid_bitset.word(i);
        while(word && n > 0) {
            int bit = ctz_u64(word);
            word &= word - 1;
            int slot = (int)(((unsigned)(i * 64 + bit)) % NEGOTIATE_SLEEP_SLOTS);
            if (has_pref && is_woken(slot)) continue;
            s_sleep_slots[slot].cell.wake_one();
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
// dt2 of the most recent negotiate() call — used by the adaptive
// fairness gate.  Tags disambiguate sibling counters sharing the same
// scalar type.
namespace {
struct STagAdaptDt2LastUs;
struct STagAdaptCLast;
struct STagAdaptLastPriorityTid;
struct STagAdaptBounceCount;
struct STagAdaptNegotiateCalls;
struct STagAdaptSkipHits;
struct STagAdaptSkipPer1k;
}
XThreadLocal<uint64_t, STagAdaptDt2LastUs>        s_adapt_dt2_last_us;
XThreadLocal<int,      STagAdaptCLast>            s_adapt_C_last;
XThreadLocal<uint32_t, STagAdaptLastPriorityTid>  s_adapt_last_priority_tid;
XThreadLocal<uint32_t, STagAdaptBounceCount>      s_adapt_bounce_count;
XThreadLocal<uint64_t, STagAdaptNegotiateCalls>   s_adapt_negotiate_calls;

// Per-Linkage privilege diagnostic counters declared in transaction_impl.h
// (g_neg_claim_attempts / g_neg_claim_successes /
//  g_neg_internal_calls_non_priv / g_neg_internal_calls_priv).
extern std::atomic<uint64_t> g_neg_claim_attempts;
extern std::atomic<uint64_t> g_neg_claim_successes;
extern std::atomic<uint64_t> g_neg_internal_calls_non_priv;
extern std::atomic<uint64_t> g_neg_internal_calls_priv;
XThreadLocal<uint64_t, STagAdaptSkipHits>  s_adapt_skip_hits;   // lease-skip fires
XThreadLocal<uint32_t, STagAdaptSkipPer1k> s_adapt_skip_per1k;  // skip_hits/calls × 1000
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
    if(entry_pr == Priority::LOWEST ||
        entry_pr == Priority::UI_DEFERRABLE || entry_pr == Priority::SCRIPTING)
        return false;
    // transaction_started_time is tid+kind+us-packed; diff_us_packed
    // extracts the µs and applies modular subtraction (wrap-safe).
    auto adapt_dt2_last_us =
        NegotiationCounter::diff_us_packed(
            now_us_entry, transaction_started_time);

#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
    *s_adapt_dt2_last_us = adapt_dt2_last_us;
    *s_adapt_C_last = sig_C;
    if(ps.tid && ps.tid != *s_adapt_last_priority_tid) {
        ++*s_adapt_bounce_count;
        *s_adapt_last_priority_tid = ps.tid;
    }
    ++*s_adapt_negotiate_calls;
    *s_adapt_skip_per1k = *s_adapt_negotiate_calls > 0
                             ? (uint32_t)((uint64_t)(*s_adapt_skip_hits) * 1000
                                           / *s_adapt_negotiate_calls)
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
    // Mask at STAMP_TID_BITS width so the comparison against ps.tid
    // (which carries the truncated TID under compact mode) is consistent.
    unsigned my_tid = Node<XN>::NegotiationCounter::my_tid_lo();
#if KAME_STM_MIN_RUNNERS != 0
    const int min_r_pre = effective_min_runners(1);
    if(NegotiationCounter::numThreadsRunning((unsigned)min_r_pre) < (unsigned)min_r_pre)
#endif
    if(my_tid == ps.tid
        && adapt_dt2_last_us < (uint64_t)KAME_DT2_FAIRNESS_US) {
        // Age in µs via modular 32-bit subtraction (wrap-safe up to ~35 min).
        uint32_t age_us = (uint32_t)now_us_entry - ps.start_us;
        if(age_us < (uint32_t)ps.lease_us) {
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
            ++*s_adapt_skip_hits;
#endif
            if(entry_pr == Priority::HIGHEST || entry_pr == Priority::NORMAL
                    || entry_pr == Priority::SCRIPTING)
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

    const uint8_t mk = (uint8_t)*detail::s_current_op_kind & 0x3u;
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
    {
        const unsigned max_r = (unsigned)effective_max_runners(C_obs);
        runners_ok = NegotiationCounter::numThreadsRunning(max_r) < max_r;
    }
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
#if KAME_STM_NEG_DIAG
    ++detail::neg_diag().entries;
#endif
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
#ifndef NDEBUG
    // Reached only under real contention, i.e. exactly when this call may sleep.
    // Sleeping here while holding a plain lock that a peer's transaction can
    // block on is the 2026-07-10 deadlock; see the foreign-lock block in
    // transaction_detail.h.  Reported once per thread, never fatal — a
    // diagnostic has no business aborting a measurement.
    if(foreignLockDepth() > 0) [[unlikely]] {
        static thread_local bool s_told = false;
        if( !s_told) {
            s_told = true;
            std::fprintf(stderr,
                "kamestm: negotiating (and possibly sleeping) while holding %d "
                "foreign lock(s).  A peer transaction blocking on that lock "
                "cannot finish, and neither can this one — the 2026-07-10 "
                "negotiation stall.  Copy what you need under a short lock, "
                "release it, then take the Snapshot/Transaction.\n",
                foreignLockDepth());
        }
    }
#endif
    //! Wait budget (absolute µs, 0 = none) — see Transactional::ScopedWaitBudget.
    //!
    //! Read here, once per call, and NOT captured in the Snapshot.  An earlier
    //! version stored it in a `Snapshot::m_wait_limit` filled by the ctors that
    //! stamp `m_started_time`, on the theory that the sleep loop should read a
    //! member rather than TLS.  Measured, that cost 1.9 % at 8 threads and
    //! 0.9 % at 4 (7 interleaved reps; lower in 6 of 7 at 4 threads) — because
    //! the ctors are the hot path and this function is not.  The member bought
    //! nothing anyway: the value is hoisted into this local once per call, so
    //! the loop never touched TLS in either version.  Reading live is also the
    //! more honest semantics for an ambient budget.
    //!
    //! Every use below is guarded on it being nonzero, so a thread that never
    //! constructs a ScopedWaitBudget executes exactly the pre-existing code.
#if KAME_STM_WAIT_BUDGET
    // Filled from the same single TLS read that yields `entry_pr` below.
    int64_t _wb_limit = 0;
#else
    // Compiled out: every `if(_wb_limit && ...)` below folds to nothing, so
    // one gate covers all five use sites without scattering #if through the
    // sleep loop.
    constexpr int64_t _wb_limit = 0;
#endif
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
#if KAME_STM_WAIT_BUDGET
    const Priority entry_pr = [&]{ const auto &c = currentTxContext();
                                   _wb_limit = c.wait_limit;
                                   return c.priority; }();
#else
    const Priority entry_pr = getCurrentPriorityMode();
#endif

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
            const auto my_id = NegotiationCounter::strip_kind(snap.m_started_time);
            const auto my_priv = NegotiationCounter::with_kind(
                snap.m_started_time, detail::StampKind::Reserved);
            // No holder-class side word to pre-publish: `my_priv` is
            // `m_started_time` with the kind bits swapped for Reserved, so it
            // still carries this Tx's PRIO field and the Reserved stamp
            // describes its own class.  (It reports the class the Tx STARTED
            // at rather than `entry_pr`, the class at negotiation entry.  Those
            // differ only for a thread that changed tier mid-Tx, and the stamp
            // is the one the peers will read.)
            for (auto &l : snap.m_tagged_linkages) {
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
            // Global mode plants no per-Linkage Reserved stamps, so
            // tag_as_contender's Rule 0 (which requires one) stays inert —
            // conservative by construction.
            claimed = NegotiationCounter::try_register_privileged_tidstamp(
                          entry_pr, snap.m_started_time);
#endif
#if KAME_STM_NEG_DIAG
            ++detail::neg_diag().priv_tries;
            if(claimed) ++detail::neg_diag().priv_grants;
#endif
            if (claimed) {
                snap.m_registered_privileged = true;
                // Pair with the decrement in
                // `Snapshot::drop_tags_n_privilege`.
                s_num_privileged_threads.fetch_add(1, std::memory_order_relaxed);
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
                g_neg_claim_successes.fetch_add(1, std::memory_order_relaxed);
#endif
#if KAME_STM_PRIV_DIAG
                std::fprintf(stderr,
                    "[ll-probe] privileged_tid=%u age=%lld us prio=%d N=%d\n",
                    (unsigned)NegotiationCounter::stamp_tid(snap.m_started_time),
                    (long long)_ll_age_us, (int)entry_pr,
                    (int)NegotiationCounter::numThreadsRunning());
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
    // Self-tagged short-circuit (KAME_STM_OPTIONAL_OPTIMIZATION).
    // If the slot's TID matches ours, this thread tagged the linkage
    // — the lease drift, numThreadsRunning probe in `_neg_apply_lease`,
    // backoff init, and the `dt <= 0` loop bail-out below are all
    // wasted work on a self-encounter.  Skip them all here.
#if defined(KAME_STM_OPTIONAL_OPTIMIZATION) && KAME_STM_OPTIONAL_OPTIMIZATION
    if(NegotiationCounter::stamp_tid(transaction_started_time)
       == NegotiationCounter::stamp_tid(started_time))
        return;
#endif
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

    // Per-call hang counter (counts how many times we've hit the
    // "ms > 5000" sleep-cap branch).  After KAME_STM_HANG_ABORT_N such hits
    // we abort() for a core dump + stack trace; 0 disables the abort while
    // keeping the [HANG] dumps.
    //
    // **Release default is 0 — the watchdog reports, it does not kill —
    // since 2026-07-31 (user).**  The abort was tuned for true deadlocks,
    // but the 2026-07-30/31 field incidents showed it executing recoverable
    // states: one freeze self-recovered at 11 s (4 s short of the abort),
    // another was a LIVE privilege holder legitimately grinding for 33+ s —
    // waiting behind it is the completion guarantee working, and killing the
    // process took the unsaved measurement with it.  With the orphaned-stamp
    // class fixed at the source (ctor exception safety) and the fair-mode
    // immunities removed (STM-HIGHEST retired, budget exempted), a >15 s
    // wait behind a live holder is contract-legitimate; a true deadlock is
    // diagnosed from the [HANG] dumps + `sample` and killed by the operator,
    // who first gets to save.  Debug builds keep 3: there the core dump IS
    // the point.
#ifndef KAME_STM_HANG_ABORT_N
    #ifdef NDEBUG
        #define KAME_STM_HANG_ABORT_N 0
    #else
        #define KAME_STM_HANG_ABORT_N 3
    #endif
#endif
    int _hang_hits = 0;

    for(int ms = 0;;) {
#if KAME_STM_NEG_DIAG
        ++detail::neg_diag().rounds;
#endif
        // Wait budget, checked at the TOP because that is the only point every
        // path through a round passes.  A tail-only check is bypassed by the
        // fair-spin `continue` below, which measured 14.23 rounds/commit and
        // 0.18 sleeps/commit against a 1 us budget.
        //
        // GATED on fair_mode_blocks_me since 2026-07-31 — this reverses the
        // original rule, and the reversal is measurement, not caution.  The
        // old rationale ("returning is not barging — the caller's CAS loses
        // to a committing holder the same as any other loser") assumed the
        // holder commits in microseconds.  A privilege holder with a long
        // closure (the 20 ms PNR analysis) broke it: budget-expired record
        // paths became fair-mode-IMMUNE spinners — the same disease that
        // retired STM-HIGHEST the same day — re-invalidating the holder every
        // closure (re-runs 1.1 -> 2.3) while honest negotiators pinned behind
        // its privilege for 12+ s (372 HANG dumps vs 0 without budgets, in
        // the field-parameter harness).  Principle: privilege is the
        // completion guarantee, and NOTHING may be immune to it.  The budget
        // bounds every OTHER wait (lottery, runner gate, ladder); the wait
        // behind a live privileged peer is contractually exempt — declining
        // it is what freezes the system.  (Expired-lowprio stamps unblock
        // inside fair_mode_blocks_me as always, so a dead holder cannot pin
        // a budgeted thread either.)
        if(_wb_limit && NegotiationCounter::now_us() >= _wb_limit
                && !NegotiationCounter::fair_mode_blocks_me(started_time, self))
            break;
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
            return (uint8_t)*detail::s_current_op_kind & 0x3u;
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
        // The budget's sleep clamps are suspended while fair-blocked (see the
        // loop-top comment): otherwise an expired budget shrinks the CV waits
        // to zero and the thread busy-spins behind the holder instead of
        // waiting — cheaper than barging, but still a wasted core.
        const int64_t _wb_round = _fair_blocks ? 0 : _wb_limit;
#if KAME_STM_NEG_DIAG
        {   auto &_d = detail::neg_diag();
            _d.exempt_round = (_wb_limit && !_wb_round) ? 1u : 0u;
            if(_d.exempt_round) ++_d.rounds_exempt; }
#endif

#if KAME_NEGSITE_ENABLED
        NegSite::last_was_gate_return() = false;
#endif
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
            const detail::StampKind my_kind = *detail::s_current_op_kind;
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
                if(NegotiationCounter::numThreadsRunning((unsigned)max_r) < (unsigned)max_r)
#endif
                    break; // gate: earned priority — always proceeds, never capped
            }
    // KAME_STM_DISABLE_LOTTERY lives in transaction_definitions.h.
#if !KAME_STM_DISABLE_LOTTERY
            // (b) C fairness lottery: LOTTERY_MULT*C threads bypass per iteration.
            //     Prevents all threads from being stuck in the gate simultaneously.
#if KAME_STM_MIN_RUNNERS != 0
            const int min_r_lot = effective_min_runners(C_obs);
            if(NegotiationCounter::numThreadsRunning((unsigned)min_r_lot) < (unsigned)min_r_lot) {
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
            ++_hang_hits;
            // Comprehensive hang-state dump.  Repeats every hit so a
            // sustained hang prints a trail of identical/evolving state;
            // abort after KAME_STM_HANG_ABORT_N hits if enabled.
            using NC = NegotiationCounter;
            fprintf(stderr,
                "Nested transaction?, Negotiating, %f sec. requested, "
                "limited to 5s. for BP@%p\n"
                "  [HANG#%d] self_tid=%u self.started_us=%lld self.kind=%u "
                "priv_flag=%d age_us=%lld\n"
                "  [HANG#%d] slot=0x%llx slot.tid=%u slot.started_us=%lld "
                "slot.kind=%u is_priv=%d\n"
                "  [HANG#%d] dt=%lld dt2=%lld sig_C=%d fair_blocks=%d "
                "tagged_n=%d tx_retry=%u commit=%llu\n",
                ms * 1e-3, (void*)self,
                _hang_hits,
                (unsigned)NC::stamp_tid(started_time),
                (long long)NC::stamp_us(started_time),
                (unsigned)NC::stamp_kind(started_time),
                (int)snap.m_registered_privileged,
                (long long)_ll_age_us,
                _hang_hits,
                (unsigned long long)transaction_started_time,
                (unsigned)NC::stamp_tid(transaction_started_time),
                (long long)NC::stamp_us(transaction_started_time),
                (unsigned)NC::stamp_kind(transaction_started_time),
                (int)NC::is_priv_stamp(transaction_started_time),
                _hang_hits,
                (long long)dt, (long long)dt2, sig_C, (int)_fair_blocks,
                (int)snap.m_tagged_linkages.size(),
                (unsigned)snap.m_tx_retry_count,
                (unsigned long long)self->m_tx_commit_count);
            // Dump every tagged Linkage's current slot stamp so we can
            // see who else is holding which slot.
            int _idx = 0;
            for(auto &_l : snap.m_tagged_linkages) {
                if(!_l) { ++_idx; continue; }
                auto _cur = _l->m_transaction_started_time.load(
                    std::memory_order_relaxed);
                fprintf(stderr,
                    "  [HANG#%d]   link[%d]=%p stamp=0x%llx tid=%u "
                    "started_us=%lld kind=%u priv=%d\n",
                    _hang_hits, _idx, (void*)_l.get(),
                    (unsigned long long)_cur,
                    (unsigned)NC::stamp_tid(_cur),
                    (long long)NC::stamp_us(_cur),
                    (unsigned)NC::stamp_kind(_cur),
                    (int)NC::is_priv_stamp(_cur));
                ++_idx;
            }
            fflush(stderr);
#if KAME_STM_HANG_ABORT_N
            if(_hang_hits >= KAME_STM_HANG_ABORT_N) {
                fprintf(stderr,
                    "  [HANG] %d consecutive 5s-cap hits — aborting for "
                    "core dump.  Set -DKAME_STM_HANG_ABORT_N=0 to disable.\n",
                    _hang_hits);
                fflush(stderr);
                std::abort();
            }
#endif
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
                //
                // Bounded: previously unbounded ("expose programming
                // error if peer priv held forever").  Diagnosis
                // showed real cycles where peer-priv stays Reserved
                // because the holder is itself CV-sleeping waiting
                // on us — the spin would run forever.  Cap at
                // KAME_STM_FAIR_SPIN_MAX_US (default 2 ms); on
                // timeout fall through to the CV-sleep path below,
                // which (with Fix A) drops our own Reserved before
                // sleeping so the cycle can break.
#ifndef KAME_STM_FAIR_SPIN_MAX_US
#define KAME_STM_FAIR_SPIN_MAX_US 2000
#endif
                const int64_t _spin_start_us =
                    (int64_t)NegotiationCounter::now_us();
                // A 2 ms busy-spin is longer than most budgets; end it at
                // whichever comes first.
                const int64_t _spin_deadline_us =
                    (_wb_round && _wb_round - _spin_start_us
                                  < KAME_STM_FAIR_SPIN_MAX_US)
                        ? _wb_round
                        : _spin_start_us + KAME_STM_FAIR_SPIN_MAX_US;
                unsigned iter = 0;
                bool _spin_timed_out = false;
                do {
                    pause4spin();
                    if((++iter & 0x3FFFFu) == 0) {
                        std::this_thread::yield();
                        if((int64_t)NegotiationCounter::now_us()
                           > _spin_deadline_us) {
                            _spin_timed_out = true;
                            break;
                        }
                    }
                } while(NegotiationCounter::fair_mode_blocks_me(
                                started_time, self));
                s_fair_spinners.fetch_sub(1, std::memory_order_relaxed);
#if KAME_STM_NEG_DIAG
                {   auto &_d = detail::neg_diag();
                    ++_d.spins;
                    _d.spin_ns += (std::uint64_t)
                        ((int64_t)NegotiationCounter::now_us()
                         - _spin_start_us) * 1000ull; }
#endif
                if( !_spin_timed_out)
                    continue;
                // Timed out: fall through to CV-sleep section so we
                // can yield + (Fix A) drop our own Reserved.
            }
        }
#endif

        // Sleep ms in 1-ms CV chunks + random ±1ms de-phasing jitter.
        // Jitter breaks the synchronized-wakeup oscillation that forms when
        // all threads enter and exit negotiate_sleep at the same 1 ms tick.
        //
        // (Fix A) priv-no-sleep: if we hold privilege and are about
        // to CV-sleep, that breaks the design invariant "priv should
        // not yield".  Real deadlocks observed via HANG dumps showed
        // mutual cycles: priv holder CV-sleeping on Linkage X owned
        // by an older peer, while the older peer fair-spins on one
        // of our Reserved Linkages.  Drop our Reserved here so the
        // peer's fair-spin breaks; on wake-up the claim path can
        // re-fire if still needed.
        if(snap.m_registered_privileged) {
            using NC = NegotiationCounter;
            const auto _my_id = NC::strip_kind(snap.m_started_time);
            for(auto &_l : snap.m_tagged_linkages) {
                auto _cur = _l->m_transaction_started_time.load(
                    std::memory_order_relaxed);
                if(NC::is_priv_stamp(_cur)
                   && NC::strip_kind(_cur) == _my_id) {
                    auto _exp = _cur;
                    _l->m_transaction_started_time.compare_exchange_strong(
                        _exp, (typename NC::cnt_t)0,
                        std::memory_order_release,
                        std::memory_order_relaxed);
                }
            }
            snap.m_registered_privileged = false;
            NC::release_priv_count_slot();
        }
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
        const unsigned _nrun_y = NegotiationCounter::numThreadsRunning(3);
        if(_nrun_y <= 2 && ms <= 1) {
            if(_nrun_y <= 1) {
                // Sole runner: every other contender is CV-asleep, so a
                // yield hands the core to no STM-runnable thread AND the
                // ReleaseOneCount would drop the running count to 0 —
                // risking a scheduler gap where nobody makes progress
                // (the asleep peers only wake on a notify we are not
                // sending here, or on their ~1 ms CV timeout).  We are
                // the only thread that can make forward progress, so keep
                // the running slot and stay on-CPU with a brief pause,
                // then retry the CAS — no voluntary yield.  (This does
                // NOT defend against the OS *involuntarily* time-slicing
                // the sole runner; that stall is bounded only by the
                // sleepers' CV-chunk timeout / privilege expiry.)
                pause4spin();
            }
            else {
                // Genuine 2-thread alternation: yield so the OS can swap
                // to the other runnable contender, which then commits.
                typename NegotiationCounter::ReleaseOneCount onedown;
                std::this_thread::yield();
            }
        }
        else {
            // ---- Deadline tail: do not sleep through the end of a budget.
            //
            // A timed wait cannot deliver a wake-up more precisely than the
            // host's idle-exit + timer-slack cost, and near the end of a
            // budget that cost is LARGER THAN THE WAIT ITSELF.  Measured on
            // the PREEMPT_RT reference host (i5-7500, acquisition thread
            // alone on an isolated core, 200 us budget): the last chunk was
            // clamped to 198 us exactly as designed, and `cell.wait()`
            // returned 695 us later, with 6 us of STM work in the whole
            // commit.  The "MAX = budget + ~200 us constant" recorded in
            // design/RT_READINESS.md was never the STM and never the
            // documented budget-exempt wait behind a privileged peer —
            // `rounds_exempt` is 0 across 17,274 slow commits, and stays 0
            // under every scheduling class, C-state setting and budget tried.
            //
            // What the wake-up is made of, measured directly as the worst
            // single cell.wait() overshoot (20 s arms, 20 ms budget, root):
            //
            //     plain        662 us     fifo            124 us
            //     slack 1 us   475 us     pmqos + fifo     20 us
            //     pmqos        605 us   (pm-qos verified: cpu3 C8 entries 0)
            //
            // Read the ordering, because it is counter-intuitive and the
            // obvious guess is wrong: the SCHEDULING CLASS dominates (5.3x),
            // and holding PM-QoS at 0 buys almost nothing on its own
            // (662 -> 605) while buying 6x on top of SCHED_FIFO
            // (124 -> 20).  The two are super-additive, so testing either
            // alone understates it.  An earlier reading of this attributed
            // the constant to the deepest C-state's 200 us exit latency
            // because that number matches the observed overshoot almost
            // exactly; the pmqos row above refutes it.  Timer slack (50 us by
            // default, and zero for any RT class) is a component of what FIFO
            // buys, but not most of it.
            //
            // So spend the remainder on-CPU instead.  Polling is strictly
            // better than waiting here on every axis that matters:
            //   * it observes the blocker clearing IMMEDIATELY rather than at
            //     the next wake-up, so the common case gets FASTER, not just
            //     more predictable;
            //   * the cost is bounded by the threshold and paid only by a
            //     thread that has already declared a deadline;
            //   * it removes both the C-state and the slack from the deadline
            //     path without needing root, a PM-QoS hold, or a tuned kernel.
            //
            // Gated on `_wb_round` — i.e. on the caller having constructed a
            // ScopedWaitBudget AND on not being fair-blocked — so ordinary
            // throughput callers, who have no deadline to protect and would
            // only lose a core to this, are untouched.
            //
            // Measured (same host, 25 s arms, acq at NORMAL — the shipped
            // tier — pinned alone on the isolated core), MAX-budget with the
            // reserve off -> on:
            //
            //     budget    MAX-budget          acq/s        UI/s   SCRIPT/s
            //      20 ms   122 us -> 7.1 us   +2 %          -2 %     +2 %
            //       1 ms   216 us -> 34 us    +10 %        -13 %    -15 %
            //     200 us   721 us -> 19 us    +1 %       **-94 %** **-98 %**
            //
            // …and in the configuration KAME should actually ship
            // (SCHED_FIFO + isolation + PM-QoS held at 0, 20 ms budget):
            // **76 us -> 3.0 us**, with UI and SCRIPTING both slightly UP.
            // 3 us is below this host's own 17 us floor (rtla osnoise), i.e.
            // the STM's contribution to the record commit's tail is now
            // smaller than the machine's noise.
            //
            // THE 200 us ROW IS A CLIFF, NOT A TREND, and it is the reason
            // this constant may not simply be raised.  Once the reserve
            // reaches the whole budget the thread never sleeps at all: it
            // stops backing off the linkage, wins every CAS from its own
            // uncontended core, and the deferrable roles stop committing (UI
            // 24.1k -> 1.5k /s, SCRIPTING 67.5k -> 1.1k /s).  A budget is the
            // deadline-bearer's patience, and spending all of it on-CPU is
            // indistinguishable from having none.  So: keep this WELL BELOW
            // the smallest budget in play, and before recommending budgets
            // near it, cap the reserve at a FRACTION of the budget span
            // (which means plumbing the span, not just the deadline, through
            // ScopedWaitBudget).  At KAME's shipped 20 ms the reserve is
            // 1.5 % of the budget and the row above is free.
#ifndef KAME_NEG_SPIN_TAIL_US
#define KAME_NEG_SPIN_TAIL_US 300
#endif
#if KAME_NEG_SPIN_TAIL_US > 0
            if(_wb_round) {
                int64_t _rem = _wb_round
                    - (int64_t)NegotiationCounter::now_us();
                if(_rem <= 0) goto _exit_cv_sleep;
                if(_rem <= (int64_t)KAME_NEG_SPIN_TAIL_US) {
                    // Keep the running-count slot: we ARE running.  (The
                    // sleep path below releases it precisely because it is
                    // about to stop running.)
                    unsigned _it = 0;
                    for(;;) {
                        pause4spin();
                        // Leave the moment the blocker is gone or we became
                        // the oldest — the whole point of not sleeping.
                        auto _v = self->m_transaction_started_time.load(
                            std::memory_order_relaxed);
                        if( !NegotiationCounter::is_active_stamp(_v)
                            || NegotiationCounter::signed_diff_us_packed(
                                   started_time, _v) <= 0)
                            break;
                        // now_us() is far dearer than a PAUSE; amortise it.
                        if((++_it & 0x3Fu) == 0
                           && (int64_t)NegotiationCounter::now_us()
                              >= _wb_round)
                            break;
                    }
#if KAME_STM_NEG_DIAG
                    {   auto &_d = detail::neg_diag();
                        ++_d.tail_spins;
                        _d.tail_spin_ns += (std::uint64_t)
                            (((int64_t)NegotiationCounter::now_us()
                              - (_wb_round - _rem)) * 1000); }
#endif
                    goto _exit_cv_sleep;
                }
            }
#endif
            // Do NOT drop this Tx's tags before sleeping.  It looks free —
            // a sleeper holding a tag keeps `fair_mode_blocks_me` true for
            // every peer on that linkage, measured at 38 % of sleeps in the
            // mixed arm, and clearing them measured +10 % at 128 threads.  It
            // is not free: the livelock verdict reads `tags_total` from
            // `snap.m_tagged_linkages` (see `_ll_total` above), so a Tx that
            // arrives at its next negotiator entry with an empty tag list can
            // never satisfy `tags_total > 0` and can therefore never claim
            // privilege.  Clearing here trades away the only escape from
            // starvation for throughput in a regime (128 contending threads)
            // that measured no benefit at all at 4.  This was knob
            // KAME_STM_CLEAR_TAGS_BEFORE_SLEEP; removed, see design/RT_READINESS.md.
            int ms_actual = ms;
#if KAME_STM_NEG_DIAG
            { auto &_d = detail::neg_diag();
              _d.ms_sum += (std::uint64_t)ms_actual;
              if((std::uint64_t)ms_actual > _d.ms_max)
                  _d.ms_max = (std::uint64_t)ms_actual; }
#endif
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
            // A round may not outlive the caller's wait budget.  This alone is
            // not enough — a chunk can still overshoot by its own length — so
            // the per-chunk clamp below handles the sub-millisecond tail.
            // Both stop KAME_NEG_SPIN_TAIL_US SHORT of the budget, leaving
            // that much for the deadline-tail spin above to cover: a wait
            // clamped to land exactly ON the deadline hands its own wake-up
            // latency straight to the caller's tail, which is the entire
            // measured overshoot (see the tail-spin comment).
            if(_wb_round && t_end > _wb_round - KAME_NEG_SPIN_TAIL_US)
                t_end = _wb_round - KAME_NEG_SPIN_TAIL_US;
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
                    //     Low-contention systems rarely hit this; high-contention systems
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
                        // Tenant verification: only wake when the slot's
                        // current tenant matches the blocker (same tid +
                        // same started_us).  Comparison strips the kind
                        // bits because the linkage slot is stamped via
                        // `with_kind(started_time, op_kind)` in
                        // `tag_as_contender` while the sleep slot stores
                        // the bare `started_time` (kind=NONE).  The stamp
                        // is read lock-free (acquire) — a racy mismatch
                        // only mis-targets this wake; the blocker falls
                        // back to its natural timeout.
                        if(NegotiationCounter::strip_kind(
                               _st.stamp.load(std::memory_order_acquire))
                           == NegotiationCounter::strip_kind(_slot_val)) {
                            _st.cell.wake_one();
                        }
                    }
                }
#if KAME_STM_DISABLE_JITTER
#if KAME_STM_NEG_DIAG
                {   // (diag) do we still own tags while going to sleep?
                    int _held = 0;
                    const auto _mine = NegotiationCounter::strip_kind(
                                            snap.m_started_time);
                    for(auto &&_l : snap.m_tagged_linkages) {
                        if( !_l) continue;
                        if(NegotiationCounter::strip_kind(
                               _l->m_transaction_started_time.load(
                                   std::memory_order_relaxed)) == _mine)
                            ++_held;
                    }
                    auto &_d = detail::neg_diag();
                    _d.tagged_list_at_sleep += snap.m_tagged_linkages.size();
                    if(_held > 0) { ++_d.sleeps_holding;
                                    _d.tags_held_at_sleep += (unsigned)_held; }
                    if(snap.m_registered_privileged) ++_d.sleeps_priv;
                }
#endif
                {
                    unsigned _chunk_us_ov = 0;
                    if(_wb_round) {
                        const int64_t _rem = _wb_round
                            - KAME_NEG_SPIN_TAIL_US
                            - (int64_t)NegotiationCounter::now_us();
                        if(_rem <= 0) goto _exit_cv_sleep;
                        else if(_rem < (int64_t)KAME_NEG_SLEEP_US_PER_MS)
                            _chunk_us_ov = (unsigned)_rem;
                    }
                    NegotiationCounter::negotiate_sleep(1, started_time,
                                                        _chunk_us_ov);
                }
#else
#if KAME_STM_NEG_DIAG
                {   // (diag) do we still own tags while going to sleep?
                    int _held = 0;
                    const auto _mine = NegotiationCounter::strip_kind(
                                            snap.m_started_time);
                    for(auto &&_l : snap.m_tagged_linkages) {
                        if( !_l) continue;
                        if(NegotiationCounter::strip_kind(
                               _l->m_transaction_started_time.load(
                                   std::memory_order_relaxed)) == _mine)
                            ++_held;
                    }
                    auto &_d = detail::neg_diag();
                    _d.tagged_list_at_sleep += snap.m_tagged_linkages.size();
                    if(_held > 0) { ++_d.sleeps_holding;
                                    _d.tags_held_at_sleep += (unsigned)_held; }
                    if(snap.m_registered_privileged) ++_d.sleeps_priv;
                }
#endif
                {
                    int _chunk_ms = 1 + (int)(s_backoff_seed >> 31);
                    unsigned _chunk_us_ov = 0;
                    if(_wb_round) {
                        // …minus the tail reserve, so this wait's wake-up
                        // jitter lands INSIDE the budget rather than past it.
                        const int64_t _rem = _wb_round
                            - KAME_NEG_SPIN_TAIL_US
                            - (int64_t)NegotiationCounter::now_us();
                        if(_rem <= 0)
                            goto _exit_cv_sleep;   // budget spent: never sleep
                        else if(_rem < (int64_t)_chunk_ms
                                       * (int64_t)KAME_NEG_SLEEP_US_PER_MS)
                            _chunk_us_ov = (unsigned)_rem;
                    }
                    NegotiationCounter::negotiate_sleep(
                        _chunk_ms, started_time, _chunk_us_ov);
                }
#endif
            } while(Node<XN>::NegotiationCounter::now_us() < t_end);
#else
#if KAME_STM_NEG_DIAG
                {   // (diag) do we still own tags while going to sleep?
                    int _held = 0;
                    const auto _mine = NegotiationCounter::strip_kind(
                                            snap.m_started_time);
                    for(auto &&_l : snap.m_tagged_linkages) {
                        if( !_l) continue;
                        if(NegotiationCounter::strip_kind(
                               _l->m_transaction_started_time.load(
                                   std::memory_order_relaxed)) == _mine)
                            ++_held;
                    }
                    auto &_d = detail::neg_diag();
                    _d.tagged_list_at_sleep += snap.m_tagged_linkages.size();
                    if(_held > 0) { ++_d.sleeps_holding;
                                    _d.tags_held_at_sleep += (unsigned)_held; }
                    if(snap.m_registered_privileged) ++_d.sleeps_priv;
                }
#endif
            {
                unsigned _us_ov = 0;
                if(_wb_round) {
                    const int64_t _rem = _wb_round
                        - KAME_NEG_SPIN_TAIL_US
                        - (int64_t)NegotiationCounter::now_us();
                    if(_rem <= 0) goto _exit_cv_sleep;
                    else if(_rem < (int64_t)ms_actual
                                   * (int64_t)KAME_NEG_SLEEP_US_PER_MS)
                        _us_ov = (unsigned)_rem;
                }
                NegotiationCounter::negotiate_sleep(ms_actual, started_time,
                                                    _us_ov);
            }
#endif
        }
        // Wait budget, again at the tail: the round may have expired it after
        // the top-of-loop check.  Same rule as the top — the wait behind a
        // live privileged peer is exempt (_wb_round is zeroed while
        // fair-blocked).
        if(_wb_round && NegotiationCounter::now_us() >= _wb_round)
            break;
    }
_exit_cv_sleep:;
  } // end adaptive-path scope
}

} // namespace Transactional

#endif /* TRANSACTION_NEG_IMPL_H */
