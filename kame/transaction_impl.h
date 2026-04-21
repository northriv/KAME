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
#include <cstdint>
#include <cmath>
#include <algorithm>

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
    // Paper ablation: entire backoff layer disabled; retry loops become pure
    // CAS-retry spin with no CPU relaxation. Paired with negotiate() early
    // return in transaction.h.
    (void)retry;
    return;
#endif
    // No cap: spin count grows linearly with retry depth. Livelock-heavy
    // paths need unbounded growth so at some retry depth the pause
    // duration exceeds coherence ping-pong cycle time, opening a winner
    // window. Empirically cap=1024 is insufficient on iMac Pro x86
    // 32-thread 3level stress; uncapped resolves it.
    uint32_t h = (uint32_t)retry * 0x9E3779B1u;
    h ^= (uint32_t)(uintptr_t)&retry;       // per-thread entropy from stack addr
    int spins = 1 + (int)(((uint64_t)(h >> 16) * (uint32_t)retry) >> 16);
    for(int k = 0; k < spins; ++k) pause4spin();
}

// Unified retry-loop backoff: always call retry_pause + negotiate.
// retry=0 → retry_pause issues 1 pause (negligible), negotiate fast-returns
// on the zero m_transaction_started_time marker. No threshold gate.
template <class XN>
void
Node<XN>::Linkage::negotiate_after_retry_pause(
    int retry,
    typename NegotiationCounter::cnt_t &started_time,
    TidBitset &tid_bitset,
    float mult_wait) noexcept {
    retry_pause(retry);
    negotiate(started_time, tid_bitset, mult_wait);
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

XThreadLocal<Priority> stl_currentPriority;

inline Priority getCurrentPriorityMode() {
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


// Adaptive-lease constants. The active lease window (ns) is stored per-Linkage
// as Linkage::m_adapt_lease_ns, so contention sites converge independently
// (hot Linkage → longer lease; cold Linkage → short).
// Clamped to [KAME_LEASE_NS_MIN, KAME_LEASE_NS_MAX] = [1 µs, 500 µs].
// Fixed-threshold drift beat proportional-rate variants (C-1, C-2) in
// sandbox because the workload's C distribution is skewed: many calls with
// C = 0..1 (early tx), fewer with high C. Keeping C = 1 neutral avoids
// dragging the lease down during the early-tx phase that dominates call
// count.
#ifndef KAME_LEASE_NS_MIN
#define KAME_LEASE_NS_MIN  1000     // 1 µs
#endif
#ifndef KAME_LEASE_NS_MAX
#define KAME_LEASE_NS_MAX  2000000   // 2 ms
#endif


// Optional diagnostic counters (opt-in via -DKAME_ADAPT_INSTRUMENT=1).
// Inspect with gdb while a test runs: `thread apply all print <name>`.
// Off by default to keep per-call overhead minimal in production builds.
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
// dt2 of the most recent negotiate() call — used by the adaptive fairness
// gate (always-on, zero cost beyond one thread_local store).
inline thread_local uint64_t s_adapt_dt2_last_ns       = 0;
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
Node<XN>::Linkage::negotiate_internal(typename NegotiationCounter::cnt_t &started_time,
                                      TidBitset &tid_bitset,
                                      float mult_wait) noexcept {
    // (removed: std::this_thread::yield() at entry.
    //  At high thread count (128+), it forms a yield-to-yielding-thread chain
    //  — gdb sampling showed 15% of samples stuck in sched_yield with
    //  negotiate_internal as immediate caller. The chain never terminates
    //  because all runnable threads are also in negotiate_internal's yield.
    //  Removed; the subsequent m_transaction_started_time load + priority/dt
    //  checks serve as the cheap-collision-clear check anyway.)

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
#ifdef KAME_PRIORITY_LEASE
    auto adapt_dt2_last_ns =
        (typename NegotiationCounter::cnt_t)
        (Node<XN>::NegotiationCounter::now() - transaction_started_time);
    // Compute the observed co-committer count C = popcount(tid_bitset)
    // and the competing tx's elapsed time dt2. C drives the adaptive
    // lease length; dt2 drives the fairness gate that suppresses the
    // owner-skip when another tx has been waiting too long.
    // Additional thread_local signals (bounce, call count, skip rate)
    // are exposed under KAME_ADAPT_INSTRUMENT for gdb tuning.
    int sig_C = 0;
    for(int i = 0; i < TID_BITSET_WORDS; ++i)
        sig_C += popcount_u64(tid_bitset[i]);


#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
    s_adapt_dt2_last_ns = adapt_dt2_last_ns;
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
    // if the value actually changes.
    static constexpr uint16_t LEASE_US_MIN =
        (uint16_t)(KAME_LEASE_NS_MIN / 1000 ? KAME_LEASE_NS_MIN / 1000 : 1);
    static constexpr uint16_t LEASE_US_MAX =
        (uint16_t)(KAME_LEASE_NS_MAX / 1000);
    // Asymmetric drift: growth scales with C (capped) so heavy
    // contention climbs the lease quickly; shrink is a smaller fixed
    // step so a single C=0 call doesn't undo many C>=2 adjustments —
    // equilibrium biases toward higher lease where Linkages do
    // see contention at all. Override the schedule via macros.
#ifndef KAME_LEASE_GROW_PER_C_PERCENT
#define KAME_LEASE_GROW_PER_C_PERCENT 30   // additive per unit of (C-1)
#endif
#ifndef KAME_LEASE_GROW_MAX_PERCENT
#define KAME_LEASE_GROW_MAX_PERCENT   80   // cap on per-call growth
#endif
#ifndef KAME_LEASE_SHRINK_PERCENT
#define KAME_LEASE_SHRINK_PERCENT    30   // shrink step when C == 0
#endif
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
    // Quantized write: commit the atomic store only when the lease has
    // actually moved by at least KAME_LEASE_QWRITE_US µs. Once the lease
    // pins at a rail (MIN or MAX), the clamped computation produces
    // new_lease_us == ps.lease_us — we must NOT still write, or every
    // call pointlessly rewrites the rail value and ping-pongs the
    // m_priority_state cache line across cores (observed as a heavy
    // storePriority cost on M3).
#ifndef KAME_LEASE_QWRITE_US
#define KAME_LEASE_QWRITE_US 1
#endif
    int delta = (int)new_lease_us - (int)ps.lease_us;
    if(delta >= (int)KAME_LEASE_QWRITE_US || delta <= -(int)KAME_LEASE_QWRITE_US) {
        PriorityState drifted = ps;
        drifted.lease_us = new_lease_us;
        storePriority(drifted);
        ps.lease_us = new_lease_us;
    }

    // Adaptive gate: suppress owner-skip when dt2 (time the competing
    // tx has been waiting) exceeds KAME_DT2_FAIRNESS_NS. dt2 > fairness
    // threshold indicates long-held conflicting tx (e.g. test_negotiation
    // with msecsleep inside iterate_commit, dt2 ≈ 60-150 µs). Chained
    // owner-skip in that regime starves the waiter. For payload-style
    // hot-loop commits (dt2 ≈ 2-8 µs) the gate stays open and the skip
    // delivers its benefit.
#ifndef KAME_DT2_FAIRNESS_NS
#define KAME_DT2_FAIRNESS_NS 40000  // 40 µs; override via -D
#endif
    unsigned my_tid = ProcessCounter::id() & 0xFFFFu;
    if(my_tid == ps.tid
        && adapt_dt2_last_ns < (uint64_t)KAME_DT2_FAIRNESS_NS) {
        // Age in µs via modular 32-bit subtraction (wrap-safe up to ~35 min).
        uint32_t age_us = nsToUs(Node<XN>::NegotiationCounter::now()) - ps.start_us;
        if(age_us < (uint32_t)ps.lease_us) {
#if defined(KAME_ADAPT_INSTRUMENT) && KAME_ADAPT_INSTRUMENT
            ++s_adapt_skip_hits;
#endif
            Priority pr = getCurrentPriorityMode();
            if(pr == Priority::HIGHEST || pr == Priority::NORMAL)
                return; //skips
        }
    }
#endif

    // Thread-local LCG for sleep-duration jitter randomization.
    static thread_local uint32_t s_backoff_seed = 0x9E3779B9u
        ^ (uint32_t)(uintptr_t)&started_time;

    for(int ms = 0;;) {
        Priority pr = getCurrentPriorityMode();
        if(pr == Priority::HIGHEST)
            break;
        auto dt = started_time - transaction_started_time;
        if(dt <= 0) {
            if((pr == Priority::NORMAL) || (ms > 5 * mult_wait))
                break; //This thread is the oldest.
        }
        auto transaction_started_time =
            m_transaction_started_time.load(std::memory_order_acquire);
        if( !transaction_started_time)
            break; //collision has not been detected.

        auto dt2 = Node<XN>::NegotiationCounter::now() - transaction_started_time;

        if(pr != Priority::LOWEST) {
            if(mult_wait * 2 * dt < dt2)
                break;
        }

        ms = std::max((int)(dt2 * mult_wait / 10000),  ms + 1);
        if(ms > 5000) {
            fprintf(stderr, "Nested transaction?, ");
            fprintf(stderr, "Negotiating, %f sec. requested, limited to 5s.", ms*1e-3);
            fprintf(stderr, "for BP@%p\n", this);
            ms = 5000;
        }

#ifndef KAME_STM_DISABLE_JITTER
#define KAME_STM_DISABLE_JITTER 0  // 1 → sleep = nominal ms (no adaptive jitter), for paper comparison
#endif
#if KAME_STM_DISABLE_JITTER
        // Paper ablation variant: nominal ms, no jitter randomization.
        int ms_actual = (ms < 1) ? 1 : ms;
        (void)tid_bitset;  // unused when jitter disabled
        (void)s_backoff_seed;
#else
        // Live-contention estimate: distinct TIDs observed this transaction.
        int C = 0;
        for(int i = 0; i < TID_BITSET_WORDS; ++i)
            C += popcount_u64(tid_bitset[i]);
        if(C < 1) C = 1;

        // √C-scaled range. C=1 (no observed contention) → sqrtC=1 →
        // no jitter (sleep exactly `ms`); C>1 fans out proportionally.
        int sqrtC = (int)std::sqrt((double)C);
        if(sqrtC < 1) sqrtC = 1;

        // Asymmetric jitter range [ms/√C, ms]. Upper bound = nominal ms;
        // spread only downward to keep mean sleep < ms (CPU stays busy)
        // while still breaking lock-step (threads wake at distinct points).
        // Proportional scaling (ms grows with dt2) already carries the
        // "wait longer when contender is old" behavior, so no need to
        // expand above ms.
        int low = std::max(1, ms / sqrtC);
        int high = ms;
        int ms_actual;
        if(high <= low) {
            ms_actual = low;   // degenerate: no jitter (sqrtC=1)
        } else {
            uint32_t span = (uint32_t)(high - low);
            s_backoff_seed = s_backoff_seed * 1103515245u + 12345u;
            uint32_t rand16 = (s_backoff_seed >> 16) & 0xFFFFu;
            ms_actual = low + (int)(((uint64_t)span * rand16) >> 16);
        }
        if(ms_actual < 1) ms_actual = 1;
#endif
        msecsleep(ms_actual);
    }
}

template <class XN>
Node<XN>::Node() : m_link(std::make_shared<Linkage>()) {
    local_shared_ptr<Packet> packet(new Packet());
    m_link->reset(new PacketWrapper(packet, SerialGenerator::gen()));
    //Use create() for this hack.
    packet->m_payload.reset(( *stl_funcPayloadCreator)(static_cast<XN&>( *this)));
    *stl_funcPayloadCreator = nullptr;
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
        for(;;) {
            local_shared_ptr<Packet> subpacket_new;
            local_shared_ptr<PacketWrapper> subwrapper;
            subwrapper = *var->m_link;
            BundledStatus status = bundle_subpacket(0, var, subwrapper, subpacket_new,
                tr.m_started_time, tr.m_tid_bitset, tr.m_serial);
            if(status != BundledStatus::SUCCESS) {
                continue;
            }
            if( !subpacket_new)
                //Inserted twice inside the package.
                break;

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
            if( !var->m_link->compareAndSet(subwrapper, newwrapper)) {
                tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
                return false;
            }
            var->m_link->tags_successful_cas();
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
Node<XN>::eraseSerials(local_shared_ptr<Packet> &packet, int64_t serial) {
    if(packet->size() && packet->subpackets()->m_serial == serial)
        packet->subpackets()->m_serial = SerialGenerator::SERIAL_NULL;
    if(packet->payload()->m_serial == serial)
        packet->payload()->m_serial = SerialGenerator::SERIAL_NULL;

    for(;;) {
        local_shared_ptr<PacketWrapper> wrapper( *packet->node().m_link);
        if(wrapper->m_bundle_serial != serial)
            break;
        local_shared_ptr<PacketWrapper> newwrapper(new PacketWrapper( *wrapper, SerialGenerator::SERIAL_NULL));
        if(packet->node().m_link->compareAndSet(wrapper, newwrapper))
            break;
    }
    for(int i = 0; i < packet->size(); ++i) {
        local_shared_ptr<Packet> &subpacket(( *packet->subpackets())[i]);
        if(subpacket)
            eraseSerials(subpacket, serial);
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
    local_shared_ptr<PacketWrapper> nullsubwrapper, newsubwrapper;
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

    eraseSerials(packet, tr.m_serial);

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
    if( !var->m_link->compareAndSet(nullsubwrapper, newsubwrapper)) {
        tr.m_oldpacket.reset(new Packet( *tr.m_oldpacket)); //Following commitment should fail.
        return false;
    }
    var->m_link->tags_successful_cas();
//		printf("r");
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
//   - Saves child_wrapper for staleness check.
//   - Updates root_wrapper in-place to the parent's wrapper (= *bundledBy).
//   - Copies root_wrapper to parent_wrapper as a snapshot for this level.
// Returns a WalkUpResult with find_status indicating success/failure.
// Other fields (status, is_root_level, parent_packet) are set later by
// walkUpChainImpl.
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
    r.child_wrapper = root_wrapper;
    r.reverse_index = root_wrapper->reverseIndex();
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

    assert( !r.parent_wrapper->packet() ||
        (r.parent_wrapper->packet()->node().m_link == r.parent_linkage));
    assert(( *r.parent_packet)->node().m_link == r.parent_linkage);

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
Node<XN>::snapshot(Snapshot<XN> &snapshot, bool multi_nodal,
                   typename NegotiationCounter::cnt_t started_time,
                   TidBitset &tid_bitset) const {
    local_shared_ptr<PacketWrapper> target;
    for(int retry = 0;; ++retry) {
        if(retry) {
            m_link->negotiate_after_retry_pause(retry, started_time, tid_bitset,
                                                 2.0f);
            snapshot.tag_as_contender(m_link);
        }
        target = *m_link;
        snapshot.m_serial = SerialGenerator::gen(target->m_bundle_serial);
        if(target->hasPriority()) {
            if( !multi_nodal)
                break;
            if( !target->packet()->missing()) {
                STRICT_assert(target->packet()->checkConsistensy(target->packet()));
                break;
            }
        }
        else {
            // Taking a snapshot inside the super packet.
            shared_ptr<Linkage > linkage(m_link);
            local_shared_ptr<PacketWrapper> superwrapper(target);
            local_shared_ptr<Packet> *foundpacket;
            auto status = walkUpChain(linkage, superwrapper, &foundpacket);
            switch(status) {
            case SnapshotStatus::SUCCESS: {
                    if( !( *foundpacket)->missing() || !multi_nodal) {
                        snapshot.m_packet = *foundpacket;
                        STRICT_assert(snapshot.m_packet->checkConsistensy(snapshot.m_packet));
                        return;
                    }
                    // The packet is imperfect, and then re-bundling the subpackets.
                    UnbundledStatus status = unbundle(nullptr, started_time, tid_bitset, m_link, target,
                        nullptr, nullptr, nullptr, &snapshot);
                    switch(status) {
                    case UnbundledStatus::W_NEW_SUBVALUE:
                    case UnbundledStatus::COLLIDED:
                    case UnbundledStatus::SUBVALUE_HAS_CHANGED:
                    default:
                        break;
                    }
                    continue;
                }
            case SnapshotStatus::DISTURBED:
            default:
                continue;
            case SnapshotStatus::NODE_MISSING:
            case SnapshotStatus::VOID_PACKET:
                //The packet has been released.
                if( !target->packet()->missing() || !multi_nodal) {
                    snapshot.m_packet = target->packet();
                    return;
                }
                break;
            }
        }
        BundledStatus status = const_cast<Node *>(this)->bundle(
            target, started_time, tid_bitset, snapshot.m_serial, true);
        switch (status) {
        case BundledStatus::SUCCESS:
            assert( !target->packet()->missing());
            STRICT_assert(target->packet()->checkConsistensy(target->packet()));
            snapshot.m_serial = SerialGenerator::gen(); //Capture Lamport advances from bundle().
            break;
        default:
            continue;
        }
    }
    snapshot.m_packet = target->packet();
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
    typename NegotiationCounter::cnt_t &started_time,
    TidBitset &tid_bitset,
    int64_t bundle_serial) {

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
            local_shared_ptr<PacketWrapper> subwrapper_new;
            UnbundledStatus status = unbundle(detect_collision ? &bundle_serial : nullptr, started_time, tid_bitset,
                subnode->m_link, subwrapper, nullptr, &subwrapper_new, superwrapper);
            switch(status) {
            case UnbundledStatus::W_NEW_SUBVALUE:
                subwrapper = subwrapper_new;
                break;
            case UnbundledStatus::COLLIDED:
                //The subpacket has already been included in the snapshot.
                subpacket_new.reset();
                return BundledStatus::SUCCESS;
            case UnbundledStatus::SUBVALUE_HAS_CHANGED:
            default:
                return BundledStatus::DISTURBED;
            }
        }
    }
    if(subwrapper->packet()->missing()) {
        assert(subwrapper->packet()->size());
        BundledStatus status = subnode->bundle(subwrapper, started_time, tid_bitset, bundle_serial, false);
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
Node<XN>::bundle(local_shared_ptr<PacketWrapper> &oldsuperwrapper,
    typename NegotiationCounter::cnt_t &started_time,
    TidBitset &tid_bitset,
    int64_t bundle_serial, bool is_bundle_root) {

    assert(oldsuperwrapper->packet());
    assert(oldsuperwrapper->packet()->size());
    assert(oldsuperwrapper->packet()->missing());

    Node &supernode(oldsuperwrapper->packet()->node());

    if( !oldsuperwrapper->hasPriority() ||
        (oldsuperwrapper->m_bundle_serial != bundle_serial)) {
        //Tags serial.
        local_shared_ptr<PacketWrapper> superwrapper(
            new PacketWrapper(oldsuperwrapper->packet(), bundle_serial));
        if( !supernode.m_link->compareAndSet(oldsuperwrapper, superwrapper)) {
            return BundledStatus::DISTURBED;
        }
        oldsuperwrapper = std::move(superwrapper);
    }

    fast_vector<local_shared_ptr<PacketWrapper>, 16> subwrappers_org(oldsuperwrapper->packet()->subpackets()->size());

    for(int retry = 0;; ++retry) {
        if(retry)
            supernode.m_link->negotiate_after_retry_pause(retry, started_time, tid_bitset,
                                                          2.0f);
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
                if(child_retry)
                    child->m_link->negotiate_after_retry_pause(child_retry, started_time, tid_bitset,
                        2.0f / subpackets->size());
                subwrapper = *child->m_link;
                if(subwrapper == subwrappers_org[i])
                    break;
                SerialGenerator::gen(subwrapper->m_bundle_serial); //Lamport: advance past sub-node serial.
                BundledStatus status = bundle_subpacket( &oldsuperwrapper,
                    child, subwrapper, subpacket_new, started_time, tid_bitset, bundle_serial);
                switch(status) {
                case BundledStatus::SUCCESS:
                    break;
                case BundledStatus::DISTURBED:
                default:
                    if(oldsuperwrapper == *supernode.m_link)
                        continue;
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
                break;
            }
        }
        if(is_bundle_root) {
            assert( &supernode == this);
            missing = false;
        }
        newpacket->m_missing = true;

        //--- Phase 2: first checkpoint — CAS the parent PacketWrapper ---
        supernode.m_link->negotiate(started_time, tid_bitset, 2.0f);
        if( !supernode.m_link->compareAndSet(oldsuperwrapper, superwrapper)) {
//			superwrapper = *supernode.m_link;
//			if(superwrapper->m_bundle_serial != bundle_serial)
            return BundledStatus::DISTURBED;
//			oldsuperwrapper = superwrapper;
//			continue;
        }
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

            //this negotiation may decrease a commiting rate.
            child->m_link->negotiate(started_time, tid_bitset, 2.0f / subnodes->size());
            assert( !bundled_ref->hasPriority());
            //Second checkpoint, the written bundle is valid or not.
            if( !child->m_link->compareAndSet(subwrappers_org[i], bundled_ref)) {
                if(local_shared_ptr<PacketWrapper>( *child->m_link)->m_bundle_serial != bundle_serial)
                    return BundledStatus::DISTURBED;
                if(oldsuperwrapper != *supernode.m_link)
                    return BundledStatus::DISTURBED;
                changed_during_bundling = true;
                break;
            }
        }
        if(changed_during_bundling)
            continue;

        //--- Phase 4: finalize — clear missing flag if all sub-packets are present ---
        superwrapper.reset(new PacketWrapper( *superwrapper, bundle_serial));
        if( !missing) {
            local_shared_ptr<Packet> &newpacket(
                reverseLookup(superwrapper->packet(), true, SerialGenerator::gen()));
            newpacket->m_missing = false;
            STRICT_assert(newpacket->checkConsistensy(newpacket));
        }

        if( !supernode.m_link->compareAndSet(oldsuperwrapper, superwrapper))
            return BundledStatus::DISTURBED;
        oldsuperwrapper = std::move(superwrapper);

        for(unsigned int i = 0; i < subnodes->size(); i++) {
            shared_ptr<Node> child(( *subnodes)[i]);
            //this tagging significantly icreased a commiting rate.
            child->m_link->tags_successful_cas(started_time);
        }
        supernode.m_link->tags_successful_cas(started_time);
        break;
    }
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
    local_shared_ptr<PacketWrapper> wrapper;
    for(int retry = 0;; ++retry) {
        if(retry)
            m_link->negotiate_after_retry_pause(retry, tr.m_started_time, tr.m_tid_bitset,
                                                 2.0f);
        wrapper = *m_link;
        if(wrapper->hasPriority()) {
            //Committing directly to the node.
            if(wrapper->packet() != tr.m_oldpacket) {
                if( !tr.isMultiNodal() && (wrapper->packet()->payload() == tr.m_oldpacket->payload())) {
                    //Single-node mode, the payload in the snapshot is unchanged.
                    tr.m_packet->subpackets() = wrapper->packet()->subpackets();
                    tr.m_packet->m_missing = wrapper->packet()->missing();
                }
                else {
                    STRICT_TEST(s_serial_abandoned = tr.m_serial);
//					fprintf(stderr, "F");
                    return false;
                }
            }
//			STRICT_TEST(std::deque<local_shared_ptr<PacketWrapper> > subwrappers);
//			STRICT_TEST(fetchSubpackets(subwrappers, wrapper->packet()));
            STRICT_assert(tr.m_packet->checkConsistensy(tr.m_packet));

            // m_link->negotiate(tr.m_started_time, tr.m_tid_bitset, 4.0f);
            if(m_link->compareAndSet(wrapper, newwrapper)) {
//				STRICT_TEST(if(wrapper->isBundled())
//					for(typename std::deque<local_shared_ptr<PacketWrapper> >::const_iterator
//					it = subwrappers.begin(); it != subwrappers.end(); ++it)
//					assert( !( *it)->hasPriority()));
                //this decreases a commit rate.
                // m_link->tags_successful_cas(tr.m_started_time);
                return true;
            }
            continue;
        }
        //Unbundling this node from the super packet.
        UnbundledStatus status = unbundle(nullptr, tr.m_started_time, tr.m_tid_bitset,
            m_link, wrapper,
            tr.isMultiNodal() ? &tr.m_oldpacket : nullptr, tr.isMultiNodal() ? &newwrapper : nullptr,
            nullptr, &tr);
        switch(status) {
        case UnbundledStatus::W_NEW_SUBVALUE:
            if(tr.isMultiNodal())
                return true;
            continue;
        case UnbundledStatus::SUBVALUE_HAS_CHANGED: {
                STRICT_TEST(s_serial_abandoned = tr.m_serial);
//				fprintf(stderr, "F");
                return false;
            }
        case UnbundledStatus::DISTURBED:
        default:
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
//   - sublinkage: the child's Linkage (currently a back-reference).
//   - bundled_ref: current value of sublinkage (the back-reference wrapper).
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
//      ancestor that needs its PacketWrapper updated.
//   2. If oldsubpacket was given and doesn't match, return
//      SUBVALUE_HAS_CHANGED (the transaction must fail).
//   3. Execute the CAS list from cas_infos bottom-up: each ancestor's
//      Linkage gets a new PacketWrapper with a copied packet that marks
//      the child's slot as missing (the child is leaving the bundle).
//   4. CAS the child's Linkage: replace the back-reference (bundled_ref)
//      with a new PacketWrapper that holds the extracted sub-packet
//      directly (hasPriority = true). The serial is advanced past the
//      super-node's bundle serial via gen().
//=============================================================================
template <class XN>
typename Node<XN>::UnbundledStatus
Node<XN>::unbundle(const int64_t *bundle_serial, typename NegotiationCounter::cnt_t &time_started,
    TidBitset &tid_bitset,
    const shared_ptr<Linkage> &sublinkage, const local_shared_ptr<PacketWrapper> &bundled_ref,
    const local_shared_ptr<Packet> *oldsubpacket, local_shared_ptr<PacketWrapper> *newsubwrapper_returned,
    local_shared_ptr<PacketWrapper> *oldsuperwrapper,
    Snapshot<XN> *snap) {

    assert( !bundled_ref->hasPriority());

// Taking a snapshot inside the super packet.
    local_shared_ptr<PacketWrapper> superwrapper(bundled_ref);
    local_shared_ptr<Packet> *newsubpacket;
    CASInfoList cas_infos;
    SnapshotStatus status = snapshotForUnbundle(sublinkage, superwrapper, &newsubpacket,
        bundle_serial ? *bundle_serial : SerialGenerator::SERIAL_NULL, &cas_infos);
    if(status == SnapshotStatus::DISTURBED)
        return UnbundledStatus::DISTURBED;
    if(status == SnapshotStatus::VOID_PACKET || status == SnapshotStatus::NODE_MISSING) {
        newsubpacket = const_cast<local_shared_ptr<Packet> *>( &bundled_ref->packet());
        assert(newsubpacket);
    }
    if(status == SnapshotStatus::NODE_MISSING_AND_COLLIDED) {
        newsubpacket = const_cast<local_shared_ptr<Packet> *>( &bundled_ref->packet());
        assert(newsubpacket);
        status = SnapshotStatus::COLLIDED;
    }
    // SUCCESS, COLLIDED → fall through

    if(oldsubpacket && ( *newsubpacket != *oldsubpacket))
        return UnbundledStatus::SUBVALUE_HAS_CHANGED;

    for(auto it = cas_infos.begin(); it != cas_infos.end(); ++it) {
        it->linkage->negotiate(time_started, tid_bitset, 2.0 / cas_infos.size());
        if( !it->linkage->compareAndSet(it->old_wrapper, it->new_wrapper))
            return UnbundledStatus::DISTURBED;
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

    if( !sublinkage->compareAndSet(bundled_ref, newsubwrapper))
        return UnbundledStatus::SUBVALUE_HAS_CHANGED;

    sublinkage->tags_successful_cas(time_started);
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

