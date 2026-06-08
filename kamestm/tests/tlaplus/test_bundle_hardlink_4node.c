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
/*
 * C11 test generated mechanically from BundleUnbundle_hardlink_4node.tla.
 *
 * Hard-link 4-node model -- C is a child of BOTH A and B, both of which
 * are children of the bundle root:
 *
 *           Root  (bundle root)
 *          /    \
 *         A      B            <- A and B are both children of Root
 *          \    /
 *           C                  <- C is hard-linked: child of BOTH A and B
 *
 * At any instant C's packet lives in exactly one of A.sub[C] or B.sub[C];
 * the other slot is Null (the hard-link reference).  Structurally C is
 * recorded in BOTH A's and B's subnodes lists throughout -- the booleans
 * cInA / cInB track that, independently of which slot currently carries
 * the packet.
 *
 * Bug surface (TLA+ header): bundle(Root) Phase 1 collects A and B in TWO
 * separate atomic snapshots (Phase1A reads Root+A, Phase1B reads B).  A
 * concurrent release(B,C) can fire between the two snapshots, clearing
 * B.sub[C] before its packet has been migrated to A.sub[C].  The bundler
 * then re-publishes a Root packet with BOTH A.sub[C] and B.sub[C] = Null
 * while C is still structurally in A and B (cInA or cInB) -> C is
 * unreachable from Root -> reverseLookup(C, Root) throws 871.
 *
 * FIX (BundlePhase4 reachability gate): before clearing Root.missing, gate
 * on reachability -- if C must be reachable (cInA \/ cInB) but neither
 * A.sub[C] nor B.sub[C] is present in the collected Root packet, do NOT
 * finalize; restart the whole bundle (TLA+ "DISTURBED"-equivalent retry).
 * Weak fairness on the peer's MigrateCToA guarantees the race eventually
 * resolves (A.sub[C] gets the packet) and the next bundle attempt succeeds.
 *
 * ENCODING: reuses the test_bundle_2level_LLfree.c / hardlink_self_collision
 * idiom VERBATIM:
 *   - slot-pool packet ring buffer (immutable once written)
 *   - TID-encoded base-B Lamport serial (gen_serial / encode_serial)
 *   - wrapper pack/unpack into a single atomic uint64_t per node
 *   - wrapper CAS (cas_w) over linkage[]
 *   - LL-free negotiate priority-tag machinery (priority_tag[], gate(),
 *     can_proceed_with_preempt, tag_after_fail/success, clear_my_tags)
 *
 * DIFFERENCES from hardlink_self_collision (per this TLA+ model):
 *   - 4 FIXED nodes Root, A, B, C.
 *   - A and B BOTH start with OWN priority (not bundled); Root starts
 *     missing=TRUE (pre-bundle).  C starts bundledRef -> B; B.sub[C] holds
 *     C's packet, A.sub[C] = Null.
 *   - Bundle Phase1 is split into Phase1A (snapshot Root+A) and Phase1B
 *     (snapshot B) so a peer release can interleave between them.
 *   - The peer operation is a 2-step release(B,C): ReleaseBCNoMigrate (CAS
 *     B to clear sub[C], cInB:=FALSE) then MigrateCToA (CAS A to fill
 *     sub[C] with C's packet + CAS C -> bundledRef A).  The two steps are
 *     one logical release; the thread cannot start anything else between.
 *   - cInA / cInB are separate structural booleans (g_cInA / g_cInB),
 *     NOT derived from the packet slots.
 *   - Post-join terminal invariant = SnapshotConsistency + HardlinkExclusive
 *     (the two INVARIANTs in BundleUnbundle_hardlink_4node_2thr_mc.cfg).
 *
 * Atomicity modes (MODE_COARSE / FINE / SUPERFINE) carry over; the bundle
 * pipeline phases are already CAS-granular so FINE is the natural default,
 * COARSE wraps the whole pipeline (and the release op) in one mutex.
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- Compile-time mode selection --- */
#define MODE_COARSE    1
#define MODE_FINE      2
#define MODE_SUPERFINE 3

#ifndef MODE
#define MODE MODE_FINE
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 2
#endif

#ifndef STRESS_SECONDS
#define STRESS_SECONDS 0
#endif

#ifndef MAX_COMMITS
#  if STRESS_SECONDS > 0
#    define MAX_COMMITS 0x7fffffff
#  else
#    define MAX_COMMITS 1
#  endif
#endif

#ifndef MAX_PAYLOAD
#define MAX_PAYLOAD 3
#endif

/* Packet pool: ring-buffer of packets (immutable once written).  Larger than
 * max in-flight references prevents pool-wrap silent corruption. */
#ifndef PACKET_POOL_ENTRIES
#define PACKET_POOL_ENTRIES 134217727u
#endif
#if (PACKET_POOL_ENTRIES) > 134217727u
#  error "PACKET_POOL_ENTRIES must fit in 27 bits (<=134217727)"
#endif

/* --- Node IDs (4 fixed nodes: Root, A, B, C) --- */
#define NODE_ROOT   0
#define NODE_A      1
#define NODE_B      2
#define NODE_C      3
#define NUM_NODES   4

/* bundled_by is a 2-bit field.  Reachable bundlers for C are A (1) and
 * B (2).  3 == NULL_NODE means "priority / no bundler".  TLA+
 * BundledRefWrapper stores parent in {A, B}; priority wrappers carry
 * bundledBy = Null.  (Root is never bundled.) */
#define NULL_NODE   0x3u

/* --- Serial + slot widths (symmetric, matches 2/3-level) --- */
#define SER_BITS   27u
#define SER_MOD    (1u << SER_BITS)
#define SER_MASK   (SER_MOD - 1u)
#define SLOT_BITS  27u
#define SLOT_MASK  ((1u << SLOT_BITS) - 1u)
#define SLOT_NULL  0u   /* reserved == TLA+ Null sub-slot */

/* --- Modular serial comparison (TLA+ ModGT) --- */
static inline bool ser_gt(uint32_t a, uint32_t b) {
    uint32_t diff = (a - b) & SER_MASK;
    return diff > 0 && diff < (SER_MOD >> 1);
}

/* TID-encoded base-B Lamport serial (verbatim from 2level_LLfree). */
#define SERIAL_BASE  ((uint32_t)(NUM_THREADS + 1))   /* > max tid */

static inline uint32_t serial_counter(uint32_t s) { return s / SERIAL_BASE; }
__attribute__((unused))
static inline uint32_t serial_tid(uint32_t s)     { return s % SERIAL_BASE; }
static inline uint32_t encode_serial(uint32_t cnt, uint32_t tid) {
    return ((cnt * SERIAL_BASE) + tid) & SER_MASK;
}

static inline uint32_t gen_serial(uint32_t thread_ser, uint32_t last_ser, uint32_t my_tid) {
    uint32_t last_cnt = serial_counter(last_ser);
    uint32_t my_cnt   = serial_counter(thread_ser);
    uint32_t base_cnt = ser_gt(last_cnt, my_cnt) ? last_cnt : my_cnt;
    uint32_t new_cnt  = base_cnt + 1u;
    return encode_serial(new_cnt, my_tid);
}

/* ============================================================================
 * Packet pool.  Packet bits:
 *   payload(8) | sub_slot[0](27) | sub_slot[1](27) | missing(1)
 *
 * sub_slot[0] / sub_slot[1] meaning is node-specific:
 *   Root packet : slot0 = Root.sub[A] (A's sub-packet),
 *                 slot1 = Root.sub[B] (B's sub-packet)
 *   A packet    : slot0 = A.sub[C] (C's sub-packet),  slot1 = SLOT_NULL
 *   B packet    : slot0 = B.sub[C] (C's sub-packet),  slot1 = SLOT_NULL
 *   C packet    : slot0 = slot1 = SLOT_NULL (leaf)
 * ============================================================================ */
static _Atomic(uint64_t) packet_pool[PACKET_POOL_ENTRIES + 1];
static _Atomic(uint32_t) global_slot_counter;

static inline uint64_t pkt_pack(uint8_t payload, uint32_t s0, uint32_t s1, bool missing) {
    uint64_t v = 0;
    v |= (uint64_t)payload;
    v |= ((uint64_t)(s0 & SLOT_MASK)) << 8;
    v |= ((uint64_t)(s1 & SLOT_MASK)) << 35;
    v |= ((uint64_t)(missing ? 1u : 0u)) << 62;
    return v;
}
static inline void pkt_unpack(uint64_t v, uint8_t *payload,
                              uint32_t *s0, uint32_t *s1, bool *missing) {
    if (payload) *payload = (uint8_t)(v & 0xFFu);
    if (s0)      *s0      = (uint32_t)((v >> 8)  & (uint64_t)SLOT_MASK);
    if (s1)      *s1      = (uint32_t)((v >> 35) & (uint64_t)SLOT_MASK);
    if (missing) *missing = (v >> 62) & 1u;
}

static inline uint32_t alloc_slot(uint8_t payload, uint32_t s0, uint32_t s1, bool missing) {
    uint32_t c = atomic_fetch_add_explicit(&global_slot_counter, 1u, memory_order_relaxed);
    uint32_t s = (c % PACKET_POOL_ENTRIES) + 1u;
    atomic_store_explicit(&packet_pool[s],
        pkt_pack(payload, s0, s1, missing), memory_order_release);
    return s;
}

static inline uint64_t load_pkt_raw(uint32_t slot) {
    return atomic_load_explicit(&packet_pool[slot], memory_order_acquire);
}
__attribute__((unused))
static inline uint8_t load_pkt_payload(uint32_t slot) {
    uint8_t p; pkt_unpack(load_pkt_raw(slot), &p, NULL, NULL, NULL);
    return p;
}
static inline uint32_t load_pkt_slot0(uint32_t slot) {
    uint32_t s0; pkt_unpack(load_pkt_raw(slot), NULL, &s0, NULL, NULL);
    return s0;
}

/* ============================================================================
 * Wrapper: serial(27) | has_priority(1) | bundled_by(2) | packet_slot(27).
 * ============================================================================ */
typedef struct {
    bool     has_priority;
    uint32_t serial;
    uint8_t  bundled_by;
    uint32_t packet_slot;
} Wrapper;

static inline uint64_t wrapper_pack(Wrapper w) {
    uint64_t v = 0;
    v |= (uint64_t)(w.serial & SER_MASK);
    v |= ((uint64_t)(w.has_priority ? 1u : 0u)) << 27;
    v |= ((uint64_t)(w.bundled_by & 0x3u)) << 28;
    v |= ((uint64_t)(w.packet_slot & SLOT_MASK)) << 30;
    return v;
}
static inline Wrapper wrapper_unpack(uint64_t v) {
    Wrapper w;
    w.serial       = (uint32_t)(v & SER_MASK);
    w.has_priority = (v >> 27) & 1u;
    w.bundled_by   = (uint8_t)((v >> 28) & 0x3u);
    w.packet_slot  = (uint32_t)((v >> 30) & (uint64_t)SLOT_MASK);
    return w;
}

/* --- Shared state --- */
static _Atomic(uint64_t) linkage[NUM_NODES];   /* TLA+ linkage[Nodes] */
static _Atomic(bool)     g_cInA;               /* TLA+ cInA           */
static _Atomic(bool)     g_cInB;               /* TLA+ cInB           */
static _Atomic(bool)     g_stop;

/* --- Spin / race-detection counters (stress diagnostics) --- */
static _Atomic(uint64_t) spin_bundle;       /* bundle pipeline restarts   */
static _Atomic(uint64_t) spin_phase1;       /* Phase1A/B snapshot retries */
static _Atomic(uint64_t) spin_phase2;       /* Phase2 CAS Root failures   */
static _Atomic(uint64_t) spin_phase4;       /* Phase4 gate/CAS retries    */
static _Atomic(uint64_t) spin_release;      /* release/migrate CAS retries*/
static _Atomic(uint64_t) spin_stale_read;   /* pool-wrap detector fires   */
static _Atomic(uint64_t) spin_negotiate;    /* can_proceed gated waits    */
static _Atomic(uint64_t) spin_preempt;      /* tag preemptions issued     */
static _Atomic(uint64_t) bundles_completed; /* successful BundlePhase5    */
static _Atomic(uint64_t) phase4_disturbed;  /* Phase4 reachability gate fires */
#define SPIN_INC(name) atomic_fetch_add_explicit(&(name), 1u, memory_order_relaxed)

/* ============================================================================
 * LL-free negotiate machinery (verbatim from 2level_LLfree / self_collision).
 * ============================================================================ */
typedef uint64_t Tag;
#define TAG_NULL   ((Tag)0)
#define TAG_VALID  (((Tag)1) << 63)
#define TAG_ITER_MASK 0x7FFFFFFFu

static inline Tag make_tag(uint32_t iter, uint32_t tid) {
    return TAG_VALID | ((Tag)(iter & TAG_ITER_MASK) << 32) | (Tag)tid;
}
static inline bool     tag_is_null(Tag t) { return t == TAG_NULL; }
static inline uint32_t tag_iter(Tag t)    { return (uint32_t)((t >> 32) & TAG_ITER_MASK); }
static inline uint32_t tag_tid(Tag t)     { return (uint32_t)(t & 0xFFFFFFFFu); }
static inline bool     tag_older(Tag a, Tag b) {
    if (tag_iter(a) != tag_iter(b)) return tag_iter(a) < tag_iter(b);
    return tag_tid(a) < tag_tid(b);
}

static _Atomic(Tag)  priority_tag[NUM_NODES];

static bool can_proceed_with_preempt(int n, uint32_t my_iter, uint32_t my_tid) {
    Tag mine = make_tag(my_iter, my_tid);
    for (;;) {
        Tag cur = atomic_load_explicit(&priority_tag[n], memory_order_acquire);
        if (tag_is_null(cur))             return true;
        if (tag_tid(cur) == my_tid)       return true;
        if (tag_older(mine, cur)) {
            if (atomic_compare_exchange_weak_explicit(
                    &priority_tag[n], &cur, mine,
                    memory_order_acq_rel, memory_order_relaxed)) {
                SPIN_INC(spin_preempt);
                return true;
            }
            continue;
        }
        return false;
    }
}

static void tag_after_fail(int n, uint32_t my_iter, uint32_t my_tid) {
    Tag mine = make_tag(my_iter, my_tid);
    for (;;) {
        Tag cur = atomic_load_explicit(&priority_tag[n], memory_order_acquire);
        Tag desired;
        if (tag_is_null(cur))                 desired = mine;
        else if (tag_tid(cur) == my_tid)      desired = mine;
        else if (tag_older(mine, cur))        desired = mine;
        else                                  return;
        if (atomic_compare_exchange_weak_explicit(
                &priority_tag[n], &cur, desired,
                memory_order_acq_rel, memory_order_relaxed)) return;
    }
}

static inline void tag_after_success(int n, uint32_t my_tid) {
    (void)n; (void)my_tid;
}

static void clear_my_tags(uint32_t my_tid) {
    for (int n = 0; n < NUM_NODES; n++) {
        Tag cur = atomic_load_explicit(&priority_tag[n], memory_order_acquire);
        while (!tag_is_null(cur) && tag_tid(cur) == my_tid) {
            if (atomic_compare_exchange_weak_explicit(
                    &priority_tag[n], &cur, TAG_NULL,
                    memory_order_acq_rel, memory_order_relaxed)) break;
        }
    }
}

static inline void gate(int n, uint32_t my_iter, uint32_t my_tid) {
    while (!can_proceed_with_preempt(n, my_iter, my_tid)) {
        SPIN_INC(spin_negotiate);
#if defined(__x86_64__) || defined(__i386__)
        __asm__ __volatile__("pause");
#endif
    }
}

typedef struct {
    uint32_t tid;          /* 1-indexed */
    uint32_t iter;         /* completed iterations */
    uint32_t serial;       /* Lamport thread-local clock */
    bool     is_bundler;   /* TLA+ T1: bundle(Root); else T2: release(B,C) */
} ThreadCtx;

/* Pool-wrap / stale-snapshot detector (Option Z). */
#define POOL_SAFETY_MARGIN 64u
#define TRANSACTION_WINDOW 4096u
static inline uint32_t cur_slot_counter(void) {
    return atomic_load_explicit(&global_slot_counter, memory_order_relaxed);
}
static inline bool pool_stale(uint32_t start_counter) {
    uint32_t advanced = (uint32_t)(cur_slot_counter() - start_counter);
    uint32_t pool_bound = PACKET_POOL_ENTRIES - POOL_SAFETY_MARGIN;
    uint32_t threshold = pool_bound < TRANSACTION_WINDOW ? pool_bound : TRANSACTION_WINDOW;
    if (advanced >= threshold) {
        SPIN_INC(spin_stale_read);
        return true;
    }
    return false;
}

#if MODE == MODE_COARSE
static pthread_mutex_t coarse_mtx = PTHREAD_MUTEX_INITIALIZER;
#  define OP_LOCK()   pthread_mutex_lock(&coarse_mtx)
#  define OP_UNLOCK() pthread_mutex_unlock(&coarse_mtx)
#else
#  define OP_LOCK()   ((void)0)
#  define OP_UNLOCK() ((void)0)
#endif

static inline Wrapper load_w(int n) {
    return wrapper_unpack(atomic_load_explicit(&linkage[n], memory_order_acquire));
}

static inline bool cas_w(int n, Wrapper *expected, Wrapper desired) {
    uint64_t e = wrapper_pack(*expected);
    uint64_t d = wrapper_pack(desired);
    bool ok = atomic_compare_exchange_strong_explicit(
        &linkage[n], &e, d, memory_order_acq_rel, memory_order_relaxed);
    if (!ok) *expected = wrapper_unpack(e);
    return ok;
}

/* --- Wrapper builders --- */
static inline Wrapper make_priority_pkt(uint32_t serial, uint32_t slot) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .packet_slot  = slot,
    };
}
/* TLA+ BundledRefWrapper(parent): packet=Null, hasPriority=FALSE, bundledBy=parent */
static inline Wrapper make_bundled(uint8_t parent_node, uint32_t serial) {
    return (Wrapper){
        .has_priority = false, .serial = serial, .bundled_by = parent_node,
        .packet_slot  = SLOT_NULL,
    };
}

/* sched_yield without pulling in <sched.h> portability headaches: a cheap
 * hint that lets the peer thread run between the two bundle snapshots. */
static inline void sched_yield_compat(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause");
#endif
}

/* ============================================================================
 * ReachableFromRoot(rootPkt) -- TLA+ helper.  C is reachable from Root's
 * published packet iff either Root.sub[A].sub[C] != Null OR
 * Root.sub[B].sub[C] != Null.
 * ============================================================================ */
static bool reachable_from_root(uint32_t r_sA, uint32_t r_sB) {
    if (r_sA != SLOT_NULL && load_pkt_slot0(r_sA) != SLOT_NULL) return true; /* A.sub[C] */
    if (r_sB != SLOT_NULL && load_pkt_slot0(r_sB) != SLOT_NULL) return true; /* B.sub[C] */
    return false;
}

/* ============================================================================
 * bundle_Root(t) -- the 5-phase bundle pipeline on Root (is_bundle_root=TRUE).
 * Mirrors TLA+ BundleStart -> BundlePhase1A -> BundlePhase1B -> BundlePhase2
 * -> BundlePhase4 -> BundlePhase5.  On any CAS failure / DISTURBED gate the
 * pipeline restarts from Phase1 (TLA+ retry edge to "bundle_phase1").
 * Returns true once Phase5 completes (bundleDone[t] := TRUE).
 *
 * Phase1 is split into 1A (read Root+A) and 1B (read B) with a deliberate
 * yield between them so a peer release can interleave (the multi-snapshot
 * pattern that exposes the race).
 * ============================================================================ */
static bool bundle_Root(ThreadCtx *ctx) {
    OP_LOCK();
    for (;;) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) { OP_UNLOCK(); return false; }
        uint32_t start_counter = cur_slot_counter();

        /* ---- Phase 1A: snapshot Root and A.  Both must be priority
         * (TLA+ rw.hasPriority /\ aw.hasPriority).  Collect A.sub[C]. ---- */
        Wrapper rw = load_w(NODE_ROOT);
        if (!rw.has_priority) { SPIN_INC(spin_phase1); SPIN_INC(spin_bundle); continue; }
        Wrapper aw = load_w(NODE_A);
        if (!aw.has_priority) { SPIN_INC(spin_phase1); SPIN_INC(spin_bundle); continue; }

        uint8_t r_pp; bool r_miss;
        pkt_unpack(load_pkt_raw(rw.packet_slot), &r_pp, NULL, NULL, &r_miss);

        uint8_t a_pp; uint32_t a_cSub;       /* A.sub[C] (=A packet slot0) */
        pkt_unpack(load_pkt_raw(aw.packet_slot), &a_pp, &a_cSub, NULL, NULL);

        /* ---- yield: peer release(B,C) may interleave between 1A and 1B
         * (DIFFERENT atomic snapshot in the TLA+ model). ---- */
        sched_yield_compat();

        /* ---- Phase 1B: snapshot B (separate atomic).  B must be priority
         * (TLA+ bw.hasPriority).  Collect B.sub[C]. ---- */
        Wrapper bw = load_w(NODE_B);
        if (!bw.has_priority) { SPIN_INC(spin_phase1); SPIN_INC(spin_bundle); continue; }
        uint8_t b_pp; uint32_t b_cSub;       /* B.sub[C] (=B packet slot0) */
        pkt_unpack(load_pkt_raw(bw.packet_slot), &b_pp, &b_cSub, NULL, NULL);

        if (pool_stale(start_counter)) continue;

        /* ---- Phase 2 (CAS Root): rebuild Root.sub from the collected A and
         * B sub[C] slots and publish missing=TRUE.  This is the "buggy" path:
         * one or both collected C-slots can be Null (the race), and Phase 2
         * republishes them verbatim without a reachability check.  CAS expects
         * the Phase1A Root wrapper (rw); on mismatch restart from Phase 1. ---- */
        uint32_t bundle_ser = gen_serial(ctx->serial, rw.serial, ctx->tid);

        /* newAPkt = MakePacket(A, [C |-> aCSubPkt]); slot0 = collected A.sub[C]. */
        uint32_t new_a_slot = alloc_slot(a_pp, a_cSub, SLOT_NULL, false);
        /* newBPkt = MakePacket(B, [C |-> bCSubPkt]); slot0 = collected B.sub[C]. */
        uint32_t new_b_slot = alloc_slot(b_pp, b_cSub, SLOT_NULL, false);
        /* newPkt = MakePacket(Root, {A:newAPkt, B:newBPkt}, missing=TRUE). */
        uint32_t r2_slot = alloc_slot(r_pp, new_a_slot, new_b_slot, true /*missing*/);
        Wrapper p2 = make_priority_pkt(bundle_ser, r2_slot);

        gate(NODE_ROOT, ctx->iter, ctx->tid);
        Wrapper exp_p2 = rw;
        if (!cas_w(NODE_ROOT, &exp_p2, p2)) {
            tag_after_fail(NODE_ROOT, ctx->iter, ctx->tid);
            SPIN_INC(spin_phase2); SPIN_INC(spin_bundle);
            continue;                              /* TLA+ -> bundle_phase1 */
        }
        tag_after_success(NODE_ROOT, ctx->tid);
        ctx->serial = bundle_ser;

        /* ---- Phase 4 (Finalize, is_bundle_root override): clear Root.missing.
         *   reachable        = ReachableFromRoot(p2.packet)
         *   cMustBeReachable = cInA \/ cInB
         *   canFinalize      = reachable \/ ~cMustBeReachable
         * If !canFinalize -> DISTURBED: do NOT publish; restart whole bundle
         * (leaving Root in the Phase2 missing=TRUE intermediate state, so no
         * inconsistent ~missing publication ever occurs).  CAS expects p2. ---- */
        bool reachable = reachable_from_root(new_a_slot, new_b_slot);
        bool c_in_a = atomic_load_explicit(&g_cInA, memory_order_acquire);
        bool c_in_b = atomic_load_explicit(&g_cInB, memory_order_acquire);
        bool c_must_be_reachable = c_in_a || c_in_b;
        bool can_finalize = reachable || !c_must_be_reachable;

        if (!can_finalize) {
            /* DISTURBED-equivalent: tag Root and restart.  Weak fairness on
             * the peer MigrateCToA guarantees A.sub[C] eventually becomes
             * non-Null, after which the next attempt finalizes. */
            tag_after_fail(NODE_ROOT, ctx->iter, ctx->tid);
            SPIN_INC(phase4_disturbed); SPIN_INC(spin_phase4); SPIN_INC(spin_bundle);
            continue;
        }

        /* finalPkt = MakePacket(Root, p2.sub, missing=FALSE). */
        uint32_t fin_slot = alloc_slot(r_pp, new_a_slot, new_b_slot, false /*missing*/);
        uint32_t f_ser = gen_serial(ctx->serial, p2.serial, ctx->tid);
        Wrapper p4 = make_priority_pkt(f_ser, fin_slot);
        gate(NODE_ROOT, ctx->iter, ctx->tid);
        Wrapper exp_p4 = p2;
        if (!cas_w(NODE_ROOT, &exp_p4, p4)) {
            tag_after_fail(NODE_ROOT, ctx->iter, ctx->tid);
            SPIN_INC(spin_phase4); SPIN_INC(spin_bundle);
            continue;                              /* TLA+ -> bundle_phase1 */
        }
        ctx->serial = f_ser;
        tag_after_success(NODE_ROOT, ctx->tid);

        /* ---- Phase 5 (Publish): completion.  Wrapper already published in
         * Phase4; mark bundleDone and release tags (Tx-end). ---- */
        clear_my_tags(ctx->tid);
        SPIN_INC(bundles_completed);
        OP_UNLOCK();
        return true;
    }
}

/* ============================================================================
 * release_B_C(t) -- the 2-step release(B,C) (TLA+ ReleaseBCNoMigrate then
 * MigrateCToA).  Single logical operation: the thread does both steps before
 * returning (matches the C++ release() API).  Runs at most once
 * (~releaseDone guard via the worker's one-shot call).
 *
 *   Step 1 (ReleaseBCNoMigrate): requires cInB, B priority, B.sub[C] != Null.
 *     CAS B -> packet with sub[C]=Null; cInB := FALSE.
 *   Step 2 (MigrateCToA): requires ~cInB, cInA, A priority, A.sub[C]=Null.
 *     CAS A -> packet with sub[C]=CPacket; CAS C -> BundledRefWrapper(A).
 *
 * Returns when both steps committed (releaseDone := TRUE).
 * ============================================================================ */
static void release_B_C(ThreadCtx *ctx) {
    OP_LOCK();

    /* ---- Step 1: ReleaseBCNoMigrate ---- */
    for (;;) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) { OP_UNLOCK(); return; }
        if (!atomic_load_explicit(&g_cInB, memory_order_acquire)) break; /* already cleared */
        Wrapper bw = load_w(NODE_B);
        if (!bw.has_priority) { SPIN_INC(spin_release); continue; }  /* TLA+ bw.hasPriority */
        uint8_t b_pp; uint32_t b_cSub;
        pkt_unpack(load_pkt_raw(bw.packet_slot), &b_pp, &b_cSub, NULL, NULL);
        if (b_cSub == SLOT_NULL) break;     /* TLA+ bw.packet.sub[C] /= Null guard */

        /* newBPkt = MakePacket(B, [C |-> Null]). */
        uint32_t ser = gen_serial(ctx->serial, bw.serial, ctx->tid);
        uint32_t new_b_slot = alloc_slot(b_pp, SLOT_NULL, SLOT_NULL, false);
        Wrapper new_bw = make_priority_pkt(ser, new_b_slot);
        gate(NODE_B, ctx->iter, ctx->tid);
        Wrapper exp = bw;
        if (cas_w(NODE_B, &exp, new_bw)) {
            ctx->serial = ser;
            /* cInB' = FALSE -- structural: C no longer in B's subnodes. */
            atomic_store_explicit(&g_cInB, false, memory_order_release);
            tag_after_success(NODE_B, ctx->tid);
            break;
        }
        tag_after_fail(NODE_B, ctx->iter, ctx->tid);
        SPIN_INC(spin_release);
    }

    /* ---- Step 2: MigrateCToA ---- */
    for (;;) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) { OP_UNLOCK(); return; }
        /* TLA+ guards: ~cInB /\ cInA. */
        if (atomic_load_explicit(&g_cInB, memory_order_acquire)) { SPIN_INC(spin_release); continue; }
        if (!atomic_load_explicit(&g_cInA, memory_order_acquire)) { OP_UNLOCK(); return; }
        Wrapper aw = load_w(NODE_A);
        if (!aw.has_priority) { SPIN_INC(spin_release); continue; }  /* TLA+ aw.hasPriority */
        uint8_t a_pp; uint32_t a_cSub;
        pkt_unpack(load_pkt_raw(aw.packet_slot), &a_pp, &a_cSub, NULL, NULL);
        if (a_cSub != SLOT_NULL) break;     /* TLA+ aw.packet.sub[C] = Null guard already filled */

        /* newAPkt = MakePacket(A, [C |-> CPacket]); CPacket leaf payload 0. */
        uint32_t c_leaf = alloc_slot(0, SLOT_NULL, SLOT_NULL, false);
        uint32_t a_ser  = gen_serial(ctx->serial, aw.serial, ctx->tid);
        uint32_t new_a_slot = alloc_slot(a_pp, c_leaf, SLOT_NULL, false);
        Wrapper new_aw = make_priority_pkt(a_ser, new_a_slot);
        gate(NODE_A, ctx->iter, ctx->tid);
        Wrapper exp = aw;
        if (cas_w(NODE_A, &exp, new_aw)) {
            ctx->serial = a_ser;
            /* linkage[C] = BundledRefWrapper(A) -- C now homed at A. */
            uint32_t c_ser = gen_serial(ctx->serial, load_w(NODE_C).serial, ctx->tid);
            Wrapper new_cw = make_bundled(NODE_A, c_ser);
            atomic_store_explicit(&linkage[NODE_C], wrapper_pack(new_cw), memory_order_release);
            ctx->serial = c_ser;
            tag_after_success(NODE_A, ctx->tid);
            /* releaseDone' = TRUE -- both steps committed. */
            break;
        }
        tag_after_fail(NODE_A, ctx->iter, ctx->tid);
        SPIN_INC(spin_release);
    }

    OP_UNLOCK();
}

/* ============================================================================
 * Thread worker.
 *   Bundler thread(s): run the one-shot bundle(Root) pipeline MAX_COMMITS
 *     times -- mirrors TLA+ T1.
 *   Releaser thread(s): perform release(B,C) once (its single ~releaseDone
 *     guard), then idle (continue spinning the bundle is NOT its role; the
 *     TLA+ T2 only releases).  We make the releaser also help-bundle after
 *     its release so AllDone (= \A t: bundleDone[t]) can be observed for the
 *     terminal check, matching the model's AllDone over ALL threads.
 *
 * TLA+ terminal condition AllDone == \A t : bundleDone[t]; every thread must
 * complete a bundle.  The releaser does its release first (Phase race), then
 * bundles to reach bundleDone.
 * ============================================================================ */
static void *worker(void *arg) {
    ThreadCtx ctx = *(ThreadCtx*)arg;

    for (uint32_t i = 0; i < (uint32_t)MAX_COMMITS; i++) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;
        if (!ctx.is_bundler && i == 0) {
            /* T2: release(B,C) once before any bundle -- this is the peer
             * action that races the bundler's Phase1A/1B window. */
            release_B_C(&ctx);
        }
        /* Every thread must eventually reach bundleDone (TLA+ AllDone). */
        bundle_Root(&ctx);
        ctx.iter++;
    }
    return NULL;
}

/* ============================================================================
 * Post-join invariant checks (mirror the two INVARIANTs in
 * BundleUnbundle_hardlink_4node_2thr_mc.cfg):
 *   SnapshotConsistency, HardlinkExclusive.
 * ============================================================================ */
static void check_invariants(void) {
    Wrapper rw = load_w(NODE_ROOT);
    /* Root is the bundle root -> always priority (never bundled). */
    assert(rw.has_priority);

    uint8_t r_pp; uint32_t r_sA, r_sB; bool r_miss;
    pkt_unpack(load_pkt_raw(rw.packet_slot), &r_pp, &r_sA, &r_sB, &r_miss);

    bool c_in_a = atomic_load_explicit(&g_cInA, memory_order_acquire);
    bool c_in_b = atomic_load_explicit(&g_cInB, memory_order_acquire);

    /* SnapshotConsistency:
     *   (rw.hasPriority /\ ~rw.packet.missing /\ (cInA \/ cInB))
     *     => ReachableFromRoot(rw.packet) */
    if (rw.has_priority && !r_miss && (c_in_a || c_in_b)) {
        assert(reachable_from_root(r_sA, r_sB));
    }

    /* HardlinkExclusive:
     *   rw.hasPriority =>
     *     ~( Root.sub[A] /= Null /\ Root.sub[B] /= Null
     *        /\ Root.sub[A].sub[C] /= Null /\ Root.sub[B].sub[C] /= Null ) */
    if (rw.has_priority) {
        bool both = (r_sA != SLOT_NULL && r_sB != SLOT_NULL &&
                     load_pkt_slot0(r_sA) != SLOT_NULL &&
                     load_pkt_slot0(r_sB) != SLOT_NULL);
        assert(!both);
    }

    /* BundleRefConsistency (structural sanity):
     *   ~cw.hasPriority => cw.bundledBy in {A, B} */
    Wrapper cw = load_w(NODE_C);
    if (!cw.has_priority) assert(cw.bundled_by == NODE_A || cw.bundled_by == NODE_B);
}

int main(void) {
    atomic_store(&global_slot_counter, 0u);

    /* Init (TLA+ Init):
     *   linkage[Root] = PriorityWrapper(MakePacket(Root, {A:Null,B:Null}, missing=TRUE))
     *   linkage[A]    = PriorityWrapper(MakePacket(A, [C|->Null],   FALSE))  (A.sub[C]=Null)
     *   linkage[B]    = PriorityWrapper(MakePacket(B, [C|->CPacket],FALSE))  (B.sub[C]=cpkt)
     *   linkage[C]    = BundledRefWrapper(B)
     *   cInA = TRUE, cInB = TRUE
     *
     * Build bottom-up:
     *   c_leaf  : C's leaf packet (payload 0)
     *   a_pkt   : A's packet with sub[C] = Null
     *   b_pkt   : B's packet with sub[C] = c_leaf
     *   root_pkt: Root's packet with sub[A]=Null, sub[B]=Null, missing=TRUE
     */
    uint32_t s_C_leaf = alloc_slot(0, SLOT_NULL, SLOT_NULL, false);
    uint32_t s_A_pkt  = alloc_slot(0, SLOT_NULL, SLOT_NULL, false);   /* A.sub[C]=Null */
    uint32_t s_B_pkt  = alloc_slot(0, s_C_leaf,  SLOT_NULL, false);   /* B.sub[C]=c_leaf */
    uint32_t s_R_pkt  = alloc_slot(0, SLOT_NULL, SLOT_NULL, true);    /* Root missing, no subs */

    Wrapper init_R = make_priority_pkt(0, s_R_pkt);
    Wrapper init_A = make_priority_pkt(0, s_A_pkt);
    Wrapper init_B = make_priority_pkt(0, s_B_pkt);
    Wrapper init_C = make_bundled(NODE_B, 0);   /* C bundledBy B */

    atomic_store(&linkage[NODE_ROOT], wrapper_pack(init_R));
    atomic_store(&linkage[NODE_A],    wrapper_pack(init_A));
    atomic_store(&linkage[NODE_B],    wrapper_pack(init_B));
    atomic_store(&linkage[NODE_C],    wrapper_pack(init_C));
    atomic_store(&g_cInA, true);
    atomic_store(&g_cInB, true);
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&priority_tag[i], TAG_NULL);
    atomic_store(&g_stop, false);

    pthread_t threads[NUM_THREADS];
    ThreadCtx ctxs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        ctxs[i].tid        = (uint32_t)(i + 1);   /* 1-indexed; tid 0 reserved for Null */
        ctxs[i].iter       = 0;
        ctxs[i].serial     = 0;
        /* T1 (tid 1) bundles; T2 (tid 2) releases then bundles.  With more
         * than 2 threads the extra threads are bundlers (peers stress Root). */
        ctxs[i].is_bundler = (i != 1);
        pthread_create(&threads[i], NULL, worker, &ctxs[i]);
    }

#if STRESS_SECONDS > 0
    struct timespec ts = { .tv_sec = STRESS_SECONDS, .tv_nsec = 0 };
    nanosleep(&ts, NULL);
    atomic_store_explicit(&g_stop, true, memory_order_release);
#endif

    for (int i = 0; i < NUM_THREADS; i++) pthread_join(threads[i], NULL);

    check_invariants();

    uint64_t completed = atomic_load(&bundles_completed);
    bool c_in_a = atomic_load(&g_cInA);
    bool c_in_b = atomic_load(&g_cInB);

    /* The terminal/safety invariant: after every thread finishes its bundle
     * pipeline, Root is published priority, not missing, and -- since C is
     * still structurally in A and/or B (cInA \/ cInB) -- C is reachable from
     * Root via at least one of the A or B 2-paths (SnapshotConsistency).
     * Mirror the reference's final published-state assertion. */
    Wrapper rw = load_w(NODE_ROOT);
    uint8_t r_pp; uint32_t r_sA, r_sB; bool r_miss;
    pkt_unpack(load_pkt_raw(rw.packet_slot), &r_pp, &r_sA, &r_sB, &r_miss);
    bool terminal_ok = rw.has_priority && !r_miss &&
                       (!(c_in_a || c_in_b) || reachable_from_root(r_sA, r_sB));
    if (!terminal_ok) {
        uint32_t a0 = (r_sA != SLOT_NULL) ? load_pkt_slot0(r_sA) : 0;
        uint32_t b0 = (r_sB != SLOT_NULL) ? load_pkt_slot0(r_sB) : 0;
        fprintf(stderr,
            "FAIL: hardlink_4node\n"
            "  cInA=%d cInB=%d bundles_completed=%llu\n"
            "  Root : prio=%d ser=%u slot=%u pkt=(pl=%u sA=%u sB=%u miss=%d)\n"
            "         A.sub[C]=%u B.sub[C]=%u\n"
            "  A : prio=%d bundled_by=%u ser=%u slot=%u\n"
            "  B : prio=%d bundled_by=%u ser=%u slot=%u\n"
            "  C : prio=%d bundled_by=%u ser=%u slot=%u\n",
            c_in_a, c_in_b, (unsigned long long)completed,
            rw.has_priority, rw.serial, rw.packet_slot, r_pp, r_sA, r_sB, r_miss,
            a0, b0,
            load_w(NODE_A).has_priority, load_w(NODE_A).bundled_by,
            load_w(NODE_A).serial, load_w(NODE_A).packet_slot,
            load_w(NODE_B).has_priority, load_w(NODE_B).bundled_by,
            load_w(NODE_B).serial, load_w(NODE_B).packet_slot,
            load_w(NODE_C).has_priority, load_w(NODE_C).bundled_by,
            load_w(NODE_C).serial, load_w(NODE_C).packet_slot);
        abort();
    }

#if STRESS_SECONDS > 0
    const char *mode_str =
#  if MODE == MODE_COARSE
        "COARSE";
#  elif MODE == MODE_SUPERFINE
        "SUPERFINE";
#  else
        "FINE";
#  endif
    printf("[hardlink_4node stress %s %ds pool=%u] bundles_completed=%llu cInA=%d cInB=%d\n",
           mode_str, STRESS_SECONDS, (unsigned)PACKET_POOL_ENTRIES,
           (unsigned long long)completed, c_in_a, c_in_b);
    printf("  spin: phase1=%llu phase2=%llu phase4=%llu disturbed=%llu release=%llu bundle_restart=%llu stale_read=%llu\n",
           (unsigned long long)atomic_load(&spin_phase1),
           (unsigned long long)atomic_load(&spin_phase2),
           (unsigned long long)atomic_load(&spin_phase4),
           (unsigned long long)atomic_load(&phase4_disturbed),
           (unsigned long long)atomic_load(&spin_release),
           (unsigned long long)atomic_load(&spin_bundle),
           (unsigned long long)atomic_load(&spin_stale_read));
    printf("  llfree: negotiate_wait=%llu preempt=%llu\n",
           (unsigned long long)atomic_load(&spin_negotiate),
           (unsigned long long)atomic_load(&spin_preempt));
#else
    /* Bounded unit run: every thread completed its bundle each iteration. */
    assert(completed >= (uint64_t)NUM_THREADS * (uint64_t)MAX_COMMITS);
    (void)completed;
#endif

    return 0;
}
