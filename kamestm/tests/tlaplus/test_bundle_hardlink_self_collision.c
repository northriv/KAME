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
 * C11 test generated mechanically from
 * BundleUnbundle_hardlink_self_collision.tla.
 *
 * Hardlink self-collision minimal model (3 fixed nodes R, A, C; N threads).
 *
 *   Topology:
 *         R                R is bundle root.
 *        / \               A is direct child of R.
 *       A   \              C is direct child of BOTH A and R (hard-link).
 *        \  |              Once InsertHardLink runs, R logically holds
 *         \ /              both A and C, but R.sub[C] stays Null because
 *          C              C's packet lives under A.sub[C].
 *
 * Initial state (pre-hardlink): a past bundle(R) completed,
 *   - linkage[A].bundledBy = R, A's packet lives in R.sub[A]
 *   - linkage[C].bundledBy = A, C's packet lives in A.sub[C]
 *   - R.sub = (A :> a_pkt, C :> Null)  <- C slot Null until hardlink inserted
 *
 * Race target:
 *   T1: InsertHardLink (registers C as direct child of R) + bundle(R)
 *   T2: concurrent bundle(R)
 *
 * The bundle(R) walk reaches C twice -- once via direct R->C and once via
 * R->A->C.  The is_bundle_root Phase 4 m_missing override is the bug surface.
 *
 * ENCODING: reuses the test_bundle_2level_LLfree.c idiom VERBATIM:
 *   - slot-pool packet ring buffer (immutable once written)
 *   - TID-encoded base-B Lamport serial (gen_serial / encode_serial)
 *   - wrapper pack/unpack into a single atomic uint64_t per node
 *   - wrapper CAS (cas_w) over linkage[]
 *   - LL-free negotiate priority-tag machinery (priority_tag[], gate(),
 *     can_proceed_with_preempt, tag_after_fail/success, clear_my_tags)
 *
 * DIFFERENCES from the 2level base (per the TLA+ model):
 *   - 3 FIXED nodes R, A, C (not Parent + 2 symmetric children).
 *   - R.sub holds two children slots: [A]=s_A, [C]=s_C.  The hardlink
 *     keeps R.sub[C] = SLOT_NULL legitimately (C reachable via A).
 *   - A.sub holds C: A's packet sub-slot [C] = s_AC.
 *   - The operation is NOT a repeating commit loop.  Each thread runs the
 *     one-shot bundle(R) pipeline (5 phases) at most MAX_COMMITS times,
 *     one designated thread additionally performing InsertHardLink first.
 *   - Post-join invariant = SnapshotConsistency, mirroring the TLA+
 *     SnapshotConsistency: any Null slot in published R.sub must be
 *     reachable via ReachableFrom (direct R-slot OR R->A->C 2-path).
 *
 * Atomicity modes (MODE_COARSE / FINE / SUPERFINE) carry over.  In this
 * model the bundle pipeline phases are already CAS-granular, so FINE is
 * the natural default; COARSE wraps the whole pipeline in a mutex.
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

/* --- Node IDs (3 fixed nodes: R, A, C) --- */
#define NODE_R      0
#define NODE_A      1
#define NODE_C      2
#define NUM_NODES   3

/* bundled_by is a 2-bit field.  Reachable bundlers: R (0) and A (1).
 * 3 == NULL_NODE means "priority / no bundler".  TLA+ BundledRefWrapper
 * stores parent in {R, A}; priority wrappers carry bundledBy = Null. */
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
 *   R packet : slot0 = R.sub[A] (A's sub-packet), slot1 = R.sub[C]
 *              (direct C slot, normally SLOT_NULL under hardlink)
 *   A packet : slot0 = A.sub[C] (C's sub-packet),  slot1 = SLOT_NULL
 *   C packet : slot0 = slot1 = SLOT_NULL (leaf)
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
static _Atomic(bool)     g_hardlink_done;      /* TLA+ hardlinkDone   */
static _Atomic(bool)     g_stop;

/* --- Spin / race-detection counters (stress diagnostics) --- */
static _Atomic(uint64_t) spin_bundle;       /* bundle pipeline restarts   */
static _Atomic(uint64_t) spin_phase2;       /* Phase2 CAS R failures      */
static _Atomic(uint64_t) spin_phase3;       /* Phase3 CAS A failures      */
static _Atomic(uint64_t) spin_phase4;       /* Phase4 CAS R failures      */
static _Atomic(uint64_t) spin_stale_read;   /* pool-wrap detector fires   */
static _Atomic(uint64_t) spin_negotiate;    /* can_proceed gated waits    */
static _Atomic(uint64_t) spin_preempt;      /* tag preemptions issued     */
static _Atomic(uint64_t) bundles_completed; /* successful BundlePhase5    */
#define SPIN_INC(name) atomic_fetch_add_explicit(&(name), 1u, memory_order_relaxed)

/* ============================================================================
 * LL-free negotiate machinery (verbatim from 2level_LLfree).
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
    bool     do_hardlink;  /* designated InsertHardLink thread (T1) */
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

/* ============================================================================
 * InsertHardLink(t) -- TLA+ action.  Registers C as a direct child of R.
 * R.sub[C] keeps SLOT_NULL (C's packet still lives under A.sub[C]); the
 * registration is recorded by the g_hardlink_done flag.  Single CAS of R,
 * idempotent: fires only while ~g_hardlink_done.
 * ============================================================================ */
static void insert_hardlink(ThreadCtx *ctx) {
    for (;;) {
        if (atomic_load_explicit(&g_hardlink_done, memory_order_acquire)) return;
        Wrapper rw = load_w(NODE_R);
        if (!rw.has_priority) return;               /* TLA+ rw.hasPriority guard */
        uint8_t pp; uint32_t s_A, s_C; bool rmiss;
        pkt_unpack(load_pkt_raw(rw.packet_slot), &pp, &s_A, &s_C, &rmiss);
        if (s_C != SLOT_NULL) return;               /* TLA+ rw.packet.sub[C] = Null guard */

        /* newSub == [sub EXCEPT ![C] = Null] -> identical sub, fresh wrapper.
         * Keep R.sub[A]=s_A, R.sub[C]=Null, missing unchanged (=FALSE here). */
        uint32_t ser = gen_serial(ctx->serial, rw.serial, ctx->tid);
        uint32_t new_slot = alloc_slot(pp, s_A, SLOT_NULL, rmiss);
        Wrapper new_rw = make_priority_pkt(ser, new_slot);

        gate(NODE_R, ctx->iter, ctx->tid);
        Wrapper exp = rw;
        if (cas_w(NODE_R, &exp, new_rw)) {
            ctx->serial = ser;
            /* hardlinkDone' = TRUE.  Set after the wrapper publish so any
             * thread observing g_hardlink_done sees the new R wrapper too. */
            atomic_store_explicit(&g_hardlink_done, true, memory_order_release);
            return;
        }
        tag_after_fail(NODE_R, ctx->iter, ctx->tid);
        /* CAS lost: retry; another thread may have set hardlink_done. */
    }
}

/* ============================================================================
 * bundle_R(t) -- the 5-phase bundle pipeline on R (is_bundle_root=TRUE).
 * Mirrors TLA+ BundleStart -> BundlePhase1..5.  On any CAS failure the
 * pipeline restarts from Phase1 (TLA+ retry edge to "bundle_phase1").
 * Returns true once Phase5 completes (bundleDone[t] := TRUE).
 * ============================================================================ */
static bool bundle_R(ThreadCtx *ctx) {
    OP_LOCK();
    for (;;) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) { OP_UNLOCK(); return false; }
        uint32_t start_counter = cur_slot_counter();

        /* ---- Phase 1 (Collect): read R, A, C wrappers ---- */
        Wrapper rw = load_w(NODE_R);
        if (!rw.has_priority) { SPIN_INC(spin_bundle); continue; }  /* TLA+ rw.hasPriority */
        uint8_t r_pp; uint32_t r_sA, r_sC; bool r_miss;
        pkt_unpack(load_pkt_raw(rw.packet_slot), &r_pp, &r_sA, &r_sC, &r_miss);
        if (r_miss) { SPIN_INC(spin_bundle); continue; }            /* TLA+ ~rw.packet.missing */

        Wrapper aw = load_w(NODE_A);
        Wrapper cw = load_w(NODE_C);

        /* aSubPkt: A's packet to bundle under R.
         *   aw.hasPriority -> aw.packet ; else rw.packet.sub[A]. */
        uint32_t a_sub_slot;
        if (aw.has_priority)            a_sub_slot = aw.packet_slot;
        else                            a_sub_slot = r_sA;   /* bundledBy R: under R.sub[A] */
        if (a_sub_slot == SLOT_NULL) { SPIN_INC(spin_bundle); continue; }

        /* cSubPkt: C's packet (if visible) -- two collection paths.
         *   cw.hasPriority                         -> cw.packet
         *   cw.bundledBy=A & aw.hasPriority         -> aw.packet.sub[C]
         *   cw.bundledBy=A & ~aw.hasPriority & R.sub[A]!=Null -> R.sub[A].sub[C]
         *   else                                   -> Null */
        uint32_t c_sub_slot;
        if (cw.has_priority) {
            c_sub_slot = cw.packet_slot;
        } else if (cw.bundled_by == NODE_A && aw.has_priority) {
            uint32_t a0;
            pkt_unpack(load_pkt_raw(aw.packet_slot), NULL, &a0, NULL, NULL);
            c_sub_slot = a0;                       /* A.sub[C] */
        } else if (cw.bundled_by == NODE_A && !aw.has_priority && r_sA != SLOT_NULL) {
            uint32_t ra0;
            pkt_unpack(load_pkt_raw(r_sA), NULL, &ra0, NULL, NULL);
            c_sub_slot = ra0;                      /* R.sub[A].sub[C] */
        } else {
            c_sub_slot = SLOT_NULL;
        }
        if (c_sub_slot == SLOT_NULL) {
            /* C not yet visible along any path; tag R and restart (the peer
             * may be mid-flight).  Mirrors TLA+ Phase1 staying guarded. */
            tag_after_fail(NODE_R, ctx->iter, ctx->tid);
            SPIN_INC(spin_bundle); continue;
        }

        if (pool_stale(start_counter)) continue;

        bool hardlink_done = atomic_load_explicit(&g_hardlink_done, memory_order_acquire);

        /* ---- Phase 2 (CAS R): R packet missing=TRUE with re-packed sub[].
         * newASub = [C |-> cPkt]; newAPkt = MakePacket(A, newASub).
         * newRSub[A] = newAPkt ; newRSub[C] = (hardlink ? Null : oldW.sub[C]).
         * ---- */
        uint32_t bundle_ser = gen_serial(ctx->serial, rw.serial, ctx->tid);
        ctx->serial = bundle_ser;

        /* A's fresh packet carrying C in its sub[C] (=slot0). */
        uint8_t a_pp = load_pkt_payload(a_sub_slot);
        uint32_t new_a_slot = alloc_slot(a_pp, c_sub_slot, SLOT_NULL, false);

        uint32_t new_r_sC = hardlink_done ? SLOT_NULL : r_sC;
        uint32_t r2_slot  = alloc_slot(r_pp, new_a_slot, new_r_sC, true /*missing*/);
        Wrapper p2 = make_priority_pkt(bundle_ser, r2_slot);

        gate(NODE_R, ctx->iter, ctx->tid);
        Wrapper exp_p2 = rw;
        if (!cas_w(NODE_R, &exp_p2, p2)) {
            tag_after_fail(NODE_R, ctx->iter, ctx->tid);
            SPIN_INC(spin_phase2); SPIN_INC(spin_bundle);
            continue;                              /* TLA+ -> bundle_phase1 */
        }
        tag_after_success(NODE_R, ctx->tid);

        /* ---- Phase 3 (CAS child): A -> BundledRefWrapper(R).
         * C stays bundledBy=A (its packet remains under A.sub[C]); the
         * direct R.sub[C]=Null is the legitimate hardlink reference.
         * If A already bundledBy R -> skip.  If A diverged & not at R ->
         * BundlePhase3Fail: restart from Phase1. ---- */
        Wrapper aw_now = load_w(NODE_A);
        if (aw_now.bundled_by == NODE_R && !aw_now.has_priority) {
            /* TLA+ disjunct: A already at R -> skip Phase 3. */
        } else if (aw_now.has_priority &&
                   aw_now.serial == aw.serial &&
                   aw_now.packet_slot == aw.packet_slot) {
            /* CAS A (collected wrapper unchanged) -> bundled-ref to R. */
            uint32_t a_ser = gen_serial(ctx->serial, aw.serial, ctx->tid);
            Wrapper b = make_bundled(NODE_R, a_ser);
            gate(NODE_A, ctx->iter, ctx->tid);
            Wrapper exp_a = aw;
            if (!cas_w(NODE_A, &exp_a, b)) {
                tag_after_fail(NODE_A, ctx->iter, ctx->tid);
                SPIN_INC(spin_phase3); SPIN_INC(spin_bundle);
                continue;                          /* TLA+ -> bundle_phase1 */
            }
            ctx->serial = a_ser;
            tag_after_success(NODE_A, ctx->tid);
        } else {
            /* BundlePhase3Fail: A's wrapper diverged from collected and is
             * not at R -> restart from Phase1. */
            tag_after_fail(NODE_A, ctx->iter, ctx->tid);
            SPIN_INC(spin_phase3); SPIN_INC(spin_bundle);
            continue;
        }

        /* ---- Phase 4 (Finalize): clear R's missing flag.
         * finalPkt = MakePacket(R, oldW.packet.sub, FALSE) -- same sub[] as
         * the Phase2 wrapper, missing=FALSE.  is_bundle_root override: the
         * Null R.sub[C] must NOT be flagged missing because C is reachable
         * via A.  CAS expects the Phase2 wrapper (p2). ---- */
        uint32_t fin_slot = alloc_slot(r_pp, new_a_slot, new_r_sC, false /*missing*/);
        uint32_t f_ser = gen_serial(ctx->serial, p2.serial, ctx->tid);
        Wrapper p4 = make_priority_pkt(f_ser, fin_slot);
        gate(NODE_R, ctx->iter, ctx->tid);
        Wrapper exp_p4 = p2;
        if (!cas_w(NODE_R, &exp_p4, p4)) {
            tag_after_fail(NODE_R, ctx->iter, ctx->tid);
            SPIN_INC(spin_phase4); SPIN_INC(spin_bundle);
            continue;                              /* TLA+ -> bundle_phase1 */
        }
        ctx->serial = f_ser;
        tag_after_success(NODE_R, ctx->tid);

        /* ---- Phase 5 (Publish): completion.  Wrapper already published in
         * Phase4; mark bundleDone and release tags (Tx-end). ---- */
        clear_my_tags(ctx->tid);
        SPIN_INC(bundles_completed);
        OP_UNLOCK();
        return true;
    }
}

/* ============================================================================
 * Thread worker.  Each thread runs the one-shot bundle(R) pipeline at most
 * MAX_COMMITS times.  The designated thread (do_hardlink) performs
 * InsertHardLink before its bundle on the first iteration -- mirroring the
 * TLA+ T1: InsertHardLink + bundle(R), T2: bundle(R).
 * ============================================================================ */
static void *worker(void *arg) {
    ThreadCtx ctx = *(ThreadCtx*)arg;

    for (uint32_t i = 0; i < (uint32_t)MAX_COMMITS; i++) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;
        if (ctx.do_hardlink) insert_hardlink(&ctx);
        bundle_R(&ctx);
        ctx.iter++;
    }
    return NULL;
}

/* ============================================================================
 * ReachableFrom(rootPkt, child) -- TLA+ helper for SnapshotConsistency.
 * TRUE iff child reachable from R's published packet via either the direct
 * R-slot or the R->A->C two-path.  Here we only ever query child = C.
 * ============================================================================ */
static bool reachable_from_R(uint32_t r_sA, uint32_t r_sC, int child) {
    if (child == NODE_A) {
        return r_sA != SLOT_NULL;                  /* direct R.sub[A] */
    }
    if (child == NODE_C) {
        /* direct R.sub[C] != Null  OR  (R.sub[A] != Null && R.sub[A].sub[C] != Null) */
        if (r_sC != SLOT_NULL) return true;
        if (r_sA != SLOT_NULL) {
            uint32_t a0;
            pkt_unpack(load_pkt_raw(r_sA), NULL, &a0, NULL, NULL);
            return a0 != SLOT_NULL;                /* A.sub[C] */
        }
        return false;
    }
    return false;
}

/* ============================================================================
 * Post-join invariant checks (mirror the TLA+ INVARIANTs in the .cfg):
 *   SnapshotConsistency, NoMissingHole, BundleRefConsistency, HardlinkExclusive.
 * ============================================================================ */
static void check_invariants(void) {
    Wrapper rw = load_w(NODE_R);
    /* R is the bundle root -> always priority (never bundled). */
    assert(rw.has_priority);

    Wrapper aw = load_w(NODE_A);
    Wrapper cw = load_w(NODE_C);

    /* BundleRefConsistency:
     *   ~aw.hasPriority => aw.bundledBy = R
     *   ~cw.hasPriority => cw.bundledBy in {A, R} */
    if (!aw.has_priority) assert(aw.bundled_by == NODE_R);
    if (!cw.has_priority) assert(cw.bundled_by == NODE_A || cw.bundled_by == NODE_R);

    uint8_t r_pp; uint32_t r_sA, r_sC; bool r_miss;
    pkt_unpack(load_pkt_raw(rw.packet_slot), &r_pp, &r_sA, &r_sC, &r_miss);

    if (!r_miss) {
        /* SnapshotConsistency: when R is published (priority, not missing),
         * every Null child slot must be reachable via ReachableFrom. */
        if (r_sA == SLOT_NULL) assert(reachable_from_R(r_sA, r_sC, NODE_A));
        if (r_sC == SLOT_NULL) assert(reachable_from_R(r_sA, r_sC, NODE_C));

        /* NoMissingHole: present children sub-packets are not themselves
         * missing. */
        if (r_sA != SLOT_NULL) {
            bool am; pkt_unpack(load_pkt_raw(r_sA), NULL, NULL, NULL, &am);
            assert(!am);
        }
        if (r_sC != SLOT_NULL) {
            bool cm; pkt_unpack(load_pkt_raw(r_sC), NULL, NULL, NULL, &cm);
            assert(!cm);
        }
    }

    /* HardlinkExclusive: never C placed BOTH directly under R AND under
     * R.sub[A].sub[C] while ~hardlinkDone (double placement before insert). */
    bool hl = atomic_load_explicit(&g_hardlink_done, memory_order_acquire);
    if (!hl) {
        bool double_placed = false;
        if (r_sC != SLOT_NULL && r_sA != SLOT_NULL) {
            uint32_t a0;
            pkt_unpack(load_pkt_raw(r_sA), NULL, &a0, NULL, NULL);
            double_placed = (a0 != SLOT_NULL);
        }
        assert(!double_placed);
    }
}

int main(void) {
    atomic_store(&global_slot_counter, 0u);

    /* Init (TLA+ Init):
     *   linkage[R] = PriorityWrapper(MakePacket(R, RSubInit, FALSE))
     *     RSubInit = [A |-> a_pkt(with A.sub[C]=c_pkt), C |-> Null]
     *   linkage[A] = BundledRefWrapper(R)
     *   linkage[C] = BundledRefWrapper(A)
     *   hardlinkDone = FALSE
     *
     * Build bottom-up:
     *   c_leaf_pkt : C's leaf packet (payload 0)
     *   a_pkt      : A's packet with sub[C] = c_leaf_pkt
     *   r_pkt      : R's packet with sub[A] = a_pkt, sub[C] = Null, missing=FALSE
     */
    uint32_t s_C_leaf = alloc_slot(0, SLOT_NULL,  SLOT_NULL, false);
    uint32_t s_A_pkt  = alloc_slot(0, s_C_leaf,   SLOT_NULL, false);  /* A.sub[C]=s_C_leaf */
    uint32_t s_R_pkt  = alloc_slot(0, s_A_pkt,    SLOT_NULL, false);  /* R.sub[A]=s_A, [C]=Null */

    Wrapper init_R = make_priority_pkt(0, s_R_pkt);
    Wrapper init_A = make_bundled(NODE_R, 0);   /* A bundledBy R */
    Wrapper init_C = make_bundled(NODE_A, 0);   /* C bundledBy A */

    atomic_store(&linkage[NODE_R], wrapper_pack(init_R));
    atomic_store(&linkage[NODE_A], wrapper_pack(init_A));
    atomic_store(&linkage[NODE_C], wrapper_pack(init_C));
    atomic_store(&g_hardlink_done, false);
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&priority_tag[i], TAG_NULL);
    atomic_store(&g_stop, false);

    pthread_t threads[NUM_THREADS];
    ThreadCtx ctxs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        ctxs[i].tid         = (uint32_t)(i + 1);  /* 1-indexed; tid 0 reserved for Null */
        ctxs[i].iter        = 0;
        ctxs[i].serial      = 0;
        ctxs[i].do_hardlink = (i == 0);           /* T1 inserts the hardlink */
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
    bool hl = atomic_load(&g_hardlink_done);

    /* The terminal/safety invariant: after every thread finishes its bundle
     * pipeline, R is published priority, not missing, and the Null direct
     * C-slot is justified by reachability via A (SnapshotConsistency).  Mirror
     * the reference's final published-state assertion. */
    Wrapper rw = load_w(NODE_R);
    uint8_t r_pp; uint32_t r_sA, r_sC; bool r_miss;
    pkt_unpack(load_pkt_raw(rw.packet_slot), &r_pp, &r_sA, &r_sC, &r_miss);
    if (!(rw.has_priority && !r_miss && r_sA != SLOT_NULL &&
          reachable_from_R(r_sA, r_sC, NODE_C))) {
        fprintf(stderr,
            "FAIL: hardlink_self_collision\n"
            "  hardlinkDone=%d bundles_completed=%llu\n"
            "  R : prio=%d ser=%u slot=%u pkt=(pl=%u sA=%u sC=%u miss=%d)\n"
            "  A : prio=%d bundled_by=%u ser=%u slot=%u\n"
            "  C : prio=%d bundled_by=%u ser=%u slot=%u\n",
            hl, (unsigned long long)completed,
            rw.has_priority, rw.serial, rw.packet_slot, r_pp, r_sA, r_sC, r_miss,
            load_w(NODE_A).has_priority, load_w(NODE_A).bundled_by,
            load_w(NODE_A).serial, load_w(NODE_A).packet_slot,
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
    printf("[hardlink_self_collision stress %s %ds pool=%u] bundles_completed=%llu hardlinkDone=%d\n",
           mode_str, STRESS_SECONDS, (unsigned)PACKET_POOL_ENTRIES,
           (unsigned long long)completed, hl);
    printf("  spin: phase2=%llu phase3=%llu phase4=%llu bundle_restart=%llu stale_read=%llu\n",
           (unsigned long long)atomic_load(&spin_phase2),
           (unsigned long long)atomic_load(&spin_phase3),
           (unsigned long long)atomic_load(&spin_phase4),
           (unsigned long long)atomic_load(&spin_bundle),
           (unsigned long long)atomic_load(&spin_stale_read));
    printf("  llfree: negotiate_wait=%llu preempt=%llu\n",
           (unsigned long long)atomic_load(&spin_negotiate),
           (unsigned long long)atomic_load(&spin_preempt));
#else
    /* Bounded unit run: every thread completed its one-shot bundle each
     * iteration, and the hardlink was inserted exactly once. */
    assert(hl);
    assert(completed >= (uint64_t)NUM_THREADS * (uint64_t)MAX_COMMITS);
    (void)completed;
#endif

    return 0;
}
