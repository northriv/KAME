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
 * C11 test generated mechanically from BundleUnbundle_hardlink_dynamic.tla.
 *
 * Hardlink model -- DynChild1 is referenced from BOTH Parent1 and Parent2:
 *
 *   Parent1 --+-- DynChild1 (hardlinked; packet currently homed at ONE parent)
 *   Parent2 --+----^
 *
 * Extends the REF_DYN port (test_bundle_2level_LLfree_dynamic.c) with the
 * second-parent migration cascade.  Same lock-free packet-pool + LL-free
 * negotiate idiom; the new mechanic is the per-thread op-target parent
 * (`opParent[t]`) and the migration of the child packet from one parent's
 * sub[] into the other's during a bundle:
 *
 *   ReadParent      read opParent's wrapper, route to bundle / commit.
 *   BundlePhase1    collect child sub-packet from:
 *                     (a) child's own priority wrapper, OR
 *                     (b) opParent's own sub[] (child bundledBy = opParent), OR
 *                     (c) the OTHER parent's sub[] (child bundledBy = otherP)
 *                         -- the MIGRATION case (sub_from_other = true).
 *   BundlePhase2    CAS opParent -> missing=TRUE, sub[c] populated.
 *   MigrateClearOther  (only if collected from otherP) CAS otherP -> sub[c]=Null.
 *   BundlePhase3    CAS child.bundledBy = opParent.
 *   BundlePhase4    CAS opParent -> missing=FALSE.
 *
 * RootThreads run CommitParent: pick a parent (opParent), snapshot it
 * (bundling/migrating the child into it as needed), then +1 to the child's
 * sub-packet payload.  unbundle (CommitChild) walks to child.bundledBy (a
 * single value -- no ambiguity).
 *
 * Atomicity: BundleCollectAtomic / BundlePhase3Atomic are "superfine" in the
 * shipped configs.  This port keeps the SUPERFINE-style independent-CAS
 * granularity throughout (each linkage CAS is its own step, full
 * interleaving), matching REF_DYN's MODE_SUPERFINE path; the coarse/fine
 * knobs from REF_DYN are retained for completeness.
 *
 * Terminal/safety invariants (post-join):
 *   SnapshotConsistency   -- a published (~missing) parent that homes the
 *                            child (child.bundledBy = p) has sub[c] /= Null.
 *   HardlinkExclusive     -- at most one parent's sub[] holds the child packet.
 *   BundleRefConsistency  -- child.bundledBy parent has priority + holds it.
 *   NoPriorityLoss        -- child is priority OR bundled under a parent.
 *   TerminalPayloadCheck  -- ChildPayload(DynChild1) = commitCount.
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
#define MODE MODE_SUPERFINE
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

/* Mirrors TLA+ CONSTANTS InsertThreads / RootThreads / LeafThreads /
 * ReleaseThreads (hardlink dynamic).  The shipped configs run
 *   InsertThreads = {}, RootThreads = Threads, LeafThreads = {},
 *   ReleaseThreads = {}
 * (DynChild1 is pre-inserted as a hardlink at Init, so no run-time insert is
 * needed to exercise the migration race).
 *   ROLE_ROOT    : CommitParent on a chosen opParent (snapshot + bundle/migrate)
 *   ROLE_LEAF    : CommitChild (direct commit or unbundle-walk to bundledBy)
 *   ROLE_INSERT  : participates in run-time insert (off by default)
 *   ROLE_RELEASE : participates in release (off by default) */
#define ROLE_ROOT     0x1u
#define ROLE_LEAF     0x2u
#define ROLE_INSERT   0x4u
#define ROLE_RELEASE  0x8u

/* Per-role thread counts.  Default = all threads are ROLE_ROOT (matching
 * RootThreads = Threads, LeafThreads = {}).  INSERT/RELEASE default to 0
 * since the shipped configs leave InsertThreads / ReleaseThreads empty. */
#ifndef NUM_ROOT_ONLY
#define NUM_ROOT_ONLY NUM_THREADS
#endif
#ifndef NUM_LEAF_ONLY
#define NUM_LEAF_ONLY 0
#endif
#ifndef NUM_INSERT_THREADS
#define NUM_INSERT_THREADS 0
#endif
#ifndef NUM_RELEASE_THREADS
#define NUM_RELEASE_THREADS 0
#endif
#if (NUM_ROOT_ONLY + NUM_LEAF_ONLY) > NUM_THREADS
#  error "NUM_ROOT_ONLY + NUM_LEAF_ONLY exceeds NUM_THREADS"
#endif
#if NUM_INSERT_THREADS > NUM_THREADS || NUM_RELEASE_THREADS > NUM_THREADS
#  error "Insert/Release thread count exceeds NUM_THREADS"
#endif
#define NUM_BOTH         (NUM_THREADS - NUM_ROOT_ONLY - NUM_LEAF_ONLY)
#define NUM_ROOT_THREADS (NUM_ROOT_ONLY + NUM_BOTH)
#define NUM_LEAF_THREADS (NUM_LEAF_ONLY + NUM_BOTH)
#if (NUM_ROOT_THREADS + NUM_LEAF_THREADS) < NUM_THREADS
#  error "TLA+ ASSUME violation: RootThreads must cover Threads"
#endif

/* Packet pool ring (immutable once written). */
#ifndef PACKET_POOL_ENTRIES
#define PACKET_POOL_ENTRIES 134217727u
#endif
#if (PACKET_POOL_ENTRIES) > 134217727u
#  error "PACKET_POOL_ENTRIES must fit in 27 bits (<=134217727)"
#endif

/* --- Node IDs (TLA+ Parent1, Parent2, DynChild1) --- */
#define NODE_PARENT1 0
#define NODE_PARENT2 1
#define NODE_CHILD1  2
#define NUM_NODES    3

/* bundled_by is a 2-bit field: 0=Parent1, 1=Parent2, 3=NULL_NODE (priority).
 * The child may be bundled by EITHER parent (hardlink). */
#define NULL_NODE   0x3u

/* OtherParent(p) -- TLA+ helper. */
static inline int other_parent(int p) {
    return (p == NODE_PARENT1) ? NODE_PARENT2 : NODE_PARENT1;
}

/* --- Serial + slot widths --- */
#define SER_BITS   27u
#define SER_MOD    (1u << SER_BITS)
#define SER_MASK   (SER_MOD - 1u)
#define SLOT_BITS  27u
#define SLOT_MASK  ((1u << SLOT_BITS) - 1u)
#define SLOT_NULL  0u   /* reserved */

static inline bool ser_gt(uint32_t a, uint32_t b) {
    uint32_t diff = (a - b) & SER_MASK;
    return diff > 0 && diff < (SER_MOD >> 1);
}

/* TID-encoded base-B Lamport serial -- mirrors REF_DYN / C++
 * SerialGenerator::gen().  Counter in upper bits + TID in lower bits. */
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
 *   payload(8) | sub_slot[child](27) | missing(1)
 * Parents carry one sub-slot (the DynChild1 packet, or SLOT_NULL).  The child
 * packet itself has no sub (its slot field stays SLOT_NULL).
 * ============================================================================ */
static _Atomic(uint64_t) packet_pool[PACKET_POOL_ENTRIES + 1];
static _Atomic(uint32_t) global_slot_counter;

static inline uint64_t pkt_pack(uint8_t payload, uint32_t s0, bool missing) {
    uint64_t v = 0;
    v |= (uint64_t)payload;
    v |= ((uint64_t)(s0 & SLOT_MASK)) << 8;
    v |= ((uint64_t)(missing ? 1u : 0u)) << 35;
    return v;
}
static inline void pkt_unpack(uint64_t v, uint8_t *payload, uint32_t *s0, bool *missing) {
    if (payload) *payload = (uint8_t)(v & 0xFFu);
    if (s0)      *s0      = (uint32_t)((v >> 8) & (uint64_t)SLOT_MASK);
    if (missing) *missing = (v >> 35) & 1u;
}

static inline uint32_t alloc_slot(uint8_t payload, uint32_t s0, bool missing) {
    uint32_t c = atomic_fetch_add_explicit(&global_slot_counter, 1u, memory_order_relaxed);
    uint32_t s = (c % PACKET_POOL_ENTRIES) + 1u;
    atomic_store_explicit(&packet_pool[s], pkt_pack(payload, s0, missing),
                          memory_order_release);
    return s;
}
static inline uint64_t load_pkt_raw(uint32_t slot) {
    return atomic_load_explicit(&packet_pool[slot], memory_order_acquire);
}
static inline uint8_t load_pkt_payload(uint32_t slot) {
    uint8_t p; pkt_unpack(load_pkt_raw(slot), &p, NULL, NULL);
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
static _Atomic(uint64_t) linkage[NUM_NODES];
static _Atomic(uint32_t) commit_count[NUM_NODES];   /* only CHILD1 used */
static _Atomic(bool)     g_stop;

/* Hardlink dynamic state.  insertedIn[c,p] is per (child,parent) pair; for the
 * single child we track it per parent.  ever_inserted likewise.  Both start
 * TRUE at Init (child pre-inserted under both parents). */
static _Atomic(bool)     inserted_in[NUM_NODES];     /* indexed by parent node */
static _Atomic(bool)     ever_inserted_in[NUM_NODES];/* indexed by parent node */
static _Atomic(uint32_t) insert_target_claim[NUM_NODES];  /* per parent (claim) */
static _Atomic(uint32_t) release_target_claim[NUM_NODES]; /* per parent (claim) */

/* --- Spin counters --- */
static _Atomic(uint64_t) spin_bundle;
static _Atomic(uint64_t) spin_commit_parent;
static _Atomic(uint64_t) spin_commit_child;
static _Atomic(uint64_t) spin_stale_read;
static _Atomic(uint64_t) spin_negotiate;
static _Atomic(uint64_t) spin_preempt;
static _Atomic(uint64_t) spin_migrate;     /* MigrateClearOther retries */
#define SPIN_INC(name) atomic_fetch_add_explicit(&(name), 1u, memory_order_relaxed)

/* ============================================================================
 * LL-free negotiate machinery (verbatim from REF_DYN).
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
static inline void tag_after_success(int n, uint32_t my_tid) { (void)n; (void)my_tid; }
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
    uint32_t iter;
    uint32_t serial;
    uint32_t role;
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
    if (advanced >= threshold) { SPIN_INC(spin_stale_read); return true; }
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
static inline Wrapper make_priority_leaf(uint8_t payload, uint32_t serial, bool pkt_missing) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .packet_slot  = alloc_slot(payload, SLOT_NULL, pkt_missing),
    };
}
static inline Wrapper make_priority_parent(uint8_t payload, uint32_t serial,
                                           uint32_t s0, bool pkt_missing) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .packet_slot  = alloc_slot(payload, s0, pkt_missing),
    };
}
__attribute__((unused))
static inline Wrapper make_priority_from_slot(uint32_t packet_slot, uint32_t serial) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .packet_slot  = packet_slot,
    };
}
static inline Wrapper make_bundled(uint8_t parent_node, uint32_t serial) {
    return (Wrapper){
        .has_priority = false, .serial = serial, .bundled_by = parent_node,
        .packet_slot  = SLOT_NULL,
    };
}

/* =========================================================================
 * try_bundle(opParent): run when opParent's packet.missing=true.  Collects
 * the child's sub-packet -- possibly MIGRATING it out of the OTHER parent's
 * sub[] -- then CASes opParent -> Phase2 (missing=TRUE, sub populated),
 * optionally MigrateClearOther (CAS otherP sub[c]=Null), flips the child to
 * bundled-by-opParent (Phase3), then CASes opParent -> Phase4 (missing=FALSE).
 * Returns finalized opParent wrapper in *out_final on success.
 * ========================================================================= */
static bool try_bundle(ThreadCtx *ctx, int opp, Wrapper *out_final) {
    int op2 = other_parent(opp);

    Wrapper pw = load_w(opp);
    if (!pw.has_priority) { SPIN_INC(spin_bundle); return false; }

    uint8_t  pp; uint32_t old_s0; bool pw_missing;
    pkt_unpack(load_pkt_raw(pw.packet_slot), &pp, &old_s0, &pw_missing);

    if (!pw_missing) { *out_final = pw; return true; }

    uint32_t bundle_ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
    ctx->serial = bundle_ser;

#if MODE == MODE_SUPERFINE
    /* BundlePhase1 serial-sync prestamp (TLA+ parentW.serial /= ser). */
    if (pw.serial != bundle_ser) {
        gate(opp, ctx->iter, ctx->tid);
        Wrapper prestamp = make_priority_from_slot(pw.packet_slot, bundle_ser);
        Wrapper exp = pw;
        if (!cas_w(opp, &exp, prestamp)) {
            tag_after_fail(opp, ctx->iter, ctx->tid);
            SPIN_INC(spin_bundle); return false;
        }
        tag_after_success(opp, ctx->tid);
        pw = prestamp;
    }
#endif

    /* Phase1 collect (single child).  ActiveOn(opParent): insertedIn[c,opp]. */
    bool c_active = atomic_load_explicit(&inserted_in[opp], memory_order_acquire);
    if (!c_active) {
        /* TLA+ activeOnP = {} -> straight to Phase4 (no children). */
        gate(opp, ctx->iter, ctx->tid);
        Wrapper p4 = make_priority_parent(pp, bundle_ser, SLOT_NULL, false);
        Wrapper exp_p4 = pw;
        if (!cas_w(opp, &exp_p4, p4)) {
            tag_after_fail(opp, ctx->iter, ctx->tid);
            SPIN_INC(spin_bundle); return false;
        }
        tag_after_success(opp, ctx->tid);
        *out_final = p4;
        return true;
    }

    Wrapper cw = load_w(NODE_CHILD1);
    uint32_t cpkt = SLOT_NULL;
    bool from_other = false;
    if (cw.has_priority) {
        cpkt = cw.packet_slot;                         /* (a) own priority */
    } else if (cw.bundled_by == opp) {
        cpkt = (old_s0 != SLOT_NULL) ? old_s0 : SLOT_NULL;  /* (b) opParent.sub */
    } else if (cw.bundled_by == op2) {
        /* (c) MIGRATION: read otherP's sub[c]. */
        Wrapper op2w = load_w(op2);
        if (op2w.has_priority) {
            uint32_t o_s0; pkt_unpack(load_pkt_raw(op2w.packet_slot), NULL, &o_s0, NULL);
            if (o_s0 != SLOT_NULL) { cpkt = o_s0; from_other = true; }
        }
    }
    if (cpkt == SLOT_NULL) {
        /* Collect-fail: tag opParent + child, retry from snapshot. */
        tag_after_fail(opp, ctx->iter, ctx->tid);
        tag_after_fail(NODE_CHILD1, ctx->iter, ctx->tid);
        SPIN_INC(spin_bundle); return false;
    }

    /* Phase2: opParent packet = (pp, sub=cpkt, missing=TRUE). */
    gate(opp, ctx->iter, ctx->tid);
    Wrapper p2 = make_priority_parent(pp, bundle_ser, cpkt, true);
    Wrapper exp_p2 = pw;
    if (!cas_w(opp, &exp_p2, p2)) {
        tag_after_fail(opp, ctx->iter, ctx->tid);
        SPIN_INC(spin_bundle); return false;
    }
    tag_after_success(opp, ctx->tid);

    /* MigrateClearOther: only when the packet was pulled from otherP's sub[].
     * CAS otherP -> sub[c]=Null, validating otherP still holds our packet. */
    if (from_other) {
        for (;;) {
            gate(op2, ctx->iter, ctx->tid);
            Wrapper op2w = load_w(op2);
            uint32_t o_s0; bool o_m;
            bool valid = false;
            if (op2w.has_priority) {
                pkt_unpack(load_pkt_raw(op2w.packet_slot), NULL, &o_s0, &o_m);
                /* op2W.packet.sub[c] /= Null /\ = local subpacket */
                valid = (o_s0 != SLOT_NULL && o_s0 == cpkt);
            }
            if (!valid) {
                /* MigrateClearOtherFail: otherP changed -> restart pipeline. */
                tag_after_fail(op2, ctx->iter, ctx->tid);
                SPIN_INC(spin_migrate); return false;
            }
            uint8_t o_pp; pkt_unpack(load_pkt_raw(op2w.packet_slot), &o_pp, NULL, NULL);
            uint32_t new_ser = gen_serial(ctx->serial, op2w.serial, ctx->tid);
            Wrapper new_op2 = make_priority_parent(o_pp, new_ser, SLOT_NULL, o_m);
            Wrapper exp_op2 = op2w;
            if (cas_w(op2, &exp_op2, new_op2)) {
                ctx->serial = new_ser;
                tag_after_success(op2, ctx->tid);
                break;
            }
            tag_after_fail(op2, ctx->iter, ctx->tid);
            SPIN_INC(spin_migrate);
            /* op2w changed under us -> re-validate next loop; if it no longer
             * holds our packet, the valid-check above aborts the pipeline. */
        }
    }

    /* Phase3: child -> bundled-by-opParent. */
    gate(NODE_CHILD1, ctx->iter, ctx->tid);
    Wrapper b = make_bundled((uint8_t)opp, bundle_ser);
    Wrapper exp_c = cw;
    if (!cas_w(NODE_CHILD1, &exp_c, b)) {
        tag_after_fail(NODE_CHILD1, ctx->iter, ctx->tid);
#if MODE == MODE_SUPERFINE
        tag_after_fail(opp, ctx->iter, ctx->tid);
#endif
        SPIN_INC(spin_bundle); return false;
    }
    tag_after_success(NODE_CHILD1, ctx->tid);

    /* Phase4: opParent packet.missing=FALSE (fresh slot prevents ABA). */
    gate(opp, ctx->iter, ctx->tid);
    Wrapper p4 = make_priority_parent(pp, bundle_ser, cpkt, false);
    Wrapper exp_p4 = p2;
    if (!cas_w(opp, &exp_p4, p4)) {
        tag_after_fail(opp, ctx->iter, ctx->tid);
        SPIN_INC(spin_bundle); return false;
    }
    tag_after_success(opp, ctx->tid);

    *out_final = p4;
    return true;
}

/* snapshot(opParent): loop until opParent is priority + non-missing. */
static void snapshot(ThreadCtx *ctx, int opp, Wrapper *out) {
    for (;;) {
        Wrapper pw = load_w(opp);
        if (!pw.has_priority) continue;   /* parents are roots, always priority */
        bool m; pkt_unpack(load_pkt_raw(pw.packet_slot), NULL, NULL, &m);
        if (!m) { *out = pw; return; }
        Wrapper tmp;
        if (try_bundle(ctx, opp, &tmp)) { *out = tmp; return; }
    }
}

/* =========================================================================
 * CommitParent(opParent): snapshot opParent (bundling/migrating the child in),
 * +1 to the child's sub-packet, CAS opParent.  TLA+ SnapRead chose opParent.
 * ========================================================================= */
static void commit_parent(ThreadCtx *ctx, int opp) {
    OP_LOCK();
    for (;;) {
        uint32_t start_counter = cur_slot_counter();
        Wrapper pw;
        snapshot(ctx, opp, &pw);

        uint8_t pp; uint32_t c_slot; bool pm;
        pkt_unpack(load_pkt_raw(pw.packet_slot), &pp, &c_slot, &pm);
        if (pm) { SPIN_INC(spin_commit_parent); continue; }

        bool c_in_snap = (c_slot != SLOT_NULL);
        if (!c_in_snap) {
            /* TLA+ snapChildren = {} -> skip iteration. */
            clear_my_tags(ctx->tid);
            OP_UNLOCK();
            return;
        }
        uint8_t c_payload = load_pkt_payload(c_slot);
        if (pool_stale(start_counter)) continue;

        uint32_t new_c = alloc_slot((uint8_t)((c_payload + 1u) % MAX_PAYLOAD),
                                    SLOT_NULL, false);
        uint32_t ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
        Wrapper new_pw = make_priority_parent(pp, ser, new_c, false);
        gate(opp, ctx->iter, ctx->tid);
        Wrapper exp = pw;
        if (cas_w(opp, &exp, new_pw)) {
            ctx->serial = ser;
            atomic_fetch_add_explicit(&commit_count[NODE_CHILD1], 1, memory_order_relaxed);
            clear_my_tags(ctx->tid);
            OP_UNLOCK();
            return;
        }
        tag_after_fail(opp, ctx->iter, ctx->tid);
        SPIN_INC(spin_commit_parent);
    }
}

/* =========================================================================
 * CommitChild: direct commit if priority, else unbundle-walk to the child's
 * bundledBy parent (a single value -- no ambiguity in this model).
 * ========================================================================= */
static void commit_child(ThreadCtx *ctx) {
    OP_LOCK();
    for (;;) {
        /* CommitSkip: child released from every parent. */
        if (!atomic_load_explicit(&inserted_in[NODE_PARENT1], memory_order_acquire) &&
            !atomic_load_explicit(&inserted_in[NODE_PARENT2], memory_order_acquire)) {
            clear_my_tags(ctx->tid);
            OP_UNLOCK();
            return;
        }
        uint32_t start_counter = cur_slot_counter();
        Wrapper cw = load_w(NODE_CHILD1);

        if (cw.has_priority) {
            uint8_t old_payload = load_pkt_payload(cw.packet_slot);
            if (pool_stale(start_counter)) continue;
            uint8_t new_payload = (uint8_t)((old_payload + 1u) % MAX_PAYLOAD);
            uint32_t ser = gen_serial(ctx->serial, cw.serial, ctx->tid);
            Wrapper new_cw = make_priority_leaf(new_payload, ser, false);
            gate(NODE_CHILD1, ctx->iter, ctx->tid);
            Wrapper exp = cw;
            if (cas_w(NODE_CHILD1, &exp, new_cw)) {
                ctx->serial = ser;
                atomic_fetch_add_explicit(&commit_count[NODE_CHILD1], 1, memory_order_relaxed);
                clear_my_tags(ctx->tid);
                OP_UNLOCK();
                return;
            }
            tag_after_fail(NODE_CHILD1, ctx->iter, ctx->tid);
            SPIN_INC(spin_commit_child);
            continue;
        }

        /* UnbundleWalk: walk to child.bundledBy parent. */
        int bp = cw.bundled_by;
        if (bp != NODE_PARENT1 && bp != NODE_PARENT2) {
            SPIN_INC(spin_commit_child); continue;
        }
        Wrapper pw = load_w(bp);
        if (!pw.has_priority) { SPIN_INC(spin_commit_child); continue; }

        uint8_t pp; uint32_t p_s0; bool pm;
        pkt_unpack(load_pkt_raw(pw.packet_slot), &pp, &p_s0, &pm);
        if (p_s0 == SLOT_NULL) {
            /* sub[c]=Null at the bundledBy parent + child not inserted -> abort. */
            if (!atomic_load_explicit(&inserted_in[bp], memory_order_acquire)) {
                clear_my_tags(ctx->tid);
                OP_UNLOCK();
                return;
            }
            SPIN_INC(spin_commit_child); continue;
        }
        uint8_t old_payload = load_pkt_payload(p_s0);
        if (pool_stale(start_counter)) continue;
        uint8_t new_payload = (uint8_t)((old_payload + 1u) % MAX_PAYLOAD);

        /* UnbundleCASAncestor: bundledBy parent packet missing=true. */
        uint32_t anc_ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
        Wrapper new_pw = make_priority_parent(pp, anc_ser, p_s0, true);
        gate(bp, ctx->iter, ctx->tid);
        Wrapper exp_pw = pw;
        if (!cas_w(bp, &exp_pw, new_pw)) {
            tag_after_fail(bp, ctx->iter, ctx->tid);
            SPIN_INC(spin_commit_child); continue;
        }
        tag_after_success(bp, ctx->tid);
        ctx->serial = anc_ser;

        /* UnbundleCASChild: child becomes priority with new payload. */
        uint32_t c_ser = gen_serial(ctx->serial, cw.serial, ctx->tid);
        Wrapper new_child = make_priority_leaf(new_payload, c_ser, false);
        gate(NODE_CHILD1, ctx->iter, ctx->tid);
        Wrapper exp_child = cw;
        if (cas_w(NODE_CHILD1, &exp_child, new_child)) {
            ctx->serial = c_ser;
            atomic_fetch_add_explicit(&commit_count[NODE_CHILD1], 1, memory_order_relaxed);
            clear_my_tags(ctx->tid);
            OP_UNLOCK();
            return;
        }
        tag_after_fail(NODE_CHILD1, ctx->iter, ctx->tid);
        SPIN_INC(spin_commit_child);
    }
}

/* =========================================================================
 * insert_child(parent): TLA+ InsertStart .. InsertFinal on a chosen parent.
 * Off by default (InsertThreads = {}).  Claims the parent slot, runs the
 * insert pipeline on opParent = parent.  Kept faithful to REF_DYN; single
 * child.  Returns true on success.
 * ========================================================================= */
static bool insert_child(int parent, ThreadCtx *ctx) {
    uint32_t expected_claim = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &insert_target_claim[parent], &expected_claim, ctx->tid,
            memory_order_acq_rel, memory_order_relaxed))
        return false;
    if (atomic_load_explicit(&ever_inserted_in[parent], memory_order_acquire)) {
        atomic_store_explicit(&insert_target_claim[parent], 0u, memory_order_release);
        return false;
    }
    OP_LOCK();
    for (;;) {
        Wrapper pw;
        snapshot(ctx, parent, &pw);

        uint32_t s_ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
        Wrapper stamped = make_priority_from_slot(pw.packet_slot, s_ser);
        gate(parent, ctx->iter, ctx->tid);
        Wrapper exp_pw = pw;
        if (!cas_w(parent, &exp_pw, stamped)) {
            tag_after_fail(parent, ctx->iter, ctx->tid);
            continue;
        }
        tag_after_success(parent, ctx->tid);
        ctx->serial = s_ser;
        pw = stamped;

        Wrapper cw = load_w(NODE_CHILD1);
        uint32_t saved_child_pkt;
        if (cw.has_priority) {
            saved_child_pkt = cw.packet_slot;
            Wrapper inserted_ref = (Wrapper){
                .has_priority = false, .serial = s_ser,
                .bundled_by = (uint8_t)parent, .packet_slot = saved_child_pkt,
            };
            gate(NODE_CHILD1, ctx->iter, ctx->tid);
            Wrapper exp_c = cw;
            if (!cas_w(NODE_CHILD1, &exp_c, inserted_ref)) {
                tag_after_fail(NODE_CHILD1, ctx->iter, ctx->tid);
                continue;
            }
            tag_after_success(NODE_CHILD1, ctx->tid);
        } else if (cw.bundled_by == parent) {
            if (cw.packet_slot != SLOT_NULL) {
                saved_child_pkt = cw.packet_slot;
            } else {
                uint32_t p_s0;
                pkt_unpack(load_pkt_raw(pw.packet_slot), NULL, &p_s0, NULL);
                saved_child_pkt = p_s0;
                if (saved_child_pkt == SLOT_NULL) continue;
            }
        } else {
            /* bundled under the OTHER parent -- second-insert needs the full
             * migration path; TLA+ InsertReadChild restarts at insert_snap. */
            continue;
        }

        uint8_t saved_payload; uint32_t saved_s0_sub; bool saved_missing;
        pkt_unpack(load_pkt_raw(saved_child_pkt),
                   &saved_payload, &saved_s0_sub, &saved_missing);
        uint32_t new_child_slot = alloc_slot(
            (uint8_t)((saved_payload + 1u) % MAX_PAYLOAD), saved_s0_sub, saved_missing);

        uint8_t pp; bool pm;
        pkt_unpack(load_pkt_raw(pw.packet_slot), &pp, NULL, &pm);
        uint32_t f_ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
        Wrapper new_pw = make_priority_parent(pp, f_ser, new_child_slot, true);
        gate(parent, ctx->iter, ctx->tid);
        Wrapper exp_pw2 = pw;
        if (!cas_w(parent, &exp_pw2, new_pw)) {
            tag_after_fail(parent, ctx->iter, ctx->tid);
            continue;
        }
        ctx->serial = f_ser;
        atomic_store_explicit(&inserted_in[parent], true, memory_order_release);
        atomic_store_explicit(&ever_inserted_in[parent], true, memory_order_release);
        clear_my_tags(ctx->tid);
        atomic_store_explicit(&insert_target_claim[parent], 0u, memory_order_release);
        OP_UNLOCK();
        return true;
    }
}

/* =========================================================================
 * release_child(parent): TLA+ ReleaseStart .. ReleaseCASChild on a parent.
 * Off by default (ReleaseThreads = {}).  CAS parent dropping sub[c]; if the
 * child was homed here, restore it to priority.
 * ========================================================================= */
static bool release_child(int parent, ThreadCtx *ctx) {
    uint32_t expected_claim = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &release_target_claim[parent], &expected_claim, ctx->tid,
            memory_order_acq_rel, memory_order_relaxed))
        return false;
    if (!atomic_load_explicit(&inserted_in[parent], memory_order_acquire)) {
        atomic_store_explicit(&release_target_claim[parent], 0u, memory_order_release);
        return false;
    }
    OP_LOCK();
    for (;;) {
        Wrapper pw;
        snapshot(ctx, parent, &pw);

        uint8_t pp; uint32_t p_s0; bool pm;
        pkt_unpack(load_pkt_raw(pw.packet_slot), &pp, &p_s0, &pm);
        uint32_t saved_child_pkt = p_s0;
        if (saved_child_pkt == SLOT_NULL) {
            atomic_store_explicit(&inserted_in[parent], false, memory_order_release);
            clear_my_tags(ctx->tid);
            atomic_store_explicit(&release_target_claim[parent], 0u, memory_order_release);
            OP_UNLOCK();
            return true;
        }
        uint32_t r_ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
        Wrapper new_pw = make_priority_parent(pp, r_ser, SLOT_NULL, pm);
        gate(parent, ctx->iter, ctx->tid);
        Wrapper exp_pw = pw;
        if (!cas_w(parent, &exp_pw, new_pw)) {
            tag_after_fail(parent, ctx->iter, ctx->tid);
            continue;
        }
        ctx->serial = r_ser;
        atomic_store_explicit(&inserted_in[parent], false, memory_order_release);

        /* ReleaseReadChild/CASChild: if the child was homed here (bundledBy =
         * parent), restore it to priority.  If homed elsewhere or already
         * priority, nothing more to do. */
        for (;;) {
            Wrapper cw = load_w(NODE_CHILD1);
            if (cw.has_priority) break;
            if (cw.bundled_by != parent) break;   /* homed at other parent */
            uint32_t c_ser = gen_serial(ctx->serial, cw.serial, ctx->tid);
            Wrapper restored = make_priority_from_slot(saved_child_pkt, c_ser);
            gate(NODE_CHILD1, ctx->iter, ctx->tid);
            Wrapper exp_c = cw;
            if (cas_w(NODE_CHILD1, &exp_c, restored)) { ctx->serial = c_ser; break; }
            tag_after_fail(NODE_CHILD1, ctx->iter, ctx->tid);
        }
        clear_my_tags(ctx->tid);
        atomic_store_explicit(&release_target_claim[parent], 0u, memory_order_release);
        OP_UNLOCK();
        return true;
    }
}

/* =========================================================================
 * Thread worker (hardlink dynamic): role-gated phases.  RootThreads commit
 * parents; opParent is chosen per iteration (TLA+ SnapRead \E p \in Parents).
 * For determinism in the unit run, thread i prefers parent (i % 2).
 * ========================================================================= */
static void *worker(void *arg) {
    ThreadCtx ctx = *(ThreadCtx*)arg;

    /* --- Phase A: Insert (off by default) --- */
    if (ctx.role & ROLE_INSERT) {
        for (int p = NODE_PARENT1; p <= NODE_PARENT2; p++) {
            if (atomic_load_explicit(&g_stop, memory_order_relaxed)) goto release_phase;
            (void)insert_child(p, &ctx);
        }
    }

    /* --- Phase B: Commit loop --- */
    if (ctx.role & (ROLE_ROOT | ROLE_LEAF)) {
        /* SnapRead/BeginChildIteration precondition: child ever-inserted under
         * all parents (everInsertedIn). */
        for (;;) {
            if (atomic_load_explicit(&g_stop, memory_order_relaxed)) goto release_phase;
            if (atomic_load_explicit(&ever_inserted_in[NODE_PARENT1], memory_order_acquire) &&
                atomic_load_explicit(&ever_inserted_in[NODE_PARENT2], memory_order_acquire))
                break;
#if defined(__x86_64__) || defined(__i386__)
            __asm__ __volatile__("pause");
#endif
        }

        /* opParent preference: stagger across parents so two root threads
         * exercise the migration race (each tries to pull the child to its
         * own parent).  TLA+ SnapRead's \E p \in Parents is a free choice. */
        int pref = ((int)(ctx.tid - 1) & 1) ? NODE_PARENT2 : NODE_PARENT1;

        for (uint32_t i = 0; i < (uint32_t)MAX_COMMITS; i++) {
            if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;
            /* SkipIteration: child released from every parent. */
            if (!atomic_load_explicit(&inserted_in[NODE_PARENT1], memory_order_acquire) &&
                !atomic_load_explicit(&inserted_in[NODE_PARENT2], memory_order_acquire))
                break;
            if (ctx.role & ROLE_ROOT) {
                /* Choose an opParent the child is currently inserted under,
                 * preferring `pref`. */
                int opp = pref;
                if (!atomic_load_explicit(&inserted_in[opp], memory_order_acquire))
                    opp = other_parent(opp);
                if (atomic_load_explicit(&inserted_in[opp], memory_order_acquire))
                    commit_parent(&ctx, opp);
            }
            if (ctx.role & ROLE_LEAF) commit_child(&ctx);
            ctx.iter++;
        }
    }

release_phase:
    /* --- Phase C: Release (off by default) --- */
    if (ctx.role & ROLE_RELEASE) {
        for (int p = NODE_PARENT1; p <= NODE_PARENT2; p++) {
            if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;
            (void)release_child(p, &ctx);
        }
    }
    return NULL;
}

/* =========================================================================
 * Post-join invariant checks (TLA+ safety invariants).
 * ========================================================================= */
static void check_invariants(void) {
    Wrapper p1w = load_w(NODE_PARENT1);
    Wrapper p2w = load_w(NODE_PARENT2);
    Wrapper cw  = load_w(NODE_CHILD1);

    /* Parents are roots -- always priority. */
    assert(p1w.has_priority);
    assert(p2w.has_priority);

    /* NoPriorityLoss: child is priority OR bundled under a parent. */
    assert(cw.has_priority ||
           cw.bundled_by == NODE_PARENT1 || cw.bundled_by == NODE_PARENT2);

    uint32_t s1, s2; bool m1, m2;
    pkt_unpack(load_pkt_raw(p1w.packet_slot), NULL, &s1, &m1);
    pkt_unpack(load_pkt_raw(p2w.packet_slot), NULL, &s2, &m2);

    /* HardlinkExclusive: at most one parent's sub[] holds the child packet. */
    assert(!((s1 != SLOT_NULL) && (s2 != SLOT_NULL)));

    /* BundleRefConsistency: child.bundledBy parent has priority and either is
     * missing OR holds the child's packet. */
    if (!cw.has_priority) {
        int bp = cw.bundled_by;
        assert(bp == NODE_PARENT1 || bp == NODE_PARENT2);
        Wrapper bpw = (bp == NODE_PARENT1) ? p1w : p2w;
        bool bm = (bp == NODE_PARENT1) ? m1 : m2;
        uint32_t bs = (bp == NODE_PARENT1) ? s1 : s2;
        assert(bpw.has_priority);
        assert(bm || bs != SLOT_NULL);
    }

    /* SnapshotConsistency: a published (~missing) parent that homes the child
     * (child bundled & bundledBy = p) has sub[c] /= Null. */
    if (p1w.has_priority && !m1 && !cw.has_priority && cw.bundled_by == NODE_PARENT1)
        assert(s1 != SLOT_NULL);
    if (p2w.has_priority && !m2 && !cw.has_priority && cw.bundled_by == NODE_PARENT2)
        assert(s2 != SLOT_NULL);
}

int main(void) {
    atomic_store(&global_slot_counter, 0u);

    /* Init (TLA+):
     *   Parent1 = PriorityWrapper(MakePacket(Parent1, 0, [c|->childPkt], FALSE))
     *   Parent2 = PriorityWrapper(MakePacket(Parent2, 0, EmptySub, FALSE))
     *   DynChild1 = BundledRefWrapper(Parent1)   (homed at Parent1)
     *   insertedIn[c,p] = everInsertedIn[c,p] = TRUE for both parents
     *   commitCount[c] = 0
     * The child's packet (payload 0) lives in Parent1.sub[child]. */
    uint32_t s_childpkt = alloc_slot(0, SLOT_NULL, false);   /* child's packet */
    uint32_t s_p1 = alloc_slot(0, s_childpkt, false);        /* Parent1 holds child */
    uint32_t s_p2 = alloc_slot(0, SLOT_NULL, false);         /* Parent2 empty sub */

    Wrapper init_p1 = (Wrapper){.has_priority=true, .serial=0,
                                .bundled_by=NULL_NODE, .packet_slot=s_p1};
    Wrapper init_p2 = (Wrapper){.has_priority=true, .serial=0,
                                .bundled_by=NULL_NODE, .packet_slot=s_p2};
    Wrapper init_c  = (Wrapper){.has_priority=false, .serial=0,
                                .bundled_by=NODE_PARENT1, .packet_slot=SLOT_NULL};

    atomic_store(&linkage[NODE_PARENT1], wrapper_pack(init_p1));
    atomic_store(&linkage[NODE_PARENT2], wrapper_pack(init_p2));
    atomic_store(&linkage[NODE_CHILD1],  wrapper_pack(init_c));
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&commit_count[i], 0);
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&priority_tag[i], TAG_NULL);
    /* insertedIn / everInsertedIn indexed by parent node; TRUE for both. */
    atomic_store(&inserted_in[NODE_PARENT1], true);
    atomic_store(&inserted_in[NODE_PARENT2], true);
    atomic_store(&ever_inserted_in[NODE_PARENT1], true);
    atomic_store(&ever_inserted_in[NODE_PARENT2], true);
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&insert_target_claim[i], 0u);
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&release_target_claim[i], 0u);
    atomic_store(&g_stop, false);

    /* Sanity: Init satisfies the safety invariants. */
    check_invariants();

    pthread_t threads[NUM_THREADS];
    ThreadCtx ctxs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        ctxs[i].tid    = (uint32_t)(i + 1);   /* 1-indexed; tid 0 reserved */
        ctxs[i].iter   = 0;
        ctxs[i].serial = 0;
        if (i < NUM_ROOT_ONLY)                       ctxs[i].role = ROLE_ROOT;
        else if (i < NUM_ROOT_ONLY + NUM_LEAF_ONLY)  ctxs[i].role = ROLE_LEAF;
        else                                         ctxs[i].role = ROLE_ROOT | ROLE_LEAF;
        if (i < NUM_INSERT_THREADS)  ctxs[i].role |= ROLE_INSERT;
        if (i < NUM_RELEASE_THREADS) ctxs[i].role |= ROLE_RELEASE;
        pthread_create(&threads[i], NULL, worker, &ctxs[i]);
    }

#if STRESS_SECONDS > 0
    struct timespec ts = { .tv_sec = STRESS_SECONDS, .tv_nsec = 0 };
    nanosleep(&ts, NULL);
    atomic_store_explicit(&g_stop, true, memory_order_release);
#endif

    for (int i = 0; i < NUM_THREADS; i++) pthread_join(threads[i], NULL);

    check_invariants();

    uint32_t cc = atomic_load(&commit_count[NODE_CHILD1]);

    /* TLA+ ChildPayload(DynChild1):
     *   priority child -> child.packet.payload
     *   bundled child  -> bundledBy parent's sub[c].payload (Null -> 0).
     * The child starts at payload 0 (Init); each successful commit adds 1
     * (mod MAX_PAYLOAD).  TerminalPayloadCheck: ChildPayload = commitCount. */
    Wrapper cw  = load_w(NODE_CHILD1);
    uint32_t child_payload;
    if (cw.has_priority) {
        child_payload = load_pkt_payload(cw.packet_slot);
    } else {
        int bp = cw.bundled_by;
        Wrapper bpw = load_w(bp);
        uint32_t bs; pkt_unpack(load_pkt_raw(bpw.packet_slot), NULL, &bs, NULL);
        child_payload = (bs != SLOT_NULL) ? load_pkt_payload(bs) : 0;
    }

    uint32_t expected = cc % MAX_PAYLOAD;   /* +0 init, +1 per commit */
    if ((uint32_t)child_payload != expected) {
        Wrapper p1 = load_w(NODE_PARENT1);
        Wrapper p2 = load_w(NODE_PARENT2);
        uint8_t pl1=0,pl2=0; uint32_t ss1=0,ss2=0; bool mm1=false,mm2=false;
        pkt_unpack(load_pkt_raw(p1.packet_slot), &pl1, &ss1, &mm1);
        pkt_unpack(load_pkt_raw(p2.packet_slot), &pl2, &ss2, &mm2);
        fprintf(stderr,
            "FAIL: MaxPayload=%d pool=%u\n"
            "  Parent1: prio=%d ser=%u slot=%u pkt=(pl=%u sub=%u m=%d)\n"
            "  Parent2: prio=%d ser=%u slot=%u pkt=(pl=%u sub=%u m=%d)\n"
            "  Child  : prio=%d bundled_by=%u ser=%u slot=%u payload=%u cc=%u (cc%%M=%u)\n",
            MAX_PAYLOAD, (unsigned)PACKET_POOL_ENTRIES,
            p1.has_priority, p1.serial, p1.packet_slot, pl1, ss1, mm1,
            p2.has_priority, p2.serial, p2.packet_slot, pl2, ss2, mm2,
            cw.has_priority, cw.bundled_by, cw.serial, cw.packet_slot,
            child_payload, cc, cc % MAX_PAYLOAD);
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
    printf("[hardlink-dynamic stress %s %ds pool=%u threads=%d] Child=%u commits\n",
           mode_str, STRESS_SECONDS, (unsigned)PACKET_POOL_ENTRIES, NUM_THREADS, cc);
    printf("  spin: commit_parent=%llu commit_child=%llu bundle=%llu migrate=%llu stale=%llu\n",
           (unsigned long long)atomic_load(&spin_commit_parent),
           (unsigned long long)atomic_load(&spin_commit_child),
           (unsigned long long)atomic_load(&spin_bundle),
           (unsigned long long)atomic_load(&spin_migrate),
           (unsigned long long)atomic_load(&spin_stale_read));
    printf("  llfree: negotiate_wait=%llu preempt=%llu\n",
           (unsigned long long)atomic_load(&spin_negotiate),
           (unsigned long long)atomic_load(&spin_preempt));
#else
    printf("[hardlink-dynamic unit threads=%d commits=%d] Child=%u commits OK\n",
           NUM_THREADS, (int)MAX_COMMITS, cc);
#endif

    return 0;
}
