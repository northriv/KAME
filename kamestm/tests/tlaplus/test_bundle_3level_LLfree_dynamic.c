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
 * C11 test generated mechanically from BundleUnbundle_3level_LLfree_dynamic.tla.
 * Dynamic-child variant of the 3-level LL-free model.
 *
 * Tree (dynamic): Grand --+-- Parent --+-- DynChild1  (inserted/released)
 *                                       +-- DynChild2  (inserted/released)
 *
 * Children start unattached. Insert/Release happen at Parent level only
 * (Grand is fixed). Combines 3L static recursive bundle with 2L dynamic
 * insert/release machinery.
 *
 *   Phase A — Insert (ROLE_INSERT): claim child via CAS, run insert
 *     protocol (snapshot Parent via try_inner_bundle, then CAS Parent
 *     serial-stamp → CAS Child to InsertedRef → CAS Parent installing
 *     new sub[c] with payload+1).
 *   Phase B — Commit (ROLE_ROOT does CommitGrand bumping ALL inserted
 *     leaves atomically; ROLE_LEAF does CommitChild per inserted child).
 *   Phase C — Release (ROLE_RELEASE): claim child, CAS Parent dropping
 *     sub[c], then restore Child wrapper to priority.
 *
 * Each child's final payload = 1 (insert) + commit_count[c]. Atomicity
 * modes / Option Z / diag counters carried over from 3L static verbatim.
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
#define MAX_PAYLOAD 256u  /* uint8 natural wrap (TLA+ MOD removed: spec uses unbounded Nat; C11 wraps at uint8) */
#endif

/* Mirrors TLA+ CONSTANT Privilege:
 *   1 (default) : LL-free priority_tag gating active (gate() blocks,
 *                 PreemptTag fires via can_proceed_with_preempt,
 *                 clear_my_tags releases at Tx-end).
 *   0           : tags neutralized (gate is a no-op, tag_after_fail
 *                 / clear_my_tags / tag_after_success do nothing,
 *                 PreemptTag dormant).  Equivalent to the pre-LLfree
 *                 algorithm — useful for sanity-checking against
 *                 test_bundle_3level.c semantics. */
#ifndef LLFREE_PRIVILEGE
#define LLFREE_PRIVILEGE 1
#endif

/* Mirrors TLA+ CONSTANTS RootThreads / LeafThreads (3-level model).
 *   ROLE_ROOT : thread does CommitGrand (snapshot Grand, +1 to ALL leaves).
 *   ROLE_LEAF : thread does CommitChild for each child (per-child direct +1).
 *   ROLE_BOTH : thread does both phases per iteration (default).
 * Each child receives MAX_COMMITS * (|root-threads| + |leaf-threads|)
 * increments total — matches the spec's TerminalPayloadCheck.
 *
 * Configuration via three counts (must sum to NUM_THREADS):
 *   NUM_ROOT_ONLY : threads in RootThreads only (CommitGrand only)
 *   NUM_LEAF_ONLY : threads in LeafThreads only (CommitChild only)
 *   NUM_BOTH      : threads in both sets (defaults to NUM_THREADS - others)
 * Assignment: tids 1..NUM_ROOT_ONLY = ROOT_ONLY,
 *             tids NUM_ROOT_ONLY+1..NUM_ROOT_ONLY+NUM_LEAF_ONLY = LEAF_ONLY,
 *             remaining tids = BOTH. */
#define ROLE_ROOT     0x1u
#define ROLE_LEAF     0x2u
#define ROLE_INSERT   0x4u
#define ROLE_RELEASE  0x8u
#define ROLE_BOTH     (ROLE_ROOT | ROLE_LEAF)

#ifndef NUM_ROOT_ONLY
#define NUM_ROOT_ONLY 0
#endif
#ifndef NUM_LEAF_ONLY
#define NUM_LEAF_ONLY 0
#endif
#ifndef NUM_INSERT_THREADS
#define NUM_INSERT_THREADS NUM_THREADS   /* TLA+ ASSUME InsertThreads /= {} */
#endif
#ifndef NUM_RELEASE_THREADS
#define NUM_RELEASE_THREADS 0
#endif
#if (NUM_ROOT_ONLY + NUM_LEAF_ONLY) > NUM_THREADS
#  error "NUM_ROOT_ONLY + NUM_LEAF_ONLY exceeds NUM_THREADS"
#endif
#if NUM_INSERT_THREADS < 1
#  error "TLA+ ASSUME violation: InsertThreads must be non-empty"
#endif
#if NUM_INSERT_THREADS > NUM_THREADS || NUM_RELEASE_THREADS > NUM_THREADS
#  error "Insert/Release thread count exceeds NUM_THREADS"
#endif
#define NUM_BOTH         (NUM_THREADS - NUM_ROOT_ONLY - NUM_LEAF_ONLY)
#define NUM_ROOT_THREADS (NUM_ROOT_ONLY + NUM_BOTH)
#define NUM_LEAF_THREADS (NUM_LEAF_ONLY + NUM_BOTH)
#if (NUM_ROOT_THREADS + NUM_LEAF_THREADS) < NUM_THREADS
#  error "TLA+ ASSUME violation: RootThreads ∪ LeafThreads must equal Threads"
#endif

/* Packet pool: ring-buffer of packets (immutable once written).
 * Must be large enough that a slot cannot be recycled during any
 * transaction's dereference window — otherwise a wrapper field could end up
 * pointing at a pool slot whose content has been overwritten, yielding
 * silent corruption.  With 27-bit slot, max pool is 134217727 (~1GB as
 * uint64 — generous headroom for many-thread Ohtaka-scale runs).  Default
 * 65536 stays L2-friendly for small runs; raise via -DPACKET_POOL_ENTRIES
 * for hot multi-thread workloads. */
/* Safe default: max 27-bit pool (widest wrap margin).  Reduce via
 * -DPACKET_POOL_ENTRIES for cache locality tuning. */
#ifndef PACKET_POOL_ENTRIES
#define PACKET_POOL_ENTRIES 134217727u
#endif
#if (PACKET_POOL_ENTRIES) > 134217727u
#  error "PACKET_POOL_ENTRIES must fit in 27 bits (<=134217727)"
#endif

/* --- Node IDs --- */
#define NODE_GRAND   0
#define NODE_PARENT  1
#define NODE_CHILD1  2
#define NODE_CHILD2  3
#define NUM_NODES    4

/* bundled_by is a 2-bit field: 0=GRAND, 1=PARENT, 3=NULL_NODE (priority).
 * NODE_GRAND/NODE_PARENT happen to fit in those 2 bits already. */
#define NULL_NODE   0x3u

/* --- Serial + slot widths --- */
#define SER_BITS   27u
#define SER_MOD    (1u << SER_BITS)
#define SER_MASK   (SER_MOD - 1u)
#define SLOT_BITS  27u
#define SLOT_MASK  ((1u << SLOT_BITS) - 1u)
#define SLOT_NULL  0u    /* slot 0 reserved as "no packet" */

/* --- Modular serial comparison (TLA+ ModGT) --- */
static inline bool ser_gt(uint32_t a, uint32_t b) {
    uint32_t diff = (a - b) & SER_MASK;
    return diff > 0 && diff < (SER_MOD >> 1);
}

/* TID-encoded base-B Lamport serial — mirrors C++
 * SerialGenerator::gen() (transaction.h:547-576) which packs a TLS
 * counter in the upper bits + TID in the lower bits of a 64-bit
 * value.  Same counter on two threads produces DIFFERENT serial
 * values because the lower bits differ — this gives wrapper
 * thread-uniqueness without any global atomic.
 *
 * TLA+ encoding: serial = counter * SerialBase + tid, with
 *   SerialBase = 1 + |Threads|   \>  max TID
 * GenSerial(t, lastSer):
 *   newCnt = max(SerialCounter(lastSer), SerialCounter(serial[t])) + 1
 *   return EncodeSerial(newCnt, t)
 * No globalSerial — purely TLS + linkage-serial Lamport. */
#define SERIAL_BASE  ((uint32_t)(NUM_THREADS + 1))   /* > max tid (tids ∈ 1..NUM_THREADS) */

static inline uint32_t serial_counter(uint32_t s) { return s / SERIAL_BASE; }
__attribute__((unused))
static inline uint32_t serial_tid(uint32_t s)     { return s % SERIAL_BASE; }
static inline uint32_t encode_serial(uint32_t cnt, uint32_t tid) {
    return ((cnt * SERIAL_BASE) + tid) & SER_MASK;
}

/* gen_serial: TID-encoded base-B Lamport step.
 *   thread_ser : caller's TLS serial slot (ctx->serial)
 *   last_ser   : witnessed wrapper serial (e.g. linkage[node].serial)
 *   my_tid     : caller's tid (1-indexed, < SERIAL_BASE)
 * Same counter on different threads → different result (low-bits TID).
 * Counter range: SER_MASK / SerialBase (e.g. ~44M for 2 threads on 27-bit). */
static inline uint32_t gen_serial(uint32_t thread_ser, uint32_t last_ser, uint32_t my_tid) {
    uint32_t last_cnt = serial_counter(last_ser);
    uint32_t my_cnt   = serial_counter(thread_ser);
    /* Modular max on counters (mirrors TLA+ "IF lastCnt > myCnt"; ser_gt
     * still applies on counter range — compare mod 2^27 / SerialBase). */
    uint32_t base_cnt = ser_gt(last_cnt, my_cnt) ? last_cnt : my_cnt;
    uint32_t new_cnt  = base_cnt + 1u;
    return encode_serial(new_cnt, my_tid);
}

/* ============================================================================
 * Packet pool.
 *   packet bits: payload(8) | sub_slot[0](14) | sub_slot[1](14) | missing(1)
 * ============================================================================ */
static _Atomic(uint64_t) packet_pool[PACKET_POOL_ENTRIES + 1];
static _Atomic(uint32_t) global_slot_counter;   /* monotonic allocator */

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
static inline uint8_t load_pkt_payload(uint32_t slot) {
    uint8_t p; pkt_unpack(load_pkt_raw(slot), &p, NULL, NULL, NULL);
    return p;
}

/* ============================================================================
 * Wrapper:
 *   serial(14) | has_priority(1) | bundled_by(2) | packet_slot(14) | pad
 * No wrapper-level missing — canonical missing lives in the Packet (TLA+).
 * ============================================================================ */
typedef struct {
    bool     has_priority;
    uint32_t serial;
    uint8_t  bundled_by;    /* NULL_NODE iff has_priority */
    uint32_t packet_slot;   /* SLOT_NULL iff !has_priority (bundled-ref) */
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
static _Atomic(uint32_t) commit_count[NUM_NODES];
static _Atomic(bool)     g_stop;

/* Dynamic-child state (mirrors 2L dynamic). */
static _Atomic(bool)     inserted[NUM_NODES];
static _Atomic(bool)     ever_inserted[NUM_NODES];
static _Atomic(uint32_t) insert_target_claim[NUM_NODES];
static _Atomic(uint32_t) release_target_claim[NUM_NODES];

/* --- Spin / race-detection counters (stress-mode diagnostics) --- */
static _Atomic(uint64_t) spin_inner_bundle;      /* try_inner_bundle bailouts */
static _Atomic(uint64_t) spin_outer_bundle;      /* try_outer_bundle bailouts */
static _Atomic(uint64_t) spin_commit_grand;      /* commit_grand CAS failures  */
static _Atomic(uint64_t) spin_commit_child;      /* commit_child CAS failures  */
static _Atomic(uint64_t) spin_stale_read;        /* wrapper-changed between
                                                    read and CAS (pre-CAS
                                                    verification detection) */
static _Atomic(uint64_t) spin_negotiate;         /* can_proceed gated waits */
static _Atomic(uint64_t) spin_preempt;           /* tag preemptions issued  */
#define SPIN_INC(name) atomic_fetch_add_explicit(&(name), 1u, memory_order_relaxed)

/* ============================================================================
 * LL-free negotiate machinery (mirrors test_bundle_2level_LLfree.c).
 *   Tag = uint64_t. Bit 63 = "valid" flag.  Bits 32-62 = iter (31-bit).
 *   Bits 0-31 = tid.  Tag value 0 == Null (no holder).
 *
 *   When LLFREE_PRIVILEGE == 0, gate / tag_after_fail / clear_my_tags
 *   etc. become no-ops (matches TLA+ Privilege=FALSE).  The priority_tag
 *   array is still allocated (cheap) but never read or written.
 * ============================================================================ */
typedef uint64_t Tag;
#define TAG_NULL      ((Tag)0)
#define TAG_VALID     (((Tag)1) << 63)
#define TAG_ITER_MASK 0x7FFFFFFFu

__attribute__((unused))
static inline Tag      make_tag(uint32_t iter, uint32_t tid) {
    return TAG_VALID | ((Tag)(iter & TAG_ITER_MASK) << 32) | (Tag)tid;
}
__attribute__((unused))
static inline bool     tag_is_null(Tag t) { return t == TAG_NULL; }
__attribute__((unused))
static inline uint32_t tag_iter(Tag t)    { return (uint32_t)((t >> 32) & TAG_ITER_MASK); }
__attribute__((unused))
static inline uint32_t tag_tid(Tag t)     { return (uint32_t)(t & 0xFFFFFFFFu); }
__attribute__((unused))
static inline bool     tag_older(Tag a, Tag b) {
    if (tag_iter(a) != tag_iter(b)) return tag_iter(a) < tag_iter(b);
    return tag_tid(a) < tag_tid(b);
}

static _Atomic(Tag)  priority_tag[NUM_NODES];

/* TLA+ Privilege simplification (no zombie / inactive-thread check):
 * tags are released ONLY on commit success (ClearMyTags in CommitDone),
 * so a thread can never reach an "inactive" state with stale tags
 * outliving its lifetime — the spec's invariant.  Removed
 * `thread_active[]` tracking entirely. */

#if LLFREE_PRIVILEGE
/* CanProceed + PreemptTag fused: returns true if (tag null) OR (tag mine)
 * OR (we successfully preempted an older active holder).  TLA+ literal
 * (no zombie branch). */
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

/* tag_after_fail: set my tag at n on CAS failure (TLA+ TagAfterFail).
 * No zombie branch: null/mine/older → mine; younger → preserve. */
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

/* tag_after_success: NO-OP (Tx-scope persistence; clear at Tx-success only). */
static inline void tag_after_success(int n, uint32_t my_tid) { (void)n; (void)my_tid; }

/* clear_my_tags: TLA+ ClearMyTags(t).  Called ONLY on commit success
 * (CommitDone success / CommitGrand success).  On CAS failure, tags are
 * preserved across retries — matches C++ operator++() which does not
 * call drop_tags_n_privilege(). */
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

#else  /* LLFREE_PRIVILEGE == 0 : tag machinery neutralized */

static inline void tag_after_fail(int n, uint32_t my_iter, uint32_t my_tid) {
    (void)n; (void)my_iter; (void)my_tid;
}
static inline void tag_after_success(int n, uint32_t my_tid) { (void)n; (void)my_tid; }
static inline void clear_my_tags(uint32_t my_tid) { (void)my_tid; }
static inline void gate(int n, uint32_t my_iter, uint32_t my_tid) {
    (void)n; (void)my_iter; (void)my_tid;
}

#endif /* LLFREE_PRIVILEGE */

typedef struct {
    uint32_t tid;          /* 1-indexed */
    uint32_t iter;         /* completed iterations */
    uint32_t serial;       /* Lamport thread-local clock */
    uint32_t role;         /* ROLE_ROOT | ROLE_LEAF | ROLE_INSERT | ROLE_RELEASE */
} ThreadCtx;

/* Pool-wrap staleness detector (Option Z). */
#define POOL_SAFETY_MARGIN 64u
static inline uint32_t cur_slot_counter(void) {
    return atomic_load_explicit(&global_slot_counter, memory_order_relaxed);
}
static inline bool pool_stale(uint32_t start_counter) {
    uint32_t advanced = (uint32_t)(cur_slot_counter() - start_counter);
    if (advanced >= PACKET_POOL_ENTRIES - POOL_SAFETY_MARGIN) {
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

/* --- Wrapper builders (each allocates a fresh pool slot unless noted) --- */

/* Leaf: no sub-packets. */
static inline Wrapper make_priority_leaf(uint8_t payload, uint32_t serial, bool pkt_missing) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .packet_slot  = alloc_slot(payload, SLOT_NULL, SLOT_NULL, pkt_missing),
    };
}

/* Inner node with explicit packet content. */
static inline Wrapper make_priority_inner(uint8_t payload, uint32_t serial,
                                          uint32_t s0, uint32_t s1, bool pkt_missing) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .packet_slot  = alloc_slot(payload, s0, s1, pkt_missing),
    };
}

/* Reuse an existing packet slot with a new serial (UnbundleRestoreParent).
 * No fresh allocation: preserves the stored packet identity. */
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
 * Bundle: inner level (Parent missing -> bundle Child1, Child2 under Parent)
 * ========================================================================= */
static bool try_inner_bundle(ThreadCtx *ctx, Wrapper *pw_out) {
    Wrapper pw = load_w(NODE_PARENT);
    if (!pw.has_priority) { SPIN_INC(spin_inner_bundle); return false; }

    uint8_t  pp;
    uint32_t old_s0, old_s1;
    bool     pw_missing;
    pkt_unpack(load_pkt_raw(pw.packet_slot), &pp, &old_s0, &old_s1, &pw_missing);

    if (!pw_missing) { *pw_out = pw; return true; }

    uint32_t bundle_ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
    ctx->serial = bundle_ser;

#if MODE == MODE_SUPERFINE
    if (pw.serial != bundle_ser) {
        gate(NODE_PARENT, ctx->iter, ctx->tid);
        Wrapper prestamp = make_priority_from_slot(pw.packet_slot, bundle_ser);
        Wrapper exp = pw;
        if (!cas_w(NODE_PARENT, &exp, prestamp)) {
            tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
            SPIN_INC(spin_inner_bundle); return false;
        }
        tag_after_success(NODE_PARENT, ctx->tid);
        pw = prestamp;
    }
#endif

    /* Dynamic: only collect for ActiveChildren (inserted[c]==true). */
    bool c1_active = atomic_load_explicit(&inserted[NODE_CHILD1], memory_order_acquire);
    bool c2_active = atomic_load_explicit(&inserted[NODE_CHILD2], memory_order_acquire);
    Wrapper cw1 = load_w(NODE_CHILD1);
    Wrapper cw2 = load_w(NODE_CHILD2);
    uint32_t sp1 = SLOT_NULL, sp2 = SLOT_NULL;
    if (c1_active) {
        if (cw1.has_priority)                                           sp1 = cw1.packet_slot;
        else if (cw1.bundled_by == NODE_PARENT && old_s0 != SLOT_NULL)  sp1 = old_s0;
    }
    if (c2_active) {
        if (cw2.has_priority)                                           sp2 = cw2.packet_slot;
        else if (cw2.bundled_by == NODE_PARENT && old_s1 != SLOT_NULL)  sp2 = old_s1;
    }
    if ((c1_active && sp1 == SLOT_NULL) || (c2_active && sp2 == SLOT_NULL)) {
        tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
        SPIN_INC(spin_inner_bundle); return false;
    }

    /* Already-bundled-here guard: reused sub-packet must not itself be missing. */
    if (c1_active && cw1.bundled_by == NODE_PARENT) {
        bool m; pkt_unpack(load_pkt_raw(sp1), NULL, NULL, NULL, &m);
        if (m) {
            tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
            SPIN_INC(spin_inner_bundle); return false;
        }
    }
    if (c2_active && cw2.bundled_by == NODE_PARENT) {
        bool m; pkt_unpack(load_pkt_raw(sp2), NULL, NULL, NULL, &m);
        if (m) {
            tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
            SPIN_INC(spin_inner_bundle); return false;
        }
    }

    /* InnerPhase2: Parent packet = (payload, sp1, sp2, missing=true). */
    gate(NODE_PARENT, ctx->iter, ctx->tid);
    Wrapper p2 = make_priority_inner(pp, bundle_ser, sp1, sp2, true);
    Wrapper exp_p2 = pw;
    if (!cas_w(NODE_PARENT, &exp_p2, p2)) {
        tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
        SPIN_INC(spin_inner_bundle); return false;
    }
    tag_after_success(NODE_PARENT, ctx->tid);

    /* InnerPhase3: Children -> bundled-by-Parent.  TLA+ InnerPhase3
     * fail (line 578-579) tags BOTH the failed grandchild and the
     * inner-bundling node `c` (= Parent here) UNCONDITIONALLY,
     * regardless of outer atomicity mode.  Symmetric with outer
     * BundlePhase3 SUPERFINE DISTURBED.  When LLFREE_PRIVILEGE=0 the
     * tag_after_fail calls are no-ops anyway. */
    Wrapper b = make_bundled(NODE_PARENT, bundle_ser);
    if (c1_active) {
        gate(NODE_CHILD1, ctx->iter, ctx->tid);
        Wrapper exp_c1 = cw1;
        if (!cas_w(NODE_CHILD1, &exp_c1, b)) {
            tag_after_fail(NODE_CHILD1, ctx->iter, ctx->tid);
            tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
            SPIN_INC(spin_inner_bundle); return false;
        }
        tag_after_success(NODE_CHILD1, ctx->tid);
    }
    if (c2_active) {
        gate(NODE_CHILD2, ctx->iter, ctx->tid);
        Wrapper exp_c2 = cw2;
        if (!cas_w(NODE_CHILD2, &exp_c2, b)) {
            tag_after_fail(NODE_CHILD2, ctx->iter, ctx->tid);
            tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
            SPIN_INC(spin_inner_bundle); return false;
        }
        tag_after_success(NODE_CHILD2, ctx->tid);
    }

    /* InnerPhase4: flip missing=false on Parent's packet (fresh slot). */
    gate(NODE_PARENT, ctx->iter, ctx->tid);
    Wrapper p4 = make_priority_inner(pp, bundle_ser, sp1, sp2, false);
    Wrapper exp_p4 = p2;
    if (!cas_w(NODE_PARENT, &exp_p4, p4)) {
        tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
        SPIN_INC(spin_inner_bundle); return false;
    }
    tag_after_success(NODE_PARENT, ctx->tid);

    *pw_out = p4;
    return true;
}

/* =========================================================================
 * Bundle: outer level (Grand missing -> bundle Parent under Grand).
 * ========================================================================= */
static bool try_outer_bundle(ThreadCtx *ctx, Wrapper *gw_out) {
    Wrapper gw = load_w(NODE_GRAND);
    if (!gw.has_priority) { SPIN_INC(spin_outer_bundle); return false; }

    uint8_t  gp;
    uint32_t g_s0, g_s1;
    bool     gw_missing;
    pkt_unpack(load_pkt_raw(gw.packet_slot), &gp, &g_s0, &g_s1, &gw_missing);

    if (!gw_missing) { *gw_out = gw; return true; }

    Wrapper pw = load_w(NODE_PARENT);
    uint32_t parent_pkt_slot = SLOT_NULL;

    /* Outer collect: also tag NODE_GRAND (= bundleNode) eagerly on
     * any collection failure so peers can't race in during the next
     * snap restart.  Mirrors TLA+ BundlePhase1 fine collect-fail
     * eager-tag (port from 2-level commit 5ff3226). */
    if (pw.has_priority) {
        Wrapper pw_fresh = pw;
        bool pw_pkt_missing;
        pkt_unpack(load_pkt_raw(pw.packet_slot), NULL, NULL, NULL, &pw_pkt_missing);
        if (pw_pkt_missing) {
            if (!try_inner_bundle(ctx, &pw_fresh)) {
                tag_after_fail(NODE_GRAND, ctx->iter, ctx->tid);
                SPIN_INC(spin_outer_bundle); return false;
            }
        }
        parent_pkt_slot = pw_fresh.packet_slot;
        pw = pw_fresh;
    } else if (pw.bundled_by == NODE_GRAND) {
        if (g_s0 == SLOT_NULL) {
            tag_after_fail(NODE_GRAND, ctx->iter, ctx->tid);
            SPIN_INC(spin_outer_bundle); return false;
        }
        bool stored_parent_missing;
        pkt_unpack(load_pkt_raw(g_s0), NULL, NULL, NULL, &stored_parent_missing);
        if (stored_parent_missing) {
            tag_after_fail(NODE_GRAND, ctx->iter, ctx->tid);
            SPIN_INC(spin_outer_bundle); return false;
        }
        parent_pkt_slot = g_s0;
    } else {
        tag_after_fail(NODE_GRAND, ctx->iter, ctx->tid);
        SPIN_INC(spin_outer_bundle); return false;
    }

    uint32_t bundle_ser = gen_serial(ctx->serial, gw.serial, ctx->tid);
    ctx->serial = bundle_ser;

#if MODE == MODE_SUPERFINE
    if (gw.serial != bundle_ser) {
        gate(NODE_GRAND, ctx->iter, ctx->tid);
        Wrapper prestamp = make_priority_from_slot(gw.packet_slot, bundle_ser);
        Wrapper exp = gw;
        if (!cas_w(NODE_GRAND, &exp, prestamp)) {
            tag_after_fail(NODE_GRAND, ctx->iter, ctx->tid);
            SPIN_INC(spin_outer_bundle); return false;
        }
        tag_after_success(NODE_GRAND, ctx->tid);
        gw = prestamp;
    }
#endif

    /* Phase2: Grand packet = (gp, parent_pkt_slot, NULL, missing=true). */
    gate(NODE_GRAND, ctx->iter, ctx->tid);
    Wrapper g2 = make_priority_inner(gp, bundle_ser, parent_pkt_slot, SLOT_NULL, true);
    Wrapper exp_g2 = gw;
    if (!cas_w(NODE_GRAND, &exp_g2, g2)) {
        tag_after_fail(NODE_GRAND, ctx->iter, ctx->tid);
        SPIN_INC(spin_outer_bundle); return false;
    }
    tag_after_success(NODE_GRAND, ctx->tid);

    /* Phase3: Parent -> bundled-by-Grand.  TLA+ port: ALWAYS CAS Parent
     * regardless of pw.hasPriority — when pw was already bundled-by-Grand,
     * the CAS REFRESHES bundle_ser, invalidating any peer's stale
     * snapshotForUnbundle pointer that still references the old wrapper.
     * Without the refresh, peer's UnbundleCASChild value-CAS may succeed
     * against the structurally identical (old-ser) wrapper → lost
     * increment.  Mirrors C++ which always emplaces a new bundled_ref
     * wrapper per inner-bundle iteration (transaction_impl.h:2487-2511). */
    gate(NODE_PARENT, ctx->iter, ctx->tid);
    Wrapper bref = make_bundled(NODE_GRAND, bundle_ser);
    Wrapper exp_pw = pw;
    if (!cas_w(NODE_PARENT, &exp_pw, bref)) {
        tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
#if MODE == MODE_SUPERFINE
        tag_after_fail(NODE_GRAND, ctx->iter, ctx->tid);
#endif
        SPIN_INC(spin_outer_bundle); return false;
    }
    tag_after_success(NODE_PARENT, ctx->tid);

    /* Phase4: Grand.missing=false. */
    gate(NODE_GRAND, ctx->iter, ctx->tid);
    Wrapper g4 = make_priority_inner(gp, bundle_ser, parent_pkt_slot, SLOT_NULL, false);
    Wrapper exp_g4 = g2;
    if (!cas_w(NODE_GRAND, &exp_g4, g4)) {
        tag_after_fail(NODE_GRAND, ctx->iter, ctx->tid);
        SPIN_INC(spin_outer_bundle); return false;
    }
    tag_after_success(NODE_GRAND, ctx->tid);

    *gw_out = g4;
    return true;
}

/* snapshot_grand(): loop until Grand is priority, packet non-missing, and the
 * stored Parent slot is present.  Returns the Grand packet contents alongside
 * the wrapper so the caller does not need to re-read the pool slot (which
 * could have been recycled). */
static void snapshot_grand(ThreadCtx *ctx, Wrapper *gw_out,
                           uint8_t *gp_out, uint32_t *g_s0_out, uint32_t *g_s1_out) {
    for (;;) {
        Wrapper gw = load_w(NODE_GRAND);
        if (!gw.has_priority) continue;
        uint8_t gp; uint32_t gs0, gs1; bool m;
        pkt_unpack(load_pkt_raw(gw.packet_slot), &gp, &gs0, &gs1, &m);
        if (!m && gs0 != SLOT_NULL) {
            *gw_out = gw; *gp_out = gp; *g_s0_out = gs0; *g_s1_out = gs1;
            return;
        }
        Wrapper gb;
        if (try_outer_bundle(ctx, &gb)) {
            uint8_t gp2; uint32_t gs02, gs12; bool m2;
            pkt_unpack(load_pkt_raw(gb.packet_slot), &gp2, &gs02, &gs12, &m2);
            if (!m2 && gs02 != SLOT_NULL) {
                *gw_out = gb; *gp_out = gp2; *g_s0_out = gs02; *g_s1_out = gs12;
                return;
            }
        }
    }
}

/* =========================================================================
 * CommitGrand: force full recursive bundle, then bump BOTH leaf payloads by
 * allocating fresh Child packets, a fresh Parent packet pointing to them, and
 * a fresh Grand packet pointing to that Parent. One top-level CAS on Grand
 * installs the new chain atomically.
 * ========================================================================= */
static void commit_grand(ThreadCtx *ctx) {
    OP_LOCK();
    for (;;) {
        uint32_t start_counter = cur_slot_counter();
        Wrapper gw;
        uint8_t  gp;
        uint32_t parent_slot, unused_s1;
        snapshot_grand(ctx, &gw, &gp, &parent_slot, &unused_s1);
        (void)unused_s1;
        if (parent_slot == SLOT_NULL) { SPIN_INC(spin_commit_grand); continue; }

        uint8_t  pp; uint32_t c1_slot, c2_slot; bool pm;
        pkt_unpack(load_pkt_raw(parent_slot), &pp, &c1_slot, &c2_slot, &pm);
        if (pm) { SPIN_INC(spin_commit_grand); continue; }

        /* TLA+ dynamic: discover children from snapshot
         *   snapChildren = {c : snapPkt.sub[c] != Null} */
        bool c1_in = (c1_slot != SLOT_NULL);
        bool c2_in = (c2_slot != SLOT_NULL);
        if (!c1_in && !c2_in) {
            /* All released since snapshot; skip iteration cleanly. */
            clear_my_tags(ctx->tid);
            OP_UNLOCK();
            return;
        }

        uint8_t c1_payload = c1_in ? load_pkt_payload(c1_slot) : 0;
        uint8_t c2_payload = c2_in ? load_pkt_payload(c2_slot) : 0;

        if (pool_stale(start_counter)) continue;

        uint32_t new_c1 = c1_in
            ? alloc_slot((uint8_t)(c1_payload + 1u), SLOT_NULL, SLOT_NULL, false)
            : SLOT_NULL;
        uint32_t new_c2 = c2_in
            ? alloc_slot((uint8_t)(c2_payload + 1u), SLOT_NULL, SLOT_NULL, false)
            : SLOT_NULL;
        uint32_t new_p  = alloc_slot(pp, new_c1, new_c2, false);

        uint32_t ser = gen_serial(ctx->serial, gw.serial, ctx->tid);
        Wrapper new_gw = make_priority_inner(gp, ser, new_p, SLOT_NULL, false);

        gate(NODE_GRAND, ctx->iter, ctx->tid);
        Wrapper exp = gw;
        if (cas_w(NODE_GRAND, &exp, new_gw)) {
            ctx->serial = ser;
            if (c1_in)
                atomic_fetch_add_explicit(&commit_count[NODE_CHILD1], 1, memory_order_relaxed);
            if (c2_in)
                atomic_fetch_add_explicit(&commit_count[NODE_CHILD2], 1, memory_order_relaxed);
            clear_my_tags(ctx->tid);
            OP_UNLOCK();
            return;
        }
        tag_after_fail(NODE_GRAND, ctx->iter, ctx->tid);
        SPIN_INC(spin_commit_grand);
    }
}

/* =========================================================================
 * CommitChild: direct commit on a leaf; unbundle walk 1 or 2 levels deep.
 * ========================================================================= */
static void commit_child(int child_node, ThreadCtx *ctx) {
    OP_LOCK();
    for (;;) {
        /* TLA+ CommitSkip: child released since queue setup. */
        if (!atomic_load_explicit(&inserted[child_node], memory_order_acquire)) {
            clear_my_tags(ctx->tid);
            OP_UNLOCK();
            return;
        }
        uint32_t start_counter = cur_slot_counter();
        Wrapper cw = load_w(child_node);

        /* --- Direct commit on priority leaf --- */
        if (cw.has_priority) {
            uint8_t old_payload = load_pkt_payload(cw.packet_slot);
            if (pool_stale(start_counter)) continue;
            uint8_t new_payload = (uint8_t)(old_payload + 1u);
            uint32_t ser = gen_serial(ctx->serial, cw.serial, ctx->tid);
            Wrapper new_cw = make_priority_leaf(new_payload, ser, false);
            gate(child_node, ctx->iter, ctx->tid);
            Wrapper exp = cw;
            if (cas_w(child_node, &exp, new_cw)) {
                ctx->serial = ser;
                atomic_fetch_add_explicit(&commit_count[child_node], 1, memory_order_relaxed);
                /* Tx-end (CommitDone success). */
                clear_my_tags(ctx->tid);
                OP_UNLOCK();
                return;
            }
            tag_after_fail(child_node, ctx->iter, ctx->tid);
            SPIN_INC(spin_commit_child);
            continue;
        }

        if (cw.bundled_by != NODE_PARENT) { SPIN_INC(spin_commit_child); continue; }

        Wrapper pw = load_w(NODE_PARENT);

        /* --- 1-level unbundle --- */
        if (pw.has_priority) {
            uint8_t  pp; uint32_t p_s0, p_s1; bool pm;
            pkt_unpack(load_pkt_raw(pw.packet_slot), &pp, &p_s0, &p_s1, &pm);
            uint32_t old_child_slot = (child_node == NODE_CHILD1) ? p_s0 : p_s1;
            if (old_child_slot == SLOT_NULL) { SPIN_INC(spin_commit_child); continue; }
            uint8_t old_payload = load_pkt_payload(old_child_slot);
            if (pool_stale(start_counter)) continue;
            uint8_t new_payload = (uint8_t)(old_payload + 1u);

            /* UnbundleCASLoop (1-step at Parent): missing=true, sub preserved
             * (old child slots).  TLA+ literal — do NOT advance sub[child]
             * here; that happens atomically with the child CAS below
             * (UnbundleCASChild Parent sync). */
            uint32_t anc_ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
            Wrapper new_pw = make_priority_inner(pp, anc_ser, p_s0, p_s1, true);
            gate(NODE_PARENT, ctx->iter, ctx->tid);
            Wrapper exp_pw = pw;
            if (!cas_w(NODE_PARENT, &exp_pw, new_pw)) {
                tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
                SPIN_INC(spin_commit_child); continue;
            }
            tag_after_success(NODE_PARENT, ctx->tid);
            ctx->serial = anc_ser;

            /* UnbundleCASChild: TLA+ Fix 3 — atomically update child
             * wrapper AND immediate parent's packet.sub[child].  C11
             * cannot do a true atomic dual-CAS, so we sequentialize:
             *   (1) CAS Child wrapper (this is the spec's
             *       `linkage[node] = oldChildW` gate → newChildW).
             *   (2) On success, CAS Parent wrapper to a new wrapper
             *       whose packet.sub[child] points at the new child
             *       packet (the "parentSync" branch of the spec).
             *   The Parent CAS uses our just-installed new_pw as
             *   expected (we still hold its identity).  If it fails
             *   (peer touched Parent), we leave Parent stale and
             *   accept the residual lost-increment risk that the
             *   spec's atomic version closes — best the C11 can do
             *   without DCAS.  cc++ is gated only on the child CAS,
             *   matching TLA+ commitOk / iterBudget--. */
            uint32_t c_ser = gen_serial(ctx->serial, cw.serial, ctx->tid);
            uint32_t new_child_slot = alloc_slot(new_payload, SLOT_NULL, SLOT_NULL, false);
            Wrapper new_child = make_priority_from_slot(new_child_slot, c_ser);
            gate(child_node, ctx->iter, ctx->tid);
            Wrapper exp_child = cw;
            if (cas_w(child_node, &exp_child, new_child)) {
                /* Parent sync: best-effort CAS Parent to advance sub[child]. */
                uint32_t sync_p_s0 = (child_node == NODE_CHILD1) ? new_child_slot : p_s0;
                uint32_t sync_p_s1 = (child_node == NODE_CHILD2) ? new_child_slot : p_s1;
                Wrapper sync_pw = make_priority_inner(pp, c_ser, sync_p_s0, sync_p_s1, true);
                Wrapper exp_sync = new_pw;
                (void)cas_w(NODE_PARENT, &exp_sync, sync_pw);
                ctx->serial = c_ser;
                atomic_fetch_add_explicit(&commit_count[child_node], 1, memory_order_relaxed);
                clear_my_tags(ctx->tid);
                OP_UNLOCK();
                return;
            }
            tag_after_fail(child_node, ctx->iter, ctx->tid);
            SPIN_INC(spin_commit_child);
            continue;
        }

        if (pw.bundled_by != NODE_GRAND) { SPIN_INC(spin_commit_child); continue; }

        /* --- 2-level unbundle --- */
        Wrapper gw = load_w(NODE_GRAND);
        if (!gw.has_priority) { SPIN_INC(spin_commit_child); continue; }
        uint8_t  gp; uint32_t g_s0, g_s1; bool gm;
        pkt_unpack(load_pkt_raw(gw.packet_slot), &gp, &g_s0, &g_s1, &gm);
        if (g_s0 == SLOT_NULL) { SPIN_INC(spin_commit_child); continue; }

        uint8_t  saved_pp; uint32_t saved_p_s0, saved_p_s1; bool saved_pm;
        pkt_unpack(load_pkt_raw(g_s0),
                   &saved_pp, &saved_p_s0, &saved_p_s1, &saved_pm);

        uint32_t saved_child_slot = (child_node == NODE_CHILD1) ? saved_p_s0 : saved_p_s1;
        if (saved_child_slot == SLOT_NULL) { SPIN_INC(spin_commit_child); continue; }
        uint8_t saved_child_payload = load_pkt_payload(saved_child_slot);
        if (pool_stale(start_counter)) continue;
        uint8_t new_payload = (uint8_t)(saved_child_payload + 1u);

        /* UnbundleCASLoop step at Grand (root-first, casTargets[1]=Grand).
         * TLA+ literal:
         *   extracted = walkWrapper.packet  (= Grand's old packet content)
         *   newPkt    = MakePacket(Grand, extracted.payload,
         *                          extracted.sub, TRUE)
         * extracted.sub[Parent] is the OLD g_s0 slot — preserved verbatim,
         * not re-allocated.  Grand becomes priority+missing=TRUE; peers
         * reading Grand re-collect via the bundle path. */
        uint32_t gp_ser = gen_serial(ctx->serial, gw.serial, ctx->tid);
        Wrapper new_gw = make_priority_inner(gp, gp_ser, g_s0, g_s1, true);
        gate(NODE_GRAND, ctx->iter, ctx->tid);
        Wrapper exp_gw = gw;
        if (!cas_w(NODE_GRAND, &exp_gw, new_gw)) {
            tag_after_fail(NODE_GRAND, ctx->iter, ctx->tid);
            SPIN_INC(spin_commit_child); continue;
        }
        tag_after_success(NODE_GRAND, ctx->tid);
        ctx->serial = gp_ser;
        /* TLA+ walkWrapper update: after CAS of root casNode == superNode,
         * walkWrapper := newW.  In C11 we use new_gw directly as the
         * super-witness for the next iter's superFresh check below. */

        /* superFresh check (TLA+ UnbundleCASLoop fine):
         *   superFresh == linkage[superNode] = walkWrapper
         * Verify Grand still matches our just-installed new_gw before the
         * next iter (Parent CAS).  If a peer touched Grand → DISTURBED →
         * tag_after_fail(Parent) + restart from commit_read. */
        Wrapper g_now = load_w(NODE_GRAND);
        if (wrapper_pack(g_now) != wrapper_pack(new_gw)) {
            tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
            SPIN_INC(spin_commit_child); continue;
        }

        /* UnbundleCASLoop step at Parent (casTargets[2]=Parent).
         * TLA+ literal:
         *   extracted = walkWrapper.packet.sub[Parent]  (= stored Parent pkt
         *               content, navigated via the .sub chain)
         *   newPkt    = MakePacket(Parent, extracted.payload,
         *                          extracted.sub, TRUE)
         * Allocate a fresh slot for Parent's wrapper.packet (DIFFERENT slot
         * from Grand.sub[Parent] = g_s0).  sub[child] still points at the
         * OLD child slots; UnbundleCASChild Parent sync advances sub[child]
         * to new_child_slot atomically with the child CAS. */
        uint32_t p_ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
        uint32_t new_parent_pkt_slot =
            alloc_slot(saved_pp, saved_p_s0, saved_p_s1, true);
        Wrapper restored_pw = make_priority_from_slot(new_parent_pkt_slot, p_ser);
        gate(NODE_PARENT, ctx->iter, ctx->tid);
        Wrapper exp_pw = pw;
        if (!cas_w(NODE_PARENT, &exp_pw, restored_pw)) {
            tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
            SPIN_INC(spin_commit_child); continue;
        }
        tag_after_success(NODE_PARENT, ctx->tid);
        ctx->serial = p_ser;

        /* UnbundleCASChild: TLA+ Fix 3 — atomically update child wrapper AND
         * immediate parent's packet.sub[child].  C11 cannot do a true atomic
         * dual-CAS, so we sequentialize:
         *   (1) CAS Child wrapper.
         *   (2) On success, CAS Parent wrapper to a new wrapper whose
         *       packet.sub[child] points at the new child packet.
         * The Parent CAS uses our just-installed restored_pw as expected.
         * If it fails (peer touched Parent), we leave Parent stale and
         * accept residual lost-increment risk — best the C11 can do
         * without DCAS. cc++ is gated only on the child CAS, matching
         * TLA+ commitOk / iterBudget--. */
        uint32_t c_ser = gen_serial(ctx->serial, cw.serial, ctx->tid);
        uint32_t new_child_slot = alloc_slot(new_payload, SLOT_NULL, SLOT_NULL, false);
        Wrapper new_child = make_priority_from_slot(new_child_slot, c_ser);
        gate(child_node, ctx->iter, ctx->tid);
        Wrapper exp_child = cw;
        if (cas_w(child_node, &exp_child, new_child)) {
            /* Parent sync: best-effort CAS Parent to advance sub[child]. */
            uint32_t sync_p_s0 = (child_node == NODE_CHILD1) ? new_child_slot : saved_p_s0;
            uint32_t sync_p_s1 = (child_node == NODE_CHILD2) ? new_child_slot : saved_p_s1;
            /* TLA+ literal: newParentPkt = [parentW.packet EXCEPT !.sub[node] = local.newpacket]
             * — parentW.packet here is the missing=true slot we installed at
             * UnbundleCASLoop step 2 (restored_pw), so missing field is
             * preserved as TRUE. */
            uint32_t new_parent_slot = alloc_slot(saved_pp, sync_p_s0, sync_p_s1, true);
            Wrapper sync_pw = make_priority_from_slot(new_parent_slot, c_ser);
            Wrapper exp_sync = restored_pw;
            (void)cas_w(NODE_PARENT, &exp_sync, sync_pw);
            ctx->serial = c_ser;
            atomic_fetch_add_explicit(&commit_count[child_node], 1, memory_order_relaxed);
            clear_my_tags(ctx->tid);
            OP_UNLOCK();
            return;
        }
        tag_after_fail(child_node, ctx->iter, ctx->tid);
        SPIN_INC(spin_commit_child);
    }
}

/* =========================================================================
 * snapshot_parent_priority: ensure Parent is priority + !missing.  Used by
 * insert/release.  TLA+ ReadParent has 3 branches:
 *   Parent priority+!missing  → proceed
 *   Parent priority+missing   → run inner bundle (no outer; don't push
 *                               Parent under Grand)
 *   Parent bundled-by-Grand   → stutter (wait for some commit_child path
 *                               to unbundle Parent)
 * Insert is typically done BEFORE any commit, so Parent stays unbundled
 * for the duration of all inserts.  After commits start, this function
 * may spin briefly when Parent is in a bundled-by-Grand window.
 * ========================================================================= */
static void snapshot_parent_priority(ThreadCtx *ctx, Wrapper *out_pw) {
    for (;;) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) {
            /* Caller is responsible for cleanup. */
            *out_pw = load_w(NODE_PARENT);
            return;
        }
        Wrapper pw = load_w(NODE_PARENT);
        if (!pw.has_priority) {
            /* Parent bundled-by-Grand — stutter. */
#if defined(__x86_64__) || defined(__i386__)
            __asm__ __volatile__("pause");
#endif
            continue;
        }
        bool pm;
        pkt_unpack(load_pkt_raw(pw.packet_slot), NULL, NULL, NULL, &pm);
        if (pm) {
            Wrapper tmp;
            if (!try_inner_bundle(ctx, &tmp)) continue;
            pw = tmp;
        }
        *out_pw = pw;
        return;
    }
}

/* =========================================================================
 * insert_child(c): TLA+ InsertStart..InsertFinal.  Inserts a new child
 * into Parent.sub[c] with payload+1.  Marks inserted[c] = TRUE.
 * ========================================================================= */
static bool insert_child(int child_node, ThreadCtx *ctx) {
    uint32_t expected_claim = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &insert_target_claim[child_node], &expected_claim, ctx->tid,
            memory_order_acq_rel, memory_order_relaxed))
        return false;
    if (atomic_load_explicit(&ever_inserted[child_node], memory_order_acquire)) {
        atomic_store_explicit(&insert_target_claim[child_node], 0u, memory_order_release);
        return false;
    }
    OP_LOCK();
    for (;;) {
        Wrapper pw;
        snapshot_parent_priority(ctx, &pw);

        /* InsertCASParent: serial-bump on Parent. */
        uint32_t s_ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
        Wrapper stamped = make_priority_from_slot(pw.packet_slot, s_ser);
        gate(NODE_PARENT, ctx->iter, ctx->tid);
        Wrapper exp_pw = pw;
        if (!cas_w(NODE_PARENT, &exp_pw, stamped)) {
            tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
            continue;
        }
        tag_after_success(NODE_PARENT, ctx->tid);
        ctx->serial = s_ser;
        pw = stamped;

        /* InsertReadChild + InsertCASChild. */
        Wrapper cw = load_w(child_node);
        uint32_t saved_child_pkt;
        if (cw.has_priority) {
            saved_child_pkt = cw.packet_slot;
            Wrapper inserted_ref = (Wrapper){
                .has_priority = false, .serial = s_ser,
                .bundled_by = NODE_PARENT, .packet_slot = saved_child_pkt,
            };
            gate(child_node, ctx->iter, ctx->tid);
            Wrapper exp_c = cw;
            if (!cas_w(child_node, &exp_c, inserted_ref)) {
                tag_after_fail(child_node, ctx->iter, ctx->tid);
                continue;
            }
            tag_after_success(child_node, ctx->tid);
        } else if (cw.bundled_by == NODE_PARENT) {
            if (cw.packet_slot != SLOT_NULL) {
                saved_child_pkt = cw.packet_slot;
            } else {
                uint32_t p_s0, p_s1;
                pkt_unpack(load_pkt_raw(pw.packet_slot), NULL, &p_s0, &p_s1, NULL);
                saved_child_pkt = (child_node == NODE_CHILD1) ? p_s0 : p_s1;
                if (saved_child_pkt == SLOT_NULL) continue;
            }
        } else {
            continue;
        }

        /* InsertFinal: CAS Parent with new sub[c] = (payload+1, sub, missing). */
        uint8_t saved_payload;
        uint32_t saved_s0_sub, saved_s1_sub;
        bool saved_missing;
        pkt_unpack(load_pkt_raw(saved_child_pkt),
                   &saved_payload, &saved_s0_sub, &saved_s1_sub, &saved_missing);
        uint32_t new_child_slot = alloc_slot(
            (uint8_t)(saved_payload + 1u),
            saved_s0_sub, saved_s1_sub, saved_missing);

        uint8_t pp; uint32_t p_s0, p_s1; bool pm;
        pkt_unpack(load_pkt_raw(pw.packet_slot), &pp, &p_s0, &p_s1, &pm);
        uint32_t new_s0 = (child_node == NODE_CHILD1) ? new_child_slot : p_s0;
        uint32_t new_s1 = (child_node == NODE_CHILD2) ? new_child_slot : p_s1;

        uint32_t f_ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
        Wrapper new_pw = make_priority_inner(pp, f_ser, new_s0, new_s1, true);
        gate(NODE_PARENT, ctx->iter, ctx->tid);
        Wrapper exp_pw2 = pw;
        if (!cas_w(NODE_PARENT, &exp_pw2, new_pw)) {
            tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
            continue;
        }
        ctx->serial = f_ser;
        atomic_store_explicit(&inserted[child_node], true, memory_order_release);
        atomic_store_explicit(&ever_inserted[child_node], true, memory_order_release);
        clear_my_tags(ctx->tid);
        atomic_store_explicit(&insert_target_claim[child_node], 0u, memory_order_release);
        OP_UNLOCK();
        return true;
    }
}

/* =========================================================================
 * release_child(c): TLA+ ReleaseStart..ReleaseCASChild.
 * ========================================================================= */
static bool release_child(int child_node, ThreadCtx *ctx) {
    uint32_t expected_claim = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &release_target_claim[child_node], &expected_claim, ctx->tid,
            memory_order_acq_rel, memory_order_relaxed))
        return false;
    if (!atomic_load_explicit(&inserted[child_node], memory_order_acquire)) {
        atomic_store_explicit(&release_target_claim[child_node], 0u, memory_order_release);
        return false;
    }
    OP_LOCK();
    for (;;) {
        Wrapper pw;
        snapshot_parent_priority(ctx, &pw);

        uint8_t pp; uint32_t p_s0, p_s1; bool pm;
        pkt_unpack(load_pkt_raw(pw.packet_slot), &pp, &p_s0, &p_s1, &pm);
        uint32_t saved_child_pkt = (child_node == NODE_CHILD1) ? p_s0 : p_s1;
        if (saved_child_pkt == SLOT_NULL) {
            atomic_store_explicit(&inserted[child_node], false, memory_order_release);
            clear_my_tags(ctx->tid);
            atomic_store_explicit(&release_target_claim[child_node], 0u, memory_order_release);
            OP_UNLOCK();
            return true;
        }
        uint32_t new_s0 = (child_node == NODE_CHILD1) ? SLOT_NULL : p_s0;
        uint32_t new_s1 = (child_node == NODE_CHILD2) ? SLOT_NULL : p_s1;
        uint32_t r_ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
        Wrapper new_pw = make_priority_inner(pp, r_ser, new_s0, new_s1, pm);
        gate(NODE_PARENT, ctx->iter, ctx->tid);
        Wrapper exp_pw = pw;
        if (!cas_w(NODE_PARENT, &exp_pw, new_pw)) {
            tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
            continue;
        }
        ctx->serial = r_ser;
        atomic_store_explicit(&inserted[child_node], false, memory_order_release);

        /* ReleaseReadChild + ReleaseCASChild: restore child to priority. */
        for (;;) {
            Wrapper cw = load_w(child_node);
            if (cw.has_priority) break;
            uint32_t c_ser = gen_serial(ctx->serial, cw.serial, ctx->tid);
            Wrapper restored = make_priority_from_slot(saved_child_pkt, c_ser);
            gate(child_node, ctx->iter, ctx->tid);
            Wrapper exp_c = cw;
            if (cas_w(child_node, &exp_c, restored)) {
                ctx->serial = c_ser;
                break;
            }
            tag_after_fail(child_node, ctx->iter, ctx->tid);
        }
        clear_my_tags(ctx->tid);
        atomic_store_explicit(&release_target_claim[child_node], 0u, memory_order_release);
        OP_UNLOCK();
        return true;
    }
}

/* =========================================================================
 * Thread worker (dynamic): Insert → Commit loop → Release.
 * ========================================================================= */
static void *worker(void *arg) {
    ThreadCtx ctx = *(ThreadCtx*)arg;

    /* --- Phase A: Insert --- */
    if (ctx.role & ROLE_INSERT) {
        for (int c = NODE_CHILD1; c <= NODE_CHILD2; c++) {
            if (atomic_load_explicit(&g_stop, memory_order_relaxed)) goto release_phase;
            (void)insert_child(c, &ctx);
        }
    }

    /* --- Phase B: Commit loop --- */
    if (ctx.role & (ROLE_ROOT | ROLE_LEAF)) {
        /* Waiting: spin until all AllChildren ever_inserted. */
        for (;;) {
            if (atomic_load_explicit(&g_stop, memory_order_relaxed)) goto release_phase;
            bool all_ever = true;
            for (int c = NODE_CHILD1; c <= NODE_CHILD2; c++) {
                if (!atomic_load_explicit(&ever_inserted[c], memory_order_acquire)) {
                    all_ever = false; break;
                }
            }
            if (all_ever) break;
#if defined(__x86_64__) || defined(__i386__)
            __asm__ __volatile__("pause");
#endif
        }

        for (uint32_t i = 0; i < (uint32_t)MAX_COMMITS; i++) {
            if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;
            bool any_active = false;
            for (int c = NODE_CHILD1; c <= NODE_CHILD2; c++) {
                if (atomic_load_explicit(&inserted[c], memory_order_acquire)) {
                    any_active = true; break;
                }
            }
            if (!any_active) break;
            if (ctx.role & ROLE_ROOT) commit_grand(&ctx);
            if (ctx.role & ROLE_LEAF) {
                for (int c = NODE_CHILD1; c <= NODE_CHILD2; c++) {
                    if (atomic_load_explicit(&inserted[c], memory_order_acquire))
                        commit_child(c, &ctx);
                }
            }
            ctx.iter++;
        }
    }

release_phase:
    /* --- Phase C: Release --- */
    if (ctx.role & ROLE_RELEASE) {
        for (int c = NODE_CHILD1; c <= NODE_CHILD2; c++) {
            if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;
            (void)release_child(c, &ctx);
        }
    }
    return NULL;
}

/* =========================================================================
 * Post-join invariant checks.
 * ========================================================================= */
static void check_invariants(void) {
    Wrapper gw = load_w(NODE_GRAND);
    assert(gw.has_priority);   /* GrandAlwaysPriority */

    Wrapper pw = load_w(NODE_PARENT);
    assert(pw.has_priority || pw.bundled_by == NODE_GRAND);

    for (int c = NODE_CHILD1; c <= NODE_CHILD2; c++) {
        Wrapper w = load_w(c);
        assert(w.has_priority || w.bundled_by == NODE_PARENT);
    }

    /* SnapshotConsistency (dynamic): only ActiveChildren require sub[c]!=Null. */
    if (pw.has_priority) {
        uint32_t s0, s1; bool pm;
        pkt_unpack(load_pkt_raw(pw.packet_slot), NULL, &s0, &s1, &pm);
        if (!pm) {
            if (atomic_load(&inserted[NODE_CHILD1])) assert(s0 != SLOT_NULL);
            if (atomic_load(&inserted[NODE_CHILD2])) assert(s1 != SLOT_NULL);
        }
    }
    if (gw.has_priority) {
        uint32_t s0, s1; bool gm;
        pkt_unpack(load_pkt_raw(gw.packet_slot), NULL, &s0, &s1, &gm);
        if (!gm) assert(s0 != SLOT_NULL);
    }
}

int main(void) {
    /* Slot 0 reserved. Start counter such that first allocation returns slot 1. */
    atomic_store(&global_slot_counter, 0u);

    /* Init packets: leaves priority with payload=0.  Inner nodes (Parent,
     * Grand) start as priority but with packet.missing=true and no sub slots —
     * this forces the first snapshot to go through a full recursive bundle,
     * so Children are collected (via try_inner_bundle) and then bundled
     * under Grand before any commit chain is installed. */
    /* Init (dynamic): children start unattached (priority, payload=0).
     * Parent starts priority+missing=TRUE with empty sub (forces first
     * snapshot to run a trivial bundle settling sub=Null). Grand same. */
    uint32_t s_c1 = alloc_slot(0, SLOT_NULL, SLOT_NULL, false);
    uint32_t s_c2 = alloc_slot(0, SLOT_NULL, SLOT_NULL, false);
    uint32_t s_p  = alloc_slot(0, SLOT_NULL, SLOT_NULL, true);
    uint32_t s_g  = alloc_slot(0, SLOT_NULL, SLOT_NULL, true);

    Wrapper init_grand  = (Wrapper){.has_priority=true, .serial=0,
                                    .bundled_by=NULL_NODE, .packet_slot=s_g};
    Wrapper init_parent = (Wrapper){.has_priority=true, .serial=0,
                                    .bundled_by=NULL_NODE, .packet_slot=s_p};
    Wrapper init_c1     = (Wrapper){.has_priority=true, .serial=0,
                                    .bundled_by=NULL_NODE, .packet_slot=s_c1};
    Wrapper init_c2     = (Wrapper){.has_priority=true, .serial=0,
                                    .bundled_by=NULL_NODE, .packet_slot=s_c2};

    atomic_store(&linkage[NODE_GRAND],  wrapper_pack(init_grand));
    atomic_store(&linkage[NODE_PARENT], wrapper_pack(init_parent));
    atomic_store(&linkage[NODE_CHILD1], wrapper_pack(init_c1));
    atomic_store(&linkage[NODE_CHILD2], wrapper_pack(init_c2));
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&commit_count[i], 0);
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&priority_tag[i], TAG_NULL);
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&inserted[i], false);
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&ever_inserted[i], false);
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&insert_target_claim[i], 0u);
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&release_target_claim[i], 0u);
    atomic_store(&g_stop, false);

    pthread_t threads[NUM_THREADS];
    ThreadCtx ctxs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        ctxs[i].tid    = (uint32_t)(i + 1);   /* tid 0 reserved for Null */
        ctxs[i].iter   = 0;
        ctxs[i].serial = 0;
        if (i < NUM_ROOT_ONLY)                       ctxs[i].role = ROLE_ROOT;
        else if (i < NUM_ROOT_ONLY + NUM_LEAF_ONLY)  ctxs[i].role = ROLE_LEAF;
        else                                         ctxs[i].role = ROLE_BOTH;
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

    uint32_t cc1 = atomic_load(&commit_count[NODE_CHILD1]);
    uint32_t cc2 = atomic_load(&commit_count[NODE_CHILD2]);

    Wrapper c1 = load_w(NODE_CHILD1);
    Wrapper c2 = load_w(NODE_CHILD2);
    /* TLA+ ChildPayload(c): walk the bundle chain to read the effective
     * payload regardless of bundle state.
     *   priority c            → c.packet.payload
     *   c bundled, P priority → Parent.packet.sub[c].payload
     *   both bundled          → Grand.packet.sub[Parent].sub[c].payload
     * GrandAlwaysPriority guarantees the chain terminates. */
    Wrapper gw_t = load_w(NODE_GRAND);
    Wrapper pw_t = load_w(NODE_PARENT);
    uint8_t c1_payload = 0, c2_payload = 0;
    /* TLA+ ChildPayload(c) with dynamic-aware Null handling: after release
     * Parent.sub[c]=Null and child has been restored to priority. */
    if (c1.has_priority) {
        c1_payload = load_pkt_payload(c1.packet_slot);
    } else if (pw_t.has_priority) {
        uint32_t s0, s1; pkt_unpack(load_pkt_raw(pw_t.packet_slot), NULL, &s0, &s1, NULL);
        if (s0 != SLOT_NULL) c1_payload = load_pkt_payload(s0);
    } else {
        uint32_t gs0, gs1; pkt_unpack(load_pkt_raw(gw_t.packet_slot), NULL, &gs0, &gs1, NULL);
        if (gs0 != SLOT_NULL) {
            uint32_t s0, s1; pkt_unpack(load_pkt_raw(gs0), NULL, &s0, &s1, NULL);
            if (s0 != SLOT_NULL) c1_payload = load_pkt_payload(s0);
        }
    }
    if (c2.has_priority) {
        c2_payload = load_pkt_payload(c2.packet_slot);
    } else if (pw_t.has_priority) {
        uint32_t s0, s1; pkt_unpack(load_pkt_raw(pw_t.packet_slot), NULL, &s0, &s1, NULL);
        if (s1 != SLOT_NULL) c2_payload = load_pkt_payload(s1);
    } else {
        uint32_t gs0, gs1; pkt_unpack(load_pkt_raw(gw_t.packet_slot), NULL, &gs0, &gs1, NULL);
        if (gs0 != SLOT_NULL) {
            uint32_t s0, s1; pkt_unpack(load_pkt_raw(gs0), NULL, &s0, &s1, NULL);
            if (s1 != SLOT_NULL) c2_payload = load_pkt_payload(s1);
        }
    }
    /* Dynamic terminal: ChildPayload(c) = 1 (insert) + commit_count[c]. */
    uint32_t expected1 = (1u + cc1) % MAX_PAYLOAD;
    uint32_t expected2 = (1u + cc2) % MAX_PAYLOAD;
    bool e1_ok = atomic_load(&ever_inserted[NODE_CHILD1]);
    bool e2_ok = atomic_load(&ever_inserted[NODE_CHILD2]);
    if (!e1_ok || !e2_ok ||
        (uint32_t)c1_payload != expected1 ||
        (uint32_t)c2_payload != expected2) {
        Wrapper gdbg = load_w(NODE_GRAND);
        Wrapper pdbg = load_w(NODE_PARENT);
        uint8_t  g_pl=0,p_pl=0; uint32_t g_s0=0,g_s1=0,p_s0=0,p_s1=0;
        bool g_m=false, p_m=false;
        if (gdbg.has_priority)
            pkt_unpack(load_pkt_raw(gdbg.packet_slot), &g_pl, &g_s0, &g_s1, &g_m);
        if (pdbg.has_priority)
            pkt_unpack(load_pkt_raw(pdbg.packet_slot), &p_pl, &p_s0, &p_s1, &p_m);
        fprintf(stderr,
            "FAIL: MaxPayload=%d pool=%u\n"
            "  Grand  : prio=%d ser=%u slot=%u pkt=(pl=%u s0=%u s1=%u m=%d)\n"
            "  Parent : prio=%d bundled_by=%u ser=%u slot=%u pkt=(pl=%u s0=%u s1=%u m=%d)\n"
            "  Child1 : prio=%d bundled_by=%u ser=%u slot=%u payload=%u cc=%u (cc%%M=%u)\n"
            "  Child2 : prio=%d bundled_by=%u ser=%u slot=%u payload=%u cc=%u (cc%%M=%u)\n",
            MAX_PAYLOAD, (unsigned)PACKET_POOL_ENTRIES,
            gdbg.has_priority, gdbg.serial, gdbg.packet_slot, g_pl, g_s0, g_s1, g_m,
            pdbg.has_priority, pdbg.bundled_by, pdbg.serial, pdbg.packet_slot,
            p_pl, p_s0, p_s1, p_m,
            c1.has_priority, c1.bundled_by, c1.serial, c1.packet_slot, c1_payload,
            cc1, cc1 % MAX_PAYLOAD,
            c2.has_priority, c2.bundled_by, c2.serial, c2.packet_slot, c2_payload,
            cc2, cc2 % MAX_PAYLOAD);
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
    printf("[3level-LLfree stress %s %ds pool=%u] Child1=%u commits, Child2=%u commits (total=%u)\n",
           mode_str, STRESS_SECONDS, (unsigned)PACKET_POOL_ENTRIES,
           cc1, cc2, cc1 + cc2);
    printf("  spin: commit_grand=%llu commit_child=%llu inner=%llu outer=%llu stale_read=%llu\n",
           (unsigned long long)atomic_load(&spin_commit_grand),
           (unsigned long long)atomic_load(&spin_commit_child),
           (unsigned long long)atomic_load(&spin_inner_bundle),
           (unsigned long long)atomic_load(&spin_outer_bundle),
           (unsigned long long)atomic_load(&spin_stale_read));
    printf("  llfree: negotiate_wait=%llu preempt=%llu\n",
           (unsigned long long)atomic_load(&spin_negotiate),
           (unsigned long long)atomic_load(&spin_preempt));
#else
    /* Dynamic terminal: payload check done above via ChildPayload chain
     * walk (= 1 + commit_count[c]).  No static expected_total. */
#endif

    return 0;
}
