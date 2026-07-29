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
 * C11 test generated mechanically from BundleUnbundle_2level_LLfree.tla.
 * (Layer 2, 2-level, livelock-free negotiate variant.)
 *
 * Tree: Parent --+-- Child1
 *                +-- Child2
 *
 * Encoding: slot-pool + Lamport serial (same as test_bundle_2level.c).
 * On top of the base 2-level protocol this variant adds the LL-free
 * "negotiate" priority-tag mechanism per node:
 *
 *   - priority_tag[n] : atomic <<iter, tid>> tag set on CAS failure at n
 *     and cleared on CAS success.  Other threads consult the tag before
 *     attempting a CAS at n; they may proceed only if the tag is null or
 *     theirs, else they must first preempt.  Older transactions
 *     (smaller iter first, then smaller tid) win contention — older
 *     active threads preempt the tag (PreemptTag in TLA+, folded into
 *     the can_proceed_with_preempt path here).
 *   - tags are cleared ONLY on commit success (clear_my_tags); no
 *     thread_active[]/zombie tracking — a thread reaches inactive state
 *     only via the success path, so no stale tag outlives its owner.
 *   - per-thread iter counter advances at the END of each iteration
 *     (mirrors TLA+ CommitDone -> iterBudget--, with iter(t) ==
 *     MaxCommits - iterBudget[t]).
 *
 * Atomicity modes / Option Z / diag counters carry over verbatim.
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
 * max in-flight references prevents pool-wrap silent corruption.  Default
 * 65536 (L2-friendly); raise via -DPACKET_POOL_ENTRIES up to 134217727. */
/* Safe default: max 27-bit pool (widest wrap margin). */
#ifndef PACKET_POOL_ENTRIES
#define PACKET_POOL_ENTRIES 134217727u
#endif
#if (PACKET_POOL_ENTRIES) > 134217727u
#  error "PACKET_POOL_ENTRIES must fit in 27 bits (<=134217727)"
#endif

/* --- Node IDs --- */
#define NODE_PARENT 0
#define NODE_CHILD1 1
#define NODE_CHILD2 2
#define NUM_NODES   3

/* bundled_by is a 2-bit field: 0=PARENT, 3=NULL_NODE (priority).  The
 * 2-level tree only has Parent as a possible bundler (Children have no
 * other ancestor); values 1 and 2 are unused. */
#define NULL_NODE   0x3u

/* --- Serial + slot widths (symmetric, matches 3-level) --- */
#define SER_BITS   27u
#define SER_MOD    (1u << SER_BITS)
#define SER_MASK   (SER_MOD - 1u)
#define SLOT_BITS  27u
#define SLOT_MASK  ((1u << SLOT_BITS) - 1u)
#define SLOT_NULL  0u   /* reserved */

/* --- Modular serial comparison (TLA+ ModGT) --- */
static inline bool ser_gt(uint32_t a, uint32_t b) {
    uint32_t diff = (a - b) & SER_MASK;
    return diff > 0 && diff < (SER_MOD >> 1);
}

/* TID-encoded base-B Lamport serial — mirrors C++
 * SerialGenerator::gen() (transaction.h:547-576).  Counter in upper
 * bits + TID in lower bits → same counter on two threads produces
 * DIFFERENT serials, so wrappers are thread-unique without any global
 * atomic.
 *
 * TLA+: serial = counter * SerialBase + tid, SerialBase > max TID.
 * GenSerial: newCnt = max(SerialCounter(lastSer), SerialCounter(serial[t])) + 1
 *            return EncodeSerial(newCnt, t)
 * Pure TLS + linkage-serial Lamport — no globalSerial. */
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
static inline uint8_t load_pkt_payload(uint32_t slot) {
    uint8_t p; pkt_unpack(load_pkt_raw(slot), &p, NULL, NULL, NULL);
    return p;
}

/* ============================================================================
 * Wrapper: serial(27) | has_priority(1) | bundled_by(2) | packet_slot(27).
 * No wrapper-level missing (lives in Packet).
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
static _Atomic(uint32_t) commit_count[NUM_NODES];
static _Atomic(bool)     g_stop;

/* --- Spin / race-detection counters (stress diagnostics) --- */
static _Atomic(uint64_t) spin_bundle;        /* try_bundle bailouts        */
static _Atomic(uint64_t) spin_commit_parent; /* commit_parent CAS failures */
static _Atomic(uint64_t) spin_commit_child;  /* commit_child  CAS failures */
static _Atomic(uint64_t) spin_stale_read;    /* pool-wrap detector fires   */
static _Atomic(uint64_t) spin_negotiate;     /* can_proceed gated waits    */
static _Atomic(uint64_t) spin_preempt;       /* tag preemptions issued     */
#define SPIN_INC(name) atomic_fetch_add_explicit(&(name), 1u, memory_order_relaxed)

/* ============================================================================
 * LL-free negotiate machinery.
 *   Tag = uint64_t. Bit 63 = "valid" flag. Bits 32-62 = iter (31 bit).
 *   Bits 0-31 = tid.  Tag value 0 == Null (no holder).
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
/* Strict <iter, tid> lexicographic order — smaller is older. */
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

/* can_proceed_with_preempt: TLA+ CanProceed merged with PreemptTag.
 * Returns true if (tag null) OR (tag mine) OR (we successfully preempted
 * a younger active holder).  No zombie branch. */
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

/* tag_after_fail: register/refresh/preempt my tag at n.  TLA+ rule:
 *   null     -> mine
 *   mine     -> mine
 *   I'm older-> mine (preempt)
 *   else     -> keep cur */
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

/* tag_after_success: NO-OP (Tx-scope persistence; tags persist within a
 * Transaction across all CASes and are released ONLY at Tx-success via
 * clear_my_tags). */
static inline void tag_after_success(int n, uint32_t my_tid) {
    (void)n; (void)my_tid;
}

/* clear_my_tags: TLA+ ClearMyTags(t).  Called ONLY on commit success.
 * On CAS failure, tags are preserved across retries — matches C++
 * operator++() which does not call drop_tags_n_privilege(). */
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

/* gate(): block until proceed allowed at node n; counts negotiate spins. */
static inline void gate(int n, uint32_t my_iter, uint32_t my_tid) {
    while (!can_proceed_with_preempt(n, my_iter, my_tid)) {
        SPIN_INC(spin_negotiate);
        /* short backoff — keeps CPU available to the older holder */
#if defined(__x86_64__) || defined(__i386__)
        __asm__ __volatile__("pause");
#endif
    }
}

typedef struct {
    uint32_t tid;          /* 1-indexed */
    uint32_t iter;         /* completed iterations */
    uint32_t serial;       /* Lamport thread-local clock */
} ThreadCtx;

/* Pool-wrap / stale-snapshot detector (Option Z).  See test_bundle_3level.c
 * for the rationale — (A) pool recycle safety, (B) transaction-age bound. */
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
static inline Wrapper make_priority_leaf(uint8_t payload, uint32_t serial, bool pkt_missing) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .packet_slot  = alloc_slot(payload, SLOT_NULL, SLOT_NULL, pkt_missing),
    };
}
static inline Wrapper make_priority_parent(uint8_t payload, uint32_t serial,
                                           uint32_t s0, uint32_t s1, bool pkt_missing) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .packet_slot  = alloc_slot(payload, s0, s1, pkt_missing),
    };
}
/* Reuse an existing packet slot (SUPERFINE prestamp / identity-stable path).
 * Only invoked in SUPERFINE mode; quiet -Wunused-function in other builds. */
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
 * try_bundle (Phase1..Phase4): run when Parent's packet.missing=true.
 * Collects Child sub-packet slots, CASes Parent to Phase2 state, flips
 * Children to bundled-ref, then CASes Parent to Phase4 (missing=false).
 * Returns the finalized Parent wrapper in *out_final on success.
 * ========================================================================= */
static bool try_bundle(ThreadCtx *ctx, Wrapper *out_final) {
    Wrapper pw = load_w(NODE_PARENT);
    if (!pw.has_priority) { SPIN_INC(spin_bundle); return false; }

    uint8_t  pp;
    uint32_t old_s0, old_s1;
    bool     pw_missing;
    pkt_unpack(load_pkt_raw(pw.packet_slot), &pp, &old_s0, &old_s1, &pw_missing);

    if (!pw_missing) { *out_final = pw; return true; }

    uint32_t bundle_ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
    ctx->serial = bundle_ser;

#if MODE == MODE_SUPERFINE
    if (pw.serial != bundle_ser) {
        gate(NODE_PARENT, ctx->iter, ctx->tid);
        Wrapper prestamp = make_priority_from_slot(pw.packet_slot, bundle_ser);
        Wrapper exp = pw;
        if (!cas_w(NODE_PARENT, &exp, prestamp)) {
            tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
            SPIN_INC(spin_bundle); return false;
        }
        tag_after_success(NODE_PARENT, ctx->tid);
        pw = prestamp;
    }
#endif

    /* Phase1 collect: for each child, determine its sub-packet slot. */
    Wrapper cw1 = load_w(NODE_CHILD1);
    Wrapper cw2 = load_w(NODE_CHILD2);
    uint32_t sp1, sp2;
    if (cw1.has_priority)                                           sp1 = cw1.packet_slot;
    else if (cw1.bundled_by == NODE_PARENT && old_s0 != SLOT_NULL)  sp1 = old_s0;
    else                                                            sp1 = SLOT_NULL;
    if (cw2.has_priority)                                           sp2 = cw2.packet_slot;
    else if (cw2.bundled_by == NODE_PARENT && old_s1 != SLOT_NULL)  sp2 = old_s1;
    else                                                            sp2 = SLOT_NULL;
    /* Collect-fail: eagerly tag Parent (= bundleNode) before caller
     * restarts from snap_read.  Mirrors TLA+ BundleUnbundle_2level_LLfree
     * line 421-426 (commit 5ff3226 fix). */
    if (sp1 == SLOT_NULL || sp2 == SLOT_NULL) {
        tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
        SPIN_INC(spin_bundle); return false;
    }

    /* Phase2: Parent packet = (pp, sp1, sp2, missing=TRUE). */
    gate(NODE_PARENT, ctx->iter, ctx->tid);
    Wrapper p2 = make_priority_parent(pp, bundle_ser, sp1, sp2, true);
    Wrapper exp_p2 = pw;
    if (!cas_w(NODE_PARENT, &exp_p2, p2)) {
        tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
        SPIN_INC(spin_bundle); return false;
    }
    tag_after_success(NODE_PARENT, ctx->tid);

    /* Phase3: Children -> bundled-ref.
     *
     * On CAS failure in SUPERFINE mode the TLA+ DISTURBED path eagerly
     * tags BOTH the child and Parent before returning to snap_read, so
     * peers cannot race in during the next snapshot attempt (mirrors
     * C++ outer-scope ScopedNegotiateLinkage at the snapshot() retry
     * loop).  In FINE mode only the child is tagged (no DISTURBED check). */
#if MODE == MODE_SUPERFINE
#  define PHASE3_FAIL_TAG_PARENT() tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid)
#else
#  define PHASE3_FAIL_TAG_PARENT() ((void)0)
#endif
    Wrapper b = make_bundled(NODE_PARENT, bundle_ser);
    gate(NODE_CHILD1, ctx->iter, ctx->tid);
    Wrapper exp_c1 = cw1;
    bool ok1 = cas_w(NODE_CHILD1, &exp_c1, b);
    if (!ok1) {
        tag_after_fail(NODE_CHILD1, ctx->iter, ctx->tid);
        PHASE3_FAIL_TAG_PARENT();
        SPIN_INC(spin_bundle); return false;
    }
    tag_after_success(NODE_CHILD1, ctx->tid);
    gate(NODE_CHILD2, ctx->iter, ctx->tid);
    Wrapper exp_c2 = cw2;
    bool ok2 = cas_w(NODE_CHILD2, &exp_c2, b);
    if (!ok2) {
        tag_after_fail(NODE_CHILD2, ctx->iter, ctx->tid);
        PHASE3_FAIL_TAG_PARENT();
        SPIN_INC(spin_bundle); return false;
    }
    tag_after_success(NODE_CHILD2, ctx->tid);
#undef PHASE3_FAIL_TAG_PARENT

    /* Phase4: Parent packet.missing=false (fresh slot prevents ABA). */
    gate(NODE_PARENT, ctx->iter, ctx->tid);
    Wrapper p4 = make_priority_parent(pp, bundle_ser, sp1, sp2, false);
    Wrapper exp_p4 = p2;
    if (!cas_w(NODE_PARENT, &exp_p4, p4)) {
        tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
        SPIN_INC(spin_bundle); return false;
    }
    tag_after_success(NODE_PARENT, ctx->tid);

    *out_final = p4;
    return true;
}

/* snapshot(): loop until Parent is priority with a non-missing packet. */
static void snapshot(ThreadCtx *ctx, Wrapper *out) {
    for (;;) {
        Wrapper pw = load_w(NODE_PARENT);
        if (!pw.has_priority) continue;   /* Parent never bundled (root). */
        bool m;
        pkt_unpack(load_pkt_raw(pw.packet_slot), NULL, NULL, NULL, &m);
        if (!m) { *out = pw; return; }
        Wrapper tmp;
        if (try_bundle(ctx, &tmp)) { *out = tmp; return; }
    }
}

/* =========================================================================
 * CommitParent: snapshot Parent, bump BOTH children's sub-packets (fresh
 * slots) and CAS Parent with the new chain.  Gated by negotiate tag.
 * ========================================================================= */
static void commit_parent(ThreadCtx *ctx) {
    OP_LOCK();
    for (;;) {
        uint32_t start_counter = cur_slot_counter();
        Wrapper pw;
        snapshot(ctx, &pw);

        uint8_t  pp; uint32_t c1_slot, c2_slot; bool pm;
        pkt_unpack(load_pkt_raw(pw.packet_slot), &pp, &c1_slot, &c2_slot, &pm);
        if (pm || c1_slot == SLOT_NULL || c2_slot == SLOT_NULL) {
            SPIN_INC(spin_commit_parent); continue;
        }

        uint8_t c1_payload = load_pkt_payload(c1_slot);
        uint8_t c2_payload = load_pkt_payload(c2_slot);

        if (pool_stale(start_counter)) continue;

        uint8_t c1_new = (uint8_t)((c1_payload + 1u) % MAX_PAYLOAD);
        uint8_t c2_new = (uint8_t)((c2_payload + 1u) % MAX_PAYLOAD);
        uint32_t new_c1 = alloc_slot(c1_new, SLOT_NULL, SLOT_NULL, false);
        uint32_t new_c2 = alloc_slot(c2_new, SLOT_NULL, SLOT_NULL, false);

        uint32_t ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
        Wrapper new_pw = make_priority_parent(pp, ser, new_c1, new_c2, false);
        gate(NODE_PARENT, ctx->iter, ctx->tid);
        Wrapper exp = pw;
        if (cas_w(NODE_PARENT, &exp, new_pw)) {
            ctx->serial = ser;
            atomic_fetch_add_explicit(&commit_count[NODE_CHILD1], 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&commit_count[NODE_CHILD2], 1, memory_order_relaxed);
            /* Transaction-end: release ALL my tags (TLA+ ClearMyTags). */
            clear_my_tags(ctx->tid);
            OP_UNLOCK();
            return;
        }
        /* CAS fail: TLA+ literal —
         *   priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail]
         * Tags preserved across retries (no ClearMyTags on fail).
         * Mirrors C++ operator++() which does not call drop_tags. */
        tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
        SPIN_INC(spin_commit_parent);
    }
}

/* =========================================================================
 * CommitChild: direct commit if priority, else 1-level unbundle.  All CAS
 * sites are gated by the LL-free negotiate tag.
 * ========================================================================= */
static void commit_child(int child_node, ThreadCtx *ctx) {
    OP_LOCK();
    for (;;) {
        uint32_t start_counter = cur_slot_counter();
        Wrapper cw = load_w(child_node);

        if (cw.has_priority) {
            uint8_t old_payload = load_pkt_payload(cw.packet_slot);
            if (pool_stale(start_counter)) continue;
            uint8_t new_payload = (uint8_t)((old_payload + 1u) % MAX_PAYLOAD);
            uint32_t ser = gen_serial(ctx->serial, cw.serial, ctx->tid);
            Wrapper new_cw = make_priority_leaf(new_payload, ser, false);
            gate(child_node, ctx->iter, ctx->tid);
            Wrapper exp = cw;
            if (cas_w(child_node, &exp, new_cw)) {
                ctx->serial = ser;
                atomic_fetch_add_explicit(&commit_count[child_node], 1, memory_order_relaxed);
                /* Transaction-end (CommitDone success): release my tags. */
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
        if (!pw.has_priority) { SPIN_INC(spin_commit_child); continue; }

        uint8_t  pp; uint32_t p_s0, p_s1; bool pm;
        pkt_unpack(load_pkt_raw(pw.packet_slot), &pp, &p_s0, &p_s1, &pm);
        uint32_t old_child_slot = (child_node == NODE_CHILD1) ? p_s0 : p_s1;
        if (old_child_slot == SLOT_NULL) { SPIN_INC(spin_commit_child); continue; }
        uint8_t old_payload = load_pkt_payload(old_child_slot);
        if (pool_stale(start_counter)) continue;
        uint8_t new_payload = (uint8_t)((old_payload + 1u) % MAX_PAYLOAD);

        /* UnbundleCASAncestor: Parent packet missing=true, subs preserved. */
        uint32_t anc_ser = gen_serial(ctx->serial, pw.serial, ctx->tid);
        Wrapper new_pw = make_priority_parent(pp, anc_ser, p_s0, p_s1, true);
        gate(NODE_PARENT, ctx->iter, ctx->tid);
        Wrapper exp_pw = pw;
        if (!cas_w(NODE_PARENT, &exp_pw, new_pw)) {
            tag_after_fail(NODE_PARENT, ctx->iter, ctx->tid);
            SPIN_INC(spin_commit_child); continue;
        }
        tag_after_success(NODE_PARENT, ctx->tid);
        ctx->serial = anc_ser;

        /* UnbundleCASChild: child becomes priority with new payload. */
        uint32_t c_ser = gen_serial(ctx->serial, cw.serial, ctx->tid);
        Wrapper new_child = make_priority_leaf(new_payload, c_ser, false);
        gate(child_node, ctx->iter, ctx->tid);
        Wrapper exp_child = cw;
        if (cas_w(child_node, &exp_child, new_child)) {
            ctx->serial = c_ser;
            atomic_fetch_add_explicit(&commit_count[child_node], 1, memory_order_relaxed);
            /* Transaction-end (CommitDone success): release my tags. */
            clear_my_tags(ctx->tid);
            OP_UNLOCK();
            return;
        }
        tag_after_fail(child_node, ctx->iter, ctx->tid);
        SPIN_INC(spin_commit_child);
    }
}

/* =========================================================================
 * Thread worker.  iter advances at end of each iteration (TLA+ rule:
 * iter(t) = MaxCommits - iterBudget[t]).  No thread_active tracking needed:
 * the new Privilege model releases tags only on commit success, and every
 * iteration ends with success → no stale tags survive a thread's lifetime.
 * ========================================================================= */
static void *worker(void *arg) {
    ThreadCtx ctx = *(ThreadCtx*)arg;

    for (uint32_t i = 0; i < (uint32_t)MAX_COMMITS; i++) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;
        commit_parent(&ctx);
        commit_child(NODE_CHILD1, &ctx);
        commit_child(NODE_CHILD2, &ctx);
        ctx.iter++;
    }
    return NULL;
}

/* =========================================================================
 * Post-join invariant checks.
 * ========================================================================= */
static void check_invariants(void) {
    Wrapper pw = load_w(NODE_PARENT);
    assert(pw.has_priority);   /* Parent is root, always priority. */

    for (int c = NODE_CHILD1; c <= NODE_CHILD2; c++) {
        Wrapper w = load_w(c);
        assert(w.has_priority || w.bundled_by == NODE_PARENT);
    }

    if (pw.has_priority) {
        uint32_t s0, s1; bool pm;
        pkt_unpack(load_pkt_raw(pw.packet_slot), NULL, &s0, &s1, &pm);
        if (!pm) {
            assert(s0 != SLOT_NULL);
            assert(s1 != SLOT_NULL);
        }
    }
}

int main(void) {
    atomic_store(&global_slot_counter, 0u);

    /* Init: leaves priority with payload=0, Parent priority with packet
     * missing=true (forces the first snapshot to run a full bundle). */
    uint32_t s_c1 = alloc_slot(0, SLOT_NULL, SLOT_NULL, false);
    uint32_t s_c2 = alloc_slot(0, SLOT_NULL, SLOT_NULL, false);
    uint32_t s_p  = alloc_slot(0, SLOT_NULL, SLOT_NULL, true);

    Wrapper init_parent = (Wrapper){.has_priority=true, .serial=0,
                                    .bundled_by=NULL_NODE, .packet_slot=s_p};
    Wrapper init_c1     = (Wrapper){.has_priority=true, .serial=0,
                                    .bundled_by=NULL_NODE, .packet_slot=s_c1};
    Wrapper init_c2     = (Wrapper){.has_priority=true, .serial=0,
                                    .bundled_by=NULL_NODE, .packet_slot=s_c2};

    atomic_store(&linkage[NODE_PARENT], wrapper_pack(init_parent));
    atomic_store(&linkage[NODE_CHILD1], wrapper_pack(init_c1));
    atomic_store(&linkage[NODE_CHILD2], wrapper_pack(init_c2));
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&commit_count[i], 0);
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&priority_tag[i], TAG_NULL);
    atomic_store(&g_stop, false);

    pthread_t threads[NUM_THREADS];
    ThreadCtx ctxs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        ctxs[i].tid    = (uint32_t)(i + 1);   /* 1-indexed; tid 0 reserved for Null */
        ctxs[i].iter   = 0;
        ctxs[i].serial = 0;
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
    uint8_t c1_payload = c1.has_priority ? load_pkt_payload(c1.packet_slot) : 0xFF;
    uint8_t c2_payload = c2.has_priority ? load_pkt_payload(c2.packet_slot) : 0xFF;
    if (!(c1.has_priority && (uint32_t)c1_payload == cc1 % MAX_PAYLOAD) ||
        !(c2.has_priority && (uint32_t)c2_payload == cc2 % MAX_PAYLOAD)) {
        Wrapper pdbg = load_w(NODE_PARENT);
        uint8_t  p_pl=0; uint32_t p_s0=0,p_s1=0; bool p_m=false;
        if (pdbg.has_priority)
            pkt_unpack(load_pkt_raw(pdbg.packet_slot), &p_pl, &p_s0, &p_s1, &p_m);
        fprintf(stderr,
            "FAIL: MaxPayload=%d pool=%u\n"
            "  Parent : prio=%d ser=%u slot=%u pkt=(pl=%u s0=%u s1=%u m=%d)\n"
            "  Child1 : prio=%d bundled_by=%u ser=%u slot=%u payload=%u cc=%u (cc%%M=%u)\n"
            "  Child2 : prio=%d bundled_by=%u ser=%u slot=%u payload=%u cc=%u (cc%%M=%u)\n",
            MAX_PAYLOAD, (unsigned)PACKET_POOL_ENTRIES,
            pdbg.has_priority, pdbg.serial, pdbg.packet_slot,
            p_pl, p_s0, p_s1, p_m,
            c1.has_priority, c1.bundled_by, c1.serial, c1.packet_slot,
            c1_payload, cc1, cc1 % MAX_PAYLOAD,
            c2.has_priority, c2.bundled_by, c2.serial, c2.packet_slot,
            c2_payload, cc2, cc2 % MAX_PAYLOAD);
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
    printf("[2level-LLfree stress %s %ds pool=%u] Child1=%u commits, Child2=%u commits (total=%u)\n",
           mode_str, STRESS_SECONDS, (unsigned)PACKET_POOL_ENTRIES,
           cc1, cc2, cc1 + cc2);
    printf("  spin: commit_parent=%llu commit_child=%llu bundle=%llu stale_read=%llu\n",
           (unsigned long long)atomic_load(&spin_commit_parent),
           (unsigned long long)atomic_load(&spin_commit_child),
           (unsigned long long)atomic_load(&spin_bundle),
           (unsigned long long)atomic_load(&spin_stale_read));
    printf("  llfree: negotiate_wait=%llu preempt=%llu\n",
           (unsigned long long)atomic_load(&spin_negotiate),
           (unsigned long long)atomic_load(&spin_preempt));
#else
    uint32_t expected_total = 2u * (uint32_t)MAX_COMMITS * (uint32_t)NUM_THREADS;
    assert(cc1 == expected_total);
    assert(cc2 == expected_total);
#endif

    return 0;
}
