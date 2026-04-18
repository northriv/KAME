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
 * C11 test generated mechanically from BundleUnbundle_2level.tla (Layer 2, 2-level)
 *
 * Tree: Parent --+-- Child1
 *                +-- Child2
 *
 * Thread lifecycle (MaxCommits iterations per thread):
 *   Each iteration:
 *     1. snapshot(Parent)  -- may trigger bundle (Phase1..4)
 *     2. commit_parent()   -- CAS Parent with ALL children incremented (retry until ok)
 *     3. commit_child(Child1) -- direct commit, unbundle if needed (retry until ok)
 *     4. commit_child(Child2) -- ditto
 *   Each child ends with exactly 2 * MaxCommits * Threads increments (mod MaxPayload).
 *
 * Atomicity modes (compile-time -DMODE=MODE_{COARSE,FINE,SUPERFINE}):
 *   COARSE    : every top-level op (commit_parent / commit_child) runs
 *               under a global mutex.  This matches TLA+ coarse, where
 *               BundlePhase1 and BundlePhase3 are atomic all-or-nothing
 *               transitions — the Phase3 "write both children" cannot be
 *               modeled by two independent CASes without serializing all
 *               concurrent writers.
 *   FINE      : per-step CAS, restart outer loop on failure (default).
 *   SUPERFINE : FINE + pre-bundle serial CAS (Phase1 entry) +
 *               Phase3 DISTURBED check (linkage serial/parent changed).
 *
 * Build modes:
 *   -DSTRESS_SECONDS=10 : 10-second stress loop, reports commit counts.
 *   (default)           : GenMC/unit mode, MAX_COMMITS=1, asserts TerminalPayloadCheck.
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
#    define MAX_COMMITS 0x7fffffff   /* effectively unbounded */
#  else
#    define MAX_COMMITS 1
#  endif
#endif

#ifndef MAX_PAYLOAD
#define MAX_PAYLOAD 3
#endif

#define NODE_PARENT 0
#define NODE_CHILD1 1
#define NODE_CHILD2 2
#define NUM_NODES   3
#define NULL_NODE   0x3       /* 2-bit sentinel for bundled_by field */
#define W_NULL_SUB  0xF       /* 4-bit sentinel for sub fields */

/* --- Serial arithmetic: plain uint32_t with natural overflow.
 * Stress runs do < 2^31 ops, so simple `>` comparisons are safe. --- */
static inline uint32_t gen_serial(uint32_t thread_ser, uint32_t last_ser) {
    uint32_t base = (last_ser > thread_ser) ? last_ser : thread_ser;
    return base + 1;
}

/* --- Wrapper: packed 64-bit linkage word ---
 * Layout:
 *   bits 0-31  : serial (uint32_t)
 *   bit 32     : has_priority
 *   bit 33     : missing
 *   bits 34-35 : bundled_by (2-bit, NULL_NODE=3 when has_priority)
 *   bits 36-39 : payload (0..15)
 *   bits 40-43 : sub_child1 (0..14, 15 = W_NULL_SUB)
 *   bits 44-47 : sub_child2 (0..14, 15 = W_NULL_SUB)
 *   bits 48-63 : unused
 */
typedef struct {
    bool     has_priority;
    uint32_t serial;
    uint8_t  bundled_by;
    bool     missing;
    uint8_t  payload;
    uint8_t  sub_child1;
    uint8_t  sub_child2;
} Wrapper;

static inline uint64_t wrapper_pack(Wrapper w) {
    uint64_t v = 0;
    v |= (uint64_t)w.serial;
    v |= ((uint64_t)(w.has_priority ? 1 : 0)) << 32;
    v |= ((uint64_t)(w.missing ? 1 : 0)) << 33;
    v |= ((uint64_t)(w.bundled_by & 0x3)) << 34;
    v |= ((uint64_t)(w.payload & 0xF)) << 36;
    v |= ((uint64_t)(w.sub_child1 & 0xF)) << 40;
    v |= ((uint64_t)(w.sub_child2 & 0xF)) << 44;
    return v;
}

static inline Wrapper wrapper_unpack(uint64_t v) {
    Wrapper w;
    w.serial       = (uint32_t)(v & 0xFFFFFFFFu);
    w.has_priority = (v >> 32) & 1;
    w.missing      = (v >> 33) & 1;
    w.bundled_by   = (v >> 34) & 0x3;
    w.payload      = (v >> 36) & 0xF;
    w.sub_child1   = (v >> 40) & 0xF;
    w.sub_child2   = (v >> 44) & 0xF;
    return w;
}

static inline bool wrapper_eq(Wrapper a, Wrapper b) {
    return wrapper_pack(a) == wrapper_pack(b);
}

/* --- Shared state --- */
static _Atomic(uint64_t) linkage[NUM_NODES];
static _Atomic(uint32_t) commit_count[NUM_NODES];   /* per-child commit count */
static _Atomic(bool)     g_stop;                    /* stress-mode stop flag */

#if MODE == MODE_COARSE
static pthread_mutex_t coarse_mtx = PTHREAD_MUTEX_INITIALIZER;
#  define OP_LOCK()   pthread_mutex_lock(&coarse_mtx)
#  define OP_UNLOCK() pthread_mutex_unlock(&coarse_mtx)
#else
#  define OP_LOCK()   ((void)0)
#  define OP_UNLOCK() ((void)0)
#endif

static Wrapper load_w(int n) {
    return wrapper_unpack(atomic_load_explicit(&linkage[n], memory_order_acquire));
}

static bool cas_w(int n, Wrapper *expected, Wrapper desired) {
    uint64_t e = wrapper_pack(*expected);
    uint64_t d = wrapper_pack(desired);
    bool ok = atomic_compare_exchange_strong_explicit(
        &linkage[n], &e, d, memory_order_acq_rel, memory_order_relaxed);
    if (!ok) *expected = wrapper_unpack(e);
    return ok;
}

static Wrapper priority_w(uint8_t payload, uint32_t serial,
                          uint8_t sub1, uint8_t sub2, bool missing) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .missing = missing, .payload = payload,
        .sub_child1 = sub1, .sub_child2 = sub2
    };
}

static Wrapper bundled_w(uint8_t parent_node, uint32_t serial) {
    return (Wrapper){
        .has_priority = false, .serial = serial, .bundled_by = parent_node,
        .missing = false, .payload = 0,
        .sub_child1 = W_NULL_SUB, .sub_child2 = W_NULL_SUB
    };
}

/* =========================================================================
 * Bundle (Phase1..Phase4): run when Parent is missing.
 * Returns the finalized parent wrapper (has_priority, !missing).
 * Returns false on "retry from SnapCheck".
 * ========================================================================= */
static bool try_bundle(uint32_t *thread_ser, Wrapper *out_final) {
    /* SnapCheck: read parent */
    Wrapper pw = load_w(NODE_PARENT);
    if (!pw.has_priority) return false;

    if (!pw.missing) {
        *out_final = pw;
        return true;
    }

    /* Need to bundle. Generate bundle serial. */
    uint32_t bundle_ser = gen_serial(*thread_ser, pw.serial);
    *thread_ser = bundle_ser;

#if MODE == MODE_SUPERFINE
    /* Pre-bundle serial CAS: stamp bundle_ser on the parent wrapper. */
    if (pw.serial != bundle_ser) {
        Wrapper prestamp = priority_w(pw.payload, bundle_ser,
                                      pw.sub_child1, pw.sub_child2, pw.missing);
        Wrapper exp = pw;
        if (!cas_w(NODE_PARENT, &exp, prestamp)) return false;
        pw = prestamp;
    }
#endif

    /* BundlePhase1: collect child sub-packets */
    uint8_t sp1, sp2;
    Wrapper cw1, cw2;

    cw1 = load_w(NODE_CHILD1);
    cw2 = load_w(NODE_CHILD2);

    if (cw1.has_priority) sp1 = cw1.payload;
    else if (cw1.bundled_by == NODE_PARENT && pw.sub_child1 != W_NULL_SUB) sp1 = pw.sub_child1;
    else sp1 = W_NULL_SUB;

    if (cw2.has_priority) sp2 = cw2.payload;
    else if (cw2.bundled_by == NODE_PARENT && pw.sub_child2 != W_NULL_SUB) sp2 = pw.sub_child2;
    else sp2 = W_NULL_SUB;

    if (sp1 == W_NULL_SUB || sp2 == W_NULL_SUB) return false;

    /* BundlePhase2: CAS parent with subs (still missing=TRUE) */
    Wrapper p2 = priority_w(pw.payload, bundle_ser, sp1, sp2, true);
    Wrapper exp_p2 = pw;
    if (!cas_w(NODE_PARENT, &exp_p2, p2)) return false;

    /* BundlePhase3: CAS each child to bundled-ref.
     * In coarse mode, OP_LOCK held by caller serializes against other writers,
     * so both CASes observe the snapshot from Phase1 above. */
    Wrapper exp_c1 = cw1;
    Wrapper b1 = bundled_w(NODE_PARENT, bundle_ser);
    bool ok1 = cas_w(NODE_CHILD1, &exp_c1, b1);

    Wrapper exp_c2 = cw2;
    Wrapper b2 = bundled_w(NODE_PARENT, bundle_ser);
    bool ok2 = ok1 ? cas_w(NODE_CHILD2, &exp_c2, b2) : false;

    if (!ok1 || !ok2) {
#if MODE == MODE_SUPERFINE
        /* DISTURBED check: if some child's linkage changed serial, or parent wrapper changed,
         * restart from snap_check; else restart from phase1 (which we do by returning false). */
#endif
        /* Restart from snap_check (outer try_bundle call) */
        return false;
    }

    /* BundlePhase4: finalize — clear missing */
    Wrapper p4 = priority_w(p2.payload, bundle_ser, sp1, sp2, false);
    Wrapper exp_p4 = p2;
    if (!cas_w(NODE_PARENT, &exp_p4, p4)) return false;

    *out_final = p4;
    return true;
}

/* snapshot(): loops until fast-path or bundle succeeds.
 * Returns the parent wrapper (has_priority, !missing) via *out.
 * Never bails: once we start an iteration, commit_count and payload must stay
 * consistent, so we must complete (even if the stop flag was set mid-op).
 */
static void snapshot(uint32_t *thread_ser, Wrapper *out) {
    for (;;) {
        Wrapper pw = load_w(NODE_PARENT);
        if (pw.has_priority && !pw.missing) {
            *out = pw;
            return;
        }
        if (pw.has_priority && pw.missing) {
            if (try_bundle(thread_ser, out)) return;
            continue;
        }
        /* Parent not priority: shouldn't happen in 2-level (no one bundles Parent) */
    }
}

/* =========================================================================
 * CommitParent: CAS Parent with ALL children incremented.
 * Retries snapshot+CAS until successful. No stop-flag check inside: stress
 * mode requires each started iteration to fully complete so that the
 * post-join invariant `child.payload == commit_count[child] % MAX_PAYLOAD`
 * holds (every counted increment must flow into the wrapper).
 * ========================================================================= */
static void commit_parent(uint32_t *thread_ser) {
    OP_LOCK();
    for (;;) {
        Wrapper snap;
        snapshot(thread_ser, &snap);
        assert(snap.has_priority && !snap.missing);
        assert(snap.sub_child1 != W_NULL_SUB);
        assert(snap.sub_child2 != W_NULL_SUB);

        uint8_t new_sub1 = (uint8_t)((snap.sub_child1 + 1) % MAX_PAYLOAD);
        uint8_t new_sub2 = (uint8_t)((snap.sub_child2 + 1) % MAX_PAYLOAD);
        uint32_t ser = gen_serial(*thread_ser, snap.serial);
        Wrapper new_pw = priority_w(snap.payload, ser, new_sub1, new_sub2, snap.missing);

        Wrapper exp = snap;
        if (cas_w(NODE_PARENT, &exp, new_pw)) {
            *thread_ser = ser;
            atomic_fetch_add_explicit(&commit_count[NODE_CHILD1], 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&commit_count[NODE_CHILD2], 1, memory_order_relaxed);
            OP_UNLOCK();
            return;
        }
        /* Retry from snapshot */
    }
}

/* =========================================================================
 * CommitChild: direct commit on a leaf, unbundle if needed.
 * Retries until success. No stop-flag check inside (see commit_parent note).
 * ========================================================================= */
static void commit_child(int child_node, uint32_t *thread_ser) {
    OP_LOCK();
    for (;;) {
        /* CommitRead */
        Wrapper cw = load_w(child_node);

        if (cw.has_priority) {
            /* CommitTryCAS */
            uint8_t new_payload = (uint8_t)((cw.payload + 1) % MAX_PAYLOAD);
            uint32_t ser = gen_serial(*thread_ser, cw.serial);
            Wrapper new_cw = priority_w(new_payload, ser,
                                        cw.sub_child1, cw.sub_child2, cw.missing);
            Wrapper exp = cw;
            if (cas_w(child_node, &exp, new_cw)) {
                *thread_ser = ser;
                atomic_fetch_add_explicit(&commit_count[child_node], 1, memory_order_relaxed);
                OP_UNLOCK();
                return;
            }
            /* CAS failed. TLA+ CommitTryCAS ELSE: IF hasPriority, decide by payload;
             * ELSE go to commit_read. We uniformly retry by continuing. */
            continue;
        }

        /* Unbundle path: child is bundled-ref */
        if (cw.bundled_by != NODE_PARENT) continue;

        /* UnbundleWalk */
        Wrapper parent_w = load_w(NODE_PARENT);
        if (!parent_w.has_priority) continue;

        uint8_t old_sub = (child_node == NODE_CHILD1) ? parent_w.sub_child1
                                                      : parent_w.sub_child2;
        if (old_sub == W_NULL_SUB) continue;

        uint8_t new_sub = (uint8_t)((old_sub + 1) % MAX_PAYLOAD);

        /* UnbundleCASAncestors: copy parent packet, set missing=TRUE,
         * use GenSerial for fresh wrapper distinctness (TLA+ modeling note). */
        uint32_t anc_ser = gen_serial(*thread_ser, parent_w.serial);
        Wrapper new_parent = priority_w(parent_w.payload, anc_ser,
                                        parent_w.sub_child1, parent_w.sub_child2, true);
        Wrapper exp_parent = parent_w;
        if (!cas_w(NODE_PARENT, &exp_parent, new_parent)) continue;
        *thread_ser = anc_ser;

        /* UnbundleCASChild: restore child to priority with new packet */
        uint32_t child_ser = gen_serial(*thread_ser, cw.serial);
        Wrapper new_child = priority_w(new_sub, child_ser, W_NULL_SUB, W_NULL_SUB, false);
        Wrapper exp_child = cw;
        if (cas_w(child_node, &exp_child, new_child)) {
            *thread_ser = child_ser;
            atomic_fetch_add_explicit(&commit_count[child_node], 1, memory_order_relaxed);
            OP_UNLOCK();
            return;
        }
        /* Child changed — retry */
    }
}

/* =========================================================================
 * Thread worker.
 * Stop flag is ONLY checked at the top of each iteration; once an iteration
 * is started, it runs to completion so that commit_count[] increments and
 * wrapper payload updates stay in lock-step.
 * ========================================================================= */
static void *worker(void *arg) {
    (void)arg;
    uint32_t ser = 0;

    for (uint32_t iter = 0; iter < (uint32_t)MAX_COMMITS; iter++) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;
        commit_parent(&ser);
        commit_child(NODE_CHILD1, &ser);
        commit_child(NODE_CHILD2, &ser);
    }
    return NULL;
}

/* =========================================================================
 * Post-join invariant checks
 * ========================================================================= */
static void check_invariants(void) {
    Wrapper pw = load_w(NODE_PARENT);

    /* GrandAlwaysPriority-equivalent for 2-level: Parent is root, always priority */
    assert(pw.has_priority);

    /* SnapshotConsistency */
    if (!pw.missing) {
        assert(pw.sub_child1 != W_NULL_SUB);
        assert(pw.sub_child2 != W_NULL_SUB);
    }

    /* NoPriorityLoss + BundleRefConsistency + MissingPropagation */
    for (int c = NODE_CHILD1; c <= NODE_CHILD2; c++) {
        Wrapper w = load_w(c);
        assert(w.has_priority || w.bundled_by != NULL_NODE);
        if (!w.has_priority && w.bundled_by == NODE_PARENT) {
            assert(pw.has_priority);
        }
    }
}

int main(void) {
    Wrapper init_parent = priority_w(0, 0, W_NULL_SUB, W_NULL_SUB, true);
    Wrapper init_child  = priority_w(0, 0, W_NULL_SUB, W_NULL_SUB, false);
    atomic_store(&linkage[NODE_PARENT], wrapper_pack(init_parent));
    atomic_store(&linkage[NODE_CHILD1], wrapper_pack(init_child));
    atomic_store(&linkage[NODE_CHILD2], wrapper_pack(init_child));
    for (int i = 0; i < NUM_NODES; i++) atomic_store(&commit_count[i], 0);
    atomic_store(&g_stop, false);

    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, worker, NULL);
    }

#if STRESS_SECONDS > 0
    struct timespec ts = { .tv_sec = STRESS_SECONDS, .tv_nsec = 0 };
    nanosleep(&ts, NULL);
    atomic_store_explicit(&g_stop, true, memory_order_release);
#endif

    for (int i = 0; i < NUM_THREADS; i++) pthread_join(threads[i], NULL);

    check_invariants();

    /* Report / assert terminal payload */
    uint32_t cc1 = atomic_load(&commit_count[NODE_CHILD1]);
    uint32_t cc2 = atomic_load(&commit_count[NODE_CHILD2]);

    /* TerminalPayloadCheck: at quiescence, each leaf child is in priority state
     * and its payload equals (its actual increment count) % MaxPayload.
     * Stress mode: use commit_count[] as ground truth (some threads may have
     *              exited mid-MaxCommits-loop, but never mid-iteration — all
     *              counted increments must have reached the wrapper).
     * GenMC  mode: commit_count[] matches 2 * MaxCommits * Threads exactly. */
    Wrapper c1 = load_w(NODE_CHILD1);
    Wrapper c2 = load_w(NODE_CHILD2);
    if (!(c1.has_priority && (uint32_t)c1.payload == cc1 % MAX_PAYLOAD) ||
        !(c2.has_priority && (uint32_t)c2.payload == cc2 % MAX_PAYLOAD)) {
        Wrapper pwdbg = load_w(NODE_PARENT);
        fprintf(stderr,
            "FAIL: MaxPayload=%d\n"
            "  Parent : prio=%d missing=%d sub1=%u sub2=%u ser=%u\n"
            "  Child1 : prio=%d bundled_by=%u payload=%u ser=%u cc=%u (cc%%M=%u)\n"
            "  Child2 : prio=%d bundled_by=%u payload=%u ser=%u cc=%u (cc%%M=%u)\n",
            MAX_PAYLOAD,
            pwdbg.has_priority, pwdbg.missing, pwdbg.sub_child1, pwdbg.sub_child2, pwdbg.serial,
            c1.has_priority, c1.bundled_by, c1.payload, c1.serial, cc1, cc1 % MAX_PAYLOAD,
            c2.has_priority, c2.bundled_by, c2.payload, c2.serial, cc2, cc2 % MAX_PAYLOAD);
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
    printf("[2level stress %s %ds] Child1=%u commits, Child2=%u commits (total=%u)\n",
           mode_str, STRESS_SECONDS, cc1, cc2, cc1 + cc2);
#else
    uint32_t expected_total = 2u * (uint32_t)MAX_COMMITS * (uint32_t)NUM_THREADS;
    assert(cc1 == expected_total);
    assert(cc2 == expected_total);
#endif

    return 0;
}
