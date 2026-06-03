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
 * C11 test generated mechanically from BundleUnbundle.tla (Layer 2, 3-level)
 *
 * Tree: Grand --+-- Parent --+-- Child1
 *                             +-- Child2
 *
 * Thread lifecycle (MaxCommits iterations per thread):
 *   Each iteration:
 *     1. commit_grand() -- snapshot Grand (recursive bundle Parent, then Children),
 *                          then atomic CAS on Grand bumping BOTH leaf sub-packets.
 *     2. commit_child(Child1) -- direct commit or 1/2-level unbundle.
 *     3. commit_child(Child2) -- ditto.
 *   Each child ends with exactly 2 * MaxCommits * Threads increments (mod MaxPayload).
 *
 * Atomicity modes (compile-time -DMODE=MODE_{COARSE,FINE,SUPERFINE}):
 *   COARSE    : every top-level op runs under a global mutex; matches TLA+
 *               coarse (BundleCollectAtomic + BundlePhase3Atomic + Unbundle*
 *               atomic).
 *   FINE      : per-step CAS, restart outer loop on failure (default).
 *   SUPERFINE : FINE + pre-bundle serial CAS on both inner and outer bundles.
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
#    define MAX_COMMITS 0x7fffffff
#  else
#    define MAX_COMMITS 1
#  endif
#endif

#ifndef MAX_PAYLOAD
#define MAX_PAYLOAD 3
#endif

/* --- Node IDs --- */
#define NODE_GRAND   0
#define NODE_PARENT  1
#define NODE_CHILD1  2
#define NODE_CHILD2  3
#define NUM_NODES    4

/* bundled_by is a 2-bit field: 0=GRAND, 1=PARENT, 3=NULL_NODE (priority).
 * NODE_GRAND/NODE_PARENT happen to fit in those 2 bits already. */
#define NULL_NODE   0x3
#define W_NULL_SUB  0xF       /* 4-bit sentinel for sub / nested-sub fields */

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
 *   bits 34-35 : bundled_by (NULL_NODE=3 when has_priority)
 *   bits 36-39 : payload (0..14)
 *   bits 40-43 : sub[0]         (Parent: Child1 payload;  Grand: Parent payload)
 *   bits 44-47 : sub[1]         (Parent: Child2 payload;  Grand: unused)
 *   bits 48-51 : sub_nested[0]  (Grand only: Child1 payload under bundled Parent)
 *   bits 52-55 : sub_nested[1]  (Grand only: Child2 payload under bundled Parent)
 *   bits 56-63 : unused
 */
typedef struct {
    bool     has_priority;
    uint32_t serial;
    uint8_t  bundled_by;
    bool     missing;
    uint8_t  payload;
    uint8_t  sub0;              /* Parent: Child1 payload;  Grand: Parent payload */
    uint8_t  sub1;              /* Parent: Child2 payload;  Grand: unused */
    uint8_t  nested0;           /* Grand only: Child1 payload under bundled Parent */
    uint8_t  nested1;           /* Grand only: Child2 payload under bundled Parent */
} Wrapper;

static inline uint64_t wrapper_pack(Wrapper w) {
    uint64_t v = 0;
    v |= (uint64_t)w.serial;
    v |= ((uint64_t)(w.has_priority ? 1 : 0)) << 32;
    v |= ((uint64_t)(w.missing ? 1 : 0)) << 33;
    v |= ((uint64_t)(w.bundled_by & 0x3)) << 34;
    v |= ((uint64_t)(w.payload & 0xF)) << 36;
    v |= ((uint64_t)(w.sub0 & 0xF)) << 40;
    v |= ((uint64_t)(w.sub1 & 0xF)) << 44;
    v |= ((uint64_t)(w.nested0 & 0xF)) << 48;
    v |= ((uint64_t)(w.nested1 & 0xF)) << 52;
    return v;
}

static inline Wrapper wrapper_unpack(uint64_t v) {
    Wrapper w;
    w.serial             = (uint32_t)(v & 0xFFFFFFFFu);
    w.has_priority       = (v >> 32) & 1;
    w.missing            = (v >> 33) & 1;
    w.bundled_by         = (v >> 34) & 0x3;
    w.payload            = (v >> 36) & 0xF;
    w.sub0               = (v >> 40) & 0xF;
    w.sub1               = (v >> 44) & 0xF;
    w.nested0            = (v >> 48) & 0xF;
    w.nested1            = (v >> 52) & 0xF;
    return w;
}

/* --- Shared state --- */
static _Atomic(uint64_t) linkage[NUM_NODES];
static _Atomic(uint32_t) commit_count[NUM_NODES];
static _Atomic(bool)     g_stop;

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

/* Construct a priority wrapper for a LEAF child: no sub[] / sub_nested[]. */
static Wrapper priority_leaf(uint8_t payload, uint32_t serial, bool missing) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .missing = missing, .payload = payload,
        .sub0 = W_NULL_SUB, .sub1 = W_NULL_SUB,
        .nested0 = W_NULL_SUB, .nested1 = W_NULL_SUB,
    };
}

/* Construct a priority Parent wrapper: sub[] = (Child1 payload, Child2 payload). */
static Wrapper priority_parent(uint8_t payload, uint32_t serial,
                               uint8_t s0, uint8_t s1, bool missing) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .missing = missing, .payload = payload,
        .sub0 = s0, .sub1 = s1,
        .nested0 = W_NULL_SUB, .nested1 = W_NULL_SUB,
    };
}

/* Construct a priority Grand wrapper: sub[0] = Parent payload;
 * sub_nested = (Child1, Child2) payloads nested under the stored Parent packet. */
static Wrapper priority_grand(uint8_t payload, uint32_t serial,
                              uint8_t parent_sub,
                              uint8_t n0, uint8_t n1,
                              bool missing) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .missing = missing, .payload = payload,
        .sub0 = parent_sub, .sub1 = W_NULL_SUB,
        .nested0 = n0, .nested1 = n1,
    };
}

/* Construct a bundled-ref wrapper (has_priority=false). */
static Wrapper bundled_w(uint8_t parent_node, uint32_t serial) {
    return (Wrapper){
        .has_priority = false, .serial = serial, .bundled_by = parent_node,
        .missing = false, .payload = 0,
        .sub0 = W_NULL_SUB, .sub1 = W_NULL_SUB,
        .nested0 = W_NULL_SUB, .nested1 = W_NULL_SUB,
    };
}

static inline uint8_t get_child_sub(Wrapper w, int child_node) {
    return (child_node == NODE_CHILD1) ? w.sub0 : w.sub1;
}
static inline uint8_t get_child_nested(Wrapper w, int child_node) {
    return (child_node == NODE_CHILD1) ? w.nested0 : w.nested1;
}

/* =========================================================================
 * Bundle: inner level (Parent missing -> bundle Child1, Child2 under Parent)
 *
 * Phase1 collect: read cw1, cw2; compute sp1, sp2 (child payload if priority,
 *                 else fall back to pw.sub[i] if already bundled-by-Parent).
 * Phase2 CAS   : Parent -> (missing=true, sub=(sp1,sp2), serial=bundle_ser).
 * Phase3 CAS   : child -> bundled-by-Parent, ser=bundle_ser, one per child.
 *                On failure, return (restart from snap_check).
 * Phase4 CAS   : Parent missing=false.
 * Returns true on success (pw_out = finalized Parent wrapper).
 * ========================================================================= */
static bool try_inner_bundle(uint32_t *thread_ser, Wrapper *pw_out) {
    Wrapper pw = load_w(NODE_PARENT);
    if (!pw.has_priority) return false;
    if (!pw.missing) { *pw_out = pw; return true; }

    uint32_t bundle_ser = gen_serial(*thread_ser, pw.serial);
    *thread_ser = bundle_ser;

#if MODE == MODE_SUPERFINE
    if (pw.serial != bundle_ser) {
        Wrapper prestamp = priority_parent(pw.payload, bundle_ser,
                                           pw.sub0, pw.sub1, pw.missing);
        Wrapper exp = pw;
        if (!cas_w(NODE_PARENT, &exp, prestamp)) return false;
        pw = prestamp;
    }
#endif

    Wrapper cw1 = load_w(NODE_CHILD1);
    Wrapper cw2 = load_w(NODE_CHILD2);
    uint8_t sp1, sp2;
    if (cw1.has_priority) sp1 = cw1.payload;
    else if (cw1.bundled_by == NODE_PARENT && pw.sub0 != W_NULL_SUB) sp1 = pw.sub0;
    else sp1 = W_NULL_SUB;
    if (cw2.has_priority) sp2 = cw2.payload;
    else if (cw2.bundled_by == NODE_PARENT && pw.sub1 != W_NULL_SUB) sp2 = pw.sub1;
    else sp2 = W_NULL_SUB;
    if (sp1 == W_NULL_SUB || sp2 == W_NULL_SUB) return false;

    Wrapper p2 = priority_parent(pw.payload, bundle_ser, sp1, sp2, true);
    Wrapper exp_p2 = pw;
    if (!cas_w(NODE_PARENT, &exp_p2, p2)) return false;

    Wrapper b = bundled_w(NODE_PARENT, bundle_ser);
    Wrapper exp_c1 = cw1;
    bool ok1 = cas_w(NODE_CHILD1, &exp_c1, b);
    Wrapper exp_c2 = cw2;
    bool ok2 = ok1 ? cas_w(NODE_CHILD2, &exp_c2, b) : false;
    if (!ok1 || !ok2) return false;

    Wrapper p4 = priority_parent(p2.payload, bundle_ser, sp1, sp2, false);
    Wrapper exp_p4 = p2;
    if (!cas_w(NODE_PARENT, &exp_p4, p4)) return false;

    *pw_out = p4;
    return true;
}

/* =========================================================================
 * Bundle: outer level (Grand missing -> bundle Parent under Grand).
 *
 * Preconditions:
 *   Grand is priority, missing=true.
 *   Parent is either priority (possibly missing) OR bundled-by-Grand.
 * If Parent is missing, we call try_inner_bundle first so that Parent's
 * own sub-packets are present before we snapshot it into Grand.
 *
 * Phase2: CAS Grand -> (missing=true, sub[0]=parent.payload,
 *                      sub_nested=[parent.sub0, parent.sub1],
 *                      ser=bundle_ser).
 * Phase3: CAS Parent -> bundled-by-Grand.
 * Phase4: CAS Grand missing=false.
 * ========================================================================= */
static bool try_outer_bundle(uint32_t *thread_ser, Wrapper *gw_out) {
    Wrapper gw = load_w(NODE_GRAND);
    if (!gw.has_priority) return false;
    if (!gw.missing) { *gw_out = gw; return true; }

    Wrapper pw = load_w(NODE_PARENT);
    uint8_t parent_payload, nested0, nested1;

    if (pw.has_priority) {
        if (pw.missing) {
            Wrapper pw_fresh;
            if (!try_inner_bundle(thread_ser, &pw_fresh)) return false;
            pw = pw_fresh;
        }
        parent_payload = pw.payload;
        nested0 = pw.sub0;
        nested1 = pw.sub1;
    } else if (pw.bundled_by == NODE_GRAND) {
        if (gw.sub0 == W_NULL_SUB) return false;
        if (gw.nested0 == W_NULL_SUB || gw.nested1 == W_NULL_SUB) return false;
        parent_payload = gw.sub0;
        nested0 = gw.nested0;
        nested1 = gw.nested1;
    } else {
        return false;
    }
    if (parent_payload == W_NULL_SUB ||
        nested0 == W_NULL_SUB || nested1 == W_NULL_SUB) return false;

    uint32_t bundle_ser = gen_serial(*thread_ser, gw.serial);
    *thread_ser = bundle_ser;

#if MODE == MODE_SUPERFINE
    if (gw.serial != bundle_ser) {
        Wrapper prestamp = priority_grand(gw.payload, bundle_ser,
                                          gw.sub0,
                                          gw.nested0, gw.nested1,
                                          gw.missing);
        Wrapper exp = gw;
        if (!cas_w(NODE_GRAND, &exp, prestamp)) return false;
        gw = prestamp;
    }
#endif

    /* Phase2: CAS Grand with Parent's snapshot, missing=true. */
    Wrapper g2 = priority_grand(gw.payload, bundle_ser,
                                parent_payload, nested0, nested1,
                                true);
    Wrapper exp_g2 = gw;
    if (!cas_w(NODE_GRAND, &exp_g2, g2)) return false;

    /* Phase3: CAS Parent to bundled-by-Grand (only if Parent still matches
     * what we computed above — i.e. priority + same payload/subs). */
    if (pw.has_priority) {
        Wrapper bref = bundled_w(NODE_GRAND, bundle_ser);
        Wrapper exp_pw = pw;
        if (!cas_w(NODE_PARENT, &exp_pw, bref)) return false;
    }
    /* else: Parent already bundled-by-Grand; Phase3 is a no-op. */

    /* Phase4: CAS Grand missing=false. */
    Wrapper g4 = g2;
    g4.missing = false;
    Wrapper exp_g4 = g2;
    if (!cas_w(NODE_GRAND, &exp_g4, g4)) return false;

    *gw_out = g4;
    return true;
}

/* snapshot_grand(): loop until Grand is priority && !missing, with all
 * sub-packets (parent + nested) present. */
static void snapshot_grand(uint32_t *thread_ser, Wrapper *gw_out) {
    for (;;) {
        Wrapper gw = load_w(NODE_GRAND);
        if (gw.has_priority && !gw.missing
            && gw.sub0 != W_NULL_SUB
            && gw.nested0 != W_NULL_SUB
            && gw.nested1 != W_NULL_SUB) {
            *gw_out = gw;
            return;
        }
        if (gw.has_priority) {
            /* Grand is missing or sub-packets incomplete → (re-)bundle. */
            if (try_outer_bundle(thread_ser, gw_out)) return;
            continue;
        }
        /* Grand is never bundled (root). */
    }
}

/* =========================================================================
 * CommitGrand: snapshot Grand (forcing the full recursive bundle) then do
 * ONE CAS on Grand that simultaneously bumps BOTH leaf sub-packets
 * (sub_nested[0] and sub_nested[1]).  This matches the TLA+ CommitGrand
 * action which updates Grand.packet.sub[Parent].sub[c] for every c in
 * ParentChildren atomically.
 * ========================================================================= */
static void commit_grand(uint32_t *thread_ser) {
    OP_LOCK();
    for (;;) {
        Wrapper gw;
        snapshot_grand(thread_ser, &gw);
        assert(gw.has_priority && !gw.missing);
        assert(gw.sub0 != W_NULL_SUB);
        assert(gw.nested0 != W_NULL_SUB);
        assert(gw.nested1 != W_NULL_SUB);

        uint8_t n0 = (uint8_t)((gw.nested0 + 1) % MAX_PAYLOAD);
        uint8_t n1 = (uint8_t)((gw.nested1 + 1) % MAX_PAYLOAD);
        uint32_t ser = gen_serial(*thread_ser, gw.serial);
        Wrapper new_gw = priority_grand(gw.payload, ser,
                                        gw.sub0, n0, n1,
                                        false);
        Wrapper exp = gw;
        if (cas_w(NODE_GRAND, &exp, new_gw)) {
            *thread_ser = ser;
            atomic_fetch_add_explicit(&commit_count[NODE_CHILD1], 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&commit_count[NODE_CHILD2], 1, memory_order_relaxed);
            OP_UNLOCK();
            return;
        }
        /* Grand changed — retry from snapshot */
    }
}

/* =========================================================================
 * CommitChild: direct commit on a leaf; unbundle walk 1 or 2 levels deep
 * if the leaf is bundled.
 *
 * 1-level unbundle (Parent is priority, Child bundled-by-Parent):
 *   UnbundleCASAncestor: Parent missing=true, preserve sub[].
 *   UnbundleCASChild   : child -> priority, payload=(old_sub+1).
 *
 * 2-level unbundle (Parent bundled-by-Grand, Child bundled-by-Parent):
 *   UnbundleCASGP      : Grand missing=true, preserve sub[] / sub_nested[].
 *   UnbundleRestoreParent: Parent bundled-by-Grand -> priority, payload=
 *                          Grand.sub0, sub=(Grand.sub_nested[]),
 *                          missing=true.
 *   UnbundleCASChild   : child -> priority, payload=(old_sub+1).
 * ========================================================================= */
static void commit_child(int child_node, uint32_t *thread_ser) {
    OP_LOCK();
    for (;;) {
        Wrapper cw = load_w(child_node);

        if (cw.has_priority) {
            uint8_t new_payload = (uint8_t)((cw.payload + 1) % MAX_PAYLOAD);
            uint32_t ser = gen_serial(*thread_ser, cw.serial);
            Wrapper new_cw = priority_leaf(new_payload, ser, cw.missing);
            Wrapper exp = cw;
            if (cas_w(child_node, &exp, new_cw)) {
                *thread_ser = ser;
                atomic_fetch_add_explicit(&commit_count[child_node], 1, memory_order_relaxed);
                OP_UNLOCK();
                return;
            }
            continue;
        }

        if (cw.bundled_by != NODE_PARENT) continue;

        Wrapper pw = load_w(NODE_PARENT);

        if (pw.has_priority) {
            /* 1-level unbundle */
            uint8_t old_sub = get_child_sub(pw, child_node);
            if (old_sub == W_NULL_SUB) continue;
            uint8_t new_sub = (uint8_t)((old_sub + 1) % MAX_PAYLOAD);

            uint32_t anc_ser = gen_serial(*thread_ser, pw.serial);
            Wrapper new_pw = priority_parent(pw.payload, anc_ser,
                                             pw.sub0, pw.sub1, true);
            Wrapper exp_pw = pw;
            if (!cas_w(NODE_PARENT, &exp_pw, new_pw)) continue;
            *thread_ser = anc_ser;

            uint32_t child_ser = gen_serial(*thread_ser, cw.serial);
            Wrapper new_child = priority_leaf(new_sub, child_ser, false);
            Wrapper exp_child = cw;
            if (cas_w(child_node, &exp_child, new_child)) {
                *thread_ser = child_ser;
                atomic_fetch_add_explicit(&commit_count[child_node], 1, memory_order_relaxed);
                OP_UNLOCK();
                return;
            }
            continue;
        }

        if (pw.bundled_by != NODE_GRAND) continue;

        /* 2-level unbundle */
        Wrapper gw = load_w(NODE_GRAND);
        if (!gw.has_priority) continue;
        if (gw.sub0 == W_NULL_SUB) continue;
        uint8_t old_nested = get_child_nested(gw, child_node);
        if (old_nested == W_NULL_SUB) continue;
        uint8_t new_sub = (uint8_t)((old_nested + 1) % MAX_PAYLOAD);

        /* UnbundleCASGP: Grand missing=true (TLA+ UnbundleCASLoop). */
        uint32_t gp_ser = gen_serial(*thread_ser, gw.serial);
        Wrapper new_gw = priority_grand(gw.payload, gp_ser,
                                        gw.sub0,
                                        gw.nested0, gw.nested1,
                                        true);
        Wrapper exp_gw = gw;
        if (!cas_w(NODE_GRAND, &exp_gw, new_gw)) continue;
        *thread_ser = gp_ser;

        /* UnbundleRestoreParent: bundled Parent -> priority, missing=true. */
        uint32_t p_ser = gen_serial(*thread_ser, pw.serial);
        Wrapper restored_pw = priority_parent(gw.sub0, p_ser,
                                              gw.nested0, gw.nested1,
                                              true);
        Wrapper exp_pw = pw;
        if (!cas_w(NODE_PARENT, &exp_pw, restored_pw)) continue;
        *thread_ser = p_ser;

        /* UnbundleCASChild: child -> priority with new payload. */
        uint32_t c_ser = gen_serial(*thread_ser, cw.serial);
        Wrapper new_child = priority_leaf(new_sub, c_ser, false);
        Wrapper exp_child = cw;
        if (cas_w(child_node, &exp_child, new_child)) {
            *thread_ser = c_ser;
            atomic_fetch_add_explicit(&commit_count[child_node], 1, memory_order_relaxed);
            OP_UNLOCK();
            return;
        }
        /* child changed — retry */
    }
}

/* =========================================================================
 * Thread worker.
 * Stop flag checked only at the top of each iteration; once an iteration
 * is started, it runs to completion so that commit_count[] increments and
 * wrapper payload updates stay in lock-step.
 *
 * Per iteration per thread: commit_grand (cc[C1]+=1, cc[C2]+=1) +
 *   commit_child(C1) (cc[C1]+=1) + commit_child(C2) (cc[C2]+=1).
 *   Total per iter: cc[Ci] += 2.
 *
 * GenMC total (MAX_COMMITS=1): cc[Ci] = 2 * NUM_THREADS.
 * ========================================================================= */
static void *worker(void *arg) {
    (void)arg;
    uint32_t ser = 0;

    for (uint32_t iter = 0; iter < (uint32_t)MAX_COMMITS; iter++) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;
        commit_grand(&ser);
        commit_child(NODE_CHILD1, &ser);
        commit_child(NODE_CHILD2, &ser);
    }
    return NULL;
}

/* =========================================================================
 * Post-join invariant checks
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

    /* SnapshotConsistency: if !missing at an inner node, its sub-packets exist. */
    if (pw.has_priority && !pw.missing) {
        assert(pw.sub0 != W_NULL_SUB);
        assert(pw.sub1 != W_NULL_SUB);
    }
    if (gw.has_priority && !gw.missing) {
        assert(gw.sub0 != W_NULL_SUB);
    }
}

int main(void) {
    Wrapper init_grand  = priority_grand(0, 0, W_NULL_SUB, W_NULL_SUB, W_NULL_SUB, true);
    Wrapper init_parent = priority_parent(0, 0, W_NULL_SUB, W_NULL_SUB, true);
    Wrapper init_child  = priority_leaf(0, 0, false);
    atomic_store(&linkage[NODE_GRAND],  wrapper_pack(init_grand));
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

    uint32_t cc1 = atomic_load(&commit_count[NODE_CHILD1]);
    uint32_t cc2 = atomic_load(&commit_count[NODE_CHILD2]);

    /* TerminalPayloadCheck: each leaf is priority, payload == cc[c] % MaxPayload.
     * Stress mode: commit_count[] is ground truth (each started iter completes).
     * GenMC  mode: commit_count[] == 2 * MaxCommits * Threads exactly. */
    Wrapper c1 = load_w(NODE_CHILD1);
    Wrapper c2 = load_w(NODE_CHILD2);
    if (!(c1.has_priority && (uint32_t)c1.payload == cc1 % MAX_PAYLOAD) ||
        !(c2.has_priority && (uint32_t)c2.payload == cc2 % MAX_PAYLOAD)) {
        Wrapper gdbg = load_w(NODE_GRAND);
        Wrapper pdbg = load_w(NODE_PARENT);
        fprintf(stderr,
            "FAIL: MaxPayload=%d\n"
            "  Grand  : prio=%d missing=%d sub[0]=%u n0=%u n1=%u ser=%u\n"
            "  Parent : prio=%d missing=%d bundled_by=%u sub0=%u sub1=%u ser=%u\n"
            "  Child1 : prio=%d bundled_by=%u payload=%u ser=%u cc=%u (cc%%M=%u)\n"
            "  Child2 : prio=%d bundled_by=%u payload=%u ser=%u cc=%u (cc%%M=%u)\n",
            MAX_PAYLOAD,
            gdbg.has_priority, gdbg.missing, gdbg.sub0,
            gdbg.nested0, gdbg.nested1, gdbg.serial,
            pdbg.has_priority, pdbg.missing, pdbg.bundled_by,
            pdbg.sub0, pdbg.sub1, pdbg.serial,
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
    printf("[3level stress %s %ds] Child1=%u commits, Child2=%u commits (total=%u)\n",
           mode_str, STRESS_SECONDS, cc1, cc2, cc1 + cc2);
#else
    uint32_t expected_total = 2u * (uint32_t)MAX_COMMITS * (uint32_t)NUM_THREADS;
    assert(cc1 == expected_total);
    assert(cc2 == expected_total);
#endif

    return 0;
}
