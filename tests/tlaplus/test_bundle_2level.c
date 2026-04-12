/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
/*
 * C11 test generated mechanically from BundleUnbundle_2level.tla (Layer 2, 2-level)
 *
 * Tree structure:
 *   Parent --+-- Child1
 *            +-- Child2
 *
 * Models:
 *   - 4-phase bundle protocol (collect → CAS parent → CAS children → finalize)
 *   - Unbundle for commit (1-level walk)
 *   - Concurrent snapshot + commit interference
 *
 * TLA+ variable mapping:
 *   linkage[n]           → _Atomic(uint64_t) linkage[n]
 *     Packed as: (hasPriority:1 | serial:15 | payload:16 | sub_child1_present:1 |
 *                 sub_child2_present:1 | missing:1 | bundledBy:4 | reserved:25)
 *     For simplicity, we use a struct-based approach with per-field atomics.
 *
 *   Since TLA+ linkage is a complex record (packet with sub-packets, hasPriority,
 *   bundledBy, serial), we model each node's state as an atomic pointer to a
 *   versioned wrapper struct, using compare_exchange on the pointer.
 *
 *   serial[t]            → thread-local serial counter
 *   local[t].*           → thread-local snapshot state
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>
#include <string.h>

/* --- Node IDs --- */
#define NODE_PARENT 0
#define NODE_CHILD1 1
#define NODE_CHILD2 2
#define NUM_NODES   3
#define NULL_NODE   0xFF

/* --- Modular serial arithmetic --- */
#define MAX_SERIAL  16  /* must be even */
#define MAX_PAYLOAD 4

static inline bool mod_gt(uint8_t a, uint8_t b) {
    uint8_t diff = (a - b + MAX_SERIAL) % MAX_SERIAL;
    return diff > 0 && diff < MAX_SERIAL / 2;
}

static inline uint8_t gen_serial(uint8_t thread_ser, uint8_t last_ser) {
    uint8_t base = mod_gt(last_ser, thread_ser) ? last_ser : thread_ser;
    return (base + 1) % MAX_SERIAL;
}

/* --- Wrapper: represents a PacketWrapper for one node --- */
/* We use a 64-bit packed representation for atomic CAS:
 *   bits 63:    hasPriority (1=priority, 0=bundled ref)
 *   bits 62-56: serial (7 bits, enough for MAX_SERIAL=16)
 *   bits 55-48: bundledBy node id (0xFF = Null)
 *   bits 47:    missing flag
 *   bits 46-40: payload (7 bits)
 *   bits 39-32: sub_child1 payload (0xFF = Null)
 *   bits 31-24: sub_child2 payload (0xFF = Null)
 *   bits 23-16: sub_child1_missing
 *   bits 15-8:  sub_child2_missing
 *   bits 7-0:   reserved
 */

#define W_NULL_SUB 0xFF

typedef struct {
    bool     has_priority;
    uint8_t  serial;
    uint8_t  bundled_by;    /* NODE_PARENT etc., or NULL_NODE */
    bool     missing;
    uint8_t  payload;
    uint8_t  sub_child1;    /* payload of child1's sub-packet, W_NULL_SUB if absent */
    uint8_t  sub_child2;    /* payload of child2's sub-packet, W_NULL_SUB if absent */
} Wrapper;

/* Pack/unpack for atomic CAS */
static inline uint64_t wrapper_pack(Wrapper w) {
    uint64_t v = 0;
    v |= ((uint64_t)w.has_priority) << 63;
    v |= ((uint64_t)(w.serial & 0x7F)) << 56;
    v |= ((uint64_t)w.bundled_by) << 48;
    v |= ((uint64_t)w.missing) << 47;
    v |= ((uint64_t)w.payload) << 40;
    v |= ((uint64_t)w.sub_child1) << 32;
    v |= ((uint64_t)w.sub_child2) << 24;
    return v;
}

static inline Wrapper wrapper_unpack(uint64_t v) {
    Wrapper w;
    w.has_priority = (v >> 63) & 1;
    w.serial       = (v >> 56) & 0x7F;
    w.bundled_by   = (v >> 48) & 0xFF;
    w.missing      = (v >> 47) & 1;
    w.payload      = (v >> 40) & 0x7F;
    w.sub_child1   = (v >> 32) & 0xFF;
    w.sub_child2   = (v >> 24) & 0xFF;
    return w;
}

/* --- Shared state: one atomic word per node --- */
_Atomic(uint64_t) linkage[NUM_NODES];

/* --- Helpers to read/CAS a node's wrapper --- */
static Wrapper load_wrapper(int node) {
    return wrapper_unpack(atomic_load_explicit(&linkage[node], memory_order_acquire));
}

static bool cas_wrapper(int node, Wrapper *expected, Wrapper desired) {
    uint64_t exp_val = wrapper_pack(*expected);
    uint64_t des_val = wrapper_pack(desired);
    bool ok = atomic_compare_exchange_strong_explicit(&linkage[node],
            &exp_val, des_val, memory_order_acq_rel, memory_order_relaxed);
    if (!ok) *expected = wrapper_unpack(exp_val);
    return ok;
}

/* --- Initial wrapper constructors --- */
static Wrapper priority_wrapper(uint8_t payload, uint8_t serial,
                                 uint8_t sub1, uint8_t sub2, bool missing) {
    return (Wrapper){
        .has_priority = true, .serial = serial, .bundled_by = NULL_NODE,
        .missing = missing, .payload = payload,
        .sub_child1 = sub1, .sub_child2 = sub2
    };
}

static Wrapper bundled_ref_wrapper(uint8_t parent_node, uint8_t serial) {
    return (Wrapper){
        .has_priority = false, .serial = serial, .bundled_by = parent_node,
        .missing = false, .payload = 0,
        .sub_child1 = W_NULL_SUB, .sub_child2 = W_NULL_SUB
    };
}

/* ========================================================================== */
/* Snapshot: SnapRead → SnapCheck → [Bundle 4-phase] → SnapDone             */
/* ========================================================================== */

/* Perform a full snapshot of Parent, bundling children if needed.
 * Returns the parent's payload on success.
 * TLA+ actions: SnapRead, SnapCheck, BundlePhase1-4, SnapDone
 */
static uint8_t snapshot(uint8_t *thread_ser) {
    for (;;) {
        /* SnapCheck: read parent */
        Wrapper pw = load_wrapper(NODE_PARENT);

        if (pw.has_priority && !pw.missing) {
            /* Fast path: parent has all sub-packets */
            /* SnapshotConsistency: if not missing, subs must be present */
            assert(pw.sub_child1 != W_NULL_SUB);
            assert(pw.sub_child2 != W_NULL_SUB);
            return pw.payload;
        }

        if (pw.has_priority && pw.missing) {
            /* Need to bundle children */
            uint8_t bundle_ser = gen_serial(*thread_ser, pw.serial);
            *thread_ser = bundle_ser;

            /* BundlePhase1: collect sub-packets from children */
            Wrapper cw1 = load_wrapper(NODE_CHILD1);
            Wrapper cw2 = load_wrapper(NODE_CHILD2);
            uint8_t sp1 = W_NULL_SUB, sp2 = W_NULL_SUB;

            if (cw1.has_priority) {
                sp1 = cw1.payload;
            } else if (cw1.bundled_by == NODE_PARENT) {
                sp1 = pw.sub_child1;  /* use parent's existing sub-packet */
            }
            if (cw2.has_priority) {
                sp2 = cw2.payload;
            } else if (cw2.bundled_by == NODE_PARENT) {
                sp2 = pw.sub_child2;
            }

            if (sp1 == W_NULL_SUB || sp2 == W_NULL_SUB) {
                continue;  /* can't collect all → retry */
            }

            /* BundlePhase2: CAS parent with collected subs (still missing=TRUE) */
            Wrapper new_parent = priority_wrapper(pw.payload, bundle_ser,
                                                   sp1, sp2, true);
            Wrapper expected_parent = pw;
            if (!cas_wrapper(NODE_PARENT, &expected_parent, new_parent)) {
                continue;  /* disturbed → retry */
            }

            /* BundlePhase3: CAS each child to bundled-ref */
            Wrapper expected_c1 = cw1;
            Wrapper bundled1 = bundled_ref_wrapper(NODE_PARENT, bundle_ser);
            Wrapper expected_c2 = cw2;
            Wrapper bundled2 = bundled_ref_wrapper(NODE_PARENT, bundle_ser);

            if (!cas_wrapper(NODE_CHILD1, &expected_c1, bundled1) ||
                !cas_wrapper(NODE_CHILD2, &expected_c2, bundled2)) {
                /* Child modified → restart from phase1 */
                continue;
            }

            /* BundlePhase4: finalize — clear missing flag */
            Wrapper final_parent = priority_wrapper(new_parent.payload, bundle_ser,
                                                     sp1, sp2, false);
            Wrapper exp_p4 = new_parent;
            if (!cas_wrapper(NODE_PARENT, &exp_p4, final_parent)) {
                continue;  /* disturbed → retry */
            }

            /* SnapshotConsistency after finalize */
            assert(final_parent.sub_child1 != W_NULL_SUB);
            assert(final_parent.sub_child2 != W_NULL_SUB);
            return final_parent.payload;
        }

        /* Parent is bundled elsewhere (shouldn't happen for root) → retry */
        continue;
    }
}

/* ========================================================================== */
/* Commit on a child node                                                     */
/* ========================================================================== */

/*
 * Commit: increment child's payload by 1.
 * TLA+ actions: CommitStart, CommitRead, CommitTryCAS, UnbundleWalk,
 *               UnbundleCASAncestors, UnbundleCASChild, CommitDone
 *
 * Returns true on success, false on true conflict.
 */
static bool commit_child(int child_node, uint8_t *thread_ser) {
    for (;;) {
        /* CommitRead: read child's wrapper */
        Wrapper cw = load_wrapper(child_node);

        if (cw.has_priority) {
            /* Direct commit path: CommitTryCAS */
            uint8_t new_payload = (cw.payload + 1) % MAX_PAYLOAD;
            uint8_t ser = gen_serial(*thread_ser, cw.serial);
            Wrapper new_cw = priority_wrapper(new_payload, ser,
                                               cw.sub_child1, cw.sub_child2,
                                               cw.missing);
            Wrapper expected = cw;
            if (cas_wrapper(child_node, &expected, new_cw)) {
                *thread_ser = ser;
                return true;
            }
            /* CAS failed: check if payload unchanged (single-node optimization) */
            if (expected.has_priority && expected.payload == cw.payload) {
                /* Adopt new children, retry CAS */
                continue;
            }
            if (expected.has_priority) {
                /* True conflict */
                return false;
            }
            /* Got bundled → fall through to unbundle */
            cw = expected;
        }

        /* Unbundle path: child is bundled → walk to parent */
        /* UnbundleWalk */
        if (cw.bundled_by != NODE_PARENT) {
            continue;  /* unexpected → retry */
        }

        Wrapper parent_w = load_wrapper(NODE_PARENT);
        if (!parent_w.has_priority) {
            continue;  /* parent also bundled → retry */
        }

        uint8_t old_child_payload;
        if (child_node == NODE_CHILD1) {
            if (parent_w.sub_child1 == W_NULL_SUB) { continue; }
            old_child_payload = parent_w.sub_child1;
        } else {
            if (parent_w.sub_child2 == W_NULL_SUB) { continue; }
            old_child_payload = parent_w.sub_child2;
        }

        uint8_t new_child_payload = (old_child_payload + 1) % MAX_PAYLOAD;

        /* UnbundleCASAncestors: CAS parent to mark child's slot as Null */
        uint8_t ser = gen_serial(*thread_ser, parent_w.serial);
        Wrapper new_parent;
        if (child_node == NODE_CHILD1) {
            new_parent = priority_wrapper(parent_w.payload, ser,
                                           W_NULL_SUB, parent_w.sub_child2, true);
        } else {
            new_parent = priority_wrapper(parent_w.payload, ser,
                                           parent_w.sub_child1, W_NULL_SUB, true);
        }
        Wrapper exp_parent = parent_w;
        if (!cas_wrapper(NODE_PARENT, &exp_parent, new_parent)) {
            continue;  /* disturbed → retry */
        }
        *thread_ser = ser;

        /* UnbundleCASChild: restore child to priority with new packet */
        uint8_t child_ser = gen_serial(*thread_ser, cw.serial);
        Wrapper new_child = priority_wrapper(new_child_payload, child_ser,
                                              W_NULL_SUB, W_NULL_SUB, false);
        Wrapper exp_child = cw;
        if (cas_wrapper(child_node, &exp_child, new_child)) {
            *thread_ser = child_ser;
            return true;
        }
        /* Child changed → retry from CommitRead */
    }
}

/* ========================================================================== */
/* Test threads                                                               */
/* ========================================================================== */

/*
 * Thread 0: snapshot Parent (exercises the full bundle protocol)
 */
static void *thread_snapshot(void *arg) {
    (void)arg;
    uint8_t my_serial = 0;
    uint8_t result = snapshot(&my_serial);
    (void)result;
    return NULL;
}

/*
 * Thread 1: commit on Child1 (exercises commit + unbundle)
 */
static void *thread_commit(void *arg) {
    (void)arg;
    uint8_t my_serial = 0;
    bool ok = commit_child(NODE_CHILD1, &my_serial);
    (void)ok;
    return NULL;
}

int main(void) {
    /* Init: Parent has priority, missing=TRUE (no sub-packets yet)
     *       Child1, Child2 have priority, missing=FALSE (leaf nodes) */
    Wrapper init_parent = priority_wrapper(0, 0, W_NULL_SUB, W_NULL_SUB, true);
    Wrapper init_child1 = priority_wrapper(0, 0, W_NULL_SUB, W_NULL_SUB, false);
    Wrapper init_child2 = priority_wrapper(0, 0, W_NULL_SUB, W_NULL_SUB, false);

    atomic_store(&linkage[NODE_PARENT], wrapper_pack(init_parent));
    atomic_store(&linkage[NODE_CHILD1], wrapper_pack(init_child1));
    atomic_store(&linkage[NODE_CHILD2], wrapper_pack(init_child2));

    pthread_t t0, t1;
    pthread_create(&t0, NULL, thread_snapshot, NULL);
    pthread_create(&t1, NULL, thread_commit, NULL);

    pthread_join(t0, NULL);
    pthread_join(t1, NULL);

    /* --- Post-join safety invariants --- */

    /* SnapshotConsistency: if parent is not missing, all subs present */
    Wrapper pw = load_wrapper(NODE_PARENT);
    if (pw.has_priority && !pw.missing) {
        assert(pw.sub_child1 != W_NULL_SUB);
        assert(pw.sub_child2 != W_NULL_SUB);
    }

    /* NoPriorityLoss: each child is either priority or has bundledBy */
    for (int c = NODE_CHILD1; c <= NODE_CHILD2; c++) {
        Wrapper w = load_wrapper(c);
        assert(w.has_priority || w.bundled_by != NULL_NODE);
    }

    /* BundleRefConsistency: if child bundledBy==Parent, parent has priority */
    for (int c = NODE_CHILD1; c <= NODE_CHILD2; c++) {
        Wrapper w = load_wrapper(c);
        if (!w.has_priority && w.bundled_by == NODE_PARENT) {
            assert(pw.has_priority);
        }
    }

    return 0;
}
