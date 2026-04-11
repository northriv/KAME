/*
 * TLA+ → C11 mechanical translation: BundleUnbundle (Layer 2, 3-level)
 *
 * Generated from BundleUnbundle.tla @c11_action annotations.
 * Purpose: verify TLA+ model logic by compiling and running.
 * NOT for GenMC (too complex) — just logical correctness check.
 *
 * Abstracts atomic_shared_ptr (Layer 0) as correct atomic operations.
 * STM commit (Layer 1) as correct CAS on packed state.
 *
 * Tree: Grand -> Parent -> {Child1, Child2}
 *
 * Compile: gcc -std=c11 -pthread -o layer2_test genmc_layer2_from_tla.c
 * Run: ./layer2_test
 */

#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* === Configuration === */
#define MAX_SERIAL   6   /* even, for ModGT */
#define MAX_PAYLOAD  2
#define NUM_THREADS  2

/* === Modular arithmetic (matches TLA+ ModGT) === */
static int mod_gt(int a, int b) {
    int diff = ((a - b) + MAX_SERIAL) % MAX_SERIAL;
    return diff > 0 && diff < MAX_SERIAL / 2;
}

static int serial_succ(int s) {
    return (s + 1) % MAX_SERIAL;
}

/* === Node IDs === */
enum NodeId { GRAND = 0, PARENT = 1, CHILD1 = 2, CHILD2 = 3, NUM_NODES = 4 };

static const char *node_name[] = {"Grand", "Parent", "Child1", "Child2"};

/* === @c11_var linkage[n]: packed into atomic uint64_t === */
/*
 * PacketWrapper representation (packed):
 *   hasPriority: 1 bit
 *   bundledBy: 3 bits (node ID or 0xF for Null)
 *   serial: 8 bits
 *   missing: 1 bit
 *   payload: 8 bits
 *   sub[0] exists: 1 bit (for inner nodes)
 *   sub[1] exists: 1 bit
 *   sub[0] payload: 8 bits
 *   sub[1] payload: 8 bits
 *   sub[0] missing: 1 bit
 *   sub[1] missing: 1 bit
 *
 * Simplified: use a struct and atomic pointer to it.
 */

/* Use a simpler approach: each linkage is an atomic pointer to an
 * immutable PacketWrapper struct (COW semantics, like the real C++) */

#define MAX_CHILDREN 2
#define NULL_NODE 0xFF

typedef struct PacketWrapper {
    int hasPriority;
    int bundledBy;      /* node ID or NULL_NODE */
    int serial;
    int payload;
    int missing;
    int node;           /* which node this packet belongs to */
    /* Sub-packets (for inner nodes) */
    int has_sub[MAX_CHILDREN];
    int sub_payload[MAX_CHILDREN];
    int sub_missing[MAX_CHILDREN];
    /* Sub-sub-packets (for Grand's children's children) */
    int has_subsub[MAX_CHILDREN][MAX_CHILDREN];
    int subsub_payload[MAX_CHILDREN][MAX_CHILDREN];
} PacketWrapper;

/* Immutable pool (COW: allocate new, never modify) */
#define POOL_SIZE 4096
static PacketWrapper g_pool[POOL_SIZE];
static _Atomic(int) g_pool_next = 0;

static PacketWrapper *alloc_wrapper(void) {
    int idx = atomic_fetch_add(&g_pool_next, 1);
    assert(idx < POOL_SIZE);
    memset(&g_pool[idx], 0, sizeof(PacketWrapper));
    return &g_pool[idx];
}

static PacketWrapper *clone_wrapper(const PacketWrapper *src) {
    PacketWrapper *w = alloc_wrapper();
    *w = *src;
    return w;
}

/* @c11_var linkage[n]: atomic pointer to PacketWrapper */
static _Atomic(PacketWrapper *) linkage[NUM_NODES];

/* @c11_var serial[t]: thread-local Lamport clock */
static int thr_serial[NUM_THREADS];

/* Global serial (simplified Lamport) */
static _Atomic(int) global_serial = 0;

static int gen_serial(int t, int last_ser) {
    int base = mod_gt(last_ser, thr_serial[t]) ? last_ser : thr_serial[t];
    int s = serial_succ(base);
    thr_serial[t] = s;
    atomic_store(&global_serial, s);
    return s;
}

/* === Helper: parent/children relationships === */
static int parent_of(int n) {
    if (n == CHILD1 || n == CHILD2) return PARENT;
    if (n == PARENT) return GRAND;
    return NULL_NODE;
}

static int child_index(int parent, int child) {
    if (parent == GRAND && child == PARENT) return 0;
    if (parent == PARENT && child == CHILD1) return 0;
    if (parent == PARENT && child == CHILD2) return 1;
    return -1;
}

/* === Invariant checks === */

/* @invariant SnapshotConsistency: if hasPriority && !missing, all subs exist */
static void check_snapshot_consistency(void) {
    /* Check Grand */
    PacketWrapper *gw = atomic_load(&linkage[GRAND]);
    if (gw->hasPriority && !gw->missing) {
        assert(gw->has_sub[0]); /* Parent sub-packet exists */
    }
    /* Check Parent */
    PacketWrapper *pw = atomic_load(&linkage[PARENT]);
    if (pw->hasPriority && !pw->missing) {
        assert(pw->has_sub[0]); /* Child1 sub-packet */
        assert(pw->has_sub[1]); /* Child2 sub-packet */
    }
}

/* @invariant NoPriorityLoss: non-root has priority or bundledBy */
static void check_no_priority_loss(void) {
    for (int n = 1; n < NUM_NODES; n++) {
        PacketWrapper *w = atomic_load(&linkage[n]);
        assert(w->hasPriority || w->bundledBy != NULL_NODE);
    }
}

/* @invariant GrandAlwaysPriority */
static void check_grand_priority(void) {
    PacketWrapper *gw = atomic_load(&linkage[GRAND]);
    assert(gw->hasPriority);
}

/* @invariant BundledByCorrect: bundledBy points to structural parent */
static void check_bundled_by_correct(void) {
    for (int n = 1; n < NUM_NODES; n++) {
        PacketWrapper *w = atomic_load(&linkage[n]);
        if (!w->hasPriority) {
            assert(w->bundledBy == parent_of(n));
        }
    }
}

static void check_all_invariants(void) {
    check_snapshot_consistency();
    check_no_priority_loss();
    check_grand_priority();
    check_bundled_by_correct();
}

/* === @c11_action SnapRead + SnapCheck + Bundle + SnapDone === */
/* Simplified: snapshot an inner node, bundling if needed */
static int do_snapshot(int t, int node) {
    for (int retry = 0; retry < 100; retry++) {
        PacketWrapper *w = atomic_load(&linkage[node]);

        /* @c11_action SnapCheck */
        if (w->hasPriority && !w->missing) {
            /* Fast path: already bundled */
            return 1;
        }
        if (w->hasPriority && w->missing) {
            /* Need to bundle */
            int ser = gen_serial(t, w->serial);

            /* @c11_action BundlePhase1: collect children */
            int num_children = (node == GRAND) ? 1 : 2;
            PacketWrapper *child_wrappers[MAX_CHILDREN];
            int collected = 1;
            for (int i = 0; i < num_children; i++) {
                int child = (node == GRAND) ? PARENT : (CHILD1 + i);
                child_wrappers[i] = atomic_load(&linkage[child]);
                if (!child_wrappers[i]->hasPriority) {
                    if (child_wrappers[i]->bundledBy == node) {
                        /* Already bundled here — use existing sub */
                    } else {
                        collected = 0;
                        break;
                    }
                }
            }
            if (!collected) continue;

            /* @c11_action BundlePhase2: CAS node with new packet (missing=TRUE) */
            PacketWrapper *new_w = clone_wrapper(w);
            new_w->serial = ser;
            for (int i = 0; i < num_children; i++) {
                if (child_wrappers[i]->hasPriority) {
                    new_w->has_sub[i] = 1;
                    new_w->sub_payload[i] = child_wrappers[i]->payload;
                    new_w->sub_missing[i] = child_wrappers[i]->missing;
                }
            }
            new_w->missing = 1;

            PacketWrapper *expected = w;
            if (!atomic_compare_exchange_strong(&linkage[node], &expected, new_w))
                continue;

            /* @c11_action BundlePhase3: CAS children to BundledRef */
            int phase3_ok = 1;
            for (int i = 0; i < num_children; i++) {
                int child = (node == GRAND) ? PARENT : (CHILD1 + i);
                PacketWrapper *child_expected = child_wrappers[i];
                PacketWrapper *bundled = alloc_wrapper();
                bundled->hasPriority = 0;
                bundled->bundledBy = node;
                bundled->serial = ser;
                if (!atomic_compare_exchange_strong(&linkage[child],
                        &child_expected, bundled)) {
                    phase3_ok = 0;
                    break;
                }
            }
            if (!phase3_ok) continue;

            /* @c11_action BundlePhase4: finalize (missing=FALSE) */
            PacketWrapper *final_w = clone_wrapper(new_w);
            final_w->missing = 0;
            expected = new_w;
            if (!atomic_compare_exchange_strong(&linkage[node], &expected, final_w))
                continue;

            return 1;
        }
        /* Bundled elsewhere — retry */
    }
    return 0; /* gave up */
}

/* === @c11_action CommitStart + CommitRead + CommitTryCAS === */
static int do_commit(int t, int node) {
    for (int retry = 0; retry < 100; retry++) {
        /* @c11_action CommitRead */
        PacketWrapper *w = atomic_load(&linkage[node]);

        if (w->hasPriority) {
            /* @c11_action CommitTryCAS */
            int ser = gen_serial(t, w->serial);
            PacketWrapper *new_w = clone_wrapper(w);
            new_w->payload = (w->payload + 1) % MAX_PAYLOAD;
            new_w->serial = ser;

            PacketWrapper *expected = w;
            if (atomic_compare_exchange_strong(&linkage[node], &expected, new_w))
                return 1;

            /* Single-node optimization: adopt new children */
            PacketWrapper *cur = atomic_load(&linkage[node]);
            if (cur->hasPriority && cur->payload == w->payload) {
                new_w = clone_wrapper(cur);
                new_w->payload = (cur->payload + 1) % MAX_PAYLOAD;
                new_w->serial = gen_serial(t, cur->serial);
                expected = cur;
                if (atomic_compare_exchange_strong(&linkage[node], &expected, new_w))
                    return 1;
            }
            continue;
        }

        /* Bundled — need unbundle */
        /* @c11_action UnbundleWalk */
        int parent = w->bundledBy;
        if (parent == NULL_NODE) continue;

        PacketWrapper *pw = atomic_load(&linkage[parent]);
        if (!pw->hasPriority) {
            /* 2-level unbundle needed (parent also bundled) */
            /* @c11_action UnbundleCASGP */
            int gp = pw->bundledBy;
            if (gp == NULL_NODE) continue;
            PacketWrapper *gpw = atomic_load(&linkage[gp]);
            if (!gpw->hasPriority) continue;

            int pi = child_index(gp, parent);
            if (pi < 0 || !gpw->has_sub[pi]) continue;
            int ci = child_index(parent, node);
            /* Extract from GP's nested sub-packets */
            /* Simplified: just retry for now */
            continue;
        }

        /* @c11_action UnbundleCASAncestor: mark child slot Null in parent */
        int ci = child_index(parent, node);
        if (ci < 0 || !pw->has_sub[ci]) continue;

        int ser = gen_serial(t, pw->serial);
        PacketWrapper *new_pw = clone_wrapper(pw);
        new_pw->has_sub[ci] = 0;
        new_pw->missing = 1;
        new_pw->serial = ser;

        PacketWrapper *expected = pw;
        if (!atomic_compare_exchange_strong(&linkage[parent], &expected, new_pw))
            continue;

        /* @c11_action UnbundleCASChild: restore child to priority */
        PacketWrapper *child_w = alloc_wrapper();
        child_w->hasPriority = 1;
        child_w->bundledBy = NULL_NODE;
        child_w->payload = (pw->sub_payload[ci] + 1) % MAX_PAYLOAD;
        child_w->serial = gen_serial(t, w->serial);
        child_w->node = node;
        child_w->missing = (node == PARENT) ? 1 : 0;

        expected = w;
        if (atomic_compare_exchange_strong(&linkage[node], &expected, child_w))
            return 1;
    }
    return 0;
}

/* === Thread function === */
static void *thread_func(void *arg) {
    int t = (int)(intptr_t)arg;

    /* Alternate between snapshot and commit */
    for (int i = 0; i < 3; i++) {
        /* Snapshot Grand */
        do_snapshot(t, GRAND);
        check_all_invariants();

        /* Commit to a child */
        int target = (t == 0) ? CHILD1 : CHILD2;
        do_commit(t, target);
        check_all_invariants();
    }
    return NULL;
}

/* === Init (matches TLA+ Init) === */
static void init(void) {
    /* Grand: priority, missing=TRUE (no children collected yet) */
    PacketWrapper *gw = alloc_wrapper();
    gw->hasPriority = 1; gw->bundledBy = NULL_NODE;
    gw->serial = 0; gw->payload = 0; gw->missing = 1;
    gw->node = GRAND;
    atomic_store(&linkage[GRAND], gw);

    /* Parent: priority, missing=TRUE */
    PacketWrapper *pw = alloc_wrapper();
    pw->hasPriority = 1; pw->bundledBy = NULL_NODE;
    pw->serial = 0; pw->payload = 0; pw->missing = 1;
    pw->node = PARENT;
    atomic_store(&linkage[PARENT], pw);

    /* Children: priority, missing=FALSE (leaves) */
    for (int i = 0; i < 2; i++) {
        PacketWrapper *cw = alloc_wrapper();
        cw->hasPriority = 1; cw->bundledBy = NULL_NODE;
        cw->serial = 0; cw->payload = 0; cw->missing = 0;
        cw->node = CHILD1 + i;
        atomic_store(&linkage[CHILD1 + i], cw);
    }
}

int main(void) {
    init();
    check_all_invariants();

    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, thread_func, (void *)(intptr_t)i);

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    check_all_invariants();
    printf("Layer 2: All invariants passed.\n");
    return 0;
}
