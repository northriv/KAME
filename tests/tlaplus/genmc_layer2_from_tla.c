/*
 * TLA+ → C11 mechanical translation: BundleUnbundle (Layer 2) v2
 *
 * Changes from v1:
 * - Invariant checks after every operation (not just thread end)
 * - UnbundleCASGP 2-level unbundle implemented (was retry fallback)
 * - destroyed made _Atomic
 * - ModGT for serial comparison
 */

#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SERIAL   6
#define MAX_PAYLOAD  2
#define NUM_THREADS  2
#define NULL_NODE    0xFF

enum NodeId { GRAND = 0, PARENT = 1, CHILD1 = 2, CHILD2 = 3, NUM_NODES = 4 };
#define MAX_CHILDREN 2

/* === Modular arithmetic === */
static int mod_gt(int a, int b) {
    int diff = ((a - b) + MAX_SERIAL) % MAX_SERIAL;
    return diff > 0 && diff < MAX_SERIAL / 2;
}

/* === PacketWrapper (immutable, COW) === */
typedef struct PacketWrapper {
    int hasPriority;
    int bundledBy;
    int serial;
    int payload;
    _Atomic(int) missing;  /* atomic for safe reads in invariant checks */
    int node;
    int has_sub[MAX_CHILDREN];
    int sub_payload[MAX_CHILDREN];
    int sub_missing[MAX_CHILDREN];
    int has_subsub[MAX_CHILDREN][MAX_CHILDREN];
    int subsub_payload[MAX_CHILDREN][MAX_CHILDREN];
} PacketWrapper;

#define POOL_SIZE 8192
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
    memcpy(w, src, sizeof(PacketWrapper));
    return w;
}

static _Atomic(PacketWrapper *) linkage[NUM_NODES];
static int thr_serial[NUM_THREADS];
static _Atomic(int) global_serial_val = 0;

static int gen_serial(int t, int last_ser) {
    int base = mod_gt(last_ser, thr_serial[t]) ? last_ser : thr_serial[t];
    int s = (base + 1) % MAX_SERIAL;
    thr_serial[t] = s;
    atomic_store(&global_serial_val, s);
    return s;
}

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

/* === Invariant checks (called after every operation) === */

static void check_all_invariants(void) {
    /* @invariant SnapshotConsistency */
    for (int n = 0; n < 2; n++) { /* GRAND and PARENT */
        PacketWrapper *w = atomic_load(&linkage[n]);
        if (w->hasPriority && !atomic_load(&w->missing)) {
            int nc = (n == GRAND) ? 1 : 2;
            for (int i = 0; i < nc; i++)
                assert(w->has_sub[i]);
        }
    }
    /* @invariant NoPriorityLoss */
    for (int n = 1; n < NUM_NODES; n++) {
        PacketWrapper *w = atomic_load(&linkage[n]);
        assert(w->hasPriority || w->bundledBy != NULL_NODE);
    }
    /* @invariant GrandAlwaysPriority */
    assert(atomic_load(&linkage[GRAND])->hasPriority);
    /* @invariant BundledByCorrect */
    for (int n = 1; n < NUM_NODES; n++) {
        PacketWrapper *w = atomic_load(&linkage[n]);
        if (!w->hasPriority)
            assert(w->bundledBy == parent_of(n));
    }
    /* @invariant BundleChainValid */
    for (int n = 1; n < NUM_NODES; n++) {
        PacketWrapper *w = atomic_load(&linkage[n]);
        if (!w->hasPriority && w->bundledBy != NULL_NODE) {
            PacketWrapper *pw = atomic_load(&linkage[w->bundledBy]);
            assert(pw->hasPriority || pw->bundledBy != NULL_NODE);
        }
    }
}

/* === Snapshot (with bundling) === */
static int do_snapshot(int t, int node) {
    for (int retry = 0; retry < 200; retry++) {
        PacketWrapper *w = atomic_load(&linkage[node]);
        if (w->hasPriority && !atomic_load(&w->missing)) {
            check_all_invariants();
            return 1;
        }
        if (w->hasPriority && atomic_load(&w->missing)) {
            int ser = gen_serial(t, w->serial);
            int num_children = (node == GRAND) ? 1 : 2;
            PacketWrapper *child_wrappers[MAX_CHILDREN];
            int collected = 1;
            for (int i = 0; i < num_children; i++) {
                int child = (node == GRAND) ? PARENT : (CHILD1 + i);
                child_wrappers[i] = atomic_load(&linkage[child]);
                if (!child_wrappers[i]->hasPriority) {
                    if (child_wrappers[i]->bundledBy == node) {
                        /* already bundled here */
                    } else { collected = 0; break; }
                }
            }
            if (!collected) continue;

            /* Phase 2: CAS parent */
            PacketWrapper *new_w = clone_wrapper(w);
            new_w->serial = ser;
            for (int i = 0; i < num_children; i++) {
                if (child_wrappers[i]->hasPriority) {
                    new_w->has_sub[i] = 1;
                    new_w->sub_payload[i] = child_wrappers[i]->payload;
                    new_w->sub_missing[i] = atomic_load(&child_wrappers[i]->missing);
                }
            }
            atomic_store(&new_w->missing, 1);
            PacketWrapper *expected = w;
            if (!atomic_compare_exchange_strong(&linkage[node], &expected, new_w))
                continue;
            check_all_invariants();

            /* Phase 3: CAS children */
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
                    phase3_ok = 0; break;
                }
            }
            if (!phase3_ok) continue;
            check_all_invariants();

            /* Phase 4: finalize */
            PacketWrapper *final_w = clone_wrapper(new_w);
            atomic_store(&final_w->missing, 0);
            expected = new_w;
            if (!atomic_compare_exchange_strong(&linkage[node], &expected, final_w))
                continue;
            check_all_invariants();
            return 1;
        }
        /* bundled elsewhere — retry */
    }
    return 0;
}

/* === Commit (with unbundling) === */
static int do_commit(int t, int node) {
    for (int retry = 0; retry < 200; retry++) {
        PacketWrapper *w = atomic_load(&linkage[node]);

        if (w->hasPriority) {
            /* Direct CAS commit */
            int ser = gen_serial(t, w->serial);
            PacketWrapper *new_w = clone_wrapper(w);
            new_w->payload = (w->payload + 1) % MAX_PAYLOAD;
            new_w->serial = ser;
            PacketWrapper *expected = w;
            if (atomic_compare_exchange_strong(&linkage[node], &expected, new_w)) {
                check_all_invariants();
                return 1;
            }
            /* Single-node optimization: adopt new children */
            PacketWrapper *cur = atomic_load(&linkage[node]);
            if (cur->hasPriority && cur->payload == w->payload) {
                new_w = clone_wrapper(cur);
                new_w->payload = (cur->payload + 1) % MAX_PAYLOAD;
                new_w->serial = gen_serial(t, cur->serial);
                expected = cur;
                if (atomic_compare_exchange_strong(&linkage[node], &expected, new_w)) {
                    check_all_invariants();
                    return 1;
                }
            }
            continue;
        }

        /* Bundled — unbundle */
        int par = w->bundledBy;
        if (par == NULL_NODE) continue;
        PacketWrapper *pw = atomic_load(&linkage[par]);

        if (pw->hasPriority) {
            /* 1-level unbundle: CAS parent, then restore child */
            int ci = child_index(par, node);
            if (ci < 0 || !pw->has_sub[ci]) continue;

            int ser = gen_serial(t, pw->serial);
            PacketWrapper *new_pw = clone_wrapper(pw);
            new_pw->has_sub[ci] = 0;
            atomic_store(&new_pw->missing, 1);
            new_pw->serial = ser;
            PacketWrapper *expected = pw;
            if (!atomic_compare_exchange_strong(&linkage[par], &expected, new_pw))
                continue;
            check_all_invariants();

            /* Restore child to priority */
            PacketWrapper *child_w = alloc_wrapper();
            child_w->hasPriority = 1;
            child_w->bundledBy = NULL_NODE;
            child_w->payload = (pw->sub_payload[ci] + 1) % MAX_PAYLOAD;
            child_w->serial = gen_serial(t, w->serial);
            child_w->node = node;
            atomic_store(&child_w->missing, (node == PARENT) ? 1 : 0);
            expected = w;
            if (atomic_compare_exchange_strong(&linkage[node], &expected, child_w)) {
                check_all_invariants();
                return 1;
            }
            continue;
        }

        /* 2-level unbundle: parent is also bundled (to grandparent) */
        int gp = pw->bundledBy;
        if (gp == NULL_NODE) continue;
        PacketWrapper *gpw = atomic_load(&linkage[gp]);
        if (!gpw->hasPriority) continue;

        int pi = child_index(gp, par);
        if (pi < 0 || !gpw->has_sub[pi]) continue;
        int ci = child_index(par, node);

        /* CAS grandparent: clear parent's child slot */
        int ser = gen_serial(t, gpw->serial);
        PacketWrapper *new_gpw = clone_wrapper(gpw);
        /* Modify the sub-packet for parent to clear child's slot */
        new_gpw->sub_missing[pi] = 1;
        atomic_store(&new_gpw->missing, 1);
        new_gpw->serial = ser;
        PacketWrapper *expected = gpw;
        if (!atomic_compare_exchange_strong(&linkage[gp], &expected, new_gpw))
            continue;
        check_all_invariants();

        /* Restore parent to priority */
        PacketWrapper *par_w = alloc_wrapper();
        par_w->hasPriority = 1;
        par_w->bundledBy = NULL_NODE;
        par_w->payload = gpw->sub_payload[pi];
        par_w->serial = gen_serial(t, pw->serial);
        par_w->node = par;
        atomic_store(&par_w->missing, 1);
        /* Copy sub-packets from GP's bundle */
        if (ci == 0) { par_w->has_sub[1] = 1; par_w->sub_payload[1] = gpw->sub_payload[pi]; }
        if (ci == 1) { par_w->has_sub[0] = 1; par_w->sub_payload[0] = gpw->sub_payload[pi]; }

        expected = pw;
        if (!atomic_compare_exchange_strong(&linkage[par], &expected, par_w)) {
            continue;
        }
        check_all_invariants();

        /* Restore child to priority with committed value */
        PacketWrapper *child_w = alloc_wrapper();
        child_w->hasPriority = 1;
        child_w->bundledBy = NULL_NODE;
        child_w->payload = (gpw->sub_payload[pi] + 1) % MAX_PAYLOAD; /* simplified */
        child_w->serial = gen_serial(t, w->serial);
        child_w->node = node;
        atomic_store(&child_w->missing, 0);
        expected = w;
        if (atomic_compare_exchange_strong(&linkage[node], &expected, child_w)) {
            check_all_invariants();
            return 1;
        }
    }
    return 0;
}

/* === Thread function === */
static void *thread_func(void *arg) {
    int t = (int)(intptr_t)arg;
    for (int i = 0; i < 3; i++) {
        do_snapshot(t, GRAND);
        check_all_invariants();
        int target = (t == 0) ? CHILD1 : CHILD2;
        do_commit(t, target);
        check_all_invariants();
        do_snapshot(t, PARENT);
        check_all_invariants();
        do_commit(t, PARENT);
        check_all_invariants();
    }
    return NULL;
}

static void init(void) {
    PacketWrapper *gw = alloc_wrapper();
    gw->hasPriority = 1; gw->bundledBy = NULL_NODE;
    gw->serial = 0; gw->payload = 0; atomic_store(&gw->missing, 1);
    gw->node = GRAND;
    atomic_store(&linkage[GRAND], gw);

    PacketWrapper *pw = alloc_wrapper();
    pw->hasPriority = 1; pw->bundledBy = NULL_NODE;
    pw->serial = 0; pw->payload = 0; atomic_store(&pw->missing, 1);
    pw->node = PARENT;
    atomic_store(&linkage[PARENT], pw);

    for (int i = 0; i < 2; i++) {
        PacketWrapper *cw = alloc_wrapper();
        cw->hasPriority = 1; cw->bundledBy = NULL_NODE;
        cw->serial = 0; cw->payload = 0; atomic_store(&cw->missing, 0);
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
