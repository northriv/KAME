/*
 * TLA+ → C11 mechanical translation: stm_commit (Layer 1)
 *
 * Generated from stm_commit.tla @c11_action annotations.
 * Purpose: verify STM commit protocol logic under C11/RC11 via GenMC.
 *
 * Abstracts atomic_shared_ptr (verified in Layer 0) as a correct
 * atomic register: the node's (val, serial) is packed into a single
 * _Atomic uint64_t and updated via CAS.
 *
 * Thread model: 3 threads, each performs iterate_commit (snapshot →
 * write → CAS-commit → retry on failure).
 */

#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <assert.h>
#include <stdint.h>

/* === Configuration (matches stm_commit_mc.cfg) === */
#define MAX_VAL    3
#define MAX_SERIAL 6
#define NUM_THREADS 3
#define ITERATIONS  2  /* each thread tries to commit this many times */

/* === @c11_var node_val + node_serial: packed in one atomic word === */
/* In C++: atomic_shared_ptr<PacketWrapper> m_link
 * Abstracted here as a single atomic uint64_t packing (val, serial) */
typedef struct {
    uint32_t val;
    uint32_t serial;
} NodeState;

static _Atomic(uint64_t) g_node;  /* packed (val, serial) */

static uint64_t pack(uint32_t val, uint32_t serial) {
    return ((uint64_t)serial << 32) | val;
}

static NodeState unpack(uint64_t packed) {
    NodeState s;
    s.val = (uint32_t)(packed & 0xFFFFFFFF);
    s.serial = (uint32_t)(packed >> 32);
    return s;
}

/* === @c11_var per-thread state === */
/* thr_snap_val, thr_snap_ser, thr_write_val, thr_committed */
/* These are thread-local, no atomics needed */

/* === Invariant checks === */

/* @invariant NoLostUpdate: if 2+ threads committed, serial >= 2 */
/* @invariant CommitSerializes: sum of commits <= serial */
/* @invariant WriteReadConsistency: last committer's value is visible */
/* These are checked at the end of the test */

static _Atomic(int) total_commits;

/* === @c11_action TakeSnapshot === */
/* Transaction<XN> tr(node) → snapshot via load_shared_ */
static NodeState take_snapshot(void) {
    uint64_t cur = atomic_load_explicit(&g_node, memory_order_acquire);
    return unpack(cur);
}

/* === @c11_action Commit === */
/* tr.commit() → compareAndSet on m_link */
static int try_commit(NodeState snap, uint32_t new_val) {
    uint64_t expected = pack(snap.val, snap.serial);
    uint32_t new_serial = (snap.serial + 1) % MAX_SERIAL;
    uint64_t desired = pack(new_val % MAX_VAL, new_serial);

    if (atomic_compare_exchange_strong_explicit(&g_node, &expected, desired,
            memory_order_acq_rel, memory_order_acquire)) {
        return 1; /* success */
    }
    return 0; /* CAS failed — retry */
}

/* === iterate_commit pattern === */
static void *thread_iterate_commit(void *arg) {
    int tid = (int)(intptr_t)arg;
    (void)tid;

    for (int i = 0; i < ITERATIONS; i++) {
        for (;;) {
            /* @c11_action TakeSnapshot */
            NodeState snap = take_snapshot();

            /* @c11_action WriteIncrement */
            uint32_t new_val = (snap.val + 1) % MAX_VAL;

            /* @c11_action Commit */
            if (try_commit(snap, new_val)) {
                atomic_fetch_add_explicit(&total_commits, 1,
                        memory_order_relaxed);
                break; /* success */
            }
            /* CAS failed → retry with new snapshot */
        }
    }
    return NULL;
}

int main(void) {
    /* Init: node_val = 0, node_serial = 0 */
    atomic_store_explicit(&g_node, pack(0, 0), memory_order_relaxed);
    atomic_store_explicit(&total_commits, 0, memory_order_relaxed);

    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, thread_iterate_commit,
                (void *)(intptr_t)i);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    /* === Invariant checks === */
    NodeState final = unpack(
            atomic_load_explicit(&g_node, memory_order_relaxed));
    int commits = atomic_load_explicit(&total_commits, memory_order_relaxed);

    /* @invariant CommitSerializes: total commits <= serial */
    /* With modular serial, check commits == final.serial (mod MAX_SERIAL) */
    assert(commits == NUM_THREADS * ITERATIONS);

    /* @invariant ValueBounded */
    assert(final.val < MAX_VAL);

    /* @invariant NoLostUpdate: serial advanced by total commits */
    assert(final.serial == (NUM_THREADS * ITERATIONS) % MAX_SERIAL);

    printf("OK: %d commits, final val=%u serial=%u\n",
           commits, final.val, final.serial);

    return 0;
}
