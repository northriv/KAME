/*
 * TLA+ → C11 mechanical translation: stm_commit (Layer 1) v2
 *
 * Changes from v1:
 * - Assertions after each operation (not just main end)
 * - destroyed made _Atomic
 */

#include <pthread.h>
#include <stdatomic.h>
#include <assert.h>
#include <stdint.h>

#define MAX_VAL    3
#define MAX_SERIAL 6
#define NUM_THREADS 3
#define ITERATIONS  2

typedef struct { uint32_t val; uint32_t serial; } NodeState;

static _Atomic(uint64_t) g_node;
static _Atomic(int) total_commits;

static uint64_t pack(uint32_t val, uint32_t serial) {
    return ((uint64_t)serial << 32) | val;
}
static NodeState unpack(uint64_t packed) {
    NodeState s;
    s.val = (uint32_t)(packed & 0xFFFFFFFF);
    s.serial = (uint32_t)(packed >> 32);
    return s;
}

/* @invariant ValueBounded */
static void check_value_bounded(void) {
    NodeState s = unpack(atomic_load_explicit(&g_node, memory_order_relaxed));
    assert(s.val < MAX_VAL);
}

/* @invariant WriteReadConsistency: after commit, value is visible */
static void check_write_read(uint32_t written_val, uint32_t snap_serial) {
    NodeState cur = unpack(atomic_load_explicit(&g_node, memory_order_acquire));
    /* If we were the last writer (serial == snap_serial + 1), our value should be there */
    if (cur.serial == (snap_serial + 1) % MAX_SERIAL) {
        assert(cur.val == written_val);
    }
}

static NodeState take_snapshot(void) {
    uint64_t cur = atomic_load_explicit(&g_node, memory_order_acquire);
    return unpack(cur);
}

static int try_commit(NodeState snap, uint32_t new_val) {
    uint64_t expected = pack(snap.val, snap.serial);
    uint32_t new_serial = (snap.serial + 1) % MAX_SERIAL;
    uint64_t desired = pack(new_val % MAX_VAL, new_serial);

    if (atomic_compare_exchange_strong_explicit(&g_node, &expected, desired,
            memory_order_acq_rel, memory_order_acquire)) {
        /* @invariant ValueBounded — check immediately after commit */
        check_value_bounded();
        return 1;
    }
    return 0;
}

static void *thread_iterate_commit(void *arg) {
    int tid = (int)(intptr_t)arg;
    (void)tid;

    for (int i = 0; i < ITERATIONS; i++) {
        for (;;) {
            NodeState snap = take_snapshot();
            /* @invariant ValueBounded after snapshot */
            assert(snap.val < MAX_VAL);

            uint32_t new_val = (snap.val + 1) % MAX_VAL;

            if (try_commit(snap, new_val)) {
                atomic_fetch_add_explicit(&total_commits, 1,
                        memory_order_relaxed);
                /* @invariant WriteReadConsistency */
                check_write_read(new_val, snap.serial);
                break;
            }
        }
    }
    return NULL;
}

int main(void) {
    atomic_store_explicit(&g_node, pack(0, 0), memory_order_relaxed);
    atomic_store_explicit(&total_commits, 0, memory_order_relaxed);

    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, thread_iterate_commit,
                (void *)(intptr_t)i);
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    NodeState final_state = unpack(
            atomic_load_explicit(&g_node, memory_order_relaxed));
    int commits = atomic_load_explicit(&total_commits, memory_order_relaxed);

    /* @invariant CommitSerializes */
    assert(commits == NUM_THREADS * ITERATIONS);
    /* @invariant ValueBounded */
    assert(final_state.val < MAX_VAL);
    /* @invariant NoLostUpdate */
    assert(final_state.serial == (NUM_THREADS * ITERATIONS) % MAX_SERIAL);

    return 0;
}
