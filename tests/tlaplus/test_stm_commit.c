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
 * C11 test generated mechanically from stm_commit.tla (Layer 1)
 *
 * Models the optimistic concurrency control cycle:
 *   1. Snapshot: read current value+serial atomically
 *   2. Write: thread-local copy-on-write (increment)
 *   3. Commit: CAS on shared state
 *   4. Retry: on failure, re-snapshot and repeat
 *
 * TLA+ variable mapping:
 *   node_val      → lower bits of _Atomic(uint64_t) node
 *   node_serial   → upper bits of _Atomic(uint64_t) node
 *   thr_snap_val  → thread-local snap_val
 *   thr_snap_ser  → thread-local snap_ser
 *   thr_write_val → thread-local write_val
 *   thr_committed → thread-local committed counter
 *
 * Each thread does: snapshot → write(+1) → commit, with retry on conflict.
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>

/* --- Pack (serial, val) into a single uint64_t --- */
#define MAX_VAL    100
#define MAX_SERIAL 100

static inline uint64_t pack(uint32_t serial, uint32_t val) {
    return ((uint64_t)serial << 32) | (uint64_t)val;
}
static inline uint32_t unpack_val(uint64_t packed) {
    return (uint32_t)(packed & 0xFFFFFFFF);
}
static inline uint32_t unpack_serial(uint64_t packed) {
    return (uint32_t)(packed >> 32);
}

/* The shared node: atomic register holding (serial, val) */
_Atomic(uint64_t) node;

/* Per-thread committed counts (written only by owning thread, read post-join) */
_Atomic(uint32_t) committed[2];

/*
 * iterate_commit cycle for one thread.
 * TLA+ actions: TakeSnapshot → WriteIncrement → Commit (with retry)
 *
 * Performs NUM_ITERS iterations of snapshot-write-commit.
 */
#define NUM_ITERS 2

static void *thread_func(void *arg) {
    int tid = (int)(intptr_t)arg;
    uint32_t my_committed = 0;

    for (int iter = 0; iter < NUM_ITERS; iter++) {
        for (;;) {
            /* TakeSnapshot(t): read current node_val and node_serial */
            uint64_t snap = atomic_load_explicit(&node, memory_order_acquire);
            uint32_t snap_val = unpack_val(snap);
            uint32_t snap_ser = unpack_serial(snap);

            /* WriteIncrement(t): write_val = snap_val + 1 (thread-local) */
            uint32_t write_val = snap_val + 1;
            assert(write_val <= MAX_VAL);  /* ValueBounded */

            /* Commit(t): CAS node from (snap_ser, snap_val) → (snap_ser+1, write_val) */
            uint64_t expected = pack(snap_ser, snap_val);
            uint64_t desired  = pack(snap_ser + 1, write_val);

            if (atomic_compare_exchange_strong_explicit(&node, &expected, desired,
                    memory_order_acq_rel, memory_order_relaxed)) {
                /* CAS succeeded */
                my_committed++;

                /* SnapshotBeforeCommit: snap_ser < new serial */
                assert(snap_ser < snap_ser + 1);

                /* WriteReadConsistency: if we just committed, node_val == write_val
                 * (only valid if no other commit happened since — checked via serial) */
                uint64_t cur = atomic_load_explicit(&node, memory_order_acquire);
                if (unpack_serial(cur) == snap_ser + 1) {
                    assert(unpack_val(cur) == write_val);
                }
                break;
            }
            /* CAS failed → Commit fail path: re-snapshot (retry from top) */
        }
    }

    atomic_store_explicit(&committed[tid], my_committed, memory_order_relaxed);
    return NULL;
}

int main(void) {
    /* Init: node_val=0, node_serial=0 */
    atomic_store(&node, pack(0, 0));
    atomic_store(&committed[0], 0);
    atomic_store(&committed[1], 0);

    pthread_t t0, t1;
    pthread_create(&t0, NULL, thread_func, (void *)(intptr_t)0);
    pthread_create(&t1, NULL, thread_func, (void *)(intptr_t)1);

    pthread_join(t0, NULL);
    pthread_join(t1, NULL);

    /* --- Post-join invariants (safe to read multiple shared variables) --- */
    uint64_t final = atomic_load(&node);
    uint32_t final_val = unpack_val(final);
    uint32_t final_ser = unpack_serial(final);
    uint32_t c0 = atomic_load(&committed[0]);
    uint32_t c1 = atomic_load(&committed[1]);

    /* NoLostUpdate: if both threads committed, serial >= 2 */
    if (c0 > 0 && c1 > 0) {
        assert(final_ser >= 2);
    }

    /* CommitSerializes: total commits <= serial */
    assert(c0 + c1 <= final_ser);

    /* ValueBounded */
    assert(final_val <= MAX_VAL);

    /* With pure increment operations: node_val == node_serial */
    assert(final_val == final_ser);

    /* QuiescentCheck: all threads idle →
     *   SumSet(Threads) % MaxSerial == node_serial
     *   SumSet(Threads) % MaxVal == node_val */
    assert((c0 + c1) % MAX_SERIAL == final_ser);
    assert((c0 + c1) % MAX_VAL == final_val);

    return 0;
}
