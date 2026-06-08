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
 * C11 test generated mechanically from OrphanReuse_optC.tla.
 *
 * "Option C" single-chunk lifecycle (negative control): the owner thread
 * exits (clears BIT_OWNED) while cross-thread freers drain the live-slot
 * count.  The chunk is released exactly once — either by an empty owner
 * exit, or by the final cross-free whose atomicDecAndTest returns true
 * (count hit 0 AND owner already gone).  A double release is a bad_release.
 *
 * This is the scaffold port for the orphan family: a packed-atomic chunk
 * word (bit_owned + mask_cnt), an atomic release flag, an atomic
 * bad_release counter, a pthread worker harness with a per-thread
 * iteration budget, and a post-join terminal assert encoding the TLA+
 * safety invariants.
 *
 * TLA+ variable mapping:
 *   bit_owned    -> BIT_OWNED bit of _Atomic(uint32_t) g_chunk
 *   mask_cnt     -> low MASKCNT bits of _Atomic(uint32_t) g_chunk
 *   released     -> _Atomic(int) g_released (0/1, must be set once)
 *   bad_release  -> _Atomic(int) g_bad_release (must stay 0)
 *
 * TLA+ action mapping (atomicity preserved on the packed g_chunk word):
 *   OwnerExit_NonEmpty / OwnerExit_Empty   -> owner_exit()
 *       fetch_and(~BIT_OWNED) then branch on the live count seen.
 *   FreeDec                                -> free_dec(), simple count > 1 dec
 *   FreeDecToZero_Releases/_Spurious       -> free_dec_to_zero()
 *       atomicDecAndTest: dec the last slot; "test" = (count was 1 AND
 *       owner bit already clear) => this freer direct-releases.
 *
 * Init has mask_cnt = 2 (worst-case depth), so a single chunk needs
 * exactly two cross-frees plus one owner exit to fully drain.
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ---------------------------------------------------------------------- */
/* Configuration knobs (consistent with the kamestm reference idiom).     */
/* ---------------------------------------------------------------------- */
#ifndef NUM_THREADS
#define NUM_THREADS   2          /* unit default */
#endif

#ifndef MAX_COMMITS
#define MAX_COMMITS   1          /* per-thread iteration budget (chunks/thread) */
#endif

#ifndef STRESS_SECONDS
#define STRESS_SECONDS 0         /* 0 = bounded unit run; >0 = wall-clock stress */
#endif

/* Init constant from the TLA+ spec: mask_cnt = 2 (two live slots). */
#define INIT_MASK_CNT 2u

/* ---------------------------------------------------------------------- */
/* Packed chunk word: BIT_OWNED | mask_cnt.                                */
/* ---------------------------------------------------------------------- */
#define BIT_OWNED    ((uint32_t)0x80000000u)
#define MASKCNT_MASK ((uint32_t)0x7fffffffu)

static inline bool     chunk_owned(uint32_t w)   { return (w & BIT_OWNED) != 0u; }
static inline uint32_t chunk_cnt(uint32_t w)     { return w & MASKCNT_MASK; }

/* ---------------------------------------------------------------------- */
/* Per-chunk lifecycle state (the four TLA+ variables for one chunk).      */
/* Each "commit" iteration runs one independent chunk through its full     */
/* lifecycle, so the variables live in a per-chunk struct.                 */
/* ---------------------------------------------------------------------- */
typedef struct {
    _Atomic(uint32_t) word;        /* bit_owned + mask_cnt */
    _Atomic(int)      released;    /* 0 -> 1, exactly once */
    _Atomic(int)      bad_release; /* must remain 0 */
} chunk_t;

/* Globals describing the chunk currently in flight + the role coordination. */
static chunk_t        *g_chunks;          /* one slot per concurrent chunk */
static _Atomic(int)    g_global_bad;      /* OR of any chunk's bad_release */

/* ---------------------------------------------------------------------- */
/* Action: OwnerExit (fetch_and(~BIT_OWNED) + branch).                     */
/*                                                                         */
/* Atomically clears BIT_OWNED and observes the live count in the same     */
/* operation (mirrors the TLA+ single-step owner exit).  If the count is   */
/* already 0 at exit, the owner is the unique releaser (OwnerExit_Empty);  */
/* otherwise the chunk lingers OWNED-cleared and a later cross-free         */
/* releases it (OwnerExit_NonEmpty).                                        */
/* ---------------------------------------------------------------------- */
static void owner_exit(chunk_t *c) {
    uint32_t old = atomic_fetch_and_explicit(&c->word, ~BIT_OWNED,
                                              memory_order_acq_rel);
    /* old is the pre-clear word; we know bit_owned was set on entry. */
    assert(chunk_owned(old));                 /* owner runs exit once */
    uint32_t cnt = chunk_cnt(old);

    if (cnt == 0u) {
        /* OwnerExit_Empty: owner is the unique releaser. */
        int prev = atomic_exchange_explicit(&c->released, 1,
                                            memory_order_acq_rel);
        if (prev != 0) {
            /* Would have double-released -> the DEBUG ABORT path. */
            atomic_store_explicit(&c->bad_release, 1, memory_order_relaxed);
            atomic_store_explicit(&g_global_bad, 1, memory_order_relaxed);
        }
    }
    /* OwnerExit_NonEmpty: nothing else to do; final cross-free releases. */
}

/* ---------------------------------------------------------------------- */
/* Action: cross-thread free of one slot.                                  */
/*                                                                         */
/* Models atomicDecAndTest on mask_cnt.  Returns "true" (direct release)   */
/* iff this decrement brought the live count to 0 AND BIT_OWNED was        */
/* already clear in the same observed word (FreeDecToZero_Releases).  If   */
/* the count was > 1 it is a plain FreeDec.  If it hit 0 while still OWNED  */
/* it is FreeDecToZero_Spurious — the owner will release on exit.          */
/* ---------------------------------------------------------------------- */
static void free_one_slot(chunk_t *c) {
    for (;;) {
        uint32_t cur = atomic_load_explicit(&c->word, memory_order_acquire);
        uint32_t cnt = chunk_cnt(cur);
        assert(cnt > 0u);                     /* never over-free */

        uint32_t desired = (cur & BIT_OWNED) | (cnt - 1u);
        if (!atomic_compare_exchange_weak_explicit(
                &c->word, &cur, desired,
                memory_order_acq_rel, memory_order_acquire)) {
            continue;                          /* CAS lost a race; retry */
        }

        if (cnt - 1u == 0u && !chunk_owned(cur)) {
            /* FreeDecToZero_Releases: atomicDecAndTest returned true. */
            int prev = atomic_exchange_explicit(&c->released, 1,
                                                memory_order_acq_rel);
            if (prev != 0) {
                atomic_store_explicit(&c->bad_release, 1, memory_order_relaxed);
                atomic_store_explicit(&g_global_bad, 1, memory_order_relaxed);
            }
        }
        /* else: FreeDec, or FreeDecToZero_Spurious (owner still set). */
        return;
    }
}

/* ---------------------------------------------------------------------- */
/* Per-chunk terminal invariant (mirrors the TLA+ theorems after every     */
/* chunk has fully drained and the owner has exited).                      */
/* ---------------------------------------------------------------------- */
static void check_terminal_invariants(const chunk_t *c) {
    uint32_t w   = atomic_load_explicit(&c->word, memory_order_acquire);
    int rel      = atomic_load_explicit(&c->released, memory_order_acquire);
    int bad      = atomic_load_explicit(&c->bad_release, memory_order_acquire);

    /* Inv_NoBadRelease: \neg bad_release */
    assert(bad == 0);

    /* TypeOK: mask_cnt \in 0..2, bit_owned cleared at terminal. */
    assert(chunk_cnt(w) <= INIT_MASK_CNT);

    /* Inv_ReleasedSticky: released => (\neg bit_owned /\ mask_cnt = 0).
     * At a fully-drained terminal state the chunk must be released. */
    assert(!chunk_owned(w));
    assert(chunk_cnt(w) == 0u);
    assert(rel == 1);
}

/* ---------------------------------------------------------------------- */
/* Thread roles.                                                           */
/*                                                                         */
/* Thread 0 is the OWNER: for each chunk it performs the owner exit.       */
/* All threads (including 0) act as cross-thread freers, draining slots    */
/* via free_one_slot until INIT_MASK_CNT frees have been issued per chunk. */
/* A shared atomic per-chunk "frees remaining" counter hands out the       */
/* exactly-INIT_MASK_CNT free operations across the worker threads so the  */
/* count never under/over-flows (the TLA+ guards mask_cnt > 0).            */
/* ---------------------------------------------------------------------- */
typedef struct {
    int id;
} thr_arg_t;

/* Per-chunk dispenser of the remaining cross-free operations. */
static _Atomic(uint32_t) *g_free_budget;   /* one per chunk, init INIT_MASK_CNT */

static int  g_num_chunks;

static void *worker(void *p) {
    thr_arg_t *a = (thr_arg_t *)p;
    bool is_owner = (a->id == 0);

    for (int k = 0; k < g_num_chunks; k++) {
        chunk_t *c = &g_chunks[k];

        /* The owner thread performs the single owner exit for this chunk. */
        if (is_owner) {
            owner_exit(c);
        }

        /* Every thread races to claim and execute cross-free slots. */
        for (;;) {
            uint32_t budget = atomic_load_explicit(&g_free_budget[k],
                                                   memory_order_acquire);
            if (budget == 0u) break;
            if (!atomic_compare_exchange_weak_explicit(
                    &g_free_budget[k], &budget, budget - 1u,
                    memory_order_acq_rel, memory_order_acquire)) {
                continue;                      /* lost the claim; retry */
            }
            free_one_slot(c);                  /* we own one free op */
        }
    }
    return NULL;
}

/* ---------------------------------------------------------------------- */
/* One bounded round: set up g_num_chunks fresh chunks, run the workers,   */
/* join, and check the terminal invariant for every chunk.                 */
/* ---------------------------------------------------------------------- */
static void run_round(int num_chunks) {
    g_num_chunks = num_chunks;
    g_chunks      = calloc((size_t)num_chunks, sizeof(*g_chunks));
    g_free_budget = calloc((size_t)num_chunks, sizeof(*g_free_budget));
    assert(g_chunks && g_free_budget);

    for (int k = 0; k < num_chunks; k++) {
        /* Init: bit_owned = TRUE, mask_cnt = 2, released/bad = FALSE. */
        atomic_store_explicit(&g_chunks[k].word, BIT_OWNED | INIT_MASK_CNT,
                              memory_order_relaxed);
        atomic_store_explicit(&g_chunks[k].released, 0, memory_order_relaxed);
        atomic_store_explicit(&g_chunks[k].bad_release, 0, memory_order_relaxed);
        atomic_store_explicit(&g_free_budget[k], INIT_MASK_CNT,
                              memory_order_relaxed);
    }
    atomic_store_explicit(&g_global_bad, 0, memory_order_relaxed);

    pthread_t  th[NUM_THREADS];
    thr_arg_t  args[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        args[t].id = t;
        int rc = pthread_create(&th[t], NULL, worker, &args[t]);
        assert(rc == 0);
    }
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(th[t], NULL);
    }

    /* Post-join terminal invariants. */
    assert(atomic_load_explicit(&g_global_bad, memory_order_acquire) == 0);
    for (int k = 0; k < num_chunks; k++) {
        check_terminal_invariants(&g_chunks[k]);
    }

    free(g_chunks);
    free(g_free_budget);
    g_chunks = NULL;
    g_free_budget = NULL;
}

int main(void) {
    /* Each "commit" budget unit runs one chunk through its full lifecycle.
     * Spread MAX_COMMITS chunks across the worker threads concurrently. */
    if (STRESS_SECONDS > 0) {
        struct timespec start;
        clock_gettime(CLOCK_MONOTONIC, &start);
        unsigned long rounds = 0;
        for (;;) {
            run_round(MAX_COMMITS > 0 ? MAX_COMMITS : 1);
            rounds++;
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            if (now.tv_sec - start.tv_sec >= STRESS_SECONDS) break;
        }
        printf("OrphanReuse_optC stress OK: %lu rounds, %d threads\n",
               rounds, NUM_THREADS);
    } else {
        int total = MAX_COMMITS > 0 ? MAX_COMMITS : 1;
        run_round(total);
        printf("OrphanReuse_optC unit OK: %d chunk(s), %d threads\n",
               total, NUM_THREADS);
    }
    return 0;
}
