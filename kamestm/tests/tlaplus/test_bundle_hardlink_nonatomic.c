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
 * C11 test generated mechanically from BundleUnbundle_hardlink_nonatomic.tla.
 *
 * Abstract 3-node micro-model of the hard-link non-atomic test pattern
 * (commit b23fa954 on claude/refactor-negotiate-scoped-f7de2).  There is
 * NO packet pool, NO Lamport serial, NO negotiate tag, NO slot-pool — the
 * TLA+ model is purely an abstract scope/limbo state machine, so this port
 * deliberately stays minimal and does NOT force-fit the slot-pool idiom of
 * the other ports.
 *
 * Nodes per thread (logical): Root --hardlink--> A --sub--> C.
 *
 * Per-thread loop iteration (TLA+ DoStep1..DoStep4 + LoopIterEnd):
 *   ❶ NonTxInsertAC    : local op on A (no Root scope)            -> s1_done
 *   ❷ TxInsertHardlink : acquire Root scope (rootBusy), then drop -> s2_done
 *   ❸ NonTxReleaseAC   : local op on A; leaves C in LIMBO         -> s3_done
 *                        (cLimbo[t] := TRUE — C.bundledBy=A stale)
 *   ❹ TxReleaseHardlink: acquire Root scope, drop; A enters LIMBO -> s4_done
 *                        (aLimbo[t] := TRUE — A.bundledBy=Root stale)
 *   LoopIterEnd        : iterCount++ ; back to idle
 *
 * Destructor (TLA+ StartFinalizeC..FinalizeACas), fired once the per-thread
 * loop has drained (iterCount == MaxIter):
 *   FinalizeC  : resolve C's limbo.
 *                - walk variant (master, UseFixVariant=FALSE):
 *                    needs Root scope (~rootBusy) — walks bundledBy chain
 *                    C -> A -> Root.
 *                - cas  variant (b23fa954, UseFixVariant=TRUE):
 *                    direct CAS on C's linkage; no Root scope needed.
 *   FinalizeA  : resolve A's limbo (A.bundledBy=Root).
 *                - walk variant: needs Root scope (~rootBusy).
 *                - cas  variant: no Root scope needed.
 *
 * The single shared mutex `rootBusy` enforces RootMutex: at most one thread
 * may hold Root scope (be mid step ❷, mid step ❹, or mid a *_walk finalize)
 * at any instant.
 *
 * Property under test (TLA+ EventuallyAllDone): every thread's destructor
 * completes — every thread reaches pc="done" with cFinalized && aFinalized.
 * That is the terminal invariant asserted post-join.  Both variants are
 * live here because Root scope is held only transiently (acquire/release
 * within a single critical section), so a *_walk finalize always eventually
 * observes ~rootBusy.
 *
 * Compile-time knobs (consistent with the sibling ports):
 *   NUM_THREADS    (default 2)
 *   MAX_ITER       (per-thread loop iterations; default 1 unit, large stress)
 *   STRESS_SECONDS (default 0)
 *   USE_FIX_VARIANT(TLA+ UseFixVariant; default 0 = master walk variant)
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef NUM_THREADS
#define NUM_THREADS 2
#endif

#ifndef STRESS_SECONDS
#define STRESS_SECONDS 0
#endif

/* TLA+ MaxIter: per-thread loop iteration count.  Unit = 1, stress = large. */
#ifndef MAX_ITER
#  if STRESS_SECONDS > 0
#    define MAX_ITER 0x7fffffff
#  else
#    define MAX_ITER 1
#  endif
#endif

/* TLA+ UseFixVariant: 0 = master bundle-fallthrough (walk), 1 = b23fa954
 * self-promote (direct CAS).  Both must reach AllDone. */
#ifndef USE_FIX_VARIANT
#define USE_FIX_VARIANT 0
#endif

/* --- TLA+ pc steps --- */
enum step {
    PC_IDLE,
    PC_S1_DONE,
    PC_S2_IN_TX,
    PC_S2_DONE,
    PC_S3_DONE,
    PC_S4_IN_TX,
    PC_S4_DONE,
    PC_FIN_C_WALK,
    PC_FIN_C_CAS,
    PC_FIN_A_WALK,
    PC_FIN_A_CAS,
    PC_DONE
};

/* ============================================================================
 * Shared state (TLA+ VARIABLES).
 *   rootBusy : single mutex flag; TRUE while any thread holds Root scope.
 * Per-thread arrays are written only by the owning thread except where the
 * model would read them cross-thread (none here — they are purely local
 * progress flags), so plain atomics suffice for the post-join reads.
 * ============================================================================ */
static _Atomic(bool)     root_busy;              /* TLA+ rootBusy            */
static _Atomic(int)      pc[NUM_THREADS];        /* TLA+ pc[t]               */
static _Atomic(uint32_t) iter_count[NUM_THREADS];/* TLA+ iterCount[t]        */
static _Atomic(bool)     a_limbo[NUM_THREADS];   /* TLA+ aLimbo[t]           */
static _Atomic(bool)     c_limbo[NUM_THREADS];   /* TLA+ cLimbo[t]           */
static _Atomic(bool)     c_finalized[NUM_THREADS];/* TLA+ cFinalized[t]      */
static _Atomic(bool)     a_finalized[NUM_THREADS];/* TLA+ aFinalized[t]      */

static _Atomic(bool)     g_stop;

/* RootMutex live check: count of threads currently holding Root scope.
 * Must never exceed 1 (TLA+ invariant RootMutex). */
static _Atomic(int)      root_holders;

/* Diagnostics. */
static _Atomic(uint64_t) spin_root_acquire;  /* spins waiting for ~rootBusy  */
#define SPIN_INC(name) atomic_fetch_add_explicit(&(name), 1u, memory_order_relaxed)

static inline void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause");
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

/* Acquire Root scope: atomically flip rootBusy FALSE->TRUE.  This is the C11
 * realization of TLA+ "~rootBusy /\ rootBusy' = TRUE" — the enabling guard
 * and the state update fused into one atomic CAS so that RootMutex holds. */
static void acquire_root(void) {
    bool expected = false;
    while (!atomic_compare_exchange_weak_explicit(
               &root_busy, &expected, true,
               memory_order_acq_rel, memory_order_relaxed)) {
        expected = false;
        SPIN_INC(spin_root_acquire);
        cpu_relax();
        /* The holder always releases Root within a bounded critical section
         * (it never blocks while holding), so this spin terminates — this is
         * the C11 realization of the model's transient Root scope. */
    }
    int h = atomic_fetch_add_explicit(&root_holders, 1, memory_order_acq_rel) + 1;
    assert(h == 1);   /* TLA+ RootMutex: only one Root holder at a time. */
}

static void release_root(void) {
    atomic_fetch_sub_explicit(&root_holders, 1, memory_order_acq_rel);
    atomic_store_explicit(&root_busy, false, memory_order_release);
}

/* ============================================================================
 * One thread = one independent (Root_t, A_t, C_t) chain.  The TLA+ model is
 * fully partitioned per thread EXCEPT for the shared rootBusy mutex, so each
 * worker drives its own pc/limbo/finalize state and contends only on
 * rootBusy.  Mirrors the per-thread Next actions literally.
 * ============================================================================ */
static void *worker(void *arg) {
    int t = (int)(intptr_t)arg;

    /* --- Loop phase: DoStep1..DoStep4 + LoopIterEnd, MaxIter times. --- */
    while (atomic_load_explicit(&iter_count[t], memory_order_relaxed) < (uint32_t)MAX_ITER) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;

        /* ❶ DoStep1: pc idle -> s1_done.  Local A op, no Root scope. */
        atomic_store_explicit(&pc[t], PC_S1_DONE, memory_order_relaxed);

        /* ❷ DoStep2Enter/Exit: TX touching Root.  Acquire then release. */
        acquire_root();
        atomic_store_explicit(&pc[t], PC_S2_IN_TX, memory_order_relaxed);
        /* (mid-tx body is abstract: nothing observable) */
        atomic_store_explicit(&pc[t], PC_S2_DONE, memory_order_relaxed);
        release_root();

        /* ❸ DoStep3: pc s2_done -> s3_done.  Local A op; C enters LIMBO. */
        atomic_store_explicit(&c_limbo[t], true, memory_order_relaxed);
        atomic_store_explicit(&pc[t], PC_S3_DONE, memory_order_relaxed);

        /* ❹ DoStep4Enter/Exit: TX touching Root.  A enters LIMBO on exit. */
        acquire_root();
        atomic_store_explicit(&pc[t], PC_S4_IN_TX, memory_order_relaxed);
        atomic_store_explicit(&a_limbo[t], true, memory_order_relaxed);
        atomic_store_explicit(&pc[t], PC_S4_DONE, memory_order_relaxed);
        release_root();

        /* LoopIterEnd: iterCount++ ; back to idle. */
        atomic_fetch_add_explicit(&iter_count[t], 1u, memory_order_relaxed);
        atomic_store_explicit(&pc[t], PC_IDLE, memory_order_relaxed);
    }

    /* --- Destructor phase: fires once iterCount == MaxIter (loop drained).
     * StartFinalizeC: pc idle -> fin_c_{cas,walk}. --- */
    atomic_store_explicit(&pc[t],
        USE_FIX_VARIANT ? PC_FIN_C_CAS : PC_FIN_C_WALK, memory_order_relaxed);

#if USE_FIX_VARIANT
    /* FinalizeCCas: direct CAS on C's linkage only — no Root scope. */
    atomic_store_explicit(&c_limbo[t], false, memory_order_relaxed);
    atomic_store_explicit(&c_finalized[t], true, memory_order_relaxed);
    atomic_store_explicit(&pc[t], PC_FIN_A_CAS, memory_order_relaxed);

    /* FinalizeACas: direct CAS on A's linkage — no Root scope. */
    atomic_store_explicit(&a_limbo[t], false, memory_order_relaxed);
    atomic_store_explicit(&a_finalized[t], true, memory_order_relaxed);
    atomic_store_explicit(&pc[t], PC_DONE, memory_order_relaxed);
#else
    /* FinalizeCWalk: C.bundledBy=A -> A.bundledBy=Root -> needs Root scope.
     * Walk acquires Root scope, resolves C's limbo, releases. */
    acquire_root();
    atomic_store_explicit(&c_limbo[t], false, memory_order_relaxed);
    atomic_store_explicit(&c_finalized[t], true, memory_order_relaxed);
    atomic_store_explicit(&pc[t], PC_FIN_A_WALK, memory_order_relaxed);
    release_root();

    /* FinalizeAWalk: A.bundledBy=Root -> needs Root scope. */
    acquire_root();
    atomic_store_explicit(&a_limbo[t], false, memory_order_relaxed);
    atomic_store_explicit(&a_finalized[t], true, memory_order_relaxed);
    atomic_store_explicit(&pc[t], PC_DONE, memory_order_relaxed);
    release_root();
#endif

    return NULL;
}

/* ============================================================================
 * Post-join invariant checks (TLA+ AllDone + RootMutex terminal state).
 * Only invoked in the bounded/unit build; quiet -Wunused-function in the
 * STRESS_SECONDS>0 build where threads may be stopped before draining. */
__attribute__((unused))
static void check_invariants(void) {
    /* Quiescent: no thread holds Root scope, rootBusy clear. */
    assert(atomic_load(&root_holders) == 0);
    assert(atomic_load(&root_busy) == false);

    for (int t = 0; t < NUM_THREADS; t++) {
        /* TLA+ AllDone: cFinalized[t] /\ aFinalized[t]. */
        assert(atomic_load(&c_finalized[t]) == true);
        assert(atomic_load(&a_finalized[t]) == true);
        /* Destructor resolved both limbo states. */
        assert(atomic_load(&c_limbo[t]) == false);
        assert(atomic_load(&a_limbo[t]) == false);
        /* Thread reached terminal pc. */
        assert(atomic_load(&pc[t]) == PC_DONE);
        /* Loop fully drained. */
        assert(atomic_load(&iter_count[t]) == (uint32_t)MAX_ITER);
    }
}

int main(void) {
    /* Init (TLA+ Init). */
    atomic_store(&root_busy, false);
    atomic_store(&root_holders, 0);
    atomic_store(&g_stop, false);
    for (int t = 0; t < NUM_THREADS; t++) {
        atomic_store(&pc[t], PC_IDLE);
        atomic_store(&iter_count[t], 0u);
        atomic_store(&a_limbo[t], false);
        atomic_store(&c_limbo[t], false);
        atomic_store(&c_finalized[t], false);
        atomic_store(&a_finalized[t], false);
    }

    pthread_t threads[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_create(&threads[t], NULL, worker, (void *)(intptr_t)t);
    }

#if STRESS_SECONDS > 0
    struct timespec ts = { .tv_sec = STRESS_SECONDS, .tv_nsec = 0 };
    nanosleep(&ts, NULL);
    atomic_store_explicit(&g_stop, true, memory_order_release);
#endif

    for (int t = 0; t < NUM_THREADS; t++) pthread_join(threads[t], NULL);

#if STRESS_SECONDS > 0
    /* In stress mode threads may have been stopped mid-loop before draining,
     * so AllDone need not hold.  Only RootMutex (quiescent) is guaranteed.
     * Drive each thread's finalize deterministically here is NOT done — we
     * simply report and check the mutex invariant. */
    assert(atomic_load(&root_holders) == 0);
    assert(atomic_load(&root_busy) == false);
    const char *variant = USE_FIX_VARIANT ? "fix(cas)" : "master(walk)";
    uint64_t done = 0;
    for (int t = 0; t < NUM_THREADS; t++)
        if (atomic_load(&pc[t]) == PC_DONE) done++;
    printf("[hardlink_nonatomic stress %ds variant=%s threads=%d] "
           "done=%llu/%d root_acquire_spins=%llu\n",
           STRESS_SECONDS, variant, NUM_THREADS,
           (unsigned long long)done, NUM_THREADS,
           (unsigned long long)atomic_load(&spin_root_acquire));
#else
    /* Unit/bounded: every thread must reach AllDone (TLA+ EventuallyAllDone). */
    check_invariants();
#endif

    return 0;
}
