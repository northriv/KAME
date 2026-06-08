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
 * C11 test generated mechanically from ChunkRecycle_microscopic.tla.
 *
 * Microscopic model of kamepoolalloc's ADDRESS-LEVEL chunk lookup during
 * reclaim + recycle (allocator.cpp lookup_chunk / deallocate vs
 * deallocate_chunk / allocate_chunk).  Hunts the sister investigation's
 * bit-state = -3 signature: a deallocate(p) computing an out-of-range slot
 * index because address p's chunk was RECLAIMED and RECYCLED at a different
 * layout while the deallocate's two unsynchronised loads (back_offset,
 * palloc) were in flight.
 *
 * ============================================================
 *  TLA+ -> C11 mapping
 * ============================================================
 *
 * Shared "memory" (per-unit metadata lookup_chunk reads / reclaim+recycle
 * write).  One region of NUM_UNITS units.  Each is an _Atomic so the
 * deallocate side's two loads can observe writes from a concurrent
 * reclaim/recycle slipping between them:
 *
 *   claim[u]     -> _Atomic(bool)     g_claim[u]
 *   ready[u]     -> _Atomic(bool)     g_ready[u]
 *   backOff[u]   -> _Atomic(int)      g_backoff[u]   (load #1)
 *   palloc[u]    -> _Atomic(int)      g_palloc[u]    (load #2; gen id, 0=free)
 *   chunkSpan[u] -> _Atomic(int)      g_span[u]
 *   gen          -> _Atomic(int)      g_gen          (monotone, next gen id)
 *   allocGenOf[u]-> _Atomic(int)      g_alloc_gen[u] (ghost; NULL_GEN=Null)
 *   staleRead    -> _Atomic(bool)     g_stale_read   (THE HUNTED BUG flag)
 *
 * TLA+ actions -> C functions (each TLA+ pc-edge = one atomic memory op):
 *   A_Allocate(t) -> do_allocate()   (claim+backoff+header+ready, one step;
 *                                     registers a live slot ghost at base)
 *   R_Reclaim(t)  -> do_reclaim()    (ready=F; palloc/span=0; backoff=0;
 *                                     claim=F -- four writes, applied as the
 *                                     TLA+ action does, between which a peer
 *                                     thread's d_loadback / d_loadpal can run)
 *   D_Start(t)    -> d_start()       (capture dUnit/dExpectGen; FreeBeforeLookup
 *                                     => free the slot ghost now -> chunk
 *                                     reclaimable mid-lookup)
 *   D_LoadBack(t) -> d_loadback()    (load #1: base_off=backOff[uP];
 *                                     base=uP-base_off)
 *   D_LoadPal(t)  -> d_loadpal()     (load #2: palloc_v=palloc[base])
 *   D_Resolve(t)  -> d_resolve()     (palloc_v==0 => foreign/safe; else if
 *                                     palloc_v != dExpectGen => staleRead=TRUE;
 *                                     !FreeBeforeLookup => free slot ghost here)
 *
 * Atomicity granularity: the TLA+ spec exposes the deallocate side at one
 * atomic op per pc-edge (D_Start/LoadBack/LoadPal/Resolve) and applies the
 * allocate/reclaim writes as a single TLA+ action each.  The model has NO
 * MODE_COARSE/FINE/SUPERFINE variants, so this port omits them.  Each
 * per-thread deallocate is driven through its 4 pc-edges as 4 distinct
 * scheduling points; allocate/reclaim run as their single atomic step.
 *
 * ============================================================
 *  CONFIG knobs (TLA+ CONSTANTS)
 * ============================================================
 *   FreeBeforeLookup (default 1, PERMISSIVE):  D_Start frees the slot
 *       immediately -> chunk reclaimable mid-lookup -> exposes
 *       lookup_chunk's missing generation check.  staleRead CAN fire.
 *   FreeBeforeLookup=0 (FAITHFUL): slot stays live until D_Resolve.
 *   SinglePayout (default 0): if 1, forbid two threads mid-deallocate of
 *       the SAME unit (= no double-payout of a slot to two threads).
 *
 *   The cfg header comments call FAITHFUL-alone a negative control, but
 *   that is wrong: TLC on this very spec (verified, see the NEG_CONTROL
 *   table at the assertion site) reports Inv_NoStaleRead VIOLATED for
 *   every config EXCEPT (FreeBeforeLookup=0 AND SinglePayout=1).  Reason:
 *   with double-payout allowed, one thread's faithful D_Resolve clears the
 *   slot, letting the chunk be reclaimed+recycled while a peer's lookup is
 *   mid-flight.  So the ONLY config where staleRead must stay FALSE -- the
 *   true negative control -- is FAITHFUL AND SinglePayout together.
 *
 * ============================================================
 *  Terminal / safety invariant (post-join assert)
 * ============================================================
 * The SANITY invariants Inv_LiveSlotImpliesClaimed and
 * Inv_BackOffPointsToBase hold in ALL configurations and are asserted
 * post-join (and continuously during the run) -- these are the model's
 * always-true terminal safety invariants.  Inv_NoStaleRead
 * (staleRead == FALSE) is the bug-detector: asserted ONLY under the true
 * negative-control config (FAITHFUL && SinglePayout); under every other
 * config staleRead becoming TRUE is the EXPECTED hunted bug (matching
 * TLC's counterexample) and is reported as a diagnostic, not asserted.
 *
 * Compile:
 *   cc -std=c11 -O2 -Wall -Wextra -pthread test_ChunkRecycle_microscopic.c
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* --- TLA+ CONSTANTS as compile-time knobs --- */
#ifndef NUM_THREADS
#define NUM_THREADS 2
#endif

#ifndef NUM_UNITS
#define NUM_UNITS 2            /* TLA+ NumUnits (recommend 2) */
#endif

#ifndef MAX_GEN
#define MAX_GEN 3              /* TLA+ MaxGen (cfg uses 3) */
#endif

/* FreeBeforeLookup: 1 = PERMISSIVE (default, exposes the bug),
 *                   0 = FAITHFUL (negative control, no bug). */
#ifndef FREE_BEFORE_LOOKUP
#define FREE_BEFORE_LOOKUP 1
#endif

/* SinglePayout: 1 = forbid two threads mid-dealloc of the same unit
 *               (negative control); 0 = allow (bug appears). */
#ifndef SINGLE_PAYOUT
#define SINGLE_PAYOUT 0
#endif

#ifndef STRESS_SECONDS
#define STRESS_SECONDS 0
#endif

/* Per-round step budget.  The TLA+ Next relation is a single nondet step;
 * the C harness drives a bounded number of microscopic steps per round to
 * widen the interleaving search, resetting the region each round.  Default
 * MAX_COMMITS=1 reproduces a minimal bounded run; large for stress. */
#ifndef MAX_COMMITS
#  if STRESS_SECONDS > 0
#    define MAX_COMMITS 0x7fffffff
#  else
#    define MAX_COMMITS 1
#  endif
#endif

/* Steps each worker performs per round before the round resets. */
#ifndef STEPS_PER_ROUND
#define STEPS_PER_ROUND 64
#endif

/* TLA+ Null for generations (gen ids start at 1, so 0 is a safe sentinel
 * distinct from any live generation; palloc 0 = released/free). */
#define NULL_GEN 0

/* TLA+ Null for the per-thread deallocate registers (dUnit / dBase ...). */
#define NULL_UNIT (-1)

/* ============================================================
 * Shared per-unit metadata (TLA+ VARIABLES).
 * ============================================================ */
static _Atomic(bool) g_claim[NUM_UNITS];      /* claim[u]     */
static _Atomic(bool) g_ready[NUM_UNITS];      /* ready[u]     */
static _Atomic(int)  g_backoff[NUM_UNITS];    /* backOff[u]   (load #1) */
static _Atomic(int)  g_palloc[NUM_UNITS];     /* palloc[u]    (load #2; gen) */
static _Atomic(int)  g_span[NUM_UNITS];       /* chunkSpan[u] */
static _Atomic(int)  g_gen;                   /* gen (monotone next id) */
static _Atomic(int)  g_alloc_gen[NUM_UNITS];  /* allocGenOf[u] ghost */

/* THE HUNTED BUG: a deallocate resolved a non-foreign chunk whose
 * generation != the address's allocation generation (TLA+ staleRead). */
static _Atomic(bool) g_stale_read;

/* Per-thread deallocate-in-flight registers (TLA+ dUnit/dBaseOff/dBase/
 * dPalloc/dExpectGen + pc).  Indexed by thread slot. */
typedef enum {
    PC_IDLE = 0,
    PC_D_LOADBACK,
    PC_D_LOADPAL,
    PC_D_RESOLVE
} Pc;

typedef struct {
    int tid;          /* 1-indexed (matches TLA+ Threads = {1,2,...}) */
    unsigned rng;     /* per-thread PRNG for the TLA+ \E nondeterminism */
    /* deallocate registers (TLA+ d* state for this thread) */
    Pc  pc;
    int d_unit;       /* dUnit[t]      */
    int d_baseoff;    /* dBaseOff[t]   */
    int d_base;       /* dBase[t]      */
    int d_palloc;     /* dPalloc[t]    */
    int d_expect_gen; /* dExpectGen[t] */
} ThreadCtx;

/* dUnit[t] published shared so SinglePayout's cross-thread guard
 * (\A other: dUnit[other] # u) can be evaluated.  NULL_UNIT = idle. */
static _Atomic(int) g_d_unit[NUM_THREADS + 1];   /* indexed by tid (1-based) */

/* --- diagnostics --- */
static _Atomic(unsigned long long) cnt_alloc;
static _Atomic(unsigned long long) cnt_reclaim;
static _Atomic(unsigned long long) cnt_dealloc;
static _Atomic(unsigned long long) cnt_foreign;     /* d_resolve palloc==0 */
static _Atomic(unsigned long long) cnt_stale_fires; /* staleRead set events */

static _Atomic(bool) g_stop;

/* A single coarse lock serialises whole TLA+ actions: each action in the
 * spec is atomic (one Next step), and the model's interleaving freedom is
 * BETWEEN actions, not within them.  We realise the spec's step granularity
 * by holding the lock for the duration of ONE action (do_allocate /
 * do_reclaim / one d_* edge) and releasing it between actions, so a peer
 * thread's next action can interleave at exactly the spec's pc boundaries
 * (D_Start | D_LoadBack | D_LoadPal | D_Resolve).  This is the faithful way
 * to reproduce a TLA+ small-step model in C without false data races. */
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

/* ============================================================
 * TLA+ helpers.
 * ============================================================ */

/* xorshift32 PRNG to realize the TLA+ `\E` nondeterministic choices. */
static inline unsigned xs32(unsigned *s) {
    unsigned x = *s ? *s : 0x9e3779b9u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

/* TLA+ SpanFree(u, span): all units in [u, u+span) free (unclaimed) and in
 * range.  Caller holds g_lock. */
static bool span_free(int u, int span) {
    if (u + span > NUM_UNITS)
        return false;
    for (int k = 0; k < span; k++)
        if (atomic_load_explicit(&g_claim[u + k], memory_order_relaxed))
            return false;
    return true;
}

/* TLA+ ReclaimableBase(u): palloc[u] != 0 AND allocGenOf[u] = Null. */
static bool reclaimable_base(int u) {
    return atomic_load_explicit(&g_palloc[u], memory_order_relaxed) != NULL_GEN
        && atomic_load_explicit(&g_alloc_gen[u], memory_order_relaxed) == NULL_GEN;
}

/* ============================================================
 * A_Allocate(t): recycle a free unit run into a new chunk.
 *
 * TLA+: pick u, span in {1,2} with SpanFree; apply claim/backOff/palloc/
 * chunkSpan/ready/allocGenOf and gen++ as one atomic action.  Returns true
 * if it fired (an enabled \E was found).  Caller holds g_lock.
 * ============================================================ */
static bool do_allocate(ThreadCtx *ctx) {
    int cur_gen = atomic_load_explicit(&g_gen, memory_order_relaxed);
    if (cur_gen > MAX_GEN)
        return false;                       /* TLA+ guard: gen <= MaxGen */

    /* TLA+ \E u, span : SpanFree(u,span).  Collect candidates, pick one at
     * random (the nondeterministic choice that lets a 2-unit C0 recycle
     * into a 1-unit C1 reinterpreting the continuation unit). */
    int cu[NUM_UNITS * 2][2];
    int n = 0;
    for (int u = 0; u < NUM_UNITS; u++) {
        for (int span = 1; span <= 2; span++) {
            if (span_free(u, span)) {
                cu[n][0] = u;
                cu[n][1] = span;
                n++;
            }
        }
    }
    if (n == 0)
        return false;                       /* no enabled allocate */

    int pick = (int)(xs32(&ctx->rng) % (unsigned)n);
    int u = cu[pick][0];
    int span = cu[pick][1];

    /* C++ allocate_chunk order: claim -> back_offset -> header (palloc,
     * span) -> ready; applied as one TLA+ action here (the racy READ
     * interleaving lives on the deallocate side). */
    for (int k = u; k < u + span; k++) {
        atomic_store_explicit(&g_claim[k], true, memory_order_relaxed);
        atomic_store_explicit(&g_backoff[k], k - u, memory_order_relaxed);
    }
    atomic_store_explicit(&g_palloc[u], cur_gen, memory_order_relaxed);
    atomic_store_explicit(&g_span[u], span, memory_order_relaxed);
    atomic_store_explicit(&g_ready[u], true, memory_order_relaxed);
    /* register a live slot at the base unit, tagged with this gen */
    atomic_store_explicit(&g_alloc_gen[u], cur_gen, memory_order_relaxed);
    atomic_store_explicit(&g_gen, cur_gen + 1, memory_order_relaxed);

    atomic_fetch_add_explicit(&cnt_alloc, 1, memory_order_relaxed);
    return true;
}

/* ============================================================
 * R_Reclaim(t): deallocate_chunk on a reclaimable base.
 *
 * TLA+: pick u in ChunkBaseUnits with ReclaimableBase(u); apply ready=F,
 * palloc=0, span=0, backOff[span units]=0, claim[span units]=F as one
 * action.  Caller holds g_lock.
 * ============================================================ */
static bool do_reclaim(ThreadCtx *ctx) {
    int cand[NUM_UNITS];
    int n = 0;
    for (int u = 0; u < NUM_UNITS; u++) {
        if (atomic_load_explicit(&g_palloc[u], memory_order_relaxed) != NULL_GEN
            && reclaimable_base(u))
            cand[n++] = u;
    }
    if (n == 0)
        return false;

    int u = cand[(int)(xs32(&ctx->rng) % (unsigned)n)];
    int span = atomic_load_explicit(&g_span[u], memory_order_relaxed);

    /* The four reclaim writes (TLA+ applies them together at this
     * abstraction; their order matters only relative to a concurrent
     * deallocate's loads, which can interleave at the pc boundaries
     * between actions). */
    atomic_store_explicit(&g_ready[u], false, memory_order_relaxed);
    atomic_store_explicit(&g_palloc[u], NULL_GEN, memory_order_relaxed);
    atomic_store_explicit(&g_span[u], 0, memory_order_relaxed);
    for (int k = u; k < u + span; k++) {
        atomic_store_explicit(&g_backoff[k], 0, memory_order_relaxed);
        atomic_store_explicit(&g_claim[k], false, memory_order_relaxed);
    }

    atomic_fetch_add_explicit(&cnt_reclaim, 1, memory_order_relaxed);
    return true;
}

/* ============================================================
 * D_Start(t): begin a deallocate of a unit holding a live slot this thread
 * "owns" (any live slot; one in-flight per thread).
 *
 * TLA+: pick u with allocGenOf[u] # Null; SinglePayout guard; capture
 * dUnit/dExpectGen; if FreeBeforeLookup, clear allocGenOf[u] now (chunk
 * reclaimable mid-lookup); pc -> d_loadback.  Caller holds g_lock.
 * ============================================================ */
static bool d_start(ThreadCtx *ctx) {
    int cand[NUM_UNITS];
    int n = 0;
    for (int u = 0; u < NUM_UNITS; u++) {
        if (atomic_load_explicit(&g_alloc_gen[u], memory_order_relaxed) == NULL_GEN)
            continue;
#if SINGLE_PAYOUT
        /* TLA+: SinglePayout => \A other != t : dUnit[other] # u */
        bool taken = false;
        for (int o = 1; o <= NUM_THREADS; o++) {
            if (o == ctx->tid)
                continue;
            if (atomic_load_explicit(&g_d_unit[o], memory_order_relaxed) == u) {
                taken = true;
                break;
            }
        }
        if (taken)
            continue;
#endif
        cand[n++] = u;
    }
    if (n == 0)
        return false;

    int u = cand[(int)(xs32(&ctx->rng) % (unsigned)n)];
    ctx->d_unit = u;
    ctx->d_expect_gen = atomic_load_explicit(&g_alloc_gen[u], memory_order_relaxed);
    atomic_store_explicit(&g_d_unit[ctx->tid], u, memory_order_relaxed);

#if FREE_BEFORE_LOOKUP
    /* Permissive: logically free now -> chunk reclaimable mid-lookup. */
    atomic_store_explicit(&g_alloc_gen[u], NULL_GEN, memory_order_relaxed);
#endif
    ctx->pc = PC_D_LOADBACK;
    return true;
}

/* D_LoadBack(t): load #1 (back_offset[uP]); base = uP - base_off.
 * Caller holds g_lock. */
static bool d_loadback(ThreadCtx *ctx) {
    int u = ctx->d_unit;
    int bo = atomic_load_explicit(&g_backoff[u], memory_order_relaxed);
    ctx->d_baseoff = bo;
    ctx->d_base = u - bo;
    ctx->pc = PC_D_LOADPAL;
    return true;
}

/* D_LoadPal(t): load #2 (palloc[base]).  Caller holds g_lock.
 *
 * Note: base is derived from load #1.  In the TLA+ model base is always in
 * range here (Inv_BackOffPointsToBase guarantees u - backOff[u] in Units
 * for any claimed unit; for an unclaimed/reclaimed unit backOff is 0 so
 * base == u, also in range). */
static bool d_loadpal(ThreadCtx *ctx) {
    int base = ctx->d_base;
    /* Defensive: base is in [0, NUM_UNITS) by the spec; assert it (this is
     * exactly Inv_BackOffPointsToBase evaluated at the load site). */
    assert(base >= 0 && base < NUM_UNITS);
    ctx->d_palloc = atomic_load_explicit(&g_palloc[base], memory_order_relaxed);
    ctx->pc = PC_D_RESOLVE;
    return true;
}

/* D_Resolve(t): resolve the deallocate.  Caller holds g_lock. */
static bool d_resolve(ThreadCtx *ctx) {
    if (ctx->d_palloc == NULL_GEN) {
        /* foreign: palloc cleared (chunk released) -> libsystem free path.
         * Safe: deallocate does not touch the bitmap. */
        atomic_fetch_add_explicit(&cnt_foreign, 1, memory_order_relaxed);
    } else {
        /* non-foreign: thread treats the resolved chunk as p's owner.
         * BUG if that chunk's generation != the one p was allocated at. */
        if (ctx->d_palloc != ctx->d_expect_gen) {
            atomic_store_explicit(&g_stale_read, true, memory_order_relaxed);
            atomic_fetch_add_explicit(&cnt_stale_fires, 1, memory_order_relaxed);
        }
    }

#if !FREE_BEFORE_LOOKUP
    /* Faithful: the slot/bit is cleared HERE, after the lookup resolved the
     * chunk -- so the chunk could not have been reclaimed during it. */
    atomic_store_explicit(&g_alloc_gen[ctx->d_unit], NULL_GEN, memory_order_relaxed);
#endif

    /* pc -> idle; clear the d* registers (TLA+ D_Resolve resets them). */
    ctx->pc = PC_IDLE;
    atomic_store_explicit(&g_d_unit[ctx->tid], NULL_UNIT, memory_order_relaxed);
    ctx->d_unit = NULL_UNIT;
    ctx->d_baseoff = NULL_UNIT;
    ctx->d_base = NULL_UNIT;
    ctx->d_palloc = NULL_GEN;
    ctx->d_expect_gen = NULL_GEN;

    atomic_fetch_add_explicit(&cnt_dealloc, 1, memory_order_relaxed);
    return true;
}

/* ============================================================
 * Continuous (during-run) sanity invariants.  These hold in EVERY config
 * (the count==0 reclaim precondition is GRANTED by R_Reclaim) and are the
 * model's terminal/safety invariants.  Checked under g_lock between actions
 * (a quiescent point equivalent to a TLA+ state).
 *
 *   Inv_LiveSlotImpliesClaimed : allocGenOf[u]#Null => claim[u] /\ palloc[u]#0
 *   Inv_BackOffPointsToBase    : claim[u] => (u - backOff[u]) in Units
 * ============================================================ */
static void check_invariants_locked(void) {
    for (int u = 0; u < NUM_UNITS; u++) {
        if (atomic_load_explicit(&g_alloc_gen[u], memory_order_relaxed) != NULL_GEN) {
            assert(atomic_load_explicit(&g_claim[u], memory_order_relaxed) == true);
            assert(atomic_load_explicit(&g_palloc[u], memory_order_relaxed) != NULL_GEN);
        }
        if (atomic_load_explicit(&g_claim[u], memory_order_relaxed) == true) {
            int base = u - atomic_load_explicit(&g_backoff[u], memory_order_relaxed);
            assert(base >= 0 && base < NUM_UNITS);
        }
    }
}

/* ============================================================
 * Worker: drive this thread's enabled TLA+ actions.  One action per loop
 * iteration, picked nondeterministically among those enabled, holding
 * g_lock for the action's duration (= one TLA+ Next step).  A thread with a
 * deallocate in flight (pc != idle) MUST advance it (the d_* edges are the
 * only enabled actions for a non-idle thread, mirroring the spec).
 * ============================================================ */
static void worker_step(ThreadCtx *ctx) {
    pthread_mutex_lock(&g_lock);

    if (ctx->pc == PC_D_LOADBACK) {
        d_loadback(ctx);
    } else if (ctx->pc == PC_D_LOADPAL) {
        d_loadpal(ctx);
    } else if (ctx->pc == PC_D_RESOLVE) {
        d_resolve(ctx);
    } else {
        /* idle: pick among A_Allocate / R_Reclaim / D_Start at random,
         * trying the others if the first pick is not enabled.  (TLA+ Next
         * is the disjunction of all enabled actions for thread t.) */
        unsigned r = xs32(&ctx->rng) % 3u;
        bool fired = false;
        for (unsigned k = 0; k < 3u && !fired; k++) {
            switch ((r + k) % 3u) {
                case 0: fired = do_allocate(ctx); break;
                case 1: fired = do_reclaim(ctx);  break;
                case 2: fired = d_start(ctx);     break;
            }
        }
        /* If nothing was enabled (e.g. region empty and gen exhausted),
         * the thread is at a TLA+ deadlock-for-this-thread; just return. */
        (void)fired;
    }

    check_invariants_locked();
    pthread_mutex_unlock(&g_lock);
}

static void *worker(void *arg) {
    ThreadCtx *ctx = (ThreadCtx *)arg;
    for (int s = 0; s < STEPS_PER_ROUND; s++) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed))
            break;
        worker_step(ctx);
    }
    /* Drain any in-flight deallocate so the post-join state is quiescent
     * (all threads at pc=idle), matching a TLA+ stuttering-terminal state. */
    while (ctx->pc != PC_IDLE) {
        pthread_mutex_lock(&g_lock);
        if (ctx->pc == PC_D_LOADBACK)      d_loadback(ctx);
        else if (ctx->pc == PC_D_LOADPAL)  d_loadpal(ctx);
        else if (ctx->pc == PC_D_RESOLVE)  d_resolve(ctx);
        check_invariants_locked();
        pthread_mutex_unlock(&g_lock);
    }
    return NULL;
}

/* TLA+ Init: region empty, all threads idle. */
static void reset_region(ThreadCtx *ctxs) {
    for (int u = 0; u < NUM_UNITS; u++) {
        atomic_store(&g_claim[u], false);
        atomic_store(&g_ready[u], false);
        atomic_store(&g_backoff[u], 0);
        atomic_store(&g_palloc[u], NULL_GEN);
        atomic_store(&g_span[u], 0);
        atomic_store(&g_alloc_gen[u], NULL_GEN);
    }
    atomic_store(&g_gen, 1);            /* generation ids start at 1 */
    for (int t = 0; t <= NUM_THREADS; t++)
        atomic_store(&g_d_unit[t], NULL_UNIT);
    for (int i = 0; i < NUM_THREADS; i++) {
        ctxs[i].pc = PC_IDLE;
        ctxs[i].d_unit = NULL_UNIT;
        ctxs[i].d_baseoff = NULL_UNIT;
        ctxs[i].d_base = NULL_UNIT;
        ctxs[i].d_palloc = NULL_GEN;
        ctxs[i].d_expect_gen = NULL_GEN;
    }
    /* NB: g_stale_read is NOT reset per round -- it is a monotone "the bug
     * was observed at least once" flag across the whole run, matching the
     * TLA+ invariant which fails the moment staleRead becomes TRUE. */
}

int main(void) {
    atomic_store(&g_stale_read, false);

    ThreadCtx ctxs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        ctxs[i].tid = i + 1;       /* 1-indexed; tid 0 reserved (Null) */
        ctxs[i].rng = 0x1234567u + 0x9e3779b9u * (unsigned)(i + 1);
    }

    int rounds = 0;

#if STRESS_SECONDS > 0
    struct timespec t0, now;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec - t0.tv_sec >= STRESS_SECONDS)
            break;
#else
    for (int round = 0; round < (int)MAX_COMMITS; round++) {
#endif
        reset_region(ctxs);

        pthread_t threads[NUM_THREADS];
        for (int i = 0; i < NUM_THREADS; i++)
            pthread_create(&threads[i], NULL, worker, &ctxs[i]);
        for (int i = 0; i < NUM_THREADS; i++)
            pthread_join(threads[i], NULL);

        /* Post-join: quiescent state.  Sanity invariants must hold. */
        check_invariants_locked();
        rounds++;
    }

    bool stale = atomic_load(&g_stale_read);

    /* Which configs are the TRUE negative control (Inv_NoStaleRead holds)?
     * Verified against TLC on this very spec (ground truth, not the cfg
     * header comments, which overstate the faithful-only case):
     *
     *   FREE_BEFORE_LOOKUP=1 SINGLE_PAYOUT=0 : VIOLATED (the hunted bug)
     *   FREE_BEFORE_LOOKUP=0 SINGLE_PAYOUT=0 : VIOLATED (cfg comment is
     *       aspirational/wrong -- double-payout still allowed: one thread's
     *       faithful D_Resolve clears the slot, the chunk is reclaimed +
     *       recycled while the peer's lookup is mid-flight -> stale read.
     *       TLC's ChunkRecycle_microscopic_faithful_mc.cfg DOES report
     *       Inv_NoStaleRead violated.)
     *   FREE_BEFORE_LOOKUP=1 SINGLE_PAYOUT=1 : VIOLATED (the slot is freed
     *       early at D_Start, so SinglePayout's mid-deallocate guard does
     *       not stop a reclaim+recycle racing the lookup loads -- TLC also
     *       reports a violation here).
     *   FREE_BEFORE_LOOKUP=0 SINGLE_PAYOUT=1 : SAFE -- the ONLY negative
     *       control: the slot stays live until D_Resolve AND no other
     *       thread can be mid-deallocate of the same unit, so the chunk
     *       cannot be reclaimed during the lookup.  TLC finds NO invariant
     *       violation (reaches a benign deadlock terminal state).
     *
     * So Inv_NoStaleRead is asserted iff (faithful AND single-payout). */
#define NEG_CONTROL (!FREE_BEFORE_LOOKUP && SINGLE_PAYOUT)

#if NEG_CONTROL
    /* NEGATIVE CONTROL: Inv_NoStaleRead must hold (no stale read possible). */
    if (stale) {
        fprintf(stderr,
            "FAIL: staleRead set under negative-control config "
            "(FREE_BEFORE_LOOKUP=%d SINGLE_PAYOUT=%d) -- "
            "Inv_NoStaleRead violated where TLC expects NO error.\n",
            FREE_BEFORE_LOOKUP, SINGLE_PAYOUT);
        abort();
    }
    assert(!stale);   /* Inv_NoStaleRead (TLA+ terminal safety invariant) */
#endif

#if STRESS_SECONDS > 0
    printf("[ChunkRecycle_microscopic stress %ds "
           "FREE_BEFORE_LOOKUP=%d SINGLE_PAYOUT=%d units=%d gen<=%d thr=%d] "
           "rounds=%d\n",
           STRESS_SECONDS, FREE_BEFORE_LOOKUP, SINGLE_PAYOUT,
           NUM_UNITS, MAX_GEN, NUM_THREADS, rounds);
    printf("  alloc=%llu reclaim=%llu dealloc=%llu foreign=%llu "
           "stale_fires=%llu  staleRead=%s\n",
           (unsigned long long)atomic_load(&cnt_alloc),
           (unsigned long long)atomic_load(&cnt_reclaim),
           (unsigned long long)atomic_load(&cnt_dealloc),
           (unsigned long long)atomic_load(&cnt_foreign),
           (unsigned long long)atomic_load(&cnt_stale_fires),
           stale ? "TRUE (bug observed)" : "false");
#  if !NEG_CONTROL
    /* BUG-EXPECTING config: staleRead becoming TRUE is the EXPECTED hunted
     * bug (Inv_NoStaleRead is violated, matching TLC's counterexample); it
     * is reported, not asserted away. */
#  endif
#else
    /* Unit: the bounded run terminated and the sanity invariants held. */
    assert(rounds >= 1);
    (void)rounds;
#  if !NEG_CONTROL
    /* BUG-EXPECTING config: report whether the bug surfaced in this short
     * bounded run.  (It may or may not, depending on the interleaving the
     * scheduler hit -- the bounded unit search is far smaller than TLC's
     * exhaustive one; the negative-control build is the one that ASSERTS
     * no stale read.) */
    fprintf(stderr,
        "[unit] BUG-EXPECTING config (FREE_BEFORE_LOOKUP=%d SINGLE_PAYOUT=%d): "
        "staleRead=%s (Inv_NoStaleRead is the hunted bug here; reported, "
        "not asserted)\n",
        FREE_BEFORE_LOOKUP, SINGLE_PAYOUT,
        stale ? "TRUE (stale-generation read reproduced)" : "false");
#  endif
#endif

    return 0;
}
