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
 * C11 test generated mechanically from OrphanReuse_36b_REOWNED.tla.
 *
 * §36b CANDIDATE FIX (the "REOWNED" variant).  The companion bug port
 * test_OrphanReuse_36b.c reproduces the ORPHAN_RELEASE_BAD ABA race:
 * orphan_claim_for_release sees disp = OWNED and CAS-releases a chunk
 * whose bit_owned is TRUE, because PoP2 wrote OWNED into disp AFTER PoP1
 * set bit_owned — disp = OWNED is overloaded (never-pushed vs re-owned).
 *
 * THE PROPOSED FIX (one extra disp value, minimal diff):
 *   * Add REOWNED to the disp domain.
 *   * PoP2 transitions disp = 1+k -> REOWNED (NOT OWNED).
 *   * orphan_claim_for_release at d = REOWNED bails (no release): the
 *     chunk is live again; its new owner manages its lifetime.
 *   * Owner exit on a {OWNED, REOWNED} chunk runs orphan_push normally
 *     (REOWNED -> PUSHING, just like OWNED -> PUSHING).  The fused
 *     fetch_and(~BIT_OWNED)+branch is modelled as OwnerExit_NonEmpty /
 *     OwnerExit_Empty: one atomic step that clears bit_owned and branches
 *     on the consistent snapshot of mask_cnt.
 *
 * IMPORTANT — WHAT THIS PORT ACTUALLY MIRRORS.  The .tla header claims the
 * fix is "verified" (Inv_NoBadRelease holds).  It does NOT.  Running TLC on
 * OrphanReuse_36b_REOWNED_2thr_mc.cfg produces a counter-example of depth 9:
 *     ... PopP0/PopP1/PopP2 (re-own, disp->REOWNED) ;
 *         ReownAllocSlot (mask_cnt 0->1) ;
 *         OwnerExit_NonEmpty (disp REOWNED->PUSHING, bit_owned->FALSE) ;
 *         PushP1 (disp->1+k, slot k <- C) ;
 *         ClaimSLOT(t1,0) releases an on-array chunk while mask_cnt = 1
 *         => bad_release' = (mask_cnt > 0) = TRUE.
 * The REOWNED value only closes the disp=OWNED-overload path that drives
 * ClaimOWNED in the bug model; a DIFFERENT path (a freer's stale
 * pending_claim surviving a full re-own + re-push, then ClaimSLOT-releasing
 * an on-array chunk whose count was re-bumped) still trips the MASK_CNT
 * half of the DEBUG guard.  So this candidate is NOT a complete fix.
 *
 * This port is a faithful mechanical translation, so it mirrors TLC: it
 * RECORDS bad_release as an observed counter (exactly like the bug port)
 * rather than assert(bad==0) — asserting the absent invariant would be
 * unfaithful to the actual model.  The terminal assert checks only the
 * invariants the model genuinely upholds (TypeOK, released-implies-disp-
 * RELEASED).
 *
 * ATOMICITY MODEL (identical discipline to the bug port).  Each TLA+
 * Next-state action is one atomic step over several variables; a single
 * lock-free word cannot keep disp/bit_owned/mask_cnt/slots consistent, so
 * each action runs under a per-chunk mutex (lock = begin step, unlock =
 * end).  Concurrency is at the interleaving level: threads acquire the
 * lock in arbitrary order, and the 3-step re-own (PoP0/PoP1/PoP2) is three
 * separately-locked steps with any other thread's actions interleaved
 * between them — which is exactly what lets the residual ClaimSLOT race be
 * exercised, matching the TLC counter-example above.
 *
 * Differences from test_OrphanReuse_36b.c:
 *   + DISP_REOWNED sentinel (= 300).
 *   + pop_p2() stores REOWNED instead of OWNED.
 *   + claim_bail() also bails on REOWNED.
 *   + push_p0() replaced by owner_exit_nonempty() / owner_exit_empty().
 *   + pending_claim is PER-THREAD (TLA+ [Threads -> BOOLEAN]).
 *
 * TLA+ variable mapping (all under c->lock):
 *   slots/disp/bit_owned/mask_cnt/released/bad_release  — as the bug port.
 *   pending_claim[t]          -> c->pending_claim[NUM_THREADS] (per-thread
 *                                 stack-local release_me flag)
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
#define MAX_COMMITS   1          /* per-round chunk lifecycles (rounds) */
#endif

#ifndef STRESS_SECONDS
#define STRESS_SECONDS 0         /* 0 = bounded unit run; >0 = wall-clock stress */
#endif

#ifndef ALLOW_REOWN
#define ALLOW_REOWN   1          /* TLA+ CONSTANT AllowReown (TRUE default) */
#endif

/* TLA+ CONSTANT K (slot array size, recommend 2). */
#ifndef K_SLOTS
#define K_SLOTS       2
#endif

#ifndef MAX_STEPS_PER_THREAD
#define MAX_STEPS_PER_THREAD 200000
#endif

/* --- disp sentinel values (mirror the C++ enum / TLA+ definitions) --- */
#define DISP_OWNED     0
#define DISP_PUSHING   100
#define DISP_RELEASED  200
#define DISP_REOWNED   300       /* THE FIX: distinct from OWNED */
/* 1 .. K_SLOTS  => on-array at slot (disp-1) */

/* --- per-thread program counters --- */
enum reown_pc { RP_IDLE = 0, RP_GOT_C, RP_ARMED_OWNED };
enum push_pc  { PP_IDLE = 0, PP_PUSHING };

/* --- slot tagged word: ptr ("C"/"Null") + monotone version --- */
typedef struct { bool is_c; uint64_t ver; } slot_t;

/* ---------------------------------------------------------------------- */
/* Per-chunk lifecycle state.  pending_claim is now per-thread.            */
/* Every field is touched only while holding c->lock.                      */
/* ---------------------------------------------------------------------- */
typedef struct {
    pthread_mutex_t lock;
    slot_t   slots[K_SLOTS];
    int32_t  disp;
    bool     bit_owned;
    int      mask_cnt;
    bool     released;
    bool     pending_claim[NUM_THREADS]; /* per-thread (the fix's model) */
    bool     bad_release;                /* must stay false in the fix */
} chunk_t;

static chunk_t      *g_chunks;
static int           g_num_chunks;
static _Atomic(int)  g_bad_observed;

/* ---------------------------------------------------------------------- */
/* Thread-local PRNG.                                                       */
/* ---------------------------------------------------------------------- */
static inline uint32_t xorshift(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x ? x : 0x1234567u;
    return *s;
}

/* The ORPHAN_RELEASE_BAD DEBUG guard (BIT_OWNED clear AND MASK_CNT 0),
 * evaluated atomically in the same step as the release (we hold c->lock). */
static inline void note_bad_release(chunk_t *c) {
    if (c->bit_owned || c->mask_cnt > 0) {
        c->bad_release = true;
        atomic_store_explicit(&g_bad_observed, 1, memory_order_relaxed);
    }
}

/* ====================================================================== */
/* ACTIONS  (each = one TLA+ atomic step; caller holds c->lock)            */
/* ====================================================================== */

/* OwnerExit_NonEmpty(t): disp in {OWNED,REOWNED} && bit_owned && mask_cnt>0
 * && push_pc=idle && !released.  fetch_and(~BIT_OWNED)+branch into push:
 * bit_owned -> FALSE, disp -> PUSHING, push_pc -> pushing.  The clear and
 * the branch are one atomic step (consistent snapshot). */
static bool owner_exit_nonempty(chunk_t *c, enum push_pc *ppc) {
    if (*ppc != PP_IDLE) return false;
    if (c->released) return false;
    if (c->disp != DISP_OWNED && c->disp != DISP_REOWNED) return false;
    if (!c->bit_owned) return false;
    if (c->mask_cnt <= 0) return false;
    c->bit_owned = false;
    c->disp = DISP_PUSHING;
    *ppc = PP_PUSHING;
    return true;
}

/* OwnerExit_Empty(t): disp in {OWNED,REOWNED} && bit_owned && mask_cnt=0
 * && !released.  Sole-owner empty exit, the unique-releaser path:
 * bit_owned -> FALSE, disp -> RELEASED, released -> TRUE. */
static bool owner_exit_empty(chunk_t *c) {
    if (c->released) return false;
    if (c->disp != DISP_OWNED && c->disp != DISP_REOWNED) return false;
    if (!c->bit_owned) return false;
    if (c->mask_cnt != 0) return false;
    c->bit_owned = false;
    c->disp = DISP_RELEASED;
    c->released = true;
    note_bad_release(c);            /* never fires here (owned just cleared,
                                     * mask_cnt==0); kept for symmetry. */
    return true;
}

/* PushP1(t,k): empty slot (Null,vk) -> (C,vk+1); disp = k+1. */
static bool push_p1(chunk_t *c, enum push_pc *ppc, int k) {
    if (*ppc != PP_PUSHING) return false;
    if (c->disp != DISP_PUSHING) return false;
    if (c->slots[k].is_c) return false;
    c->slots[k].is_c = true;
    c->slots[k].ver += 1;
    c->disp = (int32_t)(k + 1);
    *ppc = PP_IDLE;
    return true;
}

/* PushP1GiveUp(t): all slots full -> disp = OWNED. */
static bool push_p1_giveup(chunk_t *c, enum push_pc *ppc) {
    if (*ppc != PP_PUSHING) return false;
    if (c->disp != DISP_PUSHING) return false;
    for (int k = 0; k < K_SLOTS; k++)
        if (!c->slots[k].is_c) return false;
    c->disp = DISP_OWNED;
    *ppc = PP_IDLE;
    return true;
}

/* PopP0(t,k): take on-array chunk (C,vk) -> (Null,vk+1).  disp NOT updated. */
static bool pop_p0(chunk_t *c, enum reown_pc *rpc, int k) {
    if (*rpc != RP_IDLE) return false;
    if (c->released) return false;
    if (!c->slots[k].is_c) return false;
    c->slots[k].is_c = false;
    c->slots[k].ver += 1;
    *rpc = RP_GOT_C;
    return true;
}

/* PopP1(t): claim BIT_OWNED (false -> true). */
static bool pop_p1(chunk_t *c, enum reown_pc *rpc) {
    if (*rpc != RP_GOT_C) return false;
    if (c->released) return false;
    if (c->bit_owned) return false;
    c->bit_owned = true;
    *rpc = RP_ARMED_OWNED;
    return true;
}

/* PopP2(t): publish disp = REOWNED (THE FIX), closing the re-own. */
static bool pop_p2(chunk_t *c, enum reown_pc *rpc) {
    if (*rpc != RP_ARMED_OWNED) return false;
    c->disp = DISP_REOWNED;
    *rpc = RP_IDLE;
    return true;
}

/* FreeDecToZero_TriggersClaim / _Spurious: per-thread pending_claim. */
static bool free_dec_to_zero(chunk_t *c, int tid) {
    if (c->released) return false;
    if (c->mask_cnt != 1) return false;
    if (!c->bit_owned) {
        /* TriggersClaim requires \neg pending_claim[t]. */
        if (c->pending_claim[tid]) return false;
        c->mask_cnt = 0;
        c->pending_claim[tid] = true;
        return true;
    }
    /* _Spurious (bit_owned set). */
    c->mask_cnt = 0;
    return true;
}

/* FreeDecCommon: mask_cnt > 1 -> mask_cnt - 1. */
static bool free_dec_common(chunk_t *c) {
    if (c->released) return false;
    if (c->mask_cnt <= 1) return false;
    c->mask_cnt -= 1;
    return true;
}

/* ClaimOWNED(t): pending_claim[t] && disp=OWNED -> RELEASED, release. */
static bool claim_owned(chunk_t *c, int tid) {
    if (!c->pending_claim[tid]) return false;
    if (c->released) return false;
    if (c->disp != DISP_OWNED) return false;
    c->disp = DISP_RELEASED;
    c->released = true;
    c->pending_claim[tid] = false;
    note_bad_release(c);
    return true;
}

/* ClaimSLOT(t,k): pending_claim[t] && disp=k+1 && slot[k]="C" -> release. */
static bool claim_slot(chunk_t *c, int tid, int k) {
    if (!c->pending_claim[tid]) return false;
    if (c->released) return false;
    if (c->disp != (int32_t)(k + 1)) return false;
    if (!c->slots[k].is_c) return false;
    c->slots[k].is_c = false;
    c->slots[k].ver += 1;
    c->disp = DISP_RELEASED;
    c->released = true;
    c->pending_claim[tid] = false;
    note_bad_release(c);
    return true;
}

/* ClaimSLOTBail(t,k): pending_claim[t] && disp=k+1 && slot[k]!="C" -> bail. */
static bool claim_slot_bail(chunk_t *c, int tid, int k) {
    if (!c->pending_claim[tid]) return false;
    if (c->released) return false;
    if (c->disp != (int32_t)(k + 1)) return false;
    if (c->slots[k].is_c) return false;
    c->pending_claim[tid] = false;
    return true;
}

/* ClaimBail(t): pending_claim[t] && disp in {PUSHING,RELEASED,REOWNED} ->
 * bail (THE FIX: REOWNED bails, does NOT release). */
static bool claim_bail(chunk_t *c, int tid) {
    if (!c->pending_claim[tid]) return false;
    if (c->released) return false;
    if (c->disp != DISP_PUSHING && c->disp != DISP_RELEASED
        && c->disp != DISP_REOWNED) return false;
    c->pending_claim[tid] = false;
    return true;
}

/* ReownAllocSlot(t): AllowReown && reown_pc=armed_owned && mask_cnt<2 ->
 * mask_cnt + 1. */
static bool reown_alloc_slot(chunk_t *c, enum reown_pc *rpc) {
#if ALLOW_REOWN
    if (*rpc != RP_ARMED_OWNED) return false;
    if (c->mask_cnt >= 2) return false;
    c->mask_cnt += 1;
    return true;
#else
    (void)c; (void)rpc;
    return false;
#endif
}

/* ---------------------------------------------------------------------- */
/* try_action: one atomic step under the per-chunk lock.                   */
/* ---------------------------------------------------------------------- */
static bool try_action(chunk_t *c, enum reown_pc *rpc, enum push_pc *ppc,
                       int tid, int kind, int k) {
    bool fired = false;
    pthread_mutex_lock(&c->lock);
    switch (kind) {
        case 0:  fired = owner_exit_nonempty(c, ppc);  break;
        case 1:  fired = owner_exit_empty(c);          break;
        case 2:  fired = push_p1(c, ppc, k);           break;
        case 3:  fired = push_p1_giveup(c, ppc);       break;
        case 4:  fired = pop_p0(c, rpc, k);            break;
        case 5:  fired = pop_p1(c, rpc);               break;
        case 6:  fired = pop_p2(c, rpc);               break;
        case 7:  fired = free_dec_to_zero(c, tid);     break;
        case 8:  fired = free_dec_common(c);           break;
        case 9:  fired = claim_owned(c, tid);          break;
        case 10: fired = claim_slot(c, tid, k);        break;
        case 11: fired = claim_slot_bail(c, tid, k);   break;
        case 12: fired = claim_bail(c, tid);           break;
        default: break;
    }
    pthread_mutex_unlock(&c->lock);
    return fired;
}

static bool drive_self(chunk_t *c, enum reown_pc *rpc, enum push_pc *ppc) {
    bool moved = false;
    pthread_mutex_lock(&c->lock);
    if (*ppc == PP_PUSHING) {
        moved = push_p1(c, ppc, 0) || push_p1(c, ppc, 1)
                || push_p1_giveup(c, ppc);
    } else if (*rpc == RP_GOT_C) {
        moved = pop_p1(c, rpc);
    } else if (*rpc == RP_ARMED_OWNED) {
        moved = reown_alloc_slot(c, rpc) || pop_p2(c, rpc);
    }
    pthread_mutex_unlock(&c->lock);
    return moved;
}

static bool chunk_released(chunk_t *c) {
    pthread_mutex_lock(&c->lock);
    bool r = c->released;
    pthread_mutex_unlock(&c->lock);
    return r;
}

/* ---------------------------------------------------------------------- */
/* Worker.                                                                  */
/* ---------------------------------------------------------------------- */
typedef struct { int id; } thr_arg_t;

#define NUM_ACTION_KINDS 13

static void *worker(void *p) {
    thr_arg_t *a = (thr_arg_t *)p;
    int tid = a->id;
    uint32_t rng = 0x9e3779b9u ^ (uint32_t)(tid * 2654435761u) ^ 0x2u;

    for (int round = 0; round < g_num_chunks; round++) {
        chunk_t *c = &g_chunks[round];
        enum reown_pc rpc = RP_IDLE;
        enum push_pc  ppc = PP_IDLE;

        long steps = 0;
        for (;;) {
            if (chunk_released(c) && rpc == RP_IDLE && ppc == PP_IDLE) break;
            if (++steps > MAX_STEPS_PER_THREAD) break;

            int kind = (int)(xorshift(&rng) % NUM_ACTION_KINDS);
            int k    = (int)(xorshift(&rng) % K_SLOTS);
            if (!try_action(c, &rpc, &ppc, tid, kind, k))
                drive_self(c, &rpc, &ppc);
        }

        while (ppc == PP_PUSHING || rpc == RP_GOT_C || rpc == RP_ARMED_OWNED) {
            if (!drive_self(c, &rpc, &ppc)) break;
        }
    }
    return NULL;
}

/* ---------------------------------------------------------------------- */
/* Terminal invariants.  TLC refutes Inv_NoBadRelease for this model (see  */
/* header), so we do NOT assert !bad_release — that would be unfaithful.   */
/* We assert only the invariants the model genuinely upholds and report    */
/* bad_release as an observed counter, exactly like the bug port.          */
/* ---------------------------------------------------------------------- */
static void check_terminal_invariants(chunk_t *c) {
    pthread_mutex_lock(&c->lock);
    int32_t  d   = c->disp;
    bool     rel = c->released;
    int      mc  = c->mask_cnt;
    pthread_mutex_unlock(&c->lock);

    /* TypeOK: disp in DispDomain (incl. REOWNED); mask_cnt in 0..2. */
    bool disp_ok = (d == DISP_OWNED || d == DISP_PUSHING || d == DISP_RELEASED
                    || d == DISP_REOWNED || (d >= 1 && d <= K_SLOTS));
    assert(disp_ok);
    assert(mc >= 0 && mc <= 2);

    /* Released implies disp = RELEASED (the only paths that set released
     * also publish disp = RELEASED). */
    if (rel) assert(d == DISP_RELEASED);
}

/* ---------------------------------------------------------------------- */
/* One bounded round.                                                       */
/* ---------------------------------------------------------------------- */
static int run_round(int num_chunks) {
    g_num_chunks = num_chunks;
    g_chunks = calloc((size_t)num_chunks, sizeof(*g_chunks));
    assert(g_chunks);

    for (int r = 0; r < num_chunks; r++) {
        chunk_t *c = &g_chunks[r];
        pthread_mutex_init(&c->lock, NULL);
        /* Init: slot0 = (C, ver 1), slot1.. = (Null, ver 0); disp = 1;
         * bit_owned = FALSE; mask_cnt = 1; released/pending/bad = FALSE. */
        c->slots[0].is_c = true;  c->slots[0].ver = 1;
        for (int k = 1; k < K_SLOTS; k++) { c->slots[k].is_c = false; c->slots[k].ver = 0; }
        c->disp = 1;
        c->bit_owned = false;
        c->mask_cnt = 1;
        c->released = false;
        for (int t = 0; t < NUM_THREADS; t++) c->pending_claim[t] = false;
        c->bad_release = false;
    }

    pthread_t  th[NUM_THREADS];
    thr_arg_t  args[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        args[t].id = t;
        int rc = pthread_create(&th[t], NULL, worker, &args[t]);
        assert(rc == 0);
    }
    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(th[t], NULL);

    /* Post-join: check the invariants the model genuinely upholds; record
     * (not assert) bad_release, since TLC refutes Inv_NoBadRelease here. */
    int round_bad = 0;
    for (int r = 0; r < num_chunks; r++) {
        check_terminal_invariants(&g_chunks[r]);
        if (g_chunks[r].bad_release) round_bad = 1;
        pthread_mutex_destroy(&g_chunks[r].lock);
    }

    free(g_chunks);
    g_chunks = NULL;
    return round_bad;
}

int main(void) {
    atomic_store_explicit(&g_bad_observed, 0, memory_order_relaxed);

    if (STRESS_SECONDS > 0) {
        struct timespec start;
        clock_gettime(CLOCK_MONOTONIC, &start);
        unsigned long rounds = 0, bad_rounds = 0;
        for (;;) {
            if (run_round(MAX_COMMITS > 0 ? MAX_COMMITS : 1)) bad_rounds++;
            rounds++;
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            if (now.tv_sec - start.tv_sec >= STRESS_SECONDS) break;
        }
        printf("OrphanReuse_36b_REOWNED stress DONE: %lu rounds, %lu with "
               "ORPHAN_RELEASE_BAD, %d threads (candidate is NOT a complete "
               "fix: bad_release still reachable per TLC)\n",
               rounds, bad_rounds, NUM_THREADS);
    } else {
        int total = MAX_COMMITS > 0 ? MAX_COMMITS : 1;
        int bad = run_round(total);
        printf("OrphanReuse_36b_REOWNED unit OK: %d chunk(s), %d threads; "
               "ORPHAN_RELEASE_BAD observed this run = %s "
               "(candidate is NOT a complete fix; either outcome is valid)\n",
               total, NUM_THREADS, bad ? "yes" : "no");
    }
    return 0;
}
