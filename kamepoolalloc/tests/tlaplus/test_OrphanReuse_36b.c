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
 * C11 test generated mechanically from OrphanReuse_36b.tla.
 *
 * §36b proposed orphan-reuse design (THE BUG VERSION).  This is the
 * as-documented protocol whose stress test still exhibits ~1.67 %
 * corruption (4x ORPHAN_RELEASE_BAD ABA + 1x sentinel mismatch).  The
 * model's purpose is to EXHIBIT the race, not to prove it absent — so
 * unlike the optC negative control, bad_release CAN legitimately fire
 * here.  This port records bad_release as an observed counter and the
 * terminal assert checks ONLY the invariants the design truly upholds
 * (TypeOK, Inv_NoDoubleRelease, released-implies-disp-RELEASED).
 *
 * ATOMICITY MODEL.  Every TLA+ Next-state action is ONE atomic step that
 * reads guards and applies effects over several variables (disp, bit_owned,
 * mask_cnt, slots[]) as an indivisible transition.  A single lock-free
 * word cannot keep those multiple variables consistent, so each action
 * here runs under a per-chunk mutex: lock = begin atomic step, unlock =
 * end.  This makes "one action = one atomic memory transition" literal,
 * exactly the TLA+ granularity.  Concurrency lives at the *interleaving*
 * level: real threads acquire the lock in arbitrary order, and the 3-step
 * re-own (PoP0/PoP1/PoP2) is three separately-locked steps with any other
 * thread's actions interleaved between them — which is precisely the
 * documented stale-disp window the model probes.
 *
 * One chunk lifecycle, K = 2 slots.  The owner thread already exited and
 * PUSHED the chunk into slot 0 (per the TLA+ Init): disp = 1+0 = 1,
 * bit_owned = FALSE, mask_cnt = 1, slot0 = (C, ver 1).
 *
 * TLA+ variable mapping (all under the per-chunk lock):
 *   slots[k] = <<ptr, ver>>   -> g.slots[k] : {bool is_c; uint64_t ver}
 *   disp                      -> g.disp  (OWNED/PUSHING/RELEASED or 1..K)
 *   bit_owned                 -> g.bit_owned (bool)
 *   mask_cnt                  -> g.mask_cnt  (0..2)
 *   released                  -> g.released  (0->1 once)
 *   pending_claim             -> g.pending_claim (single global flag, 36b)
 *   reown_pc[t]               -> thread-local enum reown_pc
 *   push_pc[t]                -> thread-local enum push_pc
 *   bad_release               -> g.bad_release (sticky)
 *
 * TLA+ action mapping (each = one locked critical section):
 *   PushP0 / PushP1 / PushP1GiveUp                 -> push_*()
 *   PopP0 / PopP1 / PopP2                          -> pop_*()
 *   FreeDecToZero_TriggersClaim / _Spurious        -> free_dec_to_zero()
 *   FreeDecCommon                                  -> free_dec_common()
 *   ClaimOWNED / ClaimSLOT / ClaimSLOTBail / ClaimBail -> claim_*()
 *   ReownAllocSlot                                 -> reown_alloc_slot()
 *
 * Each worker thread repeatedly picks (pseudo-randomly) one ENABLED action
 * and fires it under the lock, mirroring TLC's nondeterministic
 * interleaving via real thread scheduling.  A round ends when the chunk is
 * released or the per-thread step budget runs out.
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

/* Per-thread bound on action steps, so a livelocked interleaving cannot
 * spin forever (the TLA+ state space is finite; this caps the real-thread
 * analogue).  Generous: the longest legal trace is short. */
#ifndef MAX_STEPS_PER_THREAD
#define MAX_STEPS_PER_THREAD 200000
#endif

/* --- disp sentinel values (mirror the C++ enum / TLA+ definitions) --- */
#define DISP_OWNED     0
#define DISP_PUSHING   100
#define DISP_RELEASED  200
/* 1 .. K_SLOTS  => on-array at slot (disp-1) */

/* --- per-thread program counters --- */
enum reown_pc { RP_IDLE = 0, RP_GOT_C, RP_ARMED_OWNED };
enum push_pc  { PP_IDLE = 0, PP_PUSHING };

/* --- slot tagged word: ptr ("C"/"Null") + monotone version --- */
typedef struct { bool is_c; uint64_t ver; } slot_t;

/* ---------------------------------------------------------------------- */
/* Per-chunk lifecycle state (the TLA+ variables for one chunk).  Every    */
/* field is touched only while holding c->lock, so each action is atomic.  */
/* ---------------------------------------------------------------------- */
typedef struct {
    pthread_mutex_t lock;
    slot_t   slots[K_SLOTS];      /* slots[k] = <<ptr,ver>>            */
    int32_t  disp;                /* OWNED/PUSHING/RELEASED or 1..K    */
    bool     bit_owned;           /* BOOLEAN                           */
    int      mask_cnt;            /* 0..2                              */
    bool     released;            /* sticky FALSE -> TRUE              */
    bool     pending_claim;       /* single global flag (36b spec)     */
    bool     bad_release;         /* sticky; CAN fire in this model    */
} chunk_t;

static chunk_t      *g_chunks;        /* one per round-chunk               */
static int           g_num_chunks;
static _Atomic(int)  g_bad_observed;  /* OR over all chunks (reported)     */

/* ---------------------------------------------------------------------- */
/* Cheap thread-local PRNG for action selection (no shared state).         */
/* ---------------------------------------------------------------------- */
static inline uint32_t xorshift(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x ? x : 0x1234567u;
    return *s;
}

/* The ORPHAN_RELEASE_BAD DEBUG guard.  In the real C++ this is the assert
 * in orphan_release_self: at release, BIT_OWNED must be clear AND
 * MASK_CNT must be 0.  Evaluated atomically (we hold c->lock), exactly as
 * the TLA+ bad_release' = (... \/ bit_owned \/ mask_cnt>0) is computed in
 * the same step as the release.  In this BUG model the guard can fire; we
 * record it stickily instead of aborting. */
static inline void note_bad_release(chunk_t *c) {
    if (c->bit_owned || c->mask_cnt > 0) {
        c->bad_release = true;
        atomic_store_explicit(&g_bad_observed, 1, memory_order_relaxed);
    }
}

/* ====================================================================== */
/* ACTIONS  (each = one TLA+ atomic step; caller holds c->lock)            */
/* Each returns true iff it fired (was enabled and applied its effect).    */
/* ====================================================================== */

/* PushP0(t): disp OWNED -> PUSHING. */
static bool push_p0(chunk_t *c, enum push_pc *ppc) {
    if (*ppc != PP_IDLE) return false;
    if (c->released) return false;
    if (c->disp != DISP_OWNED) return false;
    c->disp = DISP_PUSHING;
    *ppc = PP_PUSHING;
    return true;
}

/* PushP1(t,k): empty slot (Null,vk) -> (C,vk+1); disp = k+1. */
static bool push_p1(chunk_t *c, enum push_pc *ppc, int k) {
    if (*ppc != PP_PUSHING) return false;
    if (c->disp != DISP_PUSHING) return false;
    if (c->slots[k].is_c) return false;           /* SlotPtr(k) = "Null" req */
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
        if (!c->slots[k].is_c) return false;      /* some slot empty */
    c->disp = DISP_OWNED;
    *ppc = PP_IDLE;
    return true;
}

/* PopP0(t,k): take on-array chunk (C,vk) -> (Null,vk+1).  disp NOT updated
 * (the documented stale-disp window). */
static bool pop_p0(chunk_t *c, enum reown_pc *rpc, int k) {
    if (*rpc != RP_IDLE) return false;
    if (c->released) return false;
    if (!c->slots[k].is_c) return false;          /* SlotPtr(k) = "C" req */
    c->slots[k].is_c = false;
    c->slots[k].ver += 1;
    *rpc = RP_GOT_C;
    return true;
}

/* PopP1(t): claim BIT_OWNED (false -> true), preserving MASK_CNT. */
static bool pop_p1(chunk_t *c, enum reown_pc *rpc) {
    if (*rpc != RP_GOT_C) return false;
    if (c->released) return false;
    if (c->bit_owned) return false;
    c->bit_owned = true;
    *rpc = RP_ARMED_OWNED;
    return true;
}

/* PopP2(t): publish disp = OWNED, closing the re-own. */
static bool pop_p2(chunk_t *c, enum reown_pc *rpc) {
    if (*rpc != RP_ARMED_OWNED) return false;
    c->disp = DISP_OWNED;
    *rpc = RP_IDLE;
    return true;
}

/* FreeDecToZero_TriggersClaim / _Spurious: atomicDecAndTest of the last
 * slot.  mask_cnt = 1 -> 0.  If bit_owned was clear AND pending_claim was
 * clear at this step, arm pending_claim (TriggersClaim); if bit_owned was
 * set, it is the spurious dec (no claim path). */
static bool free_dec_to_zero(chunk_t *c) {
    if (c->released) return false;
    if (c->mask_cnt != 1) return false;
    if (!c->bit_owned) {
        /* FreeDecToZero_TriggersClaim requires \neg pending_claim. */
        if (c->pending_claim) {
            /* Neither TriggersClaim (pending set) nor _Spurious (owned)
             * is enabled — this exact action is disabled in TLA+. */
            return false;
        }
        c->mask_cnt = 0;
        c->pending_claim = true;
        return true;
    }
    /* bit_owned set: FreeDecToZero_Spurious (no pending_claim guard). */
    c->mask_cnt = 0;
    return true;
}

/* FreeDecCommon: mask_cnt > 1 -> mask_cnt - 1 (benign multi-survivor dec). */
static bool free_dec_common(chunk_t *c) {
    if (c->released) return false;
    if (c->mask_cnt <= 1) return false;
    c->mask_cnt -= 1;
    return true;
}

/* ClaimOWNED(t): pending_claim && disp=OWNED -> RELEASED, release.
 * bad_release fires iff bit_owned OR mask_cnt>0 at THIS atomic step. */
static bool claim_owned(chunk_t *c) {
    if (!c->pending_claim) return false;
    if (c->released) return false;
    if (c->disp != DISP_OWNED) return false;
    c->disp = DISP_RELEASED;
    c->released = true;
    c->pending_claim = false;
    note_bad_release(c);                          /* DEBUG_GUARD, same step */
    return true;
}

/* ClaimSLOT(t,k): pending_claim && disp=k+1 && slot[k]="C" ->
 * (C,v)->(Null,v+1), disp=RELEASED, release. */
static bool claim_slot(chunk_t *c, int k) {
    if (!c->pending_claim) return false;
    if (c->released) return false;
    if (c->disp != (int32_t)(k + 1)) return false;
    if (!c->slots[k].is_c) return false;          /* SlotPtr(k) = "C" req */
    c->slots[k].is_c = false;
    c->slots[k].ver += 1;
    c->disp = DISP_RELEASED;
    c->released = true;
    c->pending_claim = false;
    note_bad_release(c);
    return true;
}

/* ClaimSLOTBail(t,k): pending_claim && disp=k+1 && slot[k]!="C" -> bail. */
static bool claim_slot_bail(chunk_t *c, int k) {
    if (!c->pending_claim) return false;
    if (c->released) return false;
    if (c->disp != (int32_t)(k + 1)) return false;
    if (c->slots[k].is_c) return false;           /* still "C": ClaimSLOT path */
    c->pending_claim = false;                     /* leak direction; no release */
    return true;
}

/* ClaimBail(t): pending_claim && disp in {PUSHING,RELEASED} -> bail. */
static bool claim_bail(chunk_t *c) {
    if (!c->pending_claim) return false;
    if (c->released) return false;
    if (c->disp != DISP_PUSHING && c->disp != DISP_RELEASED) return false;
    c->pending_claim = false;
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
/* try_action: take the lock, attempt one action kind, release.  Returns   */
/* whether it fired.  This is the single point where "one action = one     */
/* atomic step" is enforced.                                                */
/* ---------------------------------------------------------------------- */
static bool try_action(chunk_t *c, enum reown_pc *rpc, enum push_pc *ppc,
                       int kind, int k) {
    bool fired = false;
    pthread_mutex_lock(&c->lock);
    switch (kind) {
        case 0:  fired = push_p0(c, ppc);            break;
        case 1:  fired = push_p1(c, ppc, k);         break;
        case 2:  fired = push_p1_giveup(c, ppc);     break;
        case 3:  fired = pop_p0(c, rpc, k);          break;
        case 4:  fired = pop_p1(c, rpc);             break;
        case 5:  fired = pop_p2(c, rpc);             break;
        case 6:  fired = free_dec_to_zero(c);        break;
        case 7:  fired = free_dec_common(c);         break;
        case 8:  fired = claim_owned(c);             break;
        case 9:  fired = claim_slot(c, k);           break;
        case 10: fired = claim_slot_bail(c, k);      break;
        case 11: fired = claim_bail(c);              break;
        default: break;
    }
    pthread_mutex_unlock(&c->lock);
    return fired;
}

/* Drive THIS thread's own pending PC forward (used to avoid a stalled
 * round: a thread mid-re-own/mid-push always has an enabled continuation
 * in TLA+, so we ensure it eventually advances). */
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
/* Worker: each thread holds its own reown_pc/push_pc (per-thread PC).     */
/* ---------------------------------------------------------------------- */
typedef struct { int id; } thr_arg_t;

#define NUM_ACTION_KINDS 12

static void *worker(void *p) {
    thr_arg_t *a = (thr_arg_t *)p;
    uint32_t rng = 0x9e3779b9u ^ (uint32_t)(a->id * 2654435761u) ^ 0x1u;

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
            if (!try_action(c, &rpc, &ppc, kind, k)) {
                /* Nothing fired for the random pick: keep our own pending
                 * op moving so the round cannot stall on a thread mid-op. */
                drive_self(c, &rpc, &ppc);
            }
        }

        /* Drive any still-open per-thread op to completion so the round can
         * reach a quiescent terminal state (TLA+ progress guarantee). */
        while (ppc == PP_PUSHING || rpc == RP_GOT_C || rpc == RP_ARMED_OWNED) {
            if (!drive_self(c, &rpc, &ppc)) break;
        }
    }
    return NULL;
}

/* ---------------------------------------------------------------------- */
/* Terminal invariants.  For the BUG model we do NOT assert !bad_release   */
/* (the model exists to exhibit it).  We assert the invariants the design  */
/* genuinely upholds, plus consistency of any release that did happen.     */
/* ---------------------------------------------------------------------- */
static void check_terminal_invariants(chunk_t *c) {
    pthread_mutex_lock(&c->lock);
    int32_t  d   = c->disp;
    bool     rel = c->released;
    int      mc  = c->mask_cnt;
    bool     bo  = c->bit_owned;
    pthread_mutex_unlock(&c->lock);

    /* TypeOK: disp in DispDomain; mask_cnt in 0..2; bit_owned boolean. */
    bool disp_ok = (d == DISP_OWNED || d == DISP_PUSHING || d == DISP_RELEASED
                    || (d >= 1 && d <= K_SLOTS));
    assert(disp_ok);
    assert(mc >= 0 && mc <= 2);
    (void)bo;

    /* Inv_NoDoubleRelease (TLA+ keeps it trivially TRUE; the not-released
     * guards on the Claim actions enforce a single release).  released is
     * sticky FALSE->TRUE. */
    (void)rel;

    /* If released, disp must be RELEASED (the only paths that set released
     * also set disp = RELEASED). */
    if (rel) assert(d == DISP_RELEASED);
}

/* ---------------------------------------------------------------------- */
/* One bounded round: fresh chunk(s) at the TLA+ Init state, run workers,  */
/* join, check terminals.                                                   */
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
        c->pending_claim = false;
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
        printf("OrphanReuse_36b stress DONE: %lu rounds, %lu with ORPHAN_RELEASE_BAD, "
               "%d threads (bug model: bad_release is EXPECTED)\n",
               rounds, bad_rounds, NUM_THREADS);
    } else {
        int total = MAX_COMMITS > 0 ? MAX_COMMITS : 1;
        int bad = run_round(total);
        printf("OrphanReuse_36b unit OK: %d chunk(s), %d threads; "
               "ORPHAN_RELEASE_BAD observed this run = %s "
               "(bug model: either outcome is valid)\n",
               total, NUM_THREADS, bad ? "yes" : "no");
    }
    return 0;
}
