/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            kamepoolalloc/LICENSE-APACHE-2.0)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (see kamepoolalloc/LICENSE-GPL-2.0).
 ***************************************************************************/
/*
 * C11 test generated mechanically from OrphanChain_pathB.tla.
 *
 * Path-B (SEPARATE-COUNTS) variant of the orphan-chain chunk-release design.
 * Sibling of OrphanChain_atomicshared, but the intrusive refcount carries NO
 * count-based self-ref and there is no self-reset: the refcount is managed
 * PURELY by structural references —
 *
 *      Refcnt(n) = owner-ref(owned[n]) + chain-ref(ChainIn(n))
 *
 *   owner-ref  = the local_shared_ptr the owner's per-thread DLL holds.
 *   chain-ref  = head / a predecessor's m_orphan_next (atomic_shared_ptr).
 *   filled[n]  = MASK_CNT, a SEPARATE counter; Alloc/Free move it but NEVER
 *                touch the intrusive refcount (the whole point of Path B —
 *                no manual refcount ops, so the split local-tag counting of
 *                atomic_smart_ptr is never bypassed).
 *
 * "Don't release a non-empty chunk" is STRUCTURAL, not a self-ref: the sweeper
 * removes ONLY dead (filled=0) nodes and relink preserves successors, so a
 * non-empty node never loses its last incoming chain-ref; an owned chunk has
 * an owner-ref.  Hence Refcnt->0 => filled==0.
 *
 * As in the sibling port, TLA+ threads are NOT modelled explicitly: each Next
 * disjunct is ONE atomic step on a snapshot satisfying its guard.  We mirror
 * that with NUM_THREADS real worker threads, each repeatedly picking a node +
 * an action and executing exactly ONE TLA+ step.  TLA+ step atomicity is
 * encoded by serializing each whole action under a single global op-lock (the
 * model has no MODE_COARSE/FINE/SUPERFINE granularity to preserve, so there is
 * nothing finer to interleave: what matters is *which* atomic step fires next,
 * which the random per-thread choice + lock exercises faithfully).
 *
 * TLA+ variable mapping:
 *   head                 -> _Atomic(int) g_head           (node id or NIL)
 *   nxt[n]               -> _Atomic(int) g_nxt[n]
 *   filled[n]            -> _Atomic(int) g_filled[n]       (0 or 1) [MASK_CNT]
 *   owned[n]             -> _Atomic(int) g_owned[n]        (0 or 1) [owner-ref]
 *   released[n]          -> _Atomic(int) g_released[n]     (0 or 1)
 *   gen                  -> _Atomic(int) g_gen             (adopts; <= MaxGen)
 *   pushes               -> _Atomic(int) g_pushes          (owner-exit pushes; <= MaxPush)
 *   bad_release          -> _Atomic(int) g_bad_release     (sticky DEBUG_GUARD)
 *
 * Reference accounting (atomic_shared_ptr abstracted):
 *   ChainIn(n)   = #{m : nxt[m]==n} + (head==n)
 *   Refcnt(n)    = (owned[n] ? 1 : 0) + ChainIn(n)        [NO self-ref term]
 * computed by scanning the (small, fixed) node set inside the locked action.
 * Release fires only at Refcnt==0 (the mutual exclusion the spec leans on);
 * Adopt is gated on ~owned (try_promote of a head node, chain-ref -> owner-ref).
 *
 * KNOB AllowLiveRemoval (default FALSE = the design): when TRUE the sweeper /
 * head-advance may drop a non-empty node, which then loses its last ref and is
 * released while filled>0 — the negative control.  In that build we assert the
 * load-bearing failure (bad_release) actually manifested instead of asserting
 * ~bad_release.
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

/* ============================ Configuration ============================== */

/* TLA+ model values N1,N2,N3 / NIL. */
#define N1   0
#define N2   1
#define N3   2
#define NUM_NODES 3
#define NIL  (-1)        /* end-of-list / reset sentinel, distinct from nodes */

/* TLA+ CONSTANTS (pathB cfg defaults): AllowReown=TRUE, AllowLiveRemoval=FALSE,
 * MaxGen=1, MaxPush=3.  Override at compile time to reproduce the other
 * .cfg knobs:
 *   -DALLOW_LIVE_REMOVAL=1        -> liveremoval cfg (expect bad_release)
 */
#ifndef ALLOW_REOWN
#define ALLOW_REOWN 1
#endif
#ifndef ALLOW_LIVE_REMOVAL
#define ALLOW_LIVE_REMOVAL 0
#endif
#ifndef MAX_GEN
#define MAX_GEN 1
#endif
#ifndef MAX_PUSH
#define MAX_PUSH 3
#endif

/* Number of concurrent worker threads interleaving atomic steps. */
#ifndef NUM_THREADS
#define NUM_THREADS 2
#endif

/* Stress duration in seconds (0 = bounded unit run). */
#ifndef STRESS_SECONDS
#define STRESS_SECONDS 0
#endif

/* Per-thread iteration budget (each iteration = one attempted TLA+ step).
 * Default large-but-bounded for unit (the state space is microscopic and the
 * terminal "all reclaimable chunks released" attractor is reached fast);
 * unbounded for stress. */
#ifndef MAX_COMMITS
#  if STRESS_SECONDS > 0
#    define MAX_COMMITS 0x7fffffff
#  else
#    define MAX_COMMITS 200000
#  endif
#endif

/* When AllowLiveRemoval=TRUE the spec EXPECTS Inv_NoBadRelease to be violated
 * (a live node falling off the chain is released).  In that build we therefore
 * do NOT assert ~bad_release at the end — we assert it was observed instead
 * (load-bearing dead-only-removal demonstration). */
#ifndef EXPECT_BAD_RELEASE
#  if ALLOW_LIVE_REMOVAL
#    define EXPECT_BAD_RELEASE 1
#  else
#    define EXPECT_BAD_RELEASE 0
#  endif
#endif

/* ============================ Shared state =============================== */

static _Atomic(int) g_head;
static _Atomic(int) g_nxt[NUM_NODES];
static _Atomic(int) g_filled[NUM_NODES];
static _Atomic(int) g_owned[NUM_NODES];
static _Atomic(int) g_released[NUM_NODES];
static _Atomic(int) g_gen;
static _Atomic(int) g_pushes;
static _Atomic(int) g_bad_release;
static _Atomic(int) g_stop;

/* Single global op-lock: serializes each whole TLA+ action so it observes a
 * consistent snapshot of (head,nxt,filled,owned,released) and commits
 * atomically — the encoding of "each action = one atomic memory access". */
static pthread_mutex_t op_mtx = PTHREAD_MUTEX_INITIALIZER;
#define OP_LOCK()   pthread_mutex_lock(&op_mtx)
#define OP_UNLOCK() pthread_mutex_unlock(&op_mtx)

/* Relaxed reads/writes are fine: all mutation happens under op_mtx, whose
 * lock/unlock provide the release/acquire fences.  Post-join reads are after
 * all joins (a happens-before edge), so plain loads suffice there too. */
static inline int LD(_Atomic(int) *p) {
    return atomic_load_explicit(p, memory_order_relaxed);
}
static inline void ST(_Atomic(int) *p, int v) {
    atomic_store_explicit(p, v, memory_order_relaxed);
}

/* ===================== Reference accounting (Layer 0) ==================== */
/* ChainIn(n): incoming structural refs = #{m : nxt[m]==n} + (head==n).
 * Must be called with op_mtx held (consistent snapshot). */
static int chain_in(int n) {
    int c = 0;
    for (int m = 0; m < NUM_NODES; m++)
        if (LD(&g_nxt[m]) == n) c++;
    if (LD(&g_head) == n) c++;
    return c;
}

/* Refcnt(n) = owner-ref(owned[n]) + ChainIn(n).  NO self-ref term — this is
 * the Path-B separate-counts variant. */
static int refcnt(int n) {
    int r = chain_in(n);
    if (LD(&g_owned[n])) r++;
    return r;
}

/* ============================== Actions ================================== */
/* Each returns 1 if it fired (guard was enabled and committed), 0 otherwise.
 * Mirrors one TLA+ Next disjunct.  All run under OP_LOCK held by the caller.
 * refcnt moves ONLY on owner-ref / chain-ref edges, NEVER on Alloc/Free. */

/* Alloc(n): owned /\ ~released /\ filled=0 -> filled 0->1.  Refcnt UNCHANGED. */
static int act_alloc(int n) {
    if (!LD(&g_owned[n])) return 0;
    if (LD(&g_released[n])) return 0;
    if (LD(&g_filled[n]) != 0) return 0;
    ST(&g_filled[n], 1);
    return 1;
}

/* Free(n): filled=1 /\ ~released -> filled 1->0.  Refcnt UNCHANGED (separate
 * counts — free does NOT touch the intrusive refcount). */
static int act_free(int n) {
    if (LD(&g_filled[n]) != 1) return 0;
    if (LD(&g_released[n])) return 0;
    ST(&g_filled[n], 0);
    return 1;
}

/* OwnerExitEmpty(n): owned /\ filled=0 /\ ~released -> owned:=FALSE.
 * Drops the owner-ref (no chain involvement). */
static int act_owner_exit_empty(int n) {
    if (!LD(&g_owned[n])) return 0;
    if (LD(&g_filled[n]) != 0) return 0;
    if (LD(&g_released[n])) return 0;
    ST(&g_owned[n], 0);
    return 1;
}

/* OwnerExitNonEmpty(n): owned /\ filled=1 /\ ~released /\ ChainIn(n)=0 /\
 *   pushes<MaxPush  ->  owned:=FALSE; nxt[n]:=head; head:=n; pushes++.
 * Transfer owner-ref -> chain-ref via a Treiber head push (atomic). */
static int act_owner_exit_nonempty(int n) {
    if (!LD(&g_owned[n])) return 0;
    if (LD(&g_filled[n]) != 1) return 0;
    if (LD(&g_released[n])) return 0;
    if (chain_in(n) != 0) return 0;          /* not already on the chain */
    if (LD(&g_pushes) >= MAX_PUSH) return 0;
    ST(&g_owned[n], 0);
    ST(&g_nxt[n], LD(&g_head));
    ST(&g_head, n);
    ST(&g_pushes, LD(&g_pushes) + 1);
    return 1;
}

/* SweepRelink(p): nxt[p] in Nodes /\ ~released[p] /\ ~released[nxt[p]] /\
 *   (filled[nxt[p]]=0 \/ AllowLiveRemoval)  ->  nxt[p] := nxt[nxt[p]].
 * Path B removes ONLY dead (filled=0) successors; relink preserves the
 * successor's successor, so a live node is never dropped. */
static int act_sweep_relink(int p) {
    int c = LD(&g_nxt[p]);
    if (c == NIL) return 0;                   /* nxt[p] not a Node */
    if (LD(&g_released[p])) return 0;
    if (LD(&g_released[c])) return 0;
#if !ALLOW_LIVE_REMOVAL
    if (LD(&g_filled[c]) != 0) return 0;      /* dead-only guard (the design) */
#endif
    ST(&g_nxt[p], LD(&g_nxt[c]));
    return 1;
}

/* HeadAdvance: head in Nodes /\ ~released[head] /\
 *   (filled[head]=0 \/ AllowLiveRemoval)  ->  head := nxt[head]. */
static int act_head_advance(void) {
    int h = LD(&g_head);
    if (h == NIL) return 0;
    if (LD(&g_released[h])) return 0;
#if !ALLOW_LIVE_REMOVAL
    if (LD(&g_filled[h]) != 0) return 0;      /* dead-only guard (the design) */
#endif
    ST(&g_head, LD(&g_nxt[h]));
    return 1;
}

/* Adopt(n): AllowReown /\ gen<MaxGen /\ head=n /\ ~released[n] /\ ~owned[n]
 *   ->  owned:=TRUE; head:=nxt[n]; gen++.  Pop the single head node and
 * re-own it (chain-ref -> owner-ref).  NOT the whole list. */
static int act_adopt(int n) {
#if ALLOW_REOWN
    if (LD(&g_gen) >= MAX_GEN) return 0;
    if (LD(&g_head) != n) return 0;
    if (LD(&g_released[n])) return 0;
    if (LD(&g_owned[n])) return 0;
    ST(&g_owned[n], 1);
    ST(&g_head, LD(&g_nxt[n]));
    ST(&g_gen, LD(&g_gen) + 1);
    return 1;
#else
    (void)n; return 0;
#endif
}

/* Release(n): ~released /\ Refcnt(n)=0  ->  released:=TRUE; nxt:=NIL;
 *   bad_release |= (filled[n]>0).  The deleter (refcount hit 0). */
static int act_release(int n) {
    if (LD(&g_released[n])) return 0;
    if (refcnt(n) != 0) return 0;
    ST(&g_released[n], 1);
    ST(&g_nxt[n], NIL);
    if (LD(&g_filled[n]) > 0)
        ST(&g_bad_release, 1);                /* sticky DEBUG_GUARD */
    return 1;
}

/* ====================== Per-step invariant checks ======================== */
/* The TLA+ safety invariants, checked while op_mtx is held so the state is a
 * consistent post-step snapshot (mirrors TLC checking invariants on each
 * reachable state).  Under AllowLiveRemoval=FALSE all must hold throughout;
 * under AllowLiveRemoval=TRUE we skip Inv_NoBadRelease / Inv_NonEmptyHasRef
 * (the model itself flags those as the expected violations). */
static void check_invariants_locked(void) {
#if !EXPECT_BAD_RELEASE
    /* (A) Inv_NoBadRelease */
    assert(!LD(&g_bad_release));
    /* (B) Inv_NonEmptyHasRef: ~released[n] /\ filled[n]>0 => Refcnt(n) >= 1 */
    for (int n = 0; n < NUM_NODES; n++)
        if (!LD(&g_released[n]) && LD(&g_filled[n]) > 0)
            assert(refcnt(n) >= 1);
#endif
    /* (C) Inv_NoDanglingNext: ~released[m] /\ nxt[m] in Nodes => ~released[nxt[m]] */
    for (int m = 0; m < NUM_NODES; m++) {
        int x = LD(&g_nxt[m]);
        if (!LD(&g_released[m]) && x != NIL)
            assert(!LD(&g_released[x]));
    }
    /* (D) Inv_HeadAlive: head in Nodes => ~released[head] */
    {
        int h = LD(&g_head);
        if (h != NIL) assert(!LD(&g_released[h]));
    }
    /* (E) Inv_ReleasedNoRefs: released[n] => Refcnt(n)=0 */
    for (int n = 0; n < NUM_NODES; n++)
        if (LD(&g_released[n])) assert(refcnt(n) == 0);
    /* (F) Inv_Acyclic: following nxt |Nodes| times from any node reaches NIL */
    for (int n = 0; n < NUM_NODES; n++) {
        int cur = n;
        for (int i = 0; i < NUM_NODES && cur != NIL; i++)
            cur = LD(&g_nxt[cur]);
        assert(cur == NIL);
    }
}

/* ============================== Worker =================================== */

/* xorshift32 PRNG — per-thread, no shared RNG state. */
static inline uint32_t xs32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return (*s = x);
}

typedef struct { uint32_t tid; uint32_t seed; } ThreadCtx;

/* Try one randomly-chosen TLA+ Next disjunct.  Returns 1 if a step fired.
 * Each call = one atomic step (the lock makes the chosen action atomic). */
static int try_one_step(uint32_t *seed) {
    int n = (int)(xs32(seed) % NUM_NODES);
    int which = (int)(xs32(seed) % 8);
    int fired = 0;

    OP_LOCK();
    switch (which) {
        case 0: fired = act_alloc(n);              break;
        case 1: fired = act_free(n);               break;
        case 2: fired = act_owner_exit_empty(n);   break;
        case 3: fired = act_owner_exit_nonempty(n);break;
        case 4: fired = act_sweep_relink(n);       break;
        case 5: fired = act_adopt(n);              break;
        case 6: fired = act_release(n);            break;
        case 7: fired = act_head_advance();        break;
        default: break;
    }
    /* Check the safety invariants on every reachable post-step state. */
    check_invariants_locked();
    OP_UNLOCK();
    return fired;
}

static void *worker(void *arg) {
    ThreadCtx *ctx = (ThreadCtx *)arg;
    uint32_t seed = ctx->seed;
    for (uint32_t i = 0; i < (uint32_t)MAX_COMMITS; i++) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) break;
        (void)try_one_step(&seed);
    }
    return NULL;
}

/* ============================ Terminal check ============================= */
/* Drive the model to quiescence: from any reachable state, repeatedly fire the
 * reclaim actions (owner-exit-empty, sweep-relink, head-advance, release) plus
 * a bounded Free of the still-live nodes and an owner-exit of any survivors,
 * until no reclaim step is enabled.  This mirrors the SpecLive fairness (WF on
 * HeadAdvance/SweepRelink/Release/OwnerExit*) that the *_live cfg checks for
 * Liveness (no leak): every chunk that becomes empty and unreferenced is
 * eventually released.  Single-threaded, after all workers joined. */
static void drive_to_quiescence(void) {
    int progress = 1;
    int guard = 0;
    const int GUARD_MAX = 100000;    /* livelock tripwire */
    while (progress && guard++ < GUARD_MAX) {
        progress = 0;
        OP_LOCK();
        /* Free every live node so the no-leak terminus is reachable (the
         * "stays empty forever" antecedent of Liveness made concrete). */
        for (int n = 0; n < NUM_NODES; n++) progress |= act_free(n);
        /* Drop owner-refs of now-empty owned nodes so they become releasable. */
        for (int n = 0; n < NUM_NODES; n++) progress |= act_owner_exit_empty(n);
        /* Reclaim: relink past / advance-head past dead nodes, then release
         * whatever now has Refcnt==0. */
        for (int p = 0; p < NUM_NODES; p++) progress |= act_sweep_relink(p);
        progress |= act_head_advance();
        for (int n = 0; n < NUM_NODES; n++) progress |= act_release(n);
        check_invariants_locked();
        OP_UNLOCK();
    }
    assert(guard < GUARD_MAX);   /* did not livelock reaching quiescence */
}

#if EXPECT_BAD_RELEASE
/* Negative control (AllowLiveRemoval=TRUE).  Re-initialise to a clean Init
 * state and replay the deterministic adversarial schedule that the spec
 * predicts will violate Inv_NoBadRelease — exactly the trace TLC reports for
 * the liveremoval cfg.  This is a single-threaded, post-join demonstration of
 * a model-predicted counterexample, so replaying from Init under a fixed
 * schedule is the faithful encoding (same as TLC exploring that path):
 *
 *   Init               head=NIL, N1 owned+filled, ...
 *   OwnerExitNonEmpty(N1)  push N1 to head: head=N1, owned[N1]=0, filled[N1]=1,
 *                          nxt[N1]=NIL.  Now Refcnt(N1)=1 (chain-ref via head).
 *   HeadAdvance            (live removal lifted) head:=nxt[N1]=NIL.  Now N1 is
 *                          off the chain and not owned: Refcnt(N1)=0, filled=1.
 *   Release(N1)            Refcnt==0 fires Release on a FILLED node ->
 *                          bad_release := TRUE.  VIOLATION.
 *
 * Asserting on the *_locked invariant set is suppressed for (A)/(B) under
 * EXPECT_BAD_RELEASE, so the intentional violation does not abort the run;
 * the structural invariants (C)-(F) still hold and are checked. */
static void replay_bad_release_trace(void) {
    OP_LOCK();
    /* Re-init (clean Init state). */
    ST(&g_head, NIL);
    for (int n = 0; n < NUM_NODES; n++) ST(&g_nxt[n], NIL);
    ST(&g_filled[N1], 1);
    ST(&g_filled[N2], 1);
    ST(&g_filled[N3], 0);
    for (int n = 0; n < NUM_NODES; n++) ST(&g_owned[n], 1);
    for (int n = 0; n < NUM_NODES; n++) ST(&g_released[n], 0);
    ST(&g_gen, 0);
    ST(&g_pushes, 0);
    /* g_bad_release intentionally NOT cleared — sticky across the whole run. */

    /* OwnerExitNonEmpty(N1): transfer owner-ref -> chain-ref (head push). */
    int r1 = act_owner_exit_nonempty(N1);
    assert(r1);
    check_invariants_locked();
    /* HeadAdvance with AllowLiveRemoval: drop the FILLED head N1 off the chain. */
    int r2 = act_head_advance();
    assert(r2);
    check_invariants_locked();
    /* Release(N1): Refcnt now 0 though filled>0 -> sets bad_release. */
    int r3 = act_release(N1);
    assert(r3);
    check_invariants_locked();
    OP_UNLOCK();
}
#endif

int main(void) {
    /* TLA+ Init: head=NIL; nxt all NIL; filled: N3=0 else 1; owned all TRUE;
     *            released=FALSE; gen=pushes=0; bad_release=FALSE. */
    ST(&g_head, NIL);
    for (int n = 0; n < NUM_NODES; n++) ST(&g_nxt[n], NIL);
    ST(&g_filled[N1], 1);
    ST(&g_filled[N2], 1);
    ST(&g_filled[N3], 0);
    for (int n = 0; n < NUM_NODES; n++) ST(&g_owned[n], 1);
    for (int n = 0; n < NUM_NODES; n++) ST(&g_released[n], 0);
    ST(&g_gen, 0);
    ST(&g_pushes, 0);
    ST(&g_bad_release, 0);
    ST(&g_stop, 0);

    /* Initial state must already satisfy the invariants (TLC checks Init). */
    OP_LOCK();
    check_invariants_locked();
    OP_UNLOCK();

    pthread_t threads[NUM_THREADS];
    ThreadCtx ctxs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        ctxs[i].tid  = (uint32_t)(i + 1);
        ctxs[i].seed = 0x9e3779b9u ^ (uint32_t)(i * 2654435761u) ^ (uint32_t)time(NULL);
        if (ctxs[i].seed == 0) ctxs[i].seed = 1u;   /* xorshift32 needs nonzero */
        pthread_create(&threads[i], NULL, worker, &ctxs[i]);
    }

#if STRESS_SECONDS > 0
    struct timespec ts = { .tv_sec = STRESS_SECONDS, .tv_nsec = 0 };
    nanosleep(&ts, NULL);
    atomic_store_explicit(&g_stop, 1, memory_order_release);
#endif

    for (int i = 0; i < NUM_THREADS; i++) pthread_join(threads[i], NULL);

    /* Concurrent phase done: all safety invariants held at every step
     * (asserted inside the lock).  Now drive to the no-leak terminus. */
    drive_to_quiescence();

    /* ---------------------- Post-join terminal invariant ---------------- */
    /* Re-check the full safety invariant set on the final quiescent state. */
    OP_LOCK();
    check_invariants_locked();
    OP_UNLOCK();

#if EXPECT_BAD_RELEASE
    /* Negative control (AllowLiveRemoval=TRUE): the spec EXPECTS
     * Inv_NoBadRelease to be violated — a non-empty node dropped from the
     * chain loses its last ref and is released.  The random phase may or may
     * not have stumbled into that window, so replay the deterministic
     * adversarial schedule TLC reports for the liveremoval cfg, then confirm
     * the load-bearing failure manifested. */
    if (!LD(&g_bad_release))
        replay_bad_release_trace();
    assert(LD(&g_bad_release));
#else
    /* TLA+ Inv_NoBadRelease: no chunk released with live slots. */
    assert(!LD(&g_bad_release));

    /* Liveness terminus (the *_live cfg's no-leak property, made concrete):
     * after quiescence every chunk that is empty AND unreferenced is released,
     * and every released chunk has Refcnt==0, nxt==NIL, filled==0.  A chunk
     * still filled (re-owned via Adopt and never re-freed) legitimately
     * survives — drive_to_quiescence frees it, so post-drive every node is
     * empty; an owned-empty node is owner-exited and released too. */
    for (int n = 0; n < NUM_NODES; n++) {
        if (LD(&g_filled[n]) == 0 && refcnt(n) == 0) {
            /* empty and unreferenced and quiescent => reclaimed (no
             * permanently-stranded empty orphan): mirrors Liveness
             * <>[](filled=0 /\ Refcnt=0) => <>released. */
            assert(LD(&g_released[n]));
        }
        if (LD(&g_released[n])) {
            assert(refcnt(n) == 0);         /* Inv_ReleasedNoRefs */
            assert(LD(&g_nxt[n]) == NIL);   /* freed header pins nothing */
            assert(LD(&g_filled[n]) == 0);  /* Inv_NonEmptyHasRef corollary */
        }
    }
    /* head is NIL (whole chain reclaimed) or a live, unreleased node. */
    {
        int h = LD(&g_head);
        if (h != NIL) assert(!LD(&g_released[h]));   /* Inv_HeadAlive */
    }
#endif

    /* Bounds (TLA+ TypeOK on the counters). */
    assert(LD(&g_gen) >= 0 && LD(&g_gen) <= MAX_GEN);
    assert(LD(&g_pushes) >= 0 && LD(&g_pushes) <= MAX_PUSH);

    return 0;
}
