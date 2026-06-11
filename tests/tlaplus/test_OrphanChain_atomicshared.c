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
 * C11 test generated mechanically from OrphanChain_atomicshared.tla.
 *
 * Intrusive orphan/reuse chain refcounted by an abstracted atomic_shared_ptr
 * (Layer 1).  The chain is a singly-linked list of 3 chunks
 *      head -> N1 -> N2(dead) -> N3 -> NIL,   filled: N1=1,N2=0,N3=1.
 * Every modelled operation in the TLA+ spec is a SINGLE atomic memory access
 * with NO thread-local program counter ("Threads are NOT modelled explicitly:
 * the interleaving of node-indexed atomic steps already covers N concurrent
 * sweepers/freers/adopters/pushers").  We mirror that here: NUM_THREADS real
 * worker threads each repeatedly pick a node + an action and execute exactly
 * ONE TLA+ step.  TLA+ step atomicity (a Next disjunct fires atomically on a
 * snapshot satisfying its guard) is encoded by serializing each whole action
 * under a single global op-lock — the legitimate encoding of "each action =
 * one atomic memory access" for a model that has no MODE_COARSE/FINE/SUPERFINE
 * granularity to preserve.  Because the model's steps are coarse (whole-state
 * predicates over head/nxt/filled), there is nothing finer to interleave: the
 * concurrency that matters is the interleaving of *which* atomic step fires
 * next, which the random per-thread choice + lock exercises faithfully.
 *
 * TLA+ variable mapping:
 *   head                 -> _Atomic(int) g_head           (node id or NIL)
 *   nxt[n]               -> _Atomic(int) g_nxt[n]
 *   filled[n]            -> _Atomic(int) g_filled[n]       (0 or 1)
 *   released[n]          -> _Atomic(int) g_released[n]     (0 or 1)
 *   gen                  -> _Atomic(int) g_gen            (revivals; <= MaxGen)
 *   pushes               -> _Atomic(int) g_pushes         (head inserts; <= MaxPush)
 *   bad_release          -> _Atomic(int) g_bad_release    (sticky DEBUG_GUARD)
 *
 * Reference accounting (atomic_shared_ptr abstracted, from the §atomic_shared
 * port's global_rc idiom):
 *   ChainIn(n)   = #{m : nxt[m]==n} + (head==n)
 *   StructRefs(n)= ChainIn(n) + (SelfRef && filled[n]>0 ? 1 : 0)
 * computed by scanning the (small, fixed) node set inside the locked action,
 * exactly like obj_refcnt is read in the reference's release path.  The
 * refcount is the "captured generation": Revive is gated on StructRefs>0
 * (a successful try_promote), Release fires only at StructRefs==0 — the
 * mutual exclusion the spec leans on.
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

/* TLA+ CONSTANTS (selfref cfg defaults): SelfRef=TRUE, AllowReown=TRUE,
 * MaxGen=1, MaxPush=0.  Override at compile time to reproduce the other
 * .cfg knobs:
 *   -DSELF_REF=0                 -> noselfref cfg (expect bad_release, see below)
 *   -DMAX_PUSH=1                 -> push cfg (head-insert race enabled)
 */
#ifndef SELF_REF
#define SELF_REF 1
#endif
#ifndef ALLOW_REOWN
#define ALLOW_REOWN 1
#endif
#ifndef MAX_GEN
#define MAX_GEN 1
#endif
#ifndef MAX_PUSH
#define MAX_PUSH 0
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
 * Default 1 for unit (kept tiny — the state space is microscopic and the
 * terminal "all reclaimable chunks released" attractor is reached fast);
 * large for stress. */
#ifndef MAX_COMMITS
#  if STRESS_SECONDS > 0
#    define MAX_COMMITS 0x7fffffff
#  else
#    define MAX_COMMITS 200000
#  endif
#endif

/* When SELF_REF=0 the spec EXPECTS Inv_NoBadRelease to be violated (the
 * negative control: a live node falling off the chain is released).  In that
 * build we therefore do NOT assert ~bad_release at the end — we assert it was
 * observed instead (load-bearing self-ref demonstration). */
#ifndef EXPECT_BAD_RELEASE
#  if SELF_REF
#    define EXPECT_BAD_RELEASE 0
#  else
#    define EXPECT_BAD_RELEASE 1
#  endif
#endif

/* ============================ Shared state =============================== */

static _Atomic(int) g_head;
static _Atomic(int) g_nxt[NUM_NODES];
static _Atomic(int) g_filled[NUM_NODES];
static _Atomic(int) g_released[NUM_NODES];
static _Atomic(int) g_gen;
static _Atomic(int) g_pushes;
static _Atomic(int) g_bad_release;
static _Atomic(int) g_stop;

/* Single global op-lock: serializes each whole TLA+ action so it observes a
 * consistent snapshot of (head,nxt,filled,released) and commits atomically —
 * the encoding of "each action = one atomic memory access". */
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

/* StructRefs(n) = ChainIn(n) + self-ref (iff enabled and filled). */
static int struct_refs(int n) {
    int r = chain_in(n);
#if SELF_REF
    if (LD(&g_filled[n]) > 0) r++;
#endif
    return r;
}

/* ============================== Actions ================================== */
/* Each returns 1 if it fired (guard was enabled and committed), 0 otherwise.
 * Mirrors one TLA+ Next disjunct.  All run under OP_LOCK held by the caller.
 * Marked unused so any knob combination (e.g. the negative control, which
 * only replays SelfResetNext + Release) stays at zero warnings. */
#define ACTION __attribute__((unused)) static int

/* Free(n): filled 1->0, ~released.  Drops the self-ref. */
ACTION act_free(int n) {
    if (LD(&g_filled[n]) != 1) return 0;
    if (LD(&g_released[n])) return 0;
    ST(&g_filled[n], 0);
    return 1;
}

/* Revive(n): AllowReown /\ gen<MaxGen /\ filled=0 /\ ~released /\
 *            StructRefs(n)>0  ->  filled 0->1, gen++.  (try_promote gate.) */
ACTION act_revive(int n) {
#if ALLOW_REOWN
    if (LD(&g_gen) >= MAX_GEN) return 0;
    if (LD(&g_filled[n]) != 0) return 0;
    if (LD(&g_released[n])) return 0;
    if (struct_refs(n) <= 0) return 0;      /* gated: still has a live ref */
    ST(&g_filled[n], 1);
    ST(&g_gen, LD(&g_gen) + 1);
    return 1;
#else
    (void)n; return 0;
#endif
}

/* SweepRelink(p): nxt[p] in Nodes /\ filled[nxt[p]]=0 /\ ~released[p] /\
 *                 ~released[nxt[p]]  ->  nxt[p] := nxt[nxt[p]]. */
ACTION act_sweep_relink(int p) {
    int c = LD(&g_nxt[p]);
    if (c == NIL) return 0;                 /* nxt[p] not a Node */
    if (LD(&g_filled[c]) != 0) return 0;    /* successor not dead */
    if (LD(&g_released[p])) return 0;
    if (LD(&g_released[c])) return 0;
    ST(&g_nxt[p], LD(&g_nxt[c]));
    return 1;
}

/* SelfResetNext(c): filled=0 /\ ~released /\ nxt[c] in Nodes -> nxt[c]:=NIL. */
ACTION act_self_reset_next(int c) {
    if (LD(&g_filled[c]) != 0) return 0;
    if (LD(&g_released[c])) return 0;
    if (LD(&g_nxt[c]) == NIL) return 0;     /* nxt[c] not a Node */
    ST(&g_nxt[c], NIL);
    return 1;
}

/* HeadAdvance: head in Nodes /\ filled[head]=0 /\ ~released[head] ->
 *              head := nxt[head]. */
ACTION act_head_advance(void) {
    int h = LD(&g_head);
    if (h == NIL) return 0;
    if (LD(&g_filled[h]) != 0) return 0;
    if (LD(&g_released[h])) return 0;
    ST(&g_head, LD(&g_nxt[h]));
    return 1;
}

/* Push(n): pushes<MaxPush /\ filled[n]=1 /\ ~released[n] /\ ChainIn(n)=0 ->
 *          nxt[n] := head; head := n; pushes++. */
ACTION act_push(int n) {
    if (LD(&g_pushes) >= MAX_PUSH) return 0;
    if (LD(&g_filled[n]) != 1) return 0;
    if (LD(&g_released[n])) return 0;
    if (chain_in(n) != 0) return 0;         /* n is off the shared chain */
    ST(&g_nxt[n], LD(&g_head));
    ST(&g_head, n);
    ST(&g_pushes, LD(&g_pushes) + 1);
    return 1;
}

/* Release(n): ~released /\ StructRefs(n)=0  ->  released:=TRUE; nxt:=NIL;
 *             bad_release |= (filled[n]>0).  The deleter (refcount hit 0). */
ACTION act_release(int n) {
    if (LD(&g_released[n])) return 0;
    if (struct_refs(n) != 0) return 0;
    ST(&g_released[n], 1);
    ST(&g_nxt[n], NIL);
    if (LD(&g_filled[n]) > 0)
        ST(&g_bad_release, 1);              /* sticky DEBUG_GUARD */
    return 1;
}

/* ====================== Per-step invariant checks ======================== */
/* The TLA+ safety invariants, checked while op_mtx is held so the state is a
 * consistent post-step snapshot (mirrors TLC checking invariants on each
 * reachable state).  Under SELF_REF=1 all must hold throughout; under
 * SELF_REF=0 we skip Inv_NoBadRelease / Inv_LiveNeverReleased (the model
 * itself flags those as the expected violations). */
static void check_invariants_locked(void) {
#if !EXPECT_BAD_RELEASE
    /* (A) Inv_NoBadRelease */
    assert(!LD(&g_bad_release));
    /* (B) Inv_LiveNeverReleased: filled>0 => ~released */
    for (int n = 0; n < NUM_NODES; n++)
        assert(!(LD(&g_filled[n]) > 0 && LD(&g_released[n])));
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
    /* (E) Inv_ReleasedNoIncoming: released[n] => StructRefs(n)=0 */
    for (int n = 0; n < NUM_NODES; n++)
        if (LD(&g_released[n])) assert(struct_refs(n) == 0);
    /* (F) Inv_Acyclic: following nxt |Nodes| times from any node reaches NIL */
    for (int n = 0; n < NUM_NODES; n++) {
        int cur = n;
        for (int i = 0; i < NUM_NODES && cur != NIL; i++)
            cur = LD(&g_nxt[cur]);
        assert(cur == NIL);
    }
}

/* ============================== Worker =================================== */
/* The concurrent random-interleaving harness drives the POSITIVE builds
 * (SELF_REF=1).  The negative control replays a fixed trace instead, so this
 * whole section is compiled out there (keeps -Wunused-function at zero). */
#if !EXPECT_BAD_RELEASE

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
    int which = (int)(xs32(seed) % 7);
    int fired = 0;

    OP_LOCK();
    switch (which) {
        case 0: fired = act_free(n);            break;
        case 1: fired = act_revive(n);          break;
        case 2: fired = act_sweep_relink(n);    break;
        case 3: fired = act_self_reset_next(n); break;
        case 4: fired = act_push(n);            break;
        case 5: fired = act_release(n);         break;
        case 6: fired = act_head_advance();     break;
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
/* Drive the model to quiescence: from any reachable state, repeatedly fire
 * the reclaim actions (HeadAdvance / SweepRelink / Release) plus a bounded
 * Free of the still-live nodes, until no reclaim/free step is enabled.  This
 * mirrors the SpecLive fairness (WF on HeadAdvance/SweepRelink/Release) that
 * the *_live cfg checks for Liveness (no leak): every chunk that can become
 * empty and unreferenced is eventually released.  Single-threaded, after all
 * workers joined. */
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
        /* Reclaim: relink past / null-out / advance-head past dead nodes,
         * then release whatever now has StructRefs==0. */
        for (int p = 0; p < NUM_NODES; p++) progress |= act_sweep_relink(p);
        for (int c = 0; c < NUM_NODES; c++) progress |= act_self_reset_next(c);
        progress |= act_head_advance();
        for (int n = 0; n < NUM_NODES; n++) progress |= act_release(n);
        check_invariants_locked();
        OP_UNLOCK();
    }
    assert(guard < GUARD_MAX);   /* did not livelock reaching quiescence */
}

#endif  /* !EXPECT_BAD_RELEASE */

#if EXPECT_BAD_RELEASE
/* Negative control (SELF_REF=0): deterministically REPLAY the canonical TLC
 * counterexample for Inv_NoBadRelease, then assert the guard fired.  TLC
 * reports the violation as a *reachable* trace, not an inevitable one, so the
 * faithful C demonstration replays that exact path rather than hoping the
 * random concurrent schedule stumbles onto it.
 *
 * Trace (from the noselfref cfg comment): a dead predecessor self-resets its
 * `next` before a sweeper relinks, so a LIVE successor's only incoming ref
 * disappears -> StructRefs=0 -> Release fires with m_filled>0.
 *
 *   Init: head=N1, N1->N2->N3->NIL, filled N1=1,N2=0,N3=1.
 *   step1 SelfResetNext(N2): N2->NIL.  N3 now has NO incoming ref
 *                            (N2 was its only one); with SELF_REF=0,
 *                            StructRefs(N3)=0 though filled[N3]=1.
 *   step2 Release(N3):       StructRefs(N3)=0 -> released, bad_release set
 *                            (filled[N3]>0 = the DEBUG_GUARD violation).
 * NOTE: per-step Inv_NoBadRelease is NOT checked in this build (see
 * check_invariants_locked), so the violating Release can fire — exactly the
 * point of the negative control. */
static void replay_negative_control(void) {
    OP_LOCK();
    int s1 = act_self_reset_next(N2);
    assert(s1 == 1);                 /* guard enabled: N2 dead, nxt[N2] in Nodes */
    assert(struct_refs(N3) == 0);    /* live N3 stranded (no self-ref to save it) */
    int s2 = act_release(N3);
    assert(s2 == 1);                 /* Release(N3) fires: StructRefs(N3)==0   */
    OP_UNLOCK();
}
#endif

int main(void) {
    /* TLA+ Init: head=N1; nxt: N1->N2, N2->N3, N3->NIL;
     *            filled: N2=0 else 1; released=FALSE; gen=pushes=0;
     *            bad_release=FALSE. */
    ST(&g_head, N1);
    ST(&g_nxt[N1], N2);
    ST(&g_nxt[N2], N3);
    ST(&g_nxt[N3], NIL);
    ST(&g_filled[N1], 1);
    ST(&g_filled[N2], 0);
    ST(&g_filled[N3], 1);
    for (int n = 0; n < NUM_NODES; n++) ST(&g_released[n], 0);
    ST(&g_gen, 0);
    ST(&g_pushes, 0);
    ST(&g_bad_release, 0);
    ST(&g_stop, 0);

    /* Initial state must already satisfy the invariants (TLC checks Init). */
    OP_LOCK();
    check_invariants_locked();
    OP_UNLOCK();

#if EXPECT_BAD_RELEASE
    /* Negative control: replay the violating trace deterministically. */
    replay_negative_control();
    assert(LD(&g_bad_release));        /* the load-bearing self-ref demo fired */
    /* TypeOK bounds still hold. */
    assert(LD(&g_gen) >= 0 && LD(&g_gen) <= MAX_GEN);
    assert(LD(&g_pushes) >= 0 && LD(&g_pushes) <= MAX_PUSH);
    return 0;
#else
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

    /* TLA+ Inv_NoBadRelease: no chunk released with live slots. */
    assert(!LD(&g_bad_release));

    /* Liveness terminus (the *_live cfg's no-leak property, made concrete):
     * after quiescence every chunk that is empty is released, and every
     * released chunk has StructRefs==0 and nxt==NIL.  A chunk that is still
     * filled (revived and never re-freed) legitimately survives. */
    for (int n = 0; n < NUM_NODES; n++) {
        if (LD(&g_filled[n]) == 0) {
            /* empty and quiescent => reclaimed (no permanently-stranded
             * empty orphan): mirrors Liveness <>[](filled=0) => <>released. */
            assert(LD(&g_released[n]));
        }
        if (LD(&g_released[n])) {
            assert(struct_refs(n) == 0);    /* Inv_ReleasedNoIncoming */
            assert(LD(&g_nxt[n]) == NIL);   /* freed header pins nothing */
            assert(LD(&g_filled[n]) == 0);  /* Inv_LiveNeverReleased */
        }
    }
    /* head is NIL (whole chain reclaimed) or a live, unreleased node. */
    {
        int h = LD(&g_head);
        if (h != NIL) {
            assert(!LD(&g_released[h]));    /* Inv_HeadAlive */
            assert(LD(&g_filled[h]) > 0);   /* a non-empty head survives */
        }
    }

    /* Bounds (TLA+ TypeOK on the counters). */
    assert(LD(&g_gen) >= 0 && LD(&g_gen) <= MAX_GEN);
    assert(LD(&g_pushes) >= 0 && LD(&g_pushes) <= MAX_PUSH);

    return 0;
#endif  /* EXPECT_BAD_RELEASE */
}
