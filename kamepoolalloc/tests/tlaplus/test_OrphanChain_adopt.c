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
 * C11 test generated mechanically from OrphanChain_adopt.tla.
 *
 * Models the ACTUAL implemented adopt mechanism (commit bb6d691d): the 2-step
 * orphan_chain_pop -> BIT_OWNED claim, the oc_hold parked-ref, the
 * atomic_intrusive_dispose BIT_OWNED gate, and residual scrub pins.  Refines
 * OrphanChain_pathB (this sibling port shares its harness style) but is FAITHFUL
 * to atomic_shared_ptr's SYNCHRONOUS disposal: disposal is folded INTO each
 * reference-dropping action, so refcnt==0 <=> released by construction.  There
 * is therefore NO standalone Release action here (unlike pathB) — each action
 * that removes a node's last reference disposes it inline, computing the
 * POST-state refcnt of the affected node under the hypothetical mutation.
 *
 * Raw-DLL design, modelled as index arrays:
 *   Refcnt(n) = ChainIn(n) + pins[+ owner-ref iff OwnerRef].
 *   ChainIn(n) = #{m : nxt[m]==n} + (head==n)   (head / predecessor m_orphan_next)
 *   pins       = (pop_ref==n) [oc_hold] + (scrub_pin==n) [scrubber load_shared].
 *   OWNED chunks are RAW (not refcounted) under the shipped design — no owner-ref
 *   term unless OwnerRef (the proposed fix) is enabled.
 *
 * TWO distinct free mechanisms (the crux the model exposes):
 *   (1) smart_ptr disposal — refcnt->0 routes to atomic_intrusive_dispose, GATED
 *       on ~owned (allocator_prv.h:1917).  Folded into ScrubUnpin / ScrubUnlink /
 *       HeadAdvance / AdoptDropRef.  Sets bad_release if it ever releases an
 *       owned/non-empty chunk.
 *   (2) the OWNER's DIRECT free — release_dll_chunks_for_thread empty branch
 *       (allocator.cpp:3026 newv==0): ~PoolAllocator + deallocate_chunk gated ONLY
 *       on MASK_CNT==0, NOT on the intrusive refcnt.  Sets bad_ownerfree if a
 *       residual smart_ptr (scrub pin / pinned predecessor's nxt) still references
 *       n at owner-free.  Modelled by OwnerExitEmpty.
 *
 * As in the sibling ports, TLA+ threads are NOT modelled explicitly: each Next
 * disjunct is ONE atomic step on a snapshot satisfying its guard.  We mirror that
 * with NUM_THREADS real worker threads, each repeatedly picking a node + an action
 * and executing exactly ONE TLA+ step.  TLA+ step atomicity is encoded by
 * serializing each whole action under a single global op-lock (the model has no
 * MODE_COARSE/FINE/SUPERFINE granularity to preserve, so there is nothing finer to
 * interleave: what matters is *which* atomic step fires next, exercised by the
 * random per-thread choice + lock).
 *
 * TLA+ variable mapping:
 *   head           -> _Atomic(int) g_head            (node id or NIL)
 *   nxt[n]         -> _Atomic(int) g_nxt[n]          (m_orphan_next; node id or NIL)
 *   filled[n]      -> _Atomic(int) g_filled[n]       (0 or 1) [MASK_CNT]
 *   owned[n]       -> _Atomic(int) g_owned[n]        (0 or 1) [BIT_OWNED]
 *   released[n]    -> _Atomic(int) g_released[n]     (0 or 1)
 *   pop_ref        -> _Atomic(int) g_pop_ref         (oc_hold; node id or NIL)
 *   scrub_pin      -> _Atomic(int) g_scrub_pin       (load_shared pin; node id/NIL)
 *   gen            -> _Atomic(int) g_gen             (adopts; <= MaxGen)
 *   bad_release    -> _Atomic(int) g_bad_release     (sticky; gate-1 violated)
 *   bad_ownerfree  -> _Atomic(int) g_bad_ownerfree   (sticky; owner-free violated)
 *
 * KNOBS (mirror the three .cfg files):
 *   GATE_ON_OWNED (default 1)  TRUE = the design's BIT_OWNED disposal gate.
 *                              FALSE (-DGATE_ON_OWNED=0, the nogate cfg) drops the
 *                              gate => AdoptDropRef frees an owned chunk =>
 *                              Inv_NoBadRelease VIOLATION (expected, demonstrated).
 *   OWNER_REF     (default 0)  FALSE = the shipped design (owner frees DIRECTLY on
 *                              MASK_CNT==0, ignoring refcnt) => Inv_NoBadOwnerFree
 *                              CAN be VIOLATED (the finding, demonstrated, not
 *                              asserted clean).  TRUE (-DOWNER_REF=1, the ownerref
 *                              cfg = the proposed fix) routes the owner free
 *                              through the refcnt; bad_ownerfree can never set =>
 *                              all invariants CLEAN (asserted).
 *   MAX_GEN       (default 2)  adopt bound (TLA+ MaxGen).
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

#ifndef GATE_ON_OWNED
#define GATE_ON_OWNED 1
#endif
#ifndef OWNER_REF
#define OWNER_REF 0
#endif
#ifndef MAX_GEN
#define MAX_GEN 2
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
 * The state space is microscopic; a large-but-bounded default reaches the
 * terminal "all reclaimable chunks released" attractor fast.  Unbounded for
 * stress. */
#ifndef MAX_COMMITS
#  if STRESS_SECONDS > 0
#    define MAX_COMMITS 0x7fffffff
#  else
#    define MAX_COMMITS 200000
#  endif
#endif

/* Under GateOnOwned=FALSE (nogate cfg) the spec EXPECTS Inv_NoBadRelease to be
 * violated (AdoptDropRef disposes an owned chunk).  In that build we demonstrate
 * the load-bearing failure rather than assert it clean. */
#ifndef EXPECT_BAD_RELEASE
#  if !GATE_ON_OWNED
#    define EXPECT_BAD_RELEASE 1
#  else
#    define EXPECT_BAD_RELEASE 0
#  endif
#endif

/* Under OwnerRef=FALSE (shipped mc cfg) the spec EXPECTS Inv_NoBadOwnerFree to be
 * VIOLATABLE — the documented finding: the owner's direct deallocate_chunk frees a
 * chunk a residual smart_ptr still references.  We demonstrate it (replay the
 * adversarial schedule if the random phase missed the window) rather than assert
 * it clean.  Under OwnerRef=TRUE (the fix) it can never set and we assert clean. */
#ifndef EXPECT_BAD_OWNERFREE
#  if !OWNER_REF
#    define EXPECT_BAD_OWNERFREE 1
#  else
#    define EXPECT_BAD_OWNERFREE 0
#  endif
#endif

/* ============================ Shared state =============================== */

static _Atomic(int) g_head;
static _Atomic(int) g_nxt[NUM_NODES];
static _Atomic(int) g_filled[NUM_NODES];
static _Atomic(int) g_owned[NUM_NODES];
static _Atomic(int) g_released[NUM_NODES];
static _Atomic(int) g_pop_ref;
static _Atomic(int) g_scrub_pin;
static _Atomic(int) g_gen;
static _Atomic(int) g_bad_release;
static _Atomic(int) g_bad_ownerfree;
static _Atomic(int) g_stop;

/* Single global op-lock: serializes each whole TLA+ action so it observes a
 * consistent snapshot and commits atomically — the encoding of "each action =
 * one atomic memory access" at TLA+ granularity. */
static pthread_mutex_t op_mtx = PTHREAD_MUTEX_INITIALIZER;
#define OP_LOCK()   pthread_mutex_lock(&op_mtx)
#define OP_UNLOCK() pthread_mutex_unlock(&op_mtx)

/* Relaxed reads/writes: all mutation happens under op_mtx, whose lock/unlock
 * provide the release/acquire fences.  Post-join reads are after all joins
 * (a happens-before edge), so plain loads suffice there too. */
static inline int LD(_Atomic(int) *p) {
    return atomic_load_explicit(p, memory_order_relaxed);
}
static inline void ST(_Atomic(int) *p, int v) {
    atomic_store_explicit(p, v, memory_order_relaxed);
}

/* ===================== Reference accounting (Layer 0) ==================== */
/* All of these must be called with op_mtx held (consistent snapshot). */

/* OwnerRefTerm(n) — +1 iff OwnerRef knob set AND ownd[n]. */
static inline int owner_ref_term(int ownd_n) {
#if OWNER_REF
    return ownd_n ? 1 : 0;
#else
    (void)ownd_n;
    return 0;
#endif
}

/* ChainIn(n) = #{m : nxt[m]==n} + (head==n).  Live snapshot. */
static int chain_in(int n) {
    int c = 0;
    for (int m = 0; m < NUM_NODES; m++)
        if (LD(&g_nxt[m]) == n) c++;
    if (LD(&g_head) == n) c++;
    return c;
}

/* Refcnt(n) = ChainIn(n) + (pop_ref==n) + (scrub_pin==n) + OwnerRefTerm(n).
 * The current-state refcount (TLA+ Refcnt). */
static int refcnt(int n) {
    int r = chain_in(n);
    if (LD(&g_pop_ref) == n)   r++;
    if (LD(&g_scrub_pin) == n) r++;
    r += owner_ref_term(LD(&g_owned[n]));
    return r;
}

/* RefcntAt(n, h, pr, sp, ownd_n): the POST-state refcount of n under a
 * hypothetical (head=h, pop_ref=pr, scrub_pin=sp, owned[n]=ownd_n) with the
 * nxt[] snapshot taken from the supplied predecessor count.  TLA+ RefcntAt
 * excludes the self-edge (m # n) in the chain term, matching the synchronous
 * disposer which has already cleared the dropping reference.
 *
 * pred_count = #{m : m != n /\ nx[m]==n} for the hypothetical nxt[] (caller
 * supplies it because the hypothetical nxt may differ from the live g_nxt). */
static int refcnt_at(int n, int pred_count, int h, int pr, int sp, int ownd_n) {
    int r = pred_count;
    if (h  == n) r++;
    if (pr == n) r++;
    if (sp == n) r++;
    r += owner_ref_term(ownd_n);
    return r;
}

/* Live predecessor count over g_nxt EXCLUDING the self-edge (m != n). */
static int pred_count_live(int n) {
    int c = 0;
    for (int m = 0; m < NUM_NODES; m++)
        if (m != n && LD(&g_nxt[m]) == n) c++;
    return c;
}

/* ============================== Actions ================================== */
/* Each returns 1 if it fired (guard enabled and committed), 0 otherwise.
 * Mirrors one TLA+ Next disjunct.  All run under OP_LOCK held by the caller. */

/* Free(n): filled=1 /\ ~released -> filled 1->0.  Refcnt UNCHANGED. */
static int act_free(int n) {
    if (LD(&g_filled[n]) != 1) return 0;
    if (LD(&g_released[n])) return 0;
    ST(&g_filled[n], 0);
    return 1;
}

/* Alloc(n): owned /\ filled=0 /\ ~released -> filled 0->1.  Refcnt UNCHANGED. */
static int act_alloc(int n) {
    if (!LD(&g_owned[n])) return 0;
    if (LD(&g_filled[n]) != 0) return 0;
    if (LD(&g_released[n])) return 0;
    ST(&g_filled[n], 1);
    return 1;
}

/* ScrubPin(n): scrub_pin=NIL /\ ~released /\ ChainIn(n)>0 -> scrub_pin := n. */
static int act_scrub_pin(int n) {
    if (LD(&g_scrub_pin) != NIL) return 0;
    if (LD(&g_released[n])) return 0;
    if (chain_in(n) <= 0) return 0;
    ST(&g_scrub_pin, n);
    return 1;
}

/* ScrubUnpin: scrub_pin /= NIL -> drop the pin; if that was n's last reference
 * AND (GateOnOwned => ~owned[n]) dispose synchronously (released:=TRUE;
 * nxt[n]:=NIL; bad_release |= filled[n]>0 \/ owned[n]).
 * RefcntAt(n, head, pop_ref, NIL, owned)  -- pin removed (sp=NIL). */
static int act_scrub_unpin(void) {
    int n = LD(&g_scrub_pin);
    if (n == NIL) return 0;
    int rc  = refcnt_at(n, pred_count_live(n), LD(&g_head), LD(&g_pop_ref),
                        NIL, LD(&g_owned[n]));
    int rel = (rc == 0);
#if GATE_ON_OWNED
    if (rel && LD(&g_owned[n])) rel = 0;     /* gate: BIT_OWNED => disposer returns */
#endif
    ST(&g_scrub_pin, NIL);
    if (rel) {
        ST(&g_released[n], 1);
        ST(&g_nxt[n], NIL);
        if (LD(&g_filled[n]) > 0 || LD(&g_owned[n]))
            ST(&g_bad_release, 1);           /* sticky DEBUG_GUARD */
    }
    return 1;
}

/* ScrubUnlink(p): nxt[p] in Nodes /\ filled[nxt[p]]=0 /\ ~owned[nxt[p]] /\
 *   ~released[p] /\ ~released[nxt[p]]  ->  relink nxt[p] := nxt[c] (c=nxt[p]);
 *   if c then has Refcnt==0 dispose c synchronously (nxt[c]:=NIL; released:=TRUE).
 * c is empty & unowned by precondition, so the gate is moot here. */
static int act_scrub_unlink(int p) {
    int c = LD(&g_nxt[p]);
    if (c == NIL) return 0;                  /* nxt[p] not a Node */
    if (LD(&g_filled[c]) != 0) return 0;
    if (LD(&g_owned[c])) return 0;
    if (LD(&g_released[p])) return 0;
    if (LD(&g_released[c])) return 0;

    int nxt_c = LD(&g_nxt[c]);               /* successor preserved by relink */
    /* Apply the relink: nxt[p] := nxt[c]. */
    ST(&g_nxt[p], nxt_c);

    /* Hypothetical post-relink predecessor count of c, excluding self-edge.
     * After relink, nxt[p] no longer points at c; any OTHER m != c with
     * nxt[m]==c still counts.  g_nxt now already reflects the relink, so a live
     * pred scan excluding the self-edge gives the post-state count. */
    int pred_c = pred_count_live(c);
    int rc  = refcnt_at(c, pred_c, LD(&g_head), LD(&g_pop_ref),
                        LD(&g_scrub_pin), LD(&g_owned[c]));
    if (rc == 0) {
        ST(&g_nxt[c], NIL);
        ST(&g_released[c], 1);
    }
    return 1;
}

/* HeadAdvance: head in Nodes /\ filled[head]=0 /\ ~owned[head] /\ ~released[head]
 *   ->  head := nxt[head]; if old head then has Refcnt==0 dispose synchronously
 *   (nxt[h]:=NIL; released:=TRUE).
 * RefcntAt(h, s, nxt, pop_ref, scrub_pin, owned) with head moved to s. */
static int act_head_advance(void) {
    int h = LD(&g_head);
    if (h == NIL) return 0;
    if (LD(&g_filled[h]) != 0) return 0;
    if (LD(&g_owned[h])) return 0;
    if (LD(&g_released[h])) return 0;

    int s = LD(&g_nxt[h]);
    ST(&g_head, s);

    /* Post-state refcnt of h: head now = s, nxt unchanged (still g_nxt). */
    int pred_h = pred_count_live(h);
    int rc  = refcnt_at(h, pred_h, s, LD(&g_pop_ref), LD(&g_scrub_pin),
                        LD(&g_owned[h]));
    if (rc == 0) {
        ST(&g_nxt[h], NIL);
        ST(&g_released[h], 1);
    }
    return 1;
}

/* AdoptPop(n): gen<MaxGen /\ head=n /\ ~released /\ ~owned /\ pop_ref=NIL
 *   ->  head:=nxt[n]; nxt[n]:=NIL; pop_ref:=n; gen++.
 * Net refcnt change is zero (loses head-ref, gains pop_ref) — no disposal. */
static int act_adopt_pop(int n) {
    if (LD(&g_gen) >= MAX_GEN) return 0;
    if (LD(&g_head) != n) return 0;
    if (LD(&g_released[n])) return 0;
    if (LD(&g_owned[n])) return 0;
    if (LD(&g_pop_ref) != NIL) return 0;
    ST(&g_head, LD(&g_nxt[n]));
    ST(&g_nxt[n], NIL);
    ST(&g_pop_ref, n);
    ST(&g_gen, LD(&g_gen) + 1);
    return 1;
}

/* AdoptClaim(n): pop_ref=n /\ ~owned /\ ~released -> owned:=TRUE.  oc_hold still
 * held (pop_ref) so refcnt >= 1; no disposal. */
static int act_adopt_claim(int n) {
    if (LD(&g_pop_ref) != n) return 0;
    if (LD(&g_owned[n])) return 0;
    if (LD(&g_released[n])) return 0;
    ST(&g_owned[n], 1);
    return 1;
}

/* AdoptDropRef(n): pop_ref=n /\ owned -> drop oc_hold (pop_ref:=NIL).  THIS IS
 * THE smart_ptr DISPOSAL/GATE SITE.  If n then has Refcnt==0 AND
 * (GateOnOwned => ~owned[n]) dispose synchronously (released:=TRUE; nxt:=NIL;
 * bad_release |= filled>0 \/ owned).
 * RefcntAt(n, head, nxt, NIL, scrub_pin, owned) -- oc_hold removed (pr=NIL). */
static int act_adopt_drop_ref(int n) {
    if (LD(&g_pop_ref) != n) return 0;
    if (!LD(&g_owned[n])) return 0;
    int rc  = refcnt_at(n, pred_count_live(n), LD(&g_head), NIL,
                        LD(&g_scrub_pin), LD(&g_owned[n]));
    int rel = (rc == 0);
#if GATE_ON_OWNED
    if (rel && LD(&g_owned[n])) rel = 0;     /* gate: BIT_OWNED => disposer returns */
#endif
    ST(&g_pop_ref, NIL);
    if (rel) {
        ST(&g_released[n], 1);
        ST(&g_nxt[n], NIL);
        if (LD(&g_filled[n]) > 0 || LD(&g_owned[n]))
            ST(&g_bad_release, 1);           /* sticky DEBUG_GUARD */
    }
    return 1;
}

/* OwnerExitEmpty(n): owned /\ filled=0 /\ ~released /\ pop_ref != n
 *   ->  owned:=FALSE (drop owner-ref);
 *   OwnerRef=TRUE (fix): release ONLY if refcnt then hits 0 (no residual pins);
 *     nxt[n]:=NIL on release.  No direct free => bad_ownerfree never set.
 *   OwnerRef=FALSE (shipped): owner frees DIRECTLY (released:=TRUE always),
 *     ignoring refcnt; bad_ownerfree |= (Refcnt(n) > 0) [pre-drop refcnt with the
 *     residual smart_ptr still counted].
 * RefcntAt(n, head, nxt, pop_ref, scrub_pin, ownd2) with owner-ref dropped. */
static int act_owner_exit_empty(int n) {
    if (!LD(&g_owned[n])) return 0;
    if (LD(&g_filled[n]) != 0) return 0;
    if (LD(&g_released[n])) return 0;
    if (LD(&g_pop_ref) == n) return 0;

#if OWNER_REF
    /* TLA+: rc = RefcntAt(n, head, nxt, pop_ref, scrub_pin, ownd2);  rel = (rc=0).
     * Compute BEFORE flipping owned so the ownd2 term uses owned[n]=FALSE. */
    int rc  = refcnt_at(n, pred_count_live(n), LD(&g_head), LD(&g_pop_ref),
                        LD(&g_scrub_pin), 0 /* owned' = FALSE */);
    int rel = (rc == 0);
    ST(&g_owned[n], 0);
    if (rel) {
        ST(&g_released[n], 1);
        ST(&g_nxt[n], NIL);
    }
    /* bad_ownerfree' = bad_ownerfree (unchanged) under OwnerRef. */
#else
    /* Shipped: direct deallocate_chunk on MASK_CNT==0, ignoring refcnt.
     * bad_ownerfree' = bad_ownerfree \/ (Refcnt(n) > 0) -- Refcnt is the
     * current-state refcnt WITH the owner-ref term (here 0 since OwnerRef off),
     * i.e. residual chain-ref / pin / pop-ref.  pop_ref!=n by guard. */
    if (refcnt(n) > 0)
        ST(&g_bad_ownerfree, 1);             /* sticky: freed a still-referenced chunk */
    ST(&g_owned[n], 0);
    ST(&g_released[n], 1);
#endif
    return 1;
}

/* OwnerExitNonEmpty(n): owned /\ filled=1 /\ ~released /\ pop_ref != n /\
 *   ChainIn(n)=0  ->  owned:=FALSE; nxt[n]:=head; head:=n.  Re-push to chain
 *   (orphan_chain_push); chunk stays alive (owner-ref -> chain-ref). */
static int act_owner_exit_nonempty(int n) {
    if (!LD(&g_owned[n])) return 0;
    if (LD(&g_filled[n]) != 1) return 0;
    if (LD(&g_released[n])) return 0;
    if (LD(&g_pop_ref) == n) return 0;
    if (chain_in(n) != 0) return 0;
    ST(&g_owned[n], 0);
    ST(&g_nxt[n], LD(&g_head));
    ST(&g_head, n);
    return 1;
}

/* ====================== Per-step invariant checks ======================== */
/* The TLA+ safety invariants, checked while op_mtx is held so the state is a
 * consistent post-step snapshot (mirrors TLC checking invariants on each
 * reachable state).
 *
 * Under the shipped design Inv_NoBadOwnerFree is a DOCUMENTED VIOLATION, so it
 * is NOT asserted when EXPECT_BAD_OWNERFREE; likewise Inv_NoBadRelease is the
 * expected violation under the nogate build (EXPECT_BAD_RELEASE).  The structural
 * invariants are checked according to which .cfg's INVARIANTS list applies:
 *   - Inv_OwnedNotChained is OMITTED in every cfg (the ownerref cfg comment notes
 *     a pin'd predecessor can transiently point at an owned node in BOTH designs),
 *     so we do NOT check it.
 *   - Inv_NoDanglingNext / Inv_ReleasedNoRefs / Inv_Acyclic are listed only in the
 *     ownerref cfg; they also hold structurally in the shipped clean trace, but to
 *     mirror the cfgs exactly we assert them unconditionally EXCEPT where an
 *     intentional violation (bad_ownerfree / bad_release) has corrupted the state. */
static void check_invariants_locked(void) {
#if !EXPECT_BAD_RELEASE
    /* Inv_NoBadRelease */
    assert(!LD(&g_bad_release));
#endif
#if !EXPECT_BAD_OWNERFREE
    /* Inv_NoBadOwnerFree */
    assert(!LD(&g_bad_ownerfree));
#endif

    /* The structural invariants below are guaranteed only when neither
     * intentional violation has fired.  bad_ownerfree means the owner directly
     * freed a still-referenced chunk (released[n]=TRUE while refcnt>0), which by
     * construction breaks Inv_ReleasedNoRefs / Inv_NoDanglingNext — that IS the
     * finding, so skip them once it has fired.  bad_release similarly corrupts
     * the chain. */
    if (LD(&g_bad_ownerfree) || LD(&g_bad_release))
        return;

    /* Inv_NoDanglingNext: ~released[m] /\ nxt[m] in Nodes => ~released[nxt[m]]. */
    for (int m = 0; m < NUM_NODES; m++) {
        int x = LD(&g_nxt[m]);
        if (!LD(&g_released[m]) && x != NIL)
            assert(!LD(&g_released[x]));
    }
    /* Inv_ReleasedNoRefs: released[n] => Refcnt(n)=0. */
    for (int n = 0; n < NUM_NODES; n++)
        if (LD(&g_released[n])) assert(refcnt(n) == 0);
    /* Inv_Acyclic: following nxt |Nodes| times from any node reaches NIL. */
    for (int n = 0; n < NUM_NODES; n++) {
        int cur = n;
        for (int i = 0; i < NUM_NODES && cur != NIL; i++)
            cur = LD(&g_nxt[cur]);
        assert(cur == NIL);
    }
    /* head, if a Node, is not released (UAF surrogate on the chain head). */
    {
        int h = LD(&g_head);
        if (h != NIL) assert(!LD(&g_released[h]));
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
    int which = (int)(xs32(seed) % 9);
    int fired = 0;

    OP_LOCK();
    switch (which) {
        case 0: fired = act_free(n);                break;
        case 1: fired = act_alloc(n);               break;
        case 2: fired = act_scrub_pin(n);           break;
        case 3: fired = act_scrub_unlink(n);        break;
        case 4: fired = act_adopt_pop(n);           break;
        case 5: fired = act_adopt_claim(n);         break;
        case 6: fired = act_adopt_drop_ref(n);      break;
        case 7: fired = act_owner_exit_empty(n);    break;
        case 8: fired = act_owner_exit_nonempty(n); break;
        default: break;
    }
    /* The two non-node-parameterised disjuncts: ScrubUnpin, HeadAdvance.
     * Fire them opportunistically too so every Next disjunct is reachable. */
    if (!fired) {
        if (xs32(seed) & 1) fired = act_scrub_unpin();
        else                fired = act_head_advance();
    }
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

/* ============================ Terminal driver ============================ */
/* Drive the model toward quiescence after the concurrent phase: repeatedly fire
 * the reclaim / drain actions until no step is enabled.  Frees live nodes,
 * completes any in-flight adopt (claim then drop-ref), unpins, exits owners,
 * advances head and relinks past dead nodes.  This mirrors the SpecLive fairness
 * the model would check (no permanently-stranded reclaimable chunk).
 * Single-threaded, after all workers joined. */
static void drive_to_quiescence(void) {
    int progress = 1;
    int guard = 0;
    const int GUARD_MAX = 200000;            /* livelock tripwire */
    while (progress && guard++ < GUARD_MAX) {
        progress = 0;
        OP_LOCK();
        /* Complete any parked adopt so pop_ref drains (claim, then drop-ref). */
        for (int n = 0; n < NUM_NODES; n++) progress |= act_adopt_claim(n);
        for (int n = 0; n < NUM_NODES; n++) progress |= act_adopt_drop_ref(n);
        /* Drop any scrub pin (and dispose if it was the last ref). */
        progress |= act_scrub_unpin();
        /* Free every live node so the no-leak terminus is reachable. */
        for (int n = 0; n < NUM_NODES; n++) progress |= act_free(n);
        /* Owner-exit now-empty owned nodes (frees / drops owner-ref). */
        for (int n = 0; n < NUM_NODES; n++) progress |= act_owner_exit_empty(n);
        /* Reclaim the chain: relink past / advance past dead nodes. */
        for (int p = 0; p < NUM_NODES; p++) progress |= act_scrub_unlink(p);
        progress |= act_head_advance();
        check_invariants_locked();
        OP_UNLOCK();
    }
    assert(guard < GUARD_MAX);               /* did not livelock reaching quiescence */
}

/* ===================== Negative-control replays ========================== */
/* Reset to the TLA+ Init state (head->N1->N2(empty)->N3->NIL, all orphans). */
static void reinit_locked(void) {
    ST(&g_head, N1);
    ST(&g_nxt[N1], N2);
    ST(&g_nxt[N2], N3);
    ST(&g_nxt[N3], NIL);
    ST(&g_filled[N1], 1);
    ST(&g_filled[N2], 0);
    ST(&g_filled[N3], 1);
    for (int n = 0; n < NUM_NODES; n++) {
        ST(&g_owned[n], 0);
        ST(&g_released[n], 0);
    }
    ST(&g_pop_ref, NIL);
    ST(&g_scrub_pin, NIL);
    ST(&g_gen, 0);
    /* sticky flags intentionally NOT cleared. */
}

#if EXPECT_BAD_OWNERFREE
/* OwnerRef=FALSE (shipped) negative control: the owner's direct deallocate_chunk
 * frees a chunk a residual smart_ptr still references.  Deterministic schedule
 * (the TLC counterexample for the mc cfg):
 *
 *   Init                  head=N1 -> N2(empty) -> N3 -> NIL.
 *   AdoptPop(N1)          head:=N2; nxt[N1]:=NIL; pop_ref:=N1; gen=1.
 *   AdoptClaim(N1)        owned[N1]:=TRUE.   (N1 re-owned, held by oc_hold)
 *   AdoptDropRef(N1)      pop_ref:=NIL; gate keeps N1 alive (owned), no dispose.
 *                         Now N1 is owned, off the chain, refcnt=0 (raw, shipped).
 *   ScrubPin(N2)          N2 still on chain (head=N2): scrub_pin:=N2, refcnt(N2)+=1.
 *   AdoptPop(N2)          gen<MaxGen: head:=N3; nxt[N2]:=NIL; pop_ref:=N2; gen=2.
 *                         Now N2: chain-ref gone, scrub_pin pins it (refcnt=2 with
 *                         pop_ref) — a RESIDUAL smart_ptr (the scrub pin) holds N2.
 *   Free(N1)/already empty; Alloc skipped.  Make N1 empty (it is, filled stayed 1
 *                         actually — N1 was filled at Init).  Free(N1): filled:=0.
 *   OwnerExitEmpty(N1)    owned[N1] & filled[N1]=0 & pop_ref(=N2)!=N1: shipped
 *                         owner frees N1 directly.  But we want the violation on a
 *                         node a pin references — drive it on the pinned node:
 *
 * Simplest faithful violation (matches the model's bad_ownerfree definition —
 * OwnerExitEmpty(n) with Refcnt(n)>0): re-own a node, leave a residual scrub pin
 * on it, then owner-exit-empty it.
 *
 *   Init
 *   ScrubPin(N1)          head=N1 on chain: scrub_pin:=N1.   (residual pin)
 *   AdoptPop(N1)          head:=N2; nxt[N1]:=NIL; pop_ref:=N1; gen=1.  (refcnt N1:
 *                         scrub_pin still pins it => ChainIn=0,pop_ref=1,pin=1)
 *   AdoptClaim(N1)        owned[N1]:=TRUE.
 *   AdoptDropRef(N1)      pop_ref:=NIL; gate (owned) keeps N1 alive.  scrub_pin
 *                         STILL pins N1 (refcnt(N1)=1 via the pin).
 *   Free(N1)              filled[N1]:=0.
 *   OwnerExitEmpty(N1)    owned & empty & pop_ref!=N1: shipped frees N1 DIRECTLY
 *                         though scrub_pin(=N1) still references it
 *                         => Refcnt(N1)=1>0 => bad_ownerfree := TRUE.  VIOLATION. */
static void replay_bad_ownerfree_trace(void) {
    OP_LOCK();
    reinit_locked();
    assert(act_scrub_pin(N1));         /* residual pin on the head node */
    assert(act_adopt_pop(N1));         /* pop N1: pop_ref=N1, head=N2 */
    assert(act_adopt_claim(N1));       /* claim BIT_OWNED */
    assert(act_adopt_drop_ref(N1));    /* drop oc_hold; gate keeps owned N1 alive */
    assert(act_free(N1));              /* N1 becomes empty (MASK_CNT 0) */
    /* scrub_pin == N1 still holds, so refcnt(N1) > 0 at owner-free. */
    assert(refcnt(N1) > 0);
    assert(act_owner_exit_empty(N1));  /* shipped owner frees directly => violation */
    assert(LD(&g_bad_ownerfree));
    OP_UNLOCK();
}
#endif

#if EXPECT_BAD_RELEASE
/* GateOnOwned=FALSE (nogate) negative control: AdoptDropRef disposes an owned
 * chunk because the BIT_OWNED gate is removed.
 *
 *   Init
 *   AdoptPop(N1)          head:=N2; nxt[N1]:=NIL; pop_ref:=N1; gen=1.
 *   AdoptClaim(N1)        owned[N1]:=TRUE.   (refcnt(N1)=1 via pop_ref only)
 *   AdoptDropRef(N1)      pop_ref:=NIL; RefcntAt(N1)=0; NO gate => dispose an
 *                         OWNED chunk => bad_release := TRUE (owned[n] term).
 *                         VIOLATION. */
static void replay_bad_release_trace(void) {
    OP_LOCK();
    reinit_locked();
    assert(act_adopt_pop(N1));
    assert(act_adopt_claim(N1));
    assert(act_adopt_drop_ref(N1));    /* nogate: disposes owned N1 => bad_release */
    assert(LD(&g_bad_release));
    OP_UNLOCK();
}
#endif

int main(void) {
    /* TLA+ Init: head=N1; N1->N2->N3->NIL; filled N2=0 else 1; owned/released all
     * FALSE; pop_ref=scrub_pin=NIL; gen=0; bad_* = FALSE. */
    OP_LOCK();
    reinit_locked();
    ST(&g_bad_release, 0);
    ST(&g_bad_ownerfree, 0);
    ST(&g_stop, 0);
    /* Initial state must already satisfy the invariants (TLC checks Init). */
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

    /* Concurrent phase done: all enabled safety invariants held at every step.
     * Drive to the no-leak terminus. */
    drive_to_quiescence();

    /* ---------------------- Post-join terminal invariant ---------------- */
    OP_LOCK();
    check_invariants_locked();
    OP_UNLOCK();

#if EXPECT_BAD_RELEASE
    /* nogate cfg: the spec EXPECTS Inv_NoBadRelease violated.  Demonstrate it. */
    if (!LD(&g_bad_release)) replay_bad_release_trace();
    assert(LD(&g_bad_release));
#endif

#if EXPECT_BAD_OWNERFREE
    /* shipped cfg: the spec EXPECTS Inv_NoBadOwnerFree violable (the finding).
     * Demonstrate it deterministically if the random phase missed the window. */
    if (!LD(&g_bad_ownerfree)) replay_bad_ownerfree_trace();
    assert(LD(&g_bad_ownerfree));
#endif

#if !EXPECT_BAD_RELEASE
    /* The BIT_OWNED disposal gate is on: smart_ptr disposal never released an
     * owned/non-empty chunk (TLA+ Inv_NoBadRelease). */
    assert(!LD(&g_bad_release));
#endif
#if !EXPECT_BAD_OWNERFREE
    /* OwnerRef fix: the owner free never freed a still-referenced chunk
     * (TLA+ Inv_NoBadOwnerFree).  Under the fix the full structural invariant
     * set is clean — verify the no-leak terminus too. */
    assert(!LD(&g_bad_ownerfree));
    for (int n = 0; n < NUM_NODES; n++) {
        if (LD(&g_released[n])) {
            assert(refcnt(n) == 0);             /* Inv_ReleasedNoRefs */
            assert(LD(&g_nxt[n]) == NIL);       /* freed header pins nothing */
        }
    }
    {
        int h = LD(&g_head);
        if (h != NIL) assert(!LD(&g_released[h]));
    }
#endif

    /* Bounds (TLA+ TypeOK on the counters). */
    assert(LD(&g_gen) >= 0 && LD(&g_gen) <= MAX_GEN);

    return 0;
}
