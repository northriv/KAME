(***************************************************************************
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
 ***************************************************************************)
--------------------- MODULE OrphanChain_atomicshared ---------------------
(*
 * Microscopic TLA+ model of the atomic_shared_ptr-refcounted intrusive
 * singly-linked ORPHAN CHAIN — the successor design to §36b.
 *
 * ============================================================
 *  WHERE THIS SITS IN THE SUITE
 * ============================================================
 * `OrphanReuse_36b.tla`   hand-rolled versioned slot array + m_orphan_disp
 *                         arbiter.  VIOLATION (Inv_NoBadRelease): the
 *                         disp/version pair cannot capture a full
 *                         pop->re-own->exit->re-push generation, so the
 *                         freer's claim-for-release CAS releases a live
 *                         chunk (ABA).
 * `OrphanReuse_optC.tla`  drop the array; restore release-on-empty.
 *                         CLEAN — but gives up in-place reuse of non-empty
 *                         orphans.
 * `OrphanChain_atomicshared.tla`  (this spec) recover reuse WITHOUT the
 *                         ABA, by letting the verified atomic_shared_ptr
 *                         REFCOUNT be the "captured generation" the
 *                         README's §36b verdict said was the only sound
 *                         completion.  The refcount + the local_shared_ptr
 *                         pin held across the dec/claim IS the generation
 *                         capture; try_promote (acquire-if-nonzero) is the
 *                         generation-matched claim.
 *
 * ============================================================
 *  DESIGN BEING MODELLED  (converged design, June 2026)
 * ============================================================
 * The shared orphan/reuse chain is an intrusive singly-linked list whose
 * links are atomic_shared_ptr (head) / atomic_shared_ptr next.  Layer 0
 * (atomic_shared_ptr.tla) already proves the refcount/CAS primitive sound;
 * THIS layer abstracts it and verifies the chain protocol + lifetime gate:
 *
 *   - SELF-REF ON m_filled.  A chunk holds one structural ref to ITSELF
 *     while it has live slots (m_filled > 0), dropped at the m_filled->0
 *     transition (the "nonzero_cnt zero test").  This is ONE ref-bump per
 *     non-empty/empty edge, NOT per slot — the hot path never refcounts.
 *     => a chunk's lifetime is DECOUPLED from its chain membership.
 *
 *   - RELEASE = refcount -> 0  (the deleter fires).  With the self-ref,
 *     refcount 0  <=>  no chain ref AND no pin AND m_filled == 0.
 *
 *   - MULTIPLE SWEEPERS, SAFE-SIDE ONLY.  Any thread may relink past a
 *     dead (m_filled==0) successor (CAS, reachability-preserving) and may
 *     idempotently null a dead node's own `next`.  Because lifetime is
 *     decoupled, every structural race outcome (lost update, tail-loss,
 *     a live node falling off the chain, a dead node surviving one extra
 *     round) costs only a REUSE HINT — never a leak or a use-after-free.
 *     No single-sweeper serialization is required.
 *
 *   - REVIVAL (reuse) vs RELEASE is the one genuinely non-safe-side edge.
 *     A dead orphan may be re-adopted (m_filled 0->1) ONLY via a
 *     try_promote that succeeds, i.e. only while it still has a live ref
 *     (StructRefs > 0).  Release fires only at StructRefs = 0.  The two
 *     predicates are mutually exclusive in every state, so a revived
 *     chunk is never released and a released chunk is never revived; the
 *     atomicity of the try_promote-vs-final-unref boundary is Layer 0's
 *     guarantee, cited not re-proved.
 *
 *   - HEAD INSERT (Push).  Owner-exit publishes a non-empty orphan by
 *     CASing it at the head (Treiber push: new->next = head; head = new).
 *     Push-vs-Push and the head-cell ABA are Layer 0; Push touches only
 *     {head, new->next} so it is disjoint from interior sweeps; and Push
 *     only ADDS refs / forward links (monotone), so it cannot of itself
 *     cause a bad release / dangling / cycle.  The one genuinely new
 *     interleaving is Push vs HeadAdvance (both CAS `head`) — modelled
 *     here so TLC confirms the protocol invariants survive that
 *     combination.  A fallen-off LIVE orphan being re-pushed also shows
 *     fall-off is recoverable.
 *
 * Empty-eager madvise (the RSS reclaim) does not touch the chain and is
 * not modelled here; "release" is the chunk-header/unit reclaim (refcnt 0).
 *
 * ============================================================
 *  STATE-SPACE DISCIPLINE
 * ============================================================
 * One microscopic chain of 3 chunks, owner already exited (orphans).
 *   head -> N1 -> N2 -> N3 -> NIL,   m_filled: N1=1, N2=0, N3=1.
 * N2 is a dead MIDDLE node with LIVE neighbours — the minimum topology
 * that exercises (a) middle relink, (b) the self-reset-before-relink
 * tail-loss that drops a LIVE successor (N3) off the chain, and (c)
 * adjacent dead-node removal once a neighbour drains.  m_filled is modelled
 * as {0,1} because only the zero boundary drives the protocol.  Revivals
 * and head-inserts are bounded by MaxGen / MaxPush for finiteness
 * (precondition counters, no StateConstraint — per the suite convention).
 *
 * Threads are NOT modelled explicitly: every modelled operation is a
 * SINGLE atomic memory access with no thread-local program counter, so the
 * interleaving of node-indexed atomic steps already covers N concurrent
 * sweepers/freers/adopters/pushers.  (Contrast §36b, whose 3-step re-own
 * needed a per-thread pc — exactly what the atomic try_promote collapses.)
 *
 * KNOBS:
 *   SelfRef.   TRUE  = the design (expect CLEAN).
 *              FALSE = drop the self-ref — expect Inv_NoBadRelease
 *                      VIOLATION (a live node falling off the chain is
 *                      released).  Shows the self-ref is load-bearing.
 *   MaxPush.   0 = no head insert.  >0 = enable Push (head-insert race).
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    N1, N2, N3,     \* the three modelled chunks (distinct model values)
    NIL,            \* end-of-list / reset sentinel (distinct from chunks)
    SelfRef,        \* TRUE = chunk self-refs while m_filled>0 (the design)
    AllowReown,     \* TRUE = a dead orphan may be re-adopted (reuse path)
    MaxGen,         \* bound on total revivals (finiteness)
    MaxPush         \* bound on total head-inserts (0 = Push disabled)

Nodes == { N1, N2, N3 }

ASSUME NIL \notin Nodes
ASSUME SelfRef \in BOOLEAN
ASSUME AllowReown \in BOOLEAN
ASSUME MaxGen \in Nat
ASSUME MaxPush \in Nat

VARIABLES
    head,           \* Nodes \cup {NIL} — chain head (atomic_shared_ptr)
    nxt,            \* [Nodes -> Nodes \cup {NIL}] — forward link per chunk
    filled,         \* [Nodes -> 0..1] — m_filled (0 = dead/empty)
    released,       \* [Nodes -> BOOLEAN] — chunk reclaimed (refcnt hit 0)
    gen,            \* Nat — revivals so far (bound: MaxGen)
    pushes,         \* Nat — head-inserts so far (bound: MaxPush)
    bad_release     \* BOOLEAN sticky — a release fired with m_filled>0
                    \*   (the C++ DEBUG_GUARD on the deleter)

vars == << head, nxt, filled, released, gen, pushes, bad_release >>

\* --- Reference accounting (atomic_shared_ptr abstracted) ------------------
\* Incoming structural refs to n: every chunk whose `next` points at n, plus
\* the head slot if it points at n.  A released chunk has nxt = NIL (its
\* freed header pins nothing), so it never contributes.
ChainIn(n) == Cardinality({ m \in Nodes : nxt[m] = n })
              + (IF head = n THEN 1 ELSE 0)

\* Total refcount = incoming chain refs + the self-ref (iff enabled & live).
\* (Pins held by an operating thread are folded into the action
\*  preconditions: any action touching n requires ~released[n], and Revive
\*  requires StructRefs(n) > 0 = a successful try_promote.)
StructRefs(n) == ChainIn(n) + (IF SelfRef /\ filled[n] > 0 THEN 1 ELSE 0)

TypeOK ==
    /\ head \in Nodes \cup {NIL}
    /\ nxt \in [Nodes -> Nodes \cup {NIL}]
    /\ filled \in [Nodes -> 0..1]
    /\ released \in [Nodes -> BOOLEAN]
    /\ gen \in 0..MaxGen
    /\ pushes \in 0..MaxPush
    /\ bad_release \in BOOLEAN

\* --- Initial state: head -> N1 -> N2(dead) -> N3 -> NIL -------------------
Init ==
    /\ head = N1
    /\ nxt = [x \in Nodes |-> IF x = N1 THEN N2
                              ELSE IF x = N2 THEN N3 ELSE NIL]
    /\ filled = [x \in Nodes |-> IF x = N2 THEN 0 ELSE 1]
    /\ released = [x \in Nodes |-> FALSE]
    /\ gen = 0
    /\ pushes = 0
    /\ bad_release = FALSE

\* ============================================================
\*  ACTIONS  (each = one atomic memory access)
\* ============================================================

\* Cross-thread free draining a chunk's last live slot: m_filled 1 -> 0.
\* Drops the self-ref (under SelfRef).  Live slot => chunk not released.
Free(n) ==
    /\ filled[n] = 1
    /\ ~released[n]
    /\ filled' = [filled EXCEPT ![n] = 0]
    /\ UNCHANGED << head, nxt, released, gen, pushes, bad_release >>

\* Reuse: re-adopt a dead orphan (m_filled 0 -> 1), re-adding the self-ref.
\* Gated by try_promote: enabled ONLY while the chunk still has a live ref
\* (StructRefs > 0).  A chunk already swept fully off the chain (StructRefs
\* = 0) is about-to-be / already released and CANNOT be revived — the
\* release-vs-revival mutual exclusion.
Revive(n) ==
    /\ AllowReown
    /\ gen < MaxGen
    /\ filled[n] = 0
    /\ ~released[n]
    /\ StructRefs(n) > 0
    /\ filled' = [filled EXCEPT ![n] = 1]
    /\ gen' = gen + 1
    /\ UNCHANGED << head, nxt, released, pushes, bad_release >>

\* Sweeper: relink past a dead successor.  c = nxt[p] is dead => p skips to
\* c's successor (reachability-preserving CAS).  Concurrent sweepers are the
\* interleaving of this action over different p; TLA+ step atomicity is the
\* CAS (a loser re-evaluates).  Drops c's incoming ref from p.
SweepRelink(p) ==
    /\ nxt[p] \in Nodes
    /\ filled[nxt[p]] = 0
    /\ ~released[p]
    /\ ~released[nxt[p]]
    /\ nxt' = [nxt EXCEPT ![p] = nxt[nxt[p]]]
    /\ UNCHANGED << head, filled, released, gen, pushes, bad_release >>

\* A dead chunk idempotently nulls its OWN next ("self-reset").  May run
\* while a predecessor still points at it — the tail-loss hazard: a live
\* successor reachable only through c then falls off the chain.  Safe-side
\* under SelfRef (the successor survives via its self-ref); a bad_release
\* under ~SelfRef.
SelfResetNext(c) ==
    /\ filled[c] = 0
    /\ ~released[c]
    /\ nxt[c] \in Nodes
    /\ nxt' = [nxt EXCEPT ![c] = NIL]
    /\ UNCHANGED << head, filled, released, gen, pushes, bad_release >>

\* Sweeper: advance head past a leading dead chunk (drops head's head-ref).
HeadAdvance ==
    /\ head \in Nodes
    /\ filled[head] = 0
    /\ ~released[head]
    /\ head' = nxt[head]
    /\ UNCHANGED << nxt, filled, released, gen, pushes, bad_release >>

\* Head insert (Treiber push): owner-exit re-publishes a LIVE, currently
\* off-chain orphan at the head.  new->next := head ; head := new.
\* Precondition ChainIn(n)=0 = n is not in the shared chain (it sits in the
\* owner's private DLL); overwriting nxt[n] drops any stale forward pin n
\* still held.  Races against HeadAdvance (both CAS `head`) and is bounded
\* by MaxPush.  Push-vs-Push / head ABA are Layer 0.
Push(n) ==
    /\ pushes < MaxPush
    /\ filled[n] = 1
    /\ ~released[n]
    /\ ChainIn(n) = 0
    /\ nxt' = [nxt EXCEPT ![n] = head]
    /\ head' = n
    /\ pushes' = pushes + 1
    /\ UNCHANGED << filled, released, gen, bad_release >>

\* The deleter: refcount has hit 0 (no incoming ref, no self-ref, unpinned).
\* Reclaims the chunk; its freed header's `next` ceases to pin anything.
\* bad_release records the C++ DEBUG_GUARD: releasing with m_filled > 0.
Release(n) ==
    /\ ~released[n]
    /\ StructRefs(n) = 0
    /\ released' = [released EXCEPT ![n] = TRUE]
    /\ nxt' = [nxt EXCEPT ![n] = NIL]
    /\ bad_release' = (bad_release \/ (filled[n] > 0))
    /\ UNCHANGED << head, filled, gen, pushes >>

Next ==
    \/ \E n \in Nodes :
         \/ Free(n)
         \/ Revive(n)
         \/ SweepRelink(n)
         \/ SelfResetNext(n)
         \/ Push(n)
         \/ Release(n)
    \/ HeadAdvance

Spec == Init /\ [][Next]_vars

\* Fairness for the liveness (no-leak) check: reclaim actions must progress.
\* Push is NOT made fair — head insert is optional progress, not required.
Fairness ==
    /\ WF_vars(HeadAdvance)
    /\ \A n \in Nodes : WF_vars(SweepRelink(n))
    /\ \A n \in Nodes : WF_vars(Release(n))

SpecLive == Init /\ [][Next]_vars /\ Fairness

\* ============================================================
\*  INVARIANTS  (safety)
\* ============================================================

\* (A) THE headline guard — the C++ deleter's DEBUG_GUARD.  A chunk is
\*     never released while it still has live slots.  CLEAN under SelfRef;
\*     VIOLATED if the self-ref is dropped (SelfRef=FALSE).
Inv_NoBadRelease == ~bad_release

\* (B) Lifetime decoupling, as a state invariant: a non-empty chunk is
\*     never in the released state.  (Equivalent to (A) holding throughout.)
Inv_LiveNeverReleased == \A n \in Nodes : filled[n] > 0 => ~released[n]

\* (C) No use-after-free via a stale link: a live chunk's `next` never
\*     points at a released chunk.
Inv_NoDanglingNext ==
    \A m \in Nodes : (~released[m] /\ nxt[m] \in Nodes) => ~released[nxt[m]]

\* (D) The chain head is never a released chunk.
Inv_HeadAlive == head \in Nodes => ~released[head]

\* (E) A released chunk has no incoming refs (its header pins nothing and
\*     nothing pins it) — refcount/structure stays coherent post-release.
Inv_ReleasedNoIncoming == \A n \in Nodes : released[n] => StructRefs(n) = 0

\* (F) The chain is acyclic (relinks/pushes only ever advance forward).
\*     Following `next` |Nodes| times from any chunk must reach NIL.
RECURSIVE Hops(_, _)
Hops(n, i) == IF i = 0 \/ n = NIL THEN n ELSE Hops(nxt[n], i - 1)
Inv_Acyclic == \A n \in Nodes : Hops(n, Cardinality(Nodes)) = NIL

\* ============================================================
\*  LIVENESS  (no leak — checked in the *_live cfg under Fairness)
\* ============================================================
\* Any chunk that stays empty forever is eventually released — empty-eager
\* + sweep compaction leaves no permanently-stranded empty orphan.
Liveness == \A n \in Nodes : (<>[](filled[n] = 0)) => <>released[n]

THEOREM Spec => []TypeOK
THEOREM Spec => []Inv_NoBadRelease
THEOREM Spec => []Inv_LiveNeverReleased
THEOREM Spec => []Inv_NoDanglingNext
THEOREM Spec => []Inv_HeadAlive
THEOREM Spec => []Inv_ReleasedNoIncoming
THEOREM Spec => []Inv_Acyclic
THEOREM SpecLive => Liveness

================================================================================
