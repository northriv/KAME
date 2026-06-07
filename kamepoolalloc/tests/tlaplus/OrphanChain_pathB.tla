(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        Dual-licensed Apache 2.0 OR GPL-2.0-or-later — see OrphanChain_atomicshared.tla.
 ***************************************************************************)
------------------------- MODULE OrphanChain_pathB -------------------------
(*
 * Path B model of the orphan-chain chunk-release design — the SEPARATE-COUNTS
 * variant decided after OrphanChain_atomicshared.tla (which modelled the
 * self-ref + self-reset variant).
 *
 * Difference from OrphanChain_atomicshared:
 *   - There is NO count-based self-ref on the intrusive refcnt, and NO
 *     self-reset.  MASK_CNT (filled) is a SEPARATE counter; the intrusive
 *     refcnt is managed purely by structural references:
 *
 *         refcnt(n) = owner-ref(owned[n]) + chain-ref(ChainIn(n)) [+ pins]
 *
 *     owner-ref  = a local_shared_ptr the owner's per-thread DLL holds.
 *     chain-ref  = head / a predecessor's m_orphan_next (atomic_shared_ptr).
 *     (pins = transient load_shared handles — abstracted: every action that
 *      touches n requires ~released[n], i.e. it reached n via a live ref;
 *      atomic_shared_ptr's local-tag SMR is Layer 0, not re-modelled.)
 *
 *   - "Don't release a non-empty chunk" is STRUCTURAL, not a self-ref:
 *     the sweeper removes ONLY dead (filled=0) nodes and relink preserves
 *     successors, so a non-empty node never loses its incoming chain-ref;
 *     an owned chunk has an owner-ref.  Hence refcnt->0  =>  filled==0,
 *     and the disposer's MASK_CNT==0 assert is a safety net, not load-bearing
 *     by itself.
 *
 *   - refcnt moves ONLY on owner-ref / chain-ref transitions, NEVER on
 *     alloc/free (Alloc/Free leave refcnt untouched — the whole point of
 *     separate counts: no manual ops on the intrusive refcnt, so
 *     atomic_smart_ptr's split (local-tag) counting is never bypassed).
 *
 * KNOB AllowLiveRemoval: the Path-B load-bearing rule is "remove only DEAD
 *   nodes".  FALSE = the design (expect CLEAN).  TRUE = let the sweeper /
 *   head-advance also drop a non-empty node — expect Inv_NoBadRelease
 *   VIOLATION (a non-empty orphan loses its last ref and is released).
 *   This is the Path-B analogue of OrphanChain_atomicshared's SelfRef knob.
 *
 * Microscopic: 3 chunks, filled in {0,1}.  Full lifecycle: owned chunks
 * alloc/free, owner-exit (empty=>release, non-empty=>push to chain), orphans
 * drain + are swept (dead-only) + adopted (single head pop).  Threads not
 * modelled (each op = one atomic step; interleaving covers N sweepers/
 * adopters/freers).  Revivals/pushes bounded by MaxGen/MaxPush (precondition
 * counters; no StateConstraint).
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    N1, N2, N3,     \* the three modelled chunks
    NIL,            \* end-of-list / reset sentinel
    AllowReown,     \* TRUE = a head orphan may be adopted (reuse)
    AllowLiveRemoval, \* FALSE = remove only DEAD nodes (the design)
    MaxGen,         \* bound on adopts
    MaxPush         \* bound on owner-exit pushes

Nodes == { N1, N2, N3 }

ASSUME NIL \notin Nodes
ASSUME AllowReown \in BOOLEAN
ASSUME AllowLiveRemoval \in BOOLEAN
ASSUME MaxGen \in Nat /\ MaxPush \in Nat

VARIABLES
    head,           \* Nodes \cup {NIL}
    nxt,            \* [Nodes -> Nodes \cup {NIL}]
    filled,         \* [Nodes -> 0..1]  (MASK_CNT — SEPARATE counter)
    owned,          \* [Nodes -> BOOLEAN]  (owner-ref present)
    released,       \* [Nodes -> BOOLEAN]
    gen, pushes,    \* Nat — adopt / push counters
    bad_release     \* BOOLEAN sticky — a release fired with filled>0

vars == << head, nxt, filled, owned, released, gen, pushes, bad_release >>

\* Structural refcount — owner-ref + chain-ref.  NO self-ref, NO pins term
\* (pins are folded into the ~released preconditions).
ChainIn(n) == Cardinality({ m \in Nodes : nxt[m] = n })
              + (IF head = n THEN 1 ELSE 0)
Refcnt(n)  == (IF owned[n] THEN 1 ELSE 0) + ChainIn(n)

TypeOK ==
    /\ head \in Nodes \cup {NIL}
    /\ nxt \in [Nodes -> Nodes \cup {NIL}]
    /\ filled \in [Nodes -> 0..1]
    /\ owned \in [Nodes -> BOOLEAN]
    /\ released \in [Nodes -> BOOLEAN]
    /\ gen \in 0..MaxGen /\ pushes \in 0..MaxPush
    /\ bad_release \in BOOLEAN

\* Init: all three OWNED (in some thread's DLL), chain empty.  N1/N2 non-empty,
\* N3 empty-but-owned (reusable).  Owner-exits will populate the chain.
Init ==
    /\ head = NIL
    /\ nxt = [x \in Nodes |-> NIL]
    /\ filled = [x \in Nodes |-> IF x = N3 THEN 0 ELSE 1]
    /\ owned = [x \in Nodes |-> TRUE]
    /\ released = [x \in Nodes |-> FALSE]
    /\ gen = 0 /\ pushes = 0
    /\ bad_release = FALSE

\* ===========================================================
\*  ACTIONS  (each = one atomic step; refcnt moves only on owner/chain edges)
\* ===========================================================

\* Owner allocates a slot into its (empty) chunk: MASK_CNT 0->1.  refcnt UNCHANGED.
Alloc(n) ==
    /\ owned[n] /\ ~released[n] /\ filled[n] = 0
    /\ filled' = [filled EXCEPT ![n] = 1]
    /\ UNCHANGED << head, nxt, owned, released, gen, pushes, bad_release >>

\* A slot is freed (owner or cross-thread): MASK_CNT 1->0.  refcnt UNCHANGED
\* (separate counts — free does NOT touch the intrusive refcnt).
Free(n) ==
    /\ filled[n] = 1 /\ ~released[n]
    /\ filled' = [filled EXCEPT ![n] = 0]
    /\ UNCHANGED << head, nxt, owned, released, gen, pushes, bad_release >>

\* Owner-exit, chunk empty: drop the owner-ref.  If that was the last ref,
\* Release fires (separately).  No chain involvement.
OwnerExitEmpty(n) ==
    /\ owned[n] /\ filled[n] = 0 /\ ~released[n]
    /\ owned' = [owned EXCEPT ![n] = FALSE]
    /\ UNCHANGED << head, nxt, filled, released, gen, pushes, bad_release >>

\* Owner-exit, chunk non-empty: transfer owner-ref -> chain-ref by pushing at
\* the head (Treiber).  owned:=false AND head/nxt updated atomically (the real
\* code: drop the DLL local_shared_ptr after the head CAS adopts the ref).
OwnerExitNonEmpty(n) ==
    /\ owned[n] /\ filled[n] = 1 /\ ~released[n]
    /\ ChainIn(n) = 0           \* not already on the chain
    /\ pushes < MaxPush
    /\ owned' = [owned EXCEPT ![n] = FALSE]
    /\ nxt' = [nxt EXCEPT ![n] = head]
    /\ head' = n
    /\ pushes' = pushes + 1
    /\ UNCHANGED << filled, released, gen, bad_release >>

\* Sweeper: relink p past its successor c.  Path B removes ONLY dead (filled=0)
\* c; relink preserves c's successor (pred now points to it), so a live node is
\* never dropped.  AllowLiveRemoval=TRUE lifts the dead-only guard (the unsafe
\* variant) to demonstrate the rule is load-bearing.
SweepRelink(p) ==
    /\ nxt[p] \in Nodes
    /\ ~released[p] /\ ~released[nxt[p]]
    /\ (filled[nxt[p]] = 0 \/ AllowLiveRemoval)
    /\ nxt' = [nxt EXCEPT ![p] = nxt[nxt[p]]]
    /\ UNCHANGED << head, filled, owned, released, gen, pushes, bad_release >>

\* Sweeper: advance head past a leading DEAD node (drops its chain-ref).
HeadAdvance ==
    /\ head \in Nodes /\ ~released[head]
    /\ (filled[head] = 0 \/ AllowLiveRemoval)
    /\ head' = nxt[head]
    /\ UNCHANGED << nxt, filled, owned, released, gen, pushes, bad_release >>

\* Adopt (reuse): pop the SINGLE head node (try_promote = ~released gate),
\* re-own it (chain-ref -> owner-ref).  Empty or non-empty.  NOT the whole list.
Adopt(n) ==
    /\ AllowReown /\ gen < MaxGen
    /\ head = n /\ n \in Nodes
    /\ ~released[n] /\ ~owned[n]
    /\ owned' = [owned EXCEPT ![n] = TRUE]
    /\ head' = nxt[n]
    /\ gen' = gen + 1
    /\ UNCHANGED << nxt, filled, released, pushes, bad_release >>

\* The deleter: refcnt hit 0 (not owned, off chain, unpinned).  Reclaim; the
\* freed header's next stops pinning its successor.  bad_release = the C++
\* DEBUG_GUARD: released while filled>0.
Release(n) ==
    /\ ~released[n] /\ Refcnt(n) = 0
    /\ released' = [released EXCEPT ![n] = TRUE]
    /\ nxt' = [nxt EXCEPT ![n] = NIL]
    /\ bad_release' = (bad_release \/ (filled[n] > 0))
    /\ UNCHANGED << head, filled, owned, gen, pushes >>

Next ==
    \/ \E n \in Nodes :
         \/ Alloc(n) \/ Free(n)
         \/ OwnerExitEmpty(n) \/ OwnerExitNonEmpty(n)
         \/ SweepRelink(n) \/ Adopt(n) \/ Release(n)
    \/ HeadAdvance

Spec == Init /\ [][Next]_vars

Fairness ==
    /\ WF_vars(HeadAdvance)
    /\ \A n \in Nodes : WF_vars(SweepRelink(n)) /\ WF_vars(Release(n))
    /\ \A n \in Nodes : WF_vars(OwnerExitEmpty(n)) /\ WF_vars(OwnerExitNonEmpty(n))
SpecLive == Init /\ [][Next]_vars /\ Fairness

\* ===========================================================
\*  INVARIANTS
\* ===========================================================

\* (A) Headline: never release a non-empty chunk.  CLEAN with dead-only
\*     removal; VIOLATED under AllowLiveRemoval.
Inv_NoBadRelease == ~bad_release

\* (B) The Path-B structural guarantee: a live non-empty chunk ALWAYS has a
\*     ref (owner-ref or chain-ref), so it is never releasable.  This is what
\*     replaces the self-ref.  (Equivalent to (A) holding throughout.)
Inv_NonEmptyHasRef ==
    \A n \in Nodes : (~released[n] /\ filled[n] > 0) => Refcnt(n) >= 1

\* (C) No use-after-free via a stale link.
Inv_NoDanglingNext ==
    \A m \in Nodes : (~released[m] /\ nxt[m] \in Nodes) => ~released[nxt[m]]

\* (D) Head is never a released chunk.
Inv_HeadAlive == head \in Nodes => ~released[head]

\* (E) A released chunk has no incoming refs.
Inv_ReleasedNoRefs == \A n \in Nodes : released[n] => Refcnt(n) = 0

\* (F) Acyclic chain.
RECURSIVE Hops(_, _)
Hops(n, i) == IF i = 0 \/ n = NIL THEN n ELSE Hops(nxt[n], i - 1)
Inv_Acyclic == \A n \in Nodes : Hops(n, Cardinality(Nodes)) = NIL

\* No-leak (liveness, *_live cfg): a chunk that ends empty AND unreferenced is
\* eventually released.
Liveness ==
    \A n \in Nodes : (<>[](filled[n] = 0 /\ Refcnt(n) = 0)) => <>released[n]

THEOREM Spec => []TypeOK
THEOREM Spec => []Inv_NoBadRelease
THEOREM Spec => []Inv_NonEmptyHasRef
THEOREM Spec => []Inv_NoDanglingNext
THEOREM Spec => []Inv_HeadAlive
THEOREM Spec => []Inv_ReleasedNoRefs
THEOREM Spec => []Inv_Acyclic
THEOREM SpecLive => Liveness

================================================================================
