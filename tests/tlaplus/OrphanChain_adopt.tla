(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        Dual-licensed Apache 2.0 OR GPL-2.0-or-later — see OrphanChain_atomicshared.tla.
 ***************************************************************************)
--------------------------- MODULE OrphanChain_adopt ---------------------------
(*
 * Models the ACTUAL implemented adopt mechanism (commit bb6d691d), closing the
 * Linux-review watch-item: the 2-step pop->BIT_OWNED claim + the oc_hold
 * parked-ref + the atomic_intrusive_dispose BIT_OWNED gate + residual scrub
 * pins.  Refines OrphanChain_pathB.tla, which (a) used the dropped 3-ref
 * owner-ref model and (b) modelled adopt as a single atomic step.
 *
 * FAITHFUL to atomic_shared_ptr's SYNCHRONOUS disposal: disposal is folded INTO
 * each reference-dropping action (HeadAdvance/ScrubUnlink/ScrubUnpin/
 * AdoptDropRef), so refcnt=0 <=> released by construction — no node sits at
 * refcnt 0 still pointing at its successor (atomic_shared_ptr's deleter runs at
 * the instant refcnt hits 0, atomically clearing m_orphan_next).
 *
 * Faithful to the raw-DLL design:
 *   - refcnt(n) = chain-ref (head / predecessor m_orphan_next) + pins.
 *     OWNED chunks are RAW (not refcounted) — no owner-ref term.
 *   - pins: a scrubber `load_shared`s on-chain nodes (scrub_pin); an adopter
 *     holds the popped node through the claim (pop_ref = oc_hold).
 *
 * TWO distinct free mechanisms (this is the crux the model exposes):
 *   (1) smart_ptr disposal — refcnt->0 routes to atomic_intrusive_dispose,
 *       which is GATED on ~owned (allocator_prv.h:1917 `if(BIT_OWNED) return;`).
 *   (2) the OWNER's DIRECT free — release_dll_chunks_for_thread's empty branch
 *       (allocator.cpp:3026 `newv==0`) calls ~PoolAllocator()+deallocate_chunk
 *       based on MASK_CNT==0 alone.  It does NOT read the intrusive refcnt.
 *
 * The dispose comment (allocator_prv.h:1915) states the design's PRECONDITION
 * for (2): "the owner releases it via deallocate_chunk on empty (refcnt then
 * stays 0, off every smart_ptr)".  This module tests whether that precondition
 * is ENFORCED or merely TIMING-BASED:
 *   - Inv_NoBadRelease  — the gate (1) never releases an owned/non-empty chunk.
 *   - Inv_NoBadOwnerFree — the owner (2) never frees a chunk with refcnt > 0.
 *     bad_ownerfree fires iff OwnerExitEmpty deallocates n while a residual
 *     scrub pin or a lingering pinned-predecessor's m_orphan_next still points
 *     at n => the freed chunk's control block is later decremented by that
 *     dangling smart_ptr (UAF / refcnt corruption on the recycled chunk).
 *
 * KNOB GateOnOwned: TRUE = the design.  FALSE = drop the gate (1) — proves
 * the BIT_OWNED disposal gate is load-bearing (Inv_NoBadRelease VIOLATION).
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS N1, N2, N3, NIL, GateOnOwned, MaxGen

Nodes == { N1, N2, N3 }

ASSUME NIL \notin Nodes
ASSUME GateOnOwned \in BOOLEAN
ASSUME MaxGen \in Nat

VARIABLES
    head,        \* Nodes \cup {NIL}
    nxt,         \* [Nodes -> Nodes \cup {NIL}]   chain forward link (m_orphan_next)
    filled,      \* [Nodes -> 0..1]               MASK_CNT (0 = empty)
    owned,       \* [Nodes -> BOOLEAN]            BIT_OWNED (re-owned off the chain)
    released,    \* [Nodes -> BOOLEAN]
    pop_ref,     \* Nodes \cup {NIL}              oc_hold: the node an adopter holds mid-2-step
    scrub_pin,   \* Nodes \cup {NIL}              a node a scrubber currently load_shared-pins
    gen,         \* Nat                            adopts so far (bound)
    bad_release, \* BOOLEAN sticky — smart_ptr disposal released an owned/non-empty chunk
    bad_ownerfree \* BOOLEAN sticky — owner deallocate_chunk'd a chunk with refcnt > 0

vars == << head, nxt, filled, owned, released, pop_ref, scrub_pin, gen,
           bad_release, bad_ownerfree >>

\* refcnt = chain-in + pins.  NO owner-ref (raw DLL): a re-owned chunk the owner
\* holds is NOT counted here — exactly why the owner's free can't consult it.
ChainIn(n) == Cardinality({ m \in Nodes : nxt[m] = n }) + (IF head = n THEN 1 ELSE 0)
Refcnt(n)  == ChainIn(n)
              + (IF pop_ref   = n THEN 1 ELSE 0)
              + (IF scrub_pin = n THEN 1 ELSE 0)

\* Post-state refcnt of node n under a hypothetical (h, nx, pr, sp) — used to
\* decide synchronous disposal INSIDE the action that drops n's last reference.
RefcntAt(n, h, nx, pr, sp) ==
    Cardinality({ m \in Nodes : m # n /\ nx[m] = n })
    + (IF h  = n THEN 1 ELSE 0)
    + (IF pr = n THEN 1 ELSE 0)
    + (IF sp = n THEN 1 ELSE 0)

TypeOK ==
    /\ head \in Nodes \cup {NIL}
    /\ nxt \in [Nodes -> Nodes \cup {NIL}]
    /\ filled \in [Nodes -> 0..1]
    /\ owned \in [Nodes -> BOOLEAN]
    /\ released \in [Nodes -> BOOLEAN]
    /\ pop_ref \in Nodes \cup {NIL}
    /\ scrub_pin \in Nodes \cup {NIL}
    /\ gen \in 0..MaxGen
    /\ bad_release \in BOOLEAN
    /\ bad_ownerfree \in BOOLEAN

\* Init: head -> N1 -> N2(empty) -> N3 -> NIL, all orphans (not owned).
Init ==
    /\ head = N1
    /\ nxt = [x \in Nodes |-> IF x = N1 THEN N2 ELSE IF x = N2 THEN N3 ELSE NIL]
    /\ filled = [x \in Nodes |-> IF x = N2 THEN 0 ELSE 1]
    /\ owned = [x \in Nodes |-> FALSE]
    /\ released = [x \in Nodes |-> FALSE]
    /\ pop_ref = NIL
    /\ scrub_pin = NIL
    /\ gen = 0
    /\ bad_release = FALSE
    /\ bad_ownerfree = FALSE

\* --- cross-thread free of a slot (orphan drains) ---
Free(n) ==
    /\ filled[n] = 1 /\ ~released[n]
    /\ filled' = [filled EXCEPT ![n] = 0]
    /\ UNCHANGED << head, nxt, owned, released, pop_ref, scrub_pin, gen,
                    bad_release, bad_ownerfree >>

\* --- owner allocates into a re-owned chunk (refill) ---
Alloc(n) ==
    /\ owned[n] /\ filled[n] = 0 /\ ~released[n]
    /\ filled' = [filled EXCEPT ![n] = 1]
    /\ UNCHANGED << head, nxt, owned, released, pop_ref, scrub_pin, gen,
                    bad_release, bad_ownerfree >>

\* --- scrubber load_shared-pins an on-chain node (transient) ---
ScrubPin(n) ==
    /\ scrub_pin = NIL /\ ~released[n] /\ ChainIn(n) > 0
    /\ scrub_pin' = n
    /\ UNCHANGED << head, nxt, filled, owned, released, pop_ref, gen,
                    bad_release, bad_ownerfree >>

\* ScrubUnpin drops the pin.  If that was the node's last reference and it is
\* unowned (gate), dispose synchronously: mark released, clear its m_orphan_next.
ScrubUnpin ==
    /\ scrub_pin /= NIL
    /\ LET n   == scrub_pin
           rc  == RefcntAt(n, head, nxt, pop_ref, NIL)   \* pin removed
           rel == (rc = 0) /\ (GateOnOwned => ~owned[n])
       IN /\ scrub_pin' = NIL
          /\ released'    = IF rel THEN [released EXCEPT ![n] = TRUE] ELSE released
          /\ nxt'         = IF rel THEN [nxt EXCEPT ![n] = NIL] ELSE nxt
          /\ bad_release' = IF rel THEN (bad_release \/ (filled[n] > 0) \/ owned[n])
                                   ELSE bad_release
    /\ UNCHANGED << head, filled, owned, pop_ref, gen, bad_ownerfree >>

\* --- scrub unlinks a DEAD (empty), non-owned orphan; relink preserves successor ---
ScrubUnlink(p) ==
    /\ nxt[p] \in Nodes
    /\ filled[nxt[p]] = 0 /\ ~owned[nxt[p]]
    /\ ~released[p] /\ ~released[nxt[p]]
    /\ LET c        == nxt[p]
           relinked == [nxt EXCEPT ![p] = nxt[c]]
           rc       == RefcntAt(c, head, relinked, pop_ref, scrub_pin)
           rel      == (rc = 0)   \* c empty & unowned by precondition
       IN /\ nxt'      = IF rel THEN [relinked EXCEPT ![c] = NIL] ELSE relinked
          /\ released' = IF rel THEN [released EXCEPT ![c] = TRUE] ELSE released
    /\ UNCHANGED << head, filled, owned, pop_ref, scrub_pin, gen,
                    bad_release, bad_ownerfree >>

\* HeadAdvance: scrub advances s_orphan_chain_head off a dead head; dispose the
\* old head synchronously if it then has no references (pin/predecessor).
HeadAdvance ==
    /\ head \in Nodes /\ filled[head] = 0 /\ ~owned[head] /\ ~released[head]
    /\ LET h   == head
           s   == nxt[head]
           rc  == RefcntAt(h, s, nxt, pop_ref, scrub_pin)   \* head moved to s
           rel == (rc = 0)
       IN /\ head'     = s
          /\ nxt'      = IF rel THEN [nxt EXCEPT ![h] = NIL] ELSE nxt
          /\ released' = IF rel THEN [released EXCEPT ![h] = TRUE] ELSE released
    /\ UNCHANGED << filled, owned, pop_ref, scrub_pin, gen,
                    bad_release, bad_ownerfree >>

\* --- adopt: 3 separate atomic steps (orphan_chain_pop -> claim -> drop oc_hold) ---
\* (1) pop head: remove from chain, take pop_ref (oc_hold), clear m_orphan_next.
\*     Net refcnt change is zero (loses head-ref, gains pop_ref) — no disposal.
AdoptPop(n) ==
    /\ gen < MaxGen
    /\ head = n /\ n \in Nodes /\ ~released[n] /\ ~owned[n]
    /\ pop_ref = NIL
    /\ head' = nxt[n]
    /\ nxt' = [nxt EXCEPT ![n] = NIL]
    /\ pop_ref' = n
    /\ gen' = gen + 1
    /\ UNCHANGED << filled, owned, released, scrub_pin, bad_release, bad_ownerfree >>
\* (2) claim BIT_OWNED (re-own).  oc_hold (pop_ref) still held => refcnt >= 1.
AdoptClaim(n) ==
    /\ pop_ref = n /\ ~owned[n] /\ ~released[n]
    /\ owned' = [owned EXCEPT ![n] = TRUE]
    /\ UNCHANGED << head, nxt, filled, released, pop_ref, scrub_pin, gen,
                    bad_release, bad_ownerfree >>
\* (3) drop oc_hold AFTER claim.  THIS IS THE smart_ptr DISPOSAL/GATE SITE.
AdoptDropRef(n) ==
    /\ pop_ref = n /\ owned[n]
    /\ LET rc  == RefcntAt(n, head, nxt, NIL, scrub_pin)   \* oc_hold removed
           rel == (rc = 0) /\ (GateOnOwned => ~owned[n])
       IN /\ pop_ref'    = NIL
          /\ released'    = IF rel THEN [released EXCEPT ![n] = TRUE] ELSE released
          /\ nxt'         = IF rel THEN [nxt EXCEPT ![n] = NIL] ELSE nxt
          /\ bad_release' = IF rel THEN (bad_release \/ (filled[n] > 0) \/ owned[n])
                                   ELSE bad_release
    /\ UNCHANGED << head, filled, owned, scrub_pin, gen, bad_ownerfree >>

\* --- owner releases a re-owned chunk at owner-exit (release_dll_chunks_for_thread) ---
\* empty branch (newv==0): DIRECT ~PoolAllocator + deallocate_chunk, gated ONLY on
\* MASK_CNT (filled=0) — NOT on the intrusive refcnt.  bad_ownerfree fires if a
\* residual smart_ptr (scrub pin / lingering pinned predecessor) still references
\* n at this moment: that pointer becomes dangling and later corrupts the freed/
\* recycled chunk's control block.  (Refcnt here excludes pop_ref by precondition.)
OwnerExitEmpty(n) ==
    /\ owned[n] /\ filled[n] = 0 /\ ~released[n] /\ pop_ref /= n
    /\ released' = [released EXCEPT ![n] = TRUE]
    /\ owned' = [owned EXCEPT ![n] = FALSE]
    /\ bad_ownerfree' = (bad_ownerfree \/ (Refcnt(n) > 0))
    /\ UNCHANGED << head, nxt, filled, pop_ref, scrub_pin, gen, bad_release >>
\* non-empty branch: re-push to the chain (orphan_chain_push); chunk stays alive.
OwnerExitNonEmpty(n) ==
    /\ owned[n] /\ filled[n] = 1 /\ ~released[n] /\ pop_ref /= n /\ ChainIn(n) = 0
    /\ owned' = [owned EXCEPT ![n] = FALSE]
    /\ nxt' = [nxt EXCEPT ![n] = head]
    /\ head' = n
    /\ UNCHANGED << filled, released, pop_ref, scrub_pin, gen, bad_release, bad_ownerfree >>

Next ==
    \/ \E n \in Nodes :
         \/ Free(n) \/ Alloc(n) \/ ScrubPin(n) \/ ScrubUnlink(n)
         \/ AdoptPop(n) \/ AdoptClaim(n) \/ AdoptDropRef(n)
         \/ OwnerExitEmpty(n) \/ OwnerExitNonEmpty(n)
    \/ ScrubUnpin \/ HeadAdvance

Spec == Init /\ [][Next]_vars

\* ============================================================
\*  INVARIANTS
\* ============================================================

\* (A) smart_ptr disposal never releases a non-empty OR re-owned chunk.  The
\*     BIT_OWNED disposal gate makes the "owned" half hold; CLEAN under
\*     GateOnOwned, VIOLATED without it.
Inv_NoBadRelease == ~bad_release

\* (A') The OWNER's direct deallocate_chunk never frees a chunk that a smart_ptr
\*     still references (refcnt > 0).  Tests the dispose-comment precondition
\*     "refcnt then stays 0, off every smart_ptr" — is it enforced or timing?
Inv_NoBadOwnerFree == ~bad_ownerfree

\* (B) A re-owned chunk is OFF the chain (no live link references an owned node).
Inv_OwnedNotChained == \A n \in Nodes : owned[n] => ChainIn(n) = 0

\* (C) No live link points at a released chunk (UAF surrogate).
Inv_NoDanglingNext ==
    \A m \in Nodes : (~released[m] /\ nxt[m] \in Nodes) => ~released[nxt[m]]

\* (D) A released chunk has no incoming refs.
Inv_ReleasedNoRefs == \A n \in Nodes : released[n] => Refcnt(n) = 0

\* (E) Acyclic chain.
RECURSIVE Hops(_, _)
Hops(n, i) == IF i = 0 \/ n = NIL THEN n ELSE Hops(nxt[n], i - 1)
Inv_Acyclic == \A n \in Nodes : Hops(n, Cardinality(Nodes)) = NIL

THEOREM Spec => []TypeOK
THEOREM Spec => []Inv_NoBadRelease
THEOREM Spec => []Inv_NoBadOwnerFree

================================================================================
