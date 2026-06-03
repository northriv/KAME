(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version.

        Pick whichever license suits your project.
 ***************************************************************************)
----------------------------- MODULE ChunkClaim -----------------------------
(*
 * TLA+ model of kamepoolalloc's region->unit chunk-claim protocol, focused
 * on the s_back_offset[] publication race fixed in commit
 * "fix back_offset speculative-write data race".
 *
 * Background (see kamepoolalloc/tests/CHUNK_CLAIM_TLA_NOTES.md):
 *   - A region is a row of NUnits 256 KiB units.
 *   - Multiple allocator templates with DIFFERENT CHUNK_UNITS strides
 *     (here CUof[p] in {2,4}) claim multi-unit chunks from the SAME
 *     shared bitmap + back_offset table.
 *   - A chunk occupies CU contiguous, CU-aligned units.  Claiming is a
 *     CAS over the units' claim bits; s_back_offset[u] for each unit u of
 *     the chunk is set to (u - base) so lookup_chunk(addr-in-u) can recover
 *     the chunk base in O(1).
 *
 * The bug: s_back_offset[] was published BEFORE the claim CAS.  A CAS
 * loser (different stride) had already written its speculative back_offset
 * entries, clobbering the entries the CAS WINNER legitimately owns.  A
 * later lookup_chunk() then resolved the wrong chunk base.
 *
 * `Speculative` selects the protocol variant:
 *   TRUE  = pre-fix  : WriteBackoff happens in the "scanned" phase, BEFORE
 *                      the CAS; on CAS failure the writes are NOT rolled
 *                      back.  TLC finds an INV2 (BackoffConsistent)
 *                      counterexample.
 *   FALSE = post-fix : back_offset is written only AFTER the CAS wins (the
 *                      units are then exclusively owned, so no other proc
 *                      writes those entries).  INV2 holds.
 *
 * Finiteness: each proc attempts at most ONE claim, then goes to "done"
 * (no retry loop) -> bounded state space with NO StateConstraint, per the
 * project's TLA+ convention.
 *)

EXTENDS Integers, FiniteSets, TLC

\* `Speculative` is the only model parameter (toggled by the .cfg):
\*   TRUE  = pre-fix  (write back_offset before the claim CAS)
\*   FALSE = post-fix (write back_offset only after winning the CAS)
CONSTANT Speculative

\* Topology hardcoded (the TLC .cfg parser does not accept `:>`/`@@`
\* function literals in CONSTANT assignments, so these live here as
\* definitions).  Two procs of DIFFERENT stride (CU=2, CU=4) in a
\* 4-unit region — the minimal configuration that exposes the
\* cross-stride back_offset clobber.
Procs  == {"p2", "p4"}
CUof   == ("p2" :> 2) @@ ("p4" :> 4)
NUnits == 4
MaxCU  == 4
FREE   == "FREE"

VARIABLES
    claim,          \* [0..NUnits-1 -> Procs \cup {FREE}] : claim-bit owner
    backoff,        \* [0..NUnits-1 -> 0..MaxCU-1]        : s_back_offset[]
    phase,          \* [Procs -> phase string]
    myBase          \* [Procs -> 0..NUnits-1]             : chosen base unit

vars == <<claim, backoff, phase, myBase>>

Units == 0 .. (NUnits - 1)

\* Units a proc with base b and stride cu would occupy.
Range(b, cu) == b .. (b + cu - 1)

\* A base is valid for proc p if it is CU-aligned and the chunk fits.
ValidBase(p, b) == /\ b % CUof[p] = 0
                   /\ b + CUof[p] <= NUnits

\* All units of [b, b+cu) are currently unclaimed (models "OCC_MASK == 0").
RangeFree(b, cu) == \A u \in Range(b, cu) : claim[u] = FREE

----------------------------------------------------------------------------
(* Initial state. *)
Init ==
    /\ claim   = [u \in Units |-> FREE]
    /\ backoff = [u \in Units |-> 0]
    /\ phase   = [p \in Procs |-> "idle"]
    /\ myBase  = [p \in Procs |-> 0]

----------------------------------------------------------------------------
(* Scan + choose a base from the CURRENT (possibly soon-stale) bitmap view.
   Models loading `v` and finding a CHUNK_STRIDE-aligned zero run. *)
ScanChoose(p) ==
    /\ phase[p] = "idle"
    /\ \E b \in Units :
         /\ ValidBase(p, b)
         /\ RangeFree(b, CUof[p])
         /\ myBase' = [myBase EXCEPT ![p] = b]
    /\ phase' = [phase EXCEPT ![p] = "scanned"]
    /\ UNCHANGED <<claim, backoff>>

(* PRE-FIX: publish back_offset speculatively, before the CAS. *)
WriteBackoffPre(p) ==
    /\ Speculative = TRUE
    /\ phase[p] = "scanned"
    /\ backoff' = [u \in Units |->
                     IF u \in Range(myBase[p], CUof[p])
                     THEN u - myBase[p] ELSE backoff[u]]
    /\ phase' = [phase EXCEPT ![p] = "wrote"]
    /\ UNCHANGED <<claim, myBase>>

(* PRE-FIX: the claim CAS, after the speculative back_offset write. *)
CasPre(p) ==
    /\ Speculative = TRUE
    /\ phase[p] = "wrote"
    /\ IF RangeFree(myBase[p], CUof[p])
       THEN /\ claim' = [u \in Units |->
                           IF u \in Range(myBase[p], CUof[p]) THEN p ELSE claim[u]]
            /\ phase' = [phase EXCEPT ![p] = "claimed"]
            /\ UNCHANGED <<backoff, myBase>>
       ELSE \* CAS failed: give up.  NOTE the speculative back_offset
            \* writes from WriteBackoffPre are NOT rolled back -> they may
            \* now describe units another proc legitimately owns.
            /\ phase' = [phase EXCEPT ![p] = "done"]
            /\ UNCHANGED <<claim, backoff, myBase>>

(* POST-FIX: win the CAS and publish back_offset together.
   In the real code these are two instructions (CAS, then the
   back_offset write loop), but the write is UNCONTENDED: after the CAS,
   claim[u]=p for exactly this proc's units, owners' unit-ranges are
   disjoint, and no other proc's ScanChoose can pick these units (not
   free) — so no other proc ever writes these back_offset entries, and
   no lookup observes them until create() (modelled as part of becoming
   "claimed").  Bundling them into one transition is therefore a sound
   reduction; it just removes the benign "won but not-yet-written"
   transient that is unobservable in the real code. *)
CasPost(p) ==
    /\ Speculative = FALSE
    /\ phase[p] = "scanned"
    /\ IF RangeFree(myBase[p], CUof[p])
       THEN /\ claim' = [u \in Units |->
                           IF u \in Range(myBase[p], CUof[p]) THEN p ELSE claim[u]]
            /\ backoff' = [u \in Units |->
                             IF u \in Range(myBase[p], CUof[p])
                             THEN u - myBase[p] ELSE backoff[u]]
            /\ phase' = [phase EXCEPT ![p] = "claimed"]
            /\ UNCHANGED myBase
       ELSE /\ phase' = [phase EXCEPT ![p] = "done"]
            /\ UNCHANGED <<claim, backoff, myBase>>

(* An idle proc with no claimable run gives up (terminal).  Models "this
   stride found no free unit-run in this region" — in the real allocator
   it would mmap a fresh region; here it simply stops.  Keeps the bounded
   protocol deadlock-free so TLC explores all interleavings for the
   safety invariants. *)
GiveUp(p) ==
    /\ phase[p] = "idle"
    /\ ~\E b \in Units : ValidBase(p, b) /\ RangeFree(b, CUof[p])
    /\ phase' = [phase EXCEPT ![p] = "done"]
    /\ UNCHANGED <<claim, backoff, myBase>>

Next ==
    \E p \in Procs :
        \/ ScanChoose(p)
        \/ WriteBackoffPre(p)
        \/ CasPre(p)
        \/ CasPost(p)
        \/ GiveUp(p)

\* Stutter when all procs are terminal (claimed/done) so TLC has no
\* deadlock once the bounded protocol finishes.
Done == \A p \in Procs : phase[p] \in {"claimed", "done"}
Terminating == Done /\ UNCHANGED vars

Spec == Init /\ [][Next \/ Terminating]_vars

----------------------------------------------------------------------------
(* --- Invariants --- *)

TypeOK ==
    /\ claim   \in [Units -> Procs \cup {FREE}]
    /\ backoff \in [Units -> 0 .. (MaxCU - 1)]
    /\ phase   \in [Procs -> {"idle","scanned","wrote","claimed","done"}]
    /\ myBase  \in [Procs -> Units]

\* INV1: no unit is owned by two procs.  Trivially true (claim is a
\* function), kept as documentation of the intended property.
NoOverlap == \A u \in Units : claim[u] \in Procs \cup {FREE}

\* INV2 (the property the bug violated): for every claimed unit u, the
\* back_offset table must resolve u to the base of the chunk that owns u.
\* base-of-owner = myBase[claim[u]] (the proc owns exactly one CU-range).
\* lookup_chunk computes base = u - backoff[u]; correctness <=>
\*   backoff[u] = u - myBase[claim[u]].
BackoffConsistent ==
    \A u \in Units :
        claim[u] # FREE => backoff[u] = u - myBase[claim[u]]

\* Equivalent lookup phrasing, for readability in counterexamples.
LookupBase(u) == u - backoff[u]
LookupResolvesOwner ==
    \A u \in Units :
        claim[u] # FREE => LookupBase(u) = myBase[claim[u]]

=============================================================================
