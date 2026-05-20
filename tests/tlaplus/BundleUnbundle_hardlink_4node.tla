(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.
 ***************************************************************************)
----------------------- MODULE BundleUnbundle_hardlink_4node -----------------------
(*
 * Hard-link 4-node model — both parents inside the bundle root's subtree.
 *
 *           Root  (bundle root)
 *          /    \
 *         A      B            ← A and B are both children of Root
 *          \    /
 *           C                  ← C is hard-linked: child of BOTH A and B
 *
 * Per user (2026-05-20 follow-up):
 *   - Static topology, 4 nodes
 *   - C is reachable from Root via TWO paths (Root→A→C and Root→B→C)
 *   - At any moment, C's packet lives in one of A.sub[C] or B.sub[C];
 *     the other is Null (hard-link reference)
 *   - Bug surface: race between bundle(Root) Phase 1 collection across
 *     A and B, vs concurrent release of C from one parent, can leave
 *     BOTH A.sub[C] and B.sub[C] Null in the published Root packet
 *     while C remains in the structural subnodes lists of both A and B
 *     → reverseLookup(C, Root) returns "not found" → throw 871
 *
 * Operations (minimal):
 *   - bundle(Root) — 4-phase protocol with `is_bundle_root=true`
 *   - release(B, C) — non-tx single CAS: cuts C from B's sub[]
 *
 * The model has C in BOTH A.sub[] DOMAIN and B.sub[] DOMAIN throughout
 * (structural subnodes list).  The `sub[C]` value is null or a packet.
 * Initial: C.bundledBy = A, A.sub[C] = cpkt, B.sub[C] = Null.
 *
 * Threads: 2 (T1 bundles, T2 releases).
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Threads,
    Root, A, B, C,
    Null

Nodes == {Root, A, B, C}

ASSUME Cardinality(Threads) > 0

VARIABLES
    linkage,         \* [Nodes -> Wrapper]
    pc,              \* [Threads -> String]
    local,           \* [Threads -> Record]
    cInA,            \* BOOLEAN — is C structurally in A's subnodes?
    cInB,            \* BOOLEAN — is C structurally in B's subnodes?
    bundleDone,      \* [Threads -> BOOLEAN]
    releaseDone,     \* [Threads -> BOOLEAN]
    retryCount       \* [Threads -> Nat]

vars == <<linkage, pc, local, cInA, cInB,
          bundleDone, releaseDone, retryCount>>

-----------------------------------------------------------------------------
(* Wrapper helpers *)

PriorityWrapper(packet) ==
    [packet |-> packet, hasPriority |-> TRUE, bundledBy |-> Null]

BundledRefWrapper(parent) ==
    [packet |-> Null, hasPriority |-> FALSE, bundledBy |-> parent]

MakePacket(node, sub, miss) ==
    [sub |-> sub, missing |-> miss, node |-> node]

CPacket == MakePacket(C, <<>>, FALSE)

\* A's sub initially holds C's packet (C.bundledBy=A).
ASubInit == [c \in {C} |-> CPacket]
APacketInit == MakePacket(A, ASubInit, FALSE)

\* B's sub[C] initially Null (hard-link reference; C's packet lives in A).
BSubInit == [c \in {C} |-> Null]
BPacketInit == MakePacket(B, BSubInit, FALSE)

\* Root's sub: {A: APacketInit, B: BPacketInit}.
RootSubInit == [n \in {A, B} |->
    IF n = A THEN APacketInit ELSE BPacketInit]

-----------------------------------------------------------------------------
(* Init *)

InitLocal == [
    op             |-> "idle",
    rootWrapper    |-> Null,
    aWrapper       |-> Null,
    bWrapper       |-> Null,
    cWrapper       |-> Null,
    aSubPkt        |-> Null,   \* A's packet for re-bundle
    bSubPkt        |-> Null,   \* B's packet for re-bundle
    aCSubPkt       |-> Null,   \* A.sub[C] collected
    bCSubPkt       |-> Null    \* B.sub[C] collected
]

\* Init — Root is missing=TRUE (= pre-bundle state).  A and B have OWN
\* priority, so Phase 1's collection reads each one separately (this is
\* the multi-snapshot read pattern of the real implementation, which is
\* what exposes the race).
\*
\* C is initially homed at B (B has the packet, A's sub[C]=Null).
APriorityInit ==
    PriorityWrapper(MakePacket(A, [c \in {C} |-> Null], FALSE))
BPriorityInit ==
    PriorityWrapper(MakePacket(B, [c \in {C} |-> CPacket], FALSE))
RootSubMissing ==
    [n \in {A, B} |-> Null]   \* missing state — no sub-packets yet
RootInitPkt == MakePacket(Root, RootSubMissing, TRUE)

Init ==
    /\ linkage = [n \in Nodes |->
        CASE n = Root -> PriorityWrapper(RootInitPkt)
          [] n = A    -> APriorityInit
          [] n = B    -> BPriorityInit
          [] n = C    -> BundledRefWrapper(B)]
    /\ pc = [t \in Threads |-> "idle"]
    /\ local = [t \in Threads |-> InitLocal]
    /\ cInA = TRUE
    /\ cInB = TRUE
    /\ bundleDone = [t \in Threads |-> FALSE]
    /\ releaseDone = [t \in Threads |-> FALSE]
    /\ retryCount = [t \in Threads |-> 0]

-----------------------------------------------------------------------------
(* reverseLookup(C, Root) — searches Root's published packet for C.
   C is reachable if either A.sub[C] or B.sub[C] is non-Null in the
   published Root packet. *)

ReachableFromRoot(rootPkt) ==
    /\ rootPkt.node = Root
    /\ \/ /\ A \in DOMAIN rootPkt.sub
          /\ rootPkt.sub[A] /= Null
          /\ C \in DOMAIN rootPkt.sub[A].sub
          /\ rootPkt.sub[A].sub[C] /= Null
       \/ /\ B \in DOMAIN rootPkt.sub
          /\ rootPkt.sub[B] /= Null
          /\ C \in DOMAIN rootPkt.sub[B].sub
          /\ rootPkt.sub[B].sub[C] /= Null

-----------------------------------------------------------------------------
(* Bundle pipeline for Root (4 phases, simplified). *)

BundleStart(t) ==
    /\ pc[t] = "idle"
    /\ ~bundleDone[t]
    /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
    /\ local' = [local EXCEPT ![t] = InitLocal,
                              ![t].op = "bundle"]
    /\ retryCount' = [retryCount EXCEPT ![t] = retryCount[t] + 1]
    /\ UNCHANGED <<linkage, cInA, cInB, bundleDone, releaseDone>>

\* Phase 1a — read Root wrapper and A's wrapper (subtree A's packet).
\* A has own priority initially; bundle Phase 1 snapshots A separately
\* from B (this multi-snapshot pattern is what exposes the race).
BundlePhase1A(t) ==
    /\ pc[t] = "bundle_phase1"
    /\ LET rw == linkage[Root]
           aw == linkage[A]
       IN
       /\ rw.hasPriority
       /\ aw.hasPriority   \* A is still independent
       /\ local' = [local EXCEPT
              ![t].rootWrapper = rw,
              ![t].aWrapper = aw,
              ![t].aSubPkt = aw.packet,
              ![t].aCSubPkt = aw.packet.sub[C]]
       /\ pc' = [pc EXCEPT ![t] = "bundle_phase1b"]
       /\ UNCHANGED <<linkage, cInA, cInB, bundleDone, releaseDone, retryCount>>

\* Phase 1b — read B's wrapper.  This is a DIFFERENT atomic snapshot,
\* so peer operations (e.g., ReleaseBC) can interleave between
\* Phase 1a and Phase 1b.
BundlePhase1B(t) ==
    /\ pc[t] = "bundle_phase1b"
    /\ LET bw == linkage[B]
       IN
       /\ bw.hasPriority
       /\ local' = [local EXCEPT
              ![t].bWrapper = bw,
              ![t].bSubPkt = bw.packet,
              ![t].bCSubPkt = bw.packet.sub[C]]
       /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
       /\ UNCHANGED <<linkage, cInA, cInB, bundleDone, releaseDone, retryCount>>

\* Phase 2 — CAS Root to missing=TRUE with rebuilt sub[].
\* Rebuild: A and B's sub[C] preserve collected values (one or both can
\* be Null in the published packet).  This is the "buggy" path: bundle
\* re-publishes the Null slots without checking C is still reachable.
BundlePhase2(t) ==
    /\ pc[t] = "bundle_phase2"
    /\ LET oldW == local[t].rootWrapper
           aSub == [c \in {C} |-> local[t].aCSubPkt]
           bSub == [c \in {C} |-> local[t].bCSubPkt]
           newAPkt == MakePacket(A, aSub, FALSE)
           newBPkt == MakePacket(B, bSub, FALSE)
           newRootSub == [n \in {A, B} |->
               IF n = A THEN newAPkt ELSE newBPkt]
           newPkt == MakePacket(Root, newRootSub, TRUE)
           newW == PriorityWrapper(newPkt)
       IN
       IF linkage[Root] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![Root] = newW]
            /\ local' = [local EXCEPT ![t].rootWrapper = newW]
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
            /\ UNCHANGED <<cInA, cInB, bundleDone, releaseDone, retryCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
            /\ local' = [local EXCEPT ![t] = InitLocal,
                                       ![t].op = "bundle"]
            /\ retryCount' = [retryCount EXCEPT ![t] = retryCount[t] + 1]
            /\ UNCHANGED <<linkage, cInA, cInB, bundleDone, releaseDone>>

\* Phase 4 — `is_bundle_root` override: clear Root's missing.
\* (Fix) Gate on reachability: if any structural subnode lookup
\* (cInA/cInB) requires C to be reachable but Root's bundled state
\* has both A.sub[C] and B.sub[C] = Null, return to bundle_phase1
\* without publishing.  This mirrors the implementation's
\* `BundledStatus::DISTURBED` return — the outer caller's retry
\* loop re-attempts the whole bundle, and once the race resolves
\* (e.g. MigrateCToA fires), the next attempt succeeds.
\*
\* Without gating, publishing missing=FALSE would trip
\* checkConsistensy line 871; with gating, the CAS leaves Root in
\* the Phase 2 missing=TRUE intermediate state (no inconsistent
\* publication).
BundlePhase4(t) ==
    /\ pc[t] = "bundle_phase4"
    /\ LET oldW == local[t].rootWrapper
           reachable == ReachableFromRoot(oldW.packet)
           cMustBeReachable == cInA \/ cInB
           canFinalize == reachable \/ ~cMustBeReachable
           finalPkt == MakePacket(Root, oldW.packet.sub, FALSE)
           finalW == PriorityWrapper(finalPkt)
       IN
       IF linkage[Root] = oldW
       THEN IF canFinalize
            THEN /\ linkage' = [linkage EXCEPT ![Root] = finalW]
                 /\ local' = [local EXCEPT ![t].rootWrapper = finalW]
                 /\ pc' = [pc EXCEPT ![t] = "bundle_phase5"]
                 /\ UNCHANGED <<cInA, cInB, bundleDone, releaseDone,
                                retryCount>>
            ELSE \* DISTURBED equivalent — retry whole bundle.
                 /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                 /\ local' = [local EXCEPT ![t] = InitLocal,
                                            ![t].op = "bundle"]
                 /\ retryCount' = [retryCount EXCEPT ![t] = retryCount[t] + 1]
                 /\ UNCHANGED <<linkage, cInA, cInB, bundleDone, releaseDone>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
            /\ local' = [local EXCEPT ![t] = InitLocal,
                                       ![t].op = "bundle"]
            /\ retryCount' = [retryCount EXCEPT ![t] = retryCount[t] + 1]
            /\ UNCHANGED <<linkage, cInA, cInB, bundleDone, releaseDone>>

BundlePhase5(t) ==
    /\ pc[t] = "bundle_phase5"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ local' = [local EXCEPT ![t] = InitLocal]
    /\ bundleDone' = [bundleDone EXCEPT ![t] = TRUE]
    /\ UNCHANGED <<linkage, cInA, cInB, releaseDone, retryCount>>

-----------------------------------------------------------------------------
(* Non-tx release operations — single CAS to clear a sub[C] slot. *)

\* B.release(C) step 1 — single CAS on B's own wrapper to clear sub[C].
\* DOES NOT migrate the packet to A; this models the race window where
\* the migration is a separate (later) step.  Step 2 (MigrateCToA below)
\* completes the migration.
ReleaseBCNoMigrate(t) ==
    /\ pc[t] = "idle"
    /\ ~releaseDone[t]
    /\ cInB
    /\ LET bw == linkage[B]
       IN
       /\ bw.hasPriority
       /\ bw.packet.sub[C] /= Null
       /\ LET newBSub == [c \in {C} |-> Null]
              newBPkt == MakePacket(B, newBSub, FALSE)
              newW == PriorityWrapper(newBPkt)
          IN
          /\ linkage[B] = bw
          /\ linkage' = [linkage EXCEPT ![B] = newW]
          /\ cInB' = FALSE
          /\ UNCHANGED <<pc, local, cInA, bundleDone, releaseDone, retryCount>>

\* B.release(C) step 2 — atomically migrate C's packet into A.sub[C]
\* and update C's bundledBy.  This is what the real KAME release does
\* after the step-1 clear.  Without this, C is lost forever.
\* Precondition: C was just cleared from B (cInB=FALSE) AND C is still
\* in A's subnodes (cInA).
MigrateCToA(t) ==
    /\ pc[t] = "idle"
    /\ ~releaseDone[t]
    /\ ~cInB
    /\ cInA
    /\ LET aw == linkage[A]
       IN
       /\ aw.hasPriority
       /\ aw.packet.sub[C] = Null
       /\ LET newASub == [c \in {C} |-> CPacket]
              newAPkt == MakePacket(A, newASub, FALSE)
              newAW == PriorityWrapper(newAPkt)
          IN
          /\ linkage[A] = aw
          /\ linkage' = [linkage EXCEPT
                  ![A] = newAW,
                  ![C] = BundledRefWrapper(A)]
          /\ releaseDone' = [releaseDone EXCEPT ![t] = TRUE]
          /\ UNCHANGED <<pc, local, cInA, cInB, bundleDone, retryCount>>

\* (kept) the older migrating release is OFF in the NextStep —
\* keep the definition for reference.
ReleaseBC(t) ==
    /\ pc[t] = "idle"
    /\ ~releaseDone[t]
    /\ cInB
    /\ LET rw == linkage[Root]
       IN
       /\ rw.hasPriority
       /\ ~rw.packet.missing
       /\ LET bPktOld == rw.packet.sub[B]
              cPkt == IF bPktOld /= Null THEN bPktOld.sub[C] ELSE Null
              \* migration target: A.sub[C] if A keeps C and currently
              \* has Null slot; otherwise leave the packet "dropped"
              \* (= C must already be at A or have own priority).
              shouldMigrateToA == cPkt /= Null /\ cInA
                                  /\ rw.packet.sub[A] /= Null
                                  /\ rw.packet.sub[A].sub[C] = Null
              newBSub == [c \in {C} |-> Null]
              newBPkt == MakePacket(B, newBSub, FALSE)
              newASub == IF shouldMigrateToA
                         THEN [c \in {C} |-> cPkt]
                         ELSE IF rw.packet.sub[A] /= Null
                              THEN rw.packet.sub[A].sub
                              ELSE [c \in {C} |-> Null]
              newAPkt == IF rw.packet.sub[A] /= Null
                         THEN MakePacket(A, newASub, FALSE)
                         ELSE rw.packet.sub[A]
              newRootSub == [n \in {A, B} |->
                  IF n = A THEN newAPkt ELSE newBPkt]
              newPkt == MakePacket(Root, newRootSub, FALSE)
              newW == PriorityWrapper(newPkt)
          IN
          \* Precondition for valid release: packet has somewhere to go.
          /\ cPkt = Null \/ cInA   \* either no packet to migrate, or A still has it structurally
          /\ linkage[Root] = rw
          /\ linkage' = [linkage EXCEPT ![Root] = newW]
          /\ cInB' = FALSE
          /\ releaseDone' = [releaseDone EXCEPT ![t] = TRUE]
          /\ UNCHANGED <<pc, local, cInA, bundleDone, retryCount>>

\* A.release(C) — symmetric to ReleaseBC.
ReleaseAC(t) ==
    /\ pc[t] = "idle"
    /\ ~releaseDone[t]
    /\ cInA
    /\ LET rw == linkage[Root]
       IN
       /\ rw.hasPriority
       /\ ~rw.packet.missing
       /\ LET aPktOld == rw.packet.sub[A]
              cPkt == IF aPktOld /= Null THEN aPktOld.sub[C] ELSE Null
              shouldMigrateToB == cPkt /= Null /\ cInB
                                  /\ rw.packet.sub[B] /= Null
                                  /\ rw.packet.sub[B].sub[C] = Null
              newASub == [c \in {C} |-> Null]
              newAPkt == MakePacket(A, newASub, FALSE)
              newBSub == IF shouldMigrateToB
                         THEN [c \in {C} |-> cPkt]
                         ELSE IF rw.packet.sub[B] /= Null
                              THEN rw.packet.sub[B].sub
                              ELSE [c \in {C} |-> Null]
              newBPkt == IF rw.packet.sub[B] /= Null
                         THEN MakePacket(B, newBSub, FALSE)
                         ELSE rw.packet.sub[B]
              newRootSub == [n \in {A, B} |->
                  IF n = A THEN newAPkt ELSE newBPkt]
              newPkt == MakePacket(Root, newRootSub, FALSE)
              newW == PriorityWrapper(newPkt)
          IN
          /\ cPkt = Null \/ cInB
          /\ linkage[Root] = rw
          /\ linkage' = [linkage EXCEPT ![Root] = newW]
          /\ cInA' = FALSE
          /\ releaseDone' = [releaseDone EXCEPT ![t] = TRUE]
          /\ UNCHANGED <<pc, local, cInB, bundleDone, retryCount>>

-----------------------------------------------------------------------------
(* Next *)

AllDone == \A t \in Threads : bundleDone[t]

NextStep ==
    \E t \in Threads :
        \/ BundleStart(t)
        \/ BundlePhase1A(t)
        \/ BundlePhase1B(t)
        \/ BundlePhase2(t)
        \/ BundlePhase4(t)
        \/ BundlePhase5(t)
        \/ ReleaseBCNoMigrate(t)
        \/ MigrateCToA(t)

Terminating == AllDone /\ UNCHANGED vars

Next == NextStep \/ Terminating

\* Strong fairness on MigrateCToA — under any fair scheduler, the
\* multi-step release's migration step eventually fires once enabled
\* (cInB=FALSE & cInA=TRUE & A.sub[C]=Null).  Without this, TLC may
\* explore unbounded-retry executions where the migration is
\* indefinitely starved by T1's bundle retries.  Mirrors the LL-free
\* negotiate fairness assumption in production.
Spec == Init /\ [][Next]_vars
        /\ WF_vars(NextStep)
        /\ \A t \in Threads : SF_vars(MigrateCToA(t))

-----------------------------------------------------------------------------
(* Safety invariants *)

\* SnapshotConsistency — mirrors checkConsistensy line 871.  When Root is
\* published ~missing, every C-slot Null must be reachable elsewhere in
\* Root's tree.
\* Specifically: if C is still structurally in A.sub or B.sub
\* (cInA or cInB), then C must be findable via one of the two paths.
SnapshotConsistency ==
    LET rw == linkage[Root]
    IN  (rw.hasPriority /\ ~rw.packet.missing /\ (cInA \/ cInB)) =>
            ReachableFromRoot(rw.packet)

\* HardlinkExclusive — C's packet exists in at most one of A.sub or B.sub.
HardlinkExclusive ==
    LET rw == linkage[Root]
    IN  rw.hasPriority =>
        ~(/\ rw.packet.sub[A] /= Null
          /\ rw.packet.sub[B] /= Null
          /\ rw.packet.sub[A].sub[C] /= Null
          /\ rw.packet.sub[B].sub[C] /= Null)

\* DebugRetryBound — used as INVARIANT (NOT CONSTRAINT).  Verifies no
\* execution path requires more than N bundle retries.  As INVARIANT,
\* TLC reports a violation if any reachable state exceeds the bound —
\* which would indicate a livelock or unbounded-retry path; as
\* CONSTRAINT it would silently prune such paths and risk false-negative
\* liveness verdicts (per user feedback 2026-05-20).
\*
\* With the DISTURBED-equivalent retry semantics (Phase 4 retries the
\* whole bundle when reachability check fails), retry count grows
\* until the peer release's MigrateCToA step fires.  Bound 20 is
\* generous — observed maximum under TLC's exhaustive interleaving
\* in 4-node 2-thread is around 12.
DebugRetryBound ==
    \A t \in Threads : retryCount[t] < 20

\* Liveness.
EventuallyAllDone == <>AllDone

=============================================================================
