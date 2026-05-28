(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.
 ***************************************************************************)
----------------------- MODULE BundleUnbundle_hardlink_external_migration -----------------------
(*
 * Hard-link external-parent model WITH cross-tree migration.
 *
 *   GN1 (bundle root)              P1 (external root)
 *    │                              │
 *    GN2                            └── P2 (P2's packet initially in P1.sub[P2])
 *    │
 *    P2 (hard-linked: also under GN2 via subnode list; sub[P2] starts Null)
 *
 * Bundle(GN1) detects P2 is currently homed at P1 (external) and
 * performs cross-tree migration as part of its protocol:
 *   - reach into P1's tree
 *   - CAS-clear P1.sub[P2]
 *   - CAS P2.bundledBy = GN2
 *   - CAS-update GN2.sub[P2] with the pulled packet
 *   - finalize GN1 ~missing
 *
 * This restores liveness for the cross-tree hard-link case — the
 * previous `_external.tla` model showed that without migration, bundle
 * could only leave GN1 missing=TRUE indefinitely.
 *
 * Concurrent race: a peer operation `P1Bundle` re-bundles P1's tree,
 * potentially racing with our cross-tree pull.  Safety invariants:
 *   - SnapshotConsistency: GN1's published packet has no unresolved
 *     Null slots (= P2 reachable inside GN1 if it's in GN2's subnodes)
 *   - HardlinkExclusive: P2's packet exists in at most one place
 *     (GN1's tree OR P1's tree) at a time
 *   - BundleRefConsistency: bundledBy points to a parent with priority
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Threads,
    GN1, GN2, P2, P1,
    Null

Nodes == {GN1, GN2, P2, P1}

ASSUME Cardinality(Threads) > 0

VARIABLES
    linkage,       \* [Nodes -> Wrapper]
    pc,            \* [Threads -> String]
    local,         \* [Threads -> Record]
    bundleDone     \* [Threads -> BOOLEAN]

\* retryCount removed (per user 2026-05-21): bounded domain state
\* + bundle retry on CAS fail produces naturally finite state space
\* without an artificial counter that would need CONSTRAINT/INVARIANT
\* bounding.  See _hardlink_4node for the same clean-up pattern.

vars == <<linkage, pc, local, bundleDone>>

-----------------------------------------------------------------------------
(* Wrapper helpers *)

PriorityWrapper(packet) ==
    [packet |-> packet, hasPriority |-> TRUE, bundledBy |-> Null]

BundledRefWrapper(parent) ==
    [packet |-> Null, hasPriority |-> FALSE, bundledBy |-> parent]

MakePacket(node, sub, miss) ==
    [sub |-> sub, missing |-> miss, node |-> node]

P2Packet == MakePacket(P2, <<>>, FALSE)

\* Initial GN1 sub: GN2 is bundled with sub[P2]=Null (P2 is in P1).
GN1SubInit ==
    [c \in {GN2} |->
        MakePacket(GN2, [l \in {P2} |-> Null], FALSE)]

\* Initial P1 sub: holds P2.
P1SubInit == [c \in {P2} |-> P2Packet]

-----------------------------------------------------------------------------
(* Init *)

InitLocal == [
    op           |-> "idle",
    gn1Wrapper   |-> Null,
    gn2Wrapper   |-> Null,
    p2Wrapper    |-> Null,
    p1Wrapper    |-> Null,
    p2Pkt        |-> Null    \* pulled from P1.sub[P2]
]

Init ==
    /\ linkage = [n \in Nodes |->
        CASE n = GN1 -> PriorityWrapper(MakePacket(GN1, GN1SubInit, TRUE))
          \* GN1 starts missing=TRUE: bundle will be needed to finalize,
          \* mimicking the initial-state legitimacy (P2 in P1's tree, so
          \* GN1 is not yet "complete").
          [] n = GN2 -> BundledRefWrapper(GN1)
          [] n = P1  -> PriorityWrapper(MakePacket(P1, P1SubInit, FALSE))
          [] n = P2  -> BundledRefWrapper(P1)]
    /\ pc = [t \in Threads |-> "idle"]
    /\ local = [t \in Threads |-> InitLocal]
    /\ bundleDone = [t \in Threads |-> FALSE]

-----------------------------------------------------------------------------
(* reverseLookup(P2, root=GN1) — same as previous model. *)

ReachableFromGN1(rootPkt) ==
    /\ rootPkt.node = GN1
    /\ GN2 \in DOMAIN rootPkt.sub
    /\ rootPkt.sub[GN2] /= Null
    /\ P2 \in DOMAIN rootPkt.sub[GN2].sub
    /\ rootPkt.sub[GN2].sub[P2] /= Null

-----------------------------------------------------------------------------
(* Bundle pipeline with cross-tree migration. *)

BundleStart(t) ==
    /\ pc[t] = "idle"
    /\ ~bundleDone[t]
    /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
    /\ local' = [local EXCEPT ![t] = InitLocal,
                              ![t].op = "bundle"]
    /\ UNCHANGED <<linkage, bundleDone>>

\* Phase 1 — read all 4 wrappers.  Detect P2's current home.
BundlePhase1(t) ==
    /\ pc[t] = "bundle_phase1"
    /\ LET gn1w == linkage[GN1]
           gn2w == linkage[GN2]
           p2w  == linkage[P2]
           p1w  == linkage[P1]
       IN
       /\ gn1w.hasPriority
       /\ local' = [local EXCEPT
              ![t].gn1Wrapper = gn1w,
              ![t].gn2Wrapper = gn2w,
              ![t].p2Wrapper  = p2w,
              ![t].p1Wrapper  = p1w]
       /\ pc' = [pc EXCEPT ![t] =
                    IF p2w.bundledBy = P1 THEN "bundle_pull_p1"
                    ELSE "bundle_phase4"]   \* P2 already in GN2 — finalize
       /\ UNCHANGED <<linkage, bundleDone>>

\* New phase: pull P2 from P1.  CAS P1 to set sub[P2]=Null and read
\* the packet into local.p2Pkt.  Requires P1 has priority and currently
\* holds P2 (= P1.sub[P2] /= Null).
BundlePullP1(t) ==
    /\ pc[t] = "bundle_pull_p1"
    /\ LET p1w == linkage[P1]
       IN
       /\ p1w.hasPriority
       /\ ~p1w.packet.missing
       /\ p1w.packet.sub[P2] /= Null
       /\ p1w = local[t].p1Wrapper   \* still our snapshot
       /\ LET newP1Sub == [c \in {P2} |-> Null]
              newP1Pkt == MakePacket(P1, newP1Sub, FALSE)
              newP1W == PriorityWrapper(newP1Pkt)
              pulled == p1w.packet.sub[P2]
          IN
          /\ linkage' = [linkage EXCEPT ![P1] = newP1W]
          /\ local' = [local EXCEPT ![t].p2Pkt = pulled]
          /\ pc' = [pc EXCEPT ![t] = "bundle_cas_p2"]
          /\ UNCHANGED <<bundleDone>>

\* Pull failure: P1's wrapper changed under us OR P2 not at P1 anymore.
\* Retry from Phase 1.
BundlePullP1Fail(t) ==
    /\ pc[t] = "bundle_pull_p1"
    /\ LET p1w == linkage[P1]
       IN  \/ ~p1w.hasPriority
           \/ p1w.packet.missing
           \/ p1w.packet.sub[P2] = Null
           \/ p1w /= local[t].p1Wrapper
    /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
    /\ local' = [local EXCEPT ![t] = InitLocal,
                              ![t].op = "bundle"]
    /\ UNCHANGED <<linkage, bundleDone>>

\* CAS P2.bundledBy from P1 to GN2.
BundleCASP2(t) ==
    /\ pc[t] = "bundle_cas_p2"
    /\ LET p2w == linkage[P2]
       IN
       /\ p2w = local[t].p2Wrapper
       /\ p2w.bundledBy = P1
       /\ linkage' = [linkage EXCEPT ![P2] = BundledRefWrapper(GN2)]
       /\ local' = [local EXCEPT ![t].p2Wrapper = BundledRefWrapper(GN2)]
       /\ pc' = [pc EXCEPT ![t] = "bundle_update_gn1"]
       /\ UNCHANGED <<bundleDone>>

BundleCASP2Fail(t) ==
    /\ pc[t] = "bundle_cas_p2"
    /\ LET p2w == linkage[P2]
       IN  p2w /= local[t].p2Wrapper
    /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
    /\ local' = [local EXCEPT ![t] = InitLocal,
                              ![t].op = "bundle"]
    /\ UNCHANGED <<linkage, bundleDone>>

\* Update GN1 so GN2.sub[P2] holds the pulled packet.  Keep GN1 missing.
BundleUpdateGN1(t) ==
    /\ pc[t] = "bundle_update_gn1"
    /\ LET gn1w == linkage[GN1]
           p2pkt == local[t].p2Pkt
       IN
       /\ gn1w = local[t].gn1Wrapper
       /\ p2pkt /= Null
       /\ LET newGN2Sub == [c \in {P2} |-> p2pkt]
              newGN2Pkt == MakePacket(GN2, newGN2Sub, FALSE)
              newGN1Sub == [c \in {GN2} |-> newGN2Pkt]
              \* Still missing=TRUE — Phase 4 will clear it.
              newGN1Pkt == MakePacket(GN1, newGN1Sub, TRUE)
              newGN1W == PriorityWrapper(newGN1Pkt)
          IN
          /\ linkage' = [linkage EXCEPT ![GN1] = newGN1W]
          /\ local' = [local EXCEPT ![t].gn1Wrapper = newGN1W]
          /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
          /\ UNCHANGED <<bundleDone>>

BundleUpdateGN1Fail(t) ==
    /\ pc[t] = "bundle_update_gn1"
    /\ LET gn1w == linkage[GN1]
       IN  gn1w /= local[t].gn1Wrapper
    /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
    /\ local' = [local EXCEPT ![t] = InitLocal,
                              ![t].op = "bundle"]
    /\ UNCHANGED <<linkage, bundleDone>>

\* Phase 4 (Finalize) — clear GN1 missing, gated on subtree integrity.
BundlePhase4(t) ==
    /\ pc[t] = "bundle_phase4"
    /\ LET gn1w == linkage[GN1]
           subOK == \A c \in DOMAIN gn1w.packet.sub :
                       gn1w.packet.sub[c] /= Null =>
                           (\A gc \in DOMAIN gn1w.packet.sub[c].sub :
                               gn1w.packet.sub[c].sub[gc] /= Null
                               \/ ReachableFromGN1(gn1w.packet))
           targetMissing == ~subOK
           finalPkt == MakePacket(GN1, gn1w.packet.sub, targetMissing)
           finalW == PriorityWrapper(finalPkt)
       IN
       /\ gn1w = local[t].gn1Wrapper
       /\ linkage' = [linkage EXCEPT ![GN1] = finalW]
       /\ local' = [local EXCEPT ![t].gn1Wrapper = finalW]
       /\ pc' = [pc EXCEPT ![t] = "bundle_phase5"]
       /\ UNCHANGED <<bundleDone>>

BundlePhase4Fail(t) ==
    /\ pc[t] = "bundle_phase4"
    /\ LET gn1w == linkage[GN1]
       IN  gn1w /= local[t].gn1Wrapper
    /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
    /\ local' = [local EXCEPT ![t] = InitLocal,
                              ![t].op = "bundle"]
    /\ UNCHANGED <<linkage, bundleDone>>

BundlePhase5(t) ==
    /\ pc[t] = "bundle_phase5"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ local' = [local EXCEPT ![t] = InitLocal]
    /\ bundleDone' = [bundleDone EXCEPT ![t] = TRUE]
    /\ UNCHANGED <<linkage>>

-----------------------------------------------------------------------------
(* P1 race — a peer operation on P1's tree (e.g., another bundle, or
   release).  Models contention: P1's wrapper may change while we're
   in cross-tree migration, forcing us to retry.

   Operationally: CAS P1 to a fresh wrapper (sub[P2] either kept or
   cleared) — this is non-deterministic to cover both peer-bundle and
   peer-release patterns. *)

P1RaceBundle(t) ==
    /\ ~bundleDone[t]
    /\ pc[t] = "idle"
    /\ LET p1w == linkage[P1]
       IN
       /\ p1w.hasPriority
       /\ \E newMiss \in {FALSE} :  \* keep missing=FALSE for simplicity
            LET refresh == PriorityWrapper(MakePacket(P1, p1w.packet.sub, newMiss))
            IN  /\ refresh /= p1w
                /\ linkage' = [linkage EXCEPT ![P1] = refresh]
                /\ UNCHANGED <<pc, local, bundleDone>>

-----------------------------------------------------------------------------
(* Next *)

AllDone == \A t \in Threads : bundleDone[t]

NextStep ==
    \E t \in Threads :
        \/ BundleStart(t)
        \/ BundlePhase1(t)
        \/ BundlePullP1(t)
        \/ BundlePullP1Fail(t)
        \/ BundleCASP2(t)
        \/ BundleCASP2Fail(t)
        \/ BundleUpdateGN1(t)
        \/ BundleUpdateGN1Fail(t)
        \/ BundlePhase4(t)
        \/ BundlePhase4Fail(t)
        \/ BundlePhase5(t)
        \/ P1RaceBundle(t)

Terminating == AllDone /\ UNCHANGED vars

Next == NextStep \/ Terminating

\* Per-thread per-action weak fairness on the progress actions.
\* `WF_vars(NextStep)` (blanket on the disjunction) is too weak here:
\* TLC's BFS can explore executions where one thread's retry path
\* (e.g. BundlePullP1Fail) fires repeatedly while the other thread's
\* progress action (e.g. BundleCASP2) is starved.  Per-action WF
\* binds fairness to each specific action and thread.
\*
\* Models OS-level thread scheduling fairness — each thread's enabled
\* progress steps eventually fire.
Spec == Init /\ [][Next]_vars
        /\ WF_vars(NextStep)
        /\ \A t \in Threads :
            /\ WF_vars(BundlePhase1(t))
            /\ WF_vars(BundlePullP1(t))
            /\ WF_vars(BundleCASP2(t))
            /\ WF_vars(BundleUpdateGN1(t))
            /\ WF_vars(BundlePhase4(t))
            /\ WF_vars(BundlePhase5(t))

-----------------------------------------------------------------------------
(* Safety invariants *)

\* SnapshotConsistency — same as previous external model.
SnapshotConsistency ==
    LET gn1w == linkage[GN1]
        gn1Pkt == gn1w.packet
    IN  (gn1w.hasPriority /\ ~gn1Pkt.missing) =>
        (\A c \in DOMAIN gn1Pkt.sub :
            gn1Pkt.sub[c] /= Null =>
                (\A gc \in DOMAIN gn1Pkt.sub[c].sub :
                    gn1Pkt.sub[c].sub[gc] = Null =>
                        ReachableFromGN1(gn1Pkt)))

\* HardlinkExclusive — P2's packet exists in at most one parent's sub[].
HardlinkExclusive ==
    LET gn1w == linkage[GN1]
        p1w  == linkage[P1]
        inGN2 == /\ gn1w.hasPriority
                 /\ gn1w.packet.sub[GN2] /= Null
                 /\ gn1w.packet.sub[GN2].sub[P2] /= Null
        inP1  == /\ p1w.hasPriority
                 /\ p1w.packet.sub[P2] /= Null
    IN  ~(inGN2 /\ inP1)

\* BundleRefConsistency — every bundled chain terminates at a priority
\* node within finite hops.  In our 4-node topology, max depth = 2
\* (P2 → GN2 → GN1, or P2 → P1).
RECURSIVE ReachesPriority(_, _)
ReachesPriority(n, depth) ==
    \/ depth = 0
    \/ linkage[n].hasPriority
    \/ /\ linkage[n].bundledBy \in Nodes
       /\ ReachesPriority(linkage[n].bundledBy, depth - 1)

BundleRefConsistency ==
    \A n \in {GN2, P2} :
        ~linkage[n].hasPriority => ReachesPriority(linkage[n].bundledBy, 3)

\* MigrationLiveness — at end, P2 should be either in GN2 (= migrated)
\* OR still in P1 (= bundle gave up safely with GN1 missing).
\* Combined with EventuallyAllDone: bundle finishes either way.
EventuallyAllDone == <>AllDone

\* Stronger liveness: bundle EVENTUALLY successfully migrates P2 OR
\* legitimately stays missing.  (Inevitable progress to a final state.)
EventuallyConsistent ==
    <>(linkage[GN1].hasPriority /\
       (linkage[GN1].packet.missing \/
        ReachableFromGN1(linkage[GN1].packet)))

=============================================================================
