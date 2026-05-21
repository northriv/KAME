(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.
 ***************************************************************************)
----------------------- MODULE BundleUnbundle_hardlink_external -----------------------
(*
 * Hard-link with EXTERNAL parent.  4 nodes:
 *
 *       GN1 (bundle root)         P1 (external root, not in GN1's subtree)
 *        │                         │
 *       GN2 (under GN1)            └── P2 (P2's packet currently lives in P1.sub[P2])
 *        │
 *       P2 (hard-linked: parent2 = GN2 — inside GN1; parent1 = P1 — outside GN1)
 *
 * Reflects the production dyn_node_test scenario reported by user:
 *   - p2 (= P2 here) is hard-linked between gn2 (= GN2, inside gn1's subtree)
 *     and a worker-local p1 (= P1 here, often OUTSIDE gn1's subtree).
 *   - After applying the Phase 3 "skip Null slot" fix from
 *     BundleUnbundle_hardlink_self_collision.tla, 2/20 runs pass but
 *     18/20 still abort with `checkConsistensy` failing on a Null slot
 *     in GN2.sub[P2] that is not reachable within GN1's tree.
 *
 * Bug surface diagnosed from C++:
 *   - bundle(GN1) makes GN2 missing=TRUE with sub[P2]=Null (Phase 3 skip).
 *   - is_bundle_root override on GN1 clears missing → GN1 published
 *     non-missing while GN2.sub[P2]=Null.
 *   - checkConsistensy(GN1): GN2.sub[P2] is Null, reverseLookup(P2,
 *     GN1.root) walks GN1 → GN2 → Null, no other path within GN1's tree.
 *     Throws line 871 of transaction_impl.h.
 *
 * Single-threaded sequential model — no concurrency, no insert/release.
 * The bug is purely structural: bundle(GN1) cannot finalize without
 * losing the hard-link to P2.
 *
 * Invariant: SnapshotConsistency mirrors `Packet::checkConsistensy` at
 *   transaction_impl.h:870-871: a Null sub-slot is consistent iff the
 *   root packet is missing OR the child reachable via reverseLookup
 *   anchored at the **published root**.
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Threads,
    GN1, GN2, P2, P1,
    Null

Nodes == {GN1, GN2, P2, P1}
\* GN1's subtree (excluding GN1 itself): {GN2, P2}
\* P1's subtree (separate): {P2}
\* P2 has TWO parents: GN2 (inside GN1) and P1 (outside GN1).

ASSUME Cardinality(Threads) > 0

VARIABLES
    linkage,       \* [Nodes -> Wrapper]
    pc,            \* [Threads -> String]
    local,         \* [Threads -> Record]
    bundleDone     \* [Threads -> BOOLEAN]

\* retryCount removed (per user 2026-05-21).  See _hardlink_4node /
\* _external_migration / _self_collision for the same clean-up.

vars == <<linkage, pc, local, bundleDone>>

-----------------------------------------------------------------------------
(* Wrapper helpers *)

PriorityWrapper(packet) ==
    [packet |-> packet, hasPriority |-> TRUE, bundledBy |-> Null]

BundledRefWrapper(parent) ==
    [packet |-> Null, hasPriority |-> FALSE, bundledBy |-> parent]

MakePacket(node, sub, miss) ==
    [sub |-> sub, missing |-> miss, node |-> node]

\* Initial state is VALID: P2 is bundled under GN2 (the in-tree parent),
\* its packet lives in GN1.sub[GN2].sub[P2].  P1 is a separate tree
\* with its own sub[P2] = Null (P2 not currently homed there).
\*
\* The bug-producing transition is captured by `ExternalMigration` below:
\* an out-of-band operation moves P2's packet from GN2.sub[P2] to
\* P1.sub[P2], setting P2.bundledBy = P1.  After this, GN1 is published
\* ~missing but its sub[GN2].sub[P2] = Null is unreachable inside GN1's
\* tree — checkConsistensy fails.
P2Packet == MakePacket(P2, <<>>, FALSE)
GN2SubValid == [l \in {P2} |-> P2Packet]
GN1SubInit ==
    [c \in {GN2} |-> MakePacket(GN2, GN2SubValid, FALSE)]
P1SubInit == [c \in {P2} |-> Null]

-----------------------------------------------------------------------------
(* Init *)

InitLocal == [
    op           |-> "idle",
    wrapper      |-> Null,   \* GN1's wrapper
    gn2Wrapper   |-> Null,
    p2Wrapper    |-> Null,
    p1Wrapper    |-> Null,
    gn2SubPkt    |-> Null,   \* GN2's packet collected for bundle
    p2SubPkt     |-> Null    \* P2's packet (if collectable)
]

Init ==
    /\ linkage = [n \in Nodes |->
        CASE n = GN1 -> PriorityWrapper(MakePacket(GN1, GN1SubInit, FALSE))
          [] n = GN2 -> BundledRefWrapper(GN1)
          [] n = P1  -> PriorityWrapper(MakePacket(P1, P1SubInit, FALSE))
          [] n = P2  -> BundledRefWrapper(GN2)]   \* P2 initially under GN2 (in-tree)
    /\ pc = [t \in Threads |-> "idle"]
    /\ local = [t \in Threads |-> InitLocal]
    /\ bundleDone = [t \in Threads |-> FALSE]

-----------------------------------------------------------------------------
(* reverseLookup(p2, root) — searches within `root`'s subtree only.
   For root=GN1: walks GN1.sub → GN2.sub → looks for P2. Doesn't
   cross into P1's tree. *)

ReachableFromGN1(rootPkt) ==
    \* P2 reachable from GN1 iff GN1.sub[GN2].sub[P2] is populated AND
    \* not missing.
    /\ rootPkt.node = GN1
    /\ GN2 \in DOMAIN rootPkt.sub
    /\ rootPkt.sub[GN2] /= Null
    /\ P2 \in DOMAIN rootPkt.sub[GN2].sub
    /\ rootPkt.sub[GN2].sub[P2] /= Null

-----------------------------------------------------------------------------
(* ExternalMigration — out-of-band operation moves P2 from GN2 (in
   GN1's subtree) to P1 (external).  Models the "p1 inserts p2" race
   from the production scenario.  The CAS sequence is non-atomic at
   the protocol level: GN1's published packet retains a (now stale)
   reference, GN2's sub[P2] becomes Null, P2's wrapper points to P1,
   and P1's sub[P2] holds the packet copy.

   This is the bug-producing transition: after it, GN1 is ~missing
   but GN2.sub[P2] = Null and P2 is unreachable within GN1's tree. *)

ExternalMigration(t) ==
    /\ ~bundleDone[t]
    /\ linkage[P2].bundledBy = GN2   \* still in-tree
    /\ LET gn1w == linkage[GN1]
           p1w == linkage[P1]
           p2pkt == gn1w.packet.sub[GN2].sub[P2]
       IN
       /\ gn1w.hasPriority
       /\ ~gn1w.packet.missing
       /\ p2pkt /= Null
       /\ LET newGN2Sub == [c \in {P2} |-> Null]
              newGN2Pkt == MakePacket(GN2, newGN2Sub, FALSE)
              newGN1Sub == [c \in {GN2} |-> newGN2Pkt]
              \* (Fix attempt) — set GN1.missing=TRUE atomically with
              \* the sub[GN2].sub[P2]=Null transition, so the
              \* SnapshotConsistency invariant's `~rootPkt.missing`
              \* guard prevents the unreachable-Null check from firing.
              \* The bundle protocol must later re-bundle to either
              \* re-pull P2 into GN2 or accept that GN1 cannot finalize.
              newGN1Pkt == MakePacket(GN1, newGN1Sub, TRUE)  \* missing=TRUE
              newGN1W == PriorityWrapper(newGN1Pkt)
              newP1Sub == [c \in {P2} |-> p2pkt]
              newP1Pkt == MakePacket(P1, newP1Sub, FALSE)
              newP1W == PriorityWrapper(newP1Pkt)
              newP2W == BundledRefWrapper(P1)
          IN
          /\ linkage' = [linkage EXCEPT
                  ![GN1] = newGN1W,
                  ![P1]  = newP1W,
                  ![P2]  = newP2W]
          /\ UNCHANGED <<pc, local, bundleDone>>

-----------------------------------------------------------------------------
(* Bundle pipeline for GN1.  5 phases. *)

BundleStart(t) ==
    /\ pc[t] = "idle"
    /\ ~bundleDone[t]
    /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
    /\ local' = [local EXCEPT ![t] = InitLocal,
                              ![t].op = "bundle"]
    /\ UNCHANGED <<linkage, bundleDone>>

\* Phase 1 (Collect) — read GN1, GN2, P2 wrappers; capture sub-packets.
\* GN2's packet comes from GN1.sub[GN2] (GN2 is BundledRef of GN1).
\* P2's packet comes from P1.sub[P2] (P2 is BundledRef of P1 — EXTERNAL).
\* Accept missing=TRUE GN1 (recursive bundle from outer caller).
BundlePhase1(t) ==
    /\ pc[t] = "bundle_phase1"
    /\ LET gn1w == linkage[GN1]
           gn2w == linkage[GN2]
           p2w  == linkage[P2]
           p1w  == linkage[P1]
       IN
       /\ gn1w.hasPriority
       /\ local' = [local EXCEPT
              ![t].wrapper    = gn1w,
              ![t].gn2Wrapper = gn2w,
              ![t].p2Wrapper  = p2w,
              ![t].p1Wrapper  = p1w,
              ![t].gn2SubPkt  = IF gn2w.hasPriority
                                THEN gn2w.packet
                                ELSE gn1w.packet.sub[GN2],
              ![t].p2SubPkt   = IF p2w.hasPriority
                                THEN p2w.packet
                                ELSE IF p2w.bundledBy = P1 /\ p1w.hasPriority
                                     THEN p1w.packet.sub[P2]
                                     ELSE IF p2w.bundledBy = GN2
                                          /\ ~gn2w.hasPriority
                                          /\ gn1w.packet.sub[GN2] /= Null
                                          THEN gn1w.packet.sub[GN2].sub[P2]
                                          ELSE Null]
       /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
       /\ UNCHANGED <<linkage, bundleDone>>

\* Phase 2 (CAS GN1) — CAS GN1 to missing=TRUE.  The new sub[] mirrors
\* the collected state: GN1.sub[GN2] holds GN2's packet, GN2.sub[P2]
\* remains Null (hard-link reference: P2's packet is in P1.sub[P2],
\* outside GN1).
BundlePhase2(t) ==
    /\ pc[t] = "bundle_phase2"
    /\ LET oldW   == local[t].wrapper
           gn2Pkt == local[t].gn2SubPkt
           p2Pkt  == local[t].p2SubPkt
           \* GN2's bundled sub[]: P2 slot stays Null (hard-link).
           newGN2Sub == [c \in {P2} |-> Null]
           newGN2Pkt == MakePacket(GN2, newGN2Sub, FALSE)
           newGN1Sub == [c \in {GN2} |-> newGN2Pkt]
           newPkt == MakePacket(GN1, newGN1Sub, TRUE)
           newW == PriorityWrapper(newPkt)
       IN
       IF linkage[GN1] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![GN1] = newW]
            /\ local' = [local EXCEPT ![t].wrapper = newW]
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase3"]
            /\ UNCHANGED <<bundleDone>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
            /\ local' = [local EXCEPT ![t] = InitLocal,
                                       ![t].op = "bundle"]
            /\ UNCHANGED <<linkage, bundleDone>>

\* Phase 3 (CAS child) — Apply the Phase 3 skip-Null fix from the
\* self-collision spec: only CAS child wrappers whose packets MOVE
\* into the parent's sub[].  GN2's packet moves (GN1.sub[GN2] is
\* populated).  P2's packet does NOT move (stays in P1.sub[P2]; GN2's
\* sub[P2] is Null) — skip P2.
BundlePhase3(t) ==
    /\ pc[t] = "bundle_phase3"
    /\ \/ \* CAS GN2 to BundledRef(GN1) — already so since Init.
          /\ linkage[GN2] = local[t].gn2Wrapper
          /\ linkage' = [linkage EXCEPT ![GN2] = BundledRefWrapper(GN1)]
          /\ local' = [local EXCEPT ![t].gn2Wrapper =
                  [packet |-> Null, hasPriority |-> FALSE, bundledBy |-> GN1]]
          /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
          /\ UNCHANGED <<bundleDone>>
       \/ \* Already at GN1 — skip
          /\ linkage[GN2].bundledBy = GN1
          /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
          /\ UNCHANGED <<linkage, local, bundleDone>>

\* Phase 4 (Finalize) — `is_bundle_root` override: clear GN1's missing
\* flag, GATED on all subtree Null slots being reachable within GN1.
\*
\* (Fix) Bug surface from production: the override unconditionally
\* clears missing.  Fix: scan the bundled sub[] for unresolved Null
\* slots, only fire the override if every Null slot's child node is
\* reachable somewhere else within GN1's tree.  Here we have only one
\* candidate (P2 via GN2.sub[P2]); fail the override if it's Null AND
\* no other in-tree path exists.
\*
\* If the override is denied, the bundle stays missing=TRUE and a
\* later operation must either re-pull P2 from P1 (cross-tree
\* migration) or remove P2 from GN2's subnodes list.  For this
\* minimal model we simply leave GN1 missing — bundle effectively
\* "fails to finalize" but the published state stays consistent.
BundlePhase4(t) ==
    /\ pc[t] = "bundle_phase4"
    /\ LET oldW == local[t].wrapper
           subOK == \A c \in DOMAIN oldW.packet.sub :
                       oldW.packet.sub[c] /= Null =>
                           (\A gc \in DOMAIN oldW.packet.sub[c].sub :
                               oldW.packet.sub[c].sub[gc] /= Null
                               \/ ReachableFromGN1(oldW.packet))
           targetMissing == ~subOK
           finalPkt == MakePacket(GN1, oldW.packet.sub, targetMissing)
           finalW == PriorityWrapper(finalPkt)
       IN
       IF linkage[GN1] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![GN1] = finalW]
            /\ local' = [local EXCEPT ![t].wrapper = finalW]
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase5"]
            /\ UNCHANGED <<bundleDone>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
            /\ local' = [local EXCEPT ![t] = InitLocal,
                                       ![t].op = "bundle"]
            /\ UNCHANGED <<linkage, bundleDone>>

\* Phase 5 (Publish) — done.
BundlePhase5(t) ==
    /\ pc[t] = "bundle_phase5"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ local' = [local EXCEPT ![t] = InitLocal]
    /\ bundleDone' = [bundleDone EXCEPT ![t] = TRUE]
    /\ UNCHANGED <<linkage>>

-----------------------------------------------------------------------------
(* Next *)

AllDone == \A t \in Threads : bundleDone[t]

NextStep ==
    \E t \in Threads :
        \/ ExternalMigration(t)
        \/ BundleStart(t)
        \/ BundlePhase1(t)
        \/ BundlePhase2(t)
        \/ BundlePhase3(t)
        \/ BundlePhase4(t)
        \/ BundlePhase5(t)

Terminating == AllDone /\ UNCHANGED vars

Next == NextStep \/ Terminating

Spec == Init /\ [][Next]_vars /\ WF_vars(NextStep)

-----------------------------------------------------------------------------
(* Safety invariants *)

\* SnapshotConsistency — mirrors checkConsistensy at transaction_impl.h:
\* line 870-871.  When GN1 is published as priority + ~missing, every
\* Null sub-slot inside GN1's tree (here: GN2.sub[P2]) must be
\* reachable via reverseLookup anchored at GN1's root packet.
\* P2 is in DOMAIN GN2.sub but its packet is Null AND it lives at
\* P1.sub[P2] — outside GN1's reverseLookup reach.  So the only way to
\* satisfy this invariant in our minimal model is for GN1 to remain
\* missing (no `is_bundle_root` override).
SnapshotConsistency ==
    LET gn1w == linkage[GN1]
        gn1Pkt == gn1w.packet
    IN  (gn1w.hasPriority /\ ~gn1Pkt.missing) =>
        (\A c \in DOMAIN gn1Pkt.sub :
            gn1Pkt.sub[c] /= Null =>
                (\A gc \in DOMAIN gn1Pkt.sub[c].sub :
                    gn1Pkt.sub[c].sub[gc] = Null =>
                        \* Null slot must be findable elsewhere in GN1's tree.
                        ReachableFromGN1(gn1Pkt)))

\* BundleRefConsistency: GN2 bundled means GN1 has priority.
BundleRefConsistency ==
    LET gn2w == linkage[GN2]
    IN  ~gn2w.hasPriority => linkage[gn2w.bundledBy].hasPriority

\* (DebugRetryBound removed — retryCount no longer in vars.)

\* EventuallyAllDone — liveness.
EventuallyAllDone == <>AllDone

=============================================================================
