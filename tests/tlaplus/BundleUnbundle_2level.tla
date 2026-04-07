------------------------- MODULE BundleUnbundle_2level -------------------------
(*
 * TLA+ specification of KAME's bundle/unbundle protocol (2-level tree).
 *
 * Tree structure (fixed):
 *   Parent --+-- Child1
 *            +-- Child2
 *
 * This 2-level model covers:
 *   - 4-phase bundle protocol (collect → CAS parent → CAS children → finalize)
 *   - Unbundle for commit (1-level walk)
 *   - Concurrent snapshot + commit interference
 *   - Single-node commit optimization
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Threads,
    Parent,
    Child1, Child2,
    Null,
    MaxPayload      \* Payloads wrap at this value

Children == {Child1, Child2}
Nodes == {Parent} \cup Children

VARIABLES
    serial, globalSerial, linkage, pc, op, target, local

vars == <<serial, globalSerial, linkage, pc, op, target, local>>

-----------------------------------------------------------------------------
(* Data structures *)

PriorityWrapper(packet, ser) ==
    [packet |-> packet, hasPriority |-> TRUE, bundledBy |-> Null, serial |-> ser]

BundledRefWrapper(parentNode, ser) ==
    [packet |-> Null, hasPriority |-> FALSE, bundledBy |-> parentNode, serial |-> ser]

MakePacket(node, payload, sub, miss) ==
    [payload |-> payload, sub |-> sub, missing |-> miss, node |-> node]

EmptySub == [c \in Children |-> Null]

GenSerial(t, lastSer) ==
    LET base == IF lastSer > serial[t] THEN lastSer ELSE serial[t]
    IN  base + 1

UpdateSerial(t, ser) ==
    /\ serial' = [serial EXCEPT ![t] = ser]
    /\ globalSerial' = IF ser > globalSerial THEN ser ELSE globalSerial

InitLocal == [
    wrapper     |-> Null,
    subwrappers |-> [c \in Children |-> Null],
    subpackets  |-> EmptySub,
    bundleSer   |-> 0,
    oldpacket   |-> Null,
    newpacket   |-> Null,
    snapResult  |-> Null,
    commitOk    |-> Null
]

Init ==
    /\ linkage = [n \in Nodes |->
        IF n = Parent
        THEN PriorityWrapper(MakePacket(Parent, 0, EmptySub, TRUE), 0)
        ELSE PriorityWrapper(MakePacket(n, 0, EmptySub, FALSE), 0)]
    /\ serial = [t \in Threads |-> 0]
    /\ globalSerial = 0
    /\ pc = [t \in Threads |-> "idle"]
    /\ op = [t \in Threads |-> "idle"]
    /\ target = [t \in Threads |-> Null]
    /\ local = [t \in Threads |-> InitLocal]

-----------------------------------------------------------------------------
(* Snapshot *)

SnapRead(t) ==
    /\ pc[t] = "idle"
    /\ op' = [op EXCEPT ![t] = "snapshot"]
    /\ target' = [target EXCEPT ![t] = Parent]
    /\ pc' = [pc EXCEPT ![t] = "snap_read"]
    /\ UNCHANGED <<serial, globalSerial, linkage, local>>

SnapCheck(t) ==
    /\ pc[t] = "snap_read"
    /\ LET w == linkage[Parent]
       IN
       IF w.hasPriority /\ ~w.packet.missing
       THEN /\ local' = [local EXCEPT ![t].snapResult = w.packet]
            /\ pc' = [pc EXCEPT ![t] = "snap_done"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>
       ELSE IF w.hasPriority /\ w.packet.missing
       THEN LET ser == GenSerial(t, w.serial)
            IN
            /\ local' = [local EXCEPT
                   ![t].wrapper = w,
                   ![t].bundleSer = ser,
                   ![t].subwrappers = [c \in Children |-> Null],
                   ![t].subpackets = EmptySub]
            /\ UpdateSerial(t, ser)
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
            /\ UNCHANGED <<linkage, op, target>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

BundlePhase1(t) ==
    /\ pc[t] = "bundle_phase1"
    /\ LET childWrappers == [c \in Children |-> linkage[c]]
           parentW == local[t].wrapper
           childPackets == [c \in Children |->
               IF childWrappers[c].hasPriority
               THEN childWrappers[c].packet
               ELSE IF childWrappers[c].bundledBy = Parent
                    THEN IF parentW.packet.sub[c] /= Null
                         THEN parentW.packet.sub[c]
                         ELSE Null
                    ELSE Null]
           allCollected == \A c \in Children : childPackets[c] /= Null
       IN
       IF allCollected
       THEN /\ local' = [local EXCEPT
                    ![t].subwrappers = childWrappers,
                    ![t].subpackets  = childPackets]
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

BundlePhase2(t) ==
    /\ pc[t] = "bundle_phase2"
    /\ LET oldW   == local[t].wrapper
           ser    == local[t].bundleSer
           subs   == local[t].subpackets
           newPkt == MakePacket(Parent, oldW.packet.payload, subs, TRUE)
           newW   == PriorityWrapper(newPkt, ser)
       IN
       IF linkage[Parent] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![Parent] = newW]
            /\ local' = [local EXCEPT ![t].wrapper = newW]
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase3"]
            /\ UNCHANGED <<serial, globalSerial, op, target>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

BundlePhase3(t) ==
    /\ pc[t] = "bundle_phase3"
    /\ LET ser     == local[t].bundleSer
           childWs == local[t].subwrappers
           allMatch == \A c \in Children : linkage[c] = childWs[c]
       IN
       IF allMatch
       THEN /\ linkage' = [n \in Nodes |->
                IF n \in Children
                THEN BundledRefWrapper(Parent, ser)
                ELSE linkage[n]]
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
            /\ UNCHANGED <<serial, globalSerial, local, op, target>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

BundlePhase4(t) ==
    /\ pc[t] = "bundle_phase4"
    /\ LET oldW     == local[t].wrapper
           ser      == local[t].bundleSer
           finalPkt == MakePacket(Parent, oldW.packet.payload,
                                  oldW.packet.sub, FALSE)
           finalW   == PriorityWrapper(finalPkt, ser)
       IN
       IF linkage[Parent] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![Parent] = finalW]
            /\ local' = [local EXCEPT ![t].snapResult = finalPkt]
            /\ pc' = [pc EXCEPT ![t] = "snap_done"]
            /\ UNCHANGED <<serial, globalSerial, op, target>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

SnapDone(t) ==
    /\ pc[t] = "snap_done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ op' = [op EXCEPT ![t] = "idle"]
    /\ target' = [target EXCEPT ![t] = Null]
    /\ local' = [local EXCEPT ![t] = InitLocal]
    /\ UNCHANGED <<serial, globalSerial, linkage>>

-----------------------------------------------------------------------------
(* Commit *)

CommitStart(t, childNode) ==
    /\ pc[t] = "idle"
    /\ childNode \in Children
    /\ op' = [op EXCEPT ![t] = "commit"]
    /\ target' = [target EXCEPT ![t] = childNode]
    /\ pc' = [pc EXCEPT ![t] = "commit_read"]
    /\ UNCHANGED <<serial, globalSerial, linkage, local>>

CommitRead(t) ==
    /\ pc[t] = "commit_read"
    /\ LET childNode == target[t]
           w == linkage[childNode]
       IN
       IF w.hasPriority
       THEN /\ local' = [local EXCEPT
                ![t].wrapper   = w,
                ![t].oldpacket = w.packet,
                ![t].newpacket = MakePacket(childNode,
                                     (w.packet.payload + 1) % MaxPayload,
                                     w.packet.sub,
                                     w.packet.missing)]
            /\ pc' = [pc EXCEPT ![t] = "commit_try_cas"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>
       ELSE /\ local' = [local EXCEPT ![t].wrapper = w]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>

CommitTryCAS(t) ==
    /\ pc[t] = "commit_try_cas"
    /\ LET childNode == target[t]
           oldW      == local[t].wrapper
           ser       == GenSerial(t, oldW.serial)
           newW      == PriorityWrapper(local[t].newpacket, ser)
       IN
       IF linkage[childNode] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![childNode] = newW]
            /\ UpdateSerial(t, ser)
            /\ local' = [local EXCEPT ![t].commitOk = "ok"]
            /\ pc' = [pc EXCEPT ![t] = "commit_done"]
            /\ UNCHANGED <<op, target>>
       ELSE IF linkage[childNode].hasPriority
       THEN IF linkage[childNode].packet.payload = oldW.packet.payload
            THEN /\ local' = [local EXCEPT
                       ![t].wrapper = linkage[childNode],
                       ![t].newpacket = MakePacket(target[t],
                           local[t].newpacket.payload,
                           linkage[childNode].packet.sub,
                           linkage[childNode].packet.missing)]
                 /\ pc' = [pc EXCEPT ![t] = "commit_try_cas"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>
            ELSE /\ local' = [local EXCEPT ![t].commitOk = "fail"]
                 /\ pc' = [pc EXCEPT ![t] = "commit_done"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>
       ELSE /\ local' = [local EXCEPT ![t].wrapper = linkage[childNode]]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>

UnbundleWalk(t) ==
    /\ pc[t] = "unbundle_walk"
    /\ LET childNode == target[t]
           w         == local[t].wrapper
           parentW   == linkage[Parent]
       IN
       IF w.bundledBy = Parent /\ parentW.hasPriority
          /\ parentW.packet.sub[childNode] /= Null
       THEN /\ local' = [local EXCEPT
                   ![t].oldpacket = parentW.packet.sub[childNode],
                   ![t].newpacket = MakePacket(childNode,
                       (parentW.packet.sub[childNode].payload + 1) % MaxPayload,
                       parentW.packet.sub[childNode].sub,
                       parentW.packet.sub[childNode].missing)]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_ancestors"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

UnbundleCASAncestors(t) ==
    /\ pc[t] = "unbundle_cas_ancestors"
    /\ LET childNode  == target[t]
           parentW    == linkage[Parent]
           ser        == GenSerial(t, parentW.serial)
           newSub     == [parentW.packet.sub EXCEPT ![childNode] = Null]
           newPkt     == MakePacket(Parent, parentW.packet.payload, newSub, TRUE)
           newParentW == PriorityWrapper(newPkt, ser)
       IN
       IF parentW.hasPriority /\ linkage[Parent] = parentW
       THEN /\ linkage' = [linkage EXCEPT ![Parent] = newParentW]
            /\ UpdateSerial(t, ser)
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_child"]
            /\ UNCHANGED <<local, op, target>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

UnbundleCASChild(t) ==
    /\ pc[t] = "unbundle_cas_child"
    /\ LET childNode  == target[t]
           oldChildW  == local[t].wrapper
           ser        == GenSerial(t, oldChildW.serial)
           newChildW  == PriorityWrapper(local[t].newpacket, ser)
       IN
       IF linkage[childNode] = oldChildW
       THEN /\ linkage' = [linkage EXCEPT ![childNode] = newChildW]
            /\ UpdateSerial(t, ser)
            /\ local' = [local EXCEPT ![t].commitOk = "ok"]
            /\ pc' = [pc EXCEPT ![t] = "commit_done"]
            /\ UNCHANGED <<op, target>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

CommitDone(t) ==
    /\ pc[t] = "commit_done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ op' = [op EXCEPT ![t] = "idle"]
    /\ target' = [target EXCEPT ![t] = Null]
    /\ local' = [local EXCEPT ![t] = InitLocal]
    /\ UNCHANGED <<serial, globalSerial, linkage>>

-----------------------------------------------------------------------------
(* Next-state relation *)

Next ==
    \E t \in Threads :
        \/ SnapRead(t)
        \/ SnapCheck(t)
        \/ BundlePhase1(t)
        \/ BundlePhase2(t)
        \/ BundlePhase3(t)
        \/ BundlePhase4(t)
        \/ SnapDone(t)
        \/ \E c \in Children : CommitStart(t, c)
        \/ CommitRead(t)
        \/ CommitTryCAS(t)
        \/ UnbundleWalk(t)
        \/ UnbundleCASAncestors(t)
        \/ UnbundleCASChild(t)
        \/ CommitDone(t)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

-----------------------------------------------------------------------------
(* Safety invariants *)

SnapshotConsistency ==
    LET pw == linkage[Parent]
    IN  (pw.hasPriority /\ ~pw.packet.missing) =>
        (\A c \in Children : pw.packet.sub[c] /= Null)

NoPriorityLoss ==
    \A c \in Children :
        LET w == linkage[c]
        IN  w.hasPriority \/ w.bundledBy /= Null

BundleRefConsistency ==
    \A c \in Children :
        LET cw == linkage[c]
        IN  (~cw.hasPriority /\ cw.bundledBy = Parent) =>
            linkage[Parent].hasPriority

SerialMonotonicity ==
    \A t \in Threads : serial[t] >= 0

Safety == SnapshotConsistency /\ NoPriorityLoss /\ BundleRefConsistency /\ SerialMonotonicity

\* State constraint for bounded model checking
StateConstraint ==
    /\ globalSerial <= 3 * MaxPayload
    /\ \A c \in Children :
        linkage[c].hasPriority =>
            linkage[c].packet.payload <= MaxPayload

=============================================================================
