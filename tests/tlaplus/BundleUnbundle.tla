--------------------------- MODULE BundleUnbundle ---------------------------
(*
 * TLA+ specification of KAME's bundle/unbundle protocol (3-level tree).
 *
 * Tree structure (fixed):
 *   Grand --+-- Parent --+-- Child1
 *                         +-- Child2
 *
 * This 3-level hierarchy exercises:
 *   - Recursive bundling: snapshot(Grand) bundles Parent, which bundles Children
 *   - Multi-level unbundle: commit(Child) when bundled 2 levels deep
 *   - snapshotSupernode() walking up through bundledBy chain
 *
 * Serials use modular arithmetic to keep the state space finite.
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Threads,
    Grand,          \* Grandparent node
    Parent,         \* Parent node (child of Grand)
    Child1, Child2, \* Leaf nodes (children of Parent)
    Null,
    MaxPayload      \* Payloads wrap at this value (e.g. 2)

GrandChildren == {Parent}         \* Grand's children
ParentChildren == {Child1, Child2} \* Parent's children
AllNodes == {Grand, Parent, Child1, Child2}
LeafNodes == {Child1, Child2}
InnerNodes == {Grand, Parent}

\* Which node is the parent of which
ParentOf(n) ==
    IF n \in ParentChildren THEN Parent
    ELSE IF n = Parent THEN Grand
    ELSE Null

\* Children of a given node
ChildrenOf(n) ==
    IF n = Grand THEN GrandChildren
    ELSE IF n = Parent THEN ParentChildren
    ELSE {}

VARIABLES
    serial, globalSerial, linkage, pc, op, target, local

vars == <<serial, globalSerial, linkage, pc, op, target, local>>

-----------------------------------------------------------------------------
(* Modular serial arithmetic *)

SerialSucc(s) == s + 1

\* Advance past lastSer then increment (Lamport step)
GenSerial(t, lastSer) ==
    LET base == IF lastSer > serial[t] THEN lastSer ELSE serial[t]
    IN  base + 1

UpdateSerial(t, ser) ==
    /\ serial' = [serial EXCEPT ![t] = ser]
    /\ globalSerial' = ser  \* simplified: just track latest

-----------------------------------------------------------------------------
(* Data structures *)

\* A Packet holds payload version, sub-packets for children, and missing flag.
\* sub is a function [ChildrenOf(node) -> Packet or Null].
MakePacket(node, payload, sub, miss) ==
    [payload |-> payload, sub |-> sub, missing |-> miss, node |-> node]

\* PacketWrapper: wraps a Packet with priority/bundle metadata
PriorityWrapper(packet, ser) ==
    [packet |-> packet, hasPriority |-> TRUE, bundledBy |-> Null, serial |-> ser]

BundledRefWrapper(parentNode, ser) ==
    [packet |-> Null, hasPriority |-> FALSE, bundledBy |-> parentNode, serial |-> ser]

\* Empty sub-packet maps for each inner node
EmptySubFor(n) ==
    IF n = Grand THEN [c \in GrandChildren |-> Null]
    ELSE IF n = Parent THEN [c \in ParentChildren |-> Null]
    ELSE [c \in {} |-> Null]

\* Initial local state
InitLocal == [
    wrapper     |-> Null,
    subwrappers |-> Null,      \* function: children -> wrapper
    subpackets  |-> Null,      \* function: children -> packet
    bundleSer   |-> 0,
    bundleNode  |-> Null,      \* which node we're bundling
    oldpacket   |-> Null,
    newpacket   |-> Null,
    snapResult  |-> Null,
    commitOk    |-> Null
]

Init ==
    /\ linkage = [n \in AllNodes |->
        IF n = Grand
        THEN PriorityWrapper(
                MakePacket(Grand, 0, [c \in GrandChildren |-> Null], TRUE), 0)
        ELSE IF n = Parent
        THEN PriorityWrapper(
                MakePacket(Parent, 0, [c \in ParentChildren |-> Null], TRUE), 0)
        ELSE PriorityWrapper(
                MakePacket(n, 0, EmptySubFor(n), FALSE), 0)]
    /\ serial = [t \in Threads |-> 0]
    /\ globalSerial = 0
    /\ pc = [t \in Threads |-> "idle"]
    /\ op = [t \in Threads |-> "idle"]
    /\ target = [t \in Threads |-> Null]
    /\ local = [t \in Threads |-> InitLocal]

-----------------------------------------------------------------------------
(* Snapshot: read node, bundle if missing *)

SnapRead(t, node) ==
    /\ pc[t] = "idle"
    /\ node \in InnerNodes
    /\ op' = [op EXCEPT ![t] = "snapshot"]
    /\ target' = [target EXCEPT ![t] = node]
    /\ pc' = [pc EXCEPT ![t] = "snap_check"]
    /\ UNCHANGED <<serial, globalSerial, linkage, local>>

SnapCheck(t) ==
    /\ pc[t] = "snap_check"
    /\ LET node == target[t]
           w == linkage[node]
       IN
       IF w.hasPriority /\ ~w.packet.missing
       THEN \* Fast path: complete
            /\ local' = [local EXCEPT ![t].snapResult = w.packet]
            /\ pc' = [pc EXCEPT ![t] = "snap_done"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>
       ELSE IF w.hasPriority /\ w.packet.missing
       THEN \* Need to bundle this node
            LET ser == GenSerial(t, w.serial)
                children == ChildrenOf(node)
            IN
            /\ local' = [local EXCEPT
                   ![t].wrapper    = w,
                   ![t].bundleSer  = ser,
                   ![t].bundleNode = node,
                   ![t].subwrappers = [c \in children |-> Null],
                   ![t].subpackets  = [c \in children |-> Null]]
            /\ UpdateSerial(t, ser)
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
            /\ UNCHANGED <<linkage, op, target>>
       ELSE \* Node is bundled — need to unbundle first (snapshot from bundled state)
            \* For simplicity, just retry (the real code calls snapshotSupernode)
            /\ pc' = [pc EXCEPT ![t] = "snap_check"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

-----------------------------------------------------------------------------
(* Bundle: 4-phase protocol *)

\* Phase 1: Collect sub-packets from children.
\* If a child is bundled elsewhere, we can't collect — restart.
\* If a child is bundled HERE (same node), use the existing sub-packet.
\* If a child has priority but is missing, we need to recursively bundle it first.
BundlePhase1(t) ==
    /\ pc[t] = "bundle_phase1"
    /\ LET node     == local[t].bundleNode
           children  == ChildrenOf(node)
           parentW   == local[t].wrapper
           childWs   == [c \in children |-> linkage[c]]
           childPkts == [c \in children |->
               IF childWs[c].hasPriority
               THEN IF ~childWs[c].packet.missing
                    THEN childWs[c].packet   \* child is complete
                    ELSE Null                \* child is missing — would need recursive bundle
               ELSE IF childWs[c].bundledBy = node
                    THEN parentW.packet.sub[c] \* already bundled here
                    ELSE Null]               \* bundled elsewhere
           allCollected == \A c \in children : childPkts[c] /= Null
       IN
       IF allCollected
       THEN /\ local' = [local EXCEPT
                    ![t].subwrappers = childWs,
                    ![t].subpackets  = childPkts]
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>
       ELSE \* Can't collect all — restart
            /\ pc' = [pc EXCEPT ![t] = "snap_check"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

\* Phase 2: CAS node's linkage with new packet (still missing=TRUE)
BundlePhase2(t) ==
    /\ pc[t] = "bundle_phase2"
    /\ LET node   == local[t].bundleNode
           oldW   == local[t].wrapper
           ser    == local[t].bundleSer
           subs   == local[t].subpackets
           newPkt == MakePacket(node, oldW.packet.payload, subs, TRUE)
           newW   == PriorityWrapper(newPkt, ser)
       IN
       IF linkage[node] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![node] = newW]
            /\ local' = [local EXCEPT ![t].wrapper = newW]
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase3"]
            /\ UNCHANGED <<serial, globalSerial, op, target>>
       ELSE \* Disturbed
            /\ pc' = [pc EXCEPT ![t] = "snap_check"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

\* Phase 3: CAS each child to bundled-ref pointing to this node
BundlePhase3(t) ==
    /\ pc[t] = "bundle_phase3"
    /\ LET node     == local[t].bundleNode
           ser      == local[t].bundleSer
           children == ChildrenOf(node)
           childWs  == local[t].subwrappers
           allMatch == \A c \in children : linkage[c] = childWs[c]
       IN
       IF allMatch
       THEN /\ linkage' = [n \in AllNodes |->
                IF n \in children
                THEN BundledRefWrapper(node, ser)
                ELSE linkage[n]]
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
            /\ UNCHANGED <<serial, globalSerial, local, op, target>>
       ELSE \* Child modified — restart from phase 1
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

\* Phase 4: Finalize — set missing=FALSE, CAS
BundlePhase4(t) ==
    /\ pc[t] = "bundle_phase4"
    /\ LET node     == local[t].bundleNode
           oldW     == local[t].wrapper
           ser      == local[t].bundleSer
           finalPkt == MakePacket(node, oldW.packet.payload,
                                  oldW.packet.sub, FALSE)
           finalW   == PriorityWrapper(finalPkt, ser)
       IN
       IF linkage[node] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![node] = finalW]
            /\ local' = [local EXCEPT ![t].snapResult = finalPkt]
            /\ pc' = [pc EXCEPT ![t] = "snap_done"]
            /\ UNCHANGED <<serial, globalSerial, op, target>>
       ELSE \* Disturbed
            /\ pc' = [pc EXCEPT ![t] = "snap_check"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

SnapDone(t) ==
    /\ pc[t] = "snap_done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ op' = [op EXCEPT ![t] = "idle"]
    /\ target' = [target EXCEPT ![t] = Null]
    /\ local' = [local EXCEPT ![t] = InitLocal]
    /\ UNCHANGED <<serial, globalSerial, linkage>>

-----------------------------------------------------------------------------
(* Commit on a leaf or inner node *)

CommitStart(t, node) ==
    /\ pc[t] = "idle"
    /\ node \in AllNodes \ {Grand}  \* Can commit to Parent or Children
    /\ op' = [op EXCEPT ![t] = "commit"]
    /\ target' = [target EXCEPT ![t] = node]
    /\ pc' = [pc EXCEPT ![t] = "commit_read"]
    /\ UNCHANGED <<serial, globalSerial, linkage, local>>

CommitRead(t) ==
    /\ pc[t] = "commit_read"
    /\ LET node == target[t]
           w == linkage[node]
       IN
       IF w.hasPriority
       THEN /\ local' = [local EXCEPT
                ![t].wrapper   = w,
                ![t].oldpacket = w.packet,
                ![t].newpacket = MakePacket(node,
                                     (w.packet.payload + 1) % MaxPayload,
                                     w.packet.sub,
                                     w.packet.missing)]
            /\ pc' = [pc EXCEPT ![t] = "commit_try_cas"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>
       ELSE \* Bundled — need unbundle
            /\ local' = [local EXCEPT ![t].wrapper = w]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>

CommitTryCAS(t) ==
    /\ pc[t] = "commit_try_cas"
    /\ LET node == target[t]
           oldW == local[t].wrapper
           ser  == GenSerial(t, oldW.serial)
           newW == PriorityWrapper(local[t].newpacket, ser)
       IN
       IF linkage[node] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![node] = newW]
            /\ UpdateSerial(t, ser)
            /\ local' = [local EXCEPT ![t].commitOk = "ok"]
            /\ pc' = [pc EXCEPT ![t] = "commit_done"]
            /\ UNCHANGED <<op, target>>
       ELSE IF linkage[node].hasPriority
       THEN IF linkage[node].packet.payload = oldW.packet.payload
            THEN \* Single-node optimization: adopt new children
                 /\ local' = [local EXCEPT
                       ![t].wrapper = linkage[node],
                       ![t].newpacket = MakePacket(target[t],
                           local[t].newpacket.payload,
                           linkage[node].packet.sub,
                           linkage[node].packet.missing)]
                 /\ pc' = [pc EXCEPT ![t] = "commit_try_cas"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>
            ELSE \* True conflict
                 /\ local' = [local EXCEPT ![t].commitOk = "fail"]
                 /\ pc' = [pc EXCEPT ![t] = "commit_done"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>
       ELSE \* Got bundled — unbundle
            /\ local' = [local EXCEPT ![t].wrapper = linkage[node]]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>

-----------------------------------------------------------------------------
(* Unbundle — recursive walk up the bundledBy chain *)

\* Walk up: find the top-level ancestor that has priority,
\* then extract our node's packet from the bundle.
\* This models snapshotSupernode() walking up 1 or 2 levels.
UnbundleWalk(t) ==
    /\ pc[t] = "unbundle_walk"
    /\ LET node    == target[t]
           w       == local[t].wrapper
           parent  == w.bundledBy
       IN
       IF parent /= Null
       THEN LET parentW == linkage[parent]
            IN
            IF parentW.hasPriority
            THEN \* Parent has priority — extract sub-packet
                 IF parentW.packet.sub[node] /= Null
                 THEN /\ local' = [local EXCEPT
                           ![t].oldpacket = parentW.packet.sub[node],
                           ![t].newpacket = MakePacket(node,
                               (parentW.packet.sub[node].payload + 1) % MaxPayload,
                               parentW.packet.sub[node].sub,
                               parentW.packet.sub[node].missing)]
                      /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_ancestor"]
                      /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>
                 ELSE \* Sub-packet is Null (not yet bundled) — retry
                      /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                      /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>
            ELSE \* Parent itself is bundled — need to walk up further
                 \* The grandparent must hold the full bundle
                 LET grandparent == parentW.bundledBy
                 IN
                 IF grandparent /= Null
                 THEN LET gpW == linkage[grandparent]
                      IN
                      IF gpW.hasPriority /\ gpW.packet.sub[parent] /= Null
                         /\ gpW.packet.sub[parent].sub[node] /= Null
                      THEN \* Found 2 levels up — extract
                           LET subPkt == gpW.packet.sub[parent].sub[node]
                           IN
                           /\ local' = [local EXCEPT
                                 ![t].oldpacket = subPkt,
                                 ![t].newpacket = MakePacket(node,
                                     (subPkt.payload + 1) % MaxPayload,
                                     subPkt.sub,
                                     subPkt.missing)]
                           /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_gp"]
                           /\ UNCHANGED <<serial, globalSerial, linkage, op, target>>
                      ELSE \* Can't find — retry
                           /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                           /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>
                 ELSE \* No grandparent — retry
                      /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                      /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>
       ELSE \* bundledBy is Null — retry
            /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

\* CAS ancestor (1 level up): mark child's slot as Null in parent, restore child
UnbundleCASAncestor(t) ==
    /\ pc[t] = "unbundle_cas_ancestor"
    /\ LET node     == target[t]
           parent   == local[t].wrapper.bundledBy
           parentW  == linkage[parent]
           ser      == GenSerial(t, parentW.serial)
           newSub   == [parentW.packet.sub EXCEPT ![node] = Null]
           newPkt   == MakePacket(parent, parentW.packet.payload, newSub, TRUE)
           newPW    == PriorityWrapper(newPkt, ser)
       IN
       IF parentW.hasPriority /\ linkage[parent] = parentW
       THEN /\ linkage' = [linkage EXCEPT ![parent] = newPW]
            /\ UpdateSerial(t, ser)
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_child"]
            /\ UNCHANGED <<local, op, target>>
       ELSE \* Disturbed
            /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

\* CAS grandparent (2 levels up): mark parent's child slot as Null in grandparent,
\* then mark child's slot as Null in parent, restore both.
\* This is a simplified model — the real code does CAS list bottom-up.
UnbundleCASGP(t) ==
    /\ pc[t] = "unbundle_cas_gp"
    /\ LET node       == target[t]
           parentNode == ParentOf(node)
           gpNode     == ParentOf(parentNode)
           gpW        == linkage[gpNode]
           ser        == GenSerial(t, gpW.serial)
       IN
       IF gpW.hasPriority /\ linkage[gpNode] = gpW
          /\ gpW.packet.sub[parentNode] /= Null
       THEN \* CAS grandparent: clear parent's slot
            LET parentPkt   == gpW.packet.sub[parentNode]
                newParentSub == [parentPkt.sub EXCEPT ![node] = Null]
                newParentPkt == MakePacket(parentNode, parentPkt.payload,
                                           newParentSub, TRUE)
                newGPSub     == [gpW.packet.sub EXCEPT ![parentNode] = newParentPkt]
                newGPPkt     == MakePacket(gpNode, gpW.packet.payload,
                                           newGPSub, TRUE)
                newGPW       == PriorityWrapper(newGPPkt, ser)
            IN
            /\ linkage' = [linkage EXCEPT ![gpNode] = newGPW]
            /\ UpdateSerial(t, ser)
            /\ pc' = [pc EXCEPT ![t] = "unbundle_restore_parent"]
            /\ UNCHANGED <<local, op, target>>
       ELSE \* Disturbed
            /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

\* After GP CAS: restore parent node to priority
UnbundleRestoreParent(t) ==
    /\ pc[t] = "unbundle_restore_parent"
    /\ LET node       == target[t]
           parentNode == ParentOf(node)
           parentW    == local[t].wrapper  \* child's wrapper, bundledBy = parentNode
           gpNode     == ParentOf(parentNode)
           gpW        == linkage[gpNode]
           ser        == GenSerial(t, linkage[parentNode].serial)
       IN
       \* Parent's linkage should still be a bundled-ref to GP
       IF ~linkage[parentNode].hasPriority /\ linkage[parentNode].bundledBy = gpNode
       THEN \* Extract parent's packet from GP and restore as priority
            IF gpW.hasPriority /\ gpW.packet.sub[parentNode] /= Null
            THEN LET parentPkt == gpW.packet.sub[parentNode]
                     newPW == PriorityWrapper(parentPkt, ser)
                 IN
                 /\ linkage' = [linkage EXCEPT ![parentNode] = newPW]
                 /\ UpdateSerial(t, ser)
                 /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_child"]
                 /\ UNCHANGED <<local, op, target>>
            ELSE \* Disturbed
                 /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>
       ELSE \* Parent already has priority (maybe another thread restored it)
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_child"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target>>

\* Final: CAS child's linkage to restore priority with new (committed) packet
UnbundleCASChild(t) ==
    /\ pc[t] = "unbundle_cas_child"
    /\ LET node     == target[t]
           oldChildW == local[t].wrapper
           ser      == GenSerial(t, oldChildW.serial)
           newChildW == PriorityWrapper(local[t].newpacket, ser)
       IN
       IF linkage[node] = oldChildW
       THEN /\ linkage' = [linkage EXCEPT ![node] = newChildW]
            /\ UpdateSerial(t, ser)
            /\ local' = [local EXCEPT ![t].commitOk = "ok"]
            /\ pc' = [pc EXCEPT ![t] = "commit_done"]
            /\ UNCHANGED <<op, target>>
       ELSE \* Child changed — retry
            /\ pc' = [pc EXCEPT ![t] = "commit_read"]
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
        \/ \E n \in InnerNodes : SnapRead(t, n)
        \/ SnapCheck(t)
        \/ BundlePhase1(t)
        \/ BundlePhase2(t)
        \/ BundlePhase3(t)
        \/ BundlePhase4(t)
        \/ SnapDone(t)
        \/ \E n \in AllNodes \ {Grand} : CommitStart(t, n)
        \/ CommitRead(t)
        \/ CommitTryCAS(t)
        \/ UnbundleWalk(t)
        \/ UnbundleCASAncestor(t)
        \/ UnbundleCASGP(t)
        \/ UnbundleRestoreParent(t)
        \/ UnbundleCASChild(t)
        \/ CommitDone(t)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* State constraint: bound serials for finite state space (replaces modular arithmetic)
StateConstraint ==
    /\ globalSerial <= 3 * MaxPayload

-----------------------------------------------------------------------------
(* Safety invariants *)

\* INV1: If a node's packet is not missing, all sub-packets exist
SnapshotConsistency ==
    \A n \in InnerNodes :
        LET w == linkage[n]
        IN  (w.hasPriority /\ ~w.packet.missing) =>
            (\A c \in ChildrenOf(n) : w.packet.sub[c] /= Null)

\* INV2: Every non-root node is either priority or has a valid bundledBy
NoPriorityLoss ==
    \A n \in AllNodes \ {Grand} :
        LET w == linkage[n]
        IN  w.hasPriority \/ w.bundledBy /= Null

\* INV3: If node is bundled, its bundledBy target must have priority
\*        OR be itself bundled to a node with priority (chain of 2)
BundleChainValid ==
    \A n \in AllNodes \ {Grand} :
        LET w == linkage[n]
        IN  (~w.hasPriority /\ w.bundledBy /= Null) =>
            LET pw == linkage[w.bundledBy]
            IN  pw.hasPriority \/ pw.bundledBy /= Null

\* INV4: bundledBy always points to the correct structural parent
BundledByCorrect ==
    \A n \in AllNodes \ {Grand} :
        LET w == linkage[n]
        IN  ~w.hasPriority => w.bundledBy = ParentOf(n)

\* INV5: Grand always has priority (it's the root, nobody bundles it)
GrandAlwaysPriority ==
    linkage[Grand].hasPriority

Safety ==
    /\ SnapshotConsistency
    /\ NoPriorityLoss
    /\ BundleChainValid
    /\ BundledByCorrect
    /\ GrandAlwaysPriority

=============================================================================
