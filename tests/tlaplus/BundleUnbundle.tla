(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************)
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
    MaxPayload,     \* Payloads wrap at this value (e.g. 2)
    MaxSerial       \* Serial wrap-around (must be even)

GrandChildren == {Parent}         \* Grand's children
ParentChildren == {Child1, Child2} \* Parent's children
AllNodes == {Grand, Parent, Child1, Child2}
LeafNodes == {Child1, Child2}
InnerNodes == {Grand, Parent}

(* Symmetry sets for state space reduction *)
ThreadSymmetry == Permutations(Threads)

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

\* ==========================================================================
\* @c11_mapping -- Variable-to-C++ correspondence (Layer 2, 3-level tree)
\*
\* Same mapping as BundleUnbundle_2level.tla but with 3-level hierarchy:
\*   Grand -> Parent -> {Child1, Child2}
\*
\* @c11_var linkage[n]:       atomic_shared_ptr<PacketWrapper> n->m_link
\*   Read: load_shared_(), Write: compareAndSet() -- both Layer 0 ops
\* @c11_var linkage[n].packet.sub[c]:  Packet::subpackets()[i]
\* @c11_var linkage[n].bundledBy:      PacketWrapper::bundledBy()
\* @c11_var serial[t]:        thread-local Lamport clock (SerialGenerator)
\*   Modular arithmetic: (base + 1) % MaxSerial. Comparisons via ModGT
\*   (signed-difference mod MaxSerial), matching C++ unsigned subtraction
\*   reinterpreted as signed. MaxSerial must be even.
\*
\* 3-level adds: recursive bundle (Grand bundles Parent which bundles Children),
\* 2-level unbundle walk (Child->Parent->Grand via snapshotSupernode recursion),
\* and UnbundleCASGP / UnbundleRestoreParent for the grandparent path.
\* Serial arithmetic is modular (no StateConstraint needed for finiteness).
\*
\* Source: kame/transaction.h, kame/transaction_impl.h
\* ==========================================================================

VARIABLES
    serial, globalSerial, linkage, pc, op, target, local,
    commit_count    \* per-node commit counter (model-only, for QuiescentCheck)

vars == <<serial, globalSerial, linkage, pc, op, target, local, commit_count>>

-----------------------------------------------------------------------------
(* Modular serial arithmetic *)

\* Modular serial comparison (same as C++ signed-difference comparison)
ModGT(a, b) == LET diff == (a - b + MaxSerial) % MaxSerial
               IN  diff > 0 /\ diff < MaxSerial \div 2

\* Advance past lastSer then increment (Lamport step)
GenSerial(t, lastSer) ==
    LET base == IF ModGT(lastSer, serial[t]) THEN lastSer ELSE serial[t]
    IN  (base + 1) % MaxSerial

UpdateSerial(t, ser) ==
    /\ serial' = [serial EXCEPT ![t] = ser]
    /\ globalSerial' = IF ModGT(ser, globalSerial) THEN ser ELSE globalSerial

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
    parentWrapper |-> Null,    \* saved parent wrapper for unbundle CAS
    gpWrapper   |-> Null,      \* saved grandparent wrapper for 2-level unbundle CAS
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
    /\ commit_count = [n \in AllNodes |-> 0]

-----------------------------------------------------------------------------
(* Snapshot: read node, bundle if missing *)

\* @c11_action SnapRead(t, node):
\*   Entry for snapshot(node) on any inner node (Grand or Parent).
\*   Source: transaction_impl.h:842-870
SnapRead(t, node) ==
    /\ pc[t] = "idle"
    /\ node \in InnerNodes
    /\ op' = [op EXCEPT ![t] = "snapshot"]
    /\ target' = [target EXCEPT ![t] = node]
    /\ pc' = [pc EXCEPT ![t] = "snap_check"]
    /\ UNCHANGED <<serial, globalSerial, linkage, local, commit_count>>

\* @c11_action SnapCheck(t):
\*   local_shared_ptr<PacketWrapper> w(*node->m_link);
\*   if (hasPriority && !missing) -> snap_done   // fast path
\*   if (hasPriority && missing)  -> bundle_phase1  // need bundle
\*   else                         -> retry       // bundled elsewhere
\*   Source: transaction_impl.h:842-870
SnapCheck(t) ==
    /\ pc[t] = "snap_check"
    /\ LET node == target[t]
           w == linkage[node]
       IN
       IF w.hasPriority /\ ~w.packet.missing
       THEN \* Fast path: complete
            /\ local' = [local EXCEPT ![t].snapResult = w.packet]
            /\ pc' = [pc EXCEPT ![t] = "snap_done"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>
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
            /\ UNCHANGED <<linkage, op, target, commit_count>>
       ELSE \* Node is bundled — need to unbundle first (snapshot from bundled state)
            \* For simplicity, just retry (the real code calls snapshotSupernode)
            /\ pc' = [pc EXCEPT ![t] = "snap_check"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

-----------------------------------------------------------------------------
(* Bundle: 4-phase protocol *)

\* Phase 1: Collect sub-packets from children.
\* If a child is bundled elsewhere, we can't collect — restart.
\* If a child is bundled HERE (same node), use the existing sub-packet.
\* If a child has priority but is missing, we need to recursively bundle it first.
\* @c11_action BundlePhase1(t):
\*   Phase 1: collect sub-packets. For each child of bundleNode:
\*     subw = *child->m_link;  bundle_subpacket(...);
\*   Recursively bundles children that are themselves missing.
\*   Source: transaction_impl.h:1077-1114
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
                    THEN IF parentW.packet.sub[c] /= Null
                            /\ ~parentW.packet.sub[c].missing
                         THEN parentW.packet.sub[c] \* already bundled here, complete
                         ELSE Null               \* bundled here but missing — need recursive bundle (retry)
                    ELSE Null]               \* bundled elsewhere
           allCollected == \A c \in children : childPkts[c] /= Null
       IN
       IF allCollected
       THEN /\ local' = [local EXCEPT
                    ![t].subwrappers = childWs,
                    ![t].subpackets  = childPkts]
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>
       ELSE \* Can't collect all — restart
            /\ pc' = [pc EXCEPT ![t] = "snap_check"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action BundlePhase2(t):
\*   // Phase 2: CAS bundleNode's linkage with new packet (still missing=TRUE)
\*   bundleNode->m_link->compareAndSet(oldsuperwrapper, superwrapper);
\*   Source: transaction_impl.h:1121-1130
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
            /\ UNCHANGED <<serial, globalSerial, op, target, commit_count>>
       ELSE \* Disturbed
            /\ pc' = [pc EXCEPT ![t] = "snap_check"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action BundlePhase3(t):
\*   // Phase 3: CAS each child to bundled-ref, ONE AT A TIME.
\*   // C++ loops: for each child, compareAndSet(subwrappers[i], bundled_ref).
\*   // On failure at child i, rollback children 0..i-1 and restart.
\*   // Modeled as: pick one un-bundled child, CAS it. Repeat until all done.
\*   Source: transaction_impl.h:1132-1154
BundlePhase3(t) ==
    /\ pc[t] = "bundle_phase3"
    /\ LET node     == local[t].bundleNode
           ser      == local[t].bundleSer
           children == ChildrenOf(node)
           childWs  == local[t].subwrappers
       IN
       \* Success path: pick one matching child, CAS it to bundled-ref.
       \* No hasPriority guard: Phase1 may have found the child already bundled (BundledRef),
       \* in which case the CAS refreshes its serial. This also handles the case where
       \* two threads race through Phase2 and one finds children already bundled.
       \/ /\ \E c \in children :
             /\ linkage[c] = childWs[c]       \* CAS precondition: matches saved value
             /\ linkage' = [linkage EXCEPT ![c] = BundledRefWrapper(node, ser)]
             \* Check if all children are now bundled
             /\ LET allDone == \A c2 \in children \ {c} :
                                   ~linkage[c2].hasPriority
                IN
                IF allDone
                THEN pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                ELSE pc' = [pc EXCEPT ![t] = "bundle_phase3"]
             /\ UNCHANGED <<serial, globalSerial, local, op, target, commit_count>>
       \* Failure path: some child changed (linkage no longer matches saved value).
       \* No hasPriority guard: the CAS fails whenever linkage[c] /= childWs[c],
       \* regardless of whether the new value has priority (covers the case where
       \* another thread bundled the child, leaving it as BundledRef with hasPriority=FALSE).
       \/ /\ \E c \in children :
                /\ childWs[c] /= Null
                /\ linkage[c] /= childWs[c]
          /\ linkage' = [n \in AllNodes |->
                IF n \in children
                   /\ ~linkage[n].hasPriority
                   /\ linkage[n].bundledBy = node
                THEN childWs[n]
                ELSE linkage[n]]
          /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
          /\ UNCHANGED <<serial, globalSerial, local, op, target, commit_count>>

\* @c11_action BundlePhase4(t):
\*   // Phase 4: finalize -- clear missing flag, CAS bundleNode
\*   superwrapper = new PacketWrapper(*superwrapper, bundle_serial);
\*   newpacket->m_missing = false;  // all sub-packets present
\*   bundleNode->m_link->compareAndSet(oldsuperwrapper, superwrapper);
\*   Source: transaction_impl.h:1158-1171
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
            /\ UNCHANGED <<serial, globalSerial, op, target, commit_count>>
       ELSE \* Disturbed
            /\ pc' = [pc EXCEPT ![t] = "snap_check"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action SnapDone(t):
\*   Return snapshot result to caller. Thread-local only.
SnapDone(t) ==
    /\ pc[t] = "snap_done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ op' = [op EXCEPT ![t] = "idle"]
    /\ target' = [target EXCEPT ![t] = Null]
    /\ local' = [local EXCEPT ![t] = InitLocal]
    /\ UNCHANGED <<serial, globalSerial, linkage, commit_count>>

-----------------------------------------------------------------------------
(* Commit on a leaf or inner node *)

\* @c11_action CommitStart(t, node):
\*   Entry: Transaction<XN> tr(node);
\*   Source: transaction.h:607-613
CommitStart(t, node) ==
    /\ pc[t] = "idle"
    /\ node \in AllNodes \ {Grand}  \* Can commit to Parent or Children
    /\ op' = [op EXCEPT ![t] = "commit"]
    /\ target' = [target EXCEPT ![t] = node]
    /\ pc' = [pc EXCEPT ![t] = "commit_read"]
    /\ UNCHANGED <<serial, globalSerial, linkage, local, commit_count>>

\* @c11_action CommitRead(t):
\*   local_shared_ptr<PacketWrapper> wrapper(*node->m_link);  -- load_shared_
\*   if (wrapper->hasPriority())
\*       -> commit_try_cas       // direct commit path
\*   else
\*       -> unbundle_walk        // bundled, need unbundle first
\*   Source: transaction_impl.h:1241-1276
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
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>
       ELSE \* Bundled — need unbundle
            /\ local' = [local EXCEPT ![t].wrapper = w]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>

\* @c11_action CommitTryCAS(t):
\*   // Direct commit (hasPriority path)
\*   local_shared_ptr<PacketWrapper> newwrapper(
\*       new PacketWrapper(tr.m_packet, tr.m_serial));
\*   if (m_link->compareAndSet(wrapper, newwrapper))
\*       return true;           // success
\*   // payload unchanged -> single-node optimization: adopt new children
\*   // payload changed   -> true conflict, fail
\*   Source: transaction_impl.h:1245-1270
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
            /\ commit_count' = [commit_count EXCEPT ![node] = @ + 1]
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
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>
            ELSE \* True conflict
                 /\ local' = [local EXCEPT ![t].commitOk = "fail"]
                 /\ pc' = [pc EXCEPT ![t] = "commit_done"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

-----------------------------------------------------------------------------
(* Unbundle — recursive walk up the bundledBy chain *)

\* @c11_action UnbundleWalk(t):
\*   // Walk up bundledBy chain (1 or 2 levels) to find sub-packet
\*   snapshotSupernode(sublinkage, superwrapper, &subpacket,
\*       FOR_UNBUNDLE, serial, &cas_infos);
\*   1-level: extract from parent->packet.sub[node]
\*   2-level: extract from grandparent->packet.sub[parent].sub[node]
\*   Source: transaction_impl.h:1314-1344, 696-755
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
                           ![t].parentWrapper = parentW,
                           ![t].oldpacket = parentW.packet.sub[node],
                           ![t].newpacket = MakePacket(node,
                               (parentW.packet.sub[node].payload + 1) % MaxPayload,
                               parentW.packet.sub[node].sub,
                               parentW.packet.sub[node].missing)]
                      /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_ancestor"]
                      /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>
                 ELSE \* Sub-packet is Null (not yet bundled) — retry
                      /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                      /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>
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
                                 ![t].gpWrapper = gpW,
                                 ![t].oldpacket = subPkt,
                                 ![t].newpacket = MakePacket(node,
                                     (subPkt.payload + 1) % MaxPayload,
                                     subPkt.sub,
                                     subPkt.missing)]
                           /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_gp"]
                           /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>
                      ELSE \* Can't find — retry
                           /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                           /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>
                 ELSE \* No grandparent — retry
                      /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                      /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>
       ELSE \* bundledBy is Null — retry
            /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action UnbundleCASAncestor(t):
\*   // CAS parent: set missing=TRUE (sub-packets preserved, NOT nulled)
\*   // C++: p->reset(new Packet(**p)); (*p)->m_missing = true;
\*   for each cas_info in cas_infos:
\*     it->linkage->compareAndSet(it->old_wrapper, it->new_wrapper);
\*   Source: transaction_impl.h:1367-1379, snapshotSupernode FOR_UNBUNDLE
\* CAS ancestor (1 level up): set missing=TRUE in parent, restore child
UnbundleCASAncestor(t) ==
    /\ pc[t] = "unbundle_cas_ancestor"
    /\ LET node     == target[t]
           parent   == local[t].wrapper.bundledBy
           oldParentW == local[t].parentWrapper
           ser      == GenSerial(t, oldParentW.serial)
           \* C++ keeps sub-packets intact; only sets missing=TRUE
           newPkt   == MakePacket(parent, oldParentW.packet.payload,
                                  oldParentW.packet.sub, TRUE)
           newPW    == PriorityWrapper(newPkt, ser)
       IN
       IF oldParentW.hasPriority /\ linkage[parent] = oldParentW
       THEN /\ linkage' = [linkage EXCEPT ![parent] = newPW]
            /\ UpdateSerial(t, ser)
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_child"]
            /\ UNCHANGED <<local, op, target, commit_count>>
       ELSE \* Disturbed
            /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action UnbundleCASGP(t):
\*   // CAS grandparent (2 levels up): clear child's slot in the parent sub-packet
\*   // within the grandparent's bundle. CAS list is processed bottom-up in C++.
\*   for each cas_info in cas_infos:
\*     it->linkage->compareAndSet(it->old_wrapper, it->new_wrapper);
\*   Source: transaction_impl.h:1367-1379, 696-755
\* CAS grandparent (2 levels up): mark parent's child slot as Null in grandparent,
\* then mark child's slot as Null in parent, restore both.
\* This is a simplified model — the real code does CAS list bottom-up.
UnbundleCASGP(t) ==
    /\ pc[t] = "unbundle_cas_gp"
    /\ LET node       == target[t]
           parentNode == ParentOf(node)
           gpNode     == ParentOf(parentNode)
           oldGPW     == local[t].gpWrapper
           ser        == GenSerial(t, oldGPW.serial)
       IN
       IF oldGPW.hasPriority /\ linkage[gpNode] = oldGPW
          /\ oldGPW.packet.sub[parentNode] /= Null
       THEN \* CAS grandparent: set missing=TRUE (sub-packets preserved)
            \* C++: p->reset(new Packet(**p)); (*p)->m_missing = true;
            LET parentPkt   == oldGPW.packet.sub[parentNode]
                \* Parent sub-packet: keep child's slot intact, just set missing
                newParentPkt == MakePacket(parentNode, parentPkt.payload,
                                           parentPkt.sub, TRUE)
                newGPSub     == [oldGPW.packet.sub EXCEPT ![parentNode] = newParentPkt]
                newGPPkt     == MakePacket(gpNode, oldGPW.packet.payload,
                                           newGPSub, TRUE)
                newGPW       == PriorityWrapper(newGPPkt, ser)
            IN
            /\ linkage' = [linkage EXCEPT ![gpNode] = newGPW]
            /\ UpdateSerial(t, ser)
            /\ pc' = [pc EXCEPT ![t] = "unbundle_restore_parent"]
            /\ UNCHANGED <<local, op, target, commit_count>>
       ELSE \* Disturbed
            /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action UnbundleRestoreParent(t):
\*   // After GP unbundle CAS: restore parent to priority with its sub-packet
\*   // extracted from grandparent's bundle.
\*   newsubwrapper = new PacketWrapper(*parentsubpacket, gen(serial));
\*   parent->m_link->compareAndSet(bundled_ref, newsubwrapper);
\*   Source: transaction_impl.h:1383-1389
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
                 /\ UNCHANGED <<local, op, target, commit_count>>
            ELSE \* Disturbed
                 /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>
       ELSE \* Parent already has priority (maybe another thread restored it)
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_child"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action UnbundleCASChild(t):
\*   // Restore child to priority with extracted sub-packet
\*   newsubwrapper = new PacketWrapper(*subpacket, gen(superwrapper->m_bundle_serial));
\*   sublinkage->compareAndSet(bundled_ref, newsubwrapper);
\*   Source: transaction_impl.h:1383-1389
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
            /\ commit_count' = [commit_count EXCEPT ![node] = @ + 1]
            /\ pc' = [pc EXCEPT ![t] = "commit_done"]
            /\ UNCHANGED <<op, target>>
       ELSE \* Child changed — retry
            /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action CommitDone(t):
\*   Return commit result to caller. Thread-local only.
CommitDone(t) ==
    /\ pc[t] = "commit_done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ op' = [op EXCEPT ![t] = "idle"]
    /\ target' = [target EXCEPT ![t] = Null]
    /\ local' = [local EXCEPT ![t] = InitLocal]
    /\ UNCHANGED <<serial, globalSerial, linkage, commit_count>>

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

\* INV6: MissingPropagation — mirrors Node<XN>::Packet::checkConsistensy.
\* If a node's packet is NOT missing, all its sub-packets must also be NOT missing.
MissingPropagation ==
    \A n \in InnerNodes :
        LET w == linkage[n]
        IN  (w.hasPriority /\ ~w.packet.missing) =>
            (\A c \in ChildrenOf(n) :
                w.packet.sub[c] /= Null => ~w.packet.sub[c].missing)

Safety ==
    /\ SnapshotConsistency
    /\ NoPriorityLoss
    /\ BundleChainValid
    /\ BundledByCorrect
    /\ GrandAlwaysPriority
    /\ MissingPropagation

\* QuiescentCheck: when all threads are idle, each non-Grand node with priority
\* has payload = commit_count mod MaxPayload.
QuiescentCheck ==
    (\A t \in Threads : pc[t] = "idle") =>
        \A n \in AllNodes \ {Grand} :
            linkage[n].hasPriority =>
                linkage[n].packet.payload = commit_count[n] % MaxPayload

=============================================================================
