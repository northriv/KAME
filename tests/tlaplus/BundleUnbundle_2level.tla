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
    MaxPayload,     \* Payloads wrap at this value
    MaxSerial       \* Serial wrap-around (must be even)

Children == {Child1, Child2}
Nodes == {Parent} \cup Children

(* Symmetry sets for state space reduction *)
ThreadSymmetry == Permutations(Threads)

\* ==========================================================================
\* @c11_mapping -- Variable-to-C++ correspondence (Layer 2, 2-level tree)
\*
\* This layer models the bundle/unbundle protocol that manages hierarchical
\* snapshots in KAME's STM. Each node has an atomic_shared_ptr<PacketWrapper>
\* called m_link (Linkage). Layer 1 (stm_commit) is the single-node commit;
\* this layer adds tree structure with bundle/unbundle on top.
\*
\* TLA+ variable              C++ type & expression
\* --------------------------------------------------------------------------
\* @c11_var linkage[n]:       atomic_shared_ptr<PacketWrapper> n->m_link
\*   -- each node's current PacketWrapper, read/written via Layer 0 ops
\*   Read:  local_shared_ptr<PacketWrapper> w(*n->m_link)  -- load_shared_
\*   Write: n->m_link->compareAndSet(old, new)             -- Layer 0 CAS
\*
\*   PacketWrapper states:
\*     hasPriority=TRUE:  wrapper owns its Packet directly (priority wrapper)
\*       -- PriorityWrapper(packet, serial)
\*     hasPriority=FALSE: wrapper is a back-reference to parent (bundled ref)
\*       -- BundledRefWrapper(parentNode, serial)
\*       -- C++: PacketWrapper(m_link_of_parent, reverse_index, bundle_serial)
\*
\* @c11_var linkage[n].packet.payload:  Packet::payload()->m_x
\* @c11_var linkage[n].packet.sub[c]:   Packet::subpackets()[i]
\* @c11_var linkage[n].packet.missing:  Packet::m_missing
\* @c11_var linkage[n].serial:          PacketWrapper::m_bundle_serial
\* @c11_var linkage[n].bundledBy:       PacketWrapper::bundledBy() -- shared_ptr<Linkage>
\*
\* @c11_var serial[t]:        thread-local Lamport clock (SerialGenerator per thread)
\*   Modular arithmetic: (base + 1) % MaxSerial. Comparisons via ModGT
\*   (signed-difference mod MaxSerial), matching C++ unsigned subtraction
\*   reinterpreted as signed. MaxSerial must be even.
\* @c11_var globalSerial:     max serial seen (simplified; C++ uses per-thread gen)
\*
\* @c11_var local[t].wrapper:     thread-local snapshot of linkage[target]
\* @c11_var local[t].subwrappers: thread-local snapshots of child linkages
\* @c11_var local[t].subpackets:  thread-local collected child packets
\*
\* All reads/writes to linkage go through atomic_shared_ptr:
\*   load_shared_() for reads, compareAndSet() for CAS updates.
\* Serial arithmetic is modular (no StateConstraint needed for finiteness).
\*
\* Source: kame/transaction.h, kame/transaction_impl.h
\* ==========================================================================

VARIABLES
    serial, globalSerial, linkage, pc, op, target, local,
    commit_count    \* per-node commit counter (model-only, for QuiescentCheck)

vars == <<serial, globalSerial, linkage, pc, op, target, local, commit_count>>

-----------------------------------------------------------------------------
(* Data structures *)

PriorityWrapper(packet, ser) ==
    [packet |-> packet, hasPriority |-> TRUE, bundledBy |-> Null, serial |-> ser]

BundledRefWrapper(parentNode, ser) ==
    [packet |-> Null, hasPriority |-> FALSE, bundledBy |-> parentNode, serial |-> ser]

MakePacket(node, payload, sub, miss) ==
    [payload |-> payload, sub |-> sub, missing |-> miss, node |-> node]

EmptySub == [c \in Children |-> Null]

\* Modular serial comparison (same as C++ signed-difference comparison)
ModGT(a, b) == LET diff == (a - b + MaxSerial) % MaxSerial
               IN  diff > 0 /\ diff < MaxSerial \div 2

GenSerial(t, lastSer) ==
    LET base == IF ModGT(lastSer, serial[t]) THEN lastSer ELSE serial[t]
    IN  (base + 1) % MaxSerial

UpdateSerial(t, ser) ==
    /\ serial' = [serial EXCEPT ![t] = ser]
    /\ globalSerial' = IF ModGT(ser, globalSerial) THEN ser ELSE globalSerial

InitLocal == [
    wrapper     |-> Null,
    parentWrapper |-> Null,  \* saved parent wrapper for unbundle CAS
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
    /\ commit_count = [n \in Nodes |-> 0]

-----------------------------------------------------------------------------
(* Snapshot *)

\* @c11_action SnapRead(t):
\*   Entry point for snapshot(node). Initiates a read of node's m_link.
\*   Source: transaction_impl.h:842-870, transaction.h:328
SnapRead(t) ==
    /\ pc[t] = "idle"
    /\ op' = [op EXCEPT ![t] = "snapshot"]
    /\ target' = [target EXCEPT ![t] = Parent]
    /\ pc' = [pc EXCEPT ![t] = "snap_read"]
    /\ UNCHANGED <<serial, globalSerial, linkage, local, commit_count>>

\* @c11_action SnapCheck(t):
\*   local_shared_ptr<PacketWrapper> w(*parent->m_link);  -- load_shared_
\*   if (w->hasPriority() && !w->packet()->missing())
\*       return w->packet();           -- fast path, no bundle needed
\*   else if (w->hasPriority() && w->packet()->missing())
\*       -> bundle_phase1              -- need to bundle children
\*   else
\*       -> retry                      -- node is bundled, need unbundle
\*   Source: transaction_impl.h:842-870
SnapCheck(t) ==
    /\ pc[t] = "snap_read"
    /\ LET w == linkage[Parent]
       IN
       IF w.hasPriority /\ ~w.packet.missing
       THEN /\ local' = [local EXCEPT ![t].snapResult = w.packet]
            /\ pc' = [pc EXCEPT ![t] = "snap_done"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>
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
            /\ UNCHANGED <<linkage, op, target, commit_count>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action BundlePhase1(t):
\*   // Phase 1: collect sub-packets from children
\*   for each child:
\*     local_shared_ptr<PacketWrapper> subw(*child->m_link);  -- load_shared_
\*     bundle_subpacket(&superwrapper, child, subw, subpacket, ...);
\*   If child hasPriority: use its packet directly.
\*   If child bundledBy==Parent: use parent's existing sub-packet for it.
\*   If child bundledBy==other: unbundle first (-> retry).
\*   Source: transaction_impl.h:1077-1114
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
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action BundlePhase2(t):
\*   // Phase 2: CAS parent's linkage with new packet (still missing=TRUE)
\*   parent->m_link->compareAndSet(oldsuperwrapper, superwrapper);
\*   Source: transaction_impl.h:1121-1130
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
            /\ UNCHANGED <<serial, globalSerial, op, target, commit_count>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action BundlePhase3(t):
\*   // Phase 3: CAS each child to bundled-ref, ONE AT A TIME.
\*   // C++ loops: for each child, compareAndSet(subwrappers[i], bundled_ref).
\*   // On failure at child i, rollback children 0..i-1 and restart.
\*   // Modeled as: pick one un-bundled child, CAS it. Repeat until all done.
\*   Source: transaction_impl.h:1132-1154
BundlePhase3(t) ==
    /\ pc[t] = "bundle_phase3"
    /\ LET ser     == local[t].bundleSer
           childWs == local[t].subwrappers
       IN
       \* Pick a matching child, CAS it to bundled-ref.
       \* No hasPriority guard: Phase1 may have found the child already bundled,
       \* in which case the CAS refreshes its serial.
       \/ \E c \in Children :
             /\ linkage[c] = childWs[c]           \* CAS precondition: matches saved value
             /\ linkage' = [linkage EXCEPT ![c] = BundledRefWrapper(Parent, ser)]
             \* Check if all children are now bundled
             /\ LET allDone == \A c2 \in Children \ {c} :
                                   ~linkage[c2].hasPriority
                IN
                IF allDone
                THEN pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                ELSE pc' = [pc EXCEPT ![t] = "bundle_phase3"]  \* continue with next child
             /\ UNCHANGED <<serial, globalSerial, local, op, target, commit_count>>
       \/ \* Any child's CAS fails → rollback all bundled children, restart.
          \* No hasPriority guard: CAS fails whenever linkage[c] /= childWs[c],
          \* regardless of whether the new value has priority.
          /\ \E c \in Children :
                /\ childWs[c] /= Null
                /\ linkage[c] /= childWs[c]
          /\ \* Rollback: restore any already-bundled children to their saved wrappers
             linkage' = [n \in Nodes |->
                 IF n \in Children /\ ~linkage[n].hasPriority
                    /\ linkage[n].bundledBy = Parent
                 THEN childWs[n]  \* rollback
                 ELSE linkage[n]]
          /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
          /\ UNCHANGED <<serial, globalSerial, local, op, target, commit_count>>

\* @c11_action BundlePhase4(t):
\*   // Phase 4: finalize -- clear missing flag, CAS parent
\*   superwrapper = new PacketWrapper(*superwrapper, bundle_serial);
\*   newpacket->m_missing = false;  // all sub-packets present
\*   parent->m_link->compareAndSet(oldsuperwrapper, superwrapper);
\*   Source: transaction_impl.h:1158-1171
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
            /\ UNCHANGED <<serial, globalSerial, op, target, commit_count>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

SnapDone(t) ==
    /\ pc[t] = "snap_done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ op' = [op EXCEPT ![t] = "idle"]
    /\ target' = [target EXCEPT ![t] = Null]
    /\ local' = [local EXCEPT ![t] = InitLocal]
    /\ UNCHANGED <<serial, globalSerial, linkage, commit_count>>

-----------------------------------------------------------------------------
(* Commit *)

\* @c11_action CommitStart(t, childNode):
\*   Entry: Transaction<XN> tr(childNode);
\*   Source: transaction.h:607-613
CommitStart(t, childNode) ==
    /\ pc[t] = "idle"
    /\ childNode \in Children
    /\ op' = [op EXCEPT ![t] = "commit"]
    /\ target' = [target EXCEPT ![t] = childNode]
    /\ pc' = [pc EXCEPT ![t] = "commit_read"]
    /\ UNCHANGED <<serial, globalSerial, linkage, local, commit_count>>

\* @c11_action CommitRead(t):
\*   local_shared_ptr<PacketWrapper> wrapper(*child->m_link);  -- load_shared_
\*   if (wrapper->hasPriority())
\*       -> commit_try_cas       // direct commit path
\*   else
\*       -> unbundle_walk        // bundled, need unbundle first
\*   Source: transaction_impl.h:1241-1276
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
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>
       ELSE /\ local' = [local EXCEPT ![t].wrapper = w]
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
    /\ LET childNode == target[t]
           oldW      == local[t].wrapper
           ser       == GenSerial(t, oldW.serial)
           newW      == PriorityWrapper(local[t].newpacket, ser)
       IN
       IF linkage[childNode] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![childNode] = newW]
            /\ UpdateSerial(t, ser)
            /\ local' = [local EXCEPT ![t].commitOk = "ok"]
            /\ commit_count' = [commit_count EXCEPT ![childNode] = @ + 1]
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
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>
            ELSE /\ local' = [local EXCEPT ![t].commitOk = "fail"]
                 /\ pc' = [pc EXCEPT ![t] = "commit_done"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action UnbundleWalk(t):
\*   // Walk up bundledBy chain to find sub-packet in parent's bundle
\*   snapshotSupernode(sublinkage, superwrapper, &subpacket,
\*       FOR_UNBUNDLE, serial, &cas_infos);
\*   Extract child's packet from parent's packet.sub[child].
\*   Source: transaction_impl.h:1314-1344, 696-755
UnbundleWalk(t) ==
    /\ pc[t] = "unbundle_walk"
    /\ LET childNode == target[t]
           w         == local[t].wrapper
           parentW   == linkage[Parent]
       IN
       IF w.bundledBy = Parent /\ parentW.hasPriority
          /\ parentW.packet.sub[childNode] /= Null
       THEN /\ local' = [local EXCEPT
                   ![t].parentWrapper = parentW,
                   ![t].oldpacket = parentW.packet.sub[childNode],
                   ![t].newpacket = MakePacket(childNode,
                       (parentW.packet.sub[childNode].payload + 1) % MaxPayload,
                       parentW.packet.sub[childNode].sub,
                       parentW.packet.sub[childNode].missing)]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_ancestors"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, commit_count>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action UnbundleCASAncestors(t):
\*   // CAS parent: set missing=TRUE (sub-packets are preserved, NOT nulled)
\*   // C++: new PacketWrapper(*shot_upper, shot_upper->m_bundle_serial)
\*   //   -- copy with the SAME serial; no gen() call here.
\*   //   -- p->reset(new Packet(**p)); (*p)->m_missing = true;
\*   for each cas_info in cas_infos:
\*     it->linkage->compareAndSet(it->old_wrapper, it->new_wrapper);
\*   Source: transaction_impl.h:1367-1379, snapshotSupernode FOR_UNBUNDLE
UnbundleCASAncestors(t) ==
    /\ pc[t] = "unbundle_cas_ancestors"
    /\ LET childNode  == target[t]
           oldParentW == local[t].parentWrapper
           \* C++: same serial as old wrapper (no gen() for ancestor CAS)
           \* C++ keeps sub-packets intact; only sets missing=TRUE
           newPkt     == MakePacket(Parent, oldParentW.packet.payload,
                                   oldParentW.packet.sub, TRUE)
           newParentW == PriorityWrapper(newPkt, oldParentW.serial)
       IN
       IF oldParentW.hasPriority /\ linkage[Parent] = oldParentW
       THEN /\ linkage' = [linkage EXCEPT ![Parent] = newParentW]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_child"]
            /\ UNCHANGED <<serial, globalSerial, local, op, target, commit_count>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

\* @c11_action UnbundleCASChild(t):
\*   // Restore child to priority with extracted sub-packet
\*   newsubwrapper = new PacketWrapper(*subpacket, gen(superwrapper->m_bundle_serial));
\*   sublinkage->compareAndSet(bundled_ref, newsubwrapper);
\*   Source: transaction_impl.h:1383-1389
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
            /\ commit_count' = [commit_count EXCEPT ![childNode] = @ + 1]
            /\ pc' = [pc EXCEPT ![t] = "commit_done"]
            /\ UNCHANGED <<op, target>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, commit_count>>

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
    \A t \in Threads : serial[t] \in 0..(MaxSerial - 1)

\* MissingPropagation: mirrors Node<XN>::Packet::checkConsistensy check #4.
\* If parent packet is NOT missing, all its sub-packets must also be NOT missing.
\* (Contrapositive: child.missing => parent.missing)
MissingPropagation ==
    LET pw == linkage[Parent]
    IN  (pw.hasPriority /\ ~pw.packet.missing) =>
        (\A c \in Children :
            pw.packet.sub[c] /= Null => ~pw.packet.sub[c].missing)

Safety == SnapshotConsistency /\ NoPriorityLoss /\ BundleRefConsistency
          /\ SerialMonotonicity /\ MissingPropagation

\* QuiescentCheck: when all threads are idle, each child with priority
\* has payload = commit_count mod MaxPayload.
QuiescentCheck ==
    (\A t \in Threads : pc[t] = "idle") =>
        \A n \in Children :
            linkage[n].hasPriority =>
                linkage[n].packet.payload = commit_count[n] % MaxPayload

=============================================================================
