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
 *
 * Thread lifecycle (MaxCommits iterations per thread):
 *   Each iteration:
 *     1. CommitParent: snapshot Parent, increment ALL children, CAS (retry until success).
 *     2. CommitChild for EACH child: direct commit (retry until success).
 * Terminal state: all iterBudget=0, all pc=idle. Run TLC with -deadlock.
 * Each child receives exactly 2 * MaxCommits * |Threads| increments total.
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Threads,
    Parent,
    Child1, Child2,
    Null,
    MaxPayload,          \* Payloads wrap at this value
    MaxSerial,           \* Serial modulus. Must exceed 2x the max serial consumed per run.
                         \* Fine-mode uses ~64 serials/commit; set >= 64*MaxCommits*|Threads|.
                         \* Modular wrap-around is required for TLC termination: commit retry
                         \* loops increment serial without consuming iterBudget.
    MaxCommits,          \* Per-child direct commit budget per thread
    BundleCollectAtomic, \* "coarse" / "fine" / "superfine"
    BundlePhase3Atomic   \* "coarse" / "fine" / "superfine"
                         \* superfine: fine + C++-faithful failure handling
                         \*   Phase1: retry same child if parent unchanged
                         \*   Phase3: check bundle_serial/parent → DISTURBED

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
\* Serial arithmetic is modular (required for finite state space; see MaxSerial comment).
\*
\* Source: kame/transaction.h, kame/transaction_impl.h
\* ==========================================================================

VARIABLES
    serial, globalSerial, linkage, pc, op, target, local,
    iterBudget,  \* [Threads -> 0..MaxCommits]: remaining full iterations per thread
    childQueue   \* [Threads -> SUBSET Children]: children pending CommitChild in current iteration

vars == <<serial, globalSerial, linkage, pc, op, target, local, iterBudget, childQueue>>

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
    /\ iterBudget = [t \in Threads |-> MaxCommits]
    /\ childQueue = [t \in Threads |-> {}]

-----------------------------------------------------------------------------
(* Snapshot *)

\* @c11_action SnapRead(t):
\*   Entry point for snapshot(node). Initiates a read of node's m_link.
\*   Guard: iterBudget[t] > 0 /\ childQueue[t] = {} -- start of each iteration.
\*   Source: transaction_impl.h:982-1000, transaction.h:328
SnapRead(t) ==
    /\ pc[t] = "idle"
    /\ iterBudget[t] > 0
    /\ childQueue[t] = {}
    /\ op' = [op EXCEPT ![t] = "snapshot"]
    /\ target' = [target EXCEPT ![t] = Parent]
    /\ pc' = [pc EXCEPT ![t] = "snap_read"]
    /\ UNCHANGED <<serial, globalSerial, linkage, local, iterBudget, childQueue>>

\* @c11_action SnapCheck(t):
\*   local_shared_ptr<PacketWrapper> w(*parent->m_link);  -- load_shared_
\*   if (w->hasPriority() && !w->packet()->missing())
\*       return w->packet();           -- fast path, no bundle needed
\*   else if (w->hasPriority() && w->packet()->missing())
\*       -> bundle_phase1              -- need to bundle children
\*   else
\*       -> retry                      -- node is bundled, need unbundle
\*   Source: transaction_impl.h:982-1000
SnapCheck(t) ==
    /\ pc[t] = "snap_read"
    /\ LET w == linkage[Parent]
       IN
       IF w.hasPriority /\ ~w.packet.missing
       THEN /\ local' = [local EXCEPT ![t].snapResult = w.packet]
            /\ pc' = [pc EXCEPT ![t] = "commit_parent"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
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
            /\ UNCHANGED <<linkage, op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>

\* @c11_action BundlePhase1(t):
\*   // Phase 1: collect sub-packets from children
\*   for each child:
\*     local_shared_ptr<PacketWrapper> subw(*child->m_link);  -- load_shared_
\*     bundle_subpacket(&superwrapper, child, subw, subpacket, ...);
\*   If child hasPriority: use its packet directly.
\*   If child bundledBy==Parent: use parent's existing sub-packet for it.
\*   If child bundledBy==other: unbundle first (-> retry).
\*   Source: transaction_impl.h:1200-1240
BundlePhase1(t) ==
    /\ pc[t] = "bundle_phase1"
    /\ IF BundleCollectAtomic = "coarse"
       THEN \* Coarse: read all children atomically
            LET childWrappers == [c \in Children |-> linkage[c]]
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
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
            ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>
       ELSE \* Fine: collect one unprocessed child per step
            LET parentW == local[t].wrapper
                ser     == local[t].bundleSer
            IN
            \* --- #1 superfine: Pre-bundle serial CAS (C++ bundle() entry, line 1182-1191) ---
            IF BundleCollectAtomic = "superfine" /\ parentW.serial /= ser
            THEN LET newW == PriorityWrapper(parentW.packet, ser)
                 IN
                 IF linkage[Parent] = parentW
                 THEN /\ linkage' = [linkage EXCEPT ![Parent] = newW]
                      /\ local' = [local EXCEPT ![t].wrapper = newW]
                      /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                      /\ UNCHANGED <<serial, globalSerial, op, target, iterBudget, childQueue>>
                 ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
                      /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>
            ELSE
            \/ \* Pick one unprocessed child (subwrappers[c] = Null)
               \E c \in Children :
                  /\ local[t].subwrappers[c] = Null
                  /\ LET cw == linkage[c]
                         cpkt == IF cw.hasPriority THEN cw.packet
                                 ELSE IF cw.bundledBy = Parent
                                      THEN IF parentW.packet.sub[c] /= Null
                                           THEN parentW.packet.sub[c]
                                           ELSE Null
                                      ELSE Null
                     IN
                     IF cpkt /= Null
                     THEN /\ local' = [local EXCEPT
                                  ![t].subwrappers[c] = cw,
                                  ![t].subpackets[c]  = cpkt]
                          /\ LET allDone == \A c2 \in Children \ {c} :
                                                local[t].subwrappers[c2] /= Null
                             IN
                             IF allDone
                             THEN pc' = [pc EXCEPT ![t] = "bundle_phase2"]
                             ELSE pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                          /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
                     ELSE \* Collection failed.
                          IF BundleCollectAtomic = "superfine"
                             /\ linkage[Parent] = parentW
                          THEN \* superfine + parent unchanged — retry same child.
                               /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                               /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>
                          ELSE \* fine: always restart; superfine: parent changed.
                               /\ pc' = [pc EXCEPT ![t] = "snap_read"]
                               /\ local' = [local EXCEPT
                                       ![t].subwrappers = [c2 \in Children |-> Null],
                                       ![t].subpackets  = EmptySub]
                               /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>

\* @c11_action BundlePhase2(t):
\*   // Phase 2: CAS parent's linkage with new packet (still missing=TRUE)
\*   parent->m_link->compareAndSet(oldsuperwrapper, superwrapper);
\*   Source: transaction_impl.h:1249-1258
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
            /\ UNCHANGED <<serial, globalSerial, op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>

\* @c11_action BundlePhase3(t):
\*   // Phase 3: CAS each child to bundled-ref, ONE AT A TIME.
\*   // C++ loops: for each child, compareAndSet(subwrappers[i], bundled_ref).
\*   // On failure at child i, rollback children 0..i-1 and restart.
\*   // Modeled as: pick one un-bundled child, CAS it. Repeat until all done.
\*   Source: transaction_impl.h:1260-1282
BundlePhase3(t) ==
    /\ pc[t] = "bundle_phase3"
    /\ LET ser     == local[t].bundleSer
           childWs == local[t].subwrappers
           allMatch == \A c \in Children : linkage[c] = childWs[c]
       IN
       IF BundlePhase3Atomic = "coarse"
       THEN \* Coarse: all children CAS'd atomically (all-or-nothing)
            IF allMatch
            THEN /\ linkage' = [n \in Nodes |->
                     IF n \in Children
                     THEN BundledRefWrapper(Parent, ser)
                     ELSE linkage[n]]
                 /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                 /\ UNCHANGED <<serial, globalSerial, local, op, target, iterBudget, childQueue>>
            ELSE /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>
       ELSE \* Fine: one child per step
            \/ \E c \in Children :
                  /\ linkage[c] = childWs[c]
                  /\ linkage' = [linkage EXCEPT ![c] = BundledRefWrapper(Parent, ser)]
                  /\ LET allDone == \A c2 \in Children \ {c} :
                                        linkage[c2] = BundledRefWrapper(Parent, ser)
                     IN
                     IF allDone
                     THEN pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                     ELSE pc' = [pc EXCEPT ![t] = "bundle_phase3"]
                  /\ UNCHANGED <<serial, globalSerial, local, op, target, iterBudget, childQueue>>
            \* Failure path: some child changed — rollback and restart.
            \* superfine: check bundle_serial/parent first (C++ transaction_impl.h:1274-1280).
            \/ /\ \E c \in Children :
                      /\ childWs[c] /= Null
                      /\ linkage[c] /= childWs[c]
               /\ LET disturbed ==
                       BundlePhase3Atomic = "superfine"
                       /\ (\/ \E c \in Children :
                                  /\ linkage[c] /= childWs[c]
                                  /\ linkage[c].serial /= ser
                            \/ linkage[Parent] /= local[t].wrapper)
                  IN
                  IF disturbed
                  THEN \* superfine DISTURBED — restart from snapshot
                       /\ pc' = [pc EXCEPT ![t] = "snap_read"]
                       /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>
                  ELSE \* #3: No rollback — restart Phase1 (re-collect re-adopts bundled children)
                       /\ local' = [local EXCEPT
                              ![t].subwrappers = [c \in Children |-> Null],
                              ![t].subpackets  = EmptySub]
                       /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                       /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>

\* @c11_action BundlePhase4(t):
\*   // Phase 4: finalize -- clear missing flag, CAS parent
\*   superwrapper = new PacketWrapper(*superwrapper, bundle_serial);
\*   newpacket->m_missing = false;  // all sub-packets present
\*   parent->m_link->compareAndSet(oldsuperwrapper, superwrapper);
\*   Source: transaction_impl.h:1286-1299
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
            /\ pc' = [pc EXCEPT ![t] = "commit_parent"]
            /\ UNCHANGED <<serial, globalSerial, op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>

-----------------------------------------------------------------------------
(* Commit *)

\* @c11_action CommitParent(t):
\*   Commit ALL children's payload changes under Parent's scope (one transaction).
\*   C++: Transaction<XN> tr(Parent); for each c: tr[c].m_x += 1; tr.commit();
\*   Uses snapResult (from completed snapshot) to build new packet with ALL children
\*   incremented, then CAS. Sets childQueue to Children for subsequent per-child commits.
CommitParent(t) ==
    /\ pc[t] = "commit_parent"
    /\ childQueue[t] = {}
    /\ local[t].snapResult /= Null
    /\ \A c \in Children : local[t].snapResult.sub[c] /= Null
    /\ LET pw      == linkage[Parent]
           snapPkt == local[t].snapResult
           newSub  == [c \in Children |->
               MakePacket(c,
                   (snapPkt.sub[c].payload + 1) % MaxPayload,
                   snapPkt.sub[c].sub,
                   snapPkt.sub[c].missing)]
           newPkt  == MakePacket(Parent, snapPkt.payload, newSub, snapPkt.missing)
           ser     == GenSerial(t, pw.serial)
           newPW   == PriorityWrapper(newPkt, ser)
       IN
       \/ \* CAS success: commit and move to per-child phase
          /\ pw.hasPriority
          /\ pw.packet = snapPkt
          /\ linkage' = [linkage EXCEPT ![Parent] = newPW]
          /\ UpdateSerial(t, ser)
          /\ childQueue' = [childQueue EXCEPT ![t] = Children]
          /\ pc' = [pc EXCEPT ![t] = "idle"]
          /\ op' = [op EXCEPT ![t] = "idle"]
          /\ target' = [target EXCEPT ![t] = Null]
          /\ local' = [local EXCEPT ![t] = InitLocal]
          /\ UNCHANGED iterBudget
       \/ \* CAS failure: retry from snapshot (iterate_commit semantics)
          /\ ~(pw.hasPriority /\ pw.packet = snapPkt)
          /\ pc' = [pc EXCEPT ![t] = "idle"]
          /\ op' = [op EXCEPT ![t] = "idle"]
          /\ target' = [target EXCEPT ![t] = Null]
          /\ local' = [local EXCEPT ![t] = InitLocal]
          /\ UNCHANGED <<serial, globalSerial, linkage, iterBudget, childQueue>>

\* @c11_action CommitStart(t, childNode):
\*   Entry: Transaction<XN> tr(childNode);
\*   Guard: childNode \in childQueue[t] -- only targets remaining in current iteration.
\*   Source: transaction.h:607-613
CommitStart(t, childNode) ==
    /\ pc[t] = "idle"
    /\ childNode \in childQueue[t]
    /\ op' = [op EXCEPT ![t] = "commit"]
    /\ target' = [target EXCEPT ![t] = childNode]
    /\ pc' = [pc EXCEPT ![t] = "commit_read"]
    /\ UNCHANGED <<serial, globalSerial, linkage, local, iterBudget, childQueue>>

\* @c11_action CommitRead(t):
\*   local_shared_ptr<PacketWrapper> wrapper(*child->m_link);  -- load_shared_
\*   if (wrapper->hasPriority())
\*       -> commit_try_cas       // direct commit path
\*   else
\*       -> unbundle_walk        // bundled, need unbundle first
\*   Source: transaction_impl.h:1364-1420
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
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
       ELSE /\ local' = [local EXCEPT ![t].wrapper = w]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>

\* @c11_action CommitTryCAS(t):
\*   // Direct commit (hasPriority path)
\*   Source: transaction_impl.h:1368-1400
\*
\*   Fidelity note (#7): C++ creates newwrapper once with tr.m_serial and reuses it
\*   across inner retries. TLA+ calls GenSerial each time, which may produce a higher
\*   serial after adopting a wrapper with a newer serial. No correctness impact.
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
            /\ UNCHANGED <<op, target, iterBudget, childQueue>>
       ELSE IF linkage[childNode].hasPriority
       THEN IF linkage[childNode].packet.payload = oldW.packet.payload
            THEN /\ local' = [local EXCEPT
                       ![t].wrapper = linkage[childNode],
                       ![t].newpacket = MakePacket(target[t],
                           local[t].newpacket.payload,
                           linkage[childNode].packet.sub,
                           linkage[childNode].packet.missing)]
                 /\ pc' = [pc EXCEPT ![t] = "commit_try_cas"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
            ELSE /\ local' = [local EXCEPT ![t].commitOk = "fail"]
                 /\ pc' = [pc EXCEPT ![t] = "commit_done"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>

\* @c11_action UnbundleWalk(t):
\*   // Walk up bundledBy chain to find sub-packet in parent's bundle
\*   snapshotForUnbundle(sublinkage, superwrapper, &subpacket,
\*       serial, &cas_infos);
\*   Extract child's packet from parent's packet.sub[child].
\*   Source: transaction_impl.h:1459-1490 (unbundle), 825-960 (walkUpChainImpl/snapshotForUnbundle)
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
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>

\* @c11_action UnbundleCASAncestors(t):
\*   // CAS parent: set missing=TRUE (sub-packets are preserved, NOT nulled)
\*   // C++: new PacketWrapper(*shot_upper, shot_upper->m_bundle_serial)
\*   //   -- copy with the SAME bundle serial value; no gen() call here.
\*   //   -- p->reset(new Packet(**p)); (*p)->m_missing = true;
\*   for each cas_info in cas_infos:
\*     it->linkage->compareAndSet(it->old_wrapper, it->new_wrapper);
\*   Source: transaction_impl.h:1494-1506, snapshotForUnbundle CAS preparation
\*
\*   TLA+ modeling note: C++ compareAndSet uses POINTER equality (local_shared_ptr identity).
\*   A new PacketWrapper allocation is always distinguishable from the old pointer, even if
\*   the semantic content is identical (e.g., missing was already TRUE). In TLA+, records
\*   compare by value, so we use GenSerial to obtain a fresh serial, making the new wrapper
\*   value-distinct. This correctly prevents BundlePhase4 from CAS-ing over an ancestor-CAS'd
\*   parent wrapper when missing was already TRUE.
UnbundleCASAncestors(t) ==
    /\ pc[t] = "unbundle_cas_ancestors"
    /\ LET childNode  == target[t]
           oldParentW == local[t].parentWrapper
           \* Use fresh serial to ensure new wrapper is value-distinct from old wrapper.
           \* C++ achieves the same via new allocation (pointer identity); TLA+ uses serial.
           ser        == GenSerial(t, oldParentW.serial)
           newPkt     == MakePacket(Parent, oldParentW.packet.payload,
                                   oldParentW.packet.sub, TRUE)
           newParentW == PriorityWrapper(newPkt, ser)
       IN
       IF oldParentW.hasPriority /\ linkage[Parent] = oldParentW
       THEN /\ linkage' = [linkage EXCEPT ![Parent] = newParentW]
            /\ UpdateSerial(t, ser)
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_child"]
            /\ UNCHANGED <<local, op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>

\* @c11_action UnbundleCASChild(t):
\*   // Restore child to priority with extracted sub-packet
\*   newsubwrapper = new PacketWrapper(*subpacket, gen(superwrapper->m_bundle_serial));
\*   sublinkage->compareAndSet(bundled_ref, newsubwrapper);
\*   Source: transaction_impl.h:1510-1520
\*
\*   Fidelity note (#8): C++ uses gen(superwrapper->m_bundle_serial) where superwrapper
\*   is the root wrapper after walk (may have higher serial). TLA+ uses oldChildW.serial
\*   (child's bundled_ref serial). Both equal at bundle time; diverge if root was
\*   re-committed. C++ produces higher serial. No correctness impact.
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
            /\ UNCHANGED <<op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>

\* @c11_action CommitDone(t):
\*   Return commit result to caller. Thread-local only.
\*   On success: remove target from childQueue. When childQueue empties, the iteration
\*   is complete: decrement iterBudget. Thread restarts (SnapRead) when iterBudget > 0;
\*   terminates (stays idle) when iterBudget = 0.
CommitDone(t) ==
    /\ pc[t] = "commit_done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ op' = [op EXCEPT ![t] = "idle"]
    /\ LET node     == target[t]
           success  == local[t].commitOk = "ok"
           newQueue == IF success THEN childQueue[t] \ {node} ELSE childQueue[t]
       IN
       /\ target' = [target EXCEPT ![t] = Null]
       /\ local' = [local EXCEPT ![t] = InitLocal]
       /\ childQueue' = [childQueue EXCEPT ![t] = newQueue]
       /\ IF success /\ newQueue = {}
          THEN iterBudget' = [iterBudget EXCEPT ![t] = iterBudget[t] - 1]
          ELSE UNCHANGED iterBudget
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
        \/ CommitParent(t)
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

\* NoSerialWrapAround: all "active" serials must be totally ordered by ModGT.
\* Active serials = thread-local serials + hasPriority node serials.
\* Bundled (hasPriority=FALSE) nodes' serials are excluded: they can go stale
\* without affecting correctness, because their wrappers are compared by full
\* structural equality, not ModGT.
\* When wrap-around makes |a - b| = MaxSerial/2, ModGT(a,b) and ModGT(b,a) are
\* both FALSE — the serial space is exhausted. Increase MaxSerial.
NoSerialWrapAround ==
    LET activeSerials == {serial[t] : t \in Threads}
                         \cup {linkage[n].serial :
                               n \in {m \in Nodes : linkage[m].hasPriority}}
    IN \A a \in activeSerials : \A b \in activeSerials :
        a = b \/ ModGT(a, b) \/ ModGT(b, a)

\* MissingPropagation: mirrors Node<XN>::Packet::checkConsistensy check #4.
\* If parent packet is NOT missing, all its sub-packets must also be NOT missing.
\* (Contrapositive: child.missing => parent.missing)
MissingPropagation ==
    LET pw == linkage[Parent]
    IN  (pw.hasPriority /\ ~pw.packet.missing) =>
        (\A c \in Children :
            pw.packet.sub[c] /= Null => ~pw.packet.sub[c].missing)

Safety == SnapshotConsistency /\ NoPriorityLoss /\ BundleRefConsistency
          /\ NoSerialWrapAround /\ MissingPropagation

\* TerminalPayloadCheck: at termination (all threads: iterBudget=0 and idle),
\* each child received exactly 2 * MaxCommits * |Threads| payload increments:
\*   - MaxCommits * |Threads| from CommitParent (ALL children incremented per iteration)
\*   - MaxCommits * |Threads| from CommitChild (one direct commit per child per iteration)
\* The expected final payload is deterministic, so no tracking variable is needed.
TerminalPayloadCheck ==
    (\A t \in Threads : iterBudget[t] = 0 /\ pc[t] = "idle") =>
        \A c \in Children :
            /\ linkage[c].hasPriority
            /\ linkage[c].packet.payload =
                   (2 * MaxCommits * Cardinality(Threads)) % MaxPayload

=============================================================================
