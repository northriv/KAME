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
 *   - walkUpChain() / snapshotForUnbundle() walking up through bundledBy chain
 *
 * Serials use modular arithmetic to keep the state space finite.
 *
 * Thread lifecycle (phase variable):
 * Thread lifecycle (MaxCommits iterations per thread):
 *   Each iteration:
 *     1. CommitGrand: snapshot Grand, increment ALL leaf children (2 levels deep), CAS
 *                     (retry until success). Exercises the deepest commit path.
 *     2. CommitChild for EACH child: direct commit (retry until success). Exercises the
 *                     full 2-level unbundle walk triggered by the earlier CommitGrand.
 * Each child receives exactly 2 * MaxCommits * |Threads| increments total.
 * Serial wrap-around is covered when MaxCommits >= MaxSerial / 2.
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Threads,
    Grand,          \* Grandparent node
    Parent,         \* Parent node (child of Grand)
    Child1, Child2, \* Leaf nodes (children of Parent)
    Null,
    MaxPayload,     \* Payloads wrap at this value (e.g. 2)
    MaxSerial,      \* Serial wrap-around (must be even)
    MaxCommits,     \* Max CommitStart/CommitDone cycles per thread in "child" phase
    \* ------------------------------------------------------------
    \* Atomicity granularity for each of 4 bulk-operation sites.
    \* "coarse" = single atomic action (current behavior, matches
    \*            the pure recursive operator)
    \* "fine"   = one load/CAS per action, interleaving allowed
    \*            (models real C++ where each atomic_shared_ptr
    \*            load/CAS is a separate memory event)
    \* ------------------------------------------------------------
    UnbundleWalkAtomic,   \* #1: SnapshotForUnbundle walk (coarse) vs 1 level/action (fine)
    UnbundleCASAtomic,    \* #2: unbundle CAS loop (coarse=all-at-once, fine=1/action)
    BundleCollectAtomic,  \* #3: BundlePhase1 child collection (coarse=all children, fine=1 child/action)
    BundlePhase3Atomic    \* #4: BundlePhase3 child CAS (coarse=all-at-once, fine=1/action)

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
\* 2-level unbundle walk (Child->Parent->Grand via walkUpChain/snapshotForUnbundle recursion),
\* and UnbundleCASGP / UnbundleRestoreParent for the grandparent path.
\* Serial arithmetic is modular (no StateConstraint needed for finiteness).
\*
\* Source: kame/transaction.h, kame/transaction_impl.h
\* ==========================================================================

VARIABLES
    serial, globalSerial, linkage, pc, op, target, local,
    iterBudget,  \* [Threads -> 0..MaxCommits]: remaining full iterations per thread
    childQueue   \* [Threads -> SUBSET ParentChildren]: children pending CommitChild in current iteration

vars == <<serial, globalSerial, linkage, pc, op, target, local, iterBudget, childQueue>>

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
    commitOk    |-> Null,
    casTargets  |-> <<>>,      \* sequence of ancestor nodes for unbundle CAS loop
    casIdx      |-> 0,         \* current index in casTargets loop
    walkNode    |-> Null,      \* (fine UnbundleWalk) current node in chain walk
    walkWrapper |-> Null       \* (fine UnbundleWalk) wrapper saved for walkNode
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
    /\ iterBudget = [t \in Threads |-> MaxCommits]
    /\ childQueue = [t \in Threads |-> {}]

-----------------------------------------------------------------------------
(* Snapshot: read node, bundle if missing *)

\* @c11_action SnapRead(t, node):
\*   Entry for snapshot(node). Only Grand is snapshotted in this model.
\*   Source: transaction_impl.h:982-1000
\*   Guard: iterBudget[t] > 0 /\ childQueue[t] = {} -- start of each iteration.
SnapRead(t, node) ==
    /\ pc[t] = "idle"
    /\ node = Grand
    /\ iterBudget[t] > 0
    /\ childQueue[t] = {}
    /\ op' = [op EXCEPT ![t] = "snapshot"]
    /\ target' = [target EXCEPT ![t] = node]
    /\ pc' = [pc EXCEPT ![t] = "snap_check"]
    /\ UNCHANGED <<serial, globalSerial, linkage, local, iterBudget, childQueue>>

\* @c11_action SnapCheck(t):
\*   local_shared_ptr<PacketWrapper> w(*node->m_link);
\*   if (hasPriority && !missing) -> snap_done   // fast path
\*   if (hasPriority && missing)  -> bundle_phase1  // need bundle
\*   else                         -> retry       // bundled elsewhere
\*   Source: transaction_impl.h:982-1000
SnapCheck(t) ==
    /\ pc[t] = "snap_check"
    /\ LET node == target[t]
           w == linkage[node]
       IN
       IF w.hasPriority /\ ~w.packet.missing
       THEN \* Fast path: complete
            /\ local' = [local EXCEPT ![t].snapResult = w.packet]
            /\ pc' = [pc EXCEPT ![t] = "commit_grand"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
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
            /\ UNCHANGED <<linkage, op, target, iterBudget, childQueue>>
       ELSE \* Node is bundled — need to unbundle first (snapshot from bundled state)
            \* For simplicity, just retry (the real code calls walkUpChain)
            /\ pc' = [pc EXCEPT ![t] = "snap_check"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>

-----------------------------------------------------------------------------
(* Bundle: 4-phase protocol *)

\* --------------------------------------------------------------------------
\* CollectSubpacket(node, child, parentW, bundleSer) — RECURSIVE operator
\* matching C++ bundle_subpacket()
\*
\* For a single child, determines its sub-packet for bundling:
\*   - hasPriority + !missing: use packet directly (fast path)
\*   - hasPriority + missing:  recursively bundle the child first,
\*                              then use the bundled packet
\*   - bundledBy == node:      already bundled here, use parent's sub-packet
\*                              (if complete; otherwise unbundle+rebundle needed → Null)
\*   - bundledBy == other:     bundled elsewhere → Null (caller retries)
\*
\* Returns the child's packet, or Null if not collectible.
\*
\* C++ correspondence: transaction_impl.h bundle_subpacket(), lines 1064-1125
\* --------------------------------------------------------------------------
RECURSIVE CollectSubpacket(_, _, _, _)
CollectSubpacket(node, child, parentW, bundleSer) ==
    LET cw == linkage[child]
    IN
    IF cw.hasPriority
    THEN IF ~cw.packet.missing
         THEN cw.packet                 \* Complete — use directly
         ELSE \* Child needs recursive bundling (collect its children).
              \* For leaf nodes, ChildrenOf returns {} so this trivially succeeds.
              LET grandchildren == ChildrenOf(child)
                  gcPkts == [gc \in grandchildren |->
                      CollectSubpacket(child, gc, cw, bundleSer)]
                  allOk == \A gc \in grandchildren : gcPkts[gc] /= Null
              IN
              IF allOk
              THEN MakePacket(child, cw.packet.payload, gcPkts, FALSE)
              ELSE Null             \* Couldn't collect grandchild; retry
    ELSE IF cw.bundledBy = node
         THEN IF parentW.packet.sub[child] /= Null
                 /\ ~parentW.packet.sub[child].missing
              THEN parentW.packet.sub[child]  \* Already bundled here, complete
              ELSE Null                       \* Bundled here but missing; retry
         ELSE Null                            \* Bundled elsewhere; retry

\* @c11_action BundlePhase1(t):
\*   Phase 1: collect sub-packets using CollectSubpacket for each child.
\*   When a child has hasPriority+missing, C++ recursively calls bundle() on it
\*   (inner bundle), which CASes the child's own children to BundledRefWrapper.
\*   The coarse model performs the inner bundle atomically in the same action:
\*     - grandchildren → BundledRefWrapper(child, bundleSer)
\*     - child → PriorityWrapper(collected_packet, bundleSer)
\*   subwrappers saves the POST-inner-bundle wrapper so Phase3 CAS matches.
\*   Source: transaction_impl.h:1200-1240, bundle() recursion at 1064-1125
\*
\* Granularity: controlled by BundleCollectAtomic
\*   "coarse" = all children loaded & collected in one action (including inner bundle CAS)
\*   "fine"   = one child loaded & collected per action (including inner bundle CAS)
BundlePhase1(t) ==
    /\ pc[t] = "bundle_phase1"
    /\ IF BundleCollectAtomic = "coarse"
       THEN LET node     == local[t].bundleNode
                children  == ChildrenOf(node)
                parentW   == local[t].wrapper
                ser       == local[t].bundleSer
                childWs   == [c \in children |-> linkage[c]]
                childPkts == [c \in children |->
                    CollectSubpacket(node, c, parentW, ser)]
                allCollected == \A c \in children : childPkts[c] /= Null
                \* Children needing inner bundle (recursive bundle in C++)
                innerBundled == {c \in children :
                    childWs[c].hasPriority /\ childWs[c].packet.missing
                    /\ ChildrenOf(c) /= {}}
            IN
            IF allCollected
            THEN /\ local' = [local EXCEPT
                         ![t].subwrappers = [c \in children |->
                             IF c \in innerBundled
                             THEN PriorityWrapper(childPkts[c], ser)
                             ELSE childWs[c]],
                         ![t].subpackets  = childPkts]
                 \* Inner bundle side-effects: CAS grandchildren to BundledRefWrapper,
                 \* finalize inner-bundled children's wrappers
                 /\ linkage' = [n \in AllNodes |->
                       IF \E c \in innerBundled : n \in ChildrenOf(c)
                          /\ linkage[n].hasPriority
                       THEN BundledRefWrapper(ParentOf(n), ser)
                       ELSE IF n \in innerBundled
                       THEN PriorityWrapper(childPkts[n], ser)
                       ELSE linkage[n]]
                 /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
                 /\ UNCHANGED <<serial, globalSerial, op, target, iterBudget, childQueue>>
            ELSE \* Can't collect all — restart
                 /\ pc' = [pc EXCEPT ![t] = "snap_check"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>
       ELSE \* fine: process one child at a time
            LET node     == local[t].bundleNode
                children  == ChildrenOf(node)
                parentW   == local[t].wrapper
                ser       == local[t].bundleSer
                \* Pick an unprocessed child (subwrappers[c] = Null for unprocessed)
                unprocessed == {c \in children : local[t].subwrappers[c] = Null}
            IN
            IF unprocessed = {}
            THEN \* All children collected — proceed to Phase2.
                 /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>
            ELSE \E c \in unprocessed :
                     LET cw   == linkage[c]
                         pkt  == CollectSubpacket(node, c, parentW, ser)
                         needsInner == cw.hasPriority /\ cw.packet.missing
                                       /\ ChildrenOf(c) /= {}
                     IN
                     IF pkt = Null
                     THEN \* Disturbed while collecting this child — restart.
                          /\ pc' = [pc EXCEPT ![t] = "snap_check"]
                          /\ local' = [local EXCEPT
                                 ![t].subwrappers = [cc \in children |-> Null],
                                 ![t].subpackets  = [cc \in children |-> Null]]
                          /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
                     ELSE /\ local' = [local EXCEPT
                                 ![t].subwrappers[c] =
                                     IF needsInner
                                     THEN PriorityWrapper(pkt, ser)
                                     ELSE cw,
                                 ![t].subpackets[c]  = pkt]
                          \* Inner bundle CAS for this child's grandchildren
                          /\ linkage' = [n \in AllNodes |->
                                IF needsInner /\ n \in ChildrenOf(c)
                                   /\ linkage[n].hasPriority
                                THEN BundledRefWrapper(c, ser)
                                ELSE IF needsInner /\ n = c
                                THEN PriorityWrapper(pkt, ser)
                                ELSE linkage[n]]
                          /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]  \* stay, process next
                          /\ UNCHANGED <<serial, globalSerial, op, target, iterBudget, childQueue>>

\* @c11_action BundlePhase2(t):
\*   // Phase 2: CAS bundleNode's linkage with new packet (still missing=TRUE)
\*   bundleNode->m_link->compareAndSet(oldsuperwrapper, superwrapper);
\*   Source: transaction_impl.h:1249-1258
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
            /\ UNCHANGED <<serial, globalSerial, op, target, iterBudget, childQueue>>
       ELSE \* Disturbed
            /\ pc' = [pc EXCEPT ![t] = "snap_check"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>

\* @c11_action BundlePhase3(t):
\*   // Phase 3: CAS each child to bundled-ref, ONE AT A TIME.
\*   // C++ loops: for each child, compareAndSet(subwrappers[i], bundled_ref).
\*   // On failure at child i, rollback children 0..i-1 and restart.
\*   // Modeled as: pick one un-bundled child, CAS it. Repeat until all done.
\*   Source: transaction_impl.h:1260-1282
\*
\* Granularity: controlled by BundlePhase3Atomic
\*   "coarse" = all children CASed at once (single atomic action)
\*   "fine"   = one child per action, interleaving allowed
BundlePhase3(t) ==
    /\ pc[t] = "bundle_phase3"
    /\ IF BundlePhase3Atomic = "coarse"
       THEN LET node     == local[t].bundleNode
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
                 /\ UNCHANGED <<serial, globalSerial, local, op, target, iterBudget, childQueue>>
            ELSE \* Some child changed — restart Phase 1
                 /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>
       ELSE \* fine: one child per action
            LET node     == local[t].bundleNode
                ser      == local[t].bundleSer
                children == ChildrenOf(node)
                childWs  == local[t].subwrappers
            IN
            \* Success path: pick one matching child, CAS it to bundled-ref.
            \/ /\ \E c \in children :
                     /\ linkage[c] = childWs[c]
                     /\ linkage' = [linkage EXCEPT ![c] = BundledRefWrapper(node, ser)]
                     /\ LET allDone == \A c2 \in children \ {c} :
                                           ~linkage[c2].hasPriority
                        IN
                        IF allDone
                        THEN pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                        ELSE pc' = [pc EXCEPT ![t] = "bundle_phase3"]
                     /\ UNCHANGED <<serial, globalSerial, local, op, target, iterBudget, childQueue>>
            \* Failure path: some child changed — rollback and restart.
            \/ /\ \E c \in children :
                     /\ childWs[c] /= Null
                     /\ linkage[c] /= childWs[c]
               /\ linkage' = [n \in AllNodes |->
                      IF n \in children
                         /\ linkage[n] = BundledRefWrapper(node, ser)
                      THEN childWs[n]
                      ELSE linkage[n]]
               /\ local' = [local EXCEPT
                      ![t].subwrappers = [c \in children |-> Null],
                      ![t].subpackets  = [c \in children |-> Null]]
               /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
               /\ UNCHANGED <<serial, globalSerial, op, target, iterBudget, childQueue>>

\* @c11_action BundlePhase4(t):
\*   // Phase 4: finalize -- clear missing flag, CAS bundleNode
\*   superwrapper = new PacketWrapper(*superwrapper, bundle_serial);
\*   newpacket->m_missing = false;  // all sub-packets present
\*   bundleNode->m_link->compareAndSet(oldsuperwrapper, superwrapper);
\*   Source: transaction_impl.h:1286-1299
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
            /\ pc' = [pc EXCEPT ![t] = "commit_grand"]
            /\ UNCHANGED <<serial, globalSerial, op, target, iterBudget, childQueue>>
       ELSE \* Disturbed
            /\ pc' = [pc EXCEPT ![t] = "snap_check"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>

-----------------------------------------------------------------------------
(* Commit on a leaf or inner node *)

\* @c11_action CommitGrand(t):
\*   Commit ALL leaf children's payload changes under Grand's scope (one transaction).
\*   C++: Transaction<XN> tr(Grand); for each c: tr[c].m_x += 1; tr.commit();
\*   Snapshot of Grand bundles Parent+Children; CAS Grand with updated
\*   packet.sub[Parent].sub[c] for ALL c in ParentChildren — deepest commit path.
\*   After success, all children are bundled under Grand, exercising the full 3-level
\*   UnbundleWalk when each child is later committed directly.
\*   Sets childQueue to ParentChildren for subsequent per-child CommitChild cycles.
CommitGrand(t) ==
    /\ pc[t] = "commit_grand"
    /\ target[t] = Grand
    /\ childQueue[t] = {}
    /\ local[t].snapResult /= Null
    /\ local[t].snapResult.sub[Parent] /= Null
    /\ \A c \in ParentChildren : local[t].snapResult.sub[Parent].sub[c] /= Null
    /\ LET w            == linkage[Grand]
           snapPkt      == local[t].snapResult
           parentPkt    == snapPkt.sub[Parent]
           newParentSub == [c \in ParentChildren |->
               MakePacket(c,
                   (parentPkt.sub[c].payload + 1) % MaxPayload,
                   parentPkt.sub[c].sub, parentPkt.sub[c].missing)]
           newParentPkt == MakePacket(Parent, parentPkt.payload,
                               newParentSub, parentPkt.missing)
           newGrandSub  == [snapPkt.sub EXCEPT ![Parent] = newParentPkt]
           newPkt       == MakePacket(Grand, snapPkt.payload,
                               newGrandSub, snapPkt.missing)
           ser          == GenSerial(t, w.serial)
           newW         == PriorityWrapper(newPkt, ser)
       IN
       \/ \* CAS success: commit and move to per-child phase
          /\ w.hasPriority
          /\ w.packet = snapPkt
          /\ linkage' = [linkage EXCEPT ![Grand] = newW]
          /\ UpdateSerial(t, ser)
          /\ childQueue' = [childQueue EXCEPT ![t] = ParentChildren]
          /\ pc' = [pc EXCEPT ![t] = "idle"]
          /\ op' = [op EXCEPT ![t] = "idle"]
          /\ target' = [target EXCEPT ![t] = Null]
          /\ local' = [local EXCEPT ![t] = InitLocal]
          /\ UNCHANGED iterBudget
       \/ \* CAS failure: retry from snapshot (iterate_commit semantics)
          /\ ~(w.hasPriority /\ w.packet = snapPkt)
          /\ pc' = [pc EXCEPT ![t] = "idle"]
          /\ op' = [op EXCEPT ![t] = "idle"]
          /\ target' = [target EXCEPT ![t] = Null]
          /\ local' = [local EXCEPT ![t] = InitLocal]
          /\ UNCHANGED <<serial, globalSerial, linkage, iterBudget, childQueue>>

\* @c11_action CommitStart(t, node):
\*   Entry: Transaction<XN> tr(node);
\*   Source: transaction.h:607-613
\*   Guard: node \in childQueue[t] -- only targets remaining in current iteration.
\*          node \in ParentChildren -- leaf children only (Parent is not targeted here).
CommitStart(t, node) ==
    /\ pc[t] = "idle"
    /\ node \in childQueue[t]
    /\ op' = [op EXCEPT ![t] = "commit"]
    /\ target' = [target EXCEPT ![t] = node]
    /\ pc' = [pc EXCEPT ![t] = "commit_read"]
    /\ UNCHANGED <<serial, globalSerial, linkage, local, iterBudget, childQueue>>

\* @c11_action CommitRead(t):
\*   local_shared_ptr<PacketWrapper> wrapper(*node->m_link);  -- load_shared_
\*   if (wrapper->hasPriority())
\*       -> commit_try_cas       // direct commit path
\*   else
\*       -> unbundle_walk        // bundled, need unbundle first
\*   Source: transaction_impl.h:1364-1420
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
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
       ELSE \* Bundled — need unbundle
            /\ local' = [local EXCEPT ![t].wrapper = w]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
            /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>

\* @c11_action CommitTryCAS(t):
\*   // Direct commit (hasPriority path)
\*   local_shared_ptr<PacketWrapper> newwrapper(
\*       new PacketWrapper(tr.m_packet, tr.m_serial));
\*   if (m_link->compareAndSet(wrapper, newwrapper))
\*       return true;           // success
\*   // payload unchanged -> single-node optimization: adopt new children
\*   // payload changed   -> true conflict, fail
\*   Source: transaction_impl.h:1368-1400
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
            /\ UNCHANGED <<op, target, iterBudget, childQueue>>
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
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
            ELSE \* True conflict
                 /\ local' = [local EXCEPT ![t].commitOk = "fail"]
                 /\ pc' = [pc EXCEPT ![t] = "commit_done"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>

-----------------------------------------------------------------------------
(* Unbundle — recursive walk up the bundledBy chain *)

\* @c11_action UnbundleWalk(t):
\*   // Walk up bundledBy chain via snapshotForUnbundle (recursive).
\*   // Finds sub-packet and builds CAS info list.
\*   Source: transaction_impl.h:1459-1490 (unbundle), 825-960 (walkUpChainImpl/snapshotForUnbundle)

\* --------------------------------------------------------------------------
\* WalkUpChain(node) — RECURSIVE operator matching C++ walkUpChain()
\*
\* Walks up the bundledBy chain from `node` to the root (hasPriority node).
\* Returns a record:
\*   status:      "SUCCESS" | "NODE_MISSING" | "VOID_PACKET" | "DISTURBED"
\*   upperpacket: the packet at the level where the sub-packet was found
\*   subpacket:   the found sub-packet (or Null)
\*   root:        the root node that has priority
\*
\* C++ correspondence: transaction_impl.h walkUpChainImpl(), lines 825-865
\* --------------------------------------------------------------------------
RECURSIVE WalkUpChain(_)
WalkUpChain(node) ==
    LET w == linkage[node]
    IN
    IF w.hasPriority
    THEN \* This node is the root — return its packet
         [status |-> "NODE_MISSING", packet |-> w.packet, root |-> node,
          subpacket |-> Null, wrapper |-> w]
    ELSE IF w.bundledBy = Null
         THEN [status |-> "DISTURBED", packet |-> Null, root |-> Null,
               subpacket |-> Null, wrapper |-> Null]
         ELSE
         \* Recurse up
         LET parentNode == w.bundledBy
             upper == WalkUpChain(parentNode)
         IN
         \* --- Status conversion (C++ switch at line 736) ---
         IF upper.status = "DISTURBED"
         THEN upper
         ELSE LET \* For VOID_PACKET/NODE_MISSING: this parent is the root.
                  \* Use its packet directly (C++: root_wrapper = parent_wrapper).
                  \* For SUCCESS/COLLIDED: recursive call found a sub-packet for
                  \* parentNode in the grandparent's packet. That sub-packet IS
                  \* this parent's packet (C++: parent_packet = *child_subpacket_out
                  \* set by findChildSlot in the recursive call).
                  upperpacket ==
                    IF upper.status \in {"VOID_PACKET", "NODE_MISSING"}
                    THEN linkage[parentNode].packet
                    ELSE upper.subpacket
                  effStatus ==
                    IF upper.status \in {"VOID_PACKET", "NODE_MISSING"}
                    THEN "SUCCESS"
                    ELSE IF upper.status = "NODE_MISSING_AND_COLLIDED"
                    THEN "COLLIDED"
                    ELSE upper.status
              IN
              \* --- Staleness check: *linkage != oldwrapper ---
              IF linkage[node] /= w
              THEN [status |-> "DISTURBED", packet |-> Null, root |-> Null,
                    subpacket |-> Null, wrapper |-> Null]
              ELSE
              \* --- Child slot search (C++ findChildSlot) ---
              IF upperpacket.sub[node] /= Null
              THEN [status |-> effStatus,
                    packet |-> upperpacket,
                    subpacket |-> upperpacket.sub[node],
                    root |-> upper.root, wrapper |-> upper.wrapper]
              ELSE IF upperpacket.missing
              THEN [status |-> "VOID_PACKET",
                    packet |-> upperpacket, subpacket |-> Null,
                    root |-> upper.root, wrapper |-> upper.wrapper]
              ELSE [status |-> "NODE_MISSING",
                    packet |-> upperpacket, subpacket |-> Null,
                    root |-> upper.root, wrapper |-> upper.wrapper]

\* --------------------------------------------------------------------------
\* SnapshotForUnbundle(node, serial) — RECURSIVE operator
\* matching C++ snapshotForUnbundle()
\*
\* Same chain walk as WalkUpChain, plus at each level:
\*   - Serial collision detection
\*   - CAS info construction (modeled as list of ancestors to CAS)
\*
\* Returns: {status, subpacket, casTargets}
\*   casTargets: sequence of ancestor nodes whose linkage needs CAS
\*
\* C++ correspondence: transaction_impl.h snapshotForUnbundle(), lines 888-960
\* --------------------------------------------------------------------------
RECURSIVE SnapshotForUnbundle(_, _)
SnapshotForUnbundle(node, ser) ==
    LET w == linkage[node]
    IN
    IF w.hasPriority
    THEN \* Root reached
         [status |-> "NODE_MISSING", packet |-> w.packet,
          subpacket |-> Null, casTargets |-> <<>>, wrapper |-> w]
    ELSE IF w.bundledBy = Null
         THEN [status |-> "DISTURBED", packet |-> Null,
               subpacket |-> Null, casTargets |-> <<>>, wrapper |-> Null]
         ELSE
         LET parentNode == w.bundledBy
             upper == SnapshotForUnbundle(parentNode, ser)
         IN
         IF upper.status = "DISTURBED"
         THEN upper
         ELSE LET upperpacket ==
                    IF upper.status \in {"VOID_PACKET", "NODE_MISSING"}
                    THEN linkage[parentNode].packet
                    ELSE upper.subpacket
                  effStatus ==
                    IF upper.status \in {"VOID_PACKET", "NODE_MISSING"}
                    THEN "SUCCESS"
                    ELSE IF upper.status = "NODE_MISSING_AND_COLLIDED"
                    THEN "COLLIDED"
                    ELSE upper.status
              IN
              \* --- Staleness check ---
              IF linkage[node] /= w
              THEN [status |-> "DISTURBED", packet |-> Null,
                    subpacket |-> Null, casTargets |-> <<>>, wrapper |-> Null]
              ELSE
              \* --- Child slot search ---
              IF upperpacket.sub[node] /= Null
              THEN \* Found sub-packet
                   IF effStatus = "COLLIDED"
                   THEN [status |-> "COLLIDED", packet |-> upperpacket,
                         subpacket |-> upperpacket.sub[node],
                         casTargets |-> upper.casTargets, wrapper |-> upper.wrapper]
                   ELSE \* Serial collision check (C++ line 795)
                        IF ser /= 0 /\ linkage[parentNode].serial = ser
                        THEN [status |-> "COLLIDED", packet |-> upperpacket,
                              subpacket |-> upperpacket.sub[node],
                              casTargets |-> upper.casTargets, wrapper |-> upper.wrapper]
                        ELSE \* Build CAS info
                             LET newTargets == Append(upper.casTargets, parentNode)
                             IN
                             [status |-> "SUCCESS", packet |-> upperpacket,
                              subpacket |-> upperpacket.sub[node],
                              casTargets |-> newTargets, wrapper |-> upper.wrapper]
              ELSE IF upperpacket.missing
              THEN \* VOID_PACKET: clear cas_infos (C++ line 777)
                   [status |-> "VOID_PACKET", packet |-> upperpacket,
                    subpacket |-> Null, casTargets |-> <<>>, wrapper |-> upper.wrapper]
              ELSE \* NODE_MISSING
                   IF effStatus = "COLLIDED"
                   THEN [status |-> "NODE_MISSING", packet |-> upperpacket,
                         subpacket |-> Null, casTargets |-> upper.casTargets,
                         wrapper |-> upper.wrapper]
                   ELSE \* --- CAS preparation (C++ line 791-829) ---
                        \* Serial collision check
                        IF ser /= 0 /\ linkage[parentNode].serial = ser
                        THEN [status |-> "NODE_MISSING", packet |-> upperpacket,
                              subpacket |-> Null, casTargets |-> upper.casTargets,
                              wrapper |-> upper.wrapper]
                        ELSE \* Build CAS info for this ancestor
                             LET newTargets == Append(upper.casTargets, parentNode)
                             IN
                             \* Check NODE_MISSING_AND_COLLIDED (C++ line 825-828)
                             IF ser /= 0
                                /\ ~w.hasPriority /\ w.serial = ser
                             THEN [status |-> "NODE_MISSING_AND_COLLIDED",
                                   packet |-> upperpacket, subpacket |-> Null,
                                   casTargets |-> newTargets, wrapper |-> upper.wrapper]
                             ELSE [status |-> "NODE_MISSING",
                                   packet |-> upperpacket, subpacket |-> Null,
                                   casTargets |-> newTargets, wrapper |-> upper.wrapper]

\* --------------------------------------------------------------------------
\* UnbundleWalk action — uses SnapshotForUnbundle operator
\* C++ correspondence: unbundle() calling snapshotForUnbundle + status dispatch
\* Source: transaction_impl.h:1459-1490
\*
\* Granularity: controlled by UnbundleWalkAtomic
\*   "coarse" = walk the entire chain atomically (single action)
\*   "fine"   = one level per action (walkNode advances up one level each step)
\* --------------------------------------------------------------------------
UnbundleWalk(t) ==
    /\ pc[t] = "unbundle_walk"
    /\ IF UnbundleWalkAtomic = "coarse"
       THEN \* coarse: whole SnapshotForUnbundle recursion in one action
            LET node    == target[t]
                w       == local[t].wrapper
                parent  == w.bundledBy
            IN
            IF parent = Null
            THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>
            ELSE LET result == SnapshotForUnbundle(node, local[t].bundleSer)
                 IN
                 IF result.status \in {"DISTURBED", "COLLIDED"}
                 THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                      /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>
                 ELSE LET subPkt ==
                            IF result.status \in {"VOID_PACKET", "NODE_MISSING"}
                            THEN w.packet
                            ELSE result.subpacket
                          casTargets == result.casTargets
                      IN
                      IF subPkt = Null
                      THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                           /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>
                      ELSE /\ local' = [local EXCEPT
                                ![t].oldpacket = subPkt,
                                ![t].newpacket = MakePacket(node,
                                    (subPkt.payload + 1) % MaxPayload,
                                    subPkt.sub, subPkt.missing),
                                ![t].casTargets = casTargets,
                                ![t].casIdx = 1]
                           /\ pc' = [pc EXCEPT ![t] =
                                IF Len(casTargets) >= 1
                                THEN "unbundle_cas_loop"
                                ELSE "unbundle_cas_child"]
                           /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
       ELSE \* fine: walk one level per action.
            \* local[t].walkNode: current node in the chain (starts at target[t])
            \* local[t].walkWrapper: wrapper saved for this node
            \* local[t].casTargets: ancestors collected so far
            \* On each step: read linkage[walkNode], check bundledBy,
            \*   move walkNode to parent, append to casTargets.
            \* Terminates when walkNode.parent has priority (root reached) or failure.
            LET wn == local[t].walkNode
            IN
            IF wn = Null
            THEN \* First step: initialize walk from target.
                 /\ local' = [local EXCEPT
                        ![t].walkNode = target[t],
                        ![t].walkWrapper = linkage[target[t]],
                        ![t].casTargets = <<>>]
                 /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]  \* re-enter
                 /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
            ELSE LET ww == local[t].walkWrapper
                     pNode == ww.bundledBy
                 IN
                 IF pNode = Null
                 THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                      /\ local' = [local EXCEPT ![t].walkNode = Null]
                      /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
                 ELSE \* Staleness check: has wn's linkage changed since we loaded ww?
                      IF linkage[wn] /= ww
                      THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                           /\ local' = [local EXCEPT ![t].walkNode = Null]
                           /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
                      ELSE LET pw == linkage[pNode]
                               newTargets == Append(local[t].casTargets, pNode)
                           IN
                           IF pw.hasPriority
                           THEN \* Root reached.
                                LET node    == target[t]
                                    w       == linkage[node]
                                    result  == SnapshotForUnbundle(node, local[t].bundleSer)
                                    subPkt  == IF result.status \in {"VOID_PACKET", "NODE_MISSING"}
                                               THEN w.packet
                                               ELSE result.subpacket
                                IN
                                IF result.status \in {"DISTURBED", "COLLIDED"} \/ subPkt = Null
                                THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                                     /\ local' = [local EXCEPT ![t].walkNode = Null]
                                     /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
                                ELSE /\ local' = [local EXCEPT
                                            ![t].oldpacket = subPkt,
                                            ![t].newpacket = MakePacket(node,
                                                (subPkt.payload + 1) % MaxPayload,
                                                subPkt.sub, subPkt.missing),
                                            ![t].casTargets = newTargets,
                                            ![t].casIdx = 1,
                                            ![t].walkNode = Null]
                                     /\ pc' = [pc EXCEPT ![t] =
                                            IF Len(newTargets) >= 1
                                            THEN "unbundle_cas_loop"
                                            ELSE "unbundle_cas_child"]
                                     /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>
                           ELSE \* Continue walking up.
                                /\ local' = [local EXCEPT
                                       ![t].walkNode = pNode,
                                       ![t].walkWrapper = pw,
                                       ![t].casTargets = newTargets]
                                /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
                                /\ UNCHANGED <<serial, globalSerial, linkage, op, target, iterBudget, childQueue>>

\* @c11_action UnbundleCASLoop(t):
\*   // CAS each ancestor in cas_infos, bottom-up (root first in the list).
\*   // C++: for(auto it = cas_infos.begin(); it != cas_infos.end(); ++it)
\*   //        it->linkage->compareAndSet(it->old_wrapper, it->new_wrapper);
\*   // Each CAS: copy ancestor's packet, set missing=TRUE.
\*   Source: transaction_impl.h:1488-1500, snapshotForUnbundle CAS preparation
\*
\* Granularity: controlled by UnbundleCASAtomic
\*   "coarse" = all CAS at once (single atomic action)
\*   "fine"   = one CAS per action, interleaving allowed
\*
\* TLA+ modeling note: C++ compareAndSet uses POINTER equality (local_shared_ptr identity).
\* A new PacketWrapper allocation is always distinguishable from the old pointer, even when
\* missing is already TRUE (same packet content). In TLA+, records compare by value, so we
\* use GenSerial for a fresh serial, making the new wrapper value-distinct. This prevents
\* BundlePhase4 from CAS-ing over an ancestor-CAS'd wrapper when missing was already TRUE.
UnbundleCASLoop(t) ==
    /\ pc[t] = "unbundle_cas_loop"
    /\ IF UnbundleCASAtomic = "coarse"
       THEN \* All CAS done atomically in one action.
            LET targets == local[t].casTargets
                allOk == \A i \in 1..Len(targets) :
                           LET n == targets[i]
                               w == linkage[n]
                           IN  w.hasPriority
                \* One fresh serial for all ancestor wrappers -- each ancestor's newPkt
                \* contains a different 'node' field, so wrappers remain distinct.
                ser == GenSerial(t, globalSerial)
            IN
            IF allOk
            THEN /\ linkage' = [n \in AllNodes |->
                       IF \E i \in 1..Len(targets) : targets[i] = n
                       THEN LET newPkt == MakePacket(n, linkage[n].packet.payload,
                                                     linkage[n].packet.sub, TRUE)
                            IN  PriorityWrapper(newPkt, ser)
                       ELSE linkage[n]]
                 /\ UpdateSerial(t, ser)
                 /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_child"]
                 /\ UNCHANGED <<local, op, target, iterBudget, childQueue>>
            ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>
       ELSE \* fine: one CAS per action
            LET idx       == local[t].casIdx
                casNode   == local[t].casTargets[idx]
                oldW      == linkage[casNode]
                ser       == GenSerial(t, oldW.serial)
                newPkt    == MakePacket(casNode, oldW.packet.payload,
                                        oldW.packet.sub, TRUE)
                newW      == PriorityWrapper(newPkt, ser)
                nextIdx   == idx + 1
                done      == nextIdx > Len(local[t].casTargets)
            IN
            IF oldW.hasPriority /\ linkage[casNode] = oldW
            THEN /\ linkage' = [linkage EXCEPT ![casNode] = newW]
                 /\ UpdateSerial(t, ser)
                 /\ local' = [local EXCEPT ![t].casIdx = nextIdx]
                 /\ pc' = [pc EXCEPT ![t] =
                      IF done
                      THEN "unbundle_cas_child"
                      ELSE "unbundle_cas_loop"]
                 /\ UNCHANGED <<op, target, iterBudget, childQueue>>
            ELSE \* Disturbed
                 /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                 /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>

\* @c11_action UnbundleCASChild(t):
\*   // Restore child to priority with extracted sub-packet
\*   newsubwrapper = new PacketWrapper(*subpacket, gen(superwrapper->m_bundle_serial));
\*   sublinkage->compareAndSet(bundled_ref, newsubwrapper);
\*   Source: transaction_impl.h:1504-1514
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
            /\ UNCHANGED <<op, target, iterBudget, childQueue>>
       ELSE \* Child changed — retry
            /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, globalSerial, linkage, local, op, target, iterBudget, childQueue>>

\* @c11_action CommitDone(t):
\*   Return commit result to caller. Thread-local only.
\*   On success: remove target from childQueue. When childQueue empties, the iteration
\*   is complete: decrement iterBudget. Thread restarts (SnapRead Grand) when
\*   iterBudget > 0; terminates (stays idle) when iterBudget = 0.
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
        \/ \E n \in InnerNodes : SnapRead(t, n)
        \/ SnapCheck(t)
        \/ BundlePhase1(t)
        \/ BundlePhase2(t)
        \/ BundlePhase3(t)
        \/ BundlePhase4(t)
        \/ CommitGrand(t)
        \/ \E n \in ParentChildren : CommitStart(t, n)
        \/ CommitRead(t)
        \/ CommitTryCAS(t)
        \/ UnbundleWalk(t)
        \/ UnbundleCASLoop(t)
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

\* NoSerialWrapAround: all "active" serials must be totally ordered by ModGT.
\* Active serials = thread-local serials + hasPriority node serials.
\* Bundled (hasPriority=FALSE) nodes' serials are excluded: they can go stale
\* (e.g. grandchildren in 3-level bundling) without affecting correctness,
\* because their wrappers are compared by full structural equality, not ModGT.
\* When wrap-around makes |a - b| = MaxSerial/2, ModGT(a,b) and ModGT(b,a) are
\* both FALSE — the serial space is exhausted. Increase MaxSerial.
NoSerialWrapAround ==
    LET activeSerials == {serial[t] : t \in Threads}
                         \cup {linkage[n].serial :
                               n \in {m \in AllNodes : linkage[m].hasPriority}}
    IN \A a \in activeSerials : \A b \in activeSerials :
        a = b \/ ModGT(a, b) \/ ModGT(b, a)

Safety ==
    /\ SnapshotConsistency
    /\ NoPriorityLoss
    /\ BundleChainValid
    /\ BundledByCorrect
    /\ GrandAlwaysPriority
    /\ MissingPropagation
    /\ NoSerialWrapAround

\* TerminalPayloadCheck: at termination (all threads: iterBudget=0 and idle),
\* each child received exactly 2 * MaxCommits * |Threads| payload increments:
\*   - MaxCommits * |Threads| from CommitGrand (ALL children incremented per iteration)
\*   - MaxCommits * |Threads| from CommitChild (one direct commit per child per iteration)
\* The expected final payload is deterministic, so no tracking variable is needed.
\* Checking per-child (not total sum) catches "commit moved between children" bugs.
TerminalPayloadCheck ==
    (\A t \in Threads : iterBudget[t] = 0 /\ pc[t] = "idle") =>
        \A c \in ParentChildren :
            /\ linkage[c].hasPriority
            /\ linkage[c].packet.payload =
                   (2 * MaxCommits * Cardinality(Threads)) % MaxPayload

=============================================================================
