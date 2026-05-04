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
-------------- MODULE BundleUnbundle_3level_LLfree_dynamic --------------
(*
 * 3-level LL-free bundle/unbundle with dynamic child insertion/release.
 * Combines BundleUnbundle_3level_LLfree.tla (static 3-level tree) with
 * BundleUnbundle_2level_LLfree_dynamic.tla (dynamic children).
 *
 * Tree structure (dynamic):
 *   Grand --+-- Parent --+-- DynChild1  (inserted/released at runtime)
 *                         +-- DynChild2  (inserted/released at runtime)
 *
 * DynChild1 and DynChild2 start unattached (own priority wrappers).
 * InsertThreads perform insert(online=true) for each child.
 * RootThreads do CommitGrand (snapshot Grand, increment discovered leaves, CAS).
 * LeafThreads do CommitChild for each discovered leaf (direct commit).
 * ReleaseThreads release inserted children (after own commits done).
 *
 * Key differences from 3L static spec:
 *   - ChildrenOf(Parent) = ActiveChildren (state-dependent)
 *   - Sub-packet domain for Parent is always AllChildren (with Null for non-active)
 *   - BundlePhase1 for bundleNode=Parent uses 2L-dynamic logic (shrink disjunct)
 *   - BundlePhase1 for bundleNode=Grand uses 3L-static logic (InnerPhase)
 *   - CollectSubpacket uses SubDomainOf (always AllChildren for Parent sub)
 *   - CommitGrand discovers children dynamically from snapshot
 *   - TerminalPayloadCheck/QuiescentCheck use commitCount (like 2L dynamic)
 *   - Threads may be insert-only / release-only (iterBudget=0 initially)
 *   - Waiting action: idle threads stutter while waiting for other threads
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Threads,
    Grand,          \* Grandparent node (static root)
    Parent,         \* Parent node (child of Grand, parent of dynamic children)
    DynChild1,      \* Dynamic leaf node 1 (inserted/released at runtime)
    DynChild2,      \* Dynamic leaf node 2 (inserted/released at runtime)
    Null,
    MaxCommits,     \* Max iterations per thread (RootThreads/LeafThreads)
    BundleCollectAtomic,  \* "coarse" / "fine" / "superfine"
    BundlePhase3Atomic,   \* "coarse" / "fine" / "superfine"
    UnbundleWalkAtomic,   \* "coarse" / "fine" / "superfine"
    UnbundleCASAtomic,    \* "coarse" / "fine"
    Privilege,            \* TRUE: LL-free priority gating active
    RootThreads,          \* \subseteq Threads: threads performing CommitGrand
    LeafThreads,          \* \subseteq Threads: threads performing CommitChild
    InsertThreads,        \* \subseteq Threads: threads performing insert(online=true)
    ReleaseThreads        \* \subseteq Threads: threads performing release

GrandChildren == {Parent}
AllChildren   == {DynChild1, DynChild2}
AllNodes      == {Grand, Parent} \cup AllChildren
InnerNodes    == {Grand, Parent}
LeafNodes     == AllChildren

ASSUME InsertThreads \subseteq Threads /\ InsertThreads /= {}
ASSUME RootThreads \subseteq Threads /\ LeafThreads \subseteq Threads
ASSUME ReleaseThreads \subseteq Threads

ThreadSymmetry == Permutations(Threads)

\* ParentOf: structural parent in the static tree topology
ParentOf(n) ==
    IF n \in AllChildren THEN Parent
    ELSE IF n = Parent THEN Grand
    ELSE Null

-----------------------------------------------------------------------------
(* Dynamic state operators — evaluated in current state via inserted variable *)
-----------------------------------------------------------------------------

\* NOTE: ActiveChildren, ChildrenOf, SubDomainOf reference the 'inserted' state
\* variable and are therefore state-dependent operators.  They are defined after
\* the VARIABLES block but used throughout actions.  In TLA+ this is valid since
\* operator definitions are not sequentially scoped.

-----------------------------------------------------------------------------
(* Variables *)

VARIABLES
    serial,        \* [Threads -> Nat]: per-thread Lamport clock (TID-encoded)
    linkage,       \* [AllNodes -> Wrapper]: per-node atomic PacketWrapper
    pc,            \* [Threads -> String]: program counter
    op,            \* [Threads -> String]: current operation
    target,        \* [Threads -> AllNodes|Null]: CAS target node
    local,         \* [Threads -> Record]: thread-local state
    iterBudget,    \* [Threads -> 0..MaxCommits]: remaining commit iterations
    childQueue,    \* [Threads -> SUBSET AllChildren]: pending per-child commits
    priorityTag,   \* [AllNodes -> Null|<<iter,tid>>]: LL-free per-node tag
    inserted,      \* [AllChildren -> BOOLEAN]: currently inserted
    insertTarget,  \* [Threads -> AllChildren|Null]: child being inserted
    releaseTarget, \* [Threads -> AllChildren|Null]: child being released
    everInserted,  \* [AllChildren -> BOOLEAN]: TRUE once child was ever inserted
    commitCount    \* [AllChildren -> Nat]: successful commits per child

vars == <<serial, linkage, pc, op, target, local, iterBudget, childQueue, priorityTag,
          inserted, insertTarget, releaseTarget, everInserted, commitCount>>

\* Dynamic set of currently inserted children
ActiveChildren == {c \in AllChildren : inserted[c]}

\* State-dependent children of a given node
ChildrenOf(n) ==
    IF n = Grand THEN GrandChildren
    ELSE IF n = Parent THEN ActiveChildren
    ELSE {}

\* Sub-packet domain — always AllChildren for Parent (includes slots for non-active)
SubDomainOf(n) ==
    IF n = Grand THEN GrandChildren
    ELSE IF n = Parent THEN AllChildren
    ELSE {}

EmptySubFor(n) ==
    IF n = Grand THEN [c \in GrandChildren |-> Null]
    ELSE IF n = Parent THEN [c \in AllChildren |-> Null]
    ELSE [c \in {} |-> Null]

-----------------------------------------------------------------------------
(* Lamport serial — C++-faithful TID-encoded base-B arithmetic *)

SerialBase == 1 + Cardinality(Threads)
SerialCounter(s) == s \div SerialBase
SerialTID(s)     == s % SerialBase
EncodeSerial(cnt, tid) == cnt * SerialBase + tid

GenSerial(t, lastSer) ==
    LET lastCnt == SerialCounter(lastSer)
        myCnt   == SerialCounter(serial[t])
        newCnt  == (IF lastCnt > myCnt THEN lastCnt ELSE myCnt) + 1
    IN  EncodeSerial(newCnt, t)

UpdateSerial(t, ser) ==
    serial' = [serial EXCEPT ![t] = ser]

SerialBound == TRUE

-----------------------------------------------------------------------------
(* LL-free negotiate helpers *)

iter(t) == MaxCommits - iterBudget[t]
MyTag(t) == <<iter(t), t>>

TagOlder(a, b) ==
    \/ a[1] < b[1]
    \/ (a[1] = b[1] /\ a[2] < b[2])

CanProceed(t, n) ==
    \/ ~Privilege
    \/ LET tag == priorityTag[n] IN
       \/ tag = Null
       \/ tag /= Null /\ tag[2] = t

TagAfterFail(t, n) ==
    IF ~Privilege
    THEN priorityTag[n]
    ELSE IF priorityTag[n] = Null
         THEN MyTag(t)
         ELSE IF priorityTag[n][2] = t
              THEN MyTag(t)
              ELSE IF TagOlder(MyTag(t), priorityTag[n])
                   THEN MyTag(t)
                   ELSE priorityTag[n]

TagAfterSuccess(t, n) == priorityTag[n]

ClearMyTags(t) ==
    IF ~Privilege
    THEN priorityTag
    ELSE [n \in AllNodes |->
              IF priorityTag[n] /= Null /\ priorityTag[n][2] = t
              THEN Null
              ELSE priorityTag[n]]

-----------------------------------------------------------------------------
(* Data structures *)

MakePacket(node, payload, sub, miss) ==
    [payload |-> payload, sub |-> sub, missing |-> miss, node |-> node]

PriorityWrapper(packet, ser) ==
    [packet |-> packet, hasPriority |-> TRUE, bundledBy |-> Null, serial |-> ser]

BundledRefWrapper(parentNode, ser) ==
    [packet |-> Null, hasPriority |-> FALSE, bundledBy |-> parentNode, serial |-> ser]

\* InsertedRef: wrapper placed on child after insert CAS (non-priority, bundledBy=Parent)
InsertedRef(parentNode, ser, pkt) ==
    [packet |-> pkt, hasPriority |-> FALSE, bundledBy |-> parentNode, serial |-> ser]

-----------------------------------------------------------------------------
(* Init *)

\* Thread-local record.  subwrappers/subpackets/innerSubWs use AllNodes domain
\* so that both Grand-level (subwrappers[Parent]) and Parent-level
\* (subwrappers[DynChild1/2]) bundling share the same record field.
InitLocal == [
    wrapper       |-> Null,
    parentWrapper |-> Null,
    gpWrapper     |-> Null,
    subwrappers   |-> [n \in AllNodes |-> Null],
    subpackets    |-> [n \in AllNodes |-> Null],
    bundleSer     |-> 0,
    bundleNode    |-> Null,
    oldpacket     |-> Null,
    newpacket     |-> Null,
    snapResult    |-> Null,
    commitOk      |-> Null,
    casTargets    |-> <<>>,
    casOldWrappers |-> <<>>,
    casIdx        |-> 0,
    walkNode      |-> Null,
    walkWrapper   |-> Null,
    innerChild    |-> Null,
    innerWrapper  |-> Null,
    innerSubWs    |-> [n \in AllNodes |-> Null]
]

Init ==
    /\ linkage = [n \in AllNodes |->
        IF n = Grand
        THEN PriorityWrapper(
                MakePacket(Grand, 0, [c \in GrandChildren |-> Null], TRUE), 0)
        ELSE IF n = Parent
        THEN PriorityWrapper(
                MakePacket(Parent, 0, [c \in AllChildren |-> Null], FALSE), 0)
        ELSE PriorityWrapper(
                MakePacket(n, 0, [c \in {} |-> Null], FALSE), 0)]
    /\ serial = [t \in Threads |-> EncodeSerial(0, t)]
    /\ pc = [t \in Threads |-> "idle"]
    /\ op = [t \in Threads |-> "idle"]
    /\ target = [t \in Threads |-> Null]
    /\ local = [t \in Threads |-> InitLocal]
    /\ iterBudget = [t \in Threads |->
        IF t \in RootThreads \cup LeafThreads THEN MaxCommits ELSE 0]
    /\ childQueue = [t \in Threads |-> {}]
    /\ priorityTag = [n \in AllNodes |-> Null]
    /\ inserted = [c \in AllChildren |-> FALSE]
    /\ insertTarget = [t \in Threads |-> Null]
    /\ releaseTarget = [t \in Threads |-> Null]
    /\ everInserted = [c \in AllChildren |-> FALSE]
    /\ commitCount = [c \in AllChildren |-> 0]

-----------------------------------------------------------------------------
(* BundleRetryPC: restart target after bundle failure, depends on op *)

BundleRetryPC(t) ==
    IF op[t] = "insert" THEN "insert_snap"
    ELSE IF op[t] = "release" THEN "release_snap"
    ELSE "snap_check"

-----------------------------------------------------------------------------
(* Insert actions *)

InsertStart(t) ==
    /\ pc[t] = "idle"
    /\ t \in InsertThreads
    /\ insertTarget[t] = Null
    /\ releaseTarget[t] = Null
    /\ childQueue[t] = {}
    /\ \E c \in AllChildren :
        /\ ~inserted[c]
        /\ ~everInserted[c]
        /\ \A t2 \in Threads \ {t} : insertTarget[t2] /= c
        /\ insertTarget' = [insertTarget EXCEPT ![t] = c]
        /\ op' = [op EXCEPT ![t] = "insert"]
        /\ pc' = [pc EXCEPT ![t] = "insert_snap"]
    /\ UNCHANGED <<serial, linkage, target, local, priorityTag, inserted,
                   releaseTarget, everInserted, iterBudget, childQueue, commitCount>>

\* ReadParent: unified action for insert_snap / release_snap.
\* Reads Parent wrapper and routes to the appropriate next step.
ReadParent(t) ==
    /\ pc[t] \in {"insert_snap", "release_snap"}
    /\ target' = [target EXCEPT ![t] = Parent]
    /\ LET w == linkage[Parent]
       IN
       IF w.hasPriority /\ ~w.packet.missing
       THEN /\ local' = [local EXCEPT ![t].wrapper = w, ![t].snapResult = w.packet]
            /\ pc' = [pc EXCEPT ![t] =
                  IF op[t] = "insert" THEN "insert_cas_parent"
                  ELSE "release_cas_parent"]
            /\ UNCHANGED <<serial, linkage, op, priorityTag, inserted,
                           insertTarget, releaseTarget, everInserted,
                           iterBudget, childQueue, commitCount>>
       ELSE IF w.hasPriority /\ w.packet.missing
       THEN LET ser == GenSerial(t, w.serial)
            IN
            /\ local' = [local EXCEPT
                   ![t].wrapper     = w,
                   ![t].bundleSer   = ser,
                   ![t].bundleNode  = Parent,
                   ![t].subwrappers = [n \in AllNodes |-> Null],
                   ![t].subpackets  = [n \in AllNodes |-> Null]]
            /\ UpdateSerial(t, ser)
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
            /\ UNCHANGED <<linkage, op, priorityTag, inserted,
                           insertTarget, releaseTarget, everInserted,
                           iterBudget, childQueue, commitCount>>
       ELSE \* bundled elsewhere — retry
            /\ pc' = [pc EXCEPT ![t] = pc[t]]
            /\ UNCHANGED <<serial, linkage, local, op, priorityTag, inserted,
                           insertTarget, releaseTarget, everInserted,
                           iterBudget, childQueue, commitCount>>

InsertCASParent(t) ==
    /\ pc[t] = "insert_cas_parent"
    /\ CanProceed(t, Parent)
    /\ LET oldW == local[t].wrapper
           ser  == GenSerial(t, oldW.serial)
           newW == PriorityWrapper(oldW.packet, ser)
       IN
       IF linkage[Parent] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![Parent] = newW]
            /\ UpdateSerial(t, ser)
            /\ local' = [local EXCEPT ![t].wrapper = newW]
            /\ pc' = [pc EXCEPT ![t] = "insert_read_child"]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterSuccess(t, Parent)]
            /\ UNCHANGED <<op, target, inserted, insertTarget, releaseTarget,
                           everInserted, iterBudget, childQueue, commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "insert_snap"]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
            /\ UNCHANGED <<serial, linkage, op, target, inserted, insertTarget,
                           releaseTarget, everInserted, iterBudget,
                           childQueue, commitCount>>

InsertReadChild(t) ==
    /\ pc[t] = "insert_read_child"
    /\ LET c  == insertTarget[t]
           cw == linkage[c]
       IN
       IF cw.hasPriority
       THEN /\ local' = [local EXCEPT
                  ![t].subwrappers[c] = cw,
                  ![t].subpackets[c]  = cw.packet]
            /\ pc' = [pc EXCEPT ![t] = "insert_cas_child"]
            /\ UNCHANGED <<serial, linkage, op, target, priorityTag, inserted,
                           insertTarget, releaseTarget, everInserted,
                           iterBudget, childQueue, commitCount>>
       ELSE IF cw.bundledBy = Parent
       THEN /\ IF cw.packet /= Null
               THEN local' = [local EXCEPT ![t].subpackets[c] = cw.packet]
               ELSE local' = [local EXCEPT
                        ![t].subpackets[c] = local[t].wrapper.packet.sub[c]]
            /\ pc' = [pc EXCEPT ![t] = "insert_final"]
            /\ UNCHANGED <<serial, linkage, op, target, priorityTag, inserted,
                           insertTarget, releaseTarget, everInserted,
                           iterBudget, childQueue, commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "insert_snap"]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ UNCHANGED <<serial, linkage, op, target, priorityTag, inserted,
                           insertTarget, releaseTarget, everInserted,
                           iterBudget, childQueue, commitCount>>

InsertCASChild(t) ==
    /\ pc[t] = "insert_cas_child"
    /\ LET c     == insertTarget[t]
           oldCW == local[t].subwrappers[c]
           ser   == local[t].wrapper.serial
           cpkt  == local[t].subpackets[c]
       IN
       /\ CanProceed(t, c)
       /\ IF linkage[c] = oldCW
          THEN /\ linkage' = [linkage EXCEPT ![c] = InsertedRef(Parent, ser, cpkt)]
               /\ pc' = [pc EXCEPT ![t] = "insert_final"]
               /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterSuccess(t, c)]
               /\ UNCHANGED <<serial, local, op, target, inserted, insertTarget,
                              releaseTarget, everInserted, iterBudget,
                              childQueue, commitCount>>
          ELSE /\ pc' = [pc EXCEPT ![t] = "insert_snap"]
               /\ local' = [local EXCEPT ![t] = InitLocal]
               /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
               /\ UNCHANGED <<serial, linkage, op, target, inserted, insertTarget,
                              releaseTarget, everInserted, iterBudget,
                              childQueue, commitCount>>

\* InsertFinal: CAS Parent's sub[c] = newCPkt (payload+1 from insert).
\* Sets inserted[c]=TRUE, everInserted[c]=TRUE on success.
InsertFinal(t) ==
    /\ pc[t] = "insert_final"
    /\ CanProceed(t, Parent)
    /\ LET c       == insertTarget[t]
           oldW    == local[t].wrapper
           cpkt    == local[t].subpackets[c]
           newCPkt == MakePacket(c, cpkt.payload + 1, cpkt.sub, cpkt.missing)
           newSub  == [oldW.packet.sub EXCEPT ![c] = newCPkt]
           newPkt  == MakePacket(Parent, oldW.packet.payload, newSub, TRUE)
           ser     == GenSerial(t, oldW.serial)
           newW    == PriorityWrapper(newPkt, ser)
       IN
       IF linkage[Parent] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![Parent] = newW]
            /\ UpdateSerial(t, ser)
            /\ inserted' = [inserted EXCEPT ![c] = TRUE]
            /\ everInserted' = [everInserted EXCEPT ![c] = TRUE]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ pc' = [pc EXCEPT ![t] = "idle"]
            /\ op' = [op EXCEPT ![t] = "idle"]
            /\ target' = [target EXCEPT ![t] = Null]
            /\ insertTarget' = [insertTarget EXCEPT ![t] = Null]
            /\ priorityTag' = ClearMyTags(t)
            /\ UNCHANGED <<releaseTarget, iterBudget, childQueue, commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "insert_snap"]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
            /\ UNCHANGED <<serial, linkage, op, target, inserted, insertTarget,
                           releaseTarget, everInserted, iterBudget,
                           childQueue, commitCount>>

-----------------------------------------------------------------------------
(* Snapshot — entry point for CommitGrand (RootThreads) *)

SnapRead(t) ==
    /\ pc[t] = "idle"
    /\ t \in RootThreads
    /\ iterBudget[t] > 0
    /\ childQueue[t] = {}
    /\ insertTarget[t] = Null
    /\ releaseTarget[t] = Null
    /\ \A c \in AllChildren : everInserted[c]
    /\ op' = [op EXCEPT ![t] = "snapshot"]
    /\ target' = [target EXCEPT ![t] = Grand]
    /\ pc' = [pc EXCEPT ![t] = "snap_check"]
    /\ UNCHANGED <<serial, linkage, local, iterBudget, childQueue,
                   priorityTag, inserted, insertTarget, releaseTarget,
                   everInserted, commitCount>>

SnapCheck(t) ==
    /\ pc[t] = "snap_check"
    /\ LET node == target[t]
           w    == linkage[node]
       IN
       IF w.hasPriority /\ ~w.packet.missing
       THEN \* Fast path: Grand has complete snapshot — proceed to commit
            /\ local' = [local EXCEPT ![t].snapResult = w.packet]
            /\ pc' = [pc EXCEPT ![t] = "commit_grand"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                           priorityTag, inserted, insertTarget, releaseTarget,
                           everInserted, commitCount>>
       ELSE IF w.hasPriority /\ w.packet.missing
       THEN \* Need to bundle Grand
            LET ser == GenSerial(t, w.serial)
            IN
            /\ local' = [local EXCEPT
                   ![t].wrapper     = w,
                   ![t].bundleSer   = ser,
                   ![t].bundleNode  = node,
                   ![t].subwrappers = [n \in AllNodes |-> Null],
                   ![t].subpackets  = [n \in AllNodes |-> Null]]
            /\ UpdateSerial(t, ser)
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
            /\ UNCHANGED <<linkage, op, target, iterBudget, childQueue,
                           priorityTag, inserted, insertTarget, releaseTarget,
                           everInserted, commitCount>>
       ELSE \* Grand is bundled elsewhere — retry
            /\ pc' = [pc EXCEPT ![t] = "snap_check"]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                           priorityTag, inserted, insertTarget, releaseTarget,
                           everInserted, commitCount>>

-----------------------------------------------------------------------------
(* CollectSubpacket — recursive, adapted for dynamic sub-packet domain *)
\*
\* For a leaf child (ChildrenOf = {}): returns packet directly.
\* For an inner child (child=Parent) with missing=FALSE: returns packet directly.
\* For an inner child with missing=TRUE: recursively bundles, using SubDomainOf
\*   to ensure the resulting packet's sub has AllChildren domain.
\* Returns Null if not collectible (disturbed or grandchild collection failed).

RECURSIVE CollectSubpacket(_, _, _, _)
CollectSubpacket(node, child, parentW, bundleSer) ==
    LET cw == linkage[child]
    IN
    IF cw.hasPriority
    THEN IF ~cw.packet.missing
         THEN cw.packet                 \* Complete — use directly
         ELSE \* Child needs recursive bundling (collect its active children).
              \* Uses SubDomainOf(child) as sub domain (AllChildren for child=Parent)
              \* so the resulting packet is consistent with the dynamic scheme.
              LET activeGC  == ChildrenOf(child)   \* = ActiveChildren for child=Parent
                  fullDom   == SubDomainOf(child)   \* = AllChildren for child=Parent
                  gcPkts    == [gc \in fullDom |->
                      IF gc \in activeGC
                      THEN CollectSubpacket(child, gc, cw, bundleSer)
                      ELSE Null]
                  allOk == \A gc \in activeGC : gcPkts[gc] /= Null
              IN
              IF allOk
              THEN MakePacket(child, cw.packet.payload, gcPkts, FALSE)
              ELSE Null
    ELSE IF cw.bundledBy = node
         THEN IF parentW.packet.sub[child] /= Null
                 /\ ~parentW.packet.sub[child].missing
              THEN parentW.packet.sub[child]
              ELSE Null
         ELSE Null

-----------------------------------------------------------------------------
(* Bundle: 4-phase protocol *)
\*
\* BundlePhase1 dispatches on local[t].bundleNode:
\*   bundleNode=Grand:  3L-static logic (child=Parent may need InnerPhase)
\*   bundleNode=Parent: 2L-dynamic logic (dynamic children, shrink disjunct)

BundlePhase1(t) ==
    /\ pc[t] = "bundle_phase1"
    /\ LET node == local[t].bundleNode
       IN
       \* ================================================================
       \* CASE A: bundleNode=Grand — collect Parent (may need InnerPhase)
       \* ================================================================
       IF node = Grand
       THEN
         IF BundleCollectAtomic = "coarse"
         THEN
           LET parentW   == local[t].wrapper
               ser       == local[t].bundleSer
               \* childWs and childPkts have AllNodes domain for uniform EXCEPT ops
               childWs   == [n \in AllNodes |->
                   IF n \in GrandChildren THEN linkage[n] ELSE Null]
               childPkts == [n \in AllNodes |->
                   IF n \in GrandChildren
                   THEN CollectSubpacket(Grand, n, parentW, ser)
                   ELSE Null]
               allCollected == \A c \in GrandChildren : childPkts[c] /= Null
               \* innerBundled: GrandChildren needing inner bundle CAS
               innerBundled == {c \in GrandChildren :
                   childWs[c].hasPriority /\ childWs[c].packet.missing
                   /\ ChildrenOf(c) /= {}}
           IN
           IF allCollected
           THEN /\ local' = [local EXCEPT
                        ![t].subwrappers = [n \in AllNodes |->
                            IF n \in GrandChildren
                            THEN (IF n \in innerBundled
                                  THEN PriorityWrapper(childPkts[n], ser)
                                  ELSE childWs[n])
                            ELSE Null],
                        ![t].subpackets = childPkts]
                /\ linkage' = [n \in AllNodes |->
                       IF \E c \in innerBundled : n \in ChildrenOf(c)
                       THEN BundledRefWrapper(ParentOf(n), ser)
                       ELSE IF n \in innerBundled
                       THEN PriorityWrapper(childPkts[n], ser)
                       ELSE linkage[n]]
                /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
                /\ UNCHANGED <<serial, op, target, iterBudget, childQueue,
                               priorityTag, inserted, insertTarget, releaseTarget,
                               everInserted, commitCount>>
           ELSE \* Can't collect — restart
                /\ pc' = [pc EXCEPT ![t] = "snap_check"]
                /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                               priorityTag, inserted, insertTarget, releaseTarget,
                               everInserted, commitCount>>
         ELSE \* fine/superfine for bundleNode=Grand
           LET parentW == local[t].wrapper
               ser     == local[t].bundleSer
               children == GrandChildren
           IN
           IF BundleCollectAtomic = "superfine" /\ parentW.serial /= ser
           THEN /\ CanProceed(t, Grand)
                /\ LET newW == PriorityWrapper(parentW.packet, ser)
                   IN
                   IF linkage[Grand] = parentW
                   THEN /\ linkage' = [linkage EXCEPT ![Grand] = newW]
                        /\ local' = [local EXCEPT ![t].wrapper = newW]
                        /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                        /\ UNCHANGED <<serial, op, target, iterBudget, childQueue,
                                       priorityTag, inserted, insertTarget, releaseTarget,
                                       everInserted, commitCount>>
                   ELSE /\ pc' = [pc EXCEPT ![t] = "snap_check"]
                        /\ priorityTag' = [priorityTag EXCEPT
                               ![Grand] = TagAfterFail(t, Grand)]
                        /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                                       inserted, insertTarget, releaseTarget,
                                       everInserted, commitCount>>
           ELSE
           LET unprocessed == {c \in children : local[t].subwrappers[c] = Null}
           IN
           IF unprocessed = {}
           THEN /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
                /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                               priorityTag, inserted, insertTarget, releaseTarget,
                               everInserted, commitCount>>
           ELSE \E c \in unprocessed :
                    LET cw   == linkage[c]
                        pkt  == CollectSubpacket(Grand, c, parentW, ser)
                        needsInner == cw.hasPriority /\ cw.packet.missing
                                      /\ ChildrenOf(c) /= {}
                    IN
                    IF pkt = Null
                    THEN IF BundleCollectAtomic = "superfine"
                            /\ linkage[Grand] = parentW
                         THEN /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                              /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget,
                                             childQueue, priorityTag, inserted, insertTarget,
                                             releaseTarget, everInserted, commitCount>>
                         ELSE /\ pc' = [pc EXCEPT ![t] = "snap_check"]
                              /\ local' = [local EXCEPT
                                     ![t].subwrappers = [n \in AllNodes |-> Null],
                                     ![t].subpackets  = [n \in AllNodes |-> Null]]
                              /\ priorityTag' = [priorityTag EXCEPT
                                     ![Grand] = TagAfterFail(t, Grand)]
                              /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                             inserted, insertTarget, releaseTarget,
                                             everInserted, commitCount>>
                    ELSE IF needsInner
                    THEN \* Enter inner bundle phases for child=Parent
                         /\ local' = [local EXCEPT
                                ![t].subpackets[c] = pkt,
                                ![t].innerChild    = c,
                                ![t].innerWrapper  = cw,
                                ![t].innerSubWs    = [n \in AllNodes |->
                                    IF n \in ChildrenOf(c) THEN linkage[n] ELSE Null]]
                         /\ pc' = [pc EXCEPT ![t] = "inner_phase2"]
                         /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                        priorityTag, inserted, insertTarget, releaseTarget,
                                        everInserted, commitCount>>
                    ELSE \* Leaf child of Parent or Parent with no active children
                         /\ local' = [local EXCEPT
                                ![t].subwrappers[c] = cw,
                                ![t].subpackets[c]  = pkt]
                         /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                         /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                        priorityTag, inserted, insertTarget, releaseTarget,
                                        everInserted, commitCount>>
       \* ================================================================
       \* CASE B: bundleNode=Parent — collect dynamic children (no InnerPhase)
       \* ================================================================
       ELSE \* node = Parent
         IF BundleCollectAtomic = "coarse"
         THEN
           LET parentW   == local[t].wrapper
               ser       == local[t].bundleSer
               childWs   == [n \in AllNodes |->
                   IF n \in ActiveChildren THEN linkage[n] ELSE Null]
               childPkts == [n \in AllNodes |->
                   IF n \in ActiveChildren
                   THEN IF linkage[n].hasPriority THEN linkage[n].packet
                        ELSE IF linkage[n].bundledBy = Parent
                             /\ parentW.packet.sub[n] /= Null
                             THEN parentW.packet.sub[n]
                             ELSE Null
                   ELSE Null]
               allCollected == \A c \in ActiveChildren : childPkts[c] /= Null
           IN
           IF ActiveChildren = {}
           THEN /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                /\ UNCHANGED <<serial, linkage, local, op, target, priorityTag,
                               inserted, insertTarget, releaseTarget, everInserted,
                               iterBudget, childQueue, commitCount>>
           ELSE
           \/ IF allCollected
              THEN /\ local' = [local EXCEPT
                           ![t].subwrappers = childWs,
                           ![t].subpackets  = childPkts]
                   /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
                   /\ UNCHANGED <<serial, linkage, op, target, priorityTag,
                                  inserted, insertTarget, releaseTarget, everInserted,
                                  iterBudget, childQueue, commitCount>>
              ELSE /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
                   /\ UNCHANGED <<serial, linkage, local, op, target, priorityTag,
                                  inserted, insertTarget, releaseTarget, everInserted,
                                  iterBudget, childQueue, commitCount>>
           \* Second disjunct: ActiveChildren shrank mid-collection — drop stale entries
           \/ /\ \A c \in ActiveChildren : local[t].subwrappers[c] /= Null
              /\ local' = [local EXCEPT
                       ![t].subwrappers = [n \in AllNodes |->
                           IF n \in ActiveChildren THEN local[t].subwrappers[n] ELSE Null],
                       ![t].subpackets  = [n \in AllNodes |->
                           IF n \in ActiveChildren THEN local[t].subpackets[n] ELSE Null]]
              /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
              /\ UNCHANGED <<serial, linkage, op, target, priorityTag,
                             inserted, insertTarget, releaseTarget, everInserted,
                             iterBudget, childQueue, commitCount>>
         ELSE \* fine/superfine for bundleNode=Parent
           LET parentW == local[t].wrapper
               ser     == local[t].bundleSer
           IN
           IF BundleCollectAtomic = "superfine" /\ parentW.serial /= ser
           THEN /\ CanProceed(t, Parent)
                /\ LET newW == PriorityWrapper(parentW.packet, ser)
                   IN
                   IF linkage[Parent] = parentW
                   THEN /\ linkage' = [linkage EXCEPT ![Parent] = newW]
                        /\ local' = [local EXCEPT ![t].wrapper = newW]
                        /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                        /\ UNCHANGED <<serial, op, target, priorityTag, inserted,
                                       insertTarget, releaseTarget, everInserted,
                                       iterBudget, childQueue, commitCount>>
                   ELSE /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
                        /\ priorityTag' = [priorityTag EXCEPT
                               ![Parent] = TagAfterFail(t, Parent)]
                        /\ UNCHANGED <<serial, linkage, local, op, target, inserted,
                                       insertTarget, releaseTarget, everInserted,
                                       iterBudget, childQueue, commitCount>>
           ELSE
           IF ActiveChildren = {}
           THEN /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                /\ UNCHANGED <<serial, linkage, local, op, target, priorityTag,
                               inserted, insertTarget, releaseTarget, everInserted,
                               iterBudget, childQueue, commitCount>>
           ELSE
           \/ \E c \in ActiveChildren :
                  /\ local[t].subwrappers[c] = Null
                  /\ LET cw   == linkage[c]
                         cpkt == IF cw.hasPriority THEN cw.packet
                                 ELSE IF cw.bundledBy = Parent
                                      /\ parentW.packet.sub[c] /= Null
                                      THEN parentW.packet.sub[c]
                                      ELSE Null
                     IN
                     IF cpkt /= Null
                     THEN /\ local' = [local EXCEPT
                                  ![t].subwrappers[c] = cw,
                                  ![t].subpackets[c]  = cpkt]
                          /\ LET allDone == \A c2 \in ActiveChildren \ {c} :
                                                local[t].subwrappers[c2] /= Null
                             IN
                             IF allDone
                             THEN pc' = [pc EXCEPT ![t] = "bundle_phase2"]
                             ELSE pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                          /\ UNCHANGED <<serial, linkage, op, target, priorityTag,
                                         inserted, insertTarget, releaseTarget, everInserted,
                                         iterBudget, childQueue, commitCount>>
                     ELSE IF BundleCollectAtomic = "superfine"
                             /\ linkage[Parent] = parentW
                          THEN /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                               /\ priorityTag' = [priorityTag EXCEPT
                                      ![c] = TagAfterFail(t, c)]
                               /\ UNCHANGED <<serial, linkage, local, op, target, inserted,
                                              insertTarget, releaseTarget, everInserted,
                                              iterBudget, childQueue, commitCount>>
                          ELSE /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
                               /\ local' = [local EXCEPT
                                       ![t].subwrappers = [n \in AllNodes |-> Null],
                                       ![t].subpackets  = [n \in AllNodes |-> Null]]
                               /\ priorityTag' = [priorityTag EXCEPT
                                      ![Parent] = TagAfterFail(t, Parent)]
                               /\ UNCHANGED <<serial, linkage, op, target, inserted,
                                              insertTarget, releaseTarget, everInserted,
                                              iterBudget, childQueue, commitCount>>
           \* Shrink disjunct (fine mode)
           \/ /\ \A c \in ActiveChildren : local[t].subwrappers[c] /= Null
              /\ local' = [local EXCEPT
                       ![t].subwrappers = [n \in AllNodes |->
                           IF n \in ActiveChildren THEN local[t].subwrappers[n] ELSE Null],
                       ![t].subpackets  = [n \in AllNodes |->
                           IF n \in ActiveChildren THEN local[t].subpackets[n] ELSE Null]]
              /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
              /\ UNCHANGED <<serial, linkage, op, target, priorityTag,
                             inserted, insertTarget, releaseTarget, everInserted,
                             iterBudget, childQueue, commitCount>>

\* InnerPhase2: CAS innerChild (=Parent) with collected sub-packets (missing=TRUE)
InnerPhase2(t) ==
    /\ pc[t] = "inner_phase2"
    /\ LET c    == local[t].innerChild
           oldW == local[t].innerWrapper
           ser  == local[t].bundleSer
           pkt  == local[t].subpackets[c]
           newPkt == MakePacket(c, oldW.packet.payload, pkt.sub, TRUE)
           newW == PriorityWrapper(newPkt, ser)
       IN
       /\ CanProceed(t, c)
       /\ IF linkage[c] = oldW
          THEN /\ linkage' = [linkage EXCEPT ![c] = newW]
               /\ local' = [local EXCEPT ![t].innerWrapper = newW]
               /\ pc' = [pc EXCEPT ![t] = "inner_phase3"]
               /\ UNCHANGED <<serial, op, target, iterBudget, childQueue,
                              priorityTag, inserted, insertTarget, releaseTarget,
                              everInserted, commitCount>>
          ELSE \* Disturbed — restart outer bundle
               /\ LET bundleN == local[t].bundleNode
                  IN
                  local' = [local EXCEPT
                      ![t].wrapper     = Null,
                      ![t].subwrappers = [n \in AllNodes |-> Null],
                      ![t].subpackets  = [n \in AllNodes |-> Null]]
               /\ pc' = [pc EXCEPT ![t] = "snap_check"]
               /\ priorityTag' = [
                      [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                          EXCEPT ![local[t].bundleNode] = TagAfterFail(t, local[t].bundleNode)]
               /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                              inserted, insertTarget, releaseTarget, everInserted, commitCount>>

\* InnerPhase3: CAS each active grandchild to BundledRefWrapper.
\* Uses local[t].innerSubWs to identify grandchildren at InnerPhase entry time
\* (snapshot of ActiveChildren at that moment, not current ActiveChildren).
InnerPhase3(t) ==
    /\ pc[t] = "inner_phase3"
    /\ LET c    == local[t].innerChild
           ser  == local[t].bundleSer
           gcWs == local[t].innerSubWs
           \* gcs: grandchildren active at InnerPhase entry (fixed snapshot)
           gcs  == {gc \in AllChildren : gcWs[gc] /= Null}
       IN
       \* Success: CAS one grandchild to BundledRefWrapper
       \/ \E gc \in gcs :
              /\ CanProceed(t, gc)
              /\ linkage[gc] = gcWs[gc]
              /\ linkage' = [linkage EXCEPT ![gc] = BundledRefWrapper(c, ser)]
              /\ LET allDone == \A gc2 \in gcs \ {gc} :
                                    linkage[gc2] = BundledRefWrapper(c, ser)
                 IN
                 IF allDone
                 THEN pc' = [pc EXCEPT ![t] = "inner_phase4"]
                 ELSE pc' = [pc EXCEPT ![t] = "inner_phase3"]
              /\ local' = [local EXCEPT ![t].innerSubWs[gc] = BundledRefWrapper(c, ser)]
              /\ UNCHANGED <<serial, op, target, iterBudget, childQueue,
                             priorityTag, inserted, insertTarget, releaseTarget,
                             everInserted, commitCount>>
       \* Failure: some grandchild changed — restart outer bundle
       \/ \E gc \in gcs :
              /\ CanProceed(t, gc)
              /\ gcWs[gc] /= Null
              /\ linkage[gc] /= gcWs[gc]
              /\ local' = [local EXCEPT
                     ![t].wrapper     = Null,
                     ![t].subwrappers = [n \in AllNodes |-> Null],
                     ![t].subpackets  = [n \in AllNodes |-> Null]]
              /\ pc' = [pc EXCEPT ![t] = "snap_check"]
              /\ priorityTag' = [
                     [priorityTag EXCEPT ![gc] = TagAfterFail(t, gc)]
                         EXCEPT ![local[t].bundleNode] = TagAfterFail(t, local[t].bundleNode)]
              /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                             inserted, insertTarget, releaseTarget, everInserted, commitCount>>

\* InnerPhase4: finalize inner bundle — set missing=FALSE on innerChild (=Parent)
InnerPhase4(t) ==
    /\ pc[t] = "inner_phase4"
    /\ LET c       == local[t].innerChild
           oldW    == local[t].innerWrapper
           ser     == local[t].bundleSer
           finalPkt == MakePacket(c, oldW.packet.payload, oldW.packet.sub, FALSE)
           finalW  == PriorityWrapper(finalPkt, ser)
       IN
       /\ CanProceed(t, c)
       /\ IF linkage[c] = oldW
          THEN \* Success — inner bundle complete. Resume outer BundlePhase1.
               /\ linkage' = [linkage EXCEPT ![c] = finalW]
               /\ local' = [local EXCEPT
                      ![t].subwrappers[c] = finalW,
                      ![t].innerChild     = Null,
                      ![t].innerWrapper   = Null,
                      ![t].innerSubWs     = [n \in AllNodes |-> Null]]
               /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
               /\ UNCHANGED <<serial, op, target, iterBudget, childQueue,
                              priorityTag, inserted, insertTarget, releaseTarget,
                              everInserted, commitCount>>
          ELSE \* Disturbed — restart outer bundle
               /\ local' = [local EXCEPT
                      ![t].wrapper     = Null,
                      ![t].subwrappers = [n \in AllNodes |-> Null],
                      ![t].subpackets  = [n \in AllNodes |-> Null]]
               /\ pc' = [pc EXCEPT ![t] = "snap_check"]
               /\ priorityTag' = [
                      [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                          EXCEPT ![local[t].bundleNode] = TagAfterFail(t, local[t].bundleNode)]
               /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                              inserted, insertTarget, releaseTarget, everInserted, commitCount>>

BundlePhase2(t) ==
    /\ pc[t] = "bundle_phase2"
    /\ LET node   == local[t].bundleNode
           oldW   == local[t].wrapper
           ser    == local[t].bundleSer
           subs   == local[t].subpackets
           \* For Parent-level bundling, sub domain = AllChildren (with Null for non-active)
           \* For Grand-level bundling, sub domain = GrandChildren
           newSub == IF node = Parent
                     THEN [c \in AllChildren |-> subs[c]]
                     ELSE [c \in GrandChildren |-> subs[c]]
           newPkt == MakePacket(node, oldW.packet.payload, newSub, TRUE)
           newW   == PriorityWrapper(newPkt, ser)
       IN
       /\ CanProceed(t, node)
       /\ IF linkage[node] = oldW
          THEN /\ linkage' = [linkage EXCEPT ![node] = newW]
               /\ local' = [local EXCEPT ![t].wrapper = newW]
               /\ pc' = [pc EXCEPT ![t] = "bundle_phase3"]
               /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterSuccess(t, node)]
               /\ UNCHANGED <<serial, op, target, inserted, insertTarget,
                              releaseTarget, everInserted, iterBudget,
                              childQueue, commitCount>>
          ELSE /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
               /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterFail(t, node)]
               /\ UNCHANGED <<serial, linkage, local, op, target, inserted,
                              insertTarget, releaseTarget, everInserted,
                              iterBudget, childQueue, commitCount>>

BundlePhase3(t) ==
    /\ pc[t] = "bundle_phase3"
    /\ LET node    == local[t].bundleNode
           ser     == local[t].bundleSer
           \* children: GrandChildren for Grand, ActiveChildren for Parent
           children == ChildrenOf(node)
           childWs  == local[t].subwrappers
       IN
       IF BundlePhase3Atomic = "coarse"
       THEN LET allMatch == \A c \in AllNodes :
                    childWs[c] /= Null => linkage[c] = childWs[c]
            IN
            /\ \A n \in AllNodes : childWs[n] /= Null => CanProceed(t, n)
            /\ IF allMatch
               THEN /\ linkage' = [n \in AllNodes |->
                         IF childWs[n] /= Null
                         THEN BundledRefWrapper(node, ser)
                         ELSE linkage[n]]
                    /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                    /\ priorityTag' = [n \in AllNodes |->
                           IF childWs[n] /= Null
                           THEN TagAfterSuccess(t, n)
                           ELSE priorityTag[n]]
                    /\ UNCHANGED <<serial, local, op, target, inserted, insertTarget,
                                   releaseTarget, everInserted, iterBudget,
                                   childQueue, commitCount>>
               ELSE \* Some child changed — regen serial and restart Phase1
                    /\ LET newSer == GenSerial(t, ser)
                       IN
                       /\ local' = [local EXCEPT
                              ![t].bundleSer   = newSer,
                              ![t].subwrappers = [n \in AllNodes |-> Null],
                              ![t].subpackets  = [n \in AllNodes |-> Null]]
                       /\ UpdateSerial(t, newSer)
                    /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                    /\ priorityTag' = [n \in AllNodes |->
                           IF childWs[n] /= Null /\ linkage[n] /= childWs[n]
                           THEN TagAfterFail(t, n)
                           ELSE priorityTag[n]]
                    /\ UNCHANGED <<linkage, op, target, inserted, insertTarget,
                                   releaseTarget, everInserted, iterBudget, childQueue,
                                   commitCount>>
       ELSE \* fine/superfine
            \/ \E c \in AllNodes :
                   /\ childWs[c] /= Null
                   /\ CanProceed(t, c)
                   /\ linkage[c] = childWs[c]
                   /\ linkage' = [linkage EXCEPT ![c] = BundledRefWrapper(node, ser)]
                   /\ LET allDone == \A c2 \in AllNodes :
                                         childWs[c2] /= Null /\ c2 /= c =>
                                             linkage[c2] = BundledRefWrapper(node, ser)
                      IN
                      IF allDone
                      THEN pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                      ELSE pc' = [pc EXCEPT ![t] = "bundle_phase3"]
                   /\ local' = [local EXCEPT ![t].subwrappers[c] = BundledRefWrapper(node, ser)]
                   /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterSuccess(t, c)]
                   /\ UNCHANGED <<serial, op, target, inserted, insertTarget,
                                  releaseTarget, everInserted, iterBudget,
                                  childQueue, commitCount>>
            \/ \E c \in AllNodes :
                   /\ childWs[c] /= Null
                   /\ CanProceed(t, c)
                   /\ linkage[c] /= childWs[c]
                   /\ LET disturbed ==
                           BundlePhase3Atomic = "superfine"
                           /\ (\/ \E c2 \in AllNodes :
                                      /\ childWs[c2] /= Null
                                      /\ linkage[c2] /= childWs[c2]
                                      /\ linkage[c2].serial /= ser
                               \/ linkage[node] /= local[t].wrapper)
                      IN
                      IF disturbed
                      THEN /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
                           /\ local' = [local EXCEPT
                                  ![t].wrapper     = Null,
                                  ![t].subwrappers = [n \in AllNodes |-> Null],
                                  ![t].subpackets  = [n \in AllNodes |-> Null]]
                           /\ priorityTag' = [
                                  [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                                      EXCEPT ![node] = TagAfterFail(t, node)]
                           /\ UNCHANGED <<serial, linkage, op, target, inserted,
                                          insertTarget, releaseTarget, everInserted,
                                          iterBudget, childQueue, commitCount>>
                      ELSE /\ LET newSer == GenSerial(t, ser)
                              IN
                              /\ local' = [local EXCEPT
                                     ![t].bundleSer   = newSer,
                                     ![t].subwrappers = [n \in AllNodes |-> Null],
                                     ![t].subpackets  = [n \in AllNodes |-> Null]]
                              /\ UpdateSerial(t, newSer)
                           /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                           /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                           /\ UNCHANGED <<linkage, op, target, inserted,
                                          insertTarget, releaseTarget, everInserted,
                                          iterBudget, childQueue, commitCount>>

BundlePhase4(t) ==
    /\ pc[t] = "bundle_phase4"
    /\ LET node     == local[t].bundleNode
           oldW     == local[t].wrapper
           ser      == local[t].bundleSer
           \* Reconstruct sub from subpackets with the correct domain
           finalSub == IF node = Parent
                       THEN [c \in AllChildren |-> local[t].subpackets[c]]
                       ELSE [c \in GrandChildren |-> local[t].subpackets[c]]
           finalPkt == MakePacket(node, oldW.packet.payload, finalSub, FALSE)
           finalW   == PriorityWrapper(finalPkt, ser)
       IN
       /\ CanProceed(t, node)
       /\ IF linkage[node] = oldW
          THEN /\ linkage' = [linkage EXCEPT ![node] = finalW]
               /\ local' = [local EXCEPT
                      ![t].wrapper    = finalW,
                      ![t].snapResult = finalPkt]
               /\ pc' = [pc EXCEPT ![t] =
                     IF op[t] = "insert" THEN "insert_cas_parent"
                     ELSE IF op[t] = "release" THEN "release_cas_parent"
                     ELSE "commit_grand"]
               /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterSuccess(t, node)]
               /\ UNCHANGED <<serial, op, target, inserted, insertTarget,
                              releaseTarget, everInserted, iterBudget,
                              childQueue, commitCount>>
          ELSE /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
               /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterFail(t, node)]
               /\ UNCHANGED <<serial, linkage, local, op, target, inserted,
                              insertTarget, releaseTarget, everInserted,
                              iterBudget, childQueue, commitCount>>

-----------------------------------------------------------------------------
(* CommitGrand: CAS Grand with dynamically discovered leaf increments *)
\*
\* Discovers active children from the snapshot: snapChildren = {c ∈ AllChildren :
\*   snapPkt.sub[Parent] /= Null ∧ snapPkt.sub[Parent].sub[c] /= Null}.
\* Increments all discovered children's payloads, updates commitCount.
\* Sets childQueue for LeafThreads (direct per-child commit follows).

CommitGrand(t) ==
    /\ pc[t] = "commit_grand"
    /\ target[t] = Grand
    /\ childQueue[t] = {}
    /\ local[t].snapResult /= Null
    /\ local[t].snapResult.sub[Parent] /= Null
    /\ LET snapPkt      == local[t].snapResult
           parentPkt    == snapPkt.sub[Parent]
           snapChildren == {c \in AllChildren : parentPkt.sub[c] /= Null}
       IN
       IF snapChildren = {}
       THEN \* No active children in snapshot — skip this iteration
            /\ iterBudget' = [iterBudget EXCEPT ![t] = iterBudget[t] - 1]
            /\ pc' = [pc EXCEPT ![t] = "idle"]
            /\ op' = [op EXCEPT ![t] = "idle"]
            /\ target' = [target EXCEPT ![t] = Null]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ priorityTag' = ClearMyTags(t)
            /\ UNCHANGED <<serial, linkage, inserted, insertTarget,
                           releaseTarget, everInserted, childQueue, commitCount>>
       ELSE
       /\ CanProceed(t, Grand)
       /\ LET w            == linkage[Grand]
              newParentSub == [c \in AllChildren |->
                  IF c \in snapChildren
                  THEN MakePacket(c, parentPkt.sub[c].payload + 1,
                                  parentPkt.sub[c].sub, parentPkt.sub[c].missing)
                  ELSE parentPkt.sub[c]]
              newParentPkt == MakePacket(Parent, parentPkt.payload,
                                  newParentSub, parentPkt.missing)
              newGrandSub  == [snapPkt.sub EXCEPT ![Parent] = newParentPkt]
              newPkt       == MakePacket(Grand, snapPkt.payload, newGrandSub, snapPkt.missing)
              ser          == GenSerial(t, w.serial)
              newW         == PriorityWrapper(newPkt, ser)
          IN
          \/ \* CAS success
             /\ w.hasPriority
             /\ w.packet = snapPkt
             /\ linkage' = [linkage EXCEPT ![Grand] = newW]
             /\ UpdateSerial(t, ser)
             /\ commitCount' = [c \in AllChildren |->
                    IF c \in snapChildren
                    THEN commitCount[c] + 1
                    ELSE commitCount[c]]
             /\ IF t \in LeafThreads
                THEN /\ childQueue' = [childQueue EXCEPT ![t] = snapChildren]
                     /\ UNCHANGED iterBudget
                ELSE /\ iterBudget' = [iterBudget EXCEPT ![t] = iterBudget[t] - 1]
                     /\ UNCHANGED childQueue
             /\ pc' = [pc EXCEPT ![t] = "idle"]
             /\ op' = [op EXCEPT ![t] = "idle"]
             /\ target' = [target EXCEPT ![t] = Null]
             /\ local' = [local EXCEPT ![t] = InitLocal]
             /\ priorityTag' = ClearMyTags(t)
             /\ UNCHANGED <<inserted, insertTarget, releaseTarget, everInserted>>
          \/ \* CAS failure
             /\ ~(w.hasPriority /\ w.packet = snapPkt)
             /\ pc' = [pc EXCEPT ![t] = "idle"]
             /\ op' = [op EXCEPT ![t] = "idle"]
             /\ target' = [target EXCEPT ![t] = Null]
             /\ local' = [local EXCEPT ![t] = InitLocal]
             /\ priorityTag' = [priorityTag EXCEPT ![Grand] = TagAfterFail(t, Grand)]
             /\ UNCHANGED <<serial, linkage, iterBudget, childQueue, inserted,
                            insertTarget, releaseTarget, everInserted, commitCount>>

-----------------------------------------------------------------------------
(* CommitChild — for LeafThreads *)

\* BeginChildIteration: leaf-only threads discover current inserted children.
BeginChildIteration(t) ==
    /\ pc[t] = "idle"
    /\ t \in LeafThreads
    /\ t \notin RootThreads
    /\ insertTarget[t] = Null
    /\ releaseTarget[t] = Null
    /\ iterBudget[t] > 0
    /\ childQueue[t] = {}
    /\ \A c \in AllChildren : everInserted[c]
    /\ LET cur == {c \in AllChildren : inserted[c]}
       IN
       /\ cur /= {}
       /\ childQueue' = [childQueue EXCEPT ![t] = cur]
    /\ UNCHANGED <<serial, linkage, pc, op, target, local, iterBudget,
                   priorityTag, inserted, insertTarget, releaseTarget,
                   everInserted, commitCount>>

CommitStart(t, childNode) ==
    /\ pc[t] = "idle"
    /\ childNode \in childQueue[t]
    /\ inserted[childNode]
    /\ op' = [op EXCEPT ![t] = "commit"]
    /\ target' = [target EXCEPT ![t] = childNode]
    /\ pc' = [pc EXCEPT ![t] = "commit_read"]
    /\ UNCHANGED <<serial, linkage, local, iterBudget, childQueue, priorityTag,
                   inserted, insertTarget, releaseTarget, everInserted, commitCount>>

\* CommitSkip: skip a child released between queue setup and commit attempt.
CommitSkip(t, childNode) ==
    /\ pc[t] = "idle"
    /\ childNode \in childQueue[t]
    /\ ~inserted[childNode]
    /\ LET newQueue == childQueue[t] \ {childNode}
       IN
       /\ childQueue' = [childQueue EXCEPT ![t] = newQueue]
       /\ IF newQueue = {}
          THEN /\ iterBudget' = [iterBudget EXCEPT ![t] = iterBudget[t] - 1]
               /\ priorityTag' = ClearMyTags(t)
          ELSE /\ UNCHANGED iterBudget
               /\ UNCHANGED priorityTag
    /\ UNCHANGED <<serial, linkage, pc, op, target, local,
                   inserted, insertTarget, releaseTarget,
                   everInserted, commitCount>>

CommitRead(t) ==
    /\ pc[t] = "commit_read"
    /\ LET node == target[t]
           w    == linkage[node]
       IN
       IF w.hasPriority
       THEN /\ local' = [local EXCEPT
                ![t].wrapper   = w,
                ![t].oldpacket = w.packet,
                ![t].newpacket = MakePacket(node,
                                     w.packet.payload + 1,
                                     w.packet.sub, w.packet.missing)]
            /\ pc' = [pc EXCEPT ![t] = "commit_try_cas"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                           priorityTag, inserted, insertTarget, releaseTarget,
                           everInserted, commitCount>>
       ELSE /\ local' = [local EXCEPT ![t].wrapper = w]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                           priorityTag, inserted, insertTarget, releaseTarget,
                           everInserted, commitCount>>

CommitTryCAS(t) ==
    /\ pc[t] = "commit_try_cas"
    /\ CanProceed(t, target[t])
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
            /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterSuccess(t, node)]
            /\ UNCHANGED <<op, target, iterBudget, childQueue, inserted,
                           insertTarget, releaseTarget, everInserted, commitCount>>
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
                 /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterFail(t, node)]
                 /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                inserted, insertTarget, releaseTarget, everInserted,
                                commitCount>>
            ELSE \* True conflict
                 /\ local' = [local EXCEPT ![t].commitOk = "fail"]
                 /\ pc' = [pc EXCEPT ![t] = "commit_done"]
                 /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterFail(t, node)]
                 /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                inserted, insertTarget, releaseTarget, everInserted,
                                commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterFail(t, node)]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                           inserted, insertTarget, releaseTarget, everInserted,
                           commitCount>>

-----------------------------------------------------------------------------
(* Unbundle — walk up bundledBy chain to find ancestor with priority *)

\* WalkUpChain(node): walk up chain from node to hasPriority root.
RECURSIVE WalkUpChain(_)
WalkUpChain(node) ==
    LET w == linkage[node]
    IN
    IF w.hasPriority
    THEN [status |-> "NODE_MISSING", packet |-> w.packet, root |-> node,
          subpacket |-> Null, wrapper |-> w]
    ELSE IF w.bundledBy = Null
         THEN [status |-> "DISTURBED", packet |-> Null, root |-> Null,
               subpacket |-> Null, wrapper |-> Null]
         ELSE
         LET parentNode == w.bundledBy
             upper == WalkUpChain(parentNode)
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
              IF linkage[node] /= w
              THEN [status |-> "DISTURBED", packet |-> Null, root |-> Null,
                    subpacket |-> Null, wrapper |-> Null]
              ELSE
              IF node \in DOMAIN upperpacket.sub /\ upperpacket.sub[node] /= Null
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

\* SnapshotForUnbundle(node, ser): walk chain + build casTargets (root-first).
\* casOldWrappers is built parallel to casTargets, recording linkage[parentNode]
\* at walk time. UnbundleCASLoop later uses it to verify each casNode hasn't
\* been modified between walk and CAS — without this guard, stale extracted
\* data can overwrite legitimate progress made by other threads.
RECURSIVE SnapshotForUnbundle(_, _)
SnapshotForUnbundle(node, ser) ==
    LET w == linkage[node]
    IN
    IF w.hasPriority
    THEN [status |-> "NODE_MISSING", packet |-> w.packet,
          subpacket |-> Null, casTargets |-> <<>>, casOldWrappers |-> <<>>,
          wrapper |-> w]
    ELSE IF w.bundledBy = Null
         THEN [status |-> "DISTURBED", packet |-> Null,
               subpacket |-> Null, casTargets |-> <<>>, casOldWrappers |-> <<>>,
               wrapper |-> Null]
         ELSE
         LET parentNode == w.bundledBy
             upper == SnapshotForUnbundle(parentNode, ser)
             parentOldW == linkage[parentNode]
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
              IF linkage[node] /= w
              THEN [status |-> "DISTURBED", packet |-> Null,
                    subpacket |-> Null, casTargets |-> <<>>,
                    casOldWrappers |-> <<>>, wrapper |-> Null]
              ELSE
              IF node \in DOMAIN upperpacket.sub /\ upperpacket.sub[node] /= Null
              THEN IF effStatus = "COLLIDED"
                   THEN [status |-> "COLLIDED", packet |-> upperpacket,
                         subpacket |-> upperpacket.sub[node],
                         casTargets |-> upper.casTargets,
                         casOldWrappers |-> upper.casOldWrappers,
                         wrapper |-> upper.wrapper]
                   ELSE IF ser /= 0 /\ linkage[parentNode].serial = ser
                        THEN [status |-> "COLLIDED", packet |-> upperpacket,
                              subpacket |-> upperpacket.sub[node],
                              casTargets |-> upper.casTargets,
                              casOldWrappers |-> upper.casOldWrappers,
                              wrapper |-> upper.wrapper]
                        ELSE LET newTargets == Append(upper.casTargets, parentNode)
                                 newOldWs == Append(upper.casOldWrappers, parentOldW)
                             IN
                             [status |-> "SUCCESS", packet |-> upperpacket,
                              subpacket |-> upperpacket.sub[node],
                              casTargets |-> newTargets,
                              casOldWrappers |-> newOldWs,
                              wrapper |-> upper.wrapper]
              ELSE IF upperpacket.missing
              THEN [status |-> "VOID_PACKET", packet |-> upperpacket,
                    subpacket |-> Null, casTargets |-> <<>>,
                    casOldWrappers |-> <<>>, wrapper |-> upper.wrapper]
              ELSE IF effStatus = "COLLIDED"
                   THEN [status |-> "NODE_MISSING", packet |-> upperpacket,
                         subpacket |-> Null, casTargets |-> upper.casTargets,
                         casOldWrappers |-> upper.casOldWrappers,
                         wrapper |-> upper.wrapper]
                   ELSE IF ser /= 0 /\ linkage[parentNode].serial = ser
                        THEN [status |-> "NODE_MISSING", packet |-> upperpacket,
                              subpacket |-> Null, casTargets |-> upper.casTargets,
                              casOldWrappers |-> upper.casOldWrappers,
                              wrapper |-> upper.wrapper]
                        ELSE LET newTargets == Append(upper.casTargets, parentNode)
                                 newOldWs == Append(upper.casOldWrappers, parentOldW)
                             IN
                             IF ser /= 0 /\ ~w.hasPriority /\ w.serial = ser
                             THEN [status |-> "NODE_MISSING_AND_COLLIDED",
                                   packet |-> upperpacket, subpacket |-> Null,
                                   casTargets |-> newTargets,
                                   casOldWrappers |-> newOldWs,
                                   wrapper |-> upper.wrapper]
                             ELSE [status |-> "NODE_MISSING",
                                   packet |-> upperpacket, subpacket |-> Null,
                                   casTargets |-> newTargets,
                                   casOldWrappers |-> newOldWs,
                                   wrapper |-> upper.wrapper]

UnbundleWalk(t) ==
    /\ pc[t] = "unbundle_walk"
    /\ IF UnbundleWalkAtomic = "coarse"
       THEN LET node   == target[t]
                w      == local[t].wrapper
                parent == w.bundledBy
            IN
            IF parent = Null
            THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                 /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                                priorityTag, inserted, insertTarget, releaseTarget,
                                everInserted, commitCount>>
            ELSE LET result == SnapshotForUnbundle(node, local[t].bundleSer)
                 IN
                 IF result.status \in {"DISTURBED", "COLLIDED"}
                 THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                      /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                                     priorityTag, inserted, insertTarget, releaseTarget,
                                     everInserted, commitCount>>
                 ELSE LET subPkt ==
                            IF result.status \in {"VOID_PACKET", "NODE_MISSING"}
                            THEN w.packet
                            ELSE result.subpacket
                      IN
                      IF subPkt = Null
                      THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                           /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                                          priorityTag, inserted, insertTarget, releaseTarget,
                                          everInserted, commitCount>>
                      ELSE /\ local' = [local EXCEPT
                                ![t].oldpacket  = subPkt,
                                ![t].newpacket  = MakePacket(node,
                                    subPkt.payload + 1, subPkt.sub, subPkt.missing),
                                ![t].casTargets = result.casTargets,
                                ![t].casOldWrappers = result.casOldWrappers,
                                ![t].casIdx     = 1,
                                ![t].walkWrapper = result.wrapper]
                           /\ pc' = [pc EXCEPT ![t] =
                                IF Len(result.casTargets) >= 1
                                THEN "unbundle_cas_loop"
                                ELSE "unbundle_cas_child"]
                           /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                          priorityTag, inserted, insertTarget, releaseTarget,
                                          everInserted, commitCount>>
       ELSE \* fine: one level per action (root-first casTargets construction)
            LET wn == local[t].walkNode
            IN
            IF wn = Null
            THEN /\ local' = [local EXCEPT
                       ![t].walkNode    = target[t],
                       ![t].walkWrapper = linkage[target[t]],
                       ![t].casTargets  = <<>>,
                       ![t].casOldWrappers = <<>>]
                 /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
                 /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                priorityTag, inserted, insertTarget, releaseTarget,
                                everInserted, commitCount>>
            ELSE LET ww    == local[t].walkWrapper
                     pNode == ww.bundledBy
                 IN
                 IF pNode = Null
                 THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                      /\ local' = [local EXCEPT ![t].walkNode = Null]
                      /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                     priorityTag, inserted, insertTarget, releaseTarget,
                                     everInserted, commitCount>>
                 ELSE IF linkage[wn] /= ww
                 THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                      /\ local' = [local EXCEPT ![t].walkNode = Null]
                      /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                     priorityTag, inserted, insertTarget, releaseTarget,
                                     everInserted, commitCount>>
                 ELSE LET pw    == linkage[pNode]
                          \* root-first: prepend so root is index 1
                          newTargets == <<pNode>> \o local[t].casTargets
                          \* parallel: prepend pw as the walk-time wrapper for pNode
                          newOldWs == <<pw>> \o local[t].casOldWrappers
                      IN
                      IF pw.hasPriority
                      THEN LET node   == target[t]
                               w      == linkage[node]
                               result == SnapshotForUnbundle(node, local[t].bundleSer)
                               subPkt == IF result.status \in {"VOID_PACKET", "NODE_MISSING"}
                                         THEN w.packet
                                         ELSE result.subpacket
                           IN
                           IF result.status \in {"DISTURBED", "COLLIDED"} \/ subPkt = Null
                           THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                                /\ local' = [local EXCEPT ![t].walkNode = Null]
                                /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                               priorityTag, inserted, insertTarget, releaseTarget,
                                               everInserted, commitCount>>
                           ELSE /\ local' = [local EXCEPT
                                       ![t].oldpacket  = subPkt,
                                       ![t].newpacket  = MakePacket(node,
                                           subPkt.payload + 1, subPkt.sub, subPkt.missing),
                                       ![t].casTargets = newTargets,
                                       ![t].casOldWrappers = newOldWs,
                                       ![t].casIdx     = 1,
                                       ![t].walkNode   = Null,
                                       ![t].walkWrapper = pw]
                                /\ pc' = [pc EXCEPT ![t] =
                                       IF Len(newTargets) >= 1
                                       THEN "unbundle_cas_loop"
                                       ELSE "unbundle_cas_child"]
                                /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                               priorityTag, inserted, insertTarget, releaseTarget,
                                               everInserted, commitCount>>
                      ELSE /\ local' = [local EXCEPT
                                  ![t].walkNode    = pNode,
                                  ![t].walkWrapper = pw,
                                  ![t].casTargets  = newTargets,
                                  ![t].casOldWrappers = newOldWs]
                           /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
                           /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                          priorityTag, inserted, insertTarget, releaseTarget,
                                          everInserted, commitCount>>

UnbundleCASLoop(t) ==
    /\ pc[t] = "unbundle_cas_loop"
    /\ LET superW    == local[t].walkWrapper
           superNode == IF superW = Null \/ superW.packet = Null THEN Null
                        ELSE superW.packet.node
           superPkt  == IF superW = Null THEN Null ELSE superW.packet
           ExtractAt(n) ==
               IF superPkt = Null THEN Null
               ELSE IF n = superNode THEN superPkt
                    ELSE IF n \in DOMAIN superPkt.sub
                         THEN superPkt.sub[n]
                         ELSE Null
       IN
       IF UnbundleCASAtomic = "coarse"
       THEN LET targets        == local[t].casTargets
                oldWs          == local[t].casOldWrappers
                allCanProceed  == \A i \in 1..Len(targets) : CanProceed(t, targets[i])
                allExtractable == \A i \in 1..Len(targets) : ExtractAt(targets[i]) /= Null
                superFresh     == /\ superNode /= Null
                                  /\ linkage[superNode] = superW
                \* allFresh: each casTarget's linkage still matches the wrapper
                \* observed at walk time. Without this, a stale CAS can overwrite
                \* legitimate updates made by other threads between walk and CAS.
                allFresh       == /\ Len(oldWs) = Len(targets)
                                  /\ \A i \in 1..Len(targets) :
                                         linkage[targets[i]] = oldWs[i]
                ser == GenSerial(t, IF superW = Null THEN 0 ELSE superW.serial)
            IN
            /\ allCanProceed
            /\ IF allExtractable /\ superFresh /\ allFresh
               THEN /\ linkage' = [n \in AllNodes |->
                         IF \E i \in 1..Len(targets) : targets[i] = n
                         THEN LET ext    == ExtractAt(n)
                                  newPkt == MakePacket(n, ext.payload, ext.sub, TRUE)
                              IN  PriorityWrapper(newPkt, ser)
                         ELSE linkage[n]]
                    /\ UpdateSerial(t, ser)
                    /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_child"]
                    /\ UNCHANGED <<local, op, target, iterBudget, childQueue,
                                   priorityTag, inserted, insertTarget, releaseTarget,
                                   everInserted, commitCount>>
               ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                    /\ priorityTag' = [n \in AllNodes |->
                           IF \E i \in 1..Len(targets) : targets[i] = n
                           THEN TagAfterFail(t, n)
                           ELSE priorityTag[n]]
                    /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                                   inserted, insertTarget, releaseTarget, everInserted,
                                   commitCount>>
       ELSE \* fine: one CAS per action
            LET idx        == local[t].casIdx
                casNode    == local[t].casTargets[idx]
                \* oldW: the wrapper observed at walk time, NOT a fresh read.
                \* Verifies the casNode hasn't been modified by another thread
                \* between walk and CAS — without this guard the CAS uses stale
                \* extracted data to overwrite legitimate concurrent updates.
                oldW       == IF idx <= Len(local[t].casOldWrappers)
                              THEN local[t].casOldWrappers[idx]
                              ELSE Null
                extracted  == ExtractAt(casNode)
                ser        == IF oldW = Null THEN 0
                              ELSE GenSerial(t, oldW.serial)
                newPkt     == IF extracted = Null THEN Null
                              ELSE MakePacket(casNode, extracted.payload, extracted.sub, TRUE)
                newW       == IF newPkt = Null \/ oldW = Null THEN Null
                              ELSE PriorityWrapper(newPkt, ser)
                superFresh == /\ superNode /= Null
                              /\ linkage[superNode] = superW
                nextIdx    == idx + 1
                done       == nextIdx > Len(local[t].casTargets)
            IN
            /\ CanProceed(t, casNode)
            /\ IF newW /= Null /\ superFresh /\ linkage[casNode] = oldW
               THEN /\ linkage' = [linkage EXCEPT ![casNode] = newW]
                    /\ UpdateSerial(t, ser)
                    /\ local' = [local EXCEPT
                           ![t].casIdx = nextIdx,
                           ![t].walkWrapper =
                               IF casNode = superNode THEN newW
                               ELSE local[t].walkWrapper]
                    /\ pc' = [pc EXCEPT ![t] =
                         IF done THEN "unbundle_cas_child" ELSE "unbundle_cas_loop"]
                    /\ UNCHANGED <<op, target, iterBudget, childQueue, priorityTag,
                                   inserted, insertTarget, releaseTarget, everInserted,
                                   commitCount>>
               ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                    /\ priorityTag' = [priorityTag EXCEPT ![casNode] = TagAfterFail(t, casNode)]
                    /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                                   inserted, insertTarget, releaseTarget, everInserted,
                                   commitCount>>

UnbundleCASChild(t) ==
    /\ pc[t] = "unbundle_cas_child"
    /\ CanProceed(t, target[t])
    /\ LET node       == target[t]
           oldChildW  == local[t].wrapper
           parentNode == ParentOf(node)
           parentW    == IF parentNode = Null THEN Null ELSE linkage[parentNode]
           ser        == GenSerial(t, oldChildW.serial)
           newChildW  == PriorityWrapper(local[t].newpacket, ser)
           parentSync ==
               /\ parentNode /= Null
               /\ parentW /= Null
               /\ parentW.hasPriority
               /\ parentW.packet /= Null
               /\ node \in DOMAIN parentW.packet.sub
           newParentPkt ==
               IF parentSync
               THEN [parentW.packet EXCEPT !.sub[node] = local[t].newpacket]
               ELSE Null
           newParentW ==
               IF parentSync THEN PriorityWrapper(newParentPkt, ser)
               ELSE Null
       IN
       IF linkage[node] = oldChildW
       THEN /\ linkage' =
                IF parentSync
                THEN [linkage EXCEPT
                          ![node]       = newChildW,
                          ![parentNode] = newParentW]
                ELSE [linkage EXCEPT ![node] = newChildW]
            /\ UpdateSerial(t, ser)
            /\ local' = [local EXCEPT ![t].commitOk = "ok"]
            /\ pc' = [pc EXCEPT ![t] = "commit_done"]
            /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterSuccess(t, node)]
            /\ UNCHANGED <<op, target, iterBudget, childQueue, inserted,
                           insertTarget, releaseTarget, everInserted, commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterFail(t, node)]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                           inserted, insertTarget, releaseTarget, everInserted,
                           commitCount>>

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
       /\ IF success
          THEN /\ commitCount' = [commitCount EXCEPT ![node] = commitCount[node] + 1]
               /\ IF newQueue = {}
                  THEN iterBudget' = [iterBudget EXCEPT ![t] = iterBudget[t] - 1]
                  ELSE UNCHANGED iterBudget
          ELSE /\ UNCHANGED <<iterBudget, commitCount>>
    /\ IF local[t].commitOk = "ok"
       THEN priorityTag' = ClearMyTags(t)
       ELSE UNCHANGED priorityTag
    /\ UNCHANGED <<serial, linkage, inserted, insertTarget, releaseTarget, everInserted>>

-----------------------------------------------------------------------------
(* Release actions — for ReleaseThreads *)

ReleaseStart(t) ==
    /\ pc[t] = "idle"
    /\ t \in ReleaseThreads
    /\ releaseTarget[t] = Null
    /\ insertTarget[t] = Null
    /\ childQueue[t] = {}
    /\ iterBudget[t] = 0
    /\ \A c \in AllChildren : everInserted[c]
    /\ \E c \in AllChildren :
        /\ inserted[c]
        /\ \A t2 \in Threads \ {t} : releaseTarget[t2] /= c
        /\ releaseTarget' = [releaseTarget EXCEPT ![t] = c]
        /\ op' = [op EXCEPT ![t] = "release"]
        /\ pc' = [pc EXCEPT ![t] = "release_snap"]
    /\ UNCHANGED <<serial, linkage, target, local, priorityTag, inserted,
                   insertTarget, everInserted, iterBudget, childQueue, commitCount>>

\* ReleaseSnap is handled by ReadParent (pc = "release_snap")

ReleaseCASParent(t) ==
    /\ pc[t] = "release_cas_parent"
    /\ CanProceed(t, Parent)
    /\ LET c        == releaseTarget[t]
           oldW     == local[t].wrapper
           snapPkt  == local[t].snapResult
           childPkt == snapPkt.sub[c]
           newSub   == [snapPkt.sub EXCEPT ![c] = Null]
           newPkt   == MakePacket(Parent, snapPkt.payload, newSub, snapPkt.missing)
           ser      == GenSerial(t, oldW.serial)
           newW     == PriorityWrapper(newPkt, ser)
       IN
       IF linkage[Parent] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![Parent] = newW]
            /\ UpdateSerial(t, ser)
            /\ inserted' = [inserted EXCEPT ![c] = FALSE]
            /\ local' = [local EXCEPT ![t].oldpacket = childPkt, ![t].wrapper = Null]
            /\ pc' = [pc EXCEPT ![t] = "release_read_child"]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterSuccess(t, Parent)]
            /\ UNCHANGED <<op, target, insertTarget, releaseTarget, everInserted,
                           iterBudget, childQueue, commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "release_snap"]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
            /\ UNCHANGED <<serial, linkage, op, target, inserted, insertTarget,
                           releaseTarget, everInserted, iterBudget,
                           childQueue, commitCount>>

ReleaseReadChild(t) ==
    /\ pc[t] = "release_read_child"
    /\ LET c  == releaseTarget[t]
           cw == linkage[c]
       IN
       IF cw.hasPriority
       THEN \* Child already unbundled — release complete
            /\ pc' = [pc EXCEPT ![t] = "idle"]
            /\ op' = [op EXCEPT ![t] = "idle"]
            /\ target' = [target EXCEPT ![t] = Null]
            /\ releaseTarget' = [releaseTarget EXCEPT ![t] = Null]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ priorityTag' = ClearMyTags(t)
            /\ UNCHANGED <<serial, linkage, inserted, insertTarget, everInserted,
                           iterBudget, childQueue, commitCount>>
       ELSE /\ local' = [local EXCEPT ![t].wrapper = cw]
            /\ pc' = [pc EXCEPT ![t] = "release_cas_child"]
            /\ UNCHANGED <<serial, linkage, op, target, priorityTag, inserted,
                           insertTarget, releaseTarget, everInserted,
                           iterBudget, childQueue, commitCount>>

ReleaseCASChild(t) ==
    /\ pc[t] = "release_cas_child"
    /\ LET c      == releaseTarget[t]
           oldCW  == local[t].wrapper
           cpkt   == local[t].oldpacket
           ser    == GenSerial(t, oldCW.serial)
           newCW  == PriorityWrapper(cpkt, ser)
       IN
       /\ CanProceed(t, c)
       /\ IF linkage[c] = oldCW
          THEN /\ linkage' = [linkage EXCEPT ![c] = newCW]
               /\ UpdateSerial(t, ser)
               /\ pc' = [pc EXCEPT ![t] = "idle"]
               /\ op' = [op EXCEPT ![t] = "idle"]
               /\ target' = [target EXCEPT ![t] = Null]
               /\ releaseTarget' = [releaseTarget EXCEPT ![t] = Null]
               /\ local' = [local EXCEPT ![t] = InitLocal]
               /\ priorityTag' = ClearMyTags(t)
               /\ UNCHANGED <<inserted, insertTarget, everInserted, iterBudget,
                              childQueue, commitCount>>
          ELSE /\ pc' = [pc EXCEPT ![t] = "release_read_child"]
               /\ local' = [local EXCEPT ![t].wrapper = Null]
               /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
               /\ UNCHANGED <<serial, linkage, op, target, inserted, insertTarget,
                              releaseTarget, everInserted, iterBudget,
                              childQueue, commitCount>>

-----------------------------------------------------------------------------
(* SkipIteration — drain remaining budget when all children released *)

SkipIteration(t) ==
    /\ pc[t] = "idle"
    /\ t \in RootThreads \cup LeafThreads
    /\ iterBudget[t] > 0
    /\ childQueue[t] = {}
    /\ insertTarget[t] = Null
    /\ releaseTarget[t] = Null
    /\ \A c \in AllChildren : everInserted[c]
    /\ ~(\E c \in AllChildren : inserted[c])
    /\ iterBudget' = [iterBudget EXCEPT ![t] = 0]
    /\ priorityTag' = ClearMyTags(t)
    /\ UNCHANGED <<serial, linkage, pc, op, target, local,
                   inserted, insertTarget, releaseTarget, everInserted,
                   childQueue, commitCount>>

-----------------------------------------------------------------------------
(* PreemptTag — older thread takes over younger thread's tag *)

PreemptTag(t, n) ==
    /\ Privilege
    /\ priorityTag[n] /= Null
    /\ priorityTag[n][2] /= t
    /\ TagOlder(MyTag(t), priorityTag[n])
    /\ \/ pc[t] /= "idle"
       \/ childQueue[t] /= {}
       \/ insertTarget[t] /= Null
       \/ releaseTarget[t] /= Null
    /\ priorityTag' = [priorityTag EXCEPT ![n] = MyTag(t)]
    /\ UNCHANGED <<serial, linkage, pc, op, target, local, iterBudget, childQueue,
                   inserted, insertTarget, releaseTarget, everInserted, commitCount>>

-----------------------------------------------------------------------------
(* Next-state relation *)

AllDone ==
    /\ \A t \in Threads : pc[t] = "idle"
    /\ \A t \in Threads : insertTarget[t] = Null
    /\ \A t \in Threads : releaseTarget[t] = Null
    /\ \A t \in Threads : childQueue[t] = {}
    /\ \A t \in Threads : iterBudget[t] = 0
    /\ IF ReleaseThreads /= {} THEN \A c \in AllChildren : ~inserted[c]
       ELSE TRUE

\* Waiting: idle thread stutter while system is not yet AllDone.
\* NOT in NextStep — WF does not require progress through this.
Waiting ==
    /\ \E t \in Threads :
        /\ pc[t] = "idle"
        /\ insertTarget[t] = Null
        /\ releaseTarget[t] = Null
        /\ childQueue[t] = {}
    /\ ~AllDone
    /\ UNCHANGED vars

NextStep ==
    \E t \in Threads :
        \/ InsertStart(t)
        \/ ReadParent(t)
        \/ InsertCASParent(t)
        \/ InsertReadChild(t)
        \/ InsertCASChild(t)
        \/ InsertFinal(t)
        \/ SnapRead(t)
        \/ SnapCheck(t)
        \/ BundlePhase1(t)
        \/ InnerPhase2(t)
        \/ InnerPhase3(t)
        \/ InnerPhase4(t)
        \/ BundlePhase2(t)
        \/ BundlePhase3(t)
        \/ BundlePhase4(t)
        \/ CommitGrand(t)
        \/ BeginChildIteration(t)
        \/ \E c \in AllChildren : CommitStart(t, c)
        \/ \E c \in AllChildren : CommitSkip(t, c)
        \/ CommitRead(t)
        \/ CommitTryCAS(t)
        \/ UnbundleWalk(t)
        \/ UnbundleCASLoop(t)
        \/ UnbundleCASChild(t)
        \/ CommitDone(t)
        \/ ReleaseStart(t)
        \/ ReleaseCASParent(t)
        \/ ReleaseReadChild(t)
        \/ ReleaseCASChild(t)
        \/ SkipIteration(t)
        \/ \E n \in AllNodes : PreemptTag(t, n)

Terminating ==
    /\ AllDone
    /\ UNCHANGED vars

Next == NextStep \/ Terminating \/ Waiting

Spec == Init /\ [][Next]_vars /\ WF_vars(NextStep)

-----------------------------------------------------------------------------
(* Safety invariants *)

SnapshotConsistency ==
    \A n \in InnerNodes :
        LET w == linkage[n]
        IN  (w.hasPriority /\ ~w.packet.missing) =>
            \A c \in ChildrenOf(n) : w.packet.sub[c] /= Null

NoPriorityLoss ==
    \A n \in AllNodes \ {Grand} :
        LET w == linkage[n]
        IN  w.hasPriority \/ w.bundledBy /= Null

BundleChainValid ==
    \A n \in AllNodes \ {Grand} :
        LET w == linkage[n]
        IN  (~w.hasPriority /\ w.bundledBy /= Null) =>
            LET pw == linkage[w.bundledBy]
            IN  pw.hasPriority \/ pw.bundledBy /= Null

BundledByCorrect ==
    \A n \in AllNodes \ {Grand} :
        LET w == linkage[n]
        IN  ~w.hasPriority => w.bundledBy = ParentOf(n)

GrandAlwaysPriority ==
    linkage[Grand].hasPriority

MissingPropagation ==
    \A n \in InnerNodes :
        LET w == linkage[n]
        IN  (w.hasPriority /\ ~w.packet.missing) =>
            \A c \in ChildrenOf(n) :
                w.packet.sub[c] /= Null => ~w.packet.sub[c].missing

Safety ==
    /\ SnapshotConsistency
    /\ NoPriorityLoss
    /\ BundleChainValid
    /\ BundledByCorrect
    /\ GrandAlwaysPriority
    /\ MissingPropagation

\* ChildPayload: extract leaf payload from wherever it currently resides
\* (own wrapper, bundled under Parent, or transitively bundled under Grand).
ChildPayload(c) ==
    IF linkage[c].hasPriority
    THEN linkage[c].packet.payload
    ELSE IF linkage[Parent].hasPriority
            /\ linkage[Parent].packet.sub[c] /= Null
         THEN linkage[Parent].packet.sub[c].payload
         ELSE IF linkage[Grand].packet.sub[Parent] /= Null
                 /\ linkage[Grand].packet.sub[Parent].sub[c] /= Null
              THEN linkage[Grand].packet.sub[Parent].sub[c].payload
              ELSE 0   \* well-definedness only; should not occur at AllDone

\* TerminalPayloadCheck: each child's payload = 1 (insert) + commitCount.
\* Equivalent to 2L dynamic: insert bumps payload by 1 (InsertFinal),
\* each CommitGrand/CommitChild bumps commitCount by 1.
TerminalPayloadCheck ==
    AllDone =>
        \A c \in AllChildren :
            /\ everInserted[c]
            /\ ChildPayload(c) = 1 + commitCount[c]

\* QuiescentCheck (DEBUG): payload correctness at every all-idle moment.
\* Uses commitCount (like 2L dynamic) rather than reconstructing from thread state.
QuiescentCheck ==
    (\A t \in Threads : pc[t] = "idle") =>
        \A c \in ActiveChildren :
            ChildPayload(c) = 1 + commitCount[c]

DebugSerialBound == TRUE

PrintTerminalSerial ==
    \/ ~AllDone
    \/ PrintT(<<[t \in Threads |-> SerialCounter(serial[t])]>>)

PrintTerminalMaxCounter ==
    \/ ~AllDone
    \/ LET counts == {SerialCounter(serial[t]) : t \in Threads}
           maxC == CHOOSE m \in counts : \A c \in counts : m >= c
       IN PrintT(maxC)

EventuallyAllDone == <>AllDone

=============================================================================
