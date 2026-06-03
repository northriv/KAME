(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************)
----------------------- MODULE BundleUnbundle_2level_LLfree_dynamic -----------------------
(*
 * Dynamic child insertion/release model for KAME's STM bundle/unbundle protocol.
 * Models insert(online_after_insertion=true), configurable CommitParent/CommitChild,
 * and release — all with per-thread role configuration and LL-free negotiate.
 *
 * Tree structure (dynamic):
 *   Parent --+-- DynChild1  (inserted/released at runtime)
 *            +-- DynChild2  (inserted/released at runtime)
 *
 * Both children start unattached (own priority wrappers).
 * Phase 1 — Insert: InsertThreads pick uninserted children and perform
 *   insert(online=true).
 * Phase 2 — Commit + Release (interleaved):
 *   RootThreads do CommitParent (snapshot Parent, increment discovered children, CAS).
 *   LeafThreads do CommitChild for EACH discovered child (direct commit per child).
 *   ReleaseThreads release inserted children (after own commits done).
 *   Children are discovered dynamically from snapshots, not hardcoded.
 *
 * C++ source: kame/transaction_impl.h
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Threads,
    Parent,
    DynChild1, DynChild2,
    Null,
    MaxCommits,
    BundleCollectAtomic,  \* "coarse" / "fine" / "superfine"
    BundlePhase3Atomic,   \* "coarse" / "fine" / "superfine"
    InsertThreads,        \* \subseteq Threads: threads performing insert(online=true)
    RootThreads,          \* \subseteq Threads: threads performing CommitParent
    LeafThreads,          \* \subseteq Threads: threads performing CommitChild
    ReleaseThreads        \* \subseteq Threads: threads performing release

AllChildren == {DynChild1, DynChild2}
Nodes == {Parent} \cup AllChildren

ASSUME InsertThreads \subseteq Threads
ASSUME RootThreads \subseteq Threads /\ LeafThreads \subseteq Threads
ASSUME InsertThreads /= {}
ASSUME ReleaseThreads \subseteq Threads

ThreadSymmetry == Permutations(Threads)

VARIABLES
    serial,        \* [Threads -> Nat]: per-thread Lamport clock
    linkage,       \* [Nodes -> Wrapper]: per-node atomic PacketWrapper
    pc,            \* [Threads -> String]: program counter
    op,            \* [Threads -> String]: current operation
    target,        \* [Threads -> Node|Null]: CAS target node
    local,         \* [Threads -> Record]: thread-local state
    priorityTag,   \* [Nodes -> Null | <<iter, tid>>]: LL-free tag
    inserted,      \* [AllChildren -> BOOLEAN]: currently inserted
    insertTarget,  \* [Threads -> AllChildren|Null]: child being inserted
    releaseTarget, \* [Threads -> AllChildren|Null]: child being released
    everInserted,  \* [AllChildren -> BOOLEAN]: TRUE once child was ever inserted
    iterBudget,    \* [Threads -> 0..MaxCommits]: remaining commit iterations
    childQueue,    \* [Threads -> SUBSET AllChildren]: children remaining in current iteration
    commitCount    \* [AllChildren -> Nat]: successful commits per child

vars == <<serial, linkage, pc, op, target, local, priorityTag,
          inserted, insertTarget, releaseTarget, everInserted,
          iterBudget, childQueue, commitCount>>

-----------------------------------------------------------------------------
(* Data structures *)

PriorityWrapper(packet, ser) ==
    [packet |-> packet, hasPriority |-> TRUE, bundledBy |-> Null, serial |-> ser]

BundledRefWrapper(parentNode, ser) ==
    [packet |-> Null, hasPriority |-> FALSE, bundledBy |-> parentNode, serial |-> ser]

InsertedRef(parentNode, ser, pkt) ==
    [packet |-> pkt, hasPriority |-> FALSE, bundledBy |-> parentNode, serial |-> ser]

MakePacket(node, payload, sub, miss) ==
    [payload |-> payload, sub |-> sub, missing |-> miss, node |-> node]

EmptySub == [c \in AllChildren |-> Null]

ActiveChildren == {c \in AllChildren : inserted[c]}

-----------------------------------------------------------------------------
(* Serial arithmetic *)

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
    LET tag == priorityTag[n] IN
    \/ tag = Null
    \/ tag /= Null /\ tag[2] = t

TagAfterFail(t, n) ==
    IF priorityTag[n] = Null
    THEN MyTag(t)
    ELSE IF priorityTag[n][2] = t
         THEN MyTag(t)
         ELSE IF TagOlder(MyTag(t), priorityTag[n])
              THEN MyTag(t)
              ELSE priorityTag[n]

TagAfterSuccess(t, n) == priorityTag[n]

ClearMyTags(t) ==
    [n \in Nodes |->
        IF priorityTag[n] /= Null /\ priorityTag[n][2] = t
        THEN Null
        ELSE priorityTag[n]]

PreemptTag(t, n) ==
    /\ pc[t] /= "idle" \/ childQueue[t] /= {} \/ insertTarget[t] /= Null
       \/ releaseTarget[t] /= Null
    /\ priorityTag[n] /= Null
    /\ priorityTag[n][2] /= t
    /\ TagOlder(MyTag(t), priorityTag[n])
    /\ priorityTag' = [priorityTag EXCEPT ![n] = MyTag(t)]
    /\ UNCHANGED <<serial, linkage, pc, op, target, local, inserted,
                   insertTarget, releaseTarget, everInserted, iterBudget,
                   childQueue, commitCount>>

-----------------------------------------------------------------------------
(* Init *)

InitLocal == [
    wrapper       |-> Null,
    subwrappers   |-> [c \in AllChildren |-> Null],
    subpackets    |-> EmptySub,
    bundleSer     |-> 0,
    snapResult    |-> Null,
    oldpacket     |-> Null,
    newpacket     |-> Null,
    parentWrapper |-> Null,
    commitOk      |-> Null
]

Init ==
    /\ linkage = [n \in Nodes |->
        IF n = Parent
        THEN PriorityWrapper(MakePacket(Parent, 0, EmptySub, FALSE), 0)
        ELSE PriorityWrapper(MakePacket(n, 0, EmptySub, FALSE), 0)]
    /\ serial = [t \in Threads |-> EncodeSerial(0, t)]
    /\ pc = [t \in Threads |-> "idle"]
    /\ op = [t \in Threads |-> "idle"]
    /\ target = [t \in Threads |-> Null]
    /\ local = [t \in Threads |-> InitLocal]
    /\ priorityTag = [n \in Nodes |-> Null]
    /\ inserted = [c \in AllChildren |-> FALSE]
    /\ insertTarget = [t \in Threads |-> Null]
    /\ releaseTarget = [t \in Threads |-> Null]
    /\ everInserted = [c \in AllChildren |-> FALSE]
    /\ iterBudget = [t \in Threads |->
        IF t \in RootThreads \cup LeafThreads THEN MaxCommits ELSE 0]
    /\ childQueue = [t \in Threads |-> {}]
    /\ commitCount = [c \in AllChildren |-> 0]

-----------------------------------------------------------------------------
(* Bundle retry target — depends on current operation *)

BundleRetryPC(t) ==
    IF op[t] = "insert" THEN "insert_snap"
    ELSE IF op[t] = "release" THEN "release_snap"
    ELSE "snap_read"

-----------------------------------------------------------------------------
(* Insert actions — only for InsertThreads *)

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

-----------------------------------------------------------------------------
(* ReadParent — unified action for insert_snap / snap_read / release_snap.
   Reads Parent's wrapper and routes to the appropriate next step. *)

ReadParent(t) ==
    /\ pc[t] \in {"insert_snap", "snap_read", "release_snap"}
    /\ target' = [target EXCEPT ![t] = Parent]
    /\ LET w == linkage[Parent]
       IN
       IF w.hasPriority /\ ~w.packet.missing
       THEN /\ local' = [local EXCEPT ![t].wrapper = w, ![t].snapResult = w.packet]
            /\ pc' = [pc EXCEPT ![t] =
                  IF op[t] = "insert" THEN "insert_cas_parent"
                  ELSE IF op[t] = "release" THEN "release_cas_parent"
                  ELSE "commit_parent"]
            /\ UNCHANGED <<serial, linkage, op, priorityTag, inserted,
                           insertTarget, releaseTarget, everInserted,
                           iterBudget, childQueue, commitCount>>
       ELSE IF w.hasPriority /\ w.packet.missing
       THEN LET ser == GenSerial(t, w.serial)
            IN
            /\ local' = [local EXCEPT
                   ![t].wrapper = w,
                   ![t].bundleSer = ser,
                   ![t].subwrappers = [c \in AllChildren |-> Null],
                   ![t].subpackets = EmptySub]
            /\ UpdateSerial(t, ser)
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
            /\ UNCHANGED <<linkage, op, priorityTag, inserted,
                           insertTarget, releaseTarget, everInserted,
                           iterBudget, childQueue, commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = pc[t]]
            /\ UNCHANGED <<serial, linkage, local, op, priorityTag, inserted,
                           insertTarget, releaseTarget, everInserted,
                           iterBudget, childQueue, commitCount>>

-----------------------------------------------------------------------------
(* Snapshot for CommitParent — entry point for RootThreads *)

SnapRead(t) ==
    /\ pc[t] = "idle"
    /\ t \in RootThreads
    /\ iterBudget[t] > 0
    /\ childQueue[t] = {}
    /\ insertTarget[t] = Null
    /\ releaseTarget[t] = Null
    /\ \A c \in AllChildren : everInserted[c]
    /\ op' = [op EXCEPT ![t] = "snapshot"]
    /\ target' = [target EXCEPT ![t] = Parent]
    /\ pc' = [pc EXCEPT ![t] = "snap_read"]
    /\ UNCHANGED <<serial, linkage, local, iterBudget, childQueue,
                   priorityTag, inserted, insertTarget, releaseTarget,
                   everInserted, commitCount>>

-----------------------------------------------------------------------------
(* Bundle actions — shared by insert, CommitParent, and release paths.
   Retry target determined by BundleRetryPC(t) based on op[t]. *)

BundlePhase1(t) ==
    /\ pc[t] = "bundle_phase1"
    /\ IF BundleCollectAtomic = "coarse"
       THEN LET childWrappers == [c \in AllChildren |->
                    IF c \in ActiveChildren THEN linkage[c] ELSE Null]
                parentW == local[t].wrapper
                childPackets == [c \in AllChildren |->
                    IF c \notin ActiveChildren THEN Null
                    ELSE IF childWrappers[c].hasPriority
                    THEN childWrappers[c].packet
                    ELSE IF childWrappers[c].bundledBy = Parent
                         THEN IF parentW.packet.sub[c] /= Null
                              THEN parentW.packet.sub[c]
                              ELSE Null
                         ELSE Null]
                allCollected == \A c \in ActiveChildren : childPackets[c] /= Null
            IN
            IF ActiveChildren = {}
            THEN /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                 /\ UNCHANGED <<serial, linkage, local, op, target, priorityTag,
                                inserted, insertTarget, releaseTarget, everInserted,
                                iterBudget, childQueue, commitCount>>
            ELSE IF allCollected
            THEN /\ local' = [local EXCEPT
                         ![t].subwrappers = childWrappers,
                         ![t].subpackets  = childPackets]
                 /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
                 /\ UNCHANGED <<serial, linkage, op, target, priorityTag,
                                inserted, insertTarget, releaseTarget, everInserted,
                                iterBudget, childQueue, commitCount>>
            ELSE /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
                 /\ UNCHANGED <<serial, linkage, local, op, target, priorityTag,
                                inserted, insertTarget, releaseTarget, everInserted,
                                iterBudget, childQueue, commitCount>>
       ELSE LET parentW == local[t].wrapper
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
                         /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterSuccess(t, Parent)]
                         /\ UNCHANGED <<serial, op, target, inserted, insertTarget,
                                        releaseTarget, everInserted, iterBudget,
                                        childQueue, commitCount>>
                    ELSE /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
                         /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
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
                               /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                               /\ UNCHANGED <<serial, linkage, local, op, target, inserted,
                                              insertTarget, releaseTarget, everInserted,
                                              iterBudget, childQueue, commitCount>>
                          ELSE /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
                               /\ local' = [local EXCEPT
                                       ![t].subwrappers = [c2 \in AllChildren |-> Null],
                                       ![t].subpackets  = EmptySub]
                               /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
                               /\ UNCHANGED <<serial, linkage, op, target, inserted,
                                              insertTarget, releaseTarget, everInserted,
                                              iterBudget, childQueue, commitCount>>
            \* All active children already have subwrappers, but ActiveChildren shrank
            \* mid-collection (a child was released). Drop stale released-child entries
            \* and advance to bundle_phase2 to complete the bundle with surviving children.
            \/ /\ \A c \in ActiveChildren : local[t].subwrappers[c] /= Null
               /\ local' = [local EXCEPT
                        ![t].subwrappers = [c2 \in AllChildren |->
                            IF c2 \in ActiveChildren THEN local[t].subwrappers[c2] ELSE Null],
                        ![t].subpackets  = [c2 \in AllChildren |->
                            IF c2 \in ActiveChildren THEN local[t].subpackets[c2] ELSE Null]]
               /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
               /\ UNCHANGED <<serial, linkage, op, target, priorityTag,
                              inserted, insertTarget, releaseTarget, everInserted,
                              iterBudget, childQueue, commitCount>>

BundlePhase2(t) ==
    /\ pc[t] = "bundle_phase2"
    /\ CanProceed(t, Parent)
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
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterSuccess(t, Parent)]
            /\ UNCHANGED <<serial, op, target, inserted, insertTarget,
                           releaseTarget, everInserted, iterBudget,
                           childQueue, commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
            /\ UNCHANGED <<serial, linkage, local, op, target, inserted,
                           insertTarget, releaseTarget, everInserted,
                           iterBudget, childQueue, commitCount>>

BundlePhase3(t) ==
    /\ pc[t] = "bundle_phase3"
    /\ LET ser     == local[t].bundleSer
       IN
       IF BundlePhase3Atomic = "coarse"
       THEN /\ \A c \in AllChildren : local[t].subwrappers[c] /= Null
                   => CanProceed(t, c)
            /\ LET allMatch == \A c \in AllChildren :
                       local[t].subwrappers[c] = Null \/ linkage[c] = local[t].subwrappers[c]
               IN
               IF allMatch
               THEN /\ linkage' = [n \in Nodes |->
                        IF n \in AllChildren /\ local[t].subwrappers[n] /= Null
                        THEN BundledRefWrapper(Parent, ser)
                        ELSE linkage[n]]
                    /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                    /\ priorityTag' = [n \in Nodes |->
                           IF n \in AllChildren /\ local[t].subwrappers[n] /= Null
                           THEN TagAfterSuccess(t, n)
                           ELSE priorityTag[n]]
                    /\ UNCHANGED <<serial, local, op, target, inserted, insertTarget,
                                   releaseTarget, everInserted, iterBudget,
                                   childQueue, commitCount>>
               ELSE /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                    /\ local' = [local EXCEPT
                           ![t].subwrappers = [c \in AllChildren |-> Null],
                           ![t].subpackets  = EmptySub]
                    /\ priorityTag' = [n \in Nodes |->
                           IF n \in AllChildren /\ local[t].subwrappers[n] /= Null
                              /\ linkage[n] /= local[t].subwrappers[n]
                           THEN TagAfterFail(t, n)
                           ELSE priorityTag[n]]
                    /\ UNCHANGED <<serial, linkage, op, target, inserted, insertTarget,
                                   releaseTarget, everInserted, iterBudget,
                                   childQueue, commitCount>>
       ELSE \/ \E c \in AllChildren :
                  /\ local[t].subwrappers[c] /= Null
                  /\ CanProceed(t, c)
                  /\ linkage[c] = local[t].subwrappers[c]
                  /\ linkage' = [linkage EXCEPT ![c] = BundledRefWrapper(Parent, ser)]
                  /\ LET allDone == \A c2 \in AllChildren \ {c} :
                             local[t].subwrappers[c2] = Null
                             \/ linkage[c2] = BundledRefWrapper(Parent, ser)
                     IN
                     IF allDone
                     THEN pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                     ELSE pc' = [pc EXCEPT ![t] = "bundle_phase3"]
                  /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterSuccess(t, c)]
                  /\ UNCHANGED <<serial, local, op, target, inserted, insertTarget,
                                 releaseTarget, everInserted, iterBudget,
                                 childQueue, commitCount>>
            \/ \E c \in AllChildren :
                  /\ local[t].subwrappers[c] /= Null
                  /\ CanProceed(t, c)
                  /\ linkage[c] /= local[t].subwrappers[c]
                  /\ LET disturbed ==
                          BundlePhase3Atomic = "superfine"
                          /\ (\/ \E c2 \in AllChildren :
                                     /\ local[t].subwrappers[c2] /= Null
                                     /\ linkage[c2] /= local[t].subwrappers[c2]
                                     /\ linkage[c2].serial /= ser
                               \/ linkage[Parent] /= local[t].wrapper)
                     IN
                     IF disturbed
                     THEN /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
                          /\ local' = [local EXCEPT
                                 ![t].wrapper = Null,
                                 ![t].subwrappers = [c2 \in AllChildren |-> Null],
                                 ![t].subpackets = EmptySub]
                          /\ priorityTag' = [
                                 [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                                     EXCEPT ![Parent] = TagAfterFail(t, Parent)]
                          /\ UNCHANGED <<serial, linkage, op, target, inserted,
                                         insertTarget, releaseTarget, everInserted,
                                         iterBudget, childQueue, commitCount>>
                     ELSE /\ local' = [local EXCEPT
                                 ![t].subwrappers = [c2 \in AllChildren |-> Null],
                                 ![t].subpackets  = EmptySub]
                          /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                          /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                          /\ UNCHANGED <<serial, linkage, op, target, inserted,
                                         insertTarget, releaseTarget, everInserted,
                                         iterBudget, childQueue, commitCount>>

\* BundlePhase4: clear missing flag. On success, route depends on op:
\*   insert  → insert_cas_parent
\*   release → release_cas_parent
\*   snapshot → commit_parent
BundlePhase4(t) ==
    /\ pc[t] = "bundle_phase4"
    /\ CanProceed(t, Parent)
    /\ LET oldW     == local[t].wrapper
           ser      == local[t].bundleSer
           finalPkt == MakePacket(Parent, oldW.packet.payload,
                                  oldW.packet.sub, FALSE)
           finalW   == PriorityWrapper(finalPkt, ser)
       IN
       IF linkage[Parent] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![Parent] = finalW]
            /\ local' = [local EXCEPT ![t].wrapper = finalW, ![t].snapResult = finalPkt]
            /\ pc' = [pc EXCEPT ![t] =
                  IF op[t] = "insert" THEN "insert_cas_parent"
                  ELSE IF op[t] = "release" THEN "release_cas_parent"
                  ELSE "commit_parent"]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterSuccess(t, Parent)]
            /\ UNCHANGED <<serial, op, target, inserted, insertTarget,
                           releaseTarget, everInserted, iterBudget,
                           childQueue, commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
            /\ UNCHANGED <<serial, linkage, local, op, target, inserted,
                           insertTarget, releaseTarget, everInserted,
                           iterBudget, childQueue, commitCount>>

-----------------------------------------------------------------------------
(* Insert CAS actions *)

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
(* CommitParent — for RootThreads, with dynamic child discovery *)

CommitParent(t) ==
    /\ pc[t] = "commit_parent"
    /\ childQueue[t] = {}
    /\ local[t].snapResult /= Null
    /\ LET snapPkt == local[t].snapResult
           snapChildren == {c \in AllChildren : snapPkt.sub[c] /= Null}
       IN
       IF snapChildren = {}
       THEN \* all children released since snapshot, skip iteration
            /\ iterBudget' = [iterBudget EXCEPT ![t] = iterBudget[t] - 1]
            /\ pc' = [pc EXCEPT ![t] = "idle"]
            /\ op' = [op EXCEPT ![t] = "idle"]
            /\ target' = [target EXCEPT ![t] = Null]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ priorityTag' = ClearMyTags(t)
            /\ UNCHANGED <<serial, linkage, inserted, insertTarget,
                           releaseTarget, everInserted, childQueue, commitCount>>
       ELSE
       /\ CanProceed(t, Parent)
       /\ LET pw      == linkage[Parent]
              newSub  == [c \in AllChildren |->
                  IF c \in snapChildren
                  THEN MakePacket(c,
                      snapPkt.sub[c].payload + 1,
                      snapPkt.sub[c].sub,
                      snapPkt.sub[c].missing)
                  ELSE snapPkt.sub[c]]
              newPkt  == MakePacket(Parent, snapPkt.payload, newSub, snapPkt.missing)
              ser     == GenSerial(t, pw.serial)
              newPW   == PriorityWrapper(newPkt, ser)
          IN
          \/ /\ pw.hasPriority
             /\ pw.packet = snapPkt
             /\ linkage' = [linkage EXCEPT ![Parent] = newPW]
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
          \/ /\ ~(pw.hasPriority /\ pw.packet = snapPkt)
             /\ pc' = [pc EXCEPT ![t] = "idle"]
             /\ op' = [op EXCEPT ![t] = "idle"]
             /\ target' = [target EXCEPT ![t] = Null]
             /\ local' = [local EXCEPT ![t] = InitLocal]
             /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
             /\ UNCHANGED <<serial, linkage, iterBudget, childQueue, inserted,
                            insertTarget, releaseTarget, everInserted, commitCount>>

-----------------------------------------------------------------------------
(* CommitChild actions — for LeafThreads *)

\* BeginChildIteration: leaf-only threads (not in RootThreads) discover children
\* dynamically from the current tree state.
BeginChildIteration(t) ==
    /\ pc[t] = "idle"
    /\ t \in LeafThreads
    /\ t \notin RootThreads
    /\ insertTarget[t] = Null
    /\ releaseTarget[t] = Null
    /\ iterBudget[t] > 0
    /\ childQueue[t] = {}
    /\ \A c \in AllChildren : everInserted[c]
    /\ LET currentChildren == {c \in AllChildren : inserted[c]}
       IN
       /\ currentChildren /= {}
       /\ childQueue' = [childQueue EXCEPT ![t] = currentChildren]
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
    /\ UNCHANGED <<serial, linkage, local, iterBudget, childQueue,
                   priorityTag, inserted, insertTarget, releaseTarget,
                   everInserted, commitCount>>

\* CommitSkip: skip a child that was released between queue setup and commit.
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
    /\ LET childNode == target[t]
           w == linkage[childNode]
       IN
       IF w.hasPriority
       THEN /\ local' = [local EXCEPT
                ![t].wrapper   = w,
                ![t].oldpacket = w.packet,
                ![t].newpacket = MakePacket(childNode,
                                     w.packet.payload + 1,
                                     w.packet.sub,
                                     w.packet.missing)]
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
            /\ priorityTag' = [priorityTag EXCEPT ![childNode] = TagAfterSuccess(t, childNode)]
            /\ UNCHANGED <<op, target, iterBudget, childQueue, inserted,
                           insertTarget, releaseTarget, everInserted, commitCount>>
       ELSE IF linkage[childNode].hasPriority
       THEN IF linkage[childNode].packet.payload = oldW.packet.payload
            THEN /\ local' = [local EXCEPT
                       ![t].wrapper = linkage[childNode],
                       ![t].newpacket = MakePacket(target[t],
                           local[t].newpacket.payload,
                           linkage[childNode].packet.sub,
                           linkage[childNode].packet.missing)]
                 /\ pc' = [pc EXCEPT ![t] = "commit_try_cas"]
                 /\ priorityTag' = [priorityTag EXCEPT ![childNode] = TagAfterFail(t, childNode)]
                 /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                inserted, insertTarget, releaseTarget, everInserted,
                                commitCount>>
            ELSE /\ local' = [local EXCEPT ![t].commitOk = "fail"]
                 /\ pc' = [pc EXCEPT ![t] = "commit_done"]
                 /\ priorityTag' = [priorityTag EXCEPT ![childNode] = TagAfterFail(t, childNode)]
                 /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                                inserted, insertTarget, releaseTarget, everInserted,
                                commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![childNode] = TagAfterFail(t, childNode)]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                           inserted, insertTarget, releaseTarget, everInserted,
                           commitCount>>

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
                       parentW.packet.sub[childNode].payload + 1,
                       parentW.packet.sub[childNode].sub,
                       parentW.packet.sub[childNode].missing)]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_ancestors"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                           priorityTag, inserted, insertTarget, releaseTarget,
                           everInserted, commitCount>>
       ELSE IF ~inserted[childNode]
       THEN \* child was released mid-commit, abort
            /\ local' = [local EXCEPT ![t].commitOk = "fail"]
            /\ pc' = [pc EXCEPT ![t] = "commit_done"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                           priorityTag, inserted, insertTarget, releaseTarget,
                           everInserted, commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                           priorityTag, inserted, insertTarget, releaseTarget,
                           everInserted, commitCount>>

UnbundleCASAncestors(t) ==
    /\ pc[t] = "unbundle_cas_ancestors"
    /\ CanProceed(t, Parent)
    /\ LET childNode  == target[t]
           oldParentW == local[t].parentWrapper
           ser        == GenSerial(t, oldParentW.serial)
           newPkt     == MakePacket(Parent, oldParentW.packet.payload,
                                   oldParentW.packet.sub, TRUE)
           newParentW == PriorityWrapper(newPkt, ser)
       IN
       IF oldParentW.hasPriority /\ linkage[Parent] = oldParentW
       THEN /\ linkage' = [linkage EXCEPT ![Parent] = newParentW]
            /\ UpdateSerial(t, ser)
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_child"]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterSuccess(t, Parent)]
            /\ UNCHANGED <<local, op, target, iterBudget, childQueue, inserted,
                           insertTarget, releaseTarget, everInserted, commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue,
                           inserted, insertTarget, releaseTarget, everInserted,
                           commitCount>>

UnbundleCASChild(t) ==
    /\ pc[t] = "unbundle_cas_child"
    /\ CanProceed(t, target[t])
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
            /\ priorityTag' = [priorityTag EXCEPT ![childNode] = TagAfterSuccess(t, childNode)]
            /\ UNCHANGED <<op, target, iterBudget, childQueue, inserted,
                           insertTarget, releaseTarget, everInserted, commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![childNode] = TagAfterFail(t, childNode)]
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
    /\ UNCHANGED <<serial, linkage, inserted, insertTarget, releaseTarget,
                   everInserted>>

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
    /\ LET c       == releaseTarget[t]
           oldW    == local[t].wrapper
           snapPkt == local[t].snapResult
           childPkt == snapPkt.sub[c]
           newSub  == [snapPkt.sub EXCEPT ![c] = Null]
           newPkt  == MakePacket(Parent, snapPkt.payload, newSub, snapPkt.missing)
           ser     == GenSerial(t, oldW.serial)
           newW    == PriorityWrapper(newPkt, ser)
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
       THEN /\ pc' = [pc EXCEPT ![t] = "idle"]
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
(* SkipIteration — drain remaining budget when all children are released *)

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
(* Next-state relation *)

AllDone ==
    /\ \A t \in Threads : pc[t] = "idle"
    /\ \A t \in Threads : insertTarget[t] = Null
    /\ \A t \in Threads : releaseTarget[t] = Null
    /\ \A t \in Threads : childQueue[t] = {}
    /\ \A t \in Threads : iterBudget[t] = 0
    /\ IF ReleaseThreads /= {} THEN \A c \in AllChildren : ~inserted[c]
       ELSE TRUE

\* Waiting: stuttering for idle threads when system is not yet AllDone.
\* Covers waiting for inserts, for other threads' commits/releases, etc.
\* NOT in NextStep, so WF does not require progress through this action.
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
        \/ BundlePhase1(t)
        \/ BundlePhase2(t)
        \/ BundlePhase3(t)
        \/ BundlePhase4(t)
        \/ InsertCASParent(t)
        \/ InsertReadChild(t)
        \/ InsertCASChild(t)
        \/ InsertFinal(t)
        \/ SnapRead(t)
        \/ CommitParent(t)
        \/ BeginChildIteration(t)
        \/ \E c \in AllChildren : CommitStart(t, c)
        \/ \E c \in AllChildren : CommitSkip(t, c)
        \/ CommitRead(t)
        \/ CommitTryCAS(t)
        \/ UnbundleWalk(t)
        \/ UnbundleCASAncestors(t)
        \/ UnbundleCASChild(t)
        \/ CommitDone(t)
        \/ ReleaseStart(t)
        \/ ReleaseCASParent(t)
        \/ ReleaseReadChild(t)
        \/ ReleaseCASChild(t)
        \/ SkipIteration(t)
        \/ \E n \in Nodes : PreemptTag(t, n)

Terminating ==
    /\ AllDone
    /\ UNCHANGED vars

Next == NextStep \/ Terminating \/ Waiting

Spec == Init /\ [][Next]_vars /\ WF_vars(NextStep)

-----------------------------------------------------------------------------
(* Safety invariants *)

SnapshotConsistency ==
    LET pw == linkage[Parent]
    IN  (pw.hasPriority /\ ~pw.packet.missing) =>
        (\A c \in ActiveChildren : pw.packet.sub[c] /= Null)

NoPriorityLoss ==
    \A c \in AllChildren :
        LET w == linkage[c]
        IN  w.hasPriority \/ w.bundledBy /= Null

BundleRefConsistency ==
    \A c \in AllChildren :
        LET cw == linkage[c]
        IN  (~cw.hasPriority /\ cw.bundledBy = Parent) =>
            linkage[Parent].hasPriority

MissingPropagation ==
    LET pw == linkage[Parent]
    IN  (pw.hasPriority /\ ~pw.packet.missing) =>
        (\A c \in ActiveChildren :
            pw.packet.sub[c] /= Null => ~pw.packet.sub[c].missing)

DebugSerialBound == TRUE

\* ChildPayload(c): authoritative payload. Reads from child's own wrapper
\* when unbundled, from parent's sub-packet when bundled.
ChildPayload(c) ==
    IF linkage[c].hasPriority
    THEN linkage[c].packet.payload
    ELSE IF linkage[Parent].packet.sub[c] /= Null
         THEN linkage[Parent].packet.sub[c].payload
         ELSE 0

\* TerminalPayloadCheck: each child's payload = 1 (insert) + commitCount.
TerminalPayloadCheck ==
    AllDone =>
        \A c \in AllChildren :
            /\ everInserted[c]
            /\ ChildPayload(c) = 1 + commitCount[c]

\* QuiescentCheck: verify payload at every all-idle moment.
QuiescentCheck ==
    (\A t \in Threads : pc[t] = "idle") =>
        \A c \in ActiveChildren :
            ChildPayload(c) = 1 + commitCount[c]

\* PrintTerminalSerial: per-thread serial at terminal state.
PrintTerminalSerial ==
    \/ ~AllDone
    \/ PrintT(<<[t \in Threads |-> SerialCounter(serial[t])]>>)

\* PrintTerminalMaxCounter: compact variant for supercomputer runs.
PrintTerminalMaxCounter ==
    \/ ~AllDone
    \/ LET counts == {SerialCounter(serial[t]) : t \in Threads}
           maxC == CHOOSE m \in counts : \A c \in counts : m >= c
       IN PrintT(maxC)

\* EventuallyAllDone: liveness — all threads complete.
EventuallyAllDone == <>AllDone

=============================================================================
