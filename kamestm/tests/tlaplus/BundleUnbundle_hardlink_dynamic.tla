(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.
 ***************************************************************************)
----------------------- MODULE BundleUnbundle_hardlink_dynamic -----------------------
(*
 * Hardlink model — DynChild1 is referenced from BOTH Parent1 and Parent2.
 *
 *   Parent1 --+-- DynChild1 (hardlinked, may be inserted under either or both)
 *   Parent2 --+----^
 *
 * Adapted from BundleUnbundle_2level_LLfree_dynamic.tla.  Superfine
 * atomicity throughout — each linkage CAS is an independent step,
 * full interleaving.
 *
 * Key new mechanic vs the 1-parent base:
 *   - The op-target parent is per-thread (`opParent[t]`).
 *   - bundle on opParent collects child packets that may currently be
 *     bundled under the OTHER parent.  Migration cascade:
 *         (a) read otherP's sub[c] in BundlePhase1
 *         (b) CAS opParent's wrapper to missing=TRUE with sub[c]
 *             populated (BundlePhase2)
 *         (c) CAS otherP's wrapper to set sub[c]=Null (NEW phase
 *             `MigrateClearOther`) — only runs when bundling captured a
 *             child from otherP's sub[]
 *         (d) CAS child.bundledBy = opParent (BundlePhase3)
 *         (e) CAS opParent missing=FALSE (BundlePhase4)
 *   - unbundle walks to child.bundledBy parent (no ambiguity since
 *     bundledBy is a single value).
 *
 * Invariants:
 *   - SnapshotConsistency(p): every parent in steady state
 *   - HardlinkExclusive: at most one parent's sub[] holds the child packet
 *   - BundleRefConsistency: child.bundledBy parent has its packet
 *   - NoPriorityLoss / MissingPropagation
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Threads,
    Parent1, Parent2,
    DynChild1,
    Null,
    MaxCommits,
    BundleCollectAtomic,    \* "fine" / "superfine"
    BundlePhase3Atomic,     \* "fine" / "superfine"
    InsertThreads,          \* threads doing insert(parent, child)
    RootThreads,            \* threads doing CommitParent
    LeafThreads,            \* threads doing CommitChild
    ReleaseThreads          \* threads doing release(parent, child)

Parents == {Parent1, Parent2}
AllChildren == {DynChild1}
Nodes == Parents \cup AllChildren

ASSUME InsertThreads \subseteq Threads
ASSUME RootThreads \subseteq Threads /\ LeafThreads \subseteq Threads
ASSUME ReleaseThreads \subseteq Threads
\* InsertThreads may be empty: the hardlink topology is set up at Init,
\* so no run-time insert is required to exercise the migration race.

ThreadSymmetry == Permutations(Threads)

VARIABLES
    serial,        \* [Threads -> Nat]: Lamport clock
    linkage,       \* [Nodes -> Wrapper]: per-node atomic
    pc,            \* [Threads -> String]: program counter
    op,            \* [Threads -> String]: current op label
    target,        \* [Threads -> Node|Null]: CAS target
    local,         \* [Threads -> Record]: thread-local
    priorityTag,   \* [Nodes -> Null | <<iter, tid>>]: LL-free tag
    insertedIn,    \* [Children × Parents -> BOOL]
    insertTarget,  \* [Threads -> Parent|Null]: which parent inserting into
    releaseTarget, \* [Threads -> Parent|Null]: which parent releasing from
    everInsertedIn,\* [Children × Parents -> BOOL]
    iterBudget,    \* [Threads -> Nat]
    childQueue,    \* [Threads -> SUBSET Children]
    commitCount,   \* [Children -> Nat]
    opParent       \* [Threads -> Parent|Null]: which parent is current op's

vars == <<serial, linkage, pc, op, target, local, priorityTag,
          insertedIn, insertTarget, releaseTarget, everInsertedIn,
          iterBudget, childQueue, commitCount, opParent>>

-----------------------------------------------------------------------------
(* Wrapper helpers *)

PriorityWrapper(packet, ser) ==
    [packet |-> packet, hasPriority |-> TRUE, bundledBy |-> Null, serial |-> ser]

BundledRefWrapper(parentNode, ser) ==
    [packet |-> Null, hasPriority |-> FALSE, bundledBy |-> parentNode, serial |-> ser]

InsertedRef(parentNode, ser, pkt) ==
    [packet |-> pkt, hasPriority |-> FALSE, bundledBy |-> parentNode, serial |-> ser]

MakePacket(node, payload, sub, miss) ==
    [payload |-> payload, sub |-> sub, missing |-> miss, node |-> node]

EmptySub == [c \in AllChildren |-> Null]

\* ActiveOn(p): children currently inserted under parent p.
ActiveOn(p) == {c \in AllChildren : insertedIn[<<c, p>>]}

OtherParent(p) == IF p = Parent1 THEN Parent2 ELSE Parent1

-----------------------------------------------------------------------------
(* Serial *)

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
    /\ UNCHANGED <<serial, linkage, pc, op, target, local, insertedIn,
                   insertTarget, releaseTarget, everInsertedIn, iterBudget,
                   childQueue, commitCount, opParent>>

-----------------------------------------------------------------------------
(* Init *)

InitLocal == [
    wrapper        |-> Null,
    parentWrapper  |-> Null,
    otherWrapper   |-> Null,    \* NEW: wrapper of OtherParent at bundle time
    subwrappers    |-> [c \in AllChildren |-> Null],
    subpackets     |-> EmptySub,
    subFromOther   |-> [c \in AllChildren |-> FALSE],  \* NEW: c came from otherP
    bundleSer      |-> 0,
    snapResult     |-> Null,
    oldpacket      |-> Null,
    newpacket      |-> Null,
    commitOk       |-> Null
]

\* Initial topology: DynChild1 is pre-inserted as a HARDLINK under
\* BOTH parents.  The child's packet currently lives in Parent1's sub[]
\* (= bundledBy = Parent1).  Parent2 references the child via insertedIn
\* but its sub[] slot is Null until a migration brings it over.
\*
\* Rationale: the insert pipeline does not handle the second-insert
\* (insert into a parent while child already bundled elsewhere) — that
\* loop case would require its own multi-CAS migration path inside
\* InsertReadChild.  For the minimal hardlink model we skip that and
\* start from the post-second-insert state, focusing on commit/release
\* migration races.
Init ==
    /\ linkage = [n \in Nodes |->
        IF n = Parent1
        THEN PriorityWrapper(
                MakePacket(Parent1, 0,
                    [c \in AllChildren |->
                        IF c = DynChild1
                        THEN MakePacket(DynChild1, 0, EmptySub, FALSE)
                        ELSE Null],
                    FALSE), 0)
        ELSE IF n = Parent2
        THEN PriorityWrapper(MakePacket(Parent2, 0, EmptySub, FALSE), 0)
        ELSE BundledRefWrapper(Parent1, 0)]
    /\ serial = [t \in Threads |-> EncodeSerial(0, t)]
    /\ pc = [t \in Threads |-> "idle"]
    /\ op = [t \in Threads |-> "idle"]
    /\ target = [t \in Threads |-> Null]
    /\ local = [t \in Threads |-> InitLocal]
    /\ priorityTag = [n \in Nodes |-> Null]
    /\ insertedIn = [pair \in AllChildren \X Parents |-> TRUE]
    /\ insertTarget = [t \in Threads |-> Null]
    /\ releaseTarget = [t \in Threads |-> Null]
    /\ everInsertedIn = [pair \in AllChildren \X Parents |-> TRUE]
    /\ iterBudget = [t \in Threads |->
        IF t \in RootThreads \cup LeafThreads THEN MaxCommits ELSE 0]
    /\ childQueue = [t \in Threads |-> {}]
    /\ commitCount = [c \in AllChildren |-> 0]
    /\ opParent = [t \in Threads |-> Null]

-----------------------------------------------------------------------------
(* Bundle retry target — depends on current operation *)

BundleRetryPC(t) ==
    IF op[t] = "insert" THEN "insert_snap"
    ELSE IF op[t] = "release" THEN "release_snap"
    ELSE "snap_read"

-----------------------------------------------------------------------------
(* Insert actions — InsertThreads pick (child, parent) pair *)

InsertStart(t) ==
    /\ pc[t] = "idle"
    /\ t \in InsertThreads
    /\ insertTarget[t] = Null
    /\ releaseTarget[t] = Null
    /\ childQueue[t] = {}
    /\ \E p \in Parents :
        /\ ~insertedIn[<<DynChild1, p>>]
        /\ ~everInsertedIn[<<DynChild1, p>>]
        /\ \A t2 \in Threads \ {t} : insertTarget[t2] /= p
        /\ insertTarget' = [insertTarget EXCEPT ![t] = p]
        /\ opParent' = [opParent EXCEPT ![t] = p]
        /\ op' = [op EXCEPT ![t] = "insert"]
        /\ pc' = [pc EXCEPT ![t] = "insert_snap"]
    /\ UNCHANGED <<serial, linkage, target, local, priorityTag, insertedIn,
                   releaseTarget, everInsertedIn, iterBudget, childQueue,
                   commitCount>>

-----------------------------------------------------------------------------
(* ReadParent — opParent[t]'s wrapper read.  Routes to next phase. *)

ReadParent(t) ==
    /\ pc[t] \in {"insert_snap", "snap_read", "release_snap"}
    /\ opParent[t] /= Null
    /\ LET p == opParent[t]
           w == linkage[p]
       IN
       /\ target' = [target EXCEPT ![t] = p]
       /\ IF w.hasPriority /\ ~w.packet.missing
          THEN /\ local' = [local EXCEPT ![t].wrapper = w,
                                          ![t].snapResult = w.packet]
               /\ pc' = [pc EXCEPT ![t] =
                       IF op[t] = "insert" THEN "insert_cas_parent"
                       ELSE IF op[t] = "release" THEN "release_cas_parent"
                       ELSE "commit_parent"]
               /\ UNCHANGED <<serial, linkage, op, priorityTag, insertedIn,
                              insertTarget, releaseTarget, everInsertedIn,
                              iterBudget, childQueue, commitCount, opParent>>
          ELSE IF w.hasPriority /\ w.packet.missing
          THEN LET ser == GenSerial(t, w.serial)
               IN
               /\ local' = [local EXCEPT
                      ![t].wrapper = w,
                      ![t].bundleSer = ser,
                      ![t].subwrappers = [c \in AllChildren |-> Null],
                      ![t].subpackets = EmptySub,
                      ![t].subFromOther = [c \in AllChildren |-> FALSE]]
               /\ UpdateSerial(t, ser)
               /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
               /\ UNCHANGED <<linkage, op, priorityTag, insertedIn,
                              insertTarget, releaseTarget, everInsertedIn,
                              iterBudget, childQueue, commitCount, opParent>>
          ELSE /\ pc' = [pc EXCEPT ![t] = pc[t]]
               /\ UNCHANGED <<serial, linkage, local, op, priorityTag,
                              insertedIn, insertTarget, releaseTarget,
                              everInsertedIn, iterBudget, childQueue,
                              commitCount, opParent>>

-----------------------------------------------------------------------------
(* SnapRead — entry point for RootThreads (CommitParent) *)

SnapRead(t) ==
    /\ pc[t] = "idle"
    /\ t \in RootThreads
    /\ iterBudget[t] > 0
    /\ childQueue[t] = {}
    /\ insertTarget[t] = Null
    /\ releaseTarget[t] = Null
    /\ \E p \in Parents :
        /\ \A c \in AllChildren : everInsertedIn[<<c, p>>]
        /\ opParent' = [opParent EXCEPT ![t] = p]
    /\ op' = [op EXCEPT ![t] = "snapshot"]
    /\ pc' = [pc EXCEPT ![t] = "snap_read"]
    /\ UNCHANGED <<serial, linkage, target, local, iterBudget, childQueue,
                   priorityTag, insertedIn, insertTarget, releaseTarget,
                   everInsertedIn, commitCount>>

-----------------------------------------------------------------------------
(* Bundle pipeline — operates on opParent[t] *)

\* BundlePhase1 — per-child collection.  For each active child under
\* opParent, gather a sub-packet from:
\*   (a) child's own priority wrapper, OR
\*   (b) opParent's own sub[] (if child bundledBy = opParent), OR
\*   (c) other parent's sub[] (if child bundledBy = otherP) — MIGRATION
BundlePhase1(t) ==
    /\ pc[t] = "bundle_phase1"
    /\ opParent[t] /= Null
    /\ LET p == opParent[t]
           op2 == OtherParent(p)
           parentW == local[t].wrapper
           ser    == local[t].bundleSer
           activeOnP == ActiveOn(p)
       IN
       \* Serial sync: if parent serial drifted, refresh.
       IF BundleCollectAtomic = "superfine" /\ parentW.serial /= ser
       THEN /\ CanProceed(t, p)
            /\ LET newW == PriorityWrapper(parentW.packet, ser)
               IN
               IF linkage[p] = parentW
               THEN /\ linkage' = [linkage EXCEPT ![p] = newW]
                    /\ local' = [local EXCEPT ![t].wrapper = newW]
                    /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                    /\ priorityTag' = [priorityTag EXCEPT ![p] = TagAfterSuccess(t, p)]
                    /\ UNCHANGED <<serial, op, target, insertedIn, insertTarget,
                                   releaseTarget, everInsertedIn, iterBudget,
                                   childQueue, commitCount, opParent>>
               ELSE /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
                    /\ priorityTag' = [priorityTag EXCEPT ![p] = TagAfterFail(t, p)]
                    /\ UNCHANGED <<serial, linkage, local, op, target,
                                   insertedIn, insertTarget, releaseTarget,
                                   everInsertedIn, iterBudget, childQueue,
                                   commitCount, opParent>>
       ELSE
       IF activeOnP = {}
       THEN /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
            /\ UNCHANGED <<serial, linkage, local, op, target, priorityTag,
                           insertedIn, insertTarget, releaseTarget, everInsertedIn,
                           iterBudget, childQueue, commitCount, opParent>>
       ELSE
       \* Per-child collection step OR proceed to phase2 when all done.
       \/ \E c \in activeOnP :
             /\ local[t].subwrappers[c] = Null
             /\ LET cw == linkage[c]
                    pkt_from_priority == cw.packet
                    pkt_from_p_sub == IF parentW.packet.sub[c] /= Null
                                      THEN parentW.packet.sub[c]
                                      ELSE Null
                    pkt_from_op2_sub == LET op2W == linkage[op2] IN
                        IF op2W.hasPriority /\ op2W.packet.sub[c] /= Null
                        THEN op2W.packet.sub[c]
                        ELSE Null
                    \* Choose source by bundledBy.
                    cpkt == IF cw.hasPriority THEN pkt_from_priority
                            ELSE IF cw.bundledBy = p THEN pkt_from_p_sub
                            ELSE IF cw.bundledBy = op2 THEN pkt_from_op2_sub
                            ELSE Null
                    fromOther == ~cw.hasPriority /\ cw.bundledBy = op2 /\ cpkt /= Null
                IN
                IF cpkt /= Null
                THEN /\ local' = [local EXCEPT
                            ![t].subwrappers[c] = cw,
                            ![t].subpackets[c] = cpkt,
                            ![t].subFromOther[c] = fromOther]
                     /\ LET allDone == \A c2 \in activeOnP \ {c} :
                                local[t].subwrappers[c2] /= Null
                        IN
                        IF allDone
                        THEN pc' = [pc EXCEPT ![t] = "bundle_phase2"]
                        ELSE pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                     /\ UNCHANGED <<serial, linkage, op, target, priorityTag,
                                    insertedIn, insertTarget, releaseTarget,
                                    everInsertedIn, iterBudget, childQueue,
                                    commitCount, opParent>>
                ELSE \* Could not read; tag child & retry from top.
                     /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
                     /\ local' = [local EXCEPT
                            ![t].subwrappers = [c2 \in AllChildren |-> Null],
                            ![t].subpackets  = EmptySub,
                            ![t].subFromOther = [c2 \in AllChildren |-> FALSE]]
                     /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                     /\ UNCHANGED <<serial, linkage, op, target, insertedIn,
                                    insertTarget, releaseTarget, everInsertedIn,
                                    iterBudget, childQueue, commitCount, opParent>>
       \* All collected — fold stale-released children, advance.
       \/ /\ \A c \in activeOnP : local[t].subwrappers[c] /= Null
          /\ local' = [local EXCEPT
                   ![t].subwrappers = [c2 \in AllChildren |->
                       IF c2 \in activeOnP THEN local[t].subwrappers[c2] ELSE Null],
                   ![t].subpackets  = [c2 \in AllChildren |->
                       IF c2 \in activeOnP THEN local[t].subpackets[c2] ELSE Null]]
          /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
          /\ UNCHANGED <<serial, linkage, op, target, priorityTag,
                         insertedIn, insertTarget, releaseTarget, everInsertedIn,
                         iterBudget, childQueue, commitCount, opParent>>

\* BundlePhase2 — CAS opParent → missing=TRUE with sub[] populated from
\* collected sub-packets.  Phase entry condition: at least one collected.
BundlePhase2(t) ==
    /\ pc[t] = "bundle_phase2"
    /\ opParent[t] /= Null
    /\ LET p == opParent[t]
       IN  CanProceed(t, p)
    /\ LET p == opParent[t]
           oldW   == local[t].wrapper
           ser    == local[t].bundleSer
           subs   == local[t].subpackets
           newPkt == MakePacket(p, oldW.packet.payload, subs, TRUE)
           newW   == PriorityWrapper(newPkt, ser)
       IN
       IF linkage[p] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![p] = newW]
            /\ local' = [local EXCEPT ![t].wrapper = newW]
            /\ pc' = [pc EXCEPT ![t] =
                    IF \E c \in AllChildren : local[t].subFromOther[c]
                    THEN "migrate_clear_other"
                    ELSE "bundle_phase3"]
            /\ priorityTag' = [priorityTag EXCEPT ![p] = TagAfterSuccess(t, p)]
            /\ UNCHANGED <<serial, op, target, insertedIn, insertTarget,
                           releaseTarget, everInsertedIn, iterBudget,
                           childQueue, commitCount, opParent>>
       ELSE /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
            /\ priorityTag' = [priorityTag EXCEPT
                   ![opParent[t]] = TagAfterFail(t, opParent[t])]
            /\ UNCHANGED <<serial, linkage, local, op, target, insertedIn,
                           insertTarget, releaseTarget, everInsertedIn,
                           iterBudget, childQueue, commitCount, opParent>>

\* MigrateClearOther — NEW phase for hardlink: CAS otherP wrapper to remove
\* migrated children from its sub[].  Runs only after BundlePhase2 if any
\* sub-packet came from otherP.
MigrateClearOther(t) ==
    /\ pc[t] = "migrate_clear_other"
    /\ opParent[t] /= Null
    /\ LET p == opParent[t]
           op2 == OtherParent(p)
       IN  CanProceed(t, op2)
    /\ LET p == opParent[t]
           op2 == OtherParent(p)
           op2W == linkage[op2]
           migrated == {c \in AllChildren : local[t].subFromOther[c]}
           ser == local[t].bundleSer
       IN
       /\ op2W.hasPriority
       /\ \A c \in migrated : op2W.packet.sub[c] /= Null
       /\ \A c \in migrated : op2W.packet.sub[c] = local[t].subpackets[c]
       /\ LET newSub == [c \in AllChildren |->
                  IF c \in migrated THEN Null ELSE op2W.packet.sub[c]]
              newPkt == MakePacket(op2, op2W.packet.payload, newSub, op2W.packet.missing)
              newSer == GenSerial(t, op2W.serial)
              newW == PriorityWrapper(newPkt, newSer)
          IN
          /\ linkage[op2] = op2W
          /\ linkage' = [linkage EXCEPT ![op2] = newW]
          /\ UpdateSerial(t, newSer)
          /\ pc' = [pc EXCEPT ![t] = "bundle_phase3"]
          /\ priorityTag' = [priorityTag EXCEPT ![op2] = TagAfterSuccess(t, op2)]
          /\ UNCHANGED <<local, op, target, insertedIn, insertTarget,
                         releaseTarget, everInsertedIn, iterBudget,
                         childQueue, commitCount, opParent>>

MigrateClearOtherFail(t) ==
    /\ pc[t] = "migrate_clear_other"
    /\ opParent[t] /= Null
    /\ LET p == opParent[t]
           op2 == OtherParent(p)
           op2W == linkage[op2]
           migrated == {c \in AllChildren : local[t].subFromOther[c]}
       IN  \/ ~op2W.hasPriority
           \/ \E c \in migrated : op2W.packet.sub[c] = Null
           \/ \E c \in migrated : op2W.packet.sub[c] /= local[t].subpackets[c]
           \/ ~CanProceed(t, op2)
    /\ LET p == opParent[t]
           op2 == OtherParent(p)
       IN
       /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
       /\ local' = [local EXCEPT ![t] = InitLocal]
       /\ priorityTag' = [priorityTag EXCEPT ![op2] = TagAfterFail(t, op2)]
       /\ UNCHANGED <<serial, linkage, op, target, insertedIn, insertTarget,
                      releaseTarget, everInsertedIn, iterBudget, childQueue,
                      commitCount, opParent>>

\* BundlePhase3 — per-child CAS to BundledRefWrapper(opParent).
BundlePhase3(t) ==
    /\ pc[t] = "bundle_phase3"
    /\ opParent[t] /= Null
    /\ LET p == opParent[t]
           ser == local[t].bundleSer
       IN
       \E c \in AllChildren :
            /\ local[t].subwrappers[c] /= Null
            /\ CanProceed(t, c)
            /\ linkage[c] = local[t].subwrappers[c]
            /\ linkage' = [linkage EXCEPT ![c] = BundledRefWrapper(p, ser)]
            /\ LET allDone == \A c2 \in AllChildren \ {c} :
                       local[t].subwrappers[c2] = Null
                       \/ linkage[c2] = BundledRefWrapper(p, ser)
               IN
               IF allDone
               THEN pc' = [pc EXCEPT ![t] = "bundle_phase4"]
               ELSE pc' = [pc EXCEPT ![t] = "bundle_phase3"]
            /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterSuccess(t, c)]
            /\ UNCHANGED <<serial, local, op, target, insertedIn, insertTarget,
                           releaseTarget, everInsertedIn, iterBudget,
                           childQueue, commitCount, opParent>>

BundlePhase3Fail(t) ==
    /\ pc[t] = "bundle_phase3"
    /\ opParent[t] /= Null
    /\ \E c \in AllChildren :
        /\ local[t].subwrappers[c] /= Null
        /\ CanProceed(t, c)
        /\ linkage[c] /= local[t].subwrappers[c]
        /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
        /\ local' = [local EXCEPT ![t] = InitLocal]
        /\ priorityTag' = [
               [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                   EXCEPT ![opParent[t]] = TagAfterFail(t, opParent[t])]
           /\ UNCHANGED <<serial, linkage, op, target, insertedIn,
                          insertTarget, releaseTarget, everInsertedIn,
                          iterBudget, childQueue, commitCount, opParent>>

\* BundlePhase4 — final CAS, opParent missing=FALSE.
BundlePhase4(t) ==
    /\ pc[t] = "bundle_phase4"
    /\ opParent[t] /= Null
    /\ LET p == opParent[t]
       IN  CanProceed(t, p)
    /\ LET p == opParent[t]
           oldW     == local[t].wrapper
           ser      == local[t].bundleSer
           finalPkt == MakePacket(p, oldW.packet.payload, oldW.packet.sub, FALSE)
           finalW   == PriorityWrapper(finalPkt, ser)
       IN
       IF linkage[p] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![p] = finalW]
            /\ local' = [local EXCEPT ![t].wrapper = finalW,
                                       ![t].snapResult = finalPkt]
            /\ pc' = [pc EXCEPT ![t] =
                    IF op[t] = "insert" THEN "insert_cas_parent"
                    ELSE IF op[t] = "release" THEN "release_cas_parent"
                    ELSE "commit_parent"]
            /\ priorityTag' = [priorityTag EXCEPT ![p] = TagAfterSuccess(t, p)]
            /\ UNCHANGED <<serial, op, target, insertedIn, insertTarget,
                           releaseTarget, everInsertedIn, iterBudget,
                           childQueue, commitCount, opParent>>
       ELSE /\ pc' = [pc EXCEPT ![t] = BundleRetryPC(t)]
            /\ priorityTag' = [priorityTag EXCEPT ![opParent[t]] = TagAfterFail(t, opParent[t])]
            /\ UNCHANGED <<serial, linkage, local, op, target, insertedIn,
                           insertTarget, releaseTarget, everInsertedIn,
                           iterBudget, childQueue, commitCount, opParent>>

-----------------------------------------------------------------------------
(* Insert CAS actions *)

InsertCASParent(t) ==
    /\ pc[t] = "insert_cas_parent"
    /\ opParent[t] /= Null
    /\ LET p == opParent[t] IN CanProceed(t, p)
    /\ LET p == opParent[t]
           oldW == local[t].wrapper
           ser  == GenSerial(t, oldW.serial)
           newW == PriorityWrapper(oldW.packet, ser)
       IN
       IF linkage[p] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![p] = newW]
            /\ UpdateSerial(t, ser)
            /\ local' = [local EXCEPT ![t].wrapper = newW]
            /\ pc' = [pc EXCEPT ![t] = "insert_read_child"]
            /\ priorityTag' = [priorityTag EXCEPT ![p] = TagAfterSuccess(t, p)]
            /\ UNCHANGED <<op, target, insertedIn, insertTarget, releaseTarget,
                           everInsertedIn, iterBudget, childQueue, commitCount,
                           opParent>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "insert_snap"]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ priorityTag' = [priorityTag EXCEPT ![opParent[t]] = TagAfterFail(t, opParent[t])]
            /\ UNCHANGED <<serial, linkage, op, target, insertedIn, insertTarget,
                           releaseTarget, everInsertedIn, iterBudget,
                           childQueue, commitCount, opParent>>

InsertReadChild(t) ==
    /\ pc[t] = "insert_read_child"
    /\ insertTarget[t] /= Null
    /\ LET p == insertTarget[t]
           c == DynChild1
           cw == linkage[c]
       IN
       IF cw.hasPriority
       THEN /\ local' = [local EXCEPT
                  ![t].subwrappers[c] = cw,
                  ![t].subpackets[c]  = cw.packet]
            /\ pc' = [pc EXCEPT ![t] = "insert_cas_child"]
            /\ UNCHANGED <<serial, linkage, op, target, priorityTag, insertedIn,
                           insertTarget, releaseTarget, everInsertedIn,
                           iterBudget, childQueue, commitCount, opParent>>
       ELSE IF cw.bundledBy = p
       THEN \* Already bundled here — packet must be in parent's sub[].
            /\ local' = [local EXCEPT
                    ![t].subpackets[c] = local[t].wrapper.packet.sub[c]]
            /\ pc' = [pc EXCEPT ![t] = "insert_final"]
            /\ UNCHANGED <<serial, linkage, op, target, priorityTag, insertedIn,
                           insertTarget, releaseTarget, everInsertedIn,
                           iterBudget, childQueue, commitCount, opParent>>
       ELSE \* bundled under other parent — retry (path needs full bundle).
            /\ pc' = [pc EXCEPT ![t] = "insert_snap"]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ UNCHANGED <<serial, linkage, op, target, priorityTag, insertedIn,
                           insertTarget, releaseTarget, everInsertedIn,
                           iterBudget, childQueue, commitCount, opParent>>

InsertCASChild(t) ==
    /\ pc[t] = "insert_cas_child"
    /\ insertTarget[t] /= Null
    /\ LET p == insertTarget[t]
           c == DynChild1
           oldCW == local[t].subwrappers[c]
           ser   == local[t].wrapper.serial
           cpkt  == local[t].subpackets[c]
       IN
       /\ CanProceed(t, c)
       /\ IF linkage[c] = oldCW
          THEN /\ linkage' = [linkage EXCEPT ![c] = InsertedRef(p, ser, cpkt)]
               /\ pc' = [pc EXCEPT ![t] = "insert_final"]
               /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterSuccess(t, c)]
               /\ UNCHANGED <<serial, local, op, target, insertedIn, insertTarget,
                              releaseTarget, everInsertedIn, iterBudget,
                              childQueue, commitCount, opParent>>
          ELSE /\ pc' = [pc EXCEPT ![t] = "insert_snap"]
               /\ local' = [local EXCEPT ![t] = InitLocal]
               /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
               /\ UNCHANGED <<serial, linkage, op, target, insertedIn,
                              insertTarget, releaseTarget, everInsertedIn,
                              iterBudget, childQueue, commitCount, opParent>>

InsertFinal(t) ==
    /\ pc[t] = "insert_final"
    /\ insertTarget[t] /= Null
    /\ LET p == insertTarget[t] IN CanProceed(t, p)
    /\ LET p == insertTarget[t]
           c == DynChild1
           oldW == local[t].wrapper
           cpkt == local[t].subpackets[c]
           newCPkt == MakePacket(c, cpkt.payload + 1, cpkt.sub, cpkt.missing)
           newSub == [oldW.packet.sub EXCEPT ![c] = newCPkt]
           newPkt == MakePacket(p, oldW.packet.payload, newSub, TRUE)
           ser == GenSerial(t, oldW.serial)
           newW == PriorityWrapper(newPkt, ser)
       IN
       IF linkage[p] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![p] = newW]
            /\ UpdateSerial(t, ser)
            /\ insertedIn' = [insertedIn EXCEPT ![<<c, p>>] = TRUE]
            /\ everInsertedIn' = [everInsertedIn EXCEPT ![<<c, p>>] = TRUE]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ pc' = [pc EXCEPT ![t] = "idle"]
            /\ op' = [op EXCEPT ![t] = "idle"]
            /\ target' = [target EXCEPT ![t] = Null]
            /\ insertTarget' = [insertTarget EXCEPT ![t] = Null]
            /\ opParent' = [opParent EXCEPT ![t] = Null]
            /\ priorityTag' = ClearMyTags(t)
            /\ UNCHANGED <<releaseTarget, iterBudget, childQueue, commitCount>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "insert_snap"]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ priorityTag' = [priorityTag EXCEPT ![insertTarget[t]] = TagAfterFail(t, insertTarget[t])]
            /\ UNCHANGED <<serial, linkage, op, target, insertedIn, insertTarget,
                           releaseTarget, everInsertedIn, iterBudget,
                           childQueue, commitCount, opParent>>

-----------------------------------------------------------------------------
(* CommitParent — RootThreads commit opParent's snapshot *)

CommitParent(t) ==
    /\ pc[t] = "commit_parent"
    /\ opParent[t] /= Null
    /\ childQueue[t] = {}
    /\ local[t].snapResult /= Null
    /\ LET p == opParent[t]
           snapPkt == local[t].snapResult
           snapChildren == {c \in AllChildren : snapPkt.sub[c] /= Null}
       IN
       IF snapChildren = {}
       THEN /\ iterBudget' = [iterBudget EXCEPT ![t] = iterBudget[t] - 1]
            /\ pc' = [pc EXCEPT ![t] = "idle"]
            /\ op' = [op EXCEPT ![t] = "idle"]
            /\ target' = [target EXCEPT ![t] = Null]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ opParent' = [opParent EXCEPT ![t] = Null]
            /\ priorityTag' = ClearMyTags(t)
            /\ UNCHANGED <<serial, linkage, insertedIn, insertTarget,
                           releaseTarget, everInsertedIn, childQueue, commitCount>>
       ELSE
       /\ CanProceed(t, p)
       /\ LET pw == linkage[p]
              newSub == [c \in AllChildren |->
                  IF c \in snapChildren
                  THEN MakePacket(c, snapPkt.sub[c].payload + 1,
                                  snapPkt.sub[c].sub, snapPkt.sub[c].missing)
                  ELSE snapPkt.sub[c]]
              newPkt == MakePacket(p, snapPkt.payload, newSub, snapPkt.missing)
              ser == GenSerial(t, pw.serial)
              newPW == PriorityWrapper(newPkt, ser)
          IN
          \/ /\ pw.hasPriority
             /\ pw.packet = snapPkt
             /\ linkage' = [linkage EXCEPT ![p] = newPW]
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
             /\ opParent' = [opParent EXCEPT ![t] = Null]
             /\ priorityTag' = ClearMyTags(t)
             /\ UNCHANGED <<insertedIn, insertTarget, releaseTarget, everInsertedIn>>
          \/ /\ ~(pw.hasPriority /\ pw.packet = snapPkt)
             /\ pc' = [pc EXCEPT ![t] = "idle"]
             /\ op' = [op EXCEPT ![t] = "idle"]
             /\ target' = [target EXCEPT ![t] = Null]
             /\ local' = [local EXCEPT ![t] = InitLocal]
             /\ opParent' = [opParent EXCEPT ![t] = Null]
             /\ priorityTag' = [priorityTag EXCEPT ![p] = TagAfterFail(t, p)]
             /\ UNCHANGED <<serial, linkage, iterBudget, childQueue, insertedIn,
                            insertTarget, releaseTarget, everInsertedIn,
                            commitCount>>

-----------------------------------------------------------------------------
(* CommitChild — for LeafThreads *)

BeginChildIteration(t) ==
    /\ pc[t] = "idle"
    /\ t \in LeafThreads
    /\ t \notin RootThreads
    /\ insertTarget[t] = Null
    /\ releaseTarget[t] = Null
    /\ iterBudget[t] > 0
    /\ childQueue[t] = {}
    /\ \A c \in AllChildren : \A p \in Parents : everInsertedIn[<<c, p>>]
    /\ LET currentChildren == {c \in AllChildren :
                                  \E p \in Parents : insertedIn[<<c, p>>]}
       IN
       /\ currentChildren /= {}
       /\ childQueue' = [childQueue EXCEPT ![t] = currentChildren]
    /\ UNCHANGED <<serial, linkage, pc, op, target, local, iterBudget,
                   priorityTag, insertedIn, insertTarget, releaseTarget,
                   everInsertedIn, commitCount, opParent>>

CommitStart(t, childNode) ==
    /\ pc[t] = "idle"
    /\ childNode \in childQueue[t]
    /\ (\E p \in Parents : insertedIn[<<childNode, p>>])
    /\ op' = [op EXCEPT ![t] = "commit"]
    /\ target' = [target EXCEPT ![t] = childNode]
    /\ pc' = [pc EXCEPT ![t] = "commit_read"]
    /\ UNCHANGED <<serial, linkage, local, iterBudget, childQueue, priorityTag,
                   insertedIn, insertTarget, releaseTarget, everInsertedIn,
                   commitCount, opParent>>

CommitSkip(t, childNode) ==
    /\ pc[t] = "idle"
    /\ childNode \in childQueue[t]
    /\ ~(\E p \in Parents : insertedIn[<<childNode, p>>])
    /\ LET newQueue == childQueue[t] \ {childNode}
       IN
       /\ childQueue' = [childQueue EXCEPT ![t] = newQueue]
       /\ IF newQueue = {}
          THEN /\ iterBudget' = [iterBudget EXCEPT ![t] = iterBudget[t] - 1]
               /\ priorityTag' = ClearMyTags(t)
          ELSE /\ UNCHANGED iterBudget
               /\ UNCHANGED priorityTag
    /\ UNCHANGED <<serial, linkage, pc, op, target, local, insertedIn,
                   insertTarget, releaseTarget, everInsertedIn, commitCount,
                   opParent>>

CommitRead(t) ==
    /\ pc[t] = "commit_read"
    /\ LET c == target[t]
           w == linkage[c]
       IN
       IF w.hasPriority
       THEN /\ local' = [local EXCEPT
                ![t].wrapper = w,
                ![t].oldpacket = w.packet,
                ![t].newpacket = MakePacket(c, w.packet.payload + 1,
                                            w.packet.sub, w.packet.missing)]
            /\ pc' = [pc EXCEPT ![t] = "commit_try_cas"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                           priorityTag, insertedIn, insertTarget, releaseTarget,
                           everInsertedIn, commitCount, opParent>>
       ELSE /\ local' = [local EXCEPT ![t].wrapper = w]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                           priorityTag, insertedIn, insertTarget, releaseTarget,
                           everInsertedIn, commitCount, opParent>>

CommitTryCAS(t) ==
    /\ pc[t] = "commit_try_cas"
    /\ CanProceed(t, target[t])
    /\ LET c == target[t]
           oldW == local[t].wrapper
           ser  == GenSerial(t, oldW.serial)
           newW == PriorityWrapper(local[t].newpacket, ser)
       IN
       IF linkage[c] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![c] = newW]
            /\ UpdateSerial(t, ser)
            /\ local' = [local EXCEPT ![t].commitOk = "ok"]
            /\ pc' = [pc EXCEPT ![t] = "commit_done"]
            /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterSuccess(t, c)]
            /\ UNCHANGED <<op, target, iterBudget, childQueue, insertedIn,
                           insertTarget, releaseTarget, everInsertedIn,
                           commitCount, opParent>>
       ELSE IF linkage[c].hasPriority
       THEN IF linkage[c].packet.payload = oldW.packet.payload
            THEN /\ local' = [local EXCEPT
                        ![t].wrapper = linkage[c],
                        ![t].newpacket = MakePacket(c,
                            local[t].newpacket.payload,
                            linkage[c].packet.sub,
                            linkage[c].packet.missing)]
                 /\ pc' = [pc EXCEPT ![t] = "commit_try_cas"]
                 /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                 /\ UNCHANGED <<serial, linkage, op, target, iterBudget,
                                childQueue, insertedIn, insertTarget,
                                releaseTarget, everInsertedIn, commitCount,
                                opParent>>
            ELSE /\ local' = [local EXCEPT ![t].commitOk = "fail"]
                 /\ pc' = [pc EXCEPT ![t] = "commit_done"]
                 /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                 /\ UNCHANGED <<serial, linkage, op, target, iterBudget,
                                childQueue, insertedIn, insertTarget,
                                releaseTarget, everInsertedIn, commitCount,
                                opParent>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget,
                           childQueue, insertedIn, insertTarget, releaseTarget,
                           everInsertedIn, commitCount, opParent>>

\* UnbundleWalk — walk up to child's bundledBy parent.
UnbundleWalk(t) ==
    /\ pc[t] = "unbundle_walk"
    /\ LET c == target[t]
           w == local[t].wrapper
           p == w.bundledBy
           pw == IF p \in Parents THEN linkage[p] ELSE Null
       IN
       IF p \in Parents /\ pw.hasPriority /\ pw.packet.sub[c] /= Null
       THEN /\ local' = [local EXCEPT
                ![t].parentWrapper = pw,
                ![t].oldpacket = pw.packet.sub[c],
                ![t].newpacket = MakePacket(c,
                    pw.packet.sub[c].payload + 1,
                    pw.packet.sub[c].sub,
                    pw.packet.sub[c].missing)]
            /\ opParent' = [opParent EXCEPT ![t] = p]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_ancestors"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                           priorityTag, insertedIn, insertTarget, releaseTarget,
                           everInsertedIn, commitCount>>
       ELSE IF ~(\E p2 \in Parents : insertedIn[<<c, p2>>])
       THEN /\ local' = [local EXCEPT ![t].commitOk = "fail"]
            /\ pc' = [pc EXCEPT ![t] = "commit_done"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue,
                           priorityTag, insertedIn, insertTarget, releaseTarget,
                           everInsertedIn, commitCount, opParent>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget,
                           childQueue, priorityTag, insertedIn, insertTarget,
                           releaseTarget, everInsertedIn, commitCount, opParent>>

UnbundleCASAncestors(t) ==
    /\ pc[t] = "unbundle_cas_ancestors"
    /\ opParent[t] /= Null
    /\ LET p == opParent[t] IN CanProceed(t, p)
    /\ LET p == opParent[t]
           c == target[t]
           oldPW == local[t].parentWrapper
           ser == GenSerial(t, oldPW.serial)
           newPkt == MakePacket(p, oldPW.packet.payload, oldPW.packet.sub, TRUE)
           newPW == PriorityWrapper(newPkt, ser)
       IN
       IF oldPW.hasPriority /\ linkage[p] = oldPW
       THEN /\ linkage' = [linkage EXCEPT ![p] = newPW]
            /\ UpdateSerial(t, ser)
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_child"]
            /\ priorityTag' = [priorityTag EXCEPT ![p] = TagAfterSuccess(t, p)]
            /\ UNCHANGED <<local, op, target, iterBudget, childQueue, insertedIn,
                           insertTarget, releaseTarget, everInsertedIn,
                           commitCount, opParent>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![opParent[t]] = TagAfterFail(t, opParent[t])]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget,
                           childQueue, insertedIn, insertTarget, releaseTarget,
                           everInsertedIn, commitCount, opParent>>

UnbundleCASChild(t) ==
    /\ pc[t] = "unbundle_cas_child"
    /\ CanProceed(t, target[t])
    /\ LET c == target[t]
           oldCW == local[t].wrapper
           ser   == GenSerial(t, oldCW.serial)
           newCW == PriorityWrapper(local[t].newpacket, ser)
       IN
       IF linkage[c] = oldCW
       THEN /\ linkage' = [linkage EXCEPT ![c] = newCW]
            /\ UpdateSerial(t, ser)
            /\ local' = [local EXCEPT ![t].commitOk = "ok"]
            /\ pc' = [pc EXCEPT ![t] = "commit_done"]
            /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterSuccess(t, c)]
            /\ UNCHANGED <<op, target, iterBudget, childQueue, insertedIn,
                           insertTarget, releaseTarget, everInsertedIn,
                           commitCount, opParent>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget,
                           childQueue, insertedIn, insertTarget, releaseTarget,
                           everInsertedIn, commitCount, opParent>>

CommitDone(t) ==
    /\ pc[t] = "commit_done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ op' = [op EXCEPT ![t] = "idle"]
    /\ LET node == target[t]
           success == local[t].commitOk = "ok"
           newQueue == IF success THEN childQueue[t] \ {node} ELSE childQueue[t]
       IN
       /\ target' = [target EXCEPT ![t] = Null]
       /\ local' = [local EXCEPT ![t] = InitLocal]
       /\ opParent' = [opParent EXCEPT ![t] = Null]
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
    /\ UNCHANGED <<serial, linkage, insertedIn, insertTarget, releaseTarget,
                   everInsertedIn>>

-----------------------------------------------------------------------------
(* Release actions *)

ReleaseStart(t) ==
    /\ pc[t] = "idle"
    /\ t \in ReleaseThreads
    /\ releaseTarget[t] = Null
    /\ insertTarget[t] = Null
    /\ childQueue[t] = {}
    /\ iterBudget[t] = 0
    /\ \A c \in AllChildren : \A p \in Parents : everInsertedIn[<<c, p>>]
    /\ \E p \in Parents :
        /\ insertedIn[<<DynChild1, p>>]
        /\ \A t2 \in Threads \ {t} : releaseTarget[t2] /= p
        /\ releaseTarget' = [releaseTarget EXCEPT ![t] = p]
        /\ opParent' = [opParent EXCEPT ![t] = p]
        /\ op' = [op EXCEPT ![t] = "release"]
        /\ pc' = [pc EXCEPT ![t] = "release_snap"]
    /\ UNCHANGED <<serial, linkage, target, local, priorityTag, insertedIn,
                   insertTarget, everInsertedIn, iterBudget, childQueue,
                   commitCount>>

ReleaseCASParent(t) ==
    /\ pc[t] = "release_cas_parent"
    /\ releaseTarget[t] /= Null
    /\ LET p == releaseTarget[t] IN CanProceed(t, p)
    /\ LET p == releaseTarget[t]
           c == DynChild1
           oldW == local[t].wrapper
           snapPkt == local[t].snapResult
           childPkt == snapPkt.sub[c]
           newSub == [snapPkt.sub EXCEPT ![c] = Null]
           newPkt == MakePacket(p, snapPkt.payload, newSub, snapPkt.missing)
           ser == GenSerial(t, oldW.serial)
           newW == PriorityWrapper(newPkt, ser)
       IN
       IF linkage[p] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![p] = newW]
            /\ UpdateSerial(t, ser)
            /\ insertedIn' = [insertedIn EXCEPT ![<<c, p>>] = FALSE]
            /\ local' = [local EXCEPT ![t].oldpacket = childPkt,
                                       ![t].wrapper = Null]
            /\ pc' = [pc EXCEPT ![t] = "release_read_child"]
            /\ priorityTag' = [priorityTag EXCEPT ![p] = TagAfterSuccess(t, p)]
            /\ UNCHANGED <<op, target, insertTarget, releaseTarget, everInsertedIn,
                           iterBudget, childQueue, commitCount, opParent>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "release_snap"]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ priorityTag' = [priorityTag EXCEPT ![releaseTarget[t]] = TagAfterFail(t, releaseTarget[t])]
            /\ UNCHANGED <<serial, linkage, op, target, insertedIn, insertTarget,
                           releaseTarget, everInsertedIn, iterBudget,
                           childQueue, commitCount, opParent>>

ReleaseReadChild(t) ==
    /\ pc[t] = "release_read_child"
    /\ releaseTarget[t] /= Null
    /\ LET c == DynChild1
           cw == linkage[c]
       IN
       IF cw.hasPriority
       THEN /\ pc' = [pc EXCEPT ![t] = "idle"]
            /\ op' = [op EXCEPT ![t] = "idle"]
            /\ target' = [target EXCEPT ![t] = Null]
            /\ releaseTarget' = [releaseTarget EXCEPT ![t] = Null]
            /\ opParent' = [opParent EXCEPT ![t] = Null]
            /\ local' = [local EXCEPT ![t] = InitLocal]
            /\ priorityTag' = ClearMyTags(t)
            /\ UNCHANGED <<serial, linkage, insertedIn, insertTarget,
                           everInsertedIn, iterBudget, childQueue, commitCount>>
       ELSE /\ local' = [local EXCEPT ![t].wrapper = cw]
            /\ pc' = [pc EXCEPT ![t] = "release_cas_child"]
            /\ UNCHANGED <<serial, linkage, op, target, priorityTag, insertedIn,
                           insertTarget, releaseTarget, everInsertedIn,
                           iterBudget, childQueue, commitCount, opParent>>

ReleaseCASChild(t) ==
    /\ pc[t] = "release_cas_child"
    /\ releaseTarget[t] /= Null
    /\ LET c == DynChild1
           oldCW == local[t].wrapper
           cpkt == local[t].oldpacket
           ser == GenSerial(t, oldCW.serial)
           newCW == PriorityWrapper(cpkt, ser)
       IN
       /\ CanProceed(t, c)
       /\ IF linkage[c] = oldCW
          THEN /\ linkage' = [linkage EXCEPT ![c] = newCW]
               /\ UpdateSerial(t, ser)
               /\ pc' = [pc EXCEPT ![t] = "idle"]
               /\ op' = [op EXCEPT ![t] = "idle"]
               /\ target' = [target EXCEPT ![t] = Null]
               /\ releaseTarget' = [releaseTarget EXCEPT ![t] = Null]
               /\ opParent' = [opParent EXCEPT ![t] = Null]
               /\ local' = [local EXCEPT ![t] = InitLocal]
               /\ priorityTag' = ClearMyTags(t)
               /\ UNCHANGED <<insertedIn, insertTarget, everInsertedIn,
                              iterBudget, childQueue, commitCount>>
          ELSE /\ pc' = [pc EXCEPT ![t] = "release_read_child"]
               /\ local' = [local EXCEPT ![t].wrapper = Null]
               /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
               /\ UNCHANGED <<serial, linkage, op, target, insertedIn,
                              insertTarget, releaseTarget, everInsertedIn,
                              iterBudget, childQueue, commitCount, opParent>>

-----------------------------------------------------------------------------
(* SkipIteration — drain when all children released (from all parents) *)

SkipIteration(t) ==
    /\ pc[t] = "idle"
    /\ t \in RootThreads \cup LeafThreads
    /\ iterBudget[t] > 0
    /\ childQueue[t] = {}
    /\ insertTarget[t] = Null
    /\ releaseTarget[t] = Null
    /\ \A c \in AllChildren : \A p \in Parents : everInsertedIn[<<c, p>>]
    /\ ~(\E c \in AllChildren : \E p \in Parents : insertedIn[<<c, p>>])
    /\ iterBudget' = [iterBudget EXCEPT ![t] = 0]
    /\ priorityTag' = ClearMyTags(t)
    /\ UNCHANGED <<serial, linkage, pc, op, target, local, insertedIn,
                   insertTarget, releaseTarget, everInsertedIn, childQueue,
                   commitCount, opParent>>

-----------------------------------------------------------------------------
(* Next *)

AllDone ==
    /\ \A t \in Threads : pc[t] = "idle"
    /\ \A t \in Threads : insertTarget[t] = Null
    /\ \A t \in Threads : releaseTarget[t] = Null
    /\ \A t \in Threads : childQueue[t] = {}
    /\ \A t \in Threads : iterBudget[t] = 0
    /\ IF ReleaseThreads /= {}
       THEN \A c \in AllChildren : \A p \in Parents : ~insertedIn[<<c, p>>]
       ELSE TRUE

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
        \/ MigrateClearOther(t)
        \/ MigrateClearOtherFail(t)
        \/ BundlePhase3(t)
        \/ BundlePhase3Fail(t)
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

Terminating == AllDone /\ UNCHANGED vars

Next == NextStep \/ Terminating \/ Waiting

Spec == Init /\ [][Next]_vars /\ WF_vars(NextStep)

-----------------------------------------------------------------------------
(* Safety invariants *)

\* SnapshotConsistency(p): when p has priority and is fully bundled
\* (~missing), every child reachable from p (insertedIn=TRUE) AND
\* homed at p (bundledBy = p) has its sub-packet populated.
SnapshotConsistency ==
    \A p \in Parents :
        LET pw == linkage[p]
        IN  (pw.hasPriority /\ ~pw.packet.missing) =>
            (\A c \in AllChildren :
                (insertedIn[<<c, p>>]
                 /\ ~linkage[c].hasPriority
                 /\ linkage[c].bundledBy = p) =>
                    pw.packet.sub[c] /= Null)

\* HardlinkExclusive: at most one parent's sub[] holds child's packet.
HardlinkExclusive ==
    \A c \in AllChildren :
        ~(\E p1, p2 \in Parents :
            /\ p1 /= p2
            /\ linkage[p1].packet.sub[c] /= Null
            /\ linkage[p2].packet.sub[c] /= Null)

\* BundleRefConsistency: child.bundledBy parent must have priority.
\* When the parent is not missing, child's packet exists either in the
\* parent's sub[] OR in the child wrapper itself (InsertedRef state — a
\* legitimate intermediate during insert before the final parent CAS).
BundleRefConsistency ==
    \A c \in AllChildren :
        LET cw == linkage[c]
        IN  (~cw.hasPriority /\ cw.bundledBy \in Parents) =>
                /\ linkage[cw.bundledBy].hasPriority
                /\ (linkage[cw.bundledBy].packet.missing
                    \/ linkage[cw.bundledBy].packet.sub[c] /= Null
                    \/ cw.packet /= Null)

\* NoPriorityLoss: child is owned (priority) OR bundled.
NoPriorityLoss ==
    \A c \in AllChildren :
        LET w == linkage[c]
        IN  w.hasPriority \/ w.bundledBy \in Parents

\* MissingPropagation: when a parent has bundled child packet, child is
\* itself not missing.
MissingPropagation ==
    \A p \in Parents :
        LET pw == linkage[p]
        IN  (pw.hasPriority /\ ~pw.packet.missing) =>
            (\A c \in AllChildren :
                pw.packet.sub[c] /= Null => ~pw.packet.sub[c].missing)

\* ChildPayload(c): authoritative payload.
ChildPayload(c) ==
    LET cw == linkage[c]
    IN  IF cw.hasPriority
        THEN cw.packet.payload
        ELSE IF cw.bundledBy \in Parents
             THEN LET ps == linkage[cw.bundledBy].packet.sub[c]
                  IN  IF ps /= Null THEN ps.payload ELSE 0
             ELSE 0

\* TerminalPayloadCheck: at termination, payload = commitCount.
\* (Pre-init topology starts each child at payload=0; no insert
\* increments since insert pipeline is disabled in this minimal model.)
TerminalPayloadCheck ==
    AllDone =>
        \A c \in AllChildren :
            (\E p \in Parents : everInsertedIn[<<c, p>>])
            => ChildPayload(c) = commitCount[c]

\* QuiescentCheck: at every all-idle moment with child still in tree,
\* payload matches commitCount.
QuiescentCheck ==
    (\A t \in Threads : pc[t] = "idle") =>
        \A c \in AllChildren :
            (\E p \in Parents : insertedIn[<<c, p>>]) =>
                ChildPayload(c) = commitCount[c]

\* EventuallyAllDone: liveness check.
EventuallyAllDone == <>AllDone

=============================================================================
