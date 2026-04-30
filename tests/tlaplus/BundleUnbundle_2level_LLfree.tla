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
------------------------- MODULE BundleUnbundle_2level_LLfree -------------------------
(*
 * Variant of BundleUnbundle_2level that mirrors KAME's livelock-free
 * negotiate mechanism explicitly so TLC's exhaustive search reaches a
 * finite state space without modular-serial wraparound, while still
 * enforcing the strict TerminalPayloadCheck count assertion.
 *
 * Key differences from the base 2level spec:
 *   - Lamport serial is plain Naturals (no `% MaxSerial`). The LL-free
 *     priority-tag mechanism bounds total wrapper churn so SerialBound
 *     CONSTRAINT can be relaxed (or kept large as a safety net).
 *   - New variable priorityTag[n]: per-node (Null | <<iter, tid>>) tag.
 *     Set by a thread when its CAS fails at a "negotiate point" (the
 *     places C++ calls m_link->negotiate()); other threads see the tag
 *     and only proceed if it's Null or matches their own (iter, tid).
 *     Older transactions (smaller iter, then smaller tid) win.
 *   - Tag is cleared by the holder when its CAS finally succeeds —
 *     mirroring C++ release_privileged_tidstamp().
 *   - C++ optimization to skip tag operations under low contention
 *     (count-based threshold) is omitted; the model always tags on
 *     failure, matching the worst-case behavior verifiable by TLC.
 *   - iter is a derived quantity: iter(t) == MaxCommits - iterBudget[t].
 *     No separate variable.
 *
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
    MaxSerial,           \* Serial upper bound for CONSTRAINT SerialBound (not a modulus).
                         \* TLC prunes branches where any serial reaches MaxSerial.
                         \* Serial arithmetic is plain natural-number (no wrap); set large
                         \* enough that realistic paths are not pruned prematurely.
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
\*   C++-faithful: TID encoded in lower digit, counter in upper.
\*   See GenSerial / EncodeSerial (mirrors transaction.h:547-576).
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
    serial, linkage, pc, op, target, local,
    iterBudget,   \* [Threads -> 0..MaxCommits]: remaining full iterations per thread
    childQueue,   \* [Threads -> SUBSET Children]: children pending CommitChild in current iteration
    priorityTag   \* [Nodes -> Null | <<iter, tid>>]: LL-free negotiate tag per node.
                  \* Set by a thread when its CAS at this node fails;
                  \* gates other threads via CanProceed. Cleared on success
                  \* by the holder. Older transactions (smaller iter, then
                  \* smaller tid) preempt younger ones. Mirrors C++
                  \* m_priority_tidstamp at every negotiate point — no
                  \* count-based skip optimisation.

vars == <<serial, linkage, pc, op, target, local, iterBudget, childQueue, priorityTag>>

-----------------------------------------------------------------------------
(* Data structures *)

PriorityWrapper(packet, ser) ==
    [packet |-> packet, hasPriority |-> TRUE, bundledBy |-> Null, serial |-> ser]

BundledRefWrapper(parentNode, ser) ==
    [packet |-> Null, hasPriority |-> FALSE, bundledBy |-> parentNode, serial |-> ser]

MakePacket(node, payload, sub, miss) ==
    [payload |-> payload, sub |-> sub, missing |-> miss, node |-> node]

EmptySub == [c \in Children |-> Null]

\* C++ FIDELITY [transaction.h:547-576, SerialGenerator]:
\*   gen(last_serial) uses TLS counter (upper 48 bits) + TID (lower 16 bits).
\*   Thread uniqueness comes from TID-in-lower-bits; same upper-bits counter
\*   on two threads still yields different serial values. TLA+ records
\*   compare by value, so we reproduce this via base-B arithmetic with
\*   B = SerialBase ≥ max(TID)+1. Bit width differs (TLA+ uses arbitrary
\*   integers) but ordering and uniqueness properties are identical.
\*   No globalSerial — C++ has none.
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

\* SerialBound: kept for cfg back-compat. Always TRUE — Lamport unbounded,
\* LL-free guarantees termination.
SerialBound == TRUE

\* ==========================================================================
\* LL-free negotiate helpers
\* ==========================================================================
\*
\* iter(t) is derived: completed full iterations so far. Combined with t
\* gives a total order <<iter, tid>> where smaller is "older" (= a
\* longer-running Transaction that should win contention to prevent
\* livelock). Mirrors C++ now_us_tagged() Lamport stamp at the
\* coarse-grained per-iteration level the model actually distinguishes.
iter(t) == MaxCommits - iterBudget[t]
MyTag(t) == <<iter(t), t>>

\* Strict tag-ordering: a is older than b. Total order on (iter, tid).
TagOlder(a, b) ==
    \/ a[1] < b[1]
    \/ (a[1] = b[1] /\ a[2] < b[2])

\* ActiveThread(t): thread t is mid-Transaction or has more iterations.
\* Stale tags from inactive threads must be ignored — otherwise a thread
\* that finished mid-protocol leaves zombie tags blocking peers (proof_semantics §4).
ActiveThread(t) == iterBudget[t] > 0 \/ pc[t] /= "idle"

\* CanProceed: gate for any CAS attempt at node n by thread t.
\* Matches C++: a thread proceeds when:
\*   (a) no privileged tidstamp registered, OR
\*   (b) the registered tidstamp's tid is mine (any iter — my own previous
\*       tag from same iteration's earlier CAS, or a stale tag I never
\*       cleared because the protocol moved me elsewhere), OR
\*   (c) the tag holder is no longer active (done all iterations and
\*       returned to idle) — its tag is provably zombie.
\* Older active threads first preempt the tag (see PreemptTag), then proceed.
CanProceed(t, n) ==
    LET tag == priorityTag[n] IN
    \/ tag = Null
    \/ /\ tag /= Null
       /\ \/ tag[2] = t
          \/ ~ActiveThread(tag[2])

\* TagAfterFail(t, n): the value priorityTag[n] should hold after thread t's
\* CAS at n fails. Cases:
\*   - No tag: register mine.
\*   - My own tag (any iter): refresh to current MyTag(t).
\*   - Other thread's tag, but holder inactive: take over with mine.
\*   - Other active thread's tag: keep older; younger thread's failure
\*     does not overwrite an older active thread's registration.
TagAfterFail(t, n) ==
    IF priorityTag[n] = Null
    THEN MyTag(t)
    ELSE IF priorityTag[n][2] = t
         THEN MyTag(t)
         ELSE IF ~ActiveThread(priorityTag[n][2])
              THEN MyTag(t)
              ELSE IF TagOlder(MyTag(t), priorityTag[n])
                   THEN MyTag(t)
                   ELSE priorityTag[n]

\* TagAfterSuccess(t, n): on CAS success at n, KEEP the tag — Transaction-
\* scope persistence (matches C++ ScopedNegotiateLinkage outer-scope which
\* releases only at scope end, not at each inner CAS). Per-CAS clearing
\* allows a peer Transaction to race in between CAS phases of the holding
\* Transaction and force endless re-bundling. The actual release happens
\* at Transaction boundaries via ClearMyTags below.
TagAfterSuccess(t, n) == priorityTag[n]

\* ClearMyTags(t): release ALL of thread t's tags at every node. Called at
\* Transaction-end transitions (CommitParent fail/success, CommitDone) to
\* implement "release on Transaction commit/abort" semantics.
ClearMyTags(t) ==
    [n \in Nodes |->
        IF priorityTag[n] = Null
        THEN priorityTag[n]
        ELSE IF priorityTag[n][2] = t
             THEN Null
             ELSE priorityTag[n]]

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
    /\ serial = [t \in Threads |-> EncodeSerial(0, t)]
    /\ pc = [t \in Threads |-> "idle"]
    /\ op = [t \in Threads |-> "idle"]
    /\ target = [t \in Threads |-> Null]
    /\ local = [t \in Threads |-> InitLocal]
    /\ iterBudget = [t \in Threads |-> MaxCommits]
    /\ childQueue = [t \in Threads |-> {}]
    /\ priorityTag = [n \in Nodes |-> Null]

\* @c11_action PreemptTag(t, n):
\*   An older thread can replace a younger thread's tag at a node, allowing
\*   it to subsequently proceed via CanProceed. Mirrors C++
\*   try_register_privileged_tidstamp() succeeding for an older Transaction.
\*   Without this action, a younger thread that grabbed the tag first could
\*   permanently lock out an older thread.
PreemptTag(t, n) ==
    /\ ActiveThread(t)
    /\ priorityTag[n] /= Null
    /\ priorityTag[n][2] /= t       \* not already mine
    /\ ActiveThread(priorityTag[n][2])  \* if holder inactive, CanProceed handles it
    /\ TagOlder(MyTag(t), priorityTag[n])
    /\ priorityTag' = [priorityTag EXCEPT ![n] = MyTag(t)]
    /\ UNCHANGED <<serial, linkage, pc, op, target, local, iterBudget, childQueue>>

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
    /\ UNCHANGED <<serial, linkage, local, iterBudget, childQueue, priorityTag>>

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
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
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
            /\ UNCHANGED <<linkage, op, target, iterBudget, childQueue, priorityTag>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue, priorityTag>>

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
                 /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
            ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
                 /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue, priorityTag>>
       ELSE \* Fine: collect one unprocessed child per step
            LET parentW == local[t].wrapper
                ser     == local[t].bundleSer
            IN
            \* --- #1 superfine: Pre-bundle serial CAS (C++ bundle() entry, line 1182-1191) ---
            IF BundleCollectAtomic = "superfine" /\ parentW.serial /= ser
            THEN /\ CanProceed(t, Parent)
                 /\ LET newW == PriorityWrapper(parentW.packet, ser)
                    IN
                    IF linkage[Parent] = parentW
                    THEN /\ linkage' = [linkage EXCEPT ![Parent] = newW]
                         /\ local' = [local EXCEPT ![t].wrapper = newW]
                         /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                         /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterSuccess(t, Parent)]
                         /\ UNCHANGED <<serial, op, target, iterBudget, childQueue>>
                    ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
                         /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
                         /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>
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
                          /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
                     ELSE \* Collection failed.
                          IF BundleCollectAtomic = "superfine"
                             /\ linkage[Parent] = parentW
                          THEN \* superfine + parent unchanged — retry same child.
                               \* Eager Child tag on retry: mirrors C++
                               \* ScopedNegotiateLinkage at transaction_impl.h
                               \* :2431 placed BEFORE the fast-path read
                               \* check (reverted to original layout 2026-04-29
                               \* per benchmark; eager tag fires on
                               \* child_retry > 0 even when the read turns
                               \* into a fast-path no-op).
                               /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                               /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                               /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>
                          ELSE \* fine: always restart; superfine: parent changed.
                               \* Eager Parent tag: mirrors C++ outer-scope
                               \* ScopedNegotiateLinkage at line 2407
                               \* (eager tag on retry > 0 of bundle's retry
                               \* loop). Without this, peer can CAS Parent
                               \* during this thread's restart cycle, causing
                               \* unbounded re-bundling.
                               /\ pc' = [pc EXCEPT ![t] = "snap_read"]
                               /\ local' = [local EXCEPT
                                       ![t].subwrappers = [c2 \in Children |-> Null],
                                       ![t].subpackets  = EmptySub]
                               /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
                               /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue>>

\* @c11_action BundlePhase2(t):
\*   // Phase 2: CAS parent's linkage with new packet (still missing=TRUE)
\*   parent->m_link->compareAndSet(oldsuperwrapper, superwrapper);
\*   Source: transaction_impl.h:1249-1258
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
            /\ UNCHANGED <<serial, op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>

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
       THEN \* Coarse: all children CAS'd atomically (all-or-nothing).
            \* Gate on every child individually (matches C++ per-child negotiate
            \* semantics — each child is a CAS target).
            /\ \A c \in Children : CanProceed(t, c)
            /\ IF allMatch
               THEN /\ linkage' = [n \in Nodes |->
                        IF n \in Children
                        THEN BundledRefWrapper(Parent, ser)
                        ELSE linkage[n]]
                    /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                    /\ priorityTag' = [n \in Nodes |->
                           IF n \in Children
                           THEN TagAfterSuccess(t, n)
                           ELSE priorityTag[n]]
                    /\ UNCHANGED <<serial, local, op, target, iterBudget, childQueue>>
               ELSE /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                    /\ priorityTag' = [n \in Nodes |->
                           IF n \in Children /\ linkage[n] /= childWs[n]
                           THEN TagAfterFail(t, n)
                           ELSE priorityTag[n]]
                    /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>
       ELSE \* Fine: one child per step
            \/ \E c \in Children :
                  /\ CanProceed(t, c)
                  /\ linkage[c] = childWs[c]
                  /\ linkage' = [linkage EXCEPT ![c] = BundledRefWrapper(Parent, ser)]
                  /\ LET allDone == \A c2 \in Children \ {c} :
                                        linkage[c2] = BundledRefWrapper(Parent, ser)
                     IN
                     IF allDone
                     THEN pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                     ELSE pc' = [pc EXCEPT ![t] = "bundle_phase3"]
                  /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterSuccess(t, c)]
                  /\ UNCHANGED <<serial, local, op, target, iterBudget, childQueue>>
            \* Failure path: some child changed — rollback and restart.
            \* superfine: check bundle_serial/parent first (C++ transaction_impl.h:1274-1280).
            \/ \E c \in Children :
                  /\ CanProceed(t, c)
                  /\ childWs[c] /= Null
                  /\ linkage[c] /= childWs[c]
                  /\ LET disturbed ==
                          BundlePhase3Atomic = "superfine"
                          /\ (\/ \E c2 \in Children :
                                     /\ linkage[c2] /= childWs[c2]
                                     /\ linkage[c2].serial /= ser
                               \/ linkage[Parent] /= local[t].wrapper)
                     IN
                     IF disturbed
                     THEN \* superfine DISTURBED — restart from snapshot.
                          \* Clear local bundle state so the next SnapCheck
                          \* sees a fresh start (matches C++ where bundle()
                          \* returning DISTURBED tears down its stack frame
                          \* and the caller restarts with fresh locals).
                          \* Also eagerly tag Parent (mirrors C++ outer-scope
                          \* ScopedNegotiateLinkage at line 2179 of snapshot()
                          \* retry loop on retry > 0 — DISTURBED return
                          \* triggers outer retry, which eagerly tags Parent
                          \* so peer cannot race in during the next snapshot
                          \* attempt).
                          /\ pc' = [pc EXCEPT ![t] = "snap_read"]
                          /\ local' = [local EXCEPT
                                 ![t].wrapper = Null,
                                 ![t].subwrappers = [c2 \in Children |-> Null],
                                 ![t].subpackets = EmptySub]
                          /\ priorityTag' = [
                                 [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                                     EXCEPT ![Parent] = TagAfterFail(t, Parent)]
                          /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue>>
                     ELSE \* #3: No rollback — restart Phase1 (re-collect re-adopts bundled children)
                          /\ local' = [local EXCEPT
                                 ![t].subwrappers = [c2 \in Children |-> Null],
                                 ![t].subpackets  = EmptySub]
                          /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                          /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                          /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue>>

\* @c11_action BundlePhase4(t):
\*   // Phase 4: finalize -- clear missing flag, CAS parent
\*   superwrapper = new PacketWrapper(*superwrapper, bundle_serial);
\*   newpacket->m_missing = false;  // all sub-packets present
\*   parent->m_link->compareAndSet(oldsuperwrapper, superwrapper);
\*   Source: transaction_impl.h:1286-1299
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
            /\ local' = [local EXCEPT ![t].snapResult = finalPkt]
            /\ pc' = [pc EXCEPT ![t] = "commit_parent"]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterSuccess(t, Parent)]
            /\ UNCHANGED <<serial, op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "snap_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>

-----------------------------------------------------------------------------
(* Commit *)

\* @c11_action CommitParent(t):
\*   Commit ALL children's payload changes under Parent's scope (one transaction).
\*   C++: Transaction<XN> tr(Parent); for each c: tr[c].m_x += 1; tr.commit();
\*   Uses snapResult (from completed snapshot) to build new packet with ALL children
\*   incremented, then CAS. Sets childQueue to Children for subsequent per-child commits.
CommitParent(t) ==
    /\ pc[t] = "commit_parent"
    /\ CanProceed(t, Parent)
    /\ childQueue[t] = {}
    /\ local[t].snapResult /= Null
    /\ \A c \in Children : local[t].snapResult.sub[c] /= Null
    /\ LET pw      == linkage[Parent]
           snapPkt == local[t].snapResult
           newSub  == [c \in Children |->
               MakePacket(c,
                   snapPkt.sub[c].payload + 1,
                   snapPkt.sub[c].sub,
                   snapPkt.sub[c].missing)]
           newPkt  == MakePacket(Parent, snapPkt.payload, newSub, snapPkt.missing)
           ser     == GenSerial(t, pw.serial)
           newPW   == PriorityWrapper(newPkt, ser)
       IN
       \/ \* CAS success: commit and move to per-child phase
          \* Transaction-end: release all my tags via ClearMyTags.
          /\ pw.hasPriority
          /\ pw.packet = snapPkt
          /\ linkage' = [linkage EXCEPT ![Parent] = newPW]
          /\ UpdateSerial(t, ser)
          /\ childQueue' = [childQueue EXCEPT ![t] = Children]
          /\ pc' = [pc EXCEPT ![t] = "idle"]
          /\ op' = [op EXCEPT ![t] = "idle"]
          /\ target' = [target EXCEPT ![t] = Null]
          /\ local' = [local EXCEPT ![t] = InitLocal]
          /\ priorityTag' = ClearMyTags(t)
          /\ UNCHANGED iterBudget
       \/ \* CAS failure: retry from snapshot (iterate_commit semantics).
          \* On TagAfterFail at Parent, then end Transaction (back to snap_read
          \* via idle); we keep the tag so peers see our priority on retry.
          \* But all OTHER nodes' tags from this Transaction are released
          \* (via ClearMyTags then EXCEPT to refresh Parent).
          /\ ~(pw.hasPriority /\ pw.packet = snapPkt)
          /\ pc' = [pc EXCEPT ![t] = "idle"]
          /\ op' = [op EXCEPT ![t] = "idle"]
          /\ target' = [target EXCEPT ![t] = Null]
          /\ local' = [local EXCEPT ![t] = InitLocal]
          /\ priorityTag' = [ClearMyTags(t) EXCEPT ![Parent] = TagAfterFail(t, Parent)]
          /\ UNCHANGED <<serial, linkage, iterBudget, childQueue>>

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
    /\ UNCHANGED <<serial, linkage, local, iterBudget, childQueue, priorityTag>>

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
                                     w.packet.payload + 1,
                                     w.packet.sub,
                                     w.packet.missing)]
            /\ pc' = [pc EXCEPT ![t] = "commit_try_cas"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
       ELSE /\ local' = [local EXCEPT ![t].wrapper = w]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>

\* @c11_action CommitTryCAS(t):
\*   // Direct commit (hasPriority path)
\*   Source: transaction_impl.h:1368-1400
\*
\*   Fidelity note (#7): C++ creates newwrapper once with tr.m_serial and reuses it
\*   across inner retries. TLA+ calls GenSerial each time, which may produce a higher
\*   serial after adopting a wrapper with a newer serial. No correctness impact.
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
                 /\ priorityTag' = [priorityTag EXCEPT ![childNode] = TagAfterFail(t, childNode)]
                 /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue>>
            ELSE /\ local' = [local EXCEPT ![t].commitOk = "fail"]
                 /\ pc' = [pc EXCEPT ![t] = "commit_done"]
                 /\ priorityTag' = [priorityTag EXCEPT ![childNode] = TagAfterFail(t, childNode)]
                 /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![childNode] = TagAfterFail(t, childNode)]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>

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
                       parentW.packet.sub[childNode].payload + 1,
                       parentW.packet.sub[childNode].sub,
                       parentW.packet.sub[childNode].missing)]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_ancestors"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue, priorityTag>>

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
    /\ CanProceed(t, Parent)
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
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterSuccess(t, Parent)]
            /\ UNCHANGED <<local, op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![Parent] = TagAfterFail(t, Parent)]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>

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
            /\ UNCHANGED <<op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![childNode] = TagAfterFail(t, childNode)]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>

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
    \* Transaction-end: release all my tags. On success, the next per-child
    \* Transaction starts fresh; on fail, retry will re-tag on its own CAS
    \* failure (no need to preserve stale tag here).
    /\ priorityTag' = ClearMyTags(t)
    /\ UNCHANGED <<serial, linkage>>

-----------------------------------------------------------------------------
(* Next-state relation *)

\* AllDone: terminal condition — every thread has consumed its full
\* iteration budget and returned to idle. Required for the Terminating
\* disjunct (proof_semantics.md §6) so cfg can drop the `-deadlock` flag
\* and use default `CHECK_DEADLOCK TRUE` to detect *real* stuck states.
AllDone == \A t \in Threads : pc[t] = "idle" /\ iterBudget[t] = 0

\* NextStep: every action that can make real progress. PreemptTag is
\* included here because an older thread proceeding via tag preemption
\* IS forward progress in the LL-free protocol.
NextStep ==
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
        \/ \E n \in Nodes : PreemptTag(t, n)

\* Terminating: self-loop at AllDone so deadlock detection can stay on
\* (no spurious deadlock at the legitimate terminal state). Excluded
\* from WF below — otherwise WF would be vacuously satisfied by
\* eternal stuttering, defeating any liveness check.
Terminating ==
    /\ AllDone
    /\ UNCHANGED vars

Next == NextStep \/ Terminating

Spec == Init /\ [][Next]_vars /\ WF_vars(NextStep)

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

\* NoSerialWrapAround: retired. With plain natural-number serial arithmetic and
\* CONSTRAINT SerialBound, serials are always totally ordered by > and never
\* ambiguous. This predicate is trivially TRUE; use SerialBound as CONSTRAINT
\* instead of this as INVARIANT.
NoSerialWrapAround == TRUE

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

\* DebugSerialBound: NEUTERED — Lamport-style GenSerial (TID-encoded
\* counter, mirrors C++ SerialGenerator) advances unboundedly per wrapper
\* allocation. LL-free priority gating now guarantees termination. Kept
\* TRUE so existing cfg files referencing this name don't break.
DebugSerialBound == TRUE

\* PrintTerminalSerial: debug invariant. Always TRUE; emits the
\* per-thread serial array to TLC stdout the first time any AllDone
\* state is visited.
PrintTerminalSerial ==
    \/ ~AllDone
    \/ PrintT(<<"Terminal serial[t]:", serial>>)

\* EventuallyAllDone: under WF, every execution eventually reaches AllDone.
\* This is the key LL-free property — if priority gating is correct, no
\* thread can be starved indefinitely, so all threads eventually finish.
\* Mirrors proof_semantics.md §6.
EventuallyAllDone == <>AllDone

\* TerminalPayloadCheck: at termination (all threads: iterBudget=0 and idle),
\* each child received exactly 2 * MaxCommits * |Threads| payload increments:
\*   - MaxCommits * |Threads| from CommitParent (ALL children incremented per iteration)
\*   - MaxCommits * |Threads| from CommitChild (one direct commit per child per iteration)
\* The expected final payload is deterministic, so no tracking variable is needed.
\* Kept as iter-arithmetic sanity check; QuiescentCheck (below) is the
\* primary correctness check now.
TerminalPayloadCheck ==
    (\A t \in Threads : iterBudget[t] = 0 /\ pc[t] = "idle") =>
        \A c \in Children :
            /\ linkage[c].hasPriority
            /\ linkage[c].packet.payload =
                   2 * MaxCommits * Cardinality(Threads)

\* QuiescentCheck: expected child payload at any all-idle moment, derived
\* from existing state without an auxiliary commit_count variable. Fires
\* earlier than TerminalPayloadCheck (at every iteration boundary, not
\* just AllDone), so lost-increment regressions surface in O(thousands)
\* of states instead of O(millions). Always-on production check.
\*
\* Per-thread per-child contribution at all-idle:
\*   completed(t) = MaxCommits - iterBudget[t]    (full iterations done)
\*   midIter(t)   = childQueue[t] /= {}           (CommitParent done in current
\*                                                 iter, some per-child commits
\*                                                 may still pend)
\*   For child c, t's contribution =
\*     2 * completed(t)
\*     + (1 if midIter(t) else 0)        partial: parent commit in current iter
\*     + (1 if midIter(t) /\ c \notin childQueue[t] else 0)
\*                                       partial: direct commit done for c
RECURSIVE SumPayloadOver(_, _)
SumPayloadOver(S, c) ==
    IF S = {} THEN 0
    ELSE LET t == CHOOSE x \in S : TRUE
             completed == MaxCommits - iterBudget[t]
             midIter   == childQueue[t] /= {}
             grandThis == IF midIter THEN 1 ELSE 0
             directThis == IF midIter /\ c \notin childQueue[t] THEN 1 ELSE 0
         IN  2 * completed + grandThis + directThis
              + SumPayloadOver(S \ {t}, c)

QuiescentCheck ==
    (\A t \in Threads : pc[t] = "idle") =>
        \A c \in Children :
            \* Implication: only check priority (= unbundled) children;
            \* mid-iteration children may still be bundled.
            linkage[c].hasPriority =>
                linkage[c].packet.payload = SumPayloadOver(Threads, c)

=============================================================================
