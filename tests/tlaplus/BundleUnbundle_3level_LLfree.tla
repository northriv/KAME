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
------------------------- MODULE BundleUnbundle_3level_LLfree -------------------------
(*
 * 3-level LL-free variant of BundleUnbundle.tla. Mirrors the LL-free
 * design proven in BundleUnbundle_2level_LLfree.tla:
 *   - Lamport serial as plain Nat (no MOD); state-space finiteness via
 *     priority gating, not CONSTRAINT SerialBound.
 *   - priorityTag[n] per node (Null | <<iter, tid>>); set on CAS fail,
 *     released only at Transaction-end via ClearMyTags.
 *   - CanProceed gate at every CAS site; older-Tx wins; PreemptTag
 *     allows older active threads to take over younger tags.
 *   - ActiveThread guards against zombie tags from finished threads.
 *   - Eager Parent/Grand tag on snap_read restart paths (mirrors C++
 *     ScopedNegotiateLinkage outer scope at retry > 0).
 *   - Terminating disjunct + EventuallyAllDone PROPERTY for liveness
 *     checking with default CHECK_DEADLOCK TRUE (no `-deadlock` flag).
 *   - MaxPayload MOD removed; payload is monotone Nat counter.
 *
 * 3-level specifics (vs 2-level):
 *   - Tree: Grand → Parent → {Child1, Child2}
 *   - CommitGrand replaces 2-level CommitParent (commits the root)
 *   - WalkUpChain / SnapshotForUnbundle traverse 2 levels (Child→Parent→Grand)
 *   - UnbundleCASLoop walks the ancestor chain (vs single UnbundleCASAncestors)
 *   - Recursive bundle: Grand bundles Parent, Parent bundles Children
 *
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
    MaxSerial,      \* Serial upper bound for safety-net CONSTRAINT (not a modulus).
                    \* Serial arithmetic is plain natural-number (no wrap); finite
                    \* state space comes from priorityTag gating, not this bound.
    MaxCommits,     \* Max CommitStart/CommitDone cycles per thread in "child" phase
    \* ------------------------------------------------------------
    \* Atomicity granularity for each of 4 bulk-operation sites.
    \* "coarse"    = single atomic action (matches recursive operator)
    \* "fine"      = one load/CAS per action, simplified failure paths
    \* "superfine" = (BundleCollectAtomic, BundlePhase3Atomic only)
    \*              fine + C++-faithful failure handling:
    \*              Phase1: retry same child if parent unchanged (#6)
    \*              Phase3: check bundle_serial/parent → DISTURBED (#4)
    \* ------------------------------------------------------------
    UnbundleWalkAtomic,   \* #1: SnapshotForUnbundle walk (coarse/fine/superfine)
                          \*     superfine: casTargets in root-first order (matching C++)
    UnbundleCASAtomic,    \* #2: unbundle CAS loop (coarse=all-at-once, fine=1/action)
    BundleCollectAtomic,  \* #3: BundlePhase1 child collection (coarse=all children, fine=1 child/action)
    BundlePhase3Atomic,   \* #4: BundlePhase3 child CAS (coarse=all-at-once, fine=1/action)
    Privilege             \* TRUE: LL-free priorityTag gating active (CanProceed gates CAS,
                          \*       PreemptTag fires, ClearMyTags releases at Tx-end). Main mode.
                          \* FALSE: tags neutralized (CanProceed=TRUE, TagAfterFail/ClearMyTags
                          \*        no-op, PreemptTag disabled). Sanity-check mode for comparing
                          \*        against old BundleUnbundle.tla semantics. Without privilege,
                          \*        finite state space is NOT guaranteed (no LL-free retry bound),
                          \*        so use only with small configs / CONSTRAINT SerialBound /
                          \*        SAFETY-only invariants (no liveness PROPERTY).

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
\*   Plain natural-number arithmetic (matching C++ 48-bit unsigned, no modular wrap).
\*   Comparisons use regular >. Bounded by CONSTRAINT SerialBound.
\*
\* 3-level adds: recursive bundle (Grand bundles Parent which bundles Children),
\* 2-level unbundle walk (Child->Parent->Grand via walkUpChain/snapshotForUnbundle recursion),
\* and UnbundleCASGP / UnbundleRestoreParent for the grandparent path.
\*
\* Source: kame/transaction.h, kame/transaction_impl.h
\* ==========================================================================

VARIABLES
    serial, linkage, pc, op, target, local,
    iterBudget,  \* [Threads -> 0..MaxCommits]: remaining full iterations per thread
    childQueue,  \* [Threads -> SUBSET ParentChildren]: children pending CommitChild in current iteration
    priorityTag  \* [AllNodes -> Null | <<iter, tid>>]: LL-free per-node tag.
                 \* Set on CAS fail; released only at Tx end via ClearMyTags.

vars == <<serial, linkage, pc, op, target, local, iterBudget, childQueue, priorityTag>>

-----------------------------------------------------------------------------
(* Lamport serial — C++-faithful encoding *)

\* C++ FIDELITY [transaction.h:547-576]:
\*   SerialGenerator stores a TLS counter in upper 48 bits + TID in lower 16
\*   bits of a 64-bit value. gen(last_serial) advances the TLS counter past
\*   last_serial's counter (Lamport step), increments, and returns the
\*   composed value. Same counter on two threads produces DIFFERENT serials
\*   because the lower bits differ — this is what makes wrappers thread-
\*   unique even when the Lamport timestamps collide.
\*
\*   TLA+ records compare by value, so we must reproduce the TID-in-low-bits
\*   trick. We use base-B arithmetic with B = SerialBase (>= max TID + 1):
\*     serial = counter * SerialBase + tid
\*   Bit width differs from C++ (TLA+ uses arbitrary integers) but the
\*   ordering and uniqueness properties are identical.
\*
\*   No globalSerial — C++ has no such variable. Thread uniqueness is
\*   guaranteed by the TID-in-lower-bits encoding alone.
SerialBase == 1 + Cardinality(Threads)   \* > max TID; thread IDs ∈ Threads

\* Decoding helpers — extract counter (upper) and tid (lower).
SerialCounter(s) == s \div SerialBase
SerialTID(s)     == s % SerialBase

\* Encoding helper — compose (counter, tid) → serial value.
EncodeSerial(cnt, tid) == cnt * SerialBase + tid

\* GenSerial(t, lastSer): mirrors C++ SerialGenerator::gen(last_serial).
\*   1. last_counter = SerialCounter(lastSer)        \* C++: last_serial & ~0xFFFF
\*   2. my_counter   = SerialCounter(serial[t])      \* C++: v.m_var upper bits
\*   3. if last_counter > my_counter: my_counter := last_counter   \* Lamport step
\*   4. my_counter   = my_counter + 1                \* C++: v++
\*   5. return EncodeSerial(my_counter, t)           \* C++: v (m_var)
\* Uses TLS (serial[t]) + lastSer only — no global state.
GenSerial(t, lastSer) ==
    LET lastCnt == SerialCounter(lastSer)
        myCnt   == SerialCounter(serial[t])
        newCnt  == (IF lastCnt > myCnt THEN lastCnt ELSE myCnt) + 1
    IN  EncodeSerial(newCnt, t)

\* UpdateSerial: persist thread t's TLS counter slot.
\* C++ stores back into stl_serial inside gen(); TLA+ does it explicitly.
UpdateSerial(t, ser) ==
    serial' = [serial EXCEPT ![t] = ser]

\* SerialBound: kept for cfg back-compat. With Lamport (unbounded) and
\* LL-free liveness guaranteeing termination, no hard cap is needed —
\* always TRUE.
SerialBound == TRUE

-----------------------------------------------------------------------------
(* LL-free helpers (mirrors BundleUnbundle_2level_LLfree.tla §190-235) *)

\* iter(t): completed full iterations so far. Combined with t gives
\* total order <<iter, tid>> where smaller is "older".
iter(t) == MaxCommits - iterBudget[t]
MyTag(t) == <<iter(t), t>>

\* Strict tag-ordering: a is older than b. Total order on (iter, tid).
TagOlder(a, b) ==
    \/ a[1] < b[1]
    \/ (a[1] = b[1] /\ a[2] < b[2])

\* ActiveThread(t): thread t is mid-Transaction or has more iterations.
\* Stale tags from inactive threads must be ignored.
ActiveThread(t) == iterBudget[t] > 0 \/ pc[t] /= "idle"

\* CanProceed: gate for any CAS attempt at node n by thread t.
\* When Privilege = FALSE, gate is disabled (always TRUE).
CanProceed(t, n) ==
    \/ ~Privilege
    \/ LET tag == priorityTag[n] IN
       \/ tag = Null
       \/ /\ tag /= Null
          /\ \/ tag[2] = t
             \/ ~ActiveThread(tag[2])

\* TagAfterFail(t, n): the value priorityTag[n] should hold after CAS fail.
\* When Privilege = FALSE, no-op (returns existing tag).
TagAfterFail(t, n) ==
    IF ~Privilege
    THEN priorityTag[n]
    ELSE IF priorityTag[n] = Null
         THEN MyTag(t)
         ELSE IF priorityTag[n][2] = t
              THEN MyTag(t)
              ELSE IF ~ActiveThread(priorityTag[n][2])
                   THEN MyTag(t)
                   ELSE IF TagOlder(MyTag(t), priorityTag[n])
                        THEN MyTag(t)
                        ELSE priorityTag[n]

\* TagAfterSuccess: NO-OP (Transaction-scope persistence; release at Tx end).
TagAfterSuccess(t, n) == priorityTag[n]

\* ClearMyTags(t): release ALL of thread t's tags. Called at Tx-end.
\* When Privilege = FALSE, no-op (priorityTag unchanged).
ClearMyTags(t) ==
    IF ~Privilege
    THEN priorityTag
    ELSE [n \in AllNodes |->
              IF priorityTag[n] = Null
              THEN priorityTag[n]
              ELSE IF priorityTag[n][2] = t
                   THEN Null
                   ELSE priorityTag[n]]

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
    walkWrapper |-> Null,      \* (fine UnbundleWalk) wrapper saved for walkNode
    innerChild  |-> Null,      \* (fine inner bundle) child being inner-bundled
    innerWrapper |-> Null,     \* (fine inner bundle) saved wrapper for innerChild
    innerSubWs  |-> Null       \* (fine inner bundle) [grandchildren -> wrapper]
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
    \* Initial TLS slots: counter=0, TID encoded in lower digit (C++ stl_serial
    \* constructor: m_var = ProcessCounter::id()).  EncodeSerial(0, t) = t.
    /\ serial = [t \in Threads |-> EncodeSerial(0, t)]
    /\ pc = [t \in Threads |-> "idle"]
    /\ op = [t \in Threads |-> "idle"]
    /\ target = [t \in Threads |-> Null]
    /\ local = [t \in Threads |-> InitLocal]
    /\ iterBudget = [t \in Threads |-> MaxCommits]
    /\ childQueue = [t \in Threads |-> {}]
    /\ priorityTag = [n \in AllNodes |-> Null]

\* @c11_action PreemptTag(t, n): older active thread takes over a younger
\* active thread's tag at node n. Mirrors C++
\* try_register_privileged_tidstamp() succeeding for an older Transaction.
PreemptTag(t, n) ==
    /\ Privilege  \* disabled when Privilege = FALSE
    /\ ActiveThread(t)
    /\ priorityTag[n] /= Null
    /\ priorityTag[n][2] /= t
    /\ ActiveThread(priorityTag[n][2])
    /\ TagOlder(MyTag(t), priorityTag[n])
    /\ priorityTag' = [priorityTag EXCEPT ![n] = MyTag(t)]
    /\ UNCHANGED <<serial, linkage, pc, op, target, local, iterBudget, childQueue>>

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
    /\ UNCHANGED <<serial, linkage, local, iterBudget, childQueue, priorityTag>>

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
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
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
            /\ UNCHANGED <<linkage, op, target, iterBudget, childQueue, priorityTag>>
       ELSE \* Node is bundled — need to unbundle first (snapshot from bundled state)
            \* For simplicity, just retry (the real code calls walkUpChain)
            /\ pc' = [pc EXCEPT ![t] = "snap_check"]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue, priorityTag>>

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
                 \* finalize inner-bundled children's wrappers.
                 \* C++ FIDELITY (transaction_impl.h:2487-2511, Phase 3 of inner
                 \* recursive bundle): each grandchild is ALWAYS CAS'd to a NEW
                 \* bundled_ref wrapper with the new bundle_serial, regardless of
                 \* whether it was previously priority or already bundled-by-us
                 \* (with an older bundle_serial). The serial refresh is what
                 \* invalidates a peer's stale snapshotForUnbundle pointer, so
                 \* the peer's final UnbundleCASChild CAS fails → returns
                 \* SUBVALUE_HAS_CHANGED → iterate_commit retries with the new
                 \* (higher) value, preventing lost increments. Earlier code
                 \* gated this update by `linkage[n].hasPriority`, which left
                 \* already-bundled grandchildren untouched and produced a TLA+
                 \* lost-increment that does NOT occur in C++.
                 /\ linkage' = [n \in AllNodes |->
                       IF \E c \in innerBundled : n \in ChildrenOf(c)
                       THEN BundledRefWrapper(ParentOf(n), ser)
                       ELSE IF n \in innerBundled
                       THEN PriorityWrapper(childPkts[n], ser)
                       ELSE linkage[n]]
                 /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
                 /\ UNCHANGED <<serial, op, target, iterBudget, childQueue, priorityTag>>
            ELSE \* Can't collect all — restart
                 /\ pc' = [pc EXCEPT ![t] = "snap_check"]
                 /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue, priorityTag>>
       ELSE \* fine/superfine: process one child at a time
            LET node     == local[t].bundleNode
                children  == ChildrenOf(node)
                parentW   == local[t].wrapper
                ser       == local[t].bundleSer
            IN
            \* --- #1 superfine: Pre-bundle serial CAS (C++ bundle() entry, line 1182-1191) ---
            \* Stamp bundle_serial on the parent's wrapper before collection.
            \* Without negotiate(), this causes livelock in fine mode (both threads
            \* alternate pre-CAS → Phase2 fail → retry, exhausting MaxSerial).
            IF BundleCollectAtomic = "superfine" /\ parentW.serial /= ser
            THEN /\ CanProceed(t, node)
                 /\ LET newW == PriorityWrapper(parentW.packet, ser)
                    IN
                    IF linkage[node] = parentW
                    THEN /\ linkage' = [linkage EXCEPT ![node] = newW]
                         /\ local' = [local EXCEPT ![t].wrapper = newW]
                         /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                         /\ UNCHANGED <<serial, op, target, iterBudget, childQueue, priorityTag>>
                    ELSE \* Parent changed — restart with eager tag.
                         /\ pc' = [pc EXCEPT ![t] = "snap_check"]
                         /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterFail(t, node)]
                         /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>
            ELSE \* Pre-CAS done, proceed to collection
            LET unprocessed == {c \in children : local[t].subwrappers[c] = Null}
            IN
            IF unprocessed = {}
            THEN \* All children collected — proceed to Phase2.
                 /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
                 /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue, priorityTag>>
            ELSE \E c \in unprocessed :
                     LET cw   == linkage[c]
                         pkt  == CollectSubpacket(node, c, parentW, ser)
                         needsInner == cw.hasPriority /\ cw.packet.missing
                                       /\ ChildrenOf(c) /= {}
                     IN
                     IF pkt = Null
                     THEN \* Disturbed while collecting this child.
                          IF BundleCollectAtomic = "superfine"
                             /\ linkage[node] = parentW
                          THEN \* superfine + parent unchanged — retry same child.
                               /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                               /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue, priorityTag>>
                          ELSE \* fine: always restart; superfine: parent changed.
                               \* Eager bundleNode tag: ports BundleUnbundle_2level_LLfree.tla
                               \* commit 5ff3226 fix (line 421-426 there).  Mirrors C++
                               \* outer-scope ScopedNegotiateLinkage at line 2407
                               \* (eager tag on retry > 0 of bundle's retry loop).
                               \* Without this, peer can CAS bundleNode during this
                               \* thread's restart cycle, causing unbounded re-bundling.
                               /\ pc' = [pc EXCEPT ![t] = "snap_check"]
                               /\ local' = [local EXCEPT
                                      ![t].subwrappers = [cc \in children |-> Null],
                                      ![t].subpackets  = [cc \in children |-> Null]]
                               /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterFail(t, node)]
                               /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue>>
                     ELSE IF needsInner
                     THEN \* C++-faithful: ALWAYS enter inner bundle phases
                          \* when child has priority+missing+children, regardless
                          \* of BundleCollectAtomic. The inner phases provide a
                          \* Children-CAS integrity check (InnerPhase3 CASes
                          \* each grandchild) which catches a peer's partial
                          \* unbundle of the grandchild between our collection
                          \* and inner CAS — without this, peer's
                          \* UnbundleCASChild updating a grandchild value goes
                          \* unnoticed, leading to the lost-increment race.
                          \* Mirrors C++ bundle() recursion when collecting a
                          \* priority+missing child (transaction_impl.h:1064-1125).
                          /\ local' = [local EXCEPT
                                 ![t].subpackets[c] = pkt,
                                 ![t].innerChild  = c,
                                 ![t].innerWrapper = cw,
                                 ![t].innerSubWs  = [gc \in ChildrenOf(c) |-> linkage[gc]]]
                          /\ pc' = [pc EXCEPT ![t] = "inner_phase2"]
                          /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
                     ELSE \* leaf child (no children) — direct collection
                          /\ local' = [local EXCEPT
                                 ![t].subwrappers[c] = cw,
                                 ![t].subpackets[c]  = pkt]
                          /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                          /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>

\* --- #2: Inner bundle phases (C++ bundle() called recursively from bundle_subpacket) ---
\* These model the inner bundle of a child that has hasPriority + missing + sub-children.
\* In the 3-level model, this is Grand bundling Parent, which triggers inner bundle of Parent
\* (collecting Child1, Child2).

\* InnerPhase2: CAS innerChild with collected sub-packets (missing=TRUE)
\* C++ bundle() Phase2 (line 1249-1258) for inner child.
InnerPhase2(t) ==
    /\ pc[t] = "inner_phase2"
    /\ LET c    == local[t].innerChild
           oldW == local[t].innerWrapper
           ser  == local[t].bundleSer
           pkt  == local[t].subpackets[c]     \* collected inner packet
           newPkt == MakePacket(c, oldW.packet.payload, pkt.sub, TRUE)
           newW == PriorityWrapper(newPkt, ser)
       IN
       /\ CanProceed(t, c)
       /\ IF linkage[c] = oldW
          THEN /\ linkage' = [linkage EXCEPT ![c] = newW]
               /\ local' = [local EXCEPT ![t].innerWrapper = newW]
               /\ pc' = [pc EXCEPT ![t] = "inner_phase3"]
               /\ UNCHANGED <<serial, op, target, iterBudget, childQueue, priorityTag>>
          ELSE \* Disturbed — restart outer bundle from snapshot
               /\ pc' = [pc EXCEPT ![t] = "snap_check"]
               /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
               /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>

\* InnerPhase3: CAS each grandchild to BundledRefWrapper (one per action)
\* C++ bundle() Phase3 (line 1260-1282) for inner child.
InnerPhase3(t) ==
    /\ pc[t] = "inner_phase3"
    /\ LET c    == local[t].innerChild
           ser  == local[t].bundleSer
           gcs  == ChildrenOf(c)
           gcWs == local[t].innerSubWs
       IN
       \* Success: CAS one grandchild
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
              \* Update gcWs[gc] to the new bundled-ref wrapper. Without this,
              \* the failure branch below sees `linkage[gc] /= gcWs[gc]` (the
              \* saved pre-CAS wrapper) and falsely fires for the gc we just
              \* successfully CAS'd, restarting inner_phase2 → infinite loop
              \* in single-thread (no real disturbance, just our own CAS being
              \* misread as a peer's interference). With Lamport-style
              \* GenSerial, every restart gets a fresh serial so distinct
              \* TLA+ states are produced unboundedly → state-space explosion.
              /\ local' = [local EXCEPT ![t].innerSubWs[gc] = BundledRefWrapper(c, ser)]
              /\ UNCHANGED <<serial, op, target, iterBudget, childQueue, priorityTag>>
       \* Failure: some grandchild changed — restart inner from Phase2.
       \* Symmetric eager tag: tag the failed grandchild AND the inner-
       \* bundling node c (mirrors outer BundlePhase3 SUPERFINE DISTURBED
       \* and matches the C11 generator's symmetric tagging — earlier
       \* TLA+ tagged only gc; C11 added c-tagging for consistency).
       \/ \E gc \in gcs :
              /\ CanProceed(t, gc)
              /\ gcWs[gc] /= Null
              /\ linkage[gc] /= gcWs[gc]
              \* Re-read grandchild wrappers for next attempt (no rollback, C++ behavior #3)
              /\ local' = [local EXCEPT
                     ![t].innerSubWs = [g \in gcs |-> linkage[g]]]
              /\ pc' = [pc EXCEPT ![t] = "inner_phase2"]
              /\ priorityTag' = [
                     [priorityTag EXCEPT ![gc] = TagAfterFail(t, gc)]
                         EXCEPT ![c] = TagAfterFail(t, c)]
              /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue>>

\* InnerPhase4: finalize — set missing=FALSE on innerChild
\* C++ bundle() Phase4 (line 1286-1299) for inner child.
InnerPhase4(t) ==
    /\ pc[t] = "inner_phase4"
    /\ LET c    == local[t].innerChild
           oldW == local[t].innerWrapper
           ser  == local[t].bundleSer
           finalPkt == MakePacket(c, oldW.packet.payload, oldW.packet.sub, FALSE)
           finalW == PriorityWrapper(finalPkt, ser)
       IN
       /\ CanProceed(t, c)
       /\ IF linkage[c] = oldW
          THEN \* Success — inner bundle complete. Save result and return to outer Phase1.
               /\ linkage' = [linkage EXCEPT ![c] = finalW]
               /\ local' = [local EXCEPT
                      ![t].subwrappers[c] = finalW,
                      ![t].innerChild  = Null,
                      ![t].innerWrapper = Null,
                      ![t].innerSubWs  = Null]
               /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
               /\ UNCHANGED <<serial, op, target, iterBudget, childQueue, priorityTag>>
          ELSE \* Disturbed — restart outer bundle from snapshot
               /\ pc' = [pc EXCEPT ![t] = "snap_check"]
               /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
               /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>

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
       /\ CanProceed(t, node)
       /\ IF linkage[node] = oldW
          THEN /\ linkage' = [linkage EXCEPT ![node] = newW]
               /\ local' = [local EXCEPT ![t].wrapper = newW]
               /\ pc' = [pc EXCEPT ![t] = "bundle_phase3"]
               /\ UNCHANGED <<serial, op, target, iterBudget, childQueue, priorityTag>>
          ELSE \* Disturbed
               /\ pc' = [pc EXCEPT ![t] = "snap_check"]
               /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterFail(t, node)]
               /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>

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
            /\ \A c \in children : CanProceed(t, c)
            /\ IF allMatch
               THEN /\ linkage' = [n \in AllNodes |->
                         IF n \in children
                         THEN BundledRefWrapper(node, ser)
                         ELSE linkage[n]]
                    /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                    /\ UNCHANGED <<serial, local, op, target, iterBudget, childQueue, priorityTag>>
               ELSE \* Some child changed — restart Phase 1
                    \* C++ FIDELITY: bundle() retry loop generates a NEW
                    \* bundle_serial each iteration so the retried Phase 3
                    \* allocates fresh PacketWrappers (distinct pointers in
                    \* C++; distinct values in TLA+). Without regen here,
                    \* the retried bundle reuses the same `ser`, and
                    \* BundledRefWrapper(node, ser) is structurally identical
                    \* to the wrapper a peer's earlier bundle wrote — so
                    \* the "refresh" CAS is a value-level no-op and a peer's
                    \* stale snapshotForUnbundle pointer compares equal,
                    \* allowing a final UnbundleCASChild that should have
                    \* failed to succeed (lost-increment race).
                    /\ LET newSer == GenSerial(t, ser)
                       IN
                       /\ local' = [local EXCEPT ![t].bundleSer = newSer]
                       /\ UpdateSerial(t, newSer)
                    /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                    /\ priorityTag' = [n \in AllNodes |->
                           IF n \in children /\ linkage[n] /= childWs[n]
                           THEN TagAfterFail(t, n)
                           ELSE priorityTag[n]]
                    /\ UNCHANGED <<linkage, op, target, iterBudget, childQueue>>
       ELSE \* fine: one child per action
            LET node     == local[t].bundleNode
                ser      == local[t].bundleSer
                children == ChildrenOf(node)
                childWs  == local[t].subwrappers
            IN
            \* Success path: pick one matching child, CAS it to bundled-ref.
            \/ \E c \in children :
                   /\ CanProceed(t, c)
                   /\ linkage[c] = childWs[c]
                   /\ linkage' = [linkage EXCEPT ![c] = BundledRefWrapper(node, ser)]
                   /\ LET allDone == \A c2 \in children \ {c} :
                                         linkage[c2] = BundledRefWrapper(node, ser)
                      IN
                      IF allDone
                      THEN pc' = [pc EXCEPT ![t] = "bundle_phase4"]
                      ELSE pc' = [pc EXCEPT ![t] = "bundle_phase3"]
                   \* Update subwrappers[c] to the new wrapper. Without this,
                   \* the failure branch below sees `linkage[c] /= childWs[c]`
                   \* (the saved pre-CAS wrapper) and falsely fires for the c
                   \* we just successfully CAS'd, restarting → infinite loop
                   \* in single-thread (Lamport-style serial regen on each
                   \* restart explodes the state space without convergence).
                   /\ local' = [local EXCEPT ![t].subwrappers[c] = BundledRefWrapper(node, ser)]
                   /\ UNCHANGED <<serial, op, target, iterBudget, childQueue, priorityTag>>
            \* Failure path: some child changed.
            \* superfine #4: check bundle_serial/parent → DISTURBED.
            \/ \E c \in children :
                   /\ CanProceed(t, c)
                   /\ childWs[c] /= Null
                   /\ linkage[c] /= childWs[c]
                   /\ LET disturbed ==
                           BundlePhase3Atomic = "superfine"
                           /\ (\/ \E c2 \in children :
                                      /\ linkage[c2] /= childWs[c2]
                                      /\ linkage[c2].serial /= ser
                                \/ linkage[node] /= local[t].wrapper)
                      IN
                      IF disturbed
                      THEN \* superfine DISTURBED — restart from snapshot.
                           \* Ports BundleUnbundle_2level_LLfree.tla line 511-531:
                           \* (a) Clear local bundle state so the next SnapCheck
                           \*     sees a fresh start (matches C++ where bundle()
                           \*     returning DISTURBED tears down its stack frame
                           \*     and the caller restarts with fresh locals).
                           \* (b) Eagerly tag BOTH the failed child AND the
                           \*     bundleNode — mirrors C++ outer-scope
                           \*     ScopedNegotiateLinkage at line 2179 of snapshot()
                           \*     retry loop on retry > 0 — DISTURBED return
                           \*     triggers outer retry, which eagerly tags the
                           \*     bundleNode so a peer cannot race in during the
                           \*     next snapshot attempt.
                           /\ pc' = [pc EXCEPT ![t] = "snap_check"]
                           /\ local' = [local EXCEPT
                                  ![t].wrapper     = Null,
                                  ![t].subwrappers = [cc \in children |-> Null],
                                  ![t].subpackets  = [cc \in children |-> Null]]
                           /\ priorityTag' = [
                                  [priorityTag EXCEPT ![c]    = TagAfterFail(t, c)]
                                      EXCEPT ![node] = TagAfterFail(t, node)]
                           /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue>>
                      ELSE \* No rollback — restart Phase1.
                           \* C++ FIDELITY: regenerate bundleSer (see comment
                           \* in coarse Phase 3 disturbed branch above).
                           /\ LET newSer == GenSerial(t, ser)
                              IN
                              /\ local' = [local EXCEPT
                                     ![t].bundleSer = newSer,
                                     ![t].subwrappers = [cc \in children |-> Null],
                                     ![t].subpackets  = [cc \in children |-> Null]]
                              /\ UpdateSerial(t, newSer)
                           /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
                           /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
                           /\ UNCHANGED <<linkage, op, target, iterBudget, childQueue>>

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
       /\ CanProceed(t, node)
       /\ IF linkage[node] = oldW
          THEN /\ linkage' = [linkage EXCEPT ![node] = finalW]
               /\ local' = [local EXCEPT ![t].snapResult = finalPkt]
               /\ pc' = [pc EXCEPT ![t] = "commit_grand"]
               /\ UNCHANGED <<serial, op, target, iterBudget, childQueue, priorityTag>>
          ELSE \* Disturbed
               /\ pc' = [pc EXCEPT ![t] = "snap_check"]
               /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterFail(t, node)]
               /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>

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
                   parentPkt.sub[c].payload + 1,
                   parentPkt.sub[c].sub, parentPkt.sub[c].missing)]
           newParentPkt == MakePacket(Parent, parentPkt.payload,
                               newParentSub, parentPkt.missing)
           newGrandSub  == [snapPkt.sub EXCEPT ![Parent] = newParentPkt]
           newPkt       == MakePacket(Grand, snapPkt.payload,
                               newGrandSub, snapPkt.missing)
           ser          == GenSerial(t, w.serial)
           newW         == PriorityWrapper(newPkt, ser)
       IN
       \/ \* CAS success: commit and move to per-child phase.
          \* Transaction-end: release all my tags via ClearMyTags.
          /\ CanProceed(t, Grand)
          /\ w.hasPriority
          /\ w.packet = snapPkt
          /\ linkage' = [linkage EXCEPT ![Grand] = newW]
          /\ UpdateSerial(t, ser)
          /\ childQueue' = [childQueue EXCEPT ![t] = ParentChildren]
          /\ pc' = [pc EXCEPT ![t] = "idle"]
          /\ op' = [op EXCEPT ![t] = "idle"]
          /\ target' = [target EXCEPT ![t] = Null]
          /\ local' = [local EXCEPT ![t] = InitLocal]
          /\ priorityTag' = ClearMyTags(t)
          /\ UNCHANGED iterBudget
       \/ \* CAS failure: retry. ClearMyTags then refresh Grand tag for retry.
          /\ CanProceed(t, Grand)
          /\ ~(w.hasPriority /\ w.packet = snapPkt)
          /\ pc' = [pc EXCEPT ![t] = "idle"]
          /\ op' = [op EXCEPT ![t] = "idle"]
          /\ target' = [target EXCEPT ![t] = Null]
          /\ local' = [local EXCEPT ![t] = InitLocal]
          /\ priorityTag' = [ClearMyTags(t) EXCEPT ![Grand] = TagAfterFail(t, Grand)]
          /\ UNCHANGED <<serial, linkage, iterBudget, childQueue>>

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
    /\ UNCHANGED <<serial, linkage, local, iterBudget, childQueue, priorityTag>>

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
                                     w.packet.payload + 1,
                                     w.packet.sub,
                                     w.packet.missing)]
            /\ pc' = [pc EXCEPT ![t] = "commit_try_cas"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
       ELSE \* Bundled — need unbundle
            /\ local' = [local EXCEPT ![t].wrapper = w]
            /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
            /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>

\* @c11_action CommitTryCAS(t):
\*   // Direct commit (hasPriority path)
\*   local_shared_ptr<PacketWrapper> newwrapper(
\*       new PacketWrapper(tr.m_packet, tr.m_serial));
\*   if (m_link->compareAndSet(wrapper, newwrapper))
\*       return true;           // success
\*   // payload unchanged -> single-node optimization: adopt new children
\*   // payload changed   -> true conflict, fail
\*   Source: transaction_impl.h:1368-1400
\*
\*   Fidelity note (#7): C++ creates newwrapper with tr.m_serial once and reuses
\*   it across inner retries (adopt-children path). TLA+ calls GenSerial each time
\*   CommitTryCAS fires, which may produce a higher serial after adopting a wrapper
\*   with a newer serial. Effect: TLA+ consumes slightly more serial space per
\*   commit. No correctness impact.
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
            /\ UNCHANGED <<op, target, iterBudget, childQueue, priorityTag>>
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
                 /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue>>
            ELSE \* True conflict
                 /\ local' = [local EXCEPT ![t].commitOk = "fail"]
                 /\ pc' = [pc EXCEPT ![t] = "commit_done"]
                 /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterFail(t, node)]
                 /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterFail(t, node)]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>

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
                 /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue, priorityTag>>
            ELSE LET result == SnapshotForUnbundle(node, local[t].bundleSer)
                 IN
                 IF result.status \in {"DISTURBED", "COLLIDED"}
                 THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                      /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue, priorityTag>>
                 ELSE LET subPkt ==
                            IF result.status \in {"VOID_PACKET", "NODE_MISSING"}
                            THEN w.packet
                            ELSE result.subpacket
                          casTargets == result.casTargets
                      IN
                      IF subPkt = Null
                      THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                           /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue, priorityTag>>
                      ELSE /\ local' = [local EXCEPT
                                ![t].oldpacket = subPkt,
                                ![t].newpacket = MakePacket(node,
                                    subPkt.payload + 1,
                                    subPkt.sub, subPkt.missing),
                                ![t].casTargets = casTargets,
                                ![t].casIdx = 1,
                                \* SAVE root_wrapper for UnbundleCASLoop
                                \* per-ancestor packet extraction (mirrors
                                \* C++ snapshotForUnbundle root_wrapper
                                \* persistence; transaction_impl.h:2118-2127).
                                ![t].walkWrapper = result.wrapper]
                           /\ pc' = [pc EXCEPT ![t] =
                                IF Len(casTargets) >= 1
                                THEN "unbundle_cas_loop"
                                ELSE "unbundle_cas_child"]
                           /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
       ELSE \* fine: walk one level per action.
            \* local[t].walkNode: current node in the chain (starts at target[t])
            \* local[t].walkWrapper: wrapper saved for this node
            \* local[t].casTargets: ancestors collected so far
            \* On each step: read linkage[walkNode], check bundledBy,
            \*   move walkNode to parent, build casTargets.
            \*   fine: Append (leaf-first); superfine: prepend (root-first, matching C++).
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
                 /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
            ELSE LET ww == local[t].walkWrapper
                     pNode == ww.bundledBy
                 IN
                 IF pNode = Null
                 THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                      /\ local' = [local EXCEPT ![t].walkNode = Null]
                      /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
                 ELSE \* Staleness check: has wn's linkage changed since we loaded ww?
                      IF linkage[wn] /= ww
                      THEN /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                           /\ local' = [local EXCEPT ![t].walkNode = Null]
                           /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
                      ELSE LET pw == linkage[pNode]
                               \* C++ FIDELITY: walkUpChainImpl is recursive,
                               \* the deepest call (at root) emplace_back's
                               \* into cas_infos FIRST, then each shallower
                               \* level appends. So cas_infos is root-first.
                               \* With leaf-first, a peer CommitGrand can
                               \* interject between t1's Parent CAS and t1's
                               \* Grand CAS (in fine mode where each CAS is
                               \* one action), and never see Grand changed →
                               \* its stale snapshot CAS succeeds, producing
                               \* a lost increment that does NOT occur in
                               \* C++. We unconditionally use root-first for
                               \* casTargets construction even in fine mode;
                               \* the other fine vs superfine differences
                               \* (BundlePhase1 pre-bundle CAS, BundlePhase3
                               \* DISTURBED detection) are preserved.
                               newTargets == <<pNode>> \o local[t].casTargets
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
                                     /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
                                ELSE /\ local' = [local EXCEPT
                                            ![t].oldpacket = subPkt,
                                            ![t].newpacket = MakePacket(node,
                                                subPkt.payload + 1,
                                                subPkt.sub, subPkt.missing),
                                            ![t].casTargets = newTargets,
                                            ![t].casIdx = 1,
                                            ![t].walkNode = Null,
                                            \* SAVE the priority super-wrapper
                                            \* (root reached at this step) so
                                            \* UnbundleCASLoop can extract
                                            \* per-ancestor packets via the
                                            \* recursive .sub[] chain — mirrors
                                            \* C++ snapshotForUnbundle which
                                            \* keeps root_wrapper across
                                            \* cas_info construction
                                            \* (transaction_impl.h:2118-2127).
                                            ![t].walkWrapper = pw]
                                     /\ pc' = [pc EXCEPT ![t] =
                                            IF Len(newTargets) >= 1
                                            THEN "unbundle_cas_loop"
                                            ELSE "unbundle_cas_child"]
                                     /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>
                           ELSE \* Continue walking up.
                                /\ local' = [local EXCEPT
                                       ![t].walkNode = pNode,
                                       ![t].walkWrapper = pw,
                                       ![t].casTargets = newTargets]
                                /\ pc' = [pc EXCEPT ![t] = "unbundle_walk"]
                                /\ UNCHANGED <<serial, linkage, op, target, iterBudget, childQueue, priorityTag>>

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
\* UnbundleCASLoop — matches C++ snapshotForUnbundle CAS infrastructure:
\*   transaction_impl.h:2118-2127 builds new_wrapper for each ancestor
\*   from extracted packet content (root_wrapper.packet for root, drilled
\*   down via .sub[] for descendants), NOT from current linkage[n].
\*   transaction_impl.h:2722-2738 CASes each cas_info entry, including
\*   non-priority ancestors. C++ has no hasPriority gate.
\*
\* Implementation: extract per-ancestor packet from local[t].walkWrapper
\* (= super-wrapper saved at walk-end) by .sub[] navigation. Non-priority
\* Parent in 3-level chain gets a valid extracted packet, allowing CAS.
\*
\* When walkWrapper is unavailable (legacy code paths), fall back to old
\* logic (linkage[n].packet + hasPriority gate) — degrades gracefully
\* but may cycle on bundled-ref ancestors (matches old BundleUnbundle.tla
\* SAFETY-only semantics).
UnbundleCASLoop(t) ==
    /\ pc[t] = "unbundle_cas_loop"
    /\ LET superW    == local[t].walkWrapper
           superNode == IF superW = Null \/ superW.packet = Null
                       THEN Null
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
       THEN \* All CAS done atomically in one action.
            LET targets       == local[t].casTargets
                allCanProceed == \A i \in 1..Len(targets) :
                                     CanProceed(t, targets[i])
                allExtractable == \A i \in 1..Len(targets) :
                                      ExtractAt(targets[i]) /= Null
                superFresh    == /\ superNode /= Null
                                 /\ linkage[superNode] = superW
                \* Use the saved walk wrapper's serial as Lamport reference
                \* (mirrors C++ snapshotForUnbundle which advances past the
                \* root wrapper's m_bundle_serial; transaction_impl.h:2752).
                ser == GenSerial(t, IF superW = Null THEN 0 ELSE superW.serial)
            IN
            /\ allCanProceed
            /\ IF allExtractable /\ superFresh
               THEN /\ linkage' = [n \in AllNodes |->
                          IF \E i \in 1..Len(targets) : targets[i] = n
                          THEN LET ext == ExtractAt(n)
                                   newPkt == MakePacket(n, ext.payload,
                                                        ext.sub, TRUE)
                               IN  PriorityWrapper(newPkt, ser)
                          ELSE linkage[n]]
                    /\ UpdateSerial(t, ser)
                    /\ pc' = [pc EXCEPT ![t] = "unbundle_cas_child"]
                    /\ UNCHANGED <<local, op, target, iterBudget, childQueue, priorityTag>>
               ELSE /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                    /\ priorityTag' = [n \in AllNodes |->
                          IF \E i \in 1..Len(targets) : targets[i] = n
                          THEN TagAfterFail(t, n)
                          ELSE priorityTag[n]]
                    /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>
       ELSE \* fine: one CAS per action.
            LET idx       == local[t].casIdx
                casNode   == local[t].casTargets[idx]
                oldW      == linkage[casNode]
                extracted == ExtractAt(casNode)
                ser       == GenSerial(t, oldW.serial)
                newPkt    == IF extracted = Null THEN Null
                             ELSE MakePacket(casNode, extracted.payload,
                                             extracted.sub, TRUE)
                newW      == IF newPkt = Null THEN Null
                             ELSE PriorityWrapper(newPkt, ser)
                superFresh == /\ superNode /= Null
                              /\ linkage[superNode] = superW
                nextIdx   == idx + 1
                done      == nextIdx > Len(local[t].casTargets)
            IN
            /\ CanProceed(t, casNode)
            /\ IF newW /= Null /\ superFresh /\ linkage[casNode] = oldW
               THEN /\ linkage' = [linkage EXCEPT ![casNode] = newW]
                    /\ UpdateSerial(t, ser)
                    \* If the casNode we just CAS'd IS the super (root we
                    \* walked to), update walkWrapper to the new wrapper so
                    \* the next iteration's `superFresh` check doesn't trip
                    \* on our own CAS. Without this, root-first ordering CAS
                    \* of Grand makes the second iter (Parent CAS) see
                    \* `linkage[Grand] /= walkWrapper` → DISTURBED →
                    \* commit_read → re-walk → CAS Grand again → infinite
                    \* loop in single-thread (Lamport regen makes each loop
                    \* iter a distinct TLA+ state, exploding state space).
                    /\ local' = [local EXCEPT
                           ![t].casIdx = nextIdx,
                           ![t].walkWrapper =
                               IF casNode = superNode
                               THEN newW
                               ELSE local[t].walkWrapper]
                    /\ pc' = [pc EXCEPT ![t] =
                         IF done
                         THEN "unbundle_cas_child"
                         ELSE "unbundle_cas_loop"]
                    /\ UNCHANGED <<op, target, iterBudget, childQueue, priorityTag>>
               ELSE \* Disturbed (super stale or extraction failed)
                    /\ pc' = [pc EXCEPT ![t] = "commit_read"]
                    /\ priorityTag' = [priorityTag EXCEPT ![casNode] = TagAfterFail(t, casNode)]
                    /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>

\* @c11_action UnbundleCASChild(t):
\*   // Restore child to priority with extracted sub-packet
\*   newsubwrapper = new PacketWrapper(*subpacket, gen(superwrapper->m_bundle_serial));
\*   sublinkage->compareAndSet(bundled_ref, newsubwrapper);
\*   Source: transaction_impl.h:1504-1514
\*
\*   Fidelity note (#8): C++ uses gen(superwrapper->m_bundle_serial) where superwrapper
\*   is the root wrapper after snapshotForUnbundle walk (may have a higher serial if
\*   another thread committed the root since bundling). TLA+ uses oldChildW.serial
\*   (the child's bundled_ref serial). Both equal bundle_serial at bundle time; they
\*   diverge only when the root was re-committed. Effect: C++ produces a higher serial
\*   (stronger Lamport guarantee). No correctness impact.
\* Final: CAS child's linkage to restore priority with new (committed) packet.
\* Atomically sync the immediate Parent's packet.sub[node] to the new
\* child packet IF Parent is currently priority+missing=TRUE (left in
\* that state by our earlier UnbundleCASLoop). Mirrors C++
\* snapshotForUnbundle which links each cas_info's new_wrapper.packet
\* into the chain via `p = &newwrapper->packet()` (transaction_impl.h
\* :2130), so when sublinkage is CASed to newsubwrapper, the
\* ancestor's newwrapper.packet.sub[child] already points to the same
\* new packet — consistent across the chain.
\*
\* Without this sync, peer's BundlePhase1 collecting from Parent reads
\* stale sub[child]=old_payload, leading to a lost-increment race
\* (peer's CommitGrand effectively no-ops at child level).
UnbundleCASChild(t) ==
    /\ pc[t] = "unbundle_cas_child"
    /\ CanProceed(t, target[t])
    /\ LET node       == target[t]
           oldChildW  == local[t].wrapper
           parentNode == ParentOf(node)
           parentW    == IF parentNode = Null THEN Null ELSE linkage[parentNode]
           ser        == GenSerial(t, oldChildW.serial)
           newChildW  == PriorityWrapper(local[t].newpacket, ser)
           \* Sync Parent.packet.sub[node] only if Parent is currently
           \* priority and structurally compatible with the chain we just
           \* built in UnbundleCASLoop.
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
            /\ UNCHANGED <<op, target, iterBudget, childQueue, priorityTag>>
       ELSE \* Child changed — retry
            /\ pc' = [pc EXCEPT ![t] = "commit_read"]
            /\ priorityTag' = [priorityTag EXCEPT ![node] = TagAfterFail(t, node)]
            /\ UNCHANGED <<serial, linkage, local, op, target, iterBudget, childQueue>>

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
    \* Transaction-end: release all my tags.
    /\ priorityTag' = ClearMyTags(t)
    /\ UNCHANGED <<serial, linkage>>

-----------------------------------------------------------------------------
(* Next-state relation *)

\* AllDone: terminal condition.
AllDone == \A t \in Threads : pc[t] = "idle" /\ iterBudget[t] = 0

\* NextStep: real-progress actions. PreemptTag included as forward progress.
NextStep ==
    \E t \in Threads :
        \/ \E n \in InnerNodes : SnapRead(t, n)
        \/ SnapCheck(t)
        \/ BundlePhase1(t)
        \/ InnerPhase2(t)
        \/ InnerPhase3(t)
        \/ InnerPhase4(t)
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
        \/ \E n \in AllNodes : PreemptTag(t, n)

\* Terminating: self-loop at AllDone so default CHECK_DEADLOCK TRUE is OK.
\* Excluded from WF — otherwise eternal stuttering would satisfy WF
\* without progress, making liveness vacuous.
Terminating ==
    /\ AllDone
    /\ UNCHANGED vars

Next == NextStep \/ Terminating

Spec == Init /\ [][Next]_vars /\ WF_vars(NextStep)

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

\* NoSerialWrapAround: retired. With plain natural-number serial arithmetic and
\* CONSTRAINT SerialBound, serials are always totally ordered by > and never
\* ambiguous. Trivially TRUE; use SerialBound as CONSTRAINT instead.
NoSerialWrapAround == TRUE

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
\* MaxPayload MOD removed (payload Nat cumulative); strict cumulative assertion.
\*
\* Production check. For early detection of lost-increment / value-loss bugs,
\* enable QuiescentCheck below in the cfg too — it fires at every all-idle
\* moment (intermediate iteration boundaries) instead of only at AllDone, so
\* a violation is caught in O(thousands) of states rather than O(millions).
TerminalPayloadCheck ==
    (\A t \in Threads : iterBudget[t] = 0 /\ pc[t] = "idle") =>
        \A c \in ParentChildren :
            /\ linkage[c].hasPriority
            /\ linkage[c].packet.payload =
                   2 * MaxCommits * Cardinality(Threads)

\* QuiescentCheck (DEBUG-ONLY): expected child payload at any all-idle
\* moment, derived from existing state without an auxiliary commit_count
\* variable. Fires earlier than TerminalPayloadCheck (at every iteration
\* boundary, not just AllDone), making lost-increment races visible
\* in just a few thousand states instead of millions. NOT enabled in
\* production cfgs — opt in by adding `INVARIANT QuiescentCheck` to a cfg
\* when chasing a regression.
\*
\* Per-thread per-child contribution at all-idle:
\*   completed(t) = MaxCommits - iterBudget[t]    (full iterations done)
\*   midIter(t)   = childQueue[t] /= {}           (CommitGrand done in current
\*                                                 iter, some per-child commits
\*                                                 may still pend)
\*   For child c, t's contribution =
\*     2 * completed(t)                  full iters: +1 grand, +1 direct each
\*     + (1 if midIter(t) else 0)        partial: grand commit in current iter
\*     + (1 if midIter(t) /\ c \notin childQueue[t] else 0)
\*                                       partial: direct commit done for c in
\*                                       current iter
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
        \A c \in ParentChildren :
            \* Implication (not conjunction): mid-iteration children may
            \* still be bundled (no priority) at all-idle moments — those
            \* states have no observable child.payload to compare against.
            \* Only verify payload when child is priority (= unbundled).
            linkage[c].hasPriority =>
                linkage[c].packet.payload = SumPayloadOver(Threads, c)

\* DebugSerialBound: NEUTERED — Lamport-style GenSerial (TID-encoded
\* counter, mirrors C++ SerialGenerator) advances unboundedly per wrapper
\* allocation, so a hard threshold no longer correlates with livelock.
\* LL-free priority gating now guarantees termination. Kept as TRUE so
\* existing cfg files referencing this name don't break.
DebugSerialBound == TRUE

\* PrintTerminalSerial: debug invariant. Always TRUE, but emits the
\* per-thread serial array to TLC stdout the first time any AllDone state
\* is visited. Useful for gauging how high Lamport counters grow under
\* each config without instrumenting an external trace.
PrintTerminalSerial ==
    \/ ~(\A t \in Threads : iterBudget[t] = 0 /\ pc[t] = "idle")
    \/ PrintT(<<"Terminal serial[t]:", serial>>)

\* EventuallyAllDone: under WF, every execution reaches AllDone.
\* The key LL-free liveness property — priority gating must guarantee
\* no thread is starved indefinitely.
EventuallyAllDone == <>AllDone

=============================================================================
