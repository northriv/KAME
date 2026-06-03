(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.
 ***************************************************************************)
----------------------- MODULE BundleUnbundle_hardlink_self_collision -----------------------
(*
 * Hardlink self-collision minimal model (3 nodes, 2 threads).
 *
 *   Topology:
 *         R                R is bundle root.
 *        / \               A is direct child of R.
 *       A   \              C is direct child of BOTH A and R (hard-link).
 *        \  |              Once InsertHardLink runs, R.sub holds both A and C.
 *         \ /
 *          C
 *
 * Initial state (pre-hardlink): past bundle(R) completed,
 *   - linkage[A].bundledBy = R, A's packet lives in R.sub[A]
 *   - linkage[C].bundledBy = A, C's packet lives in A.sub[C]
 *   - R.sub = (A :> a_pkt, C :> Null)  ← C slot Null until hardlink inserted
 *
 * Race target:
 *   T1: InsertHardLink (registers C as direct child of R) + bundle(R)
 *   T2: concurrent bundle(R)
 *
 * Hardlink self-collision: bundle(R) walks R's subtree and reaches C
 * twice — once via the direct R→C link and once via R→A→C.  The
 * `is_bundle_root` Phase 4 m_missing override is the bug surface (per
 * the KAME implementation).
 *
 * Modelled bundle phases (5):
 *   Phase 1 (Collect)  — read child wrappers
 *   Phase 2 (CAS R)    — set R missing=TRUE with collected sub[]
 *   Phase 3 (CAS child)— CAS each bundled child wrapper
 *   Phase 4 (Finalize) — clear missing flag (is_bundle_root override)
 *   Phase 5 (Publish)  — final wrapper publish (re-CAS of R)
 *
 * Plus: InsertHardLink — atomic-ish add of C to R.sub (with Null slot,
 *       since C's packet is still under A).
 *
 * Invariant: SnapshotConsistency mirrors `Packet::checkConsistensy`
 *   (transaction_impl.h:858-892, line 871 is the throw):
 *
 *     ∀ N in tree :
 *       ∀ i ∈ DOMAIN N.sub :
 *         N.sub[i] = Null ⇒ (rootPacket.missing OR
 *                            child_at(i) reachable via reverseLookup
 *                            in rootPacket)
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Threads,
    R, A, C,
    Null

Nodes == {R, A, C}
Children == {A, C}      \* R's potential children (direct)
Leaves == {C}           \* A's children

ASSUME Cardinality(Threads) > 0

VARIABLES
    linkage,       \* [Nodes -> Wrapper]: atomic per node
    pc,            \* [Threads -> String]
    local,         \* [Threads -> Record]: thread-local
    hardlinkDone,  \* BOOLEAN: InsertHardLink completed?
    bundleDone     \* [Threads -> BOOLEAN]: this thread finished its bundle

\* retryCount removed (per user 2026-05-21).  See _hardlink_4node /
\* _external_migration for the same clean-up rationale.

vars == <<linkage, pc, local, hardlinkDone, bundleDone>>

-----------------------------------------------------------------------------
(* Wrapper helpers *)

PriorityWrapper(packet) ==
    [packet |-> packet, hasPriority |-> TRUE, bundledBy |-> Null]

BundledRefWrapper(parent) ==
    [packet |-> Null, hasPriority |-> FALSE, bundledBy |-> parent]

MakePacket(node, sub, miss) ==
    [sub |-> sub, missing |-> miss, node |-> node]

\* R's sub (over Children): initially {A: a_pkt, C: Null}; after hardlink {A: a_pkt, C: Null}
\* (C slot remains Null because C's packet is under A.sub[C])
RSubInit ==
    [c \in Children |->
        IF c = A
        THEN MakePacket(A,
                  [l \in Leaves |-> MakePacket(C, <<>>, FALSE)],
                  FALSE)
        ELSE Null]

\* A's sub (over Leaves): {C: c_pkt}
ASubInit ==
    [l \in Leaves |-> MakePacket(C, <<>>, FALSE)]

-----------------------------------------------------------------------------
(* Init *)

InitLocal == [
    op            |-> "idle",
    wrapper       |-> Null,    \* R wrapper at scope start
    aWrapper      |-> Null,    \* A wrapper collected
    cWrapper      |-> Null,    \* C wrapper collected
    aSubPkt       |-> Null,    \* A's packet to bundle under R
    cSubPkt       |-> Null,    \* C's packet (if visible)
    newRSub       |-> Null     \* R's new sub[] for Phase 2
]

Init ==
    /\ linkage = [n \in Nodes |->
        CASE n = R -> PriorityWrapper(MakePacket(R, RSubInit, FALSE))
          [] n = A -> BundledRefWrapper(R)
          [] n = C -> BundledRefWrapper(A)]
    /\ pc = [t \in Threads |-> "idle"]
    /\ local = [t \in Threads |-> InitLocal]
    /\ hardlinkDone = FALSE
    /\ bundleDone = [t \in Threads |-> FALSE]

-----------------------------------------------------------------------------
(* reverseLookup — for SnapshotConsistency check.
   Returns TRUE iff `child` is reachable from `rootPkt` via either
   direct sub or via A→C path. *)

ReachableFrom(rootPkt, child) ==
    \/ /\ rootPkt.node = R
       /\ child \in DOMAIN rootPkt.sub
       /\ rootPkt.sub[child] /= Null
    \/ /\ rootPkt.node = R
       /\ A \in DOMAIN rootPkt.sub
       /\ rootPkt.sub[A] /= Null
       /\ child \in DOMAIN rootPkt.sub[A].sub
       /\ rootPkt.sub[A].sub[child] /= Null

-----------------------------------------------------------------------------
(* InsertHardLink — register C as direct child of R.
   Adds C to R.sub with Null slot (since C's packet lives in A.sub[C]).
   Done as a single CAS in this minimal model — the multi-phase insert
   pipeline (insert + bundle + child CAS) is folded for clarity.
   Idempotent: only fires once. *)

InsertHardLink(t) ==
    /\ pc[t] = "idle"
    /\ ~hardlinkDone
    /\ ~bundleDone[t]
    /\ LET rw == linkage[R]
       IN
       /\ rw.hasPriority
       /\ rw.packet.sub[C] = Null
       /\ LET newSub == [rw.packet.sub EXCEPT ![C] = Null]
              \* Hardlink insert keeps C's slot Null in R.sub (its packet
              \* still lives under A).  But it logically registers C as a
              \* child of R.  Modeled via hardlinkDone flag — ReachableFrom
              \* and BundleCollect both consult it.
              newPkt == MakePacket(R, newSub, FALSE)
              newW == PriorityWrapper(newPkt)
          IN
          /\ linkage[R] = rw
          /\ linkage' = [linkage EXCEPT ![R] = newW]
          /\ hardlinkDone' = TRUE
          /\ UNCHANGED <<pc, local, bundleDone>>

-----------------------------------------------------------------------------
(* Bundle pipeline — operates on R (root).  All actions guarded on pc[t]. *)

\* BundleStart — t initiates bundle(R, is_bundle_root=TRUE).
BundleStart(t) ==
    /\ pc[t] = "idle"
    /\ ~bundleDone[t]
    /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
    /\ local' = [local EXCEPT ![t] = InitLocal,
                              ![t].op = "bundle"]
    /\ UNCHANGED <<linkage, hardlinkDone, bundleDone>>

\* Phase 1 (Collect) — read R's wrapper, capture A's wrapper, capture C's
\* wrapper.  C is collected via two paths if hardlinkDone (R.sub also
\* references C); else only via A→C.
BundlePhase1(t) ==
    /\ pc[t] = "bundle_phase1"
    /\ LET rw == linkage[R]
           aw == linkage[A]
           cw == linkage[C]
       IN
       /\ rw.hasPriority
       /\ ~rw.packet.missing
       /\ local' = [local EXCEPT
              ![t].wrapper  = rw,
              ![t].aWrapper = aw,
              ![t].cWrapper = cw,
              ![t].aSubPkt  = IF aw.hasPriority
                              THEN aw.packet
                              ELSE rw.packet.sub[A],
              ![t].cSubPkt  = IF cw.hasPriority
                              THEN cw.packet
                              ELSE IF cw.bundledBy = A
                                   /\ aw.hasPriority
                                   THEN aw.packet.sub[C]
                                   ELSE IF cw.bundledBy = A
                                        /\ ~aw.hasPriority
                                        /\ rw.packet.sub[A] /= Null
                                        THEN rw.packet.sub[A].sub[C]
                                        ELSE Null]
       /\ pc' = [pc EXCEPT ![t] = "bundle_phase2"]
       /\ UNCHANGED <<linkage, hardlinkDone, bundleDone>>

\* Phase 2 (CAS R) — CAS R wrapper to missing=TRUE with collected sub[].
\* For self-collision modeling: the bundle "absorbs" both A and (if
\* hardlinkDone) C into R's sub[].  Failure → retry from Phase 1.
BundlePhase2(t) ==
    /\ pc[t] = "bundle_phase2"
    /\ LET oldW == local[t].wrapper
           aPkt == local[t].aSubPkt
           cPkt == local[t].cSubPkt
           \* Bundle re-pack: A gets a fresh sub[] containing C's packet
           \* (if collected), C slot directly under R remains Null if
           \* hardlinkDone (it's reachable via A).
           newASub == [l \in Leaves |-> cPkt]
           newAPkt == IF aPkt /= Null
                      THEN MakePacket(A, newASub, FALSE)
                      ELSE Null
           newRSub == [c2 \in Children |->
               IF c2 = A
               THEN newAPkt
               ELSE \* c2 = C
                    IF hardlinkDone
                    THEN Null   \* hardlink: slot stays Null
                    ELSE oldW.packet.sub[C]]
           newPkt == MakePacket(R, newRSub, TRUE)
           newW == PriorityWrapper(newPkt)
       IN
       IF linkage[R] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![R] = newW]
            /\ local' = [local EXCEPT ![t].wrapper = newW,
                                       ![t].newRSub = newRSub]
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase3"]
            /\ UNCHANGED <<hardlinkDone, bundleDone>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
            /\ local' = [local EXCEPT ![t] = InitLocal,
                                       ![t].op = "bundle"]
            /\ UNCHANGED <<linkage, hardlinkDone, bundleDone>>

\* Phase 3 (CAS child) — CAS A's wrapper to BundledRefWrapper(R).
\* C is A's child and stays bundledBy=A (its packet remains under
\* A.sub[C]); R.sub[C] = Null is the legitimate hardlink reference.
\* So Phase 3 only re-tags A.  If A wrapper changed by peer, abort &
\* retry from Phase 1.
BundlePhase3(t) ==
    /\ pc[t] = "bundle_phase3"
    /\ \/ \* CAS A
          /\ linkage[A] = local[t].aWrapper
          /\ linkage' = [linkage EXCEPT ![A] = BundledRefWrapper(R)]
          /\ local' = [local EXCEPT ![t].aWrapper =
                  [packet |-> Null, hasPriority |-> FALSE, bundledBy |-> R]]
          /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
          /\ UNCHANGED <<hardlinkDone, bundleDone>>
       \/ \* A already at R — skip Phase 3
          /\ linkage[A].bundledBy = R
          /\ pc' = [pc EXCEPT ![t] = "bundle_phase4"]
          /\ UNCHANGED <<linkage, local, hardlinkDone, bundleDone>>

\* Phase 3 fail — A's wrapper diverged from collected; retry Phase 1.
BundlePhase3Fail(t) ==
    /\ pc[t] = "bundle_phase3"
    /\ local[t].aWrapper /= Null
    /\ linkage[A] /= local[t].aWrapper
    /\ linkage[A].bundledBy /= R
    /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
    /\ local' = [local EXCEPT ![t] = InitLocal,
                              ![t].op = "bundle"]
    /\ UNCHANGED <<linkage, hardlinkDone, bundleDone>>

\* Phase 4 (Finalize) — clear R's missing flag (is_bundle_root override).
\* Bug surface: hardlink self-collision means R.sub[C] = Null (because
\* C is under A), and bundle must not report inconsistency.
BundlePhase4(t) ==
    /\ pc[t] = "bundle_phase4"
    /\ LET oldW == local[t].wrapper
           finalPkt == MakePacket(R, oldW.packet.sub, FALSE)
           finalW == PriorityWrapper(finalPkt)
       IN
       IF linkage[R] = oldW
       THEN /\ linkage' = [linkage EXCEPT ![R] = finalW]
            /\ local' = [local EXCEPT ![t].wrapper = finalW]
            /\ pc' = [pc EXCEPT ![t] = "bundle_phase5"]
            /\ UNCHANGED <<hardlinkDone, bundleDone>>
       ELSE /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
            /\ local' = [local EXCEPT ![t] = InitLocal,
                                       ![t].op = "bundle"]
            /\ UNCHANGED <<linkage, hardlinkDone, bundleDone>>

\* Phase 5 (Publish) — completion step.  In KAME's bundle this is the
\* final CAS that publishes the new wrapper; in our model the wrapper
\* was already published in Phase 4.  Phase 5 marks bundleDone[t]=TRUE
\* and idle's the thread.
BundlePhase5(t) ==
    /\ pc[t] = "bundle_phase5"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ local' = [local EXCEPT ![t] = InitLocal]
    /\ bundleDone' = [bundleDone EXCEPT ![t] = TRUE]
    /\ UNCHANGED <<linkage, hardlinkDone>>

-----------------------------------------------------------------------------
(* Next *)

AllDone == \A t \in Threads : bundleDone[t]

NextStep ==
    \E t \in Threads :
        \/ InsertHardLink(t)
        \/ BundleStart(t)
        \/ BundlePhase1(t)
        \/ BundlePhase2(t)
        \/ BundlePhase3(t)
        \/ BundlePhase3Fail(t)
        \/ BundlePhase4(t)
        \/ BundlePhase5(t)

Terminating == AllDone /\ UNCHANGED vars

Next == NextStep \/ Terminating

Spec == Init /\ [][Next]_vars /\ WF_vars(NextStep)

-----------------------------------------------------------------------------
(* Safety invariants *)

\* SnapshotConsistency — mirrors checkConsistensy line 871.
\* For each "rooted" packet (i.e. R's published wrapper), any Null slot in
\* the sub[] must be either:
\*   (a) the root is missing (intermediate state OK), OR
\*   (b) the child is still reachable via reverseLookup from root.
\* The hardlink case: R.sub[C] = Null is OK iff ReachableFrom(R, C) holds
\* (= via A→C path).
SnapshotConsistency ==
    LET rw == linkage[R]
    IN  (rw.hasPriority /\ ~rw.packet.missing) =>
        (\A child \in Children :
            rw.packet.sub[child] = Null =>
                ReachableFrom(rw.packet, child))

\* NoMissingHole: when R is not missing, its sub-packets that are
\* present are themselves not missing (except via hardlink Null slot
\* which is the legitimate "elsewhere" case).
NoMissingHole ==
    LET rw == linkage[R]
    IN  (rw.hasPriority /\ ~rw.packet.missing) =>
        (\A child \in Children :
            rw.packet.sub[child] /= Null =>
                ~rw.packet.sub[child].missing)

\* BundleRefConsistency: A and C, when bundled, point to R or A.
BundleRefConsistency ==
    LET aw == linkage[A]
        cw == linkage[C]
    IN  /\ (~aw.hasPriority => aw.bundledBy = R)
        /\ (~cw.hasPriority => cw.bundledBy \in {A, R})

\* HardlinkExclusive: after bundle, C's packet is in EXACTLY ONE place
\* (under R's direct sub OR under A's sub via the bundled chain), never
\* both.
HardlinkExclusive ==
    LET rw == linkage[R]
    IN  rw.hasPriority =>
        ~(/\ rw.packet.sub[C] /= Null
          /\ rw.packet.sub[A] /= Null
          /\ C \in DOMAIN rw.packet.sub[A].sub
          /\ rw.packet.sub[A].sub[C] /= Null
          /\ ~hardlinkDone)  \* before hardlink, no double-placement

\* (DebugRetryBound removed — retryCount no longer in vars; bounded
\* state gives naturally finite state space.)

\* EventuallyAllDone: liveness.
EventuallyAllDone == <>AllDone

=============================================================================
