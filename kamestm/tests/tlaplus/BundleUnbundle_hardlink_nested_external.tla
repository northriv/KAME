(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.
 ***************************************************************************)
------------------- MODULE BundleUnbundle_hardlink_nested_external -------------------
(*
 * Hard-link NESTED-bundle model — the foster parent is bundled as a
 * recursive (non-root) sub-bundle, so the Phase-4 reachability gate uses
 * the LOCAL sub-bundle root instead of the true global root.
 *
 *        R  (global root; top-level bundle, is_bundle_root=TRUE)
 *       / \
 *      A   M   <- A and M are both children of R
 *      |  / \
 *      C D   C   (M.sub[C] = Null hard-link; C's real packet lives in A.sub[C])
 *
 *   - C is hard-linked: HOME under A (A.sub[C] = packet), FOSTER under M
 *     (M.sub[C] = Null reference).  C is reachable from R only via A.
 *   - M is `missing`, so R's Phase-1 collection delegates to a NESTED
 *     bundle  M->bundle(..., is_bundle_root=FALSE)  (transaction_impl.h
 *     :2291 recurses on a missing() child).
 *   - That nested bundle collects D (present) and preserves the C Null
 *     slot (a legit hard-link keeps M non-missing), reaching Phase 4.
 *   - Phase-4 gate: allSubReachable(newpacket) is called SINGLE-ARG
 *     (transaction_impl.h:2645), so globalroot defaults to {} and
 *     degenerates to groot = M's own (local) packet (:1055).  C's home A
 *     is a SIBLING of M, outside M's subtree, so reverseLookup(C, M)
 *     fails -> restore m_missing=true + return DISTURBED (:2649-2652) ->
 *     the outer bundle retries -> M is still missing -> same gate failure
 *     -> retry-forever (LIVELOCK).
 *
 * The defect is a SCOPE bug (which root the gate checks), so the model
 * abstracts reverseLookup to subtree membership over the static home-edge
 * tree.  No concurrency / CAS is needed — the livelock is single-threaded.
 *
 * A/B via CONSTANT UseGlobalRoot:
 *   FALSE = faithful to current C++ (single-arg gate; groot = local M)
 *           -> EventuallyAllDone VIOLATED (nested-scope livelock).
 *   TRUE  = the fix (thread the true global root R into the gate)
 *           -> EventuallyAllDone HOLDS.
 *   SnapshotConsistency (never publish an unreachable Null slot) HOLDS in
 *   BOTH configs — the local-root gate is over-strict, never unsafe; the
 *   bug is liveness-only.
 *
 * CAVEAT (author residual, dossier §8.2-1): this model ASSUMES the
 * topology via Init — a `missing` M carrying an external hard-link,
 * sub-bundled.  It settles the CONDITIONAL "IF that state arises, THEN the
 * local-root gate livelocks and the global-root gate fixes it"; it does
 * NOT establish that the STM can construct that state from a clean tree
 * via legitimate operations.  That structural reachability is the
 * remaining author question.
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Threads,
    R, A, M, D, C,
    UseGlobalRoot

Nodes == {R, A, M, D, C}

ASSUME Cardinality(Threads) > 0
ASSUME UseGlobalRoot \in BOOLEAN

VARIABLES
    pc,           \* [Threads -> String]
    mFinalized,   \* BOOLEAN — has M's nested bundle published ~missing?
    bundleDone    \* [Threads -> BOOLEAN] — R's outer bundle completed

vars == <<pc, mFinalized, bundleDone>>

-----------------------------------------------------------------------------
(* Static home-edge tree (where each node's PACKET actually lives).
   C's home edge is A->C.  M->C is a Null foster slot, NOT a home edge —
   that is exactly why the local-root gate cannot see C. *)

HomeChildren(n) ==
    CASE n = R -> {A, M}
      [] n = A -> {C}
      [] n = M -> {D}
      [] OTHER -> {}

\* Reachability over home edges (5-node tree, depth <= 2; a depth-3
\* closure is exact and idempotent thereafter).
Step(s) == s \cup UNION { HomeChildren(n) : n \in s }
Reach(root) == Step(Step(Step({root})))
CReachableFrom(root) == C \in Reach(root)

\* The gate's root: the current C++ single-arg call degenerates to the
\* LOCAL sub-bundle root (M); the fix threads the true global root (R).
GateRoot == IF UseGlobalRoot THEN R ELSE M
CGateReachable == CReachableFrom(GateRoot)

-----------------------------------------------------------------------------
(* Init — the assumed topology: R is mid-bundle, M is missing (pending its
   nested sub-bundle), the C hard-link is homed at A. *)

Init ==
    /\ pc = [t \in Threads |-> "idle"]
    /\ mFinalized = FALSE
    /\ bundleDone = [t \in Threads |-> FALSE]

-----------------------------------------------------------------------------
(* Outer bundle at R, delegating the missing child M to a nested bundle. *)

BundleStart(t) ==
    /\ pc[t] = "idle"
    /\ ~bundleDone[t]
    /\ pc' = [pc EXCEPT ![t] = "outer_collect"]
    /\ UNCHANGED <<mFinalized, bundleDone>>

\* R Phase-1 collection.  A is present; M is missing unless its nested
\* bundle has already finalized it.  A missing M is delegated to the
\* nested bundle whose Phase-4 gate is InnerGate below.
OuterCollect(t) ==
    /\ pc[t] = "outer_collect"
    /\ IF mFinalized
       THEN pc' = [pc EXCEPT ![t] = "outer_finalize"]
       ELSE pc' = [pc EXCEPT ![t] = "inner_gate"]
    /\ UNCHANGED <<mFinalized, bundleDone>>

\* Nested bundle  M->bundle(...,FALSE)  has collected D and preserved the
\* C Null slot; this is its Phase-4 gate (allSubReachable + finalize).
InnerGate(t) ==
    /\ pc[t] = "inner_gate"
    /\ IF CGateReachable
       THEN /\ mFinalized' = TRUE                        \* publish M ~missing (finalize)
            /\ pc' = [pc EXCEPT ![t] = "outer_collect"]   \* re-collect R (M now present)
       ELSE /\ mFinalized' = FALSE                       \* DISTURBED: restore m_missing=true
            /\ pc' = [pc EXCEPT ![t] = "outer_collect"]   \* outer bundle retries from Phase 1
    /\ UNCHANGED <<bundleDone>>

OuterFinalize(t) ==
    /\ pc[t] = "outer_finalize"
    /\ bundleDone' = [bundleDone EXCEPT ![t] = TRUE]
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<mFinalized>>

-----------------------------------------------------------------------------
AllDone == \A t \in Threads : bundleDone[t]

NextStep ==
    \E t \in Threads :
        \/ BundleStart(t)
        \/ OuterCollect(t)
        \/ InnerGate(t)
        \/ OuterFinalize(t)

Terminating == AllDone /\ UNCHANGED vars
Next == NextStep \/ Terminating

Spec == Init /\ [][Next]_vars /\ WF_vars(NextStep)

-----------------------------------------------------------------------------
(* Safety: M is never published ~missing (finalized) while C's Null slot
   is unreachable from the TRUE global root.  Holds in BOTH configs — the
   local-root gate is over-strict (a liveness bug) but never unsafe. *)
SnapshotConsistency == mFinalized => CReachableFrom(R)

(* Liveness: the outer bundle eventually completes.
     VIOLATED under UseGlobalRoot=FALSE  (nested-scope livelock);
     HOLDS   under UseGlobalRoot=TRUE   (global-root fix). *)
EventuallyAllDone == <>AllDone

=============================================================================
