(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp
 ***************************************************************************)
----------------------- MODULE BundleUnbundle_hardlink_nonatomic -----------------------
(*
 * Hard-link non-atomic test pattern model
 *
 * Companion to BundleUnbundle_hardlink_4node.tla.  Models the test pattern
 * from the b23fa954 commit on claude/refactor-negotiate-scoped-f7de2 that
 * reverted to non-transactional p1->insert/release(p2) interleaved with
 * transactional gn1/gn2 operations.
 *
 * Per-thread sequence:
 *   ❶ NonTxInsertAC  : A.sub[C] ← Cpkt,  C.bundledBy ← A   (non-tx)
 *   ❷ TxInsertHardlink: B.sub[C] ← Cpkt  ∧  Root.sub[A] ← Apkt   (atomic tx)
 *   ❸ NonTxReleaseAC : A.sub[C] ← Null   (non-tx;  C.bundledBy stays = A)
 *                       ↑ LIMBO STATE — C now claims to be bundled to A,
 *                         but A no longer references C structurally.
 *   ❹ TxReleaseHardlink: B.sub[C] ← Null ∧  Root.sub[A] ← Null   (atomic tx)
 *   Destructor (end of thread):
 *      finalize C (and A)
 *
 * The crucial state is the LIMBO at step ❸: C.bundledBy points at A
 * (stale, since A no longer has C in its sub[]).  The destructor's
 * snapshot()/bundle() on C must resolve this.
 *
 * Two finalize variants compared (CONSTANT UseFixVariant):
 *   FALSE — master bundle-fallthrough:
 *           snapshot() NODE_MISSING + missing local packet falls through
 *           to bundle().  bundle() walks bundledBy chain → must acquire
 *           scope on A and (transitively) Root to finalise.
 *   TRUE  — b23fa954 self-promote:
 *           snapshot() NODE_MISSING handler CASes a priority wrapper
 *           directly onto C's linkage, bypassing the chain.  Needs scope
 *           on C's linkage only.
 *
 * Property under test: EventuallyAllDone (every thread's destructor
 * completes) under per-action WF.  If both variants hold, master is
 * theoretically live (b23fa954 fix is a performance/scheduling
 * optimization).  If only the fix variant holds, master has a real
 * liveness gap.
 *)

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Threads,
    MaxIter,            \* per-thread iteration count
    Null,
    UseFixVariant       \* TRUE = b23fa954 self-promote; FALSE = master fall-through

ASSUME Cardinality(Threads) >= 1
ASSUME MaxIter \in Nat /\ MaxIter >= 1
ASSUME UseFixVariant \in BOOLEAN

VARIABLES
    rootBusy,           \* BOOLEAN — TRUE while any thread is mid-tx touching Root
                        \*           (= mid-step-❷ or mid-step-❹ or mid-finalize_walk)
    aLimbo,             \* [Threads -> BOOLEAN] — A_t in stale-bundledBy=Root limbo
    cLimbo,             \* [Threads -> BOOLEAN] — C_t in stale-bundledBy=A limbo
    pc,                 \* [Threads -> step]
    iterCount,          \* [Threads -> 0..MaxIter]
    cFinalized,         \* [Threads -> BOOLEAN]
    aFinalized          \* [Threads -> BOOLEAN]

vars == <<rootBusy, aLimbo, cLimbo, pc, iterCount, cFinalized, aFinalized>>

Steps == {"idle",
          "s1_done",       \* after ❶
          "s2_in_tx",      \* mid ❷ (Root busy)
          "s2_done",
          "s3_done",       \* after ❸ — cLimbo TRUE
          "s4_in_tx",      \* mid ❹ (Root busy)
          "s4_done",       \* loop iteration complete
          "fin_c_walk",    \* master variant: trying to finalize C via chain
          "fin_c_cas",     \* fix variant: trying to finalize C via direct CAS
          "fin_a_walk",    \* finalize A (only walk variant; A.bundledBy=Root)
          "fin_a_cas",     \* fix variant for A
          "done"}

Init ==
    /\ rootBusy = FALSE
    /\ aLimbo = [t \in Threads |-> FALSE]
    /\ cLimbo = [t \in Threads |-> FALSE]
    /\ pc = [t \in Threads |-> "idle"]
    /\ iterCount = [t \in Threads |-> 0]
    /\ cFinalized = [t \in Threads |-> FALSE]
    /\ aFinalized = [t \in Threads |-> FALSE]

-----------------------------------------------------------------------------
(* One iteration of the loop *)

\* ❶ NonTxInsertAC — local CAS on A_t.linkage, no Root involvement.
DoStep1(t) ==
    /\ pc[t] = "idle"
    /\ iterCount[t] < MaxIter
    /\ pc' = [pc EXCEPT ![t] = "s1_done"]
    /\ UNCHANGED <<rootBusy, aLimbo, cLimbo, iterCount, cFinalized, aFinalized>>

\* ❷ TxInsertHardlink — TX touches Root.  Two atomic actions: enter tx and exit tx.
\*   Enter tx: must acquire scope on Root (rootBusy must be FALSE).
DoStep2Enter(t) ==
    /\ pc[t] = "s1_done"
    /\ ~rootBusy
    /\ rootBusy' = TRUE
    /\ pc' = [pc EXCEPT ![t] = "s2_in_tx"]
    /\ UNCHANGED <<aLimbo, cLimbo, iterCount, cFinalized, aFinalized>>

DoStep2Exit(t) ==
    /\ pc[t] = "s2_in_tx"
    /\ rootBusy' = FALSE
    /\ pc' = [pc EXCEPT ![t] = "s2_done"]
    /\ UNCHANGED <<aLimbo, cLimbo, iterCount, cFinalized, aFinalized>>

\* ❸ NonTxReleaseAC — local CAS on A_t.linkage; leaves C in limbo.
DoStep3(t) ==
    /\ pc[t] = "s2_done"
    /\ pc' = [pc EXCEPT ![t] = "s3_done"]
    /\ cLimbo' = [cLimbo EXCEPT ![t] = TRUE]   \* C's bundledBy=A still set
    /\ UNCHANGED <<rootBusy, aLimbo, iterCount, cFinalized, aFinalized>>

\* ❹ TxReleaseHardlink — TX touches Root.
DoStep4Enter(t) ==
    /\ pc[t] = "s3_done"
    /\ ~rootBusy
    /\ rootBusy' = TRUE
    /\ pc' = [pc EXCEPT ![t] = "s4_in_tx"]
    /\ UNCHANGED <<aLimbo, cLimbo, iterCount, cFinalized, aFinalized>>

DoStep4Exit(t) ==
    /\ pc[t] = "s4_in_tx"
    /\ rootBusy' = FALSE
    /\ pc' = [pc EXCEPT ![t] = "s4_done"]
    \* After ❹ both A and C are in limbo (bundledBy stale): A.bundledBy=Root,
    \* C.bundledBy=A.  (cLimbo was set in ❸; aLimbo set here.)
    /\ aLimbo' = [aLimbo EXCEPT ![t] = TRUE]
    /\ UNCHANGED <<cLimbo, iterCount, cFinalized, aFinalized>>

\* Loop iteration complete — back to idle for next iter.
LoopIterEnd(t) ==
    /\ pc[t] = "s4_done"
    /\ iterCount[t] < MaxIter
    /\ iterCount' = [iterCount EXCEPT ![t] = iterCount[t] + 1]
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<rootBusy, aLimbo, cLimbo, cFinalized, aFinalized>>

-----------------------------------------------------------------------------
(* Destructor — runs when iterCount = MaxIter and loop iteration complete *)

\* StartFinalizeC: fire from "idle" when iterCount = MaxIter (loop drained).
StartFinalizeC(t) ==
    /\ pc[t] = "idle"
    /\ iterCount[t] = MaxIter
    /\ ~cFinalized[t]
    /\ pc' = [pc EXCEPT ![t] = IF UseFixVariant THEN "fin_c_cas" ELSE "fin_c_walk"]
    /\ UNCHANGED <<rootBusy, aLimbo, cLimbo, iterCount, cFinalized, aFinalized>>

\* Master variant: bundle-based finalize on C.  C.bundledBy=A → scope
\* acquisition walks to A → A.bundledBy=Root → walks to Root.  Needs
\* scope on Root (rootBusy = FALSE) to proceed.
FinalizeCWalk(t) ==
    /\ pc[t] = "fin_c_walk"
    /\ ~cFinalized[t]
    /\ ~rootBusy                          \* must acquire Root scope
    /\ cLimbo' = [cLimbo EXCEPT ![t] = FALSE]
    /\ cFinalized' = [cFinalized EXCEPT ![t] = TRUE]
    /\ pc' = [pc EXCEPT ![t] = IF UseFixVariant THEN "fin_a_cas" ELSE "fin_a_walk"]
    /\ UNCHANGED <<rootBusy, aLimbo, iterCount, aFinalized>>

\* Fix variant: direct CAS on C's linkage only — no chain walk, no Root scope.
FinalizeCCas(t) ==
    /\ pc[t] = "fin_c_cas"
    /\ ~cFinalized[t]
    /\ cLimbo' = [cLimbo EXCEPT ![t] = FALSE]
    /\ cFinalized' = [cFinalized EXCEPT ![t] = TRUE]
    /\ pc' = [pc EXCEPT ![t] = IF UseFixVariant THEN "fin_a_cas" ELSE "fin_a_walk"]
    /\ UNCHANGED <<rootBusy, aLimbo, iterCount, aFinalized>>

\* A's destructor — A.bundledBy=Root, so walk variant needs Root scope too.
FinalizeAWalk(t) ==
    /\ pc[t] = "fin_a_walk"
    /\ ~aFinalized[t]
    /\ ~rootBusy
    /\ aLimbo' = [aLimbo EXCEPT ![t] = FALSE]
    /\ aFinalized' = [aFinalized EXCEPT ![t] = TRUE]
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ UNCHANGED <<rootBusy, cLimbo, iterCount, cFinalized>>

FinalizeACas(t) ==
    /\ pc[t] = "fin_a_cas"
    /\ ~aFinalized[t]
    /\ aLimbo' = [aLimbo EXCEPT ![t] = FALSE]
    /\ aFinalized' = [aFinalized EXCEPT ![t] = TRUE]
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ UNCHANGED <<rootBusy, cLimbo, iterCount, cFinalized>>

-----------------------------------------------------------------------------
(* Next + Spec *)

AllDone == \A t \in Threads : cFinalized[t] /\ aFinalized[t]

NextStep ==
    \E t \in Threads :
        \/ DoStep1(t)
        \/ DoStep2Enter(t)
        \/ DoStep2Exit(t)
        \/ DoStep3(t)
        \/ DoStep4Enter(t)
        \/ DoStep4Exit(t)
        \/ LoopIterEnd(t)
        \/ StartFinalizeC(t)
        \/ FinalizeCWalk(t)
        \/ FinalizeCCas(t)
        \/ FinalizeAWalk(t)
        \/ FinalizeACas(t)

Terminating == AllDone /\ UNCHANGED vars

Next == NextStep \/ Terminating

\* Per-action WF on every progress step of every thread (per the recipe
\* established in BundleUnbundle_hardlink_external_migration.tla).
\* This models OS-level scheduling fairness: no thread can starve another
\* thread's progress step indefinitely.
Spec == Init /\ [][Next]_vars
        /\ WF_vars(NextStep)
        /\ \A t \in Threads :
            /\ WF_vars(DoStep1(t))
            /\ WF_vars(DoStep2Enter(t))
            /\ WF_vars(DoStep2Exit(t))
            /\ WF_vars(DoStep3(t))
            /\ WF_vars(DoStep4Enter(t))
            /\ WF_vars(DoStep4Exit(t))
            /\ WF_vars(LoopIterEnd(t))
            /\ WF_vars(StartFinalizeC(t))
            /\ WF_vars(FinalizeCWalk(t))
            /\ WF_vars(FinalizeCCas(t))
            /\ WF_vars(FinalizeAWalk(t))
            /\ WF_vars(FinalizeACas(t))

-----------------------------------------------------------------------------
(* Invariants *)

\* No two threads can hold Root simultaneously.
RootMutex == rootBusy => Cardinality({t \in Threads : pc[t] \in {"s2_in_tx", "s4_in_tx"}}) = 1

\* When in limbo, bundledBy points at parent that doesn't reference us
\* (this is the stale state) — purely structural; no invariant to check.

EventuallyAllDone == <>AllDone

-----------------------------------------------------------------------------
(* Weak-fairness-only Spec — to isolate "is master live without OS
   per-action scheduling fairness?".  Same Init/Next, only blanket
   WF_vars(NextStep) — no per-action WF. *)
WeakSpec == Init /\ [][Next]_vars /\ WF_vars(NextStep)
=====
