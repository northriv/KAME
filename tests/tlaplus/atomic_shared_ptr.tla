(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************)
--------------------------- MODULE atomic_shared_ptr ---------------------------
(*
 * TLA+ model of kame/atomic_smart_ptr.h core protocol:
 *   acquire_tag_ref_(), load_shared_(), release_tag_ref_(), compareAndSwap_(), local_shared_ptr::reset()
 *
 * Models the tagged-pointer scheme where the lower bits of an atomic word
 * store a local reference counter, while the upper bits store the pointer
 * to a Ref object that itself holds a global reference counter.
 *
 * We separate the atomic word into two variables (ptr, local_rc) for clarity.
 * Each Ref object has a global_rc field.
 *
 * Improvements over v1:
 *   - ABA problem modeling: freed objects can be recycled (same pointer value
 *     reappears), testing whether the tagged-pointer CAS prevents ABA.
 *   - acquire_tag_ref_ read modeled as single atomic load of m_ref (ptr + local_rc
 *     are extracted from the same load in C++); CAS catches stale values.
 *   - Symmetry support for state space reduction.
 *
 * Source: kame/atomic_smart_ptr.h lines 420-613
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Threads,          \* set of thread IDs
    Objects,          \* set of possible object IDs (Ref objects)
    EnableCAS,        \* TRUE to enable compareAndSwap_ operations
    EnableSwap,       \* TRUE to enable local_shared_ptr::swap(atomic_shared_ptr&)
    EnableRecycle     \* TRUE to enable ABA recycling of freed objects

ASSUME Cardinality(Objects) >= 2

NULL == CHOOSE x : x \notin Objects

(* Symmetry sets — declared here, used in .cfg via SYMMETRY *)
ThreadSymmetry == Permutations(Threads)
ObjectSymmetry == Permutations(Objects)

\* ==========================================================================
\* @c11_mapping — Variable-to-C11 correspondence
\*
\* TLA+ variable              C11 type & expression
\* --------------------------------------------------------------------------
\* @c11_var ptr:              Obj* -- upper bits of _Atomic uintptr_t g_ref
\*                            extracted as: (Obj*)(g_ref & PTR_MASK)
\*
\* @c11_var local_rc:         unsigned -- lower bits of g_ref
\*                            extracted as: (unsigned)(g_ref & TAG_MASK)
\*   NOTE: ptr and local_rc are packed in ONE atomic word g_ref.
\*   Reads: atomic_load_explicit(&g_ref, memory_order_relaxed)
\*   Writes: via atomic_compare_exchange_weak_explicit only.
\*
\* @c11_var global_rc[o]:     _Atomic uintptr_t o->refcnt
\*   — per-object intrusive reference counter
\*   Increment: atomic_fetch_add_explicit(&o->refcnt, N, memory_order_relaxed)
\*   DecAndTest: atomic_fetch_sub_explicit(&o->refcnt, 1, memory_order_acq_rel)
\*
\* @c11_var freed[o]:         o->destroyed   (non-atomic int)
\*   — set to 1 when global_rc reaches 0 via decAndTest
\*   Written only by the thread that observes refcnt==1.
\*
\* @c11_var thr_pref[t]:      thread-local Obj *pref
\* @c11_var thr_rcnt[t]:      thread-local uintptr_t rcnt_old / rcnt_new
\* @c11_var thr_old[t]:       thread-local Obj *oldr  (CAS expected value)
\* @c11_var thr_new[t]:       thread-local Obj *newr  (CAS desired value)
\* @c11_var thr_holds[t][o]:  thread-local reference count (local_shared_ptr)
\*   — non-atomic; purely thread-local bookkeeping
\*
\* C11 constants:
\*   #define CAPACITY   8u               == LOCAL_REF_CAPACITY
\*   #define PTR_MASK   ~(uintptr_t)(CAPACITY - 1)
\*   #define TAG_MASK    (uintptr_t)(CAPACITY - 1)
\*
\* GenMC tests:  tests/cds_atomic_shared_ptr/cds_test_{load,cas,multi_cas}.c
\* ==========================================================================

(* -------------------------------------------------------------------------- *)
(* Global state of the atomic_shared_ptr                                      *)
(* -------------------------------------------------------------------------- *)
VARIABLES
    ptr,            \* current pointer in the atomic word (an Object or NULL)
    local_rc,       \* local reference counter (lower bits of atomic word)

    (* Per-object state *)
    global_rc,      \* global_rc[o]: global reference counter for each object
    freed,          \* freed[o]: TRUE if object has been "deleted"

    (* Per-thread state *)
    pc,             \* program counter for each thread
    thr_op,         \* what operation thread is performing: "load", "cas", "idle"
    thr_pref,       \* pointer read by acquire_tag_ref_ (thread-local)
    thr_rcnt,       \* local refcount value returned by acquire_tag_ref_
    thr_old,        \* for CAS: the expected old object
    thr_new,        \* for CAS: the new object to install
    thr_holds,      \* thr_holds[t][o]: number of references thread t holds to object o
    thr_rtr_ctx      \* release_tag_ref_ return target: "load_done", "cas_retry", "cas_fail"

vars == <<ptr, local_rc, global_rc, freed, pc, thr_op, thr_pref, thr_rcnt,
          thr_old, thr_new, thr_holds, thr_rtr_ctx>>

(* -------------------------------------------------------------------------- *)
(* Type invariant                                                             *)
(* -------------------------------------------------------------------------- *)
TypeOK ==
    /\ ptr \in Objects \cup {NULL}
    /\ local_rc \in Nat
    /\ global_rc \in [Objects -> Nat]
    /\ freed \in [Objects -> BOOLEAN]
    /\ pc \in [Threads -> {"idle",
                           "atr_read",     \* acquire_tag_ref_: atomic read of ptr + local_rc
                           "atr_cas",      \* acquire_tag_ref_: CAS to increment local_rc
                           "ls_inc",    \* load_shared_: increment global_rc
                           "ls_release",  \* load_shared_: call release_tag_ref_
                           "rtr_read",     \* release_tag_ref_: read refcount
                           "rtr_cas",      \* release_tag_ref_: CAS to decrement local_rc
                           "rtr_global",   \* release_tag_ref_: fallback - dec global_rc
                           "done",        \* operation completed
                           "cas_pre_inc",   \* CAS: pre-increment newr's global_rc
                           "cas_acquire",   \* CAS: call acquire_tag_ref_
                           "cas_check",     \* CAS: check if pref == oldr
                           "cas_fail_done", \* CAS mismatch: rollback newr, return false
                           "cas_transfer",  \* CAS: transfer local_rc to global_rc
                           "cas_swap",      \* CAS: the actual CAS on the atomic word
                           "cas_undo",      \* CAS: undo transfer on CAS failure
                           "cas_cleanup"    \* CAS: decrement pref's global_rc on success
                          }]
    /\ thr_op \in [Threads -> {"load", "cas", "swap", "idle"}]
    /\ thr_pref \in [Threads -> Objects \cup {NULL}]
    /\ thr_rcnt \in [Threads -> Nat]
    /\ thr_old \in [Threads -> Objects \cup {NULL}]
    /\ thr_new \in [Threads -> Objects \cup {NULL}]
    /\ thr_holds \in [Threads -> [Objects -> Nat]]
    /\ thr_rtr_ctx \in [Threads -> {"load_done", "cas_retry", "cas_fail", "none"}]

(* -------------------------------------------------------------------------- *)
(* Initial state                                                              *)
(* -------------------------------------------------------------------------- *)
Init ==
    LET initObj == CHOOSE o \in Objects : TRUE
        otherObjs == Objects \ {initObj}
        otherObj == IF otherObjs /= {} THEN CHOOSE o2 \in otherObjs : TRUE ELSE NULL
        firstThread == CHOOSE t \in Threads : TRUE
    IN
    /\ ptr = initObj
    /\ local_rc = 0
    /\ global_rc = [o \in Objects |-> IF o = initObj THEN 1
                                      ELSE IF o = otherObj THEN 1
                                      ELSE 0]
    /\ freed = [o \in Objects |-> IF o = initObj \/ o = otherObj THEN FALSE ELSE TRUE]
    /\ pc = [t \in Threads |-> "idle"]
    /\ thr_op = [t \in Threads |-> "idle"]
    /\ thr_pref = [t \in Threads |-> NULL]
    /\ thr_rcnt = [t \in Threads |-> 0]
    /\ thr_old = [t \in Threads |-> NULL]
    /\ thr_new = [t \in Threads |-> NULL]
    /\ thr_holds = [t \in Threads |->
                      [o \in Objects |->
                        IF t = firstThread /\ o = otherObj /\ otherObj /= NULL
                        THEN 1
                        ELSE 0]]
    /\ thr_rtr_ctx = [t \in Threads |-> "none"]

(* ========================================================================== *)
(* ABA Recycling: a freed object can be "reallocated" at the same address     *)
(* This simulates the allocator returning the same pointer value, which is    *)
(* the precondition for an ABA problem. The recycled object gets global_rc=1  *)
(* and is given to a thread (simulating new + local_shared_ptr assignment).   *)
(* ========================================================================== *)
\* @c11_action Recycle(t):
\*   Models: Obj *newobj = malloc(sizeof(Obj));
\*   The allocator may return the same address as a previously freed object
\*   (ABA scenario). newobj->refcnt = 1; newobj->destroyed = 0;
\*   Not an atomic operation — models heap reuse / constructor.
Recycle(t) ==
    /\ EnableRecycle
    /\ pc[t] \in {"idle", "done"}
    /\ \A o2 \in Objects : thr_holds[t][o2] = 0  \* C++: old local_shared_ptr destroyed before new allocation
    /\ \E o \in Objects :
       /\ freed[o] = TRUE
       /\ global_rc[o] = 0
       /\ freed' = [freed EXCEPT ![o] = FALSE]
       /\ global_rc' = [global_rc EXCEPT ![o] = 1]
       /\ thr_holds' = [thr_holds EXCEPT ![t][o] = @ + 1]
    /\ UNCHANGED <<ptr, local_rc, pc, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_rtr_ctx>>

(* ========================================================================== *)
(* load_shared_() operation: acquire_tag_ref_ + increment global_rc + release_tag_ref_       *)
(* Models lines 494-503                                                       *)
(* ========================================================================== *)

\* @c11_action StartLoadShared(t):
\*   Obj *p = load_shared();
\*   Precondition: previous local_shared_ptr is destroyed (holds==0).
\*   Source: atomic_smart_ptr.h:500 -> cds_test_load.c:122
StartLoadShared(t) ==
    /\ pc[t] = "idle"
    /\ \A o \in Objects : thr_holds[t][o] = 0  \* C++: local_shared_ptr destroyed before next load
    /\ pc' = [pc EXCEPT ![t] = "atr_read"]
    /\ thr_op' = [thr_op EXCEPT ![t] = "load"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_rtr_ctx>>

(* ========================================================================== *)
(* acquire_tag_ref_(): atomic read of m_ref                                      *)
(* C++ m_ref.load() is a single atomic operation; ptr and local_rc are        *)
(* extracted from the same loaded value. The subsequent CAS catches any        *)
(* change that occurred after this read.                                       *)
(* Lines 462-466: pref = pref_(); rcnt_old = refcnt_();                       *)
(* ========================================================================== *)

\* @c11_action AcquireTagRefRead(t):
\* uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
\* pref = (Obj(cur & PTR_MASK);
\* rcnt_old = (unsigned)(cur & TAG_MASK);
\* Source: atomic_smart_ptr.h:466-469 → cds_test_load.c:65-67
AcquireTagRefRead(t) ==
    /\ pc[t] = "atr_read"
    /\ thr_pref' = [thr_pref EXCEPT ![t] = ptr]
    /\ thr_rcnt' = [thr_rcnt EXCEPT ![t] = local_rc]
    /\ pc' = [pc EXCEPT ![t] = "atr_cas"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_old,
                   thr_new, thr_holds, thr_rtr_ctx>>

\* @c11_action AcquireTagRefCAS(t):
\* uintptr_t expected = (uintptr_t)pref + rcnt_old;
\* uintptr_t desired  = (uintptr_t)pref + rcnt_old + 1u;
\* if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
\* memory_order_acq_rel, memory_order_relaxed))
\* break;   // success → ls_inc (load) | cas_transfer (swap) | cas_check (cas)
\* else
\* continue; // fail → atr_read (retry)
\* CAS compares the FULL word (ptr + local_rc together); if either changed, fails.
\* Source: atomic_smart_ptr.h:489-492 → cds_test_load.c:76-80
AcquireTagRefCAS(t) ==
    /\ pc[t] = "atr_cas"
    /\ thr_pref[t] /= NULL
    /\ \/ (* CAS succeeds: both ptr AND local_rc match *)
          /\ ptr = thr_pref[t]
          /\ local_rc = thr_rcnt[t]
          /\ local_rc' = thr_rcnt[t] + 1
          /\ thr_rcnt' = [thr_rcnt EXCEPT ![t] = thr_rcnt[t] + 1]
          /\ pc' = [pc EXCEPT ![t] = CASE thr_op[t] = "load" -> "ls_inc"
                                       [] thr_op[t] = "swap" -> "cas_transfer"
                                       [] OTHER -> "cas_check"]
          /\ UNCHANGED <<ptr, global_rc, freed, thr_op, thr_pref, thr_old,
                         thr_new, thr_holds, thr_rtr_ctx>>
       \/ (* CAS fails: retry from read *)
          /\ ~(ptr = thr_pref[t] /\ local_rc = thr_rcnt[t])
          /\ pc' = [pc EXCEPT ![t] = "atr_read"]
          /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                         thr_old, thr_new, thr_holds, thr_rtr_ctx>>

\* @c11_action AcquireTagRefNull(t):
\* if (!pref) { *rcnt = rcnt_old; return NULL; }
\* Source: atomic_smart_ptr.h:470-474 → cds_test_load.c:68-71
AcquireTagRefNull(t) ==
    /\ pc[t] = "atr_cas"
    /\ thr_pref[t] = NULL
    /\ pc' = [pc EXCEPT ![t] = CASE thr_op[t] = "load" -> "done"
                                  [] thr_op[t] = "swap" -> "cas_transfer"
                                  [] OTHER -> "cas_check"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_rtr_ctx>>

\* @c11_action LoadSharedIncGlobal(t):
\* atomic_fetch_add_explicit(&pref->refcnt, 1, memory_order_relaxed);
\* Source: atomic_smart_ptr.h:504 → cds_test_load.c:127
LoadSharedIncGlobal(t) ==
    /\ pc[t] = "ls_inc"
    /\ thr_op[t] = "load"
    /\ global_rc' = [global_rc EXCEPT ![thr_pref[t]] = @ + 1]
    /\ pc' = [pc EXCEPT ![t] = "ls_release"]
    /\ UNCHANGED <<ptr, local_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_rtr_ctx>>

\* @c11_action LoadSharedStartRelease(t):
\* // fall-through to release_tag_ref_(pref)
\* Source: atomic_smart_ptr.h:505 → cds_test_load.c:128
LoadSharedStartRelease(t) ==
    /\ pc[t] = "ls_release"
    /\ thr_op[t] = "load"
    /\ pc' = [pc EXCEPT ![t] = "rtr_read"]
    /\ thr_rtr_ctx' = [thr_rtr_ctx EXCEPT ![t] = "load_done"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds>>

(* ========================================================================== *)
(* release_tag_ref_() operation (shared by load_shared_ and compareAndSwap_)              *)
(* Lines 512-531                                                              *)
(* After completion, control returns based on thr_rtr_ctx:                     *)
(*   "load_done" -> done (load_shared completed)                                     *)
(*   "cas_retry" -> cas_reserve (inner CAS failed, retry)                     *)
(*   "cas_fail"  -> cas_fail_done (mismatch, return false)                    *)
(* ========================================================================== *)

\* @c11_action ReleaseTagRefRead(t):
\* rcnt_old = atomic_load_explicit(&g_ref, memory_order_relaxed) & TAG_MASK;
\* if (rcnt_old)  → rtr_cas   // try CAS path
\* else           → rtr_global // fallback: dec global_rc
\* Source: atomic_smart_ptr.h:518-520 → cds_test_load.c:92-94
ReleaseTagRefRead(t) ==
    /\ pc[t] = "rtr_read"
    /\ thr_rcnt' = [thr_rcnt EXCEPT ![t] = local_rc]
    /\ pc' = [pc EXCEPT ![t] = IF local_rc > 0 THEN "rtr_cas" ELSE "rtr_global"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref,
                   thr_old, thr_new, thr_holds, thr_rtr_ctx>>

\* @c11_action ReleaseTagRefCAS(t):
\* uintptr_t expected = (uintptr_t)pref + rcnt_old;
\* uintptr_t desired  = (uintptr_t)pref + (rcnt_old - 1u);
\* if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
\* memory_order_acq_rel, memory_order_relaxed))
\* break;                        // success → return (load_done|cas_retry|cas_fail)
\* if (get_ptr(cur) == pref)
\* continue;                     // ptr unchanged, retry CAS → rtr_read
\* else
\* goto global_fallback;         // ptr changed → rtr_global
\* Source: atomic_smart_ptr.h:521-528 → cds_test_load.c:95-104
ReleaseTagRefCAS(t) ==
    /\ pc[t] = "rtr_cas"
    /\ LET returnPC == CASE thr_rtr_ctx[t] = "load_done" -> "done"
                          [] thr_rtr_ctx[t] = "cas_retry" -> "cas_acquire"
                          [] thr_rtr_ctx[t] = "cas_fail"  -> "cas_fail_done"
                          [] OTHER -> "done"
       IN
       \/ (* CAS succeeds: ptr unchanged and local_rc matches *)
          /\ ptr = thr_pref[t]
          /\ local_rc = thr_rcnt[t]
          /\ local_rc' = thr_rcnt[t] - 1
          /\ IF thr_rtr_ctx[t] = "load_done"
             THEN thr_holds' = [thr_holds EXCEPT ![t][thr_pref[t]] = @ + 1]
             ELSE UNCHANGED thr_holds
          /\ pc' = [pc EXCEPT ![t] = returnPC]
          /\ UNCHANGED <<ptr, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                         thr_old, thr_new, thr_rtr_ctx>>
       \/ (* CAS fails, ptr still matches -> retry release_tag_ref_ *)
          /\ ~(ptr = thr_pref[t] /\ local_rc = thr_rcnt[t])
          /\ ptr = thr_pref[t]
          /\ pc' = [pc EXCEPT ![t] = "rtr_read"]
          /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                         thr_old, thr_new, thr_holds, thr_rtr_ctx>>
       \/ (* CAS fails, ptr changed -> fallback to global dec *)
          /\ ~(ptr = thr_pref[t] /\ local_rc = thr_rcnt[t])
          /\ ptr /= thr_pref[t]
          /\ pc' = [pc EXCEPT ![t] = "rtr_global"]
          /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                         thr_old, thr_new, thr_holds, thr_rtr_ctx>>

\* @c11_action ReleaseTagRefGlobal(t):
\* uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1,
\* memory_order_acq_rel);                    // decAndTest
\* if (old_rc == 1) pref->destroyed = 1;             // "delete"
\* Source: atomic_smart_ptr.h:531-533 → cds_test_load.c:108-113
ReleaseTagRefGlobal(t) ==
    /\ pc[t] = "rtr_global"
    /\ thr_pref[t] /= NULL
    /\ LET o == thr_pref[t]
           returnPC == CASE thr_rtr_ctx[t] = "load_done" -> "done"
                         [] thr_rtr_ctx[t] = "cas_retry" -> "cas_acquire"
                         [] thr_rtr_ctx[t] = "cas_fail"  -> "cas_fail_done"
                         [] OTHER -> "done"
       IN
       /\ global_rc' = [global_rc EXCEPT ![o] = @ - 1]
       /\ IF global_rc[o] = 1
          THEN freed' = [freed EXCEPT ![o] = TRUE]
          ELSE freed' = freed
       /\ IF thr_rtr_ctx[t] = "load_done"
          THEN thr_holds' = [thr_holds EXCEPT ![t][thr_pref[t]] = @ + 1]
          ELSE UNCHANGED thr_holds
       /\ pc' = [pc EXCEPT ![t] = returnPC]
       /\ UNCHANGED <<ptr, local_rc, thr_op, thr_pref, thr_rcnt, thr_old,
                      thr_new, thr_rtr_ctx>>

(* ========================================================================== *)
(* local_shared_ptr::reset()                                                  *)
\* @c11_action Reset(t):
\* (*   if (pref->refcnt == 1) {   // unique() fast path
\* (*       pref->refcnt.store(0, memory_order_relaxed);
\* (*       delete pref;
\* (*   } else {
\* (*       uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1,
\* (*               memory_order_acq_rel);                // decAndTest
\* (*       if (old_rc == 1) delete pref;
\* (*   }
\* (*   Source: atomic_smart_ptr.h:433-444 → cds_test_load.c:136-143
\* (* ==========================================================================
Reset(t) ==
    /\ pc[t] \in {"done", "idle"}
    /\ \E o \in Objects :
       /\ thr_holds[t][o] > 0
       /\ thr_holds' = [thr_holds EXCEPT ![t][o] = @ - 1]
       /\ global_rc' = [global_rc EXCEPT ![o] = @ - 1]
       /\ IF global_rc[o] = 1
          THEN freed' = [freed EXCEPT ![o] = TRUE]
          ELSE freed' = freed
    /\ UNCHANGED <<ptr, local_rc, pc, thr_op, thr_pref, thr_rcnt, thr_old,
                   thr_new, thr_rtr_ctx>>

(* ========================================================================== *)
(* compareAndSwap_() operation                                                *)
(* Lines 556-603                                                              *)
(*                                                                            *)
(* Control flow:                                                              *)
(*   1. Pre-increment newr's global_rc (once, not on retry)                   *)
(*   2. for(;;) {                                                             *)
(*        acquire_tag_ref_()                                                     *)
(*        if (pref != oldr) { release_tag_ref; rollback; return false }            *)
(*        transfer local_rc to global_rc                                      *)
(*        CAS m_ref                                                           *)
(*        if CAS fails { undo transfer; release_tag_ref; continue }               *)
(*        break                                                               *)
(*      }                                                                     *)
(*   3. Cleanup: dec pref's global_rc; return true                            *)
(* ========================================================================== *)

\* @c11_action StartCAS(t):
\* // Entry: caller holds local_shared_ptr<T> oldr and newr.
\* // Nondeterministic choice of oldr, newr models arbitrary caller state.
\* // Precondition: newr is alive and held (freed[newr]==FALSE, holds>0).
\* Source: atomic_smart_ptr.h:560 → cds_test_cas.c:122
StartCAS(t) ==
    /\ EnableCAS
    /\ pc[t] = "idle"
    /\ \E oldObj \in Objects \cup {NULL}, newObj \in Objects \cup {NULL} :
       /\ oldObj /= newObj
       /\ (newObj /= NULL => (freed[newObj] = FALSE /\ thr_holds[t][newObj] > 0))
       /\ thr_old' = [thr_old EXCEPT ![t] = oldObj]
       /\ thr_new' = [thr_new EXCEPT ![t] = newObj]
    /\ pc' = [pc EXCEPT ![t] = "cas_pre_inc"]
    /\ thr_op' = [thr_op EXCEPT ![t] = "cas"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_pref, thr_rcnt,
                   thr_holds, thr_rtr_ctx>>

\* @c11_action CASPreInc(t):
\* if (newr)
\* atomic_fetch_add_explicit(&newr->refcnt, 1, memory_order_relaxed);
\* Source: atomic_smart_ptr.h:562-564 → cds_test_cas.c:126-127
CASPreInc(t) ==
    /\ pc[t] = "cas_pre_inc"
    /\ IF thr_new[t] /= NULL
       THEN global_rc' = [global_rc EXCEPT ![thr_new[t]] = @ + 1]
       ELSE UNCHANGED global_rc
    /\ pc' = [pc EXCEPT ![t] = "cas_acquire"]
    /\ UNCHANGED <<ptr, local_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_rtr_ctx>>

\* @c11_action CASReserve(t):
\* // Enter acquire_tag_ref_ loop for CAS/swap
\* pref = acquire_tag_ref_(&rcnt_old);
\* Source: atomic_smart_ptr.h:567 → cds_test_cas.c:131
CASReserve(t) ==
    /\ pc[t] = "cas_acquire"
    /\ pc' = [pc EXCEPT ![t] = "atr_read"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_rtr_ctx>>

\* @c11_action CASCheck(t):
\* if (pref != oldr) {
\* // mismatch:
\* if (pref) {
\* // NOSWAP=false: pref->refcnt.fetch_add(1, relaxed);
\* release_tag_ref_(pref);          → rtr_read (cas_fail)
\* } else {
\* → cas_fail_done                  // NULL, skip release
\* }
\* } else {
\* → cas_transfer                       // match, proceed to CAS
\* }
\* Source: atomic_smart_ptr.h:568-587 → cds_test_cas.c:134-152
CASCheck(t) ==
    /\ pc[t] = "cas_check"
    /\ thr_op[t] = "cas"
    /\ IF thr_pref[t] /= thr_old[t]
       THEN
            IF thr_pref[t] /= NULL
            THEN /\ pc' = [pc EXCEPT ![t] = "rtr_read"]
                 /\ thr_rtr_ctx' = [thr_rtr_ctx EXCEPT ![t] = "cas_fail"]
                 /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref,
                                thr_rcnt, thr_old, thr_new, thr_holds>>
            ELSE /\ pc' = [pc EXCEPT ![t] = "cas_fail_done"]
                 /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref,
                                thr_rcnt, thr_old, thr_new, thr_holds, thr_rtr_ctx>>
       ELSE
            /\ pc' = [pc EXCEPT ![t] = "cas_transfer"]
            /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                           thr_old, thr_new, thr_holds, thr_rtr_ctx>>

\* @c11_action CASFailDone(t):
\* // Rollback newr's pre-incremented refcount
\* if (newr)
\* atomic_fetch_sub_explicit(&newr->refcnt, 1, memory_order_relaxed);
\* // NOSWAP=false: release old oldr via decAndTest, update oldr = pref
\* return false;
\* Source: atomic_smart_ptr.h:575-586 → cds_test_cas.c:142-152
CASFailDone(t) ==
    /\ pc[t] = "cas_fail_done"
    /\ thr_op[t] = "cas"
    /\ IF thr_new[t] /= NULL
       THEN /\ global_rc' = [global_rc EXCEPT ![thr_new[t]] = @ - 1]
            /\ IF global_rc[thr_new[t]] = 1
               THEN freed' = [freed EXCEPT ![thr_new[t]] = TRUE]
               ELSE freed' = freed
       ELSE UNCHANGED <<global_rc, freed>>
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ thr_op' = [thr_op EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<ptr, local_rc, thr_pref, thr_rcnt, thr_old, thr_new,
                   thr_holds, thr_rtr_ctx>>

\* @c11_action CASTransfer(t):
\* // Transfer local refcount to global before the pointer CAS
\* if (pref && (rcnt_old != 1u))
\* atomic_fetch_add_explicit(&pref->refcnt, rcnt_old - 1u,
\* memory_order_relaxed);
\* Source: atomic_smart_ptr.h:588-589 → cds_test_cas.c:156-157
\* Also used by swap: atomic_smart_ptr.h:633-634
CASTransfer(t) ==
    /\ pc[t] = "cas_transfer"
    /\ thr_op[t] \in {"cas", "swap"}
    /\ IF thr_pref[t] /= NULL /\ thr_rcnt[t] /= 1
       THEN global_rc' = [global_rc EXCEPT ![thr_pref[t]] = @ + thr_rcnt[t] - 1]
       ELSE UNCHANGED global_rc
    /\ pc' = [pc EXCEPT ![t] = "cas_swap"]
    /\ UNCHANGED <<ptr, local_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_rtr_ctx>>

\* @c11_action CASSwap(t):
\* uintptr_t expected = (uintptr_t)pref + rcnt_old;
\* uintptr_t desired  = (uintptr_t)newr;  // tag = 0
\* if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
\* memory_order_acq_rel, memory_order_relaxed)) {
\* // success:
\* //   CAS  → cas_cleanup (dec pref's global_rc)
\* //   swap → this->m_ref = pref (local write, hold transfer) → done
\* break;
\* } else {
\* → cas_undo  // undo transfer, release_tag_ref_, retry
\* }
\* Source: atomic_smart_ptr.h:592-595 → cds_test_cas.c:160-163
\* swap:   atomic_smart_ptr.h:637-639
CASSwap(t) ==
    /\ pc[t] = "cas_swap"
    /\ thr_op[t] \in {"cas", "swap"}
    /\ \/ (* CAS succeeds *)
          /\ ptr = thr_pref[t]
          /\ local_rc = thr_rcnt[t]
          /\ ptr' = thr_new[t]
          /\ local_rc' = 0
          /\ IF thr_op[t] = "swap"
             THEN \* swap: hold transfer is immediate (this->m_ref = pref is local)
                  /\ thr_holds' = [thr_holds EXCEPT
                       ![t] = [o \in Objects |->
                         CASE o = thr_pref[t] /\ o = thr_new[t] -> thr_holds[t][o]
                           [] o = thr_pref[t] -> thr_holds[t][o] + 1
                           [] o = thr_new[t]  -> thr_holds[t][o] - 1
                           [] OTHER -> thr_holds[t][o]]]
                  /\ pc' = [pc EXCEPT ![t] = "done"]
                  /\ thr_op' = [thr_op EXCEPT ![t] = "idle"]
             ELSE \* CAS: cleanup is a separate atomic step
                  /\ pc' = [pc EXCEPT ![t] = "cas_cleanup"]
                  /\ UNCHANGED <<thr_holds, thr_op>>
          /\ UNCHANGED <<global_rc, freed, thr_pref, thr_rcnt,
                         thr_old, thr_new, thr_rtr_ctx>>
       \/ (* CAS fails *)
          /\ ~(ptr = thr_pref[t] /\ local_rc = thr_rcnt[t])
          /\ pc' = [pc EXCEPT ![t] = "cas_undo"]
          /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                         thr_old, thr_new, thr_holds, thr_rtr_ctx>>

\* @c11_action CASUndo(t):
\* // Undo the transfer and release local ref
\* if (pref) {
\* if (rcnt_old != 1u)
\* atomic_fetch_add_explicit(&pref->refcnt,
\* -(uintptr_t)(rcnt_old - 1u), memory_order_relaxed);
\* release_tag_ref_(pref);              → rtr_read (cas_retry)
\* } else {
\* → cas_acquire                        // NULL, retry from top
\* }
\* Source: atomic_smart_ptr.h:596-601 → cds_test_cas.c:167-172
\* swap:   atomic_smart_ptr.h:641-646
CASUndo(t) ==
    /\ pc[t] = "cas_undo"
    /\ thr_op[t] \in {"cas", "swap"}
    /\ IF thr_pref[t] /= NULL
       THEN /\ IF thr_rcnt[t] /= 1
               THEN global_rc' = [global_rc EXCEPT ![thr_pref[t]] = @ - (thr_rcnt[t] - 1)]
               ELSE UNCHANGED global_rc
            /\ pc' = [pc EXCEPT ![t] = "rtr_read"]
            /\ thr_rtr_ctx' = [thr_rtr_ctx EXCEPT ![t] = "cas_retry"]
       ELSE /\ pc' = [pc EXCEPT ![t] = "cas_acquire"]
            /\ UNCHANGED <<global_rc, thr_rtr_ctx>>
    /\ UNCHANGED <<ptr, local_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds>>

\* @c11_action CASCleanup(t):
\* // Release m_ref's old ownership after successful CAS
\* if (pref)
\* atomic_fetch_sub_explicit(&pref->refcnt, 1,
\* memory_order_acq_rel);        // decAndTest
\* return true;
\* Source: atomic_smart_ptr.h:603-606 → cds_test_cas.c:176-177
CASCleanup(t) ==
    /\ pc[t] = "cas_cleanup"
    /\ thr_op[t] = "cas"
    /\ IF thr_pref[t] /= NULL
       THEN /\ global_rc' = [global_rc EXCEPT ![thr_pref[t]] = @ - 1]
            /\ IF global_rc[thr_pref[t]] = 1
               THEN freed' = [freed EXCEPT ![thr_pref[t]] = TRUE]
               ELSE freed' = freed
       ELSE UNCHANGED <<global_rc, freed>>
    \* Caller's local_shared_ptr (newObj) is NOT released here;
    \* it remains in thr_holds and is released later via Reset (scope destruction).
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ thr_op' = [thr_op EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<ptr, local_rc, thr_pref, thr_rcnt, thr_old, thr_new,
                   thr_holds, thr_rtr_ctx>>

(* ========================================================================== *)
(* local_shared_ptr::swap(atomic_shared_ptr&)                                 *)
(* Lines 628-649                                                              *)
(*                                                                            *)
(* Simpler than compareAndSwap_: unconditional exchange, no PreInc/Cleanup.   *)
(* The local hold becomes the installed ref, and vice versa.                  *)
(*   for(;;) {                                                                *)
(*     acquire_tag_ref_()                                                     *)
(*     transfer local_rc to global_rc                                         *)
(*     CAS m_ref (old -> this->m_ref with rcnt=0)                             *)
(*     if CAS fails { undo transfer; release_tag_ref_; continue }             *)
(*     break                                                                  *)
(*   }                                                                        *)
(*   this->m_ref = pref;  // take old pointer                                 *)
(* ========================================================================== *)

\* @c11_action StartSwap(t):
\* // Entry: caller holds local_shared_ptr lp (with newObj).
\* // No PreInc needed — swap exchanges ownership, doesn't add a new ref.
\* // Proceeds directly to cas_acquire (skip cas_pre_inc).
\* Source: atomic_smart_ptr.h:628-630
StartSwap(t) ==
    /\ EnableSwap
    /\ pc[t] = "idle"
    /\ \E newObj \in Objects \cup {NULL} :
       /\ (newObj /= NULL => (freed[newObj] = FALSE /\ thr_holds[t][newObj] > 0))
       /\ thr_new' = [thr_new EXCEPT ![t] = newObj]
    /\ pc' = [pc EXCEPT ![t] = "cas_acquire"]  \* skip PreInc
    /\ thr_op' = [thr_op EXCEPT ![t] = "swap"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_pref, thr_rcnt,
                   thr_old, thr_holds, thr_rtr_ctx>>

ReturnToIdle(t) ==
    /\ pc[t] = "done"
    /\ thr_op[t] /= "cas"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ thr_op' = [thr_op EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_rtr_ctx>>

(* ========================================================================== *)
(* Next-state relation                                                        *)
(* ========================================================================== *)
Next ==
    \E t \in Threads :
        \/ StartLoadShared(t)
        \/ AcquireTagRefRead(t)
        \/ AcquireTagRefCAS(t)
        \/ AcquireTagRefNull(t)
        \/ LoadSharedIncGlobal(t)
        \/ LoadSharedStartRelease(t)
        \/ ReleaseTagRefRead(t)
        \/ ReleaseTagRefCAS(t)
        \/ ReleaseTagRefGlobal(t)
        \/ Reset(t)
        \/ Recycle(t)
        \/ StartCAS(t)
        \/ CASPreInc(t)
        \/ CASReserve(t)
        \/ CASCheck(t)
        \/ CASFailDone(t)
        \/ CASTransfer(t)
        \/ CASSwap(t)
        \/ CASUndo(t)
        \/ CASCleanup(t)
        \/ StartSwap(t)
        \/ ReturnToIdle(t)

(* Natural bounds — verified as invariants, no StateConstraint needed *)
(* local_rc <= N: at most N threads can each increment local_rc by 1 *)
LocalRCBounded ==
    local_rc <= Cardinality(Threads)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* ========================================================================== *)
(* Safety Properties                                                          *)
(* ========================================================================== *)

(* 1. Memory Safety: after acquire_tag_ref_ succeeds (local_rc incremented),     *)
(*    the object is not freed until release_tag_ref_ completes.                    *)
(*    atr_read/atr_cas are BEFORE reservation — only a stale pointer, never  *)
(*    dereferenced; the CAS operates on the atomic word.                      *)
MemorySafety ==
    \A t \in Threads :
        (pc[t] \in {"ls_inc", "ls_release",
                     "rtr_read", "rtr_cas", "rtr_global",
                     "cas_check", "cas_transfer", "cas_swap",
                     "cas_undo", "cas_cleanup"} /\ thr_pref[t] /= NULL)
        => freed[thr_pref[t]] = FALSE

(* 2. No use-after-free: objects held by local_shared_ptrs are not freed *)
NoUseAfterFree ==
    \A t \in Threads : \A o \in Objects :
        thr_holds[t][o] > 0 => freed[o] = FALSE

(* 3. Global refcount non-negative when not freed *)
GlobalRCNonNeg ==
    \A o \in Objects : freed[o] = FALSE => global_rc[o] >= 0

(* 4. Freed objects have global_rc = 0 *)
FreedImpliesZeroRC ==
    \A o \in Objects : freed[o] = TRUE => global_rc[o] = 0

(* 5. The currently installed object (ptr) is not freed *)
InstalledNotFreed ==
    ptr /= NULL => freed[ptr] = FALSE

(* 6. ABA safety: if a CAS at cas_swap succeeds, the object identity is      *)
(*    consistent — the pref that was checked at cas_check is the same         *)
(*    "logical" object (not a recycled imposter). Since we use the same       *)
(*    Object IDs for recycled objects, this is verified indirectly by          *)
(*    MemorySafety: a recycled object that gets installed into ptr won't      *)
(*    have the same local_rc state, so the CAS at rs_cas or cas_swap          *)
(*    would fail if the object was recycled between reads.                    *)
(*    The invariants above capture this: if any invariant is violated when    *)
(*    EnableRecycle=TRUE, it demonstrates an ABA vulnerability.               *)

=============================================================================
