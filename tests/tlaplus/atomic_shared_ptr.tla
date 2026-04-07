--------------------------- MODULE atomic_shared_ptr ---------------------------
(*
 * TLA+ model of kame/atomic_smart_ptr.h core protocol:
 *   reserve_scan_(), scan_(), leave_scan_(), compareAndSwap_(), local_shared_ptr::reset()
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
 *   - Non-atomic reserve_scan_ read: pref_() and refcnt_() are two separate
 *     loads of m_ref in the C++ code; modeled as two interleaving steps.
 *   - Symmetry support for state space reduction.
 *
 * Source: kame/atomic_smart_ptr.h lines 420-613
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Threads,          \* set of thread IDs
    Objects,          \* set of possible object IDs (Ref objects)
    MaxLocalRC,       \* max local refcount (= ATOMIC_SHARED_REF_ALIGNMENT - 1)
    MaxGlobalRC,      \* bound on global_rc for state space control
    EnableCAS,        \* TRUE to enable compareAndSwap_ operations
    EnableRecycle     \* TRUE to enable ABA recycling of freed objects

ASSUME MaxLocalRC >= 2
ASSUME Cardinality(Objects) >= 2

NULL == CHOOSE x : x \notin Objects

(* Symmetry sets — declared here, used in .cfg via SYMMETRY *)
ThreadSymmetry == Permutations(Threads)
ObjectSymmetry == Permutations(Objects)

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
    thr_op,         \* what operation thread is performing: "scan", "cas", "idle"
    thr_pref,       \* pointer read by reserve_scan_ (thread-local)
    thr_rcnt,       \* local refcount value returned by reserve_scan_
    thr_old,        \* for CAS: the expected old object
    thr_new,        \* for CAS: the new object to install
    thr_holds,      \* set of objects "held" by each thread (via local_shared_ptr)
    thr_ls_ctx      \* leave_scan_ return target: "scan_done", "cas_retry", "cas_fail"

vars == <<ptr, local_rc, global_rc, freed, pc, thr_op, thr_pref, thr_rcnt,
          thr_old, thr_new, thr_holds, thr_ls_ctx>>

(* -------------------------------------------------------------------------- *)
(* Type invariant                                                             *)
(* -------------------------------------------------------------------------- *)
TypeOK ==
    /\ ptr \in Objects \cup {NULL}
    /\ local_rc \in 0..MaxLocalRC
    /\ global_rc \in [Objects -> Nat]
    /\ freed \in [Objects -> BOOLEAN]
    /\ pc \in [Threads -> {"idle",
                           "rs_read_ptr", \* reserve_scan_: 1st load — read ptr
                           "rs_read_rc",  \* reserve_scan_: 2nd load — read local_rc
                           "rs_cas",      \* reserve_scan_: CAS to increment local_rc
                           "scan_inc",    \* scan_: increment global_rc
                           "scan_leave",  \* scan_: call leave_scan_
                           "ls_read",     \* leave_scan_: read refcount
                           "ls_cas",      \* leave_scan_: CAS to decrement local_rc
                           "ls_global",   \* leave_scan_: fallback - dec global_rc
                           "done",        \* operation completed
                           "cas_pre_inc",   \* CAS: pre-increment newr's global_rc
                           "cas_reserve",   \* CAS: call reserve_scan_
                           "cas_check",     \* CAS: check if pref == oldr
                           "cas_fail_done", \* CAS mismatch: rollback newr, return false
                           "cas_transfer",  \* CAS: transfer local_rc to global_rc
                           "cas_swap",      \* CAS: the actual CAS on the atomic word
                           "cas_undo",      \* CAS: undo transfer on CAS failure
                           "cas_cleanup"    \* CAS: decrement pref's global_rc on success
                          }]
    /\ thr_op \in [Threads -> {"scan", "cas", "idle"}]
    /\ thr_pref \in [Threads -> Objects \cup {NULL}]
    /\ thr_rcnt \in [Threads -> 0..MaxLocalRC]
    /\ thr_old \in [Threads -> Objects \cup {NULL}]
    /\ thr_new \in [Threads -> Objects \cup {NULL}]
    /\ \A t \in Threads : thr_holds[t] \subseteq Objects
    /\ thr_ls_ctx \in [Threads -> {"scan_done", "cas_retry", "cas_fail", "none"}]

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
                      IF t = firstThread /\ otherObj /= NULL
                      THEN {otherObj}
                      ELSE {}]
    /\ thr_ls_ctx = [t \in Threads |-> "none"]

(* ========================================================================== *)
(* ABA Recycling: a freed object can be "reallocated" at the same address     *)
(* This simulates the allocator returning the same pointer value, which is    *)
(* the precondition for an ABA problem. The recycled object gets global_rc=1  *)
(* and is given to a thread (simulating new + local_shared_ptr assignment).   *)
(* ========================================================================== *)
Recycle(t) ==
    /\ EnableRecycle
    /\ pc[t] \in {"idle", "done"}
    /\ \E o \in Objects :
       /\ freed[o] = TRUE
       /\ global_rc[o] = 0
       /\ freed' = [freed EXCEPT ![o] = FALSE]
       /\ global_rc' = [global_rc EXCEPT ![o] = 1]
       /\ thr_holds' = [thr_holds EXCEPT ![t] = @ \cup {o}]
    /\ UNCHANGED <<ptr, local_rc, pc, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_ls_ctx>>

(* ========================================================================== *)
(* scan_() operation: reserve_scan_ + increment global_rc + leave_scan_       *)
(* Models lines 494-503                                                       *)
(* ========================================================================== *)

StartScan(t) ==
    /\ pc[t] = "idle"
    /\ pc' = [pc EXCEPT ![t] = "rs_read_ptr"]
    /\ thr_op' = [thr_op EXCEPT ![t] = "scan"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_ls_ctx>>

(* ========================================================================== *)
(* reserve_scan_(): non-atomic two-step read of m_ref                         *)
(* In C++, pref_() and refcnt_() are two separate atomic loads of m_ref.     *)
(* Between the two loads, another thread can change m_ref. The subsequent     *)
(* CAS catches any inconsistency (it compares the full word).                 *)
(* Lines 462-466: pref = pref_(); rcnt_old = refcnt_();                       *)
(* ========================================================================== *)

(* Step 1: first atomic load — extract pointer part *)
ReserveScanReadPtr(t) ==
    /\ pc[t] = "rs_read_ptr"
    /\ thr_pref' = [thr_pref EXCEPT ![t] = ptr]
    /\ pc' = [pc EXCEPT ![t] = "rs_read_rc"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_rcnt, thr_old,
                   thr_new, thr_holds, thr_ls_ctx>>

(* Step 2: second atomic load — extract local refcount *)
(* NOTE: between step 1 and step 2, ptr and/or local_rc may have changed.    *)
(* thr_rcnt may come from a different m_ref value than thr_pref.              *)
(* The CAS at rs_cas will detect and reject this inconsistency.               *)
ReserveScanReadRC(t) ==
    /\ pc[t] = "rs_read_rc"
    /\ thr_rcnt' = [thr_rcnt EXCEPT ![t] = local_rc]
    /\ pc' = [pc EXCEPT ![t] = "rs_cas"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_old,
                   thr_new, thr_holds, thr_ls_ctx>>

(* --- reserve_scan_: CAS to increment local_rc --- *)
(* Lines 485-488: CAS(pref + rcnt_old, pref + rcnt_new) *)
(* This CAS compares the FULL word (ptr + local_rc together), so if either   *)
(* changed since the reads, it fails — even if the reads were inconsistent.   *)
ReserveScanCAS(t) ==
    /\ pc[t] = "rs_cas"
    /\ thr_pref[t] /= NULL
    /\ thr_rcnt[t] + 1 < MaxLocalRC + 1
    /\ \/ (* CAS succeeds: both ptr AND local_rc match *)
          /\ ptr = thr_pref[t]
          /\ local_rc = thr_rcnt[t]
          /\ local_rc' = thr_rcnt[t] + 1
          /\ thr_rcnt' = [thr_rcnt EXCEPT ![t] = thr_rcnt[t] + 1]
          /\ pc' = [pc EXCEPT ![t] = IF thr_op[t] = "scan" THEN "scan_inc"
                                      ELSE "cas_check"]
          /\ UNCHANGED <<ptr, global_rc, freed, thr_op, thr_pref, thr_old,
                         thr_new, thr_holds, thr_ls_ctx>>
       \/ (* CAS fails: retry from first read *)
          /\ ~(ptr = thr_pref[t] /\ local_rc = thr_rcnt[t])
          /\ pc' = [pc EXCEPT ![t] = "rs_read_ptr"]
          /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                         thr_old, thr_new, thr_holds, thr_ls_ctx>>

ReserveScanNull(t) ==
    /\ pc[t] = "rs_cas"
    /\ thr_pref[t] = NULL
    /\ pc' = [pc EXCEPT ![t] = IF thr_op[t] = "scan" THEN "done"
                                ELSE "cas_check"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_ls_ctx>>

(* --- scan_: increment global reference counter --- *)
(* Line 500: pref->refcnt.fetch_add(1) *)
ScanIncGlobal(t) ==
    /\ pc[t] = "scan_inc"
    /\ thr_op[t] = "scan"
    /\ global_rc' = [global_rc EXCEPT ![thr_pref[t]] = @ + 1]
    /\ pc' = [pc EXCEPT ![t] = "scan_leave"]
    /\ UNCHANGED <<ptr, local_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_ls_ctx>>

ScanStartLeave(t) ==
    /\ pc[t] = "scan_leave"
    /\ thr_op[t] = "scan"
    /\ pc' = [pc EXCEPT ![t] = "ls_read"]
    /\ thr_ls_ctx' = [thr_ls_ctx EXCEPT ![t] = "scan_done"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds>>

(* ========================================================================== *)
(* leave_scan_() operation (shared by scan_ and compareAndSwap_)              *)
(* Lines 512-531                                                              *)
(* After completion, control returns based on thr_ls_ctx:                     *)
(*   "scan_done" -> done (scan completed)                                     *)
(*   "cas_retry" -> cas_reserve (inner CAS failed, retry)                     *)
(*   "cas_fail"  -> cas_fail_done (mismatch, return false)                    *)
(* ========================================================================== *)

LeaveScanRead(t) ==
    /\ pc[t] = "ls_read"
    /\ thr_rcnt' = [thr_rcnt EXCEPT ![t] = local_rc]
    /\ pc' = [pc EXCEPT ![t] = IF local_rc > 0 THEN "ls_cas" ELSE "ls_global"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref,
                   thr_old, thr_new, thr_holds, thr_ls_ctx>>

LeaveScanCAS(t) ==
    /\ pc[t] = "ls_cas"
    /\ LET returnPC == CASE thr_ls_ctx[t] = "scan_done" -> "done"
                          [] thr_ls_ctx[t] = "cas_retry" -> "cas_reserve"
                          [] thr_ls_ctx[t] = "cas_fail"  -> "cas_fail_done"
                          [] OTHER -> "done"
       IN
       \/ (* CAS succeeds: ptr unchanged and local_rc matches *)
          /\ ptr = thr_pref[t]
          /\ local_rc = thr_rcnt[t]
          /\ local_rc' = thr_rcnt[t] - 1
          /\ IF thr_ls_ctx[t] = "scan_done"
             THEN thr_holds' = [thr_holds EXCEPT ![t] = @ \cup {thr_pref[t]}]
             ELSE UNCHANGED thr_holds
          /\ pc' = [pc EXCEPT ![t] = returnPC]
          /\ UNCHANGED <<ptr, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                         thr_old, thr_new, thr_ls_ctx>>
       \/ (* CAS fails, ptr still matches -> retry leave_scan_ *)
          /\ ~(ptr = thr_pref[t] /\ local_rc = thr_rcnt[t])
          /\ ptr = thr_pref[t]
          /\ pc' = [pc EXCEPT ![t] = "ls_read"]
          /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                         thr_old, thr_new, thr_holds, thr_ls_ctx>>
       \/ (* CAS fails, ptr changed -> fallback to global dec *)
          /\ ~(ptr = thr_pref[t] /\ local_rc = thr_rcnt[t])
          /\ ptr /= thr_pref[t]
          /\ pc' = [pc EXCEPT ![t] = "ls_global"]
          /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                         thr_old, thr_new, thr_holds, thr_ls_ctx>>

(* --- leave_scan_: fallback - decrement global_rc --- *)
LeaveScanGlobal(t) ==
    /\ pc[t] = "ls_global"
    /\ thr_pref[t] /= NULL
    /\ LET o == thr_pref[t]
           returnPC == CASE thr_ls_ctx[t] = "scan_done" -> "done"
                         [] thr_ls_ctx[t] = "cas_retry" -> "cas_reserve"
                         [] thr_ls_ctx[t] = "cas_fail"  -> "cas_fail_done"
                         [] OTHER -> "done"
       IN
       /\ global_rc' = [global_rc EXCEPT ![o] = @ - 1]
       /\ IF global_rc[o] = 1
          THEN freed' = [freed EXCEPT ![o] = TRUE]
          ELSE freed' = freed
       /\ IF thr_ls_ctx[t] = "scan_done"
          THEN thr_holds' = [thr_holds EXCEPT ![t] = @ \cup {thr_pref[t]}]
          ELSE UNCHANGED thr_holds
       /\ pc' = [pc EXCEPT ![t] = returnPC]
       /\ UNCHANGED <<ptr, local_rc, thr_op, thr_pref, thr_rcnt, thr_old,
                      thr_new, thr_ls_ctx>>

(* ========================================================================== *)
(* local_shared_ptr::reset()                                                  *)
(* ========================================================================== *)
Reset(t) ==
    /\ pc[t] \in {"done", "idle"}
    /\ \E o \in thr_holds[t] :
       /\ thr_holds' = [thr_holds EXCEPT ![t] = @ \ {o}]
       /\ global_rc' = [global_rc EXCEPT ![o] = @ - 1]
       /\ IF global_rc[o] = 1
          THEN freed' = [freed EXCEPT ![o] = TRUE]
          ELSE freed' = freed
    /\ UNCHANGED <<ptr, local_rc, pc, thr_op, thr_pref, thr_rcnt, thr_old,
                   thr_new, thr_ls_ctx>>

(* ========================================================================== *)
(* compareAndSwap_() operation                                                *)
(* Lines 556-603                                                              *)
(*                                                                            *)
(* Control flow:                                                              *)
(*   1. Pre-increment newr's global_rc (once, not on retry)                   *)
(*   2. for(;;) {                                                             *)
(*        reserve_scan_()                                                     *)
(*        if (pref != oldr) { leave_scan; rollback; return false }            *)
(*        transfer local_rc to global_rc                                      *)
(*        CAS m_ref                                                           *)
(*        if CAS fails { undo transfer; leave_scan; continue }               *)
(*        break                                                               *)
(*      }                                                                     *)
(*   3. Cleanup: dec pref's global_rc; return true                            *)
(* ========================================================================== *)

StartCAS(t) ==
    /\ EnableCAS
    /\ pc[t] = "idle"
    /\ \E oldObj \in Objects \cup {NULL}, newObj \in Objects \cup {NULL} :
       /\ oldObj /= newObj
       /\ (newObj /= NULL => (freed[newObj] = FALSE /\ newObj \in thr_holds[t]))
       /\ thr_old' = [thr_old EXCEPT ![t] = oldObj]
       /\ thr_new' = [thr_new EXCEPT ![t] = newObj]
    /\ pc' = [pc EXCEPT ![t] = "cas_pre_inc"]
    /\ thr_op' = [thr_op EXCEPT ![t] = "cas"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_pref, thr_rcnt,
                   thr_holds, thr_ls_ctx>>

CASPreInc(t) ==
    /\ pc[t] = "cas_pre_inc"
    /\ IF thr_new[t] /= NULL
       THEN global_rc' = [global_rc EXCEPT ![thr_new[t]] = @ + 1]
       ELSE UNCHANGED global_rc
    /\ pc' = [pc EXCEPT ![t] = "cas_reserve"]
    /\ UNCHANGED <<ptr, local_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_ls_ctx>>

CASReserve(t) ==
    /\ pc[t] = "cas_reserve"
    /\ pc' = [pc EXCEPT ![t] = "rs_read_ptr"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_ls_ctx>>

CASCheck(t) ==
    /\ pc[t] = "cas_check"
    /\ thr_op[t] = "cas"
    /\ IF thr_pref[t] /= thr_old[t]
       THEN
            IF thr_pref[t] /= NULL
            THEN /\ pc' = [pc EXCEPT ![t] = "ls_read"]
                 /\ thr_ls_ctx' = [thr_ls_ctx EXCEPT ![t] = "cas_fail"]
                 /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref,
                                thr_rcnt, thr_old, thr_new, thr_holds>>
            ELSE /\ pc' = [pc EXCEPT ![t] = "cas_fail_done"]
                 /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref,
                                thr_rcnt, thr_old, thr_new, thr_holds, thr_ls_ctx>>
       ELSE
            /\ pc' = [pc EXCEPT ![t] = "cas_transfer"]
            /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                           thr_old, thr_new, thr_holds, thr_ls_ctx>>

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
                   thr_holds, thr_ls_ctx>>

CASTransfer(t) ==
    /\ pc[t] = "cas_transfer"
    /\ thr_op[t] = "cas"
    /\ IF thr_pref[t] /= NULL /\ thr_rcnt[t] /= 1
       THEN global_rc' = [global_rc EXCEPT ![thr_pref[t]] = @ + thr_rcnt[t] - 1]
       ELSE UNCHANGED global_rc
    /\ pc' = [pc EXCEPT ![t] = "cas_swap"]
    /\ UNCHANGED <<ptr, local_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_ls_ctx>>

CASSwap(t) ==
    /\ pc[t] = "cas_swap"
    /\ thr_op[t] = "cas"
    /\ \/ (* CAS succeeds *)
          /\ ptr = thr_pref[t]
          /\ local_rc = thr_rcnt[t]
          /\ ptr' = thr_new[t]
          /\ local_rc' = 0
          /\ pc' = [pc EXCEPT ![t] = "cas_cleanup"]
          /\ UNCHANGED <<global_rc, freed, thr_op, thr_pref, thr_rcnt,
                         thr_old, thr_new, thr_holds, thr_ls_ctx>>
       \/ (* CAS fails *)
          /\ ~(ptr = thr_pref[t] /\ local_rc = thr_rcnt[t])
          /\ pc' = [pc EXCEPT ![t] = "cas_undo"]
          /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_op, thr_pref, thr_rcnt,
                         thr_old, thr_new, thr_holds, thr_ls_ctx>>

CASUndo(t) ==
    /\ pc[t] = "cas_undo"
    /\ thr_op[t] = "cas"
    /\ IF thr_pref[t] /= NULL
       THEN /\ IF thr_rcnt[t] /= 1
               THEN global_rc' = [global_rc EXCEPT ![thr_pref[t]] = @ - (thr_rcnt[t] - 1)]
               ELSE UNCHANGED global_rc
            /\ pc' = [pc EXCEPT ![t] = "ls_read"]
            /\ thr_ls_ctx' = [thr_ls_ctx EXCEPT ![t] = "cas_retry"]
       ELSE /\ pc' = [pc EXCEPT ![t] = "cas_reserve"]
            /\ UNCHANGED <<global_rc, thr_ls_ctx>>
    /\ UNCHANGED <<ptr, local_rc, freed, thr_op, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds>>

CASCleanup(t) ==
    /\ pc[t] = "cas_cleanup"
    /\ thr_op[t] = "cas"
    /\ IF thr_pref[t] /= NULL
       THEN /\ global_rc' = [global_rc EXCEPT ![thr_pref[t]] = @ - 1]
            /\ IF global_rc[thr_pref[t]] = 1
               THEN freed' = [freed EXCEPT ![thr_pref[t]] = TRUE]
               ELSE freed' = freed
       ELSE UNCHANGED <<global_rc, freed>>
    /\ IF thr_new[t] /= NULL
       THEN thr_holds' = [thr_holds EXCEPT ![t] = @ \ {thr_new[t]}]
       ELSE UNCHANGED thr_holds
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ thr_op' = [thr_op EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<ptr, local_rc, thr_pref, thr_rcnt, thr_old, thr_new, thr_ls_ctx>>

ReturnToIdle(t) ==
    /\ pc[t] = "done"
    /\ thr_op[t] /= "cas"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ thr_op' = [thr_op EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<ptr, local_rc, global_rc, freed, thr_pref, thr_rcnt,
                   thr_old, thr_new, thr_holds, thr_ls_ctx>>

(* ========================================================================== *)
(* Next-state relation                                                        *)
(* ========================================================================== *)
Next ==
    \E t \in Threads :
        \/ StartScan(t)
        \/ ReserveScanReadPtr(t)
        \/ ReserveScanReadRC(t)
        \/ ReserveScanCAS(t)
        \/ ReserveScanNull(t)
        \/ ScanIncGlobal(t)
        \/ ScanStartLeave(t)
        \/ LeaveScanRead(t)
        \/ LeaveScanCAS(t)
        \/ LeaveScanGlobal(t)
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
        \/ ReturnToIdle(t)

(* State constraint to bound state space *)
StateConstraint == \A o \in Objects : global_rc[o] <= MaxGlobalRC

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* ========================================================================== *)
(* Safety Properties                                                          *)
(* ========================================================================== *)

(* 1. Memory Safety: after reserve_scan_ succeeds (local_rc incremented),     *)
(*    the object is not freed until leave_scan_ completes.                    *)
(*    rs_read_ptr/rs_read_rc/rs_cas are BEFORE reservation — only a stale    *)
(*    pointer, never dereferenced; the CAS operates on the atomic word.       *)
MemorySafety ==
    \A t \in Threads :
        (pc[t] \in {"scan_inc", "scan_leave",
                     "ls_read", "ls_cas", "ls_global",
                     "cas_check", "cas_transfer", "cas_swap",
                     "cas_undo", "cas_cleanup"} /\ thr_pref[t] /= NULL)
        => freed[thr_pref[t]] = FALSE

(* 2. No use-after-free: objects held by local_shared_ptrs are not freed *)
NoUseAfterFree ==
    \A t \in Threads : \A o \in thr_holds[t] : freed[o] = FALSE

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
