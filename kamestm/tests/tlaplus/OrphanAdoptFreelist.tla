----------------------- MODULE OrphanAdoptFreelist ------------------------
(***************************************************************************)
(* The pool allocator's per-chunk slot custody, at bitmap granularity: what     *)
(* may hold a slot, and the invariant that the bitmap never reports storage as  *)
(* available while something still holds it.                                    *)
(*                                                                             *)
(* Four custodies, which is the whole mechanism:                                *)
(*   bits  -- the m_flags bitmap.  A SET bit means "not available".              *)
(*   occ   -- slots holding a live object (the truth the bitmap must cover).     *)
(*   mask  -- the word-cache's claimed-but-undistributed bits, parked in        *)
(*            m_freelist_head[1] with its base in [2].                          *)
(*   flist -- m_freelist_head[0], the owner-side freelist.  An owner free puts  *)
(*            the slot HERE and leaves its bit SET ("slots returned to the      *)
(*            OWNER stay in m_freelist_head[0], not the bitmap"), and           *)
(*            freelist_pop hands it back without consulting the bitmap.         *)
(*   pending[t] -- a non-owner free, held in that thread's cross-dealloc batch; *)
(*            the bit stays SET until the flush applies it.                     *)
(*                                                                             *)
(* A word-grab claims a whole word at once (bits := ALL) and hands out one      *)
(* slot, parking the rest in mask, so `bits` legitimately exceeds `occ`; the    *)
(* invariants run the other way (occ / flist / mask / pending all subset bits). *)
(*                                                                             *)
(* Thread exit is TWO ordered phases, because in the code it is two objects'    *)
(* destructors: ExitBegin models ~CrossDeallocBatch (flush this thread's        *)
(* pending frees) and ExitDrain models release_dll_chunks_for_thread (drain the *)
(* mask and the freelist, then disown).  A thread that has exited is added to   *)
(* `dead` and never acts again -- without that, nothing is ever permanently     *)
(* lost and no "forgot to flush at exit" defect is detectable.                  *)
(*                                                                             *)
(* Bug knobs, and what each is expected to do (Slots=3, Threads=2):             *)
(*   none                    1185 states, no error                              *)
(*   BUG_NO_BATCH_AT_EXIT    violates NoLostEntries                             *)
(*   BUG_DRAIN_KEEPS_CELLS   violates MaskCoveredByBits                          *)
(*   BUG_WRONG_BIT           violates OccCoveredByBits                           *)
(*   BUG_NO_DRAIN            no error -- CORRECT: a non-drained mask is          *)
(*   BUG_NO_FLIST_DRAIN      inherited by whoever adopts the chunk, so it is     *)
(*                           not lost.  Under a no-adoption build they would be  *)
(*                           genuine leaks.                                     *)
(* Slots=4, Threads=3 also passes exhaustively (37639 states, depth 16).         *)
(*                                                                             *)
(* Note on the invariants: an "X subset bits" family cannot detect a failure to  *)
(* CLEAR, which is why BitsAccounted (exact accounting) and NoLostEntries are    *)
(* here -- three of the five knobs are invisible without them.                  *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Slots,
    Threads,
    BUG_NO_DRAIN,           \* exit forgets to drain the word-cache mask
    BUG_NO_FLIST_DRAIN,     \* exit forgets to drain the owner freelist
    BUG_NO_BATCH_AT_EXIT,   \* ~CrossDeallocBatch does not flush at exit
    BUG_DRAIN_KEEPS_CELLS,  \* the drain returns the bits but leaves the cells
    BUG_WRONG_BIT

VARIABLES bits, occ, mask, flist, owner, onChain, handed, pending, pc, dead

vars == <<bits, occ, mask, flist, owner, onChain, handed, pending, pc, dead>>

NONE == "none"

TypeOK ==
    /\ bits \in SUBSET Slots
    /\ occ \in SUBSET Slots
    /\ mask \in SUBSET Slots
    /\ flist \in SUBSET Slots
    /\ owner \in Threads \cup {NONE}
    /\ onChain \in BOOLEAN
    /\ handed \in SUBSET Slots
    /\ pending \in [Threads -> SUBSET Slots]
    /\ pc \in [Threads -> {"idle", "exiting"}]
    /\ dead \in SUBSET Threads

Init ==
    /\ bits = {} /\ occ = {} /\ mask = {} /\ flist = {}
    /\ owner = NONE /\ onChain = FALSE /\ handed = {}
    /\ pending = [t \in Threads |-> {}]
    /\ pc = [t \in Threads |-> "idle"]
    /\ dead = {}

Claim(t) ==
    /\ t \notin dead /\ pc[t] = "idle" /\ owner = NONE /\ ~onChain
    /\ owner' = t
    /\ UNCHANGED <<bits, occ, mask, flist, onChain, handed, pending, pc, dead>>

(********************************* allocation *******************************)
\* freelist_pop: hands out WITHOUT consulting the bitmap.  The bit is already
\* set (it was never cleared when the owner freed the slot).
AllocFromFlist(t) ==
    /\ t \notin dead /\ pc[t] = "idle" /\ owner = t /\ flist /= {}
    /\ \E s \in flist :
        /\ handed' = IF s \in occ THEN handed \cup {s} ELSE handed
        /\ occ' = occ \cup {s}
        /\ flist' = flist \ {s}
    /\ UNCHANGED <<bits, mask, owner, onChain, pending, pc, dead>>

AllocFromMask(t) ==
    /\ t \notin dead /\ pc[t] = "idle" /\ owner = t /\ mask /= {}
    /\ \E s \in mask :
        /\ handed' = IF s \in occ THEN handed \cup {s} ELSE handed
        /\ occ' = occ \cup {s}
        /\ mask' = mask \ {s}
    /\ UNCHANGED <<bits, flist, owner, onChain, pending, pc, dead>>

WordGrab(t) ==
    /\ t \notin dead /\ pc[t] = "idle" /\ owner = t
    /\ mask = {} /\ flist = {} /\ bits /= Slots
    /\ LET free == Slots \ bits IN
       /\ \E s \in free :
           /\ handed' = IF s \in occ THEN handed \cup {s} ELSE handed
           /\ occ' = occ \cup {s}
           /\ mask' = free \ {s}
       /\ bits' = Slots
    /\ UNCHANGED <<flist, owner, onChain, pending, pc, dead>>

(************************************ frees *********************************)
\* OWNER free -> chunk-local freelist, bit LEFT SET.
FreeOwner(t) ==
    /\ t \notin dead /\ pc[t] = "idle" /\ owner = t
    /\ \E s \in occ :
        /\ occ' = occ \ {s}
        /\ flist' = flist \cup {s}
    /\ UNCHANGED <<bits, mask, owner, onChain, handed, pending, pc, dead>>

\* NON-OWNER free -> cross-dealloc batch, bit left set until the flush.
FreeBatch(t) ==
    /\ t \notin dead /\ pc[t] = "idle" /\ owner /= t
    /\ \E s \in occ :
        /\ occ' = occ \ {s}
        /\ pending' = [pending EXCEPT ![t] = @ \cup {s}]
    /\ UNCHANGED <<bits, mask, flist, owner, onChain, handed, pc, dead>>

FlushBatch(t) ==
    /\ t \notin dead /\ pc[t] \in {"idle", "exiting"} /\ pending[t] /= {}
    /\ IF BUG_WRONG_BIT
         THEN \E w \in bits : bits' = bits \ {w}
         ELSE bits' = bits \ pending[t]
    /\ pending' = [pending EXCEPT ![t] = {}]
    /\ UNCHANGED <<occ, mask, flist, owner, onChain, handed, pc, dead>>

(**************************** thread exit, in two phases ********************)
\* Phase 1: ~CrossDeallocBatch.  Runs BEFORE the DLL walk, and a peer may act
\* in between -- which one atomic exit step cannot express.
ExitBegin(t) ==
    /\ t \notin dead /\ pc[t] = "idle"
    /\ pc' = [pc EXCEPT ![t] = "exiting"]
    /\ IF BUG_NO_BATCH_AT_EXIT
         THEN /\ bits' = bits /\ pending' = pending
         ELSE /\ bits' = bits \ pending[t]
              /\ pending' = [pending EXCEPT ![t] = {}]
    /\ UNCHANGED <<occ, mask, flist, owner, onChain, handed, dead>>

\* Phase 2: release_dll_chunks_for_thread -- drain mask AND freelist, then
\* disown / chain.  Only the owner has a chunk to walk.
ExitDrain(t) ==
    /\ t \notin dead /\ pc[t] = "exiting"
    /\ IF owner = t
         THEN LET dm == IF BUG_NO_DRAIN THEN {} ELSE mask
                  df == IF BUG_NO_FLIST_DRAIN THEN {} ELSE flist
              IN /\ bits' = bits \ (dm \cup df)
                 /\ mask' = IF BUG_DRAIN_KEEPS_CELLS THEN mask ELSE mask \ dm
                 /\ flist' = IF BUG_DRAIN_KEEPS_CELLS THEN flist ELSE flist \ df
                 /\ owner' = NONE
                 /\ onChain' = (occ /= {} \/ (bits \ (dm \cup df)) /= {})
         ELSE /\ UNCHANGED <<bits, mask, flist, owner, onChain>>
    \* The thread is now PERMANENTLY gone -- its `CrossDeallocBatch` has
    \* been destroyed.  The earlier model returned it to "idle", i.e. its threads
    \* were immortal, which is why no "forgot to flush at exit" knob could be
    \* detected: nothing was ever lost for good.  's deterministic
    \* `excess = -801` (one batched entry lost per thread exit) is exactly that
    \* class, so it was outside the model.
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ dead' = dead \cup {t}
    /\ UNCHANGED <<occ, handed, pending>>

Adopt(t) ==
    /\ t \notin dead /\ pc[t] = "idle" /\ onChain /\ owner = NONE
    /\ owner' = t
    /\ onChain' = FALSE
    /\ UNCHANGED <<bits, occ, mask, flist, handed, pending, pc, dead>>

Next ==
    \/ \E t \in Threads : Claim(t) \/ AllocFromFlist(t) \/ AllocFromMask(t)
                       \/ WordGrab(t) \/ FreeOwner(t) \/ FreeBatch(t)
                       \/ FlushBatch(t) \/ ExitBegin(t) \/ ExitDrain(t)
                       \/ Adopt(t)

Spec == Init /\ [][Next]_vars

(********************************* properties *******************************)
\* The bitmap never says "available" about storage in use.
OccCoveredByBits == occ \subseteq bits
\* A freelist slot must keep its bit SET -- otherwise the bitmap can hand the
\* same slot out again while the freelist still holds it.
FlistCoveredByBits == flist \subseteq bits
\* A slot pending in some batch must keep its bit SET until that batch applies.
PendingCoveredByBits == \A t \in Threads : pending[t] \subseteq bits
\* No slot is in two custodies at once.
NoDoubleCustody ==
    /\ mask \cap occ = {}
    /\ flist \cap occ = {}
    /\ flist \cap mask = {}
    /\ \A t \in Threads : pending[t] \cap occ = {}
\* A parked mask cell must keep its bit SET, or the bitmap and the mask both
\* offer the same slot -- the BUG_DRAIN_KEEPS_CELLS shape.
MaskCoveredByBits == mask \subseteq bits
\* EXACT accounting: every set bit is in exactly one custody, and every custody
\* keeps its bit set.  This is the spec-level form of the C++ conservation
\* identity ('s `checked_flush - pushes`), and it is what the three
\* "forgot to drain" knobs actually violate -- the earlier invariants were all
\* of the form X \subseteq bits, which a failure to CLEAR can never break, so
\* those knobs passed vacuously (, applied to a spec rather than a probe).
BitsAccounted ==
    bits = occ \cup mask \cup flist \cup UNION {pending[t] : t \in Threads}
\* A dead thread's batch must be empty: anything still in it can never be
\* applied, so its bits stay SET for ever and the slots are lost.
NoLostEntries == \A t \in dead : pending[t] = {}
NoDoubleHandOut == handed = {}

\* Safety only -- what a use-after-free needs.
InvSafe == TypeOK /\ OccCoveredByBits /\ FlistCoveredByBits /\ MaskCoveredByBits
           /\ PendingCoveredByBits /\ NoDoubleCustody /\ NoDoubleHandOut
\* Safety + exact accounting (adds leak detection).
Inv == InvSafe /\ BitsAccounted /\ NoLostEntries
===========================================================================
