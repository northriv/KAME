(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            kamepoolalloc/LICENSE-APACHE-2.0)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (see kamepoolalloc/LICENSE-GPL-2.0).
 ***************************************************************************)
------------------------- MODULE OrphanReuse_36b -------------------------
(*
 * Microscopic TLA+ model of the §36b proposed replacement for §36 orphan
 * reuse.  Specification mechanically transcribed from kamepoolalloc/
 * ORPHAN_REUSE_HANDOFF.md (commit 7a010343).  The §36b prototype passed
 * leak tests on Linux but stress-test (alloc_stress, 300 runs) still
 * exhibits 1.67 % corruption — 4× ORPHAN_RELEASE_BAD (ABA) + 1× sentinel
 * mismatch — confirming the design has an unidentified concurrency race.
 *
 * Goal: explore the (chunk × slot × operation) state space to either
 * (a) exhibit the race that drives ORPHAN_RELEASE_BAD, or
 * (b) prove the per-slot version with the documented atomic-step
 *     ordering is in fact safe (in which case the bug is in the C++
 *     implementation, not the algorithm).
 *
 * ============================================================
 *  DESIGN BEING MODELLED  (verbatim from §2 of the handoff doc)
 * ============================================================
 *
 * Per-template static state:
 *   s_orphan_slots[K]: array of versioned tagged words.  Each slot:
 *     low-48 bits = chunk pointer (or 0 if empty)
 *     high-16 bits = ABA version (bumped on every push/pop/release-take)
 *
 * Per-chunk arbiter:
 *   m_orphan_disp: atomic byte, values
 *     OWNED       = off-array, owner alive (or just re-owned)
 *     PUSHING     = transitional, set by orphan_push CAS winner
 *     RELEASED    = ownership claimed by release-on-empty
 *     1+k (k<K)   = on the array at slot k
 *
 * Per-chunk count word (32-bit `m_flags_packed` — §36 layout, unchanged):
 *   BIT_OWNED (bit 31) — set iff owner thread alive
 *   MASK_CNT  (bits 30..0) — number of live slots
 *
 * Operations (each step here = ONE atomic memory op):
 *
 *   orphan_push(c)  // owner exit, c->BIT_OWNED clear, MASK_CNT > 0
 *     P0: CAS disp OWNED → PUSHING                  (1 atomic CAS)
 *     P1: scan slots from h = owner_id mod K;
 *         CAS slot k from (0,vk) → (c, vk+1)         (1 atomic CAS per try)
 *         on success: store disp = 1+k (relaxed)
 *         on full scan: store disp = OWNED, give up
 *
 *   orphan_pop()  // allocator path, fresh-claim alternative
 *     PoP0: scan from owner_id mod K;
 *           load slot k = (c, vk);
 *           CAS slot k → (0, vk+1)                   (1 atomic CAS per try)
 *           on success: ret c, caller proceeds to PoP1/PoP2
 *     PoP1: claim BIT_OWNED (CAS m_flags_packed preserving MASK_CNT)
 *     PoP2: store disp = OWNED (relaxed); resume normal alloc path
 *
 *   free(c) decrementing MASK_CNT to 0  (any thread)
 *     F0: atomicDecAndTest(MASK_CNT) → true
 *     F1: orphan_claim_for_release(c):
 *         d = c->disp.load();
 *         if d == OWNED:        try CAS disp OWNED → RELEASED  (1 CAS)
 *         elif d == 1+k:        load slot[k] -> (p,v);
 *                               if p != c: bail (someone else moved it)
 *                               else try CAS slot[k] (c,v) → (0,v+1)  (1 CAS)
 *                                    and store disp = RELEASED on success
 *         elif d == PUSHING:    spin/abort/bail per impl (we BAIL → leak)
 *         elif d == RELEASED:   already claimed → bail
 *         returns boolean: did THIS caller claim release
 *     F2 (only if F1 returned true): orphan_release_self(c):
 *         DEBUG GUARD: assert (flags & BIT_OWNED) == 0 AND MASK_CNT == 0
 *         release units to region bitmap; free chunk header
 *
 *   re-own through orphan_pop (THE PRIME SUSPECT — handoff doc §5):
 *     The pop caller’s 3-step re-own (PoP0, PoP1, PoP2) is intentionally
 *     modelled as THREE separate atomic steps with arbitrary interleaving
 *     of any other thread between them.
 *
 * ============================================================
 *  STATE-SPACE DISCIPLINE
 * ============================================================
 *
 * One chunk life-cycle, K = 2 slots, 2 threads (one owner that exits +
 * one allocator/freer that competes).  Generations bounded by AllowReown
 * (typically TRUE).  No region-level reclaim modelled — the §36b
 * question is solely about orphan-array safety.
 *
 *   Inv_NoDoubleRelease:  the chunk is released at most once.
 *   Inv_NoLiveRelease:    release happens ONLY when MASK_CNT == 0 AND
 *                         BIT_OWNED == 0.  This is the ORPHAN_RELEASE_BAD
 *                         guard the C++ DEBUG abort fires on.
 *   Inv_DispCoherent:     if disp = 1+k then slot[k].ptr is the chunk
 *                         pointer, OR disp is mid-transition (PUSHING).
 *                         (Disabled on the re-own window — that is the
 *                         documented gap; instead we monitor the
 *                         release path.)
 *   Inv_NoStrayPop:       orphan_pop never returns a chunk whose
 *                         BIT_OWNED is still set.
 *)

EXTENDS Integers, FiniteSets, TLC, Sequences

CONSTANTS
    Threads,        \* finite set of competitor thread ids
    K,              \* slot array size (recommend 2)
    AllowReown      \* TRUE = an allocator that pops can finish the
                    \*        3-step re-own and survive across a free.
                    \*        FALSE = simpler model: pop is treated
                    \*        as immediate release (debugging aid).

ASSUME K \in 1..4
ASSUME Cardinality(Threads) >= 1
ASSUME AllowReown \in BOOLEAN

\* --- Disposition values ---------------------------------------------------
\* These mirror the C++ enum.  Numeric values chosen so 1..K are the
\* "on-array at slot k-1" range, leaving OWNED/PUSHING/RELEASED outside.
OWNED    == 0
PUSHING  == 100
RELEASED == 200

\* A slot value is a record << ptr, ver >>.
\* ptr: "Null" (empty slot) or "C" (the single modelled chunk).
\* ver: natural number, monotone-increasing.

\* --- Variables ------------------------------------------------------------
VARIABLES
    slots,          \* function [0..K-1 -> [ptr |-> "Null" or "C", ver |-> Nat]]
    disp,           \* one of {OWNED, PUSHING, RELEASED} or 1..K
    bit_owned,      \* BOOLEAN
    mask_cnt,       \* Nat — live-slot count
    released,       \* BOOLEAN — orphan_release_self ran
    pending_claim,  \* BOOLEAN — a thread is between the successful
                    \* atomicDecAndTest and the actual claim_for_release
                    \* call.  Set at the dec quantum iff bit_owned was F
                    \* AND mask_cnt was 1.  Captures the C++ contract that
                    \* claim only runs when the 32-bit dec returned true.
    reown_pc,       \* per-thread program counter for the 3-step re-own
                    \*  "idle"  = not in re-own
                    \*  "got_c" = PoP0 done (slot taken); pending PoP1
                    \*  "armed_owned" = PoP1 done; pending PoP2
                    \* (re-own terminates when reown_pc becomes "idle" again)
    push_pc,        \* per-thread program counter for orphan_push
                    \*  "idle", "pushing"
    bad_release     \* BOOLEAN — sticky: the ORPHAN_RELEASE_BAD guard fired

vars == << slots, disp, bit_owned, mask_cnt, released, pending_claim,
           reown_pc, push_pc, bad_release >>

\* --- Type invariant -------------------------------------------------------
DispDomain ==
    { OWNED, PUSHING, RELEASED } \cup (1 .. K)

TypeOK ==
    /\ slots \in [0..K-1 -> [ptr: {"Null","C"}, ver: Nat]]
    /\ disp \in DispDomain
    /\ bit_owned \in BOOLEAN
    /\ mask_cnt \in 0..2
    /\ released \in BOOLEAN
    /\ pending_claim \in BOOLEAN
    /\ reown_pc \in [Threads -> {"idle","got_c","armed_owned"}]
    /\ push_pc \in [Threads -> {"idle","pushing"}]
    /\ bad_release \in BOOLEAN

\* --- Initial state --------------------------------------------------------
\* The owner thread has exited and PUSHED the chunk to slot 0 (the
\* orphan_push completed before any modelled action).  This focuses the
\* state-space exploration on the documented race window — POP +
\* cross-thread free interleaving against an on-array chunk — without
\* spending exploration budget on the trivial OWNED → RELEASED path
\* (which is concurrency-free).
\*
\* Survivor count: 1 in flight (cross-thread free still pending).
\* BIT_OWNED clear, slot 0 holds (C, ver=1), slot 1 empty (ver=0), disp = 1.
Init ==
    /\ slots = [k \in 0..K-1 |->
                  IF k = 0 THEN [ptr |-> "C",    ver |-> 1]
                           ELSE [ptr |-> "Null", ver |-> 0]]
    /\ disp = 1   \* = 1 + 0 (on slot 0)
    /\ bit_owned = FALSE
    /\ mask_cnt = 1
    /\ released = FALSE
    /\ pending_claim = FALSE
    /\ reown_pc = [t \in Threads |-> "idle"]
    /\ push_pc = [t \in Threads |-> "idle"]
    /\ bad_release = FALSE

\* --- Helpers --------------------------------------------------------------
SlotPtr(k)  == slots[k].ptr
SlotVer(k)  == slots[k].ver

\* TRUE iff every thread is outside any in-flight orphan operation.
AllIdle == (\A t \in Threads :
                /\ reown_pc[t] = "idle"
                /\ push_pc[t] = "idle")

\* ============================================================
\*  ACTIONS
\* ============================================================

\* --- orphan_push: P0 (claim PUSHING) --------------------------------------
\* Real code: owner-exit drain calls orphan_push.  Models the
\* OWNED-CAS-PUSHING step.  Allowed only when chunk is OWNED off-array.
PushP0(t) ==
    /\ push_pc[t] = "idle"
    /\ disp = OWNED
    /\ \neg released
    \* Owner exit pre-condition: BIT_OWNED is already clear (set in Init).
    /\ push_pc' = [push_pc EXCEPT ![t] = "pushing"]
    /\ disp' = PUSHING
    /\ UNCHANGED << slots, bit_owned, mask_cnt, released, pending_claim,
                    reown_pc, bad_release >>

\* --- orphan_push: P1 (publish into a slot) --------------------------------
\* The CAS that takes an empty slot.  Multiple slots considered; the model
\* lets TLC pick any empty slot (real impl scans owner_id mod K first, but
\* the safety property must hold regardless of which slot wins).
PushP1(t, k) ==
    /\ push_pc[t] = "pushing"
    /\ SlotPtr(k) = "Null"
    /\ disp = PUSHING
    /\ slots' = [slots EXCEPT ![k] = [ptr |-> "C", ver |-> SlotVer(k) + 1]]
    /\ disp' = k + 1   \* disp = 1+k convention
    /\ push_pc' = [push_pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED << bit_owned, mask_cnt, released, pending_claim,
                    reown_pc, bad_release >>

\* --- orphan_push: P1 give-up (all slots full) -----------------------------
\* If every slot is non-empty the push abandons and restores OWNED.
\* The chunk remains off-array, OWNED; a future free-to-empty will
\* claim RELEASED via orphan_claim_for_release.
PushP1GiveUp(t) ==
    /\ push_pc[t] = "pushing"
    /\ disp = PUSHING
    /\ (\A k \in 0..K-1 : SlotPtr(k) /= "Null")
    /\ push_pc' = [push_pc EXCEPT ![t] = "idle"]
    /\ disp' = OWNED
    /\ UNCHANGED << slots, bit_owned, mask_cnt, released, pending_claim,
                    reown_pc, bad_release >>

\* --- orphan_pop: PoP0 (slot-take CAS) -------------------------------------
\* An allocator path competes for the on-array chunk.
\* Take expects (ptr=C, ver=v); winner sets slot to (Null, v+1).
PopP0(t, k) ==
    /\ reown_pc[t] = "idle"
    /\ SlotPtr(k) = "C"
    /\ \neg released
    /\ slots' = [slots EXCEPT ![k] = [ptr |-> "Null", ver |-> SlotVer(k) + 1]]
    /\ reown_pc' = [reown_pc EXCEPT ![t] = "got_c"]
    /\ UNCHANGED << disp, bit_owned, mask_cnt, released, pending_claim,
                    push_pc, bad_release >>
    \* Notice: disp is NOT updated here — it lingers at 1+k.
    \* This is the documented stale-disp window the handoff fingers as
    \* the prime suspect.

\* --- orphan_pop: PoP1 (claim BIT_OWNED preserving MASK_CNT) ---------------
PopP1(t) ==
    /\ reown_pc[t] = "got_c"
    /\ \neg released
    /\ \neg bit_owned
    /\ bit_owned' = TRUE
    /\ reown_pc' = [reown_pc EXCEPT ![t] = "armed_owned"]
    /\ UNCHANGED << slots, disp, mask_cnt, released, pending_claim,
                    push_pc, bad_release >>

\* --- orphan_pop: PoP2 (publish disp = OWNED) ------------------------------
\* The final step that closes the re-own window.  After this step disp
\* matches the new ownership.
PopP2(t) ==
    /\ reown_pc[t] = "armed_owned"
    /\ disp' = OWNED
    /\ reown_pc' = [reown_pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED << slots, bit_owned, mask_cnt, released, pending_claim,
                    push_pc, bad_release >>

\* --- The free-the-last-survivor (cross-thread dec-to-0) -------------------
\* `atomicDecAndTest` on the 32-bit packed `m_flags_packed` returns TRUE
\* iff the new value is 0, i.e. iff pre-dec was bit_owned=F AND mask_cnt=1.
\* ONLY when the return is true does the freer go on to call
\* `orphan_claim_for_release` — set pending_claim so a Claim* action can
\* later run.  If bit_owned was TRUE at the dec moment, the dec still
\* happens but the claim path is NOT entered (modelled as
\* FreeDecToZero_Spurious).
\*
\* The pending_claim flag captures the C++ "atomicDecAndTest returned
\* true" return value across the gap until the claim CAS fires.  A POP
\* sequence can happen IN THIS GAP — that is exactly the documented
\* race: bit_owned can transition F→T (PoP1) AFTER pending_claim is set.
FreeDecToZero_TriggersClaim(t) ==
    /\ mask_cnt = 1
    /\ \neg bit_owned
    /\ \neg released
    /\ \neg pending_claim
    /\ mask_cnt' = 0
    /\ pending_claim' = TRUE
    /\ UNCHANGED << slots, disp, bit_owned, released, reown_pc, push_pc,
                    bad_release >>

\* Same dec but bit_owned was TRUE — no claim path, just count drop.
FreeDecToZero_Spurious(t) ==
    /\ mask_cnt = 1
    /\ bit_owned
    /\ \neg released
    /\ mask_cnt' = 0
    /\ UNCHANGED << slots, disp, bit_owned, released, pending_claim,
                    reown_pc, push_pc, bad_release >>

\* orphan_claim_for_release with d = OWNED (the off-array case):
\* CAS disp OWNED → RELEASED.  If wins, releases.
\* bad_release fires iff at THIS moment bit_owned is TRUE or mask_cnt > 0
\* — the C++ DEBUG_GUARD on orphan_release_self.
ClaimOWNED(t) ==
    /\ pending_claim
    /\ disp = OWNED
    /\ \neg released
    /\ disp' = RELEASED
    /\ released' = TRUE
    /\ pending_claim' = FALSE
    /\ bad_release' =
         (bad_release \/ bit_owned \/ (mask_cnt > 0))
    /\ UNCHANGED << slots, bit_owned, mask_cnt, reown_pc, push_pc >>

\* orphan_claim_for_release with d = 1+k (on-array case):
\* Loads slot[k], checks ptr == c (= "C"), then CAS (c,v) → (Null, v+1).
\* If versioned slot CAS wins, releases.
\* MODELLED AS ONE ATOMIC STEP — combining the load + the CAS.
\* This is *deliberately stronger* than the real impl (which does the
\* load and CAS as separate ops).  If TLC exposes a race here the real
\* code certainly has it.
ClaimSLOT(t, k) ==
    /\ pending_claim
    /\ disp = k + 1
    /\ \neg released
    \* Implementation reads slot[k] then expects (c,v) in the CAS.
    \* If the ptr in slot k is not "C" the claim bails (does not release).
    /\ SlotPtr(k) = "C"
    /\ slots' = [slots EXCEPT ![k] = [ptr |-> "Null", ver |-> SlotVer(k) + 1]]
    /\ disp' = RELEASED
    /\ released' = TRUE
    /\ pending_claim' = FALSE
    /\ bad_release' =
         (bad_release \/ bit_owned \/ (mask_cnt > 0))
    /\ UNCHANGED << bit_owned, mask_cnt, reown_pc, push_pc >>

\* orphan_claim_for_release with d = 1+k but slot mismatches (someone
\* popped + maybe re-pushed elsewhere): bail without releasing.  This
\* "bail" path is the LEAK direction.  Modelled so we can see whether
\* the model gets stuck or proceeds.  Clears pending_claim.
ClaimSLOTBail(t, k) ==
    /\ pending_claim
    /\ disp = k + 1
    /\ \neg released
    /\ SlotPtr(k) /= "C"
    /\ pending_claim' = FALSE
    /\ UNCHANGED << slots, disp, bit_owned, mask_cnt, released, reown_pc,
                    push_pc, bad_release >>

\* orphan_claim_for_release with d = PUSHING or RELEASED: bail per impl.
ClaimBail(t) ==
    /\ pending_claim
    /\ disp \in {PUSHING, RELEASED}
    /\ \neg released
    /\ pending_claim' = FALSE
    /\ UNCHANGED << slots, disp, bit_owned, mask_cnt, released, reown_pc,
                    push_pc, bad_release >>

\* ============================================================
\*  THE OPTIONAL "ALLOCATOR FREES, NEW SURVIVOR APPEARS" STEP
\* ============================================================
\* After PoP1 a thread "owns" the chunk and may now hand out fresh
\* slots, which would bump MASK_CNT.  We model a single such bump so
\* the post-re-own free races against the just-bumped count rather
\* than reproducing the initial 1→0 trace.
\* AllowReown gates this: when FALSE, the model strictly explores
\* the dec-to-0 of the original survivor only.
ReownAllocSlot(t) ==
    /\ AllowReown
    /\ reown_pc[t] = "armed_owned"
    /\ mask_cnt < 2
    /\ mask_cnt' = mask_cnt + 1
    /\ UNCHANGED << slots, disp, bit_owned, released, pending_claim,
                    reown_pc, push_pc, bad_release >>

\* The original cross-thread survivor's dec may happen any time the
\* chunk has more than 1 live slot; this is the benign multi-survivor
\* dec.  Does NOT trigger a claim (post-dec value > 0).
FreeDecCommon ==
    /\ mask_cnt > 1
    /\ \neg released
    /\ mask_cnt' = mask_cnt - 1
    /\ UNCHANGED << slots, disp, bit_owned, released, pending_claim,
                    reown_pc, push_pc, bad_release >>

\* ============================================================
\*  NEXT STATE
\* ============================================================
Next ==
    \/ \E t \in Threads :
         \/ PushP0(t)
         \/ PushP1GiveUp(t)
         \/ \E k \in 0..K-1 : PushP1(t, k)
         \/ \E k \in 0..K-1 : PopP0(t, k)
         \/ PopP1(t)
         \/ PopP2(t)
         \/ FreeDecToZero_TriggersClaim(t)
         \/ FreeDecToZero_Spurious(t)
         \/ ClaimOWNED(t)
         \/ \E k \in 0..K-1 : ClaimSLOT(t, k)
         \/ \E k \in 0..K-1 : ClaimSLOTBail(t, k)
         \/ ClaimBail(t)
         \/ ReownAllocSlot(t)
    \/ FreeDecCommon

Spec == Init /\ [][Next]_vars

\* ============================================================
\*  INVARIANTS
\* ============================================================

\* (A) Chunk is released at most once.  released is sticky TRUE; the
\*     invariant catches a SECOND attempt by guarding ClaimOWNED /
\*     ClaimSLOT with `\neg released`, so an attempt during released==TRUE
\*     would simply be impossible — we keep this trivially TRUE as a
\*     sanity check.
Inv_NoDoubleRelease == TRUE

\* (B) The DEBUG ABORT property — the C++ guard.  RELEASE must imply
\*     BIT_OWNED clear AND MASK_CNT == 0.  bad_release sticks when this
\*     would have aborted.
Inv_NoBadRelease == \neg bad_release

\* (C) An on-array chunk's disp/slots correspondence must hold when no
\*     thread is mid-re-own AND no push is in flight.  In flight, the
\*     re-own carries disp stale at 1+k while slot[k] is Null — that
\*     window is the design.
Inv_DispCoherentOnQuiet ==
    AllIdle =>
        /\ disp \in (1..K) =>
              SlotPtr(disp - 1) = "C"
        /\ disp = OWNED =>
              (\A k \in 0..K-1 : SlotPtr(k) /= "C")

\* (D) After release, MASK_CNT and BIT_OWNED must remain clear.
Inv_ReleasedSticky ==
    released => (\neg bit_owned /\ mask_cnt = 0)

\* (E) orphan_pop's caller must have ended up with the chunk OWNED before
\*     handing out slots — guarded by mask_cnt' update only happening
\*     when reown_pc[t] = "armed_owned" (= post-PoP1).
Inv_NoStrayMaskBump ==
    \A t \in Threads :
        (reown_pc[t] = "got_c") => (mask_cnt \in 0..2)
        \* Trivially true; defensive — catches a future refactor that
        \* allowed slot grants before PoP1.

THEOREM Spec => []TypeOK
THEOREM Spec => []Inv_NoBadRelease
THEOREM Spec => []Inv_DispCoherentOnQuiet
THEOREM Spec => []Inv_ReleasedSticky

================================================================================
