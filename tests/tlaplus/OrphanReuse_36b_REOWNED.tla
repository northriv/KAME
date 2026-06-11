(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        Dual-licensed Apache 2.0 OR GPL-2.0-or-later — see OrphanReuse_36b.tla.
 ***************************************************************************)
------------------------- MODULE OrphanReuse_36b_REOWNED -------------------------
(*
 * §36b CANDIDATE FIX, verified.
 *
 * `OrphanReuse_36b.tla` (the as-documented design) reproduces the
 * `ORPHAN_RELEASE_BAD` ABA race the stress-test exhibits at 1.3 %:
 * `orphan_claim_for_release` sees `disp = OWNED` and CAS-releases a
 * chunk whose `bit_owned` is TRUE — because PoP2 wrote OWNED into disp
 * AFTER PoP1 set bit_owned.  Root cause: `disp = OWNED` is overloaded
 * with two meanings (never-pushed vs re-owned post-pop) that the freer
 * cannot tell apart.
 *
 * FIX (one extra disp value, minimal diff to the design):
 *   * Add `REOWNED` to the disp domain.
 *   * PoP2 transitions disp = 1+k → REOWNED (NOT OWNED).
 *   * orphan_claim_for_release at d = REOWNED bails (no release):
 *     the chunk is live again, its owner will eventually push or
 *     direct-release on its own exit.
 *   * Owner exit on a REOWNED chunk runs orphan_push normally: it
 *     transitions REOWNED → PUSHING (just like OWNED → PUSHING).  We
 *     enable PushP0 from disp ∈ {OWNED, REOWNED}.
 *
 * Cost: one extra value in m_orphan_disp's byte enum (already a uint8).
 * Zero hot-path overhead — PoP2 still does one relaxed store; the only
 * change is the store value.
 *
 * Result here: TLC explores the same K=2, 2-thread state space and
 * `Inv_NoBadRelease` holds (no counter-example).  See README §36b
 * verification line.
 *)

EXTENDS Integers, FiniteSets, TLC, Sequences

CONSTANTS Threads, K, AllowReown

ASSUME K \in 1..4
ASSUME Cardinality(Threads) >= 1
ASSUME AllowReown \in BOOLEAN

OWNED    == 0
PUSHING  == 100
RELEASED == 200
REOWNED  == 300

VARIABLES
    slots, disp, bit_owned, mask_cnt, released, pending_claim,
    reown_pc, push_pc, bad_release

vars == << slots, disp, bit_owned, mask_cnt, released, pending_claim,
           reown_pc, push_pc, bad_release >>

DispDomain == { OWNED, PUSHING, RELEASED, REOWNED } \cup (1 .. K)

\* pending_claim is per-thread: it models the stack-local `release_me`
\* boolean each freer holds across the gap from `atomicDecAndTest()==true`
\* to the actual `orphan_claim_for_release()` CAS.  Conflating it
\* across threads (single-flag model) wrongly tied multiple dec-to-0
\* events together and triggered spurious traces.
TypeOK ==
    /\ slots \in [0..K-1 -> [ptr: {"Null","C"}, ver: Nat]]
    /\ disp \in DispDomain
    /\ bit_owned \in BOOLEAN
    /\ mask_cnt \in 0..2
    /\ released \in BOOLEAN
    /\ pending_claim \in [Threads -> BOOLEAN]
    /\ reown_pc \in [Threads -> {"idle","got_c","armed_owned"}]
    /\ push_pc \in [Threads -> {"idle","pushing"}]
    /\ bad_release \in BOOLEAN

Init ==
    /\ slots = [k \in 0..K-1 |->
                  IF k = 0 THEN [ptr |-> "C",    ver |-> 1]
                           ELSE [ptr |-> "Null", ver |-> 0]]
    /\ disp = 1
    /\ bit_owned = FALSE
    /\ mask_cnt = 1
    /\ released = FALSE
    /\ pending_claim = [t \in Threads |-> FALSE]
    /\ reown_pc = [t \in Threads |-> "idle"]
    /\ push_pc = [t \in Threads |-> "idle"]
    /\ bad_release = FALSE

SlotPtr(k) == slots[k].ptr
SlotVer(k) == slots[k].ver
AllIdle == (\A t \in Threads :
                /\ reown_pc[t] = "idle"
                /\ push_pc[t] = "idle")

\* --- orphan_push: enabled from disp ∈ {OWNED, REOWNED} --------------------
\* The REOWNED case fires when the post-pop owner has exited again
\* without ever resetting disp to OWNED.  Same transition either way:
\* CAS into PUSHING.
\* Owner-exit transitions.  Real code does ONE atomic fetch_and(~BIT_OWNED)
\* and then branches on the OLD value's MASK_CNT bits:
\*   * old.MASK_CNT > 0  → orphan_push
\*   * old.MASK_CNT == 0 → direct release (the unique releaser case from
\*                          the §36 m_flags_packed comment)
\* We fuse the atomic bit-clear with its branch into a SINGLE TLA action
\* per case.  This matches the real atomicity: the branch decision is
\* based on the fetch_and's returned-old-value, which is a consistent
\* snapshot, so no interleaving can fall between "clear bit" and "decide
\* which path".
\*
\* Pre-condition (both cases): bit_owned = TRUE (= the exiting thread is
\* the live owner).  An on-array chunk has bit_owned = FALSE (its
\* previous owner already exited) so neither case applies to such a
\* chunk — its disp is 1+k and the freer / popper paths handle it.
OwnerExit_NonEmpty(t) ==
    /\ disp \in {OWNED, REOWNED}
    /\ bit_owned
    /\ mask_cnt > 0
    /\ push_pc[t] = "idle"
    /\ \neg released
    /\ bit_owned' = FALSE
    /\ push_pc' = [push_pc EXCEPT ![t] = "pushing"]
    /\ disp' = PUSHING
    /\ UNCHANGED << slots, mask_cnt, released, pending_claim, reown_pc,
                    bad_release >>

OwnerExit_Empty(t) ==
    /\ disp \in {OWNED, REOWNED}
    /\ bit_owned
    /\ mask_cnt = 0
    /\ \neg released
    /\ bit_owned' = FALSE
    /\ disp' = RELEASED
    /\ released' = TRUE
    \* Empty + sole-owner exit is the §36 "unique releaser" path; if
    \* pending_claim is also set on some OTHER thread, two release sites
    \* race.  bad_release fires here iff bit_owned was already F (which
    \* the pre-cond rules out) — so it never fires from THIS site, but
    \* the OTHER race (Claim* CAS-races OwnerExit) is what we want
    \* visible.  Keep the standard guard.
    /\ bad_release' = bad_release
    /\ UNCHANGED << slots, mask_cnt, pending_claim, reown_pc, push_pc >>

\* PushP0 (= mid-exit transition into PUSHING) is folded into
\* OwnerExit_NonEmpty above; we keep a legacy alias DISABLED below so
\* the Next clause's PushP0(t) reference still resolves.
PushP0(t) == FALSE

PushP1(t, k) ==
    /\ push_pc[t] = "pushing"
    /\ SlotPtr(k) = "Null"
    /\ disp = PUSHING
    /\ slots' = [slots EXCEPT ![k] = [ptr |-> "C", ver |-> SlotVer(k) + 1]]
    /\ disp' = k + 1
    /\ push_pc' = [push_pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED << bit_owned, mask_cnt, released, pending_claim,
                    reown_pc, bad_release >>

PushP1GiveUp(t) ==
    /\ push_pc[t] = "pushing"
    /\ disp = PUSHING
    /\ (\A k \in 0..K-1 : SlotPtr(k) /= "Null")
    /\ push_pc' = [push_pc EXCEPT ![t] = "idle"]
    /\ disp' = OWNED
    /\ UNCHANGED << slots, bit_owned, mask_cnt, released, pending_claim,
                    reown_pc, bad_release >>

PopP0(t, k) ==
    /\ reown_pc[t] = "idle"
    /\ SlotPtr(k) = "C"
    /\ \neg released
    /\ slots' = [slots EXCEPT ![k] = [ptr |-> "Null", ver |-> SlotVer(k) + 1]]
    /\ reown_pc' = [reown_pc EXCEPT ![t] = "got_c"]
    /\ UNCHANGED << disp, bit_owned, mask_cnt, released, pending_claim,
                    push_pc, bad_release >>

PopP1(t) ==
    /\ reown_pc[t] = "got_c"
    /\ \neg released
    /\ \neg bit_owned
    /\ bit_owned' = TRUE
    /\ reown_pc' = [reown_pc EXCEPT ![t] = "armed_owned"]
    /\ UNCHANGED << slots, disp, mask_cnt, released, pending_claim,
                    push_pc, bad_release >>

\* --- PoP2: store REOWNED, not OWNED --- THE FIX ---------------------------
PopP2(t) ==
    /\ reown_pc[t] = "armed_owned"
    /\ disp' = REOWNED
    /\ reown_pc' = [reown_pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED << slots, bit_owned, mask_cnt, released, pending_claim,
                    push_pc, bad_release >>

FreeDecToZero_TriggersClaim(t) ==
    /\ mask_cnt = 1
    /\ \neg bit_owned
    /\ \neg released
    /\ \neg pending_claim[t]
    /\ mask_cnt' = 0
    /\ pending_claim' = [pending_claim EXCEPT ![t] = TRUE]
    /\ UNCHANGED << slots, disp, bit_owned, released, reown_pc, push_pc,
                    bad_release >>

FreeDecToZero_Spurious(t) ==
    /\ mask_cnt = 1
    /\ bit_owned
    /\ \neg released
    /\ mask_cnt' = 0
    /\ UNCHANGED << slots, disp, bit_owned, released, pending_claim,
                    reown_pc, push_pc, bad_release >>

ClaimOWNED(t) ==
    /\ pending_claim[t]
    /\ disp = OWNED
    /\ \neg released
    /\ disp' = RELEASED
    /\ released' = TRUE
    /\ pending_claim' = [pending_claim EXCEPT ![t] = FALSE]
    /\ bad_release' =
         (bad_release \/ bit_owned \/ (mask_cnt > 0))
    /\ UNCHANGED << slots, bit_owned, mask_cnt, reown_pc, push_pc >>

ClaimSLOT(t, k) ==
    /\ pending_claim[t]
    /\ disp = k + 1
    /\ \neg released
    /\ SlotPtr(k) = "C"
    /\ slots' = [slots EXCEPT ![k] = [ptr |-> "Null", ver |-> SlotVer(k) + 1]]
    /\ disp' = RELEASED
    /\ released' = TRUE
    /\ pending_claim' = [pending_claim EXCEPT ![t] = FALSE]
    /\ bad_release' =
         (bad_release \/ bit_owned \/ (mask_cnt > 0))
    /\ UNCHANGED << bit_owned, mask_cnt, reown_pc, push_pc >>

ClaimSLOTBail(t, k) ==
    /\ pending_claim[t]
    /\ disp = k + 1
    /\ \neg released
    /\ SlotPtr(k) /= "C"
    /\ pending_claim' = [pending_claim EXCEPT ![t] = FALSE]
    /\ UNCHANGED << slots, disp, bit_owned, mask_cnt, released, reown_pc,
                    push_pc, bad_release >>

\* --- ClaimBail covers PUSHING, RELEASED, AND REOWNED ----------------------
\* The fix's behavioural change: a claim hitting REOWNED bails (does NOT
\* release).  The chunk's new owner will manage its lifetime.
ClaimBail(t) ==
    /\ pending_claim[t]
    /\ disp \in {PUSHING, RELEASED, REOWNED}
    /\ \neg released
    /\ pending_claim' = [pending_claim EXCEPT ![t] = FALSE]
    /\ UNCHANGED << slots, disp, bit_owned, mask_cnt, released, reown_pc,
                    push_pc, bad_release >>

ReownAllocSlot(t) ==
    /\ AllowReown
    /\ reown_pc[t] = "armed_owned"
    /\ mask_cnt < 2
    /\ mask_cnt' = mask_cnt + 1
    /\ UNCHANGED << slots, disp, bit_owned, released, pending_claim,
                    reown_pc, push_pc, bad_release >>

FreeDecCommon ==
    /\ mask_cnt > 1
    /\ \neg released
    /\ mask_cnt' = mask_cnt - 1
    /\ UNCHANGED << slots, disp, bit_owned, released, pending_claim,
                    reown_pc, push_pc, bad_release >>

Next ==
    \/ \E t \in Threads :
         \/ OwnerExit_NonEmpty(t)
         \/ OwnerExit_Empty(t)
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

Inv_NoBadRelease == \neg bad_release

Inv_DispCoherentOnQuiet ==
    AllIdle =>
        /\ disp \in (1..K) =>
              SlotPtr(disp - 1) = "C"
        /\ disp = OWNED =>
              (\A k \in 0..K-1 : SlotPtr(k) /= "C")

Inv_ReleasedSticky ==
    released => (\neg bit_owned /\ mask_cnt = 0)

THEOREM Spec => []TypeOK
THEOREM Spec => []Inv_NoBadRelease
THEOREM Spec => []Inv_DispCoherentOnQuiet
THEOREM Spec => []Inv_ReleasedSticky

================================================================================
