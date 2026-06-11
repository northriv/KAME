(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        Dual-licensed Apache 2.0 OR GPL-2.0-or-later — see OrphanReuse_36b.tla.
 ***************************************************************************)
------------------------- MODULE OrphanReuse_optC -------------------------
(*
 * "Option C" — the handoff doc's §8 fallback design.  Drops the orphan
 * array entirely; restores PRE-§36 release-on-empty for cross-thread
 * dec-to-0.  Owner exit non-empty leaves the chunk OWNED (bit cleared,
 * mask_cnt > 0); the chunk's last cross-thread free direct-releases it
 * via the atomicDecAndTest-returned-true path.
 *
 * Tradeoff: NO reuse — non-empty orphans don't get their free slots
 * re-handed-out by a new owner; they drain through cross-frees and the
 * chunk is released.  The chunk-recycle cache reabsorbs the units, so
 * the warm-page benefit is partly retained.
 *
 * This spec serves as a NEGATIVE CONTROL: TLC should find no
 * bad_release.  Same state-space shape as the §36b specs minus the
 * push/pop/claim machinery.
 *)

EXTENDS Integers, FiniteSets, TLC, Sequences

CONSTANTS Threads

ASSUME Cardinality(Threads) >= 1

VARIABLES
    bit_owned,     \* BOOLEAN
    mask_cnt,      \* Nat — live-slot count
    released,      \* BOOLEAN — chunk released to region bitmap
    bad_release    \* BOOLEAN — the DEBUG ABORT would have fired

vars == << bit_owned, mask_cnt, released, bad_release >>

TypeOK ==
    /\ bit_owned \in BOOLEAN
    /\ mask_cnt \in 0..2
    /\ released \in BOOLEAN
    /\ bad_release \in BOOLEAN

\* Init: owner alive, two slots live (worst case for the spec's depth).
Init ==
    /\ bit_owned = TRUE
    /\ mask_cnt = 2
    /\ released = FALSE
    /\ bad_release = FALSE

\* Owner exit — atomic fetch_and(~BIT_OWNED) + branch.
\* Non-empty: chunk lingers (no array, just sits there) — released by
\* the final cross-free.
OwnerExit_NonEmpty ==
    /\ bit_owned
    /\ mask_cnt > 0
    /\ \neg released
    /\ bit_owned' = FALSE
    /\ UNCHANGED << mask_cnt, released, bad_release >>

\* Empty: owner is unique releaser.
OwnerExit_Empty ==
    /\ bit_owned
    /\ mask_cnt = 0
    /\ \neg released
    /\ bit_owned' = FALSE
    /\ released' = TRUE
    /\ bad_release' = (bad_release \/ bit_owned')   \* trivially F here
    /\ UNCHANGED mask_cnt

\* Cross-thread free of one survivor.
FreeDec ==
    /\ mask_cnt > 1
    /\ \neg released
    /\ mask_cnt' = mask_cnt - 1
    /\ UNCHANGED << bit_owned, released, bad_release >>

\* Cross-thread free that brings mask_cnt to 0.  Two branches via
\* atomicDecAndTest: returns TRUE iff bit_owned was F AND mask_cnt was 1.
\* TRUE → direct release.  FALSE → just dec, owner will release.
FreeDecToZero_Releases ==
    /\ mask_cnt = 1
    /\ \neg bit_owned
    /\ \neg released
    /\ mask_cnt' = 0
    /\ released' = TRUE
    /\ bad_release' = (bad_release \/ bit_owned \/ FALSE)
    /\ UNCHANGED bit_owned

FreeDecToZero_Spurious ==
    /\ mask_cnt = 1
    /\ bit_owned
    /\ \neg released
    /\ mask_cnt' = 0
    /\ UNCHANGED << bit_owned, released, bad_release >>

Next ==
    \/ OwnerExit_NonEmpty
    \/ OwnerExit_Empty
    \/ FreeDec
    \/ FreeDecToZero_Releases
    \/ FreeDecToZero_Spurious

Spec == Init /\ [][Next]_vars

Inv_NoBadRelease == \neg bad_release
Inv_ReleasedSticky == released => (\neg bit_owned /\ mask_cnt = 0)

THEOREM Spec => []TypeOK
THEOREM Spec => []Inv_NoBadRelease
THEOREM Spec => []Inv_ReleasedSticky

================================================================================
