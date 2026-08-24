(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.
 ***************************************************************************)
------------------- MODULE atomic_shared_ptr_scopetoken -------------------
(*
 * Ownership-token conservation for the scoped_atomic_view hand-off layer.
 *
 * Motivated by DYNNODE_UAF_HANDOFF.md §11.3-§11.5: the dynamic_node UAF is
 * a PacketWrapper/Packet refcount reaching zero while a holder survives.
 * v6 slot identity established (Q1) two DISTINCT holders and (Q2) that the
 * second holder's INC is VISIBLE -- so the count went wrong somewhere in
 * the +1-token protocol, not at an invisible install.  Q3's single capture
 * put both releasers in scoped_atomic_view instances (CASInfo::old_wrapper
 * and a snapshot() local).  This spec models exactly that token protocol:
 *
 *   a +1 of an object's refcount travels between
 *     - the atomic linkage word itself (the installed wrapper's share),
 *     - local_shared_ptr holders               (lsp slots),
 *     - scoped_atomic_view holders             (Owned mode),
 *     - CASInfo parking slots                  (park at transaction_impl.h
 *       :2178 via consume_scoped_view; unpark at :3329 via the ScopedNeg
 *       move-in ctor),
 *   with ZERO atomic operations on each transfer ("caller is responsible
 *   ... we do NOT verify" -- atomic_smart_ptr.h, assign_from_local / the
 *   local_shared_ptr&& move-in ctor), plus the two release_() branches
 *   (Owned: plain fetch_sub; TagHeld: tag release).
 *
 * WHAT IS ABSTRACTED, AND WHY IT IS SOUND TO DO SO:
 *   - The tag/drain machinery (acquire_tag_ref_ / release_tag_ref_ /
 *     CASTransfer of local tags to the global count) is Layer-1-verified
 *     (atomic_shared_ptr.tla, 66.5M states).  This spec does NOT re-verify
 *     it: CommitCAS requires link_tag = 0 (a swap against a quiescent tag
 *     word), which makes tag accounting exact here without re-modelling
 *     the transfer path.  The bugs this spec hunts -- a zero-atomic
 *     transfer leaving both ends owning, or a release against a holder
 *     that no longer backs it -- are orthogonal to tag transfer.
 *   - The Packet tree is absent.  Per §11.5 Q2 the defect is in the token
 *     layer; the tree walk only DRIVES a sequence of transfers, which this
 *     spec generates nondeterministically (a superset of every walk).
 *
 * THE INVARIANT (§11.3, token form):
 *     rc[o]  =  linkage share  +  #lsp holders  +  #Owned views  +  #parked
 *   for every live object; a freed object has no holder of any kind; and
 *   no release ever fires against rc = 0 (NoUnderflow -- the tripwire).
 *
 * CONSTANT Bug \in {"none", "setview_noempty", "unpark_noempty"}
 *   validates the detector: each knob weakens one documented contract the
 *   way a real defect would, and MUST produce a Conservation violation.
 *   "none" must pass exhaustively.
 *)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Threads,        \* e.g. {1, 2}
    Objects,        \* wrapper identities, e.g. {"A", "B"}
    Null,
    NL,             \* lsp slots per thread   (e.g. 2)
    NV,             \* view slots per thread  (e.g. 2)
    TagCap,         \* max simultaneous tags  (e.g. 2)
    Bug,            \* "none" | "setview_noempty" | "unpark_noempty"
    TagXfer         \* FALSE: CommitCAS requires a quiescent tag word (v1).
                    \* TRUE : CommitCAS transfers outstanding tags to the
                    \*        displaced object's rc (CASTransfer), and the
                    \*        tag-release paths gain their "word changed ->
                    \*        decrement global" branch -- the Layer-1 drain
                    \*        machinery COMPOSED with the Owned-view layer,
                    \*        which neither Layer 1 (no Owned mode) nor the
                    \*        v1 run (quiescent cut) covered.  Backings are
                    \*        FUNGIBLE under transfer (a stolen word-tag and
                    \*        a transferred rc-unit swap roles), so the
                    \*        per-object equality weakens to >= and the
                    \*        equality moves to the global sum.

ASSUME Bug \in {"none", "setview_noempty", "unpark_noempty"}
ASSUME TagXfer \in BOOLEAN

VARIABLES
    rc,             \* [Objects -> Nat]  true global refcount
    freed,          \* [Objects -> BOOLEAN]
    link_cur,       \* Objects \cup {Null}   the installed wrapper
    link_tag,       \* 0..TagCap             local tag count on the word
    lsp,            \* [Threads -> [1..NL -> Objects \cup {Null}]]
    vmode,          \* [Threads -> [1..NV -> {"empty","tag","owned"}]]
    vobj,           \* [Threads -> [1..NV -> Objects \cup {Null}]]
    park,           \* [Threads -> Objects \cup {Null}]  CASInfo::old_wrapper
    underflow       \* BOOLEAN  a release fired against rc = 0

vars == <<rc, freed, link_cur, link_tag, lsp, vmode, vobj, park, underflow>>

LSlots == 1..NL
VSlots == 1..NV

(* -------------------------------------------------------------------- *)
(* Helpers                                                                *)

\* Release one +1 of o: the C++ fetch_sub(1) + delete-check.  A release
\* against rc = 0 is the DEC-UNDERFLOW tripwire.
DecEff(o) ==
    IF rc[o] = 0
    THEN [r |-> rc,                      f |-> freed,                 u |-> TRUE]
    ELSE [r |-> [rc EXCEPT ![o] = @ - 1],
          f |-> IF rc[o] = 1 THEN [freed EXCEPT ![o] = TRUE] ELSE freed,
          u |-> underflow]

Holders(o) ==
    (IF link_cur = o THEN 1 ELSE 0)
  + Cardinality({<<t, i>> \in Threads \X LSlots : lsp[t][i] = o})
  + Cardinality({<<t, v>> \in Threads \X VSlots :
                     vmode[t][v] \in {"owned", "promoting"} /\ vobj[t][v] = o})
  + Cardinality({t \in Threads : park[t] = o})

TagPins(o) ==
    Cardinality({<<t, v>> \in Threads \X VSlots :
                     vmode[t][v] \in {"tag", "promoting"} /\ vobj[t][v] = o})

(* -------------------------------------------------------------------- *)
Init ==
    /\ \E a \in Objects :
        /\ link_cur = a
        /\ rc = [o \in Objects |-> IF o = a THEN 1 ELSE 0]
        /\ freed = [o \in Objects |-> o /= a]
    /\ link_tag = 0
    /\ lsp = [t \in Threads |-> [i \in LSlots |-> Null]]
    /\ vmode = [t \in Threads |-> [v \in VSlots |-> "empty"]]
    /\ vobj = [t \in Threads |-> [v \in VSlots |-> Null]]
    /\ park = [t \in Threads |-> Null]
    /\ underflow = FALSE

(* -------------------------------------------------------------------- *)
(* local_shared_ptr ops                                                   *)

\* load_shared_(): pin + promote, net rc+1 into an empty lsp slot.
\* (The pin/promote interior is Layer-1-verified; net effect modelled.)
LoadShared(t, i) ==
    /\ link_cur /= Null
    /\ lsp[t][i] = Null
    /\ lsp' = [lsp EXCEPT ![t][i] = link_cur]
    /\ rc' = [rc EXCEPT ![link_cur] = @ + 1]
    /\ UNCHANGED <<freed, link_cur, link_tag, vmode, vobj, park, underflow>>

\* lsp copy ctor: fetch_add(1).
LspCopy(t, i, j) ==
    /\ i /= j
    /\ lsp[t][i] /= Null
    /\ lsp[t][j] = Null
    /\ lsp' = [lsp EXCEPT ![t][j] = lsp[t][i]]
    /\ rc' = [rc EXCEPT ![lsp[t][i]] = @ + 1]
    /\ UNCHANGED <<freed, link_cur, link_tag, vmode, vobj, park, underflow>>

\* lsp reset(): fetch_sub(1) + delete-check.
LspReset(t, i) ==
    /\ lsp[t][i] /= Null
    /\ LET d == DecEff(lsp[t][i]) IN
        /\ rc' = d.r /\ freed' = d.f /\ underflow' = d.u
    /\ lsp' = [lsp EXCEPT ![t][i] = Null]
    /\ UNCHANGED <<link_cur, link_tag, vmode, vobj, park>>

\* Rebirth: a freed object id is reused by a fresh allocation
\* (make_local_shared): rc = 1 owned by the creating lsp slot.
\* §11.3's hazard is precisely a rebirth while a stale holder survives --
\* Conservation catches it as holders > rc on the reborn object.
MakeNew(t, i, o) ==
    /\ lsp[t][i] = Null
    /\ freed[o]
    /\ freed' = [freed EXCEPT ![o] = FALSE]
    /\ rc' = [rc EXCEPT ![o] = 1]
    /\ lsp' = [lsp EXCEPT ![t][i] = o]
    /\ UNCHANGED <<link_cur, link_tag, vmode, vobj, park, underflow>>

(* -------------------------------------------------------------------- *)
(* scoped_atomic_view ops                                                 *)

\* Scoped acquire in TagHeld mode: pin via the word's tag bits, no rc op.
ViewAcquireTag(t, v) ==
    /\ link_cur /= Null
    /\ vmode[t][v] = "empty"
    /\ link_tag < TagCap
    /\ link_tag' = link_tag + 1
    /\ vmode' = [vmode EXCEPT ![t][v] = "tag"]
    /\ vobj' = [vobj EXCEPT ![t][v] = link_cur]
    /\ UNCHANGED <<rc, freed, link_cur, lsp, park, underflow>>

\* Promote TagHeld -> Owned, step 1: the fetch_add(1) (load_shared_-style).
\* Split from the tag release below because in C++ they are two atomic ops
\* and a CommitCAS can transfer the word's tags in between.
ViewPromoteAdd(t, v) ==
    /\ vmode[t][v] = "tag"
    /\ rc' = [rc EXCEPT ![vobj[t][v]] = @ + 1]
    /\ vmode' = [vmode EXCEPT ![t][v] = "promoting"]
    /\ UNCHANGED <<freed, link_cur, link_tag, lsp, vobj, park, underflow>>

\* Promote step 2 / release_() TagHeld branch: release_tag_ref_ semantics.
\* The word CAS succeeds only against (vobj, tag > 0); otherwise the pin's
\* backing was CASTransfer'd into rc and the release decrements global.
\* (With TagXfer = FALSE the else-branch is unreachable: a pinned word
\*  cannot change.)
TagReleaseEff(o) ==
    IF link_cur = o /\ link_tag > 0
    THEN [tag |-> link_tag - 1, r |-> rc, f |-> freed, u |-> underflow]
    ELSE LET d == DecEff(o) IN
         [tag |-> link_tag, r |-> d.r, f |-> d.f, u |-> d.u]

ViewPromoteRelease(t, v) ==
    /\ vmode[t][v] = "promoting"
    /\ LET e == TagReleaseEff(vobj[t][v]) IN
        /\ link_tag' = e.tag /\ rc' = e.r /\ freed' = e.f /\ underflow' = e.u
    /\ vmode' = [vmode EXCEPT ![t][v] = "owned"]
    /\ UNCHANGED <<link_cur, lsp, vobj, park>>

ViewReleaseTag(t, v) ==
    /\ vmode[t][v] = "tag"
    /\ LET e == TagReleaseEff(vobj[t][v]) IN
        /\ link_tag' = e.tag /\ rc' = e.r /\ freed' = e.f /\ underflow' = e.u
    /\ vmode' = [vmode EXCEPT ![t][v] = "empty"]
    /\ vobj' = [vobj EXCEPT ![t][v] = Null]
    /\ UNCHANGED <<link_cur, lsp, park>>

\* release_(), Owned branch: plain fetch_sub(1) + delete-check.
ViewReleaseOwned(t, v) ==
    /\ vmode[t][v] = "owned"
    /\ LET d == DecEff(vobj[t][v]) IN
        /\ rc' = d.r /\ freed' = d.f /\ underflow' = d.u
    /\ vmode' = [vmode EXCEPT ![t][v] = "empty"]
    /\ vobj' = [vobj EXCEPT ![t][v] = Null]
    /\ UNCHANGED <<link_cur, link_tag, lsp, park>>

\* set_view(): release the current view, then adopt the lsp's +1 with zero
\* atomic ops (assign_from_local / the local_shared_ptr&& move-in ctor:
\* "caller is responsible ... we do NOT verify").  The Bug knob
\* "setview_noempty" models the contract breaking: the source lsp is NOT
\* emptied, leaving two holders backed by one +1.
SetView(t, v, i) ==
    /\ lsp[t][i] /= Null
    /\ vmode[t][v] \in {"empty", "owned"}   \* (TagHeld source: release first)
    /\ LET src == lsp[t][i] IN
       IF vmode[t][v] = "owned"
       THEN LET d == DecEff(vobj[t][v]) IN
            /\ rc' = d.r /\ freed' = d.f /\ underflow' = d.u
            /\ vmode' = [vmode EXCEPT ![t][v] = "owned"]
            /\ vobj' = [vobj EXCEPT ![t][v] = src]
            /\ lsp' = [lsp EXCEPT ![t][i] =
                          IF Bug = "setview_noempty" THEN src ELSE Null]
       ELSE /\ vmode' = [vmode EXCEPT ![t][v] = "owned"]
            /\ vobj' = [vobj EXCEPT ![t][v] = src]
            /\ lsp' = [lsp EXCEPT ![t][i] =
                          IF Bug = "setview_noempty" THEN src ELSE Null]
            /\ UNCHANGED <<rc, freed, underflow>>
    /\ UNCHANGED <<link_cur, link_tag, park>>

\* Park: consume_scoped_view() -> CASInfo::old_wrapper (transaction_impl.h
\* :2178).  Zero atomic ops; the view is emptied.
ParkToCAS(t, v) ==
    /\ vmode[t][v] = "owned"
    /\ park[t] = Null
    /\ park' = [park EXCEPT ![t] = vobj[t][v]]
    /\ vmode' = [vmode EXCEPT ![t][v] = "empty"]
    /\ vobj' = [vobj EXCEPT ![t][v] = Null]
    /\ UNCHANGED <<rc, freed, link_cur, link_tag, lsp, underflow>>

\* Unpark: std::move(it->old_wrapper) into the ScopedNeg move-in ctor
\* (transaction_impl.h :3329).  Bug knob "unpark_noempty": the CASInfo
\* keeps its copy -- both ends owning.
UnparkFromCAS(t, v) ==
    /\ park[t] /= Null
    /\ vmode[t][v] = "empty"
    /\ vmode' = [vmode EXCEPT ![t][v] = "owned"]
    /\ vobj' = [vobj EXCEPT ![t][v] = park[t]]
    /\ park' = [park EXCEPT ![t] =
                   IF Bug = "unpark_noempty" THEN park[t] ELSE Null]
    /\ UNCHANGED <<rc, freed, link_cur, link_tag, lsp, underflow>>

\* CASInfo destroyed still owning (cas_infos cleared on a DISTURBED path):
\* fetch_sub(1) + delete-check.
ParkDrop(t) ==
    /\ park[t] /= Null
    /\ LET d == DecEff(park[t]) IN
        /\ rc' = d.r /\ freed' = d.f /\ underflow' = d.u
    /\ park' = [park EXCEPT ![t] = Null]
    /\ UNCHANGED <<link_cur, link_tag, lsp, vmode, vobj>>

(* -------------------------------------------------------------------- *)
(* The linkage write                                                      *)

\* Install a new wrapper: the writer's lsp +1 becomes the linkage share;
\* the displaced wrapper loses its linkage share.  Requires a quiescent
\* tag word (see header: the tag-transfer path is Layer-1-verified and
\* deliberately out of scope).
CommitCAS(t, i) ==
    /\ lsp[t][i] /= Null
    /\ (~TagXfer => link_tag = 0)   \* v1: quiescent word only
    /\ lsp[t][i] /= link_cur       \* installing what is installed is a no-op
    /\ LET old == link_cur
           new == lsp[t][i] IN
       /\ link_cur' = new
       /\ link_tag' = 0            \* CASTransfer zeroes the word's tags...
       /\ lsp' = [lsp EXCEPT ![t][i] = Null]   \* writer's +1 moves onto the word
       /\ IF old = Null
          THEN /\ UNCHANGED <<rc, freed, underflow>>
          ELSE IF link_tag = 0
          THEN LET d == DecEff(old) IN         \* lose linkage share
               /\ rc' = d.r /\ freed' = d.f /\ underflow' = d.u
          ELSE \* ...and adds them to the displaced object's rc
               \* (CASTransfer: global_rc[old] += local tags), net with the
               \* lost linkage share: += tag - 1, never negative here.
               /\ rc' = [rc EXCEPT ![old] = @ + link_tag - 1]
               /\ UNCHANGED <<freed, underflow>>
    /\ UNCHANGED <<vmode, vobj, park>>

(* -------------------------------------------------------------------- *)
Next ==
    \E t \in Threads :
        \/ \E i \in LSlots : LoadShared(t, i)
        \/ \E i, j \in LSlots : LspCopy(t, i, j)
        \/ \E i \in LSlots : LspReset(t, i)
        \/ \E i \in LSlots : \E o \in Objects : MakeNew(t, i, o)
        \/ \E v \in VSlots : ViewAcquireTag(t, v)
        \/ \E v \in VSlots : ViewPromoteAdd(t, v)
        \/ \E v \in VSlots : ViewPromoteRelease(t, v)
        \/ \E v \in VSlots : ViewReleaseTag(t, v)
        \/ \E v \in VSlots : ViewReleaseOwned(t, v)
        \/ \E v \in VSlots : \E i \in LSlots : SetView(t, v, i)
        \/ \E v \in VSlots : ParkToCAS(t, v)
        \/ \E v \in VSlots : UnparkFromCAS(t, v)
        \/ ParkDrop(t)
        \/ \E i \in LSlots : CommitCAS(t, i)

Spec == Init /\ [][Next]_vars

(* -------------------------------------------------------------------- *)
(* Invariants                                                             *)

\* Â§11.3, token form.  Pins in "tag"/"promoting" mode are backed either
\* by the word's tag count or (after a CASTransfer) by a transferred rc
\* unit; the two backings are FUNGIBLE, so:
\*   - per object: rc >= explicit holders (equality only when TagXfer is
\*     off, where no pin's backing ever moves into rc);
\*   - globally: sum(rc) + link_tag  =  all holders + all pins, exactly.
\* A freed object must have nothing at all pointing at it.
PinsAll ==
    Cardinality({<<t, v>> \in Threads \X VSlots :
                     vmode[t][v] \in {"tag", "promoting"}})

\* "promoting" already did its fetch_add: it is a holder AND a pin whose
\* release is still pending -- count it on the holder side and keep its
\* pending tag-release on the pin side.
HoldersAll ==
    LET objs == Objects IN
    (IF link_cur /= Null THEN 1 ELSE 0)
  + Cardinality({<<t, i>> \in Threads \X LSlots : lsp[t][i] /= Null})
  + Cardinality({<<t, v>> \in Threads \X VSlots :
                     vmode[t][v] \in {"owned", "promoting"}})
  + Cardinality({t \in Threads : park[t] /= Null})

SumRC == LET S[oo \in SUBSET Objects] ==
              IF oo = {} THEN 0
              ELSE LET x == CHOOSE q \in oo : TRUE
                   IN rc[x] + S[oo \ {x}]
         IN S[Objects]

\* NOTE a "promoting" view consumes TWO backings: its fetch_add already
\* landed (holder side) and its original pin's release is still pending
\* (pin side) -- so it is counted in BOTH HoldersAll and PinsAll.
Conservation ==
    /\ SumRC + link_tag = HoldersAll + PinsAll
    /\ \A o \in Objects :
        /\ ~freed[o] => rc[o] >= Holders(o)
        /\ (~TagXfer /\ ~freed[o]) => rc[o] = Holders(o)
        /\ freed[o] => (rc[o] = 0 /\ Holders(o) = 0 /\ TagPins(o) = 0)

\* The DEC-UNDERFLOW tripwire.
NoUnderflow == ~underflow

\* Tag pins only ever name the installed wrapper (CommitCAS waits for
\* tag = 0, so a pinned word cannot change under the pin).
TagSane ==
    ~TagXfer =>
        \A t \in Threads : \A v \in VSlots :
            vmode[t][v] = "tag" => vobj[t][v] = link_cur

\* rc bound: every rc unit is backed by a holder OR a transferred pin,
\* so per object rc <= all holders + all pins.  (The first run used a
\* holders-only bound and tripped on a transferred pin -- the bound was
\* wrong, not the protocol; Conservation held throughout.)
TypeOK ==
    /\ rc \in [Objects -> 0..(1 + (NL + NV) * Cardinality(Threads)
                                + Cardinality(Threads)
                                + NV * Cardinality(Threads))]
    /\ freed \in [Objects -> BOOLEAN]
    /\ link_cur \in Objects \cup {Null}
    /\ link_tag \in 0..TagCap
    /\ underflow \in BOOLEAN

=============================================================================
