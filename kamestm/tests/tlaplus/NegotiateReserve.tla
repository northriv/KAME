(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp
 ***************************************************************************)
----------------------------- MODULE NegotiateReserve -----------------------------
(*
 * Single-Linkage negotiate / tag / reserve protocol.
 *
 * WHY A SEPARATE SPEC.  The question this answers is not a tree question:
 * "how many times can a HIGHEST transaction lose one Linkage before it owns
 * it?"  Phases, bundle_serial collisions and hard links do not appear in the
 * claim, and the BundleUnbundle_* family is far too large to sweep the thread
 * count (3L-dyn superfine 2t = 921 M states).  Sweeping T is the whole point:
 * the claim under test has the shape "<= T-1", so one instance proves nothing.
 *
 * THE PROTOCOL, as the C++ actually orders it
 * (ScopedNegotiateLinkage's ctor, kamestm/transaction_negotiation.h):
 *
 *   peer (NORMAL):   observe tag --> acquire view --> weak CAS
 *   HIGHEST (now):   observe tag --> acquire view --> tag --> CAS
 *   HIGHEST (prop):  observe tag --> tag --> acquire view --> reserve --> CAS
 *
 * `observe tag` is negotiate: NegotiationCounter::fair_mode_blocks_me reads
 * the Linkage's m_transaction_started_time slot and yields while a foreign
 * HIGHEST Reserved stamp sits there.  It is checked ONCE, at scope
 * construction.  Everything after it runs on that one entitlement.
 *
 * THE THREE CLAIMS
 *
 *   1. one observation, K CASes.  A peer gets at most K successful CASes per
 *      tag observation and must re-observe (and then blocks) for more.  K is
 *      a CONSTANT because the C++ value is site-dependent: the main bundle
 *      scope (transaction_impl.h:2793) negotiates once and CASes twice --
 *      Phase 2's compareAndSetRetain (:2888) and Phase 4's
 *      compareAndSetWithHint (:3029), with Phase 3 in between -- while every
 *      other scope CASes once.  So K = 2 today and K = 1 is what a
 *      re-negotiate before Phase 4 would buy.  The spec prices that change
 *      instead of asking anyone to reason it out.
 *
 *   2. after the reservation, zero.  Once HIGHEST has published a reservation
 *      no peer CAS succeeds on that Linkage.  This is the claim that
 *      distinguishes the two ways of carrying the reservation, which is why
 *      ReserveMode is a CONSTANT and "sideword" is modelled at all:
 *
 *        "sideword" -- the peer tests the reservation at OBSERVE time, the
 *                      way fair_mode_blocks_me tests the tag word today.  A
 *                      peer that observed before the reservation landed still
 *                      CASes after it.  Expected to VIOLATE INV_NoBadWins.
 *        "invalue"  -- the reservation rides in the PacketWrapper the CAS
 *                      itself compares, so a peer either holds a stale
 *                      pointer (CAS fails) or holds the reserved wrapper and
 *                      can read the flag out of memory it already has.  There
 *                      is no third case and nothing to go stale.  Expected to
 *                      HOLD.
 *
 *      The two runs together are the argument for putting the reservation in
 *      the value rather than in a second word -- a distinction that also
 *      decides whether the design survives a weak memory model.
 *
 *   3. the reserve loop terminates.  HIGHEST's reserve CAS succeeds after at
 *      most Cardinality(Peers)*K losses, because each peer is spent after K
 *      CASes and blocks on its next observation.  Liveness, hence the small
 *      model.
 *
 * AND THE ONE THAT MATTERS FOR REALTIME.  Losses are not equal.  Without a
 * reservation, a peer that wins during HIGHEST's long phase invalidates work
 * already done, and HIGHEST restarts it: hRebuilds.  With the reservation
 * published immediately after the tag, the same losses land on the reserve
 * CAS instead, before any work: hLosses, each costing one re-acquire.  The
 * counts stay equal; the cost does not.  INV_NoRebuild asserts the strong
 * form -- in "invalue" mode the expensive restart never happens at all.
 *)

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Peers,              \* the NORMAL / UI / SCRIPTING threads
    H,                  \* the single HIGHEST thread
    K,                  \* successful CASes a peer may land per observation
    MaxGen,             \* bound on wrapper generations (state-space bound)
    TagBeforeAcquire,   \* TRUE  = tag then acquire (proposed)
                        \* FALSE = acquire then tag (as the C++ ctor orders it)
    ReserveMode,        \* "none" | "sideword" | "invalue"
    Null

ASSUME Cardinality(Peers) >= 1
ASSUME H \notin Peers
ASSUME K \in Nat /\ K >= 1
ASSUME MaxGen \in Nat /\ MaxGen >= 1
ASSUME TagBeforeAcquire \in BOOLEAN
ASSUME ReserveMode \in {"none", "sideword", "invalue"}

Threads == Peers \cup {H}

VARIABLES
    gen,          \* current wrapper generation on the Linkage (the CASed word)
    resAt,        \* [0..MaxGen -> BOOLEAN] — reservation carried BY generation g.
                  \*   Indexed by generation, never a free-standing word: this is
                  \*   what "the reservation is in the value" means, and writing
                  \*   it any other way would model the design we are rejecting.
    tag,          \* the m_transaction_started_time slot: Null or H
    pc,           \* [Threads -> state]
    view,         \* [Threads -> generation each thread holds]
    license,      \* [Peers -> remaining CASes on the current observation]
    seenRes,      \* [Peers -> reservation as seen AT OBSERVE TIME] ("sideword")
    peerWins,     \* successful peer CASes since HIGHEST first observed (unbounded
                  \*   by design -- before the tag lands HIGHEST has claimed
                  \*   nothing, so peers are free; kept as a diagnostic only)
    winsAfterTag, \* successful peer CASes AFTER the tag landed -- the quantity
                  \*   the "T-1" claim is actually about
    badWins,      \* successful peer CASes on an already-reserved generation
    hLosses,      \* HIGHEST's reserve-CAS losses (cheap: re-acquire and retry)
    hRebuilds,    \* HIGHEST's work-CAS losses (expensive: redo the phase)
    hStarted      \* HIGHEST has observed at least once

vars == <<gen, resAt, tag, pc, view, license, seenRes,
          peerWins, winsAfterTag, badWins, hLosses, hRebuilds, hStarted>>

Init ==
    /\ gen       = 0
    /\ resAt     = [g \in 0..MaxGen |-> FALSE]
    /\ tag       = Null
    /\ pc        = [t \in Threads |-> "idle"]
    /\ view      = [t \in Threads |-> 0]
    /\ license   = [p \in Peers |-> 0]
    /\ seenRes   = [p \in Peers |-> FALSE]
    /\ peerWins  = 0
    /\ winsAfterTag = 0
    /\ badWins   = 0
    /\ hLosses   = 0
    /\ hRebuilds = 0
    /\ hStarted  = FALSE

-----------------------------------------------------------------------------
(* Peer (NORMAL) — observe, acquire, one-shot weak CAS.                      *)

\* What the peer is allowed to test before its CAS.  The whole design question
\* is WHICH generation's reservation it gets to see.
PeerMayCAS(p) ==
    CASE ReserveMode = "none"     -> TRUE
      [] ReserveMode = "sideword" -> ~seenRes[p]        \* read at observe time
      [] ReserveMode = "invalue"  -> ~resAt[view[p]]    \* read from the held value
      [] OTHER                    -> TRUE

\* negotiate found a foreign HIGHEST Reserved stamp: yield.  Terminal here —
\* a parked peer can be woken, but it re-tests the same stamp and parks again,
\* so it can never harm HIGHEST once the tag is up.
PObserveBlocked(p) ==
    /\ pc[p] = "idle"
    /\ tag = H
    /\ pc' = [pc EXCEPT ![p] = "blocked"]
    /\ UNCHANGED <<gen, resAt, tag, view, license, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hRebuilds, hStarted>>

PObserveOk(p) ==
    /\ pc[p] = "idle"
    /\ tag # H
    /\ pc'      = [pc      EXCEPT ![p] = "obs"]
    /\ license' = [license EXCEPT ![p] = K]
    /\ seenRes' = [seenRes EXCEPT ![p] = resAt[gen]]
    /\ UNCHANGED <<gen, resAt, tag, view,
                   peerWins, winsAfterTag, badWins, hLosses, hRebuilds, hStarted>>

PAcquire(p) ==
    /\ pc[p] = "obs"
    /\ view' = [view EXCEPT ![p] = gen]
    /\ pc'   = [pc   EXCEPT ![p] = "acq"]
    /\ UNCHANGED <<gen, resAt, tag, license, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hRebuilds, hStarted>>

\* compareAndSetRetain semantics: on success the peer keeps tracking the new
\* value, which is exactly how Phase 2 hands its view to Phase 4.
PCASOk(p) ==
    /\ pc[p] = "acq"
    /\ view[p] = gen
    /\ gen < MaxGen
    /\ license[p] > 0
    /\ PeerMayCAS(p)
    /\ gen'       = gen + 1
    /\ resAt'     = [resAt EXCEPT ![gen + 1] = FALSE]   \* a peer publishes no reservation
    /\ view'      = [view    EXCEPT ![p] = gen + 1]
    /\ license'   = [license EXCEPT ![p] = license[p] - 1]
    /\ pc'        = [pc      EXCEPT ![p] =
                        IF license[p] - 1 > 0 THEN "acq" ELSE "idle"]
    /\ peerWins'     = peerWins     + (IF hStarted   THEN 1 ELSE 0)
    /\ winsAfterTag' = winsAfterTag + (IF tag = H    THEN 1 ELSE 0)
    /\ badWins'   = badWins  + (IF resAt[gen]  THEN 1 ELSE 0)
    /\ UNCHANGED <<tag, seenRes, hLosses, hRebuilds, hStarted>>

\* Lost the CAS (stale pointer) or refused it (reservation visible).  Either
\* way the caller rebuilds a fresh scope, which means a fresh observation.
PCASFail(p) ==
    /\ pc[p] = "acq"
    /\ (view[p] # gen \/ ~PeerMayCAS(p))
    /\ pc'      = [pc      EXCEPT ![p] = "idle"]
    /\ license' = [license EXCEPT ![p] = 0]
    /\ UNCHANGED <<gen, resAt, tag, view, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hRebuilds, hStarted>>

-----------------------------------------------------------------------------
(* HIGHEST — observe, tag, acquire, reserve, work.                           *)

HObserve ==
    /\ pc[H] = "idle"
    /\ pc'      = [pc EXCEPT ![H] = "obs"]
    /\ hStarted' = TRUE
    /\ UNCHANGED <<gen, resAt, tag, view, license, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hRebuilds>>

\* Proposed order: the tag goes down before the view is taken, which also makes
\* i_am_privileged_now true for our own acquire -- the C++ evaluates it at
\* transaction_negotiation.h:488 BEFORE tag_as_contender runs, so today every
\* first touch of a Linkage takes the weak acquire and the weak CAS.
HTagEarly ==
    /\ TagBeforeAcquire
    /\ pc[H] = "obs"
    /\ tag' = H
    /\ pc'  = [pc EXCEPT ![H] = "obs_t"]
    /\ UNCHANGED <<gen, resAt, view, license, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hRebuilds, hStarted>>

HAcquireTagged ==
    /\ pc[H] = "obs_t"
    /\ view' = [view EXCEPT ![H] = gen]
    /\ pc'   = [pc   EXCEPT ![H] = "acq_t"]
    /\ UNCHANGED <<gen, resAt, tag, license, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hRebuilds, hStarted>>

HAcquireUntagged ==
    /\ ~TagBeforeAcquire
    /\ pc[H] = "obs"
    /\ view' = [view EXCEPT ![H] = gen]
    /\ pc'   = [pc   EXCEPT ![H] = "acq_u"]
    /\ UNCHANGED <<gen, resAt, tag, license, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hRebuilds, hStarted>>

\* The gap between "acq_u" and here is the untagged entry -- the author's term
\* (1).  A peer CAS landing in it costs HIGHEST its view.
HTagLate ==
    /\ ~TagBeforeAcquire
    /\ pc[H] = "acq_u"
    /\ tag' = H
    /\ pc'  = [pc EXCEPT ![H] = "acq_t"]
    /\ UNCHANGED <<gen, resAt, view, license, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hRebuilds, hStarted>>

HReserveOk ==
    /\ ReserveMode # "none"
    /\ pc[H] = "acq_t"
    /\ view[H] = gen
    /\ gen < MaxGen
    /\ gen'   = gen + 1
    /\ resAt' = [resAt EXCEPT ![gen + 1] = TRUE]
    /\ view'  = [view  EXCEPT ![H] = gen + 1]
    /\ pc'    = [pc    EXCEPT ![H] = "res"]
    /\ UNCHANGED <<tag, license, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hRebuilds, hStarted>>

\* Cheap: nothing has been built yet, so re-acquire and try again.  The tag is
\* already up, so this does not re-open the observation window.
HReserveFail ==
    /\ ReserveMode # "none"
    /\ pc[H] = "acq_t"
    /\ view[H] # gen
    /\ hLosses' = hLosses + 1
    /\ pc'      = [pc EXCEPT ![H] = "obs_t"]
    /\ UNCHANGED <<gen, resAt, tag, view, license, seenRes,
                   peerWins, winsAfterTag, badWins, hRebuilds, hStarted>>

\* The work CAS after a successful reservation.  Modelled explicitly rather
\* than assumed away: if any peer can still get through, this is where it shows
\* up, as hRebuilds.
HWorkAfterReserveOk ==
    /\ pc[H] = "res"
    /\ view[H] = gen
    /\ pc' = [pc EXCEPT ![H] = "done"]
    /\ UNCHANGED <<gen, resAt, tag, view, license, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hRebuilds, hStarted>>

HWorkAfterReserveFail ==
    /\ pc[H] = "res"
    /\ view[H] # gen
    /\ hRebuilds' = hRebuilds + 1
    /\ pc'        = [pc EXCEPT ![H] = "obs_t"]
    /\ UNCHANGED <<gen, resAt, tag, view, license, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hStarted>>

\* ReserveMode = "none" -- today's protocol.  "work" is bundle Phase 1: a long
\* interval, entered on one entitlement, whose result the Phase 2 CAS publishes.
HWorkStart ==
    /\ ReserveMode = "none"
    /\ pc[H] = "acq_t"
    /\ pc' = [pc EXCEPT ![H] = "work"]
    /\ UNCHANGED <<gen, resAt, tag, view, license, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hRebuilds, hStarted>>

HWorkOk ==
    /\ pc[H] = "work"
    /\ view[H] = gen
    /\ gen < MaxGen
    /\ gen'  = gen + 1
    /\ view' = [view EXCEPT ![H] = gen + 1]
    /\ pc'   = [pc   EXCEPT ![H] = "done"]
    /\ UNCHANGED <<resAt, tag, license, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hRebuilds, hStarted>>

HWorkFail ==
    /\ pc[H] = "work"
    /\ view[H] # gen
    /\ hRebuilds' = hRebuilds + 1
    /\ pc'        = [pc EXCEPT ![H] = "obs"]
    /\ UNCHANGED <<gen, resAt, tag, view, license, seenRes,
                   peerWins, winsAfterTag, badWins, hLosses, hStarted>>

HNext ==
    \/ HObserve \/ HTagEarly \/ HAcquireTagged \/ HAcquireUntagged \/ HTagLate
    \/ HReserveOk \/ HReserveFail
    \/ HWorkAfterReserveOk \/ HWorkAfterReserveFail
    \/ HWorkStart \/ HWorkOk \/ HWorkFail

PNext ==
    \E p \in Peers :
        \/ PObserveBlocked(p) \/ PObserveOk(p) \/ PAcquire(p)
        \/ PCASOk(p) \/ PCASFail(p)

Next == HNext \/ PNext

\* Weak fairness on HIGHEST only.  Peers are deliberately unfair: they may
\* never run.  What must not depend on their cooperation is HIGHEST finishing.
Spec == Init /\ [][Next]_vars /\ WF_vars(HNext)

-----------------------------------------------------------------------------
(* Properties                                                                *)

TypeOK ==
    /\ gen \in 0..MaxGen
    /\ resAt \in [0..MaxGen -> BOOLEAN]
    /\ tag \in {Null, H}
    /\ view \in [Threads -> 0..MaxGen]
    /\ license \in [Peers -> 0..K]
    /\ peerWins \in Nat /\ winsAfterTag \in Nat /\ badWins \in Nat
    /\ hLosses \in Nat /\ hRebuilds \in Nat

\* CLAIM 2.  No peer CAS ever succeeds on a generation that carries HIGHEST's
\* reservation.  Holds for "invalue"; expected to FAIL for "sideword", and
\* that failure is the argument for keeping the reservation in the value.
INV_NoBadWins == badWins = 0

\* CLAIM 1 + 3, the counting form -- and note WHICH window it is about.  TLC
\* refuted the first attempt at this (peerWins <= Cardinality(Peers)*K over the
\* whole run): before the tag lands, PObserveOk stays enabled, so a peer can
\* re-observe and re-arm without limit and the count is unbounded.  That is not
\* a defect -- HIGHEST has claimed nothing yet -- but it means the protocol
\* bounds only what happens AFTER the tag.  From that instant PObserveOk is
\* disabled for everyone, so the only peers left are those already holding a
\* license, and each has at most K CASes on it.
INV_WinsAfterTag == winsAfterTag <= Cardinality(Peers) * K

\* The tighter reading the "3" in 3L assumed -- one interfering peer per
\* Linkage.  Expected to FAIL whenever Cardinality(Peers) > 1: the bound must
\* carry the thread count.
INV_WinsAfterTagOne == winsAfterTag <= 1

\* Same, at K = 1: one win per peer.  Prices the re-negotiate before Phase 4.
INV_WinsAfterTagPerPeer == winsAfterTag <= Cardinality(Peers)

\* THE REALTIME CLAIM.  With the reservation in the value, HIGHEST never has to
\* redo work: every loss lands on the reserve CAS, before anything was built.
INV_NoRebuild == (ReserveMode = "invalue") => (hRebuilds = 0)

\* The losses HIGHEST does take are bounded by the same window.
INV_HLosses == hLosses <= Cardinality(Peers) * K

\* CLAIM 3, the liveness form.
LIVE_Done == <>(pc[H] = "done")

=============================================================================
