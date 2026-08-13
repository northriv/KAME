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
 * ONLY HIGHEST CHANGES.  The peer row is the same in every configuration below
 * -- no peer action reads TagBeforeAcquire or ReserveMode's ordering, and none
 * is meant to.  The proposal is a HIGHEST-side reorder plus a HIGHEST-side
 * reservation; NORMAL keeps doing exactly what it does today, which is also
 * why nothing here asks peers to be well behaved.
 *
 * AND ONLY AT THE FIRST TOUCH.  HIGHEST's view-before-tag order is not the
 * steady state: it is what happens the FIRST time a Snapshot touches a given
 * Linkage.  Every later touch already finds the tag there -- the previous
 * scope's dtor plants it on observed contention
 * (transaction_negotiation.h:1214-1217), Transaction::operator++ plants it on
 * the target before re-snapshotting (transaction.h:2664), and within one
 * Snapshot a planted tag survives until drop_tags_n_privilege.  So the
 * TagBeforeAcquire=FALSE arm models one event per Linkage per Snapshot, not a
 * per-pass cost -- which is exactly the shape of the +1 it turns out to cost.
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
 * NOTHING BEFORE THE TAG IS INTERFERENCE (user, 2026-08-13).  Until HIGHEST's
 * stamp is on the slot it has claimed nothing, so peers running are peers
 * running -- not a loss, not a race, nothing to bound.  The first version of
 * this spec counted peer wins from HIGHEST's first observation and TLC duly
 * refuted the bound: PObserveOk stays enabled until the tag lands, so a peer
 * re-observes and re-arms without limit.  Only winsAfterTag is counted now.
 *
 * NO GENERATION COUNTER.  A bounded one is worse than none: with MaxGen=8 the
 * peers exhausted the range BEFORE the tag landed -- irrelevant activity by
 * the rule above -- and HIGHEST then stuttered forever behind a `gen < MaxGen`
 * guard, which TLC reported as a LIVE_Done violation that says nothing about
 * the protocol.  The counter only ever answered "is my view stale?", so that
 * is what the state holds: one boolean per thread, no bound, and liveness
 * becomes checkable.  A CAS makes every other thread's view stale and keeps
 * the winner's (compareAndSetRetain, which is how Phase 2 hands its view to
 * Phase 4).
 *
 * AND THE ONE THAT MATTERS FOR REALTIME.  Losses are not equal.  Without a
 * reservation, a peer that wins during HIGHEST's long phase invalidates work
 * already done, and HIGHEST restarts it: hRebuilds.  With the reservation
 * published immediately after the tag, the same losses land on the reserve
 * CAS instead, before any work: hLosses, each costing one re-acquire.  The
 * counts stay equal; the cost does not.  INV_NoRebuild asserts the strong
 * form -- in "invalue" mode the expensive restart never happens at all.
 *
 * RESULTS (TLC, exhaustive -- every run below completed with an empty queue).
 *
 *   Peer wins after the tag, mode "none", tag-first:
 *
 *       T  K | <=1        <=T-1      <=(T-1)K
 *       2  1 | HOLDS      HOLDS      HOLDS
 *       2  2 | VIOLATED   VIOLATED   HOLDS
 *       3  1 | VIOLATED   HOLDS      HOLDS
 *       3  2 | VIOLATED   VIOLATED   HOLDS
 *       4  1 | VIOLATED   HOLDS      HOLDS
 *       4  2 | VIOLATED   VIOLATED   HOLDS
 *
 *   (T-1)K is exact, and each column earns its place: "<=1" -- a constant
 *   independent of the thread count, which is what the "3" in 3L was --
 *   survives only the single-peer case, and "<=T-1" survives only at K=1, so
 *   the K in the formula is precisely what a re-negotiate before Phase 4 buys.
 *
 *   Expensive rebuilds per Linkage (the quantity Node::snapshot reports as
 *   snapshot_retries_max), swept over T in {3,4} and K in {1,2}:
 *
 *       HIGHEST acquires then tags            (T-1)K + 1   <- today
 *       HIGHEST tags then acquires            (T-1)K
 *       HIGHEST tags, acquires, reserves      0
 *
 *   -- peers unchanged in all three; the whole difference is HIGHEST-side.
 *
 *   The +1 is the untagged entry, isolated: HIGHEST takes its view before its
 *   stamp is down, so a peer CAS in that gap stales a view already taken.
 *   Reordering two statements in the ctor deletes it.  The reservation deletes
 *   the rest -- INV_NoRebuild holds exhaustively at every T and K tried, which
 *   is the realtime claim: the count that scales with the thread count is the
 *   CHEAP one (a re-acquire and a CAS retry), and the expensive one is zero.
 *
 *   Against the field measurement (T=4, K=2, L=8, four 30-minute soaks giving
 *   snapshot rebuild maxima of 36/39/40/50): today's bound is
 *   ((T-1)K + 1) * L = 56.  All four soaks fall under it.  Every bound tried
 *   by hand before this spec did not: 3L = 24, (T+1)L = 40 (soak 4 = 50).
 *
 *   sideword VIOLATES INV_NoBadWins at every configuration -- 613 distinct
 *   states to the counterexample.  A second word, tested when the peer
 *   negotiates, is stale by the time the peer CASes.  That is the whole
 *   argument for carrying the reservation in the value, and it is also what
 *   makes the design safe under a weak memory model: the reader's acquire on
 *   the pointer is the same acquire that orders the fields behind it.
 *)

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Peers,              \* the NORMAL / UI / SCRIPTING threads
    H,                  \* the single HIGHEST thread
    K,                  \* successful CASes a peer may land per observation
    TagBeforeAcquire,   \* HIGHEST-side only; peers are identical either way.
                        \* TRUE  = HIGHEST tags, then acquires (proposed)
                        \* FALSE = HIGHEST acquires, then tags -- the ctor's
                        \*         order at a Linkage's first touch, where
                        \*         i_am_privileged_now (:488) is evaluated
                        \*         before tag_as_contender (:517) runs
    ReserveMode,        \* "none" | "sideword" | "invalue"
    Null

ASSUME Cardinality(Peers) >= 1
ASSUME H \notin Peers
ASSUME K \in Nat /\ K >= 1
ASSUME TagBeforeAcquire \in BOOLEAN
ASSUME ReserveMode \in {"none", "sideword", "invalue"}

Threads == Peers \cup {H}

VARIABLES
    reserved,     \* does the CURRENT value on the Linkage carry HIGHEST's
                  \*   reservation?  One boolean, and it may only ever change in
                  \*   the same action that changes the value -- that identity is
                  \*   what "the reservation is in the value" means, and writing
                  \*   it as an independently updatable word would model exactly
                  \*   the design being rejected.
    stale,        \* [Threads -> BOOLEAN] — has the value moved since I acquired?
                  \*   Replaces a generation counter; see the header.
    tag,          \* the m_transaction_started_time slot: Null or H
    pc,           \* [Threads -> state]
    license,      \* [Peers -> remaining CASes on the current observation]
    seenRes,      \* [Peers -> reservation as seen AT OBSERVE TIME] ("sideword")
    winsAfterTag, \* successful peer CASes after the tag landed -- the only
                  \*   interference the protocol is answerable for
    badWins,      \* successful peer CASes against a live reservation
    hLosses,      \* HIGHEST's reserve-CAS losses (cheap: re-acquire and retry)
    hRebuilds     \* HIGHEST's work-CAS losses (expensive: redo the phase)

vars == <<reserved, stale, tag, pc, license, seenRes,
          winsAfterTag, badWins, hLosses, hRebuilds>>

\* A successful CAS by `w`: the new value is published, so every other thread's
\* view is stale.  The winner keeps tracking it (compareAndSetRetain).
Publish(w) == [t \in Threads |-> t # w]

Init ==
    /\ reserved     = FALSE
    /\ stale        = [t \in Threads |-> FALSE]
    /\ tag          = Null
    /\ pc           = [t \in Threads |-> "idle"]
    /\ license      = [p \in Peers |-> 0]
    /\ seenRes      = [p \in Peers |-> FALSE]
    /\ winsAfterTag = 0
    /\ badWins      = 0
    /\ hLosses      = 0
    /\ hRebuilds    = 0

-----------------------------------------------------------------------------
(* Peer (NORMAL) — observe, acquire, one-shot weak CAS.                      *)

\* The design question in one operator: WHICH reservation does the peer get to
\* test?  "sideword" gives it the one it read at observe time, the way
\* fair_mode_blocks_me reads the tag today -- already stale by the time it CASes.
\* "invalue" gives it the one attached to the value it holds, and since a CAS
\* can only succeed on a non-stale view, that is the current one by construction.
PeerMayCAS(p) ==
    CASE ReserveMode = "none"     -> TRUE
      [] ReserveMode = "sideword" -> ~seenRes[p]
      [] ReserveMode = "invalue"  -> ~reserved
      [] OTHER                    -> TRUE

\* negotiate found a foreign HIGHEST Reserved stamp: yield.  Terminal here — a
\* parked peer can be woken, but it re-tests the same stamp and parks again, so
\* it can never harm HIGHEST once the tag is up.
PObserveBlocked(p) ==
    /\ pc[p] = "idle"
    /\ tag = H
    /\ pc' = [pc EXCEPT ![p] = "blocked"]
    /\ UNCHANGED <<reserved, stale, tag, license, seenRes,
                   winsAfterTag, badWins, hLosses, hRebuilds>>

PObserveOk(p) ==
    /\ pc[p] = "idle"
    /\ tag # H
    /\ pc'      = [pc      EXCEPT ![p] = "obs"]
    /\ license' = [license EXCEPT ![p] = K]
    /\ seenRes' = [seenRes EXCEPT ![p] = reserved]
    /\ UNCHANGED <<reserved, stale, tag,
                   winsAfterTag, badWins, hLosses, hRebuilds>>

PAcquire(p) ==
    /\ pc[p] = "obs"
    /\ stale' = [stale EXCEPT ![p] = FALSE]
    /\ pc'    = [pc    EXCEPT ![p] = "acq"]
    /\ UNCHANGED <<reserved, tag, license, seenRes,
                   winsAfterTag, badWins, hLosses, hRebuilds>>

PCASOk(p) ==
    /\ pc[p] = "acq"
    /\ ~stale[p]
    /\ license[p] > 0
    /\ PeerMayCAS(p)
    /\ badWins'      = badWins + (IF reserved THEN 1 ELSE 0)
    /\ reserved'     = FALSE                       \* a peer publishes no reservation
    /\ stale'        = Publish(p)
    /\ license'      = [license EXCEPT ![p] = license[p] - 1]
    /\ pc'           = [pc      EXCEPT ![p] =
                          IF license[p] - 1 > 0 THEN "acq" ELSE "idle"]
    /\ winsAfterTag' = winsAfterTag + (IF tag = H THEN 1 ELSE 0)
    /\ UNCHANGED <<tag, seenRes, hLosses, hRebuilds>>

\* Lost the CAS (stale view) or refused it (reservation visible).  Either way the
\* caller builds a fresh scope, and a fresh scope is a fresh observation.
PCASFail(p) ==
    /\ pc[p] = "acq"
    /\ (stale[p] \/ ~PeerMayCAS(p))
    /\ pc'      = [pc      EXCEPT ![p] = "idle"]
    /\ license' = [license EXCEPT ![p] = 0]
    /\ UNCHANGED <<reserved, stale, tag, seenRes,
                   winsAfterTag, badWins, hLosses, hRebuilds>>

-----------------------------------------------------------------------------
(* HIGHEST — observe, tag, acquire, reserve, work.                           *)

HObserve ==
    /\ pc[H] = "idle"
    /\ pc' = [pc EXCEPT ![H] = "obs"]
    /\ UNCHANGED <<reserved, stale, tag, license, seenRes,
                   winsAfterTag, badWins, hLosses, hRebuilds>>

\* Proposed order: the tag goes down before the view is taken.  It also makes
\* i_am_privileged_now true for our own acquire -- the C++ evaluates that at
\* transaction_negotiation.h:488, BEFORE tag_as_contender runs, so today every
\* first touch of a Linkage takes the weak acquire and the weak CAS.
HTagEarly ==
    /\ TagBeforeAcquire
    /\ pc[H] = "obs"
    /\ tag' = H
    /\ pc'  = [pc EXCEPT ![H] = "obs_t"]
    /\ UNCHANGED <<reserved, stale, license, seenRes,
                   winsAfterTag, badWins, hLosses, hRebuilds>>

HAcquireTagged ==
    /\ pc[H] = "obs_t"
    /\ stale' = [stale EXCEPT ![H] = FALSE]
    /\ pc'    = [pc    EXCEPT ![H] = "acq_t"]
    /\ UNCHANGED <<reserved, tag, license, seenRes,
                   winsAfterTag, badWins, hLosses, hRebuilds>>

HAcquireUntagged ==
    /\ ~TagBeforeAcquire
    /\ pc[H] = "obs"
    /\ stale' = [stale EXCEPT ![H] = FALSE]
    /\ pc'    = [pc    EXCEPT ![H] = "acq_u"]
    /\ UNCHANGED <<reserved, tag, license, seenRes,
                   winsAfterTag, badWins, hLosses, hRebuilds>>

\* The gap between "acq_u" and here is the untagged entry -- the author's term
\* (1), and the one that tag-before-acquire deletes.
HTagLate ==
    /\ ~TagBeforeAcquire
    /\ pc[H] = "acq_u"
    /\ tag' = H
    /\ pc'  = [pc EXCEPT ![H] = "acq_t"]
    /\ UNCHANGED <<reserved, stale, license, seenRes,
                   winsAfterTag, badWins, hLosses, hRebuilds>>

HReserveOk ==
    /\ ReserveMode # "none"
    /\ pc[H] = "acq_t"
    /\ ~stale[H]
    /\ reserved' = TRUE
    /\ stale'    = Publish(H)
    /\ pc'       = [pc EXCEPT ![H] = "res"]
    /\ UNCHANGED <<tag, license, seenRes,
                   winsAfterTag, badWins, hLosses, hRebuilds>>

\* Cheap: nothing has been built yet, so re-acquire and try again.  The tag is
\* already up, so this does not re-open the observation window.
HReserveFail ==
    /\ ReserveMode # "none"
    /\ pc[H] = "acq_t"
    /\ stale[H]
    /\ hLosses' = hLosses + 1
    /\ pc'      = [pc EXCEPT ![H] = "obs_t"]
    /\ UNCHANGED <<reserved, stale, tag, license, seenRes,
                   winsAfterTag, badWins, hRebuilds>>

\* The work CAS after a successful reservation, modelled rather than assumed
\* away: if any peer can still get through, it shows up here as hRebuilds.
HWorkAfterReserveOk ==
    /\ pc[H] = "res"
    /\ ~stale[H]
    /\ pc' = [pc EXCEPT ![H] = "done"]
    /\ UNCHANGED <<reserved, stale, tag, license, seenRes,
                   winsAfterTag, badWins, hLosses, hRebuilds>>

HWorkAfterReserveFail ==
    /\ pc[H] = "res"
    /\ stale[H]
    /\ hRebuilds' = hRebuilds + 1
    /\ pc'        = [pc EXCEPT ![H] = "obs_t"]
    /\ UNCHANGED <<reserved, stale, tag, license, seenRes,
                   winsAfterTag, badWins, hLosses>>

\* ReserveMode = "none" -- today's protocol.  "work" is bundle Phase 1: a long
\* interval entered on one entitlement, whose result the Phase 2 CAS publishes.
HWorkStart ==
    /\ ReserveMode = "none"
    /\ pc[H] = "acq_t"
    /\ pc' = [pc EXCEPT ![H] = "work"]
    /\ UNCHANGED <<reserved, stale, tag, license, seenRes,
                   winsAfterTag, badWins, hLosses, hRebuilds>>

HWorkOk ==
    /\ pc[H] = "work"
    /\ ~stale[H]
    /\ stale' = Publish(H)
    /\ pc'    = [pc EXCEPT ![H] = "done"]
    /\ UNCHANGED <<reserved, tag, license, seenRes,
                   winsAfterTag, badWins, hLosses, hRebuilds>>

HWorkFail ==
    /\ pc[H] = "work"
    /\ stale[H]
    /\ hRebuilds' = hRebuilds + 1
    /\ pc'        = [pc EXCEPT ![H] = "obs"]
    /\ UNCHANGED <<reserved, stale, tag, license, seenRes,
                   winsAfterTag, badWins, hLosses>>

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

\* Weak fairness on HIGHEST only.  Peers are deliberately unfair: they may never
\* run.  What must not depend on their cooperation is HIGHEST finishing.
Spec == Init /\ [][Next]_vars /\ WF_vars(HNext)

-----------------------------------------------------------------------------
(* Properties                                                                *)

TypeOK ==
    /\ reserved \in BOOLEAN
    /\ stale \in [Threads -> BOOLEAN]
    /\ tag \in {Null, H}
    /\ license \in [Peers -> 0..K]
    /\ winsAfterTag \in Nat /\ badWins \in Nat
    /\ hLosses \in Nat /\ hRebuilds \in Nat

\* CLAIM 2.  No peer CAS ever succeeds against a live reservation.  Holds for
\* "invalue"; FAILS for "sideword", and that counterexample is the argument for
\* keeping the reservation in the value rather than in a second word.
INV_NoBadWins == badWins = 0

\* CLAIM 1 + 3, the counting form, and note WHICH window it is about.  Nothing
\* before the tag counts: HIGHEST has claimed nothing, so peers running are just
\* peers running.  From the tag onward PObserveOk is disabled for everyone, so
\* the only peers left are those already holding a license, each with at most K
\* CASes on it.
INV_WinsAfterTag == winsAfterTag <= Cardinality(Peers) * K

\* The reading the "3" in 3L assumed -- one interfering peer per Linkage.
\* Expected to FAIL once Cardinality(Peers) > 1: the bound must carry T.
INV_WinsAfterTagOne == winsAfterTag <= 1

\* At K = 1, one win per peer.  Prices the re-negotiate before Phase 4.
INV_WinsAfterTagPerPeer == winsAfterTag <= Cardinality(Peers)

\* THE REALTIME CLAIM.  With the reservation in the value, HIGHEST never redoes
\* work: every loss lands on the reserve CAS, before anything was built.
INV_NoRebuild == (ReserveMode = "invalue") => (hRebuilds = 0)

\* The losses HIGHEST does take are bounded by the same window.
INV_HLosses == hLosses <= Cardinality(Peers) * K

\* The EXPENSIVE count -- the one Node::snapshot's retry loop reports as
\* snapshot_retries_max.  Probed at three tightness levels so the sweep can
\* report the exact maximum rather than a bound someone guessed.
INV_Rebuilds_LE_0  == hRebuilds = 0
INV_Rebuilds_LE_P  == hRebuilds <= Cardinality(Peers)
INV_Rebuilds_LE_PK == hRebuilds <= Cardinality(Peers) * K
\* Tag-late costs exactly one more: HIGHEST acquires before its stamp is down,
\* so a peer CAS in that gap staled a view it had already taken.  The author's
\* term (1), isolated.
INV_Rebuilds_LE_PK1 == hRebuilds <= Cardinality(Peers) * K + 1

\* CLAIM 3, the liveness form: the reserve loop terminates.
LIVE_Done == <>(pc[H] = "done")

=============================================================================
