---------------------------- MODULE PacketRefcount ----------------------------
(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp
        Dual-licensed: Apache-2.0 OR GPL-2.0-or-later (see the LICENSE
        files beside this directory).
 ***************************************************************************)
(*
 * PACKET-LEVEL REFERENCE OWNERSHIP across bundle / unbundle / snapshot.
 *
 * Why this model exists (DYNNODE_UAF_HANDOFF.md §13.63).  The empirical
 * hunt converged on a single fact: a `Packet` reaches refcount ZERO while
 * a live `ScopedNegotiateLinkage` still reaches it, and the next site that
 * copies that `local_shared_ptr<Packet>` resurrects it (captured at
 * `bundle:2851`'s super-wrapper construction and at `bundle:2870`'s
 * PacketList copy).  Every previous TLA+ layer abstracts packets as VALUES
 * and cannot express this: none of them tracks a packet's refcount.
 *
 * The observation is logically over-determined, which is what makes it
 * modelable.  A scope pins its PacketWrapper; a live wrapper's `m_packet`
 * is a counted reference; so `rc[wrapper.m_packet] >= 1` should be a
 * theorem.  Observing 0 means exactly one of three things:
 *
 *   (1) LIFETIME    the scope does not actually keep its wrapper alive
 *                   across a concurrent CAS (view consumed / not restored);
 *   (2) DOUBLE      the packet was released through a second, uncounted
 *                   path (two wrappers sharing one count, or a release
 *                   without ownership);
 *   (3) UNCOUNTED   the wrapper's m_packet was installed WITHOUT taking a
 *                   reference (an aliasing / move-in that skips the +1).
 *
 * This model encodes the protocol faithfully and offers each of the three
 * as a BUG KNOB, so TLC both (a) verifies the faithful protocol and
 * (b) proves the invariant has teeth against each candidate.  A faithful
 * PASS means the C++ departs from the protocol somewhere (the same verdict
 * shape as the §13 scope-token model); a faithful FAIL hands over the
 * interleaving.
 *
 * Topology (matching the reproducer, which carries a hard link by
 * construction -- `p2` under `gn2`):
 *
 *        Root
 *        /  \
 *       A    B      A and B are children of Root
 *        \  /
 *         C         C is hard-linked: child of BOTH A and B
 *
 * Abstractions, stated so fidelity is auditable:
 *   - One packet identity per node ("the packet currently at this node").
 *     Packet CLONING (copy-on-write) is modelled as taking a fresh
 *     reference on the same identity, because the question is ownership
 *     accounting, not value flow.
 *   - `rc[p]` counts holders explicitly; `holders[p]` names them, so a
 *     violation is reported with the holder set intact.
 *   - A wrapper is a record: its node, whether it is alive, and which
 *     packet its `m_packet` names (or Null).
 *   - Scopes: a thread's scope holds a wrapper; `viewHeld` records whether
 *     the view still protects it (a consuming CAS clears it, `set_view`
 *     restores it -- see `transaction_negotiation.h:877`).
 *)

EXTENDS Integers, FiniteSets, Sequences, TLC

CONSTANTS
    Threads,                 \* set of thread ids
    Root, A, B, C,           \* nodes
    Null,
    (* Bug knobs -- all FALSE in the faithful configuration. *)
    BUG_LIFETIME,            \* (1) CAS consumes the view and code keeps deref'ing
    BUG_DOUBLE,              \* (2) allow a release without owning a reference
    BUG_UNCOUNTED,           \* (3) wrapper install skips the +1 on m_packet
    (* Topology switch.  TRUE = C is hard-linked (child of BOTH A and B,
     * the reproducer's shape).  FALSE = the ordinary tree.  Present so any
     * proposed FIX can be checked in both topologies: the user's constraint
     * is that a hard-link fix must not regress the non-hard-link path, and
     * that is a structural claim this model can settle. *)
    HardLink,
    (* Which nodes threads may bundle.  The faithful two-scope model's
     * state space grows fast (wrappers now live longer), so a config may
     * restrict the actors without weakening the question: the mechanism
     * under test is local to one linkage plus its child, so {Root, C} is
     * the smallest set that still contains a super/sub pair and the
     * hard-linked node.  Full {Root,A,B,C} runs are kept as the wider
     * (possibly non-exhaustive) arm. *)
    ScopeNodes

Nodes    == {Root, A, B, C}
Packets  == {"pRoot", "pA", "pB", "pC"}
PacketOf == [n \in Nodes |-> CASE n = Root -> "pRoot"
                               [] n = A    -> "pA"
                               [] n = B    -> "pB"
                               [] OTHER    -> "pC"]

\* Wrapper ids: one per (node, generation).  Two generations suffice to
\* express "a CAS replaced the wrapper while a scope viewed the old one".
Gens     == {0, 1}
Wrappers == Nodes \X Gens

VARIABLES
    rc,          \* [Packets -> Nat]            counted references
    holders,     \* [Packets -> SUBSET Holder]  who holds them (diagnostic)
    wAlive,      \* [Wrappers -> BOOLEAN]
    wPacket,     \* [Wrappers -> Packets \cup {Null}]
    wrc,         \* [Wrappers -> Nat]           wrapper's own refcount
    link,        \* [Nodes -> Wrappers \cup {Null}]  the published wrapper
    scopeW,      \* [Threads -> Wrappers \cup {Null}]  the OUTER `supscope`
    scopeIn,     \* [Threads -> Wrappers \cup {Null}]  the INNER `scope`
    viewHeld,    \* [Threads -> BOOLEAN]  does the OUTER view still protect it?
    pc,          \* [Threads -> String]
    done,        \* [Threads -> BOOLEAN]  one bundle per thread (finiteness
                 \*   by PRECONDITION, mirroring the hard-link models'
                 \*   `bundleDone` -- this project does not use
                 \*   StateConstraint for structural bounding)
    deadRead     \* BOOLEAN: a thread read a packet through a live path
                 \*          whose rc was 0 (the §13.63 observation)

vars == <<rc, holders, wAlive, wPacket, wrc, link, scopeW, scopeIn,
          viewHeld, pc, done, deadRead>>

\* Holder identities, so a counterexample names WHO held what.
HolderW(w) == <<"wrapper", w>>
HolderS(t) == <<"super", t>>
HolderL(t) == <<"listcopy", t>>

--------------------------------------------------------------------------
(* Reference-counting helpers.  Every increment names its holder; every
 * decrement removes one.  Destruction cascades exactly as the C++ does:
 * a wrapper reaching 0 runs ~PacketWrapper, which releases m_packet. *)

Inc(p, h) == /\ rc'      = [rc      EXCEPT ![p] = @ + 1]
             /\ holders' = [holders EXCEPT ![p] = @ \cup {h}]

Dec(p, h) == /\ rc'      = [rc      EXCEPT ![p] = IF @ > 0 THEN @ - 1 ELSE 0]
             /\ holders' = [holders EXCEPT ![p] = @ \ {h}]

--------------------------------------------------------------------------
Init ==
    /\ rc       = [p \in Packets |-> IF p = "pC" /\ HardLink THEN 2 ELSE 1]
       (* pC starts with 2: it is hard-linked, so BOTH A's and B's
          wrappers name it (one via a list slot, one as the live packet) --
          the topology's defining property. *)
    /\ holders  = [p \in Packets |->
                     IF p = "pC" /\ HardLink
                     THEN {HolderW(<<A,0>>), HolderW(<<B,0>>)}
                     ELSE {HolderW(<<CHOOSE n \in Nodes : PacketOf[n] = p, 0>>)}]
    /\ wAlive   = [w \in Wrappers |-> w[2] = 0]
    /\ wPacket  = [w \in Wrappers |-> IF w[2] = 0 THEN PacketOf[w[1]] ELSE Null]
    /\ wrc      = [w \in Wrappers |-> IF w[2] = 0 THEN 1 ELSE 0]
    /\ link     = [n \in Nodes |-> <<n, 0>>]
    /\ scopeW   = [t \in Threads |-> Null]
    /\ scopeIn  = [t \in Threads |-> Null]
    /\ viewHeld = [t \in Threads |-> FALSE]
    /\ pc       = [t \in Threads |-> "idle"]
    /\ done     = [t \in Threads |-> FALSE]
    /\ deadRead = FALSE

--------------------------------------------------------------------------
(* ScopedNegotiateLinkage ctor: acquire a view of the node's published
 * wrapper.  The view takes a reference on the WRAPPER (tag or owned --
 * the §13 scope-token model verified that both keep it alive), which is
 * what is supposed to make `scope->packet()` safe. *)
ScopeAcquire(t, n) ==
    /\ pc[t] = "idle"
    /\ ~done[t]
    /\ link[n] # Null
    /\ wAlive[link[n]]
    /\ scopeW'   = [scopeW   EXCEPT ![t] = link[n]]
    /\ viewHeld' = [viewHeld EXCEPT ![t] = TRUE]
    /\ wrc'      = [wrc      EXCEPT ![link[n]] = @ + 1]
    /\ pc'       = [pc       EXCEPT ![t] = "scoped"]
    /\ UNCHANGED <<rc, holders, wAlive, wPacket, link, scopeIn, done, deadRead>>

(* transaction_impl.h:2851 -- the serial-tag block constructs a SECOND
 * scope on the SAME linkage: `ScopedNegotiateLinkage scope(supernode.m_link,
 * …)`.  The pointer check at :2875 (`scope.operator->() != supscope
 * .operator->()` -> DISTURBED) means the CAS below only proceeds when BOTH
 * scopes view the same wrapper -- so that wrapper carries TWO view
 * references plus the linkage's.  Modelling this was the gap in §13.64's
 * idealisation, which had one view per thread. *)
AcquireInner(t) ==
    /\ pc[t] = "scoped"
    /\ scopeW[t] # Null
    /\ scopeIn[t] = Null
    /\ viewHeld[t]
    /\ LET n == scopeW[t][1] IN
       /\ link[n] = scopeW[t]            \* the :2875 pointer check
       /\ wAlive[link[n]]
       /\ scopeIn' = [scopeIn EXCEPT ![t] = link[n]]
       /\ wrc'     = [wrc     EXCEPT ![link[n]] = @ + 1]
    /\ UNCHANGED <<rc, holders, wAlive, wPacket, link, scopeW, viewHeld,
                   pc, done, deadRead>>

(* bundle:2851 -- `make_local_shared<PacketWrapper>(supscope->packet(), …)`
 * reads the packet THROUGH the scope's wrapper and copies the pointer.
 * This is the site §13.63 caught incrementing a zero-count Packet: the
 * deadRead flag records exactly that observation. *)
SuperWrapperCopy(t) ==
    /\ pc[t] = "scoped"
    /\ scopeW[t] # Null
    \* The `superwrapper` local exists once per bundle attempt and is
    \* destroyed at the end of it, so a thread never holds two at once.
    \* (Also keeps holder identities unique, which CountMatchesHolders needs.)
    /\ \A q \in Packets : HolderS(t) \notin holders[q]
    /\ LET w == scopeW[t] IN
       /\ (viewHeld[t] \/ BUG_LIFETIME)   \* faithful code needs a held view
       /\ wPacket[w] # Null
       /\ LET p == wPacket[w] IN
          /\ deadRead' = (deadRead \/ rc[p] = 0)   \* §13.63's signature
          /\ Inc(p, HolderS(t))
             \* (BUG_UNCOUNTED acts at the INSTALL, in CommitCAS -- putting
             \*  it here instead merely starves the protocol of commits,
             \*  which is how its first formulation came out toothless.)
    /\ pc' = [pc EXCEPT ![t] = "phase1"]
    /\ UNCHANGED <<wAlive, wPacket, wrc, link, scopeW, scopeIn, viewHeld, done>>

(* bundle:2870 -- the PacketList copy: increments every element.  Modelled
 * on the hard-linked child, the element the captures name. *)
Phase1ListCopy(t) ==
    /\ pc[t] = "phase1"
    /\ HolderL(t) \notin holders["pC"]      \* one list copy per attempt
    /\ deadRead' = (deadRead \/ rc["pC"] = 0)
    /\ Inc("pC", HolderL(t))
    /\ pc' = [pc EXCEPT ![t] = "committing"]
    /\ UNCHANGED <<wAlive, wPacket, wrc, link, scopeW, scopeIn, viewHeld, done>>

(* The commit CAS: publish a fresh (generation-1) wrapper at the node.
 * The linkage's reference moves from the old wrapper to the new one; the
 * old wrapper may then die, and ~PacketWrapper releases its m_packet.
 * For TagHeld views the CAS CONSUMES the view (`m_pref` reset) -- the
 * faithful code either restores it via set_view or stops dereferencing. *)
CommitCAS(t) ==
    /\ pc[t] = "committing"
    /\ scopeW[t] # Null
    /\ scopeIn[t] = scopeW[t]     \* :2875 established both view one wrapper
    \* The wrapper being published IS the `superwrapper` this thread built,
    \* so the thread's local reference on its packet TRANSFERS into the
    \* installed wrapper (a move/copy into the published object, not a new
    \* increment).  Modelled as a holder RENAME with rc unchanged -- making
    \* the transfer explicit is what keeps the accounting auditable.
    /\ \E p \in Packets : HolderS(t) \in holders[p] /\
       LET old == scopeW[t]
           n   == old[1]
           new == <<n, 1>>
           dies == (IF BUG_LIFETIME THEN wrc[old] - 3 <= 0
                                    ELSE wrc[old] - 2 <= 0)
       IN /\ link[n] = old            \* CAS precondition: unchanged
          /\ ~wAlive[new]             \* one generation-1 publish per node
          /\ wAlive'  = [wAlive  EXCEPT ![new] = TRUE,
                          ![old] = IF dies THEN FALSE ELSE @]
          /\ wPacket' = [wPacket EXCEPT ![new] = p,
                          ![old] = IF dies THEN Null ELSE @]
          \* The CAS drops the LINKAGE's reference (it now names `new`) and
          \* CONSUMES the inner scope's view (TagHeld -> Empty), i.e. two of
          \* old's references go away in this one step.  The outer view is
          \* released separately, by set_view (SetViewRestore below) -- which
          \* is where the C++ actually decides old's fate.
          \* new's refcount is TWO: the linkage's (taken by the CAS) plus the
          \* caller's `superwrapper` local, which `set_view` will hand to the
          \* view by zero-atomic transfer (assign_from_local).  Modelling
          \* only the linkage's made new die one release early -- a MODEL
          \* bug TLC duly reported as a protocol violation (see §13.65).
          /\ wrc'  = [wrc  EXCEPT ![new] = 2,
                       ![old] = LET k == IF BUG_LIFETIME THEN 3 ELSE 2
                                IN IF @ >= k THEN @ - k ELSE 0]
          /\ link' = [link EXCEPT ![n] = new]
          \* Transfer, then (if the old wrapper died) ~PacketWrapper
          \* releases ITS m_packet.  Both edits land in one step.
          \* BUG_UNCOUNTED: install the wrapper's m_packet WITHOUT the
          \* reference (an aliasing / move-in that skips the +1), i.e.
          \* candidate (3) -- the published wrapper names a packet it does
          \* not own, and rc drops by one at the transfer.
          /\ LET h1 == IF BUG_UNCOUNTED
                       THEN [holders EXCEPT ![p] = @ \ {HolderS(t)}]
                       ELSE [holders EXCEPT ![p] = (@ \ {HolderS(t)})
                                              \cup {HolderW(new)}]
             IN IF dies /\ wPacket[old] # Null
                   THEN /\ holders' = [h1 EXCEPT ![wPacket[old]] =
                                          @ \ {HolderW(old)}]
                        /\ rc' = [rc EXCEPT ![wPacket[old]] =
                                      IF @ > 0 THEN @ - 1 ELSE 0,
                                  ![p] = IF BUG_UNCOUNTED /\ @ > 0
                                         THEN @ - 1 ELSE @]
                   ELSE /\ holders' = h1
                        /\ rc' = [rc EXCEPT ![p] =
                                      IF BUG_UNCOUNTED /\ @ > 0
                                      THEN @ - 1 ELSE @]
          \* The CAS consumes a TagHeld view.  FAITHFUL: the scope keeps
          \* its wrapper reference (the token layer transfers the tag into
          \* the global count -- verified by the §13 scope-token model), so
          \* only the flag clears.  BUG_LIFETIME models the hypothesis that
          \* the consumption also drops the scope's protection, which is
          \* what would let a dereference reach a dead wrapper.
          /\ viewHeld' = [viewHeld EXCEPT ![t] = FALSE]
          /\ scopeIn'  = [scopeIn  EXCEPT ![t] = Null]   \* view consumed
    /\ pc' = [pc EXCEPT ![t] = "committed"]
    /\ UNCHANGED <<scopeW, done, deadRead>>

(* set_view(std::move(newwrapper)) -- restores the view after a consuming
 * CAS so the rest of the function may keep dereferencing (see
 * transaction_negotiation.h:877 / bundle_subpacket's W_NEW_SUBVALUE). *)
RestoreView(t) ==
    /\ pc[t] = "committed"
    /\ scopeW[t] # Null
    /\ LET oldv == scopeW[t]          \* the view set_view is about to drop
           n    == oldv[1]
           dies == (wrc[oldv] - 1 <= 0)
       IN /\ link[n] # Null
          /\ wAlive[link[n]]
          \* set_view = assign_from_local: RELEASE the old view, then move in
          \* the new wrapper (zero-atomic transfer of the caller's +1).  The
          \* release is what can kill the old wrapper -- and ~PacketWrapper
          \* then drops ITS m_packet, which is the reference `supscope->
          \* packet()` was reached through moments earlier.
          /\ wrc' = [wrc EXCEPT ![oldv] = IF @ > 0 THEN @ - 1 ELSE 0]
          /\ wAlive'  = [wAlive  EXCEPT ![oldv] = IF dies THEN FALSE ELSE @]
          /\ wPacket' = [wPacket EXCEPT ![oldv] = IF dies THEN Null ELSE @]
          /\ IF dies /\ wPacket[oldv] # Null
                THEN Dec(wPacket[oldv], HolderW(oldv))
                ELSE UNCHANGED <<rc, holders, done>>
          \* Move-in is ZERO-ATOMIC: the local's reference (counted at the
          \* CAS above) becomes the view's.  No increment here.
          /\ scopeW'   = [scopeW   EXCEPT ![t] = link[n]]
          /\ viewHeld' = [viewHeld EXCEPT ![t] = TRUE]
    /\ pc' = [pc EXCEPT ![t] = "scoped"]     \* may read again (Phase 1 etc.)
    /\ UNCHANGED <<link, scopeIn, done, deadRead>>

(* Dereference `scope->packet()` through a view a consuming CAS already
 * cleared -- i.e. the code failed to `set_view` (or to stop reading) after
 * the CAS.  This is the ACTION candidate (1) needs in order to be
 * expressible at all: without it, a cleared view is never used and the
 * knob cannot have teeth.  Faithful runs never take this step. *)
DerefStaleView(t) ==
    /\ BUG_LIFETIME
    /\ pc[t] = "committed"
    /\ scopeW[t] # Null
    /\ ~viewHeld[t]
    /\ LET w == scopeW[t] IN
       /\ deadRead' = (deadRead \/ ~wAlive[w]
                       \/ (wPacket[w] # Null /\ rc[wPacket[w]] = 0))
    /\ UNCHANGED <<rc, holders, wAlive, wPacket, wrc, link, scopeW,
                   scopeIn, viewHeld, pc, done>>

(* Scope destructor: drop the wrapper reference; a wrapper reaching zero
 * runs ~PacketWrapper and releases m_packet. *)
ScopeRelease(t) ==
    /\ pc[t] \in {"scoped", "phase1", "committing", "committed"}
    /\ scopeW[t] # Null
    /\ LET w == scopeW[t] IN
       /\ wrc' = [wrc EXCEPT ![w] = IF @ > 0 THEN @ - 1 ELSE 0]
       /\ wAlive' = [wAlive EXCEPT ![w] = IF wrc[w] - 1 <= 0 THEN FALSE ELSE @]
       /\ wPacket' = [wPacket EXCEPT ![w] = IF wrc[w] - 1 <= 0 THEN Null ELSE @]
       /\ IF wrc[w] - 1 <= 0 /\ wPacket[w] # Null
             THEN Dec(wPacket[w], HolderW(w))
             ELSE UNCHANGED <<rc, holders>>
    /\ scopeW'   = [scopeW   EXCEPT ![t] = Null]
    /\ scopeIn'  = [scopeIn  EXCEPT ![t] = Null]
    /\ viewHeld' = [viewHeld EXCEPT ![t] = FALSE]
    /\ done'     = [done     EXCEPT ![t] = TRUE]
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<link, done, deadRead>>

(* The bundle's own locals go out of scope: superwrapper + the list copy. *)
DropLocals(t) ==
    /\ pc[t] \in {"committed", "scoped"}
    /\ \E p \in Packets :
         /\ HolderS(t) \in holders[p] \/ HolderL(t) \in holders[p]
         /\ IF HolderS(t) \in holders[p] THEN Dec(p, HolderS(t))
                                        ELSE Dec(p, HolderL(t))
    /\ UNCHANGED <<wAlive, wPacket, wrc, link, scopeW, scopeIn, viewHeld, pc, done, deadRead>>

(* Bug knob (2): a release performed without owning a reference -- the
 * double-release candidate.  Faithful runs never enable this. *)
UncountedRelease(t) ==
    /\ BUG_DOUBLE
    /\ \E p \in Packets : rc[p] > 0 /\ Dec(p, HolderS(t))
    /\ UNCHANGED <<wAlive, wPacket, wrc, link, scopeW, scopeIn, viewHeld, pc, done, deadRead>>

Next ==
    \/ \E t \in Threads, n \in ScopeNodes : ScopeAcquire(t, n)
    \/ \E t \in Threads : AcquireInner(t)
    \/ \E t \in Threads : SuperWrapperCopy(t)
    \/ \E t \in Threads : Phase1ListCopy(t)
    \/ \E t \in Threads : CommitCAS(t)
    \/ \E t \in Threads : RestoreView(t)
    \/ \E t \in Threads : DerefStaleView(t)
    \/ \E t \in Threads : ScopeRelease(t)
    \/ \E t \in Threads : DropLocals(t)
    \/ \E t \in Threads : UncountedRelease(t)

Spec == Init /\ [][Next]_vars

--------------------------------------------------------------------------
(* INVARIANTS *)

\* The theorem the C++ is supposed to satisfy, and the one §13.63 observed
\* failing: a packet reachable through a LIVE wrapper always has rc > 0.
LiveWrapperPinsPacket ==
    \A w \in Wrappers :
        (wAlive[w] /\ wPacket[w] # Null) => rc[wPacket[w]] > 0

\* The §13.63 observation itself, as a state predicate: no thread ever
\* reads a packet whose count has already reached zero.
NoResurrection == ~deadRead

\* NOT a protocol invariant -- a MODEL-INTERNAL bookkeeping identity, and
\* one this holder-id scheme cannot maintain: ownership TRANSFER (commit
\* renames a thread-local holder into the installed wrapper) reuses ids
\* across generations, so `holders` is a set where `rc` is a count.  Kept
\* as a definition (it is useful when tightening the model with
\* unique-per-acquisition ids) but deliberately NOT asserted; asserting it
\* would report model artefacts as protocol violations.  See §13.64.
CountMatchesHolders ==
    \A p \in Packets : rc[p] = Cardinality(holders[p])

TypeOK ==
    /\ \A p \in Packets : rc[p] \in 0..16
    /\ \A w \in Wrappers : wrc[w] \in 0..16

=============================================================================
