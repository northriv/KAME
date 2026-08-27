--------------------------- MODULE ExitWalk --------------------------------
(***************************************************************************)
(* §13.215  The exit DLL WALK, multi-chunk -- the last structural gap.      *)
(*                                                                         *)
(* §13.213 closed the freelist, the two-phase exit and permanent thread     *)
(* death, and the resulting model passes exhaustively.  It still has ONE    *)
(* chunk, so it cannot express the walk:                                    *)
(*                                                                         *)
(*   c = dll_head; dll_head = NULL;                                         *)
(*   while(c) { next = c->m_dll_next; unlink c; drain c; disown c; c = next; }*)
(*                                                                         *)
(* The hazard a single-chunk model cannot see is SEQUENCING: a chunk         *)
(* disowned early in the walk goes on the orphan chain and can be ADOPTED    *)
(* by a peer while the walk is still working through later chunks.  The      *)
(* adopter allocates and frees on it, and its cross-thread frees land on     *)
(* chunks the exiting thread is still draining.                             *)
(*                                                                         *)
(* Measured facts this must respect (they are why the model is shaped so):   *)
(*   * the only code that writes a NEIGHBOUR's link (`owner_release`'s       *)
(*     unlink) is never entered in this workload -- entered = 0 -- so link   *)
(*     corruption is out and the walk is modelled as a clean queue;          *)
(*   * a chunk not yet visited still has BIT_OWNED set, so no peer can take  *)
(*     it: only chunks already visited are adoptable.                        *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS Chunks, Slots, Threads, BUG_DRAIN_AFTER_DISOWN, BUG_STALE_FLIST

VARIABLES
    owner,      \* [Chunks -> Threads \cup {NONE}]
    onChain,    \* [Chunks -> BOOLEAN]
    bits,       \* [Chunks -> SUBSET Slots]
    occ,        \* [Chunks -> SUBSET Slots]
    flist,      \* [Chunks -> SUBSET Slots]
    pending,    \* [Threads -> SUBSET (Chunks \X Slots)]
    walk,       \* [Threads -> SUBSET Chunks]  remaining chunks of an exit walk
                \*   a SET, not a sequence: leaving the order free explores every
                \*   visit order, which is stronger than fixing one.
    pc,         \* [Threads -> {"idle","exiting"}]
    dead,       \* SUBSET Threads
    drained,    \* SUBSET Chunks : drain already run in the current walk
    snap,       \* [Chunks -> SUBSET Slots] : the freelist view the drain will use
    handed      \* SUBSET (Chunks \X Slots)

vars == <<owner, onChain, bits, occ, flist, pending, walk, pc, dead, handed,
          drained, snap>>
NONE == "none"

Pairs == Chunks \X Slots

TypeOK ==
    /\ owner \in [Chunks -> Threads \cup {NONE}]
    /\ onChain \in [Chunks -> BOOLEAN]
    /\ bits \in [Chunks -> SUBSET Slots]
    /\ occ \in [Chunks -> SUBSET Slots]
    /\ flist \in [Chunks -> SUBSET Slots]
    /\ pending \in [Threads -> SUBSET Pairs]
    /\ walk \in [Threads -> SUBSET Chunks]
    /\ pc \in [Threads -> {"idle", "exiting"}]
    /\ dead \in SUBSET Threads
    /\ handed \in SUBSET Pairs
    /\ drained \in SUBSET Chunks
    /\ snap \in [Chunks -> SUBSET Slots]

Init ==
    /\ owner = [c \in Chunks |-> NONE]
    /\ onChain = [c \in Chunks |-> FALSE]
    /\ bits = [c \in Chunks |-> {}]
    /\ occ = [c \in Chunks |-> {}]
    /\ flist = [c \in Chunks |-> {}]
    /\ pending = [t \in Threads |-> {}]
    /\ walk = [t \in Threads |-> {}]
    /\ pc = [t \in Threads |-> "idle"]
    /\ dead = {}
    /\ handed = {}
    /\ drained = {}
    /\ snap = [c \in Chunks |-> {}]

Alive(t) == t \notin dead

Claim(t, c) ==
    /\ Alive(t) /\ pc[t] = "idle" /\ owner[c] = NONE /\ ~onChain[c]
    /\ owner' = [owner EXCEPT ![c] = t]
    /\ UNCHANGED <<onChain, bits, occ, flist, pending, walk, pc, dead, handed, drained, snap>>

\* one slot at a time, bit set on claim
AllocBitmap(t, c) ==
    /\ Alive(t) /\ pc[t] = "idle" /\ owner[c] = t /\ bits[c] /= Slots
    /\ \E s \in Slots \ bits[c] :
        /\ bits' = [bits EXCEPT ![c] = @ \cup {s}]
        /\ occ' = [occ EXCEPT ![c] = @ \cup {s}]
        /\ handed' = IF s \in occ[c] THEN handed \cup {<<c, s>>} ELSE handed
    /\ UNCHANGED <<owner, onChain, flist, pending, walk, pc, dead, drained, snap>>

\* freelist_pop: no bitmap consultation, bit already set
AllocFlist(t, c) ==
    /\ Alive(t) /\ pc[t] = "idle" /\ owner[c] = t /\ flist[c] /= {}
    /\ \E s \in flist[c] :
        /\ flist' = [flist EXCEPT ![c] = @ \ {s}]
        /\ occ' = [occ EXCEPT ![c] = @ \cup {s}]
        /\ handed' = IF s \in occ[c] THEN handed \cup {<<c, s>>} ELSE handed
    /\ UNCHANGED <<owner, onChain, bits, pending, walk, pc, dead, drained, snap>>

FreeOwner(t, c) ==
    /\ Alive(t) /\ pc[t] = "idle" /\ owner[c] = t /\ occ[c] /= {}
    /\ \E s \in occ[c] :
        /\ occ' = [occ EXCEPT ![c] = @ \ {s}]
        /\ flist' = [flist EXCEPT ![c] = @ \cup {s}]
    /\ UNCHANGED <<owner, onChain, bits, pending, walk, pc, dead, handed, drained, snap>>

FreeCross(t, c) ==
    /\ Alive(t) /\ pc[t] = "idle" /\ owner[c] /= t /\ occ[c] /= {}
    /\ \E s \in occ[c] :
        /\ occ' = [occ EXCEPT ![c] = @ \ {s}]
        /\ pending' = [pending EXCEPT ![t] = @ \cup {<<c, s>>}]
    /\ UNCHANGED <<owner, onChain, bits, flist, walk, pc, dead, handed, drained, snap>>

Flush(t) ==
    /\ Alive(t) /\ pending[t] /= {}
    /\ bits' = [c \in Chunks |->
                  bits[c] \ {s \in Slots : <<c, s>> \in pending[t]}]
    /\ pending' = [pending EXCEPT ![t] = {}]
    /\ UNCHANGED <<owner, onChain, occ, flist, walk, pc, dead, handed, drained, snap>>

(**************************** the exit walk, per chunk **********************)
\* Phase 1: ~CrossDeallocBatch, then the walk list is captured.
ExitBegin(t) ==
    /\ Alive(t) /\ pc[t] = "idle"
    /\ pc' = [pc EXCEPT ![t] = "exiting"]
    /\ bits' = [c \in Chunks |->
                  bits[c] \ {s \in Slots : <<c, s>> \in pending[t]}]
    /\ pending' = [pending EXCEPT ![t] = {}]
    /\ walk' = [walk EXCEPT ![t] = {c \in Chunks : owner[c] = t}]
    /\ UNCHANGED <<owner, onChain, occ, flist, dead, handed, drained, snap>>

\* Phase 2, ONE CHUNK PER STEP, and the per-chunk work SPLIT into drain and
\* disown as separate steps.
\*
\* §13.215: my first version did both in one atomic step, and both bug knobs then
\* produced an IDENTICAL state count (6915) -- inert.  That is §13.61 on a spec
\* again, and the tell is exact: a knob that does not change the distinct-state
\* count is not being exercised.  With the two halves separated a peer can adopt
\* between them, which is the sequencing hazard a single-chunk or single-step model
\* cannot express.
\*
\* `drained` marks the chunks of this walk whose drain has already run.
ExitDrainOne(t) ==
    /\ Alive(t) /\ pc[t] = "exiting"
    /\ \E c \in walk[t] :
        /\ c \notin drained
        /\ IF BUG_DRAIN_AFTER_DISOWN THEN owner[c] = NONE ELSE owner[c] = t
        \* BUG_STALE_FLIST: clear the bits of the view captured at disown time,
        \* which is what a drain that walked the list into a local before the
        \* chunk became adoptable would do.  Slots the adopter has since popped
        \* are still in that view -- and are LIVE.
        /\ LET view == IF BUG_STALE_FLIST THEN snap[c] ELSE flist[c] IN
           /\ bits' = [bits EXCEPT ![c] = @ \ view]
           /\ flist' = [flist EXCEPT ![c] = @ \ view]
        /\ drained' = drained \cup {c}
    /\ UNCHANGED <<owner, onChain, occ, pending, walk, pc, dead, handed, snap>>

ExitDisownOne(t) ==
    /\ Alive(t) /\ pc[t] = "exiting"
    /\ \E c \in walk[t] :
        /\ owner[c] = t
        /\ IF BUG_DRAIN_AFTER_DISOWN THEN TRUE ELSE c \in drained
        /\ owner' = [owner EXCEPT ![c] = NONE]
        /\ snap' = [snap EXCEPT ![c] = flist[c]]      \* the view, captured here
        /\ onChain' = [onChain EXCEPT ![c] = (occ[c] /= {} \/ bits[c] /= {})]
        /\ walk' = IF BUG_DRAIN_AFTER_DISOWN
                     THEN walk
                     ELSE [walk EXCEPT ![t] = @ \ {c}]
    /\ UNCHANGED <<bits, occ, flist, pending, pc, dead, handed, drained>>

\* With the buggy order the chunk leaves the walk only once drained.
ExitRetire(t) ==
    /\ Alive(t) /\ pc[t] = "exiting" /\ BUG_DRAIN_AFTER_DISOWN
    /\ \E c \in walk[t] :
        /\ owner[c] = NONE /\ c \in drained
        /\ walk' = [walk EXCEPT ![t] = @ \ {c}]
    /\ UNCHANGED <<owner, onChain, bits, occ, flist, pending, pc, dead, handed,
                   drained, snap>>

ExitEnd(t) ==
    /\ Alive(t) /\ pc[t] = "exiting" /\ walk[t] = {}
    /\ dead' = dead \cup {t}
    /\ drained' = {}
    /\ snap' = snap
    /\ UNCHANGED <<owner, onChain, bits, occ, flist, pending, walk, pc, handed>>

\* A peer adopts a chunk the walk has already released.  (A knob letting it take
\* an UNVISITED chunk was tried and is inert: `onChain => owner = NONE` holds
\* invariantly in this model -- adoption pops from the orphan chain and only a
\* disowned chunk is ever on it -- so the relaxation has no reachable
\* precondition.  Removed rather than kept as a knob that passes vacuously.)
Adopt(t, c) ==
    /\ Alive(t) /\ pc[t] = "idle" /\ onChain[c]
    /\ owner[c] = NONE
    /\ owner' = [owner EXCEPT ![c] = t]
    /\ onChain' = [onChain EXCEPT ![c] = FALSE]
    /\ UNCHANGED <<bits, occ, flist, pending, walk, pc, dead, handed, drained, snap>>

Next ==
    \/ \E t \in Threads, c \in Chunks :
         Claim(t, c) \/ AllocBitmap(t, c) \/ AllocFlist(t, c)
         \/ FreeOwner(t, c) \/ FreeCross(t, c) \/ Adopt(t, c)
    \/ \E t \in Threads : Flush(t) \/ ExitBegin(t) \/ ExitDrainOne(t)
                       \/ ExitDisownOne(t) \/ ExitRetire(t) \/ ExitEnd(t)

Spec == Init /\ [][Next]_vars

(********************************* properties *******************************)
OccCoveredByBits == \A c \in Chunks : occ[c] \subseteq bits[c]
FlistCoveredByBits == \A c \in Chunks : flist[c] \subseteq bits[c]
PendingCoveredByBits ==
    \A t \in Threads, c \in Chunks :
        {s \in Slots : <<c, s>> \in pending[t]} \subseteq bits[c]
NoDoubleCustody ==
    \A c \in Chunks :
        /\ flist[c] \cap occ[c] = {}
        /\ \A t \in Threads :
             {s \in Slots : <<c, s>> \in pending[t]} \cap occ[c] = {}
\* A chunk is never simultaneously owned and offered on the chain.
NoOwnedOnChain == \A c \in Chunks : ~(onChain[c] /\ owner[c] /= NONE)
NoDoubleHandOut == handed = {}
NoLostEntries == \A t \in dead : pending[t] = {}

Inv == TypeOK /\ OccCoveredByBits /\ FlistCoveredByBits /\ PendingCoveredByBits
       /\ NoDoubleCustody /\ NoOwnedOnChain /\ NoDoubleHandOut /\ NoLostEntries
===========================================================================
