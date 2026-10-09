(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        Dual-licensed Apache 2.0 OR GPL-2.0-or-later — see OrphanChain_atomicshared.tla.
 ***************************************************************************)
----------------------------- MODULE RevivalAnchor -----------------------------
(*
 * Where the revival stack's head lives (design model; see the end of this
 * comment for how the code relates to it).
 *
 * RevivalStack.tla keeps the head in a static slot table, tagged with the
 * owner's generation.  The owner's TLS is ruled out -- a freer can reach it
 * after the owner exited (the force-walk TOCTOU: a dead TLS block, or a new
 * thread's TLS at the same address whose head happens to compare equal).
 * Here the head lives in a chunk instead: an owner's first chunk becomes its
 * ANCHOR and carries the head word.  Every chunk the owner holds points at
 * the anchor through an atomic_shared_ptr (anc), so the anchor stays alive --
 * as an ordinary orphan after its owner has gone -- while any chunk or any
 * freer still refers to it, and always names the chunk's CURRENT owner:
 * adoption repoints anc, orphaning nulls it.  No tags, no slot table.
 *
 *   - A freer holding Q loads anc[c] (a counted load: ref[a] + 1), reads the
 *     anchor's head, links, CASes, and drops its reference.  anc = null: the
 *     chunk was orphaned meanwhile -> orphan path.  Head closed: the owner is
 *     leaving -> drop the reference and reroute, as in RevivalStack.
 *   - The owner never releases its anchor while it lives (every chunk points
 *     at it); at exit it closes the head, orphans its chunks (anchor
 *     included), and drops its own reference.
 *   - A chunk is released when its packed word is zero AND no reference is
 *     left; the transition that makes both true releases it (in the code: a
 *     non-zero packed word holds one reference, dropped by whoever zeroes it).
 *   - Only a FRESH chunk becomes an anchor.  AnchorFromAdopted = TRUE lets an
 *     owner with no anchor yet adopt an orphan and make it one, reopening its
 *     head: a freer that loaded the old, closed-to-be head can then CAS the
 *     reopened one and push the previous owner's chunk onto the new stack.
 *     AnchorFromUnref = TRUE allows it only for an adopted chunk that nothing
 *     references as an anchor (ref = 0): no anc points at it, so no freer
 *     holds or can still obtain it, and reopening its head is unobservable.
 *     A chunk that was never an anchor is the common case (always ref = 0).
 *   - NoAnchorRef = TRUE: the freer loads anc without taking a reference.
 *   - LazyDrain = TRUE (the code): the owner takes the stack only when the
 *     rest of its last take is used up, and pops that rest one chunk per
 *     try, running in between; the rest keeps Q.  At exit it drops Q on
 *     the rest first, then closes the head as before.
 *
 * Q protocol: the one-bit variant of RevivalStack (Q taken before the bit
 * clear, no pin, no P), whose safety invariants hold there; only Q holders
 * touch anchors, so the anchor's lifetime does not depend on the variant.
 * Room-loss is RevivalStack's subject and is not checked here.
 *
 * The code (kamepoolalloc stage 2a, "§revive") is AnchorFromUnref = TRUE,
 * LazyDrain = TRUE.  Its orphan side differs: it keeps the existing chunk-
 * wise orphan chain (every non-empty chunk is pushed at exit; Q on an orphan
 * only means a stale freer is still pushing), not the aorph chain here.
 *
 * Results (3 chunks, 2 freers, 2 owners, K = 2, symmetry):
 *   design                 clean, 296,844,574 distinct states
 *   unref  (AnchorFromUnref)              clean, 584,542,408, depth 133
 *   code   (AnchorFromUnref, LazyDrain)   clean, 795,755,368, depth 134
 *   adopted (AnchorFromAdopted)           Inv_StackOK (found in 2 s)
 *   noref  (NoAnchorRef)                  Inv_NoUseAfterRelease
 *   witnesses (an Assert(FALSE) placed in the branch, each reached):
 *   adopted chunk made an anchor (depth 13) and a closed ex-anchor reopened
 *   (14) under unref; exit holding a rest and alloc holding a rest (17)
 *   under code.
 *)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS Chunks, Freers, Owners, NIL, NONE, K, AnchorFromAdopted, AnchorFromUnref,
          NoAnchorRef, LazyDrain

ASSUME NIL \notin Chunks /\ NONE \notin Owners
ASSUME K \in Nat \ {0}
ASSUME AnchorFromAdopted \in BOOLEAN /\ AnchorFromUnref \in BOOLEAN
ASSUME NoAnchorRef \in BOOLEAN /\ LazyDrain \in BOOLEAN

Ptr == Chunks \cup {NIL}
HeadT == [ptr: Ptr, closed: BOOLEAN]
OpenEmpty == [ptr |-> NIL, closed |-> FALSE]
ClosedEmpty == [ptr |-> NIL, closed |-> TRUE]
MaxRef == Cardinality(Chunks) + Cardinality(Freers) + Cardinality(Owners)

VARIABLES
    st,      \* [Chunks -> {"fresh", "live", "released"}]
    bits,    \* [Chunks -> 0..K]    slots allocated (one bitmap word)
    mcnt,    \* [Chunks -> Nat]     MASK_CNT
    owned,   \* [Chunks -> BOOLEAN] BIT_OWNED
    own,     \* [Chunks -> Owners \cup {NONE}]  who owns it (m_owner_id)
    q,       \* [Chunks -> BOOLEAN] Q
    tok,     \* [Chunks -> 0..K]    live slots held by the application
    nx,      \* [Chunks -> Ptr]     m_revive_next
    anc,     \* [Chunks -> Ptr]     atomic_shared_ptr to the owner's anchor
    isA,     \* [Chunks -> BOOLEAN] has been made an anchor
    hd,      \* [Chunks -> HeadT]   the head word (meaningful on anchors)
    ref,     \* [Chunks -> Nat]     references to the chunk as an anchor
    aorph,   \* available-orphan chain (as a set)
    ost,     \* [Owners -> pc]
    oanc,    \* [Owners -> Ptr]     the owner's anchor (its own reference)
    avail,   \* [Owners -> SUBSET Chunks]
    omode,   \* [Owners -> {"drain", "close"}]
    ocur,    \* [Owners -> Ptr]
    onxt,    \* [Owners -> Ptr]
    oorph,   \* [Owners -> SUBSET Chunks]
    fpc,     \* [Freers -> pc]
    fc,      \* [Freers -> Ptr]     chunk whose slot it returns
    fzero,   \* [Freers -> BOOLEAN]
    fwasq,   \* [Freers -> BOOLEAN]
    fa,      \* [Freers -> Ptr]     anchor it loaded (and holds a reference to)
    fh       \* [Freers -> HeadT]   head value loaded for the CAS

vars == <<st, bits, mcnt, owned, own, q, tok, nx, anc, isA, hd, ref, aorph,
          ost, oanc, avail, omode, ocur, onxt, oorph, fpc, fc, fzero, fwasq, fa, fh>>

ChunkVars == <<st, bits, mcnt, owned, own, q, tok, nx, anc, isA, hd, ref>>
OwnVars   == <<ost, oanc, avail, omode, ocur, onxt, oorph>>
FreeVars  == <<fpc, fc, fzero, fwasq, fa, fh>>

TypeOK ==
    /\ st \in [Chunks -> {"fresh", "live", "released"}]
    /\ bits \in [Chunks -> 0..K] /\ mcnt \in [Chunks -> 0..(1 + Cardinality(Freers))]
    /\ owned \in [Chunks -> BOOLEAN] /\ own \in [Chunks -> Owners \cup {NONE}]
    /\ q \in [Chunks -> BOOLEAN] /\ tok \in [Chunks -> 0..K]
    /\ nx \in [Chunks -> Ptr] /\ anc \in [Chunks -> Ptr]
    /\ isA \in [Chunks -> BOOLEAN] /\ hd \in [Chunks -> HeadT]
    /\ ref \in [Chunks -> 0..MaxRef] /\ aorph \subseteq Chunks
    /\ oanc \in [Owners -> Ptr] /\ fa \in [Freers -> Ptr] /\ fh \in [Freers -> HeadT]

RECURSIVE Walk(_, _)
Walk(c, n) == IF c = NIL \/ n = 0 THEN <<>> ELSE <<c>> \o Walk(nx[c], n - 1)
StackSeq(a) == Walk(hd[a].ptr, Cardinality(Chunks) + 1)
StackSet(a) == {StackSeq(a)[i] : i \in 1..Len(StackSeq(a))}
OnSomeStack == UNION {StackSet(a) : a \in {x \in Chunks : isA[x] /\ st[x] = "live"}}

Init ==
    /\ st = [c \in Chunks |-> "fresh"]
    /\ bits = [c \in Chunks |-> 0] /\ mcnt = [c \in Chunks |-> 0]
    /\ owned = [c \in Chunks |-> FALSE] /\ own = [c \in Chunks |-> NONE]
    /\ q = [c \in Chunks |-> FALSE] /\ tok = [c \in Chunks |-> 0]
    /\ nx = [c \in Chunks |-> NIL] /\ anc = [c \in Chunks |-> NIL]
    /\ isA = [c \in Chunks |-> FALSE] /\ hd = [c \in Chunks |-> OpenEmpty]
    /\ ref = [c \in Chunks |-> 0] /\ aorph = {}
    /\ ost = [t \in Owners |-> "idle"] /\ oanc = [t \in Owners |-> NIL]
    /\ avail = [t \in Owners |-> {}] /\ omode = [t \in Owners |-> "drain"]
    /\ ocur = [t \in Owners |-> NIL] /\ onxt = [t \in Owners |-> NIL]
    /\ oorph = [t \in Owners |-> {}]
    /\ fpc = [f \in Freers |-> "idle"] /\ fc = [f \in Freers |-> NIL]
    /\ fzero = [f \in Freers |-> FALSE] /\ fwasq = [f \in Freers |-> FALSE]
    /\ fa = [f \in Freers |-> NIL] /\ fh = [f \in Freers |-> OpenEmpty]

(* Release c iff, with the given new values, its packed word and its
   reference count are both zero. *)
Dead(c, mc, ow, qq, rf) == mc = 0 /\ ~ow /\ ~qq /\ rf = 0
Rel(c) == st' = [st EXCEPT ![c] = "released"]

GoIdle(f) ==
    /\ fpc' = [fpc EXCEPT ![f] = "idle"] /\ fc' = [fc EXCEPT ![f] = NIL]
    /\ fzero' = [fzero EXCEPT ![f] = FALSE] /\ fwasq' = [fwasq EXCEPT ![f] = FALSE]
    /\ fa' = [fa EXCEPT ![f] = NIL] /\ fh' = [fh EXCEPT ![f] = OpenEmpty]

(******************************** owners ********************************)

OStart(t) ==
    /\ ost[t] = "idle" /\ ost' = [ost EXCEPT ![t] = "run"]
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<aorph, oanc, avail, omode, ocur, onxt, oorph>>
    /\ UNCHANGED FreeVars

(* A fresh chunk joins t; the first one becomes t's anchor. *)
OAttach(t, c) ==
    LET a == IF oanc[t] = NIL THEN c ELSE oanc[t] IN
    /\ ost[t] = "run" /\ st[c] = "fresh"
    /\ st' = [st EXCEPT ![c] = "live"]
    /\ owned' = [owned EXCEPT ![c] = TRUE] /\ own' = [own EXCEPT ![c] = t]
    /\ anc' = [anc EXCEPT ![c] = a]
    /\ IF oanc[t] = NIL
       THEN /\ isA' = [isA EXCEPT ![c] = TRUE] /\ hd' = [hd EXCEPT ![c] = OpenEmpty]
            /\ ref' = [ref EXCEPT ![c] = 2]          \* t's own + c's anc
            /\ oanc' = [oanc EXCEPT ![t] = c]
       ELSE /\ ref' = [ref EXCEPT ![a] = @ + 1] /\ UNCHANGED <<isA, hd, oanc>>
    /\ avail' = [avail EXCEPT ![t] = @ \cup {c}]
    /\ UNCHANGED <<bits, mcnt, q, tok, nx, aorph, ost, omode, ocur, onxt, oorph>>
    /\ UNCHANGED FreeVars

OAlloc(t, c) ==
    /\ ost[t] = "run" /\ c \in avail[t] /\ bits[c] < K
    /\ bits' = [bits EXCEPT ![c] = @ + 1]
    /\ mcnt' = IF bits[c] = 0 THEN [mcnt EXCEPT ![c] = @ + 1] ELSE mcnt
    /\ tok' = [tok EXCEPT ![c] = @ + 1]
    /\ avail' = [avail EXCEPT ![t] = IF bits[c] + 1 = K THEN @ \ {c} ELSE @]
    /\ UNCHANGED <<st, owned, own, q, nx, anc, isA, hd, ref, aorph>>
    /\ UNCHANGED <<ost, oanc, omode, ocur, onxt, oorph>> /\ UNCHANGED FreeVars

(* Adopt an orphan with room.  Design: only an owner that already has an
   anchor adopts.  AnchorFromAdopted: an owner without one makes the adopted
   chunk its anchor, reopening the head. *)
OAdopt(t, c) ==
    LET mk == oanc[t] = NIL
        a  == IF mk THEN c ELSE oanc[t]
    IN
    /\ ost[t] = "run" /\ c \in aorph
    /\ ~mk \/ AnchorFromAdopted \/ (AnchorFromUnref /\ ref[c] = 0)
    /\ aorph' = aorph \ {c}
    /\ owned' = [owned EXCEPT ![c] = TRUE] /\ own' = [own EXCEPT ![c] = t]
    /\ q' = [q EXCEPT ![c] = FALSE]
    /\ anc' = [anc EXCEPT ![c] = a]
    /\ IF mk
       THEN /\ isA' = [isA EXCEPT ![c] = TRUE] /\ hd' = [hd EXCEPT ![c] = OpenEmpty]
            /\ ref' = [ref EXCEPT ![c] = @ + 2]
            /\ oanc' = [oanc EXCEPT ![t] = c]
       ELSE /\ ref' = [ref EXCEPT ![a] = @ + 1] /\ UNCHANGED <<isA, hd, oanc>>
    /\ avail' = [avail EXCEPT ![t] = IF bits[c] < K THEN @ \cup {c} ELSE @]
    /\ UNCHANGED <<st, bits, mcnt, tok, nx, ost, omode, ocur, onxt, oorph>>
    /\ UNCHANGED FreeVars

(* owner_release of an empty, unlisted chunk -- never the anchor. *)
ORelease(t, c) ==
    LET a == anc[c] IN
    /\ ost[t] = "run" /\ st[c] = "live" /\ owned[c] /\ own[c] = t
    /\ c # oanc[t] /\ mcnt[c] = 0 /\ ~q[c]
    /\ owned' = [owned EXCEPT ![c] = FALSE] /\ own' = [own EXCEPT ![c] = NONE]
    /\ anc' = [anc EXCEPT ![c] = NIL]
    /\ ref' = [ref EXCEPT ![a] = @ - 1]
    /\ IF ref[c] = 0 THEN Rel(c) ELSE UNCHANGED st    \* a former anchor may be referenced
    /\ avail' = [avail EXCEPT ![t] = @ \ {c}]
    /\ UNCHANGED <<bits, mcnt, q, tok, nx, isA, hd, aorph>>
    /\ UNCHANGED <<ost, oanc, omode, ocur, onxt, oorph>> /\ UNCHANGED FreeVars

ODrainStart(t) ==
    LET a == oanc[t] IN
    /\ ost[t] = "run" /\ a # NIL
    /\ LazyDrain => ocur[t] = NIL
    /\ ocur' = [ocur EXCEPT ![t] = hd[a].ptr]
    /\ hd' = [hd EXCEPT ![a] = OpenEmpty]
    /\ omode' = [omode EXCEPT ![t] = "drain"]
    /\ ost' = [ost EXCEPT ![t] = IF LazyDrain THEN "run" ELSE "dnext"]
    /\ UNCHANGED <<st, bits, mcnt, owned, own, q, tok, nx, anc, isA, ref, aorph>>
    /\ UNCHANGED <<oanc, avail, onxt, oorph>> /\ UNCHANGED FreeVars

(* LazyDrain: pop one chunk of the rest of the last take (then back to run). *)
OPopStart(t) ==
    /\ LazyDrain /\ ost[t] = "run" /\ ocur[t] # NIL /\ omode[t] = "drain"
    /\ ost' = [ost EXCEPT ![t] = "dnext"]
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<aorph, oanc, avail, omode, ocur, onxt, oorph>>
    /\ UNCHANGED FreeVars

(* LazyDrain: exiting with a rest -- drop Q on it ("pre"), then close. *)
OExitPre(t) ==
    /\ LazyDrain /\ ost[t] = "run" /\ ocur[t] # NIL
    /\ omode' = [omode EXCEPT ![t] = "pre"] /\ ost' = [ost EXCEPT ![t] = "dnext"]
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<aorph, oanc, avail, ocur, onxt, oorph>>
    /\ UNCHANGED FreeVars

(* Exit: close the head (taking what is on it), then orphan everything. *)
OCloseStart(t) ==
    LET a == oanc[t] IN
    /\ \/ ost[t] = "run" /\ (LazyDrain => ocur[t] = NIL)
       \/ ost[t] = "closing"
    /\ IF a = NIL
       THEN /\ ost' = [ost EXCEPT ![t] = "dead"] /\ UNCHANGED <<hd, ocur, omode>>
       ELSE /\ ocur' = [ocur EXCEPT ![t] = hd[a].ptr]
            /\ hd' = [hd EXCEPT ![a] = ClosedEmpty]
            /\ omode' = [omode EXCEPT ![t] = "close"] /\ ost' = [ost EXCEPT ![t] = "dnext"]
    /\ UNCHANGED <<st, bits, mcnt, owned, own, q, tok, nx, anc, isA, ref, aorph>>
    /\ UNCHANGED <<oanc, avail, onxt, oorph>> /\ UNCHANGED FreeVars

ODrainNext(t) ==
    /\ ost[t] = "dnext"
    /\ IF ocur[t] = NIL
       THEN IF omode[t] \in {"drain", "pre"}
            THEN /\ ost' = [ost EXCEPT ![t] = IF omode[t] = "pre" THEN "closing" ELSE "run"]
                 /\ UNCHANGED <<onxt, oorph>>
            ELSE /\ ost' = [ost EXCEPT ![t] = "orph"] /\ UNCHANGED onxt
                 /\ oorph' = [oorph EXCEPT ![t] =
                                {c \in Chunks : st[c] = "live" /\ owned[c] /\ own[c] = t}]
       ELSE /\ onxt' = [onxt EXCEPT ![t] = nx[ocur[t]]]
            /\ ost' = [ost EXCEPT ![t] = "dclr"] /\ UNCHANGED oorph
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<aorph, oanc, avail, omode, ocur>>
    /\ UNCHANGED FreeVars

ODrainClear(t) ==
    LET c == ocur[t] IN
    /\ ost[t] = "dclr"
    /\ q' = [q EXCEPT ![c] = FALSE] /\ nx' = [nx EXCEPT ![c] = NIL]
    /\ ost' = [ost EXCEPT ![t] = "dspace"]
    /\ UNCHANGED <<st, bits, mcnt, owned, own, tok, anc, isA, hd, ref, aorph>>
    /\ UNCHANGED <<oanc, avail, omode, ocur, onxt, oorph>> /\ UNCHANGED FreeVars

ODrainSpace(t) ==
    LET c == ocur[t] IN
    /\ ost[t] = "dspace"
    /\ avail' = [avail EXCEPT ![t] = IF bits[c] < K THEN @ \cup {c} ELSE @]
    /\ ocur' = [ocur EXCEPT ![t] = onxt[t]] /\ onxt' = [onxt EXCEPT ![t] = NIL]
    /\ ost' = [ost EXCEPT ![t] = IF LazyDrain /\ omode[t] = "drain" THEN "run" ELSE "dnext"]
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<aorph, oanc, omode, oorph>> /\ UNCHANGED FreeVars

(* Orphan one chunk: owned := 0, anc := null (dropping its reference to the
   anchor).  Then as RevivalStack's one-bit exit: empty and unlisted ->
   release (unless referenced); Q clear -> take it and look at the room;
   Q held by a freer -> that freer will find it orphaned. *)
OOrphan(t, c) ==
    LET a   == anc[c]
        rf  == IF a = c THEN ref[c] - 1 ELSE ref[c]   \* c's own count after the drop
    IN
    /\ ost[t] = "orph" /\ c \in oorph[t]
    /\ oorph' = [oorph EXCEPT ![t] = @ \ {c}]
    /\ owned' = [owned EXCEPT ![c] = FALSE] /\ own' = [own EXCEPT ![c] = NONE]
    /\ anc' = [anc EXCEPT ![c] = NIL]
    /\ ref' = [ref EXCEPT ![a] = @ - 1]
    /\ avail' = [avail EXCEPT ![t] = @ \ {c}]
    /\ IF mcnt[c] = 0 /\ ~q[c]
       THEN /\ IF rf = 0 THEN Rel(c) ELSE UNCHANGED st
            /\ UNCHANGED <<q, ocur>> /\ ost' = [ost EXCEPT ![t] = "orph"]
       ELSE IF ~q[c]
            THEN /\ q' = [q EXCEPT ![c] = TRUE] /\ ocur' = [ocur EXCEPT ![t] = c]
                 /\ ost' = [ost EXCEPT ![t] = "ospace"] /\ UNCHANGED st
            ELSE /\ UNCHANGED <<st, q, ocur>> /\ ost' = [ost EXCEPT ![t] = "orph"]
    /\ UNCHANGED <<bits, mcnt, tok, nx, isA, hd, aorph, oanc, omode, onxt>>
    /\ UNCHANGED FreeVars

OOrphSpace(t) ==
    LET c == ocur[t] IN
    /\ ost[t] = "ospace"
    /\ IF bits[c] < K
       THEN /\ aorph' = aorph \cup {c} /\ ocur' = [ocur EXCEPT ![t] = NIL]
            /\ ost' = [ost EXCEPT ![t] = "orph"]
       ELSE /\ ost' = [ost EXCEPT ![t] = "odrop"] /\ UNCHANGED <<aorph, ocur>>
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<oanc, avail, omode, onxt, oorph>>
    /\ UNCHANGED FreeVars

OOrphDrop(t) ==
    LET c == ocur[t] IN
    /\ ost[t] = "odrop"
    /\ q' = [q EXCEPT ![c] = FALSE]
    /\ IF Dead(c, mcnt[c], owned[c], FALSE, ref[c]) THEN Rel(c) ELSE UNCHANGED st
    /\ ocur' = [ocur EXCEPT ![t] = NIL] /\ ost' = [ost EXCEPT ![t] = "orph"]
    /\ UNCHANGED <<bits, mcnt, owned, own, tok, nx, anc, isA, hd, ref, aorph>>
    /\ UNCHANGED <<oanc, avail, omode, onxt, oorph>> /\ UNCHANGED FreeVars

(* All orphaned: drop t's own reference to its anchor. *)
OOrphDone(t) ==
    LET a == oanc[t] IN
    /\ ost[t] = "orph" /\ oorph[t] = {}
    /\ ref' = [ref EXCEPT ![a] = @ - 1]
    /\ IF Dead(a, mcnt[a], owned[a], q[a], ref[a] - 1) THEN Rel(a) ELSE UNCHANGED st
    /\ oanc' = [oanc EXCEPT ![t] = NIL] /\ avail' = [avail EXCEPT ![t] = {}]
    /\ ost' = [ost EXCEPT ![t] = "dead"]
    /\ UNCHANGED <<bits, mcnt, owned, own, q, tok, nx, anc, isA, hd, aorph>>
    /\ UNCHANGED <<omode, ocur, onxt, oorph>> /\ UNCHANGED FreeVars

(******************************** freers ********************************)

FPick(f, c) ==
    /\ fpc[f] = "idle" /\ st[c] = "live" /\ tok[c] > 0
    /\ tok' = [tok EXCEPT ![c] = @ - 1] /\ fc' = [fc EXCEPT ![f] = c]
    /\ fpc' = [fpc EXCEPT ![f] = "fq1"]
    /\ UNCHANGED <<st, bits, mcnt, owned, own, q, nx, anc, isA, hd, ref, aorph>>
    /\ UNCHANGED OwnVars /\ UNCHANGED <<fzero, fwasq, fa, fh>>

(* fetch_or Q while our slot still keeps the word non-zero. *)
FTakeQ(f) ==
    LET c == fc[f] IN
    /\ fpc[f] = "fq1"
    /\ fwasq' = [fwasq EXCEPT ![f] = q[c]] /\ q' = [q EXCEPT ![c] = TRUE]
    /\ fpc' = [fpc EXCEPT ![f] = "fb"]
    /\ UNCHANGED <<st, bits, mcnt, owned, own, tok, nx, anc, isA, hd, ref, aorph>>
    /\ UNCHANGED OwnVars /\ UNCHANGED <<fc, fzero, fa, fh>>

FBitClear(f) ==
    LET c == fc[f] IN
    /\ fpc[f] = "fb"
    /\ bits' = [bits EXCEPT ![c] = @ - 1]
    /\ IF bits[c] = 1
       THEN /\ fpc' = [fpc EXCEPT ![f] = "fd"] /\ fzero' = [fzero EXCEPT ![f] = TRUE]
            /\ UNCHANGED <<fc, fwasq, fa, fh>>
       ELSE IF fwasq[f] THEN GoIdle(f)
       ELSE /\ fpc' = [fpc EXCEPT ![f] = "fgo"] /\ UNCHANGED <<fc, fzero, fwasq, fa, fh>>
    /\ UNCHANGED <<st, mcnt, owned, own, q, tok, nx, anc, isA, hd, ref, aorph>>
    /\ UNCHANGED OwnVars

FDec(f) ==
    LET c == fc[f] IN
    /\ fpc[f] = "fd"
    /\ mcnt' = [mcnt EXCEPT ![c] = @ - 1]
    /\ IF Dead(c, mcnt[c] - 1, owned[c], q[c], ref[c])
       THEN Rel(c) /\ GoIdle(f)
       ELSE /\ UNCHANGED st
            /\ IF fwasq[f] THEN GoIdle(f)
               ELSE /\ fpc' = [fpc EXCEPT ![f] = "fgo"] /\ UNCHANGED <<fc, fzero, fwasq, fa, fh>>
    /\ UNCHANGED <<bits, owned, own, q, tok, nx, anc, isA, hd, ref, aorph>>
    /\ UNCHANGED OwnVars

(* Holding Q: an owned chunk goes to its owner's stack, an orphan to the
   orphan chain. *)
FGo(f) ==
    /\ fpc[f] = "fgo"
    /\ fpc' = [fpc EXCEPT ![f] = IF owned[fc[f]] THEN "fload" ELSE "opush"]
    /\ UNCHANGED ChunkVars /\ UNCHANGED aorph /\ UNCHANGED OwnVars
    /\ UNCHANGED <<fc, fzero, fwasq, fa, fh>>

(* Counted load of the chunk's anchor pointer (atomic_shared_ptr). *)
FLoadAnc(f) ==
    LET c == fc[f]  a == anc[c] IN
    /\ fpc[f] = "fload"
    /\ IF a = NIL
       THEN /\ fpc' = [fpc EXCEPT ![f] = "reroute"] /\ UNCHANGED <<ref, fa>>
       ELSE /\ fa' = [fa EXCEPT ![f] = a]
            /\ ref' = IF NoAnchorRef THEN ref ELSE [ref EXCEPT ![a] = @ + 1]
            /\ fpc' = [fpc EXCEPT ![f] = "pread"]
    /\ UNCHANGED <<st, bits, mcnt, owned, own, q, tok, nx, anc, isA, hd, aorph>>
    /\ UNCHANGED OwnVars /\ UNCHANGED <<fc, fzero, fwasq, fh>>

FPushRead(f) ==
    LET c == fc[f]  a == fa[f] IN
    /\ fpc[f] = "pread"
    /\ fh' = [fh EXCEPT ![f] = hd[a]]
    /\ IF hd[a].closed
       THEN /\ fpc' = [fpc EXCEPT ![f] = "unref2"] /\ UNCHANGED nx
       ELSE /\ nx' = [nx EXCEPT ![c] = hd[a].ptr] /\ fpc' = [fpc EXCEPT ![f] = "pcas"]
    /\ UNCHANGED <<st, bits, mcnt, owned, own, q, tok, anc, isA, hd, ref, aorph>>
    /\ UNCHANGED OwnVars /\ UNCHANGED <<fc, fzero, fwasq, fa>>

FPushCAS(f) ==
    LET a == fa[f] IN
    /\ fpc[f] = "pcas"
    /\ IF hd[a] = fh[f]
       THEN /\ hd' = [hd EXCEPT ![a] = [ptr |-> fc[f], closed |-> FALSE]]
            /\ fpc' = [fpc EXCEPT ![f] = "unref"]
       ELSE /\ fpc' = [fpc EXCEPT ![f] = "pread"] /\ UNCHANGED hd
    /\ UNCHANGED <<st, bits, mcnt, owned, own, q, tok, nx, anc, isA, ref, aorph>>
    /\ UNCHANGED OwnVars /\ UNCHANGED <<fc, fzero, fwasq, fa, fh>>

(* Drop the anchor reference: after a push ("unref", then idle) or on a
   closed head ("unref2", then reroute). *)
FUnref(f) ==
    LET a == fa[f]  rf == IF NoAnchorRef THEN ref[a] ELSE ref[a] - 1 IN
    /\ fpc[f] \in {"unref", "unref2"}
    /\ ref' = [ref EXCEPT ![a] = rf]
    /\ IF Dead(a, mcnt[a], owned[a], q[a], rf) THEN Rel(a) ELSE UNCHANGED st
    /\ IF fpc[f] = "unref"
       THEN GoIdle(f)
       ELSE /\ fpc' = [fpc EXCEPT ![f] = "reroute"] /\ fa' = [fa EXCEPT ![f] = NIL]
            /\ fh' = [fh EXCEPT ![f] = OpenEmpty] /\ UNCHANGED <<fc, fzero, fwasq>>
    /\ UNCHANGED <<bits, mcnt, owned, own, q, tok, nx, anc, isA, hd, aorph>>
    /\ UNCHANGED OwnVars

(* Owner gone or leaving: orphaned already -> orphan chain; still owned ->
   the exiting owner will orphan it and look at its room, so drop Q. *)
FReroute(f) ==
    LET c == fc[f] IN
    /\ fpc[f] = "reroute"
    /\ IF ~owned[c]
       THEN /\ fpc' = [fpc EXCEPT ![f] = "opush"] /\ UNCHANGED <<q, fc, fzero, fwasq, fa, fh>>
       ELSE /\ q' = [q EXCEPT ![c] = FALSE] /\ GoIdle(f)
    /\ UNCHANGED <<st, bits, mcnt, owned, own, tok, nx, anc, isA, hd, ref, aorph>>
    /\ UNCHANGED OwnVars

FOrphPush(f) ==
    /\ fpc[f] = "opush"
    /\ aorph' = aorph \cup {fc[f]} /\ GoIdle(f)
    /\ UNCHANGED ChunkVars /\ UNCHANGED OwnVars

(* orphan_chain_scrub: unlink a drained orphan; release it unless referenced
   (a former anchor), in which case the last reference releases it. *)
Scrub(c) ==
    /\ c \in aorph /\ mcnt[c] = 0 /\ bits[c] = 0
    /\ aorph' = aorph \ {c} /\ q' = [q EXCEPT ![c] = FALSE]
    /\ IF ref[c] = 0 THEN Rel(c) ELSE UNCHANGED st
    /\ UNCHANGED <<bits, mcnt, owned, own, tok, nx, anc, isA, hd, ref>>
    /\ UNCHANGED OwnVars /\ UNCHANGED FreeVars

Next ==
    \/ \E t \in Owners :
         \/ OStart(t) \/ ODrainStart(t) \/ OCloseStart(t) \/ ODrainNext(t)
         \/ ODrainClear(t) \/ ODrainSpace(t) \/ OOrphSpace(t) \/ OOrphDrop(t)
         \/ OPopStart(t) \/ OExitPre(t)
         \/ OOrphDone(t)
         \/ \E c \in Chunks : OAttach(t, c) \/ OAlloc(t, c) \/ OAdopt(t, c)
                              \/ ORelease(t, c) \/ OOrphan(t, c)
    \/ \E c \in Chunks : Scrub(c)
    \/ \E f \in Freers :
         \/ \E c \in Chunks : FPick(f, c)
         \/ FTakeQ(f) \/ FBitClear(f) \/ FDec(f) \/ FGo(f) \/ FLoadAnc(f)
         \/ FPushRead(f) \/ FPushCAS(f) \/ FUnref(f) \/ FReroute(f) \/ FOrphPush(f)

Spec == Init /\ [][Next]_vars

Symm == Permutations(Chunks) \cup Permutations(Freers) \cup Permutations(Owners)

(******************************** properties ********************************)

TouchesC(f) == fpc[f] \in {"fq1", "fb", "fd", "fgo", "fload", "pread", "pcas",
                           "unref2", "reroute", "opush"}
TouchesA(f) == fpc[f] \in {"pread", "pcas", "unref", "unref2"}

(* Nobody touches, references or lists a released chunk -- anchors included. *)
Inv_NoUseAfterRelease ==
    /\ \A f \in Freers : TouchesC(f) => st[fc[f]] = "live"
    /\ \A f \in Freers : TouchesA(f) => st[fa[f]] = "live"
    /\ \A t \in Owners : oanc[t] # NIL => st[oanc[t]] = "live"
    /\ \A c \in Chunks : anc[c] # NIL => st[anc[c]] = "live"
    /\ \A c \in OnSomeStack \cup aorph : st[c] = "live"

(* Each live anchor's stack holds only chunks whose owner it is anchoring,
   listed, once; a closed head is empty. *)
Inv_StackOK ==
    \A a \in Chunks : isA[a] /\ st[a] = "live" =>
        /\ hd[a].closed => hd[a].ptr = NIL
        /\ Len(StackSeq(a)) <= Cardinality(Chunks)
        /\ \A i, j \in 1..Len(StackSeq(a)) : i # j => StackSeq(a)[i] # StackSeq(a)[j]
        /\ \A c \in StackSet(a) : owned[c] /\ q[c] /\ anc[c] = a

(* LazyDrain: while the owner runs, the rest of its last take holds only its
   own chunks, listed (Q), anchored on it, and on no stack. *)
RestSeq(t) == Walk(ocur[t], Cardinality(Chunks) + 1)
Inv_RestOK ==
    \A t \in Owners : ost[t] = "run" /\ ocur[t] # NIL =>
        /\ Len(RestSeq(t)) <= Cardinality(Chunks)
        /\ \A i \in 1..Len(RestSeq(t)) :
              LET c == RestSeq(t)[i] IN
              /\ st[c] = "live" /\ owned[c] /\ own[c] = t /\ q[c]
              /\ anc[c] = oanc[t] /\ c \notin OnSomeStack

(* No chunk on two stacks, or on a stack and the orphan chain. *)
Inv_NoDupListing ==
    /\ \A a, b \in Chunks : a # b /\ isA[a] /\ isA[b] /\ st[a] = "live" /\ st[b] = "live"
                             => StackSet(a) \cap StackSet(b) = {}
    /\ OnSomeStack \cap aorph = {}

(* References are exactly: chunks pointing at it, freers holding it, owners
   anchored on it. *)
Inv_RefOK ==
    ~NoAnchorRef => \A a \in Chunks :
        ref[a] = Cardinality({c \in Chunks : anc[c] = a})
                 + Cardinality({f \in Freers : TouchesA(f) /\ fa[f] = a})
                 + Cardinality({t \in Owners : oanc[t] = a})

=============================================================================
