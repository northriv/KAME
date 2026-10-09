(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        Dual-licensed Apache 2.0 OR GPL-2.0-or-later — see OrphanChain_atomicshared.tla.
 ***************************************************************************)
----------------------------- MODULE RevivalGroup -----------------------------
(*
 * Design model (stage 2b, not yet code): orphaning a thread's chunks as ONE
 * group.
 *
 * RevivalAnchor.tla orphans an exiting owner's chunks one at a time; between
 * the first and the last, the anchor can already be adopted while the rest
 * still point at it.  Here a chunk belongs to a GROUP, not to a thread: its
 * anc (atomic_shared_ptr) names the group's anchor for the chunk's whole life
 * in that group, and a thread owns at most one group.  Exit hands the whole
 * group over in one step.
 *
 *   - The anchor's head word collects chunks that got room back (pushed by
 *     the freer that took Q, as in RevivalStack's one-bit protocol).  It is
 *     never closed: an orphaned group keeps collecting.  Freers never touch
 *     the chains, need no reroute, and push the same way whoever owns the
 *     group.
 *   - Anchors are never pushed onto a head.  An owner checks its own anchor's
 *     room directly; an orphaned anchor's own room waits for the group to
 *     dissolve.
 *   - Orphaned groups live on one of two chains: ROOM (head non-empty when
 *     placed; popped one at a time by adopters -- in the code the serial-
 *     protected atomic_shared_ptr chain) and FULL (taken whole with one
 *     exchange by a sweeper, which runs only when ROOM is empty, i.e. just
 *     before mmap).  A group is in exactly one place: owned, on ROOM, on FULL,
 *     or held by the one thread processing it.  Only exiting owners, adopters
 *     and sweepers move groups.
 *   - Processing a held group: take its head (exchange); each chunk on it is
 *     released if empty, otherwise moved into the processor's own group
 *     (anc repointed: one reference moves).  Then the group is dissolved if
 *     it has nobody left (no other member, no freer holding it, its own slots
 *     all free), else placed on ROOM if its head is non-empty again, else on
 *     FULL.
 *   - Every live chunk is in a group, so a chunk's packed word never reaches
 *     zero by a free: freers never release anything; holders do.
 *
 * Knob (FALSE in the design): DissolveIgnoringRefs dissolves a group without
 * checking for freers that hold a reference to its anchor.
 *
 * Q protocol: RevivalStack's one-bit variant.  Its safety holds; room-loss
 * (Inv_NoLostRoom there) is not checked here.
 *
 * Results (3 chunks, 2 freers, 2 threads, K = 2, symmetry):
 *   design                 clean, 1,678,311 distinct states, depth 81
 *   dissolverefs           Inv_NoUseAfterRelease (23 steps)
 *   witnesses (each violated, i.e. reached): W_NoDissolve (5 steps),
 *   W_NoOrphanPush (17), W_NoSweptMove (21).
 * An earlier draft cleared Q on an adopted chunk before repointing anc; a
 * freer then took Q, loaded the old anchor and pushed the chunk onto the
 * group it had just left (see OAct).
 *)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS Chunks, Freers, Owners, NIL, NONE, K, DissolveIgnoringRefs

ASSUME NIL \notin Chunks /\ NONE \notin Owners /\ K \in Nat \ {0}
ASSUME DissolveIgnoringRefs \in BOOLEAN

Ptr == Chunks \cup {NIL}
Places == {"none", "owned", "room", "full", "held"}
MaxRef == Cardinality(Chunks) + Cardinality(Freers) + 1

VARIABLES
    st,      \* [Chunks -> {"fresh", "live", "released"}]
    bits,    \* [Chunks -> 0..K]
    mcnt,    \* [Chunks -> Nat]   MASK_CNT
    tok,     \* [Chunks -> 0..K]  live slots the application holds
    q,       \* [Chunks -> BOOLEAN]
    nx,      \* [Chunks -> Ptr]
    anc,     \* [Chunks -> Ptr]   the chunk's group (its anchor)
    isA,     \* [Chunks -> BOOLEAN]
    hd,      \* [Chunks -> Ptr]   head word of an anchor
    ref,     \* [Chunks -> Nat]   references to an anchor
    gloc,    \* [Chunks -> Places]  where an anchor's group is
    roomCh,  \* SUBSET Chunks
    fullCh,  \* SUBSET Chunks
    ost,     \* [Owners -> pc]
    oanc,    \* [Owners -> Ptr]   own group
    avail,   \* [Owners -> SUBSET Chunks]
    omode,   \* [Owners -> {"own", "adopt", "sweep"}]
    otgt,    \* [Owners -> Ptr]   group whose head is being processed
    ocur,    \* [Owners -> Ptr]
    onxt,    \* [Owners -> Ptr]
    ohold,   \* [Owners -> SUBSET Chunks]  groups taken by a sweep, not yet processed
    fpc, fc, fzero, fwasq, fa, fh

vars == <<st, bits, mcnt, tok, q, nx, anc, isA, hd, ref, gloc, roomCh, fullCh,
          ost, oanc, avail, omode, otgt, ocur, onxt, ohold,
          fpc, fc, fzero, fwasq, fa, fh>>
CVars == <<st, bits, mcnt, tok, q, nx, anc, isA, hd, ref, gloc, roomCh, fullCh>>
TVars == <<ost, oanc, avail, omode, otgt, ocur, onxt, ohold>>
FVars == <<fpc, fc, fzero, fwasq, fa, fh>>

TypeOK ==
    /\ st \in [Chunks -> {"fresh", "live", "released"}]
    /\ bits \in [Chunks -> 0..K] /\ mcnt \in [Chunks -> 0..(1 + Cardinality(Freers))]
    /\ tok \in [Chunks -> 0..K] /\ q \in [Chunks -> BOOLEAN]
    /\ nx \in [Chunks -> Ptr] /\ anc \in [Chunks -> Ptr] /\ isA \in [Chunks -> BOOLEAN]
    /\ hd \in [Chunks -> Ptr] /\ ref \in [Chunks -> 0..MaxRef]
    /\ gloc \in [Chunks -> Places] /\ roomCh \subseteq Chunks /\ fullCh \subseteq Chunks
    /\ oanc \in [Owners -> Ptr] /\ otgt \in [Owners -> Ptr] /\ fa \in [Freers -> Ptr]

RECURSIVE Walk(_, _)
Walk(c, n) == IF c = NIL \/ n = 0 THEN <<>> ELSE <<c>> \o Walk(nx[c], n - 1)
HeadSeq(a) == Walk(hd[a], Cardinality(Chunks) + 1)
HeadSet(a) == {HeadSeq(a)[i] : i \in 1..Len(HeadSeq(a))}
LiveAnchors == {a \in Chunks : isA[a] /\ st[a] = "live"}
OnHeads == UNION {HeadSet(a) : a \in LiveAnchors}
RECURSIVE Rest(_, _)
Rest(c, n) == IF c = NIL \/ n = 0 THEN {} ELSE {c} \cup Rest(nx[c], n - 1)
Processing(t) == ost[t] \in {"dnext", "dclr", "dact", "dqclr"}
DrainRest(t) == IF Processing(t)
                THEN Rest(ocur[t], Cardinality(Chunks) + 1) \cup Rest(onxt[t], Cardinality(Chunks) + 1)
                ELSE {}

Init ==
    /\ st = [c \in Chunks |-> "fresh"] /\ bits = [c \in Chunks |-> 0]
    /\ mcnt = [c \in Chunks |-> 0] /\ tok = [c \in Chunks |-> 0]
    /\ q = [c \in Chunks |-> FALSE] /\ nx = [c \in Chunks |-> NIL]
    /\ anc = [c \in Chunks |-> NIL] /\ isA = [c \in Chunks |-> FALSE]
    /\ hd = [c \in Chunks |-> NIL] /\ ref = [c \in Chunks |-> 0]
    /\ gloc = [c \in Chunks |-> "none"] /\ roomCh = {} /\ fullCh = {}
    /\ ost = [t \in Owners |-> "idle"] /\ oanc = [t \in Owners |-> NIL]
    /\ avail = [t \in Owners |-> {}] /\ omode = [t \in Owners |-> "own"]
    /\ otgt = [t \in Owners |-> NIL] /\ ocur = [t \in Owners |-> NIL]
    /\ onxt = [t \in Owners |-> NIL] /\ ohold = [t \in Owners |-> {}]
    /\ fpc = [f \in Freers |-> "idle"] /\ fc = [f \in Freers |-> NIL]
    /\ fzero = [f \in Freers |-> FALSE] /\ fwasq = [f \in Freers |-> FALSE]
    /\ fa = [f \in Freers |-> NIL] /\ fh = [f \in Freers |-> NIL]

Rel(c) == st' = [st EXCEPT ![c] = "released"]

(* A group nobody needs any more: no other member, no freer holding its
   anchor, the anchor's own slots all free. *)
Dissolvable(a) ==
    /\ \A c \in Chunks \ {a} : anc[c] # a
    /\ DissolveIgnoringRefs \/ ~ \E f \in Freers : fpc[f] \in {"pread", "pcas", "unref"} /\ fa[f] = a
    /\ bits[a] = 0 /\ mcnt[a] = 0

(* Put a held / exiting group where it belongs.  (The location reference
   moves with it; dissolving drops it and the anchor's self-reference.) *)
Place(a) ==
    IF Dissolvable(a)
    THEN /\ anc' = [anc EXCEPT ![a] = NIL] /\ ref' = [ref EXCEPT ![a] = 0]
         /\ gloc' = [gloc EXCEPT ![a] = "none"] /\ Rel(a)
         /\ UNCHANGED <<roomCh, fullCh>>
    ELSE IF hd[a] # NIL
         THEN /\ roomCh' = roomCh \cup {a} /\ gloc' = [gloc EXCEPT ![a] = "room"]
              /\ UNCHANGED <<anc, ref, st, fullCh>>
         ELSE /\ fullCh' = fullCh \cup {a} /\ gloc' = [gloc EXCEPT ![a] = "full"]
              /\ UNCHANGED <<anc, ref, st, roomCh>>

GoIdle(f) ==
    /\ fpc' = [fpc EXCEPT ![f] = "idle"] /\ fc' = [fc EXCEPT ![f] = NIL]
    /\ fzero' = [fzero EXCEPT ![f] = FALSE] /\ fwasq' = [fwasq EXCEPT ![f] = FALSE]
    /\ fa' = [fa EXCEPT ![f] = NIL] /\ fh' = [fh EXCEPT ![f] = NIL]

(******************************** threads ********************************)

OStart(t) ==
    /\ ost[t] = "idle" /\ ost' = [ost EXCEPT ![t] = "run"]
    /\ UNCHANGED CVars /\ UNCHANGED <<oanc, avail, omode, otgt, ocur, onxt, ohold>>
    /\ UNCHANGED FVars

(* A fresh chunk joins t's group; the first one becomes its anchor. *)
OAttach(t, c) ==
    LET a == IF oanc[t] = NIL THEN c ELSE oanc[t] IN
    /\ ost[t] = "run" /\ st[c] = "fresh"
    /\ st' = [st EXCEPT ![c] = "live"] /\ anc' = [anc EXCEPT ![c] = a]
    /\ IF oanc[t] = NIL
       THEN /\ isA' = [isA EXCEPT ![c] = TRUE] /\ ref' = [ref EXCEPT ![c] = 2]
            /\ gloc' = [gloc EXCEPT ![c] = "owned"] /\ oanc' = [oanc EXCEPT ![t] = c]
       ELSE /\ ref' = [ref EXCEPT ![a] = @ + 1] /\ UNCHANGED <<isA, gloc, oanc>>
    /\ avail' = [avail EXCEPT ![t] = @ \cup {c}]
    /\ UNCHANGED <<bits, mcnt, tok, q, nx, hd, roomCh, fullCh>>
    /\ UNCHANGED <<ost, omode, otgt, ocur, onxt, ohold>> /\ UNCHANGED FVars

OAlloc(t, c) ==
    /\ ost[t] = "run" /\ c \in avail[t] /\ bits[c] < K
    /\ bits' = [bits EXCEPT ![c] = @ + 1]
    /\ mcnt' = IF bits[c] = 0 THEN [mcnt EXCEPT ![c] = @ + 1] ELSE mcnt
    /\ tok' = [tok EXCEPT ![c] = @ + 1]
    /\ avail' = [avail EXCEPT ![t] = IF bits[c] + 1 = K THEN @ \ {c} ELSE @]
    /\ UNCHANGED <<st, q, nx, anc, isA, hd, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<ost, oanc, omode, otgt, ocur, onxt, ohold>> /\ UNCHANGED FVars

(* The owner looks at its own anchor's room directly (anchors are never on
   a head). *)
OAnchorCheck(t) ==
    LET a == oanc[t] IN
    /\ ost[t] = "run" /\ a # NIL /\ bits[a] < K /\ a \notin avail[t]
    /\ avail' = [avail EXCEPT ![t] = @ \cup {a}]
    /\ UNCHANGED CVars /\ UNCHANGED <<ost, oanc, omode, otgt, ocur, onxt, ohold>>
    /\ UNCHANGED FVars

(* Release an empty, unlisted member of t's own group (never the anchor). *)
ORelease(t, c) ==
    LET a == oanc[t] IN
    /\ ost[t] = "run" /\ st[c] = "live" /\ anc[c] = a /\ c # a
    /\ mcnt[c] = 0 /\ ~q[c]
    /\ anc' = [anc EXCEPT ![c] = NIL] /\ ref' = [ref EXCEPT ![a] = @ - 1] /\ Rel(c)
    /\ avail' = [avail EXCEPT ![t] = @ \ {c}]
    /\ UNCHANGED <<bits, mcnt, tok, q, nx, isA, hd, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<ost, oanc, omode, otgt, ocur, onxt, ohold>> /\ UNCHANGED FVars

(* Take a head with one exchange and walk it. *)
TakeHead(t, a, mode) ==
    /\ ocur' = [ocur EXCEPT ![t] = hd[a]] /\ hd' = [hd EXCEPT ![a] = NIL]
    /\ otgt' = [otgt EXCEPT ![t] = a] /\ omode' = [omode EXCEPT ![t] = mode]
    /\ ost' = [ost EXCEPT ![t] = "dnext"]

ODrainOwn(t) ==
    /\ ost[t] = "run" /\ oanc[t] # NIL
    /\ TakeHead(t, oanc[t], "own")
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, avail, onxt, ohold>> /\ UNCHANGED FVars

(* Adopt: pop one group from ROOM; it is ours to process. *)
OAdoptStart(t, a) ==
    /\ ost[t] = "run" /\ oanc[t] # NIL /\ a \in roomCh
    /\ roomCh' = roomCh \ {a} /\ gloc' = [gloc EXCEPT ![a] = "held"]
    /\ TakeHead(t, a, "adopt")
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, ref, fullCh>>
    /\ UNCHANGED <<oanc, avail, onxt, ohold>> /\ UNCHANGED FVars

(* Sweep: ROOM empty -> take the whole FULL chain with one exchange. *)
OSweepStart(t) ==
    /\ ost[t] = "run" /\ oanc[t] # NIL /\ roomCh = {} /\ fullCh # {}
    /\ ohold' = [ohold EXCEPT ![t] = fullCh] /\ fullCh' = {}
    /\ gloc' = [c \in Chunks |-> IF c \in fullCh THEN "held" ELSE gloc[c]]
    /\ ost' = [ost EXCEPT ![t] = "spick"]
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, hd, ref, roomCh>>
    /\ UNCHANGED <<oanc, avail, omode, otgt, ocur, onxt>> /\ UNCHANGED FVars

OSweepPick(t) ==
    /\ ost[t] = "spick"
    /\ IF ohold[t] = {}
       THEN /\ ost' = [ost EXCEPT ![t] = "run"]
            /\ UNCHANGED <<hd, ohold, otgt, omode, ocur>>
       ELSE \E a \in ohold[t] :
              /\ ohold' = [ohold EXCEPT ![t] = @ \ {a}]
              /\ TakeHead(t, a, "sweep")
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, avail, onxt>> /\ UNCHANGED FVars

(* Walk a taken head: read next, clear Q, act. *)
ONext(t) ==
    /\ ost[t] = "dnext"
    /\ IF ocur[t] = NIL
       THEN IF omode[t] = "own"
            THEN /\ ost' = [ost EXCEPT ![t] = "run"] /\ otgt' = [otgt EXCEPT ![t] = NIL]
                 /\ UNCHANGED <<anc, ref, st, gloc, roomCh, fullCh, onxt>>
            ELSE /\ Place(otgt[t]) /\ otgt' = [otgt EXCEPT ![t] = NIL]
                 /\ ost' = [ost EXCEPT ![t] = IF omode[t] = "sweep" THEN "spick" ELSE "run"]
                 /\ UNCHANGED onxt
       ELSE /\ onxt' = [onxt EXCEPT ![t] = nx[ocur[t]]]
            /\ ost' = [ost EXCEPT ![t] = IF omode[t] = "own" THEN "dclr" ELSE "dact"]
            /\ UNCHANGED <<anc, ref, st, gloc, roomCh, fullCh, otgt>>
    /\ UNCHANGED <<bits, mcnt, tok, q, nx, isA, hd>>
    /\ UNCHANGED <<oanc, avail, omode, ocur, ohold>> /\ UNCHANGED FVars

OClear(t) ==
    LET c == ocur[t] IN
    /\ ost[t] = "dclr"
    /\ q' = [q EXCEPT ![c] = FALSE] /\ nx' = [nx EXCEPT ![c] = NIL]
    /\ ost' = [ost EXCEPT ![t] = "dact"]
    /\ UNCHANGED <<st, bits, mcnt, tok, anc, isA, hd, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, avail, omode, otgt, ocur, onxt, ohold>> /\ UNCHANGED FVars

(* Own head (Q already cleared): room -> own list.  Another group's head --
   still HOLDING Q, so no freer pushes it meanwhile and anc cannot change
   under a freer that has loaded it: empty -> release it (Q dropped in the
   same CAS that zeroes the word); otherwise move it into our group (one
   reference moves) and list it, then clear Q ("dqclr").  Clearing Q first
   lets a freer take Q, load the OLD anchor, and push the chunk onto a group
   it has just left (found by this model). *)
OAct(t) ==
    LET c == ocur[t]  a == otgt[t]  me == oanc[t] IN
    /\ ost[t] = "dact"
    /\ IF omode[t] = "own"
       THEN /\ avail' = [avail EXCEPT ![t] = IF bits[c] < K THEN @ \cup {c} ELSE @]
            /\ UNCHANGED <<anc, ref, st, q>>
            /\ ocur' = [ocur EXCEPT ![t] = onxt[t]] /\ onxt' = [onxt EXCEPT ![t] = NIL]
            /\ ost' = [ost EXCEPT ![t] = "dnext"]
       ELSE IF bits[c] = 0 /\ mcnt[c] = 0
            THEN /\ anc' = [anc EXCEPT ![c] = NIL] /\ ref' = [ref EXCEPT ![a] = @ - 1]
                 /\ q' = [q EXCEPT ![c] = FALSE] /\ Rel(c) /\ UNCHANGED avail
                 /\ ocur' = [ocur EXCEPT ![t] = onxt[t]] /\ onxt' = [onxt EXCEPT ![t] = NIL]
                 /\ ost' = [ost EXCEPT ![t] = "dnext"]
            ELSE /\ anc' = [anc EXCEPT ![c] = me]
                 /\ ref' = [ref EXCEPT ![a] = @ - 1, ![me] = @ + 1]
                 /\ avail' = [avail EXCEPT ![t] = IF bits[c] < K THEN @ \cup {c} ELSE @]
                 /\ UNCHANGED <<st, q, ocur, onxt>>
                 /\ ost' = [ost EXCEPT ![t] = "dqclr"]
    /\ UNCHANGED <<bits, mcnt, tok, nx, isA, hd, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, omode, otgt, ohold>> /\ UNCHANGED FVars

ODqClr(t) ==
    LET c == ocur[t] IN
    /\ ost[t] = "dqclr"
    /\ q' = [q EXCEPT ![c] = FALSE] /\ nx' = [nx EXCEPT ![c] = NIL]
    /\ ocur' = [ocur EXCEPT ![t] = onxt[t]] /\ onxt' = [onxt EXCEPT ![t] = NIL]
    /\ ost' = [ost EXCEPT ![t] = "dnext"]
    /\ UNCHANGED <<st, bits, mcnt, tok, anc, isA, hd, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, avail, omode, otgt, ohold>> /\ UNCHANGED FVars

(* Exit: list what is on our own list (taking Q), then hand the group over. *)
OExitList(t, c) ==
    LET a == oanc[t] IN
    /\ ost[t] = "run" /\ c \in avail[t]
    /\ avail' = [avail EXCEPT ![t] = @ \ {c}]
    /\ IF ~q[c] /\ ~isA[c]
       THEN /\ q' = [q EXCEPT ![c] = TRUE] /\ nx' = [nx EXCEPT ![c] = hd[a]]
            /\ hd' = [hd EXCEPT ![a] = c]
       ELSE UNCHANGED <<q, nx, hd>>
    /\ ost' = [ost EXCEPT ![t] = "xlist"]
    /\ UNCHANGED <<st, bits, mcnt, tok, anc, isA, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, omode, otgt, ocur, onxt, ohold>> /\ UNCHANGED FVars

OExitMore(t, c) ==
    LET a == oanc[t] IN
    /\ ost[t] = "xlist" /\ c \in avail[t]
    /\ avail' = [avail EXCEPT ![t] = @ \ {c}]
    /\ IF ~q[c] /\ ~isA[c]
       THEN /\ q' = [q EXCEPT ![c] = TRUE] /\ nx' = [nx EXCEPT ![c] = hd[a]]
            /\ hd' = [hd EXCEPT ![a] = c]
       ELSE UNCHANGED <<q, nx, hd>>
    /\ UNCHANGED <<st, bits, mcnt, tok, anc, isA, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<ost, oanc, omode, otgt, ocur, onxt, ohold>> /\ UNCHANGED FVars

OExitPlace(t) ==
    LET a == oanc[t] IN
    /\ \/ ost[t] = "xlist" /\ avail[t] = {}
       \/ ost[t] = "run" /\ avail[t] = {} /\ a # NIL
    /\ Place(a)
    /\ oanc' = [oanc EXCEPT ![t] = NIL] /\ ost' = [ost EXCEPT ![t] = "dead"]
    /\ UNCHANGED <<bits, mcnt, tok, q, nx, isA, hd>>
    /\ UNCHANGED <<avail, omode, otgt, ocur, onxt, ohold>> /\ UNCHANGED FVars

OExitNoGroup(t) ==
    /\ ost[t] = "run" /\ oanc[t] = NIL /\ ost' = [ost EXCEPT ![t] = "dead"]
    /\ UNCHANGED CVars /\ UNCHANGED <<oanc, avail, omode, otgt, ocur, onxt, ohold>>
    /\ UNCHANGED FVars

(******************************** freers ********************************)

(* An anchor's slot: just return it (anchors are never listed). *)
FPick(f, c) ==
    /\ fpc[f] = "idle" /\ st[c] = "live" /\ tok[c] > 0
    /\ tok' = [tok EXCEPT ![c] = @ - 1] /\ fc' = [fc EXCEPT ![f] = c]
    /\ IF isA[c]
       THEN /\ fpc' = [fpc EXCEPT ![f] = "fb"] /\ fwasq' = [fwasq EXCEPT ![f] = TRUE]
       ELSE /\ fpc' = [fpc EXCEPT ![f] = "fq1"] /\ UNCHANGED fwasq
    /\ UNCHANGED <<st, bits, mcnt, q, nx, anc, isA, hd, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED TVars /\ UNCHANGED <<fzero, fa, fh>>

FTakeQ(f) ==
    LET c == fc[f] IN
    /\ fpc[f] = "fq1"
    /\ fwasq' = [fwasq EXCEPT ![f] = q[c]] /\ q' = [q EXCEPT ![c] = TRUE]
    /\ fpc' = [fpc EXCEPT ![f] = "fb"]
    /\ UNCHANGED <<st, bits, mcnt, tok, nx, anc, isA, hd, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED TVars /\ UNCHANGED <<fc, fzero, fa, fh>>

FBitClear(f) ==
    LET c == fc[f] IN
    /\ fpc[f] = "fb"
    /\ bits' = [bits EXCEPT ![c] = @ - 1]
    /\ IF bits[c] = 1
       THEN /\ fpc' = [fpc EXCEPT ![f] = "fd"] /\ fzero' = [fzero EXCEPT ![f] = TRUE]
            /\ UNCHANGED <<fc, fwasq, fa, fh>>
       ELSE IF fwasq[f] THEN GoIdle(f)
       ELSE /\ fpc' = [fpc EXCEPT ![f] = "fload"] /\ UNCHANGED <<fc, fzero, fwasq, fa, fh>>
    /\ UNCHANGED <<st, mcnt, tok, q, nx, anc, isA, hd, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED TVars

(* MASK_CNT--.  Every live chunk is in a group (BIT_OWNED), so this never
   releases. *)
FDec(f) ==
    /\ fpc[f] = "fd"
    /\ mcnt' = [mcnt EXCEPT ![fc[f]] = @ - 1]
    /\ IF fwasq[f] THEN GoIdle(f)
       ELSE /\ fpc' = [fpc EXCEPT ![f] = "fload"] /\ UNCHANGED <<fc, fzero, fwasq, fa, fh>>
    /\ UNCHANGED <<st, bits, tok, q, nx, anc, isA, hd, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED TVars

(* Holding Q: counted load of the group's anchor. *)
FLoadAnc(f) ==
    LET a == anc[fc[f]] IN
    /\ fpc[f] = "fload"
    /\ fa' = [fa EXCEPT ![f] = a] /\ ref' = [ref EXCEPT ![a] = @ + 1]
    /\ fpc' = [fpc EXCEPT ![f] = "pread"]
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, hd, gloc, roomCh, fullCh>>
    /\ UNCHANGED TVars /\ UNCHANGED <<fc, fzero, fwasq, fh>>

FPushRead(f) ==
    /\ fpc[f] = "pread"
    /\ fh' = [fh EXCEPT ![f] = hd[fa[f]]] /\ nx' = [nx EXCEPT ![fc[f]] = hd[fa[f]]]
    /\ fpc' = [fpc EXCEPT ![f] = "pcas"]
    /\ UNCHANGED <<st, bits, mcnt, tok, q, anc, isA, hd, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED TVars /\ UNCHANGED <<fc, fzero, fwasq, fa>>

FPushCAS(f) ==
    LET a == fa[f] IN
    /\ fpc[f] = "pcas"
    /\ IF hd[a] = fh[f]
       THEN /\ hd' = [hd EXCEPT ![a] = fc[f]] /\ fpc' = [fpc EXCEPT ![f] = "unref"]
       ELSE /\ fpc' = [fpc EXCEPT ![f] = "pread"] /\ UNCHANGED hd
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED TVars /\ UNCHANGED <<fc, fzero, fwasq, fa, fh>>

FUnref(f) ==
    /\ fpc[f] = "unref"
    /\ ref' = [ref EXCEPT ![fa[f]] = @ - 1] /\ GoIdle(f)
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, hd, gloc, roomCh, fullCh>>
    /\ UNCHANGED TVars

Next ==
    \/ \E t \in Owners :
         \/ OStart(t) \/ OAnchorCheck(t) \/ ODrainOwn(t) \/ OSweepStart(t)
         \/ OSweepPick(t) \/ ONext(t) \/ OClear(t) \/ OAct(t) \/ ODqClr(t)
         \/ OExitPlace(t)
         \/ OExitNoGroup(t)
         \/ \E c \in Chunks : OAttach(t, c) \/ OAlloc(t, c) \/ ORelease(t, c)
                              \/ OAdoptStart(t, c) \/ OExitList(t, c) \/ OExitMore(t, c)
    \/ \E f \in Freers :
         \/ \E c \in Chunks : FPick(f, c)
         \/ FTakeQ(f) \/ FBitClear(f) \/ FDec(f) \/ FLoadAnc(f)
         \/ FPushRead(f) \/ FPushCAS(f) \/ FUnref(f)

Spec == Init /\ [][Next]_vars

Symm == Permutations(Chunks) \cup Permutations(Freers) \cup Permutations(Owners)

(******************************** properties ********************************)

TouchesC(f) == fpc[f] \in {"fq1", "fb", "fd", "fload", "pread", "pcas"}
TouchesA(f) == fpc[f] \in {"pread", "pcas", "unref"}
Held(t) == (IF omode[t] \in {"adopt", "sweep"} /\ otgt[t] # NIL THEN {otgt[t]} ELSE {})
           \cup ohold[t]

Inv_NoUseAfterRelease ==
    /\ \A f \in Freers : TouchesC(f) => st[fc[f]] = "live"
    /\ \A f \in Freers : TouchesA(f) => st[fa[f]] = "live"
    /\ \A t \in Owners : \A c \in ({oanc[t], ocur[t], onxt[t]} \ {NIL}) \cup Held(t)
                                   \cup avail[t] : st[c] = "live"
    /\ \A c \in Chunks : anc[c] # NIL => st[anc[c]] = "live"
    /\ \A c \in OnHeads \cup roomCh \cup fullCh : st[c] = "live"

(* Every live chunk is in a live group; heads list their own group's
   non-anchor members, once each. *)
Inv_GroupOK ==
    /\ \A c \in Chunks : st[c] = "live" => anc[c] # NIL /\ isA[anc[c]]
    /\ \A a \in LiveAnchors :
         /\ anc[a] = a
         /\ Len(HeadSeq(a)) <= Cardinality(Chunks)
         /\ \A i, j \in 1..Len(HeadSeq(a)) : i # j => HeadSeq(a)[i] # HeadSeq(a)[j]
         /\ \A c \in HeadSet(a) : q[c] /\ anc[c] = a /\ ~isA[c]

(* A group is in exactly one place. *)
Inv_PlaceOK ==
    \A a \in LiveAnchors :
        LET own  == {t \in Owners : oanc[t] = a}
            held == {t \in Owners : a \in Held(t)}
        IN  /\ Cardinality(own) + Cardinality(held)
               + (IF a \in roomCh THEN 1 ELSE 0) + (IF a \in fullCh THEN 1 ELSE 0) = 1
            /\ gloc[a] = IF own # {} THEN "owned" ELSE IF held # {} THEN "held"
                         ELSE IF a \in roomCh THEN "room" ELSE "full"

Inv_RefOK ==
    \A a \in LiveAnchors :
        ref[a] = Cardinality({c \in Chunks : anc[c] = a})
                 + Cardinality({f \in Freers : TouchesA(f) /\ fa[f] = a}) + 1

Inv_AvailOK == \A t \in Owners : \A c \in avail[t] : anc[c] = oanc[t]

Inv_QAccounted ==
    \A c \in Chunks : st[c] = "live" /\ q[c] =>
        \/ c \in OnHeads
        \/ \E f \in Freers : fpc[f] \in {"fb", "fd", "fload", "pread", "pcas"}
                              /\ fc[f] = c /\ ~fwasq[f]
        \/ \E t \in Owners : c \in DrainRest(t)

(* Witnesses, NOT invariants of the design: each must be violated, showing
   the model reaches the situation (RevivalGroup_witness_*.cfg). *)
W_NoDissolve == ~ \E a \in Chunks : isA[a] /\ st[a] = "released"
W_NoOrphanPush == ~ \E f \in Freers : fpc[f] = "unref" /\ gloc[fa[f]] \in {"room", "full", "held"}
W_NoSweptMove == ~ \E t \in Owners : omode[t] = "sweep" /\ ost[t] = "dqclr"

=============================================================================
