(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        Dual-licensed Apache 2.0 OR GPL-2.0-or-later — see OrphanChain_atomicshared.tla.
 ***************************************************************************)
----------------------------- MODULE RevivalGroup -----------------------------
(*
 * Orphaning a thread's chunks as ONE group -- kamepoolalloc stage 2b
 * ("§group"); the code is the code cfg (knobs below).
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
 *     dissolve.  (AnchorListed, the code: an anchor is listed on its own
 *     group's head like a member, and a holder only unlists it there.)
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
 * The code (kamepoolalloc stage 2b, "§group") also needs, each TRUE in the
 * code cfg:
 *   TakeOver       a thread with no group takes one over -- popped off ROOM,
 *                  or the first one a sweep picks -- instead of founding one
 *                  with a fresh chunk (short-lived threads would otherwise
 *                  mmap a chunk per template; and only a sweep notices that
 *                  an orphaned anchor has emptied, so threads with one slow
 *                  path per template must be able to sweep).
 *   LazyDrain      the own head is taken only when the rest of the last take
 *                  is used up, then popped one chunk per try; the rest keeps
 *                  Q; at exit it goes back onto the head.
 *   MoveToRest     a holder keeps Q on a chunk it moves into its own group and
 *                  puts it on its rest (needs LazyDrain).
 *   ExitAnyMember  at exit any member may be listed (the code lists the DLL
 *                  chunks with room), and empty members may be released.
 *   ExitDrainsHead at exit, after listing, the owner takes its own head and
 *                  settles it as a holder would, except that what stays is
 *                  listed again: the anchor unlisted, empty chunks released.
 *                  (Otherwise a listed anchor kept every exiting group from
 *                  dissolving, and nothing was freed after the last worker:
 *                  alloc_stress_test post-workers chunks = peak.)
 *   AnchorListed   an anchor is listed on its own group's head like any
 *                  member (a free takes its Q and pushes it); a holder of an
 *                  orphaned group only unlists it, and a listed anchor's
 *                  group does not dissolve.  (Unlisted, the owner looked at
 *                  its anchor's room only when the rest of its last take ran
 *                  out, and a producer kept re-pinning the anchor its
 *                  consumer was still freeing into: bench_xthread_pool -s256
 *                  lost 18 %.)
 *
 * Q protocol: RevivalStack's one-bit variant.  Its safety holds; room-loss
 * (Inv_NoLostRoom there) is not checked here.
 *
 * Results (3 chunks, 2 freers, 2 threads, K = 2, symmetry):
 *   design                 clean, 1,678,311 distinct states
 *   code                   clean, 57,428,577 distinct states
 *   code4f1 (4 chunks, 1 freer)  clean, 51,915,141 distinct states, depth 91
 *   code with 4 chunks and 2 freers: stopped unfinished, no violation in the
 *   first 915,434,410 distinct states (breadth-first, 2 h 10 min on 22
 *   workers, queue still growing)
 *   dissolverefs           Inv_NoUseAfterRelease (23-state trace)
 *   witnesses -- each violated, i.e. reached (trace length).  Under design:
 *   W_NoDissolve (5), W_NoOrphanPush (17), W_NoSweptMove (21).  Under code:
 *   the same three (7, 14, 20) and, as an Assert in the branch: a takeover
 *   off ROOM (15), a sweep taking a group over (10), the rest pushed back at
 *   exit (12), a release while exiting (5), a popped chunk with room (15), a
 *   chunk moved onto the holder's rest (17), an anchor pushed onto its own
 *   head (11), an adopter/sweeper unlisting the group's own anchor (18), and
 *   at exit: a chunk listed again (9), an empty listed chunk released (8),
 *   the anchor unlisted (13).
 * An earlier draft cleared Q on an adopted chunk before repointing anc; a
 * freer then took Q, loaded the old anchor and pushed the chunk onto the
 * group it had just left (see OAct).
 *)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS Chunks, Freers, Owners, NIL, NONE, K, DissolveIgnoringRefs,
          TakeOver, LazyDrain, ExitAnyMember, MoveToRest, AnchorListed, ExitDrainsHead

ASSUME NIL \notin Chunks /\ NONE \notin Owners /\ K \in Nat \ {0}
ASSUME DissolveIgnoringRefs \in BOOLEAN
ASSUME TakeOver \in BOOLEAN /\ LazyDrain \in BOOLEAN /\ ExitAnyMember \in BOOLEAN
ASSUME MoveToRest \in BOOLEAN /\ (MoveToRest => LazyDrain) /\ AnchorListed \in BOOLEAN
ASSUME ExitDrainsHead \in BOOLEAN

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
    orest,   \* [Owners -> Ptr]   LazyDrain: the rest of the last take of the own head
    opop,    \* [Owners -> Ptr]   LazyDrain: the chunk just popped off orest
    fpc, fc, fzero, fwasq, fa, fh

vars == <<st, bits, mcnt, tok, q, nx, anc, isA, hd, ref, gloc, roomCh, fullCh,
          ost, oanc, avail, omode, otgt, ocur, onxt, ohold, orest, opop,
          fpc, fc, fzero, fwasq, fa, fh>>
CVars == <<st, bits, mcnt, tok, q, nx, anc, isA, hd, ref, gloc, roomCh, fullCh>>
TVars == <<ost, oanc, avail, omode, otgt, ocur, onxt, ohold, orest, opop>>
FVars == <<fpc, fc, fzero, fwasq, fa, fh>>

TypeOK ==
    /\ st \in [Chunks -> {"fresh", "live", "released"}]
    /\ bits \in [Chunks -> 0..K] /\ mcnt \in [Chunks -> 0..(1 + Cardinality(Freers))]
    /\ tok \in [Chunks -> 0..K] /\ q \in [Chunks -> BOOLEAN]
    /\ nx \in [Chunks -> Ptr] /\ anc \in [Chunks -> Ptr] /\ isA \in [Chunks -> BOOLEAN]
    /\ hd \in [Chunks -> Ptr] /\ ref \in [Chunks -> 0..MaxRef]
    /\ gloc \in [Chunks -> Places] /\ roomCh \subseteq Chunks /\ fullCh \subseteq Chunks
    /\ oanc \in [Owners -> Ptr] /\ otgt \in [Owners -> Ptr] /\ fa \in [Freers -> Ptr]
    /\ orest \in [Owners -> Ptr] /\ opop \in [Owners -> Ptr]

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
    /\ orest = [t \in Owners |-> NIL] /\ opop = [t \in Owners |-> NIL]
    /\ fpc = [f \in Freers |-> "idle"] /\ fc = [f \in Freers |-> NIL]
    /\ fzero = [f \in Freers |-> FALSE] /\ fwasq = [f \in Freers |-> FALSE]
    /\ fa = [f \in Freers |-> NIL] /\ fh = [f \in Freers |-> NIL]

Rel(c) == st' = [st EXCEPT ![c] = "released"]

(* A group nobody needs any more: no other member, no freer holding its
   anchor, the anchor's own slots all free. *)
Dissolvable(a) ==
    /\ ~q[a]
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
    /\ UNCHANGED <<orest, opop>>

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
    /\ UNCHANGED <<orest, opop>>

OAlloc(t, c) ==
    /\ ost[t] = "run" /\ c \in avail[t] /\ bits[c] < K
    /\ bits' = [bits EXCEPT ![c] = @ + 1]
    /\ mcnt' = IF bits[c] = 0 THEN [mcnt EXCEPT ![c] = @ + 1] ELSE mcnt
    /\ tok' = [tok EXCEPT ![c] = @ + 1]
    /\ avail' = [avail EXCEPT ![t] = IF bits[c] + 1 = K THEN @ \ {c} ELSE @]
    /\ UNCHANGED <<st, q, nx, anc, isA, hd, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<ost, oanc, omode, otgt, ocur, onxt, ohold>> /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

(* The owner looks at its own anchor's room directly (anchors are never on
   a head). *)
OAnchorCheck(t) ==
    LET a == oanc[t] IN
    /\ ost[t] = "run" /\ a # NIL /\ bits[a] < K /\ a \notin avail[t]
    /\ avail' = [avail EXCEPT ![t] = @ \cup {a}]
    /\ UNCHANGED CVars /\ UNCHANGED <<ost, oanc, omode, otgt, ocur, onxt, ohold>>
    /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

(* Release an empty, unlisted member of t's own group (never the anchor). *)
ORelease(t, c) ==
    LET a == oanc[t] IN
    /\ IF ExitAnyMember THEN ost[t] \in {"run", "xlist"} ELSE ost[t] = "run"
    /\ st[c] = "live" /\ anc[c] = a /\ c # a /\ a # NIL
    /\ mcnt[c] = 0 /\ ~q[c]
    /\ anc' = [anc EXCEPT ![c] = NIL] /\ ref' = [ref EXCEPT ![a] = @ - 1] /\ Rel(c)
    /\ avail' = [avail EXCEPT ![t] = @ \ {c}]
    /\ UNCHANGED <<bits, mcnt, tok, q, nx, isA, hd, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<ost, oanc, omode, otgt, ocur, onxt, ohold>> /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

(* Take a head with one exchange and walk it. *)
TakeHead(t, a, mode) ==
    /\ ocur' = [ocur EXCEPT ![t] = hd[a]] /\ hd' = [hd EXCEPT ![a] = NIL]
    /\ otgt' = [otgt EXCEPT ![t] = a] /\ omode' = [omode EXCEPT ![t] = mode]
    /\ ost' = [ost EXCEPT ![t] = "dnext"]

ODrainOwn(t) ==
    /\ ~LazyDrain /\ ost[t] = "run" /\ oanc[t] # NIL
    /\ TakeHead(t, oanc[t], "own")
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, avail, onxt, ohold>> /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

(* Adopt: pop one group from ROOM; it is ours to process. *)
OAdoptStart(t, a) ==
    /\ ost[t] = "run" /\ oanc[t] # NIL /\ a \in roomCh
    /\ roomCh' = roomCh \ {a} /\ gloc' = [gloc EXCEPT ![a] = "held"]
    /\ TakeHead(t, a, "adopt")
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, ref, fullCh>>
    /\ UNCHANGED <<oanc, avail, onxt, ohold>> /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

(* Sweep: ROOM empty -> take the whole FULL chain with one exchange. *)
OSweepStart(t) ==
    /\ ost[t] = "run" /\ (oanc[t] # NIL \/ TakeOver) /\ roomCh = {} /\ fullCh # {}
    /\ ohold' = [ohold EXCEPT ![t] = fullCh] /\ fullCh' = {}
    /\ gloc' = [c \in Chunks |-> IF c \in fullCh THEN "held" ELSE gloc[c]]
    /\ ost' = [ost EXCEPT ![t] = "spick"]
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, hd, ref, roomCh>>
    /\ UNCHANGED <<oanc, avail, omode, otgt, ocur, onxt>> /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

(* TakeOver: a sweeper without a group takes over the first group it picks
   (its head becomes the own head, drained as such), then processes the rest
   into it. *)
OSweepPick(t) ==
    /\ ost[t] = "spick"
    /\ IF ohold[t] = {}
       THEN /\ ost' = [ost EXCEPT ![t] = "run"]
            /\ UNCHANGED <<hd, ohold, otgt, omode, ocur, oanc, gloc>>
       ELSE \E a \in ohold[t] :
              /\ ohold' = [ohold EXCEPT ![t] = @ \ {a}]
              /\ IF oanc[t] = NIL
                 THEN /\ oanc' = [oanc EXCEPT ![t] = a]
                      /\ gloc' = [gloc EXCEPT ![a] = "owned"]
                      /\ UNCHANGED <<hd, otgt, omode, ocur, ost>>
                 ELSE /\ TakeHead(t, a, "sweep") /\ UNCHANGED <<oanc, gloc>>
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, ref, roomCh, fullCh>>
    /\ UNCHANGED <<avail, onxt>> /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

(* Walk a taken head: read next, clear Q, act. *)
ONext(t) ==
    /\ ost[t] = "dnext"
    /\ IF ocur[t] = NIL
       THEN IF omode[t] \in {"own", "exit"}
            THEN /\ ost' = [ost EXCEPT ![t] = IF omode[t] = "exit" THEN "xdone" ELSE "run"]
                 /\ otgt' = [otgt EXCEPT ![t] = NIL]
                 /\ UNCHANGED <<anc, ref, st, gloc, roomCh, fullCh, onxt>>
            ELSE /\ Place(otgt[t]) /\ otgt' = [otgt EXCEPT ![t] = NIL]
                 /\ ost' = [ost EXCEPT ![t] = IF omode[t] = "sweep" THEN "spick" ELSE "run"]
                 /\ UNCHANGED onxt
       ELSE /\ onxt' = [onxt EXCEPT ![t] = nx[ocur[t]]]
            /\ ost' = [ost EXCEPT ![t] = IF omode[t] = "own" THEN "dclr" ELSE "dact"]
            /\ UNCHANGED <<anc, ref, st, gloc, roomCh, fullCh, otgt>>
    /\ UNCHANGED <<bits, mcnt, tok, q, nx, isA, hd>>
    /\ UNCHANGED <<oanc, avail, omode, ocur, ohold>> /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

OClear(t) ==
    LET c == ocur[t] IN
    /\ ost[t] = "dclr"
    /\ q' = [q EXCEPT ![c] = FALSE] /\ nx' = [nx EXCEPT ![c] = NIL]
    /\ ost' = [ost EXCEPT ![t] = "dact"]
    /\ UNCHANGED <<st, bits, mcnt, tok, anc, isA, hd, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, avail, omode, otgt, ocur, onxt, ohold>> /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

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
            /\ UNCHANGED <<anc, ref, st, q, nx, orest, hd>>
            /\ ocur' = [ocur EXCEPT ![t] = onxt[t]] /\ onxt' = [onxt EXCEPT ![t] = NIL]
            /\ ost' = [ost EXCEPT ![t] = "dnext"]
       ELSE IF c = a
            THEN (* AnchorListed: the held group's own anchor stays; unlist it *)
                 /\ UNCHANGED <<anc, ref, st, q, avail, nx, orest, ocur, onxt, hd>>
                 /\ ost' = [ost EXCEPT ![t] = "dqclr"]
       ELSE IF bits[c] = 0 /\ mcnt[c] = 0
            THEN /\ anc' = [anc EXCEPT ![c] = NIL] /\ ref' = [ref EXCEPT ![a] = @ - 1]
                 /\ q' = [q EXCEPT ![c] = FALSE] /\ Rel(c) /\ UNCHANGED <<avail, nx, orest, hd>>
                 /\ ocur' = [ocur EXCEPT ![t] = onxt[t]] /\ onxt' = [onxt EXCEPT ![t] = NIL]
                 /\ ost' = [ost EXCEPT ![t] = "dnext"]
            ELSE IF omode[t] = "exit"
            THEN (* ExitDrainsHead: it stays in our group; list it again *)
                 /\ nx' = [nx EXCEPT ![c] = hd[a]] /\ hd' = [hd EXCEPT ![a] = c]
                 /\ UNCHANGED <<anc, ref, st, q, avail, orest>>
                 /\ ocur' = [ocur EXCEPT ![t] = onxt[t]] /\ onxt' = [onxt EXCEPT ![t] = NIL]
                 /\ ost' = [ost EXCEPT ![t] = "dnext"]
            ELSE IF MoveToRest
            THEN (* the code: keep Q, put it on our own rest; a later pop
                    drops Q and looks at its room *)
                 /\ anc' = [anc EXCEPT ![c] = me]
                 /\ ref' = [ref EXCEPT ![a] = @ - 1, ![me] = @ + 1]
                 /\ nx' = [nx EXCEPT ![c] = orest[t]] /\ orest' = [orest EXCEPT ![t] = c]
                 /\ UNCHANGED <<st, q, avail, hd>>
                 /\ ocur' = [ocur EXCEPT ![t] = onxt[t]] /\ onxt' = [onxt EXCEPT ![t] = NIL]
                 /\ ost' = [ost EXCEPT ![t] = "dnext"]
            ELSE /\ anc' = [anc EXCEPT ![c] = me]
                 /\ ref' = [ref EXCEPT ![a] = @ - 1, ![me] = @ + 1]
                 /\ avail' = [avail EXCEPT ![t] = IF bits[c] < K THEN @ \cup {c} ELSE @]
                 /\ UNCHANGED <<st, q, ocur, onxt, nx, orest, hd>>
                 /\ ost' = [ost EXCEPT ![t] = "dqclr"]
    /\ UNCHANGED <<bits, mcnt, tok, isA, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, omode, otgt, ohold>> /\ UNCHANGED FVars
    /\ UNCHANGED opop

ODqClr(t) ==
    LET c == ocur[t] IN
    /\ ost[t] = "dqclr"
    /\ q' = [q EXCEPT ![c] = FALSE] /\ nx' = [nx EXCEPT ![c] = NIL]
    /\ ocur' = [ocur EXCEPT ![t] = onxt[t]] /\ onxt' = [onxt EXCEPT ![t] = NIL]
    /\ ost' = [ost EXCEPT ![t] = "dnext"]
    /\ UNCHANGED <<st, bits, mcnt, tok, anc, isA, hd, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, avail, omode, otgt, ohold>> /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

(* Exit: list what is on our own list (taking Q), then hand the group over. *)
OExitList(t, c) ==
    LET a == oanc[t] IN
    /\ ~ExitAnyMember /\ ost[t] = "run" /\ c \in avail[t]
    /\ avail' = [avail EXCEPT ![t] = @ \ {c}]
    /\ IF ~q[c] /\ ~isA[c]
       THEN /\ q' = [q EXCEPT ![c] = TRUE] /\ nx' = [nx EXCEPT ![c] = hd[a]]
            /\ hd' = [hd EXCEPT ![a] = c]
       ELSE UNCHANGED <<q, nx, hd>>
    /\ ost' = [ost EXCEPT ![t] = "xlist"]
    /\ UNCHANGED <<st, bits, mcnt, tok, anc, isA, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, omode, otgt, ocur, onxt, ohold>> /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

OExitMore(t, c) ==
    LET a == oanc[t] IN
    /\ ~ExitAnyMember /\ ost[t] = "xlist" /\ c \in avail[t]
    /\ avail' = [avail EXCEPT ![t] = @ \ {c}]
    /\ IF ~q[c] /\ ~isA[c]
       THEN /\ q' = [q EXCEPT ![c] = TRUE] /\ nx' = [nx EXCEPT ![c] = hd[a]]
            /\ hd' = [hd EXCEPT ![a] = c]
       ELSE UNCHANGED <<q, nx, hd>>
    /\ UNCHANGED <<st, bits, mcnt, tok, anc, isA, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<ost, oanc, omode, otgt, ocur, onxt, ohold>> /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

(* TakeOver: a thread without a group pops one off ROOM and makes it its
   own (the location reference becomes the owner's).  It then drains the
   head as its own and checks the anchor directly, as for a group it made. *)
OTakeOver(t, a) ==
    /\ TakeOver /\ ost[t] = "run" /\ oanc[t] = NIL /\ a \in roomCh
    /\ roomCh' = roomCh \ {a} /\ gloc' = [gloc EXCEPT ![a] = "owned"]
    /\ oanc' = [oanc EXCEPT ![t] = a]
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, hd, ref, fullCh>>
    /\ UNCHANGED <<ost, avail, omode, otgt, ocur, onxt, ohold, orest, opop>>
    /\ UNCHANGED FVars

(* LazyDrain (the code): take the own head only when the rest of the last
   take is used up; pop one chunk of the rest per try, running in between.
   The rest keeps Q.  A pop reads the link, drops Q, then looks at room. *)
ODrainLazy(t) ==
    LET a == oanc[t] IN
    /\ LazyDrain /\ ost[t] = "run" /\ a # NIL /\ orest[t] = NIL
    /\ orest' = [orest EXCEPT ![t] = hd[a]] /\ hd' = [hd EXCEPT ![a] = NIL]
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<ost, oanc, avail, omode, otgt, ocur, onxt, ohold, opop>>
    /\ UNCHANGED FVars

OPopOwn(t) ==
    LET c == orest[t] IN
    /\ LazyDrain /\ ost[t] = "run" /\ c # NIL
    /\ orest' = [orest EXCEPT ![t] = nx[c]]
    /\ q' = [q EXCEPT ![c] = FALSE] /\ nx' = [nx EXCEPT ![c] = NIL]
    /\ opop' = [opop EXCEPT ![t] = c] /\ ost' = [ost EXCEPT ![t] = "ppop"]
    /\ UNCHANGED <<st, bits, mcnt, tok, anc, isA, hd, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, avail, omode, otgt, ocur, onxt, ohold>>
    /\ UNCHANGED FVars

OPopAct(t) ==
    LET c == opop[t] IN
    /\ ost[t] = "ppop"
    /\ avail' = [avail EXCEPT ![t] = IF bits[c] < K THEN @ \cup {c} ELSE @]
    /\ opop' = [opop EXCEPT ![t] = NIL] /\ ost' = [ost EXCEPT ![t] = "run"]
    /\ UNCHANGED CVars
    /\ UNCHANGED <<oanc, omode, otgt, ocur, onxt, ohold, orest>>
    /\ UNCHANGED FVars

(* ExitAnyMember (the code): exiting, list any member (taking Q if it is
   clear -- the code lists the DLL chunks with room), dropping it from avail;
   the anchor is only dropped from avail. *)
OExitAny(t, c) ==
    LET a == oanc[t] IN
    /\ ExitAnyMember /\ ost[t] \in {"run", "xlist"} /\ a # NIL
    /\ st[c] = "live" /\ anc[c] = a
    /\ avail' = [avail EXCEPT ![t] = @ \ {c}]
    /\ IF ~q[c] /\ ~isA[c]
       THEN /\ q' = [q EXCEPT ![c] = TRUE] /\ nx' = [nx EXCEPT ![c] = hd[a]]
            /\ hd' = [hd EXCEPT ![a] = c]
       ELSE UNCHANGED <<q, nx, hd>>
    /\ ost' = [ost EXCEPT ![t] = "xlist"]
    /\ UNCHANGED <<st, bits, mcnt, tok, anc, isA, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, omode, otgt, ocur, onxt, ohold, orest, opop>> /\ UNCHANGED FVars

(* LazyDrain, exiting: push the rest back onto the own head (still holding
   Q; the link is the owner's while Q is held). *)
OExitRest(t) ==
    LET a == oanc[t]  c == orest[t] IN
    /\ LazyDrain /\ ost[t] \in {"run", "xlist"} /\ a # NIL /\ c # NIL
    /\ orest' = [orest EXCEPT ![t] = nx[c]]
    /\ nx' = [nx EXCEPT ![c] = hd[a]] /\ hd' = [hd EXCEPT ![a] = c]
    /\ ost' = [ost EXCEPT ![t] = "xlist"]
    /\ UNCHANGED <<st, bits, mcnt, tok, q, anc, isA, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, avail, omode, otgt, ocur, onxt, ohold, opop>> /\ UNCHANGED FVars

(* ExitDrainsHead (the code): after listing, take the own head and settle it
   as a holder would, except that what stays is listed again on the own head
   (OAct, mode "exit"): the anchor unlisted, empty chunks released. *)
OExitHead(t) ==
    LET a == oanc[t] IN
    /\ ExitDrainsHead /\ ost[t] \in {"run", "xlist"} /\ a # NIL
    /\ orest[t] = NIL /\ opop[t] = NIL /\ avail[t] = {}
    /\ TakeHead(t, a, "exit")
    /\ UNCHANGED <<st, bits, mcnt, tok, q, nx, anc, isA, ref, gloc, roomCh, fullCh>>
    /\ UNCHANGED <<oanc, avail, onxt, ohold, orest, opop>> /\ UNCHANGED FVars

OExitPlace(t) ==
    LET a == oanc[t] IN
    /\ IF ExitDrainsHead
       THEN ost[t] = "xdone"
       ELSE \/ ost[t] = "xlist" /\ avail[t] = {}
            \/ ost[t] = "run" /\ avail[t] = {} /\ a # NIL
    /\ orest[t] = NIL /\ opop[t] = NIL
    /\ Place(a)
    /\ oanc' = [oanc EXCEPT ![t] = NIL] /\ ost' = [ost EXCEPT ![t] = "dead"]
    /\ UNCHANGED <<bits, mcnt, tok, q, nx, isA, hd>>
    /\ UNCHANGED <<avail, omode, otgt, ocur, onxt, ohold>> /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

OExitNoGroup(t) ==
    /\ ost[t] = "run" /\ oanc[t] = NIL /\ ost' = [ost EXCEPT ![t] = "dead"]
    /\ UNCHANGED CVars /\ UNCHANGED <<oanc, avail, omode, otgt, ocur, onxt, ohold>>
    /\ UNCHANGED FVars
    /\ UNCHANGED <<orest, opop>>

(******************************** freers ********************************)

(* An anchor's slot: just return it (anchors are never listed). *)
FPick(f, c) ==
    /\ fpc[f] = "idle" /\ st[c] = "live" /\ tok[c] > 0
    /\ tok' = [tok EXCEPT ![c] = @ - 1] /\ fc' = [fc EXCEPT ![f] = c]
    /\ IF isA[c] /\ ~AnchorListed
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
         \/ ODrainLazy(t) \/ OPopOwn(t) \/ OPopAct(t) \/ OExitRest(t) \/ OExitHead(t)
         \/ \E c \in Chunks : OAttach(t, c) \/ OAlloc(t, c) \/ ORelease(t, c)
                              \/ OAdoptStart(t, c) \/ OExitList(t, c) \/ OExitMore(t, c)
                              \/ OTakeOver(t, c) \/ OExitAny(t, c)
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

RestSet(t) == Rest(orest[t], Cardinality(Chunks) + 1)

Inv_NoUseAfterRelease ==
    /\ \A f \in Freers : TouchesC(f) => st[fc[f]] = "live"
    /\ \A f \in Freers : TouchesA(f) => st[fa[f]] = "live"
    /\ \A t \in Owners : \A c \in ({oanc[t], ocur[t], onxt[t]} \ {NIL}) \cup Held(t)
                                   \cup avail[t] : st[c] = "live"
    /\ \A c \in Chunks : anc[c] # NIL => st[anc[c]] = "live"
    /\ \A c \in OnHeads \cup roomCh \cup fullCh : st[c] = "live"
    /\ \A t \in Owners : \A c \in RestSet(t) \cup ({opop[t]} \ {NIL}) : st[c] = "live"

(* Every live chunk is in a live group; heads list their own group's
   non-anchor members, once each. *)
Inv_GroupOK ==
    /\ \A c \in Chunks : st[c] = "live" => anc[c] # NIL /\ isA[anc[c]]
    /\ \A a \in LiveAnchors :
         /\ anc[a] = a
         /\ Len(HeadSeq(a)) <= Cardinality(Chunks)
         /\ \A i, j \in 1..Len(HeadSeq(a)) : i # j => HeadSeq(a)[i] # HeadSeq(a)[j]
         /\ \A c \in HeadSet(a) : q[c] /\ anc[c] = a /\ (~isA[c] \/ (AnchorListed /\ c = a))

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

(* LazyDrain: the rest of a take holds only the owner's own group's members,
   listed (Q), on no head, and is a proper list. *)
Inv_RestOK ==
    \A t \in Owners : orest[t] # NIL =>
        LET R == Walk(orest[t], Cardinality(Chunks) + 1) IN
        /\ Len(R) <= Cardinality(Chunks)
        /\ \A i, j \in 1..Len(R) : i # j => R[i] # R[j]
        /\ \A i \in 1..Len(R) : q[R[i]] /\ anc[R[i]] = oanc[t]
                                /\ (~isA[R[i]] \/ (AnchorListed /\ R[i] = oanc[t]))
                                /\ R[i] \notin OnHeads

Inv_QAccounted ==
    \A c \in Chunks : st[c] = "live" /\ q[c] =>
        \/ c \in OnHeads
        \/ \E f \in Freers : fpc[f] \in {"fb", "fd", "fload", "pread", "pcas"}
                              /\ fc[f] = c /\ ~fwasq[f]
        \/ \E t \in Owners : c \in DrainRest(t) \cup RestSet(t)

(* Witnesses, NOT invariants of the design: each must be violated, showing
   the model reaches the situation (RevivalGroup_witness_*.cfg). *)
W_NoDissolve == ~ \E a \in Chunks : isA[a] /\ st[a] = "released"
W_NoOrphanPush == ~ \E f \in Freers : fpc[f] = "unref" /\ gloc[fa[f]] \in {"room", "full", "held"}
W_NoSweptMove == ~ \E t \in Owners : omode[t] = "sweep" /\ ost[t] = "dqclr"

=============================================================================
