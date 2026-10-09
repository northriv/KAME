(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        Dual-licensed Apache 2.0 OR GPL-2.0-or-later — see OrphanChain_atomicshared.tla.
 ***************************************************************************)
----------------------------- MODULE RevivalStack -----------------------------
(*
 * Design model: replacing the DLL walks with lists of chunks that have room.
 * The code (kamepoolalloc stage 2a, "§revive") uses this model's OneBit Q
 * protocol, with the head in an anchor chunk instead of the slot table
 * (RevivalAnchor.tla) and the existing orphan chain instead of the
 * available-orphan chain modelled here.
 *
 * Today a thread finds room by walking its DLL of chunks: the cursor walk in
 * allocate_chunk_path for bitmap room revived by cross-thread frees (gated by
 * the force-walk hint), and the §24 scan_dll_freelist walk for owner-freelist
 * entries.  Both are O(chunks) per slow path; the second made allocate-only
 * workloads quadratic, and neither is bounded for realtime threads.  The
 * replacement modelled here:
 *
 *   - Revival stack, one per owner, in a static slot table (the owner's TLS
 *     may be gone when a freer reaches it).  A cross-thread free that returns
 *     a slot to an owned chunk pushes the chunk; the owner takes the whole
 *     stack with one exchange in its slow path.  Producers only push and the
 *     single consumer only takes all, so the CAS needs no ABA protection for
 *     the pointer.  The head word carries the owner's tag and an open/closed
 *     state ([chunk | tag | CLOSED]); the owner closes it at exit, and a push
 *     expecting another owner's tag, or a closed head, fails.
 *   - Orphans split by room: an orphan with room is on the available-orphan
 *     chain (adoption always gains room); a full orphan is on no list, and
 *     the cross-thread free that first gives it room pushes it.
 *   - Per-chunk bits in m_flags_packed, next to MASK_CNT and BIT_OWNED:
 *       Q  "listed": on the revival stack or the orphan chain, or held by
 *          the one thread about to put it on one.  While Q is set the chunk
 *          cannot be released (the packed word is non-zero), so the holder
 *          may keep touching it.
 *       P  "room appeared while Q was held": a freer that finds Q taken sets
 *          P instead of walking away, and a holder that decided "full, drop
 *          Q" must fail its drop CAS when P is set and look again.
 *       pin  freers between "about to clear my bit" and "done with Q".  A
 *          freer's own bit keeps the chunk alive only until it clears that
 *          bit; after that another freer can empty the word and someone can
 *          release the chunk, yet this freer still has to take Q or set P.
 *          So it pins first (fetch_add on the packed word) and unpins in the
 *          same CAS that takes Q / sets P.  Release needs the whole word zero,
 *          pins included, so the last one out releases.  (Found by this model:
 *          without the pin, NoPin = TRUE, Inv_NoUseAfterRelease fails in 16
 *          steps -- a scrub releases the chunk between a freer's bit clear and
 *          its Q step.)
 *
 * A chunk has K slots in one bitmap word.  `bits` is that word's population.
 * MASK_CNT (mcnt) counts non-zero words: the allocation that takes the word
 * from zero increments it, and the free that takes it to zero decrements it in
 * a separate, later step, as allocate_pooled and batch_return_to_bitmap do --
 * so a word can be refilled before the free that emptied it has decremented.
 * The packed word is zero iff mcnt = 0, not owned, not Q and no pins; whoever
 * makes it zero releases the chunk.  Owner generations hold the single slot one after
 * another; Tag(g) is the tag generation g writes.
 *
 * Knobs (each FALSE / 0 in the design; the _bug cfgs turn one on):
 *   TagMod           0: every generation has its own tag.  n > 0: tags wrap
 *                    modulo n, so a late push can match a later owner.
 *   ClearQBeforeNext the owner clears Q before reading the chunk's next link.
 *   NoPin            no pin: a freer is unprotected between its bit clear and
 *                    its Q step.
 *   NoPending        no P bit: a freer that finds Q taken just leaves.
 *   Serial           a serial in the packed word, bumped by every write: the
 *                    holder's "full, drop Q" CAS fails if anyone wrote the
 *                    word since it looked, which is what P does by hand.
 *                    With Serial, NoPending = TRUE is expected to be clean.
 *                    Modelled as in OrphanChain_aba: no serial values, a
 *                    flag set by any write after the holder's read.
 *   NoQ              (with Serial) registration without the Q bit: a freer's
 *                    final CAS succeeds iff nobody else wrote the word since
 *                    its pin, and then it pushes.  Checks the claim that the
 *                    serial alone stops a chunk being pushed twice.  It does
 *                    not: the same freer frees a second slot after its first
 *                    push, nobody else has written the word since its new
 *                    pin, and it pushes the chunk again (nx[c1] = c1, 20
 *                    steps).  With owner release left on, NoQ fails earlier:
 *                    the owner releases a chunk a freer is about to push.
 *
 *   OneBit           Q alone, no pin, no P: a freer takes Q (fetch_or) BEFORE
 *                    clearing its bit, while its own slot still keeps the
 *                    word non-zero; the freer that set Q pushes, one that
 *                    found Q set clears its bit and leaves the chunk alone.
 *
 * Results (2 chunks, 2 freers, K = 2, symmetry; run_orphan_chain.sh style):
 *   design (P)                   clean, 2,208,624 distinct states, depth 98
 *   serial (Serial, no P)        clean -- the serial makes P unnecessary
 *   aba                          Inv_NoPushABA reached (24 steps): the push
 *                                needs no serial, the head's tag suffices
 *   tagwrap                      Inv_StackOK (21)
 *   clearfirst                   Inv_QAccounted (31)
 *   nopin, serial_nopin          Inv_NoUseAfterRelease (10)
 *   nopending                    Inv_NoLostRoom (15)
 *   serial_noq                   Inv_NoDupListing (20)
 *   onebit_safety (OneBit)       clean, 1,495,164 distinct states, depth 86
 *   onebit (+ Inv_NoLostRoom)    Inv_NoLostRoom (15): the one-bit protocol
 *                                is safe but can leave room unlisted when a
 *                                freer that found Q set clears its bit after
 *                                the holder looked -- the code accepts this
 *                                (a later free on the chunk lists it again;
 *                                with none, it waits for its owner's exit)
 *)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS Chunks, Freers, NIL, K, Gens,
          TagMod, ClearQBeforeNext, NoPin, NoPending, Serial, NoQ, OneBit

ASSUME NIL \notin Chunks
ASSUME K \in Nat \ {0} /\ Gens \in Nat \ {0} /\ TagMod \in Nat
ASSUME ClearQBeforeNext \in BOOLEAN /\ NoPin \in BOOLEAN
       /\ NoPending \in BOOLEAN /\ Serial \in BOOLEAN /\ NoQ \in BOOLEAN
       /\ OneBit \in BOOLEAN

Tag(g) == IF TagMod = 0 THEN g ELSE ((g - 1) % TagMod) + 1
Tags   == 0 .. (IF TagMod = 0 THEN Gens ELSE TagMod)
Ptr    == Chunks \cup {NIL}

VARIABLES
    st,       \* [Chunks -> {"fresh", "live", "released"}]
    bits,     \* [Chunks -> 0..K]   slots allocated in the bitmap word
    mcnt,     \* [Chunks -> Nat]    MASK_CNT
    owned,    \* [Chunks -> BOOLEAN] BIT_OWNED
    q,        \* [Chunks -> BOOLEAN] Q
    p,        \* [Chunks -> BOOLEAN] P
    pin,      \* [Chunks -> Nat]     pins (in-flight freers)
    otag,     \* [Chunks -> Tags]    owner tag recorded in the chunk (0: none)
    tok,      \* [Chunks -> 0..K]    allocated slots the application still holds
    nx,       \* [Chunks -> Ptr]     m_revive_next
    head,     \* [ptr: Ptr, tag: Tags, closed: BOOLEAN]
    avail,    \* the current owner's own list of chunks with room
    aorph,    \* the available-orphan chain (as a set)
    gen,      \* current owner generation (0: none yet)
    opc,      \* owner program counter
    omode,    \* "drain" | "close"
    ocur,     \* owner: chunk being drained / orphaned
    onxt,     \* owner: next link read while draining
    oorph,    \* owner exit: chunks still to orphan
    fpc,      \* [Freers -> pc]
    fc,       \* [Freers -> Ptr]     the chunk a freer is returning a slot to
    fzero,    \* [Freers -> BOOLEAN] its bit clear emptied the word
    fowned,   \* [Freers -> BOOLEAN] BIT_OWNED as its Q step saw it
    fwasq,    \* OneBit: [Freers -> BOOLEAN] Q was already set when it tried
    ftag,     \* [Freers -> Tags]    owner tag read after taking Q
    fh,       \* [Freers -> head value loaded for the push CAS]
    fsd,      \* NoQ: [Freers -> BOOLEAN] word written by others since the pin
    odirty,   \* Serial: the packed word of ocur was written since the holder
              \* read it in OOrphSpace (an unbounded, non-wrapping serial)
    fdirty    \* ghost [Freers -> BOOLEAN]: head changed since this freer loaded
              \* it for its push CAS (an ABA witness, see Inv_NoPushABA)

vars == <<st, bits, mcnt, owned, q, p, pin, otag, tok, nx, head, avail, aorph,
          gen, opc, omode, ocur, onxt, oorph, fpc, fc, fzero, fowned, fwasq, ftag, fh,
          fsd, odirty, fdirty>>

HeadT == [ptr: Ptr, tag: Tags, closed: BOOLEAN]
ClosedHead(t) == [ptr |-> NIL, tag |-> t, closed |-> TRUE]

TypeOK ==
    /\ st \in [Chunks -> {"fresh", "live", "released"}]
    /\ bits \in [Chunks -> 0..K] /\ mcnt \in [Chunks -> 0..(1 + Cardinality(Freers))]
    /\ owned \in [Chunks -> BOOLEAN] /\ q \in [Chunks -> BOOLEAN]
    /\ p \in [Chunks -> BOOLEAN] /\ pin \in [Chunks -> 0..Cardinality(Freers)]
    /\ otag \in [Chunks -> Tags]
    /\ tok \in [Chunks -> 0..K] /\ nx \in [Chunks -> Ptr]
    /\ head \in HeadT /\ avail \subseteq Chunks /\ aorph \subseteq Chunks
    /\ gen \in 0..Gens /\ ocur \in Ptr /\ onxt \in Ptr /\ oorph \subseteq Chunks
    /\ fc \in [Freers -> Ptr] /\ fh \in [Freers -> HeadT]

(* The chunks on the revival stack, in order (bounded walk). *)
RECURSIVE Walk(_, _)
Walk(c, n) == IF c = NIL \/ n = 0 THEN <<>> ELSE <<c>> \o Walk(nx[c], n - 1)
StackSeq == Walk(head.ptr, Cardinality(Chunks) + 1)
StackSet == {StackSeq[i] : i \in 1..Len(StackSeq)}

Init ==
    /\ st = [c \in Chunks |-> "fresh"]
    /\ bits = [c \in Chunks |-> 0] /\ mcnt = [c \in Chunks |-> 0]
    /\ owned = [c \in Chunks |-> FALSE] /\ q = [c \in Chunks |-> FALSE]
    /\ p = [c \in Chunks |-> FALSE] /\ pin = [c \in Chunks |-> 0]
    /\ otag = [c \in Chunks |-> 0]
    /\ tok = [c \in Chunks |-> 0] /\ nx = [c \in Chunks |-> NIL]
    /\ head = ClosedHead(0)
    /\ avail = {} /\ aorph = {} /\ gen = 0
    /\ opc = "open" /\ omode = "drain" /\ ocur = NIL /\ onxt = NIL /\ oorph = {}
    /\ fpc = [f \in Freers |-> "idle"] /\ fc = [f \in Freers |-> NIL]
    /\ fzero = [f \in Freers |-> FALSE]
    /\ fowned = [f \in Freers |-> FALSE]
    /\ fwasq = [f \in Freers |-> FALSE] /\ ftag = [f \in Freers |-> 0]
    /\ fh = [f \in Freers |-> ClosedHead(0)]
    /\ fsd = [f \in Freers |-> FALSE]
    /\ odirty = FALSE
    /\ fdirty = [f \in Freers |-> FALSE]

Release(c) == st' = [st EXCEPT ![c] = "released"]

FreerVars == <<fpc, fc, fzero, fowned, fwasq, ftag, fh>>

(* A freer going idle: its locals are dead; reset them so they do not split
   states. *)
GoIdle(f) ==
    /\ fpc' = [fpc EXCEPT ![f] = "idle"] /\ fc' = [fc EXCEPT ![f] = NIL]
    /\ fzero' = [fzero EXCEPT ![f] = FALSE] /\ fowned' = [fowned EXCEPT ![f] = FALSE]
    /\ ftag' = [ftag EXCEPT ![f] = 0] /\ fh' = [fh EXCEPT ![f] = ClosedHead(0)]
    /\ fwasq' = [fwasq EXCEPT ![f] = FALSE]
OwnerVars == <<gen, opc, omode, ocur, onxt, oorph>>
ChunkVars == <<st, bits, mcnt, owned, q, p, pin, otag, tok, nx>>

(*************************** owner (one at a time) ***************************)

(* A new generation takes the slot: CAS CLOSED -> open with its tag. *)
OOpen ==
    /\ opc = "open" /\ gen < Gens /\ head.closed
    /\ gen' = gen + 1
    /\ head' = [ptr |-> NIL, tag |-> Tag(gen + 1), closed |-> FALSE]
    /\ opc' = "run"
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<avail, aorph, omode, ocur, onxt, oorph>>
    /\ UNCHANGED FreerVars

(* create_allocator: a fresh chunk joins the owner's DLL and its own list. *)
OAttach(c) ==
    /\ opc = "run" /\ st[c] = "fresh"
    /\ st' = [st EXCEPT ![c] = "live"]
    /\ owned' = [owned EXCEPT ![c] = TRUE]
    /\ otag' = [otag EXCEPT ![c] = Tag(gen)]
    /\ avail' = avail \cup {c}
    /\ UNCHANGED <<bits, mcnt, q, p, pin, tok, nx, head, aorph>>
    /\ UNCHANGED OwnerVars /\ UNCHANGED FreerVars

(* allocate_pooled from a chunk on the owner's list; leaves it when full. *)
OAlloc(c) ==
    /\ opc = "run" /\ c \in avail /\ bits[c] < K
    /\ bits' = [bits EXCEPT ![c] = @ + 1]
    /\ mcnt' = IF bits[c] = 0 THEN [mcnt EXCEPT ![c] = @ + 1] ELSE mcnt
    /\ tok' = [tok EXCEPT ![c] = @ + 1]
    /\ avail' = IF bits[c] + 1 = K THEN avail \ {c} ELSE avail
    /\ UNCHANGED <<st, owned, q, p, pin, otag, nx, head, aorph>>
    /\ UNCHANGED OwnerVars /\ UNCHANGED FreerVars

(* Adopt from the available-orphan chain: pop, then CAS owned := 1, Q := 0.
   Room is read after the CAS, so a later freer finds Q clear and pushes. *)
OAdopt(c) ==
    /\ opc = "run" /\ c \in aorph
    /\ aorph' = aorph \ {c}
    /\ owned' = [owned EXCEPT ![c] = TRUE]
    /\ q' = [q EXCEPT ![c] = FALSE] /\ p' = [p EXCEPT ![c] = FALSE]
    /\ otag' = [otag EXCEPT ![c] = Tag(gen)]
    /\ avail' = IF bits[c] < K THEN avail \cup {c} ELSE avail
    /\ UNCHANGED <<st, bits, mcnt, pin, tok, nx, head>>
    /\ UNCHANGED OwnerVars /\ UNCHANGED FreerVars

(* owner_release of an empty chunk: a CAS that needs the rest of the word --
   Q and pins -- clear. *)
ORelease(c) ==
    /\ ~NoQ   \* NoQ isolates the double push: no release, no exit
    /\ opc = "run" /\ st[c] = "live" /\ owned[c] /\ otag[c] = Tag(gen)
    /\ mcnt[c] = 0 /\ ~q[c] /\ pin[c] = 0
    /\ owned' = [owned EXCEPT ![c] = FALSE] /\ Release(c)
    /\ avail' = avail \ {c}
    /\ UNCHANGED <<bits, mcnt, q, p, pin, otag, tok, nx, head, aorph>>
    /\ UNCHANGED OwnerVars /\ UNCHANGED FreerVars

(* Slow path: take the whole stack (exchange, keeping our tag). *)
ODrainStart ==
    /\ opc = "run"
    /\ ocur' = head.ptr
    /\ head' = [ptr |-> NIL, tag |-> Tag(gen), closed |-> FALSE]
    /\ omode' = "drain" /\ opc' = IF ClearQBeforeNext THEN "dclr0" ELSE "dnext"
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<avail, aorph, gen, onxt, oorph>>
    /\ UNCHANGED FreerVars

(* Exit: take the stack and close the slot in one exchange. *)
OCloseStart ==
    /\ ~NoQ /\ opc = "run"
    /\ ocur' = head.ptr
    /\ head' = ClosedHead(Tag(gen))
    /\ omode' = "close" /\ opc' = IF ClearQBeforeNext THEN "dclr0" ELSE "dnext"
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<avail, aorph, gen, onxt, oorph>>
    /\ UNCHANGED FreerVars

DrainDone ==
    IF omode = "drain"
    THEN /\ opc' = "run" /\ UNCHANGED oorph
    ELSE /\ opc' = "orph"
         /\ oorph' = {c \in Chunks : st[c] = "live" /\ owned[c]
                                     /\ otag[c] = Tag(gen)}

(* Drain, design order: read next, clear Q (and P), then read room. *)
ODrainNext ==
    /\ opc = "dnext"
    /\ IF ocur = NIL
       THEN DrainDone /\ UNCHANGED onxt
       ELSE /\ onxt' = nx[ocur] /\ opc' = "dclr" /\ UNCHANGED oorph
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<head, avail, aorph, gen, omode, ocur>>
    /\ UNCHANGED FreerVars

ODrainClear ==
    /\ opc = "dclr"
    /\ q' = [q EXCEPT ![ocur] = FALSE] /\ p' = [p EXCEPT ![ocur] = FALSE]
    /\ nx' = [nx EXCEPT ![ocur] = NIL]   \* dead once read; cleared to keep states few
    /\ opc' = "dspace"
    /\ UNCHANGED <<st, bits, mcnt, owned, pin, otag, tok>>
    /\ UNCHANGED <<head, avail, aorph, gen, omode, ocur, onxt, oorph>>
    /\ UNCHANGED FreerVars

(* Bug order (ClearQBeforeNext): clear Q first, then read next. *)
ODrainClear0 ==
    /\ opc = "dclr0"
    /\ IF ocur = NIL
       THEN DrainDone /\ UNCHANGED <<q, p>>
       ELSE /\ q' = [q EXCEPT ![ocur] = FALSE] /\ p' = [p EXCEPT ![ocur] = FALSE]
            /\ opc' = "dnext0" /\ UNCHANGED oorph
    /\ UNCHANGED <<st, bits, mcnt, owned, pin, otag, tok, nx>>
    /\ UNCHANGED <<head, avail, aorph, gen, omode, ocur, onxt>>
    /\ UNCHANGED FreerVars

ODrainNext0 ==
    /\ opc = "dnext0"
    /\ onxt' = nx[ocur] /\ opc' = "dspace"
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<head, avail, aorph, gen, omode, ocur, oorph>>
    /\ UNCHANGED FreerVars

ODrainSpace ==
    /\ opc = "dspace"
    /\ avail' = IF bits[ocur] < K THEN avail \cup {ocur} ELSE avail
    /\ ocur' = onxt /\ onxt' = NIL
    /\ opc' = IF ClearQBeforeNext THEN "dclr0" ELSE "dnext"
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<head, aorph, gen, omode, oorph>>
    /\ UNCHANGED FreerVars

(* Exit, per owned chunk: CAS owned := 0 (and m_owner_id := 0).  Word now 0
   -> release.  Empty but pinned -> the last pin releases it.  Q clear -> take
   Q and look at the room.  Q held by a freer -> leave it to that freer, which
   will find the chunk orphaned. *)
OOrphan(c) ==
    /\ opc = "orph" /\ c \in oorph
    /\ oorph' = oorph \ {c}
    /\ owned' = [owned EXCEPT ![c] = FALSE]
    /\ otag' = [otag EXCEPT ![c] = 0]
    /\ avail' = avail \ {c}
    /\ IF mcnt[c] = 0 /\ ~q[c]
       THEN /\ IF pin[c] = 0 THEN Release(c) ELSE UNCHANGED st
            /\ opc' = "orph" /\ UNCHANGED <<q, p, ocur>>
       ELSE IF ~q[c]
            THEN /\ q' = [q EXCEPT ![c] = TRUE] /\ p' = [p EXCEPT ![c] = FALSE]
                 /\ ocur' = c /\ opc' = "ospace" /\ UNCHANGED st
            ELSE /\ opc' = "orph" /\ UNCHANGED <<st, q, p, ocur>>
    /\ UNCHANGED <<bits, mcnt, pin, tok, nx, head, aorph, gen, omode, onxt>>
    /\ UNCHANGED FreerVars

(* Holding Q: room -> push on the orphan chain; full -> try to drop Q. *)
OOrphSpace ==
    /\ opc = "ospace"
    /\ IF bits[ocur] < K
       THEN /\ aorph' = aorph \cup {ocur} /\ opc' = "orph" /\ ocur' = NIL
       ELSE /\ opc' = "odrop" /\ UNCHANGED <<aorph, ocur>>
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<head, avail, gen, omode, onxt, oorph>>
    /\ UNCHANGED FreerVars

(* Drop CAS: fails on P (room appeared meanwhile) -- or, with Serial, on any
   write to the word since OOrphSpace read it -- and looks again. *)
OOrphDrop ==
    /\ opc = "odrop"
    /\ IF IF Serial THEN odirty ELSE p[ocur]
       THEN /\ p' = [p EXCEPT ![ocur] = FALSE] /\ opc' = "ospace"
            /\ UNCHANGED <<st, q, ocur>>
       ELSE /\ q' = [q EXCEPT ![ocur] = FALSE]
            /\ IF mcnt[ocur] = 0 /\ pin[ocur] = 0 THEN Release(ocur)
               ELSE UNCHANGED st
            /\ opc' = "orph" /\ ocur' = NIL /\ UNCHANGED p
    /\ UNCHANGED <<bits, mcnt, owned, pin, otag, tok, nx>>
    /\ UNCHANGED <<head, avail, aorph, gen, omode, onxt, oorph>>
    /\ UNCHANGED FreerVars

OOrphDone ==
    /\ opc = "orph" /\ oorph = {}
    /\ avail' = {}
    /\ opc' = IF gen < Gens THEN "open" ELSE "done"
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<head, aorph, gen, omode, ocur, onxt, oorph>>
    /\ UNCHANGED FreerVars

(***************************** cross-thread freers *****************************)

(* The application hands a freer one of its live slots; the slot's bit keeps
   the word non-zero, so the chunk is alive until the freer clears it. *)
FPick(f, c) ==
    /\ fpc[f] = "idle" /\ st[c] = "live" /\ tok[c] > 0
    /\ tok' = [tok EXCEPT ![c] = @ - 1]
    /\ fc' = [fc EXCEPT ![f] = c]
    /\ fpc' = [fpc EXCEPT ![f] = IF OneBit THEN "fq1" ELSE IF NoPin THEN "fb" ELSE "pin"]
    /\ UNCHANGED <<st, bits, mcnt, owned, q, p, pin, otag, nx, head, avail, aorph,
                   fzero, fowned, fwasq, ftag, fh>> /\ UNCHANGED OwnerVars

(* Pin before clearing the bit: fetch_add on the packed word. *)
FPin(f) ==
    /\ fpc[f] = "pin"
    /\ pin' = [pin EXCEPT ![fc[f]] = @ + 1]
    /\ fpc' = [fpc EXCEPT ![f] = "fb"]
    /\ UNCHANGED <<st, bits, mcnt, owned, q, p, otag, tok, nx, head, avail, aorph,
                   fc, fzero, fowned, fwasq, ftag, fh>> /\ UNCHANGED OwnerVars

(* Clear our bit (the bitmap-word CAS).  Room appears here. *)
FBitClear(f) ==
    LET c == fc[f] IN
    /\ fpc[f] = "fb"
    /\ bits' = [bits EXCEPT ![c] = @ - 1]
    /\ fzero' = [fzero EXCEPT ![f] = (bits[c] = 1)]
    /\ fpc' = [fpc EXCEPT ![f] = IF bits[c] = 1 THEN "fd"
                                ELSE IF OneBit /\ fwasq[f] THEN "idle"
                                ELSE IF OneBit THEN "f1go" ELSE "fq"]
    /\ UNCHANGED <<st, mcnt, owned, q, p, pin, otag, tok, nx, head, avail, aorph,
                   fc, fowned, fwasq, ftag, fh>> /\ UNCHANGED OwnerVars

(* Our bit emptied the word: MASK_CNT--.  Pinned, the word stays non-zero;
   unpinned (NoPin), the word may reach zero and we release. *)
FDec(f) ==
    LET c == fc[f] IN
    /\ fpc[f] = "fd"
    /\ mcnt' = [mcnt EXCEPT ![c] = @ - 1]
    /\ IF mcnt[c] = 1 /\ ~owned[c] /\ ~q[c] /\ pin[c] = 0
       THEN Release(c) /\ GoIdle(f)
       ELSE /\ UNCHANGED st
            /\ fpc' = [fpc EXCEPT ![f] = IF OneBit /\ fwasq[f] THEN "idle"
                                         ELSE IF OneBit THEN "f1go" ELSE "fq"]
            /\ UNCHANGED <<fc, fzero, fowned, fwasq, ftag, fh>>
    /\ UNCHANGED <<bits, owned, q, p, pin, otag, tok, nx, head, avail, aorph>>
    /\ UNCHANGED OwnerVars

(* One CAS on the packed word: unpin, and take Q -- or, Q being taken, set P.
   An empty, unowned, unlisted chunk is not taken: the unpin that leaves the
   word zero releases it. *)
FTakeQ(f) ==
    LET c    == fc[f]
        pin2 == IF NoPin THEN pin[c] ELSE pin[c] - 1
    IN
    /\ fpc[f] = "fq"
    /\ ~NoQ
    /\ pin' = [pin EXCEPT ![c] = pin2]
    /\ IF mcnt[c] = 0 /\ ~owned[c] /\ ~q[c]
       THEN /\ IF pin2 = 0 THEN Release(c) ELSE UNCHANGED st
            /\ GoIdle(f) /\ UNCHANGED <<q, p>>
       ELSE IF ~q[c]
            THEN /\ q' = [q EXCEPT ![c] = TRUE] /\ UNCHANGED <<st, p>>
                 /\ fpc' = [fpc EXCEPT ![f] = IF owned[c] THEN "rtag" ELSE "opush"]
                 /\ fowned' = [fowned EXCEPT ![f] = owned[c]]
                 /\ UNCHANGED <<fc, fzero, fwasq, ftag, fh>>
            ELSE /\ p' = IF NoPending THEN p ELSE [p EXCEPT ![c] = TRUE]
                 /\ GoIdle(f) /\ UNCHANGED <<st, q>>
    /\ UNCHANGED <<bits, mcnt, owned, otag, tok, nx, head, avail, aorph>>
    /\ UNCHANGED OwnerVars

(* NoQ: the final CAS expects the word as of the pin (plus our own writes).
   Written by someone else since -> fail, re-read, retry.  Success -> unpin
   and push; nothing in the word says whether the chunk is listed already. *)
FTakeQNoQ(f) ==
    LET c == fc[f] IN
    /\ NoQ /\ fpc[f] = "fq"
    /\ IF fsd[f]
       THEN /\ fpc' = [fpc EXCEPT ![f] = "fq2"]   \* CAS failed: re-read
            /\ UNCHANGED <<pin, fc, fzero, fowned, fwasq, ftag, fh>>
       ELSE /\ pin' = [pin EXCEPT ![c] = @ - 1]
            /\ fpc' = [fpc EXCEPT ![f] = IF owned[c] THEN "rtag" ELSE "opush"]
            /\ fowned' = [fowned EXCEPT ![f] = owned[c]]
            /\ UNCHANGED <<fc, fzero, fwasq, ftag, fh>>
    /\ UNCHANGED <<st, bits, mcnt, owned, q, p, otag, tok, nx, head, avail, aorph>>
    /\ UNCHANGED OwnerVars

(* OneBit: fetch_or Q before the bit clear (our slot keeps the chunk alive). *)
FTakeQ1(f) ==
    LET c == fc[f] IN
    /\ fpc[f] = "fq1"
    /\ fwasq' = [fwasq EXCEPT ![f] = q[c]]
    /\ q' = [q EXCEPT ![c] = TRUE]
    /\ fpc' = [fpc EXCEPT ![f] = "fb"]
    /\ UNCHANGED <<st, bits, mcnt, owned, p, pin, otag, tok, nx, head, avail, aorph,
                   fc, fzero, fowned, ftag, fh>> /\ UNCHANGED OwnerVars

(* OneBit, after the bit clear: Q was ours -> push it; else leave the chunk. *)
FOneBitGo(f) ==
    LET c == fc[f] IN
    /\ fpc[f] = "f1go"
    /\ IF fwasq[f]
       THEN GoIdle(f)
       ELSE /\ fpc' = [fpc EXCEPT ![f] = IF owned[c] THEN "rtag" ELSE "opush"]
            /\ fowned' = [fowned EXCEPT ![f] = owned[c]]
            /\ UNCHANGED <<fc, fzero, fwasq, ftag, fh>>
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<head, avail, aorph>>
    /\ UNCHANGED OwnerVars

FReread(f) ==
    /\ fpc[f] = "fq2" /\ fpc' = [fpc EXCEPT ![f] = "fq"]
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<head, avail, aorph, fc, fzero, fowned, fwasq, ftag, fh>>
    /\ UNCHANGED OwnerVars

(* Holding Q on an owned chunk: read the owner tag recorded in it. *)
FReadTag(f) ==
    /\ fpc[f] = "rtag"
    /\ ftag' = [ftag EXCEPT ![f] = otag[fc[f]]]
    /\ fpc' = [fpc EXCEPT ![f] = "pread"]
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<head, avail, aorph, fc, fzero, fowned, fwasq, fh>>
    /\ UNCHANGED OwnerVars

(* Load the head; closed or another owner's -> reroute; else link to it. *)
FPushRead(f) ==
    LET c == fc[f] IN
    /\ fpc[f] = "pread"
    /\ fh' = [fh EXCEPT ![f] = head]
    /\ IF head.closed \/ head.tag # ftag[f]
       THEN /\ fpc' = [fpc EXCEPT ![f] = "reroute"] /\ UNCHANGED nx
       ELSE /\ nx' = [nx EXCEPT ![c] = head.ptr]
            /\ fpc' = [fpc EXCEPT ![f] = "pcas"]
    /\ UNCHANGED <<st, bits, mcnt, owned, q, p, pin, otag, tok, head, avail, aorph,
                   fc, fzero, fowned, fwasq, ftag>> /\ UNCHANGED OwnerVars

(* The push CAS compares the whole word: pointer, tag and state. *)
FPushCAS(f) ==
    /\ fpc[f] = "pcas"
    /\ IF head = fh[f]
       THEN /\ head' = [ptr |-> fc[f], tag |-> ftag[f], closed |-> FALSE]
            /\ GoIdle(f)
       ELSE /\ fpc' = [fpc EXCEPT ![f] = "pread"] /\ UNCHANGED head
            /\ UNCHANGED <<fc, fzero, fowned, fwasq, ftag, fh>>
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<avail, aorph>>
    /\ UNCHANGED OwnerVars

(* The owner has left (or is leaving).  Orphaned already -> it is ours to put
   on the orphan chain.  Still owned -> the exiting owner will orphan it and
   look at its room itself, so drop Q (owned: the word cannot reach 0). *)
FReroute(f) ==
    LET c == fc[f] IN
    /\ fpc[f] = "reroute"
    /\ IF ~owned[c]
       THEN /\ fpc' = [fpc EXCEPT ![f] = "opush"] /\ UNCHANGED <<q, p>>
            /\ UNCHANGED <<fc, fzero, fowned, fwasq, ftag, fh>>
       ELSE /\ q' = [q EXCEPT ![c] = FALSE] /\ p' = [p EXCEPT ![c] = FALSE]
            /\ GoIdle(f)
    /\ UNCHANGED <<st, bits, mcnt, owned, pin, otag, tok, nx, head, avail, aorph>>
    /\ UNCHANGED OwnerVars

(* Holding Q on an orphan with room: push it on the orphan chain. *)
FOrphPush(f) ==
    /\ fpc[f] = "opush"
    /\ aorph' = aorph \cup {fc[f]}
    /\ GoIdle(f)
    /\ UNCHANGED ChunkVars /\ UNCHANGED <<head, avail>>
    /\ UNCHANGED OwnerVars

(* orphan_chain_scrub: unlink a drained orphan; release it unless pinned (the
   last pin then releases it). *)
Scrub(c) ==
    /\ c \in aorph /\ mcnt[c] = 0 /\ bits[c] = 0
    /\ aorph' = aorph \ {c}
    /\ q' = [q EXCEPT ![c] = FALSE] /\ p' = [p EXCEPT ![c] = FALSE]
    /\ IF pin[c] = 0 THEN Release(c) ELSE UNCHANGED st
    /\ UNCHANGED <<bits, mcnt, owned, pin, otag, tok, nx, head, avail>>
    /\ UNCHANGED OwnerVars /\ UNCHANGED FreerVars

Acts ==
    \/ OOpen \/ ODrainStart \/ OCloseStart \/ ODrainNext \/ ODrainClear
    \/ ODrainClear0 \/ ODrainNext0 \/ ODrainSpace
    \/ OOrphSpace \/ OOrphDrop \/ OOrphDone
    \/ \E c \in Chunks : OAttach(c) \/ OAlloc(c) \/ OAdopt(c) \/ ORelease(c)
                         \/ OOrphan(c) \/ Scrub(c)
    \/ \E f \in Freers :
         \/ \E c \in Chunks : FPick(f, c)
         \/ FPin(f) \/ FBitClear(f) \/ FTakeQ(f) \/ FTakeQNoQ(f) \/ FReread(f)
         \/ FTakeQ1(f) \/ FOneBitGo(f)
         \/ FDec(f) \/ FReadTag(f)
         \/ FPushRead(f) \/ FPushCAS(f) \/ FReroute(f) \/ FOrphPush(f)

(* The ghost: a freer waiting to CAS remembers whether the head has changed
   since it loaded it. *)
PackedOf(c) == <<mcnt[c], owned[c], q[c], p[c], pin[c]>>

Next ==
    /\ Acts
    /\ odirty' = IF opc' = "odrop" /\ opc = "odrop"
                 THEN odirty \/ PackedOf(ocur)' # PackedOf(ocur)
                 ELSE IF opc' = "odrop" /\ opc = "ospace"
                      THEN FALSE
                      ELSE FALSE
    /\ fsd' = [g \in Freers |->
                IF fpc'[g] \in {"fb", "fd", "fq"}
                THEN IF fpc[g] \in {"pin", "fq2"} THEN FALSE
                     ELSE fsd[g] \/ (fpc'[g] = fpc[g] /\ fc[g] # NIL
                                     /\ PackedOf(fc[g])' # PackedOf(fc[g]))
                ELSE fpc'[g] = "fq2"]
    /\ fdirty' = [g \in Freers |->
                   IF fpc'[g] = "pcas" /\ fpc[g] = "pcas"
                   THEN fdirty[g] \/ head' # head
                   ELSE FALSE]

Spec == Init /\ [][Next]_vars

Symm == Permutations(Chunks) \cup Permutations(Freers)

(********************************** properties **********************************)

(* Chunks still to be visited by the drain in progress. *)
RECURSIVE Rest(_, _)
Rest(c, n) == IF c = NIL \/ n = 0 THEN {} ELSE {c} \cup Rest(nx[c], n - 1)
DrainRest == Rest(ocur, Cardinality(Chunks) + 1) \cup Rest(onxt, Cardinality(Chunks) + 1)

FreerHolds(f) == fpc[f] \notin {"idle"}
FreerHoldsQ(f) == \/ fpc[f] \in {"rtag", "pread", "pcas", "reroute", "opush"}
                  \/ (OneBit /\ fpc[f] \in {"fb", "fd", "f1go"} /\ ~fwasq[f])

(* Pins are exactly the freers between their pin and their Q step. *)
Inv_PinAccounted ==
    ~NoPin => \A c \in Chunks :
        pin[c] = Cardinality({f \in Freers : fpc[f] \in {"fb", "fd", "fq"} /\ fc[f] = c})
OwnerRefs == ({ocur, onxt} \ {NIL})

(* Nobody acts on, or lists, a released chunk. *)
Inv_NoUseAfterRelease ==
    /\ \A f \in Freers : FreerHolds(f) => st[fc[f]] = "live"
    /\ \A c \in OwnerRefs : st[c] = "live"
    /\ \A c \in StackSet \cup aorph \cup avail : st[c] = "live"

(* The revival stack holds only the slot owner's own, listed chunks, once
   each, and nothing while the slot is closed. *)
Inv_StackOK ==
    /\ Len(StackSeq) <= Cardinality(Chunks)
    /\ \A i, j \in 1..Len(StackSeq) : i # j => StackSeq[i] # StackSeq[j]
    /\ \A c \in StackSet : owned[c] /\ q[c] /\ otag[c] = head.tag
    /\ head.closed => head.ptr = NIL

(* No chunk is on the revival stack twice, or on both lists. *)
Inv_NoDupListing ==
    /\ \A i, j \in 1..Len(StackSeq) : i # j => StackSeq[i] # StackSeq[j]
    /\ StackSet \cap aorph = {}

(* The orphan chain holds orphans only, listed, and none also on the stack. *)
Inv_OrphansOK ==
    \A c \in aorph : ~owned[c] /\ q[c] /\ c \notin StackSet

Inv_AvailOK ==
    \A c \in avail : owned[c] /\ otag[c] = Tag(gen)

(* Q never leaks: a listed chunk is on a list or held by exactly the thread
   about to put it there. *)
Inv_QAccounted ==
    \A c \in Chunks : st[c] = "live" /\ q[c] =>
        \/ c \in StackSet \/ c \in aorph
        \/ \E f \in Freers : FreerHoldsQ(f) /\ fc[f] = c
        \/ (opc \in {"dnext", "dclr", "dspace", "dclr0", "dnext0"}
            /\ c \in DrainRest)
        \/ (opc \in {"ospace", "odrop"} /\ c = ocur)

(* NOT an invariant of the design -- a witness.  A freer about to CAS a head
   that changed and changed back since it linked to it: the pointer-only CAS
   succeeds.  RevivalStack_aba_mc.cfg checks it to show TLC reaches such
   states; the design cfg, which does not, shows every invariant above holds
   in them anyway, i.e. the push needs no serial. *)
Inv_NoPushABA ==
    ~ \E f \in Freers : fpc[f] = "pcas" /\ fdirty[f] /\ head = fh[f]

(* No room is lost: once nobody is mid-operation, every owned chunk with room
   is on the owner's list or its stack, and every orphan with room is on the
   orphan chain. *)
Quiescent == (\A f \in Freers : fpc[f] = "idle") /\ opc \in {"run", "open", "done"}
Inv_NoLostRoom ==
    Quiescent =>
      \A c \in Chunks : st[c] = "live" /\ bits[c] < K =>
          IF owned[c] THEN c \in avail \/ c \in StackSet ELSE c \in aorph

=============================================================================
