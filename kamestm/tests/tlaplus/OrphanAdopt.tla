--------------------------- MODULE OrphanAdopt ----------------------------
(***************************************************************************)
(* §13.129  The adopt half of the orphan chain, at BITMAP granularity.      *)
(*                                                                         *)
(* §13.128 exonerated the release half by measurement (both dispose         *)
(* backstops never fire, on failing runs too), leaving ADOPT as the only    *)
(* live candidate inside the chain -- and it is the half that matches the   *)
(* evidence, since §13.104's DOUBLE-LIVE is a slot handed OUT and §13.113   *)
(* found its previous occupant live and unfreed.                           *)
(*                                                                         *)
(* §13.127's model cannot answer this: it abstracted a chunk's occupancy to *)
(* one bit (MASK_CNT zero or not) and had no bitmap, so "can adopt hand out *)
(* a slot that is occupied" was outside it.  This model puts the bitmap in. *)
(*                                                                         *)
(* Three sets per chunk, which is the whole mechanism:                      *)
(*   bits -- the m_flags bitmap: a set bit means "not available"            *)
(*   occ  -- slots holding a LIVE object (the truth the bitmap must cover)  *)
(*   mask -- the word-cache's claimed-but-undistributed bits, parked in     *)
(*           m_freelist_head[1] with its base in [2]                        *)
(* The word-grab claims a whole word at once (bits := ALL) and hands out    *)
(* one slot, parking the rest in mask -- so `bits` legitimately exceeds     *)
(* `occ`, and the invariant is the other direction: occ \subseteq bits.     *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Slots,              \* slots in one bitmap word
    Threads,
    BUG_NO_DRAIN,       \* owner exit forgets to drain the word-cache mask
    BUG_DRAIN_KEEPS_CELLS, \* drain returns the bits but leaves the cells set
    BUG_WRONG_BIT       \* a free clears a bit OTHER than its own slot's --
                        \* §13.116's surviving mechanism (a mis-derived
                        \* chunk_base clears a bit in the wrong chunk, so a
                        \* live object there reads as available).  Modelled
                        \* within one chunk because the invariant it breaks is
                        \* the same one, and it is the invariant that matters.

VARIABLES
    bits,       \* SUBSET Slots
    occ,        \* SUBSET Slots
    mask,       \* SUBSET Slots
    owner,      \* Threads \cup {"none"}
    onChain,    \* BOOLEAN
    handed,     \* SUBSET Slots : slots handed out at least twice (the fault)
    pending,    \* [Threads -> SUBSET Slots]  batched cross-thread frees
    pc

vars == <<bits, occ, mask, owner, onChain, handed, pending, pc>>

NONE == "none"

TypeOK ==
    /\ bits \in SUBSET Slots
    /\ occ \in SUBSET Slots
    /\ mask \in SUBSET Slots
    /\ owner \in Threads \cup {NONE}
    /\ onChain \in BOOLEAN
    /\ handed \in SUBSET Slots
    /\ pending \in [Threads -> SUBSET Slots]

Init ==
    /\ bits = {} /\ occ = {} /\ mask = {}
    /\ owner = NONE /\ onChain = FALSE /\ handed = {}
    /\ pending = [t \in Threads |-> {}]
    /\ pc = [t \in Threads |-> "idle"]

Claim(t) ==             \* an unowned, off-chain chunk is taken fresh
    /\ pc[t] = "idle" /\ owner = NONE /\ ~onChain
    /\ owner' = t
    /\ UNCHANGED <<bits, occ, mask, onChain, handed, pending, pc>>

(*********** allocation: from the parked mask, or a fresh word-grab *********)
AllocFromMask(t) ==
    /\ pc[t] = "idle" /\ owner = t
    /\ mask /= {}
    /\ \E s \in mask :
        /\ handed' = IF s \in occ THEN handed \cup {s} ELSE handed  \* the fault
        /\ occ' = occ \cup {s}
        /\ mask' = mask \ {s}
    /\ UNCHANGED <<bits, owner, onChain, pending, pc>>

WordGrab(t) ==
    /\ pc[t] = "idle" /\ owner = t
    /\ mask = {} /\ bits /= Slots
    /\ LET free == Slots \ bits IN
       /\ \E s \in free :
           /\ handed' = IF s \in occ THEN handed \cup {s} ELSE handed
           /\ occ' = occ \cup {s}
           /\ mask' = free \ {s}
       /\ bits' = Slots                 \* the whole word goes to 1 in one CAS
    /\ UNCHANGED <<owner, onChain, pending, pc>>

(********************************** frees **********************************)
FreeDirect(t) ==
    /\ pc[t] = "idle"
    /\ \E s \in occ :
        /\ occ' = occ \ {s}
        /\ IF BUG_WRONG_BIT
             THEN \E w \in bits : bits' = bits \ {w}   \* clears SOME bit, not s
             ELSE bits' = bits \ {s}
    /\ UNCHANGED <<mask, owner, onChain, handed, pending, pc>>

FreeBatch(t) ==         \* cross-thread free: held, bit still set
    /\ pc[t] = "idle"
    /\ \E s \in occ :
        /\ occ' = occ \ {s}
        /\ pending' = [pending EXCEPT ![t] = @ \cup {s}]
    /\ UNCHANGED <<bits, mask, owner, onChain, handed, pc>>

FlushBatch(t) ==
    /\ pc[t] = "idle" /\ pending[t] /= {}
    /\ bits' = bits \ pending[t]
    /\ pending' = [pending EXCEPT ![t] = {}]
    /\ UNCHANGED <<occ, mask, owner, onChain, handed, pc>>

(**************************** owner exit / adopt ****************************)
\* release_dll_chunks_for_thread: drain the parked mask BACK to the bitmap and
\* null the cells, THEN decide empty-or-orphan.
OwnerExit(t) ==
    /\ pc[t] = "idle" /\ owner = t
    /\ IF BUG_NO_DRAIN
         THEN /\ bits' = bits /\ mask' = mask
         ELSE /\ bits' = bits \ mask
              /\ mask' = IF BUG_DRAIN_KEEPS_CELLS THEN mask ELSE {}
    /\ owner' = NONE
    /\ onChain' = (occ /= {} \/ (bits \ mask) /= {})   \* non-empty -> chain
    /\ UNCHANGED <<occ, handed, pending, pc>>

Adopt(t) ==
    /\ pc[t] = "idle" /\ onChain /\ owner = NONE
    /\ owner' = t
    /\ onChain' = FALSE
    /\ UNCHANGED <<bits, occ, mask, handed, pending, pc>>

Next ==
    \/ \E t \in Threads : Claim(t) \/ AllocFromMask(t) \/ WordGrab(t)
                       \/ FreeDirect(t) \/ FreeBatch(t) \/ FlushBatch(t)
                       \/ OwnerExit(t) \/ Adopt(t)

Spec == Init /\ [][Next]_vars

(******************************* properties ********************************)
\* Every live object's bit is set: the bitmap never says "available" about
\* storage that is in use.  This is the invariant a mis-derived chunk_base or a
\* stale mask breaks, and the one whose breach IS the DOUBLE-LIVE event.
OccCoveredByBits == occ \subseteq bits

\* The parked mask must never contain a slot that is live -- otherwise the next
\* AllocFromMask hands out occupied storage.
MaskNotLive == mask \cap occ = {}

\* And the direct statement of the fault.
NoDoubleHandOut == handed = {}

Inv == TypeOK /\ OccCoveredByBits /\ MaskNotLive /\ NoDoubleHandOut
=============================================================================
