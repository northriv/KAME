(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************)
------------------------- MODULE ChunkRecycle_threadepoch -------------------------
(*
 * TLA+ design verification for the proposed chunk-claim recycle protocol
 * with PER-THREAD-LOCAL EPOCH counter (no global counter).
 *
 * Successor to ChunkRecycle_microscopic, which proved:
 *   - bit-state -3 (lookup_chunk returns a recycled chunk) requires
 *     a double-payout (two threads given the same slot) to manifest
 *     IN A SINGLE-deallocate setting (Inv_NoStaleRead holds under
 *     SinglePayout=TRUE);
 *   - the lookup itself, as currently coded (back_offset and palloc
 *     loaded separately, no epoch / version check, no consistency
 *     re-read), is the AMPLIFIER that turns an upstream double-payout
 *     into the observed out-of-range / SEGV corruption.
 *
 * This spec models the FIX candidate:
 *
 *   1. Pack `back_offset` + `epoch` into a single per-unit atomic
 *      `unitMeta[u]`.  An atomic load fetches both consistently — no
 *      seqlock or two-read protocol needed at the lookup side.
 *
 *   2. `epoch` is the pair  <<owner_tid, counter>>  where counter is
 *      a THREAD-LOCAL monotonic counter held in the allocating
 *      thread's TLS (= conceptually one field on AllocSlot, or just
 *      a `thread_local uint64_t` next to it).  No global atomic
 *      counter, no contention.  Uniqueness is structural: distinct
 *      threads contribute disjoint `tid` halves; counter ABA within
 *      a single thread requires its counter to wrap.
 *
 *   3. lookup_chunk(p) captures the unit's epoch at the deallocate's
 *      entry (D_Start: dExpectEpoch = unitMeta[uP].epoch) — i.e. the
 *      epoch of the chunk that p belongs to is RECORDED before the
 *      racy region.  The subsequent two loads (unitMeta atomic read
 *      + chunk_header.palloc read) are then validated against
 *      dExpectEpoch.  Any mismatch -> the chunk was reclaimed+recycled
 *      mid-lookup -> foreign path (libsystem free).
 *
 * Note that "capturing dExpectEpoch at D_Start" is REALISTIC: in the
 * real code, p is a live slot at the moment delete[](p) is called;
 * the unit's atomic metadata at that moment is what we capture.  The
 * lookup then proceeds with that captured value as the expectation.
 *
 * The two CONSTANTS this spec sweeps:
 *
 *   MaxLocalEpoch : per-thread counter range — set small to provoke
 *                   wrap-induced ABA; set >= total recycles to verify
 *                   the design is safe at adequate width.
 *   SinglePayout  : retained from the predecessor as a sanity knob;
 *                   in this design, with epoch+pair-atomic-meta in
 *                   place, even SinglePayout=FALSE should be safe
 *                   AT ADEQUATE EPOCH WIDTH.
 *
 * SAFETY ONLY.
 *)

EXTENDS Integers, FiniteSets, TLC, Sequences

CONSTANTS
    Threads,            \* finite set of thread ids
    NumUnits,           \* units in the single region (recommend 2)
    MaxLocalEpoch,      \* per-thread counter ceiling.  AllowWrap=FALSE
                        \*   bounds the exploration here (no wrap; models
                        \*   "the counter is wide enough to never wrap in
                        \*   the lifetime of any in-flight lookup").
                        \*   AllowWrap=TRUE lets it wrap to 1, deliberately
                        \*   re-using the <<tid, c>> value to provoke ABA.
    AllowWrap,          \* BOOLEAN; see above.
    SinglePayout,       \* TRUE forbids two threads mid-dealloc of
                        \*   same unit.  Set FALSE here to stress
                        \*   the epoch design.
    Null

ASSUME NumUnits >= 1
ASSUME MaxLocalEpoch >= 1
ASSUME Cardinality(Threads) >= 1
ASSUME SinglePayout \in BOOLEAN
ASSUME AllowWrap \in BOOLEAN

Units == 0 .. (NumUnits - 1)

(***************************************************************************
 * Epoch values.
 *
 * An epoch is a pair  <<owner, counter>>  with owner \in Threads and
 * counter \in 1..MaxLocalEpoch, OR the literal Null (= released /
 * never claimed).  The pair is what lookups compare for identity.
 ***************************************************************************)
EpochValues == { Null } \cup ( Threads \X (1..MaxLocalEpoch) )

(***************************************************************************
 * Shared per-unit metadata (the proposed unitMeta atomic).
 *
 *   unitMeta[u] = [backOff |-> 0..3, epoch |-> EpochValues]
 *
 * Both fields are atomically read & written together.  In the
 * implementation: pack into a single uint64_t (e.g. 8-bit backOff +
 * 16-bit tid + 40-bit counter), use std::atomic<uint64_t>.
 *
 * In addition to unitMeta, the chunk_header carries `palloc` and the
 * SAME epoch (so a lookup can sanity-check by reading both, but the
 * primary identity check is unitMeta.epoch == dExpectEpoch).  We
 * keep `chunkEpoch[base]` as a separate variable so an interleaving
 * that updates one but not the other is representable; the spec then
 * verifies that the protocol updates them consistently enough that
 * INV_NoStaleRead holds.
 ***************************************************************************)
VARIABLES
    \* per-unit atomic metadata
    backOff,        \* same atomic word as `epoch` below
    epoch,          \* per-unit epoch (Null = released)
    \* chunk_header fields (live only on the chunk's base unit)
    palloc,         \* base.chunk_header.palloc (0 = released)
    chunkEpoch,     \* base.chunk_header.epoch (= unitMeta[base].epoch
                    \*   under correct protocol, but kept separate to
                    \*   make ordering races representable)
    chunkSpan,
    \* per-thread state
    localEpoch,     \* thread t's TLS counter (1..MaxLocalEpoch, wraps)
    allocGenOf,     \* ghost: which epoch a live slot at base u was
                    \*   allocated under (Null = no live slot)
    pc,
    \* deallocate-in-flight registers
    dUnit,
    dExpectEpoch,   \* epoch captured at D_Start
    dMetaBackOff,   \* unitMeta.backOff captured at D_LoadMeta
    dMetaEpoch,     \* unitMeta.epoch    captured at D_LoadMeta (same atomic)
    dBase,
    dPalloc,
    dChunkEpoch,    \* chunk_header.epoch captured at D_LoadPal
    \* bug flag
    staleRead

vars == << backOff, epoch, palloc, chunkEpoch, chunkSpan,
           localEpoch, allocGenOf, pc,
           dUnit, dExpectEpoch, dMetaBackOff, dMetaEpoch, dBase,
           dPalloc, dChunkEpoch, staleRead >>

(***************************************************************************
 * Initial state: empty region, no live slots, each thread's counter
 * at 1.
 ***************************************************************************)
Init ==
    /\ backOff      = [u \in Units |-> 0]
    /\ epoch        = [u \in Units |-> Null]
    /\ palloc       = [u \in Units |-> Null]
    /\ chunkEpoch   = [u \in Units |-> Null]
    /\ chunkSpan    = [u \in Units |-> 0]
    /\ localEpoch   = [t \in Threads |-> 1]
    /\ allocGenOf   = [u \in Units |-> Null]
    /\ pc           = [t \in Threads |-> "idle"]
    /\ dUnit        = [t \in Threads |-> Null]
    /\ dExpectEpoch = [t \in Threads |-> Null]
    /\ dMetaBackOff = [t \in Threads |-> Null]
    /\ dMetaEpoch   = [t \in Threads |-> Null]
    /\ dBase        = [t \in Threads |-> Null]
    /\ dPalloc      = [t \in Threads |-> Null]
    /\ dChunkEpoch  = [t \in Threads |-> Null]
    /\ staleRead    = FALSE

(***************************************************************************
 * ALLOCATE a chunk (recycle a free unit run).
 *
 * Real-code order (after the fix to publish back_offset INSIDE the
 * CAS-success branch — d2e2c32b):
 *
 *   1. CAS claim bits.
 *   2. Write back_offset[u..u+span-1] AND epoch[u..u+span-1] (= the
 *      new unitMeta atomic, per-unit).
 *   3. Write chunk_header.palloc + chunk_header.epoch at base.
 *   4. Release ready bit (not modelled here; orthogonal to the lookup
 *      identity issue).
 *
 * We collapse 1+2+3 into one atomic step here.  The interleaving
 * point we care about is between a CONCURRENT lookup's D_LoadMeta
 * and D_LoadPal — that's what tests whether the epoch identity check
 * is sufficient.
 ***************************************************************************)

SpanFree(u, span) ==
    /\ u + span <= NumUnits
    /\ \A k \in 0..(span-1): epoch[u+k] = Null

\* Next counter for thread t.  Wraps to 1 only when AllowWrap=TRUE;
\* otherwise just keeps incrementing (the A_Allocate guard halts
\* the thread when the counter exceeds MaxLocalEpoch).
NextLocal(t) ==
    IF localEpoch[t] >= MaxLocalEpoch /\ AllowWrap
        THEN 1
        ELSE localEpoch[t] + 1

A_Allocate(t) ==
    /\ pc[t] = "idle"
    /\ (localEpoch[t] <= MaxLocalEpoch \/ AllowWrap)
    /\ \E u \in Units, span \in {1, 2}:
        /\ SpanFree(u, span)
        /\ LET ev == << t, localEpoch[t] >> IN
            /\ backOff'    = [k \in Units |->
                                IF k \in (u..(u+span-1)) THEN k - u ELSE backOff[k]]
            /\ epoch'      = [k \in Units |->
                                IF k \in (u..(u+span-1)) THEN ev ELSE epoch[k]]
            /\ palloc'     = [palloc EXCEPT ![u] = ev]
            /\ chunkEpoch' = [chunkEpoch EXCEPT ![u] = ev]
            /\ chunkSpan'  = [chunkSpan EXCEPT ![u] = span]
            /\ allocGenOf' = [allocGenOf EXCEPT ![u] = ev]
        /\ localEpoch' = [localEpoch EXCEPT ![t] = NextLocal(t)]
    /\ UNCHANGED << pc, dUnit, dExpectEpoch, dMetaBackOff, dMetaEpoch,
                    dBase, dPalloc, dChunkEpoch, staleRead >>

(***************************************************************************
 * RECLAIM a chunk.  Reclaim is granted only on chunks with no live
 * slot (allocGenOf=Null at the base) — same modelling concession as
 * the predecessor; we have already proven the bit-level reclaim
 * protocol correct in ChunkAlloc_microscopic.
 *
 * Real-code reclaim order: ready clear -> palloc=0 -> madvise ->
 * back_offset clear -> claim clear.  We apply them atomically here
 * (the interleaving point that matters is between a concurrent
 * lookup's loads, captured by separate D_* edges).
 ***************************************************************************)

ReclaimableBase(u) ==
    /\ palloc[u] # Null
    /\ allocGenOf[u] = Null

R_Reclaim(t) ==
    /\ pc[t] = "idle"
    /\ \E u \in Units:
        /\ ReclaimableBase(u)
        /\ LET span == chunkSpan[u] IN
            /\ palloc'     = [palloc EXCEPT ![u] = Null]
            /\ chunkEpoch' = [chunkEpoch EXCEPT ![u] = Null]
            /\ chunkSpan'  = [chunkSpan EXCEPT ![u] = 0]
            /\ backOff'    = [k \in Units |->
                                 IF k \in (u..(u+span-1)) THEN 0 ELSE backOff[k]]
            /\ epoch'      = [k \in Units |->
                                 IF k \in (u..(u+span-1)) THEN Null ELSE epoch[k]]
    /\ UNCHANGED << localEpoch, allocGenOf, pc, dUnit, dExpectEpoch,
                    dMetaBackOff, dMetaEpoch, dBase, dPalloc,
                    dChunkEpoch, staleRead >>

(***************************************************************************
 * DEALLOCATE(p) — split into the proposed lookup steps.
 *
 *   D_Start    : pick an address p (a unit uP currently holding a
 *                live slot owned by the deallocating thread).  Capture
 *                the epoch the unit had AT THIS MOMENT into
 *                dExpectEpoch[t] — this is the realistic "I am about
 *                to free a live slot whose chunk currently has this
 *                epoch" snapshot.  Then logically free the slot.
 *   D_LoadMeta : meta = unitMeta[uP] = <<backOff, epoch>>  (atomic
 *                load — the proposed packed unit metadata).
 *   D_LoadPal  : palloc = chunk_header[base].palloc
 *                chunkEpoch = chunk_header[base].epoch
 *                Both are loaded here; in the real code these are
 *                two separate loads at the same memory region — we
 *                collapse them since the identity check below uses
 *                only dMetaEpoch vs dExpectEpoch as primary, with
 *                chunkEpoch as secondary defense.
 *   D_Resolve  : decide.  PRIMARY check: dMetaEpoch == dExpectEpoch.
 *                If yes, the unit's meta has not changed identity
 *                since D_Start; proceed.  If no -> foreign.
 *                BUG: dMetaEpoch == dExpectEpoch BUT the resolved
 *                base/palloc actually points to a different chunk
 *                than the one p was allocated from -> staleRead.
 ***************************************************************************)

D_Start(t) ==
    /\ pc[t] = "idle"
    /\ \E u \in Units:
        /\ allocGenOf[u] # Null
        \* SinglePayout: no other thread mid-dealloc of the same unit.
        /\ (SinglePayout => \A o \in Threads \ {t}: dUnit[o] # u)
        /\ dUnit'        = [dUnit EXCEPT ![t] = u]
        /\ dExpectEpoch' = [dExpectEpoch EXCEPT ![t] = epoch[u]]
        \* Logically free the slot now -- chunk becomes reclaimable.
        /\ allocGenOf'   = [allocGenOf EXCEPT ![u] = Null]
        /\ pc'           = [pc EXCEPT ![t] = "d_loadmeta"]
    /\ UNCHANGED << backOff, epoch, palloc, chunkEpoch, chunkSpan,
                    localEpoch, dMetaBackOff, dMetaEpoch, dBase,
                    dPalloc, dChunkEpoch, staleRead >>

D_LoadMeta(t) ==
    /\ pc[t] = "d_loadmeta"
    \* Atomic pair load: captures backOff and epoch together.
    /\ dMetaBackOff' = [dMetaBackOff EXCEPT ![t] = backOff[dUnit[t]]]
    /\ dMetaEpoch'   = [dMetaEpoch EXCEPT ![t] = epoch[dUnit[t]]]
    /\ dBase'        = [dBase EXCEPT ![t] = dUnit[t] - backOff[dUnit[t]]]
    /\ pc'           = [pc EXCEPT ![t] = "d_loadpal"]
    /\ UNCHANGED << backOff, epoch, palloc, chunkEpoch, chunkSpan,
                    localEpoch, allocGenOf, dUnit, dExpectEpoch,
                    dPalloc, dChunkEpoch, staleRead >>

D_LoadPal(t) ==
    /\ pc[t] = "d_loadpal"
    /\ dPalloc'     = [dPalloc EXCEPT ![t] = palloc[dBase[t]]]
    /\ dChunkEpoch' = [dChunkEpoch EXCEPT ![t] = chunkEpoch[dBase[t]]]
    /\ pc'          = [pc EXCEPT ![t] = "d_resolve"]
    /\ UNCHANGED << backOff, epoch, palloc, chunkEpoch, chunkSpan,
                    localEpoch, allocGenOf, dUnit, dExpectEpoch,
                    dMetaBackOff, dMetaEpoch, dBase, staleRead >>

D_Resolve(t) ==
    /\ pc[t] = "d_resolve"
    \* SEQLOCK RE-READ.  Capture unitMeta[uP] again here; only accept
    \* the lookup if the meta has not changed between D_LoadMeta and
    \* this re-read.  This closes the "reclaim slipped in after our
    \* load but before we acted" window that a pair-atomic single
    \* read cannot detect on its own.
    /\ LET reReadEpoch == epoch[dUnit[t]] IN
        \/ \* Released or detected stale via re-read mismatch.
           /\ \/ dMetaEpoch[t] = Null
              \/ dPalloc[t] = Null
              \/ dMetaEpoch[t] # reReadEpoch
           /\ UNCHANGED staleRead
        \/ \* Identity accepted: meta unchanged across the lookup AND
           \* matches the captured dExpectEpoch (from D_Start).  Under
           \* correct epoch uniqueness this MUST resolve to the
           \* original chunk; staleRead should never fire.
           /\ dMetaEpoch[t] # Null
           /\ dPalloc[t] # Null
           /\ dMetaEpoch[t] = reReadEpoch              \* seqlock OK
           /\ IF dMetaEpoch[t] = dExpectEpoch[t]
                 THEN IF chunkEpoch[dBase[t]] = dExpectEpoch[t]
                         /\ palloc[dBase[t]] = dExpectEpoch[t]
                         THEN UNCHANGED staleRead       \* safe
                         ELSE staleRead' = TRUE         \* identity mismatch
                 ELSE UNCHANGED staleRead               \* rejected -> safe
    /\ pc'           = [pc EXCEPT ![t] = "idle"]
    /\ dUnit'        = [dUnit EXCEPT ![t] = Null]
    /\ dExpectEpoch' = [dExpectEpoch EXCEPT ![t] = Null]
    /\ dMetaBackOff' = [dMetaBackOff EXCEPT ![t] = Null]
    /\ dMetaEpoch'   = [dMetaEpoch EXCEPT ![t] = Null]
    /\ dBase'        = [dBase EXCEPT ![t] = Null]
    /\ dPalloc'      = [dPalloc EXCEPT ![t] = Null]
    /\ dChunkEpoch'  = [dChunkEpoch EXCEPT ![t] = Null]
    /\ UNCHANGED << backOff, epoch, palloc, chunkEpoch, chunkSpan,
                    localEpoch, allocGenOf >>

(***************************************************************************
 * Next-state.
 ***************************************************************************)
Next ==
    \E t \in Threads:
        \/ A_Allocate(t)
        \/ R_Reclaim(t)
        \/ D_Start(t)
        \/ D_LoadMeta(t)
        \/ D_LoadPal(t)
        \/ D_Resolve(t)

Spec == Init /\ [][Next]_vars

(***************************************************************************
 * Invariants.
 ***************************************************************************)

\* Primary: lookup never accepts a stale chunk.
Inv_NoStaleRead == staleRead = FALSE

\* Sanity: a unit with a live slot is claimed (epoch non-Null) and its
\* base's palloc / chunkEpoch are consistent.
Inv_LiveSlotConsistent ==
    \A u \in Units:
        allocGenOf[u] # Null =>
            /\ epoch[u] # Null
            /\ palloc[u] # Null
            /\ chunkEpoch[u] = epoch[u]

\* Sanity: a unit's backOff resolves to a unit in range.
Inv_BackOffInRange ==
    \A u \in Units:
        epoch[u] # Null =>
            LET base == u - backOff[u] IN
                /\ base >= 0
                /\ base \in Units

================================================================================
