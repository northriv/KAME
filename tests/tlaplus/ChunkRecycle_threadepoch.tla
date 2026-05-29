(***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            kamepoolalloc/LICENSE-APACHE-2.0)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see kamepoolalloc/LICENSE-GPL-2.0).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied.
 ***************************************************************************)
------------------------- MODULE ChunkRecycle_threadepoch -------------------------
(*
 * FINAL DESIGN — kamepoolalloc chunk-claim / lookup / recycle protocol
 * with thread-local epoch + per-unit pair-atomic metadata + seqlock
 * lookup.  Successor to ChunkRecycle_microscopic (which proved the
 * existing protocol's bit-state -3 corruption requires this redesign).
 *
 * ============================================================
 *  CONFIRMED DESIGN (= this spec verifies it)
 * ============================================================
 *
 * Per-region static state (one entry per unit, 128 unit/region):
 *
 *   s_claim_bitmap[u]   : 1 BIT.  (Phase 5h's `ready` bit is RETIRED —
 *                         subsumed by chunk_header.epoch.)  The 2-bit
 *                         `{claim, ready}` encoding (`82f83b16`) reverts
 *                         to the Phase 5g-era 1-bit-per-unit layout, so
 *                         a 64-bit BitmapWord holds 64 unit claims.
 *
 *   s_unit_meta[u]      : atomic<uint64_t>.  Packed pair, written and
 *                         read as a single atomic word:
 *                            8-bit  back_off     (unit -> base distance)
 *                           16-bit  owner_tid    (allocating thread)
 *                           40-bit  counter      (thread-local serial)
 *                         epoch == <<owner_tid, counter>> with the
 *                         all-zero pattern reserved as "released".
 *
 * Per-chunk-base state (lives at chunk_header, slot 0's 8-byte field):
 *
 *   chunk_header.palloc : PoolAllocator* (= chunk identity).
 *   chunk_header.epoch  : uint64_t — same value as s_unit_meta[base].
 *                         Used as the foreign-detection sentinel:
 *                            epoch == 0 → released / not yet built
 *                            (subsumes the retired `ready` bit).
 *
 * Epoch generation:
 *
 *   thread_local uint64_t tls_chunk_epoch_counter;   // 40-bit useful
 *   uint64_t make_epoch(t) {
 *       return (++tls_chunk_epoch_counter << 16) | (tid_of(t) & 0xFFFF);
 *   }
 *
 *   Wrap is structurally impossible in practice: at 1 ns/allocation
 *   per thread, a 40-bit counter wraps in ~18 minutes — model that as
 *   "never" for any in-flight lookup window (microseconds).  Bump to
 *   48-bit if conservative; this spec models the BOUNDED-no-wrap case
 *   (AllowWrap=FALSE) for safety and the WRAP case (AllowWrap=TRUE)
 *   to confirm the design DOES break under wrap (= justifies the
 *   adequate-width requirement).
 *
 * ============================================================
 *  IMPLEMENTATION ORDER (real C++ ops)
 * ============================================================
 *
 * allocate_chunk(region):
 *   1. CAS claim bits for span units    (1-bit/unit, acquire on success)
 *   2. write s_unit_meta[u..u+span-1] = {back_off, epoch}   (relaxed)
 *   3. write chunk_header.palloc + chunk_header.epoch        (release)
 *
 * deallocate_chunk(chunk_base):
 *   1. write chunk_header.epoch = 0                          (release)
 *      (single STORE both retires `ready` and stamps the
 *       header as stale for lookup_chunk to detect)
 *   2. write chunk_header.palloc = 0                         (relaxed)
 *   3. clear s_unit_meta[u..u+span-1]                        (relaxed)
 *   4. madvise (slot region only; chunk_header skipped)      (-)
 *   5. clear claim bits for span units                       (release)
 *      (last; makes the units recyclable)
 *
 * lookup_chunk(p):
 *   1. unit = (p - region_base) >> CHUNK_SHIFT
 *   2. meta1 = s_unit_meta[unit].load(acquire)
 *      if meta1.epoch == 0 → foreign  (= released sentinel)
 *   3. base    = unit - meta1.back_off
 *      ph_ep   = chunk_header[base].epoch.load(acquire)
 *      palloc  = chunk_header[base].palloc.load(acquire)
 *      atomic_thread_fence(acquire)
 *   4. meta2 = s_unit_meta[unit].load(relaxed)              (SEQLOCK)
 *   5. if meta1 != meta2 OR ph_ep != meta1.epoch → foreign
 *   6. return palloc                                         (accepted)
 *
 * ============================================================
 *  WHAT THE PRIOR SPECS PROVED ALONG THE WAY
 * ============================================================
 *
 *   ChunkAlloc_microscopic   : bit-level reclaim exclusion correct.
 *   ChunkRecycle_microscopic : bit-state -3 requires a double-payout;
 *                              lookup_chunk's two unsynchronised loads
 *                              are the amplifier.
 *   THIS SPEC                : per-AllocSlot epoch + unit-meta pair-
 *                              atomic + lookup seqlock re-read is
 *                              SOUND when the counter does not wrap
 *                              (3,693,839 states, no error).
 *
 * The CONSTANT knobs used to control the experiment:
 *
 *   MaxLocalEpoch : per-thread counter ceiling
 *   AllowWrap     : FALSE bounds exploration (no wrap; model
 *                   "the counter is wide enough"); TRUE provokes ABA
 *   SinglePayout  : FALSE allows two threads mid-dealloc of the
 *                   same unit (worst case the design must survive)
 *
 *   NumUnits=2, MaxLocalEpoch=5, AllowWrap=FALSE, SinglePayout=FALSE:
 *      NO ERROR over 3.7 M states.
 *   AllowWrap=TRUE: ABA-induced violation in ~1k states.  Expected;
 *                   justifies the wide-counter requirement.
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
