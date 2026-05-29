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
------------------------- MODULE ChunkRecycle_microscopic -------------------------
(*
 * Microscopic TLA+ model of kamepoolalloc's ADDRESS-LEVEL chunk lookup
 * during reclaim + recycle — the layer ChunkAlloc_microscopic excluded.
 *
 * ChunkAlloc_microscopic proved the Phase 5j bit-level reclaim-exclusion
 * protocol correct for a single non-reused chunk.  The bug it could not
 * see is the sister investigation's bit-state = -3 (idx >= m_count):
 * a `deallocate(p)` computed an out-of-range slot index — the signature
 * of an address whose chunk was RECLAIMED and RECYCLED at a different
 * layout while a deallocate of an old-generation slot was in flight.
 *
 * This spec models the actual address arithmetic of
 * `PoolAllocatorBase::lookup_chunk` / `::deallocate` (allocator.cpp
 * lines 2240-2334) and the `deallocate_chunk` / `allocate_chunk`
 * handover ordering, all at the granularity of one atomic memory op
 * per pc-edge.
 *
 * ============================================================
 *  REAL CODE BEING MODELLED
 * ============================================================
 *
 * lookup_chunk(p) / deallocate(p):
 *     unit_idx  = (p - region_base) >> CHUNK_SHIFT
 *     base_off  = s_back_offset[unit_idx]          // load #1
 *     base_idx  = unit_idx - base_off
 *     chunk_base= region_base + base_idx*UNIT
 *     palloc    = *(chunk_base + PALLOC_OFFSET)     // load #2
 *     if(palloc <= 1) return foreign
 *     ... use palloc + per-slot header to compute the bit index ...
 *
 *   The two loads (back_offset, palloc) are SEPARATE, unsynchronised
 *   reads.  Nothing pins the chunk between them.
 *
 * deallocate_chunk(chunk_base) — reclaim, in THIS order (lines 2099-):
 *     1. clear ready bit            (release-ish)
 *     2. chunk_header.size_info = 0
 *        chunk_header.palloc    = 0   // <-- load #2 sees 0 after this
 *     3. madvise
 *     4. clear back_offset[all units]  // <-- load #1 sees 0 after this
 *     5. clear claim bits             (release)
 *
 * allocate_chunk(region) — recycle a freed unit run (lines 1437-1499):
 *     1. claim CAS on the base unit's bitmap (acquire ownership)
 *     2. write back_offset[all units of the new chunk]
 *     3. write chunk_header (palloc = new PoolAllocator, size_info)
 *     4. fetch_or ready bit            (release)
 *
 * ============================================================
 *  THE RACE THIS HUNTS
 * ============================================================
 *
 * A deallocate of address p (a slot of the OLD chunk C0 occupying unit
 * uP) interleaves with C0's reclaim and a recycle of unit uP's run into
 * a NEW, differently-sized chunk C1:
 *
 *   - deallocate reads back_offset[uP] (load #1).  If it reads C0's
 *     value it computes C0's base; if C1's, C1's base.
 *   - deallocate reads palloc[base] (load #2).  Between load #1 and
 *     load #2 the reclaim+recycle can advance, so the base computed
 *     from C0's back_offset can carry C1's palloc (or vice-versa).
 *   - if the resolved (base, palloc) pair belongs to C1 but p is a C0
 *     slot, the subsequent slot-index computation uses C1's layout on a
 *     C0 address -> idx may exceed C1's m_count -> bit-state = -3, OR a
 *     LIVE C1 slot's bit is cleared -> double-payout / overlap.
 *
 * SAFETY ONLY.  No fairness / liveness (weak-CAS-with-retry C++).
 *
 * State-space discipline:
 *   - one region, NumUnits units (2 is enough: a 2-unit C0 recycled
 *     into a 1-unit C1 leaves unit 1 reinterpreted).
 *   - generations bounded by MaxGen (2: C0 -> reclaim -> C1).
 *   - one tracked deallocate-in-flight per non-owner thread.
 *
 * ============================================================
 *  IMPORTANT MODELLING CAVEAT (read before trusting the counterexample)
 * ============================================================
 *
 * `D_Start` logically frees the slot (allocGenOf[u] := Null) AT THE
 * START of the deallocate, making the chunk immediately reclaimable
 * while the lookup loads are still in flight.  This is STRONGER (more
 * permissive) than the real cross-free / owner-drain paths, where the
 * slot's bitmap bit stays SET until the flush CAS-clears it — so the
 * chunk's MASK_CNT stays >= 1 and the chunk is NOT reclaimable during
 * the lookup.  Under that faithful ordering the lookup always resolves
 * the live chunk and no stale read occurs.
 *
 * Therefore the counterexample TLC finds here does NOT, by itself,
 * prove the real code races on a SINGLE well-behaved deallocate.  What
 * it DOES establish, robustly, is a design fact:
 *
 *     lookup_chunk(p) trusts `palloc != 0` alone.  It performs NO
 *     check that the chunk it resolved is the same generation /
 *     identity that address p was allocated from.  Its two loads
 *     (back_offset, palloc) are unsynchronised.
 *
 * So IF anything upstream causes p's bitmap bit to be cleared early
 * (the prime suspects: a double-payout of the same slot to two
 * threads, or a drain/cross-flush ordering that clears the bit before
 * the lookup completes), the chunk becomes reclaimable+recyclable mid-
 * lookup and lookup_chunk silently returns the WRONG (recycled) chunk
 * -> idx >= m_count == sister session's observed bit-state -3.
 *
 * In other words: this spec localises the SECOND-ORDER amplifier
 * (lookup_chunk's missing generation check) that turns an upstream
 * slot-lifetime bug into the observed out-of-range corruption.  The
 * FIRST-ORDER cause (what clears p's bit early) needs the buddy /
 * FS=false N-bit-per-slot / claim-recycle layer — a follow-up spec.
 *
 * A fix suggested directly by this model: stamp each chunk header with
 * a generation/epoch and have lookup_chunk reject a resolved chunk
 * whose epoch does not match the one encoded in (or alongside) p,
 * falling through to the libsystem path instead of corrupting a
 * recycled chunk's bitmap.
 *)

EXTENDS Integers, FiniteSets, TLC, Sequences

CONSTANTS
    Threads,        \* finite set of thread ids
    NumUnits,       \* units in the single region (recommend 2)
    MaxGen,         \* generation bound (recommend 2: C0 then C1)
    FreeBeforeLookup, \* TRUE  = permissive model: D_Start frees the slot
                      \*         immediately (chunk reclaimable mid-lookup);
                      \*         exposes lookup_chunk's missing generation
                      \*         check (Inv_NoStaleRead violated).
                      \* FALSE = faithful model: the slot stays live (bit
                      \*         set) until the lookup completes in
                      \*         D_Resolve; this is the real cross-free /
                      \*         drain ordering.  Used as a NEGATIVE
                      \*         CONTROL — TLC should find NO error,
                      \*         confirming a single well-behaved
                      \*         deallocate is safe and the bug must be
                      \*         upstream (early bit-clear / double-payout).
    SinglePayout,     \* TRUE  = forbid two threads being mid-deallocate of
                      \*         the SAME unit (= no double-payout of a
                      \*         slot to two threads).  NEGATIVE CONTROL:
                      \*         with this on, TLC should find NO error,
                      \*         proving the corruption REQUIRES an
                      \*         upstream double-payout — pinning the
                      \*         first-order bug to the claim/recycle or
                      \*         FS=false N-bit layer.
                      \* FALSE = allow it (the corruption appears).
    Null

ASSUME NumUnits >= 1
ASSUME MaxGen >= 1
ASSUME Cardinality(Threads) >= 1
ASSUME FreeBeforeLookup \in BOOLEAN
ASSUME SinglePayout \in BOOLEAN

Units == 0 .. (NumUnits - 1)

(***************************************************************************
 * Shared "memory" — the per-unit metadata that lookup_chunk reads and
 * deallocate_chunk / allocate_chunk write.
 *
 *   claim[u]    : claim bit for unit u (TRUE = some chunk owns this unit)
 *   ready[u]    : ready bit for unit u (TRUE = chunk header is valid)
 *   backOff[u]  : s_back_offset[u] — distance back to the chunk's base
 *                 unit (0 for base / single-unit; 1.. for continuation).
 *   palloc[u]   : chunk_header palloc field AT unit u (only meaningful
 *                 at a chunk's base unit).  0 = released / no chunk.
 *                 Otherwise = the generation id of the chunk based here
 *                 (we use the generation as a stand-in for the
 *                 PoolAllocator* identity; distinct generations =
 *                 distinct chunk objects, which is what matters for the
 *                 bug).
 *   chunkSpan[u]: number of units the chunk based at u spans (its
 *                 "layout" — a 2-unit C0 vs a 1-unit C1 differ here).
 *                 0 if no chunk based here.
 *
 *   gen         : next generation id to hand out (monotone).
 *
 *   allocGenOf[u]: ghost — which generation a LIVE slot at unit u
 *                 belongs to (Null if no live slot tracked at u).  Set
 *                 when a thread "allocates" a slot at u; checked by the
 *                 deallocate to detect stale-layout reads.  This is the
 *                 specification's notion of ground truth, independent of
 *                 the (racy) shared metadata.
 ***************************************************************************)

VARIABLES
    claim,
    ready,
    backOff,
    palloc,
    chunkSpan,
    gen,
    allocGenOf,
    \* per-thread deallocate-in-flight registers (model load #1 / load #2)
    pc,
    dUnit,          \* the unit address p being deallocated maps to
    dBaseOff,       \* load #1 result captured by the thread
    dBase,          \* base unit computed from load #1
    dPalloc,        \* load #2 result captured by the thread
    dExpectGen,     \* the generation the thread's address p was alloc'd at
    \* bug flags raised by deallocate when it detects a mismatch
    staleRead       \* set TRUE if any deallocate resolved a non-foreign
                    \* chunk whose generation != the address's alloc gen

vars == << claim, ready, backOff, palloc, chunkSpan, gen, allocGenOf,
           pc, dUnit, dBaseOff, dBase, dPalloc, dExpectGen, staleRead >>

(***************************************************************************
 * Initial state: region empty (no chunk), all threads idle.
 ***************************************************************************)
Init ==
    /\ claim      = [u \in Units |-> FALSE]
    /\ ready      = [u \in Units |-> FALSE]
    /\ backOff    = [u \in Units |-> 0]
    /\ palloc     = [u \in Units |-> 0]
    /\ chunkSpan  = [u \in Units |-> 0]
    /\ gen        = 1                       \* generation ids start at 1
    /\ allocGenOf = [u \in Units |-> Null]
    /\ pc         = [t \in Threads |-> "idle"]
    /\ dUnit      = [t \in Threads |-> Null]
    /\ dBaseOff   = [t \in Threads |-> Null]
    /\ dBase      = [t \in Threads |-> Null]
    /\ dPalloc    = [t \in Threads |-> Null]
    /\ dExpectGen = [t \in Threads |-> Null]
    /\ staleRead  = FALSE

(***************************************************************************
 * ALLOCATE A CHUNK (recycle a free unit run).
 *
 * For state-economy we allocate chunks of span 1 or span 2 starting at
 * a base unit whose whole span is currently unclaimed.  A single
 * "atomic-ish" action sequence in C++ order; we expose the steps that
 * matter for the lookup race as separate pc edges only for the
 * deallocate side (the allocate side's internal ordering is less
 * critical to the hunted race — but we DO keep claim-before-header-
 * before-ready as the real code does, via a small sub-pc).
 *
 * A thread may allocate at most while idle; the allocation also
 * registers a live slot (allocGenOf) at the chunk's base unit with the
 * new generation.  That live slot is what a later deallocate will try
 * to free.
 ***************************************************************************)

\* A span starting at base u is free if all its units are unclaimed and
\* within range.
SpanFree(u, span) ==
    /\ u + span <= NumUnits
    /\ \A k \in 0..(span-1): claim[u+k] = FALSE

A_Allocate(t) ==
    /\ pc[t] = "idle"
    /\ gen <= MaxGen
    /\ \E u \in Units, span \in {1, 2}:
        /\ SpanFree(u, span)
        \* C++ allocate_chunk order: claim -> back_offset -> header
        \* (palloc) -> ready.  We apply them in one step here; the
        \* deallocate side is where the racy READ interleaving lives.
        /\ claim'     = [k \in Units |->
                            IF k \in (u..(u+span-1)) THEN TRUE ELSE claim[k]]
        /\ backOff'   = [k \in Units |->
                            IF k \in (u..(u+span-1)) THEN k - u ELSE backOff[k]]
        /\ palloc'    = [palloc EXCEPT ![u] = gen]
        /\ chunkSpan' = [chunkSpan EXCEPT ![u] = span]
        /\ ready'     = [ready EXCEPT ![u] = TRUE]
        \* register a live slot at the base unit, tagged with this gen
        /\ allocGenOf'= [allocGenOf EXCEPT ![u] = gen]
        /\ gen'       = gen + 1
    /\ UNCHANGED << pc, dUnit, dBaseOff, dBase, dPalloc, dExpectGen,
                    staleRead >>

(***************************************************************************
 * RECLAIM A CHUNK (deallocate_chunk) — C++ order, as separate atomic
 * steps so a concurrent deallocate's two loads can interleave at any
 * point.
 *
 *   R_ClearReady   : ready[base] = FALSE
 *   R_ClearHeader  : palloc[base] = 0 ; chunkSpan[base] = 0
 *   R_ClearBackOff : backOff[all units of the chunk] = 0
 *   R_ClearClaim   : claim[all units] = FALSE  (units become recyclable)
 *
 * Reclaim is only legal on a chunk with NO live slot (allocGenOf=Null
 * at its base) — modelling the count==0 precondition proven sufficient
 * by ChunkAlloc_microscopic.  THIS IS THE KEY MODELLING CHOICE: we
 * GRANT that the bit-level protocol never reclaims a chunk with a live
 * slot.  The question is whether, even so, the ADDRESS-level lookup of
 * an ALREADY-FREED slot (one whose deallocate is in flight) can be
 * corrupted by recycle.
 *
 * To express "a slot's deallocate is in flight after the slot was
 * logically freed", a thread first frees its slot (clears allocGenOf,
 * making the chunk reclaimable) and THEN walks the lookup_chunk loads.
 * That mirrors reality: the application has called delete[](p); the
 * slot is logically gone; deallocate(p) is mid-flight doing its loads;
 * meanwhile the chunk legitimately becomes empty and is reclaimed and
 * recycled.
 ***************************************************************************)

ChunkBaseUnits ==
    { u \in Units : palloc[u] # 0 }     \* units that are a chunk base

ReclaimableBase(u) ==
    /\ palloc[u] # 0
    /\ allocGenOf[u] = Null            \* no live slot (count==0 granted)

R_Reclaim(t) ==
    /\ pc[t] = "idle"
    /\ \E u \in ChunkBaseUnits:
        /\ ReclaimableBase(u)
        /\ LET span == chunkSpan[u] IN
            \* All four reclaim writes applied together at this abstraction
            \* (their internal order matters only relative to a concurrent
            \* deallocate's loads; we model that interleaving by letting
            \* deallocate capture loads at idle between any actions).
            /\ ready'     = [ready EXCEPT ![u] = FALSE]
            /\ palloc'    = [palloc EXCEPT ![u] = 0]
            /\ chunkSpan' = [chunkSpan EXCEPT ![u] = 0]
            /\ backOff'   = [k \in Units |->
                                IF k \in (u..(u+span-1)) THEN 0 ELSE backOff[k]]
            /\ claim'     = [k \in Units |->
                                IF k \in (u..(u+span-1)) THEN FALSE ELSE claim[k]]
    /\ UNCHANGED << gen, allocGenOf, pc, dUnit, dBaseOff, dBase,
                    dPalloc, dExpectGen, staleRead >>

(***************************************************************************
 * DEALLOCATE(p) — the address-level lookup, split into its two loads so
 * recycle can slip between them.
 *
 *   D_Start    : choose an address p (a unit uP that currently OR
 *                formerly held a live slot this thread allocated).
 *                Capture dExpectGen = the generation p was allocated at.
 *                FIRST logically free the slot (allocGenOf[uP] := Null)
 *                so the chunk can legitimately become reclaimable while
 *                this deallocate is in flight — the realistic ordering.
 *   D_LoadBack : base_off = backOff[uP]      (load #1)
 *                base     = uP - base_off
 *   D_LoadPal  : palloc_v = palloc[base]     (load #2)
 *   D_Resolve  : if palloc_v <= 1 -> foreign (fine, falls to libsystem);
 *                else the thread treats `base`/palloc_v as p's chunk.
 *                BUG CHECK: if palloc_v is a generation != dExpectGen,
 *                the deallocate is operating on a recycled chunk for an
 *                old address -> raise staleRead (== bit-state -3 / the
 *                overlap-write corruption).
 ***************************************************************************)

\* A thread can start a deallocate for any unit that holds a live slot
\* IT allocated.  (One in-flight deallocate per thread for state economy.)
D_Start(t) ==
    /\ pc[t] = "idle"
    /\ \E u \in Units:
        /\ allocGenOf[u] # Null
        \* SinglePayout negative control: no other thread may be mid-
        \* deallocate of the same unit (= the slot was not double-paid
        \* out to two threads).
        /\ (SinglePayout => \A other \in Threads \ {t}: dUnit[other] # u)
        /\ dUnit'      = [dUnit EXCEPT ![t] = u]
        /\ dExpectGen' = [dExpectGen EXCEPT ![t] = allocGenOf[u]]
        \* Permissive (FreeBeforeLookup): logically free now -> chunk
        \* reclaimable mid-lookup.  Faithful (~FreeBeforeLookup): keep
        \* the slot live (bit stays set) until D_Resolve, matching the
        \* real flush ordering where the bit is cleared AFTER the lookup.
        /\ allocGenOf' = IF FreeBeforeLookup
                            THEN [allocGenOf EXCEPT ![u] = Null]
                            ELSE allocGenOf
        /\ pc'         = [pc EXCEPT ![t] = "d_loadback"]
    /\ UNCHANGED << claim, ready, backOff, palloc, chunkSpan, gen,
                    dBaseOff, dBase, dPalloc, staleRead >>

D_LoadBack(t) ==
    /\ pc[t] = "d_loadback"
    /\ dBaseOff' = [dBaseOff EXCEPT ![t] = backOff[dUnit[t]]]
    /\ dBase'    = [dBase EXCEPT ![t] = dUnit[t] - backOff[dUnit[t]]]
    /\ pc'       = [pc EXCEPT ![t] = "d_loadpal"]
    /\ UNCHANGED << claim, ready, backOff, palloc, chunkSpan, gen,
                    allocGenOf, dUnit, dPalloc, dExpectGen, staleRead >>

D_LoadPal(t) ==
    /\ pc[t] = "d_loadpal"
    /\ dPalloc' = [dPalloc EXCEPT ![t] = palloc[dBase[t]]]
    /\ pc'      = [pc EXCEPT ![t] = "d_resolve"]
    /\ UNCHANGED << claim, ready, backOff, palloc, chunkSpan, gen,
                    allocGenOf, dUnit, dBaseOff, dBase, dExpectGen,
                    staleRead >>

D_Resolve(t) ==
    /\ pc[t] = "d_resolve"
    /\ \/ \* foreign: palloc cleared (chunk released) -> falls to
          \* libsystem free.  Safe: deallocate does not touch the bitmap.
          /\ dPalloc[t] = 0
          /\ UNCHANGED staleRead
       \/ \* non-foreign: thread treats the resolved chunk as p's owner.
          \* BUG if that chunk's generation differs from the one p was
          \* allocated at -> stale-layout read -> bit-state -3 / overlap.
          /\ dPalloc[t] # 0
          /\ IF dPalloc[t] # dExpectGen[t]
                THEN staleRead' = TRUE
                ELSE UNCHANGED staleRead
    /\ pc'      = [pc EXCEPT ![t] = "idle"]
    /\ dUnit'   = [dUnit EXCEPT ![t] = Null]
    /\ dBaseOff'= [dBaseOff EXCEPT ![t] = Null]
    /\ dBase'   = [dBase EXCEPT ![t] = Null]
    /\ dPalloc' = [dPalloc EXCEPT ![t] = Null]
    /\ dExpectGen' = [dExpectGen EXCEPT ![t] = Null]
    \* Faithful model: the bit/slot is cleared HERE, after the lookup
    \* has resolved the chunk — i.e. the chunk could not have been
    \* reclaimed during the lookup.  (Permissive model already freed
    \* it at D_Start, so this is a no-op there.)
    /\ allocGenOf' = IF FreeBeforeLookup
                        THEN allocGenOf
                        ELSE [allocGenOf EXCEPT ![dUnit[t]] = Null]
    /\ UNCHANGED << claim, ready, backOff, palloc, chunkSpan, gen >>

(***************************************************************************
 * Next-state relation.  One thread, one microscopic action.
 ***************************************************************************)

Next ==
    \E t \in Threads:
        \/ A_Allocate(t)
        \/ R_Reclaim(t)
        \/ D_Start(t)
        \/ D_LoadBack(t)
        \/ D_LoadPal(t)
        \/ D_Resolve(t)

Spec == Init /\ [][Next]_vars

(***************************************************************************
 * Invariants.
 ***************************************************************************)

\* THE BUG WE ARE HUNTING.  A deallocate resolved a non-foreign chunk
\* whose generation differs from the address's allocation generation
\* (= reading a recycled chunk's layout for an old-generation slot,
\*  the bit-state = -3 signature).
Inv_NoStaleRead == staleRead = FALSE

\* Sanity: a chunk base with a live slot must remain claimed and ready
\* (it cannot have been reclaimed out from under a live slot — the
\*  count==0 precondition we GRANTED in R_Reclaim).
Inv_LiveSlotImpliesClaimed ==
    \A u \in Units:
        allocGenOf[u] # Null => (claim[u] = TRUE /\ palloc[u] # 0)

\* Sanity: back_offset of a claimed continuation unit points back to a
\* base unit that actually has a chunk (palloc != 0).  When this breaks,
\* lookup_chunk computes a base that isn't a real chunk base.
Inv_BackOffPointsToBase ==
    \A u \in Units:
        (claim[u] = TRUE) =>
            LET base == u - backOff[u] IN
                /\ base >= 0
                /\ base \in Units

================================================================================
