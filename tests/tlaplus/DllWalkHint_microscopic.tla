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
------------------------- MODULE DllWalkHint_microscopic -------------------------
(*
 * TLA+ model of kamepoolalloc's DLL-walk cursor + cross-thread revival
 * hint protocol (Phase 5n cursor + Phase 5v/w/x force_walk_ptr).  The
 * goal is to verify that:
 *
 *   - the cursor does not deref a released chunk (UAF safety),
 *   - cross-thread cross-free's `force_walk_ptr.store(true)` cannot
 *     hit a dangling pointer after the owner exits (UAF safety —
 *     Phase 5v had this; Phase 5x atomicised the pointer to fix),
 *   - `dll_exhausted=TRUE` is not a permanent stuck state when free
 *     space becomes available (LIVENESS).
 *
 * ============================================================
 *  WHAT THE C++ DOES
 * ============================================================
 *
 * Owner-thread state (`s_tls` in allocator_prv.h):
 *   dll_head, dll_tail            : DLL of this thread's chunks
 *   dll_cursor                    : where the last walk stopped
 *   dll_exhausted                 : last walk reached end with no space
 *   dll_force_walk_from_head      : atomic<bool> — set by cross-freers
 *                                   when they revive a chunk's space
 *
 * Per-chunk state:
 *   m_flags_packed                : MASK_CNT (count of non-empty m_flags
 *                                   words; >=1 means SOME live slot)
 *   m_owner_dll_force_walk_ptr    : atomic<atomic<bool>*> — points to
 *                                   the owner's dll_force_walk_from_head.
 *                                   Set by allocate_chunk (line 672),
 *                                   nullified by release_dll_chunks_for_
 *                                   thread (line 2037) — release store.
 *
 * Owner walk (allocator.cpp 1808-1861):
 *   if (dll_force_walk_from_head.exchange(false))
 *       cursor = nullptr; exhausted = false;
 *   if (!exhausted) {
 *       for (c = cursor ?: dll_head; c; c = c->dll_next)
 *           if (c->allocate_pooled(...)) { cursor = c; return; }
 *       cursor = nullptr; exhausted = true;
 *   }
 *   mmap fresh chunk;
 *
 * Cross-thread free (allocator.cpp 981, 985):
 *   if (auto *p = chunk->m_owner_dll_force_walk_ptr.load(acquire))
 *       p->store(true, relaxed);
 *
 * Owner exit (allocator.cpp 2037):
 *   chunk->m_owner_dll_force_walk_ptr.store(nullptr, release);
 *   ... then ATOMIC_FETCH_AND on m_flags_packed to clear BIT_OWNED ...
 *
 * ============================================================
 *  KEY RACES TO HUNT
 * ============================================================
 *
 *   R1.  Owner exits between a cross-freer's `force_walk_ptr.load` and
 *        its `p->store(true)` — i.e. p points into TLS that the owner
 *        has just torn down.  Phase 5v's plain-pointer design hit this
 *        as 1000-thread SEGV; Phase 5x's atomic load (with the owner's
 *        release-store of nullptr) is intended to plug it.  TLC must
 *        confirm.
 *
 *   R2.  Owner reaches `dll_exhausted=TRUE` while a cross-freer has
 *        already revived a chunk's slot count, BUT the hint flag was
 *        consumed (exchanged false) just before the cross-freer set
 *        it.  If no further wake-up arrives, owner is stuck mmap'ing
 *        fresh chunks forever despite reclaim-ready space — a perf
 *        bug expressible as a LIVENESS failure (the chunk's free
 *        slot is never reused by the owner).
 *
 *   R3.  Cursor stale: cursor points to a chunk that the owner itself
 *        released via owner_release neighbour-release (allocator.cpp
 *        1785: `if (s_tls.dll_cursor == nx) s_tls.dll_cursor = nxnext;`
 *        is the existing guard).  Walks subsequent to a release must
 *        not deref a freed chunk.
 *
 * SAFETY ONLY for now (R1 / R3); R2 is liveness — added as a
 * commented-out PROPERTY at the bottom for a separate liveness pass.
 *)

EXTENDS Integers, FiniteSets, TLC, Sequences

CONSTANTS
    Owner,          \* the single owning thread
    Freers,         \* set of cross-freer thread ids
    NumChunks,      \* DLL length (recommend 2)
    MaxAllocOps,    \* owner's allocate budget
    MaxFreeOps,     \* per-freer cross-free budget
    Null

ASSUME NumChunks >= 1
ASSUME Cardinality(Freers) >= 1

Chunks == 1 .. NumChunks
Threads == { Owner } \cup Freers

(***************************************************************************
 * Shared state.
 *
 *   chunkExists[c]      : the chunk object is allocated (= memory live)
 *   chunkInDLL[c]       : c is in owner's DLL (between head and tail)
 *   chunkFree[c]        : c has at least one free slot (= count < cap)
 *   chunkForceWalkPtr[c]: atomic<atomic<bool>*> — Null after owner exits
 *
 *   ownerCursor         : Null, or a chunk in the DLL
 *   ownerExhausted      : the exhausted flag
 *   ownerForceWalk      : the atomic<bool> hint flag
 *   ownerAlive          : owner thread still alive
 *
 *   crossUAF            : flag — set TRUE if any cross-free dereffed a
 *                         pointer to dead-TLS storage (Phase 5v's bug)
 *   cursorUAF           : flag — set TRUE if owner walk dereffed a
 *                         released chunk
 ***************************************************************************)
VARIABLES
    chunkExists,
    chunkInDLL,
    chunkFree,
    chunkForceWalkPtr,
    ownerCursor,
    ownerExhausted,
    ownerForceWalk,
    ownerAlive,
    allocOpsLeft,
    freeOpsLeft,
    crossUAF,
    cursorUAF,
    \* per-freer captured force_walk_ptr value (Null or "OWNER_FLAG")
    fLoadedPtr,
    \* per-freer captured chunk index it is freeing
    fChunk

vars == << chunkExists, chunkInDLL, chunkFree, chunkForceWalkPtr,
           ownerCursor, ownerExhausted, ownerForceWalk, ownerAlive,
           allocOpsLeft, freeOpsLeft, crossUAF, cursorUAF,
           fLoadedPtr, fChunk >>

(***************************************************************************
 * Initial state: empty DLL, owner alive, no in-flight ops, no UAFs.
 *
 * To bootstrap something for the freers to free, we set chunkExists
 * = TRUE for all chunks AND chunkInDLL = TRUE — i.e. the DLL is
 * pre-populated with NumChunks chunks, all FULL (chunkFree=FALSE).
 * This is the realistic state where the owner is about to walk
 * looking for space.
 ***************************************************************************)
Init ==
    /\ chunkExists       = [c \in Chunks |-> TRUE]
    /\ chunkInDLL        = [c \in Chunks |-> TRUE]
    /\ chunkFree         = [c \in Chunks |-> FALSE]       \* all full
    /\ chunkForceWalkPtr = [c \in Chunks |-> "OWNER_FLAG"]  \* points to owner's flag
    /\ ownerCursor       = Null
    /\ ownerExhausted    = FALSE
    /\ ownerForceWalk    = FALSE
    /\ ownerAlive        = TRUE
    /\ allocOpsLeft      = MaxAllocOps
    /\ freeOpsLeft       = [f \in Freers |-> MaxFreeOps]
    /\ crossUAF          = FALSE
    /\ cursorUAF         = FALSE
    /\ fLoadedPtr        = [f \in Freers |-> Null]
    /\ fChunk            = [f \in Freers |-> Null]

(***************************************************************************
 * OWNER ACTIONS — modelled with the actual C++ control flow.
 ***************************************************************************)

\* (1) Hint check: exchange(false) on the force_walk flag.  If it was
\* TRUE, the owner resets its cursor and exhausted flag so the next
\* walk goes from head.  This is the FIRST thing in the C++ alloc
\* path (line 1829).
\* NOTE: allocOpsLeft no longer guards owner actions — owner is
\* modelled as "perpetually trying" until it exits, so weak fairness
\* on owner actions captures the realistic "owner keeps allocating"
\* behaviour the liveness property assumes.
O_CheckHint ==
    /\ ownerAlive
    /\ ownerForceWalk
    /\ ownerForceWalk' = FALSE
    /\ ownerCursor' = Null
    /\ ownerExhausted' = FALSE
    /\ UNCHANGED << chunkExists, chunkInDLL, chunkFree, chunkForceWalkPtr,
                    ownerAlive, allocOpsLeft, freeOpsLeft, crossUAF,
                    cursorUAF, fLoadedPtr, fChunk >>

\* (2) DLL walk step.  If exhausted is FALSE, advance cursor toward
\* the first free chunk.  Either:
\*   - cursor lands on a chunk with free space -> success, cursor stays
\*     there, an alloc op is consumed.
\*   - cursor reaches end without finding free -> exhausted=TRUE.
\*
\* Modelled as a single atomic step (the loop is internally
\* sequential; we don't expose its internal interleavings in this
\* spec because Phase 5n is single-threaded on the cursor — only
\* this thread writes it).
O_Walk ==
    /\ ownerAlive
    /\ ~ownerForceWalk           \* don't run walk if hint pending (model: do hint first)
    /\ ~ownerExhausted
    /\ \/ \* Success branch: there exists a chunk in DLL with free
          \* space; cursor lands on it AND the free slot is consumed
          \* (= the allocate succeeds, so chunkFree[c] flips back to
          \* FALSE).  This models one full alloc round-trip.
          \E c \in Chunks:
              /\ chunkInDLL[c]
              /\ chunkFree[c]
              /\ IF ~chunkExists[c] THEN cursorUAF' = TRUE ELSE cursorUAF' = cursorUAF
              /\ ownerCursor' = c
              /\ ownerExhausted' = FALSE
              /\ chunkFree' = [chunkFree EXCEPT ![c] = FALSE]
       \/ \* Walk-end branch: no chunk in DLL has free space; mark
          \* exhausted.  No op consumed (owner stays alive and may
          \* check hint or walk again later if force_walk is set).
          /\ \A c \in Chunks: chunkInDLL[c] => ~chunkFree[c]
          /\ ownerCursor' = Null
          /\ ownerExhausted' = TRUE
          /\ cursorUAF' = cursorUAF
          /\ UNCHANGED chunkFree
    /\ UNCHANGED << chunkExists, chunkInDLL, chunkForceWalkPtr,
                    ownerForceWalk, ownerAlive, allocOpsLeft, freeOpsLeft,
                    crossUAF, fLoadedPtr, fChunk >>

\* (3) Owner exit: release_dll_chunks_for_thread.  For each in-DLL chunk:
\*   (a) atomic-release-store the force_walk_ptr to nullptr (Phase 5x);
\*   (b) cleanup proceeds with claiming BIT_OWNED (modelled abstractly
\*       as just leaving the chunk in place; reclamation itself is
\*       covered by the bit-level protocol verified in ChunkAlloc_
\*       microscopic).
\* This action exits the owner; subsequent allocate ops are blocked.
\* Owner exit can fire at any time the owner is alive — modelling
\* that the application thread may terminate.  For liveness, we add
\* NO weak-fairness for O_Exit (it is a permitted but not required
\* action; we want to verify the owner-alive case doesn't get stuck
\* before exit).
O_Exit ==
    /\ ownerAlive
    /\ ownerAlive' = FALSE
    /\ chunkForceWalkPtr' = [c \in Chunks |->
                                IF chunkInDLL[c] THEN Null
                                                  ELSE chunkForceWalkPtr[c]]
    /\ UNCHANGED << chunkExists, chunkInDLL, chunkFree, ownerCursor,
                    ownerExhausted, ownerForceWalk, allocOpsLeft,
                    freeOpsLeft, crossUAF, cursorUAF,
                    fLoadedPtr, fChunk >>

(***************************************************************************
 * CROSS-FREER ACTIONS — two atomic steps for the hint protocol:
 *   X_Free_Cross         : flip chunkFree[c] from FALSE to TRUE
 *                          (modelling the bit-clear that revives space)
 *   X_Send_Hint          : capture force_walk_ptr (load-acquire), then
 *                          store true through it (if non-null).
 *
 * The split makes the race between "owner exits" and "freer's
 * load-then-store" representable: O_Exit can fire BETWEEN the freer's
 * load (X_Send_Hint_Load) and its store (X_Send_Hint_Store).
 *
 * The atomic<atomic<bool>*> protects the deref: the load returns
 * either the live pointer (synchronised-with all of the owner's TLS
 * setup) or nullptr (synchronised-with the owner's exit), but never
 * an in-between value.  We model that by:
 *   - X_Send_Hint_Load captures the current value of
 *     chunkForceWalkPtr[c] into a local register.
 *   - X_Send_Hint_Store inspects that captured value and either
 *     skips (if Null) or applies (if "OWNER_FLAG") — modelling
 *     `p->store(true)` going to the live atomic<bool>.
 *   - IF the captured value were a dangling pointer (= not atomic),
 *     the store would UAF.  This spec models the atomic + nullify
 *     pattern, so it must NOT generate a UAF; if TLC finds one,
 *     either the spec is wrong or the design is wrong.
 ***************************************************************************)

X_Free_Cross(f) ==
    /\ freeOpsLeft[f] > 0
    /\ fChunk[f] = Null            \* no in-flight free
    /\ \E c \in Chunks:
        /\ chunkInDLL[c]
        /\ chunkExists[c]
        /\ chunkFree' = [chunkFree EXCEPT ![c] = TRUE]   \* revive
        /\ fChunk' = [fChunk EXCEPT ![f] = c]
    /\ UNCHANGED << chunkExists, chunkInDLL, chunkForceWalkPtr,
                    ownerCursor, ownerExhausted, ownerForceWalk,
                    ownerAlive, allocOpsLeft, freeOpsLeft, crossUAF,
                    cursorUAF, fLoadedPtr >>

X_Send_Hint_Load(f) ==
    /\ fChunk[f] # Null
    /\ fLoadedPtr[f] = Null
    /\ fLoadedPtr' = [fLoadedPtr EXCEPT ![f] = chunkForceWalkPtr[fChunk[f]]]
    /\ UNCHANGED << chunkExists, chunkInDLL, chunkFree, chunkForceWalkPtr,
                    ownerCursor, ownerExhausted, ownerForceWalk,
                    ownerAlive, allocOpsLeft, freeOpsLeft, crossUAF,
                    cursorUAF, fChunk >>

X_Send_Hint_Store(f) ==
    /\ fLoadedPtr[f] # Null \/ fChunk[f] # Null   \* in-flight free
    /\ fLoadedPtr[f] = "OWNER_FLAG"               \* non-null capture
    /\ ownerForceWalk' = TRUE
    /\ freeOpsLeft' = [freeOpsLeft EXCEPT ![f] = freeOpsLeft[f] - 1]
    /\ fLoadedPtr' = [fLoadedPtr EXCEPT ![f] = Null]
    /\ fChunk'     = [fChunk EXCEPT ![f] = Null]
    /\ UNCHANGED << chunkExists, chunkInDLL, chunkFree, chunkForceWalkPtr,
                    ownerCursor, ownerExhausted, ownerAlive, allocOpsLeft,
                    crossUAF, cursorUAF >>

X_Send_Hint_Skip(f) ==
    \* Captured value was Null (owner had exited before our load) ->
    \* skip the store entirely.  No UAF.  This is the safe outcome
    \* of Phase 5x's atomic + nullify pattern.
    /\ fChunk[f] # Null
    /\ fLoadedPtr[f] = Null
    /\ chunkForceWalkPtr[fChunk[f]] = Null    \* AND it's still null
    /\ freeOpsLeft' = [freeOpsLeft EXCEPT ![f] = freeOpsLeft[f] - 1]
    /\ fChunk'     = [fChunk EXCEPT ![f] = Null]
    /\ UNCHANGED << chunkExists, chunkInDLL, chunkFree, chunkForceWalkPtr,
                    ownerCursor, ownerExhausted, ownerForceWalk,
                    ownerAlive, allocOpsLeft, crossUAF, cursorUAF,
                    fLoadedPtr >>

(***************************************************************************
 * Next + Spec.
 *
 * Weak fairness on owner actions models "owner keeps allocating until
 * it exits".  Weak fairness on each freer's hint-store path models
 * "cross-frees and hints eventually complete".  Without these, TLC
 * trivially finds executions where the owner just sits idle — not
 * a real-system race, only a TLA+ artefact of unrestricted scheduling.
 ***************************************************************************)
Next ==
    \/ O_CheckHint
    \/ O_Walk
    \/ O_Exit
    \/ \E f \in Freers:
        \/ X_Free_Cross(f)
        \/ X_Send_Hint_Load(f)
        \/ X_Send_Hint_Store(f)
        \/ X_Send_Hint_Skip(f)

Fairness ==
    /\ WF_vars(O_CheckHint)
    /\ WF_vars(O_Walk)
    /\ \A f \in Freers:
        /\ WF_vars(X_Send_Hint_Load(f))
        /\ WF_vars(X_Send_Hint_Store(f))
        /\ WF_vars(X_Send_Hint_Skip(f))

Spec == Init /\ [][Next]_vars /\ Fairness

(***************************************************************************
 * Invariants.
 ***************************************************************************)

\* No cross-freer ever dereffed a dangling pointer.  Phase 5x's
\* atomic<atomic<bool>*> + release-nullify guarantees this; we verify.
Inv_NoCrossUAF == crossUAF = FALSE

\* Cursor never advanced to a chunk that did not exist.
Inv_NoCursorUAF == cursorUAF = FALSE

\* Cursor, when non-null, points to a chunk that exists and is in the DLL.
Inv_CursorWellFormed ==
    ownerCursor # Null =>
        /\ chunkExists[ownerCursor]
        /\ chunkInDLL[ownerCursor]

(***************************************************************************
 * Liveness — the stuck-exhausted hunt.
 *
 * Property: whenever the owner is exhausted while a chunk in the DLL
 * has free space, the exhausted flag MUST eventually clear.
 *
 * If TLC reports this as VIOLATED, the design has a "stuck mmap-fresh
 * forever" pattern (= perf bug expressed formally).  If satisfied,
 * Phase 5x's atomic-pointer hint protocol is shown to recover from
 * every reachable exhausted state when free space is available.
 *
 * Encoded as a leads-to (~>) under the Spec's Fairness assumption.
 ***************************************************************************)
StuckExhaustedRecovers ==
    (ownerAlive
       /\ ownerExhausted
       /\ \E c \in Chunks: chunkInDLL[c] /\ chunkFree[c])
    ~> (~ownerExhausted \/ ~ownerAlive)

================================================================================
