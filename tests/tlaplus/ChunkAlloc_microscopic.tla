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
------------------------- MODULE ChunkAlloc_microscopic -------------------------
(*
 * Microscopic TLA+ model of kamepoolalloc chunk-bitmap allocate / free
 * / cross-thread flush / chunk reclaim — narrowed to expose the bug
 * class observed in `alloc_stress_test` at cross_thread_pct=100:
 * same chunk hands the same slot to two threads (overlapping payloads
 * at offset 208 reported by sister investigation session).
 *
 * SAFETY ONLY.  Liveness is intentionally out of scope — the C++
 * implementation uses weak CAS with retry and Phase 5x atomic
 * BIT_OWNED-clear handover that the TLA+ spec models with
 * non-deterministic CAS failure; there is no fairness guarantee TLC
 * could verify, only the absence of bad interleavings.
 *
 * State-space discipline (matching superfine style used by Layer 2):
 *   - chunk count = 1; the alleged bug is INTRA-chunk, not cross-chunk.
 *     A second chunk would only add cartesian product without exposing
 *     the bit-level race.
 *   - bitmap width = small (4 bits, configurable via CONSTANT NumSlots).
 *     Bits map 1:1 to slots in this minimal model (FS=true layout).
 *     The borrow-header / N-bit-per-slot story (FS=false, sister
 *     hypothesis C) is a follow-up spec — keep this first model
 *     focused on the bit-level CAS race itself.
 *   - threads = 2.  Mirrors the canonical owner / cross-freer split.
 *   - per-thread op budget = small (configurable via MaxOps).
 *
 * Microscopic actions (cf. allocator.cpp `allocate_pooled` /
 * `batch_return_to_bitmap` / `owner_release` /
 * `release_dll_chunks_for_thread`):
 *
 *   A_ReadFlags(t)     : read current `m_flags` snapshot into local
 *                        oldv for thread t's allocate attempt.
 *   A_PickBit(t)       : choose a free bit position from oldv.  Models
 *                        `find_training_zeros` selecting the lowest
 *                        free bit (any pick would do; lowest keeps the
 *                        state minimal).
 *   A_CAS(t)           : CAS m_flags : oldv → oldv | (1<<bit).  Fails
 *                        if m_flags has changed since A_ReadFlags;
 *                        thread loops back to A_ReadFlags on failure.
 *                        On success the slot is `owned[t] := owned[t] +
 *                        {bit}` and m_flags_packed.count incremented if
 *                        m_flags transitioned 0→non-0.
 *   F_OwnerPush(t)     : owner thread t pushes one of its owned bits
 *                        onto its TLS freelist (slot still bitmap-set;
 *                        Phase 5d "borrow"-style — bit cleared lazily
 *                        on cross-flush or batch-drain).
 *   F_CrossEnqueue(t,u): freer u enqueues one of t's owned bits onto
 *                        its own cross-batch destined for chunk owner
 *                        t.  The bit is NOT yet cleared — only when
 *                        the batch is flushed.
 *   F_FlushBit(u)      : freer u flushes a single entry from its
 *                        cross-batch: CAS-clears the bit on m_flags
 *                        (release).  When the word transitions to 0,
 *                        atomicDec m_flags_packed.count.  If the dec
 *                        brings flags_packed to 0 AND owned=FALSE,
 *                        u is the cross-side releaser → reclaim.
 *   O_OwnerExit(t)     : owner t clears BIT_OWNED via atomicFetchAnd.
 *                        If resulting value is 0 (= count was 0),
 *                        owner is the unique releaser → reclaim.
 *   X_Reclaim(t)       : delete-the-chunk action.  Records the
 *                        reclaiming thread in `reclaimed` history set.
 *
 * Invariants:
 *   Inv_NoDoubleClaim       : at most one thread owns any given bit
 *                             (= the original BUG we are hunting).
 *   Inv_AtMostOneReclaim    : |reclaimed| <= 1 over the whole run
 *                             (chunk-object double-delete absence).
 *   Inv_NoUseAfterReclaim   : no thread holds an owned bit after the
 *                             chunk has been reclaimed.
 *   Inv_FlagsPackedConsistency:
 *                             flagsPacked.count equals |{ bits where
 *                             m_flags has 1 in them }| modulo
 *                             in-flight (CAS pending) operations.
 *                             Approximate: checked only in quiescent
 *                             pc states.
 *
 * If TLC reports NO counterexample with this minimal model, the bug
 * is NOT in the high-level allocate/cross-flush/reclaim CAS protocol
 * itself — pointing the next investigation at the borrow-header N-bit
 * decoding (sister session hypothesis C) or the chunk-recycle path
 * (allocate_chunk reusing a released region).  If TLC FINDS a
 * counterexample, the trace pinpoints the exact interleaving.
 *)

EXTENDS Integers, FiniteSets, TLC, Sequences

CONSTANTS
    Threads,        \* finite set of thread ids
    OwnerThread,    \* \in Threads — the chunk's owning thread.  Allocate
                    \*   actions and owner-side free / owner_exit are
                    \*   restricted to this thread.  Other threads do
                    \*   cross-free only.  Matches Phase 4b+ design
                    \*   (per-chunk owner; shared registry retired).
    NumWords,       \* number of m_flags words in the chunk (recommend 2;
                    \*   the bug class we're hunting needs >= 2 words for
                    \*   the inc-delay race to open across words)
    BitsPerWord,    \* slots per word (recommend 2)
    MaxOps,         \* per-thread alloc-budget (alloc + free ops are paired)
    Null

ASSUME NumWords >= 1
ASSUME BitsPerWord >= 1
ASSUME Cardinality(Threads) >= 1
ASSUME OwnerThread \in Threads

\* Word indices and bit-positions-within-a-word.  Each slot is
\* uniquely identified by a pair <<w, b>> with w \in Words and
\* b \in BitsInWord.
Words == 0 .. (NumWords - 1)
BitsInWord == 0 .. (BitsPerWord - 1)
Slots == Words \X BitsInWord
WordOf(s) == s[1]
SlotsInWord(set, w) == { s \in set : WordOf(s) = w }
NumNonZeroWords(set) == Cardinality({ w \in Words : SlotsInWord(set, w) # {} })

\* Program counter states.  Granularity = one atomic step per pc edge.
PCs == {
    "idle",
    "alloc_read",      \* read oldv from m_flags
    "alloc_pick",      \* picked a bit from oldv
    "alloc_cas",       \* attempt CAS to claim
    "alloc_bump",      \* CAS succeeded; now do the SEPARATE atomicInc on
                       \*   m_flags_packed.count (race window vs owner_release
                       \*   pre-check + atomicFetchAnd)
    "alloc_done",      \* successful claim; advance budget
    "free_owner_push", \* owner pushed bit onto TLS freelist
    "free_cross_enq",  \* enqueued cross-free entry
    "flush_cas",       \* flushing one cross-free entry (CAS clear)
    "flush_check_rel", \* checking dec-to-zero releaser status
    "owner_exit_and",  \* owner_exit atomicFetchAnd(~BIT_OWNED)
    "owner_exit_check",\* check if AND brought packed to 0
    "reclaim",         \* execute delete + deallocate
    "exited"           \* thread completed all ops
}

\* The chunk's shared state ("m_flags" and "m_flags_packed").
\*
\* mflags        : set of bit positions currently 1 in m_flags
\* flagsPacked   : record [count: 0..NumSlots, owned: BOOLEAN, alive: BOOLEAN]
\*   .count  = MASK_CNT  (number of CURRENTLY-CLAIMED bits — owner pushes
\*             onto TLS freelist do NOT decrement; cross-flush bit-clear
\*             that brings a word to zero does)
\*   .owned  = BIT_OWNED set (TRUE while owner thread alive)
\*   .alive  = chunk object still exists.  Cleared by Reclaim.  Used
\*             only by the bug-hunt invariants — does NOT gate further
\*             actions in the model (we want TLC to find a counterexample
\*             where actions occur AFTER alive=FALSE).
\*
\* localOldv[t] : per-thread snapshot of m_flags taken at A_ReadFlags
\* localBit[t]  : per-thread chosen bit (post A_PickBit)
\* owned[t]     : set of bits currently held by t (bitmap-set OR pushed
\*                to TLS freelist).  Bit is removed only on successful
\*                cross-flush CAS-clear that consumes one of t's slots.
\* tlsFree[t]   : per-thread freelist (subset of owned[t]) — slots
\*                owner has logically freed but bit still set.
\* crossBatch[t]: bag of bits t has cross-enqueued, awaiting flush.
\*                Each entry is a bit position (the slot's owner is
\*                implicit: the chunk's owner is the only owner in
\*                this 1-chunk model, so any entry is destined to
\*                the same chunk).
\* opsLeft[t]   : remaining op budget (decrements on alloc_done).
\* reclaimed    : sequence of thread ids that executed X_Reclaim.
\*                Used by Inv_AtMostOneReclaim.

VARIABLES
    mflags,
    flagsPacked,
    pc,
    localOldv,
    localBit,
    localExitWasReleaser,  \* TRUE iff this thread's atomicFetchAnd(~BIT_OWNED)
                           \*   observed `old & ~BIT_OWNED == 0` — captured
                           \*   AT THE ATOMIC OP, not re-read later.  Matches
                           \*   the C++ pattern `old = atomicFetchAnd(...);
                           \*   newv = old & ~BIT_OWNED; if(newv == 0) ...`
    localFlushWasReleaser, \* TRUE iff this thread's atomicDecAndTest
                           \*   returned true (= word became 0 AT THE DEC).
                           \*   Matches `if(atomicDecAndTest(&m_flags_packed))
                           \*   i_am_releaser = true;`
    owned,
    tlsFree,
    crossBatch,
    opsLeft,
    reclaimed

vars == << mflags, flagsPacked, pc, localOldv, localBit,
           localExitWasReleaser, localFlushWasReleaser,
           owned, tlsFree, crossBatch, opsLeft, reclaimed >>

(***************************************************************************
 * Initial state.
 *
 *   - chunk fresh: mflags empty, flagsPacked = [count|->0, owned|->TRUE,
 *     alive|->TRUE].  Models the post-construction state at line 656 in
 *     allocator.cpp: `m_flags_packed = BIT_OWNED`.
 *   - all threads idle.
 ***************************************************************************)
Init ==
    /\ mflags = {}
    /\ flagsPacked = [count |-> 0, owned |-> TRUE, alive |-> TRUE]
    /\ pc = [t \in Threads |-> "idle"]
    /\ localOldv = [t \in Threads |-> {}]
    /\ localBit = [t \in Threads |-> Null]
    /\ localExitWasReleaser = [t \in Threads |-> FALSE]
    /\ localFlushWasReleaser = [t \in Threads |-> FALSE]
    /\ owned = [t \in Threads |-> {}]
    /\ tlsFree = [t \in Threads |-> {}]
    /\ crossBatch = [t \in Threads |-> {}]
    /\ opsLeft = [t \in Threads |-> MaxOps]
    /\ reclaimed = << >>

(***************************************************************************
 * ALLOCATE PATH
 ***************************************************************************)

A_ReadFlags(t) ==
    /\ t = OwnerThread              \* only owner allocates from this chunk
    /\ pc[t] = "idle"
    /\ opsLeft[t] > 0
    /\ flagsPacked.alive            \* don't start a new alloc on a dead chunk
    /\ pc' = [pc EXCEPT ![t] = "alloc_read"]
    /\ localOldv' = [localOldv EXCEPT ![t] = mflags]
    /\ UNCHANGED << mflags, flagsPacked, localBit, owned, tlsFree,
                    crossBatch, opsLeft, reclaimed,
                    localExitWasReleaser, localFlushWasReleaser >>

A_PickBit(t) ==
    /\ t = OwnerThread
    /\ pc[t] = "alloc_read"
    \* find_training_zeros: pick any slot not in oldv.  Choose any
    \* (TLC explores all picks).  If none free in our snapshot, retry
    \* (= goto idle, treated as a failed allocate).
    /\ \/ /\ \E s \in Slots \ localOldv[t]:
              localBit' = [localBit EXCEPT ![t] = s]
          /\ pc' = [pc EXCEPT ![t] = "alloc_pick"]
       \/ /\ Slots \ localOldv[t] = {}
          /\ pc' = [pc EXCEPT ![t] = "idle"]
          /\ localBit' = localBit
    /\ UNCHANGED << mflags, flagsPacked, localOldv, owned, tlsFree,
                    crossBatch, opsLeft, reclaimed,
                    localExitWasReleaser, localFlushWasReleaser >>

A_CAS(t) ==
    /\ t = OwnerThread
    /\ pc[t] = "alloc_pick"
    /\ \/ \* CAS succeeds.  C++ CAS operates on ONE m_flags word at a
          \* time — so success means the chosen slot's word is unchanged
          \* between A_ReadFlags and now, AND the chosen bit is still
          \* free in that word.  Other words may have changed (other
          \* threads acted on them) — those changes are NOT a CAS-fail
          \* condition.  This is the key cross-word interleaving the
          \* spec must capture.
          LET w == WordOf(localBit[t]) IN
            /\ SlotsInWord(mflags, w) = SlotsInWord(localOldv[t], w)
            /\ localBit[t] \notin mflags
            /\ mflags' = mflags \cup {localBit[t]}
            /\ owned' = [owned EXCEPT ![t] = @ \cup {localBit[t]}]
            /\ pc' = [pc EXCEPT ![t] = "alloc_bump"]
            /\ UNCHANGED << flagsPacked >>
       \/ \* CAS fails: chosen word changed OR bit got taken — retry.
          LET w == WordOf(localBit[t]) IN
            /\ \/ SlotsInWord(mflags, w) # SlotsInWord(localOldv[t], w)
               \/ localBit[t] \in mflags
            /\ pc' = [pc EXCEPT ![t] = "alloc_read"]
            /\ UNCHANGED << mflags, flagsPacked, owned >>
    /\ UNCHANGED << localOldv, localBit, tlsFree,
                    crossBatch, opsLeft, reclaimed,
                    localExitWasReleaser, localFlushWasReleaser >>

\* Phase 5j allocate_pooled (line 884):
\*   if(oldv == 0) atomicInc(&this->m_flags_packed);
\* This is the SECOND atomic op on m_flags_packed, separate from the
\* CAS that published the bit.  The gap between A_CAS and A_BumpCount
\* is the suspected race window: between them, the word IS bit-set but
\* the count has NOT been bumped — observers (owner_release pre-check
\* + AND) can see count=0 with a live bit.
A_BumpCount(t) ==
    /\ t = OwnerThread
    /\ pc[t] = "alloc_bump"
    \* C++: `if(oldv == 0) atomicInc(&this->m_flags_packed);`
    \* `oldv == 0` here is the CAS'd WORD's old value being zero — not
    \* the whole bitmap.  So inc fires iff the chosen slot's word was
    \* empty in localOldv (= snapshot at A_ReadFlags).
    \* This is the SEPARATE atomic op; observers (owner_release pre-
    \* check on m_flags_packed) can see the word non-empty but the
    \* COUNT still missing the +1 in the window between A_CAS (bit
    \* set in mflags) and A_BumpCount (count bumped).
    /\ LET w == WordOf(localBit[t]) IN
        \/ /\ SlotsInWord(localOldv[t], w) = {}    \* CAS was 0 → non-zero
           /\ flagsPacked' = [flagsPacked EXCEPT
                                 !.count = flagsPacked.count + 1]
        \/ /\ SlotsInWord(localOldv[t], w) # {}    \* added to already-set word
           /\ UNCHANGED flagsPacked
    /\ pc' = [pc EXCEPT ![t] = "alloc_done"]
    /\ UNCHANGED << mflags, localOldv, localBit, owned, tlsFree,
                    crossBatch, opsLeft, reclaimed,
                    localExitWasReleaser, localFlushWasReleaser >>

A_Done(t) ==
    /\ t = OwnerThread
    /\ pc[t] = "alloc_done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ opsLeft' = [opsLeft EXCEPT ![t] = @ - 1]
    /\ localBit' = [localBit EXCEPT ![t] = Null]
    /\ UNCHANGED << mflags, flagsPacked, localOldv, owned, tlsFree,
                    crossBatch, reclaimed,
                    localExitWasReleaser, localFlushWasReleaser >>

(***************************************************************************
 * FREE PATHS — owner-side and cross-side.
 *
 * The C++ implementation routes owner-side free into a per-thread
 * freelist (no bitmap touch until batch-drain) and cross-side free
 * into CrossDeallocBatch which is periodically flushed via
 * batch_return_to_bitmap.  Both ultimately CAS-clear the bit.
 *
 * Model: owner push and cross enqueue are no-bitmap-touch.  Only
 * flush (cross or drain) clears the bit.  We compress drain into the
 * same FlushBit action — its semantics are identical (CAS-clear the
 * bit, dec count, check releaser).
 ***************************************************************************)

\* Owner pushes a bit onto its TLS freelist.  Bitmap unchanged.  This
\* simulates an owner-thread `delete[]` that lands in the per-thread
\* freelist instead of immediately CAS-clearing.
F_OwnerPush(t) ==
    /\ t = OwnerThread
    /\ pc[t] = "idle"
    /\ \E b \in owned[t] \ tlsFree[t]:
        /\ tlsFree' = [tlsFree EXCEPT ![t] = @ \cup {b}]
        /\ pc' = [pc EXCEPT ![t] = "free_owner_push"]
    /\ UNCHANGED << mflags, flagsPacked, localOldv, localBit,
                    owned, crossBatch, opsLeft, reclaimed,
                    localExitWasReleaser, localFlushWasReleaser >>

F_OwnerPushDone(t) ==
    /\ t = OwnerThread
    /\ pc[t] = "free_owner_push"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED << mflags, flagsPacked, localOldv, localBit,
                    owned, tlsFree, crossBatch, opsLeft, reclaimed,
                    localExitWasReleaser, localFlushWasReleaser >>

\* Thread u enqueues a cross-free of one of thread t's slots.  In the
\* model "any slot from any other thread" can be cross-freed; this is
\* upper-bounding what the real test does (worker u dequeues a slot
\* originally allocated by worker t).
F_CrossEnqueue(u) ==
    /\ u # OwnerThread                  \* only non-owners cross-free
    /\ pc[u] = "idle"
    /\ \E t \in Threads \ {u}:
        \E b \in (owned[t] \ tlsFree[t]) \ crossBatch[u]:
            /\ crossBatch' = [crossBatch EXCEPT ![u] = @ \cup {b}]
            /\ \* Transfer ownership: the slot leaves t's "owned"
               \* (it's being handed to u for cross-freeing).
               \* In the real allocator the slot leaves the user's
               \* hands at the moment of `delete[]` — the bitmap bit
               \* still says claimed, but the slot is no longer in
               \* use by anyone.  We model that by removing from
               \* owned[t] when entering the cross-batch.
               owned' = [owned EXCEPT ![t] = @ \ {b}]
            /\ pc' = [pc EXCEPT ![u] = "free_cross_enq"]
    /\ UNCHANGED << mflags, flagsPacked, localOldv, localBit,
                    tlsFree, opsLeft, reclaimed,
                    localExitWasReleaser, localFlushWasReleaser >>

F_CrossEnqueueDone(u) ==
    /\ u # OwnerThread
    /\ pc[u] = "free_cross_enq"
    /\ pc' = [pc EXCEPT ![u] = "idle"]
    /\ UNCHANGED << mflags, flagsPacked, localOldv, localBit,
                    owned, tlsFree, crossBatch, opsLeft, reclaimed,
                    localExitWasReleaser, localFlushWasReleaser >>

\* Flush one bit from u's crossBatch: CAS-clear it on m_flags, dec
\* MASK_CNT.  Model the CAS as always-succeeding here (single-word
\* model; CAS can only spuriously fail in superfine F_Flush variants
\* — keep simple for now).  This matches `batch_return_to_bitmap`'s
\* OnClearFn at line 1057-1066 of allocator.cpp.
F_FlushBit(u) ==
    /\ u # OwnerThread
    /\ pc[u] = "idle"
    /\ \E s \in crossBatch[u]:
        /\ s \in mflags          \* sanity: must still be set
        \* C++ (line 1057-1066): CAS-clears the bit; on word-becomes-0
        \* (newv == 0 && oldv != 0) atomicDecAndTest m_flags_packed.
        \* "word-becomes-0" is THIS slot's word becoming empty after the
        \* clear, not the whole chunk.
        /\ mflags' = mflags \ {s}
        /\ LET w == WordOf(s) IN
            \/ \* This slot's word becomes empty → atomic dec count.
               \* Releaser iff dec brings count to 0 AND owned=FALSE.
               /\ SlotsInWord(mflags \ {s}, w) = {}
               /\ flagsPacked' = [flagsPacked EXCEPT
                                     !.count = flagsPacked.count - 1]
               /\ localFlushWasReleaser' = [localFlushWasReleaser EXCEPT
                     ![u] = (flagsPacked.count - 1 = 0)
                            /\ (~flagsPacked.owned)]
            \/ \* Word still has other bits set → no dec.
               /\ SlotsInWord(mflags \ {s}, w) # {}
               /\ UNCHANGED flagsPacked
               /\ localFlushWasReleaser' = [localFlushWasReleaser EXCEPT
                     ![u] = FALSE]
        /\ crossBatch' = [crossBatch EXCEPT ![u] = @ \ {s}]
        /\ pc' = [pc EXCEPT ![u] = "flush_check_rel"]
        /\ localBit' = [localBit EXCEPT ![u] = s]
    /\ UNCHANGED << localOldv, owned, tlsFree, opsLeft, reclaimed,
                    localExitWasReleaser >>

\* Post-dec releaser check.  Phase 5j: cross-thread freer is the
\* unique releaser iff THE dec brought m_flags_packed to 0
\* (= captured-at-the-atomic-op result, not a re-read).
F_FlushCheckRel(u) ==
    /\ u # OwnerThread
    /\ pc[u] = "flush_check_rel"
    /\ \/ /\ localFlushWasReleaser[u]
          /\ pc' = [pc EXCEPT ![u] = "reclaim"]
       \/ /\ ~localFlushWasReleaser[u]
          /\ pc' = [pc EXCEPT ![u] = "idle"]
    /\ localBit' = [localBit EXCEPT ![u] = Null]
    /\ UNCHANGED << mflags, flagsPacked, localOldv, owned,
                    tlsFree, crossBatch, opsLeft, reclaimed,
                    localExitWasReleaser, localFlushWasReleaser >>

(***************************************************************************
 * OWNER_RELEASE — Phase 4a, alive-owner empty-neighbour release.
 *
 * C++ (allocator.cpp::owner_release, lines 1916-1966):
 *
 *   if(dll_len <= LEAVE_VACANT_CHUNKS_PER_THREAD) return false;
 *   if((palloc->m_flags_packed & MASK_CNT) != 0) return false;  // pre-check
 *   uint32_t old = atomicFetchAnd(&palloc->m_flags_packed, ~BIT_OWNED);
 *   uint32_t newv = old & ~BIT_OWNED;
 *   if(newv != 0) return false;  // cross-thread brought a bit back
 *   return true;                 // owner is unique releaser
 *
 * Sister-session bit-state query shows premature release with BIT_OWNED
 * still set on victim slot — i.e. `owner_release` fires on a chunk that
 * still has live slots.  The pre-check + AND-then-newv==0 check is meant
 * to prevent that.  Question: is there an interleaving where this two-
 * step protocol misfires?  (Note: LEAVE_VACANT_CHUNKS floor is omitted
 * here as a worst-case — caller-side floor is orthogonal to the
 * MASK_CNT desync mechanism we're hunting.)
 *
 * The hypothesised window: pre-check sees count=0 (allocate path's
 * atomicInc not yet visible) → AND clears BIT_OWNED → newv=0 →
 * release.  In real C++ the atomicInc IS atomic so this should be
 * impossible — but Phase 5j abandoned BIT_RELEASED, so we re-verify.
 ***************************************************************************)

O_OwnerRelease(t) ==
    /\ t = OwnerThread
    /\ pc[t] = "idle"
    /\ flagsPacked.owned                    \* haven't exited
    /\ flagsPacked.count = 0                \* pre-check empty
    \* Atomic fetchAnd(~BIT_OWNED) — same semantics as owner_exit.
    \* localExitWasReleaser captured AT the atomic op.
    /\ flagsPacked' = [flagsPacked EXCEPT !.owned = FALSE]
    /\ localExitWasReleaser' = [localExitWasReleaser EXCEPT
          ![t] = (flagsPacked.count = 0)]
    /\ pc' = [pc EXCEPT ![t] = "owner_exit_check"]
    /\ UNCHANGED << mflags, localOldv, localBit, owned, tlsFree,
                    crossBatch, opsLeft, reclaimed,
                    localFlushWasReleaser >>

(***************************************************************************
 * OWNER EXIT — release_dll_chunks_for_thread.
 *
 * Models a single atomicFetchAnd(~BIT_OWNED).  If the resulting word
 * is 0 (count was 0), owner is the unique releaser → reclaim.
 * Otherwise leave to cross-side flush to detect dec-to-zero.
 *
 * The owner can exit at any time it's idle.  After exit it stays
 * "exited" and does not allocate further (= matches thread death).
 ***************************************************************************)

O_OwnerExit(t) ==
    /\ t = OwnerThread                  \* only owner can clear BIT_OWNED
    /\ pc[t] = "idle"
    /\ opsLeft[t] = 0                   \* don't exit until alloc budget done
    /\ flagsPacked.owned                \* haven't exited yet
    \* Atomic fetchAnd(~BIT_OWNED).  C++:
    \*   uint32_t old = atomicFetchAnd(&m_flags_packed, ~BIT_OWNED);
    \*   uint32_t newv = old & ~BIT_OWNED;
    \*   if(newv == 0) /* owner is unique releaser */
    \* Capture (newv == 0) AT the atomic op: count was 0 at the time
    \* of the AND.  Do NOT re-read in O_OwnerExitCheck.
    /\ flagsPacked' = [flagsPacked EXCEPT !.owned = FALSE]
    /\ localExitWasReleaser' = [localExitWasReleaser EXCEPT
          ![t] = (flagsPacked.count = 0)]
    /\ pc' = [pc EXCEPT ![t] = "owner_exit_check"]
    /\ UNCHANGED << mflags, localOldv, localBit, owned, tlsFree,
                    crossBatch, opsLeft, reclaimed,
                    localFlushWasReleaser >>

O_OwnerExitCheck(t) ==
    /\ t = OwnerThread
    /\ pc[t] = "owner_exit_check"
    /\ \/ /\ localExitWasReleaser[t]
          /\ pc' = [pc EXCEPT ![t] = "reclaim"]
       \/ /\ ~localExitWasReleaser[t]
          /\ pc' = [pc EXCEPT ![t] = "exited"]
    /\ UNCHANGED << mflags, flagsPacked, localOldv, localBit, owned,
                    tlsFree, crossBatch, opsLeft, reclaimed,
                    localExitWasReleaser, localFlushWasReleaser >>

(***************************************************************************
 * RECLAIM — delete + deallocate.
 *
 * Records the reclaiming thread.  If two threads ever reach reclaim
 * for the same chunk, Inv_AtMostOneReclaim fires — the original
 * double-free bug class.  Sets alive=FALSE for the use-after-reclaim
 * invariant.
 ***************************************************************************)

X_Reclaim(t) ==
    /\ pc[t] = "reclaim"
    /\ reclaimed' = Append(reclaimed, t)
    /\ flagsPacked' = [flagsPacked EXCEPT !.alive = FALSE]
    /\ pc' = [pc EXCEPT ![t] = "exited"]
    /\ UNCHANGED << mflags, localOldv, localBit, owned, tlsFree,
                    crossBatch, opsLeft,
                    localExitWasReleaser, localFlushWasReleaser >>

(***************************************************************************
 * Next-state relation.
 *
 * At each step one thread takes one microscopic action.  No fairness
 * — TLC explores all interleavings exhaustively.
 ***************************************************************************)

Next ==
    \E t \in Threads:
        \/ A_ReadFlags(t)
        \/ A_PickBit(t)
        \/ A_CAS(t)
        \/ A_BumpCount(t)
        \/ A_Done(t)
        \/ F_OwnerPush(t)
        \/ F_OwnerPushDone(t)
        \/ F_CrossEnqueue(t)
        \/ F_CrossEnqueueDone(t)
        \/ F_FlushBit(t)
        \/ F_FlushCheckRel(t)
        \/ O_OwnerRelease(t)
        \/ O_OwnerExit(t)
        \/ O_OwnerExitCheck(t)
        \/ X_Reclaim(t)

Spec == Init /\ [][Next]_vars

(***************************************************************************
 * Invariants — the bug-hunt safety properties.
 ***************************************************************************)

\* INV 1: at most one thread owns any given bit position.
\* THIS IS THE BUG WE ARE HUNTING.  If this fails, TLC's counterexample
\* shows the exact interleaving where two threads claimed the same slot.
Inv_NoDoubleClaim ==
    \A t1, t2 \in Threads:
        t1 # t2 => owned[t1] \cap owned[t2] = {}

\* INV 2: chunk is reclaimed at most once.
Inv_AtMostOneReclaim ==
    Len(reclaimed) <= 1

\* INV 3: after reclaim, no thread holds an owned bit on this chunk.
\* (Bits in tlsFree or crossBatch ARE allowed since the slot is
\*  conceptually returned even if bookkeeping lingers — but bits in
\*  `owned[t]` represent a live reference.)
Inv_NoUseAfterReclaim ==
    flagsPacked.alive \/ \A t \in Threads: owned[t] = {}

\* INV 4: flagsPacked.count matches what we'd compute from the chunk's
\* bitmap and in-flight ops.  Checked only when ALL threads are quiescent
\* (= no in-flight CAS or flush operations).  Approximate, but catches
\* gross accounting bugs.
Quiescent ==
    \A t \in Threads: pc[t] \in {"idle", "exited"}

Inv_FlagsPackedConsistency ==
    Quiescent => flagsPacked.count = NumNonZeroWords(mflags)

\* INV 5: a bit set in mflags must be in EXACTLY one of:
\* - some thread's owned[]
\* - some thread's tlsFree[]
\* - some thread's crossBatch[]
\* (= the bit accounts for exactly one slot reference somewhere)
Inv_BitmapAccountedFor ==
    Quiescent =>
        \A s \in mflags:
            \E t \in Threads:
                \/ s \in owned[t]
                \/ s \in tlsFree[t]
                \/ s \in crossBatch[t]

================================================================================
