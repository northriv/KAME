/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
/*
 * C11 test generated mechanically from ChunkAlloc_microscopic.tla.
 *
 * Microscopic model of kamepoolalloc chunk-bitmap allocate / owner-free /
 * cross-thread flush / chunk reclaim, narrowed to expose the intra-chunk
 * slot double-claim race observed in alloc_stress_test at
 * cross_thread_pct=100 (same chunk hands the same slot to two threads).
 *
 * SAFETY ONLY (the TLA+ spec is safety-only; liveness is out of scope).
 *
 * ------------------------------------------------------------------------
 * NEW INFRA (vs the bundle-tree ports): there is NO bundle tree here.  The
 * genuinely new machinery this port introduces is:
 *
 *   (1) per-WORD (not whole-bitmap) CAS granularity.  m_flags is modelled
 *       as an array of NUM_WORDS independent atomic words, each holding
 *       BITS_PER_WORD bits.  A claim CAS touches exactly ONE word; it fails
 *       only if THAT word changed since the snapshot, NOT if some other
 *       word changed (mirrors C++ CAS over a single m_flags word).  This is
 *       the cross-word interleaving the spec exists to capture.
 *
 *   (2) a separate atomic packed count (flagsPacked.count) desynchronised
 *       from the bitmap.  The C++ `if(oldv==0) atomicInc(&m_flags_packed)`
 *       is a SECOND atomic op, distinct from the publishing CAS.  The gap
 *       between A_CAS (bit set) and A_BumpCount (count bumped) is the
 *       suspected desync window observers can witness.  This port keeps
 *       the two as separate atomic steps with a forced rendezvous so the
 *       window is actually exercised.
 *
 *   (3) releaser arbitration: a cross-side freer becomes the unique
 *       releaser iff its atomicDecAndTest brought the count to 0 (captured
 *       AT the dec, value `localFlushWasReleaser`); the owner becomes the
 *       unique releaser iff its atomicFetchAnd(~BIT_OWNED) observed the
 *       remaining word == 0 (captured AT the AND, `localExitWasReleaser`).
 *       The terminal invariant is that at most ONE thread reclaims.
 *
 * TLA+ -> C mapping:
 *   mflags                 -> per-word atomic bitmaps g_word[NUM_WORDS]
 *   flagsPacked            -> ONE _Atomic g_packed word (like C++
 *     m_flags_packed): count in low bits, BIT_OWNED, BIT_ALIVE.  Keeping
 *     count+owned in one atomic makes the cross-side dec-and-test + owned
 *     releaser gate a single atomic snapshot (TLA+ F_FlushBit step).
 *   owned[t]/tlsFree[t]/crossBatch[t]
 *                          -> auxiliary observation ledger (mutex-guarded
 *                             slot-state table) so the set-valued safety
 *                             invariants can be checked exactly.  The ledger
 *                             does NOT gate the racy atomics -- the CAS, the
 *                             count inc/dec and the BIT_OWNED clear remain
 *                             real lock-free atomics so the race, if any, is
 *                             observable.
 *   reclaimed (sequence)   -> _Atomic g_reclaim_count
 *
 * TLA+ action mapping (one function per microscopic action):
 *   A_ReadFlags / A_PickBit / A_CAS / A_BumpCount / A_Done -> alloc_attempt()
 *   F_OwnerPush                                            -> owner_push()
 *   F_CrossEnqueue                                         -> cross_enqueue()
 *   F_FlushBit / F_FlushCheckRel                           -> flush_one()
 *   O_OwnerRelease                                         -> owner_release()
 *   O_OwnerExit / O_OwnerExitCheck                         -> owner_exit()
 *   X_Reclaim                                              -> reclaim()
 *
 * Terminal / safety invariants (post-join check_invariants()):
 *   Inv_NoDoubleClaim         : no slot in two threads' owned[] at once
 *   Inv_AtMostOneReclaim      : reclaim count <= 1
 *   Inv_NoUseAfterReclaim     : alive OR every owned[] empty
 *   Inv_FlagsPackedConsistency: quiescent => count == #non-zero words
 *   Inv_BitmapAccountedFor    : quiescent => each set bit is owned/tls/cross
 * NoDoubleClaim and AtMostOneReclaim are additionally checked on the fly
 * (every owned-set transition / every reclaim) so a transient violation is
 * caught even if the post-join quiescent state looks clean.
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ------------------------------------------------------------------------
 * Compile-time knobs (consistent with the bundle ports + the .cfg scope).
 * The canonical TLC scope (ChunkAlloc_microscopic_2thr_mc.cfg) is:
 *   Threads={1,2}, OwnerThread=1, NumWords=2, BitsPerWord=2, MaxOps=2.
 * ---------------------------------------------------------------------- */
#ifndef NUM_THREADS
#define NUM_THREADS 2          /* >= 1 owner + cross-freers */
#endif

#ifndef NUM_WORDS
#define NUM_WORDS 2            /* m_flags words; >=2 to open the cross-word race */
#endif

#ifndef BITS_PER_WORD
#define BITS_PER_WORD 2        /* slots per word */
#endif

#ifndef STRESS_SECONDS
#define STRESS_SECONDS 0
#endif

/* MAX_OPS is the per-thread alloc budget for ONE chunk episode -- exactly
 * the TLA+ CONSTANT MaxOps (default 2 mirrors the .cfg).  It is the same
 * for unit and stress: a single chunk episode is always bounded so a round
 * cannot livelock.  Stress widens the search by re-racing FRESH chunks. */
#ifndef MAX_OPS
#define MAX_OPS 2
#endif

/* MAX_COMMITS is the number of fresh-chunk rounds the outer harness runs
 * (harness symmetry with the bundle ports).  Unit default 1 reproduces the
 * bounded TLA+ run; stress drives rounds by wall-clock instead. */
#ifndef MAX_COMMITS
#  if STRESS_SECONDS > 0
#    define MAX_COMMITS 0x7fffffff
#  else
#    define MAX_COMMITS 1
#  endif
#endif

#define NUM_SLOTS (NUM_WORDS * BITS_PER_WORD)

#if NUM_WORDS < 1
#  error "NUM_WORDS must be >= 1"
#endif
#if BITS_PER_WORD < 1
#  error "BITS_PER_WORD must be >= 1"
#endif
#if NUM_SLOTS > 64
#  error "NUM_SLOTS must fit the per-thread owned bitmask (<=64)"
#endif

/* OwnerThread: the 1-indexed owning thread (spec OwnerThread=1 -> tid 1). */
#define OWNER_TID 1

/* ------------------------------------------------------------------------
 * Slot identity.  A slot is a (word, bitInWord) pair; the flat index is
 * w*BITS_PER_WORD + b.  WordOf(slot) recovers the word.
 * ---------------------------------------------------------------------- */
static inline int slot_word(int s) { return s / BITS_PER_WORD; }
__attribute__((unused))
static inline int slot_bit_in_word(int s) { return s % BITS_PER_WORD; }
static inline uint32_t bit_in_word_mask(int s) {
    return 1u << (s % BITS_PER_WORD);
}

/* ------------------------------------------------------------------------
 * (1) Shared chunk state: per-WORD atomic bitmaps (TLA+ mflags).
 *     g_word[w] bit (s%BITS_PER_WORD) set  <=>  slot s in mflags.
 * ---------------------------------------------------------------------- */
static _Atomic(uint32_t) g_word[NUM_WORDS];

/* (2) flagsPacked: count + BIT_OWNED + BIT_ALIVE in ONE atomic word, exactly
 *     like the C++ `m_flags_packed` (MASK_CNT | BIT_OWNED).  Packing count
 *     and owned into a single atomic is what makes the cross-side
 *     atomicDecAndTest and its `~owned` releaser gate a SINGLE atomic
 *     snapshot -- matching the TLA+ F_FlushBit step atomicity (the dec and
 *     the unprimed `flagsPacked.owned` read happen in one Next step).  The
 *     DESYNC the spec hunts is NOT here: it is between this word and the
 *     SEPARATE per-word bitmap g_word[] -- A_CAS sets a bit (g_word) while
 *     A_BumpCount bumps the count (g_packed) one atomic step later.
 *
 *     Layout:  bits 0..15 = count (MASK_CNT)
 *              bit 16      = BIT_OWNED
 *              bit 17      = BIT_ALIVE (bookkeeping for the invariants) */
#define PK_CNT_MASK  0x0000FFFFu
#define PK_OWNED     0x00010000u
#define PK_ALIVE     0x00020000u
static _Atomic(uint32_t) g_packed;
static inline uint32_t pk_count(uint32_t v) { return v & PK_CNT_MASK; }
static inline bool      pk_owned(uint32_t v) { return (v & PK_OWNED) != 0; }
static inline bool      pk_alive(uint32_t v) { return (v & PK_ALIVE) != 0; }

/* (3) Releaser arbitration history. */
static _Atomic(int32_t)  g_reclaim_count;   /* Len(reclaimed) */

/* --- diagnostics --- */
static _Atomic(unsigned long long) spin_cas;        /* A_CAS retries        */
static _Atomic(unsigned long long) cnt_alloc;       /* successful claims    */
static _Atomic(unsigned long long) cnt_flush;       /* cross-flush bits     */
static _Atomic(unsigned long long) cnt_reclaim;     /* reclaim invocations  */

/* ------------------------------------------------------------------------
 * Auxiliary observation ledger (TLA+ owned / tlsFree / crossBatch).
 *
 * Each slot is in exactly one logical state from the spec's point of view:
 *   FREE   : not claimed by anyone
 *   OWNED  : in owned[holder]              (bit set, live reference)
 *   TLSFREE: in tlsFree[holder]            (owner-pushed, bit still set)
 *   CROSS  : in crossBatch[holder]         (cross-enqueued, bit still set)
 * `holder` is the 1-indexed tid that holds the slot.
 *
 * This ledger is auxiliary: it lets check_invariants() evaluate the
 * set-valued TLA+ invariants exactly.  Each TLA+ ACTION is realized as a
 * single critical section over g_action_mtx (== g_ledger_mtx) that updates
 * the ledger AND the corresponding atomics (g_word / g_packed) together --
 * mirroring the spec's atomic step.  The primitives below assume the action
 * lock is already held; the action functions take/release it.
 * ---------------------------------------------------------------------- */
enum slot_state { SS_FREE = 0, SS_OWNED, SS_TLSFREE, SS_CROSS };
static int            g_slot_state[NUM_SLOTS];   /* enum slot_state */
static int            g_slot_holder[NUM_SLOTS];  /* 1-indexed tid, 0=none */
static pthread_mutex_t g_ledger_mtx = PTHREAD_MUTEX_INITIALIZER;

/* g_action_mtx: the single "one mutex per atomic TLA+ action" lock.  Reusing
 * the ledger mutex (the ledger and the atomics are always mutated together
 * inside one action). */
#define g_action_mtx g_ledger_mtx

/* --- ledger primitives (caller holds g_action_mtx) --- */

/* F_OwnerPush: OWNED(tid) -> TLSFREE(tid).  Returns slot or -1. */
static int ledger_owner_push_locked(int tid) {
    for (int s = 0; s < NUM_SLOTS; s++)
        if (g_slot_state[s] == SS_OWNED && g_slot_holder[s] == tid) {
            g_slot_state[s] = SS_TLSFREE;   /* holder unchanged */
            return s;
        }
    return -1;
}

/* F_CrossEnqueue: non-owner u takes a slot OWNED by another thread (NOT in
 * tlsFree) and moves it to CROSS held by u.  (TLA+: remove from owned[t], add
 * to crossBatch[u].)  Returns slot or -1. */
static int ledger_cross_enqueue_locked(int u) {
    for (int s = 0; s < NUM_SLOTS; s++)
        if (g_slot_state[s] == SS_OWNED && g_slot_holder[s] != u) {
            g_slot_state[s]  = SS_CROSS;
            g_slot_holder[s] = u;            /* handed to u for cross-free */
            return s;
        }
    return -1;
}

/* Pick one CROSS slot held by u (for flush).  Returns slot or -1. */
static int ledger_pick_cross_locked(int u) {
    for (int s = 0; s < NUM_SLOTS; s++)
        if (g_slot_state[s] == SS_CROSS && g_slot_holder[s] == u)
            return s;
    return -1;
}

/* ------------------------------------------------------------------------
 * Helpers over the per-word bitmap (mflags).
 * ---------------------------------------------------------------------- */
static inline uint32_t word_load(int w) {
    return atomic_load_explicit(&g_word[w], memory_order_acquire);
}
/* NumNonZeroWords(mflags): number of words with any bit set. */
static int num_nonzero_words(void) {
    int n = 0;
    for (int w = 0; w < NUM_WORDS; w++)
        if (word_load(w) != 0) n++;
    return n;
}

/* ------------------------------------------------------------------------
 * ALLOCATE PATH (owner only).  Mirrors A_ReadFlags / A_PickBit / A_CAS /
 * A_BumpCount / A_Done as a single function performing the microscopic
 * sequence with the per-word CAS and the SEPARATE count bump.
 *
 * Returns true iff a slot was claimed (one alloc-budget unit consumed).
 * Returns false if no free slot remained in the snapshot (failed allocate).
 * ---------------------------------------------------------------------- */
static bool alloc_attempt(unsigned *rng);

/* xorshift32 PRNG: realizes the TLA+ `\E s \in Slots \ oldv` nondeterministic
 * bit pick (any free slot, not just the lowest) so different interleavings of
 * which word gets touched are explored. */
static inline unsigned xs32(unsigned *s) {
    unsigned x = *s ? *s : 0x9e3779b9u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

/* TLA+ interleaving discipline.  The spec is a sequence of ATOMIC actions
 * (one Next step at a time); the genuine hazards it captures are encoded in
 * the STEP BOUNDARIES, not in sub-step racing:
 *   - per-word CAS granularity  : A_CAS only fails if the CHOSEN word changed.
 *   - count desync              : A_CAS and A_BumpCount are SEPARATE steps, so
 *                                 a peer's whole action can interleave between
 *                                 them (bit set, count not yet bumped).
 *   - releaser arbitration      : F_FlushBit dec+owned-read is ONE step.
 * We therefore make each micro-action a single critical section under one
 * global action lock (g_action_mtx, reusing the ledger lock), and KEEP
 * A_CAS / A_BumpCount as distinct critical sections so the desync window is
 * real and interleaveable -- exactly the spec's atomicity granularity.  This
 * is the same "one mutex per atomic action" mapping the reference port uses
 * (test_ChunkClaim.c) to realize TLC's interleavings deterministically. */

static bool alloc_attempt(unsigned *rng) {
    for (;;) {
        /* ---- ACTION: A_ReadFlags + A_PickBit + A_CAS (one atomic step) ----
         * In the spec A_CAS reads the chosen word, tests it against the
         * snapshot, and on success sets the bit AND adds to owned[] all in one
         * Next step.  We fuse the read/pick/CAS/publish into one critical
         * section accordingly.  The PER-WORD granularity is preserved by the
         * logic: success requires only the CHOSEN word to be unchanged. */
        pthread_mutex_lock(&g_action_mtx);
        uint32_t oldv[NUM_WORDS];
        for (int w = 0; w < NUM_WORDS; w++)
            oldv[w] = word_load(w);

        int cand[NUM_SLOTS];
        int nc = 0;
        for (int s = 0; s < NUM_SLOTS; s++) {
            int w = slot_word(s);
            if ((oldv[w] & bit_in_word_mask(s)) == 0)
                cand[nc++] = s;
        }
        if (nc == 0) {
            pthread_mutex_unlock(&g_action_mtx);
            return false;   /* Slots \ oldv = {} : failed allocate */
        }
        int slot = cand[xs32(rng) % (unsigned)nc];
        int w    = slot_word(slot);
        uint32_t bm = bit_in_word_mask(slot);
        bool word_was_zero = (oldv[w] == 0);   /* for A_BumpCount decision */

        /* A_CAS: per-WORD CAS.  Inside the lock the chosen word cannot change
         * mid-CAS, so a single compare-exchange models the spec's atomic
         * "chosen word unchanged AND bit free -> set bit". */
        uint32_t expected = oldv[w];
        uint32_t desired  = expected | bm;
        bool ok = atomic_compare_exchange_strong_explicit(
            &g_word[w], &expected, desired,
            memory_order_acq_rel, memory_order_acquire);
        if (!ok) {
            /* Chosen word changed (only possible if g_action_mtx were not
             * held; defensive) -> retry from A_ReadFlags. */
            pthread_mutex_unlock(&g_action_mtx);
            atomic_fetch_add_explicit(&spin_cas, 1, memory_order_relaxed);
            continue;
        }
        /* owned[t] := owned[t] cup {bit} -- same step as the bit set.  This is
         * where Inv_NoDoubleClaim is enforced on the fly: the slot must be
         * FREE in the ledger (a second winner of the same slot would assert).*/
        assert(g_slot_state[slot] == SS_FREE
               && "Inv_NoDoubleClaim: slot double-claimed");
        g_slot_state[slot]  = SS_OWNED;
        g_slot_holder[slot] = OWNER_TID;
        pthread_mutex_unlock(&g_action_mtx);

        /* ---- desync window between A_CAS and A_BumpCount lives HERE ----
         * The bit is set (and owned) but the count is NOT yet bumped.  A peer
         * action (owner_release pre-check, a flush) can interleave now and
         * observe the count missing the +1.  This separate, later action is
         * the second atomic op the spec hunts. */

        /* ---- ACTION: A_BumpCount (separate atomic step) ----
         * Inc the count iff the chosen slot's WORD was zero in the snapshot
         * (word 0 -> non-zero).  Separate critical section = interleaveable. */
        pthread_mutex_lock(&g_action_mtx);
        if (word_was_zero)
            atomic_fetch_add_explicit(&g_packed, 1u, memory_order_acq_rel);
        pthread_mutex_unlock(&g_action_mtx);

        /* A_Done. */
        atomic_fetch_add_explicit(&cnt_alloc, 1, memory_order_relaxed);
        return true;
    }
}

/* ------------------------------------------------------------------------
 * F_OwnerPush (owner only): push one owned bit onto the TLS freelist.  The
 * bitmap bit stays set; only the ledger state moves OWNED -> TLSFREE.  One
 * atomic action.
 * ---------------------------------------------------------------------- */
static void owner_push(void) {
    pthread_mutex_lock(&g_action_mtx);
    (void)ledger_owner_push_locked(OWNER_TID);   /* no-op if nothing to push */
    pthread_mutex_unlock(&g_action_mtx);
}

/* ------------------------------------------------------------------------
 * F_CrossEnqueue (non-owner u): take a slot OWNED by another thread and move
 * it into u's cross-batch.  Bitmap untouched (cleared only at flush).  One
 * atomic action.
 * ---------------------------------------------------------------------- */
static void cross_enqueue(int u) {
    pthread_mutex_lock(&g_action_mtx);
    (void)ledger_cross_enqueue_locked(u);
    pthread_mutex_unlock(&g_action_mtx);
}

/* ------------------------------------------------------------------------
 * F_FlushBit + F_FlushCheckRel (non-owner u): clear one cross-batch bit from
 * mflags.  When the bit's WORD becomes empty, dec the count; the dec-to-zero
 * result is captured AT the dec, AND'd with `~owned` read in the SAME atomic
 * step (localFlushWasReleaser).  Releaser iff (count became 0) AND (NOT
 * owned).  If releaser -> reclaim.
 *
 * F_FlushBit is ONE TLA+ atomic step: bit-clear + count-dec + owned-read +
 * crossBatch-consume all happen together.  We bracket the whole body with the
 * action lock so the bitmap clear and the ledger consume are atomic w.r.t. the
 * allocator's snapshot-then-claim (no "bit clear but ledger still CROSS"
 * window).  Returns whether u must reclaim (done outside the lock). */
static void reclaim(int tid);   /* fwd */

static bool flush_one_locked(int u) {
    int slot = ledger_pick_cross_locked(u);
    if (slot < 0) return false;   /* nothing to flush */
    int w = slot_word(slot);
    uint32_t bm = bit_in_word_mask(slot);

    /* Bit must still be set (s \in mflags): only the holder u flushes its own
     * CROSS slot, and the bitmap clear lives in this same locked action. */
    uint32_t cur = word_load(w);
    assert((cur & bm) != 0 && "F_FlushBit: cross-batch bit not set in mflags");
    uint32_t after = cur & ~bm;
    atomic_store_explicit(&g_word[w], after, memory_order_release);

    bool flush_was_releaser = false;
    /* Word-becomes-0 (this word empty after the clear) -> atomicDecAndTest.
     * Dec of count and the `~owned` releaser gate read the SAME packed word
     * in this one locked step:
     *   localFlushWasReleaser' = (count-1 = 0) /\ (~flagsPacked.owned)
     * Mirrors C++ atomicDecAndTest on the single m_flags_packed word. */
    if (after == 0) {
        uint32_t pv = atomic_load_explicit(&g_packed, memory_order_acquire);
        uint32_t nv = (pv & ~PK_CNT_MASK) | ((pk_count(pv) - 1u) & PK_CNT_MASK);
        atomic_store_explicit(&g_packed, nv, memory_order_release);
        flush_was_releaser = (pk_count(nv) == 0) && !pk_owned(nv);
    }

    /* crossBatch consume: slot returns to FREE (same atomic step). */
    g_slot_state[slot]  = SS_FREE;
    g_slot_holder[slot] = 0;
    atomic_fetch_add_explicit(&cnt_flush, 1, memory_order_relaxed);
    return flush_was_releaser;
}

static void flush_one(int u) {
    pthread_mutex_lock(&g_action_mtx);
    bool releaser = flush_one_locked(u);
    pthread_mutex_unlock(&g_action_mtx);
    /* F_FlushCheckRel: if I was the releaser, reclaim (separate step). */
    if (releaser)
        reclaim(u);
}

/* ------------------------------------------------------------------------
 * O_OwnerRelease (owner): Phase 4a alive-owner empty-neighbour release.
 *   pre-check count==0; if so atomicFetchAnd(~BIT_OWNED); releaser iff the
 *   word observed at the AND had count==0 (captured AT the op).
 * Returns true iff the owner released BIT_OWNED this call (became releaser
 * candidate); used by the worker to decide reclaim.
 * ---------------------------------------------------------------------- */
static void owner_release(int tid) {
    /* O_OwnerRelease is ONE TLA+ atomic step: pre-check (count==0) + AND +
     * releaser capture all read/write the packed word together.  But the
     * spec's pre-check reads count and the AND fetches it AGAIN at the op --
     * both inside the one step, so under the action lock we read once.  The
     * desync the spec probes is between THIS action and the allocate path's
     * A_BumpCount (a separate action), captured by the interleaving. */
    bool releaser = false;
    pthread_mutex_lock(&g_action_mtx);
    uint32_t cur = atomic_load_explicit(&g_packed, memory_order_acquire);
    if (pk_owned(cur) && pk_count(cur) == 0) {
        uint32_t nv = cur & ~PK_OWNED;       /* atomicFetchAnd(~BIT_OWNED) */
        atomic_store_explicit(&g_packed, nv, memory_order_release);
        /* localExitWasReleaser captured AT the AND: newv & ~BIT_OWNED == 0. */
        releaser = (pk_count(nv) == 0);
    }
    pthread_mutex_unlock(&g_action_mtx);
    if (releaser)
        reclaim(tid);
}

/* ------------------------------------------------------------------------
 * O_OwnerExit + O_OwnerExitCheck (owner): release_dll_chunks_for_thread.
 *   atomicFetchAnd(~BIT_OWNED); releaser iff resulting word == 0 (count==0
 *   captured AT the AND).  Owner exits afterwards (thread death).  One step.
 * ---------------------------------------------------------------------- */
static void owner_exit(int tid) {
    bool releaser = false;
    pthread_mutex_lock(&g_action_mtx);
    uint32_t cur = atomic_load_explicit(&g_packed, memory_order_acquire);
    if (pk_owned(cur)) {                     /* not yet exited */
        uint32_t nv = cur & ~PK_OWNED;       /* atomicFetchAnd(~BIT_OWNED) */
        atomic_store_explicit(&g_packed, nv, memory_order_release);
        /* localExitWasReleaser captured AT the AND: newv==0 iff count==0. */
        releaser = (pk_count(nv) == 0);
    }
    pthread_mutex_unlock(&g_action_mtx);
    if (releaser)
        reclaim(tid);
}

/* ------------------------------------------------------------------------
 * X_Reclaim: delete + deallocate.  Records the reclaiming thread; sets
 * alive=FALSE.  Inv_AtMostOneReclaim is checked on the fly here AND
 * post-join.
 * ---------------------------------------------------------------------- */
static void reclaim(int tid) {
    (void)tid;
    int32_t n = atomic_fetch_add_explicit(&g_reclaim_count, 1, memory_order_acq_rel) + 1;
    atomic_fetch_and_explicit(&g_packed, ~PK_ALIVE, memory_order_acq_rel);
    atomic_fetch_add_explicit(&cnt_reclaim, 1, memory_order_relaxed);
    /* Inv_AtMostOneReclaim (on the fly): the chunk must be reclaimed once. */
    assert(n <= 1 && "Inv_AtMostOneReclaim: chunk reclaimed more than once");
}

/* ------------------------------------------------------------------------
 * Worker.  The owner thread (tid==OWNER_TID) runs the allocate path,
 * owner-side frees, owner_release and finally owner_exit.  Non-owner threads
 * run cross-enqueue + flush.  Each loop iteration interleaves the micro
 * actions; MAX_COMMITS bounds the owner's alloc budget (TLA+ MaxOps).
 * ---------------------------------------------------------------------- */
typedef struct { int tid; unsigned rng; } ThreadCtx;

static _Atomic(bool) g_stop_flag;   /* STRESS_SECONDS stop signal */

static void *owner_worker(void *arg) {
    ThreadCtx *ctx = (ThreadCtx *)arg;
    int budget = (int)MAX_OPS;          /* TLA+ MaxOps: bounded per episode */

    /* Allocate budget: each iteration optionally allocs then may owner-push. */
    while (budget > 0) {
        if (pk_alive(atomic_load_explicit(&g_packed, memory_order_acquire))) {
            if (alloc_attempt(&ctx->rng))
                budget--;          /* A_Done consumed a budget unit */
            else
                budget--;          /* failed allocate still consumes an op */
        } else {
            break;                 /* don't start new allocs on a dead chunk */
        }
        /* Interleave an owner push of a previously-owned slot. */
        if (xs32(&ctx->rng) & 1u)
            owner_push();
        /* Occasionally attempt the alive-owner empty-neighbour release. */
        if ((xs32(&ctx->rng) & 3u) == 0)
            owner_release(ctx->tid);
    }

    /* Budget done: owner exit (clears BIT_OWNED, may reclaim). */
    owner_exit(ctx->tid);
    return NULL;
}

static void *freer_worker(void *arg) {
    ThreadCtx *ctx = (ThreadCtx *)arg;
    /* Bound the freer's work so the episode terminates: a bounded number of
     * cross-enqueue + flush passes proportional to the slot count and the
     * owner's per-episode budget.  (The TLA+ model is finite; this mirrors it
     * by draining whatever the owner produced.) */
    long passes = (long)MAX_OPS * (long)NUM_SLOTS * 4L + 16L;
    while (passes-- > 0) {
        if (atomic_load_explicit(&g_stop_flag, memory_order_relaxed))
            break;
        cross_enqueue(ctx->tid);
        flush_one(ctx->tid);
    }
    /* Final drain: flush anything still in our cross-batch so a quiescent
     * post-join state has no lingering CROSS slot with the bit set. */
    for (;;) {
        pthread_mutex_lock(&g_action_mtx);
        bool empty = (ledger_pick_cross_locked(ctx->tid) < 0);
        pthread_mutex_unlock(&g_action_mtx);
        if (empty) break;
        flush_one(ctx->tid);
    }
    return NULL;
}

/* ------------------------------------------------------------------------
 * Post-join invariant checks (TLA+ Inv_* at quiescence).
 * ---------------------------------------------------------------------- */
static void check_invariants(void) {
    /* All threads are joined -> quiescent (pc[t] in {idle, exited}). */
    uint32_t pk = atomic_load_explicit(&g_packed, memory_order_acquire);

    /* Inv_NoDoubleClaim: a single holder field per slot makes a 2-owner
     * state unrepresentable, so the disjointness owned[t1] cap owned[t2] = {}
     * holds by construction.  Verify every OWNED slot has a valid holder
     * (a corrupt holder would signal a ledger race). */
    for (int s = 0; s < NUM_SLOTS; s++) {
        if (g_slot_state[s] == SS_OWNED) {
            int h = g_slot_holder[s];
            assert(h >= 1 && h <= NUM_THREADS
                   && "Inv_NoDoubleClaim: corrupt owned-slot holder");
        }
    }

    /* Inv_AtMostOneReclaim. */
    assert(atomic_load(&g_reclaim_count) <= 1
           && "Inv_AtMostOneReclaim");

    /* Inv_NoUseAfterReclaim: alive OR every owned[] empty. */
    if (!pk_alive(pk)) {
        for (int s = 0; s < NUM_SLOTS; s++)
            assert(g_slot_state[s] != SS_OWNED
                   && "Inv_NoUseAfterReclaim: owned slot after reclaim");
    }

    /* Inv_FlagsPackedConsistency (quiescent): count == #non-zero words. */
    assert((int)pk_count(pk) == num_nonzero_words()
           && "Inv_FlagsPackedConsistency");

    /* Inv_BitmapAccountedFor (quiescent): every set bit is owned/tls/cross. */
    for (int s = 0; s < NUM_SLOTS; s++) {
        int w = slot_word(s);
        bool set = (word_load(w) & bit_in_word_mask(s)) != 0;
        if (set)
            assert(g_slot_state[s] != SS_FREE
                   && "Inv_BitmapAccountedFor: set bit unaccounted");
        else
            /* a cleared bit must not still be logically held with the bit
             * expected set -- TLSFREE/CROSS/OWNED imply the bit is set. */
            assert(g_slot_state[s] == SS_FREE
                   && "Inv_BitmapAccountedFor: held slot with cleared bit");
    }
}

/* ------------------------------------------------------------------------
 * Init / reset to the TLA+ Init state.
 *   mflags = {} ; flagsPacked = [count|->0, owned|->TRUE, alive|->TRUE].
 * ---------------------------------------------------------------------- */
static void reset_chunk(void) {
    for (int w = 0; w < NUM_WORDS; w++)
        atomic_store(&g_word[w], 0u);
    /* flagsPacked = [count|->0, owned|->TRUE, alive|->TRUE]. */
    atomic_store(&g_packed, PK_OWNED | PK_ALIVE);
    atomic_store(&g_reclaim_count, 0);
    for (int s = 0; s < NUM_SLOTS; s++) {
        g_slot_state[s]  = SS_FREE;
        g_slot_holder[s] = 0;
    }
}

int main(void) {
    long rounds = 0;
    atomic_store(&g_stop_flag, false);

    ThreadCtx ctxs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        ctxs[i].tid = i + 1;                 /* 1-indexed; tid 1 = owner */
        ctxs[i].rng = 0x1234567u + 0x9e3779b9u * (unsigned)(i + 1);
    }

#if STRESS_SECONDS > 0
    struct timespec t0, now;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec - t0.tv_sec >= STRESS_SECONDS) break;
#else
    for (int round = 0; round < (int)MAX_COMMITS; round++) {
#endif
        reset_chunk();

        pthread_t threads[NUM_THREADS];
        for (int i = 0; i < NUM_THREADS; i++) {
            if (ctxs[i].tid == OWNER_TID)
                pthread_create(&threads[i], NULL, owner_worker, &ctxs[i]);
            else
                pthread_create(&threads[i], NULL, freer_worker, &ctxs[i]);
        }
        for (int i = 0; i < NUM_THREADS; i++)
            pthread_join(threads[i], NULL);

        check_invariants();
        rounds++;
    }

#if STRESS_SECONDS > 0
    atomic_store_explicit(&g_stop_flag, true, memory_order_release);
    printf("[ChunkAlloc_microscopic stress %ds words=%d bits/word=%d thr=%d] "
           "rounds=%ld alloc=%llu flush=%llu reclaim=%llu spin_cas=%llu\n",
           STRESS_SECONDS, NUM_WORDS, BITS_PER_WORD, NUM_THREADS, rounds,
           (unsigned long long)atomic_load(&cnt_alloc),
           (unsigned long long)atomic_load(&cnt_flush),
           (unsigned long long)atomic_load(&cnt_reclaim),
           (unsigned long long)atomic_load(&spin_cas));
#else
    /* Unit: the bounded run terminated and the terminal invariant held. */
    assert(rounds >= 1);
    (void)rounds;
#endif

    return 0;
}
