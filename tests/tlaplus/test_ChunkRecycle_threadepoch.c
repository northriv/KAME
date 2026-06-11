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
 * C11 test generated mechanically from ChunkRecycle_threadepoch.tla.
 *
 * Models kamepoolalloc's chunk-claim / lookup / recycle protocol with a
 * thread-local epoch + per-unit pair-atomic metadata + a seqlock-style
 * lookup re-read.  Successor to ChunkRecycle_microscopic (which proved the
 * existing protocol's bit-state -3 corruption requires this redesign).
 *
 * ============================================================
 *  NEW C11 INFRA introduced by this port (vs. test_ChunkClaim.c)
 * ============================================================
 *   - per-thread epoch generation: epoch == <<owner_tid, counter>> packed
 *     into a single uint64_t word, all-zero reserved as "released" (Null).
 *   - per-unit pair-atomic metadata `unitMeta` = <<backOff, epoch>> packed
 *     in ONE _Atomic(uint64_t) word, read & written as one atomic load /
 *     store (the proposed std::atomic<uint64_t> unitMeta).
 *   - SEQLOCK-style lookup: the deallocate/lookup path loads unitMeta once
 *     (D_LoadMeta), reads the chunk_header (D_LoadPal), then RE-READS
 *     unitMeta (D_Resolve) and only accepts if the epoch is unchanged
 *     across the two reads (closes the "reclaim slipped in mid-lookup"
 *     window a single pair-atomic read cannot detect).
 *   - ABA-wrap provocation knob ALLOW_WRAP: when set the thread-local
 *     counter wraps back to 1, deliberately re-using an <<tid, counter>>
 *     value so the design's identity check can be fooled (= the model's
 *     AllowWrap=TRUE branch, which TLC reports DOES break).  Default 0:
 *     the bounded "counter is wide enough to never wrap" case which TLC
 *     verified over 3.7 M states with NO error.
 *
 * The slot-pool / Lamport-serial idiom of the sibling ports is NOT needed
 * here: identity is carried by the epoch pair, not a packet pool.
 *
 * ============================================================
 *  TLA+ -> C variable mapping
 * ============================================================
 *   epoch[u], backOff[u]   -> packed g_unit_meta[u]  (ONE atomic word)
 *   palloc[u]              -> g_palloc[u]      (chunk_header.palloc; 0=rel)
 *   chunkEpoch[u]          -> g_chunk_epoch[u] (chunk_header.epoch)
 *   chunkSpan[u]           -> g_chunk_span[u]
 *   localEpoch[t]          -> ThreadCtx.local_epoch (TLS counter, 1-based)
 *   allocGenOf[u]          -> g_alloc_gen[u]   (ghost: live-slot epoch, 0=none)
 *   pc[t], dUnit, dExpect, -> the deallocate worker's local control flow +
 *     dMetaEpoch, dBase,      captured registers (function locals in the
 *     dPalloc, dChunkEpoch    lookup_and_free path)
 *   staleRead              -> g_stale_read (the bug flag; terminal invariant)
 *
 * ============================================================
 *  TLA+ action -> C function mapping
 * ============================================================
 *   A_Allocate(t)  -> allocate_chunk()   (claim a free unit run, stamp epoch)
 *   R_Reclaim(t)   -> reclaim_chunk()    (free a chunk with no live slot)
 *   D_Start ..     -> lookup_and_free()  (the 4-step seqlock lookup; D_Resolve
 *     D_Resolve       sets g_stale_read on an accepted-but-wrong identity)
 *
 * Atomicity granularity: the TLA+ A_Allocate / R_Reclaim each collapse their
 * (claim, meta-write, header-write) sub-steps into ONE atomic action (the
 * spec's stated modelling concession: "the interleaving point we care about
 * is between a concurrent lookup's D_LoadMeta and D_LoadPal").  We mirror
 * that exactly: allocate / reclaim hold a per-region spin-lock so each runs
 * as one indivisible step, while the lookup's four D_* steps run lock-free
 * and CAN be interleaved by a concurrent allocate/reclaim between any two of
 * them.  There are therefore no MODE_COARSE/FINE/SUPERFINE knobs (the source
 * model has no such modes).
 *
 * Terminal / safety invariant (TLA+ Inv_NoStaleRead): a lookup NEVER accepts
 * a stale chunk -> g_stale_read stays false.  Plus Inv_LiveSlotConsistent and
 * Inv_BackOffInRange checked over the final region.  Encoded as post-join
 * assert() in check_invariants().
 *
 * SAFETY ONLY (the spec is safety-only; no liveness claim).
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* --- Configuration knobs (consistent with sibling ports) --- */
#ifndef NUM_THREADS
#define NUM_THREADS 2
#endif

#ifndef STRESS_SECONDS
#define STRESS_SECONDS 0
#endif

/* Iteration budget: each thread runs MAX_COMMITS allocate/lookup/reclaim
 * episodes.  Default 1 for the bounded unit run, huge for stress. */
#ifndef MAX_COMMITS
#  if STRESS_SECONDS > 0
#    define MAX_COMMITS 0x7fffffff
#  else
#    define MAX_COMMITS 1
#  endif
#endif

/* TLA+ CONSTANT NumUnits (recommend 2). */
#ifndef NUM_UNITS
#define NUM_UNITS 2
#endif

/* TLA+ CONSTANT MaxLocalEpoch: per-thread counter ceiling.  In the
 * no-wrap (default) case a thread stops allocating once its counter
 * exceeds this; in the wrap case it wraps back to 1 to re-use values. */
#ifndef MAX_LOCAL_EPOCH
#  if STRESS_SECONDS > 0
#    define MAX_LOCAL_EPOCH 0x7fffffffffULL   /* effectively never (40-bit) */
#  else
#    define MAX_LOCAL_EPOCH 5ULL
#  endif
#endif

/* TLA+ CONSTANT AllowWrap.  0 (default): bounded no-wrap; models "the
 * counter is wide enough that no in-flight lookup ever sees a re-used
 * epoch".  TLC: NO error.  1: deliberately wrap the counter to provoke
 * ABA; TLC reports a stale-read violation -> justifies the wide-counter
 * requirement.  Default 0 so the unit test asserts the SOUND design. */
#ifndef ALLOW_WRAP
#define ALLOW_WRAP 0
#endif

/* TLA+ CONSTANT SinglePayout.  FALSE (default) lets two threads be
 * mid-dealloc of the SAME unit (the worst case the epoch design must
 * survive).  TRUE forbids it. */
#ifndef SINGLE_PAYOUT
#define SINGLE_PAYOUT 0
#endif

/* ============================================================================
 * Epoch packing.  epoch == <<owner_tid, counter>>, all-zero == Null.
 *   bits  0..15  : owner_tid (16-bit)
 *   bits 16..55  : counter   (40-bit, thread-local serial)
 * A unit_meta word additionally carries backOff in the high byte:
 *   bits 56..63  : backOff   (8-bit)
 * ============================================================================ */
typedef uint64_t Epoch;
#define EPOCH_NULL ((Epoch)0)

static inline Epoch make_epoch(uint32_t tid, uint64_t counter) {
    /* tid in [1, 0xFFFF], counter in [1, 2^40-1] -> never all-zero. */
    return ((Epoch)(tid & 0xFFFFu)) | (((Epoch)(counter & 0xFFFFFFFFFFULL)) << 16);
}
__attribute__((unused))
static inline uint32_t epoch_tid(Epoch e)     { return (uint32_t)(e & 0xFFFFu); }
__attribute__((unused))
static inline uint64_t epoch_counter(Epoch e) { return (e >> 16) & 0xFFFFFFFFFFULL; }

/* unit_meta = backOff(8) | epoch(56).  Read/written as ONE atomic word. */
static inline uint64_t meta_pack(uint8_t back_off, Epoch e) {
    return ((uint64_t)e & 0x00FFFFFFFFFFFFFFULL) | ((uint64_t)back_off << 56);
}
static inline Epoch   meta_epoch(uint64_t m)   { return m & 0x00FFFFFFFFFFFFFFULL; }
static inline uint8_t meta_backoff(uint64_t m) { return (uint8_t)(m >> 56); }

/* ============================================================================
 * Shared per-region state.
 * ============================================================================ */
/* g_unit_meta[u] : the proposed pair-atomic unitMeta (backOff + epoch),
 *                  read & written as a single atomic word. */
static _Atomic(uint64_t) g_unit_meta[NUM_UNITS];
/* chunk_header fields, live on the chunk's base unit. */
static _Atomic(Epoch)    g_palloc[NUM_UNITS];       /* 0 == released */
static _Atomic(Epoch)    g_chunk_epoch[NUM_UNITS];
static _Atomic(uint32_t) g_chunk_span[NUM_UNITS];
/* g_alloc_gen[u] : ghost -- which epoch a LIVE slot at base u was allocated
 *                  under (0 == no live slot).  TLA+ allocGenOf. */
static _Atomic(Epoch)    g_alloc_gen[NUM_UNITS];

/* The single bug flag (TLA+ staleRead).  Never set under the sound design. */
static _Atomic(bool)     g_stale_read;

/* --- diagnostics --- */
static _Atomic(unsigned long long) cnt_alloc;
static _Atomic(unsigned long long) cnt_reclaim;
static _Atomic(unsigned long long) cnt_lookup;
static _Atomic(unsigned long long) cnt_lookup_accept;
static _Atomic(unsigned long long) cnt_lookup_foreign;
static _Atomic(unsigned long long) cnt_seqlock_reject;

static _Atomic(bool) g_stop;

/* ----------------------------------------------------------------------------
 * Region mutation lock.  TLA+ A_Allocate / R_Reclaim each fire as ONE atomic
 * action (claim+meta+header bundled).  We serialise the two MUTATORS under a
 * spin-lock so each is indivisible w.r.t. the other; the LOOKUP path takes
 * the lock only around its individual D_* sub-steps so a mutator can slip in
 * between a lookup's loads -- exactly the interleaving the spec targets.
 * --------------------------------------------------------------------------*/
static pthread_mutex_t g_mut_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    uint32_t tid;          /* 1-indexed (0 reserved for Null epoch) */
    uint64_t local_epoch;  /* TLA+ localEpoch[t]; 1-based */
    unsigned rng;          /* per-thread PRNG for nondeterministic choices */
} ThreadCtx;

static inline unsigned xs32(unsigned *s) {
    unsigned x = *s ? *s : 0x9e3779b9u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

/* TLA+ NextLocal(t): wrap to 1 only when AllowWrap; else just increment. */
static inline uint64_t next_local(uint64_t cur) {
#if ALLOW_WRAP
    return (cur >= (uint64_t)MAX_LOCAL_EPOCH) ? 1ULL : cur + 1ULL;
#else
    return cur + 1ULL;
#endif
}

/* TLA+ SpanFree(u, span): the run [u, u+span) fits and every unit is free
 * (epoch == Null).  Caller holds g_mut_lock. */
static bool span_free(int u, int span) {
    if (u + span > NUM_UNITS)
        return false;
    for (int k = 0; k < span; k++) {
        uint64_t m = atomic_load_explicit(&g_unit_meta[u + k], memory_order_relaxed);
        if (meta_epoch(m) != EPOCH_NULL)
            return false;
    }
    return true;
}

/* ============================================================================
 * A_Allocate(t).  Claim a free unit run, stamp the per-unit epoch pair and
 * the chunk_header at the base.  ONE atomic action (under g_mut_lock).
 *
 * Real-code order folded in here (CAS claim -> write unitMeta -> write
 * chunk_header).  Returns true if a chunk was allocated.
 * ============================================================================ */
static bool allocate_chunk(ThreadCtx *ctx) {
    /* TLA+ guard: localEpoch[t] <= MaxLocalEpoch \/ AllowWrap. */
#if !ALLOW_WRAP
    if (ctx->local_epoch > (uint64_t)MAX_LOCAL_EPOCH)
        return false;
#endif

    pthread_mutex_lock(&g_mut_lock);

    /* \E u \in Units, span \in {1,2}: SpanFree(u, span).  Try a randomised
     * order of (base, span) candidates to widen the interleaving search. */
    int start = (int)(xs32(&ctx->rng) % (unsigned)NUM_UNITS);
    int chosen_u = -1, chosen_span = 0;
    for (int i = 0; i < NUM_UNITS && chosen_u < 0; i++) {
        int u = (start + i) % NUM_UNITS;
        int span_first = (xs32(&ctx->rng) & 1u) ? 2 : 1;
        for (int sidx = 0; sidx < 2; sidx++) {
            int span = (sidx == 0) ? span_first : (3 - span_first);
            if (span_free(u, span)) { chosen_u = u; chosen_span = span; break; }
        }
    }
    if (chosen_u < 0) {
        pthread_mutex_unlock(&g_mut_lock);
        return false;   /* no free run; A_Allocate not enabled */
    }

    Epoch ev = make_epoch(ctx->tid, ctx->local_epoch);
    int u = chosen_u, span = chosen_span;

    /* backOff' / epoch' : per-unit unitMeta for the claimed run. */
    for (int k = 0; k < span; k++) {
        uint64_t m = meta_pack((uint8_t)k, ev);   /* backOff = (u+k) - u = k */
        atomic_store_explicit(&g_unit_meta[u + k], m, memory_order_release);
    }
    /* chunk_header at the base: palloc + chunkEpoch + span. */
    atomic_store_explicit(&g_palloc[u], ev, memory_order_release);
    atomic_store_explicit(&g_chunk_epoch[u], ev, memory_order_release);
    atomic_store_explicit(&g_chunk_span[u], (uint32_t)span, memory_order_relaxed);
    /* ghost allocGenOf at the base. */
    atomic_store_explicit(&g_alloc_gen[u], ev, memory_order_release);

    /* localEpoch' = NextLocal(t). */
    ctx->local_epoch = next_local(ctx->local_epoch);

    pthread_mutex_unlock(&g_mut_lock);
    atomic_fetch_add_explicit(&cnt_alloc, 1, memory_order_relaxed);
    return true;
}

/* ============================================================================
 * R_Reclaim(t).  Reclaim a chunk whose base has NO live slot
 * (allocGenOf == Null) but is still claimed (palloc != Null).  ONE atomic
 * action (under g_mut_lock).  This is what re-frees the units so a later
 * allocate can recycle them under a NEW epoch -- the ABA setup.
 * ============================================================================ */
static bool reclaim_chunk(ThreadCtx *ctx) {
    pthread_mutex_lock(&g_mut_lock);

    int start = (int)(xs32(&ctx->rng) % (unsigned)NUM_UNITS);
    int u = -1;
    for (int i = 0; i < NUM_UNITS; i++) {
        int b = (start + i) % NUM_UNITS;
        /* ReclaimableBase(b): palloc != Null AND allocGenOf == Null. */
        if (atomic_load_explicit(&g_palloc[b], memory_order_acquire) != EPOCH_NULL &&
            atomic_load_explicit(&g_alloc_gen[b], memory_order_acquire) == EPOCH_NULL) {
            u = b; break;
        }
    }
    if (u < 0) {
        pthread_mutex_unlock(&g_mut_lock);
        return false;   /* nothing reclaimable */
    }

    int span = (int)atomic_load_explicit(&g_chunk_span[u], memory_order_relaxed);
    if (span < 1) span = 1;

    /* palloc' = Null, chunkEpoch' = Null, chunkSpan' = 0. */
    atomic_store_explicit(&g_palloc[u], EPOCH_NULL, memory_order_release);
    atomic_store_explicit(&g_chunk_epoch[u], EPOCH_NULL, memory_order_release);
    atomic_store_explicit(&g_chunk_span[u], 0u, memory_order_relaxed);
    /* backOff' = 0, epoch' = Null for the run (clears unitMeta -> recyclable). */
    for (int k = 0; k < span && (u + k) < NUM_UNITS; k++) {
        atomic_store_explicit(&g_unit_meta[u + k], meta_pack(0, EPOCH_NULL),
                              memory_order_release);
    }

    pthread_mutex_unlock(&g_mut_lock);
    atomic_fetch_add_explicit(&cnt_reclaim, 1, memory_order_relaxed);
    return true;
}

/* ============================================================================
 * DEALLOCATE(p) lookup -- the 4-step seqlock path (TLA+ D_Start .. D_Resolve).
 *
 * Picks a unit currently holding a LIVE slot (allocGenOf != Null), captures
 * the epoch at that moment (dExpectEpoch), logically frees the slot, then
 * runs the lookup:
 *   D_LoadMeta : pair-atomic load of unitMeta -> dMetaEpoch, dMetaBackOff,
 *                dBase = unit - backOff.
 *   D_LoadPal  : load chunk_header[base].palloc + .epoch.
 *   D_Resolve  : SEQLOCK re-read of unitMeta; accept only if the epoch is
 *                unchanged across the lookup; then the PRIMARY identity check
 *                dMetaEpoch == dExpectEpoch.  If accepted-as-original but the
 *                resolved chunk_header actually points elsewhere -> staleRead.
 *
 * Each individual D_* sub-step takes g_mut_lock only for its own atomic
 * read (so a mutator can fire BETWEEN sub-steps), mirroring the spec's
 * separate D_* edges.  Returns the accepted palloc, or EPOCH_NULL (foreign).
 * ============================================================================ */
static Epoch lookup_and_free(ThreadCtx *ctx) {
    /* ---- D_Start : pick a unit with a live slot owned by anyone ----
     * (SinglePayout is enforced via the live-slot claim: the first thread to
     * grab the slot zeroes allocGenOf atomically, so a second cannot grab the
     * SAME live slot -- matching TLA+ "no other thread mid-dealloc of u" when
     * SINGLE_PAYOUT.  When !SINGLE_PAYOUT we still allow a concurrent thread
     * to be mid-lookup of the same unit because each captures its own
     * dExpectEpoch register.) */
    int picked = -1;
    Epoch d_expect = EPOCH_NULL;

    pthread_mutex_lock(&g_mut_lock);
    {
        int start = (int)(xs32(&ctx->rng) % (unsigned)NUM_UNITS);
        for (int i = 0; i < NUM_UNITS; i++) {
            int u = (start + i) % NUM_UNITS;
            Epoch ag = atomic_load_explicit(&g_alloc_gen[u], memory_order_acquire);
            if (ag != EPOCH_NULL) {
                picked = u;
                /* dExpectEpoch[t] = epoch[u] at this moment. */
                d_expect = meta_epoch(atomic_load_explicit(&g_unit_meta[u],
                                                           memory_order_acquire));
#if SINGLE_PAYOUT
                /* Logically free the slot HERE so no peer can re-pick it
                 * (TLA+ allocGenOf' = Null in D_Start). */
                atomic_store_explicit(&g_alloc_gen[u], EPOCH_NULL,
                                      memory_order_release);
#else
                /* Still free it (TLA+ frees in D_Start unconditionally); the
                 * !SINGLE_PAYOUT relaxation is that two threads may BOTH have
                 * picked the unit before either freed -- realised naturally
                 * by the concurrent lock-free interleaving of the D_* steps
                 * below across threads. */
                atomic_store_explicit(&g_alloc_gen[u], EPOCH_NULL,
                                      memory_order_release);
#endif
                break;
            }
        }
    }
    pthread_mutex_unlock(&g_mut_lock);

    if (picked < 0)
        return EPOCH_NULL;   /* D_Start not enabled: no live slot */
    atomic_fetch_add_explicit(&cnt_lookup, 1, memory_order_relaxed);

    int u = picked;

    /* ---- D_LoadMeta : single pair-atomic load of unitMeta ---- */
    uint64_t meta1 = atomic_load_explicit(&g_unit_meta[u], memory_order_acquire);
    Epoch   d_meta_epoch   = meta_epoch(meta1);
    uint8_t d_meta_backoff = meta_backoff(meta1);
    int     d_base         = u - (int)d_meta_backoff;
    if (d_base < 0 || d_base >= NUM_UNITS) {
        /* backOff resolved out of range -> treat as foreign (defensive;
         * Inv_BackOffInRange guarantees this never happens for a non-Null
         * epoch, but a Null/released read can carry a stale backOff). */
        atomic_fetch_add_explicit(&cnt_lookup_foreign, 1, memory_order_relaxed);
        return EPOCH_NULL;
    }

    /* A concurrent allocate/reclaim may fire HERE (between D_LoadMeta and
     * D_LoadPal) -- exactly the spec's interleaving point. */

    /* ---- D_LoadPal : load chunk_header[base].palloc + .epoch ---- */
    Epoch d_palloc      = atomic_load_explicit(&g_palloc[d_base], memory_order_acquire);
    Epoch d_chunk_epoch = atomic_load_explicit(&g_chunk_epoch[d_base], memory_order_acquire);
    atomic_thread_fence(memory_order_acquire);

    /* ---- D_Resolve : SEQLOCK re-read + identity decision ---- */
    uint64_t meta2 = atomic_load_explicit(&g_unit_meta[u], memory_order_relaxed);
    Epoch re_read_epoch = meta_epoch(meta2);

    /* Branch 1: released / foreign / seqlock mismatch -> reject (safe). */
    if (d_meta_epoch == EPOCH_NULL ||
        d_palloc == EPOCH_NULL ||
        d_meta_epoch != re_read_epoch) {
        if (d_meta_epoch != re_read_epoch)
            atomic_fetch_add_explicit(&cnt_seqlock_reject, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&cnt_lookup_foreign, 1, memory_order_relaxed);
        return EPOCH_NULL;
    }

    /* Branch 2: identity ACCEPTED (meta unchanged across the lookup). */
    /* d_meta_epoch == re_read_epoch (seqlock OK) && both non-Null here. */
    if (d_meta_epoch == d_expect) {
        /* PRIMARY check says "still the chunk p was allocated from".  Under
         * correct epoch uniqueness the chunk_header MUST agree.  If it does
         * NOT, the lookup has accepted a stale chunk -> THE BUG. */
        if (d_chunk_epoch == d_expect && d_palloc == d_expect) {
            /* safe */
        } else {
            atomic_store_explicit(&g_stale_read, true, memory_order_release);
        }
    }
    /* else: dMetaEpoch != dExpectEpoch -> rejected as foreign (safe). */

    if (d_meta_epoch == d_expect)
        atomic_fetch_add_explicit(&cnt_lookup_accept, 1, memory_order_relaxed);
    else
        atomic_fetch_add_explicit(&cnt_lookup_foreign, 1, memory_order_relaxed);

    return d_palloc;
}

/* ============================================================================
 * Worker.  Each thread runs MAX_COMMITS episodes; each episode fires a random
 * one of the three enabled actions (allocate / reclaim / lookup-free),
 * mirroring the TLA+ Next disjunction \E t: A_Allocate \/ R_Reclaim \/ D_*.
 * Lookups run lock-free so their D_* sub-steps interleave with peers'
 * mutators -- the whole point of the model.
 * ============================================================================ */
static void *worker(void *arg) {
    ThreadCtx ctx = *(ThreadCtx *)arg;

    for (uint32_t i = 0; i < (uint32_t)MAX_COMMITS; i++) {
        if (atomic_load_explicit(&g_stop, memory_order_relaxed))
            break;
        /* Bias toward allocate so units keep churning (alloc -> lookup/free
         * -> reclaim -> re-alloc is the cycle that exposes ABA under wrap). */
        unsigned r = xs32(&ctx.rng) % 4u;
        switch (r) {
            case 0:
            case 1:
                if (!allocate_chunk(&ctx)) { (void)lookup_and_free(&ctx); }
                break;
            case 2:
                (void)lookup_and_free(&ctx);
                break;
            default:
                if (!reclaim_chunk(&ctx)) { (void)allocate_chunk(&ctx); }
                break;
        }
    }
    return NULL;
}

/* ============================================================================
 * Post-join invariant checks (TLA+ Inv_NoStaleRead / Inv_LiveSlotConsistent
 * / Inv_BackOffInRange over the final region state).
 * ============================================================================ */
static void check_invariants(void) {
    /* Inv_NoStaleRead -- the primary terminal/safety invariant. */
    assert(atomic_load_explicit(&g_stale_read, memory_order_acquire) == false);

    for (int u = 0; u < NUM_UNITS; u++) {
        uint64_t m   = atomic_load_explicit(&g_unit_meta[u], memory_order_acquire);
        Epoch   e    = meta_epoch(m);
        uint8_t bo   = meta_backoff(m);
        Epoch   ag   = atomic_load_explicit(&g_alloc_gen[u], memory_order_acquire);
        Epoch   pal  = atomic_load_explicit(&g_palloc[u], memory_order_acquire);
        Epoch   cep  = atomic_load_explicit(&g_chunk_epoch[u], memory_order_acquire);

        /* Inv_LiveSlotConsistent: a unit with a live slot at its base is
         * claimed and its base header is consistent.  allocGenOf is only ever
         * set on a BASE unit (backOff 0), so check there. */
        if (ag != EPOCH_NULL) {
            assert(e != EPOCH_NULL);
            assert(pal != EPOCH_NULL);
            assert(cep == e);
        }

        /* Inv_BackOffInRange: a claimed unit's backOff resolves in range. */
        if (e != EPOCH_NULL) {
            int base = u - (int)bo;
            assert(base >= 0 && base < NUM_UNITS);
            (void)base;
        }
        (void)pal; (void)cep;   /* silence -Wunused under -DNDEBUG */
    }
}

/* Reset the shared region to the TLA+ Init state for a fresh batch. */
static void reset_region(void) {
    for (int u = 0; u < NUM_UNITS; u++) {
        atomic_store(&g_unit_meta[u], meta_pack(0, EPOCH_NULL));
        atomic_store(&g_palloc[u], EPOCH_NULL);
        atomic_store(&g_chunk_epoch[u], EPOCH_NULL);
        atomic_store(&g_chunk_span[u], 0u);
        atomic_store(&g_alloc_gen[u], EPOCH_NULL);
    }
    atomic_store(&g_stale_read, false);
    atomic_store(&cnt_alloc, 0ULL);
    atomic_store(&cnt_reclaim, 0ULL);
    atomic_store(&cnt_lookup, 0ULL);
    atomic_store(&cnt_lookup_accept, 0ULL);
    atomic_store(&cnt_lookup_foreign, 0ULL);
    atomic_store(&cnt_seqlock_reject, 0ULL);
    atomic_store(&g_stop, false);
}

int main(void) {
    reset_region();

    pthread_t threads[NUM_THREADS];
    ThreadCtx ctxs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        ctxs[i].tid         = (uint32_t)(i + 1);   /* 1-indexed; 0 == Null */
        ctxs[i].local_epoch = 1ULL;                /* TLA+ Init: counter = 1 */
        ctxs[i].rng         = 0x1234567u + 0x9e3779b9u * (unsigned)(i + 1);
    }

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, worker, &ctxs[i]);

#if STRESS_SECONDS > 0
    struct timespec ts = { .tv_sec = STRESS_SECONDS, .tv_nsec = 0 };
    nanosleep(&ts, NULL);
    atomic_store_explicit(&g_stop, true, memory_order_release);
#endif

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    check_invariants();

#if STRESS_SECONDS > 0
    printf("[ChunkRecycle_threadepoch stress %ds NUM_THREADS=%d NUM_UNITS=%d "
           "ALLOW_WRAP=%d SINGLE_PAYOUT=%d]\n",
           STRESS_SECONDS, NUM_THREADS, NUM_UNITS, ALLOW_WRAP, SINGLE_PAYOUT);
    printf("  alloc=%llu reclaim=%llu lookup=%llu (accept=%llu foreign=%llu "
           "seqlock_reject=%llu)\n",
           (unsigned long long)atomic_load(&cnt_alloc),
           (unsigned long long)atomic_load(&cnt_reclaim),
           (unsigned long long)atomic_load(&cnt_lookup),
           (unsigned long long)atomic_load(&cnt_lookup_accept),
           (unsigned long long)atomic_load(&cnt_lookup_foreign),
           (unsigned long long)atomic_load(&cnt_seqlock_reject));
    printf("  stale_read = %s\n",
           atomic_load(&g_stale_read) ? "TRUE (BUG -- expected only under ALLOW_WRAP)"
                                      : "false (Inv_NoStaleRead held)");
#else
    /* Unit: the bounded run terminated and the terminal invariant held. */
    (void)0;
#endif

    return 0;
}
