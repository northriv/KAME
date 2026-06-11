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
 * C11 test generated mechanically from ChunkClaim.tla.
 *
 * Models kamepoolalloc's region->unit chunk-claim protocol: the
 * s_back_offset[] publication race fixed in
 * "fix back_offset speculative-write data race".
 *
 * Topology (hardcoded in the TLA+ spec):
 *   - One region of NUNITS = 4 contiguous 256 KiB "units".
 *   - Two allocator procs of DIFFERENT stride: p2 (CU=2), p4 (CU=4),
 *     racing on the SAME shared claim bitmap + back_offset table.
 *   - A chunk occupies CU contiguous, CU-aligned units; claiming is a CAS
 *     over those units' claim bits; s_back_offset[u] = (u - base) so a
 *     later lookup_chunk(addr-in-u) recovers the base via u - backoff[u].
 *
 * The bug (SPECULATIVE==1, pre-fix): s_back_offset[] is published BEFORE
 * the claim CAS.  A CAS loser (different stride) has already written its
 * speculative back_offset entries, clobbering the entries the CAS WINNER
 * legitimately owns; a later lookup_chunk() resolves the wrong base.  TLC
 * reports a BackoffConsistent (INV2) violation.
 *
 * The fix (SPECULATIVE==0, post-fix): back_offset is written only AFTER
 * the CAS wins.  The units are then exclusively owned, so no other proc
 * writes those entries; INV2 holds.
 *
 * TLA+ variable mapping:
 *   claim[u]        -> bit u of _Atomic(uint32_t) g_claim  (0=FREE, set=claimed)
 *                      g_owner[u] records WHICH proc owns u (for the invariant)
 *   backoff[u]      -> _Atomic(uint8_t) g_backoff[u]   (s_back_offset[])
 *   phase[p]        -> control flow of each worker thread
 *   myBase[p]       -> ThreadCtx.my_base (chosen base unit)
 *
 * TLA+ action mapping:
 *   ScanChoose(p)      -> scan_choose()  (pick a valid+free CU-aligned base)
 *   WriteBackoffPre(p) -> write_backoff()  (SPECULATIVE only, before CAS)
 *   CasPre(p)/CasPost(p) -> claim_cas()  (bitmap CAS; post-fix also writes
 *                        back_offset on success)
 *   GiveUp(p)          -> scan_choose() returning "no base" terminal
 *
 * Terminal/safety invariant (TLA+ BackoffConsistent / LookupResolvesOwner):
 *   for every claimed unit u, backoff[u] == u - myBase[owner(u)], i.e. the
 *   O(1) lookup u - backoff[u] resolves the owning chunk's base.  Encoded
 *   as a post-join assert() in check_invariants().
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* --- Protocol variant (TLA+ CONSTANT Speculative) ---
 *   1 = pre-fix  : publish back_offset speculatively BEFORE the claim CAS;
 *                  a CAS loser does NOT roll its writes back -> bug.
 *   0 = post-fix : write back_offset only AFTER winning the CAS. */
#ifndef SPECULATIVE
#define SPECULATIVE 0
#endif

/* --- Configuration --- */
/* Region topology is hardcoded in the spec; the two racing procs are the
 * two threads (CU=2 and CU=4).  NUM_THREADS is fixed at 2 by the spec
 * topology but kept as a knob for harness symmetry. */
#ifndef NUM_THREADS
#define NUM_THREADS 2
#endif

#ifndef STRESS_SECONDS
#define STRESS_SECONDS 0
#endif

/* Iteration budget: the TLA+ proc attempts ONE claim then terminates.
 * For the C stress harness we let each thread re-run the whole claim
 * episode MAX_COMMITS times (resetting the region each episode) to widen
 * the interleaving search; default 1 reproduces the bounded TLA+ run. */
#ifndef MAX_COMMITS
#  if STRESS_SECONDS > 0
#    define MAX_COMMITS 0x7fffffff
#  else
#    define MAX_COMMITS 1
#  endif
#endif

#define NUNITS 4
#define MAXCU  4

/* Proc identities: 0 = FREE sentinel, p2 and p4 own bits / back_offset. */
#define PROC_FREE 0
#define PROC_P2   1   /* stride CU=2 */
#define PROC_P4   2   /* stride CU=4 */

/* CUof[p] : stride of each proc (TLA+ CUof). */
static int cu_of(int proc) { return proc == PROC_P2 ? 2 : 4; }

/* --- Shared region state (TLA+ claim / backoff) --- */
/* g_claim : bit u set <=> unit u is claimed (TLA+ claim[u] != FREE).
 *           The whole region is one atomic word so a CU-range claim is a
 *           single CAS over the range's bits (mirrors the real OCC_MASK
 *           CAS over a row of claim bits). */
static _Atomic(uint32_t) g_claim;
/* g_owner[u] : which proc owns unit u (for the invariant; not in the
 *              real bitmap but recoverable there from per-chunk metadata).
 *              Written together with the claim bits inside the winning CAS
 *              episode, so it is consistent with g_claim. */
static _Atomic(uint8_t)  g_owner[NUNITS];
/* g_backoff[u] : s_back_offset[u]  (TLA+ backoff[u]). */
static _Atomic(uint8_t)  g_backoff[NUNITS];

/* my_base[proc] : the base unit each proc settled on for the CURRENT
 *                 episode (TLA+ myBase[p]).  Indexed by proc id. */
static _Atomic(uint8_t)  g_my_base[3];
/* g_claimed[proc] : did this proc win its CAS this episode? */
static _Atomic(bool)     g_claimed[3];

/* --- diagnostics --- */
static _Atomic(unsigned long long) spin_cas;

/* Portable spin barrier (macOS libc has no pthread_barrier_t).  Used to
 * realize the TLA+ step granularity: ScanChoose and WriteBackoffPre/Cas
 * are SEPARATE atomic steps, so an interleaving can place one proc's whole
 * write+CAS between another proc's scan and its own write.  Forcing both
 * procs to rendezvous at the "scanned" phase before any of them writes
 * deterministically realizes that worst-case interleaving (which is the
 * one TLC reports for the pre-fix bug); plain free-running threads almost
 * never hit the window. */
static _Atomic(unsigned) g_barrier_count;
static _Atomic(unsigned) g_barrier_gen;
static unsigned          g_barrier_n;   /* participants this round */

static void barrier_wait(void) {
    unsigned gen = atomic_load_explicit(&g_barrier_gen, memory_order_acquire);
    if (atomic_fetch_add_explicit(&g_barrier_count, 1, memory_order_acq_rel)
            + 1 == g_barrier_n) {
        atomic_store_explicit(&g_barrier_count, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_barrier_gen, 1, memory_order_acq_rel);
    } else {
        while (atomic_load_explicit(&g_barrier_gen, memory_order_acquire) == gen)
            ; /* spin */
    }
}

typedef struct { int proc; unsigned rng; } ThreadCtx;

/* xorshift32: per-thread PRNG to realize the TLA+ `\E b` nondeterministic
 * base choice (a proc may pick ANY valid free CU-aligned base, not just
 * the lowest).  p2 picking base 2 while p4 picks base 0 is what exposes
 * the cross-stride back_offset clobber. */
static inline unsigned xs32(unsigned *s) {
    unsigned x = *s ? *s : 0x9e3779b9u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

/* TLA+ Range(b,cu) bit-mask: bits [b, b+cu). */
static inline uint32_t range_mask(int base, int cu) {
    return ((uint32_t)((1u << cu) - 1u)) << base;
}

/* TLA+ ValidBase(p,b): b is CU-aligned and the chunk fits in the region. */
static inline bool valid_base(int cu, int base) {
    return (base % cu == 0) && (base + cu <= NUNITS);
}

/* ScanChoose(p): scan the CURRENT (possibly soon-stale) bitmap and pick a
 * valid, currently-free CU-aligned base.  Returns the chosen base, or -1
 * if no claimable run exists (TLA+ GiveUp).  Mirrors loading `v` and
 * finding a CHUNK_STRIDE-aligned zero run.  The TLA+ `\E b` choice is
 * nondeterministic among ALL valid free bases, so we collect candidates
 * and pick one at random (rng != NULL); a NULL rng falls back to the
 * lowest base (deterministic). */
static int scan_choose(int cu, unsigned *rng) {
    uint32_t v = atomic_load_explicit(&g_claim, memory_order_acquire);
    int cand[NUNITS];
    int n = 0;
    for (int b = 0; b < NUNITS; b++) {
        if (!valid_base(cu, b))
            continue;
        uint32_t m = range_mask(b, cu);
        if ((v & m) == 0)   /* RangeFree(b,cu): all units of [b,b+cu) free */
            cand[n++] = b;
    }
    if (n == 0)
        return -1;  /* GiveUp: no free run for this stride */
    if (rng == NULL)
        return cand[0];
    return cand[xs32(rng) % (unsigned)n];
}

/* WriteBackoffPre(p): publish s_back_offset[u] = u - base for the chunk's
 * units, BEFORE the claim CAS (pre-fix path only).  Per the model these
 * writes are NOT rolled back on CAS failure. */
static void write_backoff(int base, int cu) {
    for (int u = base; u < base + cu; u++)
        atomic_store_explicit(&g_backoff[u], (uint8_t)(u - base),
                              memory_order_release);
}

/* The claim CAS over the CU-range's bits (TLA+ CasPre / CasPost claim').
 * Returns true iff this proc won the range (all bits were 0 and are now
 * set to claimed).  On success also records ownership.  In the post-fix
 * variant the back_offset write is bundled into the won-CAS episode. */
static bool claim_cas(int proc, int base, int cu) {
    uint32_t m = range_mask(base, cu);
    uint32_t expected = atomic_load_explicit(&g_claim, memory_order_acquire);
    for (;;) {
        if ((expected & m) != 0)
            return false;  /* a unit got claimed since the scan: CAS fails */
        uint32_t desired = expected | m;
        if (atomic_compare_exchange_weak_explicit(
                &g_claim, &expected, desired,
                memory_order_acq_rel, memory_order_acquire)) {
            /* Won: the CU-range is now exclusively ours.  Publish owner
             * and (post-fix) back_offset for our units.  These writes are
             * uncontended -- no other proc's scan can pick these set bits. */
            for (int u = base; u < base + cu; u++)
                atomic_store_explicit(&g_owner[u], (uint8_t)proc,
                                      memory_order_release);
#if !SPECULATIVE
            write_backoff(base, cu);   /* CasPost: write only after winning */
#endif
            return true;
        }
        /* CAS lost a bit-race; `expected` reloaded -> re-test (a)bove.
         * This terminates: g_claim bits only ever go 0->1 within an
         * episode, so the loop makes progress or sees a conflicting bit. */
        atomic_fetch_add_explicit(&spin_cas, 1, memory_order_relaxed);
    }
}

/* One full claim episode for proc p (TLA+ idle -> ... -> claimed/done). */
static void run_episode(int proc, unsigned *rng) {
    int cu = cu_of(proc);

    /* ScanChoose / GiveUp */
    int base = scan_choose(cu, rng);
    if (base < 0) {
        atomic_store_explicit(&g_claimed[proc], false, memory_order_release);
        barrier_wait();   /* still participate so peers don't hang */
        return;           /* GiveUp: terminal "done" */
    }
    atomic_store_explicit(&g_my_base[proc], (uint8_t)base, memory_order_release);

    /* Rendezvous at the TLA+ "scanned" phase: every proc has chosen a base
     * from a fully-free view of the region BEFORE any write/CAS happens.
     * This is the interleaving that exposes the cross-stride clobber. */
    barrier_wait();

#if SPECULATIVE
    /* WriteBackoffPre: publish back_offset speculatively, before the CAS. */
    write_backoff(base, cu);
    /* CasPre: claim CAS AFTER the speculative write.  On failure the
     * speculative back_offset entries are deliberately NOT rolled back. */
    bool won = claim_cas(proc, base, cu);
#else
    /* CasPost: claim CAS, then (inside claim_cas on success) back_offset. */
    bool won = claim_cas(proc, base, cu);
#endif

    atomic_store_explicit(&g_claimed[proc], won, memory_order_release);
}

/* Each thread runs exactly ONE claim episode per round (mirrors the TLA+
 * proc: at most one claim, then terminal).  The number of rounds is driven
 * by main() (MAX_COMMITS for unit, wall-clock for stress) with a fresh
 * region reset between rounds. */
static void *worker(void *arg) {
    ThreadCtx *ctx = (ThreadCtx *)arg;
    run_episode(ctx->proc, &ctx->rng);
    return NULL;
}

/* TLA+ BackoffConsistent / LookupResolvesOwner:
 *   for every claimed unit u, backoff[u] == u - myBase[owner(u)],
 *   i.e. lookup_chunk's base = u - backoff[u] resolves the owner's chunk
 *   base.  This is exactly the property the speculative-publication bug
 *   violates. */
static void check_invariants(void) {
    uint32_t claim = atomic_load_explicit(&g_claim, memory_order_acquire);
    for (int u = 0; u < NUNITS; u++) {
        bool claimed = (claim & (1u << u)) != 0;
        if (!claimed)
            continue;  /* claim[u] = FREE: nothing required of backoff[u] */

        int owner = atomic_load_explicit(&g_owner[u], memory_order_acquire);
        assert(owner == PROC_P2 || owner == PROC_P4);  /* NoOverlap / TypeOK */

        int base    = atomic_load_explicit(&g_my_base[owner], memory_order_acquire);
        int backoff = atomic_load_explicit(&g_backoff[u], memory_order_acquire);

        /* BackoffConsistent: backoff[u] == u - myBase[owner]. */
        assert(backoff == u - base);
        /* LookupResolvesOwner (equivalent): u - backoff[u] == myBase[owner]. */
        assert(u - backoff == base);
    }
}

/* Reset the shared region to the TLA+ Init state for a fresh episode. */
static void reset_region(void) {
    atomic_store(&g_claim, 0u);
    for (int u = 0; u < NUNITS; u++) {
        atomic_store(&g_owner[u], (uint8_t)PROC_FREE);
        atomic_store(&g_backoff[u], 0u);
    }
    atomic_store(&g_my_base[PROC_P2], 0u);
    atomic_store(&g_my_base[PROC_P4], 0u);
    atomic_store(&g_claimed[PROC_P2], false);
    atomic_store(&g_claimed[PROC_P4], false);
    atomic_store(&g_barrier_count, 0u);
    atomic_store(&g_barrier_gen, 0u);
}

int main(void) {
    int episodes = 0;
    g_barrier_n = (unsigned)NUM_THREADS;   /* barrier participants per round */

    /* Persistent per-thread state: proc role (fixed) + PRNG seed (advances
     * across rounds so the `\E b` base choice varies, eventually hitting the
     * p2-base-2 / p4-base-0 interleaving that exposes the pre-fix bug). */
    ThreadCtx ctxs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        ctxs[i].proc = (i % 2 == 0) ? PROC_P2 : PROC_P4;
        ctxs[i].rng  = 0x1234567u + 0x9e3779b9u * (unsigned)(i + 1);
    }

#if STRESS_SECONDS > 0
    /* Stress: keep re-racing fresh regions until the wall-clock budget.
     * Each round resets the region, runs both procs concurrently, joins,
     * and checks the invariant. */
    struct timespec t0, now;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec - t0.tv_sec >= STRESS_SECONDS)
            break;
#else
    /* Unit: run MAX_COMMITS rounds (default 1), fresh region each round. */
    for (int round = 0; round < (int)MAX_COMMITS; round++) {
#endif
        reset_region();

        pthread_t threads[NUM_THREADS];
        /* Two racing procs of DIFFERENT stride (spec topology).  Extra
         * threads (NUM_THREADS>2) alternate the same two proc roles. */
        for (int i = 0; i < NUM_THREADS; i++)
            pthread_create(&threads[i], NULL, worker, &ctxs[i]);
        for (int i = 0; i < NUM_THREADS; i++)
            pthread_join(threads[i], NULL);

        check_invariants();
        episodes++;
    }

#if STRESS_SECONDS > 0
    printf("[ChunkClaim stress %ds SPECULATIVE=%d] rounds=%d spin_cas=%llu\n",
           STRESS_SECONDS, SPECULATIVE, episodes,
           (unsigned long long)atomic_load(&spin_cas));
#else
    /* Unit: the bounded run terminated and the terminal invariant held. */
    assert(episodes >= 1);
    (void)episodes;
#endif

    return 0;
}
