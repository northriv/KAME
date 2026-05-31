/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            LICENSE-APACHE-2.0 in this directory)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see LICENSE-GPL-2.0 in this directory).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
***************************************************************************/
/*
 * GenMC test — §25/§26 large-recycle cache: exclusive-ownership /
 * no-premature-release (use-after-free / double-free) safety under RC11.
 *
 * This is NOT an ABA test.  Pointer-ABA on a slot (B -> null -> B) is
 * benign here: the invariant "a pointer sits in a slot  <=>  that block
 * is FREE" means a stale popper's winning CAS only ever takes a block
 * that has since been re-freed, and the size/meta is read AFTER winning
 * the take (own-then-read, never a pre-ownership peek).  So tagging is
 * unnecessary and is not modelled.
 *
 * What we DO verify (the real question): the cache never lets a block be
 * released (munmap / madvise) while another thread still owns or
 * references it, and two threads never own the same block at once:
 *
 *   (A) Exclusive ownership — a block is taken by at most one thread per
 *       "appearance".  Modelled by an atomic g_inuse[blk] flipped 0->1
 *       on take with assert(prev==0): a second concurrent owner trips it.
 *   (B) No use-after-free — every meta read / use asserts g_live[blk].
 *       g_live is a PLAIN int, so GenMC also flags any *data race* on it,
 *       which is precisely what a broken happens-before chain (e.g. a
 *       block reachable from two slots, or a missing acquire) would
 *       produce.
 *   (C) Release only by the current owner — lrc_release() exchanges
 *       g_inuse 1->0 with assert(prev==1) and asserts g_live before
 *       clearing it.
 *
 * 1:1 mapping to kamepoolalloc/allocator.cpp (global L2 path only):
 *   recycle_pop_fit()  — lines ~4793: per-slot acquire-load, weak-CAS
 *                        take (acq_rel, fail -> NEXT slot, no inner retry),
 *                        own-then-read size, sz>=need -> return, else one
 *                        put-back CAS or lrc_release.
 *   recycle_push()     — lines ~4825: sloppy byte-cap check, then
 *                        first-empty-slot weak-CAS publish (acq_rel),
 *                        band-full -> return false (caller releases).
 *   deallocate path    — `if(!recycle_push(...)) large_va_raw_unmap(...)`.
 *
 * Faithful abstractions (do not affect the ownership/release protocol):
 *   - The per-thread L1 (no atomics) is omitted — it cannot create
 *     cross-thread ownership hazards; only the shared L2 can.
 *   - lrc_idx/lrc_band log-bucketing is collapsed to a fixed band that
 *     spans ALL slots (the worst case for contention — every thread
 *     races on the same slots).
 *   - `kind` (CHUNK vs MMAP) is fixed; the chunk/mmap band clamp is irrelevant
 *     to ownership safety.
 *   - Pointers are modelled as small ints (blk+1 in a slot; 0 == empty),
 *     since the cache CASes raw nullptr/pointer values — it does not use
 *     pointer tag bits (unlike atomic_smart_ptr).
 *   - The byte cap is set high so the cap-release path is not the focus;
 *     the too-small put-back-fail RELEASE path (the "勝手に解放" path) IS
 *     exercised by giving the two blocks different sizes.
 *
 * Run:  make run     (needs GenMC, see Makefile)
 *       make smoke   (concrete gcc run — build/logic sanity, no model check)
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>

#define NSLOT 2
#define NBLK  2

/* g_lrc_slot[]: each entry stores blk+1, or 0 for empty (== nullptr). */
static _Atomic(int)  g_slot[NSLOT];
/* meta read by lrc_block_size(): block size, written at birth, stable
 * while the block is cached. */
static int           g_size[NBLK];
/* (B) PLAIN — 1 = mapped, 0 = released (munmap'd).  assert + GenMC race. */
static int           g_live[NBLK];
/* (A)/(C) double-ownership / release-owner detector. */
static _Atomic(int)  g_inuse[NBLK];
/* sloppy single-atomic byte accounting (faithful; not asserted). */
static _Atomic(long) g_bytes;
static const long    G_CAP = 1000;  /* high — no cap pressure in this test */

/* lrc_release(): release a block owned by ME.  Models large_va_raw_unmap /
 * recycle_release_chunk.  (C): must own; must still be live. */
static void lrc_release(int blk, int me) {
    (void)me;
    int prev = atomic_exchange_explicit(&g_inuse[blk], 0, memory_order_relaxed);
    assert(prev == 1);          /* (C) only the current owner releases */
    assert(g_live[blk]);        /* not already released (no double-free) */
    g_live[blk] = 0;            /* munmap — plain write, exclusive */
}

/* recycle_pop_fit(need): scan the (whole-pool) band, weak-CAS take, own,
 * read size; fit -> return blk, else put-back or release.  Returns blk>=0
 * on hit, -1 on miss.  me = thread id. */
static int recycle_pop_fit(int need, int me) {
    for (int s = 0; s < NSLOT; s++) {
        int v = atomic_load_explicit(&g_slot[s], memory_order_acquire);
        if (v == 0) continue;
        int exp = v;
        if (!atomic_compare_exchange_weak_explicit(
                &g_slot[s], &exp, 0,
                memory_order_acq_rel, memory_order_relaxed))
            continue;            /* weak: spurious / taken -> NEXT slot */
        int blk = v - 1;
        /* own it now (exclusive: it is out of every slot) */
        int prev = atomic_exchange_explicit(&g_inuse[blk], 1, memory_order_relaxed);
        assert(prev == 0);       /* (A) no concurrent second owner */
        assert(g_live[blk]);     /* (B) meta read of a still-mapped block */
        int sz = g_size[blk];    /* own-then-read (safe: owned + live) */
        atomic_fetch_sub_explicit(&g_bytes, sz, memory_order_relaxed);
        if (sz >= need)
            return blk;          /* hand to caller (user will use + free) */
        /* too small: one put-back attempt, else release */
        atomic_store_explicit(&g_inuse[blk], 0, memory_order_relaxed); /* relinquish before republish */
        int exp2 = 0;
        if (atomic_compare_exchange_weak_explicit(
                &g_slot[s], &exp2, blk + 1,
                memory_order_acq_rel, memory_order_relaxed)) {
            atomic_fetch_add_explicit(&g_bytes, sz, memory_order_relaxed); /* back in cache */
        } else {
            atomic_store_explicit(&g_inuse[blk], 1, memory_order_relaxed); /* re-own to release */
            lrc_release(blk, me);
        }
    }
    return -1;                   /* miss */
}

/* recycle_push(blk): cap-check, then publish into the first empty band
 * slot (weak-CAS).  Returns 1 if cached, 0 if the caller must release.
 * Precondition: ME owns blk. */
static int recycle_push(int blk, int me) {
    (void)me;
    if (atomic_load_explicit(&g_bytes, memory_order_relaxed) + g_size[blk] > G_CAP)
        return 0;                /* over cap -> caller releases */
    for (int s = 0; s < NSLOT; s++) {
        atomic_store_explicit(&g_inuse[blk], 0, memory_order_relaxed); /* relinquish before publish */
        int exp = 0;
        if (atomic_compare_exchange_weak_explicit(
                &g_slot[s], &exp, blk + 1,
                memory_order_acq_rel, memory_order_relaxed)) {
            atomic_fetch_add_explicit(&g_bytes, g_size[blk], memory_order_relaxed);
            return 1;            /* published; blk is free again */
        }
        atomic_store_explicit(&g_inuse[blk], 1, memory_order_relaxed); /* CAS failed -> reclaim, try next */
    }
    return 0;                    /* band full -> caller releases */
}

/* One alloc -> use -> free cycle, as a client of the cache would do. */
static void alloc_use_free(int need, int me) {
    int blk = recycle_pop_fit(need, me);
    if (blk < 0) return;                       /* cache miss (fresh mmap IRL) */
    assert(atomic_load_explicit(&g_inuse[blk], memory_order_relaxed) == 1); /* I hold it */
    assert(g_live[blk]);                       /* (B) using a mapped block */
    int use = g_size[blk]; (void)use;          /* "use" the block's memory */
    if (!recycle_push(blk, me))                /* free: cache it, or... */
        lrc_release(blk, me);                  /* ...release (deallocate path) */
}

static void *t_need2(void *a) { (void)a; alloc_use_free(2, 0); return NULL; }
static void *t_need1(void *a) { (void)a; alloc_use_free(1, 1); return NULL; }

int main(void) {
    /* Two blocks of different sizes, both initially cached (one per slot).
     * Different sizes so the need=2 thread popping the size-1 block hits
     * the too-small put-back / RELEASE path. */
    g_size[0] = 1; g_size[1] = 2;
    g_live[0] = 1; g_live[1] = 1;
    atomic_store_explicit(&g_inuse[0], 0, memory_order_relaxed);
    atomic_store_explicit(&g_inuse[1], 0, memory_order_relaxed);
    atomic_store_explicit(&g_slot[0], 1, memory_order_relaxed); /* B0 */
    atomic_store_explicit(&g_slot[1], 2, memory_order_relaxed); /* B1 */
    atomic_store_explicit(&g_bytes, 3, memory_order_relaxed);

    pthread_t a, b;
    pthread_create(&a, NULL, t_need2, NULL);
    pthread_create(&b, NULL, t_need1, NULL);
    pthread_join(a, NULL);
    pthread_join(b, NULL);

    /* Post-conditions (hold in EVERY interleaving):
     *  - no thread still owns a block (no leaked ownership),
     *  - a live block is cached in exactly the expected sense (reachable
     *    from a slot), and a released block is in no slot. */
    for (int blk = 0; blk < NBLK; blk++) {
        assert(atomic_load_explicit(&g_inuse[blk], memory_order_relaxed) == 0);
        int in_slot = (atomic_load_explicit(&g_slot[0], memory_order_relaxed) == blk + 1)
                    || (atomic_load_explicit(&g_slot[1], memory_order_relaxed) == blk + 1);
        if (g_live[blk]) assert(in_slot);    /* live -> not leaked */
        else             assert(!in_slot);   /* released -> unreachable */
    }
    return 0;
}
