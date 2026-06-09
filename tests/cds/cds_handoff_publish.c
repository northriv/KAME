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
 * GenMC test — the alloc→handoff→cross-free PUBLICATION of a slot's
 * non-atomic resolve metadata (back_offset + palloc identity).
 *
 * WHY THIS TEST EXISTS (read before editing)
 * ------------------------------------------
 * The shipped resolver (`resolve_chunk_from_slot` / the `deallocate`
 * hot path) reads a slot's chunk-resolution metadata with PLAIN /
 * relaxed loads:
 *     back_off = rmeta->back_offset[unit_idx];        // plain uint8 array
 *     palloc   = *(PoolAllocatorBase**)(chunk_base+8); // plain pointer
 * The allocating thread wrote those with PLAIN stores right after the
 * claim CAS — there is NO release/acquire on the back_offset/palloc
 * accesses themselves.  The code's stated justification (allocator.cpp,
 * "a single relaxed load … suffices") is that visibility *rides the
 * application's pointer handoff*: a thread can only `free(p)` a pointer
 * it legitimately received, and that hand-off (queue, shared var, …)
 * carries the release→acquire that publishes the claimer's writes.
 *
 * That assumption is exactly what this test pins under RC11.  It is the
 * one remaining weak-memory premise of the seqlock-FREE shipped resolver
 * (the cross-CLAIMER race is a separate concern, covered by ChunkClaim;
 * the reclaim+recycle race is closed structurally by the live-slot
 * invariant — see ChunkRecycle_threadepoch / the README).
 *
 * MODEL
 * -----
 *   g_back_offset, g_chunk_id : the slot's resolve metadata — PLAIN
 *       (non-atomic), faithful to `rmeta->back_offset[]` and the chunk
 *       header's palloc word.  Init to the 0 "stale / released" sentinels.
 *   g_handoff : the application's pointer-transfer channel (T1 → T2).
 *
 * Claimer (T1): write the metadata (plain), then publish the pointer.
 * Freer  (T2): receive the pointer; ONLY if received, resolve via the
 *       plain metadata.  INV: a freer that received the pointer reads the
 *       claimer's metadata (never the stale 0 sentinel).
 *
 * CONTROLS
 *   default              : handoff is release/acquire -> EXPECT No errors.
 *   -DNO_HANDOFF_SYNC    : handoff is relaxed (the premise violated) ->
 *       EXPECT a Non-atomic race on the plain metadata (the claimer's
 *       write races the freer's read with no happens-before).  This is
 *       the negative control proving the plain accesses are sound ONLY
 *       because the handoff synchronises — i.e. the premise is real and
 *       load-bearing.
 *
 * Run:  make run-handoff          (GenMC, positive — clean)
 *       make run-handoff-norel    (GenMC, negative — must race)
 *
 * Co-Authored-By: Claude <noreply@anthropic.com>
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

#define CORRECT_BACKOFF 3        /* the claimer's back_offset for this unit */
#define CORRECT_CHUNKID 7        /* the claimer's chunk/palloc identity      */
#define USER_PTR ((void *)0x1000)/* the pointer `p` handed T1 -> T2          */

static int               g_back_offset;  /* PLAIN: rmeta->back_offset[unit_idx] */
static int               g_chunk_id;     /* PLAIN: chunk header palloc identity */
static _Atomic(void *)   g_handoff;      /* the app's T1->T2 pointer channel    */

/* Allocating thread: claim the unit (abstracted), write its resolve
 * metadata with PLAIN stores, then publish `p` on the handoff channel. */
static void *claimer(void *a) {
    (void)a;
    g_back_offset = CORRECT_BACKOFF;   /* plain write, post-claim */
    g_chunk_id    = CORRECT_CHUNKID;   /* plain write */
#ifdef NO_HANDOFF_SYNC
    atomic_store_explicit(&g_handoff, USER_PTR, memory_order_relaxed);
#else
    atomic_store_explicit(&g_handoff, USER_PTR, memory_order_release);
#endif
    return NULL;
}

/* Freeing thread: a `free(p)` can only run on a pointer it RECEIVED.
 * Model that: read the handoff; only if we got `p` do we resolve it via
 * the plain metadata (the shipped relaxed/plain back_offset + palloc). */
static void *freer(void *a) {
    (void)a;
#ifdef NO_HANDOFF_SYNC
    void *p = atomic_load_explicit(&g_handoff, memory_order_relaxed);
#else
    void *p = atomic_load_explicit(&g_handoff, memory_order_acquire);
#endif
    if(p != USER_PTR) return NULL;     /* not handed off yet → no free happens */
    int bo = g_back_offset;            /* plain read of resolve metadata */
    int id = g_chunk_id;               /* plain read */
    assert(bo == CORRECT_BACKOFF);     /* INV: freer sees the claimer's publish */
    assert(id == CORRECT_CHUNKID);
    return NULL;
}

int main(void) {
    g_back_offset = 0;                  /* "stale / uninitialised" sentinel */
    g_chunk_id    = 0;                  /* "released" sentinel */
    atomic_store_explicit(&g_handoff, NULL, memory_order_relaxed);

    pthread_t t1, t2;
    pthread_create(&t1, NULL, claimer, NULL);
    pthread_create(&t2, NULL, freer,   NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    return 0;
}
