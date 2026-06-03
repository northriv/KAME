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
 * GenMC test — §13/§20 thread-exit DLL walk vs concurrent cross-thread
 * last-slot free (INV-6 exactly-one-releaser, INV-21 cached-next no-UAF).
 *
 * This is the protocol the separately-reported "CrossDeallocBatch flush"
 * crash investigation flagged as comment-only (INVARIANTS.md INV-21).  The
 * owning thread, on exit, walks its private chunk DLL: for each chunk it
 * caches `next = c->m_dll_next` BEFORE clearing BIT_OWNED, because once
 * BIT_OWNED is clear a cross-thread freer returning the chunk's last slot
 * can immediately release (free) the chunk — after which `c->m_dll_next`
 * is freed memory.
 *
 * `m_flags_packed`: bit 31 = BIT_OWNED, bits 0..30 = MASK_CNT (live slots).
 * Release identification WITHOUT a separate BIT_RELEASED bit:
 *   - owner exit : atomicFetchAnd(~BIT_OWNED); if result == 0 (MASK_CNT was
 *                  0) the owner is the unique releaser.
 *   - cross free : atomicDecAndTest (sub 1, == 0?); true ⇒ cross is the
 *                  unique releaser (only possible once BIT_OWNED is clear).
 * The two never both fire (proof sketch in allocator.cpp); GenMC checks it.
 *
 * Memory orders: the real code uses `__sync_fetch_and_and` /
 * `__sync_sub_and_fetch` (seq_cst full barriers).  Modelled here as acq_rel
 * — strictly weaker, so race-freedom under acq_rel implies it under the
 * stronger order the code actually emits.
 *
 * Scenario: chunk c1 starts (BIT_OWNED=1, MASK_CNT=1) — owner alive, one
 * slot held by another thread — and is linked c1 -> c2 in the owner's DLL.
 *   Thread OWNER: cache next=c1.dll_next; null c1's links; clear BIT_OWNED;
 *                 release c1 iff newv==0; continue the walk via cached next.
 *   Thread CROSS: return c1's last slot (dec MASK_CNT); release c1 iff the
 *                 dec brought the word to 0.
 *
 * Verifies under RC11, all interleavings:
 *   (INV-6)  c1 is released exactly once — never zero (leak) or twice
 *            (double-free).  A `released` counter asserts ==1 at the end.
 *   (INV-21) the owner's accesses to c1's DLL links happen while c1 is
 *            still live (a `live` ghost + assert), because they are all
 *            sequenced BEFORE the BIT_OWNED clear; and the walk advances
 *            via the CACHED next, never a re-read of freed `c1.dll_next`.
 *   No data race on c1.live / c1.dll_next (GenMC checks).
 *
 * Run:  make run-dll      (GenMC)
 *       make smoke-dll    (concrete gcc + TSAN sanity)
 *
 * Co-Authored-By: Claude <noreply@anthropic.com>
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>

#define BIT_OWNED 0x80000000u
#define MASK_CNT  0x7FFFFFFFu

typedef struct Chunk {
    _Atomic(unsigned) flags;       /* BIT_OWNED | MASK_CNT */
    struct Chunk     *dll_next;    /* owner-private link */
    struct Chunk     *dll_prev;
    int               live;        /* ghost: 1 = chunk valid, 0 = released */
} Chunk;

static Chunk c1, c2;
static _Atomic(int) g_released;    /* counts releasers of c1 — must end == 1 */

static void release_c1(void) {
    /* Exactly-one-releaser: this must run for c1 exactly once.  Model the
     * "destruct + deallocate_chunk" as flipping the live ghost and bumping
     * the releaser count. */
    int prev = atomic_fetch_add_explicit(&g_released, 1, memory_order_relaxed);
    assert(prev == 0);             /* (INV-6) never a second releaser */
    c1.live = 0;                   /* chunk memory now invalid */
}

/* OWNER thread-exit walk over c1 (cached-next discipline). */
static void *t_owner(void *a) {
    (void)a;
    Chunk *c = &c1;

    /* (INV-21) c is still OWNED here (BIT_OWNED set ⇒ not yet releasable by
     * cross), so its links are safe to read/write.  Cache next FIRST. */
    assert(c->live);
    Chunk *next = c->dll_next;     /* <<< cached BEFORE the clear */
    assert(c->live);
    c->dll_prev = NULL;            /* owner nulls its links (still owned) */
    c->dll_next = NULL;

    /* Clear BIT_OWNED.  After this, cross-thread free may release c. */
    unsigned old = atomic_fetch_and_explicit(&c->flags, ~BIT_OWNED,
                                              memory_order_acq_rel);
    unsigned newv = old & ~BIT_OWNED;
    if(newv == 0)                  /* MASK_CNT was 0 → owner is releaser */
        release_c1();
    /* else: non-empty — leave c for cross_release.  Do NOT touch c again. */

    /* Continue the walk via the CACHED next (never re-read c->dll_next). */
    assert(next == &c2);
    assert(next->live);            /* the next chunk is safely reachable */
    return NULL;
}

/* CROSS thread returns c1's last slot. */
static void *t_cross(void *a) {
    (void)a;
    /* atomicDecAndTest: sub 1, releaser iff the word hits 0 (only possible
     * once BIT_OWNED is already clear). */
    unsigned newv = atomic_fetch_sub_explicit(&c1.flags, 1u,
                                               memory_order_acq_rel) - 1u;
    if(newv == 0)
        release_c1();
    return NULL;
}

int main(void) {
    /* c1: owner alive + 1 cross-held slot; linked c1 -> c2. */
    atomic_store_explicit(&c1.flags, BIT_OWNED | 1u, memory_order_relaxed);
    c1.dll_next = &c2; c1.dll_prev = NULL; c1.live = 1;
    c2.dll_next = NULL; c2.dll_prev = &c1; c2.live = 1;
    atomic_store_explicit(&g_released, 0, memory_order_relaxed);

    pthread_t o, x;
    pthread_create(&o, NULL, t_owner, NULL);
    pthread_create(&x, NULL, t_cross, NULL);
    pthread_join(o, NULL);
    pthread_join(x, NULL);

    /* (INV-6) c1 released exactly once, by exactly one of the two paths. */
    assert(atomic_load_explicit(&g_released, memory_order_relaxed) == 1);
    assert(c1.live == 0);                       /* the releaser freed it */
    assert(atomic_load_explicit(&c1.flags, memory_order_relaxed) == 0u);
    return 0;
}
