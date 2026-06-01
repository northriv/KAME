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
 * GenMC test — §22 CrossDeallocBatch held-bit invariant (INV-6, the
 * deferred-clear half).  THIS is the protocol the separately-reported
 * "CrossDeallocBatch flush" crash investigation flagged as entirely
 * comment-only — the one whose audit concluded "the held-bit invariant
 * holds; the actual crash was the macOS madvise straddle, not a held-bit
 * violation".  This harness mechanically confirms that audit.
 *
 * Design (allocator.cpp CrossDeallocBatch::push / flush /
 * PoolAllocator<...,true>::batch_return_to_bitmap):
 *   - `push(chunk, slot)` stores the entry WITHOUT clearing the slot's
 *     m_flags bit.  The still-set bit keeps that m_flags WORD non-zero,
 *     which keeps the chunk's MASK_CNT ≥ 1, which keeps the chunk
 *     un-releasable.  THE HELD BIT IS WHAT KEEPS THE CHUNK ALIVE.
 *   - `flush()` → `batch_return_to_bitmap` CAS-clears the slot bits; only
 *     when an m_flags word goes non-zero → 0 does it decrement MASK_CNT
 *     (via atomicDecAndTest on the packed word).  Release happens iff the
 *     packed word (BIT_OWNED | MASK_CNT) reaches 0.
 *
 * Two atomic words, exactly as in the allocator:
 *   bitmap  = one m_flags word; here 2 slot bits (BIT0/BIT1).
 *   packed  = m_flags_packed = BIT_OWNED | MASK_CNT; MASK_CNT counts
 *             NON-EMPTY m_flags words (here 0 or 1).
 *
 * Scenario: a chunk with both slots live (bitmap = 0b11, MASK_CNT = 1,
 * owner alive).  Two cross-threads each hold one slot in their batch and
 * flush it; the owner exits.  Verifies under RC11, all interleavings:
 *
 *   (INV-6)      the chunk is released EXACTLY ONCE (`released` counter
 *                asserts == 1) — never zero (leak) or twice (double-free).
 *   (held-bit)   release happens ONLY when the bitmap word is fully empty
 *                (`assert(bitmap == 0)` inside release) — i.e. every held
 *                slot has been flushed.  A chunk with a slot still held in
 *                ANY batch is never released.
 *   (no-UAF)     every flush runs against a LIVE chunk (`assert(live)` at
 *                flush entry) — guaranteed because the OTHER flusher's
 *                still-set bit keeps the chunk alive until this flusher
 *                also clears its bit.  This is the property whose violation
 *                would be the flush-time virtual-call use-after-free.
 *
 * Memory orders: real code uses `__sync_*` (seq_cst); modelled as acq_rel
 * (strictly weaker ⇒ race-freedom here implies it under the stronger order).
 *
 * Run:  make run-batch     (GenMC)
 *       make smoke-batch   (concrete gcc + TSAN sanity)
 *
 * Co-Authored-By: Claude <noreply@anthropic.com>
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>

#define BIT_OWNED 0x80000000u
#define BIT0      0x1u
#define BIT1      0x2u

static _Atomic(unsigned) bitmap;     /* one m_flags word: 2 slot bits */
static _Atomic(unsigned) packed;     /* BIT_OWNED | MASK_CNT (# non-empty words) */
static int               live;       /* ghost: chunk valid */
static _Atomic(int)      released;   /* releaser count — must end == 1 */

static void release_chunk(void) {
    /* (held-bit) the bitmap word MUST be empty when we release — i.e. no
     * slot is still held in any batch. */
    assert(atomic_load_explicit(&bitmap, memory_order_relaxed) == 0u);
    int prev = atomic_fetch_add_explicit(&released, 1, memory_order_relaxed);
    assert(prev == 0);               /* (INV-6) exactly one releaser */
    live = 0;
}

/* flush ONE batch entry against `bit` of this chunk's m_flags word —
 * the core of batch_return_to_bitmap for a single slot. */
static void flush_slot(unsigned bit) {
    /* (no-UAF) the chunk must be live when its batch_return_to_bitmap runs.
     * Held by the other flusher's still-set bit until we clear ours. */
    assert(live);

    /* CAS-clear this slot's bit (the real for(;;) atomicCompareAndSet loop). */
    unsigned oldv = atomic_load_explicit(&bitmap, memory_order_relaxed), newv;
    do {
        newv = oldv & ~bit;
    } while(!atomic_compare_exchange_weak_explicit(
                &bitmap, &oldv, newv,
                memory_order_acq_rel, memory_order_relaxed));

    /* Only when the WORD goes non-zero → 0 do we decrement MASK_CNT
     * (atomicDecAndTest on the packed word). */
    if(newv == 0u) {
        unsigned p_old = atomic_fetch_sub_explicit(&packed, 1u, memory_order_acq_rel);
        if(p_old - 1u == 0u)          /* packed hit 0 ⇒ BIT_OWNED was already clear */
            release_chunk();
    }
}

static void *t_flush0(void *a) { (void)a; flush_slot(BIT0); return NULL; }
static void *t_flush1(void *a) { (void)a; flush_slot(BIT1); return NULL; }

/* Owner thread exit: clear BIT_OWNED; releaser iff MASK_CNT already 0
 * (atomicFetchAnd on the packed word). */
static void *t_owner(void *a) {
    (void)a;
    unsigned p_old = atomic_fetch_and_explicit(&packed, ~BIT_OWNED, memory_order_acq_rel);
    if((p_old & ~BIT_OWNED) == 0u)    /* MASK_CNT was 0 ⇒ owner is releaser */
        release_chunk();
    return NULL;
}

int main(void) {
    atomic_store_explicit(&bitmap, BIT0 | BIT1, memory_order_relaxed);  /* both slots live */
    atomic_store_explicit(&packed, BIT_OWNED | 1u, memory_order_relaxed); /* owner alive, 1 non-empty word */
    live = 1;
    atomic_store_explicit(&released, 0, memory_order_relaxed);

    pthread_t f0, f1, o;
    pthread_create(&f0, NULL, t_flush0, NULL);
    pthread_create(&f1, NULL, t_flush1, NULL);
    pthread_create(&o,  NULL, t_owner,  NULL);
    pthread_join(f0, NULL);
    pthread_join(f1, NULL);
    pthread_join(o,  NULL);

    /* Both slots flushed + owner exited ⇒ fully released, exactly once. */
    assert(atomic_load_explicit(&released, memory_order_relaxed) == 1);
    assert(live == 0);
    assert(atomic_load_explicit(&bitmap, memory_order_relaxed) == 0u);
    assert(atomic_load_explicit(&packed, memory_order_relaxed) == 0u);
    return 0;
}
