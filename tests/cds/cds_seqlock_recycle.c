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
 * GenMC test — the MEMORY-ORDERING core of the seqlock `lookup_chunk`
 * re-read (the candidate epoch+seqlock chunk-recycle root-cure,
 * `ChunkRecycle_threadepoch.tla`).
 *
 * WHY THIS TEST EXISTS (read before editing)
 * ------------------------------------------
 *   - The SHIPPED `lookup_chunk` is seqlock-FREE (allocator.cpp: "NO
 *     seqlock, NO epoch" — it relies on the live-slot invariant, verified
 *     via the ChunkClaim back_offset-after-CAS GenMC test).  This file does
 *     NOT test shipped code.
 *   - The epoch+seqlock is the *candidate* complete root-cure.  Its LOGICAL
 *     protocol (epoch-unchanged ⇒ consistent) is exhaustively verified by
 *     TLC (`ChunkRecycle_threadepoch.tla`, 3.69M states) and by the
 *     mechanically-derived `test_ChunkRecycle_threadepoch.c`.
 *   - BUT both of those collapse the WRITER (claim+meta-write+data-write)
 *     into ONE atomic step ("we mirror allocate as one indivisible step").
 *     That abstraction HIDES the weak-memory question a real multi-store
 *     writer raises: can a reader latch `meta1 == meta2` yet read DATA that
 *     belongs to a different epoch?  That hinges entirely on the reader's
 *     acquire fence placement — which a single-atomic-writer model cannot
 *     exercise.  This focused test fills exactly that gap.
 *
 * MODEL
 * -----
 * One recyclable unit, modelled as the canonical (odd/even) seqlock that the
 * epoch re-read reduces to:
 *   g_seq  : atomic counter.  EVEN = stable, ODD = a reclaim is mid-flight.
 *            (The allocator's monotonic `unitMeta.epoch` plays g_seq's role;
 *            the reader's `meta1 == meta2` accept maps to "g_seq even and
 *            unchanged across the two reads".)
 *   g_d0,g_d1 : the unit's payload (e.g. palloc identity + back_off).  A
 *            consistent snapshot has g_d0 == g_d1; a TORN read (g_d0 from
 *            epoch A, g_d1 from epoch B) is the safety violation the seqlock
 *            must make unobservable.
 *
 * Writer (one reclaim cycle, repeated WRITES times):
 *   seq -> odd (begin) ; release-fence ; d0=d1=v ; seq -> even (publish,release)
 * Reader (one lookup re-read):
 *   s1 = seq(acquire) ; d0,d1 = load ; [ACQUIRE FENCE] ; s2 = seq
 *   accept iff (s1 == s2 && even)  ->  assert(d0 == d1)   // INV: no torn read
 *
 * CONTROLS
 *   default            : reader has the acquire fence  -> EXPECT: No errors.
 *   -DNO_READER_FENCE  : fence removed (the exact weak-memory mistake the
 *                        re-read guards against) -> EXPECT: Safety violation
 *                        (a torn d0 != d1 latched under s1 == s2).  This is
 *                        the negative control proving the test is non-vacuous
 *                        and actually exercises the fence.
 *
 * Run:  make run-seqlock          (GenMC, positive control — clean)
 *       make run-seqlock-nofence  (GenMC, negative control — must fail)
 *
 * Co-Authored-By: Claude <noreply@anthropic.com>
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

#ifndef WRITES
#define WRITES 2          /* reclaim cycles; 2 gives a torn window, stays small */
#endif

static _Atomic(unsigned) g_seq;   /* even = stable, odd = reclaim in flight */
static _Atomic(unsigned) g_d0;    /* payload halves; equal in a consistent snapshot */
static _Atomic(unsigned) g_d1;

/* Writer: WRITES reclaim cycles, each bracketed by an odd seq (begin) and an
 * even seq (publish).  Data stores sit strictly inside the bracket, after a
 * release fence, so a reader that sees an even/unchanged seq pair is
 * guaranteed (with the reader's acquire fence) a single-epoch snapshot. */
static void *writer(void *a) {
    (void)a;
    for(unsigned v = 1; v <= WRITES; v++) {
        unsigned s = atomic_load_explicit(&g_seq, memory_order_relaxed);
        atomic_store_explicit(&g_seq, s + 1u, memory_order_relaxed);   /* odd: begin */
        atomic_thread_fence(memory_order_release);
        atomic_store_explicit(&g_d0, v, memory_order_relaxed);
        atomic_store_explicit(&g_d1, v, memory_order_relaxed);
        atomic_store_explicit(&g_seq, s + 2u, memory_order_release);   /* even: publish */
    }
    return NULL;
}

/* Reader: the seqlock lookup re-read.  Reads payload BETWEEN two seq reads;
 * accepts only a stable (even) unchanged seq.  The acquire fence keeps the
 * payload loads from sinking past the second seq read — drop it and a torn
 * snapshot can be latched. */
static void *reader(void *a) {
    (void)a;
    unsigned s1 = atomic_load_explicit(&g_seq, memory_order_acquire);
    unsigned d0 = atomic_load_explicit(&g_d0, memory_order_relaxed);
    unsigned d1 = atomic_load_explicit(&g_d1, memory_order_relaxed);
#ifndef NO_READER_FENCE
    atomic_thread_fence(memory_order_acquire);   /* the load the re-read relies on */
#endif
    unsigned s2 = atomic_load_explicit(&g_seq, memory_order_relaxed);
    if(s1 == s2 && (s1 & 1u) == 0u)              /* stable + unchanged ⇒ accept */
        assert(d0 == d1);                        /* INV_NoTornRead */
    return NULL;
}

int main(void) {
    atomic_store_explicit(&g_seq, 0u, memory_order_relaxed);
    atomic_store_explicit(&g_d0, 0u, memory_order_relaxed);
    atomic_store_explicit(&g_d1, 0u, memory_order_relaxed);

    pthread_t w, r;
    pthread_create(&w, NULL, writer, NULL);
    pthread_create(&r, NULL, reader, NULL);
    pthread_join(w, NULL);
    pthread_join(r, NULL);
    return 0;
}
