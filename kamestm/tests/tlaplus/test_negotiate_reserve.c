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
 * C11 litmus derived from NegotiateReserve.tla (TagBeforeAcquire = TRUE),
 * checking the one thing the TLA+ spec cannot: whether its bound survives
 * the RC11 memory model.  TLA+ interleaves actions sequentially-
 * consistently, so every execution it counts is SC-explainable; weak memory
 * re-admits the deleted "+1" only through an execution with NO SC
 * equivalent.  For this protocol that execution is a genuine
 * store-buffering (Dekker) pair -- BOTH sides in "wrote mine, then read
 * yours" position:
 *
 *   H (HIGHEST):  tag.store(release); tag.load(acquire) verify;
 *                 [seq_cst fence]                <- the fix under audit
 *                 view = packet.load(...);       (the pin)
 *   P (peer):     packet.CAS(acq_rel);           (a licensed win)
 *                 [seq_cst fence]                <- variant knob
 *                 t = tag.load(...);             (NEXT round's licence check)
 *
 * Forbidden outcome: H's view predates P's win (H builds on a stale view)
 * AND P's next licence check misses H's tag (P keeps a licence it should
 * have lost).  Both misses together have no SC interleaving, and that extra
 * licensed win is exactly the "+1" TagBeforeAcquire deletes under SC.
 *
 * A single-round check-then-CAS peer is deliberately NOT the shape here:
 * "P observed no tag, then won after H's view" IS SC-reachable (P's
 * observation simply precedes H's tag in the interleaving) and is already
 * charged to the (T-1)K budget.  A first draft of this file asserted
 * against it and GenMC rightly refuted the assertion.
 *
 * Why the store-verify does not substitute for H's fence: an own-location
 * acquire load is satisfied from the store buffer (store forwarding, x86
 * and ARM alike), so it proves nothing about global visibility.
 *
 * TWO VARIANTS, both meant to be run:
 *
 *   default              PROVABLE variant: both fences present.  RC11
 *                        forbids the outcome; GenMC must report no errors.
 *
 *   -DNEG_RESERVE_AS_IMPLEMENTED
 *                        As-shipped variant: H keeps its fence
 *                        (transaction_negotiation.h plants it on the
 *                        _tag_first path), P has none -- its win is an
 *                        acq_rel RMW and its next check a relaxed load, as
 *                        in the implementation.  RC11 admits the outcome,
 *                        so GenMC is EXPECTED TO REPORT THE VIOLATION (the
 *                        same checked-in-counterexample convention as
 *                        NegotiateReserve_sideword_mc.cfg).  On real
 *                        targets the residual is narrower than RC11 admits:
 *                        x86's locked RMW is a full barrier, closing P's
 *                        side outright; ARMv8's casal is not, leaving a
 *                        store-buffer-drain window in which P's next check
 *                        can miss the tag -- one extra licensed win of the
 *                        "+1" class, nanoseconds wide, bounded per drain.
 *                        Fencing it away would put a dmb on EVERY tier's
 *                        commit CAS; the accepted trade is documented here
 *                        and at the fence site.
 *
 * Run (see tests/VERIFICATION.md for the GenMC build):
 *   genmc --disable-estimation test_negotiate_reserve.c
 *   genmc --disable-estimation -- -DNEG_RESERVE_AS_IMPLEMENTED \
 *         test_negotiate_reserve.c
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>

/* The Linkage's priority slot (0 = empty) and its packet word (a
 * generation counter standing in for the wrapper pointer). */
_Atomic(uint64_t) tag;
_Atomic(uint64_t) packet;

/* Post-join observables. */
_Atomic(uint64_t) h_view;    /* the view H pinned */
_Atomic(uint64_t) p_next_t;  /* P's next-round licence check result */
_Atomic(int)      p_won;     /* P's win landed */

#define H_STAMP 0x1u   /* nonzero = a (Reserved, HIGHEST) stamp */

static void *thread_H(void *arg) {
    (void)arg;
    /* tag_as_contender, Option B: store + verify (transaction.h). */
    atomic_store_explicit(&tag, H_STAMP, memory_order_release);
    uint64_t v = atomic_load_explicit(&tag, memory_order_acquire);
    if (v == H_STAMP) {
        atomic_thread_fence(memory_order_seq_cst);   /* the fix under audit */
        uint64_t view = atomic_load_explicit(&packet, memory_order_acquire);
        atomic_store_explicit(&h_view, view, memory_order_relaxed);
    }
    return NULL;
}

static void *thread_P(void *arg) {
    (void)arg;
    /* A licensed win: P observed no tag some time ago (before this litmus
     * begins) and now lands its CAS -- the commit path's acq_rel RMW. */
    uint64_t expect = 0;
    if (atomic_compare_exchange_strong_explicit(&packet, &expect, 1,
            memory_order_acq_rel, memory_order_relaxed)) {
        atomic_store_explicit(&p_won, 1, memory_order_relaxed);
#ifndef NEG_RESERVE_AS_IMPLEMENTED
        atomic_thread_fence(memory_order_seq_cst);
#endif
        /* Next round's licence check (fair_mode_blocks_me): relaxed in the
         * implementation. */
        uint64_t t = atomic_load_explicit(&tag, memory_order_relaxed);
        atomic_store_explicit(&p_next_t, t, memory_order_relaxed);
    }
    return NULL;
}

int main(void) {
    pthread_t th, tp;
    pthread_create(&th, NULL, thread_H, NULL);
    pthread_create(&tp, NULL, thread_P, NULL);
    pthread_join(th, NULL);
    pthread_join(tp, NULL);

    /* The store-buffering outcome: H pinned the pre-win view AND P's next
     * licence check saw no tag.  No SC interleaving produces both, so any
     * execution reaching here is weak-memory-only -- the re-admitted "+1". */
    if (atomic_load_explicit(&p_won, memory_order_relaxed)) {
        uint64_t view = atomic_load_explicit(&h_view, memory_order_relaxed);
        uint64_t t    = atomic_load_explicit(&p_next_t, memory_order_relaxed);
        assert(!(view == 0 && t == 0));
    }
    return 0;
}
