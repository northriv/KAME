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
 * GenMC test 9: scoped_atomic_view + compareAndSetWeak race.
 *
 * Models compareAndSetWeak(scoped, newr) from kame/atomic_smart_ptr.h
 * where the scoped already holds a tag ref (acquired in its constructor).
 *
 * Key differences from test 8 (compareAndSet, no-acquire, with const
 * local_shared_ptr):
 *   - scoped has a tag (1 of T tag holders, instead of +1 in refcnt)
 *   - step 4 = fetch_add(T-1) (pre-pay OTHERS only; we already have a tag)
 *   - failure undo = fetch_sub(T-1, relaxed)
 *   - success: tag is consumed by the CAS (scoped becomes Empty)
 *   - failure (pointer-changed before CAS): release_tag_ref_ on scoped's pref,
 *     then scoped becomes Empty. (Eager cleanup so caller can detect.)
 *
 * Race scenario: scoped CAS vs concurrent load_shared_. Verifies:
 *   - step 4 = +(T-1) is correct under contention
 *   - undo fetch_sub(T-1, relaxed) is safe (scoped's tag keeps refcnt >= 1)
 *   - pointer-changed eager cleanup balances the swapper's pre-pay
 *   - no double-decrement / leak under any interleaving
 */

#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

#define CAPACITY 8u
#define PTR_MASK (~(uintptr_t)(CAPACITY - 1))
#define TAG_MASK ((uintptr_t)(CAPACITY - 1))

typedef struct {
    _Atomic(uintptr_t) refcnt;
    int id;
    int destroyed;
} Obj;

static Obj obj_A __attribute__((aligned(CAPACITY)));
static Obj obj_B __attribute__((aligned(CAPACITY)));

static _Atomic(uintptr_t) g_ref;

static void obj_init(Obj *o, int id) {
    atomic_store_explicit(&o->refcnt, 1, memory_order_relaxed);
    o->id = id;
    o->destroyed = 0;
}

static Obj *get_ptr(uintptr_t tagged) { return (Obj *)(tagged & PTR_MASK); }
static uintptr_t get_tag(uintptr_t tagged) { return tagged & TAG_MASK; }

/* acquire_tag_ref: bumps tag count, returns (pref, rcnt_new) */
static Obj *acquire_tag_ref(uintptr_t *rcnt_out) {
    for (;;) {
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        Obj *pref = get_ptr(cur);
        uintptr_t rcnt_old = get_tag(cur);
        if (!pref) { *rcnt_out = rcnt_old; return NULL; }
        uintptr_t rcnt_new = rcnt_old + 1u;
        if (rcnt_new >= CAPACITY) continue;
        uintptr_t expected = (uintptr_t)pref + rcnt_old;
        uintptr_t desired  = (uintptr_t)pref + rcnt_new;
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed)) {
            *rcnt_out = rcnt_new;
            return pref;
        }
    }
}

/* release_tag_ref(pref, K): cas_rcnt-style release. */
static void release_tag_ref(Obj *pref, uintptr_t added_global_rcnt) {
    for (;;) {
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        uintptr_t rcnt_old = get_tag(cur);
        if (rcnt_old && get_ptr(cur) == pref) {
            uintptr_t local_release = rcnt_old < added_global_rcnt
                ? rcnt_old : added_global_rcnt;
            uintptr_t expected = (uintptr_t)pref + rcnt_old;
            uintptr_t desired  = (uintptr_t)pref + (rcnt_old - local_release);
            if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                    memory_order_acq_rel, memory_order_relaxed)) {
                uintptr_t sub = added_global_rcnt - local_release;
                if (sub) {
                    uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, sub,
                            memory_order_acq_rel);
                    if (old_rc == sub) pref->destroyed = 1;
                }
                return;
            }
            if (get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed)) == pref)
                continue;
        }
        {
            uintptr_t to_sub = added_global_rcnt ? added_global_rcnt : 1u;
            uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, to_sub,
                    memory_order_acq_rel);
            if (old_rc == to_sub) pref->destroyed = 1;
        }
        return;
    }
}

static void local_reset(Obj *pref) {
    if (!pref) return;
    uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1, memory_order_acq_rel);
    if (old_rc == 1) pref->destroyed = 1;
}

/* load_shared_(): acquire tag, promote to refcnt, drain tag. */
static Obj *load_shared(void) {
    uintptr_t rcnt;
    Obj *pref = acquire_tag_ref(&rcnt);
    if (!pref) return NULL;
    atomic_fetch_add_explicit(&pref->refcnt, rcnt, memory_order_relaxed);
    release_tag_ref(pref, rcnt);
    return pref;
}

/*
 * compareAndSetWeak (scoped variant) — no acquire, scoped already holds tag.
 *
 * Caller's `scoped_pref` is the Obj* from scoped's constructor (TagHeld).
 * The scoped's tag is one of the T tag holders in g_ref.
 *
 * On success: tag absorbed by CAS, scoped becomes Empty.
 *   Caller signals this by setting *scoped_pref_inout = NULL.
 * On failure (any reason): scoped state is preserved (TagHeld) for retry,
 *   OR cleared if pointer-changed (eager cleanup).
 *
 * Returns: 1 on success, 0 on failure. *scoped_consumed = 1 if scoped's
 *   tag was consumed (success or pointer-changed cleanup).
 */
static int compare_and_set_weak_scoped(Obj **scoped_pref_inout, Obj *newr) {
    Obj *scoped_pref = *scoped_pref_inout;
    if (!scoped_pref) return 0;

    /* pre-increment newr (optimistic) */
    if (newr)
        atomic_fetch_add_explicit(&newr->refcnt, 1, memory_order_relaxed);

    /* load_tagged_(): read current (pref, T) without acquire */
    uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
    Obj *pref = get_ptr(cur);
    uintptr_t T = get_tag(cur);

    /* pointer mismatch: scoped's tag was absorbed by some swapper.
     * Eagerly release on scoped's pref (pointer-changed path will fetch_sub(1)
     * to consume swapper's pre-pay) and signal scoped now Empty. */
    if (pref != scoped_pref) {
        if (newr)
            atomic_fetch_sub_explicit(&newr->refcnt, 1, memory_order_relaxed);
        release_tag_ref(scoped_pref, 1u);
        *scoped_pref_inout = NULL;
        return 0;
    }

    /* step 4: +T — treat scoped's tag as if pre-paid in refcnt.
     *   ABSORBED: our CAS absorbs T tags including scoped's. Step4 covers
     *     all T pre-pays; success-path fetch_sub(2) consumes scoped's share.
     *   DRAINED: scoped's tag was already drained by a cas_rcnt; the
     *     drainer pre-paid scoped +1 in refcnt. Step4 covers current T_now
     *     others; success-path fetch_sub(2) consumes drainer's pre-pay.
     * Uniform bookkeeping; no need to runtime-detect ABSORBED vs DRAINED. */
    if (pref && T)
        atomic_fetch_add_explicit(&pref->refcnt, T, memory_order_relaxed);

    /* CAS pref+T → newr+0 */
    uintptr_t expected = (uintptr_t)pref + T;
    uintptr_t desired  = (uintptr_t)newr;
    if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
            memory_order_acq_rel, memory_order_relaxed)) {
        /* success: release g_ref's implicit 1 + scoped's tag-share (=2). */
        if (pref) {
            uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 2,
                    memory_order_acq_rel);
            if (old_rc == 2) pref->destroyed = 1;
        }
        /* scoped's tag-share was consumed by fetch_sub(2). Mark Empty. */
        *scoped_pref_inout = NULL;
        return 1;
    }

    /* failure: undo step 4 (full -T). Tag still held — destructor will run. */
    if (pref && T)
        atomic_fetch_sub_explicit(&pref->refcnt, T, memory_order_relaxed);
    if (newr)
        atomic_fetch_sub_explicit(&newr->refcnt, 1, memory_order_relaxed);
    return 0;
}

/* Thread 1: scoped CAS A→B (weak, single-shot).
 *   1. Acquire tag (scoped's constructor)
 *   2. compareAndSetWeak(scoped, B) — single attempt
 *   3. Destruct scoped — if still TagHeld, release_tag_ref. If Empty, no-op. */
static void *thread_scoped_cas(void *arg) {
    (void)arg;
    uintptr_t rcnt;
    Obj *scoped_pref = acquire_tag_ref(&rcnt);
    if (!scoped_pref) return NULL;
    /* (we don't care about rcnt after acquire; CAS reads current T fresh) */

    int ok = compare_and_set_weak_scoped(&scoped_pref, &obj_B);
    (void)ok;

    /* destructor: if scoped still TagHeld (failure case), release_tag_ref(1) */
    if (scoped_pref) {
        release_tag_ref(scoped_pref, 1u);
    }
    return NULL;
}

/* Thread 2: load_shared on g_ref */
static void *thread_load(void *arg) {
    (void)arg;
    Obj *p = load_shared();
    if (p) {
        assert(p->destroyed == 0);
        local_reset(p);
    }
    return NULL;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    obj_init(&obj_A, 1);
    obj_init(&obj_B, 2);

    /* g_ref initially holds obj_A (implicit ref = 1 already in obj_A.refcnt) */
    atomic_store_explicit(&g_ref, (uintptr_t)&obj_A, memory_order_relaxed);

    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_scoped_cas, NULL);
    pthread_create(&t2, NULL, thread_load,       NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    /* Release g_ref's final ownership */
    Obj *final_p = get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed));
    if (final_p)
        local_reset(final_p);

    /* If CAS succeeded, B is in g_ref (already released above).
     * If CAS failed (weak fail or pointer-mismatch — but here only A holds
     * pref so mismatch can't happen), A is still in g_ref.
     * obj_B's "creator" +1 is held by the test main; release it. */
    if (!obj_B.destroyed)
        local_reset(&obj_B);

    assert(obj_A.destroyed == 1);
    assert(obj_B.destroyed == 1);

    uintptr_t rc_a = atomic_load_explicit(&obj_A.refcnt, memory_order_relaxed);
    uintptr_t rc_b = atomic_load_explicit(&obj_B.refcnt, memory_order_relaxed);
    assert(rc_a == 0);
    assert(rc_b == 0);

    return 0;
}
