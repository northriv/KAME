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
 * GenMC test 7: swap with cas_rcnt optimization on CAS failure path.
 *
 * Models local_shared_ptr<T>::swap(atomic_shared_ptr<T>&) from
 * kame/atomic_smart_ptr.h:
 *   1. acquire_tag_ref → tag++
 *   2. step 4: refcnt += (rcnt_old - 1) if rcnt_old > 1
 *   3. CAS m_ref (pref+rcnt_old → newr+0)
 *      - success: break, *this = pref (m_ref's old 1 transfers to *this)
 *      - failure: release_tag_ref(pref, rcnt_old) [cas_rcnt optimization]
 *
 * Verifies that the cas_rcnt-optimized failure path preserves refcount
 * balance under swap-vs-swap and swap-vs-load contention.
 *
 * The C++ unit test (4 threads × 400K iter) reports objcnt leaks (~70-90)
 * with this pattern even after the corresponding optimization in
 * compareAndSet (test 5/6) and load_shared_ pass GenMC. This test seeks
 * the exact interleaving that causes the leak.
 *
 * Difference from compareAndSet:
 *   - No pointer-match early return (any pref proceeds to step 4)
 *   - No fetch_sub(1) post-success (m_ref's 1 ref transfers to *this)
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
static Obj obj_C __attribute__((aligned(CAPACITY)));

static _Atomic(uintptr_t) g_ref;

static void obj_init(Obj *o, int id) {
    atomic_store_explicit(&o->refcnt, 1, memory_order_relaxed);
    o->id = id;
    o->destroyed = 0;
}

static Obj *get_ptr(uintptr_t tagged) {
    return (Obj *)(tagged & PTR_MASK);
}

static uintptr_t get_tag(uintptr_t tagged) {
    return tagged & TAG_MASK;
}

static Obj *acquire_tag_ref(uintptr_t *rcnt_out) {
    Obj *pref;
    uintptr_t rcnt_new;
    for (;;) {
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        pref = get_ptr(cur);
        uintptr_t rcnt_old = get_tag(cur);
        if (!pref) {
            *rcnt_out = rcnt_old;
            return NULL;
        }
        rcnt_new = rcnt_old + 1u;
        if (rcnt_new >= CAPACITY)
            continue;
        uintptr_t expected = (uintptr_t)pref + rcnt_old;
        uintptr_t desired  = (uintptr_t)pref + rcnt_new;
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed))
            break;
    }
    *rcnt_out = rcnt_new;
    return pref;
}

/*
 * release_tag_ref with cas_rcnt parameter, mirroring atomic_smart_ptr.h.
 *   cas_rcnt > 0 (CAS failure path): release cas_rcnt total refs back.
 *     - same pointer + tag CAS success: tag-=1, undo step 4 excess (cas_rcnt-1)
 *     - pointer changed: single fetch_sub(cas_rcnt) = undo + our 1 combined
 *   cas_rcnt = 0 (normal release): release 1 ref.
 */
static void release_tag_ref(Obj *pref, uintptr_t cas_rcnt) {
    for (;;) {
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        uintptr_t rcnt_old = get_tag(cur);
        if (rcnt_old && get_ptr(cur) == pref) {
            uintptr_t rcnt_new = rcnt_old - 1u;
            uintptr_t expected = (uintptr_t)pref + rcnt_old;
            uintptr_t desired  = (uintptr_t)pref + rcnt_new;
            if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                    memory_order_acq_rel, memory_order_relaxed)) {
                /* CRITICAL: undo must be acq_rel + delete check. Concurrent
                 * local_reset can drop refcnt to (cas_rcnt-1), our undo takes
                 * it to 0 — but without delete check the object leaks. */
                if (cas_rcnt > 1u) {
                    uintptr_t sub_amount = cas_rcnt - 1u;
                    uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt,
                            sub_amount, memory_order_acq_rel);
                    if (old_rc == sub_amount)
                        pref->destroyed = 1;
                }
                break;
            }
            if (get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed)) == pref)
                continue;
        }
        {
            uintptr_t to_sub = cas_rcnt ? cas_rcnt : 1u;
            uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, to_sub,
                    memory_order_acq_rel);
            if (old_rc == to_sub)
                pref->destroyed = 1;
        }
        break;
    }
}

static void local_reset(Obj *pref) {
    if (!pref) return;
    uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1,
            memory_order_acq_rel);
    if (old_rc == 1)
        pref->destroyed = 1;
}

/*
 * swap: atomically replaces g_ref's pointer with newr, returns old pref.
 * Caller's *this (= newr) is transferred to g_ref; pref (was in g_ref) is
 * transferred to caller's *this.
 *
 * The newr's refcount is NOT incremented here — caller's *this already had
 * +1 of newr; that +1 transfers to g_ref's "implicit 1".
 *
 * Returns: old pref. Caller's *this becomes pref. Pref's "implicit 1" from
 * old g_ref transfers to caller's *this — no atomic op.
 */
static Obj *swap_op(Obj *newr) {
    Obj *pref;
    for (;;) {
        uintptr_t rcnt_old;
        pref = acquire_tag_ref(&rcnt_old);
        if (!pref) {
            /* g_ref was null. Just CAS in newr. */
            uintptr_t expected = 0;
            uintptr_t desired  = (uintptr_t)newr;
            if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                    memory_order_acq_rel, memory_order_relaxed))
                return NULL;
            continue;
        }
        /* Step 4: transfer (rcnt_old - 1) tag refs to global. */
        if (rcnt_old != 1u)
            atomic_fetch_add_explicit(&pref->refcnt, rcnt_old - 1u,
                    memory_order_relaxed);

        uintptr_t expected = (uintptr_t)pref + rcnt_old;
        uintptr_t desired  = (uintptr_t)newr;
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed))
            break;

        /* CAS failure: cas_rcnt optimized release. */
        release_tag_ref(pref, rcnt_old);
    }
    /* Success: caller's *this gets pref (m_ref's old 1 transfers). No atomic op. */
    return pref;
}

/* Thread 1: swap in obj_B, release received pref */
static void *thread_swap_B(void *arg) {
    (void)arg;
    Obj *old = swap_op(&obj_B);
    if (old) {
        assert(old->destroyed == 0);
        local_reset(old);
    }
    return NULL;
}

/* Thread 2: swap in obj_C, release received pref */
static void *thread_swap_C(void *arg) {
    (void)arg;
    Obj *old = swap_op(&obj_C);
    if (old) {
        assert(old->destroyed == 0);
        local_reset(old);
    }
    return NULL;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    obj_init(&obj_A, 1);
    obj_init(&obj_B, 2);
    obj_init(&obj_C, 3);

    /* g_ref initially holds obj_A with its initial refcnt=1 (g_ref's implicit). */
    atomic_store_explicit(&g_ref, (uintptr_t)&obj_A, memory_order_relaxed);

    /* obj_B and obj_C start with refcnt=1 (initial owner = the thread). */

    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_swap_B, NULL);
    pthread_create(&t2, NULL, thread_swap_C, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    /* Final: whatever remains in g_ref is released by main. */
    Obj *final_p = get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed));
    if (final_p)
        local_reset(final_p);

    /* All three objects must be destroyed exactly once. */
    assert(obj_A.destroyed == 1);
    assert(obj_B.destroyed == 1);
    assert(obj_C.destroyed == 1);

    /* All refcnts must be zero. */
    uintptr_t rc_a = atomic_load_explicit(&obj_A.refcnt, memory_order_relaxed);
    uintptr_t rc_b = atomic_load_explicit(&obj_B.refcnt, memory_order_relaxed);
    uintptr_t rc_c = atomic_load_explicit(&obj_C.refcnt, memory_order_relaxed);
    assert(rc_a == 0);
    assert(rc_b == 0);
    assert(rc_c == 0);

    return 0;
}
