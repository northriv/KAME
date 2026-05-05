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
 * GenMC test 5: multiple threads racing on compareAndSet with
 * release_tag_ref cas_rcnt optimization.
 *
 * Same scenario as test 3, but the CAS failure path uses
 * release_tag_ref(pref, cas_rcnt) which combines the step 4 undo
 * with the local ref release:
 *   - Same pointer: tag-1, then undo step 4 excess
 *   - Pointer changed: single fetch_sub(cas_rcnt)
 *
 * Assertions:
 *   - At most one CAS succeeds
 *   - All objects' refcounts are consistent at the end
 *   - No memory leaks (all objects reach refcnt 0)
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
static _Atomic(int) success_count;

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
 * release_tag_ref with cas_rcnt parameter.
 * cas_rcnt > 0: called from CAS failure path after step 4 transferred
 * (cas_rcnt - 1) to global.
 *   - Same pointer, tag CAS success: undo step 4 excess.
 *   - Pointer changed: single fetch_sub(cas_rcnt) = undo + release combined.
 * cas_rcnt = 0: normal release (decrement tag by 1, or decAndTest on global).
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
                /* Tag decremented — our local ref released.
                 * cas_rcnt > 0: undo step 4's excess on global. */
                if (cas_rcnt > 1u)
                    atomic_fetch_add_explicit(&pref->refcnt,
                        -(uintptr_t)(cas_rcnt - 1u), memory_order_relaxed);
                break;
            }
            if (get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed)) == pref)
                continue;
        }
        /* pointer changed: undo step 4 excess + our 1 ref = cas_rcnt (or 1) */
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
 * compareAndSet (NOSWAP=true variant) using release_tag_ref(pref, rcnt_old)
 * on the CAS failure path — eliminates the explicit undo step.
 */
static int compare_and_set(Obj *oldr, Obj *newr) {
    Obj *pref;

    if (newr)
        atomic_fetch_add_explicit(&newr->refcnt, 1, memory_order_relaxed);

    for (;;) {
        uintptr_t rcnt_old;
        pref = acquire_tag_ref(&rcnt_old);

        if (pref != oldr) {
            if (pref)
                release_tag_ref(pref, 0);
            if (newr)
                atomic_fetch_sub_explicit(&newr->refcnt, 1, memory_order_relaxed);
            return 0;
        }

        /* Step 4: transfer local refcount to global */
        if (pref && (rcnt_old != 1u))
            atomic_fetch_add_explicit(&pref->refcnt, rcnt_old - 1u, memory_order_relaxed);

        uintptr_t expected = (uintptr_t)pref + rcnt_old;
        uintptr_t desired  = (uintptr_t)newr;
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed))
            break;

        /* CAS failed — release_tag_ref with cas_rcnt combines undo + release */
        if (pref) {
            release_tag_ref(pref, rcnt_old);
        }
    }

    if (pref)
        atomic_fetch_sub_explicit(&pref->refcnt, 1, memory_order_acq_rel);

    return 1;
}

static void *thread_cas(void *arg) {
    Obj *my_obj = (Obj *)arg;

    int ok = compare_and_set(&obj_A, my_obj);
    if (ok) {
        atomic_fetch_add_explicit(&success_count, 1, memory_order_relaxed);
    }

    local_reset(&obj_A);
    local_reset(my_obj);
    return NULL;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    obj_init(&obj_A, 1);
    obj_init(&obj_B, 2);
    obj_init(&obj_C, 3);
    atomic_store_explicit(&success_count, 0, memory_order_relaxed);

    atomic_store_explicit(&g_ref, (uintptr_t)&obj_A, memory_order_relaxed);

    atomic_fetch_add_explicit(&obj_A.refcnt, 1, memory_order_relaxed); /* for thread 1 */
    atomic_fetch_add_explicit(&obj_A.refcnt, 1, memory_order_relaxed); /* for thread 2 */

    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_cas, &obj_B);
    pthread_create(&t2, NULL, thread_cas, &obj_C);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    int sc = atomic_load_explicit(&success_count, memory_order_relaxed);
    assert(sc <= 1);

    Obj *final_p = get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed));
    if (final_p)
        local_reset(final_p);

    assert(obj_A.destroyed == 1);
    assert(obj_B.destroyed == 1);
    assert(obj_C.destroyed == 1);

    uintptr_t rc_a = atomic_load_explicit(&obj_A.refcnt, memory_order_relaxed);
    uintptr_t rc_b = atomic_load_explicit(&obj_B.refcnt, memory_order_relaxed);
    uintptr_t rc_c = atomic_load_explicit(&obj_C.refcnt, memory_order_relaxed);
    assert(rc_a == 0);
    assert(rc_b == 0);
    assert(rc_c == 0);

    return 0;
}
