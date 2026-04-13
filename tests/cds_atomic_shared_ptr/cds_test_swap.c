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
 * GenMC test 4: swap + local_reset safety under concurrency
 *
 * Verifies that 2 threads concurrently performing an atomic exchange
 * (atomic_shared_ptr::swap with their own local ref as new value) and
 * subsequently releasing the received old pointer via
 * local_shared_ptr::reset() do not cause use-after-free, double-free,
 * or refcount corruption.
 *
 * Models the swap path from kame/atomic_smart_ptr.h:
 *   - CAS g_ref from (pref, tag_old) to (new_val, 0)   [acq_rel success]
 *   - Transfer outstanding local_refcnt to global:
 *       atomic_fetch_add(pref->refcnt, tag_old)         [relaxed]
 *   - Caller inherits pref's former g_ref +1 as its own global reference.
 *
 * Memory orderings match the original:
 *   - compare_exchange_weak: acq_rel success, relaxed failure
 *   - global refcnt fetch_add on tag transfer: relaxed
 *   - local reset fetch_sub (decAndTest): acq_rel
 *
 * Note: this test intentionally keeps tag_old = 0 throughout because
 * there are no concurrent load_shared_ calls here. Pure swap vs swap
 * verifies the CAS serialization and refcount ownership transfer.
 * The tag-to-global transfer path is exercised in conjunction with
 * load_shared_ by cds_test_cas and cds_test_multi_cas.
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
    int data;
    int destroyed;
} Obj;

/* Shared atomic_shared_ptr slot */
static _Atomic(uintptr_t) g_ref;

/* Three aligned objects — g_ref starts holding obj_A; threads swap in B/C */
static Obj g_obj_A __attribute__((aligned(CAPACITY)));
static Obj g_obj_B __attribute__((aligned(CAPACITY)));
static Obj g_obj_C __attribute__((aligned(CAPACITY)));

static void obj_init(Obj *o, int data) {
    /* refcnt = 1 represents "held by current owner".
     * For g_obj_A initially: owner is g_ref.
     * For g_obj_B, g_obj_C initially: owner is the thread that will swap
     * it in; that owning +1 becomes g_ref's owning +1 on successful CAS. */
    atomic_store_explicit(&o->refcnt, 1, memory_order_relaxed);
    o->data = data;
    o->destroyed = 0;
}

static Obj *get_ptr(uintptr_t tagged) {
    return (Obj *)(tagged & PTR_MASK);
}

static uintptr_t get_tag(uintptr_t tagged) {
    return tagged & TAG_MASK;
}

/*
 * swap_exchange_: atomic exchange of g_ref with a new pointer,
 * returning the old pointer with a transferred global reference.
 *
 * If the old tagged pointer had nonzero local_refcnt (in-flight loads),
 * fold those counts into pref->refcnt so that their later fall-back
 * global decrements remain balanced.
 */
static Obj *swap_exchange(Obj *new_val) {
    uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
    for (;;) {
        Obj *pref = get_ptr(cur);
        uintptr_t tag_old = get_tag(cur);
        uintptr_t desired = (uintptr_t)new_val; /* fresh tag = 0 */
        if (atomic_compare_exchange_weak_explicit(&g_ref, &cur, desired,
                memory_order_acq_rel, memory_order_relaxed)) {
            if (pref && tag_old) {
                atomic_fetch_add_explicit(&pref->refcnt, tag_old,
                        memory_order_relaxed);
            }
            return pref;
        }
    }
}

/*
 * local_reset: release a global reference (simulates local_shared_ptr::reset).
 * decAndTest with acq_rel.
 */
static void local_reset(Obj *pref) {
    if (!pref) return;
    uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1,
            memory_order_acq_rel);
    if (old_rc == 1) {
        pref->destroyed = 1;
    }
}

/* Thread 1: swap in Obj_B, release received old pointer */
static void *thread_swap_B(void *arg) {
    (void)arg;
    Obj *old = swap_exchange(&g_obj_B);
    if (old) {
        assert(old->destroyed == 0);
        int v = old->data;
        (void)v;
        local_reset(old);
    }
    return NULL;
}

/* Thread 2: swap in Obj_C, release received old pointer (symmetric with T1) */
static void *thread_swap_C(void *arg) {
    (void)arg;
    Obj *old = swap_exchange(&g_obj_C);
    if (old) {
        assert(old->destroyed == 0);
        int v = old->data;
        (void)v;
        local_reset(old);
    }
    return NULL;
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    obj_init(&g_obj_A, 101);
    obj_init(&g_obj_B, 102);
    obj_init(&g_obj_C, 103);

    atomic_store_explicit(&g_ref, (uintptr_t)&g_obj_A, memory_order_relaxed);

    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_swap_B, NULL);
    pthread_create(&t2, NULL, thread_swap_C, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    /* Final cleanup: whatever remains in g_ref is released by main */
    Obj *final_p = get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed));
    if (final_p) {
        local_reset(final_p);
    }

    /* All three objects must be destroyed exactly once */
    assert(g_obj_A.destroyed == 1);
    assert(g_obj_B.destroyed == 1);
    assert(g_obj_C.destroyed == 1);
    /* All refcnts must be zero */
    assert(atomic_load_explicit(&g_obj_A.refcnt, memory_order_relaxed) == 0);
    assert(atomic_load_explicit(&g_obj_B.refcnt, memory_order_relaxed) == 0);
    assert(atomic_load_explicit(&g_obj_C.refcnt, memory_order_relaxed) == 0);

    return 0;
}
