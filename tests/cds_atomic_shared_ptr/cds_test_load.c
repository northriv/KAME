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
 * GenMC test 1: load_shared_ / release_tag_ref_ safety
 *
 * Verifies that 3 threads concurrently calling load_shared_()
 * and subsequently releasing via local_shared_ptr::reset() do not
 * cause use-after-free or refcount corruption.
 *
 * Models the core protocol from kame/atomic_smart_ptr.h:
 *   acquire_tag_ref_ -> global refcnt fetch_add -> release_tag_ref_
 *
 * Memory orderings match the original exactly:
 *   - compare_exchange_weak: acq_rel success, relaxed failure
 *   - global refcnt fetch_add in load_shared_: relaxed
 *   - global refcnt fetch_sub in release_tag_ref_ fallback: acq_rel (decAndTest)
 *   - global refcnt fetch_sub in local reset: acq_rel (decAndTest)
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

/* Simulated object with intrusive refcount */
typedef struct {
    _Atomic(uintptr_t) refcnt;
    int data;        /* payload — read to detect use-after-free */
    int destroyed;   /* set to 1 when "deleted" */
} Obj;

/* The shared atomic_shared_ptr: tagged pointer (ptr | local_refcnt) */
static _Atomic(uintptr_t) g_ref;

/* Global object — must be aligned so low bits of address are zero */
static Obj g_obj __attribute__((aligned(CAPACITY)));

static void obj_init(Obj *o) {
    atomic_store_explicit(&o->refcnt, 1, memory_order_relaxed);  /* initial owner: g_ref */
    o->data = 42;
    o->destroyed = 0;
}

static Obj *get_ptr(uintptr_t tagged) {
    return (Obj *)(tagged & PTR_MASK);
}

static uintptr_t get_tag(uintptr_t tagged) {
    return tagged & TAG_MASK;
}

/*
 * acquire_tag_ref_: atomically increment the local refcount in the
 * tagged pointer via CAS.
 * Returns the Ref pointer; writes the new tag value to *rcnt_out.
 */
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
        if (rcnt_new >= CAPACITY) {
            continue;
        }
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
 * release_tag_ref_: try to decrement local refcount via CAS.
 * If the pointer was swapped out, fall back to global refcount decrement.
 */
static void release_tag_ref(Obj *pref) {
    for (;;) {
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        uintptr_t rcnt_old = get_tag(cur);
        if (rcnt_old && get_ptr(cur) == pref) {
            uintptr_t rcnt_new = rcnt_old - 1u;
            uintptr_t expected = (uintptr_t)pref + rcnt_old;
            uintptr_t desired  = (uintptr_t)pref + rcnt_new;
            if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                    memory_order_acq_rel, memory_order_relaxed))
                break;
            /* Re-check if pointer is still the same */
            if (get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed)) == pref)
                continue;  /* pointer unchanged, retry */
        }
        /* Local reference was transferred to global by a swapper.
         * Decrement global refcount (decAndTest semantics). */
        uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1,
                memory_order_acq_rel);
        if (old_rc == 1) {
            /* refcnt reached 0 — "delete" the object */
            pref->destroyed = 1;
        }
        break;
    }
}

/*
 * load_shared_: acquire a global reference to the pointed-to object.
 * Returns the Ref pointer with global refcount incremented by 1.
 */
static Obj *load_shared(void) {
    uintptr_t rcnt;
    Obj *pref = acquire_tag_ref(&rcnt);
    if (!pref) return NULL;
    /* Increment global refcount — relaxed, matching the original */
    atomic_fetch_add_explicit(&pref->refcnt, 1, memory_order_relaxed);
    release_tag_ref(pref);
    return pref;
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

/* Thread: load_shared, read data, then release */
static void *thread_load(void *arg) {
    (void)arg;
    Obj *p = load_shared();
    if (p) {
        /* The object must not be destroyed while we hold a reference */
        assert(p->destroyed == 0);
        /* Read payload — GenMC will flag data races here */
        int v = p->data;
        (void)v;
        local_reset(p);
    }
    return NULL;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    obj_init(&g_obj);
    /* Store pointer with local refcount = 0 */
    atomic_store_explicit(&g_ref, (uintptr_t)&g_obj, memory_order_relaxed);

    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_load, NULL);
    pthread_create(&t2, NULL, thread_load, NULL);

    /* Main thread also does a load */
    Obj *p = load_shared();
    if (p) {
        assert(p->destroyed == 0);
        local_reset(p);
    }

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    /* Final cleanup: release g_ref's ownership */
    Obj *final_p = get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed));
    if (final_p) {
        local_reset(final_p);
    }

    /* Object must be destroyed exactly once, refcnt must be 0 */
    assert(g_obj.destroyed == 1);
    uintptr_t final_rc = atomic_load_explicit(&g_obj.refcnt, memory_order_relaxed);
    assert(final_rc == 0);

    return 0;
}
