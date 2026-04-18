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
 * GenMC test 2: load_shared_ + compareAndSwap_ race
 *
 * One thread reads via load_shared_() while another swaps the pointer
 * via compareAndSwap_(). Verifies:
 *   - The reader's reference remains valid until released
 *   - Both old and new objects have correct final refcounts
 *   - No memory leaks (both objects eventually reach refcnt 0)
 *
 * Models compareAndSwap_<false> from kame/atomic_smart_ptr.h.
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
    int id;          /* identity tag */
    int destroyed;
} Obj;

/* Aligned objects — ensure low bits are zero */
static Obj obj_A __attribute__((aligned(CAPACITY)));
static Obj obj_B __attribute__((aligned(CAPACITY)));

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
            if (get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed)) == pref)
                continue;
        }
        uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1,
                memory_order_acq_rel);
        if (old_rc == 1)
            pref->destroyed = 1;
        break;
    }
}

static Obj *load_shared(void) {
    uintptr_t rcnt;
    Obj *pref = acquire_tag_ref(&rcnt);
    if (!pref) return NULL;
    atomic_fetch_add_explicit(&pref->refcnt, 1, memory_order_relaxed);
    release_tag_ref(pref);
    return pref;
}

static void local_reset(Obj *pref) {
    if (!pref) return;
    uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1,
            memory_order_acq_rel);
    if (old_rc == 1)
        pref->destroyed = 1;
}

/*
 * compareAndSwap_<false>: CAS on g_ref, replacing oldr with newr.
 * On failure, updates oldr to the current value (swap semantics).
 * Returns 1 on success, 0 on failure.
 *
 * oldr_ptr: pointer to the caller's "old" Obj* (will be updated on failure)
 * newr:     the new Obj* to install
 */
static int compare_and_swap(Obj **oldr_ptr, Obj *newr) {
    Obj *pref;

    /* Step 1: pre-increment newr's global refcount */
    if (newr)
        atomic_fetch_add_explicit(&newr->refcnt, 1, memory_order_relaxed);

    for (;;) {
        uintptr_t rcnt_old;
        pref = acquire_tag_ref(&rcnt_old);

        /* Mismatch? CAS fails */
        if (pref != *oldr_ptr) {
            if (pref) {
                /* NOSWAP=false: take a global ref for the new oldr value */
                atomic_fetch_add_explicit(&pref->refcnt, 1, memory_order_relaxed);
                release_tag_ref(pref);
            }
            /* Roll back newr's pre-incremented refcount */
            if (newr)
                atomic_fetch_sub_explicit(&newr->refcnt, 1, memory_order_relaxed);
            /* Release old oldr */
            if (*oldr_ptr) {
                uintptr_t old_rc = atomic_fetch_sub_explicit(&(*oldr_ptr)->refcnt, 1,
                        memory_order_acq_rel);
                if (old_rc == 1)
                    (*oldr_ptr)->destroyed = 1;
            }
            /* Update oldr to current */
            *oldr_ptr = pref;
            return 0;
        }

        /* Step 4: transfer local refcount to global */
        if (pref && (rcnt_old != 1u))
            atomic_fetch_add_explicit(&pref->refcnt, rcnt_old - 1u, memory_order_relaxed);

        /* Step 5: CAS m_ref */
        uintptr_t expected = (uintptr_t)pref + rcnt_old;
        uintptr_t desired  = (uintptr_t)newr; /* new tag = 0 */
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed))
            break;

        /* CAS failed — undo the transfer and retry */
        if (pref) {
            if (rcnt_old != 1u)
                atomic_fetch_add_explicit(&pref->refcnt,
                    -(uintptr_t)(rcnt_old - 1u), memory_order_relaxed);
            release_tag_ref(pref);
        }
    }

    /* Step 6: success — release g_ref's old ownership */
    if (pref)
        atomic_fetch_sub_explicit(&pref->refcnt, 1, memory_order_acq_rel);

    return 1;
}

/* Thread 1: reader — load_shared, use, release */
static void *thread_reader(void *arg) {
    (void)arg;
    Obj *p = load_shared();
    if (p) {
        assert(p->destroyed == 0);
        /* read the identity — must be either A or B */
        int id = p->id;
        assert(id == 1 || id == 2);
        local_reset(p);
    }
    return NULL;
}

/* Thread 2: writer — swap A -> B */
static void *thread_writer(void *arg) {
    (void)arg;
    /* oldr starts as a "local_shared_ptr" pointing to obj_A (with global ref) */
    Obj *oldr = &obj_A;
    atomic_fetch_add_explicit(&oldr->refcnt, 1, memory_order_relaxed); /* local_shared_ptr copy */

    int ok = compare_and_swap(&oldr, &obj_B);
    /* Whether success or failure, oldr now holds the current value */
    /* Release the local_shared_ptr */
    local_reset(oldr);
    (void)ok;
    return NULL;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    obj_init(&obj_A, 1);
    obj_init(&obj_B, 2);

    /* g_ref initially points to obj_A, global refcnt already 1 */
    atomic_store_explicit(&g_ref, (uintptr_t)&obj_A, memory_order_relaxed);

    /* obj_B is "not yet stored" — it will gain a ref when CAS succeeds.
     * It starts with refcnt=1 from obj_init (the "creator" owns it). */

    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_reader, NULL);
    pthread_create(&t2, NULL, thread_writer, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    /* Release g_ref's final ownership */
    Obj *final_p = get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed));
    if (final_p)
        local_reset(final_p);

    /* If CAS succeeded, obj_A should be destroyed and obj_B should still exist
     * (then we need to release obj_B's initial creator ref).
     * If CAS failed, obj_A is still in g_ref and obj_B was never stored
     * (release obj_B's creator ref). */

    /* Release obj_B's creator ownership if it hasn't been destroyed yet */
    if (!obj_B.destroyed)
        local_reset(&obj_B);

    /* Both objects should eventually be destroyed */
    assert(obj_A.destroyed == 1);
    assert(obj_B.destroyed == 1);

    uintptr_t rc_a = atomic_load_explicit(&obj_A.refcnt, memory_order_relaxed);
    uintptr_t rc_b = atomic_load_explicit(&obj_B.refcnt, memory_order_relaxed);
    assert(rc_a == 0);
    assert(rc_b == 0);

    return 0;
}
