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
 * GenMC test 8: compareAndSet_ (NOSWAP=true) no-acquire optimization.
 *
 * Models compareAndSet_() from kame/atomic_smart_ptr.h, which avoids
 * acquire_tag_ref_() entirely because the caller's oldr (local_shared_ptr)
 * keeps pref alive for the duration of the call.
 *
 * Key difference from compareAndSwap_ (test 2/3):
 *   - No acquire: reads (pref, T) from g_ref directly via load_tagged_()
 *   - step 4: fetch_add(T)  [+T, not +T-1; no implicit acquire ref]
 *   - CAS: (pref + T) -> (newr + 0)
 *   - success: fetch_sub(1, acq_rel)  [release g_ref's implicit ownership]
 *   - failure undo: fetch_sub(T, relaxed)  [no delete check: oldr keeps N>=2]
 *
 * Race scenario: compareAndSet vs load_shared_ (which holds a tag ref T=1
 * during the CAS window). Verifies:
 *   - step 4 +T correctly pre-pays for the T concurrent tag holders
 *   - undo fetch_sub(T) on failure is safe (oldr alive => refcnt >= 2)
 *   - no double-decrement / leak in any interleaving
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

/* acquire_tag_ref: used by load_shared_ only (not compareAndSet) */
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

static void release_tag_ref(Obj *pref, uintptr_t added_global_rcnt) {
    for (;;) {
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        uintptr_t rcnt_old = get_tag(cur);
        if (rcnt_old && get_ptr(cur) == pref) {
            uintptr_t local_release = rcnt_old < added_global_rcnt ? rcnt_old : added_global_rcnt;
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

static Obj *load_shared(void) {
    uintptr_t rcnt;
    Obj *pref = acquire_tag_ref(&rcnt);
    if (!pref) return NULL;
    atomic_fetch_add_explicit(&pref->refcnt, rcnt, memory_order_relaxed);
    release_tag_ref(pref, rcnt);
    return pref;
}

/*
 * compareAndSet_ (NOSWAP=true) — no acquire_tag_ref_.
 *
 * Safety: oldr is the caller's local_shared_ptr, kept alive for the
 * duration of this call. So when pref == oldr, pref->refcnt >= 2
 * (g_ref's implicit 1 + oldr's 1), and undo fetch_sub(T) cannot
 * trigger deletion.
 *
 * step 4 = fetch_add(T): pre-pays for ALL T existing tag holders
 * (cf. compareAndSwap_ which uses T-1 because its own acquired tag
 * is "consumed" by the CAS).
 */
static int compare_and_set(Obj *oldr, Obj *newr) {
    /* pre-increment newr (optimistic) */
    if (newr)
        atomic_fetch_add_explicit(&newr->refcnt, 1, memory_order_relaxed);

    for (;;) {
        /* load_tagged_(): read (pref, T) without acquire */
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        Obj *pref = get_ptr(cur);
        uintptr_t T = get_tag(cur);

        /* pointer mismatch: return false, no step 4 needed */
        if (pref != oldr) {
            if (newr)
                atomic_fetch_sub_explicit(&newr->refcnt, 1, memory_order_relaxed);
            return 0;
        }

        /* step 4: +T (pre-pay for all T tag holders; no implicit acquire ref) */
        if (pref && T)
            atomic_fetch_add_explicit(&pref->refcnt, T, memory_order_relaxed);

        /* CAS */
        uintptr_t expected = (uintptr_t)pref + T;
        uintptr_t desired  = (uintptr_t)newr; /* new tag = 0 */
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed)) {
            /* success: release g_ref's implicit ownership of pref */
            if (pref)
                atomic_fetch_sub_explicit(&pref->refcnt, 1, memory_order_acq_rel);
            return 1;
        }

        /* failure: undo step 4.
         * relaxed: oldr keeps pref->refcnt >= 2, so this fetch_sub
         * cannot reach 0 and trigger deletion. */
        if (pref && T)
            atomic_fetch_sub_explicit(&pref->refcnt, T, memory_order_relaxed);

        /* strong version: retry */
    }
}

/* Thread 1: compareAndSet(A -> B), oldr=A held as local_shared_ptr */
static void *thread_cas(void *arg) {
    (void)arg;
    /* oldr = local_shared_ptr copy of A: refcnt +1 */
    atomic_fetch_add_explicit(&obj_A.refcnt, 1, memory_order_relaxed);
    int ok = compare_and_set(&obj_A, &obj_B);
    /* release local copy of oldr (A) */
    local_reset(&obj_A);
    (void)ok;
    return NULL;
}

/* Thread 2: load_shared — acquires a tag ref (T becomes 1 during CAS window) */
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
    pthread_create(&t1, NULL, thread_cas,  NULL);
    pthread_create(&t2, NULL, thread_load, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    /* Release g_ref's final ownership */
    Obj *final_p = get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed));
    if (final_p)
        local_reset(final_p);

    /* Release obj_B's creator ref if CAS never stored it */
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
