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
 * GenMC test 11: release_tagheld_zeroreset_ (bulk TagHeld release).
 *
 * Models scoped_atomic_view<T>::release_tagheld_zeroreset_() from
 * kamepoolalloc/atomic_smart_ptr.h: a TagHeld view releases by pre-paying
 * the OTHER (rcnt_now - 1) tag holders into the global refcnt and then
 * draining the whole word via release_tag_ref_(pref, rcnt_now); if the
 * word changed under it (pointer swapped, or another drainer already
 * pre-paid it), a plain fetch_sub(1) releases the +1 that is then in the
 * global count.
 *
 * This mechanizes the hand audit in DYNNODE_UAF_HANDOFF.md §13.1
 * (interleavings A–F), which until now was "closed by inspection" only.
 * Three scenarios (compile with -DSCEN=1|2|3):
 *
 *   SCEN 1  zeroreset vs load_shared_ — covers A (unchanged word, full
 *           drain), B (new acquires between the view's load and its drain
 *           CAS), C (the load_shared_'s own drain shrinks the word first;
 *           the zeroreset's release_tag_ref_ then takes the excess-undo
 *           fetch_sub path with the d93c7dfe delete check).
 *   SCEN 2  zeroreset vs zeroreset (two TagHeld holders on one word) —
 *           covers E: the loser of the drain race finds the word at 0 and
 *           must undo its full pre-pay through the global count.
 *   SCEN 4  zeroreset vs load_shared_ vs swap — the three-way
 *           composition of all of the above in one state space.
 *   SCEN 3  zeroreset vs swap (CASTransfer) — covers D (pointer changed
 *           before the view's load: plain fetch_sub(1)) and F (swap lands
 *           between the pre-pay and the drain CAS: release_tag_ref_ sees
 *           the pointer changed and undoes pre-pay + own share globally,
 *           against the swapper's fetch_add(tag_old) transfer).
 *
 * Verified invariants: no refcnt ever goes below zero (assert on every
 * fetch_sub), each object destroyed exactly once, refcnt exactly zero at
 * the end, and no thread ever touches a destroyed object.
 */

#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

#ifndef SCEN
#define SCEN 1
#endif

#define CAPACITY 8u
#define PTR_MASK (~(uintptr_t)(CAPACITY - 1))
#define TAG_MASK ((uintptr_t)(CAPACITY - 1))

typedef struct {
    _Atomic(uintptr_t) refcnt;
    int id;
    int destroyed;
} Obj;

static Obj obj_A __attribute__((aligned(CAPACITY)));
#if SCEN == 3 || SCEN == 4
static Obj obj_B __attribute__((aligned(CAPACITY)));
#endif

static _Atomic(uintptr_t) g_ref;

static void obj_init(Obj *o, int id) {
    atomic_store_explicit(&o->refcnt, 1, memory_order_relaxed);
    o->id = id;
    o->destroyed = 0;
}

static Obj *get_ptr(uintptr_t tagged) { return (Obj *)(tagged & PTR_MASK); }
static uintptr_t get_tag(uintptr_t tagged) { return tagged & TAG_MASK; }

/* Every global decrement funnels through here: the underflow tripwire the
 * KAME_RC_TRACE build checks at runtime is an ASSERT under GenMC. */
static void sub_rc(Obj *pref, uintptr_t k) {
    uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, k,
            memory_order_acq_rel);
    assert(old_rc >= k);            /* DEC-UNDERFLOW tripwire */
    if (old_rc == k) {
        assert(pref->destroyed == 0);
        pref->destroyed = 1;
    }
}

/* acquire_tag_ref: bumps tag count, returns (pref, rcnt_new). */
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

/* release_tag_ref_(pref, added): drain min(rcnt_old, added) from the word,
 * release the excess through the global count (with the delete check the
 * excess-undo path acquired after GenMC test 7). */
static void release_tag_ref(Obj *pref, uintptr_t added) {
    uintptr_t sub = added;
    for (;;) {
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        uintptr_t rcnt_old = get_tag(cur);
        if (rcnt_old && get_ptr(cur) == pref) {
            uintptr_t local_release = rcnt_old < added ? rcnt_old : added;
            uintptr_t expected = (uintptr_t)pref + rcnt_old;
            uintptr_t desired  = (uintptr_t)pref + (rcnt_old - local_release);
            if (atomic_compare_exchange_weak_explicit(&g_ref, &expected,
                    desired, memory_order_acq_rel, memory_order_relaxed)) {
                sub = added - local_release;
                break;
            }
            /* CAS lost: retry only while the pointer is still ours. */
            cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
            if (get_ptr(cur) == pref && get_tag(cur))
                continue;
        }
        break;      /* pointer changed or tag fully drained by others */
    }
    if (sub)
        sub_rc(pref, sub);
}

/* release_tagheld_zeroreset_(): the function under test. */
static void zeroreset(Obj *scoped_pref) {
    uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
    Obj *pref = get_ptr(cur);
    uintptr_t rcnt_now = get_tag(cur);
    if (pref == scoped_pref && rcnt_now > 0) {
        if (rcnt_now > 1)
            atomic_fetch_add_explicit(&scoped_pref->refcnt, rcnt_now - 1,
                    memory_order_relaxed);          /* pre-pay the others */
        release_tag_ref(scoped_pref, rcnt_now);     /* drain the word */
        return;
    }
    /* pointer changed or tag drained — our +1 is in the global count. */
    sub_rc(scoped_pref, 1u);
}

static void local_reset(Obj *pref) {
    if (!pref) return;
    sub_rc(pref, 1u);
}

#if SCEN == 1 || SCEN == 4
/* load_shared_: acquire tag, promote to refcnt, drain the tag. */
static Obj *load_shared(void) {
    uintptr_t rcnt;
    Obj *pref = acquire_tag_ref(&rcnt);
    if (!pref) return NULL;
    atomic_fetch_add_explicit(&pref->refcnt, rcnt, memory_order_relaxed);
    release_tag_ref(pref, rcnt);
    return pref;
}
#endif

#if SCEN == 3 || SCEN == 4
#ifndef SWAP_TRANSFER_AFTER_CAS
/* swap: local_shared_ptr::swap(atomic_shared_ptr&) — the REAL protocol.
 * The swapper first becomes one of the tag holders (acquire_tag_ref_),
 * pre-pays the OTHER (rcnt_old - 1) holders into the global count, and
 * only THEN CASes the word out; its own tag share is compensated by the
 * implicit m_ref +1 it walks away with.  The pre-pay-BEFORE-CAS order is
 * load-bearing: see the SWAP_TRANSFER_AFTER_CAS knob below. */
static Obj *swap_exchange(Obj *new_val) {
    for (;;) {
        uintptr_t rcnt_old;
        Obj *pref = acquire_tag_ref(&rcnt_old);
        if (!pref) return NULL;                     /* not reachable here */
        if (rcnt_old != 1u)
            atomic_fetch_add_explicit(&pref->refcnt, rcnt_old - 1u,
                    memory_order_relaxed);          /* pre-pay the others */
        uintptr_t expected = (uintptr_t)pref + rcnt_old;
        uintptr_t desired  = (uintptr_t)new_val;    /* fresh tag = 0 */
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed))
            return pref;        /* we own the implicit m_ref +1 */
        release_tag_ref(pref, rcnt_old);            /* undo, retry */
    }
}
#else
/* BUG KNOB (build with -DSWAP_TRANSFER_AFTER_CAS): the transfer order
 * cds_test_swap.c (test 4) has been modelling — CAS first, then
 * fetch_add(tag_old) onto the old object.  This order is UNSAFE against a
 * concurrent TagHeld releaser: between the CAS and the fetch_add, the
 * releaser sees "pointer changed => my +1 is in the global count", its
 * fetch_sub(1) hits the implicit 1, and the object is destroyed while the
 * swapper still holds it (premature destroy -> the swapper's fetch_add and
 * local_reset touch a freed object; double destroy).  GenMC finds this in
 * milliseconds -- which both validates the test's teeth and proves the
 * implementation's pre-pay-BEFORE-CAS ordering is load-bearing.  Test 4
 * never noticed because it has no tag-holding contender (tag_old == 0 in
 * all its executions -- the transfer never fires). */
static Obj *swap_exchange(Obj *new_val) {
    uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
    for (;;) {
        Obj *pref = get_ptr(cur);
        uintptr_t tag_old = get_tag(cur);
        uintptr_t desired = (uintptr_t)new_val;     /* fresh tag = 0 */
        if (atomic_compare_exchange_weak_explicit(&g_ref, &cur, desired,
                memory_order_acq_rel, memory_order_relaxed)) {
            if (pref && tag_old)
                atomic_fetch_add_explicit(&pref->refcnt, tag_old,
                        memory_order_relaxed);
            return pref;
        }
    }
}
#endif
#endif

/* The TagHeld view holder: acquire, then bulk-release. */
static void *thread_zeroreset(void *arg) {
    (void)arg;
    uintptr_t rcnt;
    Obj *scoped_pref = acquire_tag_ref(&rcnt);
    if (!scoped_pref) return NULL;
    assert(scoped_pref->destroyed == 0);
    zeroreset(scoped_pref);
    return NULL;
}

#if SCEN == 1 || SCEN == 4
static void *thread_load(void *arg) {
    (void)arg;
    Obj *p = load_shared();
    if (p) {
        assert(p->destroyed == 0);
        local_reset(p);
    }
    return NULL;
}
#endif

#if SCEN == 3 || SCEN == 4
static void *thread_swap(void *arg) {
    (void)arg;
    Obj *old = swap_exchange(&obj_B);
    if (old) {
        assert(old->destroyed == 0);
        local_reset(old);
    }
    return NULL;
}
#endif

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    obj_init(&obj_A, 1);
#if SCEN == 3 || SCEN == 4
    obj_init(&obj_B, 2);
#endif
    /* g_ref holds obj_A (the implicit ref = the 1 in obj_A.refcnt). */
    atomic_store_explicit(&g_ref, (uintptr_t)&obj_A, memory_order_relaxed);

    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_zeroreset, NULL);
#if SCEN == 1
    pthread_create(&t2, NULL, thread_load, NULL);
#elif SCEN == 2
    pthread_create(&t2, NULL, thread_zeroreset, NULL);
#elif SCEN == 3
    pthread_create(&t2, NULL, thread_swap, NULL);
#elif SCEN == 4
    pthread_create(&t2, NULL, thread_load, NULL);
    pthread_t t3;
    pthread_create(&t3, NULL, thread_swap, NULL);
#endif
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
#if SCEN == 4
    pthread_join(t3, NULL);
#endif

    /* Release g_ref's final ownership. */
    Obj *final_p = get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed));
    if (final_p)
        local_reset(final_p);

    assert(obj_A.destroyed == 1);
    assert(atomic_load_explicit(&obj_A.refcnt, memory_order_relaxed) == 0);
#if SCEN == 3 || SCEN == 4
    assert(obj_B.destroyed == 1);
    assert(atomic_load_explicit(&obj_B.refcnt, memory_order_relaxed) == 0);
#endif
    return 0;
}
