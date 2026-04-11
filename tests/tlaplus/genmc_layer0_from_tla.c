/*
 * TLA+ → C11 mechanical translation: atomic_shared_ptr (Layer 0)
 *
 * Generated from atomic_shared_ptr.tla @c11_action annotations.
 * Purpose: verify TLA+ model logic under C11/RC11 memory model via GenMC.
 *
 * This is NOT the real C++ implementation — it is a direct translation
 * of the TLA+ state machine into C11, preserving the same actions,
 * guards, and state transitions. Comparison with the real C++ code
 * (kame/atomic_smart_ptr.h) verifies model fidelity.
 *
 * Thread model: N threads, each performs load_shared_ then reset,
 * optionally interleaved with CAS operations.
 */

#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

/* === @c11_var mappings === */

#define CAPACITY 8u
#define PTR_MASK (~(uintptr_t)(CAPACITY - 1))
#define TAG_MASK ((uintptr_t)(CAPACITY - 1))

/* @c11_var global_rc[o]: _Atomic uintptr_t o->refcnt */
/* @c11_var freed[o]:     o->destroyed (non-atomic int) */
typedef struct {
    _Atomic(uintptr_t) refcnt;
    int id;
    int destroyed;
} Obj;

/* @c11_var ptr + local_rc: packed in _Atomic uintptr_t g_ref */
static _Atomic(uintptr_t) g_ref;

static Obj obj_A __attribute__((aligned(CAPACITY)));
static Obj obj_B __attribute__((aligned(CAPACITY)));

static void obj_init(Obj *o, int id) {
    atomic_store_explicit(&o->refcnt, 1, memory_order_relaxed);
    o->id = id;
    o->destroyed = 0;
}

/* === Helper: extract ptr and tag from packed g_ref === */

static Obj *get_ptr(uintptr_t tagged) {
    return (Obj *)(tagged & PTR_MASK);
}

static uintptr_t get_tag(uintptr_t tagged) {
    return tagged & TAG_MASK;
}

/* === @c11_action AcquireTagRefRead + AcquireTagRefCAS === */
/* TLA+: AcquireTagRefRead reads ptr and local_rc atomically,
 *       AcquireTagRefCAS does CAS(old, old+1) */
static Obj *acquire_tag_ref(uintptr_t *rcnt_out) {
    Obj *pref;
    uintptr_t rcnt_new;
    for (;;) {
        /* @c11_action AcquireTagRefRead */
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        pref = get_ptr(cur);
        uintptr_t rcnt_old = get_tag(cur);

        /* @c11_action AcquireTagRefNull */
        if (!pref) {
            *rcnt_out = rcnt_old;
            return NULL;
        }

        rcnt_new = rcnt_old + 1u;
        if (rcnt_new >= CAPACITY)
            continue; /* LocalRCBounded: spin */

        /* @c11_action AcquireTagRefCAS */
        uintptr_t expected = (uintptr_t)pref + rcnt_old;
        uintptr_t desired  = (uintptr_t)pref + rcnt_new;
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed))
            break;
    }
    *rcnt_out = rcnt_new;
    return pref;
}

/* === @c11_action ReleaseTagRefRead + ReleaseTagRefCAS + ReleaseTagRefGlobal === */
static void release_tag_ref(Obj *pref) {
    for (;;) {
        /* @c11_action ReleaseTagRefRead */
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        uintptr_t rcnt_old = get_tag(cur);

        if (rcnt_old) {
            /* @c11_action ReleaseTagRefCAS */
            uintptr_t rcnt_new = rcnt_old - 1u;
            uintptr_t expected = (uintptr_t)pref + rcnt_old;
            uintptr_t desired  = (uintptr_t)pref + rcnt_new;
            if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                    memory_order_acq_rel, memory_order_relaxed))
                break;
            cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
            if (get_ptr(cur) == pref)
                continue;
        }

        /* @c11_action ReleaseTagRefGlobal — fallback */
        uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1,
                memory_order_acq_rel);
        if (old_rc == 1) {
            pref->destroyed = 1;
        }
        break;
    }
}

/* === @c11_action StartLoadShared + LoadSharedIncGlobal + LoadSharedStartRelease === */
static Obj *load_shared(void) {
    uintptr_t rcnt;
    Obj *pref = acquire_tag_ref(&rcnt);
    if (!pref) return NULL;

    /* @c11_action LoadSharedIncGlobal */
    atomic_fetch_add_explicit(&pref->refcnt, 1, memory_order_relaxed);

    /* @c11_action LoadSharedStartRelease */
    release_tag_ref(pref);
    return pref;
}

/* === @c11_action Reset === */
static void local_reset(Obj *pref) {
    if (!pref) return;
    uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1,
            memory_order_acq_rel);
    if (old_rc == 1) {
        pref->destroyed = 1;
    }
}

/* === @c11_action StartCAS + CASPreInc + CASCheck + CASTransfer +
 *                 CASSwap + CASUndo + CASCleanup + CASFailDone === */
static int compare_and_set(Obj *oldr, Obj *newr) {
    /* @c11_action CASPreInc */
    if (newr)
        atomic_fetch_add_explicit(&newr->refcnt, 1, memory_order_relaxed);

    for (;;) {
        /* @c11_action CASReserve → AcquireTagRef */
        uintptr_t rcnt_old;
        Obj *pref = acquire_tag_ref(&rcnt_old);

        /* @c11_action CASCheck */
        if (pref != oldr) {
            if (pref)
                release_tag_ref(pref);
            /* @c11_action CASFailDone — rollback */
            if (newr)
                atomic_fetch_sub_explicit(&newr->refcnt, 1, memory_order_relaxed);
            return 0;
        }

        /* @c11_action CASTransfer */
        if (pref && (rcnt_old != 1u))
            atomic_fetch_add_explicit(&pref->refcnt, rcnt_old - 1u,
                    memory_order_relaxed);

        /* @c11_action CASSwap */
        uintptr_t expected = (uintptr_t)pref + rcnt_old;
        uintptr_t desired  = (uintptr_t)newr;
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed))
            break;

        /* @c11_action CASUndo */
        if (pref) {
            if (rcnt_old != 1u)
                atomic_fetch_add_explicit(&pref->refcnt,
                    -(uintptr_t)(rcnt_old - 1u), memory_order_relaxed);
            release_tag_ref(pref);
        }
    }

    /* @c11_action CASCleanup */
    Obj *old_pref = oldr;
    if (old_pref)
        atomic_fetch_sub_explicit(&old_pref->refcnt, 1, memory_order_acq_rel);

    return 1;
}

/* ================================================================
 * Test: 2 threads — load_shared + CAS interleaving
 * Maps to TLA+ step2_scan_plus_cas configuration
 * ================================================================ */

static void *thread_load_and_reset(void *arg) {
    (void)arg;
    /* @c11_action StartLoadShared (precondition: holds == 0) */
    Obj *p = load_shared();
    if (p) {
        assert(p->destroyed == 0); /* MemorySafety */
        /* @c11_action Reset */
        local_reset(p);
    }
    return NULL;
}

static void *thread_cas(void *arg) {
    Obj *new_obj = (Obj *)arg;
    /* Caller holds reference to obj_A (pre-acquired in main) */
    int ok = compare_and_set(&obj_A, new_obj);
    (void)ok;
    /* Release caller's local_shared_ptr to obj_A */
    local_reset(&obj_A);
    /* Release creator's ref on newr */
    local_reset(new_obj);
    return NULL;
}

int main(void) {
    obj_init(&obj_A, 1);
    obj_init(&obj_B, 2);

    /* g_ref → obj_A */
    atomic_store_explicit(&g_ref, (uintptr_t)&obj_A, memory_order_relaxed);

    /* Pre-acquire references for CAS thread (C++ calling convention) */
    atomic_fetch_add_explicit(&obj_A.refcnt, 1, memory_order_relaxed);

    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_load_and_reset, NULL);
    pthread_create(&t2, NULL, thread_cas, &obj_B);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    /* Final cleanup */
    Obj *final_p = get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed));
    if (final_p)
        local_reset(final_p);

    /* All objects must be destroyed, all refcounts zero */
    assert(obj_A.destroyed == 1);
    assert(obj_B.destroyed == 1);
    assert(atomic_load_explicit(&obj_A.refcnt, memory_order_relaxed) == 0);
    assert(atomic_load_explicit(&obj_B.refcnt, memory_order_relaxed) == 0);

    return 0;
}
