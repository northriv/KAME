/*
 * TLA+ → C11 mechanical translation: atomic_shared_ptr (Layer 0) v2
 *
 * Changes from v1:
 * - Assertions inside functions (MemorySafety, NoUseAfterFree, etc.)
 * - swap() operation added (TLA+ StartSwap)
 * - destroyed is _Atomic for safe assertion reads
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
    _Atomic(int) destroyed;  /* atomic for safe assertion reads */
} Obj;

static _Atomic(uintptr_t) g_ref;

static Obj obj_A __attribute__((aligned(CAPACITY)));
static Obj obj_B __attribute__((aligned(CAPACITY)));

static void obj_init(Obj *o, int id) {
    atomic_store_explicit(&o->refcnt, 1, memory_order_relaxed);
    o->id = id;
    atomic_store_explicit(&o->destroyed, 0, memory_order_relaxed);
}

static Obj *get_ptr(uintptr_t tagged) { return (Obj *)(tagged & PTR_MASK); }
static uintptr_t get_tag(uintptr_t tagged) { return tagged & TAG_MASK; }

/* === @invariant InstalledNotFreed === */
static void check_installed_not_freed(void) {
    uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
    Obj *p = get_ptr(cur);
    if (p) assert(atomic_load_explicit(&p->destroyed, memory_order_relaxed) == 0);
}

/* === acquire_tag_ref === */
static Obj *acquire_tag_ref(uintptr_t *rcnt_out) {
    Obj *pref;
    uintptr_t rcnt_new;
    for (;;) {
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        pref = get_ptr(cur);
        uintptr_t rcnt_old = get_tag(cur);
        if (!pref) { *rcnt_out = rcnt_old; return NULL; }
        rcnt_new = rcnt_old + 1u;
        if (rcnt_new >= CAPACITY) continue;
        uintptr_t expected = (uintptr_t)pref + rcnt_old;
        uintptr_t desired  = (uintptr_t)pref + rcnt_new;
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed)) {
            /* @invariant MemorySafety: pinned object is not freed */
            assert(atomic_load_explicit(&pref->destroyed, memory_order_relaxed) == 0);
            break;
        }
    }
    *rcnt_out = rcnt_new;
    return pref;
}

/* === release_tag_ref === */
static void release_tag_ref(Obj *pref) {
    for (;;) {
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        uintptr_t rcnt_old = get_tag(cur);
        if (rcnt_old) {
            uintptr_t rcnt_new = rcnt_old - 1u;
            uintptr_t expected = (uintptr_t)pref + rcnt_old;
            uintptr_t desired  = (uintptr_t)pref + rcnt_new;
            if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                    memory_order_acq_rel, memory_order_relaxed))
                break;
            cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
            if (get_ptr(cur) == pref) continue;
        }
        /* Fallback: global_rc-- */
        uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1,
                memory_order_acq_rel);
        if (old_rc == 1) {
            atomic_store_explicit(&pref->destroyed, 1, memory_order_relaxed);
        }
        /* @invariant FreedImpliesZeroRC */
        if (atomic_load_explicit(&pref->destroyed, memory_order_relaxed))
            assert(atomic_load_explicit(&pref->refcnt, memory_order_relaxed) == 0);
        break;
    }
}

/* === load_shared_ === */
static Obj *load_shared(void) {
    uintptr_t rcnt;
    Obj *pref = acquire_tag_ref(&rcnt);
    if (!pref) return NULL;
    atomic_fetch_add_explicit(&pref->refcnt, 1, memory_order_relaxed);
    release_tag_ref(pref);
    /* @invariant NoUseAfterFree: held object is not freed */
    assert(atomic_load_explicit(&pref->destroyed, memory_order_relaxed) == 0);
    return pref;
}

/* === reset (local_shared_ptr destructor) === */
static void local_reset(Obj *pref) {
    if (!pref) return;
    uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1,
            memory_order_acq_rel);
    if (old_rc == 1) {
        atomic_store_explicit(&pref->destroyed, 1, memory_order_relaxed);
    }
}

/* === compare_and_set === */
static int compare_and_set(Obj *oldr, Obj *newr) {
    if (newr)
        atomic_fetch_add_explicit(&newr->refcnt, 1, memory_order_relaxed);

    for (;;) {
        uintptr_t rcnt_old;
        Obj *pref = acquire_tag_ref(&rcnt_old);

        if (pref != oldr) {
            if (pref) release_tag_ref(pref);
            if (newr)
                atomic_fetch_sub_explicit(&newr->refcnt, 1, memory_order_relaxed);
            return 0;
        }

        if (pref && (rcnt_old != 1u))
            atomic_fetch_add_explicit(&pref->refcnt, rcnt_old - 1u,
                    memory_order_relaxed);

        uintptr_t expected = (uintptr_t)pref + rcnt_old;
        uintptr_t desired  = (uintptr_t)newr;
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed)) {
            /* CAS succeeded — cleanup */
            if (pref)
                atomic_fetch_sub_explicit(&pref->refcnt, 1, memory_order_acq_rel);
            /* @invariant InstalledNotFreed */
            check_installed_not_freed();
            return 1;
        }

        /* CAS failed — undo transfer */
        if (pref) {
            if (rcnt_old != 1u)
                atomic_fetch_add_explicit(&pref->refcnt,
                    -(uintptr_t)(rcnt_old - 1u), memory_order_relaxed);
            release_tag_ref(pref);
        }
    }
}

/* === swap (local_shared_ptr::swap(atomic_shared_ptr&)) === */
/* TLA+ StartSwap: like CAS but skips PreInc/Check/Cleanup */
static Obj *swap_ptr(Obj *newr) {
    /* No PreInc — caller's local_shared_ptr transfers ownership */
    for (;;) {
        uintptr_t rcnt_old;
        Obj *pref = acquire_tag_ref(&rcnt_old);

        /* No Check — swap always proceeds regardless of current value */

        /* Transfer */
        if (pref && (rcnt_old != 1u))
            atomic_fetch_add_explicit(&pref->refcnt, rcnt_old - 1u,
                    memory_order_relaxed);

        uintptr_t expected = (uintptr_t)pref + rcnt_old;
        uintptr_t desired  = (uintptr_t)newr;
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed)) {
            /* No Cleanup — old value returned to caller */
            /* @invariant InstalledNotFreed */
            check_installed_not_freed();
            return pref; /* caller now owns old value */
        }

        /* Undo transfer */
        if (pref) {
            if (rcnt_old != 1u)
                atomic_fetch_add_explicit(&pref->refcnt,
                    -(uintptr_t)(rcnt_old - 1u), memory_order_relaxed);
            release_tag_ref(pref);
        }
    }
}

/* === Test threads === */

static void *thread_load_and_reset(void *arg) {
    (void)arg;
    Obj *p = load_shared();
    if (p) {
        assert(atomic_load_explicit(&p->destroyed, memory_order_relaxed) == 0);
        local_reset(p);
    }
    return NULL;
}

static void *thread_cas(void *arg) {
    Obj *new_obj = (Obj *)arg;
    int ok = compare_and_set(&obj_A, new_obj);
    (void)ok;
    local_reset(&obj_A);  /* release caller's ref to old */
    local_reset(new_obj);  /* release creator's ref to new */
    return NULL;
}

static void *thread_swap(void *arg) {
    Obj *new_obj = (Obj *)arg;
    /* swap installs new_obj, returns old */
    Obj *old = swap_ptr(new_obj);
    if (old) {
        assert(atomic_load_explicit(&old->destroyed, memory_order_relaxed) == 0);
        local_reset(old); /* release old */
    }
    return NULL;
}

int main(void) {
    obj_init(&obj_A, 1);
    obj_init(&obj_B, 2);
    atomic_store_explicit(&g_ref, (uintptr_t)&obj_A, memory_order_relaxed);

    /* Pre-acquire refs for CAS thread */
    atomic_fetch_add_explicit(&obj_A.refcnt, 1, memory_order_relaxed);
    /* Pre-acquire ref for swap thread (obj_B ownership transferred) */
    /* obj_B.refcnt already 1 from init — swap takes ownership */

    pthread_t t1, t2;

    /* Test 1: load + CAS */
    pthread_create(&t1, NULL, thread_load_and_reset, NULL);
    pthread_create(&t2, NULL, thread_cas, &obj_B);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    /* Final cleanup */
    Obj *final_p = get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed));
    if (final_p) local_reset(final_p);

    assert(atomic_load_explicit(&obj_A.destroyed, memory_order_relaxed) == 1);
    assert(atomic_load_explicit(&obj_B.destroyed, memory_order_relaxed) == 1);
    assert(atomic_load_explicit(&obj_A.refcnt, memory_order_relaxed) == 0);
    assert(atomic_load_explicit(&obj_B.refcnt, memory_order_relaxed) == 0);

    return 0;
}
