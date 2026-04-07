/*
 * GenMC test 3: multiple threads racing on compareAndSwap_
 *
 * 2 threads each attempt to CAS the same atomic_shared_ptr from
 * obj_A to their own object (obj_B, obj_C). At most one
 * should succeed; the other fails.
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
            if (get_ptr(cur) == pref)
                continue;
        }
        uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1,
                memory_order_acq_rel);
        if (old_rc == 1)
            pref->destroyed = 1;
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
 * compareAndSet (NOSWAP=true variant): simpler, doesn't update oldr on failure.
 * Returns 1 on success, 0 on failure.
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
                release_tag_ref(pref);
            if (newr)
                atomic_fetch_sub_explicit(&newr->refcnt, 1, memory_order_relaxed);
            return 0;
        }

        if (pref && (rcnt_old != 1u))
            atomic_fetch_add_explicit(&pref->refcnt, rcnt_old - 1u, memory_order_relaxed);

        uintptr_t expected = (uintptr_t)pref + rcnt_old;
        uintptr_t desired  = (uintptr_t)newr;
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed))
            break;

        if (pref) {
            if (rcnt_old != 1u)
                atomic_fetch_add_explicit(&pref->refcnt,
                    -(uintptr_t)(rcnt_old - 1u), memory_order_relaxed);
            release_tag_ref(pref);
        }
    }

    if (pref)
        atomic_fetch_sub_explicit(&pref->refcnt, 1, memory_order_acq_rel);

    return 1;
}

/* Each thread tries to CAS from obj_A to its own object.
 * Models the calling convention: compareAndSet takes const local_shared_ptr& oldr,
 * so the caller holds a global reference to the old value.
 * The reference to obj_A is acquired BEFORE thread creation (in main)
 * to match real usage where local_shared_ptr is obtained before the CAS call. */
static void *thread_cas(void *arg) {
    Obj *my_obj = (Obj *)arg;

    /* Thread already holds a global ref to obj_A (acquired in main before spawn) */
    int ok = compare_and_set(&obj_A, my_obj);
    if (ok) {
        atomic_fetch_add_explicit(&success_count, 1, memory_order_relaxed);
    }

    /* Release the caller's local_shared_ptr to obj_A */
    local_reset(&obj_A);

    /* Always release creator's ref on newr.
     * - CAS succeeded: compare_and_set pre-incremented newr for g_ref,
     *   so creator ref is separate and must be released.
     * - CAS failed: compare_and_set rolled back the pre-increment,
     *   so creator ref is the only remaining ref. */
    local_reset(my_obj);
    return NULL;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    obj_init(&obj_A, 1);
    obj_init(&obj_B, 2);
    obj_init(&obj_C, 3);
    atomic_store_explicit(&success_count, 0, memory_order_relaxed);

    /* g_ref -> obj_A (refcnt = 1 from obj_init covers g_ref ownership) */
    atomic_store_explicit(&g_ref, (uintptr_t)&obj_A, memory_order_relaxed);

    /* Acquire local_shared_ptr references to obj_A for both threads BEFORE
     * spawning them. In real code, each thread would have obtained its
     * local_shared_ptr<T> (and thus a global ref) before calling compareAndSet.
     * Doing this sequentially here avoids a use-after-free race where one thread
     * frees obj_A before the other thread has acquired its reference. */
    atomic_fetch_add_explicit(&obj_A.refcnt, 1, memory_order_relaxed); /* for thread 1 */
    atomic_fetch_add_explicit(&obj_A.refcnt, 1, memory_order_relaxed); /* for thread 2 */

    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_cas, &obj_B);
    pthread_create(&t2, NULL, thread_cas, &obj_C);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    int sc = atomic_load_explicit(&success_count, memory_order_relaxed);
    /* Both threads try CAS(obj_A -> X). Only one can succeed because
     * the successful CAS changes g_ref away from obj_A, so the other
     * thread's acquire_tag_ref sees a different pointer and fails the
     * pref != oldr check. sc is 0 or 1. */
    assert(sc <= 1);

    /* Release g_ref's final ownership.
     * - If a CAS succeeded: g_ref -> winner's obj (refcnt=1 from CAS pre-increment,
     *   creator ref already released by thread_cas). local_reset destroys it.
     * - If no CAS succeeded: g_ref -> obj_A (refcnt=1 from obj_init).
     *   local_reset destroys it. */
    Obj *final_p = get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed));
    if (final_p)
        local_reset(final_p);

    /* obj_A:
     * - CAS succeeded: CAS step 6 released g_ref's ownership (fetch_sub acq_rel).
     *   obj_A.refcnt was 1 (from obj_init for g_ref), decremented to 0 -> destroyed.
     * - No CAS succeeded: released by final_p above. */
    assert(obj_A.destroyed == 1);

    /* obj_B and obj_C: both threads always call local_reset(my_obj).
     * - Winner: creator ref released by thread, g_ref ref released by final_p above.
     * - Loser: compare_and_set rolled back pre-increment; creator ref released by thread. */
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
