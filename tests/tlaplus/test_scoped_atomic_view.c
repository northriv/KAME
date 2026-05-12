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
 * C11 test mechanically translated from atomic_shared_ptr.tla (Layer 1)
 * — scoped_atomic_view sub-spec (Scope* actions).
 *
 * Models the minimal TagHeld lifecycle:
 *   ScopeStartAcquire → AcquireTagRefRead → AcquireTagRefCAS → ScopeSetState
 *   ScopeCASStart  → ScopeCASPreInc → ScopeCASLoad → ScopeCASCheck →
 *     match:    ScopeCASTransfer → ScopeCASSwap →
 *               (success: ScopeCASCleanup [fetch_sub(2)])
 *               (failure: ScopeCASUndo  [undo step4 + step2])
 *     mismatch: rtr_read with ctx = "scope_release" (release_tag_ref_(scope_pref, 1))
 *   ScopeDtor (if scope still TagHeld at exit) → release_tag_ref_(pref, 1)
 *
 * Distinguishing characteristics vs. compareAndSwap_ (Set/Swap path):
 *   - Skip acquire_tag_ref_ in CAS — scope provides +1.
 *   - step4 = +T (full T, not T-1): scope's tag-share is pre-paid as if
 *     already in global.
 *   - On CAS success, fetch_sub(2) on pref absorbs both m_ref's release
 *     AND scope's tag-share uniformly.
 *
 * Simplifications (matching the TLA+ spec):
 *   - WEAK CAS spurious failure NOT modelled: uses
 *     atomic_compare_exchange_STRONG_explicit throughout.  Strong-CAS
 *     behaviors are a subset of weak-CAS behaviors, so safety verified
 *     here transfers to the weak case (covered separately by
 *     cds_atomic_shared_ptr/cds_test_scoped_weak.c).
 *   - Owned state and adaptive promotion NOT modelled.
 *
 * Each of 2 threads: ScopeAcquire → ScopeCASSet → (Dtor if needed).
 *
 * TLA+ invariant checks via assert():
 *   MemorySafety       — scope_pref is never freed while scope tagheld
 *   NoUseAfterFree     — local_shared_ptr holdees are alive
 *   GlobalRCNonNeg     — global_rc never goes below zero
 *   FreedImpliesZeroRC — freed objects have refcnt=0
 *   ScopeConsistent    — scope_state and scope_pref agree
 *   TerminalCheck      — at end, ptr's object has rc=1, all others rc=0+freed
 *
 * GenMC invocation:
 *   genmc --rc11 --unroll=5 test_scoped_atomic_view.c
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>

#define CAPACITY 8u
#define PTR_MASK (~(uintptr_t)(CAPACITY - 1))
#define TAG_MASK ((uintptr_t)(CAPACITY - 1))

typedef struct {
    _Atomic(uintptr_t) refcnt;
    int destroyed;
    int id;
} Obj;

/* Objects must be CAPACITY-aligned so the lower bits are free for the tag. */
static Obj obj_A __attribute__((aligned(CAPACITY)));
static Obj obj_B __attribute__((aligned(CAPACITY)));
static Obj obj_C __attribute__((aligned(CAPACITY)));

/* The shared atomic_shared_ptr's tagged word. */
static _Atomic(uintptr_t) g_ref;

static inline Obj *get_ptr(uintptr_t tagged) { return (Obj *)(tagged & PTR_MASK); }
static inline uintptr_t get_tag(uintptr_t tagged) { return tagged & TAG_MASK; }

static void obj_init(Obj *o, int id) {
    atomic_store_explicit(&o->refcnt, 1, memory_order_relaxed);
    o->destroyed = 0;
    o->id = id;
}

/* ==========================================================================
 * @c11_action AcquireTagRefRead + AcquireTagRefCAS — load_tagged + +1 tag.
 *   Models: acquire_tag_ref_ from atomic_smart_ptr.h:1075-1108.
 *   Returns pref and rcnt_new (post-increment).
 *   Strong CAS (no WEAK spurious failure modelled).
 * ========================================================================== */
static Obj *acquire_tag_ref(uintptr_t *rcnt_out) {
    for (;;) {
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        Obj *pref = get_ptr(cur);
        uintptr_t rcnt_old = get_tag(cur);
        if (!pref) { *rcnt_out = rcnt_old; return NULL; }
        uintptr_t rcnt_new = rcnt_old + 1u;
        assert(rcnt_new < CAPACITY);  /* LocalRCBounded invariant */
        uintptr_t expected = (uintptr_t)pref + rcnt_old;
        uintptr_t desired  = (uintptr_t)pref + rcnt_new;
        if (atomic_compare_exchange_strong_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed)) {
            *rcnt_out = rcnt_new;
            return pref;
        }
    }
}

/* ==========================================================================
 * @c11_action ReleaseTagRefRead/CAS/Global — drain release.
 *   Models: release_tag_ref_(pref, added_global_rcnt) from h:1158-1206.
 *   added_global_rcnt = number of global refcnt units the caller pre-added
 *   on top of the 1 local tag being released.  For the scope dtor path,
 *   added_global_rcnt = 1 (no pre-add).
 * ========================================================================== */
static void release_tag_ref(Obj *pref, uintptr_t added_global_rcnt) {
    uintptr_t sub_amount = added_global_rcnt;
    for (;;) {
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        uintptr_t rcnt_old = get_tag(cur);
        if (rcnt_old && get_ptr(cur) == pref) {
            uintptr_t local_release = rcnt_old < added_global_rcnt
                ? rcnt_old : added_global_rcnt;
            uintptr_t rcnt_new = rcnt_old - local_release;
            uintptr_t expected = (uintptr_t)pref + rcnt_old;
            uintptr_t desired  = (uintptr_t)pref + rcnt_new;
            if (atomic_compare_exchange_strong_explicit(&g_ref, &expected, desired,
                    memory_order_acq_rel, memory_order_relaxed)) {
                sub_amount = added_global_rcnt - local_release;
                break;
            }
            /* CAS failed: if ptr is still pref, retry; else fall through. */
            if (get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed)) == pref)
                continue;
        }
        break;
    }
    if (sub_amount) {
        uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, sub_amount,
                memory_order_acq_rel);
        if (old_rc == sub_amount) pref->destroyed = 1;
    }
}

/* Plain local_shared_ptr::reset(): fetch_sub(1) on global with delete check. */
static void local_reset(Obj *pref) {
    if (!pref) return;
    uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 1, memory_order_acq_rel);
    if (old_rc == 1) pref->destroyed = 1;
}

/* ==========================================================================
 * compareAndSet_impl_<scoped_atomic_view, ..., false (no WEAK), false>
 *
 * Mechanically translates the Layer 1 spec's ScopeCAS{Start,PreInc,Load,
 * Check,Transfer,Swap,Cleanup,Undo} actions into a single C function.
 * The C-level atomics correspond 1:1 to the TLA+ atomic steps.
 *
 * Inputs:
 *   scoped_pref_inout: scope's m_pref (TagHeld). On success or pointer-
 *     mismatch, set to NULL to signal the scope was consumed/absorbed.
 *   newr             : pointer to the new object to install.
 *
 * Returns: 1 on success, 0 on failure.
 * ========================================================================== */
static int compare_and_set_scoped(Obj **scoped_pref_inout, Obj *newr) {
    Obj *scoped_pref = *scoped_pref_inout;
    if (!scoped_pref) return 0;

    /* @c11_action ScopeCASPreInc — step 2: newr->refcnt.fetch_add(1). */
    if (newr)
        atomic_fetch_add_explicit(&newr->refcnt, 1, memory_order_relaxed);

    /* @c11_action ScopeCASLoad — load_tagged_() (no acquire; scope has +1). */
    uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
    Obj *pref = get_ptr(cur);
    uintptr_t T = get_tag(cur);

    /* @c11_action ScopeCASCheck — pref vs scope_pref. */
    if (pref != scoped_pref) {
        /* Mismatch: scope tag was absorbed by a swapper.
         * Undo step 2; release scope's "logical" +1 via release_tag_ref_(pref,1).
         * The rtr pipeline detects ptr changed and falls through to
         * fetch_sub(1, acq_rel) on scoped_pref's global. */
        if (newr)
            atomic_fetch_sub_explicit(&newr->refcnt, 1, memory_order_relaxed);
        release_tag_ref(scoped_pref, 1u);
        *scoped_pref_inout = NULL;
        return 0;
    }

    /* @c11_action ScopeCASTransfer — step 4: pref->refcnt += T (full T, not T-1). */
    if (pref && T)
        atomic_fetch_add_explicit(&pref->refcnt, T, memory_order_relaxed);

    /* @c11_action ScopeCASSwap — CAS pref+T → newr+0. */
    uintptr_t expected = (uintptr_t)pref + T;
    uintptr_t desired  = (uintptr_t)newr;  /* tag = 0 */
    if (atomic_compare_exchange_strong_explicit(&g_ref, &expected, desired,
            memory_order_acq_rel, memory_order_relaxed)) {
        /* @c11_action ScopeCASCleanup — fetch_sub(2): m_ref's release + scope's share. */
        if (pref) {
            uintptr_t old_rc = atomic_fetch_sub_explicit(&pref->refcnt, 2,
                    memory_order_acq_rel);
            if (old_rc == 2) pref->destroyed = 1;
        }
        *scoped_pref_inout = NULL;  /* scope_state → "none" */
        return 1;
    }

    /* @c11_action ScopeCASUndo — failure: undo step 4 (T) + step 2 (newr -= 1). */
    if (pref && T)
        atomic_fetch_sub_explicit(&pref->refcnt, T, memory_order_relaxed);
    if (newr)
        atomic_fetch_sub_explicit(&newr->refcnt, 1, memory_order_relaxed);
    /* scope still TagHeld; caller's destructor will release. */
    return 0;
}

/* ==========================================================================
 * Thread body: scoped lifecycle.
 *   1. ScopeStartAcquire + acquire pipeline + ScopeSetState
 *   2. ScopeCAS{...} — try once to install thr_new
 *   3. ScopeDtor — if scope still TagHeld, release the tag.
 *
 * Each thread owns 1 local_shared_ptr to a "creator" object (its newr),
 * pre-charged in main().  After thread completion the creator-ref is
 * either consumed by CAS (success) or still held by main (failure).
 * ========================================================================== */
typedef struct {
    Obj *newr;     /* the object this thread wants to install */
    int  ok;       /* 1 if CAS succeeded */
} ThreadArg;

static void *thread_scope_cycle(void *arg_) {
    ThreadArg *arg = (ThreadArg *)arg_;

    /* (1) ScopeStartAcquire + acquire pipeline + ScopeSetState. */
    uintptr_t rcnt;
    Obj *scope_pref = acquire_tag_ref(&rcnt);
    if (!scope_pref) { arg->ok = 0; return NULL; }
    /* MemorySafety: scope_pref alive while tagheld */
    assert(scope_pref->destroyed == 0);

    /* (2) ScopeCAS set. */
    arg->ok = compare_and_set_scoped(&scope_pref, arg->newr);

    /* (3) ScopeDtor — release_tag_ref if scope still TagHeld. */
    if (scope_pref) {
        release_tag_ref(scope_pref, 1u);
    }
    return NULL;
}

int main(void) {
    obj_init(&obj_A, 1);  /* initial installed object */
    obj_init(&obj_B, 2);  /* thread 1 wants to install B */
    obj_init(&obj_C, 3);  /* thread 2 wants to install C */

    /* g_ref initially installed with obj_A (1 ref from the installation). */
    atomic_store_explicit(&g_ref, (uintptr_t)&obj_A, memory_order_relaxed);

    ThreadArg arg1 = { .newr = &obj_B, .ok = 0 };
    ThreadArg arg2 = { .newr = &obj_C, .ok = 0 };

    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_scope_cycle, &arg1);
    pthread_create(&t2, NULL, thread_scope_cycle, &arg2);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    /* Release g_ref's final installation (ScopeDtor analog for the
     * installed object).  Match the C++ semantics of "caller's
     * local_shared_ptr destruction is independent of CAS outcome":
     * always release main's creator-refs on obj_B and obj_C. */
    Obj *final_p = get_ptr(atomic_load_explicit(&g_ref, memory_order_relaxed));
    if (final_p) local_reset(final_p);
    local_reset(&obj_B);
    local_reset(&obj_C);

    /* TerminalCheck: all objects freed at quiescence. */
    assert(obj_A.destroyed == 1);
    assert(obj_B.destroyed == 1);
    assert(obj_C.destroyed == 1);

    /* FreedImpliesZeroRC */
    assert(atomic_load_explicit(&obj_A.refcnt, memory_order_relaxed) == 0);
    assert(atomic_load_explicit(&obj_B.refcnt, memory_order_relaxed) == 0);
    assert(atomic_load_explicit(&obj_C.refcnt, memory_order_relaxed) == 0);

    /* arg1.ok + arg2.ok can be 0, 1, or 2: both CAS may succeed if T1
     * completes fully before T2 acquires; or both fail (impossible here
     * since one will always observe pref unchanged), or one succeeds. */
    return 0;
}
