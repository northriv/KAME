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
 * C11 test generated mechanically from atomic_shared_ptr.tla (Layer 0)
 *
 * Models the tagged-pointer scheme: an atomic word packs (ptr, local_rc).
 * Three threads perform load_shared / compareAndSwap_ / swap concurrently.
 * Invariants from TLA+ are checked via assert().
 *
 * TLA+ variable mapping:
 *   ptr, local_rc       → packed in _Atomic(uintptr_t) g_ref
 *   global_rc[o]        → _Atomic(uintptr_t) obj_refcnt[o]
 *   freed[o]            → int obj_freed[o]
 *   thr_pref, thr_rcnt  → thread-local variables
 *   thr_holds[t][o]     → thread-local hold counters
 */

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <assert.h>
#include <stdbool.h>

/* --- Configuration --- */
#define NUM_OBJECTS   3
#define CAPACITY      8u
#define PTR_MASK      (~(uintptr_t)(CAPACITY - 1))
#define TAG_MASK      ((uintptr_t)(CAPACITY - 1))

/* --- Object pool (simulating Ref objects) --- */
/* Object addresses must be aligned to CAPACITY so lower bits are free for tag */
static uintptr_t obj_addr[NUM_OBJECTS];    /* "addresses" of objects */
_Atomic(uintptr_t) obj_refcnt[NUM_OBJECTS]; /* global reference counts */
int obj_freed[NUM_OBJECTS];                 /* freed flags (non-atomic) */

/* --- The atomic shared pointer: packs (ptr | local_rc) --- */
_Atomic(uintptr_t) g_ref;

/* --- Helpers --- */
static int obj_index(uintptr_t addr) {
    for (int i = 0; i < NUM_OBJECTS; i++)
        if (obj_addr[i] == addr) return i;
    return -1;  /* NULL */
}

static uintptr_t get_ptr(uintptr_t val) { return val & PTR_MASK; }
static unsigned get_tag(uintptr_t val)   { return (unsigned)(val & TAG_MASK); }

static uintptr_t make_ref(uintptr_t addr, unsigned tag) {
    return addr | (uintptr_t)tag;
}

/* --- acquire_tag_ref_: atomically read ptr+tag, CAS to increment tag --- */
/* Returns pointer (or 0 for NULL), sets *out_rcnt to the post-CAS tag */
static uintptr_t acquire_tag_ref(unsigned *out_rcnt) {
    for (;;) {
        /* AcquireTagRefRead: atomic load of g_ref */
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        uintptr_t pref = get_ptr(cur);
        unsigned rcnt = get_tag(cur);

        if (pref == 0) {
            /* AcquireTagRefNull */
            *out_rcnt = rcnt;
            return 0;
        }

        /* AcquireTagRefCAS: CAS (pref+rcnt) -> (pref+rcnt+1) */
        uintptr_t expected = make_ref(pref, rcnt);
        uintptr_t desired  = make_ref(pref, rcnt + 1);
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed)) {
            *out_rcnt = rcnt + 1;
            return pref;
        }
        /* CAS failed → retry (atr_read) */
    }
}

/* --- release_tag_ref_: try CAS to decrement tag, fallback to global dec --- */
static void release_tag_ref(uintptr_t pref, bool is_load_done, int *hold_idx) {
    if (pref == 0) return;
    int idx = obj_index(pref);
    assert(idx >= 0);

    for (;;) {
        /* ReleaseTagRefRead */
        uintptr_t cur = atomic_load_explicit(&g_ref, memory_order_relaxed);
        unsigned rcnt = get_tag(cur);

        if (rcnt > 0) {
            /* ReleaseTagRefCAS: try CAS (pref+rcnt) -> (pref+rcnt-1) */
            uintptr_t expected = make_ref(pref, rcnt);
            uintptr_t desired  = make_ref(pref, rcnt - 1);
            if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                    memory_order_acq_rel, memory_order_relaxed)) {
                /* Success */
                if (is_load_done && hold_idx) *hold_idx = idx;
                return;
            }
            /* CAS failed: check if ptr changed */
            if (get_ptr(expected) == pref) {
                continue;  /* ptr unchanged, retry */
            }
            /* ptr changed → fall through to global dec */
        }

        /* ReleaseTagRefGlobal: dec global refcount */
        uintptr_t old_rc = atomic_fetch_sub_explicit(&obj_refcnt[idx], 1,
                memory_order_acq_rel);
        if (old_rc == 1) {
            obj_freed[idx] = 1;
        }
        if (is_load_done && hold_idx) *hold_idx = idx;
        return;
    }
}

/* --- load_shared_(): acquire_tag_ref + inc global + release_tag_ref --- */
/* Returns object index, or -1 for NULL */
static int load_shared(void) {
    unsigned rcnt;
    uintptr_t pref = acquire_tag_ref(&rcnt);

    if (pref == 0) return -1;

    int idx = obj_index(pref);
    assert(idx >= 0);

    /* LoadSharedIncGlobal */
    atomic_fetch_add_explicit(&obj_refcnt[idx], 1, memory_order_relaxed);

    /* LoadSharedStartRelease → release_tag_ref_ */
    int hold = -1;
    release_tag_ref(pref, true, &hold);

    /* MemorySafety: after acquire completes, object not freed */
    assert(obj_freed[idx] == 0);

    return hold;
}

/* --- reset(): release a local_shared_ptr hold --- */
static void reset_hold(int idx) {
    if (idx < 0) return;
    uintptr_t old_rc = atomic_fetch_sub_explicit(&obj_refcnt[idx], 1,
            memory_order_acq_rel);
    if (old_rc == 1) {
        obj_freed[idx] = 1;
    }
}

/* --- compareAndSwap_(): CAS the installed pointer --- */
/* Returns true on success, false on mismatch */
static bool compare_and_swap(uintptr_t old_ptr, uintptr_t new_ptr) {
    int new_idx = -1;
    if (new_ptr != 0) {
        new_idx = obj_index(new_ptr);
        assert(new_idx >= 0);
    }

    /* CASPreInc: pre-increment newr's global_rc */
    if (new_idx >= 0) {
        atomic_fetch_add_explicit(&obj_refcnt[new_idx], 1, memory_order_relaxed);
    }

    for (;;) {
        /* CASReserve → acquire_tag_ref_ */
        unsigned rcnt;
        uintptr_t pref = acquire_tag_ref(&rcnt);

        /* CASCheck: pref == oldr? */
        if (pref != old_ptr) {
            /* Mismatch → release_tag_ref, rollback newr */
            if (pref != 0) {
                release_tag_ref(pref, false, NULL);
            }
            /* CASFailDone: rollback newr's pre-increment */
            if (new_idx >= 0) {
                uintptr_t old_rc = atomic_fetch_sub_explicit(&obj_refcnt[new_idx], 1,
                        memory_order_relaxed);
                if (old_rc == 1) {
                    obj_freed[new_idx] = 1;
                }
            }
            return false;
        }

        /* CASTransfer: transfer local_rc to global_rc */
        int pref_idx = obj_index(pref);
        if (pref != 0 && rcnt != 1) {
            atomic_fetch_add_explicit(&obj_refcnt[pref_idx], rcnt - 1,
                    memory_order_relaxed);
        }

        /* CASSwap: CAS (pref+rcnt) -> (newr+0) */
        uintptr_t expected = make_ref(pref, rcnt);
        uintptr_t desired  = make_ref(new_ptr, 0);
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed)) {
            /* Success → CASCleanup: dec pref's global_rc */
            if (pref != 0 && pref_idx >= 0) {
                uintptr_t old_rc = atomic_fetch_sub_explicit(
                        &obj_refcnt[pref_idx], 1, memory_order_acq_rel);
                if (old_rc == 1) {
                    obj_freed[pref_idx] = 1;
                }
            }
            return true;
        }

        /* CAS failed → CASUndo: undo transfer, release_tag_ref, retry */
        if (pref != 0) {
            if (rcnt != 1) {
                atomic_fetch_sub_explicit(&obj_refcnt[pref_idx], rcnt - 1,
                        memory_order_relaxed);
            }
            release_tag_ref(pref, false, NULL);
        }
        /* retry from acquire_tag_ref */
    }
}

/* --- swap(): unconditional exchange, no PreInc/Cleanup --- */
/* TLA+ StartSwap: skip cas_pre_inc, go to cas_acquire.
 * CASSwap success (swap path): hold transfer is immediate
 *   (this->m_ref = pref is local write) → done. No CASCleanup.
 * Returns the index of the old object taken from g_ref, or -1 for NULL.
 */
static int swap(uintptr_t new_ptr) {
    int new_idx = -1;
    if (new_ptr != 0) {
        new_idx = obj_index(new_ptr);
        assert(new_idx >= 0);
    }

    /* No PreInc — swap exchanges ownership, doesn't add a new ref */

    for (;;) {
        /* cas_acquire → acquire_tag_ref_ */
        unsigned rcnt;
        uintptr_t pref = acquire_tag_ref(&rcnt);

        /* CASTransfer: transfer local_rc to global_rc */
        int pref_idx = (pref != 0) ? obj_index(pref) : -1;
        if (pref != 0 && rcnt != 1) {
            atomic_fetch_add_explicit(&obj_refcnt[pref_idx], rcnt - 1,
                    memory_order_relaxed);
        }

        /* CASSwap: CAS (pref+rcnt) -> (newr+0) */
        uintptr_t expected = make_ref(pref, rcnt);
        uintptr_t desired  = make_ref(new_ptr, 0);
        if (atomic_compare_exchange_weak_explicit(&g_ref, &expected, desired,
                memory_order_acq_rel, memory_order_relaxed)) {
            /* Success (swap path): hold transfer is immediate.
             * Caller gives up newObj, takes pref.
             * No CASCleanup — the refcount is already correct:
             *   pref's global_rc includes the transferred local refs,
             *   and the caller now owns that reference (hold).
             *   newObj's global_rc was NOT pre-incremented, and
             *   the caller releases its hold (the ref goes to g_ref). */
            return pref_idx;
        }

        /* CAS failed → CASUndo: undo transfer, release_tag_ref_, retry */
        if (pref != 0) {
            if (rcnt != 1) {
                atomic_fetch_sub_explicit(&obj_refcnt[pref_idx], rcnt - 1,
                        memory_order_relaxed);
            }
            release_tag_ref(pref, false, NULL);
        }
        /* retry from acquire_tag_ref */
    }
}

/* ========================================================================== */
/* Test threads                                                               */
/* ========================================================================== */

/*
 * Thread 0: performs load_shared → reset (exercises StartLoadShared through
 * ReleaseTagRef, then Reset)
 */
static void *thread_load(void *arg) {
    (void)arg;
    int hold = load_shared();
    if (hold >= 0) {
        /* NoUseAfterFree: held object not freed */
        assert(obj_freed[hold] == 0);
        reset_hold(hold);
    }
    return NULL;
}

/*
 * Thread 1: performs compareAndSwap_ (exercises StartCAS through CASCleanup)
 * Tries to CAS obj[0] → obj[1]
 */
static void *thread_cas(void *arg) {
    (void)arg;
    /* thr_old = obj[0], thr_new = obj[1] (held by this thread initially) */
    bool ok = compare_and_swap(obj_addr[0], obj_addr[1]);
    (void)ok;
    return NULL;
}

/*
 * Thread 2: performs swap (exercises StartSwap → CASTransfer → CASSwap)
 * Unconditionally exchanges g_ref with obj[2].
 * No PreInc, no CASCleanup — ownership transfer is immediate.
 */
static void *thread_swap(void *arg) {
    (void)arg;
    /* thr_new = obj[2] (held by this thread initially) */
    int old_idx = swap(obj_addr[2]);
    if (old_idx >= 0) {
        /* NoUseAfterFree: the object we took from g_ref is not freed */
        assert(obj_freed[old_idx] == 0);
        /* Release the hold we acquired (like local_shared_ptr destructor) */
        reset_hold(old_idx);
    }
    return NULL;
}

int main(void) {
    /* --- Setup: aligned "addresses" for objects --- */
    /* Use addresses that are CAPACITY-aligned so lower bits are 0 */
    obj_addr[0] = (uintptr_t)CAPACITY * 2;   /* e.g. 16 */
    obj_addr[1] = (uintptr_t)CAPACITY * 3;   /* e.g. 24 */
    obj_addr[2] = (uintptr_t)CAPACITY * 4;   /* e.g. 32 */

    /* Init: obj[0] is installed in g_ref,
     *        obj[1] is held by thread 1 (CAS),
     *        obj[2] is held by thread 2 (swap) */
    /* global_rc: obj[0]=1 (installed), obj[1]=1 (held), obj[2]=1 (held) */
    atomic_store(&obj_refcnt[0], 1);
    atomic_store(&obj_refcnt[1], 1);
    atomic_store(&obj_refcnt[2], 1);
    obj_freed[0] = 0;
    obj_freed[1] = 0;
    obj_freed[2] = 0;

    /* g_ref = obj_addr[0] | 0 (local_rc = 0) */
    atomic_store(&g_ref, make_ref(obj_addr[0], 0));

    pthread_t t0, t1, t2;
    pthread_create(&t0, NULL, thread_load, NULL);
    pthread_create(&t1, NULL, thread_cas, NULL);
    pthread_create(&t2, NULL, thread_swap, NULL);

    pthread_join(t0, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    /* --- Post-join invariants --- */

    /* InstalledNotFreed: the currently installed object is not freed */
    uintptr_t final_ref = atomic_load(&g_ref);
    uintptr_t final_ptr = get_ptr(final_ref);
    if (final_ptr != 0) {
        int idx = obj_index(final_ptr);
        assert(idx >= 0);
        assert(obj_freed[idx] == 0);
    }

    /* GlobalRCNonNeg / FreedImpliesZeroRC */
    for (int i = 0; i < NUM_OBJECTS; i++) {
        uintptr_t rc = atomic_load(&obj_refcnt[i]);
        if (!obj_freed[i]) {
            assert(rc >= 0);  /* GlobalRCNonNeg (always true for unsigned) */
        }
        if (obj_freed[i]) {
            assert(rc == 0);  /* FreedImpliesZeroRC */
        }
    }

    /* QuiescentCheck: when all threads idle:
     *   freed[o] == (global_rc[o] == 0) for all objects
     *   ptr != NULL => global_rc[ptr] >= 1 */
    for (int i = 0; i < NUM_OBJECTS; i++) {
        uintptr_t rc = atomic_load(&obj_refcnt[i]);
        assert(obj_freed[i] == (rc == 0));
    }
    if (final_ptr != 0) {
        int idx = obj_index(final_ptr);
        assert(idx >= 0);
        assert(atomic_load(&obj_refcnt[idx]) >= 1);
    }

    return 0;
}
