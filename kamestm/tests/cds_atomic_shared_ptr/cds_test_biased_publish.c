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
 * GenMC test 10: biased reference counting — private→shared DIRECT publish.
 *
 * Models the gated `-DKAME_LSP_BIASED=1` path of kamepoolalloc/atomic_smart_ptr.h
 * for the ONLY configuration the scheme is sound in: a control block that is
 * born PRIVATE on its owner thread, churned NON-atomically (relaxed load+store)
 * while private, then DIRECTLY published — its pointer installed as the value of
 * an atomic_shared_ptr slot — at which point PRIV is cleared (fetch_and, release)
 * sequenced-before the slot install, and every subsequent refcnt op is atomic.
 *
 * Negative-logic PRIV bit (MSB): SET ⇒ private (owner-only, non-atomic count);
 * CLEAR ⇒ shared (word == plain count, atomic fetch_add / decAndTest).
 *
 * Property verified across ALL RC11 interleavings:
 *   - no use-after-free / double-free (Obj.destroyed observed only as 0 by a
 *     reader holding a ref; destroyed exactly once at the end);
 *   - refcount accounting balances (final refcnt == 0);
 *   - the owner's pre-publish NON-atomic stores are never concurrent with the
 *     reader's atomic ops, because the reader cannot observe the slot pointer
 *     until the release-store that publishes it, which is sequenced-after the
 *     PRIV-clear and all private churn (publish→acquire happens-before chain).
 *
 * This DOES NOT model transitive containment (a private CB reached only by being
 * a member of a published payload) — that configuration is UNSOUND under Plan A
 * (empirically leaks / UAFs) and is out of scope here; see the soundness-boundary
 * note in atomic_smart_ptr.h.
 */

#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

#define CAPACITY 8u
#define PTR_MASK (~(uintptr_t)(CAPACITY - 1))

/* Negative-logic PRIVate bit in the strong refcnt MSB (matches KAME_LSP_PRIV). */
#define LSP_PRIV ((uintptr_t)1 << (sizeof(uintptr_t) * 8 - 1))
#define LSP_CNT(v) ((v) & ~LSP_PRIV)

typedef struct {
    _Atomic(uintptr_t) refcnt;   /* PRIV|count while private; plain count when shared */
    int data;
    int destroyed;
} Obj;

/* Shared atomic_shared_ptr slot (just a pointer; no load_shared_ tag traffic here). */
static _Atomic(uintptr_t) g_slot;

static Obj g_obj __attribute__((aligned(CAPACITY)));

static Obj *slot_ptr(uintptr_t tagged) { return (Obj *)(tagged & PTR_MASK); }

/* (§biased) owner-private copy: PRIV set ⇒ relaxed load+store (non-atomic-equiv,
 * ldr/str on arm64); else the shared atomic fetch_add. */
static void private_or_shared_copy(Obj *o) {
    uintptr_t v = atomic_load_explicit(&o->refcnt, memory_order_relaxed);
    if (v & LSP_PRIV)
        atomic_store_explicit(&o->refcnt, v + 1, memory_order_relaxed);
    else
        atomic_fetch_add_explicit(&o->refcnt, 1, memory_order_relaxed);
}

/* (§biased) reset: PRIV set ⇒ relaxed store; else decAndTest (acq_rel). */
static void private_or_shared_reset(Obj *o) {
    uintptr_t v = atomic_load_explicit(&o->refcnt, memory_order_relaxed);
    if (v & LSP_PRIV) {
        if (LSP_CNT(v) == 1) {
            atomic_store_explicit(&o->refcnt, 0, memory_order_relaxed);
            o->destroyed = 1;
        } else {
            atomic_store_explicit(&o->refcnt, v - 1, memory_order_relaxed);
        }
        return;
    }
    if (atomic_fetch_sub_explicit(&o->refcnt, 1, memory_order_acq_rel) == 1)
        o->destroyed = 1;
}

/* Owner thread: born owning 1 ref.  Make the slot's ref while still private
 * (owner-only, non-atomic), PUBLISH (clear PRIV release + install pointer
 * release), then KEEP CHURNING its own ref post-publish and finally release it.
 *
 * The post-publish churn is what gives this test teeth: after a CORRECT publish
 * PRIV is clear, so private_or_shared_copy/reset take the ATOMIC branch and can
 * safely run concurrently with the reader.  If the publish FAILED to clear PRIV
 * (delete the fetch_and below), the owner's churn and the reader's churn both
 * take the NON-atomic relaxed-store branch on the now-shared CB — GenMC then
 * finds a lost-update interleaving and the final refcnt/destroyed asserts fail. */
static void *thread_owner(void *arg) {
    (void)arg;
    Obj *o = &g_obj;
    /* add the slot's reference while still owner-private (non-atomic): 1 -> 2 */
    private_or_shared_copy(o);
    /* PUBLISH: clear PRIV (release) sequenced-before the pointer install (release). */
    atomic_fetch_and_explicit(&o->refcnt, ~LSP_PRIV, memory_order_release);
    atomic_store_explicit(&g_slot, (uintptr_t)o, memory_order_release);
    /* post-publish: churn the owner's own ref via the PRIV-branching ops. */
    private_or_shared_copy(o);
    private_or_shared_reset(o);
    /* release the owner's own ref (the slot keeps its ref). */
    private_or_shared_reset(o);
    return NULL;
}

/* Reader thread: acquire the slot; if published, take+drop a ref via the same
 * PRIV-branching ops the owner uses (post-publish these must be the atomic path). */
static void *thread_reader(void *arg) {
    (void)arg;
    Obj *o = slot_ptr(atomic_load_explicit(&g_slot, memory_order_acquire));
    if (o) {
        private_or_shared_copy(o);
        assert(o->destroyed == 0);     /* must be alive while we hold a ref */
        int v = o->data; (void)v;
        private_or_shared_reset(o);
    }
    return NULL;
}

int main(void) {
    atomic_store_explicit(&g_obj.refcnt, LSP_PRIV | 1u, memory_order_relaxed); /* born private */
    g_obj.data = 101;
    g_obj.destroyed = 0;
    atomic_store_explicit(&g_slot, 0, memory_order_relaxed);

    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_owner, NULL);
    pthread_create(&t2, NULL, thread_reader, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    /* Release the slot's reference — last holder, drives refcnt to 0. */
    Obj *final_p = slot_ptr(atomic_load_explicit(&g_slot, memory_order_acquire));
    if (final_p)
        private_or_shared_reset(final_p);

    /* Exactly-once destruction and balanced refcount across every interleaving. */
    assert(g_obj.destroyed == 1);
    assert(atomic_load_explicit(&g_obj.refcnt, memory_order_relaxed) == 0);
    return 0;
}
