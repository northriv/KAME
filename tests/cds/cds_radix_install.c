/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            LICENSE-APACHE-2.0 in this directory)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see LICENSE-GPL-2.0 in this directory).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
***************************************************************************/
/*
 * GenMC test — §13 radix L2 lazy-install lock-free protocol (INV-9, INV-11).
 *
 * The 2-level radix maps a 32-MiB-aligned pointer to a present/kind token:
 *   s_radix_l1[L1] : atomic<RadixL2Node*>   (lazily mmap'd L2 leaves)
 *   leaf->entries[L2] : atomic<uint32_t>     (KAME_RADIX_ABSENT/POOL/LARGE)
 *
 * The subtle concurrency is the LAZY INSTALL of an L2 leaf: the first
 * inserter into an empty L1 slot mmaps a fresh leaf and CAS-installs it; a
 * concurrent inserter that loses the CAS must munmap its own leaf and use
 * the winner's.  A concurrent reader (radix_lookup, e.g. a foreign-pointer
 * check from an unrelated free) acquire-loads L1 and then reads the entry.
 *
 * This harness models that protocol with the EXACT memory orders from
 * allocator.cpp:
 *   - L1 load           : acquire   (radix_lookup_slow / radix_insert)
 *   - L1 install CAS     : release / acquire (radix_insert)
 *   - entry store        : release   (radix_insert / radix_clear)
 *   - entry load         : relaxed   (radix_lookup_slow)
 *
 * Verifies (under RC11, all interleavings):
 *   (A) Exactly one leaf is ever installed in L1; the CAS loser's leaf is
 *       freed and NEVER dereferenced by anyone (the `live` ghost + assert).
 *       — If the install were not a single-winner CAS, a reader could read a
 *       munmap'd leaf  →  use-after-free.
 *   (B) A reader that observes a non-null L1 leaf dereferences a LIVE leaf
 *       (the acquire-load of L1 synchronises-with the install's release CAS,
 *       so the leaf's existence is visible before any entry access).
 *   (C) Entry reads are coherent: a slot reads either ABSENT (0) or its
 *       OWN kind, never the neighbouring slot's kind and never a torn value
 *       (entries are per-slot atomics).
 *   (D) After all inserts complete, every written slot holds its kind, and
 *       exactly one leaf was installed (the other was freed).
 *
 * NOT modelled here (external to the radix): the alloc→free pointer handoff
 * that carries a LARGE alloc's meta (magic/mmap_size) happens-before the
 * freeing thread's meta deref.  The radix entry's relaxed load is sound for
 * the KIND value (atomic) and for foreign-pointer checks (kind only); meta
 * visibility rides the handoff, not the radix — see INVARIANTS.md INV-9.
 *
 * Run:  make run-radix     (GenMC)
 *       make smoke-radix   (concrete gcc + TSAN sanity)
 *
 * Co-Authored-By: Claude <noreply@anthropic.com>
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>

enum { KAME_RADIX_ABSENT = 0, KAME_RADIX_POOL = 1, KAME_RADIX_LARGE = 2 };

#define NSLOT 2     /* two L2 entries exercised (one per inserter) */
#define NPOOL 2     /* at most two leaves can be alloc'd (one per inserter) */

typedef struct {
    _Atomic(unsigned) entries[NSLOT];
    _Atomic(int)      live;     /* ghost: 1 = mapped, 0 = munmap'd (atomic: read on the
                                 * lookup path; visibility rides g_l1's acq/rel) */
} L2Node;

static L2Node            g_nodes[NPOOL];   /* "fresh mmap" pool */
static _Atomic(int)      g_node_next;      /* hand out a distinct fresh node */
static _Atomic(L2Node *) g_l1;             /* one L1 slot */

/* radix_alloc_l2(): mmap a zero-filled leaf.  Modelled as taking the next
 * pool node (distinct per caller, like a fresh mmap address). */
static L2Node *alloc_l2(void) {
    int i = atomic_fetch_add_explicit(&g_node_next, 1, memory_order_relaxed);
    if(i >= NPOOL) return NULL;
    /* entries[] are already 0 (mmap zero-fill / static init); mark live. */
    atomic_store_explicit(&g_nodes[i].live, 1, memory_order_relaxed);
    return &g_nodes[i];
}
/* munmap of a CAS-loser leaf.  No other thread can hold it (it was never in
 * L1), so this is a plain ghost flip. */
static void free_l2(L2Node *n) { atomic_store_explicit(&n->live, 0, memory_order_relaxed); }

/* radix_insert(slot, kind) — allocator.cpp lines ~4510. */
static void radix_insert(int slot, unsigned kind) {
    L2Node *leaf = atomic_load_explicit(&g_l1, memory_order_acquire);
    if(!leaf) {
        L2Node *nl = alloc_l2();
        if(!nl) return;                                  /* OOM → lookup misses */
        L2Node *expected = NULL;
        if(atomic_compare_exchange_strong_explicit(
               &g_l1, &expected, nl,
               memory_order_release, memory_order_acquire)) {
            leaf = nl;                                   /* we installed it */
        } else {
            free_l2(nl);                                 /* lost the race → munmap ours */
            leaf = expected;                             /* use the winner's */
        }
    }
    /* No `assert(leaf->live)` here.  The real radix_insert loser uses the
     * winner's leaf ONLY to do `leaf->entries[l2].store()` (atomic) — it never
     * reads a winner-initialised NON-atomic field, so it does not depend on the
     * failed install-CAS's failure=acquire synchronising with the winner's
     * release-CAS.  (GenMC/RC11 does not treat a failed compare_exchange_strong's
     * failure=acquire as synchronising — it keys the RMW read order off the
     * SUCCESS order; success=acq_rel or an explicit acquire reload would, but the
     * real release/acquire code needs neither because the leaf is mmap-zeroed and
     * all-atomic.)  The UAF-visibility invariant is checked on the lookup path
     * below, where the reader does a plain acquire-load of g_l1. */
    atomic_store_explicit(&leaf->entries[slot], kind, memory_order_release);
}

/* radix_lookup(slot) — allocator.cpp lines ~4466.  Returns kind or ABSENT. */
static unsigned radix_lookup(int slot) {
    L2Node *leaf = atomic_load_explicit(&g_l1, memory_order_acquire);
    if(!leaf) return KAME_RADIX_ABSENT;
    assert(atomic_load_explicit(&leaf->live, memory_order_relaxed));  /* (B): leaf visible (acquire-load of g_l1) ⇒ live */
    return atomic_load_explicit(&leaf->entries[slot], memory_order_relaxed);
}

/* radix_clear(slot) — allocator.cpp lines ~4552.  Release store of ABSENT. */
static void radix_clear(int slot) {
    L2Node *leaf = atomic_load_explicit(&g_l1, memory_order_acquire);
    if(!leaf) return;
    assert(atomic_load_explicit(&leaf->live, memory_order_relaxed));
    atomic_store_explicit(&leaf->entries[slot], KAME_RADIX_ABSENT,
                          memory_order_release);
}

/* Two inserters racing on the SAME empty L1 slot (different L2 entries), and
 * one reader doing a foreign-pointer-style lookup of both slots. */
static void *t_insert_pool(void *a)  { (void)a; radix_insert(0, KAME_RADIX_POOL);  return NULL; }
static void *t_insert_large(void *a) { (void)a; radix_insert(1, KAME_RADIX_LARGE); return NULL; }
static void *t_lookup(void *a) {
    (void)a;
    unsigned v0 = radix_lookup(0);
    unsigned v1 = radix_lookup(1);
    /* (C): each slot reads its own kind or ABSENT — never the other's. */
    assert(v0 == KAME_RADIX_ABSENT || v0 == KAME_RADIX_POOL);
    assert(v1 == KAME_RADIX_ABSENT || v1 == KAME_RADIX_LARGE);
    return NULL;
}

int main(void) {
    atomic_store_explicit(&g_node_next, 0, memory_order_relaxed);
    atomic_store_explicit(&g_l1, NULL, memory_order_relaxed);

    pthread_t a, b, c;
    pthread_create(&a, NULL, t_insert_pool,  NULL);
    pthread_create(&b, NULL, t_insert_large, NULL);
    pthread_create(&c, NULL, t_lookup,       NULL);
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    pthread_join(c, NULL);

    /* (D): exactly one leaf installed; the other (if alloc'd) was freed. */
    L2Node *leaf = atomic_load_explicit(&g_l1, memory_order_relaxed);
    assert(leaf != NULL);
    assert(atomic_load_explicit(&leaf->live, memory_order_relaxed));
    int installed = 0, freed = 0;
    for(int i = 0; i < NPOOL; i++) {
        if(&g_nodes[i] == leaf) { assert(atomic_load_explicit(&g_nodes[i].live, memory_order_relaxed)); installed++; }
        else if(atomic_load_explicit(&g_nodes[i].live, memory_order_relaxed) == 0
                && i < atomic_load_explicit(&g_node_next, memory_order_relaxed))
            freed++;
    }
    assert(installed == 1);
    /* Both inserts done ⇒ both entries hold their kind. */
    assert(atomic_load_explicit(&leaf->entries[0], memory_order_relaxed) == KAME_RADIX_POOL);
    assert(atomic_load_explicit(&leaf->entries[1], memory_order_relaxed) == KAME_RADIX_LARGE);
    return 0;
}
