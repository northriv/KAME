/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            kamepoolalloc/LICENSE-APACHE-2.0)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (see kamepoolalloc/LICENSE-GPL-2.0).
***************************************************************************/

// alloc_thread_exit_unarmed_test — regression for the post-30ea1daa /
// 3145e139 narrow gap in the large/dedicated L1 recycle teardown guard.
//
// Scenario the test forces (the SUFFICIENT condition the guard must cover):
//
//   * A PRODUCER thread allocates a > 32 KiB block (dedicated tier) and
//     hands its address to the main "consumer" thread via a pthread_key TSD.
//   * The producer exits.  Its L1 recycle cache was armed (it did pool
//     allocations) and gets drained by `l1_drain`.
//   * The CONSUMER thread is a separate worker that does NO pool
//     allocations of its own (so its L1 recycle cache is NEVER armed —
//     `s_l1_drained` stays false).  Its only pool interaction is to free,
//     in its OWN pthread_key destructor at thread exit, the block the
//     producer handed it.
//   * That free is a cross-thread large/dedicated free arriving from a
//     pthread_key dtor on a never-armed thread.  Before 3145e139 it would
//     `recycle_push -> l1_push`, ARM a fresh L1 on the dying thread, and
//     strand the block (l1_drain's dtor either never registers or runs
//     first), permanently leaving the chunk units claimed.
//
// Repeat this consumer-spawn cycle.  With the guard, `chunks_live` /
// `units_live` plateau; without it, +1 chunk/cycle monotonically.
//
// Why the existing `alloc_thread_exit_free_test` does NOT catch this:
//   - Scenario A frees in-body (no teardown free path).
//   - Scenario B's free happens in the SAME thread that allocated, so the
//     thread DID arm the L1 (l1_base() was called by allocate_dedicated_chunk)
//     and s_l1_drained closes the gap (l1_push refuses post-drain).
//   - Scenario C is per-thread alloc+free in body.
// None spawn a non-allocating consumer thread whose only free is a
// teardown release of a cross-thread-origin large block.  This file does.
//
// Pool-only (kame_pool_* C API); built when USE_KAME_ALLOCATOR is ON.
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "kame_pool.h"

#include <pthread.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

static int g_fails = 0;
#define CHECK(c, ...) do { if(!(c)) { ++g_fails; \
    std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while(0)

// Dedicated tier (> 32 KiB ALLOC_MAX_BUCKETED_SIZE).  Chosen so each cycle
// allocates exactly ONE dedicated chunk.
static const std::size_t LARGE_SIZE = 48 * 1024;

// Consumer thread receives a pointer here; its pthread_key destructor frees it
// at thread exit.  The destructor runs AFTER the C++ thread_local dtors that
// run l1_drain — so the free path on the consumer side is the regression case:
//   * the consumer has never armed its L1 recycle cache (it does no
//     allocations of its own),
//   * the free is for a foreign (producer-allocated) > 32 KiB block.
// Both pre-conditions the 3145e139 guard targets.
static pthread_key_t g_consumer_tsd_key;
static void consumer_tsd_dtor(void *v) {
    if(v) kame_pool_free(v);   // dedicated free routed through deallocate
}

static std::size_t units_live() {
    kame_pool_stats_t s; std::memset(&s, 0, sizeof s); s.version = 2;
    kame_pool_get_stats(&s);
    return s.units_live;
}
static std::size_t chunks_live() {
    kame_pool_stats_t s; std::memset(&s, 0, sizeof s); s.version = 2;
    kame_pool_get_stats(&s);
    return s.chunks_live;
}

static const int CYCLES = 120, WARMUP = 20;

static bool grew(std::size_t base, std::size_t cur, std::size_t slack) {
    return cur > base + slack;
}

int main() {
    pthread_key_create(&g_consumer_tsd_key, consumer_tsd_dtor);

    std::size_t base_u = 0, base_c = 0;
    for(int cyc = 0; cyc < CYCLES; cyc++) {
        // Hand-off across two short-lived threads: producer allocates,
        // consumer frees at its OWN exit via the pthread_key dtor.  Both
        // joined in-cycle so units_live is measured at a quiescent point.
        std::atomic<void *> handoff{nullptr};

        std::thread producer([&] {
            void *p = kame_pool_malloc(LARGE_SIZE);
            // Touch first byte so the chunk is RSS-live (the leak signal is
            // on the BITMAP regardless, but this keeps the test honest).
            if(p) ((char *)p)[0] = 0xAA;
            handoff.store(p, std::memory_order_release);
        });
        producer.join();

        std::thread consumer([&] {
            // CRITICAL: this thread does NO pool allocations of its own.
            // Its `tls_l1` (recycle-cache L1) stays nullptr; `s_l1_drained`
            // stays false; `tls_l1_drain`'s thread_local dtor never armed.
            // The ONLY pool interaction is the deferred large-tier free
            // below — exactly the path 3145e139 fixed.
            void *p = handoff.load(std::memory_order_acquire);
            pthread_setspecific(g_consumer_tsd_key, p);
            // Falling off main() runs the pthread_key dtor → free(p).
        });
        consumer.join();

        if(cyc == WARMUP) { base_u = units_live(); base_c = chunks_live(); }
        if(cyc >= WARMUP && (cyc % 20 == 0 || cyc == CYCLES - 1))
            std::printf("  cyc %3d: units_live=%6zu chunks_live=%6zu\n",
                        cyc, units_live(), chunks_live());
    }

    std::size_t end_u = units_live(), end_c = chunks_live();
    // Pre-fix: ~ +1 chunk per (CYCLES - WARMUP) = +100 over 100 cycles, far
    // beyond the small fragmentation slack.  Post-fix: plateau.
    CHECK(!grew(base_u, end_u, 16),
          "unarmed-consumer dedicated free STRANDED: units_live grew %zu -> %zu "
          "over %d cycles (free of cross-thread > 32 KiB block from a never-armed "
          "consumer's pthread_key dtor refilled L1 → never drained)",
          base_u, end_u, CYCLES - WARMUP);
    CHECK(!grew(base_c, end_c, 16),
          "unarmed-consumer dedicated free STRANDED: chunks_live grew %zu -> %zu",
          base_c, end_c);

    std::printf("  [%s] unarmed-consumer plateau: units %zu -> %zu, chunks %zu -> %zu\n",
                grew(base_u, end_u, 16) ? "BAD" : "ok",
                base_u, end_u, base_c, end_c);

    std::printf(g_fails == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fails);
    return g_fails ? 1 : 0;
}
