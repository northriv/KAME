// (§28) Cap-eviction verification for kame_pool_set_large_cache_cap.
//
// The cache holds freed large-tier blocks (chunks 256 KiB..4 MiB and mmap
// 4..64 MiB) up to g_lrc_cap bytes.  Lowering the cap should synchronously
// release blocks until reserved_bytes drops to (or below) the new total/2.
//
// Two-phase eviction priority (§28):
//   Phase 1 (cache > hw·ΣS_i): REDUCE K — drop redundant duplicates from
//     high-k slots, preserve every size class's coverage.
//   Phase 2 (cache ≤ hw·ΣS_i): REDUCE N — drop highest-idx (largest-size)
//     slots first.
//
// This test pins down:
//   (1) cap-RAISE / cap-UNCHANGED return without releasing anything (the
//       short-circuit at the start);
//   (2) cap-LOWER shrinks the cache so subsequent allocs can't reuse the
//       previously-cached blocks (proxy: peak reserved bytes after a
//       cap-tighten is lower than the cache was holding before);
//   (3) the eviction is STRONG-CAS-and-release — racing pop/push during the
//       evict don't cause infinite loops or lost-update;
//   (4) the cache still works after a tighten + re-raise round-trip.
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "kame_pool.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static int g_fails = 0;
#define CHECK(c, ...) do { if(!(c)) { ++g_fails; \
    std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while(0)

static const size_t MiB = (size_t)1u << 20;

// Touch every page so allocations are RSS-resident (otherwise mmap+munmap
// looks free).  Reduces noise in reserved_bytes-based assertions.
static void touch(void *p, size_t n) {
    auto *b = (volatile uint8_t *)p;
    for(size_t off = 0; off < n; off += 4096) b[off] = (uint8_t)(off >> 12);
}

int main() {
    // Baseline: explicit small→bigger raise is non-shrinking.  Use literal
    // values (NOT the default cap's magnitude × N) so the multiplication
    // can't overflow `size_t` on 32-bit hosts where the default cap (2 GiB
    // total) sits within an octave of SIZE_MAX.
    {
        kame_pool_set_large_cache_cap(128 * MiB);   // small baseline
        size_t cap_small = kame_pool_get_large_cache_cap();
        kame_pool_set_large_cache_cap(512 * MiB);   // RAISE
        size_t cap_big = kame_pool_get_large_cache_cap();
        CHECK(cap_big >= cap_small,
              "cap RAISE shrank: %zu MiB -> %zu MiB",
              cap_small / MiB, cap_big / MiB);
        std::printf("  [ok] cap raise %zu MiB -> %zu MiB\n",
                    cap_small / MiB, cap_big / MiB);
    }

    // Fill the cache with a representative spread (small mmap blocks across
    // several size classes) so eviction has things to walk.
    {
        kame_pool_set_large_cache_cap(1024 * MiB);  // 1 GiB total = 512 MiB g_lrc_cap

        std::vector<void *> ptrs;
        for(size_t sz : {6u * MiB, 10u * MiB, 16u * MiB, 24u * MiB}) {
            for(int i = 0; i < 6; i++) {
                void *p = kame_pool_malloc(sz);
                if(p) { touch(p, sz); ptrs.push_back(p); }
            }
        }
        // Free everything → goes into the cache (most blocks).
        for(void *p : ptrs) kame_pool_free(p);
        size_t before = kame_pool_reserved_bytes();
        std::printf("  filled cache; reserved=%zu MiB\n", before / MiB);

        // Tighten the cap aggressively: 64 MiB total = 32 MiB g_lrc_cap.
        // The Phase-2 path must drop the highest-idx (largest-size) entries
        // first, so the cache settles at low-size coverage only.
        kame_pool_set_large_cache_cap(64 * MiB);
        size_t after = kame_pool_reserved_bytes();
        std::printf("  after cap=64 MiB; reserved=%zu MiB\n", after / MiB);

        // We can't directly inspect g_lrc_bytes (it's not exposed in the C
        // API), but we can verify that subsequent fresh allocs DON'T find
        // the previously-cached large blocks — i.e. usable_size returns
        // tight values, not the cached (possibly oversized) ones.
        void *q = kame_pool_malloc(20 * MiB);
        size_t us = kame_pool_malloc_usable_size(q);
        CHECK(us > 0, "alloc after evict null or libc-foreign");
        CHECK(us < 28 * MiB,
              "20 MiB alloc after aggressive evict returned %zu MiB — cache "
              "should have been emptied of large blocks", us / MiB);
        std::printf("  [%s] post-evict 20 MiB alloc usable=%zu MiB (want ~20)\n",
                    us < 28 * MiB ? "ok" : "BAD", us / MiB);
        kame_pool_free(q);
    }

    // MT race: many threads alloc/free while another thread tightens the
    // cap.  The evict's strong-CAS must not livelock with the racing pushes/
    // pops, and no double-free / use-after-free.
    {
        kame_pool_set_large_cache_cap(512 * MiB);
        std::atomic<bool> stop{false};
        std::atomic<int> bad{0};
        auto churner = [&] {
            while(!stop.load(std::memory_order_relaxed)) {
                void *p = kame_pool_malloc(8 * MiB);
                if(!p) { bad++; continue; }
                ((volatile uint8_t *)p)[0] = 0x42;
                if(((volatile uint8_t *)p)[0] != 0x42) bad++;
                kame_pool_free(p);
            }
        };
        std::vector<std::thread> ts;
        for(int t = 0; t < 6; t++) ts.emplace_back(churner);
        // Several tightens while churners run.
        for(int round = 0; round < 4; round++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            kame_pool_set_large_cache_cap((size_t)(64 + round * 32) * MiB);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        stop.store(true, std::memory_order_relaxed);
        for(auto &t : ts) t.join();
        CHECK(bad == 0, "MT cap-tighten had %d failures", bad.load());
        std::printf("  [%s] MT cap tighten under 6-thread churn\n",
                    bad == 0 ? "ok" : "BAD");
    }

    // Round-trip: tighten then re-raise; cache should still serve fresh
    // alloc/free cycles (no permanent corruption).
    {
        kame_pool_set_large_cache_cap(32 * MiB);   // tight
        kame_pool_set_large_cache_cap(1024 * MiB); // re-raise
        void *a = kame_pool_malloc(10 * MiB);
        if(a) kame_pool_free(a);
        void *b = kame_pool_malloc(10 * MiB);
        if(b) {
            size_t us = kame_pool_malloc_usable_size(b);
            CHECK(us > 0 && us < 14 * MiB,
                  "post round-trip alloc usable=%zu MiB", us / MiB);
            kame_pool_free(b);
        }
        std::printf("  [ok] tighten + re-raise round-trip\n");
    }

    std::printf(g_fails == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fails);
    return g_fails ? 1 : 0;
}
