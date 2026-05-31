// (§27) Huge-allocation (> 32 MiB) verification.
//
// Sizes above ALLOC_MIN_MMAP_SIZE (32 MiB) are served by allocate_large_va
// with a MULTI-region mmap, of which only the head 32-MiB radix slot is
// registered (the tail slots are never standalone radix_lookup targets), and
// they BYPASS the §25/§26 warm recycle cache (whose log index tops out at
// 32 MiB — above that all sizes collapse to one slot whose only pop gate is
// `cached ≥ need`, which would over-satisfy a smaller huge request and pin
// its RSS).  This test pins down:
//
//   (1) round-trip integrity across the FULL span (head + every tail slot),
//   (2) two live huge allocs get distinct, non-overlapping bases (no radix
//       slot collision),
//   (3) RSS-regression / no-over-satisfy: after freeing a BIG huge block, a
//       SMALLER huge request must NOT be satisfied from it (the cache bypass)
//       — usable size stays tight,
//   (4) the ≤ 32 MiB tier STILL warm-reuses through the cache (the bypass is
//       scoped to the huge tier only),
//   (5) a multithreaded huge alloc/free storm stays correct.
//
// Pool-only (uses the kame_pool_* C API directly); built when
// USE_KAME_ALLOCATOR is ON.  kame_pool_malloc_usable_size() returns 0 for
// foreign/libc pointers, so a non-zero tight usable size also proves the
// allocation is pool-managed (allocate_large_va), not a libc fallback.
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

// Paint sentinels at the start, every 8 MiB, and the very last byte — so a
// dropped/over-written tail radix slot (or a short mmap) is caught.
static void touch_span(void *p, size_t n) {
    volatile uint8_t *b = (volatile uint8_t *)p;
    for(size_t off = 0; off < n; off += (8u << 20)) b[off] = (uint8_t)(0xA5 ^ (off >> 20));
    b[n - 1] = 0x5A;
}
static bool check_span(void *p, size_t n) {
    volatile uint8_t *b = (volatile uint8_t *)p;
    for(size_t off = 0; off < n; off += (8u << 20))
        if(b[off] != (uint8_t)(0xA5 ^ (off >> 20))) return false;
    return b[n - 1] == 0x5A;
}

int main() {
    // (1) round-trip integrity for several huge sizes (all > 32 MiB)
    for(size_t mb : {40u, 64u, 100u, 200u}) {
        size_t n = mb * MiB;
        void *p = kame_pool_malloc(n);
        CHECK(p != nullptr, "malloc(%zu MiB) returned null", mb);
        if(!p) continue;
        size_t us = kame_pool_malloc_usable_size(p);
        // Non-zero ⇒ pool-managed (not a foreign/libc pointer); ≥ n ⇒ honoured;
        // tight ⇒ NOT over-satisfied from a cached bigger block.
        CHECK(us >= n, "usable_size(%zu MiB)=%zu < requested %zu", mb, us, n);
        CHECK(us < n + 64 * MiB, "usable_size(%zu MiB)=%zu MiB too large (cache over-satisfy?)",
              mb, us / MiB);
        touch_span(p, n);
        CHECK(check_span(p, n), "span integrity FAILED for %zu MiB (tail-slot corruption?)", mb);
        kame_pool_free(p);
        std::printf("  [ok] %zu MiB round-trip, usable=%zu MiB\n", mb, us / MiB);
    }

    // (2) two live huge allocs: distinct, non-overlapping, no cross-corruption
    {
        size_t n = 50 * MiB;
        void *a = kame_pool_malloc(n), *b = kame_pool_malloc(n);
        CHECK(a && b && a != b, "two huge allocs not distinct: %p %p", a, b);
        if(a && b) {
            uintptr_t ua = (uintptr_t)a, ub = (uintptr_t)b;
            bool overlap = (ua < ub + n) && (ub < ua + n);
            CHECK(!overlap, "two huge allocs overlap: %p %p (n=%zu)", a, b, n);
            std::memset(a, 0x11, n);
            std::memset(b, 0x22, n);
            CHECK(((uint8_t *)a)[n - 1] == 0x11 && ((uint8_t *)b)[n - 1] == 0x22,
                  "cross-corruption between two huge allocs");
        }
        kame_pool_free(a);
        kame_pool_free(b);
        std::printf("  [ok] two 50 MiB allocs distinct & non-overlapping\n");
    }

    // (3) RSS-regression: free a 512 MiB block, then request 40 MiB.  The
    //     cache being bypassed for huge, the 40 MiB request must be a fresh
    //     ~40 MiB mmap, NOT the recycled 512 MiB block.
    {
        void *big = kame_pool_malloc(512 * MiB);
        CHECK(big != nullptr, "malloc(512 MiB) null");
        kame_pool_free(big);  // would-be cache push — must be skipped for huge
        void *small = kame_pool_malloc(40 * MiB);
        CHECK(small != nullptr, "malloc(40 MiB) null");
        if(small) {
            size_t us = kame_pool_malloc_usable_size(small);
            CHECK(us < 64 * MiB,
                  "40 MiB request got %zu MiB usable — over-satisfied from the freed 512 MiB block!",
                  us / MiB);
            std::printf("  [%s] post-512MiB-free, 40 MiB request usable=%zu MiB (want ~40, not ~512)\n",
                        us < 64 * MiB ? "ok" : "BAD", us / MiB);
            kame_pool_free(small);
        }
    }

    // (4) the cacheable tier (≤ 32 MiB) STILL warm-reuses through the cache —
    //     the §27 bypass is scoped to the huge tier only.  Single-threaded so
    //     the just-freed block is the obvious reuse candidate.
    {
        void *a = kame_pool_malloc(20 * MiB);
        kame_pool_free(a);
        void *b = kame_pool_malloc(20 * MiB);
        CHECK(a == b, "20 MiB tier did NOT warm-reuse (a=%p b=%p) — cache regression", a, b);
        std::printf("  [%s] 20 MiB tier warm-reuse (cache intact for <= 32 MiB)\n",
                    a == b ? "ok" : "BAD");
        kame_pool_free(b);
    }

    // (5) MT: many threads hammering huge allocs — distinctness + integrity
    {
        std::atomic<int> bad{0};
        auto worker = [&] {
            for(int i = 0; i < 8; i++) {
                size_t n = (size_t)(33 + (i % 8) * 4) * MiB;  // 33..61 MiB, all huge
                void *p = kame_pool_malloc(n);
                if(!p) { bad++; continue; }
                std::memset(p, 0x7E, n);
                if(((uint8_t *)p)[0] != 0x7E || ((uint8_t *)p)[n - 1] != 0x7E) bad++;
                kame_pool_free(p);
            }
        };
        std::vector<std::thread> ts;
        for(int t = 0; t < 8; t++) ts.emplace_back(worker);
        for(auto &t : ts) t.join();
        CHECK(bad == 0, "MT huge alloc had %d failures", bad.load());
        std::printf("  [%s] MT 8x8 huge allocs\n", bad == 0 ? "ok" : "BAD");
    }

    std::printf(g_fails == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fails);
    return g_fails ? 1 : 0;
}
