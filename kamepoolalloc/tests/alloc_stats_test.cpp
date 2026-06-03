// (§28.2 / §28.5) kame_pool_get_stats v2 fields verification.
//
// Stats v2 adds four tier-attribution fields beyond v1's region/unit counts:
//   - cache_bytes:          bytes parked in the global L2 recycle cache
//   - dedicated_chunk_bytes: §15 dedicated chunks (INCLUDING cache-parked,
//                            §28.5: walk-derived in get_stats, no hot-path
//                            running counter)
//   - large_alloc_count    : §19/§27 large_va live count (held by program;
//                            cache-parked NOT included)
//   - large_alloc_bytes    : sum of their mmap_size (live, not parked)
//
// `cache_bytes` and `large_*` are O(1) atomic counters; `dedicated_chunk_bytes`
// is reconstructed via the same region+back_offset walk that produces
// `chunks_live`/`units_live`.  This test verifies the values move in the
// expected direction at each allocate/free transition.
//
// Pool-only (uses the kame_pool_* C API directly).
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "kame_pool.h"

#include <cstdio>
#include <cstdint>
#include <cstring>

static int g_fails = 0;
#define CHECK(c, ...) do { if(!(c)) { ++g_fails; \
    std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while(0)

static const size_t MiB = (size_t)1u << 20;

static kame_pool_stats_t snap() {
    kame_pool_stats_t s = {};
    s.version = KAME_POOL_STATS_VERSION;
    kame_pool_get_stats(&s);
    return s;
}

int main() {
    // Baseline.
    auto base = snap();
    CHECK(base.version_supported >= 2, "expected v2 support, got %u",
          base.version_supported);
    std::printf("  baseline: dedicated=%zu MiB  large_n=%zu  large_b=%zu MiB  cache=%zu MiB\n",
                base.dedicated_chunk_bytes / MiB, base.large_alloc_count,
                base.large_alloc_bytes / MiB, base.cache_bytes / MiB);

    // (1) Dedicated chunk (256 KiB..4 MiB) — verify dedicated_chunk_bytes
    //     bumps on alloc.  Post-free the chunk is cache-parked (L1 hot
    //     path), so `dedicated_chunk_bytes` STAYS bumped (§28.5: walk-derived
    //     counter, parked chunks keep bit-7 + claim bit).  The increment
    //     is reflected in `cache_bytes` once spilled to global L2 — see
    //     (1b).  Note: the FIRST kame_pool_malloc can cause a cascade of
    //     internal medium-size allocs (NUMA detection via opendir routes
    //     through our pool), so `delta_alloc` may be larger than the user
    //     request; only verify it bumped UP.
    {
        void *p = kame_pool_malloc(800 * 1024);   // → §15 dedicated, ~1 MiB chunk
        auto s1 = snap();
        CHECK(s1.dedicated_chunk_bytes > base.dedicated_chunk_bytes,
              "dedicated alloc didn't bump dedicated_chunk_bytes: %zu → %zu",
              base.dedicated_chunk_bytes, s1.dedicated_chunk_bytes);
        size_t delta_alloc = s1.dedicated_chunk_bytes - base.dedicated_chunk_bytes;
        CHECK(delta_alloc >= 800 * 1024,
              "dedicated alloc bump = %zu KiB, expected ≥ 800 KiB", delta_alloc / 1024);
        kame_pool_free(p);
        auto s2 = snap();
        // §28.5: cache-parked dedicated chunks remain counted; the chunk
        // stays in the bitmap with bit-7 set.  Verify it's still ≥ s1.
        CHECK(s2.dedicated_chunk_bytes >= s1.dedicated_chunk_bytes,
              "dedicated free unexpectedly dropped dedicated_chunk_bytes: %zu → %zu",
              s1.dedicated_chunk_bytes, s2.dedicated_chunk_bytes);
        std::printf("  [ok] dedicated alloc: dedicated +%zu KiB (parked after free)\n",
                    delta_alloc / 1024);
    }

    // (1b) L1 overflow → global L2 — push more same-size blocks than the L1
    //      can hold so the spill into the global cache is reflected in
    //      cache_bytes.
    {
        auto s0 = snap();
        constexpr int N = 64;        // > LRC_K_L1 (=32) per band
        void *buf[N];
        for(int i = 0; i < N; i++) buf[i] = kame_pool_malloc(800 * 1024);
        for(int i = 0; i < N; i++) kame_pool_free(buf[i]);
        auto s1 = snap();
        CHECK(s1.cache_bytes > s0.cache_bytes,
              "%dx dedicated free didn't bump cache_bytes (L2): %zu → %zu",
              N, s0.cache_bytes, s1.cache_bytes);
        std::printf("  [ok] L1-overflow (%dx dedicated): cache_bytes %zu → %zu KiB\n",
                    N, s0.cache_bytes / 1024, s1.cache_bytes / 1024);
    }

    // (2) Large mmap (4..32 MiB) — verify large_alloc_count and
    //     large_alloc_bytes both step.
    {
        auto s0 = snap();
        void *p = kame_pool_malloc(10 * MiB);     // → §19 large_va
        auto s1 = snap();
        CHECK(s1.large_alloc_count == s0.large_alloc_count + 1,
              "large alloc didn't bump count: %zu → %zu",
              s0.large_alloc_count, s1.large_alloc_count);
        size_t delta = s1.large_alloc_bytes - s0.large_alloc_bytes;
        CHECK(delta >= 10 * MiB && delta <= 12 * MiB,
              "large alloc bytes bump = %zu MiB, expected ~10 MiB", delta / MiB);
        kame_pool_free(p);
        auto s2 = snap();
        CHECK(s2.large_alloc_count == s0.large_alloc_count,
              "large free didn't restore count: %zu vs %zu",
              s2.large_alloc_count, s0.large_alloc_count);
        CHECK(s2.large_alloc_bytes == s0.large_alloc_bytes,
              "large free didn't restore bytes: %zu vs %zu",
              s2.large_alloc_bytes, s0.large_alloc_bytes);
        std::printf("  [ok] 10 MiB large_va alloc+free: count +1 then back, bytes +%zu MiB then back\n",
                    delta / MiB);
    }

    // (3) Large span alloc (100 MiB, §27/§35) — same counters cover it.  Since
    //     §35 this size is ≤ LRC_HI so it is CACHED on free, but the live
    //     counters (large_alloc_count/bytes) track the in-program block
    //     regardless of whether free caches or munmaps it.
    {
        auto s0 = snap();
        void *p = kame_pool_malloc(100 * MiB);
        auto s1 = snap();
        CHECK(s1.large_alloc_count == s0.large_alloc_count + 1,
              "huge alloc didn't bump count");
        size_t delta = s1.large_alloc_bytes - s0.large_alloc_bytes;
        CHECK(delta >= 100 * MiB,
              "huge alloc bytes bump = %zu MiB, expected ≥ 100 MiB", delta / MiB);
        kame_pool_free(p);
        auto s2 = snap();
        CHECK(s2.large_alloc_count == s0.large_alloc_count,
              "huge free didn't restore count");
        CHECK(s2.large_alloc_bytes == s0.large_alloc_bytes,
              "huge free didn't restore bytes");
        std::printf("  [ok] 100 MiB huge alloc+free: count and bytes both step then back\n");
    }

    // (4) cap-tighten drains the cache → cache_bytes should fall.
    {
        // Pump some content into the cache.
        void *p = kame_pool_malloc(1 * MiB);
        kame_pool_free(p);
        auto s1 = snap();
        kame_pool_set_large_cache_cap(16 * MiB);   // tight cap = drain
        auto s2 = snap();
        // After a tight cap the parked block can be released — cache_bytes
        // should not exceed (cap = total/2 = 8 MiB).
        CHECK(s2.cache_bytes <= 8 * MiB,
              "cache_bytes after tighten = %zu MiB, expected ≤ 8 MiB",
              s2.cache_bytes / MiB);
        std::printf("  [ok] cap tighten: cache_bytes %zu MiB → %zu MiB (cap=8 MiB)\n",
                    s1.cache_bytes / MiB, s2.cache_bytes / MiB);
    }

    std::printf(g_fails == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fails);
    return g_fails ? 1 : 0;
}
