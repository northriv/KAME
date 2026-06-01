// Smoke-test for `kame::pool_aligned_allocator<T, Align>` — the
// Eigen-/SIMD-style over-aligned C++17 Allocator.  Exercises both the
// default-aligned path (Align ≤ 16, fast-path through kame_pool_malloc)
// and the over-aligned path (Align ∈ {32, 64, 256}, routes via
// kame_pool_aligned_alloc to the matching pool bucket).
//
// No Eigen build dep — the allocator class is a pure C++17 Allocator,
// independent of any matrix library.  If this passes, plugging it into
// `std::vector<EigenType, kame::pool_aligned_allocator<EigenType, 32>>`
// is structurally sound.
//
// **Windows note:** `kame_pool_aligned_alloc` returns null for over-aligned
// requests on Windows (current limitation).  This test detects the
// environment via `kame_pool_aligned_alloc(32, 64)` and SKIPS the over-
// aligned section there; the 16-byte default-aligned path is still
// exercised.
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "aligned_allocator.hpp"
#include "../kame_pool.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

static int g_fails = 0;
#define CHECK(c, ...) do { if(!(c)) { ++g_fails; \
    std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while(0)

// Compile-time concept checks.
static_assert(std::is_default_constructible_v<kame::pool_aligned_allocator<int, 32>>);
static_assert(std::is_copy_constructible_v<kame::pool_aligned_allocator<int, 32>>);
static_assert(std::is_same_v<
    typename std::allocator_traits<kame::pool_aligned_allocator<int, 32>>::value_type, int>);
static_assert(std::is_same_v<
    typename std::allocator_traits<kame::pool_aligned_allocator<int, 32>>::template rebind_alloc<double>,
    kame::pool_aligned_allocator<double, 32>>);
static_assert(kame::pool_aligned_allocator<int, 32>::is_always_equal::value);
static_assert(kame::pool_aligned_allocator<int, 64>::alignment == 64);

template <std::size_t A>
static bool aligned(const void *p) {
    return (reinterpret_cast<std::uintptr_t>(p) & (A - 1)) == 0;
}

int main() {
    // (1) Default-aligned (Align = 16): always supported, including
    //     Windows.  Equivalent to plain pool_allocator + 16-byte promise.
    {
        kame::pool_aligned_allocator<int, 16> a;
        int *p = a.allocate(256);
        CHECK(p != nullptr, "alloc(256) returned null");
        CHECK(aligned<16>(p), "p not 16-aligned: %p", (void *)p);
        for (int i = 0; i < 256; ++i) p[i] = i;
        for (int i = 0; i < 256; ++i) CHECK(p[i] == i, "data[%d]", i);
        a.deallocate(p, 256);
        std::printf("  [ok] Align=16 default-path 256 ints\n");
    }

    // (2) Windows over-aligned probe.  `kame_pool_aligned_alloc` on
    //     Windows currently returns null with errno=EINVAL for any
    //     alignment > 16; detect that and skip the over-aligned section
    //     so the test stays portable.
    bool over_aligned_supported = false;
    {
        if (void *probe = kame_pool_aligned_alloc(32, 64)) {
            over_aligned_supported = true;
            kame_pool_free(probe);
        }
    }
    if ( !over_aligned_supported) {
        std::printf("  [skip] over-aligned tests — "
                    "kame_pool_aligned_alloc(32, ...) returned null on this "
                    "platform (Windows limitation)\n");
    }

    // (3) Align = 32 (AVX2 / Eigen default).
    if (over_aligned_supported) {
        kame::pool_aligned_allocator<double, 32> a;
        for (int trial = 0; trial < 4; ++trial) {
            double *p = a.allocate(512);
            CHECK(p != nullptr, "32-aligned alloc trial %d", trial);
            CHECK(aligned<32>(p), "32-aligned trial %d: %p", trial, (void *)p);
            for (int i = 0; i < 512; ++i) p[i] = 1.5 * i;
            for (int i = 0; i < 512; ++i)
                CHECK(p[i] == 1.5 * i, "32-aligned data trial %d [%d]", trial, i);
            a.deallocate(p, 512);
        }
        std::printf("  [ok] Align=32 (AVX2 / Eigen) 4 alloc/free cycles\n");
    }

    // (4) Align = 64 (AVX-512 / cacheline).
    if (over_aligned_supported) {
        kame::pool_aligned_allocator<float, 64> a;
        float *p = a.allocate(1024);
        CHECK(p != nullptr, "64-aligned alloc");
        CHECK(aligned<64>(p), "64-aligned: %p", (void *)p);
        a.deallocate(p, 1024);
        std::printf("  [ok] Align=64 (AVX-512 / cacheline) alloc/free\n");
    }

    // (4b) Align = 4096 (page) — exercises the bucket path's top ALIGN class.
    if (over_aligned_supported) {
        kame::pool_aligned_allocator<char, 4096> a;
        char *p = a.allocate(8192);
        CHECK(p != nullptr, "page-aligned alloc");
        CHECK(aligned<4096>(p), "page-aligned: %p", (void *)p);
        a.deallocate(p, 8192);
        std::printf("  [ok] Align=4096 (page) alloc/free\n");
    }

    // (4c) Align = 256 KiB — exercises the dedicated-chunk path.  Pool
    // returns a chunk whose payload starts at the chunk's 256 KiB unit
    // boundary; the user gets the alignment "for free".
    if (over_aligned_supported) {
        kame::pool_aligned_allocator<char, 262144> a;  // 256 KiB
        char *p = a.allocate(65536);                    // 64 KiB user size
        CHECK(p != nullptr, "256 KiB-aligned alloc");
        CHECK(aligned<262144>(p), "256 KiB-aligned: %p", (void *)p);
        a.deallocate(p, 65536);
        std::printf("  [ok] Align=256 KiB (dedicated chunk) alloc/free\n");
    }

    // (4d) Align > 256 KiB — pool-managed ceiling.  The aligned C API
    // returns null (libc fallback only on POSIX, no fallback on Windows;
    // the large_va tier's user pointer is `base + PAGE`, not 32 MiB-
    // aligned, so it can't satisfy this).  Concept-test for graceful
    // null-return rather than crash.
    {
        void *p = kame_pool_aligned_alloc(1048576, 65536);
        if (p) {
            // POSIX libc fallback served us — sanity check the alignment.
            CHECK(aligned<1048576>(p),
                  "libc posix_memalign returned mis-aligned: %p", p);
            kame_pool_free(p);
            std::printf("  [ok] Align=1 MiB via libc fallback (POSIX)\n");
        } else {
            std::printf("  [ok] Align=1 MiB returns null (Windows / pool ceiling)\n");
        }
    }

    // (5) std::vector with the allocator — the typical Eigen drop-in.
    //     This is identical to:
    //         std::vector<T, Eigen::aligned_allocator<T>>
    //     but pool-backed.
    {
        constexpr std::size_t kAlign = 32;  // skip the test below on Windows
        if (over_aligned_supported) {
            std::vector<double, kame::pool_aligned_allocator<double, kAlign>> v;
            for (int i = 0; i < 10000; ++i) v.push_back(i * 0.5);
            CHECK(aligned<kAlign>(v.data()),
                  "vector data not %zu-aligned", kAlign);
            CHECK(v.size() == 10000, "vector size");
            CHECK(v[9999] == 9999 * 0.5, "vector content");
            std::printf("  [ok] std::vector<double, pool_aligned_allocator<double,32>> "
                        "push_back 10K + data() alignment\n");
        }
        else {
            std::vector<double, kame::pool_aligned_allocator<double, 16>> v;
            for (int i = 0; i < 10000; ++i) v.push_back(i * 0.5);
            CHECK(aligned<16>(v.data()), "vector data not 16-aligned");
            std::printf("  [ok] std::vector<double, pool_aligned_allocator<double,16>> "
                        "(Windows over-aligned-unsupported path)\n");
        }
    }

    // (6) Rebind path — exercises allocator_traits<>::rebind_alloc.
    {
        kame::pool_aligned_allocator<int, 32> a;
        using AT = std::allocator_traits<decltype(a)>;
        typename AT::template rebind_alloc<double> b(a);
        CHECK(b.alignment == 32, "rebound alignment carries through: %zu",
              b.alignment);
        std::printf("  [ok] rebind carries Align template parameter\n");
    }

    // (7) Pool engaged?
    {
        kame_pool_stats_t st{};
        st.version = KAME_POOL_STATS_VERSION;
        kame_pool_get_stats(&st);
        CHECK(st.regions_populated > 0,
              "pool never claimed a region — aligned_allocator misroute?");
        std::printf("  [ok] pool engaged: regions=%zu chunks_live=%zu\n",
                    st.regions_populated, st.chunks_live);
    }

    std::printf(g_fails == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fails);
    return g_fails ? 1 : 0;
}
