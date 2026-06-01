// Smoke-test for `kame::pmr::pool_memory_resource` — the C++17
// `std::pmr::memory_resource` adaptor.  Exercises the most common PMR
// containers and the polymorphic_allocator path so that any container
// or third-party library consuming `std::pmr::memory_resource *` is
// covered structurally.
//
// Does NOT require Boost / Folly / abseil — just libstdc++ / libc++ with
// `<memory_resource>` enabled.  Falls back to a compile-time skip (PASS)
// on toolchains that lack the header.
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "../kame_pool.h"

#if __has_include(<memory_resource>)

#include "pmr_resource.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <deque>
#include <list>
#include <map>
#include <memory_resource>
#include <string>
#include <unordered_map>
#include <vector>

static int g_fails = 0;
#define CHECK(c, ...) do { if(!(c)) { ++g_fails; \
    std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while(0)

int main() {
    auto *mr = kame::pmr::pool_resource();
    CHECK(mr != nullptr, "pool_resource() returned null");

    // (1) Direct memory_resource API — `allocate` / `deallocate` with a
    // few different sizes and alignments.
    {
        for (std::size_t bytes : { 16u, 64u, 1024u, 65536u }) {
            void *p = mr->allocate(bytes);
            CHECK(p != nullptr, "allocate(%zu) returned null", bytes);
            std::memset(p, 0xAB, bytes);
            mr->deallocate(p, bytes);
        }
        std::printf("  [ok] direct allocate/deallocate (4 sizes)\n");
    }

    // (2) is_equal: two singletons-by-name compare equal; null memory_resource
    //     (e.g. std::pmr::null_memory_resource()) does not.
    {
        auto *mr2 = kame::pmr::pool_resource();
        CHECK(mr->is_equal(*mr2), "singleton self-equality");
        CHECK( !mr->is_equal(*std::pmr::null_memory_resource()),
              "pool != null_memory_resource");
        std::printf("  [ok] is_equal: singleton + cross-resource\n");
    }

    // (3) std::pmr::vector — the most-used PMR container.
    {
        std::pmr::vector<int> v(mr);
        for (int i = 0; i < 10000; ++i) v.push_back(i);
        for (int i = 0; i < 10000; ++i) CHECK(v[i] == i, "vec[%d]", i);
        std::printf("  [ok] std::pmr::vector<int> push_back 10K\n");
    }

    // (4) std::pmr::string — exercises rebind to char and SSO threshold.
    {
        std::pmr::string s(mr);
        for (int i = 0; i < 4096; ++i) s.push_back((char)('a' + (i & 31)));
        CHECK(s.size() == 4096, "pmr::string size");
        std::printf("  [ok] std::pmr::string push_back 4K\n");
    }

    // (5) std::pmr::unordered_map — heavy node allocation (bucket
    //     storage + node alloc per insert).  This is the kind of
    //     container ROS 2 / Drake / abseil hash-table internals look like.
    {
        std::pmr::unordered_map<int, int> m(mr);
        for (int i = 0; i < 2048; ++i) m.emplace(i, i * 3);
        for (int i = 0; i < 2048; ++i) CHECK(m[i] == i * 3, "umap[%d]", i);
        std::printf("  [ok] std::pmr::unordered_map<int,int> 2K inserts\n");
    }

    // (6) std::pmr::map — node-per-element red-black tree, exercises
    //     rebind to the internal node type.
    {
        std::pmr::map<int, std::pmr::string> m(mr);
        for (int i = 0; i < 256; ++i)
            m.emplace(i, std::pmr::string("hello", mr));
        CHECK(m.size() == 256, "pmr::map size");
        std::printf("  [ok] std::pmr::map<int,pmr::string>\n");
    }

    // (7) std::pmr::list and ::deque — node-based + segmented containers.
    {
        std::pmr::list<int>  l(mr);
        std::pmr::deque<int> d(mr);
        for (int i = 0; i < 1024; ++i) { l.push_back(i); d.push_back(i); }
        CHECK(l.size() == 1024 && d.size() == 1024,
              "list/deque size: %zu / %zu", l.size(), d.size());
        std::printf("  [ok] std::pmr::list + std::pmr::deque\n");
    }

    // (8) std::pmr::polymorphic_allocator<T> — explicit construction-from-
    //     resource path used by libraries that want to expose their own
    //     allocator-aware factory.
    {
        std::pmr::polymorphic_allocator<int> pa(mr);
        int *p = pa.allocate(16);
        for (int i = 0; i < 16; ++i) p[i] = i * 7;
        for (int i = 0; i < 16; ++i) CHECK(p[i] == i * 7, "poly[%d]", i);
        pa.deallocate(p, 16);
        std::printf("  [ok] std::pmr::polymorphic_allocator<int>\n");
    }

    // (9) std::pmr::set_default_resource path — ensures default-constructed
    //     pmr containers (no explicit resource) route through the pool.
    {
        auto *prev = std::pmr::get_default_resource();
        std::pmr::set_default_resource(mr);
        {
            std::pmr::vector<long> w;  // no explicit mr → uses default
            for (long i = 0; i < 1000; ++i) w.push_back(i * 11);
            CHECK(w[999] == 999 * 11, "default-resource vec");
        }
        std::pmr::set_default_resource(prev);
        std::printf("  [ok] set_default_resource round-trip\n");
    }

    // (10) Sanity check: pool actually engaged (not silently routing to libc).
    {
        kame_pool_stats_t st{};
        st.version = KAME_POOL_STATS_VERSION;
        kame_pool_get_stats(&st);
        CHECK(st.regions_populated > 0,
              "pool never claimed a region — pmr_resource may misroute");
        std::printf("  [ok] pool engaged: regions=%zu chunks_live=%zu\n",
                    st.regions_populated, st.chunks_live);
    }

    std::printf(g_fails == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fails);
    return g_fails ? 1 : 0;
}

#else // !__has_include(<memory_resource>)

int main() {
    std::printf("SKIP: <memory_resource> not available on this toolchain\n");
    return 0;
}

#endif
