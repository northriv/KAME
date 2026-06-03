// Smoke-test for kame::pool_allocator<T>'s C++17 Allocator concept
// conformance.  Exercises allocate / deallocate / rebind / propagate
// traits / equality via std::vector and std::list — the same Allocator
// concept rclcpp uses internally — so if this compiles + runs clean,
// the rclcpp integration is structurally sound.
//
// Does NOT depend on rclcpp / ROS 2 — keeps the kamepoolalloc test
// scaffold self-contained.  Build via the cmake scaffold:
//     add_executable(test_ros2_allocator contrib/test_ros2_allocator.cpp)
//     target_link_libraries(test_ros2_allocator kamepoolalloc Threads::Threads)
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "ros2_allocator.hpp"

#include "../kame_pool.h"

#include <cassert>
#include <cstdio>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

static int g_fails = 0;
#define CHECK(c, ...) do { if(!(c)) { ++g_fails; \
    std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while(0)

// Compile-time concept checks — the Allocator named requirements
// (C++17 [allocator.requirements]) and the rclcpp Allocator concept.
static_assert(std::is_default_constructible_v<kame::pool_allocator<int>>);
static_assert(std::is_copy_constructible_v<kame::pool_allocator<int>>);
static_assert(std::is_move_constructible_v<kame::pool_allocator<int>>);
static_assert(std::is_copy_assignable_v<kame::pool_allocator<int>>);
static_assert(std::is_nothrow_destructible_v<kame::pool_allocator<int>>);
static_assert(std::is_same_v<
    typename std::allocator_traits<kame::pool_allocator<int>>::value_type, int>);
static_assert(std::is_same_v<
    typename std::allocator_traits<kame::pool_allocator<int>>::template rebind_alloc<double>,
    kame::pool_allocator<double>>);
static_assert(kame::pool_allocator<int>::is_always_equal::value);

int main() {
    // (1) Basic allocate / deallocate round-trip.
    {
        kame::pool_allocator<int> a;
        int *p = a.allocate(128);
        CHECK(p != nullptr, "allocate(128) returned null");
        for (int i = 0; i < 128; ++i) p[i] = i;
        for (int i = 0; i < 128; ++i) CHECK(p[i] == i, "data integrity at %d", i);
        a.deallocate(p, 128);
        std::printf("  [ok] allocate/deallocate round-trip\n");
    }

    // (2) Allocator equality (`is_always_equal=true_type`).
    {
        kame::pool_allocator<int> a1;
        kame::pool_allocator<int> a2;
        kame::pool_allocator<double> a3;
        CHECK(a1 == a2, "same-T allocators should compare equal");
        CHECK(a1 == kame::pool_allocator<int>(a3), "rebound allocators equal");
        CHECK( !(a1 != a2), "operator!= broken");
        std::printf("  [ok] stateless equality\n");
    }

    // (3) std::vector — STL container that exercises the full Allocator
    // surface (resize, reserve, copy/move-assign with propagate_on_*).
    {
        std::vector<int, kame::pool_allocator<int>> v;
        for (int i = 0; i < 10000; ++i) v.push_back(i);
        for (int i = 0; i < 10000; ++i) CHECK(v[i] == i, "vec[%d]", i);

        std::vector<int, kame::pool_allocator<int>> w = v;          // copy
        CHECK(w.size() == 10000, "copy size");
        w.clear();
        w = std::move(v);                                            // move-assign
        CHECK(w.size() == 10000, "moved size");
        std::printf("  [ok] std::vector<int> push_back/copy/move\n");
    }

    // (4) std::list — node-based container, exercises one-element-at-a-time
    // allocate calls (the more common rclcpp message-pool pattern).
    {
        std::list<int, kame::pool_allocator<int>> l;
        for (int i = 0; i < 1024; ++i) l.push_back(i);
        int seen = 0;
        for (int x : l) CHECK(x == seen++, "list elem");
        CHECK(seen == 1024, "list iteration count");
        std::printf("  [ok] std::list<int> 1024 push_back\n");
    }

    // (5) std::map — exercises rebind_alloc (map node uses
    //     `rebind_alloc<__rb_tree_node>` internally).
    {
        std::map<int, int, std::less<>,
                 kame::pool_allocator<std::pair<const int, int>>> m;
        for (int i = 0; i < 256; ++i) m[i] = i * 2;
        for (int i = 0; i < 256; ++i) CHECK(m[i] == i * 2, "map[%d]", i);
        std::printf("  [ok] std::map<int,int> with rebind\n");
    }

    // (6) std::string with pool allocator — rebinds char allocator,
    // exercises SSO threshold + heap-allocated string growth.
    {
        using PoolString = std::basic_string<char, std::char_traits<char>,
                                              kame::pool_allocator<char>>;
        PoolString s;
        for (int i = 0; i < 4096; ++i) s += (char)('a' + (i & 31));
        CHECK(s.size() == 4096, "string size");
        std::printf("  [ok] std::basic_string with pool_allocator<char>\n");
    }

    // (7) std::allocate_shared — the rclcpp publisher/subscriber path uses
    // it heavily for message instances.  Exercises allocator-aware
    // shared_ptr ctor / control-block allocation.
    {
        kame::pool_allocator<int> a;
        auto p = std::allocate_shared<int>(a, 42);
        CHECK(*p == 42, "allocate_shared value");
        std::printf("  [ok] std::allocate_shared<int>\n");
    }

    // (8) Sanity: confirm we're actually using the pool, not silently
    // falling through to libc malloc.  After the loops above, the pool
    // should have at least one chunk live.
    {
        kame_pool_stats_t st{};
        st.version = KAME_POOL_STATS_VERSION;
        kame_pool_get_stats(&st);
        CHECK(st.regions_populated > 0,
              "pool never claimed a region — pool_allocator may be misrouting");
        std::printf("  [ok] pool engaged: regions=%zu chunks_live=%zu\n",
                    st.regions_populated, st.chunks_live);
    }

    std::printf(g_fails == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fails);
    return g_fails ? 1 : 0;
}
