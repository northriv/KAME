/***************************************************************************
        Copyright (C) 2008-2026 Kentaro Kitagawa
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
//
// ros2_allocator.hpp — C++17 Allocator adaptor backed by kamepoolalloc.
//
// Status: EXPERIMENTAL (contrib).  Header-only, no integration test against
// rclcpp itself ships in this repo — rclcpp / ROS 2 aren't a build
// dependency.  Reports + PRs from on-target ROS 2 real-time teams welcomed.
//
// Why use this in a ROS 2 real-time node?
//
//   * Lock-free TLS freelist pop/push on the hot path — no contention
//     between control thread, perception thread, and DDS callback worker.
//   * `kame_pool_set_realtime_mode(1)` silences ALL background maintenance
//     (lazy drain, auto-tune, thread-exit reclaim) so the allocator can
//     never inject a surprise munmap into a 1 kHz control loop.
//   * Coexists with the rest of the runtime: §31's free-family interposition
//     (ELF strong symbol on Linux, Mach-O __interpose on macOS, IAT redirect
//     on Windows) reconciles a pool-allocated message handed to a libc-
//     bound DDS stack, so no manual lifetime juggling is needed.
//   * Bounded chunk-claim latency in steady state.  Cold-claim is *not*
//     formally bounded (it can mmap a 32 MiB region on first touch), so for
//     a hard-RT loop you should pre-warm by allocating + freeing the
//     working-set sizes BEFORE entering the time-critical section — same
//     idiom you'd use with TLSF.
//
// Quick usage:
//
//     #include <kamepoolalloc/contrib/ros2_allocator.hpp>
//     #include <kame_pool.h>
//
//     int main(int argc, char **argv) {
//         rclcpp::init(argc, argv);
//
//         // (Strongly recommended for RT nodes.)  Silence the pool's
//         // background maintenance paths.
//         kame_pool_set_realtime_mode(1);
//
//         // Pre-warm: touch every size class the RT path will use.
//         // (Tune to your message sizes.)
//         for (std::size_t sz : {64u, 256u, 1024u, 4096u, 65536u}) {
//             void *p = kame_pool_malloc(sz);
//             if (p) kame_pool_free(p);
//         }
//
//         auto node = std::make_shared<MyNode>();
//
//         // Publisher / subscriber template-parameterised on the
//         // allocator.  rclcpp threads the Allocator type through to
//         // the message memory and intra-process buffer.
//         using PoolAlloc = kame::pool_allocator<void>;
//         auto pub = node->create_publisher<MsgType, PoolAlloc>(
//             "topic", qos, PoolAlloc{});
//
//         rclcpp::spin(node);
//         rclcpp::shutdown();
//     }
//
// For STL containers used inside the RT callback (e.g. a working std::vector):
//
//     std::vector<float, kame::pool_allocator<float>> work_buf;
//     work_buf.reserve(N);   // do this OUTSIDE the RT loop (one-shot mmap risk)
//
// Standards: this adaptor satisfies C++17's Allocator named requirement
// (allocator_traits<>-compatible) and the rclcpp Allocator concept
// requirements; it is `is_always_equal = true_type`, so containers and
// rclcpp internals never have to call the stateful-allocator swap path.

#ifndef KAMEPOOLALLOC_CONTRIB_ROS2_ALLOCATOR_HPP_
#define KAMEPOOLALLOC_CONTRIB_ROS2_ALLOCATOR_HPP_

#include "../kame_pool.h"

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>

namespace kame {

//! C++17 Allocator backed by kamepoolalloc.  Stateless — every instance
//! routes through the same process-wide pool — so equality is identity-
//! independent (`is_always_equal = true_type`).  Safe to default-construct,
//! copy, move, rebind, and pass into any STL container or rclcpp template.
template <typename T>
class pool_allocator {
public:
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap            = std::true_type;
    using is_always_equal                        = std::true_type;

    constexpr pool_allocator() noexcept = default;
    constexpr pool_allocator(const pool_allocator &) noexcept = default;
    constexpr pool_allocator(pool_allocator &&) noexcept = default;
    pool_allocator &operator=(const pool_allocator &) noexcept = default;
    pool_allocator &operator=(pool_allocator &&) noexcept = default;
    ~pool_allocator() = default;

    //! Rebinding constructor — allows `pool_allocator<int>` to be
    //! converted from `pool_allocator<double>` via the standard
    //! allocator_traits machinery.
    template <typename U>
    constexpr pool_allocator(const pool_allocator<U> &) noexcept {}

    //! Nested rebind alias.  `std::allocator_traits` provides a default
    //! when the templated-template form is detectable, but some older
    //! rclcpp headers (Foxy / Galactic) reach into `Allocator::rebind`
    //! directly — keep it for compatibility.
    template <typename U>
    struct rebind { using other = pool_allocator<U>; };

    //! Allocate storage for `n` Ts.  Routes through `kame_pool_malloc`,
    //! which is `noexcept` and `errno=ENOMEM` on failure — translate to
    //! `std::bad_alloc` to match the standard Allocator contract.
    [[nodiscard]] T *allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();
        void *p = kame_pool_malloc(n * sizeof(T));
        if (!p) throw std::bad_alloc();
        return static_cast<T *>(p);
    }

    //! Deallocate.  `n` is ignored — `kame_pool_free` looks up the size
    //! from the radix tree if needed.  Never throws.
    void deallocate(T *p, std::size_t /* n */) noexcept {
        kame_pool_free(p);
    }

    //! Largest `n` `allocate(n)` would accept.  Standard convenience —
    //! containers use this for growth-policy bounds.
    constexpr size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }
};

//! Equality — all instances point at the same global pool.  `true` always.
template <typename T, typename U>
constexpr bool operator==(const pool_allocator<T> &,
                          const pool_allocator<U> &) noexcept {
    return true;
}
template <typename T, typename U>
constexpr bool operator!=(const pool_allocator<T> &,
                          const pool_allocator<U> &) noexcept {
    return false;
}

} // namespace kame

#endif // KAMEPOOLALLOC_CONTRIB_ROS2_ALLOCATOR_HPP_
