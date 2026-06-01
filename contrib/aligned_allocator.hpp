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
// aligned_allocator.hpp — over-aligned C++17 Allocator backed by kamepoolalloc.
//
// Status: EXPERIMENTAL (contrib).  Header-only.  Drop-in replacement for
// `Eigen::aligned_allocator<T>` and similar alignment-aware allocators used
// by Drake / Bullet / ceres-solver / ndarray containers.  Default alignment
// 32 B matches `EIGEN_DEFAULT_ALIGN_BYTES` on x86-64 with AVX2; pass an
// explicit `Align` template parameter for AVX-512 (64) or page-aligned
// buffers (4096).
//
// Quick usage:
//
//     #include <kamepoolalloc/contrib/aligned_allocator.hpp>
//
//     // Drop-in for `Eigen::aligned_allocator<Eigen::Vector4d>`:
//     std::vector<Eigen::Vector4d,
//                 kame::pool_aligned_allocator<Eigen::Vector4d>> v;
//
//     // Custom alignment (AVX-512 = 64 B):
//     std::vector<float,
//                 kame::pool_aligned_allocator<float, 64>> w;
//
// Routing:
//   * `Align ≤ alignof(std::max_align_t)` (16 B on every supported ABI)
//     → `kame_pool_malloc()` fast path, no per-call alignment math.
//   * `Align > 16 B` → `kame_pool_aligned_alloc(Align, n*sizeof(T))`,
//     served from the pool's bucket tiers (ALIGN ∈ {32, 64, 256, 1024,
//     4096}) on POSIX.
//
// **Windows caveat:** `kame_pool_aligned_alloc` currently rejects over-
// aligned requests on Windows (the libc-fallback can't be paired with
// `kame_pool_free` without per-pointer tagging — see
// kamepoolalloc/allocator.cpp:4029).  On Windows, over-aligned allocation
// here throws `std::bad_alloc`.  Workarounds:
//   - Build for POSIX (Linux / macOS) where it Just Works.
//   - Use `Eigen::aligned_allocator<T>` (libc-backed) instead on Windows.
//   - For 16-byte-aligned types (most uses), drop `Align` to the default
//     16 — kamepoolalloc's normal slots are 16 B-aligned everywhere.

#ifndef KAMEPOOLALLOC_CONTRIB_ALIGNED_ALLOCATOR_HPP_
#define KAMEPOOLALLOC_CONTRIB_ALIGNED_ALLOCATOR_HPP_

#include "../kame_pool.h"

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>

namespace kame {

//! C++17 Allocator that guarantees `Align`-byte alignment on its returns.
//! Stateless: `is_always_equal = true_type`.  Drop-in for
//! `Eigen::aligned_allocator<T>` with `Align = EIGEN_DEFAULT_ALIGN_BYTES`
//! (defaults to 32 B = AVX2 alignment).
//!
//! Concept conformance: same as `kame::pool_allocator<T>` — value_type,
//! allocate / deallocate, rebind, propagate_on_* traits, equality operators.
template <typename T, std::size_t Align = 32>
class pool_aligned_allocator {
    static_assert(Align > 0 && (Align & (Align - 1)) == 0,
                  "kame::pool_aligned_allocator: Align must be a power of 2");

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap            = std::true_type;
    using is_always_equal                        = std::true_type;

    //! The alignment this allocator promises.  Public so Eigen-style
    //! templates that probe `Allocator::alignment` (some do) can see it.
    static constexpr std::size_t alignment = Align;

    constexpr pool_aligned_allocator() noexcept = default;
    constexpr pool_aligned_allocator(const pool_aligned_allocator &) noexcept = default;
    constexpr pool_aligned_allocator(pool_aligned_allocator &&) noexcept = default;
    pool_aligned_allocator &operator=(const pool_aligned_allocator &) noexcept = default;
    pool_aligned_allocator &operator=(pool_aligned_allocator &&) noexcept = default;
    ~pool_aligned_allocator() = default;

    //! Rebinding constructor.  Allows `pool_aligned_allocator<int>` to be
    //! converted from `pool_aligned_allocator<double>` via the standard
    //! allocator_traits machinery.  Alignment carries through.
    template <typename U>
    constexpr pool_aligned_allocator(
        const pool_aligned_allocator<U, Align> &) noexcept {}

    //! Nested rebind alias — same alignment carries through to the
    //! rebound type.
    template <typename U>
    struct rebind { using other = pool_aligned_allocator<U, Align>; };

    [[nodiscard]] T *allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();
        const std::size_t bytes = n * sizeof(T);

        // Fast path: required alignment satisfied by the pool's default
        // 16-byte malloc.  Both `Align ≤ 16` AND `alignof(T) ≤ 16` must
        // hold — if T itself demands more alignment than the template
        // parameter (unusual but legal), respect that too.
        if constexpr (Align <= alignof(std::max_align_t)
                      && alignof(T) <= alignof(std::max_align_t)) {
            if (void *p = kame_pool_malloc(bytes))
                return static_cast<T *>(p);
            throw std::bad_alloc();
        }
        // Over-aligned path.  `kame_pool_aligned_alloc` returns null on
        // Windows (over-aligned not yet supported — see header comment);
        // surface that as `std::bad_alloc` per the Allocator contract.
        const std::size_t a =
            Align > alignof(T) ? Align : alignof(T);
        if (void *p = kame_pool_aligned_alloc(a, bytes))
            return static_cast<T *>(p);
        throw std::bad_alloc();
    }

    void deallocate(T *p, std::size_t /* n */) noexcept {
        // kame_pool_free dispatches via the radix; works for both the
        // fast-path and aligned returns without per-pointer tagging.
        kame_pool_free(p);
    }

    constexpr size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }
};

template <typename T, std::size_t A1, typename U, std::size_t A2>
constexpr bool operator==(const pool_aligned_allocator<T, A1> &,
                          const pool_aligned_allocator<U, A2> &) noexcept {
    // Stateless + same backing pool → always equal across instances of
    // the template, regardless of T / Align.  (Containers only compare
    // same-T allocators in practice; the cross-T overload satisfies the
    // allocator_traits-rebind path.)
    return true;
}
template <typename T, std::size_t A1, typename U, std::size_t A2>
constexpr bool operator!=(const pool_aligned_allocator<T, A1> &,
                          const pool_aligned_allocator<U, A2> &) noexcept {
    return false;
}

} // namespace kame

#endif // KAMEPOOLALLOC_CONTRIB_ALIGNED_ALLOCATOR_HPP_
