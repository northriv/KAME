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
// pmr_resource.hpp — `std::pmr::memory_resource` backed by kamepoolalloc.
//
// Status: EXPERIMENTAL (contrib).  Header-only.  Requires `<memory_resource>`
// (C++17).  libc++ on macOS shipped <memory_resource> from clang 16+ /
// macOS 13.3+; libstdc++ from gcc 9+.  MSVC has had it since VS 2017
// 15.6.
//
// Why this matters: PMR is the standard C++17 interface for "swap out the
// allocator behind any STL container at runtime".  One adaptor covers EVERY
// `std::pmr::vector / unordered_map / string / list / ...` plus any third-
// party library that consumes `std::pmr::memory_resource*` (Boost.Container,
// Folly, abseil, etc.) — no per-ecosystem wrapper needed.
//
// Quick usage:
//
//     #include <kamepoolalloc/contrib/pmr_resource.hpp>
//
//     // Singleton — outlives any container that pins it.
//     auto *mr = kame::pmr::pool_resource();
//
//     std::pmr::vector<int>          v(mr);
//     std::pmr::unordered_map<K,V>   m(mr);
//     std::pmr::string               s(mr);
//
//     // Or as the default for a whole scope:
//     std::pmr::set_default_resource(mr);
//     std::pmr::vector<int> w;  // uses pool by default
//
// Alignment behaviour:
//
//   * alignment ≤ alignof(std::max_align_t)  (16 B on every supported ABI)
//     → `kame_pool_malloc()` — fast path, default 16-byte aligned slots.
//   * alignment > 16 B → `kame_pool_aligned_alloc()`.  Falls through to the
//     pool's bucket tier (ALIGN ∈ {32, 64, 256, 1024, 4096}) on POSIX.
//     **Windows:** `kame_pool_aligned_alloc` is currently a no-op (returns
//     null with errno=EINVAL) for over-aligned requests — see
//     kame_pool_aligned_alloc's comment in allocator.cpp.  PMR over-aligned
//     allocations therefore throw `std::bad_alloc` on Windows; either keep
//     alignment ≤ 16 there, or use a different PMR resource for those
//     containers.

#ifndef KAMEPOOLALLOC_CONTRIB_PMR_RESOURCE_HPP_
#define KAMEPOOLALLOC_CONTRIB_PMR_RESOURCE_HPP_

#include "../kame_pool.h"

#include <cstddef>
#include <memory_resource>
#include <new>

namespace kame::pmr {

//! `std::pmr::memory_resource` implementation routing through kamepoolalloc.
//! Stateless singleton — `do_is_equal()` returns true for any other instance
//! of this class, so polymorphic_allocator copy/swap paths short-circuit
//! to the no-op fast path.
class pool_memory_resource final : public std::pmr::memory_resource {
public:
    //! Process-wide singleton — outlives every container that pins it,
    //! and is cheap to take the address of from any thread.
    static pool_memory_resource *singleton() noexcept {
        static pool_memory_resource s;
        return &s;
    }

protected:
    void *do_allocate(std::size_t bytes, std::size_t alignment) override {
        if (bytes == 0) bytes = 1;  // PMR contract: distinguishable pointer
        // Fast path — kame_pool_malloc guarantees alignof(std::max_align_t)
        // (16 B on every supported ABI), which covers any "normal" PMR
        // request including doubles, ptrs, and SIMD-free types.
        if (alignment <= alignof(std::max_align_t)) {
            if (void *p = kame_pool_malloc(bytes)) return p;
            throw std::bad_alloc();
        }
        // Over-aligned (SIMD types, page-aligned buffers, etc.) — route
        // via the pool's explicit aligned API.  Returns null on Windows
        // for now (see file header); translate to bad_alloc per PMR.
        if (void *p = kame_pool_aligned_alloc(alignment, bytes)) return p;
        throw std::bad_alloc();
    }

    void do_deallocate(void *p, std::size_t /*bytes*/,
                       std::size_t /*alignment*/) override {
        // `kame_pool_free` looks up the slot kind via the radix tree, so
        // it correctly handles BOTH the fast-path (kame_pool_malloc) and
        // aligned (kame_pool_aligned_alloc) returns without per-pointer
        // tagging on our side.
        kame_pool_free(p);
    }

    bool do_is_equal(const memory_resource &other) const noexcept override {
        // Stateless singleton — every instance allocates from the same
        // pool, so any two `pool_memory_resource`s are functionally
        // equal.  Identity check first as a fast path, then RTTI for
        // the (rare) case where two instances of the class coexist via
        // separate translation units / static libs.
        if (this == &other) return true;
        return dynamic_cast<const pool_memory_resource *>(&other) != nullptr;
    }
};

//! Convenience accessor — returns a pointer to the process-wide singleton.
//! Use this anywhere you'd pass a `std::pmr::memory_resource *`.
inline std::pmr::memory_resource *pool_resource() noexcept {
    return pool_memory_resource::singleton();
}

} // namespace kame::pmr

#endif // KAMEPOOLALLOC_CONTRIB_PMR_RESOURCE_HPP_
