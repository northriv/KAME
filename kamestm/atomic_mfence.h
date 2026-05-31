/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
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
#ifndef KAMESTM_ATOMIC_MFENCE_H_
#define KAMESTM_ATOMIC_MFENCE_H_

// Unified barrier + spin-pause helpers.  Memory barriers route through
// `std::atomic_thread_fence` (pure C++17); `pause4spin` retains a tiny
// per-arch hint (PAUSE on x86, YIELD on ARM64).
//
// kamestm-owned duplicate of kamepoolalloc/atomic_mfence.h: kamestm is a
// standalone subtree mirror (one-way `git subtree split`) and must not
// strongly depend on kamepoolalloc — but both libraries need these
// barriers internally.  The body is pure stdlib + a 2-line per-arch
// pause hint; drift risk between the two copies is negligible and the
// alternative (dependency inversion or build-system gymnastics) is
// worse.  Each subtree owns its copy.

#include <atomic>

#if defined(_MSC_VER)
#  include <intrin.h>   // _mm_pause (x86) / __yield (ARM64)
#endif

inline void readBarrier() noexcept {
    std::atomic_thread_fence(std::memory_order_acquire);
}
inline void writeBarrier() noexcept {
    std::atomic_thread_fence(std::memory_order_release);
}
inline void memoryBarrier() noexcept {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

inline void pause4spin() noexcept {
#if defined(__x86_64__) || defined(__i386__) \
    || defined(_M_IX86) || defined(_M_X64)
#  if defined(_MSC_VER)
    _mm_pause();
#  else
    __builtin_ia32_pause();
#  endif
#elif defined(__aarch64__) || defined(__arm64__) \
    || defined(_M_ARM64) || defined(__arm__)
#  if defined(_MSC_VER)
    __yield();
#  else
    __asm__ __volatile__("yield" ::: "memory");
#  endif
#else
    // Unknown ISA: correctness retained, throughput hint omitted.
#endif
}

#endif /*KAMESTM_ATOMIC_MFENCE_H_*/
