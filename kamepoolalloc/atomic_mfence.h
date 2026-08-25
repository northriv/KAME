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
#ifndef KAMEPOOLALLOC_ATOMIC_MFENCE_H_
#define KAMEPOOLALLOC_ATOMIC_MFENCE_H_

// Unified barrier + spin-pause helpers.  Memory barriers route through
// `std::atomic_thread_fence` (pure C++17); `pause4spin` retains a tiny
// per-arch hint (PAUSE on x86, YIELD on ARM64).  This replaces the
// previous arch-dispatched scheme via atomic_prv_mfence_x86.h /
// atomic_prv_mfence_arm8.h (deleted) and the intermediate
// atomic_prv_mfence.h indirection (merged in-place here).

#include <atomic>

#if defined(_MSC_VER)
#  include <intrin.h>   // _mm_pause (x86) / __yield (ARM64)
#endif

// §13.43(2): TSan does not model atomic_thread_fence (-Wtsan), so
// fence-published data is invisible to it no matter how allocations are
// annotated.  Under a TSan build ONLY, the barriers additionally
// release/acquire a process-wide proxy token, which conservatively
// models the fences' pan-address ordering for the race detector (it can
// only ADD happens-before, i.e. hide, never invent, a race -- and only
// along edges the fences already claim).  Self-contained gate: this
// header is included before allocator_prv.h's KAME_TSAN_ENABLED exists.
#if defined(__SANITIZE_THREAD__)
#define KAME_MFENCE_TSAN_ 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define KAME_MFENCE_TSAN_ 1
#endif
#endif
#ifdef KAME_MFENCE_TSAN_
extern "C" {
void __tsan_acquire(void *addr);
void __tsan_release(void *addr);
}
inline unsigned char kame_tsan_fence_token_;
#define KAME_MFENCE_TSAN_ACQ_ __tsan_acquire(&kame_tsan_fence_token_)
#define KAME_MFENCE_TSAN_REL_ __tsan_release(&kame_tsan_fence_token_)
#else
#define KAME_MFENCE_TSAN_ACQ_ ((void)0)
#define KAME_MFENCE_TSAN_REL_ ((void)0)
#endif

inline void readBarrier() noexcept {
    std::atomic_thread_fence(std::memory_order_acquire);
    KAME_MFENCE_TSAN_ACQ_;
}
inline void writeBarrier() noexcept {
    KAME_MFENCE_TSAN_REL_;
    std::atomic_thread_fence(std::memory_order_release);
}
inline void memoryBarrier() noexcept {
    KAME_MFENCE_TSAN_REL_;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    KAME_MFENCE_TSAN_ACQ_;
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

#endif /*KAMEPOOLALLOC_ATOMIC_MFENCE_H_*/
