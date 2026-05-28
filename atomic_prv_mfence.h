/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
        implied.  See the License for the specific language governing
        permissions and limitations under the License.

        SPDX-License-Identifier: Apache-2.0
 ***************************************************************************/
#ifndef KAMEPOOLALLOC_ATOMIC_PRV_MFENCE_H_
#define KAMEPOOLALLOC_ATOMIC_PRV_MFENCE_H_

#include <atomic>

#if defined(_MSC_VER)
#  include <intrin.h>   // _mm_pause (x86) / __yield (ARM64)
#endif

// Unified barrier + spin-pause helpers (kamepoolalloc copy).  Bit-
// identical to kame/atomic_prv_mfence.h — duplicated here to keep
// kamepoolalloc dependency-free of kame/.  Memory barriers route
// through std::atomic_thread_fence (pure C++17); pause4spin retains
// a tiny per-arch hint.

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

#endif /*KAMEPOOLALLOC_ATOMIC_PRV_MFENCE_H_*/
