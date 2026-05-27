/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
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
