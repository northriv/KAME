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
#ifndef ATOMIC_PRV_MFENCE_H_
#define ATOMIC_PRV_MFENCE_H_

#include <atomic>

#if defined(_MSC_VER)
#  include <intrin.h>   // _mm_pause (x86) / __yield (ARM64)
#endif

// Unified barrier + spin-pause helpers.  Replaces the previous
// per-arch headers (atomic_prv_mfence_x86.h / atomic_prv_mfence_arm8.h),
// which are now thin shims for backward compatibility.
//
// Memory barriers route through std::atomic_thread_fence — pure C++17,
// no inline asm, no compiler intrinsics.  Compilers emit the equivalent
// hardware fences:
//   * x86_64: seq_cst → mfence (or `lock or`); acquire/release → no-op
//             (TSO makes them free for normal loads/stores)
//   * ARMv8:  seq_cst → dmb ish (or dmb sy); acquire → dmb ishld;
//             release → dmb ishst
//
// Note on rdtsc serialization (kame/xtime.cpp): the existing call
// sequence `memoryBarrier(); rdtsc; memoryBarrier();` relied on the
// previous _mm_mfence() emitting an actual mfence instruction.  On
// GCC/Clang/MSVC, std::atomic_thread_fence(seq_cst) on x86 also emits
// mfence (or `lock or`), so rdtsc serialization is preserved in practice.
// If a future toolchain elides the fence we'd need an x86-specific
// shim — flag with the comment at the rdtsc site if that becomes a
// concern.

inline void readBarrier() noexcept {
    std::atomic_thread_fence(std::memory_order_acquire);
}
inline void writeBarrier() noexcept {
    std::atomic_thread_fence(std::memory_order_release);
}
inline void memoryBarrier() noexcept {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

// CPU pause hint for spin loops.  The C++ standard has no portable
// equivalent (std::this_thread::yield is an OS yield, not a CPU hint),
// so this remains the one piece of arch-specific code.  Correctness
// is preserved on unknown architectures — only the throughput hint
// is lost.
inline void pause4spin() noexcept {
#if defined(__x86_64__) || defined(__i386__) \
    || defined(_M_IX86) || defined(_M_X64)
#  if defined(_MSC_VER)
    _mm_pause();                              // MSVC: <intrin.h>
#  else
    __builtin_ia32_pause();                   // GCC / Clang
#  endif
#elif defined(__aarch64__) || defined(__arm64__) \
    || defined(_M_ARM64) || defined(__arm__)
#  if defined(_MSC_VER)
    __yield();                                // MSVC ARM64
#  else
    __asm__ __volatile__("yield" ::: "memory");
#  endif
#else
    // Unknown ISA: omit hint (correctness retained, throughput lost).
#endif
}

#endif /*ATOMIC_PRV_MFENCE_H_*/
