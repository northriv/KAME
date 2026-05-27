/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef KAMEPOOLALLOC_ATOMIC_MFENCE_H_
#define KAMEPOOLALLOC_ATOMIC_MFENCE_H_

// Minimal arch-select for `readBarrier()` / `writeBarrier()` /
// `memoryBarrier()` / `pause4spin()`.  Mirrors the include shape in
// `kame/atomic_prv_basic.h` → `atomic_prv_std.h`, but pulls only the
// barrier helpers — not `std::atomic<T>` wrappers, not the
// `atomic_shared_ptr` machinery.  `kamepoolalloc` is the standalone
// pool-allocator dylib and intentionally has no upward dependency on
// `kame/`.

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__ \
    || defined __x86_64__ || defined _M_IX86 || defined _M_X64
    #include "atomic_prv_mfence_x86.h"
#elif defined(__arm64__) || defined(__aarch64__) || defined(_M_ARM64)
    // __arm64__   — Apple Silicon (Clang)
    // __aarch64__ — Linux ARM64 (GCC/Clang)
    // _M_ARM64    — MSVC ARM64
    #include "atomic_prv_mfence_arm8.h"
#else
    #error "kamepoolalloc: unsupported architecture (need x86 or ARM64)"
#endif

#endif /*KAMEPOOLALLOC_ATOMIC_MFENCE_H_*/
