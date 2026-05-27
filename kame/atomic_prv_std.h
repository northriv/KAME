/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef ATOMIC_PRV_STD_H_
#define ATOMIC_PRV_STD_H_

#include <type_traits>
#include <inttypes.h>
#include <atomic>

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__\
    || defined __x86_64__ || defined _M_IX86 || defined _M_X64
    #include "atomic_prv_mfence_x86.h"
#elif defined(__arm64__) || defined(__aarch64__) || defined(_M_ARM64)
    // __arm64__  — Apple Silicon (Clang)
    // __aarch64__ — Linux ARM64 (GCC/Clang)
    // _M_ARM64   — MSVC ARM64 (not otherwise supported)
    #include "atomic_prv_mfence_arm8.h"
#else
    #error
#endif

#if ATOMIC_LLONG_LOCK_FREE == 2
    typedef long long int_cas_max;
#elif ATOMIC_LONG_LOCK_FREE == 2
    typedef long int_cas_max;
#elif ATOMIC_INT_LOCK_FREE == 2
    typedef int int_cas_max;
#endif
typedef int_cas_max uint_cas_max;

// The STM transaction framework packs multiple fields into atomic<uint64_t>
// (m_priority_state, m_recent_ops_state, RunnerCounterEntry::v, stamps, ...).
// These must be truly lock-free; a mutex-based fallback would break the
// lock-freedom guarantees of the entire framework.
// On 64-bit x86/ARM64: always satisfied.
// On 32-bit x86 (i486+): CMPXCHG8B provides lock-free 64-bit CAS — satisfied.
// On 32-bit ARM or other platforms without native 64-bit atomics: not supported.
static_assert(ATOMIC_LLONG_LOCK_FREE == 2,
    "KAME requires always-lock-free 64-bit atomics (ATOMIC_LLONG_LOCK_FREE==2). "
    "Supported: 64-bit x86/ARM64, or 32-bit x86 with i486+ (CMPXCHG8B).");

template <typename T>
class atomic<T, typename std::enable_if<std::is_integral<T>::value || std::is_pointer<T>::value>::type>
: public std::atomic<T> {
public:
    atomic() noexcept = default;
    atomic(const atomic &t) noexcept : std::atomic<T>() { *this = (T)t;}
    atomic(const T &t) : std::atomic<T>(t) {}
    atomic& operator=(const T &t) noexcept{this->store(t); return *this; }
    bool compare_set_strong(const T &oldv, const T &newv) noexcept {
        T expected = oldv;
        return std::atomic<T>::compare_exchange_strong(expected, newv,
            std::memory_order_acq_rel, std::memory_order_acquire);
    }
    bool compare_set_weak(const T &oldv, const T &newv) noexcept {
        T expected = oldv;
        return std::atomic<T>::compare_exchange_weak(expected, newv,
            std::memory_order_acq_rel, std::memory_order_relaxed);
    }
    bool decAndTest() noexcept {
        return this->fetch_sub(1, std::memory_order_acq_rel) == 1;
    }
};

#endif /*ATOMIC_PRV_STD_H_*/
