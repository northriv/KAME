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

// Barriers + spin-pause: unified, portable C++17 (std::atomic_thread_fence).
// Per-arch dispatch is no longer needed — only pause4spin retains a small
// arch-specific hint, encapsulated inside atomic_prv_mfence.h.
#include "atomic_prv_mfence.h"

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
// On 64-bit targets this is trivially lock-free; on 32-bit targets the
// compiler maps it to a DCAS instruction (CMPXCHG8B on i486+, LDREXD/STREXD
// on ARMv7-A) when available — std::atomic<uint64_t> stays lock-free.
// Targets without hardware DCAS (i386, ARMv5/v6) previously used the
// in-tree DCAS fallback; that path should be revived rather than blocked
// here, hence no static_assert.

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
