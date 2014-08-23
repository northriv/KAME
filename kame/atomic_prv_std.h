/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef ATOMIC_PRV_STD_H_
#define ATOMIC_PRV_STD_H_

#include <type_traits>
#include <inttypes.h>
#include <atomic>

#include "atomic_prv_mfence_x86.h"

#if ATOMIC_LLONG_LOCK_FREE == 2
    typedef long long int_cas_max;
#elif ATOMIC_LONG_LOCK_FREE == 2
    typedef long int_cas_max;
#elif ATOMIC_INT_LOCK_FREE == 2
    typedef int int_cas_max;
#endif
typedef int_cas_max uint_cas_max;

template <typename T>
class atomic<T, typename std::enable_if<std::is_integral<T>::value || std::is_pointer<T>::value>::type>
: public std::atomic<T> {
public:
    atomic() : std::atomic<T>() {}
    atomic(const atomic &t) : std::atomic<T>() { *this = (T)t;}
    atomic(const T &t) : std::atomic<T>(t) {}
    atomic& operator=(const T &t) {this->store(t); return *this; }
    bool compare_exchange_strong(const T &oldv, const T &newv) {
        T expected = oldv;
        return std::atomic<T>::compare_exchange_strong(expected, newv);
    }
    bool decAndTest() {
        return (--( *this) == 0);
    }
    bool addAndTest(T t) {
        return ((( *this) += t) == 0);
    }
};

#endif /*ATOMIC_PRV_STD_H_*/
