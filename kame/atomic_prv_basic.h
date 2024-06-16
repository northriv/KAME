/***************************************************************************
        Copyright (C) 2002-2016 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef ATOMIC_PRV_BASIC_H_
#define ATOMIC_PRV_BASIC_H_

#ifndef USE_STD_ATOMIC
    #ifdef _MSC_VER
        #define USE_STD_ATOMIC
    #endif
    #if defined __clang__
        #define USE_STD_ATOMIC
    #endif
    #if defined __GNUC__ && !defined __clang__
        #if __GNUC__ >= 5 && __GNUC_MINOR__ >= 1
            #define USE_STD_ATOMIC
        #endif
    #endif
#endif

#include <stdint.h>

template <typename T, class Enable = void > class atomic;

#ifdef USE_STD_ATOMIC
    #include "atomic_prv_std.h"
#else

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__\
    || defined __x86_64__ || defined _M_IX86 || defined _M_X64
    #include "atomic_prv_x86.h"
#else
    #if defined __ppc__ || defined __POWERPC__ || defined __powerpc__
        #include "atomic_prv_ppc.h"
    #elif defined __arm__
        #include "atomic_prv_arm.h"
    #else
        #error Unsupported processor
    #endif // __ppc__
#endif // __i386__

#include <type_traits>

//! atomic access to POD type capable of CAS.
template <typename T>
class atomic_pod_cas {
public:
    atomic_pod_cas() noexcept = default;
    atomic_pod_cas(T t) noexcept : m_var(t) {}
    atomic_pod_cas(const atomic_pod_cas &t) noexcept : m_var(t) {}
    operator T() const noexcept { T x = m_var; readBarrier(); return x;}
    atomic_pod_cas &operator=(T t) noexcept {
        writeBarrier(); m_var = t; return *this;
    }
    atomic_pod_cas &operator=(const atomic_pod_cas &x) noexcept {
        writeBarrier(); m_var = x.m_var; return *this;
    }
    T exchange(T newv) noexcept {
        T old = atomicSwap(newv, &m_var);
        return old;
    }
    bool compare_set_strong(T oldv, T newv) noexcept {
        bool ret = atomicCompareAndSet(oldv, newv, &m_var);
        return ret;
    }
protected:
    T m_var;
};

//! atomic access to POD type capable of CAS2.
template <typename T>
class atomic_pod_cas2 {
public:
    atomic_pod_cas2() noexcept = default;
    atomic_pod_cas2(T t) noexcept : m_var(t) {}
    atomic_pod_cas2(const atomic_pod_cas2 &t) noexcept : m_var(t) {}
    operator T() const noexcept {
        for(;;) {
            T oldv = m_var;
            if(atomicCompareAndSet(oldv, oldv, &m_var)) {
                return oldv;
            }
        }
    }
    atomic_pod_cas2 &operator=(T t) noexcept {
        writeBarrier();
        for(;;) {
            T oldv = m_var;
            if(atomicCompareAndSet(oldv, t, &m_var))
                break;
        }
        return *this;
    }
    atomic_pod_cas2 &operator=(const atomic_pod_cas2 &x) noexcept {
        *this = (T)x;
        return *this;
    }
    T exchange(T newv) noexcept {
        for(;;) {
            T oldv = m_var;
            if(atomicCompareAndSet(oldv, newv, &m_var)) {
                return oldv;
            }
        }
    }
    bool compare_set_strong(T oldv, T newv) noexcept {
        bool ret = atomicCompareAndSet(oldv, newv, &m_var);
        return ret;
    }
protected:
    T m_var
#ifdef _MSC_VER
    __declspec(align(8));
#else
    __attribute__((aligned(8)));
#endif
};

//! atomic access to POD type capable of CAS2.
template <typename T>
class atomic<T, typename std::enable_if<
(sizeof(int_cas2) * 2 == sizeof(T)) && std::is_pod<T>::value>::type>
: public atomic_pod_cas2<T> {
public:
    atomic() noexcept = default;
    atomic(T t) noexcept : atomic_pod_cas2<T>(t) {}
    atomic(const atomic &t) noexcept = default;
};

//! atomic access to POD type capable of CAS.
template <typename T>
class atomic<T, typename std::enable_if<
(sizeof(int_cas_max) >= sizeof(T)) && std::is_pod<T>::value &&
!std::is_integral<T>::value>::type>
: public atomic_pod_cas<T> {
public:
    atomic() noexcept = default;
    atomic(T t) noexcept : atomic_pod_cas<T>(t) {}
    atomic(const atomic &t) noexcept = default;
};

//! atomic access to integer-POD-type capable of CAS.
template <typename T>
class atomic<T, typename std::enable_if<
(sizeof(int_cas_max) >= sizeof(T)) && std::is_integral<T>::value>::type >
: public atomic_pod_cas<T> {
public:
    atomic() noexcept : atomic_pod_cas<T>((T)0) {}
    atomic(T t) noexcept : atomic_pod_cas<T>(t) {}
    atomic(const atomic &t) = default;
    //! Note that the return value is atomically given.
    atomic &operator++() noexcept {writeBarrier(); atomicInc( &this->m_var); return *this;}
    //! Note that the return value is atomically given.
    atomic &operator--() noexcept {writeBarrier(); atomicDecAndTest( &this->m_var); return *this;}
    //! Note that the return value is atomically given.
    atomic &operator+=(T t) noexcept {writeBarrier(); atomicAdd( &this->m_var, t); return *this;}
    //! Note that the return value is atomically given.
    atomic &operator-=(T t) noexcept {writeBarrier(); atomicAdd( &this->m_var, -t); return *this;}
    bool decAndTest() noexcept {
        bool ret = atomicDecAndTest( &this->m_var);
        return ret;
    }
    bool addAndTest(T t) noexcept {
        bool ret = atomicAddAndTest( &this->m_var, t);
        return ret;
    }
};

#endif //!USE_STD_ATOMIC

#endif /*ATOMIC_PRV_BASIC_H_*/
