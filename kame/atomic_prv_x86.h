/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef ATOMIC_PRV_X86_H_
#define ATOMIC_PRV_X86_H_

#include <type_traits>
#include <inttypes.h>

#include "atomic_prv_mfence_x86.h"

#define HAVE_CAS_2

typedef intptr_t int_cas2;
typedef uintptr_t uint_cas2;
#ifndef ATOMIC_PRV_STD_H_
typedef int_cas2 int_cas_max;
typedef uint_cas2 uint_cas_max;
#endif

#if !defined __LP64__ && !defined __LLP64__ && !defined(_WIN64) &&! defined(__MINGW64__)
	#ifdef __SSE2__
		#define HAVE_ATOMIC_RW64
	#endif
#endif

#ifdef HAVE_ATOMIC_RW64
//! \param x must be aligned to 8bytes.
template <typename T>
inline void atomicWrite64(const T &x, T *target) noexcept {
	static_assert(sizeof(T) == 8, "");
	asm (
		" movq %0, %%xmm0;"
		" movq %%xmm0, %1;"
		:
		: "m" (x), "m" (*target)
		: "memory", "%xmm0");
}

//! \param x must be aligned to 8bytes.
template <typename T>
inline void atomicRead64(T *x, const T &target) noexcept {
	static_assert(__alignof__(T) >= 8, "");
	static_assert(sizeof(T) == 8, "");
	asm (
		" movq %0, %%xmm0;"
		" movq %%xmm0, %1;"
		:
		: "m" (target), "m" (*x)
		: "memory", "%xmm0");
}
#endif

//! Compare-And-Swap 2 long words.
//! \param oldv0 compared with \p target[0].
//! \param oldv1 compared with \p target[1].
//! \param newv0 new value to \p target[0].
//! \param newv1 new value to \p target[1].
template <typename T>
inline bool atomicCompareAndSet2(
	T oldv0, T oldv1,
    T newv0, T newv1, T *target ) noexcept {
	unsigned char ret;
#if defined __LP64__ || defined __LLP64__ || defined(_WIN64) || defined(__MINGW64__)
	asm volatile (
		" lock; cmpxchg16b (%%rsi);"
		" sete %0;" // ret = zflag ? 1 : 0
		: "=q" (ret), "=d" (oldv1), "=a" (oldv0)
		: "1" (oldv1), "2" (oldv0),
		"c" (newv1), "b" (newv0),
		"S" (target)
		: "memory");
#else
	asm volatile (
		//gcc with -fPIC cannot handle EBX correctly.
		" push %%ebx;"
		" mov %6, %%ebx;"
		" lock; cmpxchg8b (%%esi);"
		" pop %%ebx;"
		" sete %0;" // ret = zflag ? 1 : 0
		: "=q" (ret), "=d" (oldv1), "=a" (oldv0)
		: "1" (oldv1), "2" (oldv0),
		"c" (newv1), "D" (newv0),
		"S" (target)
		: "memory");
#endif
	return ret;
}
template <typename T, typename X>
inline typename std::enable_if<
std::is_pod<T>::value && (sizeof(T) == sizeof(int_cas2) * 2) && (sizeof(X) == sizeof(int_cas2) * 2), bool>::type
atomicCompareAndSet(
	T oldv,
    T newv, X *target ) noexcept {
	union {
		T x;
		uint_cas2 w[2];
	} newv_, oldv_;
	newv_.x = newv;
	oldv_.x = oldv;

	return atomicCompareAndSet2(oldv_.w[0], oldv_.w[1], newv_.w[0], newv_.w[1], (uint_cas2*)(target));
}

//! \return true if old == *target and new value is assigned
template <typename T>
inline typename std::enable_if<
std::is_pod<T>::value && (sizeof(T) <= sizeof(int_cas_max)), bool>::type
atomicCompareAndSet(T oldv, T newv, T *target ) noexcept {
	unsigned char ret;
	asm volatile (
		"  lock; cmpxchg %2,%3;"
		" sete %0" // ret = zflag ? 1 : 0
		: "=q" (ret), "=a" (oldv)
		: "q" (newv), "m" ( *target), "1" (oldv)
		: "memory");
	return ret;
}
template <typename T>
inline T atomicSwap(T v, T *target ) noexcept {
	asm volatile (
		"xchg %0,%1" //lock prefix is not needed.
		: "=q" (v)
		: "m" ( *target), "0" (v)
		: "memory" );
	return v;
}
template <typename T>
inline void atomicAdd(T *target, T x ) noexcept {
	asm (
		"lock; add %1,%0"
		:
		: "m" ( *target), "q" (x)
		: "memory" );
}
//! \return true if new value is zero.
template <typename T>
inline bool atomicAddAndTest(T *target, T x ) noexcept {
    unsigned char ret;
	asm volatile (
		"lock; add %2,%1;"
		" sete %0" // ret = zflag ? 1 : 0
		: "=q" (ret)
		: "m" ( *target), "q" (x)
		: "memory" );
	return ret;
}
template <typename T>
inline
typename std::enable_if<(4 > sizeof(T)), void>::type
atomicInc(T *target ) noexcept {
	asm (
		"lock; inc%z0 %0"
		:
		: "m" ( *target)
		: "memory" );
}
template <typename T>
inline
typename std::enable_if<(4 == sizeof(T)), void>::type //hack for buggy %z.
atomicInc(T *target ) noexcept {
	asm (
        "lock; incl %0"
		:
		: "m" ( *target)
		: "memory" );
}
template <typename T>
inline
typename std::enable_if<(8 == sizeof(T)), void>::type //hack for buggy %z.
atomicInc(T *target ) noexcept {
	asm (
		"lock; incq %0"
		:
		: "m" ( *target)
		: "memory" );
}

template <typename T>
inline
typename std::enable_if<(4 > sizeof(T)), void>::type
atomicDec(T *target ) noexcept {
	asm (
		"lock; dec%z0 %0"
		:
		: "m" ( *target)
		: "memory" );
}
template <typename T>
inline
typename std::enable_if<(4 == sizeof(T)), void>::type //hack for buggy %z.
atomicDec(T *target ) noexcept {
	asm (
        "lock; decl %0"
		:
		: "m" ( *target)
		: "memory" );
}
template <typename T>
inline
typename std::enable_if<(8 == sizeof(T)), void>::type //hack for buggy %z.
atomicDec(T *target ) noexcept {
	asm (
		"lock; decq %0"
		:
		: "m" ( *target)
		: "memory" );
}
//! \return zero flag.
template <typename T>
inline
typename std::enable_if<(4 > sizeof(T)), bool>::type
atomicDecAndTest(T *target ) noexcept {
    unsigned char ret;
	asm volatile (
		"lock; dec%z1 %1;"
		" sete %0" // ret = zflag ? 1 : 0
		: "=q" (ret)
		: "m" ( *target)
		: "memory" );
	return ret;
}
template <typename T>
inline
typename std::enable_if<(4 == sizeof(T)), bool>::type //hack for buggy %z.
atomicDecAndTest(T *target ) noexcept {
    unsigned char ret;
	asm volatile (
        "lock; decl %1;"
		" sete %0" // ret = zflag ? 1 : 0
		: "=q" (ret)
		: "m" ( *target)
		: "memory" );
	return ret;
}
template <typename T>
inline
typename std::enable_if<(8 == sizeof(T)), bool>::type //hack for buggy %z.
atomicDecAndTest(T *target ) noexcept {
    unsigned char ret;
	asm volatile (
		"lock; decq %1;"
		" sete %0" // ret = zflag ? 1 : 0
		: "=q" (ret)
		: "m" ( *target)
		: "memory" );
	return ret;
}

#endif /*ATOMIC_PRV_X86_H_*/
