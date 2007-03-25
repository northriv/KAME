/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
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

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_pod.hpp>
#include <boost/type_traits/is_integral.hpp>

//! memory barriers. 
inline void readBarrier() {
	asm volatile( "lfence" ::: "memory" );
//	asm volatile ("lock; addl $0,0(%%esp)" ::: "memory");
}
inline void writeBarrier() {
	asm volatile( "sfence" ::: "memory" );
//	asm volatile ("lock; addl $0,0(%%esp)" ::: "memory");
}
inline void memoryBarrier() {
	asm volatile( "mfence" ::: "memory" );
//	asm volatile ("lock; addl $0,0(%%esp)" ::: "memory");
}

inline void monitor(void *addr, unsigned int /*size*/) {
	uint32_t cx = 0L;
	uint32_t dx = 0L;
	ASSERT(cg_cpuSpec.hasMonitor);
	asm volatile( 
		"monitor"
//		".byte 0x0f, 0x01, 0xc8"
		:: "a" (addr), "c" (cx), "d" (dx) : "memory" );
}
inline void pause4spin() {
	asm volatile( "pause" ::: "memory" );
}
inline void mwait() {
	uint32_t ax = 0L;
	uint32_t cx = 0L;
	ASSERT(cg_cpuSpec.hasMonitor);
	asm volatile(
		"mwait"
//		".byte 0x0f, 0x01, 0xc9"
		:: "a" (ax), "c" (cx) : "memory" );
}
#if SIZEOF_VOID_P == 4
typedef int32_t int_cas2_each;
typedef int64_t int_cas2_both;
typedef int_cas2_each int_cas_max;
typedef uint32_t uint_cas2_each;
typedef uint64_t uint_cas2_both;
typedef uint_cas2_each uint_cas_max;
#define HAVE_CAS_2
//! Compare-And-Swap 2 long words.
//! \param oldv0 compared with \p target[0].
//! \param oldv1 compared with \p target[1].
//! \param newv0 new value to \p target[0].
//! \param newv1 new value to \p target[1].
template <typename T>
bool atomicCompareAndSet2(
	T oldv0, T oldv1,
	T newv0, T newv1, T *target ) {
	unsigned char ret;
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
	return ret;
}
template <typename T, typename X>
typename boost::enable_if_c<
boost::is_pod<T>::value && (sizeof(T) == sizeof(int_cas2_both)) && (sizeof(X) >= sizeof(int_cas2_each)), bool>::type
atomicCompareAndSet(
								T oldv,
								T newv, X *target ) {
	return atomicCompareAndSet2((uint_cas2_each)(*((uint_cas2_both*)&oldv) % (1uLL << 8 * sizeof(uint_cas2_each))), 
								(uint_cas2_each)(*((uint_cas2_both*)&oldv) / (1uLL << 8 * sizeof(uint_cas2_each))),
								(uint_cas2_each)(*((uint_cas2_both*)&newv) % (1uLL << 8 * sizeof(uint_cas2_each))),
								(uint_cas2_each)(*((uint_cas2_both*)&newv) / (1uLL << 8 * sizeof(uint_cas2_each))),
								(uint_cas2_each*)target);
}
#else
//!\todo x86_64.
#error "Unsupported size of int."
#endif

//! \return true if old == *target and new value is assigned
template <typename T>
typename boost::enable_if_c<
boost::is_pod<T>::value && (sizeof(T) <= sizeof(int_cas_max)), bool>::type
atomicCompareAndSet(T oldv, T newv, T *target ) {
	unsigned char ret;
	asm volatile (
		"  lock; cmpxchg%z2 %2,%3;"
		" sete %0" // ret = zflag ? 1 : 0
		: "=q" (ret), "=a" (oldv)
		: "r" (newv), "m" (*target), "1" (oldv)
		: "memory");
	return ret;
}
template <typename T>
inline T atomicSwap(T v, T *target ) {
	asm volatile (
				"xchg%z0 %0,%1" //lock prefix is not needed.
				: "=r" (v)
				: "m" (*target), "0" (v)
				: "memory" );
	return v;
}
template <typename T>
inline void atomicAdd(T *target, T x ) {
	asm volatile (
				"lock; add%z0 %1,%0"
				:
				: "m" (*target), "ir" (x)
				: "memory" );
}
//! \return true if new value is zero.
template <typename T>
inline bool atomicAddAndTest(T *target, T x ) {
	register unsigned char ret;
	asm volatile (
				"lock; add%z1 %2,%1;"
				" sete %0" // ret = zflag ? 1 : 0
				: "=q" (ret)
				: "m" (*target), "ir" (x)
				: "memory" );
	return ret;
}
template <typename T>
inline void atomicInc(T *target ) {
	asm volatile (
				"lock; inc%z0 %0"
				:
				: "m" (*target)
				: "memory" );
}
template <typename T>
inline void atomicDec(T *target ) {
	asm volatile (
				"lock; dec%z0 %0"
				:
				: "m" (*target)
				: "memory" );
}
//! \return zero flag.
template <typename T>
inline bool atomicDecAndTest(T *target ) {
	register unsigned char ret;
	asm volatile (
				"lock; dec%z1 %1;"
				" sete %0" // ret = zflag ? 1 : 0
				: "=q" (ret)
				: "m" (*target)
				: "memory" );
	return ret;
}

#endif /*ATOMIC_PRV_X86_H_*/
