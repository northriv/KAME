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

//! memory barriers. 
inline void readBarrier() {
	asm volatile( "lfence" ::: "memory" );
}
inline void writeBarrier() {
	asm volatile( "sfence" ::: "memory" );
}
inline void memoryBarrier() {
	asm volatile( "mfence" ::: "memory" );
}
//! For spinning.
inline void pauseN(unsigned int cnt) {
	for(unsigned int i = cnt; i != 0; --i) {
		asm volatile( "pause" ::: "memory" );
	}
}
//! \return true if old == *target and new value is assigned
template <typename T>
inline bool atomicCompareAndSet(T oldv, T newv, T *target ) {
	register unsigned char ret;
	asm volatile (
		"  lock; cmpxchg%z2 %2,%3;"
		" sete %0" // ret = zflag ? 1 : 0
		: "=q" (ret), "=a" (oldv)
		: "r" (newv), "m" (*target), "1" (oldv)
		: "memory");
	return ret;
}
#if SIZEOF_VOID_P == 4
#define HAVE_CAS_2
//! Compare-And-Swap 2 long words.
//! \param oldv0 compared with \p target[0].
//! \param oldv1 compared with \p target[1].
//! \param newv0 new value to \p target[0].
//! \param newv1 new value to \p target[1].
inline bool atomicCompareAndSet2(
	uint32_t oldv0, uint32_t oldv1,
	uint32_t newv0, uint32_t newv1, uint32_t *target ) {
	unsigned char ret;
	asm volatile (
		//gcc with -fPIC cannot handle EBX correctly.
		" push %%ebx;"
		" mov %6, %%ebx;"
		" lock; cmpxchg8b %7;"
		" pop %%ebx;"
		" sete %0;" // ret = zflag ? 1 : 0
		: "=r" (ret), "=d" (oldv1), "=a" (oldv0)
		: "1" (oldv1), "2" (oldv0),
		"c" (newv1), "g" (newv0),
		"m" (*target)
		: "memory");
	return ret;
}
inline bool atomicCompareAndSet2(
								int32_t oldv0, int32_t oldv1,
								int32_t newv0, int32_t newv1, int32_t *target ) {
	return atomicCompareAndSet2((uint32_t) oldv0, (uint32_t) oldv1,
								(uint32_t) newv0, (uint32_t) newv1, (uint32_t*) target);
}
#endif
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
