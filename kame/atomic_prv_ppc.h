/***************************************************************************
		Copyright (C) 2002-2011 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef ATOMIC_PRV_PPC_H_
#define ATOMIC_PRV_PPC_H_

//! memory barriers. 
inline void readBarrier() {
	asm volatile( "isync" ::: "memory" );
}
inline void writeBarrier() {
	asm volatile( "osync" ::: "memory" );
}
inline void memoryBarrier() {
	asm volatile( "sync" ::: "memory" );
}
//! For spinning.
inline void pauseN(unsigned int cnt) {
	for(unsigned int i = cnt; i != 0; --i)
		asm volatile( "nop" ::: "memory" );
}

#define MAX_SIZEOF_CAS SIZEOF_VOID_P
template <typename T>
//! \return true if old == *target and new value is assigned
inline bool atomicCompareAndSet(T oldv, T newv, T *target ) {
	T ret;
	asm volatile ( "1: \n"
				   "lwarx %[ret], 0, %[target] \n"
				   "cmpw %[ret], %[oldv] \n"
				   "bne- 2f \n"
				   "stwcx. %[newv], 0, %[target] \n"
				   "bne- 1b \n"
				   "2: "
				   : [ret] "=&r" (ret)
				   : [oldv] "r" (oldv), [newv] "r" (newv), [target] "r" (target)
				   : "cc", "memory");
	return (ret == oldv);
}
//! \return target's old value.
template <typename T>
inline T atomicSwap(T newv, T *target ) {
	T ret;
	asm volatile ( "1: \n"
				   "lwarx %[ret], 0, %[target] \n"
				   "stwcx. %[newv], 0, %[target] \n"
				   "bne- 1b"
				   : [ret] "=&r" (ret)
				   : [newv] "r" (newv), [target] "r" (target)
				   : "cc", "memory");
	return ret;
}
template <typename T>
inline void atomicInc(T *target ) {
	T ret;
	asm volatile ( "1: \n"
				   "lwarx %[ret], 0, %[target] \n"
				   "addi %[ret], %[ret], 1 \n"
				   "stwcx. %[ret], 0, %[target] \n"
				   "bne- 1b"
				   : [ret] "=&b" (ret)
				   : [target] "r" (target)
				   : "cc", "memory");
}
template <typename T>
inline void atomicDec(T *target ) {
	T ret;
	asm volatile ( "1: \n"
				   "lwarx %[ret], 0, %[target] \n"
				   "addi %[ret], %[ret], -1 \n"
				   "stwcx. %[ret], 0, %[target] \n"
				   "bne- 1b"
				   : [ret] "=&b" (ret)
				   : [target] "r" (target)
				   : "cc", "memory");
}
template <typename T>
inline void atomicAdd(T *target, T x ) {
	T ret;
	asm volatile ( "1: \n"
				   " lwarx %[ret], 0, %[target] \n"
				   "add %[ret], %[ret], %[x] \n"
				   "stwcx. %[ret], 0, %[target] \n"
				   "bne- 1b"
				   : [ret] "=&r" (ret)
				   : [target] "r" (target), [x] "r" (x)
				   : "cc", "memory");
}
//! \return true if new value is zero.
template <typename T>
inline bool atomicAddAndTest(T *target, T x ) {
	T ret;
	asm volatile ( "1: \n"
				   "lwarx %[ret], 0, %[target] \n"
				   "add %[ret], %[ret], %[x] \n"
				   "stwcx. %[ret], 0, %[target] \n"
				   "bne- 1b"
				   : [ret] "=&r" (ret)
				   : [target] "r" (target), [x] "r" (x)
				   : "cc", "memory");
	return (ret == 0);
}
//! \return zero flag.
template <typename T>
inline bool atomicDecAndTest(T *target ) {
	return atomicAddAndTest(target, (T)-1);
}

#endif /*ATOMIC_PRV_PPC_H_*/
