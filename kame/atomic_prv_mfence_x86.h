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
#ifndef ATOMIC_PRV_MFENCE_X86_H_
#define ATOMIC_PRV_MFENCE_X86_H_

#include <type_traits>
#include <inttypes.h>

//! memory barriers.
inline void readBarrier() {
#ifdef _MSC_VER
    __asm lfence
#else
	asm volatile( "lfence" ::: "memory" );
	//	asm volatile ("lock; addl $0,0(%%esp)" ::: "memory");
#endif
}
inline void writeBarrier() {
#ifdef _MSC_VER
    __asm sfence
#else
    asm volatile( "sfence" ::: "memory" );
	//	asm volatile ("lock; addl $0,0(%%esp)" ::: "memory");
#endif
}
inline void memoryBarrier() {
#ifdef _MSC_VER
    __asm mfence
#else
    asm volatile( "mfence" ::: "memory" );
	//	asm volatile ("lock; addl $0,0(%%esp)" ::: "memory");
#endif
}

inline void pause4spin() {
	asm volatile( "pause" ::: "memory" );
}

#endif /*ATOMIC_PRV_MFENCE_X86_H_*/
