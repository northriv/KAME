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
#ifndef ATOMIC_PRV_MFENCE_X86_H_
#define ATOMIC_PRV_MFENCE_X86_H_

#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <emmintrin.h>
    #include <xmmintrin.h>
    #include <x86intrin.h>
#endif

//! memory barriers.
inline void readBarrier() noexcept {
    _mm_lfence();
//	asm volatile( "lfence" ::: "memory" );
//	//	asm volatile ("lock; addl $0,0(%%esp)" ::: "memory");
}
inline void writeBarrier() noexcept {
    _mm_sfence();
//    asm volatile( "sfence" ::: "memory" );
//	//	asm volatile ("lock; addl $0,0(%%esp)" ::: "memory");
}
inline void memoryBarrier() noexcept {
    _mm_mfence();
//    asm volatile( "mfence" ::: "memory" );
//	//	asm volatile ("lock; addl $0,0(%%esp)" ::: "memory");
}

inline void pause4spin() noexcept {
    _mm_pause();
//	asm volatile( "pause" ::: "memory" );
}

#endif /*ATOMIC_PRV_MFENCE_X86_H_*/
