/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/

#include "allocator.h"
template <int>
FixedSizeAllocator<ALLOC_SIZE1> *FixedSizeAllocator<ALLOC_SIZE1>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE1>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE1>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE2> *FixedSizeAllocator<ALLOC_SIZE2>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE2>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE2>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE3> *FixedSizeAllocator<ALLOC_SIZE3>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE3>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE3>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE4> *FixedSizeAllocator<ALLOC_SIZE4>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE4>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE4>::s_allocators_cnt;

template <int SIZE>
FixedSizeAllocator<SIZE>::FixedSizeAllocator()  : m_mempool(new char[ALLOC_MEMPOOL_SIZE]), m_idx(0) {
	fprintf(stderr, "new allocator for %dB, starting @ %llx, ending @ %llx\n", SIZE,
		(unsigned long long)m_mempool, (unsigned long long)( &m_mempool[ALLOC_MEMPOOL_SIZE]));
	memset(m_flags, 0, MEMPOOL_COUNT);
	memoryBarrier();
}
template <int SIZE>
FixedSizeAllocator<SIZE>::~FixedSizeAllocator() {
	delete [] m_mempool;
}
template <int SIZE>
bool
FixedSizeAllocator<SIZE>::trySetupNewAllocator(int aidx) {
	FixedSizeAllocator *alloc = new FixedSizeAllocator;
	atomicInc( &s_allocators_cnt);
	writeBarrier();
	if(atomicCompareAndSet((FixedSizeAllocator *)0, alloc, &s_allocators[aidx]))
		return true;

	delete alloc;
	atomicDec( &s_allocators_cnt);
	return false;
}

template class FixedSizeAllocator<ALLOC_SIZE1>;
template class FixedSizeAllocator<ALLOC_SIZE2>;
template class FixedSizeAllocator<ALLOC_SIZE3>;
template class FixedSizeAllocator<ALLOC_SIZE4>;
