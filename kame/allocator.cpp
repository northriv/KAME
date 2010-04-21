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

#include "support.h"
#include "atomic.h"

template <int SIZE>
FixedSizeAllocator<SIZE>::FixedSizeAllocator()  : m_idx(0) {
	memset(m_flags, 0, MEMPOOL_COUNT);
	C_ASSERT(SIZE % ALLOC_ALIGNMENT == 0);
	ASSERT((uintptr_t)m_mempool % ALLOC_ALIGNMENT == 0);
//	memoryBarrier();
}
template <int SIZE>
FixedSizeAllocator<SIZE>::~FixedSizeAllocator() {
	for(int idx = 0; idx < MEMPOOL_COUNT; ++idx) {
		if(m_flags[idx]) {
			fprintf(stderr, "Leak found for %dB @ %llx.\n", SIZE,
				(unsigned long long) &m_mempool[idx]);
		}
	}

}

template <int SIZE>
inline void *
FixedSizeAllocator<SIZE>::allocate_pooled() {
	for(int cnt = 0; cnt < MEMPOOL_COUNT; ++cnt) {
		int idx = m_idx;
		if( !m_flags[idx]) {
//				if(atomicSwap(1, &m_flags[idx]) == 0) {
			if(atomicCompareAndSet((FUINT)0, (FUINT)1, &m_flags[idx])) {
				return &m_mempool[idx * SIZE];
			}
		}
		++idx;
		if(idx == MEMPOOL_COUNT)
			idx = 0;
		m_idx = idx;
	}
	return 0;
}
template <int SIZE>
inline bool
FixedSizeAllocator<SIZE>::deallocate_pooled(void *p) {
	int idx = static_cast<size_t>((char *)p - m_mempool) / SIZE;
	ASSERT(m_flags[idx] == 1);
	m_flags[idx] = 0;
//	m_idx = idx;
	return true;
}
template <int SIZE>
bool
FixedSizeAllocator<SIZE>::trySetupNewAllocator(int aidx) {
	FixedSizeAllocator *alloc = new FixedSizeAllocator;
	if(atomicCompareAndSet((FixedSizeAllocator *)0, alloc, &s_allocators[aidx])) {
		writeBarrier(); //for alloc.
		atomicInc( &s_allocators_cnt);
//		writeBarrier();
		fprintf(stderr, "New memory pool for %dB, starting @ %llx w/ len. of %llxB.\n", SIZE,
			(unsigned long long)alloc->m_mempool, (unsigned long long)ALLOC_MEMPOOL_SIZE);
		return true;
	}
	delete alloc;
	return false;
}
template <int SIZE>
void *
FixedSizeAllocator<SIZE>::allocate(size_t size) {
	int acnt = s_allocators_cnt;
	int aidx = std::min(s_curr_allocator_idx, acnt - 1);
	readBarrier(); //for alloc.
	for(int cnt = 0;; ++cnt) {
		if(cnt == acnt) {
			trySetupNewAllocator(acnt);
			acnt = s_allocators_cnt;
			readBarrier(); //for alloc.
			aidx = acnt - 1;
		}
		FixedSizeAllocator *alloc = s_allocators[aidx];
		if(void *p = alloc->allocate_pooled()) {
			readBarrier(); //for the pooled memory.
			for(unsigned int i = 0; i < SIZE / sizeof(uint64_t); ++i)
				static_cast<uint64_t *>(p)[i] = 0; //zero clear.
			return p;
		}
		++aidx;
		if(aidx >= acnt)
			aidx = 0;
		s_curr_allocator_idx = aidx;
	}
}
template <int SIZE>
inline bool
FixedSizeAllocator<SIZE>::deallocate(void *p) {
	int acnt = s_allocators_cnt;
	int aidx = std::min(s_curr_allocator_idx, acnt - 1);
	for(int cnt = 0; cnt < acnt; ++cnt) {
		FixedSizeAllocator *alloc = s_allocators[aidx];
		if((p >= alloc->m_mempool) && (p < &alloc->m_mempool[ALLOC_MEMPOOL_SIZE])) {
			if(alloc->deallocate_pooled(p)) {
	//				s_curr_allocator_idx = aidx;
				return true;
			}
		}
		if(aidx == 0)
			aidx = acnt - 1;
		else
			--aidx;
	}
	return false;
}
template <int SIZE>
void
FixedSizeAllocator<SIZE>::release_pools() {
	int acnt = s_allocators_cnt;
	for(int aidx = 0; aidx < acnt; ++aidx) {
		FixedSizeAllocator *alloc = s_allocators[aidx];
		delete alloc;
	}
}
template <int SIZE>
void*
FixedSizeAllocator<SIZE>::operator new(size_t size) throw() {
	return malloc(size);
}
template <int SIZE>
void
FixedSizeAllocator<SIZE>::operator delete(void* p) {
	free(p);
}

void operator delete(void* p) throw() {
	memoryBarrier(); //for the pooled memory
	if(FixedSizeAllocator<ALLOC_SIZE1>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE2>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE3>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE4>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE5>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE6>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE7>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE8>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE9>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE10>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE11>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE12>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE13>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE14>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE15>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE16>::deallocate(p))
		return;
	free(p);
}

void release_pools() {
	FixedSizeAllocator<ALLOC_SIZE1>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE2>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE3>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE4>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE5>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE6>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE7>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE8>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE9>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE10>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE11>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE12>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE13>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE14>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE15>::release_pools();
	FixedSizeAllocator<ALLOC_SIZE16>::release_pools();
}

template <int SIZE>
FixedSizeAllocator<SIZE> *FixedSizeAllocator<SIZE>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int SIZE>
int FixedSizeAllocator<SIZE>::s_curr_allocator_idx;
template <int SIZE>
int FixedSizeAllocator<SIZE>::s_allocators_cnt;

template class FixedSizeAllocator<ALLOC_SIZE1>;
template class FixedSizeAllocator<ALLOC_SIZE2>;
template class FixedSizeAllocator<ALLOC_SIZE3>;
template class FixedSizeAllocator<ALLOC_SIZE4>;
template class FixedSizeAllocator<ALLOC_SIZE5>;
template class FixedSizeAllocator<ALLOC_SIZE6>;
template class FixedSizeAllocator<ALLOC_SIZE7>;
template class FixedSizeAllocator<ALLOC_SIZE8>;
template class FixedSizeAllocator<ALLOC_SIZE9>;
template class FixedSizeAllocator<ALLOC_SIZE10>;
template class FixedSizeAllocator<ALLOC_SIZE11>;
template class FixedSizeAllocator<ALLOC_SIZE12>;
template class FixedSizeAllocator<ALLOC_SIZE13>;
template class FixedSizeAllocator<ALLOC_SIZE14>;
template class FixedSizeAllocator<ALLOC_SIZE15>;
template class FixedSizeAllocator<ALLOC_SIZE16>;

//struct _PoolReleaser {
//	~_PoolReleaser() {
//		release_pools();
//	}
//} _pool_releaser;
