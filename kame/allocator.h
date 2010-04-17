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

#ifndef ALLOCATOR_H_
#define ALLOCATOR_H_

//!\desc
//! Lock-free new/delete operators for small objects.

#include "support.h"
#include "atomic.h"

#define ALLOC_MEMPOOL_SIZE (1024 * 512)
#define ALLOC_MAX_ALLOCATORS (1024 * 1024 * 1024 / ALLOC_MEMPOOL_SIZE)
#define ALLOC_SIZE1 16
#define ALLOC_SIZE2 32
#define ALLOC_SIZE3 64
#define ALLOC_SIZE4 128

template <int SIZE>
class FixedSizeAllocator {
	enum {MEMPOOL_COUNT = ALLOC_MEMPOOL_SIZE / SIZE};
public:
	FixedSizeAllocator() : m_mempool(new char[ALLOC_MEMPOOL_SIZE]), m_idx(0) {
		fprintf(stderr, "new allocator for %dB, starting @ %llx, ending @ %llx\n", SIZE,
			(unsigned long long)m_mempool, (unsigned long long)( &m_mempool[ALLOC_MEMPOOL_SIZE]));
		memset(m_flags, 0, MEMPOOL_COUNT);
		memoryBarrier();
	}
	~FixedSizeAllocator() { delete [] m_mempool; }
	inline void *allocate_pooled() {
		for(int cnt = 0; cnt < MEMPOOL_COUNT; ++cnt) {
			int idx = m_idx;
			if( !m_flags[idx]) {
//				int ret = atomicSwap(1, &m_flags[idx]);
//				if( !ret) {
				if(atomicCompareAndSet(0, 1, &m_flags[idx])) {
					readBarrier();
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
	inline bool deallocate_pooled(void *p) {
		if((p < m_mempool) || (p >= &m_mempool[ALLOC_MEMPOOL_SIZE]))
			return false;
		int idx = static_cast<size_t>((char *)p - m_mempool) / SIZE;
		ASSERT(m_flags[idx] == 1);
		writeBarrier();
		m_flags[idx] = 0;
//		m_idx = idx;
		return true;
	}
	static inline void *allocate(size_t size) {
		int acnt = s_allocators_cnt;
		int aidx = s_curr_allocator_idx;
		for(int cnt = 0;; ++cnt) {
			FixedSizeAllocator *alloc = s_allocators[aidx];
			if(alloc) {
				if(void *p = alloc->allocate_pooled()) {
//					fprintf(stderr, "alloc %llx\n", (long long) p);
					memset( p, 0, size);
					return p;
				}
				++aidx;
				if((aidx >= acnt) && (cnt < acnt))
					aidx = 0;
			}
			else {
				alloc = new FixedSizeAllocator;
				atomicInc( &s_allocators_cnt);
				writeBarrier();
				if(atomicCompareAndSet((FixedSizeAllocator *)0, alloc, &s_allocators[aidx]))
					++acnt;
				else {
					delete alloc;
					atomicDec( &s_allocators_cnt);
				}
			}
			s_curr_allocator_idx = aidx;
		}
	}
	static inline bool deallocate(void *p) {
		int acnt = s_allocators_cnt;
		int aidx = std::min(s_curr_allocator_idx, acnt - 1);
		for(int cnt = 0; cnt < acnt; ++cnt) {
			FixedSizeAllocator *alloc = s_allocators[aidx];
			if(alloc) {
				if(alloc->deallocate_pooled(p)) {
//					fprintf(stderr, "dealloc %llx\n", (long long) p);
					s_curr_allocator_idx = aidx;
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
private:
	char *m_mempool;
	int m_idx;
	int m_flags[MEMPOOL_COUNT];
	static FixedSizeAllocator *s_allocators[ALLOC_MAX_ALLOCATORS];
	static int s_curr_allocator_idx;
	static int s_allocators_cnt;
	void* operator new(size_t size) throw() {
		return malloc(size);
	}
	void operator delete(void* p) {
		free(p);
	}
};

inline void* operator new(size_t size) throw() {
	if(size <= ALLOC_SIZE1)
		return FixedSizeAllocator<ALLOC_SIZE1>::allocate(size);
	if(size <= ALLOC_SIZE2)
		return FixedSizeAllocator<ALLOC_SIZE2>::allocate(size);
	if(size <= ALLOC_SIZE3)
		return FixedSizeAllocator<ALLOC_SIZE3>::allocate(size);
	if(size <= ALLOC_SIZE4)
		return FixedSizeAllocator<ALLOC_SIZE4>::allocate(size);
	return malloc(size);
}
inline void operator delete(void* p) {
	if(FixedSizeAllocator<ALLOC_SIZE1>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE2>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE3>::deallocate(p))
		return;
	if(FixedSizeAllocator<ALLOC_SIZE4>::deallocate(p))
		return;
	free(p);
}

#endif /* ALLOCATOR_H_ */
