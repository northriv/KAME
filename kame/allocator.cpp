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

#define ALLOC_MEMPOOL_SIZE (1024 * 1024)
#define ALLOC_MAX_ALLOCATORS (1024 * 1024 * 1024 / ALLOC_MEMPOOL_SIZE)
#define ALLOC_SIZE1 8
#define ALLOC_SIZE2 16
#define ALLOC_SIZE3 24
#define ALLOC_SIZE4 32
#define ALLOC_SIZE5 40
#define ALLOC_SIZE6 48
#define ALLOC_SIZE7 56
#define ALLOC_SIZE8 64
#define ALLOC_SIZE9 80
#define ALLOC_SIZE10 96
#define ALLOC_SIZE11 112
#define ALLOC_SIZE12 128
#define ALLOC_SIZE13 160
#define ALLOC_SIZE14 192
#define ALLOC_SIZE15 224
#define ALLOC_SIZE16 256

template <int SIZE>
class FixedSizeAllocator {
	enum {MEMPOOL_COUNT = ALLOC_MEMPOOL_SIZE / SIZE};
	typedef uint8_t FUINT;
public:
	FixedSizeAllocator();
	~FixedSizeAllocator();
	inline void *allocate_pooled() {
		for(int cnt = 0; cnt < MEMPOOL_COUNT; ++cnt) {
			int idx = m_idx;
			if( !m_flags[idx]) {
//				if(atomicSwap(1, &m_flags[idx]) == 0) {
				if(atomicCompareAndSet((FUINT)0, (FUINT)1, &m_flags[idx])) {
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
	static inline bool trySetupNewAllocator(int aidx);
	static inline void *allocate(size_t size) {
		int acnt = s_allocators_cnt;
		int aidx = s_curr_allocator_idx;
		for(int cnt = 0;; ++cnt) {
			FixedSizeAllocator *alloc = s_allocators[aidx];
			if(alloc) {
				if(void *p = alloc->allocate_pooled()) {
					for(unsigned int i = 0; i < SIZE / sizeof(uint64_t); ++i)
						static_cast<uint64_t *>(p)[i] = 0; //zero clear.
					return p;
				}
				++aidx;
				if((aidx >= acnt) && (cnt < acnt))
					aidx = 0;
			}
			else {
				trySetupNewAllocator(aidx);
				acnt = s_allocators_cnt;
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
	FUINT m_flags[MEMPOOL_COUNT];
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

template <int SIZE>
inline bool
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

void* operator new(size_t size) throw() {
	if(size <= ALLOC_SIZE1)
		return FixedSizeAllocator<ALLOC_SIZE1>::allocate(size);
	if(size <= ALLOC_SIZE2)
		return FixedSizeAllocator<ALLOC_SIZE2>::allocate(size);
	if(size <= ALLOC_SIZE3)
		return FixedSizeAllocator<ALLOC_SIZE3>::allocate(size);
	if(size <= ALLOC_SIZE4)
		return FixedSizeAllocator<ALLOC_SIZE4>::allocate(size);
	if(size <= ALLOC_SIZE5)
		return FixedSizeAllocator<ALLOC_SIZE5>::allocate(size);
	if(size <= ALLOC_SIZE6)
		return FixedSizeAllocator<ALLOC_SIZE6>::allocate(size);
	if(size <= ALLOC_SIZE7)
		return FixedSizeAllocator<ALLOC_SIZE7>::allocate(size);
	if(size <= ALLOC_SIZE8)
		return FixedSizeAllocator<ALLOC_SIZE8>::allocate(size);
	if(size <= ALLOC_SIZE9)
		return FixedSizeAllocator<ALLOC_SIZE9>::allocate(size);
	if(size <= ALLOC_SIZE10)
		return FixedSizeAllocator<ALLOC_SIZE10>::allocate(size);
	if(size <= ALLOC_SIZE11)
		return FixedSizeAllocator<ALLOC_SIZE11>::allocate(size);
	if(size <= ALLOC_SIZE12)
		return FixedSizeAllocator<ALLOC_SIZE12>::allocate(size);
	if(size <= ALLOC_SIZE13)
		return FixedSizeAllocator<ALLOC_SIZE13>::allocate(size);
	if(size <= ALLOC_SIZE14)
		return FixedSizeAllocator<ALLOC_SIZE14>::allocate(size);
	if(size <= ALLOC_SIZE15)
		return FixedSizeAllocator<ALLOC_SIZE15>::allocate(size);
	if(size <= ALLOC_SIZE16)
		return FixedSizeAllocator<ALLOC_SIZE16>::allocate(size);
	return malloc(size);
}
void operator delete(void* p) throw() {
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
template <int>
FixedSizeAllocator<ALLOC_SIZE5> *FixedSizeAllocator<ALLOC_SIZE5>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE5>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE5>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE6> *FixedSizeAllocator<ALLOC_SIZE6>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE6>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE6>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE7> *FixedSizeAllocator<ALLOC_SIZE7>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE7>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE7>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE8> *FixedSizeAllocator<ALLOC_SIZE8>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE8>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE8>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE9> *FixedSizeAllocator<ALLOC_SIZE9>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE9>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE9>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE10> *FixedSizeAllocator<ALLOC_SIZE10>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE10>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE10>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE11> *FixedSizeAllocator<ALLOC_SIZE11>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE11>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE11>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE12> *FixedSizeAllocator<ALLOC_SIZE12>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE12>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE12>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE13> *FixedSizeAllocator<ALLOC_SIZE13>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE13>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE13>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE14> *FixedSizeAllocator<ALLOC_SIZE14>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE14>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE14>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE15> *FixedSizeAllocator<ALLOC_SIZE15>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE15>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE15>::s_allocators_cnt;
template <int>
FixedSizeAllocator<ALLOC_SIZE16> *FixedSizeAllocator<ALLOC_SIZE16>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <int>
int FixedSizeAllocator<ALLOC_SIZE16>::s_curr_allocator_idx;
template <int>
int FixedSizeAllocator<ALLOC_SIZE16>::s_allocators_cnt;

template <int SIZE>
FixedSizeAllocator<SIZE>::FixedSizeAllocator()  : m_mempool(new char[ALLOC_MEMPOOL_SIZE]), m_idx(0) {
	fprintf(stderr, "new allocator for %dB, starting @ %llx, ending @ %llx\n", SIZE,
		(unsigned long long)m_mempool, (unsigned long long)( &m_mempool[ALLOC_MEMPOOL_SIZE]));
	memset(m_flags, 0, MEMPOOL_COUNT);
	C_ASSERT(SIZE % sizeof(uint64_t) == 0);
	ASSERT((uintptr_t)m_mempool % sizeof(uint64_t) == 0);
	memoryBarrier();
}
template <int SIZE>
FixedSizeAllocator<SIZE>::~FixedSizeAllocator() {
	delete [] m_mempool;
}

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
