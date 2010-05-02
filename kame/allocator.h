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

#include <new>
#include <stdint.h>
#include <stdlib.h>

#define ALLOC_MEMPOOL_SIZE (1024 * 512) //512KiB
#define ALLOC_MAX_ALLOCATORS (1024 * 8) //4GiB max.
#define ALLOC_ALIGNMENT (sizeof(double)) //i.e. 8B

//! \brief Fast lock-free allocators for small objects: new(), new[](), delete(), delete[]() operators.\n
//! Arbitrary sizes of memory in a unit of double-quad word less than 4KiB
//! can be allocated from a memory pool. The larger memory is provided by malloc().\n
//! Those memory pools won't be released once being secured in order to reduce efforts for locking of pools.
//! \sa allocator_test.cpp.
template <unsigned int ALIGN>
class PooledAllocator {
public:
	~PooledAllocator();
	template <unsigned int SIZE>
	static void *allocate() ;
	static inline bool deallocate(void *p);
	static void release_pools();

	typedef uintptr_t FUINT;
private:
	PooledAllocator(char *addr);
	template <unsigned int SIZE>
	inline void *allocate_pooled();
	inline bool deallocate_pooled(void *p);
	static bool trySetupNewAllocator(int aidx);
	enum {FLAGS_COUNT = ALLOC_MEMPOOL_SIZE / ALIGN / sizeof(FUINT) / 8};
	enum {MMAP_SPACE_SIZE = 1024 * 1024 * 16, //16MiB
		NUM_ALLOCATORS_IN_SPACE = MMAP_SPACE_SIZE / ALLOC_MEMPOOL_SIZE,
		MMAP_SPACES_COUNT = ALLOC_MAX_ALLOCATORS / NUM_ALLOCATORS_IN_SPACE};
	char *m_mempool;
	int m_idx; //a hint for searching in a sparse area.
	FUINT m_flags[FLAGS_COUNT]; //every bit indicates occupancy in m_mempool.
	int m_used_flags;
	FUINT m_sizes[FLAGS_COUNT]; //zero at the MSB indicates the end of the allocated area.
	static char *s_mmapped_spaces[MMAP_SPACES_COUNT]; //swap space given by mmap(PROT_NONE).
	static uintptr_t s_allocators[ALLOC_MAX_ALLOCATORS];
	static int s_curr_allocator_idx;
	static int s_allocators_cnt;
	void* operator new(size_t size) throw();
	void operator delete(void* p);
};

#define ALLOC_ALIGN1 (ALLOC_ALIGNMENT)
#if defined __LP64__ || defined __LLP64__
	#define ALLOC_ALIGN2 (ALLOC_ALIGNMENT * 8)
	#define ALLOC_ALIGN(size) (((size) % ALLOC_ALIGN2 != 0) || ((size) == ALLOC_ALIGN1 * 64) ? ALLOC_ALIGN1 : ALLOC_ALIGN2)
#else
	#define ALLOC_ALIGN2 (ALLOC_ALIGNMENT * 4)
	#define ALLOC_ALIGN3 (ALLOC_ALIGNMENT * 16)
	#define ALLOC_ALIGN(size) (((size) % ALLOC_ALIGN2 != 0) || ((size) == ALLOC_ALIGN1 * 32) ? ALLOC_ALIGN1 : \
		(((size) % ALLOC_ALIGN3 != 0) || ((size) == ALLOC_ALIGN2 * 32) ? ALLOC_ALIGN2 : ALLOC_ALIGN3))
#endif

#define ALLOC_SIZE1 (ALLOC_ALIGNMENT * 1)
#define ALLOC_SIZE2 (ALLOC_ALIGNMENT * 2)
#define ALLOC_SIZE3 (ALLOC_ALIGNMENT * 3)
#define ALLOC_SIZE4 (ALLOC_ALIGNMENT * 4)
#define ALLOC_SIZE5 (ALLOC_ALIGNMENT * 5)
#define ALLOC_SIZE6 (ALLOC_ALIGNMENT * 6)
#define ALLOC_SIZE7 (ALLOC_ALIGNMENT * 7)
#define ALLOC_SIZE8 (ALLOC_ALIGNMENT * 8)
#define ALLOC_SIZE9 (ALLOC_ALIGNMENT * 9)
#define ALLOC_SIZE10 (ALLOC_ALIGNMENT * 10)
#define ALLOC_SIZE11 (ALLOC_ALIGNMENT * 11)
#define ALLOC_SIZE12 (ALLOC_ALIGNMENT * 12)
#define ALLOC_SIZE13 (ALLOC_ALIGNMENT * 13)
#define ALLOC_SIZE14 (ALLOC_ALIGNMENT * 14)
#define ALLOC_SIZE16 (ALLOC_ALIGNMENT * 16)

void* __allocate_large_size_or_malloc(size_t size) throw();

#define __ALLOCATE_9_16X(X, size) {\
	if(size <= ALLOC_SIZE16 * X) {\
		if(size <= ALLOC_SIZE9 * X)\
			return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE9 * X)>::allocate<ALLOC_SIZE9 * X>();\
		if(size <= ALLOC_SIZE10 * X)\
			return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE10 * X)>::allocate<ALLOC_SIZE10 * X>();\
		if(size <= ALLOC_SIZE11 * X)\
			return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE11 * X)>::allocate<ALLOC_SIZE11 * X>();\
		if(size <= ALLOC_SIZE12 * X)\
			return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE12 * X)>::allocate<ALLOC_SIZE12 * X>();\
		if(size <= ALLOC_SIZE13 * X)\
			return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE13 * X)>::allocate<ALLOC_SIZE13 * X>();\
		if(size <= ALLOC_SIZE14 * X)\
			return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE14 * X)>::allocate<ALLOC_SIZE14 * X>();\
		return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE16 * X)>::allocate<ALLOC_SIZE16 * X>();\
	}\
}

inline void* operator new(size_t size) throw() {
	//expecting a compile-time optimization because size is usually fixed to the object size.
	if(size <= ALLOC_SIZE1)
		return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE1)>::allocate<ALLOC_SIZE1>();
	if(size <= ALLOC_SIZE2)
		return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE2)>::allocate<ALLOC_SIZE2>();
	if(size <= ALLOC_SIZE3)
		return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE3)>::allocate<ALLOC_SIZE3>();
	if(size <= ALLOC_SIZE4)
		return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE4)>::allocate<ALLOC_SIZE4>();
	if(size <= ALLOC_SIZE8) {
		if(size <= ALLOC_SIZE5)
			return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE5)>::allocate<ALLOC_SIZE5>();
		if(size <= ALLOC_SIZE6)
			return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE6)>::allocate<ALLOC_SIZE6>();
		if(size <= ALLOC_SIZE7)
			return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE7)>::allocate<ALLOC_SIZE7>();
		return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE8)>::allocate<ALLOC_SIZE8>();
	}
	__ALLOCATE_9_16X(1, size);
	return __allocate_large_size_or_malloc(size);
}
inline void* operator new[](size_t size) throw() {
	return operator new(size);
}

void __deallocate_pooled_or_free(void* p) throw();

inline void operator delete(void* p) throw() {
	return __deallocate_pooled_or_free(p);
}

inline void operator delete[](void* p) throw() {
	operator delete(p);
}

void release_pools();

#endif /* ALLOCATOR_H_ */
