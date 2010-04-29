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

#define ALLOC_MEMPOOL_SIZE (1024 * 512)
#define ALLOC_MAX_ALLOCATORS (1024 * 1024 / ALLOC_MEMPOOL_SIZE  * 1024 * 4)
#define ALLOC_ALIGNMENT (sizeof(double)) //i.e. 8

//! \brief Fast lock-free allocators for small objects: new(), new[](), delete(), delete[]() operators.\n
//! Arbitrary sizes of memory in a unit of double-quad word less than 256B can be allocated from a memory pool.\n
//! Those memory pools won't be released once being secured in order to reduce efforts for locking of pools.
//! \sa allocator_test.cpp.
class PooledAllocator {
	typedef uint32_t FUINT;
	enum {FLAGS_COUNT = ALLOC_MEMPOOL_SIZE / ALLOC_ALIGNMENT / sizeof(FUINT) / 8};
public:
	PooledAllocator();
	~PooledAllocator();
	template <unsigned int SIZE>
	inline void *allocate_pooled();
	inline bool deallocate_pooled(void *p);
	static bool trySetupNewAllocator(int aidx);
	template <unsigned int SIZE>
	static void *allocate() ;
	static inline bool deallocate(void *p);
	static void release_pools();
private:
	char m_mempool[ALLOC_MEMPOOL_SIZE];
	int m_idx; //a hint for searching in a sparse area.
	FUINT m_flags[FLAGS_COUNT]; //every bit indicates occupancy in m_mempool.
	FUINT m_sizes[FLAGS_COUNT]; //zero at the MSB indicates the end of the allocated area.
	static PooledAllocator *s_allocators[ALLOC_MAX_ALLOCATORS];
	static int s_curr_allocator_idx;
	static int s_allocators_cnt;
	void* operator new(size_t size) throw();
	void operator delete(void* p);
};

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
#define ALLOC_SIZE15 (ALLOC_ALIGNMENT * 16)
#define ALLOC_SIZE16 (ALLOC_ALIGNMENT * 18)
#define ALLOC_SIZE17 (ALLOC_ALIGNMENT * 20)
#define ALLOC_SIZE18 (ALLOC_ALIGNMENT * 22)
#define ALLOC_SIZE19 (ALLOC_ALIGNMENT * 24)
#define ALLOC_SIZE20 (ALLOC_ALIGNMENT * 28)
#define ALLOC_SIZE21 (ALLOC_ALIGNMENT * 32)

inline void* operator new(size_t size) throw() {
	//expecting a compile-time optimization because size is usually fixed to the object size.
	if(size <= ALLOC_SIZE1)
		return PooledAllocator::allocate<ALLOC_SIZE1>();
	if(size <= ALLOC_SIZE2)
		return PooledAllocator::allocate<ALLOC_SIZE2>();
	if(size <= ALLOC_SIZE3)
		return PooledAllocator::allocate<ALLOC_SIZE3>();
	if(size <= ALLOC_SIZE4)
		return PooledAllocator::allocate<ALLOC_SIZE4>();
	if(size <= ALLOC_SIZE5)
		return PooledAllocator::allocate<ALLOC_SIZE5>();
	if(size <= ALLOC_SIZE6)
		return PooledAllocator::allocate<ALLOC_SIZE6>();
	if(size <= ALLOC_SIZE7)
		return PooledAllocator::allocate<ALLOC_SIZE7>();
	if(size <= ALLOC_SIZE8)
		return PooledAllocator::allocate<ALLOC_SIZE8>();
	if(size <= ALLOC_SIZE9)
		return PooledAllocator::allocate<ALLOC_SIZE9>();
	if(size <= ALLOC_SIZE10)
		return PooledAllocator::allocate<ALLOC_SIZE10>();
	if(size <= ALLOC_SIZE11)
		return PooledAllocator::allocate<ALLOC_SIZE11>();
	if(size <= ALLOC_SIZE12)
		return PooledAllocator::allocate<ALLOC_SIZE12>();
	if(size <= ALLOC_SIZE13)
		return PooledAllocator::allocate<ALLOC_SIZE13>();
	if(size <= ALLOC_SIZE14)
		return PooledAllocator::allocate<ALLOC_SIZE14>();
	if(size <= ALLOC_SIZE15)
		return PooledAllocator::allocate<ALLOC_SIZE15>();
	if(size <= ALLOC_SIZE16)
		return PooledAllocator::allocate<ALLOC_SIZE16>();
	if(size <= ALLOC_SIZE17)
		return PooledAllocator::allocate<ALLOC_SIZE17>();
	if(size <= ALLOC_SIZE18)
		return PooledAllocator::allocate<ALLOC_SIZE18>();
	if(size <= ALLOC_SIZE19)
		return PooledAllocator::allocate<ALLOC_SIZE19>();
	if(size <= ALLOC_SIZE20)
		return PooledAllocator::allocate<ALLOC_SIZE20>();
	if(size <= ALLOC_SIZE21)
		return PooledAllocator::allocate<ALLOC_SIZE21>();
	return malloc(size);
}
inline void* operator new[](size_t size) throw() {
	return operator new(size);
}

void operator delete(void* p) throw();

inline void operator delete[](void* p) throw() {
	operator delete(p);
}

void release_pools();

#endif /* ALLOCATOR_H_ */
