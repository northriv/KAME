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

#ifndef ALLOCATOR_PRV_H_
#define ALLOCATOR_PRV_H_

#include <new>
#include <stdint.h>
#include <stdlib.h>
#include <limits>

#define ALLOC_MEMPOOL_SIZE (1024 * 256) //256KiB
#define ALLOC_MAX_ALLOCATORS (1024 * 4) //2GiB max.
#define ALLOC_ALIGNMENT (sizeof(double)) //i.e. 8B

//! \brief Memory blocks in a unit of double-quad word
//! can be allocated from fixed-size or variable-size memory pools.
//! \tparam FS determines fixed-size or variable-size.
//! \sa allocator_test.cpp.
template <unsigned int ALIGN, bool FS = false, bool DUMMY = true>
class PooledAllocator {
public:
	template <unsigned int SIZE>
	static void *allocate();
	static inline bool deallocate(void *p);
	static void release_pools();
	void report_leaks();

	typedef uintptr_t FUINT;
protected:
	PooledAllocator(char *addr);
	template <unsigned int SIZE>
	inline void *allocate_pooled(int aidx);
	inline bool deallocate_pooled(void *p);
	static bool trySetupNewAllocator(int aidx);
	static void releaseAllocator(uintptr_t alloc, int aidx);
	enum {FLAGS_COUNT = ALLOC_MEMPOOL_SIZE / ALIGN / sizeof(FUINT) / 8};
	enum {MMAP_SPACE_SIZE = 1024 * 1024 * 16, //16MiB
		NUM_ALLOCATORS_IN_SPACE = MMAP_SPACE_SIZE / ALLOC_MEMPOOL_SIZE,
		MMAP_SPACES_COUNT = ALLOC_MAX_ALLOCATORS / NUM_ALLOCATORS_IN_SPACE};
	char *m_mempool;
	int m_idx; //a hint for searching in a sparse area.
	FUINT m_flags[FLAGS_COUNT]; //every bit indicates occupancy in m_mempool.
	static char *s_mmapped_spaces[MMAP_SPACES_COUNT]; //swap space given by mmap(PROT_NONE).
	static uintptr_t s_allocators[ALLOC_MAX_ALLOCATORS];
	static int s_flags_inc_cnt[ALLOC_MAX_ALLOCATORS];
	static int s_curr_allocator_idx;
	static int s_allocators_ubound;
	void* operator new(size_t size) throw();
	void operator delete(void* p);
};

template <unsigned int ALIGN, bool DUMMY>
class PooledAllocator<ALIGN, false, DUMMY> : public PooledAllocator<ALIGN, true, false> {
public:
	void report_leaks();
	typedef typename PooledAllocator<ALIGN, true, false>::FUINT FUINT;
protected:
	enum {FLAGS_COUNT = PooledAllocator<ALIGN, true, false>::FLAGS_COUNT};
	PooledAllocator(char *addr) : PooledAllocator<ALIGN, true, false>(addr) {}
	template <unsigned int SIZE>
	inline void *allocate_pooled(int aidx);
	inline bool deallocate_pooled(void *p);
private:
	template <unsigned int, bool, bool> friend class PooledAllocator;
	FUINT m_sizes[FLAGS_COUNT]; //zero at the MSB indicates the end of the allocated area.
};

#define ALLOC_ALIGN1 (ALLOC_ALIGNMENT * 2)
#if defined __LP64__ || defined __LLP64__
	#define ALLOC_ALIGN2 (ALLOC_ALIGNMENT * 16)
	#define ALLOC_ALIGN(size) (((size) % ALLOC_ALIGN2 != 0) || ((size) == ALLOC_ALIGN1 * 64) ? ALLOC_ALIGN1 : ALLOC_ALIGN2)
//	#define ALLOC_ALIGN(size) (((size) <= ALLOC_ALIGN1 * 64) ? ALLOC_ALIGN1 : ALLOC_ALIGN2)
#else
	#define ALLOC_ALIGN2 (ALLOC_ALIGNMENT * 8)
	#define ALLOC_ALIGN3 (ALLOC_ALIGNMENT * 32)
	#define ALLOC_ALIGN(size) (((size) % ALLOC_ALIGN2 != 0) || ((size) == ALLOC_ALIGN1 * 32) ? ALLOC_ALIGN1 :\
		(((size) % ALLOC_ALIGN3 != 0) || ((size) == ALLOC_ALIGN2 * 32) ? ALLOC_ALIGN2 : ALLOC_ALIGN3))
//	#define ALLOC_ALIGN(size) (((size) <= ALLOC_ALIGN1 * 32) ? ALLOC_ALIGN1 :
//		(((size) <= ALLOC_ALIGN2 * 32) ? ALLOC_ALIGN2 : ALLOC_ALIGN3))
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

void* allocate_large_size_or_malloc(size_t size) throw();

#define ALLOCATE_9_16X(X, size) {\
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

inline void* new_redirected(std::size_t size) throw(std::bad_alloc) {
	//expecting a compile-time optimization because size is usually fixed to the object size.
	if(size <= ALLOC_SIZE1)
		return PooledAllocator<ALLOC_SIZE1, true>::allocate<ALLOC_SIZE1>();
	if(size <= ALLOC_SIZE2)
		return PooledAllocator<ALLOC_SIZE2, true>::allocate<ALLOC_SIZE2>();
	if(size <= ALLOC_SIZE3)
		return PooledAllocator<ALLOC_SIZE3, true>::allocate<ALLOC_SIZE3>();
	if(size <= ALLOC_SIZE4)
		return PooledAllocator<ALLOC_SIZE4, true>::allocate<ALLOC_SIZE4>();
	if(size <= ALLOC_SIZE8) {
		if(size <= ALLOC_SIZE5)
			return PooledAllocator<ALLOC_SIZE5, true>::allocate<ALLOC_SIZE5>();
		if(size <= ALLOC_SIZE6)
			return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE6)>::allocate<ALLOC_SIZE6>();
		if(size <= ALLOC_SIZE7)
			return PooledAllocator<ALLOC_SIZE7, true>::allocate<ALLOC_SIZE7>();
		return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE8)>::allocate<ALLOC_SIZE8>();
	}
	if(size <= ALLOC_SIZE16) {
		if(size <= ALLOC_SIZE9)
			return PooledAllocator<ALLOC_SIZE9, true>::allocate<ALLOC_SIZE9>();
		if(size <= ALLOC_SIZE10)
			return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE10)>::allocate<ALLOC_SIZE10>();
		if(size <= ALLOC_SIZE12)
			return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE12)>::allocate<ALLOC_SIZE12>();
		if(size <= ALLOC_SIZE14)
			return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE14)>::allocate<ALLOC_SIZE14>();
		return PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE16)>::allocate<ALLOC_SIZE16>();
	}
	ALLOCATE_9_16X(2, size);
	return allocate_large_size_or_malloc(size);
}

//void* operator new(std::size_t size) throw(std::bad_alloc);
//void* operator new(std::size_t size, const std::nothrow_t&) throw();
//void* operator new[](std::size_t size) throw(std::bad_alloc);
//void* operator new[](std::size_t size, const std::nothrow_t&) throw();
//
//void operator delete(void* p) throw();
//void operator delete(void* p, const std::nothrow_t&) throw();
//void operator delete[](void* p) throw();
//void operator delete[](void* p, const std::nothrow_t&) throw();

void deallocate_pooled_or_free(void* p) throw();

void release_pools();

#endif /* ALLOCATOR_PRV_H_ */
