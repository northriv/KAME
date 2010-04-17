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
//! Memory pools won't be released once being secured.

#include <new>
#include <stdint.h>
#include <stdlib.h>

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
	inline void *allocate_pooled();
	inline bool deallocate_pooled(void *p);
	static bool trySetupNewAllocator(int aidx);
	static void *allocate(size_t size) ;
	static inline bool deallocate(void *p);
private:
	char *m_mempool;
	int m_idx;
	FUINT m_flags[MEMPOOL_COUNT];
	static FixedSizeAllocator *s_allocators[ALLOC_MAX_ALLOCATORS];
	static int s_curr_allocator_idx;
	static int s_allocators_cnt;
	void* operator new(size_t size) throw();
	void operator delete(void* p);
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
void operator delete(void* p) throw();

#endif /* ALLOCATOR_H_ */
