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

#include <string.h>
#include <boost/utility/enable_if.hpp>

#include <sys/types.h>
#include <sys/mman.h>

//! Bit count / population count for 32bit.
//! Referred to H. S. Warren, Jr., "Beautiful Code", O'Reilly.
template <typename T>
inline typename boost::enable_if_c<sizeof(T) == 4, unsigned int>::type count_bits(T x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0f0f0f0fu;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0xffu;
}

//! Bit count / population count for 64bit.
template <typename T>
inline typename boost::enable_if_c<sizeof(T) == 8, unsigned int>::type count_bits(T x) {
    x = x - ((x >> 1) & 0x5555555555555555uLL);
    x = (x & 0x3333333333333333uLL) + ((x >> 2) & 0x3333333333333333uLL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fuLL;
    x = x + (x >> 8);
    x = x + (x >> 16);
    x = x + (x >> 32);
    return x & 0xffu;
}

//! \return one bit at the first zero from the LSB in \a x.
template <typename T>
inline T find_zero_forward(T x) {
	return (( ~x) & (x + 1u));
}

//! \return one bit at the first one from the LSB in \a x.
template <typename T>
inline T find_one_forward(T x) {
	return x & ( ~x + 1u);
}

//! Folds "OR" operations. O(log X).
//! Expecting inline expansions of codes.
//! \tparam X number of zeros to be looked for.
template <unsigned int X, unsigned int SHIFTS, typename T>
inline T fold_bits(T x) {
//	printf("%d, %llx\n", SHIFTS, x);
//	if(x == ~(T)0u)
//		return x; //already filled.
	if(X <  2 * SHIFTS)
		return x;
	x = (x >> SHIFTS) | x;
	if(X & SHIFTS)
		x = (x >> SHIFTS) | x;
	return (2 * SHIFTS < sizeof(T) * 8) ?
		fold_bits<X, (2 * SHIFTS < sizeof(T) * 8) ? 2 * SHIFTS : 1>(x) : x;
}

//! Bit scan forward, counting zeros in the LSBs.
//! \param x should be 2^n.
//! \sa find_zero_forward(), find_first_oen().
template <typename T>
inline unsigned int count_zeros_forward(T x) {
#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__ || defined __x86_64__
	T ret;
	asm ("bsf %1,%0": "=q" (ret) : "r" (x) :);
	return ret;
#else
	return count_bits(x - 1);
#endif
}

//template <int X, typename T>
//inline T find_training_zeros_tedious(T x) {
//	T ret = ((T)1u << X) - 1u;
//	while(x & ret)
//		ret = ret << 1;
//	ret = find_one_forward(ret);
//	if(ret > (T)1u << (sizeof(T) * 8 - X)) return 0; //checking if T has enough space in MSBs.
//	return ret;
//}

//! Finds training zeros from LSB in \a x using O(log n) algorithm.
//! \tparam X number of zeros to be looked for.
//! \return one bit at the LSB of the training zeros if enough zeros are found.
template <int X, typename T>
inline T find_training_zeros(T x) {
//	if( !x) return 1u;
	if(X == sizeof(T) * 8)
		return !x ? 1u : 0u; //a trivial case.
	x = fold_bits<X, 1>(x);
	if(x == ~(T)0u)
		return 0; //already filled.
	x = find_zero_forward(x); //picking the first zero from LSB.
	if(x > (T)1u << (sizeof(T) * 8 - X)) return 0; //checking if T has enough space in MSBs.
	return x;
}


void *malloc_mmap(size_t size) {
//		fprintf(stderr, "mmap(), %d\n", (int)size);
		void *p = (
			mmap(0, size + ALLOC_ALIGNMENT, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0));
		ASSERT(p != MAP_FAILED);
		*static_cast<size_t *>(p) = size + ALLOC_ALIGNMENT;
		return static_cast<char *>(p) + ALLOC_ALIGNMENT;
}
void free_munmap(void *p) {
		p = static_cast<void *>(static_cast<char *>(p) - ALLOC_ALIGNMENT);
		size_t size = *static_cast<size_t *>(p);
	//	fprintf(stderr, "unmmap(), %d\n", (int)size);
		int ret = munmap(p, size);
		ASSERT( !ret);
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
PooledAllocator<ALIGN, FS, DUMMY>::PooledAllocator(char *addr)  : m_mempool(addr), m_idx(0) {
	memset(m_flags, 0, FLAGS_COUNT * sizeof(FUINT));
//	memset(m_sizes, 0, FLAGS_COUNT * sizeof(FUINT));
//	memoryBarrier();
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PooledAllocator<ALIGN, FS, DUMMY>::report_leaks() {
	for(int idx = 0; idx < FLAGS_COUNT; ++idx) {
		while(m_flags[idx]) {
			int sidx = count_zeros_forward(find_one_forward(m_flags[idx]));
			fprintf(stderr, "Leak found for %dB @ %llx.\n", (int)(ALIGN),
				(unsigned long long)(uintptr_t) &m_mempool[(idx * sizeof(FUINT) * 8 + sidx) * ALIGN]);
			m_flags[idx] &= ~(1u << sidx);
		}
	}
}
template <unsigned int ALIGN, bool DUMMY>
void
PooledAllocator<ALIGN, false, DUMMY>::report_leaks() {
	for(int idx = 0; idx < FLAGS_COUNT; ++idx) {
		while(this->m_flags[idx]) {
			int sidx = count_zeros_forward(find_one_forward(this->m_flags[idx]));
			int size = count_zeros_forward(find_zero_forward(m_sizes[idx] >> sidx)) + 1;
			fprintf(stderr, "Leak found for %dB @ %llx.\n", (int)(size * ALIGN),
				(unsigned long long)(uintptr_t) &this->m_mempool[(idx * sizeof(FUINT) * 8 + sidx) * ALIGN]);
			this->m_flags[idx] &= ~((2 *(((FUINT)1u << (size - 1)) - 1u) + 1u) << sidx);
		}
	}
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
template <unsigned int SIZE>
inline void *
PooledAllocator<ALIGN, FS, DUMMY>::allocate_pooled(int aidx) {
	FUINT one;
	FUINT *pflag = &this->m_flags[this->m_idx];
	FUINT *pflag_org = pflag;
	for(;;) {
		FUINT oldv = *pflag;
		if( ~oldv) {
			one = find_zero_forward(oldv);
//			ASSERT(count_bits(one) == SIZE / ALIGN);
//			ASSERT( !(one & oldv));
			if(oldv == 0) {
				*pflag = one;
				atomicInc( &this->s_flags_inc_cnt[aidx]);
				writeBarrier(); //for the counter.
				break;
			}
			else {
				FUINT newv = oldv | one; //set a flag.
				if(atomicCompareAndSet(oldv, newv, pflag))
					break;
			}
			continue;
		}
		pflag++;
		if(pflag == &this->m_flags[FLAGS_COUNT])
			pflag = this->m_flags;
		if(pflag == pflag_org)
			return 0;
	}

	int idx = pflag - this->m_flags;
	this->m_idx = idx;

	int sidx = count_zeros_forward(one);

	return &this->m_mempool[(idx * sizeof(FUINT) * 8 + sidx) * ALIGN];
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
inline bool
PooledAllocator<ALIGN, FS, DUMMY>::deallocate_pooled(void *p) {
	int midx = static_cast<size_t>((char *)p - this->m_mempool) / ALIGN;
	int idx = midx / (sizeof(FUINT) * 8);
	unsigned int sidx = midx % (sizeof(FUINT) * 8);

	FUINT none = ~((FUINT)1u << sidx);

	writeBarrier(); //for the pooled memory
	for(;;) {
		FUINT oldv = this->m_flags[idx];
		FUINT newv = oldv & none;
//		fprintf(stderr, "d: %llx, %d, %x, %x, %x\n", (unsigned long long)(uintptr_t)p, idx, oldv, newv, ones);
		if(atomicCompareAndSet(oldv, newv, &this->m_flags[idx])) {
			ASSERT(( oldv | none) == ~(FUINT)0); //checking for double free.
//			m_idx = idx; //writing a hint for a next allocation.
			if(newv == 0)
				return true;
			break;
		}
	}
	return false;
}


template <unsigned int ALIGN, bool DUMMY>
template <unsigned int SIZE>
inline void *
PooledAllocator<ALIGN, false, DUMMY>::allocate_pooled(int aidx) {
	FUINT ones, cand;
	FUINT *pflag = &this->m_flags[this->m_idx];
	FUINT *pflag_org = pflag;
	for(;;) {
		FUINT oldv = *pflag;
		cand = find_training_zeros<SIZE / ALIGN>(oldv);
		if(cand) {
			ones = cand *
				(2u * (((FUINT)1u << (SIZE / ALIGN - 1u)) - 1u) + 1u); //N ones, not to overflow.
//			ASSERT(count_bits(ones) == SIZE / ALIGN);
//			ASSERT( !(ones & oldv));
			if(oldv == 0) {
				*pflag = ones;
				atomicInc( &this->s_flags_inc_cnt[aidx]);
				writeBarrier(); //for the counter.
				break;
			}
			else {
				FUINT newv = oldv | ones; //filling with SIZE ones.
				if(atomicCompareAndSet(oldv, newv, pflag))
					break;
			}
			continue;
		}
		pflag++;
		if(pflag == &this->m_flags[FLAGS_COUNT])
			pflag = this->m_flags;
		if(pflag == pflag_org)
			return 0;
	}

	int idx = pflag - this->m_flags;
	this->m_idx = (SIZE / ALIGN <= sizeof(FUINT) * 8 / 2) ?  idx : ((idx + 1) % FLAGS_COUNT);

	FUINT sizes_old = m_sizes[idx];
	FUINT sizes_new = (sizes_old | ones) & ~(cand << (SIZE / ALIGN - 1u));
//					ASSERT((~sizes_new & ones) == cand << (SIZE / ALIGN - 1u));
	m_sizes[idx] = sizes_new;

	int sidx = count_zeros_forward(cand);

	return &this->m_mempool[(idx * sizeof(FUINT) * 8 + sidx) * ALIGN];
}
template <unsigned int ALIGN, bool DUMMY>
inline bool
PooledAllocator<ALIGN, false, DUMMY>::deallocate_pooled(void *p) {
	int midx = static_cast<size_t>((char *)p - this->m_mempool) / ALIGN;
	int idx = midx / (sizeof(FUINT) * 8);
	unsigned int sidx = midx % (sizeof(FUINT) * 8);

	FUINT nones = find_zero_forward(m_sizes[idx] >> sidx);
	nones = ~((nones | (nones - 1u)) << sidx);

	writeBarrier(); //for the pooled memory
	for(;;) {
		FUINT oldv = this->m_flags[idx];
		FUINT newv = oldv & nones;
//		fprintf(stderr, "d: %llx, %d, %x, %x, %x\n", (unsigned long long)(uintptr_t)p, idx, oldv, newv, ones);
		if(atomicCompareAndSet(oldv, newv, &this->m_flags[idx])) {
			ASSERT(( oldv | nones) == ~(FUINT)0); //checking for double free.
//			m_idx = idx; //writing a hint for a next allocation.
			if(newv == 0)
				return true;
			break;
		}
	}
	return false;
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
bool
PooledAllocator<ALIGN, FS, DUMMY>::trySetupNewAllocator(int aidx) {
	while( !s_mmapped_spaces[aidx / NUM_ALLOCATORS_IN_SPACE]) {
		int ps = getpagesize();
		ASSERT(ALLOC_MEMPOOL_SIZE % ps == 0);
		char *p = static_cast<char *>(
			mmap(0, MMAP_SPACE_SIZE, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
		ASSERT(p != MAP_FAILED);
		if(atomicCompareAndSet((char *)0, p, &s_mmapped_spaces[aidx / NUM_ALLOCATORS_IN_SPACE])) {
			fprintf(stderr, "Reserve swap space for %dB %s, starting @ %llx w/ len. of %llxB.\n", (int)ALIGN,
				DUMMY ? "fixed" : "aligned",
				(unsigned long long)p,
				(unsigned long long)(uintptr_t)MMAP_SPACE_SIZE);
			break;
		}
		munmap(p, MMAP_SPACE_SIZE);
	}
	if(atomicCompareAndSet((uintptr_t)0u, (uintptr_t)1u, &s_allocators[aidx])) {
		char *addr =
			s_mmapped_spaces[aidx / NUM_ALLOCATORS_IN_SPACE] +
			ALLOC_MEMPOOL_SIZE * (aidx % NUM_ALLOCATORS_IN_SPACE);
		int ret = mprotect(addr, ALLOC_MEMPOOL_SIZE, PROT_READ | PROT_WRITE);
		uintptr_t alloc = reinterpret_cast<uintptr_t>(new PooledAllocator<ALIGN, DUMMY, DUMMY>(addr));
		ASSERT( !ret);
		writeBarrier(); //for alloc.
		s_allocators[aidx] = alloc;
		int acnt = s_allocators_ubound;
		while(aidx >= acnt) {
			if(atomicCompareAndSet(acnt, aidx + 1, &s_allocators_ubound))
				break;
		}
//		writeBarrier();
//		fprintf(stderr, "New memory pool for %dB aligned, starting @ %llx w/ len. of %llxB.\n", (int)ALIGN,
//			(unsigned long long)(uintptr_t)alloc->m_mempool,
//			(unsigned long long)(uintptr_t)ALLOC_MEMPOOL_SIZE);
//		printf("n");
		return true;
	}
	return false;
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
template <unsigned int SIZE>
inline void *
PooledAllocator<ALIGN, FS, DUMMY>::allocate() {
	int aidx = s_curr_allocator_idx;
	for(int cnt = 0;; ++cnt) {
		uintptr_t alloc = s_allocators[aidx];
		if(alloc && !(alloc & 1u) && (atomicCompareAndSet(alloc, alloc | 1u, &s_allocators[aidx]))) {
			readBarrier();
			if(void *p = reinterpret_cast<PooledAllocator<ALIGN, DUMMY, DUMMY> *>(alloc)->allocate_pooled<SIZE>(aidx)) {
	//			fprintf(stderr, "a: %llx\n", (unsigned long long)(uintptr_t)p);
//				for(unsigned int i = 0; i < SIZE / sizeof(uint64_t); ++i)
//					static_cast<uint64_t *>(p)[i] = 0; //zero clear.

				s_allocators[aidx] = alloc;
				s_curr_allocator_idx = aidx;
				return p;
			}
			s_allocators[aidx] = alloc;
		}
		int acnt = s_allocators_ubound;
		++aidx;
		if((aidx >= acnt) || (aidx == ALLOC_MAX_ALLOCATORS))
			aidx = 0;
		if(cnt == acnt) {
			for(aidx = 0; aidx < ALLOC_MAX_ALLOCATORS - 1; ++aidx)
				if( !s_allocators[aidx])
					break;
			if( !trySetupNewAllocator(aidx))
				continue;
			cnt = 0;
		}
//		s_curr_allocator_idx = aidx;
	}
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PooledAllocator<ALIGN, FS, DUMMY>::releaseAllocator(uintptr_t alloc, int aidx) {
//							int aidx_just_become_vacant = aidx;
//							for(int aidx = s_allocators_ubound - 1; aidx >= 0; --aidx) {
//								if(s_flags_inc_cnt[aidx])
//									continue;
//								uintptr_t alloc = s_allocators[aidx];
//								if( !alloc || (alloc & 1u) || (aidx == aidx_just_become_vacant))
//									continue;
	if(atomicCompareAndSet(alloc, alloc | 1u, &s_allocators[aidx])) {
		readBarrier();
		PooledAllocator<ALIGN, DUMMY, DUMMY> *palloc =
			reinterpret_cast<PooledAllocator<ALIGN, DUMMY, DUMMY> *>(alloc);
		//checking if the pool is really vacant.
		if( !s_flags_inc_cnt[aidx]) {
			//releasing memory.
			mprotect(reinterpret_cast<PooledAllocator *>(alloc)->m_mempool, ALLOC_MEMPOOL_SIZE, PROT_NONE);
//						fprintf(stderr, "Freed: %llx\n", (unsigned long long)(uintptr_t)reinterpret_cast<PooledAllocator *>(alloc)->m_mempool);
//							printf("r");
			delete palloc;
			writeBarrier();

			s_allocators[aidx] = 0;
			//decreasing upper boundary.
			while(int acnt = s_allocators_ubound) {
				if(s_allocators[acnt - 1])
					break;
				atomicCompareAndSet(acnt, acnt - 1, &s_allocators_ubound);
			}
		}
		else {
			s_allocators[aidx] = alloc;
		}
	}
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
inline bool
PooledAllocator<ALIGN, FS, DUMMY>::deallocate(void *p) {
	for(int cnt = 0; cnt < MMAP_SPACES_COUNT; ++cnt) {
		char *mp = s_mmapped_spaces[cnt];
		if( !mp)
			break;
		if((p >= mp) && (p < &mp[MMAP_SPACE_SIZE])) {
			int aidx = (static_cast<char *>(p) - mp) / ALLOC_MEMPOOL_SIZE + cnt * NUM_ALLOCATORS_IN_SPACE;
			uintptr_t alloc = s_allocators[aidx];
			alloc &= ~(uintptr_t)1u;
//			ASSERT((p >= alloc->m_mempool) && (p < &alloc->m_mempool[ALLOC_MEMPOOL_SIZE]));
			PooledAllocator<ALIGN, DUMMY, DUMMY> *palloc = reinterpret_cast<PooledAllocator<ALIGN, DUMMY, DUMMY> *>(alloc);
			if(palloc->deallocate_pooled(p)) {
				if(atomicDecAndTest( &s_flags_inc_cnt[aidx]) && ~(s_allocators[aidx] & 1u)) {
					releaseAllocator(alloc, aidx);
				}
			}
//			s_curr_allocator_idx = aidx;
			return true;
		}
	}
	return false;
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PooledAllocator<ALIGN, FS, DUMMY>::release_pools() {
	int acnt = s_allocators_ubound;
	for(int aidx = 0; aidx < acnt; ++aidx) {
		uintptr_t alloc = s_allocators[aidx];
		alloc &= ~(uintptr_t)1u;
		reinterpret_cast<PooledAllocator<ALIGN, DUMMY, DUMMY> *>(alloc)->report_leaks();
		delete reinterpret_cast<PooledAllocator<ALIGN, DUMMY, DUMMY> *>(alloc);
		s_allocators[aidx] = 0;
	}
	s_allocators_ubound = 0;
	for(int cnt = 0; cnt < MMAP_SPACES_COUNT; ++cnt) {
		char *mp = s_mmapped_spaces[cnt];
		if( !mp)
			break;
		int ret = munmap(mp, MMAP_SPACE_SIZE);
		s_mmapped_spaces[cnt] = 0;
//		int ret = mprotect(mp, MMAP_SPACE_SIZE, PROT_NONE);
		ASSERT( !ret);
	}
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
void*
PooledAllocator<ALIGN, FS, DUMMY>::operator new(size_t size) throw() {
	return malloc(size);
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PooledAllocator<ALIGN, FS, DUMMY>::operator delete(void* p) {
	free(p);
}

void* __allocate_large_size_or_malloc(size_t size) throw() {
	__ALLOCATE_9_16X(4, size);
	__ALLOCATE_9_16X(8, size);
	__ALLOCATE_9_16X(16, size);
	__ALLOCATE_9_16X(32, size); //to 4KB.

	return malloc(size);
}

void __deallocate_pooled_or_free(void* p) throw() {
	if(PooledAllocator<ALLOC_SIZE1, true>::deallocate(p))
		return;
	if(PooledAllocator<ALLOC_SIZE2, true>::deallocate(p))
		return;
	if(PooledAllocator<ALLOC_ALIGN1>::deallocate(p))
		return;
	if(PooledAllocator<ALLOC_SIZE3, true>::deallocate(p))
		return;
	if(PooledAllocator<ALLOC_SIZE4, true>::deallocate(p))
		return;
	if(PooledAllocator<ALLOC_SIZE5, true>::deallocate(p))
		return;
	if(PooledAllocator<ALLOC_SIZE6, true>::deallocate(p))
		return;
	if(PooledAllocator<ALLOC_SIZE7, true>::deallocate(p))
		return;
	if(PooledAllocator<ALLOC_ALIGN2>::deallocate(p))
		return;
#if defined ALLOC_ALIGN3
	if(PooledAllocator<ALLOC_ALIGN3>::deallocate(p))
		return;
#endif
	free(p);
}

void release_pools() {
	PooledAllocator<ALLOC_SIZE1, true>::release_pools();
	PooledAllocator<ALLOC_SIZE2, true>::release_pools();
	PooledAllocator<ALLOC_SIZE3, true>::release_pools();
	PooledAllocator<ALLOC_SIZE4, true>::release_pools();
	PooledAllocator<ALLOC_SIZE5, true>::release_pools();
	PooledAllocator<ALLOC_SIZE6, true>::release_pools();
	PooledAllocator<ALLOC_SIZE7, true>::release_pools();
	PooledAllocator<ALLOC_ALIGN1>::release_pools();
	PooledAllocator<ALLOC_ALIGN2>::release_pools();
#if defined ALLOC_ALIGN3
	PooledAllocator<ALLOC_ALIGN3>::release_pools();
#endif
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
uintptr_t PooledAllocator<ALIGN, FS, DUMMY>::s_allocators[ALLOC_MAX_ALLOCATORS];
template <unsigned int ALIGN, bool FS, bool DUMMY>
int PooledAllocator<ALIGN, FS, DUMMY>::s_curr_allocator_idx;
template <unsigned int ALIGN, bool FS, bool DUMMY>
int PooledAllocator<ALIGN, FS, DUMMY>::s_flags_inc_cnt[ALLOC_MAX_ALLOCATORS];
template <unsigned int ALIGN, bool FS, bool DUMMY>
int PooledAllocator<ALIGN, FS, DUMMY>::s_allocators_ubound;
template <unsigned int ALIGN, bool FS, bool DUMMY>
char *PooledAllocator<ALIGN, FS, DUMMY>::s_mmapped_spaces[MMAP_SPACES_COUNT];

template class PooledAllocator<ALLOC_ALIGN1>;
template class PooledAllocator<ALLOC_ALIGN2>;
#if defined ALLOC_ALIGN3
	template class PooledAllocator<ALLOC_ALIGN3>;
#endif

template class PooledAllocator<ALLOC_SIZE1, true>;
template class PooledAllocator<ALLOC_SIZE2, true>;
template class PooledAllocator<ALLOC_SIZE3, true>;
template class PooledAllocator<ALLOC_SIZE4, true>;
template class PooledAllocator<ALLOC_SIZE5, true>;
template class PooledAllocator<ALLOC_SIZE6, true>;
template class PooledAllocator<ALLOC_SIZE7, true>;

template void *PooledAllocator<ALLOC_SIZE1, true>::allocate<ALLOC_SIZE1>();
template void *PooledAllocator<ALLOC_SIZE2, true>::allocate<ALLOC_SIZE2>();
template void *PooledAllocator<ALLOC_SIZE3, true>::allocate<ALLOC_SIZE3>();
template void *PooledAllocator<ALLOC_SIZE4, true>::allocate<ALLOC_SIZE4>();
template void *PooledAllocator<ALLOC_SIZE5, true>::allocate<ALLOC_SIZE5>();
template void *PooledAllocator<ALLOC_SIZE6, true>::allocate<ALLOC_SIZE6>();
template void *PooledAllocator<ALLOC_SIZE7, true>::allocate<ALLOC_SIZE7>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE8)>::allocate<ALLOC_SIZE8>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE9)>::allocate<ALLOC_SIZE9>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE10)>::allocate<ALLOC_SIZE10>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE11)>::allocate<ALLOC_SIZE11>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE12)>::allocate<ALLOC_SIZE12>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE13)>::allocate<ALLOC_SIZE13>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE14)>::allocate<ALLOC_SIZE14>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE16)>::allocate<ALLOC_SIZE16>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE9 * 2)>::allocate<ALLOC_SIZE9 * 2>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE10 * 2)>::allocate<ALLOC_SIZE10 * 2>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE11 * 2)>::allocate<ALLOC_SIZE11 * 2>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE12 * 2)>::allocate<ALLOC_SIZE12 * 2>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE13 * 2)>::allocate<ALLOC_SIZE13 * 2>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE14 * 2)>::allocate<ALLOC_SIZE14 * 2>();
template void *PooledAllocator<ALLOC_ALIGN(ALLOC_SIZE16 * 2)>::allocate<ALLOC_SIZE16 * 2>();

//struct _PoolReleaser {
//	~_PoolReleaser() {
//		release_pools();
//	}
//} _pool_releaser;
