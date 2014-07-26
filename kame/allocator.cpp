/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/

//#define GUARDIAN 0xaaaaaaaauLL
//#define FILLING_AFTER_ALLOC 0x55555555uLL
#define LEAVE_VACANT_CHUNKS 1 //keep at least this # of chunks.

#include "allocator_prv.h"
#include "support.h"

#include "atomic.h"

#include <string.h>
#include <type_traits>

#include <sys/types.h>

//! Bit count / population count for 32bit.
//! Referred to H. S. Warren, Jr., "Beautiful Code", O'Reilly.
template <typename T>
inline typename std::enable_if<sizeof(T) == 4, unsigned int>::type count_bits(T x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0f0f0f0fu;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0xffu;
}

//! Bit count / population count for 64bit.
template <typename T>
inline typename std::enable_if<sizeof(T) == 8, unsigned int>::type count_bits(T x) {
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
template<typename T>
inline T fold_bits(unsigned int X, unsigned int SHIFTS, T x) {
//	printf("%d, %llx\n", SHIFTS, x);
//	if(x == ~(T)0u)
//		return x; //already filled.
	if(X <  2 * SHIFTS)
		return x;
	x = (x >> SHIFTS) | x;
	if(X & SHIFTS)
		x = (x >> SHIFTS) | x;
	return (2 * SHIFTS < sizeof(T) * 8) ?
		fold_bits(X, (2 * SHIFTS < sizeof(T) * 8) ? 2 * SHIFTS : 1, x) : x;
};

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
//! \arg X number of zeros to be looked for.
//! \return one bit at the LSB of the training zeros if enough zeros are found.
template<typename T>
inline T find_training_zeros (int X, T x) {
//	if( !x) return 1u;
	if(X == sizeof(T) * 8)
		return !x ? 1u : 0u; //a trivial case.
	x = fold_bits(X, 1, x);
	if(x == ~(T)0u)
		return 0; //already filled.
	x = find_zero_forward(x); //picking the first zero from LSB.
	if(x > (T)1u << (sizeof(T) * 8 - X)) return 0; //checking if T has enough space in MSBs.
	return x;
};

inline void *malloc_mmap(size_t size) {
//		fprintf(stderr, "mmap(), %d\n", (int)size);
		void *p = (
			mmap(0, size + ALLOC_ALIGNMENT, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0));
		assert(p != MAP_FAILED);
		*static_cast<size_t *>(p) = size + ALLOC_ALIGNMENT;
		return static_cast<char *>(p) + ALLOC_ALIGNMENT;
}
inline void free_munmap(void *p) {
		p = static_cast<void *>(static_cast<char *>(p) - ALLOC_ALIGNMENT);
		size_t size = *static_cast<size_t *>(p);
	//	fprintf(stderr, "unmmap(), %d\n", (int)size);
		int ret = munmap(p, size);
		assert( !ret);
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
inline PoolAllocator<ALIGN, FS, DUMMY>::PoolAllocator(int count, char *addr, char *ppool) :
	PoolAllocatorBase(ppool),
	m_flags(reinterpret_cast<FUINT *>( &addr[(sizeof(PoolAllocator) + sizeof(FUINT) - 1) * sizeof(FUINT)])),
	m_idx(0),
	m_count(count) {
	m_flags_nonzero_cnt = 0;
	m_flags_filled_cnt = 0;
	for(int i = count - 1; i >= 0 ; --i)
		m_flags[i] = 0; //zero clear.
#ifdef GUARDIAN
	for(unsigned int i = 0; i < count * sizeof(FUINT) * 8 * ALIGN / sizeof(uint64_t); ++i)
		reinterpret_cast<uint64_t *>(ppool)[i] = GUARDIAN; //filling
#endif
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
inline PoolAllocator<ALIGN, FS, DUMMY> *PoolAllocator<ALIGN, FS, DUMMY>::create(size_t size, char *ppool) {
	size_t size_alloc = (sizeof(PoolAllocator) + sizeof(FUINT) - 1) * sizeof(FUINT);
	int count = size / ALIGN / sizeof(FUINT) / 8;
	char *area = static_cast<char *>(malloc(size_alloc + sizeof(FUINT) * count));
	if( !area)
		return 0;
	PoolAllocator *p = new(area) PoolAllocator(count, area, ppool);
	return p;
}
template <unsigned int ALIGN, bool DUMMY>
inline PoolAllocator<ALIGN, false, DUMMY>::PoolAllocator(int count, char *addr, char *ppool) :
	PoolAllocator<ALIGN, true, false>(count, addr, ppool),
	m_sizes(reinterpret_cast<FUINT *>( &addr[(sizeof(PoolAllocator) + sizeof(FUINT) - 1) * sizeof(FUINT)
	                                         + sizeof(FUINT) * count])) {
	m_available_bits = sizeof(FUINT) * 8;
}
template <unsigned int ALIGN, bool DUMMY>
inline PoolAllocator<ALIGN, false, DUMMY> *PoolAllocator<ALIGN, false, DUMMY>::create(size_t size, char *ppool) {
	int count = size / ALIGN / sizeof(FUINT) / 8;
	size_t size_alloc = (sizeof(PoolAllocator) + sizeof(FUINT) - 1) * sizeof(FUINT);
	char *area = static_cast<char *>(malloc(size_alloc + sizeof(FUINT) * count * 2));
	if( !area)
		return 0;
	PoolAllocator *p = new(area) PoolAllocator(count, area, ppool);
	return p;
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
inline void PoolAllocator<ALIGN, FS, DUMMY>::operator delete(void *p) throw() {
	free(p);
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PoolAllocator<ALIGN, FS, DUMMY>::report_leaks() {
	for(int idx = 0; idx < m_count; ++idx) {
		while(m_flags[idx]) {
			int sidx = count_zeros_forward(find_one_forward(m_flags[idx]));
			fprintf(stderr, "Leak found for %dB @ %p.\n", (int)(ALIGN),
				&m_mempool[(idx * sizeof(FUINT) * 8 + sidx) * ALIGN]);
			m_flags[idx] &= ~(1u << sidx);
		}
	}
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PoolAllocator<ALIGN, FS, DUMMY>::report_statistics(size_t &chunk_size, size_t &used_size) {
	chunk_size = m_count * ALIGN * sizeof(FUINT) * 8;
	printf("Chunk @%p, size of 0x%llx, ", m_mempool, (unsigned long long)chunk_size);
	used_size = 0;
	for(int idx = 0; idx < m_count; ++idx) {
		used_size += count_bits(m_flags[idx]);
	}
	used_size *= ALIGN;
	printf("for fixed %dB, nonzero=%.1f%%, filled=%.1f%%, filling = %.2f%%\n", ALIGN,
		(double)m_flags_nonzero_cnt / m_count * 100.0,
		(double)m_flags_filled_cnt / m_count * 100.0,
		(double)used_size / chunk_size * 100.0);
}
template <unsigned int ALIGN, bool DUMMY>
void
PoolAllocator<ALIGN, false, DUMMY>::report_statistics(size_t &chunk_size, size_t &used_size) {
	chunk_size = this->m_count * ALIGN * sizeof(FUINT) * 8;
	printf("Chunk @%p, size of 0x%llx, ", this->m_mempool, (unsigned long long)chunk_size);
	used_size = 0;
	for(int idx = 0; idx < this->m_count; ++idx) {
		used_size += count_bits(this->m_flags[idx]);
	}
	used_size *= ALIGN;
	printf("for variable %dB, nonzero=%.2f%%, available=%d, filling = %.2f%%\n", ALIGN,
		(double)this->m_flags_nonzero_cnt / this->m_count * 100,
		m_available_bits,
		(double)used_size / chunk_size * 100.0);
}
template <unsigned int ALIGN, bool DUMMY>
void
PoolAllocator<ALIGN, false, DUMMY>::report_leaks() {
	for(int idx = 0; idx < this->m_count; ++idx) {
		while(this->m_flags[idx]) {
			int sidx = count_zeros_forward(find_one_forward(this->m_flags[idx]));
			int size = count_zeros_forward(find_zero_forward(m_sizes[idx] >> sidx)) + 1;
			fprintf(stderr, "Leak found for %dB @ %p.\n", (int)(size * ALIGN),
				&this->m_mempool[(idx * sizeof(FUINT) * 8 + sidx) * ALIGN]);
			this->m_flags[idx] &= ~((2 *(((FUINT)1u << (size - 1)) - 1u) + 1u) << sidx);
		}
	}
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
inline void *
PoolAllocator<ALIGN, FS, DUMMY>::allocate_pooled(unsigned int SIZE) {
	FUINT one;
	int idx = this->m_idx;
	for(;;) {
		FUINT *pflag = &this->m_flags[idx];
		FUINT oldv = *pflag;
		if(oldv != ~(FUINT)0u) {
			one = find_zero_forward(oldv);
//			assert(count_bits(one) == SIZE / ALIGN);
//			assert( !(one & oldv));
			if(oldv == 0) {
				*pflag = one;
				atomicInc( &this->m_flags_nonzero_cnt);
				writeBarrier(); //for the counter.
				break;
			}
			else {
				FUINT newv = oldv | one; //set a flag.
				if(atomicCompareAndSet(oldv, newv, pflag)) {
					if(newv == ~(FUINT)0u) {
						atomicInc( &this->m_flags_filled_cnt);
						writeBarrier(); //for the counter.
					}
					break;
				}
			}
			continue;
		}
		if(this->m_flags_filled_cnt == this->m_count)
			return 0;
		idx++;
		if(idx == this->m_count) {
			idx = 0;
		}
	}

	int sidx = count_zeros_forward(one);

	this->m_idx = idx;

	void *p = &this->m_mempool[(idx * sizeof(FUINT) * 8 + sidx) * ALIGN];
	return p;
}

template <unsigned int ALIGN, bool DUMMY>
inline void *
PoolAllocator<ALIGN, false, DUMMY>::allocate_pooled(unsigned int SIZE) {
	if(m_available_bits < SIZE / ALIGN)
		return 0;
	FUINT oldv, ones, cand;
	int idx = this->m_idx;
	FUINT *pflag = &this->m_flags[idx];
	for(FUINT *pend = &this->m_flags[this->m_count];;) {
		oldv = *pflag;
		cand = find_training_zeros(SIZE / ALIGN, oldv);
		if(cand) {
			ones = cand *
				(2u * (((FUINT)1u << (SIZE / ALIGN - 1u)) - 1u) + 1u); //N ones, not to overflow.
//			assert(count_bits(ones) == SIZE / ALIGN);
//			assert( !(ones & oldv));
			if(oldv == 0) {
				*pflag = ones;
				atomicInc( &this->m_flags_nonzero_cnt);
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
		if(pflag == pend) {
			if((pend != &this->m_flags[this->m_count]) || (idx == 0)) {
				if(this->m_flags_nonzero_cnt == this->m_count) {
					readBarrier();
					if(this->m_flags_nonzero_cnt == this->m_count) {
						m_available_bits = SIZE / ALIGN - 1u;
						writeBarrier();
						return 0;
					}
				}
				pflag = &this->m_flags[idx];
				pend = &this->m_flags[this->m_count];
			}
			else {
				pflag = this->m_flags;
				pend = &this->m_flags[idx];
			}
		}
	}

	idx = pflag - this->m_flags;

	FUINT sizes_old = m_sizes[idx];
	FUINT sizes_new = (sizes_old | ones) & ~(cand << (SIZE / ALIGN - 1u));
//					assert((~sizes_new & ones) == cand << (SIZE / ALIGN - 1u));
	if((sizes_old != sizes_new) || (oldv == 0)) {
		m_sizes[idx] = sizes_new;
		writeBarrier(); //for the counter and m_sizes.
	}
	int sidx = count_zeros_forward(cand);

	this->m_idx = idx;

	void *p = &this->m_mempool[(idx * sizeof(FUINT) * 8 + sidx) * ALIGN];
	return p;
}
template <unsigned int ALIGN, bool DUMMY>
bool
PoolAllocator<ALIGN, false, DUMMY>::deallocate_pooled(char *p) {
	writeBarrier(); //for the contents written by the user.

	int midx = (p - this->m_mempool) / ALIGN;
	int idx = midx / (sizeof(FUINT) * 8);
	unsigned int sidx = midx % (sizeof(FUINT) * 8);

	FUINT nones = find_zero_forward(m_sizes[idx] >> sidx);
	nones = ~((nones | (nones - 1u)) << sidx);

#ifdef GUARDIAN
	int size = count_bits( ~nones);
	for(unsigned int i = 0; i < size * ALIGN / sizeof(uint64_t); ++i)
		static_cast<uint64_t *>(p)[i] = GUARDIAN; //filling
#endif
	FUINT *pflags = &this->m_flags[idx];
	for(;;) {
		FUINT oldv = *pflags;
		FUINT newv = oldv & nones;
//		fprintf(stderr, "d: %p, %d, %x, %x, %x\n", p, idx, oldv, newv, ones);
		if(atomicCompareAndSet(oldv, newv, pflags)) {
			assert(( oldv | nones) == ~(FUINT)0); //checking for double free.
			if(newv == 0) {
				m_available_bits = sizeof(FUINT) * 8;
				if(atomicDecAndTest( &this->m_flags_nonzero_cnt)) {
					if(this->release_allocator(this)) {
						delete this; //suicide.
						return true;
					}
					else
						return false;
				}
			}
			else if(m_available_bits != sizeof(FUINT) * 8) {
				unsigned int size = count_bits( ~newv);
				for(;;) {
					unsigned int bits = m_available_bits;
					if(bits >= size)
						break;
					if(atomicCompareAndSet(bits, (unsigned int)size, &m_available_bits)) {
						break;
					}
				}
			}
			break;
		}
	}
	return false;
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
bool
PoolAllocator<ALIGN, FS, DUMMY>::deallocate_pooled(char *p) {
	writeBarrier(); //for the contents written by the user.

	int midx = (p - this->m_mempool) / ALIGN;
	int idx = midx / (sizeof(FUINT) * 8);
	unsigned int sidx = midx % (sizeof(FUINT) * 8);

	FUINT none = ~((FUINT)1u << sidx);

#ifdef GUARDIAN
	for(unsigned int i = 0; i < ALIGN / sizeof(uint64_t); ++i)
		static_cast<uint64_t *>(p)[i] = GUARDIAN; //filling
#endif
	FUINT *pflags = &this->m_flags[idx];
	for(;;) {
		FUINT oldv = *pflags;
		FUINT newv = oldv & none;
//		fprintf(stderr, "d: %p, %d, %x, %x, %x\n", p, idx, oldv, newv, ones);
		if(atomicCompareAndSet(oldv, newv, pflags)) {
			assert(( oldv | none) == ~(FUINT)0); //checking for double free.
//			m_idx = idx; //writing a hint for a next allocation.
			if(oldv == ~(FUINT)0u)
				atomicDec( &this->m_flags_filled_cnt);
			else if(newv == 0) {
				if(atomicDecAndTest( &this->m_flags_nonzero_cnt)) {
					if(this->release_allocator(this)) {
						delete this; //suicide.
						return true;
					}
					else
						return false;
				}
			}
			break;
		}
	}
	return false;
}

template <class ALLOC>
inline ALLOC *
PoolAllocatorBase::allocate_chunk() {
	int cidx = 0;
	size_t chunk_size = ALLOC_MIN_CHUNK_SIZE;
	for(;;) {
		if(cidx >= ALLOC_MAX_CHUNKS) {
			fprintf(stderr, "# of chunks exceeds the limit.\n");
			return 0;
		}
		if( !s_chunks[cidx]) {
			if(atomicCompareAndSet((PoolAllocatorBase *)0, reinterpret_cast<PoolAllocatorBase *>(1u), &s_chunks[cidx])) {
				writeBarrier();
				break;
			}
			continue;
		}
		++cidx;
		if(cidx % NUM_ALLOCATORS_IN_SPACE == 0) {
			chunk_size = GROW_CHUNK_SIZE(chunk_size);
		}
	}

	while( !s_mmapped_spaces[cidx / NUM_ALLOCATORS_IN_SPACE]) {
		size_t mmap_size = chunk_size * NUM_ALLOCATORS_IN_SPACE;
		char *p = static_cast<char *>(
			mmap(0, mmap_size, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
		if(p == MAP_FAILED) {
			fprintf(stderr, "mmap() failed.\n");
			s_chunks[cidx] = 0;
			return 0;
		}
		writeBarrier();
		if(atomicCompareAndSet((char *)0, p, &s_mmapped_spaces[cidx / NUM_ALLOCATORS_IN_SPACE])) {
			readBarrier();
			fprintf(stderr, "Reserve swap space starting @ %p w/ len. of 0x%llxB.\n", p, (unsigned long long)mmap_size);
			break;
		}
		munmap(p, mmap_size);
	}
	char *addr =
		s_mmapped_spaces[cidx / NUM_ALLOCATORS_IN_SPACE] + chunk_size * (cidx % NUM_ALLOCATORS_IN_SPACE);
	int ret = mprotect(addr, chunk_size, PROT_READ | PROT_WRITE);
	assert( !ret);

	ALLOC *palloc = ALLOC::create(chunk_size, addr);
	s_chunks[cidx] = palloc;

	return palloc;
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
bool
PoolAllocator<ALIGN, FS, DUMMY>::create_allocator(int &aidx) {
	for(aidx = 0;; ++aidx) {
		if(aidx >= ALLOC_MAX_CHUNKS_OF_TYPE) {
			fprintf(stderr, "# of chunks for %d align. exceeds the limit.\n", ALIGN);
			throw std::bad_alloc();
		}
		if( !s_chunks_of_type[aidx])
			break;
	}
	if(atomicCompareAndSet((uintptr_t)0u, (uintptr_t)1u, &s_chunks_of_type[aidx])) {
		PoolAllocator<ALIGN, DUMMY, DUMMY> *palloc =
			allocate_chunk<PoolAllocator<ALIGN, DUMMY, DUMMY> >();
		if( !palloc) {
			s_chunks_of_type[aidx] = 0;
			throw std::bad_alloc();
		}

		palloc->m_idx_of_type = aidx;

		for(;;) {
			int acnt = s_chunks_of_type_ubound;
			if((aidx < acnt) || atomicCompareAndSet(acnt, aidx + 1, &s_chunks_of_type_ubound))
				break;
		}
		writeBarrier(); //for alloc.
		s_chunks_of_type[aidx] = reinterpret_cast<uintptr_t>(palloc);
//		writeBarrier();
//		fprintf(stderr, "New memory pool for %dB aligned, starting @ %p w/ len. of %pB.\n", (int)ALIGN,
//			alloc->m_mempool, (uintptr_t)ALLOC_CHUNK_SIZE);
//		printf("n");
		return true;
	}
	return false;
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
template <unsigned int SIZE>
inline void *
PoolAllocator<ALIGN, FS, DUMMY>::allocate() {
	int aidx = s_curr_chunk_idx;
	for(int cnt = 0;; ++cnt) {
		uintptr_t *palloc = &s_chunks_of_type[aidx];
		uintptr_t alloc = *palloc;
		if(alloc && !(alloc & 1u) && (atomicCompareAndSet(alloc, alloc | 1u, palloc))) {
			readBarrier();
			if(void *p =
				reinterpret_cast<PoolAllocator<ALIGN, DUMMY, DUMMY> *>(alloc)->allocate_pooled(SIZE)) {
	//			fprintf(stderr, "a: %p\n", p);
#ifdef GUARDIAN
				for(unsigned int i = 0; i < SIZE / sizeof(uint64_t); ++i) {
					if(static_cast<uint64_t *>(p)[i] != GUARDIAN) {
						fprintf(stderr, "Memory tainted in %p:64\n", &static_cast<uint64_t *>(p)[i]);
					}
				}
#endif
#ifdef FILLING_AFTER_ALLOC
				for(unsigned int i = 0; i < SIZE / sizeof(uint64_t); ++i)
					static_cast<uint64_t *>(p)[i] = FILLING_AFTER_ALLOC; //filling
#endif
				*palloc = alloc;
				return p;
			}
			*palloc = alloc;
		}
		int acnt = s_chunks_of_type_ubound;
		if(cnt >= acnt) {
			readBarrier();
			while(cnt >= s_chunks_of_type_ubound) {
				if(create_allocator(aidx))
					break;
				readBarrier();
				continue;
			}
			s_curr_chunk_idx = aidx;
			continue;
		}
		++aidx;
		if(aidx >= acnt) {
			readBarrier();
			if(aidx >= s_chunks_of_type_ubound)
				aidx = 0;
		}
		s_curr_chunk_idx = aidx;
	}
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
bool
PoolAllocator<ALIGN, FS, DUMMY>::release_allocator(PoolAllocator *palloc) {
	if(s_chunks_of_type_ubound <= LEAVE_VACANT_CHUNKS) {
		return false;
	}

	uintptr_t alloc = reinterpret_cast<uintptr_t>(palloc);
	int aidx = palloc->m_idx_of_type;
//	if(s_curr_chunk_idx ==  aidx) {
//		s_curr_chunk_idx = (aidx > 0) ? aidx - 1 : 0;
//	}

	if(atomicCompareAndSet(alloc, alloc | 1u, &s_chunks_of_type[aidx])) {
		readBarrier();
		//checking if the pool is really vacant.
		if( !palloc->m_flags_nonzero_cnt) {
#ifdef GUARDIAN
			void *ppool = reinterpret_cast<PoolAllocator *>(alloc)->m_mempool;
			for(unsigned int i = 0; i < ALLOC_CHUNK_SIZE / sizeof(uint64_t); ++i) {
				if(static_cast<uint64_t *>(ppool)[i] != GUARDIAN) {
					fprintf(stderr, "Memory tainted in %p:64\n", &static_cast<uint64_t *>(ppool)[i]);
				}
			}
#endif

			s_chunks_of_type[aidx] = 0;
			//decreasing upper boundary.
			while(int acnt = s_chunks_of_type_ubound) {
				if(s_chunks_of_type[acnt - 1])
					break;
				atomicCompareAndSet(acnt, acnt - 1, &s_chunks_of_type_ubound);
			}
			return true;
		}
		else {
			s_chunks_of_type[aidx] = alloc;
		}
	}
	return false;
}
inline void
PoolAllocatorBase::deallocate_chunk(int cidx, size_t chunk_size) {
	void *addr =
		s_mmapped_spaces[cidx / NUM_ALLOCATORS_IN_SPACE] + chunk_size * (cidx % NUM_ALLOCATORS_IN_SPACE);
	//releasing memory.
	mprotect(addr, chunk_size, PROT_NONE);

	s_chunks[cidx] = 0;
}

template <int CCNT, size_t CHUNK_SIZE>
inline bool
PoolAllocatorBase::deallocate_(void *p) {
	char *mp = s_mmapped_spaces[CCNT];
	if((CCNT > 0) && !mp)
		return false;
	ptrdiff_t pdiff = static_cast<char *>(p) - mp;
	if((pdiff >= 0) && (pdiff < (ptrdiff_t)CHUNK_SIZE * NUM_ALLOCATORS_IN_SPACE)) {
		int cidx = pdiff / CHUNK_SIZE + CCNT * NUM_ALLOCATORS_IN_SPACE;
		PoolAllocatorBase *palloc = s_chunks[cidx];
		if(palloc->deallocate_pooled(static_cast<char *>(p))) {
			deallocate_chunk(cidx, CHUNK_SIZE);
		}
		return true;
	}
	if(CCNT + 1 == ALLOC_MAX_MMAP_ENTRIES)
		return false;
	return deallocate_<(CCNT + 1 < ALLOC_MAX_MMAP_ENTRIES) ? CCNT + 1 : CCNT,
		(CCNT + 1 < ALLOC_MAX_MMAP_ENTRIES) ? GROW_CHUNK_SIZE(CHUNK_SIZE) : CHUNK_SIZE>(p);
}
inline bool
PoolAllocatorBase::deallocate(void *p) {
	if(deallocate_<0, ALLOC_MIN_CHUNK_SIZE>(p))
		return true;
	return false;
}
void
PoolAllocatorBase::release_chunks() {
	for(int cidx = 0; cidx < ALLOC_MAX_CHUNKS; ++cidx) {
		s_chunks[cidx] = 0;
	}
	size_t chunk_size = ALLOC_MIN_CHUNK_SIZE;
	for(int cnt = 0; cnt < ALLOC_MAX_MMAP_ENTRIES; ++cnt) {
		char *mp = s_mmapped_spaces[cnt];
		if( !mp)
			break;
		size_t mmap_size = chunk_size * NUM_ALLOCATORS_IN_SPACE;
		int ret = munmap(mp, mmap_size);
		s_mmapped_spaces[cnt] = 0;
		assert( !ret);
		chunk_size = GROW_CHUNK_SIZE(chunk_size);
	}
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PoolAllocator<ALIGN, FS, DUMMY>::release_pools() {
	int acnt = s_chunks_of_type_ubound;
	for(int aidx = 0; aidx < acnt; ++aidx) {
		uintptr_t alloc = s_chunks_of_type[aidx];
		alloc &= ~(uintptr_t)1u;
		reinterpret_cast<PoolAllocator<ALIGN, DUMMY, DUMMY> *>(alloc)->report_leaks();
		delete reinterpret_cast<PoolAllocator<ALIGN, DUMMY, DUMMY> *>(alloc);
		s_chunks_of_type[aidx] = 0;
	}
	s_chunks_of_type_ubound = 0;
}

void* allocate_large_size_or_malloc(size_t size) throw() {
	ALLOCATE_9_16X(4, size);
	ALLOCATE_9_16X(8, size);
	ALLOCATE_9_16X(16, size);
	ALLOCATE_9_16X(32, size);
	ALLOCATE_9_16X(64, size);

	return malloc(size);
}

inline void deallocate_pooled_or_free(void* p) throw() {
	if(PoolAllocatorBase::deallocate(p))
		return;
	free(p);
}

void release_pools() {
	PoolAllocator<ALLOC_SIZE1, true>::release_pools();
	PoolAllocator<ALLOC_SIZE2, true>::release_pools();
	PoolAllocator<ALLOC_SIZE3, true>::release_pools();
	PoolAllocator<ALLOC_SIZE4, true>::release_pools();
	PoolAllocator<ALLOC_SIZE5, true>::release_pools();
	PoolAllocator<ALLOC_SIZE7, true>::release_pools();
	PoolAllocator<ALLOC_SIZE9, true>::release_pools();
	PoolAllocator<ALLOC_SIZE11, true>::release_pools();
	PoolAllocator<ALLOC_SIZE13, true>::release_pools();
	PoolAllocator<ALLOC_SIZE15, true>::release_pools();
	PoolAllocator<ALLOC_ALIGN1>::release_pools();
	PoolAllocator<ALLOC_ALIGN2>::release_pools();
#if defined ALLOC_ALIGN3
	PoolAllocator<ALLOC_ALIGN3>::release_pools();
#endif
	PoolAllocatorBase::release_chunks();
}
void report_statistics() {
	size_t chunk_size = 0;
	size_t used_size = 0;
	for(int cidx = 0; cidx < PoolAllocatorBase::ALLOC_MAX_CHUNKS; ++cidx) {
		PoolAllocatorBase *palloc = PoolAllocatorBase::s_chunks[cidx];
		if(palloc) {
			size_t cs, us;
			palloc->report_statistics(cs, us);
			chunk_size += cs;
			used_size += us;
		}
	}
	printf("Total chunk size = 0x%llxB, filling = %.2f%%\n", (unsigned long long)chunk_size,
		(double)used_size / chunk_size * 100.0);
}

void* operator new(std::size_t size) throw(std::bad_alloc) {
	return new_redirected(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) throw() {
	try {
		return operator new(size);
	}
	catch (std::bad_alloc &) {
		return 0;
	}
}
void* operator new[](std::size_t size) throw(std::bad_alloc) {
	return operator new(size);
}
void* operator new[](std::size_t size, const std::nothrow_t&) throw() {
	try {
		return operator new(size);
	}
	catch (std::bad_alloc &) {
		return 0;
	}
}

void operator delete(void* p) throw() {
	return deallocate_pooled_or_free(p);
}
void operator delete(void* p, const std::nothrow_t&) throw() {
	return deallocate_pooled_or_free(p);
}
void operator delete[](void* p) throw() {
	return operator delete(p);
}
void operator delete[](void* p, const std::nothrow_t&) throw() {
	return operator delete(p);
}


char *PoolAllocatorBase::s_mmapped_spaces[ALLOC_MAX_MMAP_ENTRIES];
PoolAllocatorBase *PoolAllocatorBase::s_chunks[ALLOC_MAX_CHUNKS];
template <unsigned int ALIGN, bool FS, bool DUMMY>
uintptr_t PoolAllocator<ALIGN, FS, DUMMY>::s_chunks_of_type[ALLOC_MAX_CHUNKS_OF_TYPE];
template <unsigned int ALIGN, bool FS, bool DUMMY>
int ALLOC_TLS PoolAllocator<ALIGN, FS, DUMMY>::s_curr_chunk_idx;
template <unsigned int ALIGN, bool FS, bool DUMMY>
int PoolAllocator<ALIGN, FS, DUMMY>::s_chunks_of_type_ubound;

template class PoolAllocator<ALLOC_ALIGN1>;
template class PoolAllocator<ALLOC_ALIGN2>;
#if defined ALLOC_ALIGN3
	template class PoolAllocator<ALLOC_ALIGN3>;
#endif

template class PoolAllocator<ALLOC_SIZE1, true>;
template class PoolAllocator<ALLOC_SIZE2, true>;
template class PoolAllocator<ALLOC_SIZE3, true>;
template class PoolAllocator<ALLOC_SIZE4, true>;
template class PoolAllocator<ALLOC_SIZE5, true>;
template class PoolAllocator<ALLOC_SIZE7, true>;
template class PoolAllocator<ALLOC_SIZE9, true>;
template class PoolAllocator<ALLOC_SIZE11, true>;
template class PoolAllocator<ALLOC_SIZE13, true>;
template class PoolAllocator<ALLOC_SIZE15, true>;

template void *PoolAllocator<ALLOC_SIZE1, true>::allocate<ALLOC_SIZE1>();
template void *PoolAllocator<ALLOC_SIZE2, true>::allocate<ALLOC_SIZE2>();
template void *PoolAllocator<ALLOC_SIZE3, true>::allocate<ALLOC_SIZE3>();
template void *PoolAllocator<ALLOC_SIZE4, true>::allocate<ALLOC_SIZE4>();
template void *PoolAllocator<ALLOC_SIZE5, true>::allocate<ALLOC_SIZE5>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE6)>::allocate<ALLOC_SIZE6>();
template void *PoolAllocator<ALLOC_SIZE7, true>::allocate<ALLOC_SIZE7>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE8)>::allocate<ALLOC_SIZE8>();
template void *PoolAllocator<ALLOC_SIZE9, true>::allocate<ALLOC_SIZE9>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE10)>::allocate<ALLOC_SIZE10>();
template void *PoolAllocator<ALLOC_SIZE11, true>::allocate<ALLOC_SIZE11>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE12)>::allocate<ALLOC_SIZE12>();
template void *PoolAllocator<ALLOC_SIZE13, true>::allocate<ALLOC_SIZE13>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE14)>::allocate<ALLOC_SIZE14>();
template void *PoolAllocator<ALLOC_SIZE15, true>::allocate<ALLOC_SIZE15>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE16)>::allocate<ALLOC_SIZE16>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE9 * 2)>::allocate<ALLOC_SIZE9 * 2>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE10 * 2)>::allocate<ALLOC_SIZE10 * 2>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE11 * 2)>::allocate<ALLOC_SIZE11 * 2>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE12 * 2)>::allocate<ALLOC_SIZE12 * 2>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE13 * 2)>::allocate<ALLOC_SIZE13 * 2>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE14 * 2)>::allocate<ALLOC_SIZE14 * 2>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE15 * 2)>::allocate<ALLOC_SIZE15 * 2>();
template void *PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE16 * 2)>::allocate<ALLOC_SIZE16 * 2>();

//static struct PoolReleaser {
//	~PoolReleaser() {
//		release_pools();
//	}
//} pool_releaser;
