/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/

//#define GUARDIAN 0xaaaaaaaauLL
//#define FILLING_AFTER_ALLOC 0x55555555uLL
#define LEAVE_VACANT_CHUNKS 64 //keep at least this # of chunks.

#include "allocator.h"

#ifndef USE_STD_ALLOCATOR

#include "atomic.h"
#include "threadlocal.h"

#include <algorithm>
#include <assert.h>
#include <cerrno>
#include <cstdlib>
#include <string.h>
#include <type_traits>

// Per-thread flag: set to true when AllocPinCleanup has run, signalling
// that pool-allocator TLS (s_my_chunk, freelists, pin counts) is no
// longer valid.  Trivially destructible (`ALLOC_TLS` = `__thread`) so it
// survives past all thread_local / pthread_key destructors.  Checked in
// `new_redirected()` to fall back to malloc for any heap operations
// that occur during later TLS cleanup phases (e.g. pthread_key dtors
// like RunnerCounterRegistration).
ALLOC_TLS bool s_alloc_tls_off = false;

// Forward decl — the post-thread-exit functor used by AllocPinCleanup
// to overwrite every slot of `g_thread_slots[]` before chunks are
// released.  Defined later in this TU.
namespace { void *malloc_fallback(PoolAllocatorBase *, std::size_t size) noexcept; }

// Per-thread cleanup of pinned chunks. On thread exit the destructor of
// this TLS object walks the registered atomic pin counters and decrements
// each, allowing release_allocator() to reclaim chunks the thread had
// claimed. Fixed-size — no dynamic allocation in the destructor (which
// would recurse into the allocator). Capacity covers the count of
// distinct PoolAllocator template instantiations actually in use.
namespace {
struct AllocPinCleanup {
    static constexpr int MAX = 32;
    struct Pin {
        std::atomic<int> *count_ptr;
        PoolAllocatorBase *chunk;
    };
    Pin pinned[MAX] = {};
    int count = 0;
    void add(std::atomic<int> *p, PoolAllocatorBase *c) noexcept {
        if(count < MAX) pinned[count++] = {p, c};
    }
    ~AllocPinCleanup() noexcept {
        // Flush per-chunk owner freelists FIRST so their slots are
        // returned to the bitmap before the chunk is potentially
        // released (pin count → 0).
        for(int i = 0; i < count; ++i)
            pinned[i].chunk->flush_owner_freelist();
        // Repoint every per-thread alloc slot to `malloc_fallback`
        // BEFORE the chunk pins drop to 0 — otherwise a later TLS
        // destructor that allocates could route through a chunk that's
        // about to be released by another thread's deallocate.  This
        // also makes `new_redirected`'s hot path safe with no
        // activation-flag check: the table itself encodes "pool off".
        for(int b = 0; b < ALLOC_NUM_BUCKETS; ++b) {
            g_thread_slots[b].chunk = nullptr;
            g_thread_slots[b].alloc_fn = &malloc_fallback;
        }
        // Null out per-ALIGN `s_my_chunk` TLS pointers BEFORE
        // decrementing pin counts.  Without this, later TLS destructors
        // (e.g. RunnerCounterRegistration via pthread_key_create) that
        // run after this thread_local destructor see a stale s_my_chunk
        // and push to a dead freelist — permanent slot leak, or
        // use-after-free if the chunk was released.
        for(int i = 0; i < count; ++i)
            pinned[i].chunk->clear_owner_tls();
        // Signal that pool-allocator TLS is dead.  Read by
        // `is_allocator_thread_active()` from later (pthread_key) TLS
        // dtors.  `new_redirected` itself no longer checks this flag —
        // the per-bucket slot rewrite above is its analogue.
        s_alloc_tls_off = true;
        for(int i = 0; i < count; ++i)
            pinned[i].count_ptr->fetch_sub(1, std::memory_order_release);
    }
};
// XThreadLocal (NOT raw `thread_local`).  XThreadLocal's lazy-init
// path (`tls_storage` + `XThreadLocal<T>::ctor_`) uses `std::malloc`
// directly (commit 25998cbb), bypassing the global `operator new`
// override → no re-entry into `PoolAllocator::allocate()`, so this is
// safe to first-touch from the chunk-pin path inside `allocate()`.
//
// On Windows, raw `thread_local` would give each plugin DLL its own
// TLS slot (and own AllocPinCleanup), so pins added in one DLL would
// not be cleaned up via another DLL's slot.  XThreadLocal keyed on
// the libkame-side global's address gives every DLL the same slot.
XThreadLocal<AllocPinCleanup> tls_alloc_pin_cleanup;

// Cross-thread dealloc batch — single TLS buffer per thread holding
// {chunk, slot} pairs from non-owner deallocs.  Flushes when full
// (CAP entries) or at thread exit.  Flush sorts by chunk pointer to
// group same-chunk slots, then calls chunk->batch_return_to_bitmap
// per group to merge multiple slots into ONE CAS per m_flags word.
//
// vs the previous design (immediate single-slot bitmap CAS per
// non-owner dealloc): trades CAS count for a single TLS write +
// occasional sorted flush.  Wins on:
//   * UMA: TLS write (~2-5 ns) << atomic CAS (~10-20 ns);
//   * NUMA (Ohtaka): collapses cross-socket cache-line traffic
//     from N scattered m_flags words to one batched CAS per word
//     per chunk group.
//
// Stage 1 (this commit): non-owner deallocs only.  Owner-overflow
// (freelist full) keeps the existing single-slot bitmap CAS for
// now.  Stage 2 would route owner-overflow here too.
struct CrossDeallocBatch {
    static constexpr int CAP = 64;
    struct Entry { PoolAllocatorBase *chunk; void *slot; };
    Entry buf[CAP];
    int count = 0;

    void push(PoolAllocatorBase *c, void *s) noexcept {
        // We're called from deallocate paths; if the slot count has
        // hit CAP we must drain before pushing the new entry.  Drain
        // is in-thread (no synchronization needed beyond the bitmap
        // CAS that batch_return_to_bitmap does internally).
        if(count == CAP) flush();
        buf[count++] = {c, s};
    }
    void flush() noexcept {
        if(count == 0) return;
        // Sort by chunk pointer so same-chunk entries are contiguous.
        // Stable sort isn't required (slots from same chunk are
        // interchangeable at the bitmap level).
        std::sort(buf, buf + count,
                  [](const Entry &a, const Entry &b) {
                      return a.chunk < b.chunk;
                  });
        // Run per-chunk groups.  Each group → ONE virtual call →
        // batch_return_to_bitmap groups slots by m_flags word
        // internally for further CAS coalescing.
        // Stack scratch sized to CAP — bounded.
        void *slots_buf[CAP];
        int i = 0;
        while(i < count) {
            PoolAllocatorBase *c = buf[i].chunk;
            int j = i;
            while(j < count && buf[j].chunk == c) {
                slots_buf[j - i] = buf[j].slot;
                ++j;
            }
            c->batch_return_to_bitmap(slots_buf, j - i);
            i = j;
        }
        count = 0;
    }
    ~CrossDeallocBatch() noexcept { flush(); }
};
XThreadLocal<CrossDeallocBatch> tls_cross_dealloc_batch;

} // anon namespace

// Atomic helpers moved to allocator_prv.h so the header-inlined
// `batch_clear_impl` template member of PoolAllocator can use them.
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
#else
    #include <sys/mman.h>
#endif
#include <sys/types.h>

// `count_bits` and `find_zero_forward` are now in allocator_prv.h
// (header-visible for inline use by FS=false bucket-freelist push).
// Reference: H. S. Warren, Jr., "Beautiful Code", O'Reilly.

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
//! \param x should be 2^n (a single set bit).
//! \sa find_zero_forward(), find_first_oen().
//!
//! Compiles to `bsf`/`tzcnt` on x86 and `rbit;clz` on ARM64 via
//! __builtin_ctzll, so this single implementation covers every arch the
//! pool allocator supports. The former x86 inline-asm form is preserved
//! behind the same guard as a backstop for exotic toolchains.
template <typename T>
inline unsigned int count_zeros_forward(T x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(static_cast<unsigned long long>(x));
#elif defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__ || defined __x86_64__
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
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
        void *p = malloc(size);
#else
		void *p = (
			mmap(0, size + ALLOC_ALIGNMENT, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0));
		assert(p != MAP_FAILED);
#endif
		*static_cast<size_t *>(p) = size + ALLOC_ALIGNMENT;
		return static_cast<char *>(p) + ALLOC_ALIGNMENT;
}
inline void free_munmap(void *p) {
		p = static_cast<void *>(static_cast<char *>(p) - ALLOC_ALIGNMENT);
		size_t size = *static_cast<size_t *>(p);
	//	fprintf(stderr, "unmmap(), %d\n", (int)size);
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
        free(p);
#else
        int ret = munmap(p, size);
		assert( !ret);
#endif
}

bool g_sys_image_loaded = false;

void activateAllocator() {g_sys_image_loaded = true;}

template <unsigned int ALIGN, bool FS, bool DUMMY>
inline PoolAllocator<ALIGN, FS, DUMMY>::PoolAllocator(int count, char *addr, char *ppool) :
	PoolAllocatorBase(ppool),
	// Empty-list sentinel: `(char *)this`.  See try_owner_freelist_pop.
	m_freelist_head(reinterpret_cast<char *>(this)),
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
	// Layout: [class][m_flags].  The owner-thread freelist is now a
	// linked list embedded in free slots themselves (see
	// try_owner_freelist_pop / _push and the `m_freelist_head` field
	// comment in allocator_prv.h), so no separate uint16_t-index
	// array follows m_flags.
	size_t size_alloc = (sizeof(PoolAllocator) + sizeof(FUINT) - 1) * sizeof(FUINT);
	int count = size / ALIGN / sizeof(FUINT) / 8;
	char *area = static_cast<char *>(malloc(size_alloc + sizeof(FUINT) * count));
	if( !area)
		return 0;
	PoolAllocator *p = new(area) PoolAllocator(count, area, ppool);
	return p;
}
template <unsigned int ALIGN, bool DUMMY>
inline PoolAllocator<ALIGN, false, DUMMY>::PoolAllocator(int count, char *addr, char *ppool,
                                                          uint16_t *fs_buckets) :
	PoolAllocator<ALIGN, true, false>(count, addr, ppool),
	m_sizes(reinterpret_cast<FUINT *>( &addr[(sizeof(PoolAllocator) + sizeof(FUINT) - 1) * sizeof(FUINT)
	                                         + sizeof(FUINT) * count])),
	m_fs_buckets(fs_buckets) {
	m_available_bits = sizeof(FUINT) * 8;
	for(int i = 0; i < FS_MAX_BUCKETS; ++i)
		m_fs_bucket_count[i] = 0;
}
template <unsigned int ALIGN, bool DUMMY>
inline PoolAllocator<ALIGN, false, DUMMY> *PoolAllocator<ALIGN, false, DUMMY>::create(size_t size, char *ppool) {
	int count = size / ALIGN / sizeof(FUINT) / 8;
	size_t size_alloc = (sizeof(PoolAllocator) + sizeof(FUINT) - 1) * sizeof(FUINT);
	// FS=false uses bucket-based push instead of the inherited FS=true
	// linked-list freelist (which just stays at its empty sentinel).
	// Its own per-size bucket array m_fs_buckets:
	//   FS_MAX_BUCKETS × FS_BUCKET_CAP × 2 B = 8 KiB tail-allocated
	// after the m_flags / m_sizes arrays.
	constexpr size_t fs_buckets_bytes =
	    sizeof(uint16_t) * FS_MAX_BUCKETS * FS_BUCKET_CAP;
	// Layout: [class][m_flags][m_sizes][m_fs_buckets]
	char *area = static_cast<char *>(malloc(size_alloc + sizeof(FUINT) * count * 2
	                                        + fs_buckets_bytes));
	if( !area)
		return 0;
	uint16_t *fs_buckets = reinterpret_cast<uint16_t *>(
	    area + size_alloc + sizeof(FUINT) * count * 2);
	PoolAllocator *p = new(area) PoolAllocator(count, area, ppool, fs_buckets);
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
			// Always-CAS path (formerly an oldv==0 non-atomic fast write
			// existed here). Without an external lock around the chunk —
			// which the TLS s_my_chunk fast path in allocate() removes —
			// the non-atomic store would race with another thread doing
			// the same on the same flag word, producing torn writes that
			// hand the same bit to two threads. CAS even at oldv==0 is
			// only marginally slower and keeps the chunk thread-safe.
			FUINT newv = oldv | one; //set a flag.
			if(atomicCompareAndSet(oldv, newv, pflag)) {
				if(oldv == 0)
					atomicInc( &this->m_flags_nonzero_cnt);
				if(newv == ~(FUINT)0u)
                    atomicInc( &this->m_flags_filled_cnt);
				writeBarrier(); //for the counters.
				break;
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
	// Owner-side per-size bucket fast path.  Safe to call regardless
	// of s_my_chunk: in the owner case we are the single writer; in
	// the slow-path-just-claimed case the buckets were drained on the
	// previous owner's thread exit (m_fs_bucket_count[*] all zero).
	// `m_sizes` at the popped slot is already correct (slot was
	// allocated with the same N originally).
	if(void *p = this->fs_try_bucket_pop(SIZE / ALIGN)) return p;
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
			// Always-CAS path (formerly an oldv==0 non-atomic fast write
			// existed here). See sibling allocate_pooled(FS=true) for
			// the rationale: TLS s_my_chunk fast path in allocate()
			// removes the bit0-lock around chunk access, so the
			// non-atomic store would torn-write under contention.
			FUINT newv = oldv | ones; //filling with SIZE ones.
			if(atomicCompareAndSet(oldv, newv, pflag)) {
				if(oldv == 0)
					atomicInc( &this->m_flags_nonzero_cnt);
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
	// Owner-side per-size bucket fast path.  fs_try_bucket_push
	// decodes N from m_sizes and stores slot_idx into bucket[N-1].
	// Non-owner deallocs OR N outside FS_MAX_BUCKETS OR bucket-at-cap
	// route to the cross-thread TLS batch.
	//
	// `s_my_chunk` has declared type `PoolAllocator<ALIGN, false, false>*`
	// (from the base's `PoolAllocator<ALIGN, DUMMY, DUMMY>*` with
	// DUMMY=false), while `this` has type `PoolAllocator<ALIGN, false,
	// DUMMY>*` — different template instantiations referring to the
	// same chunk object.  Compare as void* to bypass the type mismatch.
	if(static_cast<void *>(PoolAllocator<ALIGN, true, false>::s_my_chunk)
	    == static_cast<void *>(this)) {
		if(this->fs_try_bucket_push(p)) return false;
	}
	tls_cross_dealloc_batch->push(this, p);
	return false;
}

template <unsigned int ALIGN, bool DUMMY>
void
PoolAllocator<ALIGN, false, DUMMY>::flush_owner_freelist() noexcept {
	// Walk each bucket; for each entry, CAS-clear the N corresponding
	// bits in m_flags.  Per-slot CAS (vs the m_freelist_bits-style
	// "OR all then one CAS per word" optimisation) — flush is a
	// thread-exit-only path so the loop cost is amortised away.
	for(int b = 0; b < FS_MAX_BUCKETS; ++b) {
		unsigned N = (unsigned)(b + 1);
		FUINT bit_pattern = (((FUINT)1u << N) - 1u);  // N low bits set
		unsigned cnt = this->m_fs_bucket_count[b];
		for(unsigned k = 0; k < cnt; ++k) {
			uint32_t slot_idx =
			    this->m_fs_buckets[(size_t)b * FS_BUCKET_CAP + k];
			unsigned idx = slot_idx / (sizeof(FUINT) * 8);
			unsigned sidx = slot_idx % (sizeof(FUINT) * 8);
			FUINT mask = bit_pattern << sidx;
			FUINT *pflag = &this->m_flags[idx];
			for(;;) {
				FUINT oldv = *pflag;
				FUINT newv = oldv & ~mask;
				if(atomicCompareAndSet(oldv, newv, pflag)) {
					if(newv == 0 && oldv != 0) {
						this->m_available_bits = sizeof(FUINT) * 8;
						atomicDec( &this->m_flags_nonzero_cnt);
					}
					break;
				}
			}
		}
		this->m_fs_bucket_count[b] = 0;
	}
}

// FS=false batch return — multi-bit clear (slots vary in n_slots,
// recovered from m_sizes).  Reuses the inherited batch_clear_impl
// skeleton with a multi-bit MaskFn and FS=false-specific OnClearFn
// (no filled_cnt, updates m_available_bits).
template <unsigned int ALIGN, bool DUMMY>
void
PoolAllocator<ALIGN, false, DUMMY>::batch_return_to_bitmap(
    void **slot_ptrs, int n) noexcept {
	// NOTE: parameter is named `slot_ptrs`, not `slots`, to avoid clashing
	// with Qt's `slots` keyword macro (Qt6 qtmetamacros.h defines
	// `#define slots Q_SLOTS`, which expands to an empty token in
	// non-MOC builds — turning `slots[i]` into `[i]`, parsed as a lambda
	// capture list and producing baffling errors at the use sites).
	if(n <= 0) return;
	int i = 0;
	this->batch_clear_impl(
		[&]() -> char * {
			if(i >= n) return nullptr;
			return static_cast<char *>(slot_ptrs[i++]);
		},
		// MaskFn: FS=false multi-bit (decode N from m_sizes)
		[this](int idx, unsigned sidx, char *p) -> FUINT {
			FUINT nones = find_zero_forward(m_sizes[idx] >> sidx);
			FUINT slot_mask = (nones | (nones - 1u)) << sidx;
#ifdef GUARDIAN
			unsigned int n_slots = count_bits(slot_mask);
			for(unsigned int j = 0; j < n_slots * ALIGN / sizeof(uint64_t); ++j)
				reinterpret_cast<uint64_t *>(p)[j] = GUARDIAN;
#else
			(void)p;
#endif
			return slot_mask;
		},
		// OnClearFn: FS=false counter + available-bits hint
		[this](FUINT oldv, FUINT newv) {
			if(newv == 0 && oldv != 0) {
				m_available_bits = sizeof(FUINT) * 8;
				atomicDec( &this->m_flags_nonzero_cnt);
			}
		});
	// Chunk-release check.  Safe to delete this — the CrossDeallocBatch
	// flush caller advances past this chunk-group after the call.
	if(this->m_flags_nonzero_cnt == 0
	        && PoolAllocator<ALIGN, true, false>::release_allocator(this)) {
		delete this;
	}
}

// Body of `batch_clear_impl` — out-of-class definition kept in
// allocator.cpp.  The function is template-on-lambdas; bodies in the
// header would balloon allocator_prv.h with a non-trivial loop that's
// only exercised from the cross-dealloc-batch flush and the
// owner-thread freelist flush — both rare, "long" code paths.  The
// SHORT push/pop helpers (try_owner_freelist_push/pop) ARE in the
// header for inlining (per-dealloc / per-alloc hot paths).
template <unsigned int ALIGN, bool FS, bool DUMMY>
template <typename FetchSlot, typename MaskFn, typename OnClearFn>
void
PoolAllocator<ALIGN, FS, DUMMY>::batch_clear_impl(
    FetchSlot fetch_slot, MaskFn mask_fn, OnClearFn on_clear) noexcept {
	FUINT *masks = static_cast<FUINT *>(
		alloca(sizeof(FUINT) * (size_t)this->m_count));
	for(int i = 0; i < this->m_count; ++i) masks[i] = 0;
	bool any = false;
	while(char *p = fetch_slot()) {
		int midx = (p - this->m_mempool) / ALIGN;
		int idx = midx / (sizeof(FUINT) * 8);
		unsigned int sidx = midx % (sizeof(FUINT) * 8);
		masks[idx] |= mask_fn(idx, sidx, p);
		any = true;
	}
	if( !any) return;
	for(int idx = 0; idx < this->m_count; ++idx) {
		FUINT clear_mask = masks[idx];
		if( !clear_mask) continue;
		FUINT nones = ~clear_mask;
		FUINT *pflags = &this->m_flags[idx];
		for(;;) {
			FUINT oldv = *pflags;
			FUINT newv = oldv & nones;
			if(atomicCompareAndSet(oldv, newv, pflags)) {
				on_clear(oldv, newv);
				break;
			}
		}
	}
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PoolAllocator<ALIGN, FS, DUMMY>::flush_owner_freelist_to_bitmap() noexcept {
	// Drain the chunk-private owner freelist (linked list embedded in
	// the free slots — see `try_owner_freelist_pop` / `_push`) and
	// CAS-clear the matching bits in m_flags, batched by word index.
	// Shared `batch_clear_impl` skeleton — see allocator_prv.h.
	if(this->m_freelist_head == reinterpret_cast<char *>(this)) return;
	this->batch_clear_impl(
		// FetchSlot: pop one slot from the head of the embedded linked
		// list.  Each free slot's first 8 bytes hold the next pointer
		// (or `(char *)this` sentinel for the list's tail).
		[this]() -> char * {
			char *head = this->m_freelist_head;
			if(head == reinterpret_cast<char *>(this)) return nullptr;
			this->m_freelist_head = *reinterpret_cast<char **>(head);
#ifdef GUARDIAN
			for(unsigned int i = 0; i < ALIGN / sizeof(uint64_t); ++i)
				reinterpret_cast<uint64_t *>(head)[i] = GUARDIAN;
#endif
			return head;
		},
		// MaskFn: FS=true single bit (same as batch_return_to_bitmap)
		[](int /*idx*/, unsigned sidx, char * /*p*/) -> FUINT {
			return ((FUINT)1u) << sidx;
		},
		// OnClearFn: FS=true counter updates
		[this](FUINT oldv, FUINT newv) {
			if(oldv == ~(FUINT)0u)
				atomicDec( &this->m_flags_filled_cnt);
			if(newv == 0 && oldv != 0)
				atomicDec( &this->m_flags_nonzero_cnt);
		});
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PoolAllocator<ALIGN, FS, DUMMY>::flush_owner_freelist() noexcept {
	flush_owner_freelist_to_bitmap();
}

// Batched bitmap clear of slots passed via argument array (vs the
// chunk-private freelist drained by flush_owner_freelist_to_bitmap).
// Called from CrossDeallocBatch::flush — slots all belong to THIS
// chunk (caller groups by chunk pointer), so one shared direct-map
// scratch suffices.
template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PoolAllocator<ALIGN, FS, DUMMY>::batch_return_to_bitmap(
    void **slot_ptrs, int n) noexcept {
	// `slot_ptrs` not `slots` — Qt's `slots` keyword macro (see comment
	// on the FS=false sibling above) clobbers the name.
	if(n <= 0) return;
	int i = 0;
	this->batch_clear_impl(
		// FetchSlot: walk the argument array
		[&]() -> char * {
			if(i >= n) return nullptr;
			char *p = static_cast<char *>(slot_ptrs[i++]);
#ifdef GUARDIAN
			for(unsigned int j = 0; j < ALIGN / sizeof(uint64_t); ++j)
				reinterpret_cast<uint64_t *>(p)[j] = GUARDIAN;
#endif
			return p;
		},
		// MaskFn: FS=true single bit
		[](int /*idx*/, unsigned sidx, char * /*p*/) -> FUINT {
			return ((FUINT)1u) << sidx;
		},
		// OnClearFn: FS=true counter updates
		[this](FUINT oldv, FUINT newv) {
			if(oldv == ~(FUINT)0u)
				atomicDec( &this->m_flags_filled_cnt);
			if(newv == 0 && oldv != 0)
				atomicDec( &this->m_flags_nonzero_cnt);
		});
	// Chunk-release check.
	if(this->m_flags_nonzero_cnt == 0
	        && PoolAllocator<ALIGN, FS, DUMMY>::release_allocator(this)) {
		delete this;
	}
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PoolAllocator<ALIGN, FS, DUMMY>::clear_owner_tls() noexcept {
	s_my_chunk = nullptr;
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
bool
PoolAllocator<ALIGN, FS, DUMMY>::deallocate_pooled(char *p) {
	// Three-way dispatch (Stage 2 — both owner-overflow AND non-owner
	// route to the per-thread cross-dealloc TLS batch):
	//
	//   owner + freelist has room  → push to owner freelist (no atomic)
	//   owner + freelist at cap   ─┐
	//   non-owner                 ─┴→ TLS batch (sorted flush, batched
	//                                  bitmap CAS per m_flags word)
	//
	// The bitmap CAS path that used to handle owner-overflow + non-owner
	// is now entirely in batch_return_to_bitmap (called from the TLS
	// batch flush), where multiple slots merge into 1 CAS per m_flags
	// word.  release_allocator + chunk delete also moves there.
	if(s_my_chunk == this && this->try_owner_freelist_push(p)) {
		// Slot stays "allocated" in the bitmap until flushed back via
		// flush_owner_freelist_to_bitmap (thread exit) or the chunk
		// overflows.  Owner's next alloc may pop it back immediately.
		return false;
	}
	// Owner-overflow OR non-owner: TLS batch.
	tls_cross_dealloc_batch->push(this, p);
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
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
        char *p = static_cast<char *>(
            malloc(mmap_size));
#else
		char *p = static_cast<char *>(
			mmap(0, mmap_size, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0));
		if(p == MAP_FAILED) {
			fprintf(stderr, "mmap() failed.\n");
			s_chunks[cidx] = 0;
			return 0;
		}
#endif
		writeBarrier();
		if(atomicCompareAndSet((char *)0, p, &s_mmapped_spaces[cidx / NUM_ALLOCATORS_IN_SPACE])) {
			readBarrier();
			fprintf(stderr, "Reserve swap space starting @ %p w/ len. of 0x%llxB.\n", p, (unsigned long long)mmap_size);
			break;
		}
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
        free(p);
#else
        munmap(p, mmap_size);
#endif
    }
	char *addr =
		s_mmapped_spaces[cidx / NUM_ALLOCATORS_IN_SPACE] + chunk_size * (cidx % NUM_ALLOCATORS_IN_SPACE);
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
#else
    // mprotect failure is silent under NDEBUG (assert no-op); the next
    // write to addr would then SIGBUS. Treat it as fatal so the failure
    // is visible.
    int ret = mprotect(addr, chunk_size, PROT_READ | PROT_WRITE);
    if(ret != 0) {
        fprintf(stderr,
            "mprotect(%p, 0x%llx, RW) failed: errno=%d (chunk %d, "
            "page-align: addr&%d=%lu size&%d=%lu)\n",
            addr, (unsigned long long)chunk_size, errno, cidx,
            ALLOC_PAGE_SIZE, (unsigned long)((uintptr_t)addr % ALLOC_PAGE_SIZE),
            ALLOC_PAGE_SIZE, (unsigned long)(chunk_size % ALLOC_PAGE_SIZE));
        std::abort();
    }
#endif

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
void *
PoolAllocator<ALIGN, FS, DUMMY>::allocate_chunk_path(unsigned int SIZE) {
	// Cold path of allocate<SIZE>().  Called only when the
	// header-inline `try_owner_freelist_pop` fast path missed (either
	// `s_my_chunk == nullptr` on first thread access, or the owner-
	// thread freelist for this chunk was empty).
	//
	// Thread-exit cleanup is handled centrally by `AllocPinCleanup::~dtor`
	// (registered via `XThreadLocal<AllocPinCleanup>::operator*()` on the
	// first call that pins a chunk, a few lines below).  No per-template
	// thread_local guard is needed here, and the previous `(void)&s_tls_guard`
	// ODR-use is removed so we don't pay a C++ thread_local init thunk call
	// per allocation (macOS arm64 emits `bl __ZTH...11s_tls_guardE`).
	// Try the bitmap-CAS path on the pinned chunk before falling all the
	// way through to the chunk-claim loop.
	// allocate_pooled() does its own per-flag atomic CAS so concurrent
	// allocations from the same chunk by other threads are safe; the
	// expensive bit0-lock CAS on s_chunks_of_type[] is skipped entirely.
	// release_allocator() is gated on m_thread_pinned_count == 0 so the
	// chunk cannot be freed underneath us.
	if(PoolAllocator<ALIGN, DUMMY, DUMMY> *my = s_my_chunk) {
		if(void *p = my->allocate_pooled(SIZE)) {
#ifdef GUARDIAN
			for(unsigned int i = 0; i < SIZE / sizeof(uint64_t); ++i) {
				if(static_cast<uint64_t *>(p)[i] != GUARDIAN) {
					fprintf(stderr, "Memory tainted in %p:64\n", &static_cast<uint64_t *>(p)[i]);
				}
			}
#endif
#ifdef FILLING_AFTER_ALLOC
			for(unsigned int i = 0; i < SIZE / sizeof(uint64_t); ++i)
				static_cast<uint64_t *>(p)[i] = FILLING_AFTER_ALLOC;
#endif
			return p;
		}
		// Pinned chunk full — fall through to slow path to find/create
		// another. The pinned count on the old chunk is left bumped
		// (one thread's worth of extra residency), preventing release
		// while we still might dealloc objects originally allocated
		// from it. New chunk's pin replaces it as the fast-path target.
	}
	// Slow path: claim an UNCLAIMED chunk exclusively (compare_exchange
	// 0 → 1 on m_thread_pinned_count). This guarantees each thread has a
	// dedicated chunk so the chunk's bitmap CAS in allocate_pooled is
	// uncontended on the hot path. The pin is registered with
	// tls_alloc_pin_cleanup so it is decremented on thread exit, freeing
	// the chunk for reuse by future threads (otherwise long-running
	// programs with thread churn would exhaust ALLOC_MAX_CHUNKS_OF_TYPE).
	int aidx = s_curr_chunk_idx;
	for(int cnt = 0;; ++cnt) {
		uintptr_t *palloc = &s_chunks_of_type[aidx];
		uintptr_t alloc = *palloc;
		if(alloc && !(alloc & 1u)) {
			PoolAllocator<ALIGN, DUMMY, DUMMY> *chunk =
				reinterpret_cast<PoolAllocator<ALIGN, DUMMY, DUMMY> *>(alloc);
			int expected = 0;
			if(chunk->m_thread_pinned_count.compare_exchange_strong(
					expected, 1, std::memory_order_acq_rel)) {
				// Exclusive claim succeeded — register pin cleanup
				// (with chunk pointer so the cleanup hook can flush
				// the per-chunk owner-thread freelist on thread exit),
				// cache as TLS fast-path target, and allocate.
				tls_alloc_pin_cleanup->add(&chunk->m_thread_pinned_count, chunk);
				s_my_chunk = chunk;
				if(void *p = chunk->allocate_pooled(SIZE)) {
#ifdef GUARDIAN
					for(unsigned int i = 0; i < SIZE / sizeof(uint64_t); ++i) {
						if(static_cast<uint64_t *>(p)[i] != GUARDIAN) {
							fprintf(stderr, "Memory tainted in %p:64\n", &static_cast<uint64_t *>(p)[i]);
						}
					}
#endif
#ifdef FILLING_AFTER_ALLOC
					for(unsigned int i = 0; i < SIZE / sizeof(uint64_t); ++i)
						static_cast<uint64_t *>(p)[i] = FILLING_AFTER_ALLOC;
#endif
					return p;
				}
				// Claimed chunk is full — leave it pinned (we may still
				// hold objects in it) and continue searching.
			}
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
	// A chunk pinned by any thread's TLS s_my_chunk must not be released
	// — that thread's next allocate() would dereference freed memory.
	// The pin count is bumped (once per thread per chunk) on the
	// fast-path-claim slow path and never decremented in steady state.
	if(palloc->m_thread_pinned_count.load(std::memory_order_relaxed) > 0)
		return false;

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
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
#else
    mprotect(addr, chunk_size, PROT_NONE);
#endif

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
		// Defensive: cidx may map to a slot that is still in-creation
		// (s_chunks[cidx] == (PoolAllocatorBase*)1u sentinel set by
		// allocate_chunk's CAS) or freed (== nullptr) when a non-pool
		// pointer (e.g. libsystem malloc on macOS Apple Silicon, used
		// by ICU/Foundation during early process startup) happens to
		// land within our mmap virtual address range. Treat both as
		// "not our pointer" so the caller falls through to std::free.
		if((uintptr_t)palloc <= (uintptr_t)1u)
			return false;
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
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
        free(mp);
#else
        int ret = munmap(mp, mmap_size);
        assert( !ret);
#endif
		s_mmapped_spaces[cnt] = 0;
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

    return std::malloc(size);
}

// =====================================================================
// Per-thread allocation functor table.  See allocator_prv.h's comment
// above `AllocSlot` for the high-level rationale.  Lives here so the
// table's static initializer can take addresses of the per-bucket
// `bucket_first_access` template instantiations.
// =====================================================================
namespace {

//! Bucket → (ALIGN, FS, SIZE) mapping.  Specialized for buckets 1..16
//! to match the dispatch in the old if-chain `new_redirected` body.
//! `PunType` matches the `s_my_chunk` declaration in the bucket's
//! PoolAllocator instantiation (= `PoolAllocator<ALIGN, DUMMY, DUMMY>`
//! where DUMMY follows from the inheritance for FS=false partial specs).
template <int B> struct BucketTraits;

#define KAME_DECL_BUCKET(B_, ALIGN_, FS_, SIZE_) \
    template<> struct BucketTraits<B_> { \
        static constexpr unsigned int ALIGN = (ALIGN_); \
        static constexpr bool FS = (FS_); \
        static constexpr unsigned int SIZE = (SIZE_); \
        using PoolType = PoolAllocator<ALIGN, FS>; \
        using PunType = PoolAllocator<ALIGN, FS, FS>; \
    }

KAME_DECL_BUCKET( 1, ALLOC_SIZE1,                  true,  ALLOC_SIZE1 );
KAME_DECL_BUCKET( 2, ALLOC_SIZE2,                  true,  ALLOC_SIZE2 );
KAME_DECL_BUCKET( 3, ALLOC_SIZE3,                  true,  ALLOC_SIZE3 );
KAME_DECL_BUCKET( 4, ALLOC_SIZE4,                  true,  ALLOC_SIZE4 );
KAME_DECL_BUCKET( 5, ALLOC_SIZE5,                  true,  ALLOC_SIZE5 );
KAME_DECL_BUCKET( 6, ALLOC_ALIGN(ALLOC_SIZE6),    false,  ALLOC_SIZE6 );
KAME_DECL_BUCKET( 7, ALLOC_SIZE7,                  true,  ALLOC_SIZE7 );
KAME_DECL_BUCKET( 8, ALLOC_ALIGN(ALLOC_SIZE8),    false,  ALLOC_SIZE8 );
KAME_DECL_BUCKET( 9, ALLOC_SIZE9,                  true,  ALLOC_SIZE9 );
KAME_DECL_BUCKET(10, ALLOC_ALIGN(ALLOC_SIZE10),   false,  ALLOC_SIZE10);
KAME_DECL_BUCKET(11, ALLOC_SIZE11,                 true,  ALLOC_SIZE11);
KAME_DECL_BUCKET(12, ALLOC_ALIGN(ALLOC_SIZE12),   false,  ALLOC_SIZE12);
KAME_DECL_BUCKET(13, ALLOC_SIZE13,                 true,  ALLOC_SIZE13);
KAME_DECL_BUCKET(14, ALLOC_ALIGN(ALLOC_SIZE14),   false,  ALLOC_SIZE14);
KAME_DECL_BUCKET(15, ALLOC_SIZE15,                 true,  ALLOC_SIZE15);
KAME_DECL_BUCKET(16, ALLOC_ALIGN(ALLOC_SIZE16),   false,  ALLOC_SIZE16);
#undef KAME_DECL_BUCKET

//! State-machine "off" / post-cleanup slot — every g_thread_slots[]
//! entry points here after AllocPinCleanup::~dtor runs.  Also serves
//! as the dispatch target for any path that wants to fall back to
//! plain malloc (e.g. if a future bucket adds a size that overflows
//! ALLOC_MAX_CHUNKS_OF_TYPE).
void *malloc_fallback(PoolAllocatorBase * /*ignored*/, std::size_t size) noexcept {
    return std::malloc(size);
}

//! Out-of-line cold path for bucket B: freelist miss → bitmap CAS /
//! chunk-claim / create_allocator.  Marked `cold` so the compiler keeps
//! it out of the hot icache lines.  Split out of `bucket_steady_alloc`
//! so the fast path is frame-less (no callee-saved register push) —
//! the compiler can pass `chunk` straight through x0 to a tail call
//! here when the freelist misses.
template <int B>
__attribute__((cold, noinline))
void *bucket_steady_alloc_slow(PoolAllocatorBase *chunk, std::size_t /*size*/) noexcept {
    using BT = BucketTraits<B>;
    using PA = typename BT::PoolType;
    void *p = PA::allocate_chunk_path(BT::SIZE);
    // After allocate_chunk_path, s_my_chunk may have advanced to a
    // freshly-claimed chunk (the old one filled).  Keep g_thread_slots
    // in sync — the next hot-path call will then use the new chunk.
    PoolAllocatorBase *new_chunk = PA::get_pinned_chunk_base();
    if(new_chunk != chunk) g_thread_slots[B].chunk = new_chunk;
    return p;
}

//! Steady-state hot path for bucket B.  Reads `chunk` from the slot
//! (passed as arg — no per-template `s_my_chunk` TLV thunk).  Pops
//! from the chunk's owner freelist.  On miss, `musttail`-calls the
//! cold path so this function itself remains frame-less.
template <int B>
__attribute__((hot))
void *bucket_steady_alloc(PoolAllocatorBase *chunk, std::size_t /*size*/) noexcept {
    using Pun = typename BucketTraits<B>::PunType;
    if(void *p = static_cast<Pun *>(chunk)->try_owner_freelist_pop())
        return p;
    // Tail-call to slow path.  Same signature → compiler typically
    // emits a `b` (jump) rather than `bl`/`ret`, keeping this function
    // frame-less.  `musttail` is rejected here because of an implicit
    // `noexcept`-induced cleanup boundary — the optimizer still does it.
    return bucket_steady_alloc_slow<B>(chunk, 0);
}

//! First-access trampoline for bucket B.  Initial value of every slot
//! `alloc_fn` field.  Checks the activation flag (the ONE place this
//! check still lives — in the cold path), claims a chunk via the
//! existing `allocate<SIZE>()` slow path which registers AllocPinCleanup,
//! then rewrites the slot to point to `bucket_steady_alloc<B>` for all
//! subsequent allocations on this thread.
template <int B>
void *bucket_first_access(PoolAllocatorBase * /*ignored*/, std::size_t size) noexcept {
    if( !g_sys_image_loaded) {
        // Pool not activated yet (dyld / pre-main static-init phase).
        // Don't rewrite the slot — we want to retry the activation
        // check on every call until activateAllocator() is invoked.
        return std::malloc(size);
    }
    using BT = BucketTraits<B>;
    using PA = typename BT::PoolType;
    // Use the existing allocate<>() entry point: claims a chunk, pins
    // it via tls_alloc_pin_cleanup, sets `PA::s_my_chunk`, and returns
    // the allocated slot.  After this call returns, we know the chunk
    // is reserved for this thread.
    void *p = PA::template allocate<BT::SIZE>();
    PoolAllocatorBase *chunk = PA::get_pinned_chunk_base();
    if(chunk) {
        // Install steady-state functor + cache the chunk pointer so
        // subsequent allocations skip the activation check entirely.
        g_thread_slots[B].chunk = chunk;
        g_thread_slots[B].alloc_fn = &bucket_steady_alloc<B>;
    }
    return p;
}

} // anon namespace

// The per-thread table.  `__thread` (= `ALLOC_TLS` on GCC/Clang) so the
// storage lifetime extends past every TLS destructor on this thread —
// XThreadLocal would `std::free` the underlying memory mid-cleanup,
// leaving the `cached` pointer dangling for any later TLS dtor that
// allocates.  Static initializer takes addresses of `bucket_first_access`
// template instantiations (constant expressions); bucket 0 maps to
// bucket 1's 16-B allocator so size=0 allocations don't fault.
ALLOC_TLS AllocSlot g_thread_slots[ALLOC_NUM_BUCKETS] = {
    { nullptr, &bucket_first_access< 1> },  // bucket 0: size==0 → 16-B alloc
    { nullptr, &bucket_first_access< 1> },  // bucket 1: size ≤  16
    { nullptr, &bucket_first_access< 2> },  // bucket 2: size ≤  32
    { nullptr, &bucket_first_access< 3> },  // bucket 3: size ≤  48
    { nullptr, &bucket_first_access< 4> },  // bucket 4: size ≤  64
    { nullptr, &bucket_first_access< 5> },  // bucket 5: size ≤  80
    { nullptr, &bucket_first_access< 6> },  // bucket 6: size ≤  96 (FS=false)
    { nullptr, &bucket_first_access< 7> },  // bucket 7: size ≤ 112
    { nullptr, &bucket_first_access< 8> },  // bucket 8: size ≤ 128 (FS=false)
    { nullptr, &bucket_first_access< 9> },  // bucket 9: size ≤ 144
    { nullptr, &bucket_first_access<10> },  // bucket 10: size ≤ 160 (FS=false)
    { nullptr, &bucket_first_access<11> },  // bucket 11: size ≤ 176
    { nullptr, &bucket_first_access<12> },  // bucket 12: size ≤ 192 (FS=false)
    { nullptr, &bucket_first_access<13> },  // bucket 13: size ≤ 208
    { nullptr, &bucket_first_access<14> },  // bucket 14: size ≤ 224 (FS=false)
    { nullptr, &bucket_first_access<15> },  // bucket 15: size ≤ 240
    { nullptr, &bucket_first_access<16> },  // bucket 16: size ≤ 256 (FS=false)
};

// Out-of-line large-size dispatch.  Sizes > 256 B fall here from
// `new_redirected`.  Keeps the activation-flag check (cold path is the
// right place for it — only paid by larger allocations).
void *new_redirected_large(std::size_t size) noexcept {
    if( !g_sys_image_loaded || s_alloc_tls_off)
        return std::malloc(size);
    ALLOCATE_9_16X(2, size);
    return allocate_large_size_or_malloc(size);
}

inline void deallocate_pooled_or_free(void* p) throw() {
	// Mirror new_redirected's gate: when the pool isn't activated yet
	// (very early in process startup, before main() runs
	// activateAllocator()), all allocations went through malloc. The
	// pool's mmap'd regions don't even exist yet, so any pointer must
	// be a malloc'd one — skip the pool deallocate to avoid touching
	// uninitialized s_mmapped_spaces / s_chunks.
	if( !g_sys_image_loaded) {
		std::free(p);
		return;
	}
	if(PoolAllocatorBase::deallocate(p))
		return;
    std::free(p);
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

void* operator new(std::size_t size) {
    return new_redirected(size);
}
void* operator new[](std::size_t size) {
    return new_redirected(size);
}

void operator delete(void* p) noexcept {
    deallocate_pooled_or_free(p);
}
void operator delete[](void* p) noexcept {
    deallocate_pooled_or_free(p);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    return new_redirected(size);
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return new_redirected(size);
}
void operator delete(void* p, const std::nothrow_t&) noexcept {
    deallocate_pooled_or_free(p);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
    deallocate_pooled_or_free(p);
}

char *PoolAllocatorBase::s_mmapped_spaces[ALLOC_MAX_MMAP_ENTRIES];
PoolAllocatorBase *PoolAllocatorBase::s_chunks[ALLOC_MAX_CHUNKS];
template <unsigned int ALIGN, bool FS, bool DUMMY>
uintptr_t PoolAllocator<ALIGN, FS, DUMMY>::s_chunks_of_type[ALLOC_MAX_CHUNKS_OF_TYPE];
template <unsigned int ALIGN, bool FS, bool DUMMY>
int ALLOC_TLS PoolAllocator<ALIGN, FS, DUMMY>::s_curr_chunk_idx;
template <unsigned int ALIGN, bool FS, bool DUMMY>
int PoolAllocator<ALIGN, FS, DUMMY>::s_chunks_of_type_ubound;
template <unsigned int ALIGN, bool FS, bool DUMMY>
ALLOC_TLS PoolAllocator<ALIGN, DUMMY, DUMMY> *
    PoolAllocator<ALIGN, FS, DUMMY>::s_my_chunk;

// (Per-template `thread_local TlsGuard s_tls_guard` removed.
//  AllocPinCleanup::~AllocPinCleanup — fired via the pthread_key dtor
//  registered by `XThreadLocal<AllocPinCleanup>` on first allocate() —
//  is now the sole place that runs `flush_owner_freelist`,
//  `clear_owner_tls`, and sets `s_alloc_tls_off = true` at thread exit.
//  Eliminates the C++ thread_local init thunk that macOS arm64 emits
//  for `(void)&s_tls_guard` in the allocate() hot path.)

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

// (Per-SIZE explicit instantiation of allocate<SIZE>() removed —
//  allocate<SIZE>() is now header-inline in allocator_prv.h
//  (`[[gnu::always_inline]]`).  The out-of-line cold path,
//  `allocate_chunk_path(unsigned int)`, is a non-template member; it
//  is instantiated once per `(ALIGN, FS, DUMMY)` class instantiation
//  by the `template class PoolAllocator<...>;` directives above.)

//static struct PoolReleaser {
//	~PoolReleaser() {
//		release_pools();
//	}
//} pool_releaser;
#endif //USE_STD_ALLOCATOR
