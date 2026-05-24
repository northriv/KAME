/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/

#ifndef ALLOCATOR_PRV_H_
#define ALLOCATOR_PRV_H_

#ifndef USE_STD_ALLOCATOR

#include <new>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <atomic>
#include <limits>

#if defined(__GNUC__) || defined(__clang__)
	#define ALLOC_TLS __thread //TLS for allocations, could be better for NUMA.
#else
	#define ALLOC_TLS thread_local
#endif

#define ALLOC_MIN_CHUNK_SIZE (1024 * 256) //256KiB
// OS page size used to align growing chunk sizes for mprotect(). macOS
// arm64 uses 16 KiB pages; passing a non-page-aligned size to mprotect()
// fails silently (assert is no-op under NDEBUG) and the next access faults
// with SIGBUS. Linux x86_64 uses 4 KiB and Linux arm64 typically 4 KiB,
// occasionally 64 KiB. POWER usually 64 KiB. Use the largest plausible
// value per arch so chunk sizes round to a multiple in all cases.
#if defined(__APPLE__) && defined(__aarch64__)
    #define ALLOC_PAGE_SIZE 16384  // 16 KiB
#elif defined(__powerpc64__) || defined(__POWERPC__)
    #define ALLOC_PAGE_SIZE 65536  // 64 KiB
#else
    #define ALLOC_PAGE_SIZE 4096   // 4 KiB
#endif
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
    #define GROW_CHUNK_SIZE(x) ((size_t)(x / 4 * 5) / ALLOC_PAGE_SIZE * ALLOC_PAGE_SIZE)
    #define ALLOC_MIN_MMAP_SIZE ALLOC_MIN_CHUNK_SIZE
    #define ALLOC_MAX_MMAP_ENTRIES 24
#else
    #if defined __LP64__ || defined __LLP64__ || defined(_WIN64) || defined(__MINGW64__)
        #define GROW_CHUNK_SIZE(x) ((size_t)(x / 4 * 5) / ALLOC_PAGE_SIZE * ALLOC_PAGE_SIZE)
        #define ALLOC_MIN_MMAP_SIZE (1024 * 1024 * 32) //32MiB
        #define ALLOC_MAX_MMAP_ENTRIES 24 //27GiB approx.
    #else
        #define GROW_CHUNK_SIZE(x) ((size_t)(x / 8 * 9) / ALLOC_PAGE_SIZE * ALLOC_PAGE_SIZE)
        #define ALLOC_MIN_MMAP_SIZE (1024 * 1024 * 8) //8MiB
        #define ALLOC_MAX_MMAP_ENTRIES 32 //2.7GiB approx.
    #endif
#endif

#define ALLOC_ALIGNMENT 16 //bytes, not 8 but 16 for compatibility
#define ALLOC_MAX_CHUNKS_OF_TYPE \
	(ALLOC_MIN_MMAP_SIZE / ALLOC_MIN_CHUNK_SIZE * ALLOC_MAX_MMAP_ENTRIES)

class PoolAllocatorBase {
public:
	virtual ~PoolAllocatorBase() = default;
	template <int CCNT, size_t CHUNK_SIZE>
	static inline bool deallocate_(void *p);
	static inline bool deallocate(void *p);
	static void release_chunks();
	virtual void report_statistics(size_t &chunk_size, size_t &used_size) = 0;
	//! Hook called from `AllocPinCleanup` on thread exit: any items
	//! parked in the per-chunk owner-thread freelist (FS=true chunks
	//! only) are flushed back to the bitmap via the normal CAS path so
	//! they don't leak when the chunk is later released.  Default
	//! no-op; FS=true PoolAllocator overrides.
	virtual void flush_owner_freelist() noexcept {}
	//! Null out this thread's `s_my_chunk` for this chunk's ALIGN type.
	//! Called from `AllocPinCleanup` after freelist flush, before pin
	//! count decrement.  Prevents stale `s_my_chunk` from pushing to
	//! a dead freelist when later TLS destructors (e.g.
	//! `RunnerCounterRegistration` via `pthread_key`) do heap
	//! alloc/dealloc after `AllocPinCleanup` has already run.
	virtual void clear_owner_tls() noexcept {}
protected:
	PoolAllocatorBase(char *ppool) : m_mempool(ppool), m_numa_node(-1) {}
	virtual bool deallocate_pooled(char *p) = 0;

	template <class ALLOC>
	static ALLOC *allocate_chunk(int8_t numa_node);
	static void deallocate_chunk(int cidx, size_t chunk_size);

	//! A chunk, memory block.
	char * const m_mempool;
	//! NUMA node id (from `getcpu(2)` on Linux, else -1) of the
	//! thread that first-touched this chunk's memory pages via the
	//! `mprotect(RW)` + constructor write sequence in
	//! `allocate_chunk()`.  Set once at chunk creation; immutable.
	//!
	//! `PoolAllocator::allocate<>()` slow path uses this to prefer
	//! chunks on the calling thread's local NUMA — keeping per-tx
	//! allocations local even on dual-socket EPYC, where the
	//! pre-NUMA-aware allocator placed *all* chunks on the main
	//! thread's NUMA (the +27 % single-socket-pin gap observed in
	//! VTune bisects).
	//!
	//! On non-NUMA platforms (macOS, Windows) `m_numa_node` is left
	//! at -1; the slow-path filter degenerates to "match anything"
	//! → zero overhead vs the pre-NUMA-aware behaviour.
	int8_t m_numa_node;

private:
	friend void report_statistics();
	enum {NUM_ALLOCATORS_IN_SPACE = ALLOC_MIN_MMAP_SIZE / ALLOC_MIN_CHUNK_SIZE,
		ALLOC_MAX_CHUNKS = NUM_ALLOCATORS_IN_SPACE * ALLOC_MAX_MMAP_ENTRIES};
	//! Swap spaces given by anonymous mmap().
	static char *s_mmapped_spaces[ALLOC_MAX_MMAP_ENTRIES];
	static PoolAllocatorBase *s_chunks[ALLOC_MAX_CHUNKS];
};

//! Per-thread flag — true once any allocator-TLS destructor has fired
//! (`PoolAllocator<X>::TlsGuard::~dtor` or `AllocPinCleanup::~dtor`).
//! Forward-declared here so `PoolAllocator::TlsGuard` (below) can write
//! it before the full `extern` declaration further down.  The variable
//! is defined in allocator.cpp.
extern ALLOC_TLS bool s_alloc_tls_off;

//! \brief Memory blocks in a unit of double-quad word
//! can be allocated from fixed-size or variable-size memory pools.
//! \tparam FS determines fixed-size or variable-size.
//! \sa allocator_test.cpp.
template <unsigned int ALIGN, bool FS = false, bool DUMMY = true>
class PoolAllocator : public PoolAllocatorBase {
public:
	template <unsigned int SIZE>
	static void *allocate();
	static void release_pools();
	void report_leaks();
	void report_statistics(size_t &chunk_size, size_t &used_size) override;

	typedef uintptr_t FUINT;
protected:
	PoolAllocator(int count, char *addr, char *ppool,
	              void **freelist, int freelist_cap);
	inline void *allocate_pooled(unsigned int SIZE);
	bool deallocate_pooled(char *p) override;
	static bool create_allocator(int &aidx);
	static bool release_allocator(PoolAllocator *alloc);

	//! Every bit indicates occupancy in m_mempool.
	FUINT * const m_flags;
	//! A hint for searching in a chunk.
    int m_idx;
	const int m_count;
	//! # of flags that having non-zero values.
    int m_flags_nonzero_cnt;
    //! # of flags that having fully filled values.
	int m_flags_filled_cnt;
	int m_idx_of_type;

	//! Pointers to PooledAllocator. The LSB bit is set when allocation/releasing/creation is in progress.
	static uintptr_t s_chunks_of_type[ALLOC_MAX_CHUNKS_OF_TYPE];

	static int ALLOC_TLS s_curr_chunk_idx;
	static int s_chunks_of_type_ubound;
	//! Per-thread "currently owned" chunk for fast-path allocate(). When
	//! non-null, allocate<SIZE>() goes directly through this pointer
	//! instead of CAS-locking s_chunks_of_type[s_curr_chunk_idx]. The
	//! chunk-internal allocate_pooled is already thread-safe (per-flag
	//! atomic CAS on the bitmap), so multiple threads sharing the same
	//! chunk via TLS is correct, but performance is best when each
	//! thread has its own chunk (no inter-thread bitmap contention).
	//! Lifetime: claimed via the slow path; held until process exit
	//! (release_allocator is gated to skip thread-pinned chunks via
	//! m_thread_pinned_count).
	//! Type stored in s_chunks_of_type[]: chunks are of the inner-FS
	//! variant (PoolAllocator<ALIGN, DUMMY, DUMMY>), not necessarily the
	//! outer template (which may be FS=true while inner chunks are
	//! FS=false). Casting to the wrong type would mis-dispatch
	//! allocate_pooled().
	static ALLOC_TLS PoolAllocator<ALIGN, DUMMY, DUMMY> *s_my_chunk;
	//! Per-thread guard for THIS pool template's TLS state.  Each
	//! PoolAllocator template instantiation has its own thread_local
	//! `TlsGuard`, so the order in which thread_local destructors run
	//! does not matter: whichever fires first sets `s_alloc_tls_off`,
	//! and the rest are idempotent.  This removes the previous single
	//! point of failure (only `AllocPinCleanup::~dtor` set the flag).
	//!
	//! `allocate<>()` performs an ODR-use of `s_tls_guard` so the
	//! thread_local is initialised on this thread's first allocation,
	//! and its destructor is registered for thread exit.
	struct TlsGuard {
		~TlsGuard() noexcept {
			// Per-template cleanup, in case `AllocPinCleanup::~dtor`
			// runs after us or skips this template.  Idempotent — both
			// fields tolerate being re-cleared.
			if(s_my_chunk) {
				s_my_chunk->flush_owner_freelist();
				s_my_chunk = nullptr;
			}
			// Global allocator-off flag — read by `is_allocator_thread_active()`
			// from later (pthread_key) TLS dtors.
			s_alloc_tls_off = true;
		}
	};
	static thread_local TlsGuard s_tls_guard;
	//! # of threads pinning this chunk via TLS s_my_chunk. Incremented
	//! once per thread on first use; never decremented in steady state.
	//! release_allocator() returns false when this is non-zero so the
	//! chunk is not freed while any thread's TLS pointer references it.
	std::atomic<int> m_thread_pinned_count{0};

	//! Per-chunk owner-thread freelist (fixed-size FS=true chunks only —
	//! the FS=false specialization overrides `deallocate_pooled` to
	//! bypass this push because variable-size slots don't share a
	//! uniform size; for FS=false instances `m_freelist == nullptr`,
	//! `m_freelist_cap == 0`).  Only the thread whose TLS
	//! `s_my_chunk == this` pushes to / pops from these members, so
	//! they stay non-atomic.  Non-owner deallocs see `s_my_chunk !=
	//! this` (via the dealloc fast-path check at the top of
	//! `deallocate_pooled`) and skip straight to the bitmap CAS path.
	//! Owner check is `s_my_chunk == this` — `s_my_chunk` is
	//! `__thread` (gcc extension) on GCC/Clang per the `ALLOC_TLS`
	//! macro, i.e. a plain memory load, NOT the `_tlv_get_addr`
	//! runtime call macOS uses for `thread_local`.  Cleanup hook in
	//! `AllocPinCleanup` flushes residual entries on thread exit.
	//!
	//! Capacity is dynamic, scaled to ~10% of the chunk's total slot
	//! count clamped to `[FREELIST_CAP_MIN, FREELIST_CAP_MAX]`, so
	//! larger chunks absorb more same-thread alloc/dealloc cycles
	//! before hitting the bitmap CAS path.  The freelist array itself
	//! lives at the tail of the chunk-metadata malloc block; `create()`
	//! reserves the space and passes the pointer + cap into this ctor.
	enum {
		FREELIST_CAP_MIN = 32,
		FREELIST_CAP_MAX = 4096,
	};
	void ** const m_freelist;
	const int m_freelist_cap;
	int m_freelist_count = 0;

	void flush_owner_freelist() noexcept override;
	void flush_owner_freelist_to_bitmap() noexcept;
	void clear_owner_tls() noexcept override;

	void operator delete(void *) throw();
private:
	friend class PoolAllocatorBase;

	static PoolAllocator *create(size_t size, char *ppool);
};

//! Partially specialized class for variable-size allocators.
template <unsigned int ALIGN, bool DUMMY>
class PoolAllocator<ALIGN, false, DUMMY> : public PoolAllocator<ALIGN, true, false> {
public:
	void report_leaks();
	void report_statistics(size_t &chunk_size, size_t &used_size) override;
	typedef typename PoolAllocator<ALIGN, true, false>::FUINT FUINT;
protected:
	PoolAllocator(int count, char *addr, char *ppool,
	              void **freelist, int freelist_cap);
	inline void *allocate_pooled(unsigned int SIZE);
	bool deallocate_pooled(char *p) override;

private:
	friend class PoolAllocatorBase;
	template <unsigned int, bool, bool> friend class PoolAllocator;

	static PoolAllocator *create(size_t size, char *ppool);

	//! Cleared bit at the MSB indicates the end of the allocated area. \sa m_flags.
	FUINT * const m_sizes;
	unsigned int m_available_bits;
};

#define ALLOC_ALIGN1 (ALLOC_ALIGNMENT * 2)
#if defined __LP64__ || defined __LLP64__ || defined(_WIN64) || defined(__MINGW64__)
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
#define ALLOC_SIZE15 (ALLOC_ALIGNMENT * 15)
#define ALLOC_SIZE16 (ALLOC_ALIGNMENT * 16)

void* allocate_large_size_or_malloc(size_t size) throw();

#define ALLOCATE_9_16X(X, size) {\
	if(size <= ALLOC_SIZE16 * X) {\
		if(size <= ALLOC_SIZE9 * X)\
			return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE9 * X)>::allocate<ALLOC_SIZE9 * X>();\
		if(size <= ALLOC_SIZE10 * X)\
			return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE10 * X)>::allocate<ALLOC_SIZE10 * X>();\
		if(size <= ALLOC_SIZE11 * X)\
			return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE11 * X)>::allocate<ALLOC_SIZE11 * X>();\
		if(size <= ALLOC_SIZE12 * X)\
			return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE12 * X)>::allocate<ALLOC_SIZE12 * X>();\
		if(size <= ALLOC_SIZE13 * X)\
			return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE13 * X)>::allocate<ALLOC_SIZE13 * X>();\
		if(size <= ALLOC_SIZE14 * X)\
			return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE14 * X)>::allocate<ALLOC_SIZE14 * X>();\
		if(size <= ALLOC_SIZE15 * X)\
			return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE15 * X)>::allocate<ALLOC_SIZE15 * X>();\
		return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE16 * X)>::allocate<ALLOC_SIZE16 * X>();\
	}\
}

extern bool g_sys_image_loaded;
//! `s_alloc_tls_off` is forward-declared earlier in this file (around the
//! PoolAllocator class) so `PoolAllocator::TlsGuard` can reference it.

void activateAllocator();

inline void* new_redirected(std::size_t size) {
	//expecting a compile-time optimization because size is usually fixed to the object size.
    if( !g_sys_image_loaded || s_alloc_tls_off)
        return malloc(size); //Not to allocate dyld / post-TLS-cleanup objects.

	if(size <= ALLOC_SIZE1)
		return PoolAllocator<ALLOC_SIZE1, true>::allocate<ALLOC_SIZE1>();
	if(size <= ALLOC_SIZE2)
		return PoolAllocator<ALLOC_SIZE2, true>::allocate<ALLOC_SIZE2>();
	if(size <= ALLOC_SIZE3)
		return PoolAllocator<ALLOC_SIZE3, true>::allocate<ALLOC_SIZE3>();
	if(size <= ALLOC_SIZE4)
		return PoolAllocator<ALLOC_SIZE4, true>::allocate<ALLOC_SIZE4>();
	if(size <= ALLOC_SIZE8) {
		if(size <= ALLOC_SIZE5)
			return PoolAllocator<ALLOC_SIZE5, true>::allocate<ALLOC_SIZE5>();
		if(size <= ALLOC_SIZE6)
			return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE6)>::allocate<ALLOC_SIZE6>();
		if(size <= ALLOC_SIZE7)
			return PoolAllocator<ALLOC_SIZE7, true>::allocate<ALLOC_SIZE7>();
		return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE8)>::allocate<ALLOC_SIZE8>();
	}
	if(size <= ALLOC_SIZE16) {
		if(size <= ALLOC_SIZE9)
			return PoolAllocator<ALLOC_SIZE9, true>::allocate<ALLOC_SIZE9>();
		if(size <= ALLOC_SIZE10)
			return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE10)>::allocate<ALLOC_SIZE10>();
		if(size <= ALLOC_SIZE11)
			return PoolAllocator<ALLOC_SIZE11, true>::allocate<ALLOC_SIZE11>();
		if(size <= ALLOC_SIZE12)
			return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE12)>::allocate<ALLOC_SIZE12>();
		if(size <= ALLOC_SIZE13)
			return PoolAllocator<ALLOC_SIZE13, true>::allocate<ALLOC_SIZE13>();
		if(size <= ALLOC_SIZE14)
			return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE14)>::allocate<ALLOC_SIZE14>();
		if(size <= ALLOC_SIZE15)
			return PoolAllocator<ALLOC_SIZE15, true>::allocate<ALLOC_SIZE15>();
		return PoolAllocator<ALLOC_ALIGN(ALLOC_SIZE16)>::allocate<ALLOC_SIZE16>();
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

void release_pools();
void report_statistics();

#endif /* USE_STD_ALLOCATOR */

#endif /* ALLOCATOR_PRV_H_ */
