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
#include <type_traits>

// Portable atomic primitives for the custom pool allocator (formerly
// x86-only inline asm in atomic_prv_x86.h, then inline templates in
// allocator.cpp; hoisted here so header-inlined PoolAllocator member
// templates — `batch_clear_impl` etc. — can use them).  GCC/Clang
// __sync builtins work on every arch the pool supports.

//! Bit count / population count for 32bit.  Hoisted from allocator.cpp
//! so header-inlined FS=false bucket-freelist push can call it.
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

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value || std::is_pointer<T>::value, bool>::type
atomicCompareAndSet(T oldv, T newv, T *target) noexcept {
    return __sync_bool_compare_and_swap(target, oldv, newv);
}
template <typename T>
inline void atomicInc(T *target) noexcept {
    __sync_fetch_and_add(target, 1);
}
template <typename T>
inline void atomicDec(T *target) noexcept {
    __sync_fetch_and_sub(target, 1);
}
template <typename T>
inline bool atomicDecAndTest(T *target) noexcept {
    return __sync_sub_and_fetch(target, 1) == 0;
}

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
	//! Address-only chunk lookup.  Mirrors `deallocate_<>`'s s_chunks
	//! index calculation but stops at the chunk pointer — no
	//! `deallocate_pooled` call.  Returns nullptr if `p` does not
	//! belong to any pool chunk (or the chunk has been released).
	//! Used by `drain_thread_slot_freelists` to handle the case where
	//! `g_thread_slots[bucket].freelist_head` holds slots from
	//! multiple chunks of the same PoolType (e.g. FS=false buckets
	//! 6/8/10/12 share `PoolAllocator<32, false>`; a chunk transition
	//! triggered by one bucket leaves the others' `g_thread_chunks`
	//! entry stale, but the freelist may still receive both old- and
	//! new-chunk slots).
	static inline PoolAllocatorBase *lookup_chunk(void *p) noexcept;
	static void release_chunks();
	virtual void report_statistics(size_t &chunk_size, size_t &used_size) = 0;
	//! Null out this thread's `s_my_chunk` for this chunk's ALIGN type.
	//! Called from `AllocPinCleanup` after freelist flush, before pin
	//! count decrement.  Prevents stale `s_my_chunk` from pushing to
	//! a dead freelist when later TLS destructors (e.g.
	//! `RunnerCounterRegistration` via `pthread_key`) do heap
	//! alloc/dealloc after `AllocPinCleanup` has already run.
	virtual void clear_owner_tls() noexcept {}
	//! Batch return of `n` slots (all belonging to THIS chunk) to the
	//! bitmap.  Called by the cross-thread TLS batch flush — see
	//! `CrossDeallocBatch::flush` in allocator.cpp.  Each override
	//! groups slots by their m_flags[] word and clears all the bits
	//! in ONE CAS per word, vs the naïve per-slot CAS.  Pure virtual
	//! so every PoolAllocator instantiation supplies a concrete
	//! implementation.
	virtual void batch_return_to_bitmap(void **slot_ptrs, int n) noexcept = 0;
	//! Freelist-miss slow allocate.  Called from `new_redirected`'s
	//! cold path through this chunk's vtable; runs the bitmap-CAS /
	//! chunk-claim / create_allocator path with this template
	//! instantiation's compile-time ALIGN.  `bucket` is the table
	//! index of the freelist that missed, used to mirror an advanced
	//! `s_my_chunk` back into `g_thread_chunks[bucket]`.  Pure virtual
	//! so the dispatch is per-(ALIGN,FS) without a separate
	//! function-pointer table.
	virtual void *slow_allocate(unsigned bucket, std::size_t size) noexcept = 0;
protected:
	PoolAllocatorBase(char *ppool) : m_mempool(ppool) {}
	virtual bool deallocate_pooled(char *p) = 0;

	template <class ALLOC>
	static ALLOC *allocate_chunk();
	static void deallocate_chunk(int cidx, size_t chunk_size);

	//! A chunk, memory block.
	char * const m_mempool;
	//! Index into `s_chunks[]` for this chunk, and the chunk size used
	//! to address back into `s_mmapped_spaces`.  Set by `allocate_chunk`
	//! immediately after `s_chunks[cidx] = palloc`.  Read by the
	//! `batch_return_to_bitmap` chunk-release path (FS=true and FS=false
	//! overrides in allocator.cpp) so it can call `deallocate_chunk`
	//! after `release_allocator` returns true and before `delete this`
	//! — clearing `s_chunks[cidx]` and mprotect-ing the mempool back to
	//! PROT_NONE.  Without this, `s_chunks[cidx]` would dangle after the
	//! `delete this` self-suicide; a later thread's `deallocate_<>`
	//! lookup would dereference the freed PoolAllocator instance and
	//! glibc would surface it as `free(): invalid pointer` /
	//! `tcache_thread_shutdown(): unaligned tcache chunk detected`.
	int m_cidx = -1;
	size_t m_chunk_size = 0;

private:
	friend void report_statistics();
	enum {NUM_ALLOCATORS_IN_SPACE = ALLOC_MIN_MMAP_SIZE / ALLOC_MIN_CHUNK_SIZE,
		ALLOC_MAX_CHUNKS = NUM_ALLOCATORS_IN_SPACE * ALLOC_MAX_MMAP_ENTRIES};
	//! Swap spaces given by anonymous mmap().
	static char *s_mmapped_spaces[ALLOC_MAX_MMAP_ENTRIES];
	static PoolAllocatorBase *s_chunks[ALLOC_MAX_CHUNKS];
};

//! Per-thread flag — true once `AllocPinCleanup::~dtor` has fired.
//! Read by `new_redirected()` (and other allocator-TLS-aware code via
//! `is_allocator_thread_active()`) to fall back to malloc once the
//! pool-allocator TLS state is dead.  Defined in allocator.cpp.
extern ALLOC_TLS bool s_alloc_tls_off;

//! \brief Memory blocks in a unit of double-quad word
//! can be allocated from fixed-size or variable-size memory pools.
//! \tparam FS determines fixed-size or variable-size.
//! \sa allocator_test.cpp.
template <unsigned int ALIGN, bool FS = false, bool DUMMY = true>
class PoolAllocator : public PoolAllocatorBase {
public:
	//! Cold path: first-access chunk-claim + bitmap-CAS slow allocate.
	//! `[[gnu::always_inline]]` is retained so `bucket_first_access<B>`
	//! folds into a direct call to `allocate_chunk_path(SIZE)` per
	//! template instantiation, keeping SIZE compile-time inside the
	//! bitmap accounting in `allocate_pooled`.  The real hot path
	//! (owner-thread freelist pop) lives in `new_redirected` on the
	//! per-thread `AllocSlot`, not here.
	template <unsigned int SIZE>
	[[gnu::always_inline]] static void *allocate() noexcept {
		// `bucket_first_access<B>`'s hot path entry — only reached on
		// the very first allocation of (this thread, this bucket).  The
		// real hot path is `new_redirected` → AllocSlot freelist pop in
		// the header; this function just kicks the chunk-claim and the
		// bitmap CAS path.  Stays in allocator.cpp as a non-template
		// function (SIZE passed at runtime — only used inside
		// allocate_pooled's bitmap accounting; ALIGN/FS/DUMMY-specific
		// via the class).
		return allocate_chunk_path(SIZE);
	}
	//! Public accessor for the per-thread functor-table dispatcher
	//! (anon-namespace helpers in allocator.cpp).  Returns the
	//! currently-pinned chunk for this thread as a `PoolAllocatorBase*`
	//! so the dispatcher can cache it in `g_thread_slots[bucket].chunk`
	//! after `allocate_chunk_path` claimed a new one.
	static PoolAllocatorBase *get_pinned_chunk_base() noexcept {
		return static_cast<PoolAllocatorBase *>(s_my_chunk);
	}
	//! Public (was protected) so the per-thread functor-table dispatcher
	//! in allocator.cpp can call it on freelist miss without needing a
	//! friend declaration.  Tries `allocate_pooled` on the pinned chunk
	//! first, then the chunk-claim CAS loop, then `create_allocator` to
	//! mmap a new chunk.  Single function per (ALIGN, FS, DUMMY)
	//! instantiation — runtime SIZE arg, no per-SIZE explosion.
	static void *allocate_chunk_path(unsigned int SIZE);

	static void release_pools();
	void report_leaks();
	void report_statistics(size_t &chunk_size, size_t &used_size) override;

	typedef uintptr_t FUINT;
protected:
	PoolAllocator(int count, char *addr, char *ppool);
	inline void *allocate_pooled(unsigned int SIZE);
	bool deallocate_pooled(char *p) override;
	void batch_return_to_bitmap(void **slot_ptrs, int n) noexcept override;
	void *slow_allocate(unsigned bucket, std::size_t size) noexcept override;
	static bool create_allocator(int &aidx);
	static bool release_allocator(PoolAllocator *alloc);

	// === Cache line 0: owner-side hot reads & const fields.
	//! Every bit indicates occupancy in m_mempool.
	FUINT * const m_flags;
	//! A hint for searching in a chunk.
	int m_idx;
	const int m_count;
	int m_idx_of_type;

	// === Cache line 1+: cross-thread-written atomic counters.
	// `alignas(64)` on the first counter forces them onto a separate
	// cache line from the freelist + read-only members above, so an
	// `atomicInc/Dec` on `m_flags_nonzero_cnt` by another thread does
	// not invalidate the owner's freelist load/store cache line.
	alignas(64) int m_flags_nonzero_cnt;
	//! # of flags that having fully filled values.
	int m_flags_filled_cnt;

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
	// (removed: `thread_local TlsGuard s_tls_guard;` and its dtor.
	//  AllocPinCleanup::~AllocPinCleanup — fired via
	//  XThreadLocal<AllocPinCleanup>'s pthread_key dtor on thread exit —
	//  already drains the per-thread AllocSlot freelists, calls
	//  `clear_owner_tls()`, and sets `s_alloc_tls_off = true`, for
	//  every chunk this thread pinned.  The TlsGuard was redundant and
	//  its `(void)&s_tls_guard` ODR-use in `allocate<>()` emitted a
	//  per-template C++ thread_local init thunk call on every
	//  allocation (macOS arm64), with no observable correctness
	//  benefit.)
	//! # of threads pinning this chunk via TLS s_my_chunk. Incremented
	//! once per thread on first use; never decremented in steady state.
	//! release_allocator() returns false when this is non-zero so the
	//! chunk is not freed while any thread's TLS pointer references it.
	std::atomic<int> m_thread_pinned_count{0};

	void clear_owner_tls() noexcept override;


	//! Shared batched bitmap-clear skeleton (body in allocator.cpp).
	//! Parameterised on:
	//!   FetchSlot()       → `char *`  : next slot to clear (nullptr=end)
	//!   MaskFn(idx,sidx,p)→ `FUINT`   : bit-mask for one slot
	//!                                    (FS=true: 1 bit at sidx;
	//!                                    FS=false: N bits via m_sizes)
	//!   OnClearFn(oldv,newv)→ `void`  : per-word counter update
	//!
	//! Used by `batch_return_to_bitmap` (both FS=true and FS=false
	//! overrides).  Sole remaining caller now that the chunk-local
	//! freelist has been folded into AllocSlot.
	template <typename FetchSlot, typename MaskFn, typename OnClearFn>
	void batch_clear_impl(FetchSlot fetch_slot, MaskFn mask_fn,
	                     OnClearFn on_clear) noexcept;

protected:

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
	PoolAllocator(int count, char *addr, char *ppool);
	inline void *allocate_pooled(unsigned int SIZE);
	bool deallocate_pooled(char *p) override;
	void batch_return_to_bitmap(void **slot_ptrs, int n) noexcept override;
	void *slow_allocate(unsigned bucket, std::size_t size) noexcept override;

	// FS=false's previous per-chunk size-bucketed freelist
	// (m_fs_buckets, m_fs_bucket_count, fs_try_bucket_push,
	// fs_try_bucket_pop, FS_MAX_BUCKETS, FS_BUCKET_CAP, and the
	// flush_owner_freelist override) is removed: dealloc now pushes
	// to the per-thread AllocSlot freelist at
	// `g_thread_slots[bucket_for_size(N * ALIGN)]`, identically to
	// FS=true.  Allocations get a freelist hit via the inline pop in
	// `new_redirected` and never reach `allocate_pooled` on that
	// path.  Drain at thread exit sweeps `g_thread_slots[*]` and
	// routes slots through `tls_cross_dealloc_batch` →
	// `batch_return_to_bitmap`, whose FS=false override decodes N
	// from m_sizes and clears N bits per slot.
	// Saves 8 KiB per FS=false chunk (m_fs_buckets storage).

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
//! `s_alloc_tls_off` is forward-declared earlier in this file (just above
//! PoolAllocator) so new_redirected can read it.

void activateAllocator();

// ---------------------------------------------------------------------
// Per-thread allocation functor table (hot-path dispatch).
//
// Each AllocSlot owns the per-thread freelist for one size bucket.  The
// freelist is a LIFO linked list embedded in the free slots themselves:
// each free slot's first 8 bytes hold a `char *` pointer to the next
// free slot.
//
// Hot path: `new_redirected` inlines the freelist pop directly on the
// AllocSlot.  No indirect call on the freelist-hit path.  On miss, the
// slow path reads `g_thread_chunks[bucket]`; if non-null it dispatches
// through the chunk's vtable (`slow_allocate(bucket, size)`), which
// per-(ALIGN,FS) override runs the chunk-claim / bitmap CAS path.  If
// null, it falls through to `cold_first_access(bucket, size)` which
// handles activation-flag / cleanup-flag checks and the (rare)
// per-bucket first-access dispatch.
//
// sizeof(AllocSlot) == 8: a single `char *`, so `g_thread_slots[bucket]`
// indexing is a single shifted-load addressing-mode form
// `ldr x, [base, bucket, lsl #3]` — no separate slot-address computation
// needed.  8 slots share a 64-B cache line.  The chunk pointer lives in
// the parallel `g_thread_chunks[]` TLS array so the freelist-hit hot
// path touches only one cache line.
//
// State machine (encoded in `g_thread_chunks[bucket]`):
//   - `nullptr`: pre-activation OR pre-first-use OR post-cleanup.
//     Slow path goes to `cold_first_access`, which checks the
//     activation flag (`g_sys_image_loaded`) and the cleanup flag
//     (`s_alloc_tls_off`) and either returns `std::malloc(size)` or
//     dispatches per-bucket to `PoolAllocator<ALIGN,FS>::allocate<SIZE>()`
//     (which sets `g_thread_chunks[bucket]` as a side effect).
//   - non-null: steady state — `chunk->slow_allocate(bucket, size)`
//     virtual call updates `g_thread_chunks[bucket]` if `s_my_chunk`
//     has advanced to a new chunk after a fill.
//   - `AllocPinCleanup::~dtor` on thread exit clears all entries back
//     to `nullptr`, and the cleanup flag `s_alloc_tls_off` is set so
//     subsequent allocations route to `std::malloc`.
// ---------------------------------------------------------------------

struct AllocSlot {
	//! Owner-thread freelist head (LIFO).  Each free slot's first 8
	//! bytes hold the next pointer.  nullptr ⇒ empty: user data never
	//! appears on the freelist link (push always overwrites the slot's
	//! first 8 bytes with the previous head), so 0 unambiguously means
	//! "end of list".  Zero-initialised at static init.
	char *freelist_head;

	//! Owner-thread freelist push.  Single-writer (TLS pin), no atomics.
	void push(void *p) noexcept {
		*reinterpret_cast<char **>(p) = freelist_head;
		freelist_head = static_cast<char *>(p);
	}
	//! Owner-thread freelist pop.  Returns nullptr on empty;
	//! otherwise removes and returns the head slot.
	void *pop() noexcept {
		char *head = freelist_head;
		if(!head) return nullptr;
		freelist_head = *reinterpret_cast<char **>(head);
		return head;
	}
};
static_assert(sizeof(AllocSlot) == 8,
              "AllocSlot must stay 8 B — hot-path uses lsl #3 indexed addressing");

//! Bucket count.
//!   - index 0 (size = 0): reuses bucket 1's 16-B allocator
//!   - 1..16: sizes 16..256 in 16-B increments    (FS=true + FS=false mixed)
//!   - 17..24: sizes 288..512 in 32-B increments  (FS=false, was previously
//!             handled by new_redirected_large via ALLOCATE_9_16X(2, ...))
constexpr int ALLOC_NUM_BUCKETS = 25;

//! Size → bucket-index for both allocation (user size → bucket of the
//! smallest slot that fits) and deallocation (slot's actual size →
//! its bucket).  Both ranges happen to fit the same formula because
//! slot sizes coincide with the bucket boundary (the top of each
//! bucket's user-size range == the slot size for that bucket):
//!
//!   user_size 1..16   ↘
//!   slot_size = 16    ↗ → bucket 1
//!   user_size 17..32  ↘
//!   slot_size = 32    ↗ → bucket 2
//!   ...
//!   user_size 257..288  ↘
//!   slot_size = 288     ↗ → bucket 17
//!   ...
//!   user_size 481..512  ↘
//!   slot_size = 512     ↗ → bucket 24
//!
//! Sizes > ALLOC_MAX_BUCKETED_SIZE (= 512 = ALLOC_SIZE16*2) are not
//! covered; callers must check before indexing.
constexpr std::size_t ALLOC_MAX_BUCKETED_SIZE = (std::size_t)ALLOC_SIZE16 * 2u;

inline constexpr unsigned int bucket_for_size(std::size_t size) noexcept {
	// size ≤ 256: 16-B step slots.  (size+15)>>4 gives 1..16 for
	// size 1..256, and 0 for size==0 (reuses bucket 0's 16-B allocator).
	if(size <= (std::size_t)ALLOC_SIZE16)
		return static_cast<unsigned int>((size + 15u) >> 4);
	// 257..512: 32-B step slots (288, 320, ..., 512).  17 + ((size-257)>>5)
	// gives 17..24.  Works for both user size (rounding up to the
	// next slot size) and slot size (already a multiple of 32 above 256).
	return 17u + static_cast<unsigned int>((size - 257u) >> 5);
}

extern ALLOC_TLS AllocSlot g_thread_slots[ALLOC_NUM_BUCKETS];

//! Parallel TLS table holding each bucket's currently pinned chunk.
//! Initial value (and post-cleanup) is `nullptr`; the slow path in
//! `new_redirected` treats null as "first access on this (thread,
//! bucket)" — see `cold_first_access` in allocator.cpp.  Once
//! populated, the slow path dispatches through the chunk's vtable
//! (`PoolAllocatorBase::slow_allocate`), so this single parallel TLS
//! array carries both the state-machine state AND the dispatch
//! target — no separate function-pointer table is needed.
extern ALLOC_TLS PoolAllocatorBase *g_thread_chunks[ALLOC_NUM_BUCKETS];

// ---------------------------------------------------------------------
// Fast pthread-TSD bypass of the macOS TLV thunk.
//
// On macOS, C++ `__thread` / `thread_local` accesses lower to a TLV
// thunk: `adrp; add; ldr; blr tlv_get_addr` — roughly 10-15 cycles of
// dependent loads + a function call per access.  This block bypasses
// that for the two hottest TLS arrays:
//
//   1. `kame_tls_init_fast` (constructor priority 101) allocates two
//      `pthread_key_t`s, writes sentinel values into them via
//      `pthread_setspecific`, then scans the current pthread struct
//      (base = `kame_thread_pointer()`) byte-by-byte to find which
//      offsets received the sentinels.  Offsets stored in
//      `s_kame_slots_tsd_offset` / `s_kame_chunks_tsd_offset`.
//   2. Each thread's first allocation routes through the cold path
//      which writes `&g_thread_slots[0]` / `&g_thread_chunks[0]` (the
//      *per-thread* TLV-resolved addresses) into its own TSD slots.
//   3. Steady-state hot path reads `*(AllocSlot**)(TP + offset)` and
//      indexes `[bucket]`.  Two null checks — `offset != 0` (pre-init
//      guard) and `pointer != null` (per-thread first-touch guard) —
//      both predict not-taken with 100% accuracy after warmup.
//
// Linux note: on glibc/ELF, `__thread` for TUs linked into the main
// executable (the KAME case) uses the initial-exec TLS model — a
// single `mov %fs:OFFSET, %reg` instruction with no function call.
// The TLV-bypass optimisation solves a macOS-specific problem and
// adds no measurable win on Linux, while the sentinel-scan offset
// discovery is brittle (depends on glibc's `struct pthread` layout
// and on a key being placed in `specific_1stblock[]` rather than the
// dynamically allocated second-level array).  Empirically it caused
// glibc tcache corruption on x86_64 (`tcache_thread_shutdown():
// unaligned tcache chunk detected`).  Keep Linux on direct TLV.
//
// On unsupported platforms, the macros expand to direct
// `&g_thread_slots[0]` / `&g_thread_chunks[0]` references which keep
// the TLV thunk on the hot path.
// ---------------------------------------------------------------------

#if defined(__APPLE__) && (defined(__aarch64__) || defined(__x86_64__))
    #define KAME_FAST_TSD 1
#else
    #define KAME_FAST_TSD 0
#endif

#if KAME_FAST_TSD
//! Architecture-specific read of the thread-pointer register.  Returns
//! the base of the pthread struct (or TCB on glibc).  Used as the base
//! for the byte-offset TSD read.
//!
//! Important: `__builtin_thread_pointer()` on arm64 expands to
//! `mrs TPIDR_EL0`, which is the *read-write* register.  On macOS,
//! Apple's libc keeps the thread pointer in `TPIDRRO_EL0` (read-only)
//! and leaves `TPIDR_EL0` zero / unused — so the builtin returns
//! garbage there.  We always use explicit inline asm to read the
//! correct register per ABI.
static inline char *kame_thread_pointer() noexcept {
    #if defined(__APPLE__) && defined(__aarch64__)
        uintptr_t tp;
        __asm__ volatile("mrs %0, TPIDRRO_EL0" : "=r"(tp));
        return (char *)tp;
    #elif defined(__linux__) && defined(__aarch64__)
        uintptr_t tp;
        __asm__ volatile("mrs %0, TPIDR_EL0" : "=r"(tp));
        return (char *)tp;
    #elif defined(__APPLE__) && defined(__x86_64__)
        // macOS Intel: %gs:0 stores self-pointer == pthread struct base.
        uintptr_t tp;
        __asm__ volatile("movq %%gs:0, %0" : "=r"(tp));
        return (char *)tp;
    #elif defined(__linux__) && defined(__x86_64__)
        // Linux x86_64: %fs:0 stores self-pointer == TCB base.
        uintptr_t tp;
        __asm__ volatile("movq %%fs:0, %0" : "=r"(tp));
        return (char *)tp;
    #endif
}

//! Discovered TSD byte offsets.  Zero means "not yet initialised"
//! (constructor hasn't run, or `pthread_key_create` / sentinel scan
//! failed); hot path falls to TLV fallback in that case.
extern std::size_t s_kame_slots_tsd_offset;
extern std::size_t s_kame_chunks_tsd_offset;

//! Out-of-line cold paths invoked when either guard branch fails.
//! Defined in allocator.cpp.  Plant the per-thread TSD slot if
//! `s_kame_*_tsd_offset` is set, then return the TLV-resolved address.
//!
//! `[[clang::preserve_most]]`: caller-side register-spill avoidance.
//! Without it, clang must spill the live `size` and `bucket` regs (in
//! the caller-saved set) across the call, bloating `operator new`'s
//! prologue with 4-6 reg saves.  preserve_most shifts the burden into
//! the cold callee (cheap — cold).
[[clang::preserve_most]] AllocSlot *kame_slots_cold() noexcept;
[[clang::preserve_most]] PoolAllocatorBase **kame_chunks_cold() noexcept;

//! Hot accessor: returns the base of this thread's `g_thread_slots[]`.
//! Inlined into `new_redirected` and `deallocate_pooled`.
inline AllocSlot *kame_slots_base() noexcept {
    std::size_t off = s_kame_slots_tsd_offset;
    if(__builtin_expect(off != 0, 1)) {
        AllocSlot *p =
            *reinterpret_cast<AllocSlot **>(kame_thread_pointer() + off);
        if(__builtin_expect(p != nullptr, 1)) return p;
    }
    return kame_slots_cold();
}
//! Hot accessor: returns the base of this thread's `g_thread_chunks[]`.
inline PoolAllocatorBase **kame_chunks_base() noexcept {
    std::size_t off = s_kame_chunks_tsd_offset;
    if(__builtin_expect(off != 0, 1)) {
        PoolAllocatorBase **p =
            *reinterpret_cast<PoolAllocatorBase ***>(kame_thread_pointer() + off);
        if(__builtin_expect(p != nullptr, 1)) return p;
    }
    return kame_chunks_cold();
}
#else  // !KAME_FAST_TSD: fall back to direct TLV access
inline AllocSlot *kame_slots_base() noexcept { return &g_thread_slots[0]; }
inline PoolAllocatorBase **kame_chunks_base() noexcept { return &g_thread_chunks[0]; }
#endif

//! Cold slow path: invoked when `g_thread_chunks[bucket] == nullptr`
//! (first access on this (thread, bucket), or post-cleanup).  Handles
//! activation-flag / cleanup-flag checks, then dispatches per bucket
//! to the matching `PoolAllocator<ALIGN,FS>::allocate<SIZE>()`.
//! Defined in allocator.cpp; declared here so `new_redirected` can
//! tail-call it.
void *cold_first_access(unsigned bucket, std::size_t size) noexcept;

//! Out-of-line path for sizes larger than the table covers (> 256 B).
//! Handles activation-flag check + the existing if-chain for the
//! `ALLOCATE_9_16X(2, size)` range and the malloc fallback for very
//! large sizes.  Hot path (size ≤ 256 B) bypasses this entirely.
void *new_redirected_large(std::size_t size) noexcept;

inline void *new_redirected(std::size_t size) {
	// Hot path: sizes ≤ 256.  One branch + the inline `(size+15)>>4`
	// formula (the small-range half of `bucket_for_size`).  Larger
	// sizes go to `new_redirected_large`, which uses the full
	// `bucket_for_size` helper for its own 257..512 dispatch.
	if(size > (std::size_t)ALLOC_SIZE16)
		return new_redirected_large(size);
	unsigned int bucket = (static_cast<unsigned int>(size) + 15u) >> 4;
	// Fast TSD access if available (macOS arm64/x86_64, Linux
	// arm64/x86_64); else direct TLV access.
	AllocSlot &slot = kame_slots_base()[bucket];
	// Inline freelist pop — no indirect call on hit path.  Empty
	// sentinel: nullptr (push only ever writes the previous head into
	// the slot's first 8 bytes, so user data is never on the link).
	char *head = slot.freelist_head;
	if(head) {
		slot.freelist_head = *reinterpret_cast<char **>(head);
		return head;
	}
	// Freelist miss — dispatch through the pinned chunk's vtable if
	// the bucket has been activated on this thread, otherwise fall
	// to `cold_first_access` for the (rare) first-time path + the
	// pre-activation / post-cleanup malloc fallbacks.
	if(PoolAllocatorBase *chunk = kame_chunks_base()[bucket])
		return chunk->slow_allocate(bucket, size);
	return cold_first_access(bucket, size);
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
