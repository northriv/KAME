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
	//! Batch return of `n` slots (all belonging to THIS chunk) to the
	//! bitmap.  Called by the cross-thread TLS batch flush — see
	//! `CrossDeallocBatch::flush` in allocator.cpp.  Each override
	//! groups slots by their m_flags[] word and clears all the bits
	//! in ONE CAS per word, vs the naïve per-slot CAS.  Pure virtual
	//! so every PoolAllocator instantiation supplies a concrete
	//! implementation.
	virtual void batch_return_to_bitmap(void **slot_ptrs, int n) noexcept = 0;
protected:
	PoolAllocatorBase(char *ppool) : m_mempool(ppool) {}
	virtual bool deallocate_pooled(char *p) = 0;

	template <class ALLOC>
	static ALLOC *allocate_chunk();
	static void deallocate_chunk(int cidx, size_t chunk_size);

	//! A chunk, memory block.
	char * const m_mempool;

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
	//! Hot path: owner-thread freelist pop.  `[[gnu::always_inline]]`
	//! ensures clang ignores its size-based inlining heuristic so this
	//! inlines into `operator new` (cross-TU via the header), removing
	//! the per-allocate call boundary.  Freelist miss falls through to
	//! the out-of-line `allocate_chunk_path` (allocate_pooled +
	//! chunk-claim + create_allocator).
	//!
	//! SIZE is a template parameter so the slot stride used by
	//! `try_owner_freelist_pop` (= ALIGN) and the bitmap path inside
	//! `allocate_chunk_path` can be constant-folded; the freelist hit
	//! path itself uses ALIGN (not SIZE) to compute the slot address,
	//! so it's identical for all SIZE values that share an ALIGN bucket.
	template <unsigned int SIZE>
	[[gnu::always_inline]] static void *allocate() noexcept {
		// Single-instruction fast path on Apple Silicon: TLS load
		// `s_my_chunk` (one `mrs tpidr_el0` + offset), null-check, and
		// (if non-null) inlined `try_owner_freelist_pop` (5 instructions:
		// load m_freelist_curpos, compare with m_freelist, decrement,
		// load uint16_t, multiply-add for slot address).
		if(PoolAllocator<ALIGN, DUMMY, DUMMY> *my = s_my_chunk) {
			if(void *p = my->try_owner_freelist_pop())
				return p;
		}
		// Cold path — pinned-chunk bitmap CAS, chunk-claim loop, chunk
		// creation.  Stays in allocator.cpp as a non-template function
		// (SIZE passed at runtime — only used inside allocate_pooled's
		// bitmap accounting; ALIGN/FS/DUMMY-specific via the class).
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
	static bool create_allocator(int &aidx);
	static bool release_allocator(PoolAllocator *alloc);

	// === Cache line 0: owner-side hot reads & const fields.
	// Owner-thread freelist is a LIFO linked list **embedded in the
	// free slots themselves** — each free slot's first 8 bytes hold a
	// `char *` pointer to the next free slot, or the empty-sentinel
	// `(char *)this` when this slot is the list's tail.  No separate
	// uint16_t-index array, no cap, no slot-count limit: every freed
	// owner-thread slot lands on the list.
	//
	// `m_freelist_head`:
	//   - empty: equal to `(char *)this` (= the chunk metadata
	//     address, which can never be a valid slot pointer because
	//     slots live at `this + SLOT_POOL_OFFSET` and beyond);
	//   - non-empty: pointer to the current head free slot.  The
	//     slot's first 8 bytes hold the next pointer.
	//
	// On pop the critical path is just:
	//   ldr  head, [this,#head]
	//   cmp  head, this        ; sentinel check
	//   b.eq slow
	//   ldr  next, [head]      ; uses head as base; one dependent load
	//   str  next, [this,#head]
	//   mov  ret, head         ; return value ready right after first ldr
	// — no per-template `s_my_chunk` TLV thunk, no `m_mempool` field
	// load, no `m_freelist_end` field load.  Critical path to return:
	// ldr head → ret (4 cycles), vs the uint16_t-index variant's
	// ldr curpos → ldrh idx → add mempool+idx<<N (~10 cycles).
	char *m_freelist_head;
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
	//  already calls `chunk->flush_owner_freelist()`, `clear_owner_tls()`,
	//  and sets `s_alloc_tls_off = true`, for every chunk this thread
	//  pinned.  The TlsGuard was redundant and its `(void)&s_tls_guard`
	//  ODR-use in `allocate<>()` emitted a per-template C++ thread_local
	//  init thunk call on every allocation (macOS arm64), with no
	//  observable correctness benefit.)
	//! # of threads pinning this chunk via TLS s_my_chunk. Incremented
	//! once per thread on first use; never decremented in steady state.
	//! release_allocator() returns false when this is non-zero so the
	//! chunk is not freed while any thread's TLS pointer references it.
	std::atomic<int> m_thread_pinned_count{0};

	//! Per-chunk owner-thread freelist (FS=true chunks only — FS=false's
	//! `deallocate_pooled` override uses bucket-based push instead).
	//! Single-writer (the thread whose TLS `s_my_chunk == this`);
	//! non-owners see `s_my_chunk != this` and route to the cross-thread
	//! TLS batch.  Cleanup hook in `AllocPinCleanup` flushes residual
	//! entries on thread exit.  No cap, no slot-count limit: every
	//! freed owner-thread slot lands on the embedded linked list.

	void flush_owner_freelist() noexcept override;
	void flush_owner_freelist_to_bitmap() noexcept;
	void clear_owner_tls() noexcept override;

	//! Shared batched bitmap-clear skeleton (body in allocator.cpp).
	//! Parameterised on:
	//!   FetchSlot()       → `char *`  : next slot to clear (nullptr=end)
	//!   MaskFn(idx,sidx,p)→ `FUINT`   : bit-mask for one slot
	//!                                    (FS=true: 1 bit at sidx;
	//!                                    FS=false: N bits via m_sizes)
	//!   OnClearFn(oldv,newv)→ `void`  : per-word counter update
	//!
	//! Used by `flush_owner_freelist_to_bitmap` and
	//! `batch_return_to_bitmap` (both FS=true and FS=false overrides).
	template <typename FetchSlot, typename MaskFn, typename OnClearFn>
	void batch_clear_impl(FetchSlot fetch_slot, MaskFn mask_fn,
	                     OnClearFn on_clear) noexcept;

	//! Owner-thread freelist push.  Inlined here in the header — hot
	//! dealloc path.  Embeds the previous head pointer into the freed
	//! slot's first 8 bytes (overwriting whatever user data lived
	//! there; the slot is now free, user no longer owns it).  No cap
	//! check, no size-class limit — every owner-thread dealloc lands
	//! on the list.  Single-writer (TLS pin), so no atomics.
	bool try_owner_freelist_push(void *p) noexcept {
		*reinterpret_cast<char **>(p) = m_freelist_head;
		m_freelist_head = static_cast<char *>(p);
		return true;
	}
public:
	//! Owner-thread freelist pop.  Inlined here in the header — hottest
	//! alloc path.  Critical path is `ldr head, [this,#head] / cmp head,
	//! this / b.eq slow / ldr next,[head] / str next,[this,#head]`
	//! — head is the return value, ready ~4 cycles after entry.  Empty
	//! sentinel is `(char *)this` (the chunk metadata address — never a
	//! valid slot pointer, no field load needed).
	//! Public so the per-thread functor-table dispatcher (anon-namespace
	//! helpers in allocator.cpp) can call it without a friend declaration.
	void *try_owner_freelist_pop() noexcept {
		char *head = m_freelist_head;
		if(head == reinterpret_cast<char *>(this)) return nullptr;
		m_freelist_head = *reinterpret_cast<char **>(head);
		return head;
	}
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
// AllocSlot.  No indirect call on the freelist-hit path — the slow
// `alloc_fn` (in the parallel `g_thread_alloc_fn[]` TLS array) is only
// called on miss (freelist empty + slow chunk-bitmap path).
//
// sizeof(AllocSlot) == 16: keeping it a power-of-two lets the compiler
// turn `g_thread_slots[bucket]` indexing into a single `ldr x, [base,
// offset]` where `offset = (size+15) & ~0xf` (the byte offset is
// already what `bucket_for_size`'s shift-by-4 computes, just unshifted).
// At 16 B/slot, 4 slots share a 64-B cache line, so adjacent-bucket
// allocations share an L1d fetch.  `alloc_fn` is split into a separate
// TLS array because (a) it isn't on the hot path — only called on
// freelist miss — and (b) moving it out is what unlocks the 16-byte
// size and the consequent shift-free indexing.
//
// State machine (encoded in `g_thread_alloc_fn[bucket]`):
//   - Initial value (pre-activation OR before this thread's first use
//     of this bucket): `&bucket_first_access<Bucket>`.
//     That function checks `g_sys_image_loaded`; if false returns
//     `std::malloc(size)`; if true claims a chunk, stores it in
//     `slot.chunk`, and rewrites `g_thread_alloc_fn[Bucket]` to
//     `&bucket_steady_alloc<Bucket>` before tail-calling it.
//   - Steady state: `&bucket_steady_alloc<Bucket>` — bitmap-CAS path
//     for freelist misses.  Hot-path freelist pop is *not* here; it's
//     in `new_redirected` directly.
//   - After `AllocPinCleanup::~dtor` on thread exit: all entries reset
//     to `&malloc_fallback`, freelists drained back to the bitmap via
//     the cross-thread batch path.
// ---------------------------------------------------------------------

struct AllocSlot {
	//! Owner-thread freelist head (LIFO).  Each free slot's first 8
	//! bytes hold the next pointer.  nullptr ⇒ empty: user data never
	//! appears on the freelist link (push always overwrites the slot's
	//! first 8 bytes with the previous head), so 0 unambiguously means
	//! "end of list".  Zero-initialised at static init.
	char *freelist_head;
	//! Currently pinned chunk for this (thread, bucket).  Used by the
	//! slow path and by `deallocate_pooled` for the owner check
	//! (`slot.chunk == this` ⇒ push to slot's freelist; otherwise
	//! cross-thread batch).
	PoolAllocatorBase *chunk;

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
static_assert(sizeof(AllocSlot) == 16,
              "AllocSlot must stay 16 B — indexing assumes pow-of-two stride");

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

//! Parallel TLS table holding each bucket's freelist-miss slow path
//! (`bucket_first_access<B>` initially, `bucket_steady_alloc<B>` after
//! first activation, `malloc_fallback` after thread-exit cleanup).
//! Indexed by bucket — same index as `g_thread_slots`.  Not on the
//! freelist-hit hot path; lives outside AllocSlot so the latter stays
//! 16 B for shift-free indexing in `new_redirected`.
extern ALLOC_TLS void *(*g_thread_alloc_fn[ALLOC_NUM_BUCKETS])
    (AllocSlot *slot, std::size_t size) noexcept;

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
	AllocSlot &slot = g_thread_slots[bucket];
	// Inline freelist pop — no indirect call on hit path.  Empty
	// sentinel: nullptr (push only ever writes the previous head into
	// the slot's first 8 bytes, so user data is never on the link).
	char *head = slot.freelist_head;
	if(head) {
		slot.freelist_head = *reinterpret_cast<char **>(head);
		return head;
	}
	// Freelist miss — bucket-specific slow path (first_access /
	// steady_alloc / malloc_fallback), looked up in the parallel
	// TLS table (kept out of AllocSlot to keep the latter 16 B).
	return g_thread_alloc_fn[bucket](&slot, size);
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
