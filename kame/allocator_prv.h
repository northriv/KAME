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
	PoolAllocator(int count, char *addr, char *ppool,
	              uint16_t *fs_buckets);
	inline void *allocate_pooled(unsigned int SIZE);
	bool deallocate_pooled(char *p) override;
	void batch_return_to_bitmap(void **slot_ptrs, int n) noexcept override;
	void flush_owner_freelist() noexcept override;

	//! Size-bucketed owner-thread freelist for variable-size FS=false
	//! chunks.  Bucket index = N - 1 where N is the slot size in ALIGN
	//! units (1 ≤ N ≤ FS_MAX_BUCKETS = 16).  Each bucket is a uint16_t
	//! slot-index LIFO of FS_BUCKET_CAP entries.  Single-writer (owner
	//! thread), no atomics.
	//!
	//! Why per-size buckets: FS=false slots vary in size within a
	//! chunk, so a single LIFO can't satisfy a size-N pop by returning
	//! a top entry of size M ≠ N (would partial-allocate, leaking the
	//! rest).  Per-size buckets give O(1) push and O(1) pop with
	//! guaranteed size match.  m_sizes encoding is preserved (the
	//! popped slot already has the correct N-bit pattern at its
	//! sidx, since we re-allocate with the same N).
	//!
	//! Coverage: bucket array covers N = 1 to FS_MAX_BUCKETS = 16,
	//! which is the full unit range for ALIGN1=32 chunks (sizes
	//! 32 B – 512 B inclusive).  ALIGN2=256 chunks use units 1–8 in
	//! the inline dispatch (256 B – 2 KiB) and 9+ via ALLOCATE_9_16X
	//! (2 KiB+); the 9–16 range is covered, 17+ falls through to the
	//! cross-thread TLS batch.
	//!
	//! Memory: 16 × 256 × 2 B = 8 KiB per FS=false chunk.
	static constexpr int FS_MAX_BUCKETS = 16;
	static constexpr int FS_BUCKET_CAP = 256;

	//! Try-push slot to size-matched bucket.  Owner-only.  Returns
	//! true if pushed (slot retained in owner-side freelist); false
	//! if N is out of bucket range or the bucket is at cap (caller
	//! routes to TLS cross-dealloc batch).
	bool fs_try_bucket_push(char *p) noexcept {
		unsigned slot_idx = static_cast<unsigned>(
		    (p - this->m_mempool) / ALIGN);
		unsigned idx = slot_idx / (sizeof(FUINT) * 8);
		unsigned sidx = slot_idx % (sizeof(FUINT) * 8);
		FUINT nones = find_zero_forward(m_sizes[idx] >> sidx);
		FUINT slot_mask = nones | (nones - 1u);
		unsigned N = count_bits(slot_mask);
		if(N < 1 || N > (unsigned)FS_MAX_BUCKETS) return false;
		unsigned b = N - 1;
		if(m_fs_bucket_count[b] >= (uint16_t)FS_BUCKET_CAP) return false;
		m_fs_buckets[(size_t)b * FS_BUCKET_CAP + m_fs_bucket_count[b]++] =
		    static_cast<uint16_t>(slot_idx);
		return true;
	}
	//! Try-pop slot from size-matched bucket.  Owner-only.  Returns
	//! nullptr if bucket is empty; caller falls through to the
	//! bitmap allocate path.  m_sizes at the popped slot is already
	//! correct (the slot was alloc'd with the same N, pushed without
	//! clearing m_sizes, and we re-alloc with the same N).
	void *fs_try_bucket_pop(unsigned N) noexcept {
		if(N < 1 || N > (unsigned)FS_MAX_BUCKETS) return nullptr;
		unsigned b = N - 1;
		if(m_fs_bucket_count[b] == 0) return nullptr;
		uint32_t slot_idx =
		    m_fs_buckets[(size_t)b * FS_BUCKET_CAP + --m_fs_bucket_count[b]];
		return this->m_mempool + static_cast<size_t>(slot_idx) * ALIGN;
	}

private:
	friend class PoolAllocatorBase;
	template <unsigned int, bool, bool> friend class PoolAllocator;

	static PoolAllocator *create(size_t size, char *ppool);

	//! Cleared bit at the MSB indicates the end of the allocated area. \sa m_flags.
	FUINT * const m_sizes;
	unsigned int m_available_bits;

	//! Bucket storage and per-bucket counts.  Storage lives at the
	//! tail of the chunk-metadata malloc block; pointer is set in
	//! `create()`.  Counts are inline (small fixed array).
	uint16_t * const m_fs_buckets;
	uint16_t m_fs_bucket_count[FS_MAX_BUCKETS];
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
// The if-chain over size + activation-flag checks that `new_redirected`
// previously did on every allocation is replaced with:
//   1. table lookup indexed by 16-B size bucket (`size_t -> 1..16`)
//   2. indirect call through the slot's `alloc_fn`.
//
// State is encoded *in the slot's function pointer*:
//   - Initial value (pre-activation OR before this thread's first use
//     of this bucket): `&bucket_first_access<Bucket>`.
//     That function checks `g_sys_image_loaded`; if false returns
//     `std::malloc(size)`, if true claims a chunk for this thread,
//     stores it in `slot.chunk`, and rewrites `slot.alloc_fn` to
//     `&bucket_steady_alloc<Bucket>` before tail-calling it.
//   - Steady state: `&bucket_steady_alloc<Bucket>` — direct freelist
//     pop on `slot.chunk` (no TLS reads, no activation check).
//   - After `AllocPinCleanup::~dtor` on thread exit: all slots reset
//     to `&malloc_fallback` so any subsequent allocation by a later
//     TLS destructor on this thread routes safely to `std::malloc`.
//
// The whole table fits in one cache line per ~4 buckets (each slot is
// 16 B on 64-bit) and is a single `__thread` symbol — 1 TLV thunk per
// allocation on macOS arm64 instead of the per-template `s_my_chunk`
// thunk used by the old `allocate<SIZE>()` path.  Not XThreadLocal —
// the table's lifetime must extend past every TLS destructor on this
// thread, which XThreadLocal cannot guarantee (its `dtor_` frees the
// underlying `T`, leaving the `cached` thread_local pointer dangling).
// ---------------------------------------------------------------------

struct AllocSlot {
	PoolAllocatorBase *chunk;
	void *(*alloc_fn)(PoolAllocatorBase *chunk, std::size_t size) noexcept;
};

//! Bucket count: index 0 (size = 0) reuses bucket 1's 16-B allocator;
//! indices 1..16 cover sizes 16..256 in 16-B increments.
constexpr int ALLOC_NUM_BUCKETS = 17;

extern ALLOC_TLS AllocSlot g_thread_slots[ALLOC_NUM_BUCKETS];

//! Out-of-line path for sizes larger than the table covers (> 256 B).
//! Handles activation-flag check + the existing if-chain for the
//! `ALLOCATE_9_16X(2, size)` range and the malloc fallback for very
//! large sizes.  Hot path (size ≤ 256 B) bypasses this entirely.
void *new_redirected_large(std::size_t size) noexcept;

inline void *new_redirected(std::size_t size) {
	if(size <= ALLOC_SIZE16) {
		// Bucket: 0 for size=0 (handled the same as bucket 1 — its
		// alloc_fn slot also points to the 16-B allocator);
		// (size+15)>>4 gives 1..16 for size in 1..256.
		unsigned int bucket = (static_cast<unsigned int>(size) + 15u) >> 4;
		const AllocSlot &slot = g_thread_slots[bucket];
		return slot.alloc_fn(slot.chunk, size);
	}
	return new_redirected_large(size);
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
