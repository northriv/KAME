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
#include <cstdint>
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
//! Atomic fetch-and-AND.  Returns the OLD value (before AND) so the
//! caller can compute the resulting bit pattern.  Used by Phase 5j
//! BIT_OWNED clear to detect "I brought m_flags_packed to 0" → I'm
//! the unique releaser.
template <typename T>
inline T atomicFetchAnd(T *target, T value) noexcept {
    return __sync_fetch_and_and(target, value);
}

#if defined(__GNUC__) || defined(__clang__)
	#define ALLOC_TLS __thread //TLS for allocations, could be better for NUMA.
#else
	#define ALLOC_TLS thread_local
#endif

//! Allocation unit (1 chunk = N × this).  Phase 5l: every mmap region is
//! a uniform 32 MiB block carved into 128 fixed-size 256 KiB "units".
//! A chunk = 1, 2, or 4 contiguous units depending on the per-template
//! `CHUNK_UNITS` (= 1 for ALIGN < 256, = 2 for ALIGN < 1024, = 4 for
//! ALIGN ≥ 1024 = 1024).  The unit size matches the previous Phase 5g
//! minimum so cross-thread chunk residency under lazy commit stays
//! tight; the buddy approach replaces the previous 2× growth ladder.
//!
//! O(1) chunk_base lookup from any slot uses `s_back_offset[]` (1 byte
//! per unit, per region) — see `PoolAllocatorBase::s_back_offset` below.
//! `back_offset[u] = u - base_u` for a unit `u` claimed as part of a
//! chunk whose base is at `base_u`; reader does `base_u = u -
//! back_offset[u]` and then `chunk_base = region + base_u * 256K`.
#define ALLOC_MIN_CHUNK_SIZE (1024 * 256) //256 KiB unit
//! log2(ALLOC_MIN_CHUNK_SIZE) — used for fast unit-index extraction
//! `unit_idx = pdiff >> ALLOC_MIN_CHUNK_SHIFT`.  Compile-time constant.
#define ALLOC_MIN_CHUNK_SHIFT 18  //log2(256 KiB)
//! Max chunk = 4 units = 1 MiB (CHUNK_UNITS_MAX × ALLOC_MIN_CHUNK_SIZE).
//! All allocations of size ≤ ALLOC_MAX_BUCKETED_SIZE (16376 B) fit within
//! a 4-unit (1 MiB) chunk's slot region (= 1 MiB − 64 B = 1048512 B).
#define ALLOC_MAX_CHUNK_UNITS 4
#define ALLOC_MAX_CHUNK_SIZE (ALLOC_MIN_CHUNK_SIZE * ALLOC_MAX_CHUNK_UNITS)
// OS page size — relevant for the `madvise()` granularity on chunk
// release.  All chunk sizes are multiples of ALLOC_MIN_CHUNK_SIZE
// (= 256 KiB) which auto-satisfies every supported arch's page size.
#if defined(__APPLE__) && defined(__aarch64__)
    #define ALLOC_PAGE_SIZE 16384  // 16 KiB
#elif defined(__powerpc64__) || defined(__POWERPC__)
    #define ALLOC_PAGE_SIZE 65536  // 64 KiB
#else
    #define ALLOC_PAGE_SIZE 4096   // 4 KiB
#endif
//! Phase 5l: regions are uniform 32 MiB — no ladder, no growth.  The
//! Phase 5g growth-cap macro `GROW_CHUNK_SIZE` is removed; chunk size
//! is now a per-template constant (`PoolAllocator<...>::CHUNK_SIZE`).
//! `NUM_ALLOCATORS_IN_SPACE == 128` matches the bit count of the per-
//! region claim bitmap (BitmapWord × BITMAP_WORDS_PER_REGION ×
//! CHUNKS_PER_BITMAP_WORD = 128 unit slots).  Every region is 32 MiB =
//! 128 × 256 KiB regardless of host word size.
//!
//! `ALLOC_MAX_MMAP_ENTRIES` is the VA cap — each region is mmap'd
//! `PROT_READ | PROT_WRITE` upfront (Phase 5l switches the release path
//! from `mprotect(PROT_NONE)` to `madvise(MADV_FREE/DONTNEED)` so
//! reclaim is RSS-cheap without protection toggling).  Total
//! reservation = 32 MiB × N entries.
//!
//!   host         |  N  | total VA reserved
//!   -------------+-----+--------------------
//!   64-bit       |  96 | 3 GiB
//!   32-bit       |   5 | 160 MiB (= 32 MiB × 5, leaves headroom
//!                |     |  under Linux 3 GiB / Win32 2 GiB user VA)
//!   Windows      |  96 | (same as 64-bit; pool path opt-in via dylib)
//!
//! 96 was chosen to preserve Phase 5g's effective pool capacity (≈3 GiB)
//! after the ladder→uniform refactor.  Under the previous 2× ladder,
//! 24 entries × up-to-128-MiB region (= chunk_size × 128 with cap) gave
//! ≈3 GiB across the ladder; multi-template sharing per region was
//! impossible (each ladder level was a different chunk_size).  Phase
//! 5l unifies all templates into the same uniform regions, so the
//! adversarial cross-thread workload (200 thr × 32 conc × 50 % cross)
//! that previously spread ≈1500 chunks across 5+ ladder levels now
//! competes for the same set of regions — capacity-tight at 24 × 32 MiB.
#define ALLOC_MIN_MMAP_SIZE (1024 * 1024 * 32) //32 MiB = 256 KiB × 128
#if defined __LP64__ || defined __LLP64__ || defined(_WIN64) || defined(__MINGW64__)
    #define ALLOC_MAX_MMAP_ENTRIES 96 //96 × 32 MiB = 3 GiB total pool VA
#else
    //! 32-bit host: 5 regions = 160 MiB.  Largest chunk = 1 MiB; the
    //! pool's max bucket is 16 KiB so 1 MiB chunks hold 64+ slots — far
    //! more than enough.
    #define ALLOC_MAX_MMAP_ENTRIES 5
#endif

//! Reserved bytes at the head of every chunk.  Layout (Phase 5d):
//!   [ 0 ..  7]: chunk-wide SIZE info — `uint64_t`:
//!                 FS=true  (fixed-size chunk): low 32 bits = slot
//!                          size in bytes (= ALIGN; same for every
//!                          slot in the chunk).  Non-zero ⇒ "jump
//!                          straight to bucket-driven dispatch
//!                          without a per-slot header read".
//!                 FS=false (variable-size): 0.  Distinct from FS=true
//!                          and from chunk-released (palloc==0); the
//!                          dealloc path reads the per-slot
//!                          `{bucket, SIZE}` header at `p - 8`
//!                          instead (Phase 5d "borrow" scheme).
//!               High 32 bits: ALIGN (always — for non-templated
//!               dispatchers; see `chunk_header_size_info()`).
//!   [ 8 .. 15]: `PoolAllocatorBase *` palloc (chunk owner).
//!   [16 .. 23]: `DeallocateFn` — non-virtual static trampoline
//!               (per-template) that dispatches the dealloc body.
//!   [24 .. 31]: `SizeOfFn` — slot-size lookup trampoline.
//!               FS=false: reads SIZE from the per-slot
//!               `{bucket, SIZE}` header at `p - 8` (Phase 5d).
//!   [32 .. 55]: pad.
//!   [56 .. 63]: RESERVED for FS=false slot 0's `{uint32_t bucket,
//!               uint32_t SIZE}` header (Phase 5d-3 "borrow" scheme
//!               formalisation).
//!               The slot at bit 0 of m_flags[0] (= the slot whose
//!               p_user == mempool == chunk_base + ALLOC_CHUNK_HEADER)
//!               has no predecessor whose last 8 B can host its
//!               header; this 8-byte tail of the chunk-header pad is
//!               its dedicated home.
//!               `allocate_pooled` writes here uniformly via
//!               `slot_start - 8` (= chunk_base + 56) without any
//!               special-case branch — the address math reduces
//!               naturally.  For FS=true chunks this 8 B is just
//!               unused pad (FS=true has no per-slot header).
//! Slot region (`m_mempool`) starts at `chunk_base + ALLOC_CHUNK_HEADER`.
//!
//! TODO (Phase 5d-4 candidate): the [0..7] SIZE info enables a
//! "unified deallocate" that branches on `(hdr[0] != 0)` instead of
//! the indirect `DeallocateFn` call.  The high-32-b ALIGN is
//! already available, so FS=false's `p - 8` header read needs no
//! extra per-template dispatch.
#define ALLOC_CHUNK_HEADER 64
#define ALLOC_CHUNK_HEADER_SIZE_INFO_OFFSET     0   // [ 0.. 7]: chunk SIZE info
#define ALLOC_CHUNK_HEADER_PALLOC_OFFSET        8   // [ 8..15]: palloc
#define ALLOC_CHUNK_HEADER_FN_OFFSET           16   // [16..23]: DeallocateFn
#define ALLOC_CHUNK_HEADER_SIZEOF_FN_OFFSET    24   // [24..31]: SizeOfFn
// [56..63] = slot-0 header (= ALLOC_CHUNK_HEADER - 8).  No constant
// needed — `allocate_pooled` reaches it via the uniform
// `slot_start - 8` math (slot 0's slot_start == m_mempool ==
// chunk_base + ALLOC_CHUNK_HEADER).
static_assert(ALLOC_CHUNK_HEADER >= ALLOC_CHUNK_HEADER_SIZEOF_FN_OFFSET + 8 + 8,
              "chunk header must have >= 8 B of pad between SizeOfFn "
              "and the slot-0 reservation at chunk_header[-8..-1].");

#define ALLOC_ALIGNMENT 16 //bytes, not 8 but 16 for compatibility
#define ALLOC_MAX_CHUNKS_OF_TYPE \
	(ALLOC_MIN_MMAP_SIZE / ALLOC_MIN_CHUNK_SIZE * ALLOC_MAX_MMAP_ENTRIES)

class PoolAllocatorBase;
//! Cross-dealloc batch entry — paired chunk + slot pointers.  Defined
//! here so `PoolAllocatorBase::batch_return_to_bitmap` and
//! `CrossDeallocBatch::buf[]` (in allocator.cpp) share the exact same
//! layout — no per-chunk slot-pointer copy on flush, no SoA/AoS
//! translation; `batch_return_to_bitmap` reads `entries[k].slot`
//! directly from the caller's buffer.
//!
//! `CrossDeallocBatch` keeps a sentinel `{nullptr, nullptr}` entry at
//! the position one past the live count, so the chunk-side walker
//! only needs `while(entries[k].chunk == this)` — no `k < n_max` test
//! in the inner loop.  Trailing sentinel is invariant by flush
//! contract (any non-trivial chunk pointer `this` differs from
//! nullptr, so the walk always terminates at the boundary).
struct CrossDeallocEntry {
	PoolAllocatorBase *chunk;
	void              *slot;
};

class PoolAllocatorBase {
public:
	//! Signature of the per-chunk dealloc trampoline stored in the
	//! chunk header at offset `ALLOC_CHUNK_HEADER_FN_OFFSET` (= 8).
	//! Set by `allocate_chunk` to `&PoolAllocator<ALIGN,FS,DUMMY>::
	//! deallocate_pooled_static`, a non-virtual static wrapper that
	//! casts `base` to the bound derived type and calls its inline
	//! `deallocate_pooled_impl`.  Replaces vtable dispatch on the
	//! `deallocate_<>` hot path: 1 load (function pointer, same cache
	//! line as `palloc`) + 1 indirect branch, vs. the previous
	//! 2 loads (vtable + slot) + 1 indirect branch.  Saving on macOS
	//! arm64 with cache-hot vtable: ~1-2 cycles per dealloc; on
	//! NUMA / cache-cold vtable: more.
	using DeallocateFn = bool (*)(PoolAllocatorBase *base, char *slot);

	//! Signature of the per-chunk slot-size trampoline stored at chunk-
	//! header offset `ALLOC_CHUNK_HEADER_SIZEOF_FN_OFFSET` (= 16).
	//! Used by `pool_slot_size(p)` / `realloc()` to recover the size of
	//! an allocated slot without a vtable call.  Per-(ALIGN,FS,DUMMY)
	//! instantiation:
	//!   - FS=true   → return ALIGN (compile-time constant)
	//!   - FS=false  → decode N from `m_sizes[idx]>>sidx`, return N*ALIGN
	using SizeOfFn = std::size_t (*)(PoolAllocatorBase *base, char *slot);

	virtual ~PoolAllocatorBase() = default;
	//! Phase 5l: regions are uniform 32 MiB, so `deallocate_<>` no longer
	//! needs the per-level compile-time CHUNK_SIZE template parameter.
	//! Collapsed to a single non-template function with a runtime
	//! region-walk loop — eliminates the 24/96-level template recursion
	//! that previously generated one inlined copy of the body per
	//! ladder level (icache bloat scaling with ALLOC_MAX_MMAP_ENTRIES).
	static inline bool deallocate(void *p);
	//! Look up the slot size (bytes) for a pointer.  Returns 0 if `p`
	//! is not a pool slot (foreign / libsystem-malloc'd / null).  Uses
	//! the same chunk-header pattern as `deallocate` and dispatches
	//! the slot-size lookup through the chunk's `SizeOfFn`.
	static inline std::size_t size_of(void *p);
	//! Address-only chunk lookup.  Returns nullptr if `p` does not
	//! belong to any pool chunk (or the chunk has been released).
	//! Used by `drain_thread_slot_freelists` to handle the case where
	//! `g_thread_slots[bucket].freelist_head` holds slots from multiple
	//! chunks of the same PoolType (e.g. FS=false sizes 96/128/160/192
	//! all share `PoolAllocator<32, false>`; a chunk transition triggered
	//! by one bucket leaves the others' `g_thread_chunks[]` entry stale,
	//! but the freelist may still receive both old- and new-chunk slots
	//! through the shared `s_my_chunk == this` owner check).  Implemented
	//! as a for-loop walk of `s_mmapped_spaces[]` — each region is a
	//! uniform 32 MiB (Phase 5l), and `s_back_offset[]` maps any
	//! claimed unit back to its chunk base in O(1).
	static inline PoolAllocatorBase *lookup_chunk(void *p) noexcept;
	//! Total live chunks across all regions, summed from
	//! `s_claim_bitmap[]` (popcount of set bits).  Diagnostic probe for
	//! tests that want to verify release paths actually fire — leak in
	//! the chunk-release path would show as monotonic growth across
	//! repeated alloc/free cycles.  Relaxed loads (rare path, hint
	//! only; the snapshot races against concurrent
	//! claim / release CAS but each bit is consistent at the moment of
	//! its read).
	static int count_live_chunks() noexcept;
	//! Null out this thread's `s_my_chunk` for this chunk's ALIGN type.
	//! Called from `AllocThreadExitCleanup` after freelist flush, before pin
	//! count decrement.  Prevents stale `s_my_chunk` from pushing to
	//! a dead freelist when later TLS destructors (e.g.
	//! `RunnerCounterRegistration` via `pthread_key`) do heap
	//! alloc/dealloc after `AllocThreadExitCleanup` has already run.
	virtual void clear_owner_tls() noexcept {}
	//! Batch return of a contiguous run of CrossDeallocEntries whose
	//! `chunk == this` to the bitmap.  Each override walks
	//! `entries[k]` while `entries[k].chunk == this` (terminating on
	//! the trailing `{nullptr, nullptr}` sentinel or the next chunk's
	//! group), merges adjacent same-m_flags-word slots into one CAS
	//! per word, and returns the number of entries it consumed so the
	//! caller can advance past them.  Pure virtual — each
	//! `PoolAllocator<ALIGN, FS, DUMMY>` supplies its own ALIGN /
	//! per-FS-variant counter logic.
	//!
	//! Caller contract:
	//!   * `entries[0].chunk == this` on entry (else returns 0);
	//!   * `entries[k].chunk` for k ≥ count past the buffer is
	//!     `nullptr` (the sentinel) so the inner loop's chunk
	//!     comparison terminates without needing a count test.
	virtual int batch_return_to_bitmap(
	    const CrossDeallocEntry *entries) noexcept = 0;

	//! Adaptive holding hint: last `batch_return_to_bitmap` call's
	//! coalescing factor for this chunk, in fixed-point ×16
	//! (16 = 1.0× = no coalescing benefit, 24 = 1.5× = 33 % CAS
	//! saved, 32 = 2.0× = 50 % saved, etc.).  Updated `relaxed` on
	//! each batch — it's a hint, not authoritative; races are
	//! benign (next push reads slightly stale value, no
	//! correctness impact).  Read by `CrossDeallocBatch::
	//! push_direct` to decide adaptively whether to hold (route to
	//! the per-thread holding buf for further coalescing
	//! accumulation) or dispatch immediately.  Epsilon-greedy
	//! explore in the caller occasionally force-holds regardless,
	//! so a chunk whose factor dropped below threshold can be
	//! re-evaluated.
	std::atomic<uint8_t> m_last_coalesce_x16{16};
	//! Freelist-miss slow allocate.  Called from `new_redirected`'s
	//! cold path through this chunk's vtable; runs the bitmap-CAS /
	//! chunk-claim / create_allocator path with this template
	//! instantiation's compile-time ALIGN.  `bucket` is the table
	//! index of the freelist that missed, used to mirror an advanced
	//! `s_my_chunk` back into `g_thread_chunks[bucket]`.  Pure virtual
	//! so the dispatch is per-(ALIGN,FS) without a separate
	//! function-pointer table.
	virtual void *slow_allocate(unsigned bucket, std::size_t size) noexcept = 0;
	//! Public read-only accessor for `m_chunk_size` — used by
	//! anonymous-namespace helpers (e.g. `drain_thread_slot_freelists`)
	//! that need to compute `chunk_base = head & ~(chunk_size - 1)`
	//! without a per-template dispatch (the helper iterates all
	//! buckets and slots from all chunk-template instantiations may
	//! coexist on its freelists).  Returns the chunk-size stamped by
	//! `allocate_chunk()` at chunk-claim time.
	std::size_t chunk_size() const noexcept { return m_chunk_size; }
protected:
	PoolAllocatorBase(char *ppool) : m_mempool(ppool) {}
	virtual bool deallocate_pooled(char *p) = 0;

	template <class ALLOC>
	static ALLOC *allocate_chunk();
	//! Release a chunk back to PROT_NONE.  Clears the chunk header
	//! pointer at `chunk_base`, mprotect's the chunk back to PROT_NONE,
	//! and clears the matching claim bit in `s_claim_bitmap[]` (region
	//! + bit located via a walk over `s_mmapped_spaces[]`).  Called
	//! both from the owner-side `deallocate_<>` last-slot release path
	//! and from the cross-batch `batch_return_to_bitmap` suicide path.
	static void deallocate_chunk(char *chunk_base, size_t chunk_size);

	//! A chunk, memory block.
	char * const m_mempool;

	//! Chunk size for this PoolAllocator instance.  Stamped by
	//! `allocate_chunk()` from the per-level ladder value.  Read by
	//! the cross-batch `batch_return_to_bitmap` chunk-release path
	//! (FS=true and FS=false overrides) so it can call
	//! `deallocate_chunk(chunk_base, chunk_size)` after `cross_release`
	//! returns true and BEFORE the `delete this` self-suicide cascade
	//! — clearing the chunk-header pointer + claim bit, and
	//! mprotect-ing the mempool back to PROT_NONE.
	//! The owner-side dealloc path returns `true` from
	//! `deallocate_pooled` and `PoolAllocatorBase::deallocate_<>`
	//! calls `deallocate_chunk(chunk_base, CHUNK_SIZE)` directly
	//! with the template's compile-time chunk_size.
	size_t m_chunk_size = 0;

public:
	enum {NUM_ALLOCATORS_IN_SPACE = ALLOC_MIN_MMAP_SIZE / ALLOC_MIN_CHUNK_SIZE};
	static_assert(NUM_ALLOCATORS_IN_SPACE == 128,
		"NUM_ALLOCATORS_IN_SPACE expected to be 128 — total bit count "
		"of the per-region chunk-claim bitmap "
		"(BitmapWord × BITMAP_WORDS_PER_REGION).");
	//! Word type for `s_claim_bitmap[]`.  Picked per-target so the
	//! chunk-claim CAS remains genuinely lock-free:
	//!   * Hosts where `atomic<uint64_t>` is always-lock-free
	//!     (`ATOMIC_LLONG_LOCK_FREE == 2`) — 64-bit hosts, and 32-bit
	//!     hosts with hardware DCAS (x86 CMPXCHG8B / ARMv7 LDREXD):
	//!     `uint64_t`, 4 words per region.
	//!   * Hosts without DCAS — fall back to `uint32_t`, 8 words per
	//!     region.  Requires single-word atomic<uint32_t> to be
	//!     always-lock-free (true on every architecture this allocator
	//!     supports).
	//!
	//! Phase 5l: 2-bit encoding per UNIT (not per chunk).  Each 256-KiB
	//! unit owns a pair of bits (claim at bit 2N, ready at bit 2N+1):
	//!   * (0, 0): free — unit is unclaimed.
	//!   * (1, 0): claimed as a CONTINUATION unit of a multi-unit chunk
	//!     (no chunk_header here — header lives at the BASE unit at
	//!     `base_idx = unit_idx - s_back_offset[…+unit_idx]`).  Also
	//!     used transiently for a BASE unit between claim CAS and
	//!     post-init ready-bit publish.
	//!   * (1, 1): claimed as the BASE unit of a chunk, chunk_header
	//!     fully initialised and dereferenceable.
	//!   * (0, 1): impossible (would mean "ready but not claimed").
	//!
	//! For multi-unit chunks (CHUNK_UNITS = 2 or 4), the base unit and
	//! all continuation units carry claim=1 in a single atomic CAS at
	//! claim time; only the base unit ever gets ready=1.  `back_offset`
	//! tells readers which unit holds the header for any claimed unit.
	//! Total bits per region: 128 units × 2 = 256.
#if ATOMIC_LLONG_LOCK_FREE == 2 && !defined(KAME_FORCE_UINT32_BITMAP)
	using BitmapWord = uint64_t;
	static constexpr int BITMAP_WORDS_PER_REGION = 4;
#else
	using BitmapWord = uint32_t;
	static constexpr int BITMAP_WORDS_PER_REGION = 8;
	static_assert(ATOMIC_INT_LOCK_FREE == 2,
		"atomic<uint32_t> must be always-lock-free as the fallback "
		"bitmap word type — targets without 32-bit atomic CAS are "
		"not supported.");
#endif
	static constexpr int BITS_PER_BITMAP_WORD = int(sizeof(BitmapWord) * 8);
	static constexpr int CHUNKS_PER_BITMAP_WORD =
	    BITS_PER_BITMAP_WORD / 2;     // 2 bits per chunk
	static_assert(CHUNKS_PER_BITMAP_WORD * BITMAP_WORDS_PER_REGION
	                 == NUM_ALLOCATORS_IN_SPACE,
		"bitmap layout (2 bits per chunk) must cover all 128 chunks per region");
	//! Mask: bit 2N set ⇒ "claimed" bit for chunk N.  Used to test
	//! claimed-only or count live chunks.  0x5555... on 64-bit,
	//! 0x55555555 on 32-bit.
	static constexpr BitmapWord CLAIM_BITS_MASK =
	    (BITS_PER_BITMAP_WORD == 64)
	        ? static_cast<BitmapWord>(0x5555555555555555ULL)
	        : static_cast<BitmapWord>(0x55555555u);
	//! Mask: bit 2N+1 set ⇒ "ready" bit for chunk N.  0xAAAA... pair.
	static constexpr BitmapWord READY_BITS_MASK =
	    static_cast<BitmapWord>(CLAIM_BITS_MASK << 1);
private:
	//! Swap spaces given by anonymous mmap().
	static char *s_mmapped_spaces[ALLOC_MAX_MMAP_ENTRIES];
	//! Per-mmap-region per-unit claim/ready bitmap.  Each 256-KiB unit
	//! owns two bits (claim at bit 2N, ready at bit 2N+1).  Word size
	//! selected at compile time (see `BitmapWord` above) so the claim-
	//! CAS is genuinely lock-free on every supported target — `uint64_t`
	//! on 64-bit hosts and 32-bit hosts with DCAS, `uint32_t` on
	//! 32-bit hosts without DCAS.  Total bits per region stays 256
	//! (= 128 units × 2 bits).
	//!
	//! Lookups go via chunk-header dereference (see `deallocate_<>` /
	//! `lookup_chunk`), so this array is only consulted on the cold
	//! chunk-claim / release paths.  Total storage on 64-bit:
	//! 24 × 4 × 8 B = 768 B.
	static std::atomic<BitmapWord>
	    s_claim_bitmap[ALLOC_MAX_MMAP_ENTRIES * BITMAP_WORDS_PER_REGION];

public:
	//! Phase 5l: per-region back-offset table.  One byte per unit:
	//!     s_back_offset[region * 128 + u] = u - base_u
	//! where `base_u` is the unit index of the chunk whose claim covers
	//! `u`.  For single-unit chunks the entry is 0; for the base unit of
	//! a multi-unit chunk the entry is 0; for continuation units the
	//! entry is 1, 2, or 3 (max CHUNK_UNITS = 4).
	//!
	//! Written by `allocate_chunk` BEFORE the claim-bit CAS (with
	//! `writeBarrier` between), so any reader that observes
	//! `claim_bit == 1` (acquire-load) is guaranteed to see the
	//! current claimer's back_offset value.  Cleared back to 0 by
	//! `deallocate_chunk` after the release sequence; an uninitialised
	//! entry reads 0 from BSS, matching "single-unit chunk at this
	//! position" — a benign default since the first claim overwrites it.
	//!
	//! No atomic on the back_offset slot itself (plain byte read/write):
	//! the bitmap claim-bit CAS supplies the synchronisation point.
	//! Total size: 24 × 128 = 3072 B on 64-bit (5 × 128 = 640 B on 32-bit).
	static uint8_t s_back_offset[ALLOC_MAX_MMAP_ENTRIES * NUM_ALLOCATORS_IN_SPACE];
};

//! Per-thread flag — true once `AllocThreadExitCleanup::~dtor` has fired.
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
	//! Phase 5l buddy tiers.  Larger ALIGN gets larger chunks so the
	//! per-chunk slot count stays in a healthy range (>= 32 slots per
	//! chunk for the largest ALIGN3 bucket).  Multi-unit chunks are
	//! laid out in `s_mmapped_spaces` at unit-aligned positions; the
	//! chunk-claim CAS sets CHUNK_UNITS contiguous claim bits in one
	//! atomic op, and `s_back_offset[]` records the back-offset of each
	//! continuation unit so any slot pointer resolves to its chunk base
	//! in O(1) regardless of which unit it falls in.
	static constexpr unsigned int CHUNK_UNITS =
	    (ALIGN < 256u) ? 1u :
	    (ALIGN < 1024u) ? 2u :
	                      4u;
	static constexpr size_t CHUNK_SIZE = (size_t)CHUNK_UNITS * (size_t)ALLOC_MIN_CHUNK_SIZE;
	static_assert(CHUNK_UNITS <= ALLOC_MAX_CHUNK_UNITS,
	    "CHUNK_UNITS must fit within ALLOC_MAX_CHUNK_UNITS");

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

	//! Non-virtual static trampoline for the chunk-header fn pointer.
	//! `allocate_chunk` stamps the chunk header (offset
	//! `ALLOC_CHUNK_HEADER_FN_OFFSET`) with `&deallocate_pooled_static`;
	//! `deallocate_<>` reads that pointer and dispatches directly via
	//! `fn(palloc, p)` — bypassing the vtable lookup that the virtual
	//! `deallocate_pooled` override would require.  The body just
	//! down-casts `base` to this template instantiation and invokes
	//! the non-virtual qualified-name call
	//! `self->PoolAllocator::deallocate_pooled(p)`, which compiles to
	//! a direct branch.
	static bool deallocate_pooled_static(PoolAllocatorBase *base, char *p);

	//! Non-virtual static trampoline for the chunk-header `SizeOfFn`
	//! pointer.  FS=true returns the constant ALIGN — every slot in a
	//! fixed-size chunk has the same length.  The FS=false partial
	//! specialisation overrides this to read SIZE from the per-slot
	//! prefix at `p - ALIGN` (Phase 5c).
	static std::size_t size_of_static(PoolAllocatorBase * /*base*/,
	                                  char * /*p*/) noexcept {
	    return ALIGN;
	}

	//! Value written to chunk_base + ALLOC_CHUNK_HEADER_SIZE_INFO_OFFSET
	//! by `allocate_chunk()` (Phase 5c).  Layout:
	//!   * Low 32 bits = FS-distinguishing "slot SIZE":
	//!       FS=true  : ALIGN (slot size; non-zero ⇒ fixed-size chunk,
	//!                  dispatcher can derive the bucket directly).
	//!       FS=false : 0     (signal: read SIZE from per-slot prefix
	//!                  at `p - ALIGN`).
	//!   * High 32 bits = ALIGN (always — for FS=false dispatchers
	//!       that need to convert user pointer → slot_start = p - ALIGN
	//!       without dispatching through a per-template hook, e.g.
	//!       `drain_thread_slot_freelists`).
	//! Picked per-template so the chunk header carries the right
	//! discriminator + the ALIGN needed for prefix-based dispatch.
	static constexpr std::uint64_t chunk_header_size_info() noexcept {
	    return static_cast<std::uint64_t>(ALIGN)
	         | (static_cast<std::uint64_t>(ALIGN) << 32);
	}

	typedef uintptr_t FUINT;
protected:
	PoolAllocator(int count, char *addr, char *ppool);
	inline void *allocate_pooled(unsigned int SIZE);
	bool deallocate_pooled(char *p) override;
	int batch_return_to_bitmap(const CrossDeallocEntry *entries) noexcept override;
	void *slow_allocate(unsigned bucket, std::size_t size) noexcept override;
	//! Mmap a fresh chunk and register it in `s_chunks_of_type[]` for
	//! diagnostic enumeration only (`release_pools` / `report_statistics`).
	//! Mmap a fresh chunk for the current thread.  Phase 4b: no global
	//! registry — the per-thread DLL is the sole source of truth for
	//! "chunks this thread can allocate from".  Called from
	//! `allocate_chunk_path`'s slow path when the DLL scan finds no
	//! reusable chunk.  Returns a fresh chunk pointer (not in any
	//! thread's DLL yet — caller is responsible for appending) or
	//! throws `std::bad_alloc` on mmap failure.
	static PoolAllocator<ALIGN, DUMMY, DUMMY> *create_allocator();
	//! Owner-driven release of a chunk this thread owns (DLL member).
	//! Atomically claims `BIT_RELEASED` on `m_flags_packed`.  Returns
	//! true ⇒ caller must unlink from DLL + `delete palloc` +
	//! `PoolAllocatorBase::deallocate_chunk(cbase, csz)`.
	//! Returns false if the chunk is not actually empty (count > 0),
	//! `BIT_RELEASED` was already set, or this thread's DLL has fewer
	//! than `LEAVE_VACANT_CHUNKS_PER_THREAD` chunks (floor — avoid
	//! thrashing on bursty workloads).
	//!
	//! Phase 4b-final: `BIT_RELEASED` on the packed word is the
	//! single serialisation point across all release paths (owner-
	//! driven neighbour release, cross-thread last-slot release,
	//! thread-exit cleanup) — exactly one CAS wins, the winner owns
	//! the cleanup.  No global registry, no bit-0-lock CAS.
	//!
	//! `cross_release` is the cross-thread variant: additionally
	//! gates on `BIT_OWNER_EXITED == 1` (only the owning thread's
	//! exit-path or its own slow-path may release while owner is
	//! alive).  No DLL traversal — the cross-thread caller has no
	//! access to the owner's DLL.
	static bool owner_release(PoolAllocator *palloc);
	static bool cross_release(PoolAllocator *palloc);
	//! Per-thread DLL teardown for thread-exit cleanup.  Called from
	//! `AllocThreadExitCleanup::~dtor` once per (ALIGN, FS) template the
	//! thread has touched.  Walks the per-thread DLL with cached-next:
	//! for each chunk, either claims `BIT_RELEASED` (if empty — release
	//! it) or sets `BIT_OWNER_EXITED` (if non-empty — signal cross-
	//! thread frees to release on the eventual dec-to-zero).  Clears
	//! the per-thread `s_dll_head` / `s_dll_tail` / `s_my_chunk` slots
	//! before the walk so a stale read by a TLS dtor running afterwards
	//! cannot route into a released chunk.
	static void release_dll_chunks_for_thread() noexcept;

	// === Cache line 0: owner-side hot reads & const fields.
	//! Every bit indicates occupancy in m_mempool.
	FUINT * const m_flags;
	//! A hint for searching in a chunk.
	int m_idx;
	const int m_count;

	// === Cache line 1+: cross-thread-written atomic counters.
	// `alignas(64)` on the first counter forces them onto a separate
	// cache line from the freelist + read-only members above, so an
	// `atomicInc/Dec` on `m_flags_packed` by another thread does not
	// invalidate the owner's freelist load/store cache line.
	//
	// Packed counter + state bits (Phase 5j — inverts the old
	// BIT_OWNER_EXITED → BIT_OWNED and drops BIT_RELEASED):
	//   * Bits  0..30 — nonzero-flag-word count.  Max value =
	//     `m_count` ≤ ALLOC_CHUNK_SIZE / ALIGN / 64 ≈ 16 K
	//     even at ALIGN=16 / chunk-size=1 MiB; 31 bits (= 2 G) is
	//     comfortably over-provisioned.
	//   * Bit  31     — `BIT_OWNED`: set when owner thread is alive
	//     and holds this chunk in its DLL.  Cleared atomically by
	//     `release_dll_chunks_for_thread` / `owner_release` at owner
	//     exit / neighbour-release.  Inverted semantics from the old
	//     `BIT_OWNER_EXITED` so the dec-to-zero CAS uniquely
	//     identifies the releaser without a separate `BIT_RELEASED`.
	//
	// Release identification (no BIT_RELEASED needed):
	//   - Cross-thread free brings MASK_CNT → 0 via atomicDecAndTest.
	//     If returns true, m_flags_packed is now 0 → BIT_OWNED was
	//     CLEAR (owner is gone) AND MASK_CNT was 1 → I'm the unique
	//     releaser.  If returns false (BIT_OWNED still set, or
	//     MASK_CNT was > 1), I'm not.
	//   - Owner exit calls atomicFetchAnd(&m_flags_packed, ~BIT_OWNED).
	//     If `old & ~BIT_OWNED == 0` (= MASK_CNT was 0), owner is the
	//     unique releaser.  Else cross-thread will release on its next
	//     dec-to-zero.
	//   - Owner_release (Phase 4a empty-neighbour) uses the same
	//     atomicFetchAnd: chunks observed-empty in our DLL are
	//     released by us via the AND → newv == 0 check.
	//
	// The two operations (dec-to-0 vs AND-clear-BIT_OWNED) are
	// mutually exclusive because exactly one transitions m_flags_packed
	// to all-zero.
	//
	// Bit 30 is intentionally left unused (previously BIT_RELEASED);
	// available for future ABA-counter / additional state if needed.
	static constexpr uint32_t MASK_CNT  = 0x7FFFFFFFu; // bits 0..30
	static constexpr uint32_t BIT_OWNED = 0x80000000u; // bit 31
	alignas(64) uint32_t m_flags_packed;
	//! # of flags that having fully filled values.
	int m_flags_filled_cnt;

	//! Per-thread "currently owned" chunk for fast-path allocate().
	//! When non-null, `allocate<SIZE>()` calls
	//! `s_my_chunk->allocate_pooled()` directly without scanning the
	//! DLL or mmaping fresh.  Type uses `<ALIGN, DUMMY, DUMMY>` so
	//! FS=true and FS=false partial specs share the same TLS slot.
	//! Lifetime: set on chunk-claim success; cleared in
	//! `release_dll_chunks_for_thread` at thread exit.
	static ALLOC_TLS PoolAllocator<ALIGN, DUMMY, DUMMY> *s_my_chunk;
	//! Per-thread, per-template DLL head / tail.  Phase 4b: this is now
	//! the sole source of truth for "chunks this thread can allocate
	//! from".  Chunks are appended to the tail on `allocate_chunk_path`
	//! mmap-fresh success and unlinked on `owner_release` success
	//! (Phase 4a's chunk-full-trigger neighbour release) or on
	//! `release_dll_chunks_for_thread` (thread exit).  Single-writer
	//! (this thread); no atomic ops needed on the DLL pointers.
	static ALLOC_TLS PoolAllocator<ALIGN, DUMMY, DUMMY> *s_dll_head;
	static ALLOC_TLS PoolAllocator<ALIGN, DUMMY, DUMMY> *s_dll_tail;

	//! Per-thread DLL pointers.  Single-writer (the owning thread)
	//! and single-reader (same thread).  No atomic ordering needed
	//! for these two fields in steady state.
	//!
	//! Type uses the same `<ALIGN, DUMMY, DUMMY>` erasure trick as
	//! `s_my_chunk` / `s_dll_head` so FS=true and FS=false partial
	//! specialisations all link through identically-typed pointers
	//! — the FS=false partial spec inherits `m_dll_prev/next` from
	//! the `<ALIGN, true, false>` base, whose stored type then
	//! aligns with the per-thread DLL head/tail above.
	PoolAllocator<ALIGN, DUMMY, DUMMY> *m_dll_prev{nullptr};
	PoolAllocator<ALIGN, DUMMY, DUMMY> *m_dll_next{nullptr};

	// Phase 4b: the previous `std::atomic<bool> m_owner_exited` lives
	// here as `BIT_OWNER_EXITED` inside `m_flags_packed` (above).
	// Packing it together with the count lets the cross-thread
	// last-slot-returner observe both the dec-to-zero transition AND
	// the owner-gone state on one word, with no extra atomic load.

	void clear_owner_tls() noexcept override;


	//! Shared bitmap-clear skeleton (body in allocator.cpp).  Walks
	//! `entries[k]` while `entries[k].chunk == this`, terminating on
	//! the next chunk's group or the trailing `{nullptr, nullptr}`
	//! sentinel.  Returns the number of entries it consumed.
	//! Parameterised on:
	//!   MaskFn(idx,sidx,p)→ `FUINT`   : bit-mask for one slot
	//!                                    (FS=true: 1 bit at sidx;
	//!                                    FS=false: N+1 bits via the
	//!                                    per-slot prefix SIZE — Phase
	//!                                    5c.  `p` is `slot_start` (=
	//!                                    `p_user - ALIGN`), `sidx` is
	//!                                    the prefix bit position.)
	//!   OnClearFn(oldv,newv)→ `void`  : per-word counter update
	//!
	//! Precondition: the chunk run is sorted by ascending slot pointer
	//! address (== m_flags word index order).  `CrossDeallocBatch::
	//! flush` enforces the sort; the post-teardown and drain paths
	//! call with a single-entry run (trivially sorted).
	//!
	//! Used by `batch_return_to_bitmap` (both FS=true and FS=false
	//! overrides).  Sole remaining caller now that the chunk-local
	//! freelist has been folded into AllocSlot.
	template <typename MaskFn, typename OnClearFn>
	int batch_clear_impl(const CrossDeallocEntry *entries,
	                     MaskFn mask_fn, OnClearFn on_clear) noexcept;

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
	//! See `PoolAllocator<ALIGN, FS, DUMMY>::deallocate_pooled_static`.
	//! FS=false partial spec provides its own trampoline that down-
	//! casts to this leaf type and invokes the non-virtual
	//! `PoolAllocator<ALIGN, false, DUMMY>::deallocate_pooled`.
	static bool deallocate_pooled_static(PoolAllocatorBase *base, char *p);
	//! FS=false slot-size lookup (Phase 5c).  Reads SIZE from the
	//! per-slot prefix at `p - ALIGN`.  Returns the user-requested
	//! size in bytes.  Stamped into the chunk header at offset
	//! `ALLOC_CHUNK_HEADER_SIZEOF_FN_OFFSET`; overrides the FS=true
	//! constant returned by the base template's `size_of_static`.
	static std::size_t size_of_static(PoolAllocatorBase *base, char *p) noexcept;
	//! FS=false chunk-header SIZE info (Phase 5c).  Low 32 bits = 0 so
	//! dispatchers can distinguish FS=false chunks (and read the per-
	//! slot prefix at `p - ALIGN` instead of treating header[0..7] as
	//! the slot size).  High 32 bits = ALIGN so non-templated callers
	//! (e.g. `drain_thread_slot_freelists`) can recover the offset to
	//! convert `p_user → slot_start = p - ALIGN` for FS=false slots.
	static constexpr std::uint64_t chunk_header_size_info() noexcept {
	    return static_cast<std::uint64_t>(ALIGN) << 32;
	}
	typedef typename PoolAllocator<ALIGN, true, false>::FUINT FUINT;
protected:
	PoolAllocator(int count, char *addr, char *ppool);
	inline void *allocate_pooled(unsigned int SIZE);
	bool deallocate_pooled(char *p) override;
	int batch_return_to_bitmap(const CrossDeallocEntry *entries) noexcept override;
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

	// Phase 5c: m_sizes and m_available_bits dropped.  Per-slot SIZE
	// metadata now lives in the slot's own first ALIGN bytes (the
	// "+1 prefix" — bitmap claims N+1 bits, slot[0..3] stores SIZE as
	// uint32_t, returned pointer is `slot_start + ALIGN`).
	// Phase 5d-1 borrow scheme moved this to p_user - 8.
	//
	// Phase 5f: also dropped the Phase 5a 80% fragmentation cutoff +
	// the brief Phase 5f-1 `m_bits_set` counter.  allocate_pooled
	// now walks at most `m_count` FUINT words per call and bails on
	// full sweep — same worst-case cost as the upfront `count_bits`
	// scan was paying on EVERY call, but only when the walk genuinely
	// fails.  Quick check via `m_flags_filled_cnt` (inherited base)
	// catches the all-words-filled case in O(1).
};

#define ALLOC_ALIGN1 (ALLOC_ALIGNMENT * 2)
#if defined __LP64__ || defined __LLP64__ || defined(_WIN64) || defined(__MINGW64__)
	#define ALLOC_ALIGN2 (ALLOC_ALIGNMENT * 16)
	//! New on 64-bit: ALIGN3 = 1024 (= 16 × 64).  Used by buckets 31..36
	//! (sizes 3072..8192 in 1024-B step) so the FS=false machinery can
	//! cover above 2 KiB without ballooning slot-counts/chunks at the
	//! lower-ALIGN tiers.  Max in-pool slot size:
	//!   ALIGN3 × FUINT_BITS = 1024 × 64 = 65536 (we cap usage at 8192).
	#define ALLOC_ALIGN3 (ALLOC_ALIGNMENT * 64)
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

//! Sole tail of the dispatch chain for sizes > ALLOC_MAX_BUCKETED_SIZE
//! (= 16376 bytes since Phase 5d-4).  Phase 5d-4 covers up to 16 KiB
//! minus 8 B in 24 buckets via the 4-way exponential ladder; anything
//! bigger goes straight to libsystem here.  The legacy `ALLOCATE_9_16X`
//! macro and its power-of-2 PoolAllocator template explosions are
//! removed.
void* allocate_large_size_or_malloc(size_t size) throw();

extern bool g_sys_image_loaded;
//! `s_alloc_tls_off` is forward-declared earlier in this file (just above
//! PoolAllocator) so new_redirected can read it.

// `activateAllocator()` is declared by allocator.h — either as `extern`
// (inline-compiled / qmake build) or as an `inline noexcept {}` no-op
// (dylib build, where the dylib's __attribute__((constructor)) handles
// activation).  Don't redeclare here, would shadow the inline form.

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
//   - `AllocThreadExitCleanup::~dtor` on thread exit clears all entries back
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
static_assert(sizeof(AllocSlot) == sizeof(char *),
              "AllocSlot must be exactly one pointer wide — "
              "hot-path uses pointer-scaled indexed addressing (lsl #3 on 64-bit, lsl #2 on 32-bit)");

//! Bucket count (Phase 5d-4 layout).
//!   - index 0 (size = 0): reuses bucket 1's 16-B allocator
//!   - 1..16: sizes 16..256 in 16-B increments         (FS=true + FS=false mixed; unchanged)
//!   - 17..40: 4-way exponential FS=false ladder       (sizes 320..16384 total; 3 ALIGN stages)
//!       17..24: ALIGN= 64, slot total = 320, 384, 448, 512, 640, 768, 896, 1024  (N = 5..16)
//!       25..32: ALIGN=256, slot total = 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096  (N = 5..16)
//!       33..40: ALIGN=1024, slot total = 5120, 6144, 7168, 8192, 10240, 12288, 14336, 16384  (N = 5..16)
//!
//! `total` here = user_size + 8 (the Phase 5d "borrow" header).
//! user_capacity = total - 8.  Each stage doubles ALIGN; the N values
//! within a stage repeat (5, 6, 7, 8, 10, 12, 14, 16) — clean 4-way
//! 4-octave coverage with no over-provisioning (every (ALIGN, N) pair
//! exactly matches its bucket's slot total).
//!
//! ALIGN=64 (stage 1) chosen because every bucket-17..24 N value is
//! even when expressed in ALIGN=32 units — doubling the bitmap unit
//! halves the bit count per slot without losing size-class fidelity.
//!
//! Range 257..319 is intentionally folded into bucket 17 (total 320) —
//! the worst-case internal frag (63/320 ≈ 20%) is paid by a thin
//! distribution sliver in the alloc_stress histogram and avoids
//! adding yet another sub-bucket just to shave the boundary.
constexpr int ALLOC_NUM_BUCKETS = 41;

//! Size → bucket-index.  FS=true range (1..256) keeps the 16-byte-step
//! formula.  FS=false range (257..16376) uses a 4-way exponential
//! ladder via std::bit_width / __builtin_clzll:
//!
//!   total      = user_size + 8                          (header overhead)
//!   octave     = floor(log2(total))                     (msb position)
//!   sub_index  = (total >> (octave - 2)) & 0x3          (2-bit sub-bucket)
//!   if total has lower bits below sub_index: sub_index += 1
//!
//! The resulting (octave, sub) maps to bucket 16 + (octave - 8)*4 + sub:
//!   octave 8 (total 257..512): sub 1..3 → bucket 17..19 (sub 0 = 256,
//!                                          handled by the FS=true path)
//!   octave 9..14 (total 513..16384): sub 0..3 → bucket 20..40
//!
//! Sub overflow (e.g. octave 9 sub=4 after the lower-bits add) naturally
//! maps to the next octave's sub=0 via the same formula.
//!
//! Max user_size = 16376 (= 16 KiB - 8).  Sizes > ALLOC_MAX_BUCKETED_SIZE
//! fall through `new_redirected_large` to mmap.
constexpr std::size_t ALLOC_MAX_BUCKETED_SIZE = 16376u;

inline constexpr unsigned int bucket_for_size(std::size_t size) noexcept {
	// FS=true range: 1..256, 16-B step.  (size+15)>>4 yields 1..16
	// for size 1..256, and 0 for size==0 (reuses bucket 0's 16-B
	// allocator).
	if(size <= (std::size_t)ALLOC_SIZE16)
		return static_cast<unsigned int>((size + 15u) >> 4);
	// FS=false 4-way exponential range: 257..16376.
	std::size_t total = size + 8u;
	// floor(log2(total)).  __builtin_clzll undefined for 0, but
	// total >= 265 here so always > 0.
	int msb = 63 - __builtin_clzll(static_cast<unsigned long long>(total));
	// 2-bit sub-index for 4-way within the octave.
	int sub = static_cast<int>((total >> (msb - 2)) & 0x3u);
	// Any bits below the sub-index region ⇒ round up to next sub-bucket.
	std::size_t mask = (std::size_t(1) << (msb - 2)) - 1u;
	if(total & mask) ++sub;
	// bucket = 16 + (octave - 8) * 4 + sub
	// Octave 8 has FS=true at sub=0 (size 256), so FS=false starts at
	// sub=1 → bucket 17.  Sub overflow (4) naturally lifts to next
	// octave's sub=0 via the same formula.
	return 16u + static_cast<unsigned int>((msb - 8) * 4 + sub);
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
// thunk: `adrp; add; ldr; blr tlv_get_addr` — roughly 10-15 cycles
// of dependent loads + a function call per access.  This block
// bypasses that for the two hottest TLS arrays.  (Linux glibc lowers
// `__thread` to a direct `%fs:0 + offset` indexed load with no thunk
// call, so there's nothing to bypass — `KAME_FAST_TSD` is macOS-only.)
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
// On unsupported platforms (Windows; non-arm64/x86_64), the macros
// expand to direct `&g_thread_slots[0]` / `&g_thread_chunks[0]`
// references which keep the TLV thunk on the hot path.
// ---------------------------------------------------------------------

// macOS only: the TLV thunk (`tlv_get_addr` — `adrp/add/ldr/blr`) is
// what makes per-access `__thread` expensive on Apple platforms; this
// fast path replaces it with `mrs TPIDRRO_EL0` + an indexed load.
// On Linux, glibc lowers `__thread` to `%fs:0 + offset` directly with
// no thunk call (initial-exec model), so there is no thunk to bypass
// and adding the pthread-TSD redirection only buys glibc-layout
// fragility.  Restricted accordingly.
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__x86_64__))
    #define KAME_FAST_TSD 1
#else
    #define KAME_FAST_TSD 0
#endif

#if KAME_FAST_TSD
//! Architecture-specific read of the thread-pointer register (pthread
//! struct base).  Used as the base for the byte-offset TSD read.
//!
//! Important: `__builtin_thread_pointer()` on arm64 expands to
//! `mrs TPIDR_EL0`, which is the *read-write* register.  On macOS,
//! Apple's libc keeps the thread pointer in `TPIDRRO_EL0` (read-only)
//! and leaves `TPIDR_EL0` zero / unused — so the builtin returns
//! garbage there.  Always use explicit inline asm.
static inline char *kame_thread_pointer() noexcept {
    #if defined(__aarch64__)
        uintptr_t tp;
        __asm__ volatile("mrs %0, TPIDRRO_EL0" : "=r"(tp));
        return (char *)tp;
    #elif defined(__x86_64__)
        // macOS Intel: %gs:0 stores self-pointer == pthread struct base.
        uintptr_t tp;
        __asm__ volatile("movq %%gs:0, %0" : "=r"(tp));
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
	// Fast TSD access on macOS (arm64/x86_64); else direct TLV access.
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

#endif /* USE_STD_ALLOCATOR */

#endif /* ALLOCATOR_PRV_H_ */
