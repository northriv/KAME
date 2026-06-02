/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            LICENSE-APACHE-2.0 in this directory)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see LICENSE-GPL-2.0 in this directory).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
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

// ===== MSVC compatibility shim for the live pool =====
// The live pool is ON by default on MSVC (opt OUT via
// KAME_DISABLE_POOL_MSVC → USE_STD_ALLOCATOR, in which case this header
// isn't included).  The pool core is written for GCC/Clang; map the
// GCC-isms MSVC lacks.  Placed here (the pool-private header) rather than
// allocator.h so that ANY includer — incl. tests that include
// allocator_prv.h directly — gets the shim.  (The 5 __sync_* atomic
// wrappers below carry their own _Interlocked* branch; TLS already falls
// to thread_local on MSVC.)
#if defined(_MSC_VER) && !defined(__GNUC__)
    #include <intrin.h>
    // noinline/cold/used/tls_model are codegen hints — safe to drop on
    // MSVC.  `section` is macOS-only; `constructor` is handled via static-
    // init, not the attribute.  So strip every __attribute__.
    #define __attribute__(x)
    #ifndef __builtin_expect
        #define __builtin_expect(expr, c) (expr)
    #endif
    // constexpr bit-scan (C++17 relaxed constexpr) — _BitScan* aren't
    // constexpr, but these feed constexpr ladder-bucket math.
    constexpr int kame_msvc_ctzll(unsigned long long v) noexcept { if(!v) return 64; int n=0; while(!(v&1ull)){v>>=1;++n;} return n; }
    constexpr int kame_msvc_ctz(unsigned int v) noexcept { if(!v) return 32; int n=0; while(!(v&1u)){v>>=1;++n;} return n; }
    constexpr int kame_msvc_clzll(unsigned long long v) noexcept { if(!v) return 64; int n=0; while(!(v&(1ull<<63))){v<<=1;++n;} return n; }
    #define __builtin_ctzll(x) kame_msvc_ctzll(x)
    #define __builtin_ctz(x)   kame_msvc_ctz(x)
    #define __builtin_clzll(x) kame_msvc_clzll(x)
    #define __builtin_thread_pointer() ((void *)__readgsqword(0x30)) // TEB self-ptr
    static inline bool kame_msvc_mul_ovf(std::size_t a, std::size_t b, std::size_t *out) noexcept {
        unsigned long long hi; *out = (std::size_t)_umul128(a, b, &hi); return hi != 0ull;
    }
    #define __builtin_mul_overflow(a, b, outp) kame_msvc_mul_ovf((a), (b), (outp))
#endif

// Cache-line size, architecture-dependent.  Kept as a fixed macro
// (std::hardware_destructive_interference_size is ABI-fragile across
// libc++/libstdc++ and warns under GCC).  Duplicated from kamestm's
// transaction_detail.h to keep kamepoolalloc standalone-buildable.
// Documents the per-target line sizes that motivate the
// PoolAllocatorBase deallocate-fast-path hot block being held off
// chunk_header's cache line (Apple-aarch64 / PPC 128 B; Fujitsu 256 B).
// NOTE: the hot block itself is `alignas(64)`, not this value — the
// embedded object is only 64-aligned (chunk_base+ALLOC_CHUNK_HEADER), so
// a larger alignas would be UB; 64 still clears chunk_header's line for
// the 64/128 B targets because the object already starts at +64.  See
// PoolAllocatorBase::m_owner_id.
#ifndef KAME_CACHE_LINE
  #if defined(__APPLE__) && defined(__aarch64__)
    #define KAME_CACHE_LINE 128
  #elif defined(__powerpc64__) || defined(__POWERPC__)
    #define KAME_CACHE_LINE 128
  #elif defined(__aarch64__) && (defined(__FUJITSU) || defined(__CLANG_FUJITSU))
    #define KAME_CACHE_LINE 256
  #else
    #define KAME_CACHE_LINE 64
  #endif
#endif

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

#if defined(_MSC_VER) && !defined(__GNUC__)
// MSVC: the GCC __sync_* builtins don't exist.  Map to _Interlocked*
// (full-barrier, same ordering as the legacy __sync_*), dispatched by
// width.  Only 4- and 8-byte targets occur (ints, FUINT, pointers).
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value || std::is_pointer<T>::value, bool>::type
atomicCompareAndSet(T oldv, T newv, T *target) noexcept {
    if constexpr (sizeof(T) == 8) {
        long long o = (long long)(std::intptr_t)oldv;
        return _InterlockedCompareExchange64((long long volatile *)target,
                   (long long)(std::intptr_t)newv, o) == o;
    } else {
        long o = (long)(std::intptr_t)oldv;
        return _InterlockedCompareExchange((long volatile *)target,
                   (long)(std::intptr_t)newv, o) == o;
    }
}
template <typename T>
inline void atomicInc(T *target) noexcept {
    if constexpr (sizeof(T) == 8) _InterlockedIncrement64((long long volatile *)target);
    else _InterlockedIncrement((long volatile *)target);
}
template <typename T>
inline void atomicDec(T *target) noexcept {
    if constexpr (sizeof(T) == 8) _InterlockedDecrement64((long long volatile *)target);
    else _InterlockedDecrement((long volatile *)target);
}
template <typename T>
inline bool atomicDecAndTest(T *target) noexcept {
    if constexpr (sizeof(T) == 8) return _InterlockedDecrement64((long long volatile *)target) == 0;
    else return _InterlockedDecrement((long volatile *)target) == 0;
}
//! Atomic fetch-and-AND.  Returns the OLD value (before AND).
template <typename T>
inline T atomicFetchAnd(T *target, T value) noexcept {
    if constexpr (sizeof(T) == 8)
        return (T)_InterlockedAnd64((long long volatile *)target, (long long)value);
    else
        return (T)(long)_InterlockedAnd((long volatile *)target, (long)value);
}
//! (§32) Atomic fetch-and-OR (MSVC).  Returns the OLD value.
template <typename T>
inline T atomicFetchOr(T *target, T value) noexcept {
    if constexpr (sizeof(T) == 8)
        return (T)_InterlockedOr64((long long volatile *)target, (long long)value);
    else
        return (T)(long)_InterlockedOr((long volatile *)target, (long)value);
}
//! (§32) Atomic exchange (MSVC).  Returns the OLD value.
template <typename T>
inline T atomicExchange(T *target, T value) noexcept {
    if constexpr (sizeof(T) == 8)
        return (T)_InterlockedExchange64((long long volatile *)target, (long long)value);
    else
        return (T)(long)_InterlockedExchange((long volatile *)target, (long)value);
}
//! (§32) Atomic fetch-and-SUB (MSVC).  Returns the OLD value.
template <typename T>
inline T atomicFetchSub(T *target, T value) noexcept {
    if constexpr (sizeof(T) == 8)
        return (T)_InterlockedExchangeAdd64((long long volatile *)target,
                                            -(long long)value);
    else
        return (T)(long)_InterlockedExchangeAdd((long volatile *)target,
                                                 -(long)value);
}
#else
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
//! caller can compute the resulting bit pattern.  Used by an earlier change
//! BIT_OWNED clear to detect "I brought m_flags_packed to 0" → I'm
//! the unique releaser.
template <typename T>
inline T atomicFetchAnd(T *target, T value) noexcept {
    return __sync_fetch_and_and(target, value);
}
//! (§32) Atomic fetch-and-OR.  Returns the OLD value.  Used by the
//! cross-thread shadow path in `batch_return_to_bitmap` (FS=true,
//! kHasReturnShadow): bit-set into `return_flags[w]` without
//! coordination with other cross-thread freers on the same word.
template <typename T>
inline T atomicFetchOr(T *target, T value) noexcept {
    return __sync_fetch_and_or(target, value);
}
//! (§32) Atomic exchange.  Returns the OLD value, stores `value`.  Used
//! by the owner sweep `sweep_return_shadow()` to drain pending returns
//! out of `return_flags[w]` in one shot.
template <typename T>
inline T atomicExchange(T *target, T value) noexcept {
    return __atomic_exchange_n(target, value, __ATOMIC_ACQ_REL);
}
//! (§32) Atomic fetch-and-SUB.  Returns the OLD value, subtracts
//! `value`.  Used by the owner sweep to batch MASK_CNT decrements
//! (popcount(swept_bits) at a time) instead of per-bit `atomicDec`.
template <typename T>
inline T atomicFetchSub(T *target, T value) noexcept {
    return __sync_fetch_and_sub(target, value);
}
#endif

#if defined(__GNUC__) || defined(__clang__)
	#define ALLOC_TLS __thread //TLS for allocations, could be better for NUMA.
	// Hot-path TLS variables: marked initial-exec to bypass libc's
	// `__tls_get_addr` thunk (~15% of total runtime under the
	// global-dynamic default on shared libraries).  IE lowers each
	// access to a single `mov %fs:offset(,%idx,8),%reg` op.
	//
	// Cost: each IE TLS variable claims space in the program's static
	// TLS block at load time, so the library can ONLY be loaded at
	// process start (LD_PRELOAD or normal -l link) — NOT `dlopen`'d
	// after startup, since dlopen has limited surplus static-TLS
	// budget.  We restrict the IE marking to the small hot variables
	// (g_thread_freelist_ptr 384 B + s_tls_owner_id 4 B + s_alloc_tls_off
	// 1 B ≈ 400 B); the larger cold TLS (tls_cross_dealloc_batch 16 KiB)
	// stays on global-dynamic.  Total static-TLS demand of the IE
	// variables fits in the default Linux surplus budget (~4 KiB).
	//
	// Windows carve-out: `tls_model("initial-exec")` targets the
	// ELF/Mach-O `__tls_get_addr` thunk, which doesn't exist on the
	// Windows ABI — TLS access there goes through the FS/GS segment
	// pointer + `_tls_index` machinery (real `.tls` section) or
	// `__emutls_get_address` (MinGW gcc emulated TLS) instead.  MinGW
	// gcc + lld either silently ignores the attribute or emits a broken
	// section layout across the EXE/DLL boundary, with the symptom that
	// IE-marked TLS reads return garbage in modules / dlopen'd DLLs
	// (observed: kame.exe's allocator activates correctly under inline-
	// compile, but `operator new` from a module crashes the moment it
	// reads `g_thread_freelist_ptr`).  Drop the attribute on Windows;
	// the cost is one `_tls_index`-indirect read per hot-path TLS access
	// (or one `__emutls_get_address` call on emutls builds) — measurable
	// but tiny compared to the cost of a corrupted pool state.
	#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
		#define ALLOC_TLS_IE __thread
	#else
		#define ALLOC_TLS_IE __thread __attribute__((tls_model("initial-exec")))
	#endif
#else
	#define ALLOC_TLS thread_local
	#define ALLOC_TLS_IE thread_local
#endif

//! Allocation unit (1 chunk = N × this).  every mmap region is
//! a uniform 32 MiB block carved into 128 fixed-size 256 KiB "units".
//! A chunk = 1, 2, or 4 contiguous units depending on the per-template
//! `CHUNK_UNITS` (= 1 for ALIGN < 256, = 2 for ALIGN < 1024, = 4 for
//! ALIGN ≥ 1024 = 1024).  The unit size matches the previous an earlier change
//! minimum so cross-thread chunk residency under lazy commit stays
//! tight; the buddy approach replaces the previous 2× growth ladder.
//!
//! O(1) chunk_base lookup from any slot uses `s_back_offset[]` (one
//! byte per unit, per region) — see `PoolAllocatorBase::s_back_offset`
//! below.  `back_off = u - base_u`; a reader does `base_u = u -
//! back_off` and then `chunk_base = region + base_u * 256K`.
#define ALLOC_MIN_CHUNK_SIZE (1024 * 256) //256 KiB unit
//! log2(ALLOC_MIN_CHUNK_SIZE) — used for fast unit-index extraction
//! `unit_idx = pdiff >> ALLOC_MIN_CHUNK_SHIFT`.  Compile-time constant.
#define ALLOC_MIN_CHUNK_SHIFT 18  //log2(256 KiB)
//! Max chunk = 16 units = 4 MiB.  The compile-time bucket templates only
//! ever use CHUNK_UNITS ∈ {1,2,4} (≤ 1 MiB — see the CHUNK_UNITS constexpr
//! below); the larger 8/16-unit chunks are claimed only by the *runtime*
//! N-unit path used for dedicated single-slot large allocations
//! (allocate_dedicated_chunk).  A 16-unit chunk still fits within a 32 MiB
//! region's 128-unit bitmap and within one BitmapWord's claim CAS (≤ 64
//! units/word on 64-bit, ≤ 32 on 32-bit), and back_off (uint8) covers it.
#define ALLOC_MAX_CHUNK_UNITS 16
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
//! regions are uniform 32 MiB — no ladder, no growth.  The
//! an earlier change growth-cap macro `GROW_CHUNK_SIZE` is removed; chunk size
//! is now a per-template constant (`PoolAllocator<...>::CHUNK_SIZE`).
//! `NUM_ALLOCATORS_IN_SPACE == 128` matches the bit count of the per-
//! region claim bitmap (BitmapWord bits × BITMAP_WORDS_PER_REGION =
//! UNITS_PER_BITMAP_WORD × BITMAP_WORDS_PER_REGION = 128 units).  Every
//! region is 32 MiB = 128 × 256 KiB regardless of host word size.
//!
//! `ALLOC_MAX_REGIONS` is the VA cap — each region is mmap'd
//! `PROT_READ | PROT_WRITE` upfront (switches the release path
//! from `mprotect(PROT_NONE)` to `madvise(MADV_FREE/DONTNEED)` so
//! reclaim is RSS-cheap without protection toggling).  Total
//! reservation = 32 MiB × N entries.
//!
//!   host         |   N  | total VA cap
//!   -------------+------+--------------------
//!   64-bit       | 3200 | 100 GiB
//!   32-bit       |   96 |   3 GiB  (= full Linux 3G/1G user VA;
//!                |      |   lazy — only mmap'd-on-demand regions
//!                |      |   count against the actual user VA budget,
//!                |      |   so the cap is a ceiling not a reservation)
//!   Windows      | 3200 | (same as 64-bit; pool path opt-in via dylib)
//!
//! Sizing rationale: a general-purpose allocator must accommodate
//! workloads that stretch into the multi-GiB user-heap range without
//! catastrophically aborting (the earlier change's 3 GiB cap was first to expose
//! this — `alloc_only` at 8192 B for 500K iters needs ~4.5 GiB pool
//! and would hit `# of chunks exceeds the limit`).  64-bit hosts have
//! 128 TiB of user VA on macOS / Linux / Windows so a 100 GiB cap is
//! 0.08 % of available VA — trivial in reservation cost.  32-bit
//! matches the practical user-VA ceiling on Linux (3G/1G split).
//!
//! Lazy mmap: `allocate_chunk` only mmaps a region when the previous
//! ones are full.  A workload using 5 regions walks 5 entries in
//! `deallocate`'s region-loop and pays RSS only for those 5 × 32 MiB.
//! Bumping the cap doesn't accelerate steady-state allocation.
//!
//! Cache impact for `s_back_offset[]` (only the COLD chunk-claim /
//! deallocate-region-miss paths touch unused regions' table entries):
//!   * 64-bit: 3200 × 128 × 8 = 3.1 MiB BSS.  Unused entries stay
//!     zero-filled in BSS pages, never paged in.  Hot subset =
//!     populated_regions × 1 KiB; for typical KAME workloads (1-5
//!     populated regions) the working set fits in a few KiB.
//!   * 32-bit: 96 × 128 × 8 = 96 KiB BSS.
//!
//! Cache impact for `s_claim_bitmap[]`:
//!   * 64-bit: 3200 × 2 × 8 = 50 KiB.  L2 only on full scan
//!     (allocate_chunk cold path); steady-state ops touch one word.
//!   * 32-bit: 96 × 4 × 4 = 1.5 KiB.  L1d.
//!
//! `s_mmapped_spaces[]` (the region-base array walked by every
//! deallocate / lookup_chunk):
//!   * 64-bit: 3200 × 8 = 25 KiB.  First N entries (= populated
//!     regions) hot in L1d; rest unused.
//!   * 32-bit: 96 × 8 = 768 B.  L1d.
#define ALLOC_MIN_MMAP_SIZE (1024 * 1024 * 32) //32 MiB = 256 KiB × 128
//! log2(ALLOC_MIN_MMAP_SIZE).  Used as the "region index shift" for the
//! 2-level radix lookup (§13): any pointer's owning region is identified
//! by its upper `64 - ALLOC_MIN_MMAP_SHIFT` bits.
#define ALLOC_MIN_MMAP_SHIFT 25
static_assert((1ull << ALLOC_MIN_MMAP_SHIFT) == (unsigned)ALLOC_MIN_MMAP_SIZE,
              "ALLOC_MIN_MMAP_SHIFT must equal log2(ALLOC_MIN_MMAP_SIZE)");
//! (§13.3) Region-count ceiling.  No longer an array bound — the
//! per-region-index globals (`s_mmapped_spaces[]`, `s_region_has_free[]`)
//! are retired; regions live on a push-only list and their metadata
//! inside themselves.  This constant is now ONLY the default (uncapped)
//! value of the runtime `s_max_regions_cap`, set to the radix tree's full
//! VA coverage so the pool is effectively VA-limited (not array-capped).
#if defined __LP64__ || defined __LLP64__ || defined(_WIN64) || defined(__MINGW64__)
    //! 1 << RADIX_REGION_BITS = 4194304 regions × 32 MiB = 128 TiB — the
    //! 47-bit user-VA range the radix covers (static_assert'd below).
    #define ALLOC_MAX_REGIONS (1 << 22)
#else
    //! 32-bit host: 96 regions = 3 GiB, matching the Linux 3G/1G user-VA
    //! ceiling (mmap fails past this anyway).  Kept small so the
    //! `regions × 32 MiB` byte math in kame_pool_get_max_bytes can't
    //! overflow a 32-bit size_t.
    #define ALLOC_MAX_REGIONS 96
#endif

//! 2-level radix tree for O(1) pointer-to-region lookup (§13,
//! tests/CHUNK_CLAIM_TLA_NOTES.md).  Replaces the O(N) linear walk of
//! `s_mmapped_spaces[]` that every `lookup_chunk(p)` / `deallocate(p)` /
//! `size_of(p)` used to do.
//!
//! Encoding: region index = `p >> ALLOC_MIN_MMAP_SHIFT` (upper bits of
//! the pointer); split into RADIX_L1_BITS + RADIX_L2_BITS.  Picked 11+11
//! = 22-bit region index, which covers 22 + 25 = 47-bit user VA — the
//! Linux x86-64 default user-VA range.  Pointers above this range
//! (rare: 48-bit ARM64, 5-level paging kernels) fall back via the
//! defensive bound check in `radix_lookup` (returns -1, treated as
//! foreign — same outcome as today's "not in `s_mmapped_spaces[]`").
//!
//! Storage:
//!   L1: fixed `[1<<11 = 2048]` `atomic<RadixL2Node*>` in BSS (16 KiB).
//!       Top level is hot-in-L1d on the lookup path.
//!   L2: each populated node is `[1<<11 = 2048]` `atomic<uint32_t>`
//!       slots (8 KiB = 2 pages).  Slot value 0 = unpopulated; non-zero
//!       = present (a pure presence token since §13.3 — the region base
//!       is derived from the pointer, not a stored index).  L2 nodes
//!       are allocated
//!       LAZILY via mmap from `radix_alloc_l2()` (NOT through the
//!       interposed libc malloc — that would recurse).  Each L2 covers
//!       `2^11 × 32 MiB = 64 GiB` of VA, so a 1-TiB workload populates
//!       ~16 L2 nodes (128 KiB committed total).
//!
//! Concurrency: L1 slot install is one CAS (loser munmap's its loser
//! leaf).  L2 slot store is `release`-paired with the reader's
//! `acquire` load on the L1 entry that brought it into view (the L1
//! load synchronizes-with the L2 init store).  No reclamation needed
//! (slots are written once and live for the process lifetime — regions
//! are never unmapped in the current design; §13.2 may revisit).
constexpr int RADIX_L1_BITS = 11;
constexpr int RADIX_L2_BITS = 11;
constexpr unsigned RADIX_L1_SIZE = 1u << RADIX_L1_BITS;
constexpr unsigned RADIX_L2_SIZE = 1u << RADIX_L2_BITS;
constexpr int RADIX_REGION_BITS = RADIX_L1_BITS + RADIX_L2_BITS;  // 22
#if defined __LP64__ || defined __LLP64__ || defined(_WIN64) || defined(__MINGW64__)
// 64-bit: the region-count ceiling equals the radix's full VA coverage.
static_assert(ALLOC_MAX_REGIONS == (1 << RADIX_REGION_BITS),
              "ALLOC_MAX_REGIONS must equal the radix VA coverage "
              "(1 << RADIX_REGION_BITS)");
#endif

//! (§19) Radix slot value semantics — what kind of allocation lives at
//! the 32-MiB-aligned base of a present slot.  Lookup returns this
//! directly; 0 means "absent, foreign pointer".
enum RadixKind : uint32_t {
    KAME_RADIX_ABSENT = 0u,  // no allocation at this base
    KAME_RADIX_POOL   = 1u,  // a PoolAllocatorBase::RegionMeta lives here
    KAME_RADIX_LARGE  = 2u,  // a LargeAllocMeta lives here (§19 large-alloc)
};

struct RadixL2Node {
    // Member name is `entries`, NOT `slots` — Qt defines `slots` as an
    // empty preprocessor token in <QtCore>, so `T slots[N];` at class
    // scope becomes `T [N];`, which Apple Clang then mis-parses as a
    // structured binding (illegal at class scope).  Renamed to keep the
    // header includable from Qt-built TUs (`kame/main.cpp`, etc.)
    // without `#undef slots` gymnastics.
    std::atomic<uint32_t> entries[RADIX_L2_SIZE];  // (§19) RadixKind values
};
static_assert(sizeof(RadixL2Node) == RADIX_L2_SIZE * 4u,
              "RadixL2Node must be exactly RADIX_L2_SIZE * 4 bytes "
              "(atomic<uint32_t>) for a clean 8 KiB layout");

//! Reserved bytes at the head of every chunk.  Layout:
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
//!                          instead.
//!               High 32 bits: ALIGN (always — for non-templated
//!               dispatchers; see `chunk_header_size_info()`).
//!   [ 8 .. 15]: `PoolAllocatorBase *` palloc (chunk owner).
//!   [16 .. 23]: `DeallocateFn` — non-virtual static trampoline
//!               (per-template) that dispatches the dealloc body.
//!   [24 .. 31]: `SizeOfFn` — slot-size lookup trampoline.
//!               FS=false: reads SIZE from the per-slot
//!               `{bucket, SIZE}` header at `p - 8`.
//!   [32 .. 55]: pad.
//!   [56 .. 63]: RESERVED for FS=false slot 0's `{uint32_t bucket,
//!               uint32_t SIZE}` header (an earlier change "borrow" scheme
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
//! TODO: the [0..7] SIZE info enables a
//! "unified deallocate" that branches on `(hdr[0] != 0)` instead of
//! the indirect `DeallocateFn` call.  The high-32-b ALIGN is
//! already available, so FS=false's `p - 8` header read needs no
//! extra per-template dispatch.
#define ALLOC_CHUNK_HEADER 64

//! (§15) Forward-shift reservation: every chunk's first byte sits
//! K_MAX bytes BEFORE its first claimed unit's boundary.  Slot region
//! therefore starts at the unit boundary (256 KiB-aligned), giving
//! deterministic page / huge-page alignment for slot data.
//!
//!     chunk_base = unit_boundary[base_unit] - ALLOC_CHUNK_K_MAX
//!     slot region = [chunk_base + K_MAX, chunk_base + chunk_size)
//!                 = [unit_boundary[base_unit],
//!                    unit_boundary[base_unit + CHUNK_UNITS] - K_MAX)
//!
//! The K_MAX bytes of metadata (chunk_header + PoolAllocator object +
//! m_flags + padding) live in the PREVIOUS unit's last page.  Adjacent
//! chunks tile end-to-end: chunk N's last K_MAX bytes are reserved for
//! chunk N+1's metadata (if the next position is later claimed).  This
//! means EVERY chunk's effective slot region is `chunk_size - K_MAX`
//! bytes; one chunk's tail K_MAX is always next-chunk's metadata slot.
//!
//! The first chunk in a region (base_unit = 1) has its metadata in
//! unit 0 (the RegionMeta unit) — unit 0's RegionMeta lives at the
//! start (~150 B), and only the last K_MAX bytes are reserved for
//! chunk 1's metadata, so no collision.
//!
//! K_MAX is sized so that the largest template (smallest ALIGN, biggest
//! count) fits its PoolAllocator object + m_flags array + (§32) optional
//! return shadow bitmap within `K_MAX - ALLOC_CHUNK_HEADER` bytes.
//! Verified at compile time in the per-template `create()`:
//!   - ALIGN=16, CHUNK_UNITS=1: PoolAllocator (~200 B) + m_flags (~2 KiB)
//!                              + shadow (~2 KiB) + padding < 8 KiB ✓
//!   - ALIGN=256, CHUNK_UNITS=2: PoolAllocator + m_flags (~256 B) +
//!                              shadow (~256 B) ≪ 8 KiB
//!   - Dedicated chunks: chunk_header only (~64 B) ≪ 8 KiB
//!
//! (§32) Bumped 4 KiB → 8 KiB to admit a second `count`-word bitmap
//! (`return_flags()` shadow) for ALL FS=true real chunks including
//! ALIGN=16, eliminating producer↔consumer false-sharing across the
//! whole bucket range.  On Linux/Windows (PAGE = 4 KiB) the metadata
//! now occupies 2 pages instead of 1 — one extra TLB entry per chunk
//! during the dealloc hot-block read.  On macOS arm64 (PAGE = 16 KiB)
//! and POWER (PAGE = 64 KiB) the metadata still sits in 1 page.
//! Trade-off: 4 KiB extra reservation per chunk = 1.6 % overhead on
//! 256 KiB chunks, 0.8 % on 512 KiB, ≤ 0.4 % on 1 MiB+ — paid only by
//! chunks that hand out slots, not by the dedicated-chunk path.
#define ALLOC_CHUNK_K_MAX 8192

#define ALLOC_CHUNK_HEADER_SIZE_INFO_OFFSET     0   // [ 0.. 7]: chunk SIZE info
#define ALLOC_CHUNK_HEADER_PALLOC_OFFSET        8   // [ 8..15]: palloc
#define ALLOC_CHUNK_HEADER_FN_OFFSET           16   // [16..23]: DeallocateFn
#define ALLOC_CHUNK_HEADER_SIZEOF_FN_OFFSET    24   // [24..31]: SizeOfFn
// [32..39]: dedicated-chunk total byte size (only when SIZE_INFO low-32 ==
// ALLOC_CHUNK_DEDICATED_SIZEINFO).  Reuses the field freed by the retired
// recycle epoch.  A "dedicated" chunk is a single large allocation that
// occupies a whole N-unit chunk (no sub-slot bitmap); deallocate() detects
// the sentinel BEFORE bucket_for_size and releases the whole chunk.
#define ALLOC_CHUNK_HEADER_DEDICATED_SIZE_OFFSET 32  // [32..39]: dedicated chunk bytes
//! SIZE_INFO low-32 sentinel marking a dedicated single-slot large chunk.
//! Distinct from any real ALIGN (≤ 1024) and from 0 (the FS=false marker).
#define ALLOC_CHUNK_DEDICATED_SIZEINFO 0xFFFFFFFFu
// [32..55] free (was recycle epoch, retired — DLL/lookup safety comes
//           from BIT_OWNED gating + live-slot invariant, not an epoch)
// [56..63] = slot-0 header (= ALLOC_CHUNK_HEADER - 8).  No constant
// needed — `allocate_pooled` reaches it via the uniform
// `slot_start - 8` math (slot 0's slot_start == m_mempool ==
// chunk_base + ALLOC_CHUNK_HEADER).
static_assert(ALLOC_CHUNK_HEADER >= ALLOC_CHUNK_HEADER_SIZEOF_FN_OFFSET + 8 + 8,
              "chunk header must have >= 8 B of pad between SizeOfFn "
              "and the slot-0 reservation at chunk_header[-8..-1].");

#define ALLOC_ALIGNMENT 16 //bytes, not 8 but 16 for compatibility
#define ALLOC_MAX_CHUNKS_OF_TYPE \
	(ALLOC_MIN_MMAP_SIZE / ALLOC_MIN_CHUNK_SIZE * ALLOC_MAX_REGIONS)

//! Max distinct sizes a single chunk hands out — depth of its (ALIGN,FS)
//! template's size set; sizes the compact per-chunk freelist
//! `PoolAllocatorBase::m_freelist_head[]` (follow-up "(1b)" §12.3).
//! FS=true chunks serve exactly 1 size (local id 0); the widest FS=false
//! tier is ALIGN=256, which serves buckets {16, 32..39} = 9 sizes
//! (id 0..8).  Defined here (before PoolAllocatorBase) since the class
//! needs it for the array bound; `kBucketLocalId[]` (below) and a
//! BucketTraits-derived static_assert (allocator.cpp) verify no
//! (ALIGN,FS) tier exceeds this and that ids are collision-free.
constexpr int KAME_LOCAL_BUCKETS = 9;

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
	//! regions are uniform 32 MiB, so `deallocate_<>` no longer
	//! needs the per-level compile-time CHUNK_SIZE template parameter.
	//! Collapsed to a single non-template function with a runtime
	//! region-walk loop — eliminates the 24/96-level template recursion
	//! that previously generated one inlined copy of the body per
	//! ladder level (icache bloat scaling with ALLOC_MAX_REGIONS).
	static inline bool deallocate(void *p);
	//! Look up the slot size (bytes) for a pointer.  Returns 0 if `p`
	//! is not a pool slot (foreign / libsystem-malloc'd / null).  Uses
	//! the same chunk-header pattern as `deallocate` and dispatches
	//! the slot-size lookup through the chunk's `SizeOfFn`.
	static inline std::size_t size_of(void *p);
	//! Dedicated single-slot large allocation (sizes between
	//! ALLOC_MAX_BUCKETED_SIZE and a 16-unit chunk's payload = 4 MiB − 64 B).
	//! Claims a whole N-unit chunk (no sub-slot bitmap) and returns
	//! chunk_base + ALLOC_CHUNK_HEADER; freed via the s_back_offset bit7
	//! path in deallocate() / size_of().  Returns nullptr (caller falls
	//! through to std::malloc) when the payload is too large or the region
	//! cap is hit.
	static void *allocate_dedicated_chunk(std::size_t size) noexcept;
	//! (§22) Public forwarder to the protected `deallocate_chunk` (the
	//! N-bit bitmap-CAS claim-clear + madvise that truly releases a
	//! dedicated chunk).  Exists so the namespace-level large-recycle
	//! cache (allocator.cpp) can release a recycled dedicated chunk on
	//! eviction / thread-exit.  The evicting thread is the chunk's unique
	//! owner (the claim bits stay set while cached, so no other thread can
	//! re-claim it; the release is a single-winner CAS), hence race-free.
	static void recycle_release_chunk(char *chunk_base,
	                                  std::size_t chunk_size) noexcept;
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
	//! uniform 32 MiB, and `s_back_offset[]` maps any claimed unit back
	//! to its chunk base in O(1).
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

	//! Slot region start = `chunk_base + ALLOC_CHUNK_K_MAX` (§15
	//! forward-shift).  Constexpr-derivable from `this` because the
	//! PoolAllocator object is always placement-new'd at
	//! `chunk_base + ALLOC_CHUNK_HEADER`, so the slot region is
	//! `(char*)this + (K_MAX - HEADER)` — a 4032-byte offset shared by
	//! every template.  This identity replaces the former `m_mempool`
	//! field, saving 8 bytes from the cache-line-1 hot block (now packs
	//! more of `m_freelist_head[]` into the same line) and turning the
	//! hot-path `chunk_obj->m_mempool` load into a LEA off `this`.
	char *mempool() noexcept {
	    return reinterpret_cast<char *>(this)
	         + (ALLOC_CHUNK_K_MAX - ALLOC_CHUNK_HEADER);
	}
	const char *mempool() const noexcept {
	    return reinterpret_cast<const char *>(this)
	         + (ALLOC_CHUNK_K_MAX - ALLOC_CHUNK_HEADER);
	}
protected:
	PoolAllocatorBase() = default;
	virtual bool deallocate_pooled(char *p) = 0;

	template <class ALLOC>
	static ALLOC *allocate_chunk();
	//! Runtime N-unit chunk claim (1..ALLOC_MAX_CHUNK_UNITS).  Walks the
	//! region bitmaps (two passes, incl. fresh aligned mmap), wins a
	//! single-word CAS over `chunk_units` contiguous, chunk_units-aligned
	//! claim bits, writes `s_back_offset[base..] = (u | back_off_flag)`,
	//! and returns the chunk_base address (NO chunk_header / writeBarrier
	//! — the caller writes those).  Returns nullptr if the region cap is
	//! reached.  Used by the dedicated single-slot large path
	//! (`allocate_dedicated_chunk`); the compile-time bucket templates
	//! keep their own walk in `allocate_chunk<ALLOC>()` for now.
	static char *claim_chunk(unsigned chunk_units,
	                         std::uint8_t back_off_flag) noexcept;
	//! Release a chunk back to PROT_NONE.  Clears the chunk header
	//! pointer at `chunk_base`, mprotect's the chunk back to PROT_NONE,
	//! and clears the matching claim bit in `s_claim_bitmap[]` (region
	//! + bit located via a walk over `s_mmapped_spaces[]`).  Called
	//! both from the owner-side `deallocate_<>` last-slot release path
	//! and from the cross-batch `batch_return_to_bitmap` suicide path.
	//!
	//! `reclaim_pages` gates the `madvise(MADV_FREE/DONTNEED)`
	//! call.  Default `true` for the normal mid-run release paths
	//! (cross-thread last-slot, owner-side empty, allocate-failure
	//! cleanup) where eager page reclaim controls long-running-process
	//! RSS.  `release_dll_chunks_for_thread()` passes `false` — at
	//! thread-exit, `madvise(MADV_DONTNEED)` on every chunk was
	//! consuming ~30 % of bench-style `alloc_only` time (perf-confirmed
	//! on Linux: `clear_page_erms` + `free_pages_and_swap_cache` cost
	//! per chunk; ~2000 chunks × ~100 µs each); skipping it lets the
	//! kernel reclaim pages on process exit / OOM pressure rather than
	//! eagerly on every thread teardown.  Saved pages are still
	//! protected by the global `m_max_reserved_bytes` cap.
	static void deallocate_chunk(char *chunk_base, size_t chunk_size,
	                             bool reclaim_pages = true);

public:
	//! Cache-line-isolated hot block for the deallocate owner-free fast
	//! path (follow-up "(1b)", tests/CHUNK_CLAIM_TLA_NOTES.md §12.3).  All
	//! members are write-once at `allocate_chunk` and immutable for the
	//! chunk's life, so the fast path reads them with no coherence cost
	//! beyond the initial store, and NEVER touches chunk_header's cache
	//! line (chunk_base[0..63] — palloc at +8, size_info at +0).
	//!
	//! ALIGNMENT: the embedded PoolAllocator object sits at chunk_base +
	//! ALLOC_CHUNK_HEADER (= 64) and chunk_base is 256 KiB-aligned, so the
	//! object is EXACTLY 64-aligned — `alignas(128)` (Apple Silicon's
	//! KAME_CACHE_LINE) would be UB here (placement-new of a 128-aligned
	//! type at a 64-aligned address) and the member would not actually be
	//! 128-aligned.  64 suffices because the object already starts at +64:
	//! the hot block lands at object-offset >= 64 = absolute >=
	//! chunk_base+128, a line distinct from chunk_header[0..63] for both
	//! 64 B (x86-64) and 128 B (Apple Silicon) targets.  (256 B (Fujitsu)
	//! would need moving ALLOC_CHUNK_HEADER — out of scope.)
	//!
	//!   m_owner_id    : owner-thread id; freeing thread matches its
	//!                   `s_tls_owner_id` (0 = unassigned/released, never
	//!                   matches a live thread).  Also the released signal:
	//!                   `deallocate_chunk` clears it to 0 so the fast path
	//!                   rejects a released chunk WITHOUT reading palloc.
	//!   m_fs_flag     : redundant copy of chunk_header.size_info's
	//!                   FS=true/false discriminator (true => FS=true).
	//!                   On dealloc fast path it picks the local-id source:
	//!                   FS=true => local id 0 (chunk serves 1 size);
	//!                   FS=false => the slot prefix at p-8 carries it.
	//!
	//! FREELIST INDEXING (§12.3 — design "alloc: bucket-id direct;
	//! dealloc: local-id direct; no conversion-table on hot paths"):
	//!   m_freelist_head[] is indexed by COMPACT per-chunk LOCAL id, not
	//!   global bucket.  KAME_LOCAL_BUCKETS (= 9) covers the widest tier
	//!   (ALIGN=256 FS=false serves buckets {16, 32..39}).  `kBucketLocalId[]`
	//!   maps bucket -> local id — but the alloc fast path NEVER reads it:
	//!   instead, TLS `g_thread_freelist_ptr[bucket]` holds a `char **`
	//!   pointing DIRECTLY at the active chunk's m_freelist_head[local-id-
	//!   for-this-bucket], updated at chunk-switch (slow_allocate) where
	//!   kBucketLocalId IS read.  So alloc is `*tls[bucket]` (zero remap,
	//!   zero chunk-pointer-deref on hit), and dealloc gets local-id from
	//!   the chunk's m_fs_flag (FS=true) or the slot prefix (FS=false).
	//!   Both routes converge on chunk->m_freelist_head[local], single
	//!   storage; no consistency problem.
	alignas(64) uint32_t m_owner_id;
	bool      m_fs_flag;
	//! (§16) m_sizes mode discriminator/shift for the dealloc fast path.
	//!   m_sizes      : null for borrow-scheme chunks (FS=true, or FS=false
	//!                  ALIGN<1024 — per-slot {local_id,SIZE} prefix at p-8).
	//!                  Non-null for FS=false ALIGN>=1024 ("full-usable"
	//!                  mode): points at the per-chunk `m_sizes[bit]` array
	//!                  (one uint16 per bitmap bit = slot start), packed
	//!                  `(N << 8) | local_id`.  Lets a large-slot chunk hand
	//!                  out FULL `N*ALIGN`-byte slots (no 8-byte borrow
	//!                  theft) so power-of-2 page-aligned requests don't
	//!                  round up to the next size class (was 50 % waste).
	//!   m_align_shift: log2(ALIGN); fast path computes the slot's bit index
	//!                  `(p - m_mempool) >> m_align_shift` to index m_sizes.
	//! Both sit in the hot block (cache line with m_owner_id) so the
	//! borrow-mode null check is free — a borrow chunk reads m_sizes
	//! (already-loaded line), sees null, and falls to the p-8 prefix exactly
	//! as before, never touching m_mempool / the m_sizes array.
	uint8_t   m_align_shift;
	uint16_t  m_base_bucket;     // unused on hot paths; kept for diagnostics
	uint16_t *m_sizes;
	char     *m_freelist_head[KAME_LOCAL_BUCKETS];

	//! Owner-thread freelist push/pop (LIFO; freed slot's first 8 bytes
	//! hold the next pointer).  `local` is the chunk's local-id (NOT the
	//! global bucket).  No atomics — only the owner thread touches its
	//! chunk's freelists.  Both push and pop reach the SAME storage that
	//! the alloc-side TLS shortcut (`g_thread_freelist_ptr[bucket]`) is
	//! aimed at — see slow_allocate for the maintenance of that pointer.
	inline void freelist_push(unsigned local, void *p) noexcept {
		*reinterpret_cast<char **>(p) = m_freelist_head[local];
		m_freelist_head[local] = static_cast<char *>(p);
	}
	inline void *freelist_pop(unsigned local) noexcept {
		char *head = m_freelist_head[local];
		if(head)
			m_freelist_head[local] = *reinterpret_cast<char **>(head);
		return head;
	}

protected:
	// `m_mempool` retired — see `mempool()` accessor above.  Derived from
	// `this` directly because §15 pins the PoolAllocator object at
	// `chunk_base + ALLOC_CHUNK_HEADER`.

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
	//! address of the OWNER thread's `s_dll_head` TLS
	//! variable for THIS chunk's (ALIGN, FS) template.  Set by the
	//! derived `PoolAllocator<ALIGN, FS, DUMMY>` constructor to
	//! `(void *)&s_dll_head` taken in the owner thread's context.
	//!
	//! `s_dll_head` is `ALLOC_TLS` (= `__thread`); its address is
	//! per-thread (TCB base + fixed offset per template).  Comparing
	//! a stored value to `(void *)&s_dll_head` taken later identifies
	//! whether the comparing thread is the original owner.
	//!
	//! Used by the dealloc cursor-reset paths to gate
	//! `reset_dll_walk_state()`: reset only when we are the owner —
	//! resetting another thread's cursor would be wasteful no-op
	//! (their DLL is unaffected by our bitmap-clear).  Without this
	//! gate, alloc_stress's 50%-cross-thread workload paid a 10-20%
	//! perf tax on Linux from spurious cursor resets.
	void *m_owner_dll_head_addr = nullptr;

	//! pointer to owner thread's "force DLL re-walk" hint
	//! flag (TLS `std::atomic<bool>` per PoolAllocator template).
	//! Cross-thread frees set this so the owner's next
	//! `allocate_chunk_path` notices that one of its DLL chunks got
	//! a bitmap clear since the last walk and force-restarts the
	//! walk from `s_dll_head` instead of resuming from a stale
	//! `s_dll_cursor`.
	//!
	//! declared `std::atomic<std::atomic<bool> *>` (atomic
	//! pointer to atomic bool) so that owner-exit
	//! (`release_dll_chunks_for_thread`) and concurrent cross-thread
	//! frees on surviving chunks do not data-race on plain pointer
	//! access.  Owner-exit stores `nullptr` with `release` BEFORE
	//! clearing BIT_OWNED; cross-thread freers load with `acquire`
	//! and skip the deref when null.  The 1000-thread `alloc_stress`
	//! Linux SEGV that an earlier change exhibited (1000 thr × 20K × 30 %cross)
	//! is fixed by this ordering — without atomic load/store on the
	//! pointer, owner-exit's plain `= nullptr` racing with a cross-
	//! thread freer's plain `->store(...)` was UB and crashed on Linux.
	//!
	//! Inner `store(true)` (cross-thread → owner) and the owner's
	//! `exchange(false)` in `allocate_chunk_path` remain
	//! `memory_order_relaxed` — the hint itself doesn't require
	//! synchronisation, only the outer pointer's lifetime does.
	std::atomic<std::atomic<bool> *> m_owner_dll_force_walk_ptr{nullptr};

	//! runtime cap on the number of mmap regions
	//! `allocate_chunk` may claim.  Initialised to
	//! `ALLOC_MAX_REGIONS` (= no further restriction beyond the
	//! compile-time ceiling) and overridable at runtime via
	//! `kame_pool_set_max_bytes()` in allocator.h.  Loaded relaxed
	//! on the cold mmap path; never read on the alloc/free hot path.
	static std::atomic<int> s_max_regions_cap;

	//! (§21) Whether the thread-exit DLL teardown
	//! (`release_dll_chunks_for_thread`) madvise's the slot pages of the
	//! chunks it releases.  Default TRUE — RSS is returned promptly when a
	//! thread exits, closing the only "advise-skip" exception in the
	//! release protocol (mid-run releases always madvise).  Set FALSE via
	//! `kame_pool_set_thread_exit_reclaim(0)` to restore the prior
	//! perf-optimised behaviour (skip the ~30 % thread-exit madvise cost
	//! and let the kernel batch-reclaim at process exit) for workloads
	//! that spawn/exit threads rapidly and don't care about steady-state
	//! RSS.  Loaded relaxed on the (cold) thread-exit path only.
	static std::atomic<int> s_thread_exit_reclaim;

	//! read-only accessor used by `kame_pool_reserved_bytes`.
	//! (§13.3) O(1): returns the live populated-region counter maintained
	//! by `mmap_new_region` (was an O(N) walk of the retired
	//! `s_mmapped_spaces[]`).
	static std::size_t populated_region_count() noexcept {
		int n = s_region_count.load(std::memory_order_relaxed);
		return n > 0 ? (std::size_t)n : 0;
	}

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
	//!     `uint64_t`, 2 words per region.
	//!   * Hosts without DCAS — fall back to `uint32_t`, 4 words per
	//!     region.  Requires single-word atomic<uint32_t> to be
	//!     always-lock-free (true on every architecture this allocator
	//!     supports).
	//!
	//! 1-bit encoding per UNIT.  Each 256-KiB unit owns ONE bit:
	//!   * 0: free — unit is unclaimed.
	//!   * 1: claimed (base OR continuation unit of a chunk).
	//! A multi-unit chunk (CHUNK_UNITS = 2 or 4) sets CHUNK_UNITS
	//! adjacent bits in a single atomic CAS at claim time.  Which unit
	//! holds the chunk_header — and whether the chunk is fully built —
	//! is no longer encoded in the bitmap: `s_back_offset[]` carries the
	//! back-offset, and released/foreign is read from chunk_header.palloc
	//! (the recycle epoch and the WIP seqlock are retired; epoch == 0 was
	//! yet built), subsuming the retired `ready` bit.  Total bits per
	//! region: 128 units × 1 = 128.
#if ATOMIC_LLONG_LOCK_FREE == 2 && !defined(KAME_FORCE_UINT32_BITMAP)
	using BitmapWord = uint64_t;
	static constexpr int BITMAP_WORDS_PER_REGION = 2;
#else
	using BitmapWord = uint32_t;
	static constexpr int BITMAP_WORDS_PER_REGION = 4;
	static_assert(ATOMIC_INT_LOCK_FREE == 2,
		"atomic<uint32_t> must be always-lock-free as the fallback "
		"bitmap word type — targets without 32-bit atomic CAS are "
		"not supported.");
#endif
	static constexpr int BITS_PER_BITMAP_WORD = int(sizeof(BitmapWord) * 8);
	static constexpr int UNITS_PER_BITMAP_WORD =
	    BITS_PER_BITMAP_WORD;         // 1 bit per unit
	static_assert(UNITS_PER_BITMAP_WORD * BITMAP_WORDS_PER_REGION
	                 == NUM_ALLOCATORS_IN_SPACE,
		"bitmap layout (1 bit per unit) must cover all 128 units per region");

	//! (§13.2) Per-region metadata block, embedded at `region_base + 0`
	//! (i.e. inside unit 0 of every mmap region).  Replaces two of the
	//! `ALLOC_MAX_REGIONS`-sized globals — `s_claim_bitmap[]` and
	//! `s_back_offset[]` — with a struct attached to each region.  Self-
	//! contained per-region metadata lets the populated-region count
	//! scale without growing fixed BSS, paving the way to retire
	//! ALLOC_MAX_REGIONS entirely (HPC scaling step 3).
	//!
	//! Layout (144 B total on both 64- and 32-bit):
	//!   claim_bitmap[BITMAP_WORDS_PER_REGION]  // 16 B
	//!   back_offset [NUM_ALLOCATORS_IN_SPACE]  // 128 B
	//!
	//! Lives in unit 0 of the region (256 KiB virtual; only the first
	//! page = 4 KiB is touched and committed by the kernel).  Unit 0 is
	//! permanently marked claimed (bit 0 of claim_bitmap set at region
	//! init), so allocate_chunk's bitmap scan naturally skips it.  Cost:
	//! 1/128 = 0.78 % of region VA, ~4 KiB committed RSS per region —
	//! both negligible vs the BSS savings (3200 × 144 B = 450 KiB BSS
	//! returned to the rodata-free pool).
	//!
	//! Concurrency: claim_bitmap entries use atomic CAS (same as the
	//! retired global s_claim_bitmap).  back_offset bytes are plain
	//! (written by the claim path BEFORE the claim-bit CAS publishes
	//! the chunk; the bit's release semantics carries them).  At region
	//! init, mmap zero-fills the page, then `init_region_meta` sets bit
	//! 0 of claim_bitmap before publishing the region to other threads
	//! via the s_mmapped_spaces[ccnt] release-CAS — readers' acquire on
	//! that slot synchronizes-with this init.
	struct RegionMeta {
		std::atomic<BitmapWord> claim_bitmap[BITMAP_WORDS_PER_REGION];  // 16 B
		std::uint8_t            back_offset[NUM_ALLOCATORS_IN_SPACE];   // 128 B
		//! (§13.3) Per-NUMA-node region list — singly-linked, PUSH-ONLY at
		//! head of `s_region_dll_heads[numa_node]`.  Regions are never
		//! unmapped in the current design, so the list needs no removal /
		//! reclamation: walkers never see a freed node, and a lock-free
		//! Treiber push is the only mutation.
		std::atomic<RegionMeta *> dll_next;
		//! (§13.3) "may have a free chunk slot" hint (1 = maybe free).
		//! Set at region init and by `deallocate_chunk`; cleared
		//! (tentatively) by a claim path that finds no slot for ITS
		//! CHUNK_UNITS.  Race-tolerant best-effort hint.
		std::atomic<unsigned char> has_free;
		//! (§14C) NUMA node the region is bound to (== the node of the
		//! thread that mmap'd it).  Used by claim-side walkers to start
		//! with their local node's list before falling back to others, and
		//! by `mbind(MPOL_BIND, {numa_node})` at region create.  0 on
		//! single-node systems / non-Linux (no NUMA info).
		std::uint16_t numa_node;
	};
	// RegionMeta lives in the first page (4 KiB) of unit 0; it only needs
	// to FIT, not hit an exact size.  144 B of arrays + the list/hint
	// fields, well under a page.
	static_assert(sizeof(RegionMeta) <= 4096,
	              "RegionMeta must fit in the first page of region unit 0");

	//! (§19) Large-alloc metadata.  Lives in the first page of a §19
	//! large-alloc's 32-MiB-aligned mmap region; the user pointer is
	//! `base + ALLOC_PAGE_SIZE`.  The radix slot at `base >> 25` carries
	//! KAME_RADIX_LARGE so deallocate / size_of dispatch this struct
	//! instead of RegionMeta.
	//!
	//! On free: clear the radix slot (CAS), then `munmap(base, mmap_size)`.
	//! The radix clear happens BEFORE the munmap so any racing reader on
	//! the old base either sees the live slot (and reads valid meta) or
	//! sees absent (and falls through to libc free) — never a torn read
	//! against an unmapped page.
	struct LargeAllocMeta {
		std::uint64_t magic;       // sentinel; debug-time identity check
		std::size_t   alloc_size;  // user-requested bytes (post-page-rounding still ≥ this)
		std::size_t   mmap_size;   // bytes mmap'd at base (incl. meta page)
		std::uint16_t numa_node;   // bound NUMA node (0 if none/unknown)
	};
	static constexpr std::uint64_t KAME_LARGE_ALLOC_MAGIC =
	    0xCAFEBABEDEADBEEFull;
	static_assert(sizeof(LargeAllocMeta) <= 4096,
	              "LargeAllocMeta must fit in the first page of its mmap");

	//! (§19) Recover the LargeAllocMeta from any pointer inside the
	//! alloc.  The 32-MiB-aligned base IS the meta address (radix
	//! lookup confirms the slot kind is KAME_RADIX_LARGE).  Caller must
	//! have already verified `radix_lookup(p) == KAME_RADIX_LARGE`.
	static inline LargeAllocMeta *large_alloc_meta_of(void *p) noexcept {
		return reinterpret_cast<LargeAllocMeta *>(
		    (uintptr_t)p & ~((uintptr_t)ALLOC_MIN_MMAP_SIZE - 1u));
	}

	//! (§19) Allocate from the §19 large-alloc tier — a single
	//! 32-MiB-aligned mmap registered as one radix slot.  Returns the
	//! user pointer (= mmap_base + ALLOC_PAGE_SIZE) or nullptr if the
	//! mmap or radix insert fails.  Caller guarantees
	//! `size <= ALLOC_MIN_MMAP_SIZE - ALLOC_PAGE_SIZE` (one slot fit).
	static void *allocate_large_va(std::size_t size) noexcept;

	//! (§19) Free a §19 large-alloc.  `p` is the user pointer; caller has
	//! already confirmed it's a KAME_RADIX_LARGE allocation via
	//! `radix_lookup`.  Clears the radix slot then munmap's the region.
	static void deallocate_large_va(void *p) noexcept;

	//! Recover a region's metadata block from its base pointer.  Region
	//! base is `ALLOC_MIN_MMAP_SIZE`-aligned (= 32 MiB), so the cast is
	//! always valid for a populated region pointer.
	static inline RegionMeta *region_meta(char *mp) noexcept {
		return reinterpret_cast<RegionMeta *>(mp);
	}
	//! Region base from any in-region pointer (regions are 32-MiB-aligned).
	static inline RegionMeta *region_meta_of(void *p) noexcept {
		return reinterpret_cast<RegionMeta *>(
		    (uintptr_t)p & ~((uintptr_t)ALLOC_MIN_MMAP_SIZE - 1u));
	}

	//! (§13.3) Head of the push-only region list, and the populated
	//! region count (for the runtime cap + O(1) `populated_region_count`).
	//! (§14C) Compile-time cap on the number of NUMA nodes the allocator
	//! tracks separately.  Most multi-socket HPC nodes have 1-8 NUMA
	//! nodes; 16 covers larger Intel/AMD/Fujitsu systems with room to
	//! spare.  The runtime detected count `s_num_numa_nodes` is clamped
	//! to this; nodes beyond fold into node `KAME_MAX_NUMA_NODES - 1`.
	//! Cost: `KAME_MAX_NUMA_NODES × 8 B` = 128 B of BSS for the per-node
	//! list-head array (one cache line on a 128 B-line target).
	static constexpr int KAME_MAX_NUMA_NODES = 16;

	//! (§14C) Per-NUMA-node region list head + populated-region counter.
	//! On non-Linux / single-node Linux, only index 0 is used (matches
	//! the single-list §13.3 behavior).  On multi-node systems, each
	//! region is bound to its creator thread's local node via mbind and
	//! pushed onto that node's list, so chunk-claim Pass 1 finds the
	//! freshest LOCAL region first (locality preserved) without touching
	//! foreign-NUMA cache lines.
	static std::atomic<RegionMeta *>
	    s_region_dll_heads[KAME_MAX_NUMA_NODES];
	static std::atomic<int>          s_region_count;

	//! Number of NUMA nodes the allocator tracks (1..KAME_MAX_NUMA_NODES).
	//! Detected at first call via `/sys/devices/system/node/` on Linux;
	//! 1 elsewhere.  Lazily initialised by `numa_node_for_this_thread`.
	static std::atomic<int> s_num_numa_nodes;

	//! Current thread's preferred NUMA node (lazy-initialised on first
	//! call from `sched_getcpu()` + the kernel's CPU→node mapping).  Used
	//! by `mmap_new_region` (which mbind's the region to this node) and
	//! by claim-path Pass 1 (which walks this node's list first).
	//! A thread that later migrates to a different node keeps the
	//! original assignment — slight locality drift, no correctness issue.
	static int numa_node_for_this_thread() noexcept;

	//! mmap a fresh 32-MiB-aligned region, init its RegionMeta (reserve
	//! unit 0, has_free=1), register it in the radix tree, and push it on
	//! the region list.  Returns the new region's metadata (== its base
	//! pointer) or nullptr on cap-exceeded / mmap failure.  Shared by both
	//! chunk-claim Pass-2 sites (`allocate_chunk<ALLOC>` and `claim_chunk`).
	static RegionMeta *mmap_new_region() noexcept;

private:
	//! (§13.3) `s_mmapped_spaces[]` retired — region bases are derived
	//! directly from any in-region pointer (`p & ~(ALLOC_MIN_MMAP_SIZE-1)`,
	//! regions are 32-MiB-aligned) and enumerated via the push-only region
	//! list (`s_region_dll_head`).  This removes the last cap-sized global
	//! and with it ALLOC_MAX_REGIONS — the region count is now bounded
	//! only by VA (the radix covers 47-bit = 128 TiB) and the optional
	//! runtime cap `s_max_regions_cap`.

	//! 2-level radix tree top level (§13).  L1 indexed by upper
	//! RADIX_L1_BITS of a pointer's region index; entries point at a
	//! lazily-mmap'd RadixL2Node (or null).  The full layout doc is on
	//! `RadixL2Node` above.  Used by `radix_lookup(p)` to test in O(1)
	//! whether `p` falls in a populated region (slot value: 0 = absent,
	//! non-zero = present) — replacing the former O(populated-N) linear
	//! walk over `s_mmapped_spaces[]`.  Since §13.3 the slot is a pure
	//! presence token (region base is derived from `p`, not from a ccnt
	//! index), so no per-region id is stored.
	static std::atomic<RadixL2Node *> s_radix_l1[RADIX_L1_SIZE];

	//! Allocate a zero-init L2 node directly via mmap.  CANNOT use libc
	//! malloc here: kamepoolalloc interposes `malloc` / `free`, and a
	//! libc-malloc call from inside the region-claim path would
	//! recurse back into `kame_malloc` → `allocate_chunk` →
	//! `radix_insert` → libc malloc → ... .  Returns nullptr on OOM
	//! (extremely unlikely; one 8-KiB mmap per ~16 GiB of populated
	//! pool).
	static RadixL2Node *radix_alloc_l2() noexcept;

	//! Mark region base `mp` present in the radix with the given kind
	//! (KAME_RADIX_POOL for normal pool regions from `mmap_new_region`,
	//! KAME_RADIX_LARGE for §19 large-alloc registrations).
	//! Concurrent-safe: L1-leaf install via CAS (loser munmap's its
	//! leaf); the per-region presence slot is set atomically.
	static void radix_insert(char *mp, uint32_t kind) noexcept;

	//! (§19) Clear the radix slot for base `mp` (CAS-back-to-zero).
	//! Caller must ensure no concurrent reader can be DEREF'ing the meta
	//! by the time this returns — for §19 large allocs, the slot is
	//! cleared BEFORE the backing mmap is unmapped, so a racing reader
	//! either sees the live slot+meta or sees absent and falls through
	//! to libc free.
	static void radix_clear(char *mp) noexcept;

	//! Out-of-line full radix walk.  Called only on cache miss.  Returns
	//! the slot's kind (KAME_RADIX_POOL / KAME_RADIX_LARGE) if present,
	//! or KAME_RADIX_ABSENT (0) if absent.  Updates `s_last_region_base`
	//! ONLY for pool regions so a large-alloc lookup never poisons the
	//! cache (the large alloc's base disappears on munmap; another
	//! thread's stale cache entry would falsely report present).
	static int radix_lookup_slow(uintptr_t up) noexcept;

	//! O(1) lookup: return the radix slot's kind (KAME_RADIX_POOL,
	//! KAME_RADIX_LARGE) if `p` falls in a populated region, else
	//! KAME_RADIX_ABSENT (= 0).
	//! Inlined into deallocate / size_of / lookup_chunk to keep the
	//! dealloc fast path tight.
	//!
	//! Fast path = 1-entry per-thread "last region" cache
	//! (`s_last_region_base`, IE TLS).  Cache holds only pool regions
	//! (§19), so a cache hit guarantees KAME_RADIX_POOL.  A
	//! dealloc usually touches the same region as the previous one
	//! (most workloads: alloc/dealloc bursts hit one chunk's worth of
	//! VA; HPC: ditto, plus loop-carried locality).  Hit cost: two IE
	//! TLS loads + bitmask + compare = ~3 cycles.  Compare against the
	//! full radix walk (~10 cycles, two chained dependent loads).
	//!
	//! Cache miss: full walk via `radix_lookup_slow` (out-of-line so the
	//! inlined version stays icache-cheap).  Misses also update the
	//! cache (for pool regions only).
	static inline int radix_lookup(void *p) noexcept {
		uintptr_t up = (uintptr_t)p;
		uintptr_t base = up & ~((uintptr_t)ALLOC_MIN_MMAP_SIZE - 1u);
		// (§19) Cache holds ONLY pool region bases, so a hit means
		// KAME_RADIX_POOL.  base 0 (init) never matches (kernel null
		// page is never a region base).
		if(__builtin_expect(base == s_last_region_base, 1))
			return (int)KAME_RADIX_POOL;
		return radix_lookup_slow(up);
	}

	//! Per-thread 1-entry region-lookup cache (the last PRESENT region
	//! base confirmed by `radix_lookup_slow`).  Initialised to 0 which
	//! never matches a real region base.
	static ALLOC_TLS_IE uintptr_t s_last_region_base;

public:
	//! (§13.2) The former global `s_claim_bitmap[]` and `s_back_offset[]`
	//! are gone — both moved into per-region `RegionMeta` at
	//! `region_base + 0` (= unit 0 of every region; bit 0 of the
	//! bitmap permanently set to keep allocate_chunk away).  Callers
	//! reach them via `region_meta(mp)->claim_bitmap[word]` and
	//! `region_meta(mp)->back_offset[unit_idx]`.  BSS savings: ~450 KiB
	//! on 64-bit (3200 × (16+128)).  Per-region commit: 1 page = 4 KiB.

	//! (§13.3) The former global `s_region_has_free[]` skip-bitmap is
	//! retired — the per-region "has free chunk space" hint now lives in
	//! `RegionMeta::has_free`, and the chunk-claim Pass-1 walks the
	//! push-only region list (`s_region_dll_head`) checking each region's
	//! hint.  New regions are pushed at the list head, so the freshest
	//! (most likely to have free space) are visited first and the walk
	//! usually returns within a few nodes.  Same lazy best-effort
	//! semantics as the old bitmap bit (set on fresh-region init + on
	//! chunk free; tentatively cleared when a claim finds no slot for its
	//! CHUNK_UNITS).
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
	//! an earlier change buddy tiers.  Larger ALIGN gets larger chunks so the
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
		return static_cast<PoolAllocatorBase *>(s_tls.my_chunk);
	}
	//! public reset of this thread's DLL walk hints
	//! (`s_dll_cursor` + `s_dll_exhausted`) for callers outside the
	//! PoolAllocator class hierarchy — specifically `CrossDeallocBatch::
	//! push_direct` (anon-namespace, no inheritance) which calls
	//! `batch_return_to_bitmap` directly on the freeing thread and
	//! needs to signal "DLL may have revived chunks" to this thread's
	//! subsequent `allocate_chunk_path`.  See the an earlier change commit /
	//! the call sites in `deallocate_pooled` for the full rationale.
	static void reset_dll_walk_state() noexcept {
		s_tls.dll_cursor = nullptr;
		s_tls.dll_exhausted = false;
	}
	//! (§24) Scan this thread's DLL of chunks for one whose freelist at
	//! `local_id` is non-empty; if found, re-pin it as `s_tls.my_chunk`,
	//! pop one slot from its freelist, and return the popped pointer.
	//! Returns nullptr if no chunk has a freelist entry at this local id.
	//! Inside the class so it can access the protected `m_dll_next` and
	//! `m_freelist_head` fields of the same template's chunks.  Used by
	//! `slow_allocate` (FS=true and FS=false) before falling through to
	//! the bitmap-claim path — without it, freelist entries on
	//! non-active chunks become unreachable on multi-chunk working sets.
	static char *scan_dll_freelist(unsigned local_id) noexcept {
		for(PoolAllocator<ALIGN, DUMMY, DUMMY> *c = s_tls.dll_head;
		    c; c = c->m_dll_next) {
			char *head = c->m_freelist_head[local_id];
			if(head) {
				s_tls.my_chunk = c;
				c->m_freelist_head[local_id] =
				    *reinterpret_cast<char **>(head);
				return head;
			}
		}
		return nullptr;
	}

	//! public accessor for this thread's `s_dll_head` TLS
	//! address.  Used by external code (CrossDeallocBatch::push_direct
	//! in anon namespace) to compare against a chunk's stored
	//! `m_owner_dll_head_addr` and identify same-thread frees for
	//! the conditional cursor reset.
	static void *dll_head_tls_addr() noexcept {
		return static_cast<void *>(&s_tls.dll_head);
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
	//! prefix at `p - ALIGN`.
	static std::size_t size_of_static(PoolAllocatorBase * /*base*/,
	                                  char * /*p*/) noexcept {
	    return ALIGN;
	}

	//! Value written to chunk_base + ALLOC_CHUNK_HEADER_SIZE_INFO_OFFSET
	//! by `allocate_chunk()`.  Layout:
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

	//! (§32) Compile-time discriminator: does this chunk template carry a
	//! cross-thread return shadow bitmap?  True for all real FS=true
	//! chunks (`<ALIGN, true, true>`), now including ALIGN=16 — the K_MAX
	//! bump (4 KiB → 8 KiB) admits the second `count`-word bitmap at the
	//! smallest slot size where the bitmap itself is largest (~2 KiB
	//! shadow on a 256 KiB chunk).  See `return_flags()`.
	static constexpr bool kHasReturnShadow = FS && DUMMY;

	//! (§32) Accessor for the cross-thread return shadow bitmap.  Layout
	//! is fixed at chunk construction (`m_flags` end → align up to 64-byte
	//! boundary), so the pointer is recomputed on demand rather than
	//! stored on the chunk — no extra field, no extra cache line.
	//!
	//! Compile-time folds:
	//!   * ALIGN >= 32, FS=true, DUMMY=true → live pointer
	//!   * everything else (ALIGN=16, FS=false, FS=true base ctor pass) →
	//!     `nullptr`, branch predicted away at the use sites.
	//!
	//! The shadow lives `alignas(64)`-separated from `m_flags` so the two
	//! bitmaps never share a CPU cacheline.  Cross-thread frees route
	//! through `fetch_or` on the shadow (cacheline B); the owner thread
	//! reconciles via
	//!   `r = return_flags()[w].exchange(0); m_flags[w] &= ~r;`
	//! at well-defined points (allocate_pooled saturation, owner_release
	//! empty check, thread-exit cleanup) — see those call sites.
	//!
	//! Placed AFTER `typedef uintptr_t FUINT;` because the return type
	//! depends on it; placed BEFORE the `protected:` block below to keep
	//! the accessor public for cross-template use sites in allocator.cpp.
	inline FUINT *return_flags() noexcept {
		if constexpr (kHasReturnShadow) {
			char *flags_end =
			    reinterpret_cast<char *>(m_flags)
			    + static_cast<size_t>(m_count) * sizeof(FUINT);
			std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(flags_end);
			std::uintptr_t aligned =
			    (addr + std::uintptr_t(63)) & ~std::uintptr_t(63);
			return reinterpret_cast<FUINT *>(aligned);
		}
		else {
			return nullptr;
		}
	}

	//! (§32) Owner-side sweep: fold cross-thread shadow bits into m_flags.
	//!
	//! Walks `return_flags[w]` for w in [0, m_count): atomically drains the
	//! word (`exchange(0)`), ANDs the freed bit pattern out of `m_flags[w]`,
	//! and updates the two counters that drive chunk lifecycle:
	//!
	//!   * `m_flags_filled_cnt` — decremented when a word transitions from
	//!     ALL-SET (saturated) to non-saturated.  Saturation is the trigger
	//!     used by allocate_pooled to decide "this word is exhausted, try
	//!     the next one"; restoring it lets the allocator find slots again.
	//!   * `m_flags_packed` MASK_CNT — decremented (batched via popcount)
	//!     when a word transitions from non-zero to zero.  MASK_CNT==0 is
	//!     the chunk-empty release trigger.
	//!
	//! Atomic semantics:
	//!   * `return_flags[w].exchange(0, ACQ_REL)` — pairs with cross-thread
	//!     `fetch_or` in `batch_return_to_bitmap` (RELEASE on the OR);
	//!     ensures any user-side stores that preceded the cross-thread
	//!     free are visible to the owner after the exchange.
	//!   * `m_flags[w]` writes are NOT atomic — owner is the sole writer
	//!     (the allocate-path bitmap-CAS only sets bits; sweep clears them).
	//!     The contract is enforced by the kHasReturnShadow gating: with
	//!     shadow on, no cross-thread CAS ever lands on m_flags.
	//!
	//! Returns true if any bits were drained (caller can use this as a
	//! hint to retry an allocate-pooled scan).
	//!
	//! Owner-only.  Calling from a non-owner thread is UB.  Call sites:
	//!   * `allocate_pooled` saturation fallback (per-chunk).
	//!   * `owner_release` empty check (per-chunk).
	//!   * `cross_release` — N/A: cross_release runs on a cross-thread
	//!     after owner exit; the shadow path is disabled there
	//!     (`batch_return_to_bitmap` falls back to direct CAS-clear on
	//!     BIT_OWNED == 0).
	//!   * `release_dll_chunks_for_thread` — pre-release final sweep on
	//!     thread exit (per chunk in this thread's DLL).
	bool sweep_return_shadow() noexcept {
		if constexpr (!kHasReturnShadow) return false;
		FUINT *rf = return_flags();
		// Cacheline-spread CAS-based sweep — owner walks return_flags in
		// cacheline-strided passes so the OWNER's CAS attempt and a
		// concurrent CROSS-THREAD `fetch_or` on the same chunk's
		// `return_flags[]` only land on the same cacheline for one
		// inner-loop step per pass.  Strong CAS (`r → 0`) on each word;
		// CAS-fail = consumer fetch_or modified the word between our
		// load and our CAS — leave the bits in the shadow, the next
		// sweep picks them up.  Sweep is "best-effort" / "drain what we
		// can" — never blocks on contention.  Per pass: m_count /
		// CL_WORDS words touched, one per cacheline.
		//
		//   pass 0 (offset=0): words   0,  8, 16, 24, …
		//   pass 1 (offset=1): words   1,  9, 17, 25, …
		//   …
		//   pass 7 (offset=7): words   7, 15, 23, 31, …
		constexpr int CL_WORDS = (64 + sizeof(FUINT) - 1) / sizeof(FUINT);
		bool any = false;
		uint32_t packed_dec = 0;
		for(int offset = 0; offset < CL_WORDS; ++offset) {
			for(int idx = offset; idx < m_count; idx += CL_WORDS) {
				FUINT r = rf[idx];                // relaxed load
				if(r == 0) continue;
				// Strong CAS: expect r, set 0.  Fail ⇒ concurrent
				// cross-thread `fetch_or` raced us; leave the bits and
				// move on (next sweep will pick them up).  This is the
				// "skip cacheline on CAS-fail" that pairs with the
				// outer cacheline-spread pass — the contended cacheline
				// is dropped from THIS pass; the next pass (different
				// offset, different word within the same cacheline) or
				// the next sweep call retries.
				if( !atomicCompareAndSet(r, FUINT(0), &rf[idx]))
					continue;
				any = true;
				FUINT oldv = m_flags[idx];
				FUINT newv = oldv & ~r;
				m_flags[idx] = newv;
				if(oldv == ~(FUINT)0u && newv != ~(FUINT)0u)
					atomicDec(&m_flags_filled_cnt);
				if(newv == 0 && oldv != 0)
					++packed_dec;
			}
		}
		if(packed_dec)
			atomicFetchSub(&m_flags_packed, packed_dec);
		return any;
	}
protected:
	PoolAllocator(int count, char *addr);
	inline void *allocate_pooled(unsigned int SIZE);
	bool deallocate_pooled(char *p) override;
	int batch_return_to_bitmap(const CrossDeallocEntry *entries) noexcept override;
	void *slow_allocate(unsigned bucket, std::size_t size) noexcept override;
	//! Mmap a fresh chunk and register it in `s_chunks_of_type[]` for
	//! diagnostic enumeration only (`release_pools` / `report_statistics`).
	//! Mmap a fresh chunk for the current thread.  no global
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
	//! `BIT_RELEASED` on the packed word is the
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
	// Packed counter + state bits (an earlier change — inverts the old
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
	//   - Owner_release uses the same
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

	//! per-template per-thread state, consolidated into one
	//! TLS struct.  Single TLS load delivers the struct base, then
	//! all per-thread fields are at compile-time offsets — cache-line
	//! adjacent and one indirection cheaper than the previous six
	//! separate `ALLOC_TLS` statics.
	//!
	//! Each PoolAllocator<ALIGN, FS, DUMMY> instantiation gets its own
	//! `ThreadLocalState` (per-template TLS); the previous "shared TLS
	//! slot via `<ALIGN, DUMMY, DUMMY>` type trick" is preserved by
	//! using `PoolAllocator<ALIGN, DUMMY, DUMMY> *` for the chunk
	//! pointers — FS=true and FS=false partial specialisations whose
	//! DUMMY axis collapses to the same type can interoperate at the
	//! pointer level.
	//!
	//! Members:
	//!   * `my_chunk`: currently pinned chunk for the fast allocate
	//!     path.  Non-null after chunk-claim success; cleared at
	//!     thread exit.
	//!   * `dll_head` / `dll_tail`: head/tail of this thread's DLL of
	//!     chunks owned by this template.  Sole source of truth for
	//!     "chunks this thread can allocate from" since an earlier change.
	//!     Single-writer (this thread); no atomic ordering needed.
	//!   * `dll_cursor`: pointer into the DLL where the
	//!     next walk should resume.  Set on successful claim; nulled
	//!     on walk-to-end / chunk-release.
	//!   * `dll_exhausted`: if true, the previous walk
	//!     reached end without finding free space — the next
	//!     `allocate_chunk_path` skips the walk and goes straight to
	//!     mmap-fresh.  Cleared by:
	//!       - new chunk append (mmap-fresh path).
	//!       - own-side `reset_dll_walk_state()` after a
	//!         `batch_return_to_bitmap`.
	//!       - cross-thread `dll_force_walk_from_head` exchange in
	//!         `allocate_chunk_path`.
	//!   * `dll_force_walk_from_head`: cross-thread
	//!     revival hint.  `relaxed atomic`; cross-thread frees set
	//!     true via the chunk's `m_owner_dll_force_walk_ptr` (which
	//!     points into THIS struct).  `allocate_chunk_path`
	//!     exchanges it back to false at entry; if it was true,
	//!     resets cursor + exhausted so the next walk restarts from
	//!     `dll_head` and visits revived chunks.
	struct ThreadLocalState {
		PoolAllocator<ALIGN, DUMMY, DUMMY> *my_chunk;
		//! (§32) The chunk that was `my_chunk` immediately before the
		//! current pinning — "1-back".  Avoid both ALLOCATING from it
		//! (in `allocate_chunk_path` Phase 2 DLL walk) and SWEEPING
		//! its `return_flags[]` shadow (in `allocate_pooled`
		//! saturation): consumer threads' cross-thread frees are
		//! statistically concentrated on the chunk producer JUST
		//! exhausted (slots allocated most recently are the freshest
		//! and most likely to be in-flight to a consumer).  Sweeping
		//! or allocating from 1-back forces the owner's cache to
		//! grab `return_flags` cachelines that the consumer is
		//! actively writing — the false-sharing the §32 shadow split
		//! was meant to avoid.  Skipping 1-back leaves the cacheline
		//! in consumer's cache; the chunk gets revisited (and swept)
		//! after one more pinning cycle when it's "2-back" and the
		//! cacheline is cold.
		PoolAllocator<ALIGN, DUMMY, DUMMY> *dll_one_back;
		PoolAllocator<ALIGN, DUMMY, DUMMY> *dll_head;
		PoolAllocator<ALIGN, DUMMY, DUMMY> *dll_tail;
		PoolAllocator<ALIGN, DUMMY, DUMMY> *dll_cursor;
		bool dll_exhausted;
		std::atomic<bool> dll_force_walk_from_head;
	};
	static ALLOC_TLS ThreadLocalState s_tls;

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

	// the previous `std::atomic<bool> m_owner_exited` lives
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
	//! FS=false slot-size lookup.  Reads SIZE from the
	//! per-slot prefix at `p - ALIGN`.  Returns the user-requested
	//! size in bytes.  Stamped into the chunk header at offset
	//! `ALLOC_CHUNK_HEADER_SIZEOF_FN_OFFSET`; overrides the FS=true
	//! constant returned by the base template's `size_of_static`.
	static std::size_t size_of_static(PoolAllocatorBase *base, char *p) noexcept;
	//! FS=false chunk-header SIZE info.  Low 32 bits = 0 so
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
	PoolAllocator(int count, char *addr);
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

	// m_sizes and m_available_bits dropped.  Per-slot SIZE
	// metadata now lives in the slot's own first ALIGN bytes (the
	// "+1 prefix" — bitmap claims N+1 bits, slot[0..3] stores SIZE as
	// uint32_t, returned pointer is `slot_start + ALIGN`).
	// an earlier change borrow scheme moved this to p_user - 8.
	//
	// also dropped the an earlier change 80% fragmentation cutoff +
	// the brief an earlier change `m_bits_set` counter.  allocate_pooled
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
// extend the FS=true 16-step ladder to cover the 257..368
// gap that an earlier change left between bucket 16 (size 256) and FS=false
// bucket 17's max user (376).  Zero-frag for power-of-16 requests
// like 272, 288, 304, 320, 336, 352, 368 at the cost of 7 new
// PoolAllocator<ALIGN=size, true> template instantiations.
#define ALLOC_SIZE17 (ALLOC_ALIGNMENT * 17)  // 272
#define ALLOC_SIZE18 (ALLOC_ALIGNMENT * 18)  // 288
#define ALLOC_SIZE19 (ALLOC_ALIGNMENT * 19)  // 304
#define ALLOC_SIZE20 (ALLOC_ALIGNMENT * 20)  // 320
#define ALLOC_SIZE21 (ALLOC_ALIGNMENT * 21)  // 336
#define ALLOC_SIZE22 (ALLOC_ALIGNMENT * 22)  // 352
#define ALLOC_SIZE23 (ALLOC_ALIGNMENT * 23)  // 368

//! Sole tail of the dispatch chain for sizes > ALLOC_MAX_BUCKETED_SIZE
//! (= 16376 bytes since an earlier change).  an earlier change covers up to 16 KiB
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

//! Bucket count.
//!   - index 0 (size = 0): reuses bucket 1's 16-B allocator
//!   - 1..23: sizes 16..368 in 16-B increments         (FS=true + FS=false mixed; an earlier change extends 1..16 to 1..23)
//!   - 24..47: 4-way exponential FS=false ladder       (sizes 384..17408 slot; 3 ALIGN stages; an earlier change shifts old 17..40 +7)
//!       24..31: ALIGN= 64, slot = 384, 448, 512, 576, 704, 832, 960, 1088   (N = 6, 7, 8, 9, 11, 13, 15, 17)
//!       32..39: ALIGN=256, slot = 1536, 1792, 2048, 2304, 2816, 3328, 3840, 4352
//!       40..47: ALIGN=1024, slot = 6144, 7168, 8192, 9216, 11264, 13312, 15360, 17408
//!   - 48..51: ALIGN=4096 page-aligned power-of-2 tier  (slot = 4096, 8192, 16384, 32768 — N = 1, 2, 4, 8)
//!             shadows the standard ladder at exact page-multiple sizes
//!             (3841..4096, 7161..8192, 15353..16384) and extends the
//!             pool up to 32760 (was libc malloc beyond 17400).
//!
//! an earlier change — N+1 shift rationale:  in an earlier change (N ∈ {5..16}), a
//! round-number user request like 1024 B routed to bucket 25
//! (slot 1280, internal frag 256 = 25 %) because the natural fit at
//! bucket 24 (slot 1024) had user_cap = 1024-8 = 1016 — one byte too
//! small for the borrow header.  Bumping every bucket's N by 1 makes
//! bucket K's user_cap = (N+1)*ALIGN - 8 = old_slot + ALIGN - 8, so
//! 1024-byte requests now fit in bucket 24 (slot 1088, frag 64 = 6 %).
//!
//! Internal-frag improvement for power-of-2 round-number requests:
//!   user 1024:  25 % → 6 %   (bucket 25 → 24)
//!   user 2048:  25 % → 12 %  (bucket 29 → 28)
//!   user 4096:  25 % →  6 %  (bucket 33 → 32)
//!   user 8192:  25 % → 12 %  (bucket 37 → 36)
//!
//! an earlier change rationale: the earlier change's bucket 17 (slot 384) absorbed user
//! requests 257..376 with up to 32 % internal frag for the smaller end.
//! an earlier change extends the FS=true 16-step ladder seven positions
//! (sizes 272, 288, 304, 320, 336, 352, 368) so each gets a zero-frag
//! template instantiation (PoolAllocator<ALIGN=size, true>).  FS=false
//! ladder shifts +7 (old 17..40 → new 24..47).  ALLOC_NUM_BUCKETS:
//! 41 → 48.
//!
//! an earlier change frag improvements vs //!   user  272:  32 % → 0 %   (bucket 17 slot 272)
//!   user  320:  17 % → 0 %   (bucket 20 slot 320)
//!   user  368:   2 % → 0 %   (bucket 23 slot 368)
//! Buckets 48..51 (ALIGN=4096, FS=false, N=1/2/4/8 → slot 4K/8K/16K/32K)
//! are the page-aligned tier: every slot size is a power of 2 and a
//! multiple of every common page size (4 KiB, 16 KiB, 32 KiB, 64 KiB),
//! so MADV_FREE/DONTNEED, TLB-coverage and THP behaviour stay clean on
//! every supported page-size machine.  See `bucket_for_size` below for
//! the routing rules; the new buckets shadow the existing octave/sub
//! ladder only at the page-aligned sizes (4096, 8192, 16384) and extend
//! the upper end from 17408 to 32768 — sizes 17409..32768 previously
//! fell through to libc malloc and now use bucket 51.
constexpr int ALLOC_NUM_BUCKETS = 52;
// kBucketLocalId[] below is hardcoded to ALLOC_NUM_BUCKETS entries (the
// FS=false prefix's local-id field width also implicitly bounds this).
static_assert(ALLOC_NUM_BUCKETS == 52,
              "kBucketLocalId[52] must match ALLOC_NUM_BUCKETS");

//! Size → bucket-index.  FS=true/mixed range (1..368) uses the 16-byte
//! step formula.  FS=false range
//! (369..17400) uses the an earlier change N+1-shifted 4-way exponential ladder
//! with bucket indices shifted +7.  (§16) Sizes 17409..32768 route to the
//! ALIGN=4096 N=8 bucket 51 (slot 32768), now FULL-usable so the bucket
//! serves the entire 32768-byte slot (was 32760 under the borrow scheme).
//!
//! Algorithm: compute the borrow-tier bucket, and for the full-usable
//! tier (ALIGN>=1024) recompute with total=size; then page-aligned
//! overrides.  `kBucketNewSlot[]` gives each bucket's slot for the
//! step-down compare.
constexpr std::size_t ALLOC_MAX_BUCKETED_SIZE = 32768u;

//! per-bucket NEW slot size.  Indexed by bucket K.
//!   * Buckets 1..23: 16-step (FS=true + mixed FS=false).  Slot = K*16.
//!   * Buckets 24..47: FS=false N+1-shifted ladder, slot = (N+1)*ALIGN.
//!   * Buckets 48..51: ALIGN=4096 FS=false power-of-2 tier.
//!
//! Used by `bucket_for_size` to test `total <= kBucketNewSlot[K-1]`
//! (= "user fits in the bucket below") on the FS=false range step-down.
inline constexpr uint32_t kBucketNewSlot[52] = {
    // 0..23: 16-step.  Even-K buckets in 6..16 are actually FS=false
    // ALIGN=32 chunks (see KAME_DECL_BUCKET) but slot total = K*16
    // from the dispatch table's view.  Buckets 17..23
    // are FS=true with ALIGN=size, slot = size exactly.
    0, 16, 32, 48, 64, 80, 96, 112, 128,
    144, 160, 176, 192, 208, 224, 240, 256,
    272, 288, 304, 320, 336, 352, 368,
    // 24..31: ALIGN=64, N+1 ∈ {6, 7, 8, 9, 11, 13, 15, 17}.
    384, 448, 512, 576, 704, 832, 960, 1088,
    // 32..39: ALIGN=256.
    1536, 1792, 2048, 2304, 2816, 3328, 3840, 4352,
    // 40..47: ALIGN=1024.
    6144, 7168, 8192, 9216, 11264, 13312, 15360, 17408,
    // 48..51: ALIGN=4096 page-aligned tier, N ∈ {1, 2, 4, 8}.
    4096, 8192, 16384, 32768,
};

//! Global bucket -> COMPACT per-chunk local size id (§12.3).  A chunk ==
//! one `PoolAllocator<ALIGN,FS>` template, and all same-(ALIGN,FS)
//! buckets share its `s_tls.my_chunk`, so one chunk hands out exactly
//! that template's size set.  The local id is a bucket's position WITHIN
//! that set (0-based), used to index the dense
//! `PoolAllocatorBase::m_freelist_head[KAME_LOCAL_BUCKETS]` and stored in
//! the FS=false slot prefix.  A BucketTraits-derived static_assert in
//! allocator.cpp VERIFIES, at compile time, that every id here equals
//! "#lower buckets sharing my (ALIGN,FS)", which guarantees (a) two
//! sizes in one chunk never share a freelist head and (b) every id
//! < KAME_LOCAL_BUCKETS.  Tiers:
//!   FS=true (each its own ALIGN=size, one size)        -> id 0
//!   ALIGN=32  FS=false: buckets {6,8,10,12,14}         -> 0..4
//!   ALIGN=64  FS=false: buckets {24..31}               -> 0..7
//!   ALIGN=256 FS=false: buckets {16, 32..39}           -> 0, 1..8
//!   ALIGN=1024 FS=false: buckets {40..47}              -> 0..7
//!   ALIGN=4096 FS=false: buckets {48..51}              -> 0..3
//!
//! USED ONLY ON THE COLD PATH (slow_allocate / cold_first_access) to set
//! the per-thread TLS `g_thread_freelist_ptr[bucket]` shortcut so the
//! alloc HOT path needs no table lookup; dealloc HOT path reads the
//! local-id directly from chunk.m_fs_flag (FS=true) or the slot prefix
//! (FS=false).  See §12.3.
inline constexpr uint8_t kBucketLocalId[52] = {
    // 0      (size 0 reuses bucket 1's allocator; never an FS=false slot)
    0,
    // 1..5   FS=true
    0, 0, 0, 0, 0,
    // 6 FS=false ALIGN=32 #0 ; 7 FS=true ; 8 ALIGN=32 #1 ; 9 FS=true ;
    // 10 #2 ; 11 FS=true ; 12 #3 ; 13 FS=true ; 14 #4 ; 15 FS=true ;
    // 16 FS=false ALIGN=256 #0
    0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 0,
    // 17..23 FS=true
    0, 0, 0, 0, 0, 0, 0,
    // 24..31 FS=false ALIGN=64
    0, 1, 2, 3, 4, 5, 6, 7,
    // 32..39 FS=false ALIGN=256 (#1..8; #0 is bucket 16 above)
    1, 2, 3, 4, 5, 6, 7, 8,
    // 40..47 FS=false ALIGN=1024
    0, 1, 2, 3, 4, 5, 6, 7,
    // 48..51 FS=false ALIGN=4096 (N=1,2,4,8 → slot 4K/8K/16K/32K)
    0, 1, 2, 3,
};

//! (§17) Per-bucket pointer-alignment guarantee, in bytes.  Every slot in
//! bucket K is `kBucketAlign[K]`-aligned because the slot region starts at
//! a 256 KiB unit boundary (a multiple of every entry) and slot j sits at
//! `mempool + j*ALIGN`.  Used by `bucket_for_aligned` to route an
//! over-aligned (alignment > ALLOC_ALIGNMENT) request to the smallest
//! bucket whose ALIGN is a multiple of the requested alignment.
//!
//! Power-of-two entries only matter for the aligned path; FS=true buckets
//! with non-power-of-two ALIGN (3, 5, 7, 9, 11, 13, 15, 17..23) are still
//! listed truthfully and are filtered out by the `(al & (A-1)) == 0`
//! divisibility test for the (always power-of-two) caller alignment.
inline constexpr uint16_t kBucketAlign[52] = {
    // 0: size=0 reuses bucket 1 (ALIGN=16)
    16,
    // 1..5 FS=true: ALIGN = SIZE = K*16
    16, 32, 48, 64, 80,
    // 6 FS=false ALIGN=32 ; 7 FS=true ALIGN=112 ; 8 FS=false ALIGN=32 ;
    // 9 FS=true ALIGN=144 ; 10 ALIGN=32 ; 11 ALIGN=176 ; 12 ALIGN=32 ;
    // 13 ALIGN=208 ; 14 ALIGN=32 ; 15 ALIGN=240 ; 16 FS=false ALIGN=256
    32, 112, 32, 144, 32, 176, 32, 208, 32, 240, 256,
    // 17..23 FS=true ALIGN=SIZE (272..368)
    272, 288, 304, 320, 336, 352, 368,
    // 24..31 FS=false ALIGN=64
    64, 64, 64, 64, 64, 64, 64, 64,
    // 32..39 FS=false ALIGN=256
    256, 256, 256, 256, 256, 256, 256, 256,
    // 40..47 FS=false ALIGN=1024 (§16 full-usable)
    1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
    // 48..51 FS=false ALIGN=4096 (§16 full-usable)
    4096, 4096, 4096, 4096,
};

//! Usable bytes of bucket K — what `malloc_usable_size` returns for a slot
//! handed out by K.  Equals `kBucketNewSlot[K]` for FS=true (ALIGN=slot)
//! and the §16 full-usable tier (ALIGN>=1024); `kBucketNewSlot[K] - 8` for
//! the FS=false borrow tier (ALIGN<1024 with slot>ALIGN, i.e. the per-slot
//! prefix steals 8 B from the slot's tail).
inline constexpr std::size_t kame_bucket_usable(unsigned int b) noexcept {
    std::size_t slot = (b < 52u) ? kBucketNewSlot[b] : 0u;
    if(b == 0u || b >= 52u) return 0u;
    if(kBucketAlign[b] >= 1024u) return slot;          // §16 full-usable
    if((std::size_t)kBucketAlign[b] == slot) return slot;  // FS=true (ALIGN=slot)
    return slot - 8u;                                  // FS=false borrow
}

//! (§16) 4-way exponential octave/sub ladder bucket for a "total" (the
//! slot the request must fit into).  Returns an UN-stepped bucket index
//! in 24..(>47).  Shared by the borrow-tier and full-tier passes of
//! `bucket_for_size`, which differ only in `total` (size+8 vs size) and
//! the step-down / tier handling.
inline constexpr unsigned int kame_ladder_bucket(std::size_t total) noexcept {
	int msb = 63 - __builtin_clzll(static_cast<unsigned long long>(total));
	int sub = static_cast<int>((total >> (msb - 2)) & 0x3u);
	std::size_t mask = (std::size_t(1) << (msb - 2)) - 1u;
	if(total & mask) ++sub;
	return 23u + static_cast<unsigned int>((msb - 8) * 4 + sub);
}

inline constexpr unsigned int bucket_for_size(std::size_t size) noexcept {
	// FS=true / mixed range: 1..368, 16-B step.  (size+15)>>4 yields
	// 1..23 for size 1..368, and 0 for size==0 (reuses bucket 0's 16-B
	// allocator).
	if(size <= (std::size_t)ALLOC_SIZE23)
		return static_cast<unsigned int>((size + 15u) >> 4);

	// FS=false range, split into two tiers (§16):
	//   * BORROW tier  (ALIGN 64/256, buckets 24..39): the per-slot
	//     {local_id,SIZE} prefix borrows the slot's last 8 B, so the slot
	//     must hold `size + 8`.
	//   * FULL-USABLE tier (ALIGN 1024/4096, buckets 40..51): metadata
	//     lives in the chunk-header m_sizes[] array, so the slot need only
	//     hold `size`.  Routing with total=size (not size+8) keeps exact
	//     power-of-2 page requests in their own bucket instead of rounding
	//     up to the next size class (the borrow scheme's 50 % waste).
	//
	// Decide the tier from the borrow-pass bucket: <=39 ⇒ borrow,
	// otherwise full.  Only the 40↔39 boundary step-down matters for the
	// split, and it needs kBucketNewSlot[39] (in-range), so the borrow
	// pass never indexes the out-of-order page-tier slots.
	unsigned int Kb = kame_ladder_bucket(size + 8u);
	if(Kb <= 40u) {
		if(Kb > 24u && (size + 8u) <= kBucketNewSlot[Kb - 1u]) --Kb;
		if(Kb <= 39u) {
			// Page bucket 48 (ALIGN=4096 full, usable 4096) beats borrow
			// bucket 39 (slot 4352, usable 4344) for 3833..4096 — exact
			// page fit + 4 KiB alignment.
			if(Kb == 39u && size <= 4096u) return 48u;
			return Kb;
		}
		// Kb == 40 → size in (4344, 6144]; fall through to the full tier.
	}

	// FULL-USABLE tier: recompute with total = size.
	unsigned int Kf = kame_ladder_bucket(size);
	if(Kf > 47u) {
		// Beyond bucket 47 (slot 17408): the page bucket 51 (slot 32768)
		// extends the pool to 32768; larger sizes route to the dedicated
		// chunk / libc path (ALLOC_NUM_BUCKETS = "no bucket").
		return (size <= 32768u) ? 51u : ALLOC_NUM_BUCKETS;
	}
	// Full-tier step-down, bounded to stay within the full tier (Kf-1 >=
	// 40) so it never crosses into the borrow tier (whose usable is slot-8,
	// not slot, making a cross-tier kBucketNewSlot compare unsound).
	if(Kf > 40u && size <= kBucketNewSlot[Kf - 1u]) --Kf;
	// Page bucket 50 (slot 16384) beats full bucket 47 (slot 17408) for
	// 15361..16384.  (Bucket 49 ties full bucket 42 at slot 8192; plain
	// malloc stays on 42 for denser ALIGN=1024 chunks — 49 is reserved for
	// the large-alignment aligned-alloc path.)
	if(Kf == 47u && size <= 16384u) return 50u;
	return Kf;
}

//! (§17) Bucket for an over-aligned request (alignment > ALLOC_ALIGNMENT,
//! always power-of-two by caller contract).  Returns the smallest bucket
//! whose ALIGN is a multiple of `alignment` AND whose usable size covers
//! `size`; or `ALLOC_NUM_BUCKETS` if none qualifies (caller falls back to
//! `posix_memalign` / dedicated chunk).
//!
//! Soundness: every pool slot is `kBucketAlign[b]`-aligned (slot region
//! starts at a 256 KiB unit boundary — a multiple of every kBucketAlign
//! entry — and slot j sits at `mempool + j*ALIGN`).  Over-alignment
//! requests up to 4096 B are served from the existing bucket tiers
//! (ALIGN ∈ {32,64,256,1024,4096}); larger alignment OR larger size go to
//! `allocate_dedicated_chunk` (256 KiB-aligned payload, satisfies any A
//! up to 256 KiB) or libc `posix_memalign`.
//!
//! Cold path — called only from aligned-allocation entry points.  Linear
//! scan over 52 buckets, O(N) but invoked once per allocation; not on the
//! malloc hot path.
inline unsigned int bucket_for_aligned(std::size_t alignment,
                                       std::size_t size) noexcept {
	unsigned int best = (unsigned)ALLOC_NUM_BUCKETS;
	std::size_t best_usable = ~(std::size_t)0;
	std::size_t mask = alignment - 1u;
	for(unsigned int b = 1u; b < (unsigned)ALLOC_NUM_BUCKETS; ++b) {
		std::size_t al = kBucketAlign[b];
		// alignment must divide al (al >= alignment AND al % alignment == 0);
		// alignment is power-of-two so `al & (alignment-1) == 0` is exact.
		if(al < alignment) continue;
		if((al & mask) != 0u) continue;
		std::size_t u = kame_bucket_usable(b);
		if(u < size) continue;
		if(u < best_usable) { best_usable = u; best = b; }
	}
	return best;
}

extern ALLOC_TLS AllocSlot g_thread_slots[ALLOC_NUM_BUCKETS];

//! (§12.3) Direct-jump TLS shortcut for the alloc hot path: per-bucket
//! pointer to the active chunk's `m_freelist_head[kBucketLocalId[bucket]]`
//! cell.  Lets `new_redirected` pop with NO conversion-table read and
//! NO chunk-pointer indirection on a freelist hit — just one TLS read +
//! one indirect deref.  Maintained by `slow_allocate` / `cold_first_access`
//! (cold paths) which DO read `kBucketLocalId[]` to compute the offset.
//! nullptr = bucket not yet activated on this thread (first access or
//! post-cleanup) -> fast path falls to the cold path.  Cleared at thread
//! exit / chunk release.
//!
//! With this TLS, the previous `g_thread_chunks[]` parallel array is no
//! longer needed — the chunk's PoolAllocator object can be recovered
//! from any freelist_ptr by `chunk_from_freelist_ptr(fp)` below (one
//! mask + add), since chunks are ALLOC_MIN_CHUNK_SIZE-aligned and the
//! embed object lives at `chunk_base + ALLOC_CHUNK_HEADER`.
extern ALLOC_TLS_IE char **g_thread_freelist_ptr[ALLOC_NUM_BUCKETS];

//! (§12.3) Recover the chunk's PoolAllocator object from any pointer
//! inside the chunk's first unit — in particular, from a
//! `g_thread_freelist_ptr[bucket]` value, which points at the embed
//! object's `m_freelist_head[local-id]` cell.  Chunks are
//! `ALLOC_MIN_CHUNK_SIZE`-aligned (the per-region claim bitmap reserves
//! one bit per `ALLOC_MIN_CHUNK_SIZE`-byte unit), so masking with
//! `~(ALLOC_MIN_CHUNK_SIZE - 1)` lands on `chunk_base`; the embed object
//! is at `chunk_base + ALLOC_CHUNK_HEADER` (= +64).  Works for both
//! single- and multi-unit chunks because the embed object always lives
//! in the FIRST unit, where `fp` resides.
inline PoolAllocatorBase *
chunk_from_freelist_ptr(char **fp) noexcept {
    // (§15) With forward shift, fp sits in the PREVIOUS unit's last page
    // (PoolAllocator at chunk_base + 64 = unit_boundary[U] - K_MAX + 64,
    // so fp ∈ [unit_boundary[U] - K_MAX + 64, unit_boundary[U])).
    // Add K_MAX before masking so the round-down hits unit_boundary[U]
    // (the slot region start of THIS chunk), not unit U-1's boundary.
    uintptr_t unit_boundary = (reinterpret_cast<uintptr_t>(fp)
                                + (uintptr_t)ALLOC_CHUNK_K_MAX)
        & ~(static_cast<uintptr_t>(ALLOC_MIN_CHUNK_SIZE) - 1u);
    // chunk_base = unit_boundary - K_MAX;  PoolAllocator = chunk_base + 64
    return reinterpret_cast<PoolAllocatorBase *>(
        unit_boundary - (uintptr_t)ALLOC_CHUNK_K_MAX
        + (uintptr_t)ALLOC_CHUNK_HEADER);
}

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

//! Discovered TSD byte offset for `g_thread_slots`.  Zero means "not
//! yet initialised" (constructor hasn't run, or `pthread_key_create` /
//! sentinel scan failed); hot path falls to TLV fallback in that case.
//! `g_thread_chunks[]` was retired in (§12.3) — its TSD slot is gone
//! too.  `g_thread_freelist_ptr[]` uses direct TLV access.
extern std::size_t s_kame_slots_tsd_offset;

//! Out-of-line cold path invoked when either guard branch fails.
//! Defined in allocator.cpp.  Plants the per-thread TSD slot if
//! `s_kame_slots_tsd_offset` is set, then returns the TLV-resolved
//! address.
//!
//! `[[clang::preserve_most]]`: caller-side register-spill avoidance.
//! Without it, clang must spill live caller-saved regs across the call,
//! bloating `operator new`'s prologue with 4-6 reg saves.  preserve_most
//! shifts the burden into the cold callee (cheap — cold).
[[clang::preserve_most]] AllocSlot *kame_slots_cold() noexcept;

//! Hot accessor: returns the base of this thread's `g_thread_slots[]`.
//! Inlined into `deallocate_pooled` (alloc hot path no longer uses it —
//! it reads `g_thread_freelist_ptr[]` directly via TLV).
inline AllocSlot *kame_slots_base() noexcept {
    std::size_t off = s_kame_slots_tsd_offset;
    if(__builtin_expect(off != 0, 1)) {
        AllocSlot *p =
            *reinterpret_cast<AllocSlot **>(kame_thread_pointer() + off);
        if(__builtin_expect(p != nullptr, 1)) return p;
    }
    return kame_slots_cold();
}
#else  // !KAME_FAST_TSD: fall back to direct TLV access
inline AllocSlot *kame_slots_base() noexcept { return &g_thread_slots[0]; }
#endif

//! Cold slow path: invoked when `g_thread_chunks[bucket] == nullptr`
//! (first access on this (thread, bucket), or post-cleanup).  Handles
//! activation-flag / cleanup-flag checks, then dispatches per bucket
//! to the matching `PoolAllocator<ALIGN,FS>::allocate<SIZE>()`.
//! Defined in allocator.cpp; declared here so `new_redirected` can
//! tail-call it.
void *cold_first_access(unsigned bucket, std::size_t size) noexcept;

//! Out-of-line path for sizes larger than the inline 16-step range
//! (> 368 B since an earlier change; was > 256 B in an earlier change..5p).  Handles
//! activation-flag check + the FS=false ladder dispatch and the
//! malloc fallback for very large sizes.  Inline hot path
//! (size ≤ 368 B) bypasses this entirely.
void *new_redirected_large(std::size_t size) noexcept;

//! (§17) Pool-or-libc aligned-allocation entry point.  Routes to the
//! smallest pool bucket whose ALIGN is a multiple of `alignment` and
//! whose usable size covers `size`; falls back to `allocate_dedicated_chunk`
//! when alignment ≤ 256 KiB and the request exceeds the bucket range; and
//! to `posix_memalign` otherwise.  Returns nullptr on allocation failure.
//! `alignment` must be a power of two; caller guarantees this.
//!
//! All pool-returned pointers are freeable via `PoolAllocatorBase::
//! deallocate`, exactly like a normal pool slot — no separate
//! `_aligned_free` pairing.  Libc-returned pointers (posix_memalign
//! fallback) are freed via libc free, which `deallocate_pooled_or_free`
//! resolves automatically.
void *new_redirected_aligned(std::size_t alignment, std::size_t size) noexcept;

inline void *new_redirected(std::size_t size) {
	// Hot path: sizes ≤ 368.  One branch + the inline `(size+15)>>4`
	// formula (the small-range half of `bucket_for_size`).  Larger sizes
	// go to `new_redirected_large`, which uses the full `bucket_for_size`
	// for the FS=false dispatch.
	if(size > (std::size_t)ALLOC_SIZE23)
		return new_redirected_large(size);
	unsigned int bucket = (static_cast<unsigned int>(size) + 15u) >> 4;
	// (§12.3) DIRECT-JUMP fast path: bucket -> freelist-head pointer in
	// ONE TLS read.  `g_thread_freelist_ptr[bucket]` is a `char **` that
	// points DIRECTLY at the active chunk's
	// `m_freelist_head[kBucketLocalId[bucket]]` cell, maintained at
	// chunk-switch (slow_allocate / cold_first_access — see allocator.cpp)
	// where `kBucketLocalId[]` IS read.  So the alloc hot path needs
	// NEITHER a bucket->local-id remap NOR a chunk-pointer-deref chain —
	// just `*tls[bucket]` to get the head, and a normal LIFO pop.
	if(char **head_ptr = g_thread_freelist_ptr[bucket]) {
		if(char *head = *head_ptr) {
			*head_ptr = *reinterpret_cast<char **>(head);
			return head;
		}
		// Freelist empty for this bucket — chunk is still pinned (the
		// ptr is non-null); recover the chunk's PoolAllocator object via
		// `chunk_from_freelist_ptr` (one mask + add, NO second TLS read)
		// and dispatch through its vtable.
		return chunk_from_freelist_ptr(head_ptr)->slow_allocate(bucket, size);
	}
	// bucket not yet activated on this thread (first-time path +
	// pre-activation / post-cleanup malloc fallbacks).
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
