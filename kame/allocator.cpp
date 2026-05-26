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
#if KAME_FAST_TSD
    #include <pthread.h>
#endif

// Per-thread flag: set to true when AllocPinCleanup has run, signalling
// that pool-allocator TLS (s_my_chunk, freelists, pin counts) is no
// longer valid.  Trivially destructible (`ALLOC_TLS` = `__thread`) so it
// survives past all thread_local / pthread_key destructors.  Checked in
// `new_redirected()` to fall back to malloc for any heap operations
// that occur during later TLS cleanup phases (e.g. pthread_key dtors
// like RunnerCounterRegistration).
ALLOC_TLS bool s_alloc_tls_off = false;

#if KAME_FAST_TSD
// Fast pthread-TSD bypass of macOS / Linux TLV thunk.  See header for
// the design overview.  These two globals carry the discovered byte
// offsets within the pthread struct (= `kame_thread_pointer()`) where
// our two pthread_keys' TSD slots live.  Zero means "not yet
// initialised"; the hot accessor falls back to TLV in that state.
std::size_t s_kame_slots_tsd_offset = 0;
std::size_t s_kame_chunks_tsd_offset = 0;

namespace {
pthread_key_t s_kame_slots_key;
pthread_key_t s_kame_chunks_key;

// Constructor priority 101: runs early but after libc/libpthread
// constructors at priorities <= 100.  If pthread_key_create or the
// sentinel scan fails, the offsets stay 0 and the allocator stays on
// the TLV path with no further runtime overhead (degraded mode).
//
// Inter-TU ordering: other TUs' constructor(101)s may run before this
// one and call operator new; they hit the TLV fallback (offset == 0),
// which is safe.  Once we run, subsequent allocations on the main
// thread go through fast TSD.  Other threads plant their own TSD slot
// lazily on their first allocation via `kame_*_cold` below.
__attribute__((constructor(101)))
void kame_tls_init_fast() noexcept {
    if(pthread_key_create(&s_kame_slots_key, nullptr) != 0) return;
    if(pthread_key_create(&s_kame_chunks_key, nullptr) != 0) return;

    char *tp = kame_thread_pointer();
    if( !tp) return;

    // Sentinel scan: write two distinct magic values via the POSIX
    // API, then walk the pthread struct to find which byte offsets
    // received them.  POSIX doesn't expose the layout, but the
    // implementation must store the value somewhere reachable from
    // the thread pointer for `pthread_getspecific` to be fast — we
    // rely on it being a fixed offset, true for both Apple's libc
    // and glibc.
    const uintptr_t sent1 = (uintptr_t)0xDEAD600D11AA1234ull;
    const uintptr_t sent2 = (uintptr_t)0xDEAD600D11BB5678ull;
    pthread_setspecific(s_kame_slots_key,  (void *)sent1);
    pthread_setspecific(s_kame_chunks_key, (void *)sent2);

    std::size_t off1 = 0, off2 = 0;
    // 4 KiB upper bound covers all libc TSD layouts we know about
    // (Apple reserves slots 0..N, then user keys start; offsets are
    // typically < 2 KiB).  Stride 8 — slot is a pointer.
    for(std::size_t off = 0; off < 4096 && (!off1 || !off2); off += 8) {
        uintptr_t v = *reinterpret_cast<uintptr_t *>(tp + off);
        if(v == sent1 && !off1) off1 = off;
        else if(v == sent2 && !off2) off2 = off;
    }

    if(off1 && off2) {
        s_kame_slots_tsd_offset  = off1;
        s_kame_chunks_tsd_offset = off2;
        // Plant THIS thread's (= typically the main thread's) TSD
        // slots now so the next allocation hits the fast path on the
        // first try.  Touching the __thread arrays triggers TLV lazy
        // init for this thread; the resulting addresses are stable
        // for this thread's lifetime.
        pthread_setspecific(s_kame_slots_key,  &g_thread_slots[0]);
        pthread_setspecific(s_kame_chunks_key, &g_thread_chunks[0]);
    }
    else {
        // Scan failed — leave offsets at 0 (degraded TLV-only mode).
        pthread_setspecific(s_kame_slots_key,  nullptr);
        pthread_setspecific(s_kame_chunks_key, nullptr);
    }
}
} // anon namespace

// Cold paths for the fast-TSD accessors in the header.  Called when
// either guard branch fails (offset == 0 → pre-init, fall back to
// TLV; or TSD slot null → first allocation on this thread, plant the
// pointer).  `preserve_most` (matching the header decl) tells the
// caller that this call preserves nearly all caller-saved registers,
// so `operator new`'s hot-path prologue stays small.  cold + noinline
// keeps the inlining budget separate.
[[clang::preserve_most]]
__attribute__((cold, noinline))
AllocSlot *kame_slots_cold() noexcept {
    if(s_kame_slots_tsd_offset != 0) {
        // Post-init, per-thread first touch.  Plant the TSD slot for
        // this thread; `&g_thread_slots[0]` is TLV-resolved here,
        // which lazily allocates per-thread storage.  Subsequent
        // hot-path reads will see the non-null TSD value.
        pthread_setspecific(s_kame_slots_key, &g_thread_slots[0]);
    }
    return &g_thread_slots[0];
}
[[clang::preserve_most]]
__attribute__((cold, noinline))
PoolAllocatorBase **kame_chunks_cold() noexcept {
    if(s_kame_chunks_tsd_offset != 0) {
        pthread_setspecific(s_kame_chunks_key, &g_thread_chunks[0]);
    }
    return &g_thread_chunks[0];
}
#endif // KAME_FAST_TSD

// Forward decl — the post-thread-exit functor used by AllocPinCleanup
// to overwrite every slot of `g_thread_slots[]` before chunks are
// released.  Defined later in this TU.

// Forward decl: AllocPinCleanup's dtor needs to drain each per-bucket
// AllocSlot freelist back to the bitmap via the cross-thread TLS batch
// (CrossDeallocBatch is defined further down).  Hide the dependency
// behind a free function defined after CrossDeallocBatch.
namespace { void drain_thread_slot_freelists() noexcept; }

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
        // Drain each per-bucket AllocSlot freelist back to the bitmap
        // FIRST.  Slots on the linked list inside the slot pool would
        // become unreachable after the table-rewrite below (the head
        // pointer is wiped) and after the chunk pins drop (the chunk
        // may be released).  `drain_thread_slot_freelists` issues a
        // per-slot `batch_return_to_bitmap(&one, 1)` via
        // `lookup_chunk` (handles FS=false bucket-share invariant).
        drain_thread_slot_freelists();
        // Clear every per-thread chunk pointer BEFORE the chunk pins
        // drop to 0 — otherwise a later TLS destructor that allocates
        // could route through a chunk that's about to be released by
        // another thread's deallocate.  After this loop, the slow
        // path's `g_thread_chunks[bucket]` read returns nullptr, so
        // `new_redirected` falls to `cold_first_access`, which
        // observes `s_alloc_tls_off == true` (set a few lines below)
        // and returns `std::malloc(size)`.
        for(int b = 0; b < ALLOC_NUM_BUCKETS; ++b)
            g_thread_chunks[b] = nullptr;
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

// Cross-thread dealloc batch — per-thread parallel arrays of slot
// pointers and their owning chunks.  Parallel-array (SoA) layout is
// chosen over the natural AoS (`struct { chunk, slot }`) so that
// after sorting, the per-chunk `slot` subarray is *contiguous in
// memory* — directly passable to `chunk->batch_return_to_bitmap`
// without an intermediate copy.
//
// On flush:
//   1. Insertion-sort the (chunks, slots) pair by (chunk, slot)
//      lexicographically — chunk primary key for grouping, slot
//      pointer secondary key so the per-chunk slot subarray is
//      pointer-sorted (= m_flags-word-index-sorted).  In-place,
//      swap-based, no allocation.  Insertion sort is the right
//      choice at CAP=16: O(n²/2) ≈ 128 compares worst, but it's
//      branch-friendly and cache-warm on the tiny SoA arrays.
//   2. Walk chunk runs, hand each `chunk->batch_return_to_bitmap`
//      the contiguous `&slots[run_start], run_len`.  The chunk's
//      bitmap clear (in `batch_clear_impl`) walks the sorted slots
//      once, merging adjacent same-word slots into one CAS — O(n)
//      total, no temporary allocation, no m_count-proportional
//      bookkeeping.
//
// Why batching beats CAP=1 here despite the earlier ohtaka result:
// the old `batch_clear_impl` paid O(m_count) bookkeeping per call
// regardless of n, so n=1 calls were ~150 cycles of pure overhead
// per slot.  Now the bookkeeping is O(n) (slot-walk + adjacent
// same-word merge), so n>1 wins purely from coalesced CAS reduction
// whenever slots happen to share an m_flags word.
//
// CAP=16 chosen by the earlier sweep (HWM trade-off — see git log).
// Re-tune-able now that the O(n) impl removes the throughput cost
// curve.
struct CrossDeallocBatch {
    // FS=true small-slot batch.  FS=true buckets are ALIGN==SIZE
    // (16..240 B), one bit per slot in m_flags ⇒ up to 64 slots per
    // FUINT word.  Cross-thread frees of small slots are numerous AND
    // their chunks tend to repeat (a few hot per-size-class chunks
    // serve most allocs), so a deep accumulation window catches
    // same-chunk same-word "buddies" arriving over time → at flush,
    // sort + adjacent-merge coalesces them into one CAS per word.
    //
    // FS=false large-slot frees (sizes 96..512, N bits / slot) bypass
    // this buffer via `push_direct` and dispatch immediately — the
    // per-slot bit footprint is wider so word coalescing windows are
    // smaller, and large-slot chunks repeat less, so the holding cost
    // doesn't pay back.
    //
    // CAP=1024 chosen for L1d-resident accumulation:
    //   16 B / entry × 1025 entries = 16.4 KiB.
    // Most modern L1d is 32-64 KiB; the buf fits with room for other
    // working set.  Per-thread; 128 threads × 16 KiB = 2 MiB total —
    // acceptable for the throughput win expected on NUMA.
    //
    // Sort cost (~20000 cycles for 1024 entries) amortised over
    // 1024 pushes ≈ 20 cycles/push — break-even with current CAP=1
    // direct dispatch IF average coalescing factor > 1.08 (saves >
    // 8 % of CAS, which at ~250 cycles per cross-socket CAS = 20
    // cycles/push).  Realistic FS=true workload (STM Payload deep-
    // copies, identical-size objects from a few chunks) should
    // comfortably exceed this.
    static constexpr int CAP = 1024;
    CrossDeallocEntry buf[CAP + 1];   // +1 = sentinel slot
    int               count = 0;

    //! FS=true path: hold and batch.  Caller passes its own `this`
    //! as `c` (the chunk).
    void push(PoolAllocatorBase *c, void *s) noexcept {
        if(count == CAP) flush();
        buf[count++] = {c, s};
    }

    //! FS=false / FS=true ALIGN > 48 path: adaptive direct/hold
    //! dispatch.  Reads the chunk's `m_last_coalesce_x16` hint
    //! (relaxed); routes to the hold buf when the last batch
    //! returned a coalescing factor ≥ threshold, else dispatches
    //! immediately with a single-slot scratch + sentinel.
    //!
    //! Threshold is per (ALIGN, FS) — the holding cost scales with
    //! slot size, and FS=false slots are wider than their ALIGN
    //! (PoolAllocator<16, false> serves up to 256 B; <32, false>
    //! up to 512 B) AND repeat less in realistic workloads, so
    //! FS=false thresholds are 2× the FS=true ones at the same
    //! ALIGN tier:
    //!
    //!                  FS=true (×1.1)   FS=false (×2)
    //!   ALIGN ≤  64         20               36
    //!   ALIGN ≤ 128         24               44
    //!   ALIGN ≤ 256         29               52
    //!   ALIGN >  256        35               64
    //!
    //! (Bumped up from the original 18/22/26/32 baseline because
    //! ohtaka bench showed memory consumption growing too fast at
    //! the original thresholds — too-aggressive holding inflates
    //! the "held bitmap bit ⇒ chunk can't release" pressure.  ×2
    //! for FS=false and ×1.1 for FS=true gives a more conservative
    //! hold policy that keeps the speed win and trims the
    //! ReserveSwapSpace footprint.)
    //!
    //! Epsilon-greedy explore: every `EXPLORE_PERIOD`-th call,
    //! force-hold regardless of the hint.  Without this a chunk
    //! whose factor dropped below threshold (e.g., one bad batch
    //! from a worker thread) would be permanently routed direct,
    //! never giving its coalescing potential a chance to be
    //! re-measured.  Period 64 ⇒ ~1.5 % exploration overhead.
    //!
    //! Not static — the explore counter lives in the per-thread
    //! batch instance, naturally TLS-local.
    static constexpr int EXPLORE_PERIOD = 64;
    int explore_counter = 0;

    template <unsigned ALIGN, bool FS>
    void push_direct(PoolAllocatorBase *c, void *s) noexcept {
        constexpr uint8_t threshold_x16 = FS
            ? ((ALIGN <=  64) ? 20 :
               (ALIGN <= 128) ? 24 :
               (ALIGN <= 256) ? 29 : 35)
            : ((ALIGN <=  64) ? 36 :
               (ALIGN <= 128) ? 44 :
               (ALIGN <= 256) ? 52 : 64);
        bool hold;
        if(++explore_counter >= EXPLORE_PERIOD) {
            explore_counter = 0;
            hold = true;                                // explore
        }
        else {
            hold = c->m_last_coalesce_x16.load(
                       std::memory_order_relaxed) >= threshold_x16;
        }
        if(hold) {
            push(c, s);
            return;
        }
        CrossDeallocEntry tmp[2] = {{c, s}, {nullptr, nullptr}};
        c->batch_return_to_bitmap(tmp);
    }

    void flush() noexcept {
        if(count == 0) return;
        // Sort by (chunk, slot) lex — chunk primary key for grouping,
        // slot pointer secondary key so each chunk run is pointer-
        // ascending (= m_flags-word-ascending).  std::sort introsort,
        // no heap, in-place swap-based.
        std::sort(buf, buf + count,
                  [](const CrossDeallocEntry &a, const CrossDeallocEntry &b) {
                      if(a.chunk != b.chunk) return a.chunk < b.chunk;
                      return a.slot < b.slot;
                  });
        // Plant the sentinel after the live count so the chunk-side
        // walk terminates by `entries[k].chunk == this` failing,
        // without a length check.
        buf[count] = {nullptr, nullptr};
        // Walk chunk runs.  `batch_return_to_bitmap` consumes the run
        // starting at `&buf[i]` (entries[k].chunk == this until
        // sentinel / next chunk), returns the count, caller advances.
        int i = 0;
        while(i < count) {
            i += buf[i].chunk->batch_return_to_bitmap(&buf[i]);
        }
        count = 0;
    }
    ~CrossDeallocBatch() noexcept { flush(); }
};
XThreadLocal<CrossDeallocBatch> tls_cross_dealloc_batch;

// Drain each per-bucket AllocSlot's freelist back to the bitmap.
// Called from `AllocPinCleanup::~dtor`, before the table-wide
// `g_thread_chunks` clear and the chunk pin decrements.  Each free
// slot's first 8 bytes hold the next pointer (see AllocSlot doc).
//
// Why we MUST look up each slot's chunk individually (not just use
// `g_thread_chunks[b]`):
//
// Multiple FS=false buckets share a single `PoolAllocator` template
// instantiation (sizes 96/128/160/192/224/256 all use
// `PoolAllocator<32, false>`, sizes 288..512 all use
// `PoolAllocator<64, false>` etc.), sharing one `s_my_chunk` static.
// When bucket B0 fills and `slow_allocate` claims a new chunk C2,
// only `g_thread_chunks[B0]` is updated to C2; `g_thread_chunks[B1]`
// still holds the previous chunk C1.  A subsequent
// `deallocate_pooled` of a C2 slot via bucket B1's dealloc path
// passes the owner check (`s_my_chunk == this == C2`) and pushes to
// `g_thread_slots[B1].freelist_head` — but `g_thread_chunks[B1]` is
// still C1.  At drain, the bucket's freelist may therefore hold
// slots from BOTH C1 and C2.  Sending all of them at
// `g_thread_chunks[B1]` (= C1) would make `batch_return_to_bitmap`
// compute `(c2_slot - C1->m_mempool) / ALIGN` — a wild idx that
// walks off `m_flags[]` into unrelated memory → SIGSEGV.  (Caught
// by `alloc_stress_test 5000 64 5000 30` — the STM 3level_mixed
// workload missed it because its allocs are near-fixed-size and all
// land in one FS=true bucket.)
//
// Fix: per-slot `PoolAllocatorBase::lookup_chunk(p)` (address-only
// `s_chunks[cidx]` lookup) gives the slot's true owner.  Per-slot
// CAS is slower than batched but drain is rare (thread exit only).
// Direct `batch_return_to_bitmap` call — must NOT route through
// `tls_cross_dealloc_batch`, which in PerThread's LIFO TLS chain
// dies before `~AllocPinCleanup` and would corrupt freed heap (see
// the AllocPinCleanup comment).
//
// Each chunk we touch is still pinned at this point (pin counts
// drop later in the same destructor), so `release_allocator`
// returns false inside `batch_return_to_bitmap` and the chunk is
// not deleted from under us.
void drain_thread_slot_freelists() noexcept {
    // Single-slot scratch + trailing nullptr sentinel — satisfies
    // `batch_return_to_bitmap`'s `entries[k].chunk == this` walk
    // contract (one matching entry, then the sentinel terminates).
    CrossDeallocEntry tmp[2] = {};
    for(int b = 0; b < ALLOC_NUM_BUCKETS; ++b) {
        AllocSlot &slot = g_thread_slots[b];
        char *head = slot.freelist_head;
        slot.freelist_head = nullptr;
        while(head) {
            char *next = *reinterpret_cast<char **>(head);
            if(PoolAllocatorBase *c = PoolAllocatorBase::lookup_chunk(head)) {
                tmp[0] = {c, head};
                // tmp[1] stays {nullptr, nullptr} as the sentinel.
                c->batch_return_to_bitmap(tmp);
            }
            head = next;
        }
    }
}

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
	m_flags(reinterpret_cast<FUINT *>( &addr[(sizeof(PoolAllocator) + sizeof(FUINT) - 1) * sizeof(FUINT)])),
	m_idx(0),
	m_count(count) {
	m_flags_nonzero_cnt = 0;
	m_flags_filled_cnt = 0;
	for(int i = count - 1; i >= 0 ; --i)
		m_flags[i] = 0; //zero clear.
	// Initial coalesce hint by (FS, real-instance):
	//   FS=true real chunk (FS && DUMMY): start ABOVE all FS=true
	//     thresholds (max 35) → push_direct optimistically routes
	//     to hold on the first encounter, letting `batch_clear_impl`
	//     measure the actual coalescing factor and refine the hint.
	//   FS=false real chunk: leave default (16) — below all FS=false
	//     thresholds (≥ 36), so first encounter direct-dispatches.
	//     Adaptive ramps up only if the explore-period override
	//     catches a strong coalescing factor on this chunk.
	// `FS && DUMMY` distinguishes a real FS=true chunk (`<ALIGN,
	// true, true>`) from the `<ALIGN, true, false>` base used by
	// FS=false's partial spec.
	if constexpr (FS && DUMMY) {
		this->m_last_coalesce_x16.store(40, std::memory_order_relaxed);
	}
#ifdef GUARDIAN
	for(unsigned int i = 0; i < count * sizeof(FUINT) * 8 * ALIGN / sizeof(uint64_t); ++i)
		reinterpret_cast<uint64_t *>(ppool)[i] = GUARDIAN; //filling
#endif
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
inline PoolAllocator<ALIGN, FS, DUMMY> *PoolAllocator<ALIGN, FS, DUMMY>::create(size_t size, char *ppool) {
	// Layout: [class][m_flags].  The owner-thread freelist lives on the
	// per-thread `AllocSlot` (`g_thread_slots[bucket]`) — embedded
	// linked list with the next-pointer stored in each free slot's
	// first 8 bytes — so no per-chunk freelist storage follows m_flags.
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
	// Layout: [class][m_flags][m_sizes].  The previous tail-allocated
	// m_fs_buckets (FS_MAX_BUCKETS × FS_BUCKET_CAP × 2 B = 8 KiB per
	// chunk) is gone — FS=false dealloc now pushes to the per-thread
	// AllocSlot freelist in `g_thread_slots[]`, identically to FS=true.
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
	// Owner-side freelist hit is handled in `new_redirected` via the
	// per-thread `g_thread_slots[bucket].freelist_head` — by the time
	// we reach `allocate_pooled` the freelist has missed.  This path
	// runs the bitmap CAS to claim N consecutive free bits.
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
	// Owner-side dealloc: decode the slot's N (size in ALIGN units)
	// from m_sizes, compute the global bucket index for size = N*ALIGN,
	// and push to the per-thread `g_thread_slots[bucket].freelist_head`
	// — exactly the slot that `new_redirected` pops from on the next
	// allocation of that size.  Non-owner OR bucket-out-of-range
	// (slot's size exceeds what `new_redirected` covers, i.e.
	// size > 256 B) routes to the cross-thread TLS batch which
	// eventually CASes the bitmap via `batch_return_to_bitmap`.
	//
	// `s_my_chunk` has declared type `PoolAllocator<ALIGN, false, false>*`
	// (from the base's `PoolAllocator<ALIGN, DUMMY, DUMMY>*` with
	// DUMMY=false), while `this` has type `PoolAllocator<ALIGN, false,
	// DUMMY>*` — different template instantiations referring to the
	// same chunk object.  Compare as void* to bypass the type mismatch.
	if(static_cast<void *>(PoolAllocator<ALIGN, true, false>::s_my_chunk)
	    == static_cast<void *>(this)) {
		unsigned slot_idx = static_cast<unsigned>(
		    (p - this->m_mempool) / ALIGN);
		unsigned idx = slot_idx / (sizeof(FUINT) * 8);
		unsigned sidx = slot_idx % (sizeof(FUINT) * 8);
		FUINT nones = find_zero_forward(m_sizes[idx] >> sidx);
		FUINT slot_mask = nones | (nones - 1u);
		unsigned N = count_bits(slot_mask);
		std::size_t size_bytes = (std::size_t)N * ALIGN;
		// Same `bucket_for_size` as `new_redirected` — slot sizes
		// (multiples of 16 ≤ 256 or 32 between 288..512) all land
		// on their canonical bucket regardless of which formula
		// branch you came through.
		if(N >= 1 && size_bytes <= ALLOC_MAX_BUCKETED_SIZE) {
			unsigned bucket = bucket_for_size(size_bytes);
			kame_slots_base()[bucket].push(p);
			return false;
		}
	}
	// Post-teardown bypass.  `~AllocPinCleanup` sets `s_alloc_tls_off`
	// just before exiting; `~CrossDeallocBatch` ran earlier in the
	// PerThread LIFO chain so the cross-batch storage is already
	// `free()`d.  Later pthread_key dtors that delete pool slots must
	// not push to it — go through `batch_return_to_bitmap` directly
	// on a single-slot scratch + sentinel.
	if(__builtin_expect(s_alloc_tls_off, 0)) {
		CrossDeallocEntry tmp[2] = {{this, p}, {nullptr, nullptr}};
		this->batch_return_to_bitmap(tmp);
		return false;
	}
	// FS=false large slots: bypass the holding buffer by default,
	// dispatch immediately.  Each FS=false slot consumes N bits in
	// m_flags (slot pointer's idx + N-1 successor bits via m_sizes
	// decode), so per-word coalescing windows are small AND large-
	// slot chunks repeat less frequently than FS=true small-slot
	// chunks — the holding cost wouldn't pay back.  `push_direct`
	// consults the chunk's adaptive coalescing hint to override and
	// hold if this particular chunk has shown high recent merge
	// factor (epsilon-greedy explores periodically to re-evaluate).
	tls_cross_dealloc_batch->template push_direct<ALIGN, false>(this, p);
	return false;
}

// FS=false batch return — multi-bit clear (slots vary in n_slots,
// recovered from m_sizes).  Reuses the inherited batch_clear_impl
// skeleton with a multi-bit MaskFn and FS=false-specific OnClearFn
// (no filled_cnt, updates m_available_bits).
template <unsigned int ALIGN, bool DUMMY>
int
PoolAllocator<ALIGN, false, DUMMY>::batch_return_to_bitmap(
    const CrossDeallocEntry *entries) noexcept {
	// Walk entries[k] while .chunk == this — terminates on the next
	// chunk's group OR the trailing {nullptr, nullptr} sentinel that
	// `CrossDeallocBatch::flush` plants at buf[count].  No `k < n_max`
	// test in the inner loop.  Drain / post-teardown single-slot paths
	// pass a stack-local {this, slot} + sentinel pair.
	int n = this->batch_clear_impl(entries,
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
	// Chunk-release check.  Safe to delete this — the caller
	// (`tls_cross_dealloc_batch->push`, `drain_thread_slot_freelists`,
	// or the post-teardown bypass in `deallocate_pooled`) is on a
	// single-slot path and won't reference this chunk again after
	// the call.  `deallocate_chunk` MUST follow `delete this` (using
	// the pre-cached cidx/chunk_size) so `s_chunks[cidx]` is cleared
	// and the mempool mprotect'd back to PROT_NONE.  Without it, the
	// slot in `s_chunks[]` dangles past the suicide and a subsequent
	// `deallocate_<>` would dereference the freed PoolAllocator and
	// glibc would catch it as `free(): invalid pointer`.  Owner-side
	// `deallocate_pooled` returns `true` to get this same cleanup
	// via `PoolAllocatorBase::deallocate_<>`; the
	// `batch_return_to_bitmap` paths don't have that cascade and
	// must do the cleanup themselves.
	if(this->m_flags_nonzero_cnt == 0
	        && PoolAllocator<ALIGN, true, false>::release_allocator(this)) {
		int cidx = this->m_cidx;
		size_t csz = this->m_chunk_size;
		delete this;
		PoolAllocatorBase::deallocate_chunk(cidx, csz);
	}
	return n;
}

// Body of `batch_clear_impl` — out-of-class definition kept in
// allocator.cpp.  The function is template-on-lambdas; bodies in the
// header would balloon allocator_prv.h with a non-trivial loop that's
// only exercised from the cross-dealloc-batch flush (a rare, "long"
// code path).  Hot owner-thread freelist push/pop is done inline on
// `AllocSlot` in `new_redirected`, not via this helper.
template <unsigned int ALIGN, bool FS, bool DUMMY>
template <typename MaskFn, typename OnClearFn>
int
PoolAllocator<ALIGN, FS, DUMMY>::batch_clear_impl(
    const CrossDeallocEntry *entries,
    MaskFn mask_fn, OnClearFn on_clear) noexcept {
	// Walks `entries[k]` while `entries[k].chunk == this`, terminating
	// on the trailing `{nullptr, nullptr}` sentinel that
	// `CrossDeallocBatch::flush` plants at `buf[count]`, OR on the
	// next chunk's group when this is called mid-flush.  Returns the
	// number of entries consumed so the caller can advance past them.
	//
	// Precondition: entries are sorted by ascending pointer address
	// within a chunk group (== sorted by m_flags word index, since
	// word index is `(slot - mempool) / ALIGN / FUINT_BITS`, monotone
	// in slot pointer).  Adjacent same-word slots are therefore
	// contiguous in the input; one O(n) walk merges them.  No
	// alloca, no scratch buffer, no m_count-proportional bookkeeping.
	//
	// Drain / post-teardown single-slot paths pass {this, slot,
	// nullptr-sentinel} so they trivially satisfy the contract.
	//
	// This replaces the previous m_count-proportional design
	// (alloca(m_count*FUINT) + zero(m_count) + per-slot index into a
	// mask array + final m_count-word scan), which paid ~150 cycles
	// per call regardless of n.  perf on ohtaka had ~5 % wall-clock
	// in batch_clear_impl at high cross-thread rates, dominated by
	// the m_count terms — gone now.
	constexpr int FUINT_BITS = sizeof(FUINT) * 8;
	int i = 0;
	int n_words = 0;   // unique m_flags words touched — for coalesce hint
	while(entries[i].chunk == this) {
		char *p = static_cast<char *>(entries[i].slot);
		int midx = (p - this->m_mempool) / ALIGN;
		int idx = midx / FUINT_BITS;
		unsigned int sidx = midx % FUINT_BITS;
		FUINT mask = mask_fn(idx, sidx, p);
		// Merge adjacent same-word slots — pointer-sorted ⇒
		// word-index-sorted, so once we see a different idx
		// (or a different chunk) we know no later slot lands in this
		// word either.
		int j = i + 1;
		while(entries[j].chunk == this) {
			char *q = static_cast<char *>(entries[j].slot);
			int midx_q = (q - this->m_mempool) / ALIGN;
			int idx_q = midx_q / FUINT_BITS;
			if(idx_q != idx) break;
			unsigned int sidx_q = midx_q % FUINT_BITS;
			mask |= mask_fn(idx_q, sidx_q, q);
			++j;
		}
		++n_words;
		// CAS-clear `m_flags[idx] &= ~mask` with retry; on_clear gets
		// the (oldv, newv) for counter updates (per-FS-variant logic).
		FUINT nones = ~mask;
		FUINT *pflags = &this->m_flags[idx];
		for(;;) {
			FUINT oldv = *pflags;
			FUINT newv = oldv & nones;
			if(atomicCompareAndSet(oldv, newv, pflags)) {
				on_clear(oldv, newv);
				break;
			}
		}
		i = j;
	}
	// Update adaptive coalescing hint: factor_x16 = (entries × 16) /
	// unique_words.  16 = 1.0× = no benefit; > 16 = adjacent merges
	// happened.  Relaxed: it's just a hint, races benign (next push
	// reads slightly stale value, no correctness impact).
	if(n_words > 0) {
		unsigned factor = (unsigned(i) * 16u) / unsigned(n_words);
		if(factor > 255u) factor = 255u;
		this->m_last_coalesce_x16.store(uint8_t(factor),
		                                std::memory_order_relaxed);
	}
	return i;
}

// Bitmap clear of slots passed via argument array.  All slots must
// belong to THIS chunk (callers always pass single-chunk groups —
// `CrossDeallocBatch::push` issues `&one, 1`,
// `drain_thread_slot_freelists` `lookup_chunk`s each slot and dispatches
// per chunk, and the post-teardown bypass in `deallocate_pooled` issues
// `&one, 1`).  Single-chunk invariant lets us share one direct-map
// scratch.  Sole remaining consumer of `batch_clear_impl` (the
// chunk-private freelist drain that previously also used it has been
// folded into the per-thread AllocSlot drain in
// `drain_thread_slot_freelists`).
template <unsigned int ALIGN, bool FS, bool DUMMY>
int
PoolAllocator<ALIGN, FS, DUMMY>::batch_return_to_bitmap(
    const CrossDeallocEntry *entries) noexcept {
	// Walks entries[k] while .chunk == this — sentinel-terminated, no
	// length argument; see the FS=false sibling for the full rationale
	// and the contract with `CrossDeallocBatch::flush`.
#ifdef GUARDIAN
	for(int k = 0; entries[k].chunk == this; ++k) {
		char *p = static_cast<char *>(entries[k].slot);
		for(unsigned int j = 0; j < ALIGN / sizeof(uint64_t); ++j)
			reinterpret_cast<uint64_t *>(p)[j] = GUARDIAN;
	}
#endif
	int n = this->batch_clear_impl(entries,
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
	// Chunk-release check.  See FS=false sibling for the rationale —
	// `deallocate_chunk` MUST follow `delete this` so the dangling
	// `s_chunks[cidx]` slot is cleared and the mempool faults on a
	// later stray access.
	if(this->m_flags_nonzero_cnt == 0
	        && PoolAllocator<ALIGN, FS, DUMMY>::release_allocator(this)) {
		int cidx = this->m_cidx;
		size_t csz = this->m_chunk_size;
		delete this;
		PoolAllocatorBase::deallocate_chunk(cidx, csz);
	}
	return n;
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PoolAllocator<ALIGN, FS, DUMMY>::clear_owner_tls() noexcept {
	s_my_chunk = nullptr;
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
bool
PoolAllocator<ALIGN, FS, DUMMY>::deallocate_pooled(char *p) {
	// Two-way dispatch:
	//
	//   owner               → push to per-thread AllocSlot freelist (no atomic)
	//   non-owner           → TLS cross-dealloc batch (batched bitmap CAS
	//                          per m_flags word at flush time)
	//
	// Owner check: per-template `s_my_chunk` TLS only.  The previous
	// secondary `g_thread_slots[bucket].chunk == this` check was
	// redundant — `bucket_first_access` / `bucket_steady_alloc` keep
	// `g_thread_chunks[bucket]` in lockstep with `s_my_chunk` by
	// construction, so the two would always agree.  Dropping it saves
	// one TLS read on every owner-side dealloc, and freed up space
	// for the AllocSlot to shrink to 8 B (chunk pointer moved into
	// the parallel `g_thread_chunks[]` array).
	constexpr int kBucket = ALIGN / ALLOC_ALIGNMENT;
	if(static_cast<PoolAllocatorBase *>(s_my_chunk) == this) {
		// Slot stays "allocated" in the bitmap until flushed back via
		// AllocPinCleanup (thread exit) or the chunk's bitmap is
		// directly returned to (allocate_pooled goes there on freelist
		// miss).  Owner's next alloc on this bucket pops it back
		// immediately from `g_thread_slots[kBucket].freelist_head`.
		kame_slots_base()[kBucket].push(p);
		return false;
	}
	// Post-teardown bypass.  See FS=false sibling above — once
	// `s_alloc_tls_off` is set, `tls_cross_dealloc_batch` may have
	// been destroyed; route the bit-clear through
	// `batch_return_to_bitmap` directly with a single-slot scratch
	// + sentinel so later pthread_key dtors that delete pool slots
	// do not touch freed heap.
	if(__builtin_expect(s_alloc_tls_off, 0)) {
		CrossDeallocEntry tmp[2] = {{this, p}, {nullptr, nullptr}};
		this->batch_return_to_bitmap(tmp);
		return false;
	}
	// FS=true ALIGN ≤ 48 (sizes 16/32/48): hold-and-batch path.  1
	// bit per slot in m_flags ⇒ up to 64 slots per FUINT word; a
	// deep (CAP=1024) accumulation window gives same-chunk same-
	// word "buddies" arriving over time a chance to be coalesced
	// into one CAS per word at flush time.  The smallest buckets
	// are picked for two reasons:
	//
	//   * held-bytes-per-entry = slot size.  Lowest slot sizes
	//     minimise the "bitmap bit held" memory pressure that
	//     delays chunk release in the owner thread (the
	//     `ReserveSwapSpace` growth Linux Claude observed at
	//     CAP=2048/4096 scaled with avg_held_bytes × CAP).
	//   * Smallest ALIGN classes have the most slots per chunk
	//     (3072 for ALIGN=16 vs 200 for ALIGN=240), so the buf's
	//     chunk coverage is densest — buddies more likely.
	//
	// FS=true ALIGN > 48 (sizes 64..240) fall to the direct
	// dispatch path: their per-entry held-bytes payback ratio is
	// worse, and their chunks repeat less frequently in realistic
	// STM workloads (allocation distribution is heavy-tailed
	// toward smallest classes).
	if constexpr (ALIGN <= 48) {
		tls_cross_dealloc_batch->push(this, p);
	} else {
		tls_cross_dealloc_batch->template push_direct<ALIGN, true>(this, p);
	}
	return false;
}

// FS=true slow_allocate override.  Called from `new_redirected`'s cold
// path through this chunk's vtable when `g_thread_chunks[bucket]` is
// non-null.  Equivalent to the previous `bucket_steady_alloc<B>`
// function-pointer slot, but ALIGN comes from the template
// instantiation (compile-time) instead of B.  FS=true buckets are
// single-size (ALIGN == slot size), so `SIZE = ALIGN`; `bucket` is
// only used to mirror a moved `s_my_chunk` back into
// `g_thread_chunks[bucket]`.
template <unsigned int ALIGN, bool FS, bool DUMMY>
__attribute__((cold, noinline))
void *
PoolAllocator<ALIGN, FS, DUMMY>::slow_allocate(unsigned bucket,
                                               std::size_t /*size*/) noexcept {
	void *p = allocate_chunk_path(ALIGN);
	PoolAllocatorBase *new_chunk =
	    static_cast<PoolAllocatorBase *>(s_my_chunk);
	if(new_chunk != g_thread_chunks[bucket])
		g_thread_chunks[bucket] = new_chunk;
	return p;
}

// FS=false slow_allocate override.  Multiple bucket indices share one
// PoolAllocator<ALIGN, false> instantiation (e.g. buckets 6, 8, 10, 12,
// 14, 16 all live on PoolAllocator<16, false>), so the bucket's
// slot size differs from ALIGN and must be derived from `bucket` at
// runtime.  The inverse of `bucket_for_size`:
//   bucket 1..16  →  slot_size = bucket * 16     (sizes 16..256, 16-B step)
//   bucket 17..24 →  slot_size = 256 + (bucket-16)*32   (sizes 288..512)
// The FS=false `allocate_pooled` uses this SIZE to compute N=SIZE/ALIGN
// (number of consecutive ALIGN-slots to claim from the bitmap).
template <unsigned int ALIGN, bool DUMMY>
__attribute__((cold, noinline))
void *
PoolAllocator<ALIGN, false, DUMMY>::slow_allocate(unsigned bucket,
                                                  std::size_t /*size*/) noexcept {
	unsigned int slot_size = (bucket <= 16) ? (bucket * 16u)
	                                        : (256u + (bucket - 16u) * 32u);
	// Inherited static; resolves to PoolAllocator<ALIGN, true, false>::
	// allocate_chunk_path, which uses the FS=false-instantiated
	// s_my_chunk under the hood (the DUMMY=false template trick).
	void *p = PoolAllocator<ALIGN, true, false>::allocate_chunk_path(slot_size);
	PoolAllocatorBase *new_chunk = static_cast<PoolAllocatorBase *>(
	    PoolAllocator<ALIGN, true, false>::s_my_chunk);
	if(new_chunk != g_thread_chunks[bucket])
		g_thread_chunks[bucket] = new_chunk;
	return p;
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
    // Pre-warm: one byte per OS page, on the claiming thread.  Forces
    // the kernel's first-touch fault to fire HERE (cold chunk-claim
    // path, rare) instead of inside each subsequent user-side write to
    // freshly mapped slots (alloc/dealloc hot path, frequent).  Two
    // wins:
    //
    //   1. Latency: removes a 1-10 μs page-fault from the user's hot
    //      path; instead concentrates the cost into the claim, which
    //      is already a slow path with bounded-latency expectations.
    //   2. NUMA placement: anonymous pages are bound to the NUMA node
    //      of the *first writer*.  Pre-warming on the claiming thread
    //      pins the chunk's pages to that thread's node, so the same
    //      thread's subsequent user accesses are node-local (matters
    //      on ohtaka — 128 cores across multiple sockets).  Without
    //      pre-warming, pages are bound to whichever thread first
    //      writes a slot, which on a busy NUMA box is often a thread
    //      that subsequently migrates away.
    //
    // Cost is bounded: `chunk_size` is a few hundred KB to a few MB,
    // ALLOC_PAGE_SIZE is 4 KiB (Linux) or 16 KiB (Apple Silicon), so
    // 100..2000 stores per claim — negligible vs the mmap syscall and
    // page-table setup the kernel already did.  `volatile` prevents
    // the optimiser from eliding the stores.
    for(size_t off = 0; off < chunk_size; off += ALLOC_PAGE_SIZE)
        reinterpret_cast<volatile char *>(addr)[off] = 0;
#endif

	ALLOC *palloc = ALLOC::create(chunk_size, addr);
	// Stamp (cidx, chunk_size) on the instance so the cross-batch
	// chunk-release self-suicide path in `batch_return_to_bitmap` can
	// call `deallocate_chunk(cidx, chunk_size)` AFTER `delete this`
	// — clearing `s_chunks[cidx]` so a later `deallocate_<>` lookup
	// doesn't dereference the freed PoolAllocator instance.
	palloc->m_cidx = cidx;
	palloc->m_chunk_size = chunk_size;
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
	// Cold path of allocate<SIZE>().  Reached on the very first
	// allocation of (this thread, this bucket) via
	// `bucket_first_access<B>`, or whenever the per-thread `AllocSlot`
	// freelist for this bucket misses and the slow path dispatcher
	// (`bucket_steady_alloc<B>` in g_thread_alloc_fn[]) is invoked.
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

// Runtime walk of `s_mmapped_spaces[]` mirroring `deallocate_<>`'s
// progressive `GROW_CHUNK_SIZE` ladder — find which mmap space
// contains `p` and look up `s_chunks[cidx]`.  No `deallocate_pooled`
// dispatch; pure address-to-chunk mapping for the drain path.  Kept
// as a regular for-loop rather than reusing the recursive-template
// `deallocate_<>` because that template lives on the hot operator
// delete path and an extra branch arm there is undesirable.
inline PoolAllocatorBase *
PoolAllocatorBase::lookup_chunk(void *p) noexcept {
	size_t chunk_size = ALLOC_MIN_CHUNK_SIZE;
	for(int ccnt = 0; ccnt < ALLOC_MAX_MMAP_ENTRIES; ++ccnt) {
		char *mp = s_mmapped_spaces[ccnt];
		if(ccnt > 0 && !mp) break;
		if(mp) {
			ptrdiff_t pdiff = static_cast<char *>(p) - mp;
			if(pdiff >= 0
			   && pdiff < (ptrdiff_t)chunk_size * NUM_ALLOCATORS_IN_SPACE) {
				int cidx = int(pdiff / chunk_size)
				           + ccnt * NUM_ALLOCATORS_IN_SPACE;
				PoolAllocatorBase *palloc = s_chunks[cidx];
				if((uintptr_t)palloc <= (uintptr_t)1u) return nullptr;
				return palloc;
			}
		}
		chunk_size = GROW_CHUNK_SIZE(chunk_size);
	}
	return nullptr;
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
// Buckets 17..24 cover sizes 288..512 (32-B increments) — previously
// dispatched by new_redirected_large via ALLOCATE_9_16X(2, size).  All
// FS=false, ALIGN per ALLOC_ALIGN(size) (= 32 except for size 512 → 256).
KAME_DECL_BUCKET(17, ALLOC_ALIGN(ALLOC_SIZE9 * 2),  false, ALLOC_SIZE9 * 2);   // size 288
KAME_DECL_BUCKET(18, ALLOC_ALIGN(ALLOC_SIZE10 * 2), false, ALLOC_SIZE10 * 2);  // size 320
KAME_DECL_BUCKET(19, ALLOC_ALIGN(ALLOC_SIZE11 * 2), false, ALLOC_SIZE11 * 2);  // size 352
KAME_DECL_BUCKET(20, ALLOC_ALIGN(ALLOC_SIZE12 * 2), false, ALLOC_SIZE12 * 2);  // size 384
KAME_DECL_BUCKET(21, ALLOC_ALIGN(ALLOC_SIZE13 * 2), false, ALLOC_SIZE13 * 2);  // size 416
KAME_DECL_BUCKET(22, ALLOC_ALIGN(ALLOC_SIZE14 * 2), false, ALLOC_SIZE14 * 2);  // size 448
KAME_DECL_BUCKET(23, ALLOC_ALIGN(ALLOC_SIZE15 * 2), false, ALLOC_SIZE15 * 2);  // size 480
KAME_DECL_BUCKET(24, ALLOC_ALIGN(ALLOC_SIZE16 * 2), false, ALLOC_SIZE16 * 2);  // size 512
#undef KAME_DECL_BUCKET

//! First-access trampoline for bucket B.  Invoked from the
//! `cold_first_access` switch when `g_thread_chunks[B] == nullptr`.
//! Claims a chunk via the existing `allocate<>()` slow path (which
//! registers AllocPinCleanup) and records the chunk into
//! `g_thread_chunks[B]` so subsequent freelist-miss calls go straight
//! to the chunk vtable path (`PoolAllocatorBase::slow_allocate`) and
//! never come back through `cold_first_access`.
template <int B>
__attribute__((noinline))
void *bucket_first_access(std::size_t /*size*/) noexcept {
    using BT = BucketTraits<B>;
    using PA = typename BT::PoolType;
    void *p = PA::template allocate<BT::SIZE>();
    PoolAllocatorBase *chunk = PA::get_pinned_chunk_base();
    if(chunk) g_thread_chunks[B] = chunk;
    return p;
}

} // anon namespace

// Cold path entry point used by `new_redirected` when
// `g_thread_chunks[bucket] == nullptr`.  Handles three states:
//
//   1. Pre-activation (`g_sys_image_loaded == false`): return
//      std::malloc(size), don't claim a chunk.  Retried on every call
//      until activateAllocator() is invoked.
//   2. Post-cleanup (`s_alloc_tls_off == true`): same — return
//      std::malloc(size).  Set by AllocPinCleanup::~dtor on thread
//      exit; later TLS destructors that still allocate land here.
//   3. First access: switch on bucket to invoke the per-bucket
//      `bucket_first_access<B>`, which calls
//      `PA::allocate<BT::SIZE>()` with SIZE compile-time-const,
//      registers AllocPinCleanup, and populates
//      `g_thread_chunks[B]`.
//
// `__attribute__((cold))`: clang places this out-of-line so the
// freelist-miss path in `new_redirected` doesn't bloat its branch
// distance budget.  The switch lowers to a jump table on arm64.
__attribute__((cold, noinline))
void *cold_first_access(unsigned bucket, std::size_t size) noexcept {
    if( !g_sys_image_loaded || s_alloc_tls_off)
        return std::malloc(size);
    switch(bucket) {
        case  0: case  1: return bucket_first_access< 1>(size);
        case  2:          return bucket_first_access< 2>(size);
        case  3:          return bucket_first_access< 3>(size);
        case  4:          return bucket_first_access< 4>(size);
        case  5:          return bucket_first_access< 5>(size);
        case  6:          return bucket_first_access< 6>(size);
        case  7:          return bucket_first_access< 7>(size);
        case  8:          return bucket_first_access< 8>(size);
        case  9:          return bucket_first_access< 9>(size);
        case 10:          return bucket_first_access<10>(size);
        case 11:          return bucket_first_access<11>(size);
        case 12:          return bucket_first_access<12>(size);
        case 13:          return bucket_first_access<13>(size);
        case 14:          return bucket_first_access<14>(size);
        case 15:          return bucket_first_access<15>(size);
        case 16:          return bucket_first_access<16>(size);
        case 17:          return bucket_first_access<17>(size);
        case 18:          return bucket_first_access<18>(size);
        case 19:          return bucket_first_access<19>(size);
        case 20:          return bucket_first_access<20>(size);
        case 21:          return bucket_first_access<21>(size);
        case 22:          return bucket_first_access<22>(size);
        case 23:          return bucket_first_access<23>(size);
        case 24:          return bucket_first_access<24>(size);
    }
    return std::malloc(size);  // unreachable
}

// The per-thread tables.  `__thread` (= `ALLOC_TLS` on GCC/Clang) so the
// storage lifetime extends past every TLS destructor on this thread —
// XThreadLocal would `std::free` the underlying memory mid-cleanup,
// leaving the `cached` pointer dangling for any later TLS dtor that
// allocates.
//
// Two parallel tables, both indexed by bucket:
//   `g_thread_slots[]`     8 B/entry  – freelist head, the only field
//                                       on the freelist-hit hot path.
//   `g_thread_chunks[]`    8 B/entry  – currently pinned chunk.  Doubles
//                                       as the slow-path state machine:
//                                       nullptr ⇒ first_access /
//                                       post-cleanup (route through
//                                       `cold_first_access`); non-null
//                                       ⇒ steady (dispatch through the
//                                       chunk's vtable, `slow_allocate`).
// Total 400 B per thread (vs 600 B in the previous fn-pointer-table
// design).  Bucket 0 maps to bucket 1's 16-B allocator so size=0
// allocations don't fault: `cold_first_access`'s switch on bucket
// pairs `case 0:` with `case 1:` into a single label.
ALLOC_TLS AllocSlot g_thread_slots[ALLOC_NUM_BUCKETS] = {};
ALLOC_TLS PoolAllocatorBase *g_thread_chunks[ALLOC_NUM_BUCKETS] = {};

// Out-of-line large-size dispatch.  Sizes > 256 B fall here from
// `new_redirected`.  The 257..512 range dispatches via the same
// g_thread_slots[] table (buckets 17..24) as the small range, just
// from this colder function instead of inline — keeps the hot
// path (sizes ≤ 256) lean (single branch + inline freelist pop).
// Sizes > 512 fall through to allocate_large_size_or_malloc (the
// X=4 / X=8 / ... ALIGN doublings).  Activation-flag check lives
// here too (cold path is the right place — only paid by larger
// allocations).
void *new_redirected_large(std::size_t size) noexcept {
    if(size <= ALLOC_MAX_BUCKETED_SIZE) {
        unsigned int bucket = bucket_for_size(size);
        AllocSlot &slot = kame_slots_base()[bucket];
        char *head = slot.freelist_head;
        if(head) {
            slot.freelist_head = *reinterpret_cast<char **>(head);
            return head;
        }
        if(PoolAllocatorBase *chunk = kame_chunks_base()[bucket])
            return chunk->slow_allocate(bucket, size);
        return cold_first_access(bucket, size);
    }
    if( !g_sys_image_loaded || s_alloc_tls_off)
        return std::malloc(size);
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

#ifdef KAME_SIZE_HISTOGRAM
// Allocation-size histogram for size-class profiling.  Enabled by
// `-DKAME_SIZE_HISTOGRAM` at build time.  Per-bucket atomic counters
// incremented on every operator new / new[] / nothrow variant; dumped
// to stderr via atexit at process exit.
//
// Index = `(size + 15) >> 4`  →  16-byte granularity.  Covers
// 16..16384 directly; sizes above 16384 fold into the top bucket.
namespace {
constexpr int KAME_HISTO_SIZE = 1024;
std::atomic<uint64_t> g_alloc_size_histo[KAME_HISTO_SIZE];

void kame_print_histo() noexcept {
    fprintf(stderr, "=== KAME_SIZE_HISTOGRAM ===\n");
    uint64_t total = 0;
    for(int i = 0; i < KAME_HISTO_SIZE; ++i)
        total += g_alloc_size_histo[i].load(std::memory_order_relaxed);
    if( !total) { fprintf(stderr, "  (no allocations)\n"); return; }
    uint64_t cum = 0;
    fprintf(stderr, "  size_range      count        %%       cum%%\n");
    for(int i = 0; i < KAME_HISTO_SIZE; ++i) {
        uint64_t n = g_alloc_size_histo[i].load(std::memory_order_relaxed);
        if(n == 0) continue;
        cum += n;
        int lo = (i == 0) ? 0 : (i - 1) * 16 + 1;
        int hi = i * 16;
        fprintf(stderr, "  %5d..%-6d %10llu  %6.2f%%  %6.2f%%\n",
                lo, hi, (unsigned long long)n,
                100.0 * n / total, 100.0 * cum / total);
    }
    fprintf(stderr, "  total: %llu allocs\n", (unsigned long long)total);
}

struct KameHistoInstaller {
    KameHistoInstaller() noexcept { std::atexit(kame_print_histo); }
};
KameHistoInstaller g_kame_histo_installer;

inline void kame_histo_record(std::size_t size) noexcept {
    int idx = static_cast<int>((size + 15) >> 4);
    if(idx >= KAME_HISTO_SIZE) idx = KAME_HISTO_SIZE - 1;
    g_alloc_size_histo[idx].fetch_add(1, std::memory_order_relaxed);
}
} // namespace
#define KAME_HISTO_REC(size) kame_histo_record(size)
#else
#define KAME_HISTO_REC(size) ((void)0)
#endif

void* operator new(std::size_t size) {
    KAME_HISTO_REC(size);
    return new_redirected(size);
}
void* operator new[](std::size_t size) {
    KAME_HISTO_REC(size);
    return new_redirected(size);
}

void operator delete(void* p) noexcept {
    deallocate_pooled_or_free(p);
}
void operator delete[](void* p) noexcept {
    deallocate_pooled_or_free(p);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    KAME_HISTO_REC(size);
    return new_redirected(size);
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    KAME_HISTO_REC(size);
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
//  is now the sole place that drains the per-thread AllocSlot
//  freelists, runs `clear_owner_tls`, and sets `s_alloc_tls_off =
//  true` at thread exit.  Eliminates the C++ thread_local init thunk
//  that macOS arm64 emits for `(void)&s_tls_guard` in the allocate()
//  hot path.)

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
