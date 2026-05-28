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
// Phase 4b-final / Phase 5o: per-thread floor on `owner_release`.
// Stop releasing when this thread's DLL has fewer than this many
// chunks for the given (ALIGN, FS) template — avoids release / re-
// mmap thrashing on bursty workloads.
//
// Phase 5o bumps the floor 2 → 16 to repair the bucket34_repro
// regression on Linux (33.5 → 0.24 M/s under Phase 5n).  Diagnosis:
//
//   bucket34_repro uses 6 sizes (3072..8192 B) all in the ALIGN=1024
//   tier, which share a single per-thread DLL.  Steady-state live[]
//   distribution is ~12 chunks per thread (170 slots × 6 buckets /
//   113 slots/chunk).  At floor=2, owner_release fires whenever an
//   empty neighbour appears and dll_len > 2 — i.e. any time a chunk
//   transiently empties due to random size fluctuation (stddev ~12
//   per bucket).  Each release → madvise → next claim faults pages
//   anew.  Perf shows ~22 % `clear_page_erms` + ~9 % page-fault
//   accounting on this workload.
//
//   floor=16 means dll_len must EXCEED 16 to release; bucket34's
//   12-chunk DLL never triggers release in steady state, so empties
//   stay warm for reuse.  RSS cost: 16 × ALIGN_tier_max_chunk_size
//   per (thread, tier) — for ALIGN=1024 (CHUNK_UNITS=4 = 1 MiB) and
//   8 threads, max 128 MiB extra resident per tier, well below the
//   3 GiB / 100 GiB VA caps from Phase 5l-fixup.  Real workloads
//   either stay well below the floor (no release ever fires —
//   correct) or far above it (release still fires aggressively when
//   dll_len > 16 — correct).
#define LEAVE_VACANT_CHUNKS_PER_THREAD 16

#include "allocator.h"

#ifndef USE_STD_ALLOCATOR

#include "atomic_mfence.h"   // readBarrier / writeBarrier / pause4spin
                             // (kamepoolalloc-internal, mirrors kame/atomic.h's
                             //  arch-select chain but drops atomic_shared_ptr et al)

#include <algorithm>
#include <assert.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>          // std::memset / std::memcpy
                            // (glibc's `<string.h>` puts them in the
                            //  global namespace only — libc++/Apple
                            //  pull them into `std::` transitively but
                            //  libstdc++ does not.  `<cstring>` is the
                            //  portable C++ way.)
#include <type_traits>
#if defined(__APPLE__)
    #include <malloc/malloc.h>   // for malloc_zone_from_ptr / malloc_zone_free
#endif
#if KAME_FAST_TSD
    #include <pthread.h>
#endif

// Per-thread flag: set to true when AllocThreadExitCleanup has run, signalling
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

// Forward decl — the post-thread-exit functor used by AllocThreadExitCleanup
// to overwrite every slot of `g_thread_slots[]` before chunks are
// released.  Defined later in this TU.

// Forward decl: AllocThreadExitCleanup's dtor needs to drain each per-bucket
// AllocSlot freelist back to the bitmap via the cross-thread TLS batch
// (CrossDeallocBatch is defined further down).  Hide the dependency
// behind a free function defined after CrossDeallocBatch.
namespace { void drain_thread_slot_freelists() noexcept; }

// Per-thread cleanup at thread exit.  Phase 4b: chunks are no longer
// pinned via atomic counters; this destructor instead walks each
// (ALIGN, FS) template's per-thread DLL (via the registered
// `release_dll_chunks_for_thread` callbacks) and either releases
// empty chunks directly or marks non-empty chunks with
// `BIT_OWNER_EXITED` so cross-thread last-slot-returners can release
// them later.  Capacity covers the count of distinct PoolAllocator
// template instantiations actually in use by this thread.
namespace {
struct AllocThreadExitCleanup {
    static constexpr int MAX = 32;
    // `noexcept` is part of the function-pointer type since C++17 — the
    // dylib + tests + production builds (cmake `-std=gnu++17`, qmake
    // `CONFIG += c++17`) compile at C++17 so this is well-formed and
    // matches the implementation's `noexcept` declaration.
    using ReleaseDllFn = void (*)() noexcept;
    ReleaseDllFn release_fns[MAX] = {};
    int count = 0;
    //! Register a per-template DLL teardown callback.  Called once per
    //! thread per (ALIGN, FS) template from `allocate_chunk_path` on
    //! the first mmap-fresh path entry.  Dedup'd so repeated calls
    //! are O(count) but idempotent.
    void add(ReleaseDllFn fn) noexcept {
        for(int i = 0; i < count; ++i)
            if(release_fns[i] == fn) return;
        if(count < MAX) release_fns[count++] = fn;
    }
    ~AllocThreadExitCleanup() noexcept {
        // Drain each per-bucket AllocSlot freelist back to the bitmap
        // FIRST.  Slots on the linked list inside the slot pool would
        // become unreachable after the per-template DLL walk below
        // (which may release the very chunk a slot belongs to).
        // `drain_thread_slot_freelists` issues a per-slot
        // `batch_return_to_bitmap(&one, 1)` via `lookup_chunk` (handles
        // FS=false bucket-share invariant).
        drain_thread_slot_freelists();
        // Clear every per-thread bucket chunk pointer BEFORE the DLL
        // teardown walk.  Otherwise a later TLS destructor that
        // allocates could route through a chunk that's about to be
        // released.  After this loop the slow path's
        // `g_thread_chunks[bucket]` read returns nullptr, so
        // `new_redirected` falls to `cold_first_access`, which
        // observes `s_alloc_tls_off == true` (set a few lines below)
        // and returns `std::malloc(size)`.
        for(int b = 0; b < ALLOC_NUM_BUCKETS; ++b)
            g_thread_chunks[b] = nullptr;
        // Walk each registered template's per-thread DLL.  Each
        // callback wipes its own `s_my_chunk` / `s_dll_head` / `s_dll_tail`
        // first, then iterates with cached-next, setting BIT_OWNER_EXITED
        // on non-empty chunks and releasing empties directly via
        // BIT_RELEASED CAS.  See
        // `PoolAllocator<>::release_dll_chunks_for_thread` for details.
        for(int i = 0; i < count; ++i)
            release_fns[i]();
        // Signal that pool-allocator TLS is dead.  Read by
        // `is_allocator_thread_active()` from later (pthread_key) TLS
        // dtors.  `new_redirected` itself no longer checks this flag —
        // the per-bucket slot rewrite above is its analogue.
        s_alloc_tls_off = true;
    }
};
// Raw `thread_local` — the kamepoolalloc dylib boundary already
// ensures a single shared instance across all plugin DLLs/dylibs
// that link against us, so the cross-DLL slot-sharing concern that
// motivated `XThreadLocal` upstream is gone.
//
// First-touch re-entry safety: C++ thread_local lazy init on macOS
// uses `tlv_allocate_and_initialize_for_key` (libsystem) for the
// storage, and `__cxa_thread_atexit` registers the dtor via
// libcxxabi's `malloc` — both libsystem-malloc paths.  Neither
// recurses into our pool, so first-touch from `allocate()` is safe.
//
// Destruction order: C++ destroys thread_locals in reverse order of
// construction completion.  `AllocThreadExitCleanup` is touched first
// (via `tls_alloc_thread_exit_cleanup.add(...)` in the allocate() hot path),
// `CrossDeallocBatch` second (via `push(...)` in deallocate); so the
// batch is flushed before AllocThreadExitCleanup tears down chunks — the
// ordering invariant the previous XThreadLocal PerThread LIFO chain
// guaranteed.
thread_local AllocThreadExitCleanup tls_alloc_thread_exit_cleanup;

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
    // FS=true-only small-slot batch (Phase 5d-2: FS=false bypasses
    // cross-batch entirely in its `deallocate_pooled` — see that
    // function for rationale).  FS=true buckets are ALIGN==SIZE
    // (16..240 B), one bit per slot in m_flags ⇒ up to 64 slots per
    // FUINT word.  Cross-thread frees of small slots are numerous AND
    // their chunks tend to repeat (a few hot per-size-class chunks
    // serve most allocs), so a deep accumulation window catches
    // same-chunk same-word "buddies" arriving over time → at flush,
    // sort + adjacent-merge coalesces them into one CAS per word.
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

    //! Direct/adaptive dispatch path — FS=true only (Phase 5d-2:
    //! FS=false bypasses cross-batch entirely in `deallocate_pooled`
    //! and never reaches this template).
    //!
    //! FS=true: adaptive.  Reads the chunk's `m_last_coalesce_x16`
    //! hint (relaxed); routes to hold when ≥ per-ALIGN threshold
    //! (compile-time folded), else direct.  Epsilon-greedy explore
    //! force-holds once per `EXPLORE_PERIOD` to re-measure chunks
    //! whose hint dropped below threshold.
    //!
    //! FS=true thresholds (compile-time tiers):
    //!
    //!   ALIGN ≤  64  → 20  (1.25×)
    //!   ALIGN ≤ 128  → 24  (1.50×)
    //!   ALIGN ≤ 256  → 29  (1.81×)
    //!   ALIGN >  256 → 35  (2.19×)
    //!
    //! Not static — the explore counter lives in the per-thread
    //! batch instance, naturally TLS-local.
    static constexpr int EXPLORE_PERIOD = 128;
    int explore_counter = 0;

    template <unsigned ALIGN>
    void push_direct(PoolAllocatorBase *c, void *s) noexcept {
        constexpr uint8_t threshold_x16 =
            (ALIGN <=  64) ? 20 :
            (ALIGN <= 128) ? 24 :
            (ALIGN <= 256) ? 29 : 35;
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
thread_local CrossDeallocBatch tls_cross_dealloc_batch;

// Drain each per-bucket AllocSlot's freelist back to the bitmap.
// Called from `AllocThreadExitCleanup::~dtor`, before the table-wide
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
// dies before `~AllocThreadExitCleanup` and would corrupt freed heap (see
// the AllocThreadExitCleanup comment).
//
// Phase 4b: each touched chunk still has `BIT_OWNER_EXITED == 0`
// at this point (the per-template DLL walk that sets it runs
// AFTER `drain_thread_slot_freelists` in `~AllocThreadExitCleanup`), so
// the cross_release inside batch_return_to_bitmap returns false
// — the owner thread (us) is still alive, no release allowed.
void drain_thread_slot_freelists() noexcept {
    // Single-slot scratch + trailing nullptr sentinel — satisfies
    // `batch_return_to_bitmap`'s `entries[k].chunk == this` walk
    // contract (one matching entry, then the sentinel terminates).
    //
    // Phase 5d: freelists hold p_user pointers for BOTH FS=true and
    // FS=false (the "borrow scheme" puts FS=false's user pointer at
    // slot_start, same convention as FS=true).  `batch_return_to_bitmap`
    // and its MaskFn both work on `entries[k].slot == p_user` directly
    // — for FS=false they read the `{bucket, SIZE}` header from
    // `p_user - 8` (chunk_header pad for slot 0, predecessor's
    // reserved tail otherwise).  No per-FS conversion needed.
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

#if defined(KAMEPOOLALLOC_DYLIB)
// Dylib mode: auto-activate at dylib load.  `__attribute__((constructor))`
// with the priority slot we already use for `kame_tls_init_fast` (101)
// runs after libc/libpthread (which use ≤100) but before any consumer
// image's static-init — so by the time `main()` is reached, every
// `operator new` call is fully pool-routed.  No `activateAllocator()`
// call from user code is necessary; `KamePooledAllocGuard` and the
// per-test `tests/allocator.cpp` activator shim are correspondingly
// elided in dylib builds (see `KAMEPOOLALLOC_DYLIB` branches in
// `allocator.h`, and the dropped `support_SRCS` entry in
// `tests/CMakeLists.txt`).
__attribute__((constructor(101)))
static void kamepoolalloc_auto_activate() noexcept {
    g_sys_image_loaded = true;
}
#else
// Inline-compiled mode (qmake): the kame app and each standalone test
// binary contain `allocator.cpp` as a TU of its own, and the activation
// flag flip stays an explicit step — `kame/main.cpp` does it via
// `KamePooledAllocGuard`, the standalone tests via the static-init
// shim in `tests/allocator.cpp`.  Both are no-ops once the dylib build
// path is selected (which is the case for the cmake test build that
// chases LTO interpose semantics).
void activateAllocator() {g_sys_image_loaded = true;}
#endif

template <unsigned int ALIGN, bool FS, bool DUMMY>
inline PoolAllocator<ALIGN, FS, DUMMY>::PoolAllocator(int count, char *addr, char *ppool) :
	PoolAllocatorBase(ppool),
	m_flags(reinterpret_cast<FUINT *>( &addr[(sizeof(PoolAllocator) + sizeof(FUINT) - 1) * sizeof(FUINT)])),
	m_idx(0),
	m_count(count) {
	// Phase 5j: BIT_OWNED set at construction — the chunk has an
	// owner (the thread doing the chunk-claim and adding to its DLL).
	// MASK_CNT = 0 initially; allocate_pooled bumps it.  BIT_OWNED is
	// cleared by release_dll_chunks_for_thread / owner_release via
	// atomicFetchAnd, which doubles as the release-rights check
	// (newv == 0 ⇒ I'm the unique releaser).
	m_flags_packed = BIT_OWNED;
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
	PoolAllocator<ALIGN, true, false>(count, addr, ppool) {
	// Phase 5c: m_sizes and m_available_bits are gone.  Per-slot SIZE
	// is stored in the slot's own first ALIGN bytes (the "+1 prefix"
	// — see allocate_pooled below).  Nothing further to initialise
	// here: the base ctor zero-clears m_flags, and the prefix bytes
	// for each slot are written at allocate-time before the bitmap CAS
	// publishes the slot's ownership to other threads.
}
template <unsigned int ALIGN, bool DUMMY>
inline PoolAllocator<ALIGN, false, DUMMY> *PoolAllocator<ALIGN, false, DUMMY>::create(size_t size, char *ppool) {
	int count = size / ALIGN / sizeof(FUINT) / 8;
	size_t size_alloc = (sizeof(PoolAllocator) + sizeof(FUINT) - 1) * sizeof(FUINT);
	// Phase 5c layout: [class][m_flags].  The previous trailing m_sizes
	// array (count × FUINT bytes) is gone — per-slot SIZE now lives in
	// each slot's own first ALIGN-bytes prefix, claimed from the bitmap
	// as one extra bit alongside the user data (N+1 bits per slot).
	char *area = static_cast<char *>(malloc(size_alloc + sizeof(FUINT) * count));
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
					atomicInc( &this->m_flags_packed);
				if(newv == ~(FUINT)0u) {
                    atomicInc( &this->m_flags_filled_cnt);
                    // Proactive Phase 3 trigger: when this chunk hits 4/5
                    // (80 %) of its words fully filled, flush this
                    // thread's cross-dealloc batch.  Any batched frees
                    // for OTHER chunks land back in their bitmaps,
                    // letting the next chunk-full event's DLL scan
                    // (Phase 2) find recovered space before mmaping
                    // fresh memory.  Sampled at word-fill granularity
                    // (~1 in FUINT_BITS = 64 allocs) so the overhead is
                    // amortised; `flush()` is a no-op when the batch is
                    // empty so post-cross-event calls are cheap.
                    if(this->m_flags_filled_cnt * 5 >= this->m_count * 4)
                        tls_cross_dealloc_batch.flush();
                }
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
	// runs the bitmap CAS to claim N contiguous free bits (Phase 5d:
	// "borrow scheme" — the per-slot `{uint32_t bucket, uint32_t SIZE}`
	// header lives in the LAST 8 bytes of the PREVIOUS slot's ALIGN
	// area, or in `chunk_header[56..63]` for slot 0 at bit 0/word 0.
	// No separate "prefix bit" is claimed.  See allocator_prv.h's
	// chunk-header layout doc (Phase 5d-3) for the formal reservation.
	//
	// User pointer p = slot_start (ALIGN-aligned ✓).
	// Header at `p - 8`:
	//   * For slot at bit 0 of word 0: `mempool - 8` = `chunk_base +
	//     ALLOC_CHUNK_HEADER - 8` = `chunk_header[56..63]` —
	//     formally reserved by Phase 5d-3 (static_assert in
	//     allocator_prv.h confirms ≥ 8 B of pad before this region).
	//   * For slot at bit B > 0: byte position `B*ALIGN - 8` = LAST
	//     8 bytes of bit (B-1)'s ALIGN area.  Universal invariant:
	//     every allocated slot reserves its OWN last 8 bytes as
	//     storage for the next slot's header, so `user_area =
	//     N*ALIGN - 8 bytes` and the reservation is never trampled
	//     by user writes.
	//
	// Required N is the smallest value satisfying
	// `N*ALIGN - 8 >= SIZE`, i.e. `N = ceil((SIZE + 8) / ALIGN)`.
	// For the existing bucket-size schedule (SIZE = K*ALIGN), this
	// equals `K + 1` — same bit count as Phase 5c's `N_user + 1`.
	const unsigned int N = (SIZE + 8u + ALIGN - 1u) / ALIGN;
	// Phase 5d header content: bucket index + SIZE packed into 8 bytes.
	// Computed once on the cold allocate path so the dealloc hot path
	// can read bucket directly (no `bucket_for_size` call).
	const std::uint32_t bucket_idx = bucket_for_size(SIZE);
	const std::uint64_t hdr_word =
	    static_cast<std::uint64_t>(bucket_idx)
	  | (static_cast<std::uint64_t>(SIZE) << 32);
	// Phase 5f: dropped the Phase 5a 80% fragmentation cutoff — it
	// walked all m_count FUINT words via count_bits on every
	// allocate_pooled call (catastrophic on high-level chunks where
	// m_count ≈ 4096 → ~4 µs/alloc).  The walk-once-and-bail logic
	// below is also bounded by m_count and only pays that cost when
	// the chunk is truly out of N-contiguous-zero runs (rare; only
	// at chunk-fill boundary).  Quick exit when all FUINT words are
	// fully filled — `m_flags_filled_cnt` is incrementally
	// maintained (atomicInc on word-becomes-all-ones inside the CAS
	// below; atomicDec on word-becomes-zero in batch_return_to_bitmap).
	if(this->m_flags_filled_cnt == this->m_count)
		return 0;

	FUINT oldv, ones, cand;
	int idx = this->m_idx;
	FUINT *pflag = &this->m_flags[idx];
	int sidx = 0;
	char *slot_start = nullptr;
	int walked = 0;  // count of distinct m_flags words visited (max = m_count)
	for(;;) {
		oldv = *pflag;
		cand = find_training_zeros(N, oldv);
		if(cand) {
			ones = cand *
				(2u * (((FUINT)1u << (N - 1u)) - 1u) + 1u); //N ones, not to overflow.
//			assert(count_bits(ones) == N);
//			assert( !(ones & oldv));
			sidx = count_zeros_forward(cand);
			int idx_cand = pflag - this->m_flags;
			slot_start = &this->m_mempool[
			    (size_t(idx_cand) * sizeof(FUINT) * 8 + sidx) * ALIGN];
			// Phase 5d: write the {bucket, SIZE} header BEFORE the CAS
			// publishes the bit.  Header lives at `slot_start - 8`:
			//   * Bit 0 of word 0 (slot_start == mempool):
			//       slot_start - 8 = chunk_base + 56 (Phase 5d-3
			//       reserved area in chunk-header pad).
			//   * Bit B > 0: slot_start - 8 lands in bit (B-1)'s last 8
			//       bytes, either inside an allocated slot whose
			//       user_area excludes its last 8 B (universal
			//       reservation invariant) or in a free bitmap region.
			// CAS publishes the header via release semantics.
			*reinterpret_cast<std::uint64_t *>(slot_start - 8) = hdr_word;
			// Always-CAS path (cf. FS=true sibling): TLS s_my_chunk
			// fast path removes the bit0-lock around chunk access, so
			// a non-atomic store would torn-write under contention.
			FUINT newv = oldv | ones; //filling with N ones (all user bits).
			if(atomicCompareAndSet(oldv, newv, pflag)) {
				if(oldv == 0)
					atomicInc( &this->m_flags_packed);
				// Phase 5f: maintain m_flags_filled_cnt so the quick
				// "chunk is 100% full" check above stays accurate.
				if(newv == ~(FUINT)0u)
					atomicInc( &this->m_flags_filled_cnt);
				break;
			}
			continue;  // CAS race, retry same word
		}
		// No N-contiguous zeros in this word — advance to next.
		++pflag;
		++walked;
		if(walked >= this->m_count) {
			// Full sweep without finding a slot.  Chunk is too
			// fragmented for N consecutive zeros even though some
			// words have free bits.  Bail; caller picks another chunk.
			return 0;
		}
		if(pflag == &this->m_flags[this->m_count])
			pflag = this->m_flags;  // wrap to start
	}

	idx = pflag - this->m_flags;
	this->m_idx = idx;

	// Return the USER pointer: slot_start is the first claimed bit's
	// byte position, which IS the user data start (header is at
	// slot_start - 8 in the borrow scheme).
	return slot_start;
}
template <unsigned int ALIGN, bool DUMMY>
bool
PoolAllocator<ALIGN, false, DUMMY>::deallocate_pooled(char *p) {
	// Phase 5d: {bucket, SIZE} header at `p - 8` (LAST 8 bytes of the
	// prefix bit's ALIGN area).  One 64-bit load recovers bucket
	// directly — no `bucket_for_size(SIZE)` call on the hot path.
	//
	// Owner-side dealloc: push to the per-thread
	// `g_thread_slots[bucket].freelist_head` — exactly the slot that
	// `new_redirected` pops from on the next allocation of that size.
	// Non-owner OR slot 0 (bucket==0 sentinel reserved by 5d-3) routes
	// to the cross-thread path which uses slot_start for batch dispatch.
	//
	// `s_my_chunk` has declared type `PoolAllocator<ALIGN, false, false>*`
	// (from the base's `PoolAllocator<ALIGN, DUMMY, DUMMY>*` with
	// DUMMY=false), while `this` has type `PoolAllocator<ALIGN, false,
	// DUMMY>*` — different template instantiations referring to the
	// same chunk object.  Compare as void* to bypass the type mismatch.
	if(static_cast<void *>(PoolAllocator<ALIGN, true, false>::s_my_chunk)
	    == static_cast<void *>(this)) {
		std::uint64_t hdr = *reinterpret_cast<std::uint64_t *>(p - 8);
		unsigned bucket = static_cast<unsigned>(hdr);
		// Defensive bound check (bucket must be valid; misroute on
		// stale data is detected and falls through to bitmap path).
		if(bucket >= 1 && bucket < ALLOC_NUM_BUCKETS) {
			kame_slots_base()[bucket].push(p);
			return false;
		}
	}
	// Phase 5d-2: FS=false never participates in cross-thread batch
	// holding — large slots have small per-word coalescing windows AND
	// large-slot chunks repeat less frequently than FS=true small-slot
	// chunks, so holding cost wouldn't pay back.  Empirically (ohtaka)
	// even epsilon-greedy explore couldn't recover useful coalescing
	// factor for FS=false.  Drop the whole machinery; route every
	// cross-thread / non-owner / post-teardown free directly to a
	// single-entry `batch_return_to_bitmap` call.
	//
	// This also subsumes the `s_alloc_tls_off` post-teardown bypass
	// (the old code had a separate branch for it because the cross-
	// batch TLS instance had already been destroyed) — we never touch
	// `tls_cross_dealloc_batch` here so the post-teardown case is
	// implicit.
	CrossDeallocEntry tmp[2] = {{this, p}, {nullptr, nullptr}};
	this->batch_return_to_bitmap(tmp);
	return false;
}

// FS=false non-virtual static trampoline.  Sibling of the FS=true
// `deallocate_pooled_static` above — see that comment for the
// rationale (chunk-header fn pointer dispatch on the hot path).
template <unsigned int ALIGN, bool DUMMY>
bool
PoolAllocator<ALIGN, false, DUMMY>::deallocate_pooled_static(
    PoolAllocatorBase *base, char *p) {
	PoolAllocator *self = static_cast<PoolAllocator *>(base);
	return self->PoolAllocator::deallocate_pooled(p);
}

// FS=false slot-size trampoline (Phase 5d).  Reads SIZE from the
// `{bucket, SIZE}` header at `p - 8` (the LAST 8 bytes of the prefix
// bit's ALIGN area; allocate_pooled wrote it there before publishing
// the bitmap bit).  Used by `realloc()` via
// `PoolAllocatorBase::size_of()` to recover the exact allocated byte
// count for a slot.  The high 32 bits of the 64-bit header hold SIZE.
template <unsigned int ALIGN, bool DUMMY>
std::size_t
PoolAllocator<ALIGN, false, DUMMY>::size_of_static(
    PoolAllocatorBase * /*base*/, char *p) noexcept {
	return static_cast<std::size_t>(
	    *reinterpret_cast<std::uint32_t *>(p - 4));
}

// FS=false batch return (Phase 5d "borrow" scheme) — N-bit clear
// where N = ceil((SIZE + 8) / ALIGN).  Caller passes `p` = p_user
// (= slot_start in the borrow scheme).  The `{bucket, SIZE}` header
// lives at `p - 8` (= chunk_header pad for slot 0, or previous slot's
// reserved last-8 bytes otherwise).  Reuses the inherited
// batch_clear_impl skeleton with a borrow-scheme MaskFn and FS=false-
// specific OnClearFn (no filled_cnt; m_available_bits is gone since
// Phase 5c's fragmentation cutoff in allocate_pooled replaces it).
template <unsigned int ALIGN, bool DUMMY>
int
PoolAllocator<ALIGN, false, DUMMY>::batch_return_to_bitmap(
    const CrossDeallocEntry *entries) noexcept {
	// Walk entries[k] while .chunk == this — terminates on the next
	// chunk's group OR the trailing {nullptr, nullptr} sentinel that
	// `CrossDeallocBatch::flush` plants at buf[count].  No `k < n_max`
	// test in the inner loop.  Drain / post-teardown single-slot paths
	// pass a stack-local {this, p_user} + sentinel pair.
	bool i_am_releaser = false;
	int n = this->batch_clear_impl(entries,
		// MaskFn: Phase 5d FS=false N-bit clear (borrow scheme).
		// `p` is p_user (= slot_start); the `{bucket, SIZE}` header
		// is at `p - 8` (in previous slot's reserved 8 B tail OR in
		// chunk_header pad for slot 0).  SIZE is the upper 32 bits.
		// N = ceil((SIZE + 8) / ALIGN) bits starting at the slot's
		// own bit position `sidx`.
		[](int /*idx*/, unsigned sidx, char *p) -> FUINT {
			unsigned size_bytes = *reinterpret_cast<std::uint32_t *>(
			    p - 4);
			unsigned N = (size_bytes + 8u + ALIGN - 1u) / ALIGN;
			FUINT slot_mask = (((FUINT(1) << N) - FUINT(1))) << sidx;
#ifdef GUARDIAN
			for(unsigned int j = 0;
			    j < N * ALIGN / sizeof(uint64_t); ++j)
				reinterpret_cast<uint64_t *>(p)[j] = GUARDIAN;
#endif
			return slot_mask;
		},
		// OnClearFn: FS=false.  Phase 5j: use atomicDecAndTest to
		// identify the unique releaser when MASK_CNT goes 1 → 0.  If
		// dec returns 0, BIT_OWNED was already clear (owner is gone)
		// AND MASK_CNT was 1 → this caller is the sole releaser.
		[this, &i_am_releaser](FUINT oldv, FUINT newv) {
			if(newv == 0 && oldv != 0) {
				if(atomicDecAndTest(&this->m_flags_packed))
					i_am_releaser = true;
			}
			// Phase 5f: maintain m_flags_filled_cnt symmetric with
			// allocate_pooled's atomicInc on word-becomes-all-ones.
			if(oldv == ~(FUINT)0u)
				atomicDec( &this->m_flags_filled_cnt);
		});
	if(i_am_releaser) {
		// Phase 5j: atomicDecAndTest uniquely identified me as the
		// releaser (m_flags_packed transitioned to 0 = BIT_OWNED clear,
		// MASK_CNT == 0).  No CAS race possible — owner exit's
		// atomicFetchAnd(~BIT_OWNED) and the dec-to-0 are mutually
		// exclusive (only one operation brings the word to all-zero).
		char *cbase = this->m_mempool - ALLOC_CHUNK_HEADER;
		size_t csz = this->m_chunk_size;
		delete this;
		PoolAllocatorBase::deallocate_chunk(cbase, csz);
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
	// happened.  Relaxed: it's just a hint, races benign.  Skip for
	// FS=false — Phase 5d-2 bypasses cross-batch entirely on the
	// FS=false dealloc path (direct single-entry batch_return_to_bitmap
	// call), so the hint is never consulted and storing it would be
	// wasted work.
	if constexpr (FS) {
		if(n_words > 0) {
			unsigned factor = (unsigned(i) * 16u) / unsigned(n_words);
			if(factor > 255u) factor = 255u;
			this->m_last_coalesce_x16.store(uint8_t(factor),
			                                std::memory_order_relaxed);
		}
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
	bool i_am_releaser = false;
	int n = this->batch_clear_impl(entries,
		// MaskFn: FS=true single bit
		[](int /*idx*/, unsigned sidx, char * /*p*/) -> FUINT {
			return ((FUINT)1u) << sidx;
		},
		// OnClearFn: FS=true.  Phase 5j: use atomicDecAndTest to
		// identify the unique releaser when MASK_CNT goes 1 → 0.
		[this, &i_am_releaser](FUINT oldv, FUINT newv) {
			if(oldv == ~(FUINT)0u)
				atomicDec( &this->m_flags_filled_cnt);
			if(newv == 0 && oldv != 0) {
				if(atomicDecAndTest(&this->m_flags_packed))
					i_am_releaser = true;
			}
		});
	if(i_am_releaser) {
		// Phase 5j: dec brought m_flags_packed to 0 → BIT_OWNED was
		// already clear (owner gone) AND MASK_CNT was 1.  I am
		// uniquely the releaser.
		char *cbase = this->m_mempool - ALLOC_CHUNK_HEADER;
		size_t csz = this->m_chunk_size;
		delete this;
		PoolAllocatorBase::deallocate_chunk(cbase, csz);
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
		// AllocThreadExitCleanup (thread exit) or the chunk's bitmap is
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
		tls_cross_dealloc_batch.push(this, p);
	} else {
		tls_cross_dealloc_batch.template push_direct<ALIGN>(this, p);
	}
	return false;
}

// FS=true non-virtual static trampoline for the chunk-header fn
// pointer.  `allocate_chunk` stores `&PoolAllocator<ALIGN, FS, DUMMY>::
// deallocate_pooled_static` at chunk_base + ALLOC_CHUNK_HEADER_FN_OFFSET;
// `deallocate_<>`'s hot path reads it and invokes `fn(palloc, p)` —
// one indirect branch, no vtable lookup.  The qualified-name call
// `self->PoolAllocator::deallocate_pooled(p)` compiles to a direct
// branch (non-virtual) on the bound derived type's body.
template <unsigned int ALIGN, bool FS, bool DUMMY>
bool
PoolAllocator<ALIGN, FS, DUMMY>::deallocate_pooled_static(
    PoolAllocatorBase *base, char *p) {
	PoolAllocator *self = static_cast<PoolAllocator *>(base);
	return self->PoolAllocator::deallocate_pooled(p);
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
// PoolAllocator<ALIGN, false> instantiation, so the bucket's slot SIZE
// (= max user_size) differs from ALIGN and must be derived from
// `bucket` at runtime.  Phase 5d-4 4-way exponential layout:
//   bucket 1..16  →  slot_size = bucket * 16        (sizes 16..256, FS=true mixed; this branch is reached
//                                                    via the FS=false specialisation only for buckets 6, 8,
//                                                    10, 12, 14, 16 — the FS=false-half of the mixed range)
//   bucket 17..24 →  4-way octave 8/9/10 sub 1..3/0..3/0  (ALIGN=64, user_size = total - 8)
//   bucket 25..32 →  4-way octave 10/11/12 sub 1..3/0..3/0 (ALIGN=256)
//   bucket 33..40 →  4-way octave 12/13/14 sub 1..3/0..3/0 (ALIGN=1024)
//
// The FS=false `allocate_pooled` expects SIZE = user size (max user
// bytes the bucket serves).  Internally it computes
// N = ceil((SIZE + 8) / ALIGN) (Phase 5d-1 borrow scheme).
template <unsigned int ALIGN, bool DUMMY>
__attribute__((cold, noinline))
void *
PoolAllocator<ALIGN, false, DUMMY>::slow_allocate(unsigned bucket,
                                                  std::size_t /*size*/) noexcept {
	// Phase 5q inverse of `bucket_for_size`: bucket index → max
	// user_size.
	//
	// Buckets 1..23 (FS=true + FS=false mixed): slot_size = bucket * 16.
	//   - 1..16: original Phase 5d-4 range (16..256 B).
	//   - 17..23: Phase 5q FS=true extension (272..368 B).
	//
	// Buckets 24..47 (FS=false N+1 shift, Phase 5p): 4-way exponential.
	//   bucket_offset = bucket - 23 (1..24).
	//   octave = 8 + bucket_offset / 4 (8..14).
	//   sub = bucket_offset % 4 (0..3).
	//   OLD total = (1<<octave) + (1<<(octave-2)) * sub.
	//   NEW slot = OLD + ALIGN_tier (Phase 5p N+1 shift).
	//   user_size = NEW slot - 8 (Phase 5d-1 borrow header).
	unsigned int slot_size;
	if(bucket <= 23) {
		slot_size = bucket * 16u;
	}
	else {
		unsigned int off    = bucket - 23u;       // 1..24
		unsigned int octave = 8u + off / 4u;       // 8..14
		unsigned int sub    = off % 4u;            // 0..3
		unsigned int total  = (1u << octave) + (1u << (octave - 2u)) * sub;
		slot_size = total + ALIGN - 8u;
	}
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
	// Phase 5l + 5m: uniform 32 MiB regions carved into 128 × 256 KiB
	// units.  A chunk = `CHUNK_UNITS` (1, 2, or 4) contiguous units at
	// a unit-aligned position.  Per-region claim/ready bitmap stays at
	// 2 bits/unit (Phase 5h encoding) but is now per UNIT — multi-unit
	// chunks set CHUNK_UNITS adjacent claim bits in a single CAS, and
	// only the base unit ever gets `ready = 1`.
	//
	// Phase 5m: `s_region_has_free[]` skip-bitmap eliminates the O(N)
	// walk-past-full-regions cost.  Two passes:
	//   1. Walk set bits of `s_region_has_free` — try each region; on
	//      failure (region fully claimed), clear the bit and continue.
	//   2. If pass 1 exhausted, find an unallocated region, mmap it,
	//      set its bit, and claim there.
	constexpr unsigned int CHUNK_UNITS = ALLOC::CHUNK_UNITS;
	constexpr size_t CHUNK_SIZE = ALLOC::CHUNK_SIZE;
	constexpr unsigned int CHUNK_STRIDE_BITS = 2u * CHUNK_UNITS; // bits per chunk in bitmap
	// All-bits-of-this-chunk's-pairs mask, anchored at bit 0.  E.g.
	// CHUNK_UNITS=1 → 0b11; =2 → 0b1111; =4 → 0xFF.
	constexpr BitmapWord CHUNK_OCC_MASK =
	    (CHUNK_STRIDE_BITS >= sizeof(BitmapWord) * 8u)
	        ? ~BitmapWord(0)
	        : ((BitmapWord(1) << CHUNK_STRIDE_BITS) - BitmapWord(1));

	// Inline-able lambda: try to claim one chunk in a specific region.
	// Returns palloc on success, nullptr if every CHUNK_UNITS-aligned
	// slot in the region's bitmap is already claimed.
	auto try_claim_in_region = [&](int region) -> ALLOC * {
		for(int word = 0; word < BITMAP_WORDS_PER_REGION; ++word) {
			std::atomic<BitmapWord> *bm =
			    &s_claim_bitmap[region * BITMAP_WORDS_PER_REGION + word];
			for(;;) {
				BitmapWord v = bm->load(std::memory_order_relaxed);
				int free_pos = -1;
				for(unsigned k = 0; k + CHUNK_STRIDE_BITS <= (unsigned)BITS_PER_BITMAP_WORD;
				    k += CHUNK_STRIDE_BITS) {
					if(((v >> k) & CHUNK_OCC_MASK) == 0) {
						free_pos = (int)k;
						break;
					}
				}
				if(free_pos < 0) break;  // word saturated
				BitmapWord claim_mask = 0;
				for(unsigned u = 0; u < CHUNK_UNITS; ++u)
					claim_mask |= BitmapWord(1) << (free_pos + 2u * u);
				BitmapWord newv = v | claim_mask;
				int base_unit_in_word = free_pos / 2;
				int base_unit_idx =
				    word * CHUNKS_PER_BITMAP_WORD + base_unit_in_word;

				size_t bo_base = (size_t)region * NUM_ALLOCATORS_IN_SPACE
				                 + (size_t)base_unit_idx;
				for(unsigned u = 0; u < CHUNK_UNITS; ++u)
					s_back_offset[bo_base + u] = (uint8_t)u;
				writeBarrier();

				if(bm->compare_exchange_weak(v, newv,
				                             std::memory_order_acquire,
				                             std::memory_order_relaxed)) {
					char *addr = s_mmapped_spaces[region]
					           + (size_t)base_unit_idx * (size_t)ALLOC_MIN_CHUNK_SIZE;
#if !(defined __WIN32__ || defined WINDOWS || defined _WIN32)
					static const bool prewarm = [] {
						const char *e = std::getenv("KAME_ALLOC_PREWARM");
						return e && e[0] != '\0' && e[0] != '0';
					}();
					if(prewarm) {
						for(size_t off = 0; off < CHUNK_SIZE; off += ALLOC_PAGE_SIZE)
							reinterpret_cast<volatile char *>(addr)[off] = 0;
					}
#endif
					ALLOC *palloc = ALLOC::create(CHUNK_SIZE - ALLOC_CHUNK_HEADER,
					                              addr + ALLOC_CHUNK_HEADER);
					palloc->m_chunk_size = CHUNK_SIZE;
					*reinterpret_cast<std::uint64_t *>(
					    addr + ALLOC_CHUNK_HEADER_SIZE_INFO_OFFSET) =
					    ALLOC::chunk_header_size_info();
					*reinterpret_cast<PoolAllocatorBase **>(
					    addr + ALLOC_CHUNK_HEADER_PALLOC_OFFSET) = palloc;
					*reinterpret_cast<DeallocateFn *>(
					    addr + ALLOC_CHUNK_HEADER_FN_OFFSET) =
					    &ALLOC::deallocate_pooled_static;
					*reinterpret_cast<SizeOfFn *>(
					    addr + ALLOC_CHUNK_HEADER_SIZEOF_FN_OFFSET) =
					    &ALLOC::size_of_static;
					writeBarrier();
					BitmapWord ready_mask =
					    BitmapWord(1) << (free_pos + 1);
					bm->fetch_or(ready_mask, std::memory_order_release);
					return palloc;
				}
				// CAS failed: concurrent claim updated v.  Retry inner
				// loop with the new v (back_offset writes we did above
				// are inert until the claim bit publishes).
			}
			// word fully claimed for THIS CHUNK_UNITS alignment.  Try
			// the next word.
		}
		// All words in this region full for this CHUNK_UNITS.
		return nullptr;
	};

	// Pass 1: walk regions whose has-free bit is set.
	for(int rword = 0; rword < REGION_BITMAP_WORDS; ++rword) {
		uint64_t rv =
		    s_region_has_free[rword].load(std::memory_order_relaxed);
		while(rv != 0) {
			int rbit = __builtin_ctzll(rv);
			int region = rword * 64 + rbit;
			if(region >= ALLOC_MAX_MMAP_ENTRIES) break;
			if(ALLOC *palloc = try_claim_in_region(region))
				return palloc;
			// Region is full for OUR CHUNK_UNITS.  Note that other
			// templates with smaller CHUNK_UNITS may still find space
			// here — so we clear the bit only relaxed-tentatively;
			// future deallocs in this region will fetch_or it back
			// when space genuinely opens up.
			s_region_has_free[rword].fetch_and(
			    ~(uint64_t(1) << rbit),
			    std::memory_order_relaxed);
			rv &= ~(uint64_t(1) << rbit);
		}
	}

	// Pass 2: find an unallocated region, mmap it, then claim.
	for(int region = 0; region < ALLOC_MAX_MMAP_ENTRIES; ++region) {
		while( !s_mmapped_spaces[region]) {
			size_t mmap_size = ALLOC_MIN_MMAP_SIZE;
			// Region alignment = ALLOC_MAX_CHUNK_SIZE so any in-region
			// multi-unit chunk (up to CHUNK_UNITS_MAX) lands at a
			// chunk_size-aligned absolute address — still useful for
			// debug-dump / future AND-mask micro-opt.  back_offset[]
			// already gives O(1) chunk_base lookup so the AND-mask
			// trick is not on the hot path anymore.
			constexpr size_t kAlign = ALLOC_MAX_CHUNK_SIZE;
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
			char *p = static_cast<char *>(_aligned_malloc(mmap_size, kAlign));
			if( !p) {
				fprintf(stderr,
				    "_aligned_malloc(%zu, %zu) failed.\n",
				    mmap_size, kAlign);
				return 0;
			}
#else
			// mmap returns page-aligned (4-16 KiB), not kAlign-aligned.
			// Over-allocate by one kAlign and munmap the head + tail
			// so the kept region starts at a kAlign boundary.  Cost:
			// 1 × kAlign of VA wasted per region (= 1 MiB on 64-bit).
			// Region size 32 MiB ⇒ ~3 % overhead in VA reservation,
			// trivial in RSS (PROT_READ|PROT_WRITE without writes ≈
			// zero RSS until pages are first touched).
			size_t total = mmap_size + kAlign;
			char *raw = static_cast<char *>(
			    mmap(0, total, PROT_READ | PROT_WRITE,
			         MAP_ANON | MAP_PRIVATE, -1, 0));
			if(raw == MAP_FAILED) {
				fprintf(stderr, "mmap() failed.\n");
				return 0;
			}
			uintptr_t aligned =
			    ((uintptr_t)raw + kAlign - 1u) & ~(uintptr_t)(kAlign - 1u);
			char *p = reinterpret_cast<char *>(aligned);
			size_t prefix = p - raw;
			size_t suffix = total - prefix - mmap_size;
			if(prefix > 0) munmap(raw, prefix);
			if(suffix > 0) munmap(p + mmap_size, suffix);
#endif
			writeBarrier();
			if(atomicCompareAndSet((char *)0, p, &s_mmapped_spaces[region])) {
				readBarrier();
				fprintf(stderr,
				    "Reserve swap space starting @ %p w/ len. of 0x%llxB.\n",
				    p, (unsigned long long)mmap_size);
				break;
			}
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
			_aligned_free(p);
#else
			munmap(p, mmap_size);
#endif
		}

		// Region is freshly mmap'd: mark it has-free (idempotent OR)
		// BEFORE attempting claim, so other allocate_chunk callers
		// concurrently entering Pass 1 can see it.
		s_region_has_free[region / 64].fetch_or(
		    uint64_t(1) << (region % 64),
		    std::memory_order_relaxed);

		if(ALLOC *palloc = try_claim_in_region(region))
			return palloc;
		// Shouldn't happen — a freshly mmap'd region has 128 units
		// all free, which always satisfies any CHUNK_UNITS up to
		// ALLOC_MAX_CHUNK_UNITS.  Defensive: fall through to the next
		// region (race against another claimer racing us at the same
		// region somehow).
	}
	fprintf(stderr, "# of chunks exceeds the limit.\n");
	return 0;
}
// Phase 4b-final: chunk-claim is purely mmap.  No global registry —
// the per-thread DLL is the sole source of truth for "chunks this
// thread can allocate from"; per-chunk ownership is encoded in the
// chunk header's `PoolAllocatorBase *` (visible to cross-thread frees
// via `lookup_chunk`) and the chunk's `m_flags_packed` BIT_RELEASED
// race point.
template <unsigned int ALIGN, bool FS, bool DUMMY>
PoolAllocator<ALIGN, DUMMY, DUMMY> *
PoolAllocator<ALIGN, FS, DUMMY>::create_allocator() {
	PoolAllocator<ALIGN, DUMMY, DUMMY> *palloc =
		allocate_chunk<PoolAllocator<ALIGN, DUMMY, DUMMY> >();
	if( !palloc) {
		fprintf(stderr,
		    "# of chunks for %d align. exceeds the mmap region ladder.\n", ALIGN);
		throw std::bad_alloc();
	}
	return palloc;
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
	// Thread-exit cleanup is handled centrally by `AllocThreadExitCleanup::~dtor`
	// (registered via `XThreadLocal<AllocThreadExitCleanup>::operator*()` on the
	// first call that pins a chunk, a few lines below).  No per-template
	// thread_local guard is needed here, and the previous `(void)&s_tls_guard`
	// ODR-use is removed so we don't pay a C++ thread_local init thunk call
	// per allocation (macOS arm64 emits `bl __ZTH...11s_tls_guardE`).
	// Try the bitmap-CAS path on the current `s_my_chunk` before
	// falling all the way through to the DLL scan + mmap-fresh path.
	// allocate_pooled() does its own per-flag atomic CAS so concurrent
	// allocations from the same chunk by other threads are safe.
	// Phase 4b: `s_my_chunk` is a DLL member of this thread — only
	// this thread can release it (via Phase 4a `owner_release` or
	// thread-exit `release_dll_chunks_for_thread`), so no other-thread
	// guard is needed.
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
	// Phase 3 (chunk-full trigger): right after detecting that the
	// active chunk has filled, flush this thread's cross-thread
	// dealloc batch.  The flushed entries return slots to the
	// originating chunks (often this thread's own DLL members from
	// earlier in the run) — by the time the Phase 2 DLL scan below
	// looks for a chunk with room, those chunks may have just
	// recovered space.  Net effect:
	//   * mmap pressure reduced (DLL scan finds reusable chunks
	//     instead of falling through to `create_allocator`)
	//   * batched cross-thread frees don't get postponed past the
	//     point of memory growth
	//
	// Cost: O(N log N) sort + per-chunk bitmap CAS on the batch (N ≤
	// CAP = 1024).  Paid once per chunk-fill event — orders of
	// magnitude rarer than the per-allocation hot path.  Safe to call
	// with an empty batch (early-out inside `flush`).
	//
	// Other threads' chunks emptied by this flush are NOT released
	// here — the owning thread (eventually) handles those via the
	// same trigger, or via Phase 4's thread-exit cleanup.  Own
	// chunks emptied by the flush are visible to the Phase 2 scan
	// directly below.
	// Phase 5n: only reset cursor / exhausted if the batch actually
	// had entries that we are about to flush back to chunk bitmaps.
	// In pure-alloc workloads (alloc_only, no cross-frees), the batch
	// stays empty, the flush is a no-op, and we keep the cursor's
	// O(N) → O(1) advantage.  Stress workloads see real cross-frees,
	// the count > 0 path fires, and the cursor rewinds so partial
	// revivals are visited.
	if(tls_cross_dealloc_batch.count != 0) {
		tls_cross_dealloc_batch.flush();
		s_dll_cursor = nullptr;
		s_dll_exhausted = false;
	}
	// Phase 4a: own-empty-neighbour release.  Walk forward from
	// `s_my_chunk` along the DLL: if the immediate next chunk is
	// empty ((m_flags_packed & MASK_CNT) == 0), release it; if its successor
	// is *also* empty, release that too — capped at two consecutive
	// releases per trigger so the cost stays bounded.  Steady-state
	// memory growth = mmap rate (one new chunk per chunk-fill); this
	// release path balances it at the same cadence.
	//
	// Safety: Phase 4b `owner_release` CAS's `BIT_RELEASED` on the
	// chunk's `m_flags_packed` — this races safely against any
	// cross-thread `cross_release`, but cross_release additionally
	// requires `BIT_OWNER_EXITED == 1` which only the owner's
	// exit-path sets, so while we're alive only `owner_release` can
	// win the race.  Caller (us) handles the post-CAS DLL unlink +
	// `delete` + `deallocate_chunk`.
	if(s_my_chunk) {
		auto *nx = s_my_chunk->m_dll_next;
		for(int released = 0; nx && released < 2; ) {
			auto *nxnext = nx->m_dll_next;
			if((nx->m_flags_packed & PoolAllocator<ALIGN, DUMMY, DUMMY>::MASK_CNT) != 0)
				break;  // hit a non-empty
			// `nx` is `PoolAllocator<ALIGN, DUMMY, DUMMY> *` (same
			// type as `s_my_chunk`).  Use the current template's
			// `owner_release` (FS=true and FS=false specialisations
			// share the static — `m_flags_packed` lives on the
			// FS=true base).
			if(PoolAllocator<ALIGN, FS, DUMMY>::owner_release(nx)) {
				// DLL unlink (single-writer = us).
				if(nx->m_dll_prev) nx->m_dll_prev->m_dll_next = nx->m_dll_next;
				else               s_dll_head = nx->m_dll_next;
				if(nx->m_dll_next) nx->m_dll_next->m_dll_prev = nx->m_dll_prev;
				else               s_dll_tail = nx->m_dll_prev;
				char *cbase = nx->m_mempool - ALLOC_CHUNK_HEADER;
				size_t csz = nx->m_chunk_size;
				// Phase-4a stale-cache invariant: multiple FS=false
				// buckets share one PoolAllocator<ALIGN,false>
				// template, so the released chunk `nx` may still be
				// cached in `g_thread_chunks[b]` for sibling buckets
				// that never triggered a chunk-switch.  Sweep all
				// per-thread bucket slots and clear matching pointers
				// BEFORE `delete nx`; the next `new_redirected_large`
				// on those buckets will route via `cold_first_access`
				// → `bucket_first_access<B>` and re-pin against the
				// current valid `s_my_chunk` for this template.
				// Without this sweep the sibling slots become dangling
				// pointers into freed malloc memory — the next
				// virtual `chunk->slow_allocate(...)` dispatch reads a
				// trashed vtable and jumps into garbage.
				//
				// FS=true templates don't share buckets so the sweep is
				// a few comparisons of unrelated PoolAllocator pointers
				// (always misses) on those paths; cheap enough to keep
				// unconditional.
				PoolAllocatorBase *nx_pa = static_cast<PoolAllocatorBase *>(nx);
				for(int b = 0; b < ALLOC_NUM_BUCKETS; ++b) {
					if(g_thread_chunks[b] == nx_pa)
						g_thread_chunks[b] = nullptr;
				}
				// Phase 5n: if the cursor was pointing at the released
				// chunk, advance past it.  Also clear the exhaustion
				// flag — DLL was just modified; an earlier chunk
				// (now closer to head via the unlink) might have had
				// cross-thread frees we didn't see during the previous
				// walk.  Conservative reset → next allocate_chunk_path
				// rewalks from head.
				if(s_dll_cursor == nx)
					s_dll_cursor = nxnext;
				s_dll_exhausted = false;
				delete nx;
				PoolAllocatorBase::deallocate_chunk(cbase, csz);
				++released;
			}
			else {
				// `owner_release` refused (LEAVE_VACANT_CHUNKS floor
				// or a racing re-allocation).  Stop — don't try
				// further neighbours, the chunk is still in use.
				break;
			}
			nx = nxnext;
		}
	}
	// Phase 2: forward-scan this thread's already-claimed chunks for
	// one that has room.  Cross-thread frees on our previously-active
	// chunks routinely empty out bits while those chunks sit older in
	// the DLL — Phase 4b made this the SOLE chunk-reuse mechanism
	// (the per-template global chunk registry, retired in 4b-final,
	// no longer mediates cross-thread chunk reclaim).
	//
	// Phase 5n: cursor-based DLL walk.  If `s_dll_exhausted` is true,
	// the previous walk reached end without finding space — skip the
	// walk entirely (set false again when a new chunk is added below,
	// or when an owner_release clears it).  Otherwise resume from
	// `s_dll_cursor` (set by the previous successful claim or end-of-
	// walk advance), or s_dll_head on first walk after a reset.
	//
	// Forward sweep covers the full DLL from the cursor on: newer
	// chunks appear later (see the tail-append at the mmap-fresh path
	// below), so iterating from the cursor visits everything added
	// after the cursor — including the just-mmapped chunk if the
	// cursor was reset via mmap-fresh.  Skipping `s_my_chunk` avoids
	// a redundant retry of the just-failed `allocate_pooled` call
	// above.
	if( !s_dll_exhausted) {
		auto *c = s_dll_cursor ? s_dll_cursor : s_dll_head;
		while(c) {
			if(c != s_my_chunk) {
				if(void *p = c->allocate_pooled(SIZE)) {
					s_my_chunk = c;
					s_dll_cursor = c;
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
			}
			c = c->m_dll_next;
		}
		// Walk reached end without finding space — mark exhausted so
		// future allocate_chunk_path calls skip the walk until the
		// DLL is modified (new chunk added or chunk released).
		s_dll_cursor = nullptr;
		s_dll_exhausted = true;
	}
	// All own chunks full — mmap a fresh chunk.  Phase 4b retired the
	// previous pin-CAS scan of the global registry; chunks owned by
	// other (live or exited) threads are no longer reclaimable here.
	// Exited-thread chunks drain naturally via cross-thread frees +
	// `BIT_OWNER_EXITED`; live-thread chunks are private DLL members
	// of their owner.
	//
	// The fresh chunk is appended to this thread's DLL tail and cached
	// in `s_my_chunk`.  Also register the per-template DLL teardown
	// callback with `tls_alloc_thread_exit_cleanup` if not already registered
	// (deduped inside `add`), so thread-exit walks this template's
	// DLL.
	PoolAllocator<ALIGN, DUMMY, DUMMY> *chunk = create_allocator();
	tls_alloc_thread_exit_cleanup.add(
	    &PoolAllocator<ALIGN, FS, DUMMY>::release_dll_chunks_for_thread);
	s_my_chunk = chunk;
	chunk->m_dll_prev = s_dll_tail;
	chunk->m_dll_next = nullptr;
	if(s_dll_tail)
		s_dll_tail->m_dll_next = chunk;
	else
		s_dll_head = chunk;
	s_dll_tail = chunk;
	// Phase 5n: fresh chunk has full capacity — clear the exhaustion
	// flag and point the cursor at it.  Next allocate_chunk_path that
	// finds `s_my_chunk` full will resume the DLL walk from this
	// chunk; in alloc_only workloads where this chunk is the only
	// one with space, the walk does an O(1) skip-self-and-end instead
	// of the O(N) walk-all-prior-full-chunks.
	s_dll_cursor = chunk;
	s_dll_exhausted = false;
	void *p = chunk->allocate_pooled(SIZE);
	// Fresh chunk always has room for the first allocation; an mmap
	// chunk has 16K+ slots even at ALIGN=16.
#ifdef GUARDIAN
	for(unsigned int i = 0; p && i < SIZE / sizeof(uint64_t); ++i) {
		if(static_cast<uint64_t *>(p)[i] != GUARDIAN) {
			fprintf(stderr, "Memory tainted in %p:64\n", &static_cast<uint64_t *>(p)[i]);
		}
	}
#endif
#ifdef FILLING_AFTER_ALLOC
	for(unsigned int i = 0; p && i < SIZE / sizeof(uint64_t); ++i)
		static_cast<uint64_t *>(p)[i] = FILLING_AFTER_ALLOC;
#endif
	return p;
}
// Phase 4b-final: chunk release is a single CAS on the chunk's
// `m_flags_packed` (BIT_RELEASED).  Whoever wins the CAS is the
// exclusive releaser; they then call `delete` + `deallocate_chunk()`.
// The pin field, the bit-0-lock CAS on the per-template chunk
// registry, and the registry itself are all gone — `BIT_RELEASED`
// on the packed word is the single serialisation point across:
//
//   1. `owner_release(palloc)` — Phase 4a chunk-full neighbour
//      release in `allocate_chunk_path`.  No `BIT_OWNER_EXITED`
//      precondition (owner alive, releasing its own empty chunks);
//      gated by `LEAVE_VACANT_CHUNKS_PER_THREAD` floor (DLL traversal
//      to count this thread's chunks) so bursty workloads don't
//      thrash release-then-mmap.
//   2. `cross_release(palloc)` — cross-thread last-slot returner in
//      `batch_return_to_bitmap` when dec-to-zero meets
//      `BIT_OWNER_EXITED == 1`.  No floor: owner is gone, the chunk
//      has no future, release immediately.
//   3. `release_dll_chunks_for_thread()` — `AllocThreadExitCleanup::~dtor`
//      walks THIS thread's DLL: empty chunks claim `BIT_RELEASED`
//      directly, non-empty set `BIT_OWNER_EXITED`.  No floor: thread
//      is exiting, the chunks belong to nobody.
template <unsigned int ALIGN, bool FS, bool DUMMY>
bool
PoolAllocator<ALIGN, FS, DUMMY>::owner_release(PoolAllocator *palloc) {
	// Phase 5j release model:
	//   - Owner thread observes a DLL-neighbour chunk that's empty
	//     (MASK_CNT == 0).
	//   - atomicFetchAnd(~BIT_OWNED) clears the owned bit.
	//   - If `old & ~BIT_OWNED == 0`, owner is the unique releaser
	//     (= the AND brought m_flags_packed to 0 because MASK_CNT was
	//     0 and BIT_OWNED was 1).  Return true → caller deletes +
	//     deallocate_chunks.
	//   - Else MASK_CNT > 0 (in-flight cross-thread free that hadn't
	//     completed when owner observed empty — very rare since
	//     cross-thread dec is atomic and visible) → leave for cross-
	//     thread releaser.  Return false.  BIT_OWNED is now clear so
	//     the cross-thread releaser's subsequent atomicDecAndTest will
	//     bring the word to 0 and identify itself as releaser.
	//
	// Per-thread floor check: count this thread's DLL chunks and skip
	// release below the floor.  Cheap — DLL is single-writer (us) and
	// typically holds 1–3 chunks per template, far below
	// AllocThreadExitCleanup::MAX = 32.  Called from the slow path only
	// (chunk-full trigger), so the O(D) walk is invisible to the hot
	// path.
	int dll_len = 0;
	for(auto *c = s_dll_head; c; c = c->m_dll_next) ++dll_len;
	if(dll_len <= LEAVE_VACANT_CHUNKS_PER_THREAD) return false;

	// Quick pre-check: bail if not empty.  Avoids the atomicFetchAnd
	// (and the BIT_OWNED clear that'd hand release to cross-thread).
	if((palloc->m_flags_packed & MASK_CNT) != 0) return false;

	uint32_t old = atomicFetchAnd(&palloc->m_flags_packed,
	                              static_cast<uint32_t>(~BIT_OWNED));
	uint32_t newv = old & ~BIT_OWNED;
	if(newv != 0) {
		// MASK_CNT > 0 (cross-thread brought a bit back?) — no, MASK_CNT
		// monotone non-increases on non-pinned DLL chunks.  Reaching
		// here means the pre-check raced with our AND completion.  Not
		// the releaser; cross-thread will handle.
		return false;
	}
#ifdef GUARDIAN
	void *ppool = palloc->m_mempool;
	for(unsigned int i = 0; i < palloc->m_chunk_size / sizeof(uint64_t); ++i) {
		if(static_cast<uint64_t *>(ppool)[i] != GUARDIAN) {
			fprintf(stderr, "Memory tainted in %p:64\n",
				&static_cast<uint64_t *>(ppool)[i]);
		}
	}
#endif
	return true;
}

// Phase 5j: cross_release no longer needed as a separate path.
// Cross-thread releaser identification is inlined in
// batch_return_to_bitmap's OnClearFn via atomicDecAndTest — when
// dec brings m_flags_packed to 0 (= BIT_OWNED was clear AND MASK_CNT
// was 1), that thread is uniquely the releaser.  The function is
// kept declared in allocator_prv.h for ABI stability across template
// instantiations but defined as a stub here.
template <unsigned int ALIGN, bool FS, bool DUMMY>
bool
PoolAllocator<ALIGN, FS, DUMMY>::cross_release(PoolAllocator * /*palloc*/) {
	// Legacy entry — not used in Phase 5j+.  See OnClearFn release
	// branch in batch_return_to_bitmap.
	return false;
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PoolAllocator<ALIGN, FS, DUMMY>::release_dll_chunks_for_thread() noexcept {
	// Walk this thread's DLL with cached-next.  For each chunk:
	//   empty (count == 0) → CAS BIT_RELEASED, then delete + deallocate_chunk.
	//   non-empty           → CAS BIT_OWNER_EXITED, then drop reference.
	//
	// Cached-next is essential because once we set BIT_OWNER_EXITED on
	// a non-empty chunk, the cross-thread last-slot-returner can race
	// us, win BIT_RELEASED, and delete the chunk — `c->m_dll_next` is
	// freed memory.  We read `next` BEFORE the OWNER_EXITED CAS.
	//
	// `s_dll_head` / `s_dll_tail` / `s_my_chunk` are wiped FIRST so any
	// later TLS dtor that allocates (via `cold_first_access` →
	// `is_allocator_thread_active() == false` → libsystem fallback)
	// cannot route through a chunk we already released.
	// Phase 5j: single atomicFetchAnd per DLL chunk.  Clears BIT_OWNED;
	// if the resulting value is 0 (MASK_CNT was 0 → chunk was empty),
	// owner is the unique releaser.  Otherwise the chunk has live
	// slots — cross-thread free will identify itself as releaser via
	// atomicDecAndTest when it brings MASK_CNT to 0.
	//
	// Race with concurrent cross-thread free:
	//   Cross-thread dec from (BIT_OWNED=1, MASK_CNT=1) → (1, 0):
	//     atomicDecAndTest returns false (newv != 0 because BIT_OWNED).
	//     Owner's subsequent AND brings to 0 → owner releases.
	//   Owner's AND from (1, 1) → (0, 1):
	//     newv = 1 ≠ 0; owner not releaser.  Cross-thread dec from
	//     (0, 1) → 0; cross-thread releases.
	// Exactly one releaser in each interleaving.
	auto *c = s_dll_head;
	s_dll_head = nullptr;
	s_dll_tail = nullptr;
	s_my_chunk = nullptr;
	// Phase 5n: thread exit → cursor and exhausted flag both moot.
	s_dll_cursor = nullptr;
	s_dll_exhausted = false;
	while(c) {
		auto *next = c->m_dll_next;
		c->m_dll_prev = nullptr;
		c->m_dll_next = nullptr;
		uint32_t old = atomicFetchAnd(&c->m_flags_packed,
		                              static_cast<uint32_t>(~BIT_OWNED));
		uint32_t newv = old & ~BIT_OWNED;
		if(newv == 0) {
			char *cbase = c->m_mempool - ALLOC_CHUNK_HEADER;
			size_t csz = c->m_chunk_size;
			delete c;
			PoolAllocatorBase::deallocate_chunk(cbase, csz);
		}
		// else: chunk still has live slots (MASK_CNT > 0).
		// Cross-thread releaser will pick it up via atomicDecAndTest.
		c = next;
	}
}
inline void
PoolAllocatorBase::deallocate_chunk(char *chunk_base, size_t chunk_size) {
	// Phase 5l release sequence (multi-unit aware):
	//   1. Clear `ready` bit on the BASE unit (acquire).  Readers
	//      acquire-observing claim=1, ready=0 know the chunk_header is
	//      stale; combined with the chunk_header palloc=0 write below
	//      this provides the standard "released chunk" indicator.
	//   2. Clear chunk_header palloc / size_info.  Any in-flight
	//      lookup_chunk gets palloc==0 → treats as foreign.
	//   3. madvise(MADV_FREE on macOS, MADV_DONTNEED on Linux).  Reclaims
	//      the chunk's physical pages but leaves VA RW; subsequent
	//      allocate_chunk reclaim of these units gets zero-fault'd pages
	//      on first write.  No mprotect / PROT_NONE — we trade SIGBUS
	//      safety for fewer syscalls + cleaner re-allocate semantics.
	//   4. Clear back_offset entries for ALL units of this chunk.
	//   5. Clear the claim bits of ALL units (release).  Reclaimable.
	//
	// chunk_size determines the unit count (CHUNK_UNITS = chunk_size /
	// ALLOC_MIN_CHUNK_SIZE).  Region size is uniform 32 MiB.
	unsigned int chunk_units =
	    static_cast<unsigned int>(chunk_size >> ALLOC_MIN_CHUNK_SHIFT);
	for(int region = 0; region < ALLOC_MAX_MMAP_ENTRIES; ++region) {
		char *mp = s_mmapped_spaces[region];
		if(region > 0 && !mp) break;
		if(mp) {
			ptrdiff_t pdiff = chunk_base - mp;
			if(pdiff >= 0
			   && pdiff < (ptrdiff_t)ALLOC_MIN_MMAP_SIZE) {
				unsigned int base_unit_idx =
				    static_cast<unsigned int>((size_t)pdiff >> ALLOC_MIN_CHUNK_SHIFT);
				int word = base_unit_idx / CHUNKS_PER_BITMAP_WORD;
				int base_in_word = base_unit_idx % CHUNKS_PER_BITMAP_WORD;
				int base_bit = base_in_word * 2;
				std::atomic<BitmapWord> *bm =
				    &s_claim_bitmap[region * BITMAP_WORDS_PER_REGION + word];
				// Step 1: clear ready on base unit.
				bm->fetch_and(~(BitmapWord(1) << (base_bit + 1)),
				              std::memory_order_acquire);
				// Step 2: clear chunk_header.
				*reinterpret_cast<std::uint64_t *>(
				    chunk_base + ALLOC_CHUNK_HEADER_SIZE_INFO_OFFSET) = 0;
				*reinterpret_cast<PoolAllocatorBase **>(
				    chunk_base + ALLOC_CHUNK_HEADER_PALLOC_OFFSET) = nullptr;
				// Step 3: madvise reclaims physical pages.
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
				(void)chunk_size;
#elif defined(__APPLE__)
				// macOS: MADV_FREE — kernel zeros pages lazily on
				// next access (or sooner under memory pressure).
				// Apple libc's implementation is cheap (per-page
				// flag flip, no LRU-list manipulation) and reuse-
				// fast: subsequent writes to the same VA hit the
				// preserved pages without re-faulting.
				madvise(chunk_base, chunk_size, MADV_FREE);
#else
				// Linux / others: MADV_DONTNEED — eager reclaim.
				//
				// Phase 5n tried MADV_FREE on Linux for "lazy reclaim,
				// fast reuse" but it REGRESSED catastrophically on
				// reuse-heavy workloads (bucket34_repro 33.5 →
				// 0.26 M/s, fifo:1024 3072B 3.3 → 1.6 M/s,
				// alloc_stress c=32 x=50 % RSS 9 MiB → 698 MiB).
				// Root cause is kernel-specific: Linux MADV_FREE
				// adds pages to an LRU lazy-discard list (multi-
				// thread lock contention) and reusing the VA via a
				// fresh write triggers a minor page-fault to clear
				// the discard flag — net cost EXCEEDS the
				// MADV_DONTNEED + zero-fault round-trip for our
				// alloc/free cadence, while RSS bloats because
				// reclaim is delayed until memory pressure.  macOS
				// MADV_FREE does not have this problem because
				// Apple's implementation is structured differently.
				madvise(chunk_base, chunk_size, MADV_DONTNEED);
#endif
				// Step 4: clear back_offset entries.
				size_t bo_base = (size_t)region * NUM_ALLOCATORS_IN_SPACE
				                 + (size_t)base_unit_idx;
				for(unsigned u = 0; u < chunk_units; ++u)
					s_back_offset[bo_base + u] = 0;
				// Step 5: clear claim bits for all units (release).
				BitmapWord claim_mask = 0;
				for(unsigned u = 0; u < chunk_units; ++u)
					claim_mask |= BitmapWord(1) << (base_bit + 2u * u);
				bm->fetch_and(~claim_mask, std::memory_order_release);
				// Step 6 (Phase 5m): set the region's has-free bit.
				// fetch_or is idempotent; if a concurrent allocate_chunk
				// just cleared it after finding all words full, our set
				// re-publishes that this region now has free space.
				s_region_has_free[region / 64].fetch_or(
				    uint64_t(1) << (region % 64),
				    std::memory_order_relaxed);
				return;
			}
		}
	}
}

// Diagnostic probe — sum popcount across every region's claim bitmap
// to get the count of currently-live chunks.  Diagnostic only; relaxed
// loads across a possibly-concurrent claim/release race.  Used by tests
// to verify chunk release paths fire (a chunk leak would show as
// monotonic growth across repeated alloc/free cycles).
int
PoolAllocatorBase::count_live_chunks() noexcept {
	// Phase 5h: 2-bit encoding — count CLAIM bits only (bit 2N for
	// chunk N).  READY bits (bit 2N+1) are auxiliary, counted
	// together with their claim bit would double-count.
	int n = 0;
	for(int i = 0; i < ALLOC_MAX_MMAP_ENTRIES * BITMAP_WORDS_PER_REGION; ++i) {
		BitmapWord v = s_claim_bitmap[i].load(std::memory_order_relaxed);
		n += int(count_bits(v & CLAIM_BITS_MASK));
	}
	return n;
}

// Address → chunk lookup.  Walks `s_mmapped_spaces[]` (uniform 32 MiB
// regions since Phase 5l) to find which region contains `p`, then uses
// `s_back_offset[]` to resolve the chunk's base unit in O(1) — one
// byte-table read and a subtract.  Reads `palloc` from the chunk
// header at the base unit's offset 8.
inline PoolAllocatorBase *
PoolAllocatorBase::lookup_chunk(void *p) noexcept {
	for(int ccnt = 0; ccnt < ALLOC_MAX_MMAP_ENTRIES; ++ccnt) {
		char *mp = s_mmapped_spaces[ccnt];
		if(ccnt > 0 && !mp) break;
		if(mp) {
			ptrdiff_t pdiff = static_cast<char *>(p) - mp;
			if(pdiff >= 0
			   && pdiff < (ptrdiff_t)ALLOC_MIN_MMAP_SIZE) {
				unsigned int unit_idx = static_cast<unsigned int>(
				    (size_t)pdiff >> ALLOC_MIN_CHUNK_SHIFT);
				unsigned int base_off =
				    s_back_offset[ccnt * NUM_ALLOCATORS_IN_SPACE + unit_idx];
				unsigned int base_idx = unit_idx - base_off;
				char *chunk_base =
				    mp + (size_t)base_idx * (size_t)ALLOC_MIN_CHUNK_SIZE;
				PoolAllocatorBase *palloc =
				    *reinterpret_cast<PoolAllocatorBase **>(
				        chunk_base + ALLOC_CHUNK_HEADER_PALLOC_OFFSET);
				if((uintptr_t)palloc <= (uintptr_t)1u) return nullptr;
				return palloc;
			}
		}
	}
	return nullptr;
}

inline bool
PoolAllocatorBase::deallocate(void *p) {
	// `delete nullptr` is well-defined as a no-op per [expr.delete]/2.
	// Treat null as "not our pointer" so the caller (operator delete or
	// kame_free) falls through to its libsystem-free path, which itself
	// short-circuits on null.
	if( !p) return false;
	// Phase 5l: uniform 32 MiB regions, runtime walk.  Each region is
	// either fully claimed (mp != null) or empty (mp == null).  Empty
	// regions cluster at the tail of `s_mmapped_spaces[]` (claim grows
	// monotonically from index 0), so the loop short-circuits on the
	// first null past index 0.
	for(int ccnt = 0; ccnt < ALLOC_MAX_MMAP_ENTRIES; ++ccnt) {
		char *mp = s_mmapped_spaces[ccnt];
		if(ccnt > 0 && !mp) return false;
		if( !mp) continue;
		ptrdiff_t pdiff = static_cast<char *>(p) - mp;
		if(pdiff < 0 || pdiff >= (ptrdiff_t)ALLOC_MIN_MMAP_SIZE)
			continue;
		// Phase 5l: chunk_base via `s_back_offset[]`.  unit_idx is the
		// 256 KiB unit within this region; back_offset gives the
		// distance back to the chunk's base unit (0 for single-unit
		// chunks, 1..3 for continuation units of multi-unit chunks).
		// O(1) — one byte-table load and a subtract.  Synchronised by
		// `allocate_chunk`'s writeBarrier before the claim CAS, so any
		// reader observing a non-released chunk sees the current
		// claimer's back_offset value.
		unsigned int unit_idx =
		    static_cast<unsigned int>((size_t)pdiff >> ALLOC_MIN_CHUNK_SHIFT);
		unsigned int base_off =
		    s_back_offset[ccnt * NUM_ALLOCATORS_IN_SPACE + unit_idx];
		unsigned int base_idx = unit_idx - base_off;
		char *chunk_base = mp + (size_t)base_idx * (size_t)ALLOC_MIN_CHUNK_SIZE;
		PoolAllocatorBase *palloc =
		    *reinterpret_cast<PoolAllocatorBase **>(
		        chunk_base + ALLOC_CHUNK_HEADER_PALLOC_OFFSET);
		// Defensive: chunk header may hold nullptr (chunk released) or
		// 1u (in-creation sentinel — unlikely to be observed past the
		// allocate_chunk writeBarrier+s_chunks store).  Treat both as
		// "not our pointer" so the caller falls through to std::free
		// — same semantics as the previous s_chunks-based check,
		// retained for libsystem-malloc pointers that happen to land
		// inside our PROT_NONE mmap range (macOS Apple Silicon
		// ICU/Foundation early startup).
		if((uintptr_t)palloc <= (uintptr_t)1u)
			return false;
		// Phase 5e — Unified inline dispatch.  Read the chunk-header
		// SIZE info word at [0..7]:
		//   FS=true  : low 32 b = ALIGN (= slot size, non-zero)
		//   FS=false : 0 — read per-slot {bucket, SIZE} header at p - 8
		// Then derive `bucket` and check ownership via the parallel TLS
		// `g_thread_chunks[bucket]`.  When this thread is the owner
		// (cache-hot, common case) we push to the per-thread
		// `g_thread_slots[bucket].freelist_head` inline — no indirect
		// call, no per-template virtual dispatch.
		//
		// Cross-thread / non-owner / post-teardown (g_thread_chunks
		// cleared) all fall through to the existing `DeallocateFn` at
		// chunk-header offset ALLOC_CHUNK_HEADER_FN_OFFSET (= 16),
		// preserving the adaptive holding (FS=true ALIGN ≤ 48) and
		// chunk-release plumbing already in the per-template
		// `deallocate_pooled`.
		std::uint64_t hdr_size_info = *reinterpret_cast<std::uint64_t *>(
		    chunk_base + ALLOC_CHUNK_HEADER_SIZE_INFO_OFFSET);
		// FS-discriminator is the LOW 32 bits of hdr_size_info (the
		// high 32 b always carries ALIGN for BOTH FS=true and FS=false
		// per chunk_header_size_info()).  A bare `(hdr_size_info != 0)`
		// test mis-routes FS=false through the FS=true arm (low 32 b
		// == 0 but high 32 b != 0).  Test the low 32 b explicitly.
		std::uint32_t hdr_fs_size =
		    static_cast<std::uint32_t>(hdr_size_info);
		unsigned bucket;
		if(hdr_fs_size != 0) {
			// FS=true: ALIGN in low 32 b; bucket_for_size constexpr-folds
			// to (ALIGN+15)>>4 for ALIGN ≤ 256.
			bucket = bucket_for_size(
			    static_cast<std::size_t>(hdr_fs_size));
		} else {
			// FS=false: per-slot {uint32_t bucket, uint32_t SIZE} at p - 8.
			std::uint64_t hdr = *reinterpret_cast<std::uint64_t *>(
			    static_cast<char *>(p) - 8);
			bucket = static_cast<unsigned>(hdr);
			// Defensive: out-of-range bucket falls through to vtable
			// (the slow path will further validate via owner check or
			// cross-batch dispatch).  This protects against a stray
			// libsystem-malloc'd pointer that happens to land inside our
			// PROT_NONE mmap range AND has a non-zero hdr_size_info in
			// the chunk header pad.
			if(bucket == 0 || bucket >= (unsigned)ALLOC_NUM_BUCKETS)
				goto vtable_dispatch;
		}
		{
			PoolAllocatorBase **chunks_base = kame_chunks_base();
			if(__builtin_expect(chunks_base[bucket] == palloc, 1)) {
				// Owner-side inline freelist push.  No atomic, no
				// indirect call, no template dispatch.  The slot stays
				// "allocated" in the chunk's m_flags bitmap until the
				// owner thread's next allocate_pooled cycle or thread-
				// exit drain returns it via batch_return_to_bitmap.
				kame_slots_base()[bucket].push(p);
				return true;
			}
		}
	vtable_dispatch:
		// Cold path: cross-thread / non-pinned / post-teardown.  Falls
		// back to the per-template DeallocateFn at chunk_base + 16,
		// which preserves the existing adaptive-holding / cross-batch /
		// chunk-release logic in `deallocate_pooled`.
		{
		DeallocateFn fn = *reinterpret_cast<DeallocateFn *>(
		    chunk_base + ALLOC_CHUNK_HEADER_FN_OFFSET);
#ifdef KAME_DEBUG_CHUNK_HEADER
		// Diagnostic check for chunk_header corruption: verify fn
		// doesn't point into the chunk's slot region.  In a healthy
		// build fn is a code address far from chunk_base.  Enable with
		// `-DKAME_DEBUG_CHUNK_HEADER` in the kamepoolalloc build flags.
		{
			uintptr_t fn_addr = (uintptr_t)fn;
			uintptr_t cb      = (uintptr_t)chunk_base;
			// Phase 5l: chunk_size is now per-template; read from palloc's
			// runtime field.
			uintptr_t cb_end  = cb + palloc->chunk_size();
			if(fn_addr >= cb && fn_addr < cb_end) {
				fprintf(stderr,
				    "[allocator] CORRUPTION: chunk_base=%p csz=0x%llx "
				    "(CCNT=%d) palloc=%p fn=%p slot=%p\n"
				    "  fn falls inside slot region (offset 0x%llx).\n"
				    "  Header dump (chunk_base + 0..63):\n",
				    chunk_base, (unsigned long long)palloc->chunk_size(), CCNT,
				    palloc, (void *)fn_addr, p,
				    (unsigned long long)(fn_addr - cb));
				for(int i = 0; i < 64; i += 8) {
					fprintf(stderr, "    +%02d: %016llx\n", i,
					    (unsigned long long)*(uint64_t *)(chunk_base + i));
				}
				std::abort();
			}
		}
#endif
		{
			// Capture chunk_size BEFORE fn() in case fn signals release
			// (would `delete palloc` and invalidate palloc->chunk_size()).
			size_t csz = palloc->chunk_size();
			if(fn(palloc, static_cast<char *>(p))) {
				// Current `deallocate_pooled` always returns false (the
				// chunk-release-on-empty path is taken inside
				// `batch_return_to_bitmap` via the `i_am_releaser`
				// branch).  Kept as a defensive shim in case a future
				// trampoline opts to release at this site.
				deallocate_chunk(chunk_base, csz);
			}
		}
		return true;
		}
	}
	return false;
}

// `size_of` — read-only sibling of `deallocate`.  Walks the same
// per-region back_offset table and dereferences the chunk header.
// Dispatches `SizeOfFn` at offset `ALLOC_CHUNK_HEADER_SIZEOF_FN_OFFSET`
// (= 24) and returns the slot size in bytes.  Used by `kame_realloc`
// to size copies.  Returns 0 for any pointer not inside our pool
// (libsystem-malloc'd, null, or chunk released).
inline std::size_t
PoolAllocatorBase::size_of(void *p) {
	if( !p) return 0;
	for(int ccnt = 0; ccnt < ALLOC_MAX_MMAP_ENTRIES; ++ccnt) {
		char *mp = s_mmapped_spaces[ccnt];
		if(ccnt > 0 && !mp) return 0;
		if( !mp) continue;
		ptrdiff_t pdiff = static_cast<char *>(p) - mp;
		if(pdiff < 0 || pdiff >= (ptrdiff_t)ALLOC_MIN_MMAP_SIZE)
			continue;
		unsigned int unit_idx =
		    static_cast<unsigned int>((size_t)pdiff >> ALLOC_MIN_CHUNK_SHIFT);
		unsigned int base_off =
		    s_back_offset[ccnt * NUM_ALLOCATORS_IN_SPACE + unit_idx];
		unsigned int base_idx = unit_idx - base_off;
		char *chunk_base = mp + (size_t)base_idx * (size_t)ALLOC_MIN_CHUNK_SIZE;
		PoolAllocatorBase *palloc =
		    *reinterpret_cast<PoolAllocatorBase **>(
		        chunk_base + ALLOC_CHUNK_HEADER_PALLOC_OFFSET);
		if((uintptr_t)palloc <= (uintptr_t)1u)
			return 0;
		SizeOfFn fn = *reinterpret_cast<SizeOfFn *>(
		    chunk_base + ALLOC_CHUNK_HEADER_SIZEOF_FN_OFFSET);
		return fn(palloc, static_cast<char *>(p));
	}
	return 0;
}
void* allocate_large_size_or_malloc(size_t size) throw() {
	// Phase 5d-4: bucket dispatch covers sizes 1..ALLOC_MAX_BUCKETED_SIZE
	// (= 16376).  Anything bigger lands here and goes straight to
	// libsystem.  The legacy ALLOCATE_9_16X(4, size) … (64, size) chain
	// — which broke the size range into power-of-2 sub-tiers each
	// dispatched through a PoolAllocator template instantiation — is no
	// longer needed; the per-tier bucket path is more granular AND
	// avoids the explosion of PoolAllocator template instantiations
	// (`<256, false>`, `<512, false>`, …, `<16384, false>` etc.) those
	// macros implied.
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
// Phase 5q: extend the FS=true 16-step ladder to buckets 17..23
// (sizes 272..368).  Each is FS=true with ALIGN = SIZE → zero
// internal frag (one slot per ALIGN-byte region).  Closes the
// 257..368 gap that Phase 5p's bucket 17 (slot 384) absorbed at
// up to 32 % frag for the small end.
KAME_DECL_BUCKET(17, ALLOC_SIZE17,                 true,  ALLOC_SIZE17);  // 272
KAME_DECL_BUCKET(18, ALLOC_SIZE18,                 true,  ALLOC_SIZE18);  // 288
KAME_DECL_BUCKET(19, ALLOC_SIZE19,                 true,  ALLOC_SIZE19);  // 304
KAME_DECL_BUCKET(20, ALLOC_SIZE20,                 true,  ALLOC_SIZE20);  // 320
KAME_DECL_BUCKET(21, ALLOC_SIZE21,                 true,  ALLOC_SIZE21);  // 336
KAME_DECL_BUCKET(22, ALLOC_SIZE22,                 true,  ALLOC_SIZE22);  // 352
KAME_DECL_BUCKET(23, ALLOC_SIZE23,                 true,  ALLOC_SIZE23);  // 368

// Buckets 24..47 (Phase 5p N+1 shift, indices +7 from Phase 5p): 4-way
// exponential FS=false ladder.  3 ALIGN stages × 8 (octave/sub) =
// 24 buckets.  The "borrow" header is the universal 8 B at p_user - 8
// (Phase 5d-1).  Bucket `SIZE` is the MAX user_size the bucket serves
// (= slot total - 8); slow_allocate / bucket_first_access pass SIZE
// to allocate_pooled, which computes N = ceil((SIZE+8)/ALIGN).
// Slot total uniformly = (N+1)*ALIGN per Phase 5p.

// Stage 1 — ALIGN=64.  Slot totals 384..1088 (= 6..17 × 64).
KAME_DECL_BUCKET(24,  64u, false,   376u);  // octave 8 sub 1 +ALIGN, N=6,  slot= 384
KAME_DECL_BUCKET(25,  64u, false,   440u);  // octave 8 sub 2 +ALIGN, N=7,  slot= 448
KAME_DECL_BUCKET(26,  64u, false,   504u);  // octave 8 sub 3 +ALIGN, N=8,  slot= 512
KAME_DECL_BUCKET(27,  64u, false,   568u);  // octave 9 sub 0 +ALIGN, N=9,  slot= 576
KAME_DECL_BUCKET(28,  64u, false,   696u);  // octave 9 sub 1 +ALIGN, N=11, slot= 704
KAME_DECL_BUCKET(29,  64u, false,   824u);  // octave 9 sub 2 +ALIGN, N=13, slot= 832
KAME_DECL_BUCKET(30,  64u, false,   952u);  // octave 9 sub 3 +ALIGN, N=15, slot= 960
KAME_DECL_BUCKET(31,  64u, false,  1080u);  // octave 10 sub 0 +ALIGN, N=17, slot=1088

// Stage 2 — ALIGN=256.  Slot totals 1536..4352 (= 6..17 × 256).
KAME_DECL_BUCKET(32, 256u, false,  1528u);  // octave 10 sub 1 +ALIGN, N=6,  slot= 1536
KAME_DECL_BUCKET(33, 256u, false,  1784u);  // octave 10 sub 2 +ALIGN, N=7,  slot= 1792
KAME_DECL_BUCKET(34, 256u, false,  2040u);  // octave 10 sub 3 +ALIGN, N=8,  slot= 2048
KAME_DECL_BUCKET(35, 256u, false,  2296u);  // octave 11 sub 0 +ALIGN, N=9,  slot= 2304
KAME_DECL_BUCKET(36, 256u, false,  2808u);  // octave 11 sub 1 +ALIGN, N=11, slot= 2816
KAME_DECL_BUCKET(37, 256u, false,  3320u);  // octave 11 sub 2 +ALIGN, N=13, slot= 3328
KAME_DECL_BUCKET(38, 256u, false,  3832u);  // octave 11 sub 3 +ALIGN, N=15, slot= 3840
KAME_DECL_BUCKET(39, 256u, false,  4344u);  // octave 12 sub 0 +ALIGN, N=17, slot= 4352

// Stage 3 — ALIGN=1024.  Slot totals 6144..17408 (= 6..17 × 1024).
KAME_DECL_BUCKET(40, 1024u, false,  6136u);  // octave 12 sub 1 +ALIGN, N=6,  slot= 6144
KAME_DECL_BUCKET(41, 1024u, false,  7160u);  // octave 12 sub 2 +ALIGN, N=7,  slot= 7168
KAME_DECL_BUCKET(42, 1024u, false,  8184u);  // octave 12 sub 3 +ALIGN, N=8,  slot= 8192
KAME_DECL_BUCKET(43, 1024u, false,  9208u);  // octave 13 sub 0 +ALIGN, N=9,  slot= 9216
KAME_DECL_BUCKET(44, 1024u, false, 11256u);  // octave 13 sub 1 +ALIGN, N=11, slot=11264
KAME_DECL_BUCKET(45, 1024u, false, 13304u);  // octave 13 sub 2 +ALIGN, N=13, slot=13312
KAME_DECL_BUCKET(46, 1024u, false, 15352u);  // octave 13 sub 3 +ALIGN, N=15, slot=15360
KAME_DECL_BUCKET(47, 1024u, false, 17400u);  // octave 14 sub 0 +ALIGN, N=17, slot=17408
#undef KAME_DECL_BUCKET

//! First-access trampoline for bucket B.  Invoked from the
//! `cold_first_access` switch when `g_thread_chunks[B] == nullptr`.
//! Claims a chunk via the existing `allocate<>()` slow path (which
//! registers AllocThreadExitCleanup) and records the chunk into
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
//      std::malloc(size).  Set by AllocThreadExitCleanup::~dtor on thread
//      exit; later TLS destructors that still allocate land here.
//   3. First access: switch on bucket to invoke the per-bucket
//      `bucket_first_access<B>`, which calls
//      `PA::allocate<BT::SIZE>()` with SIZE compile-time-const,
//      registers AllocThreadExitCleanup, and populates
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
        case 25:          return bucket_first_access<25>(size);
        case 26:          return bucket_first_access<26>(size);
        case 27:          return bucket_first_access<27>(size);
        case 28:          return bucket_first_access<28>(size);
        case 29:          return bucket_first_access<29>(size);
        case 30:          return bucket_first_access<30>(size);
        case 31:          return bucket_first_access<31>(size);
        case 32:          return bucket_first_access<32>(size);
        case 33:          return bucket_first_access<33>(size);
        case 34:          return bucket_first_access<34>(size);
        case 35:          return bucket_first_access<35>(size);
        case 36:          return bucket_first_access<36>(size);
        case 37:          return bucket_first_access<37>(size);
        case 38:          return bucket_first_access<38>(size);
        case 39:          return bucket_first_access<39>(size);
        case 40:          return bucket_first_access<40>(size);
        case 41:          return bucket_first_access<41>(size);
        case 42:          return bucket_first_access<42>(size);
        case 43:          return bucket_first_access<43>(size);
        case 44:          return bucket_first_access<44>(size);
        case 45:          return bucket_first_access<45>(size);
        case 46:          return bucket_first_access<46>(size);
        case 47:          return bucket_first_access<47>(size);
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

// Forward non-pool pointers to the *actual* libsystem free, bypassing
// the `free` symbol — which our `__DATA,__interpose` redirects back to
// `kame_free`, causing infinite recursion.
//
// Why `dlsym(RTLD_NEXT, "free")` does NOT work: dyld applies interposing
// at *bind time*, so `dlsym` returns the bound (interposed) symbol
// address — i.e. `&kame_free` itself.  Verified empirically: on first
// dealloc, dlsym hands back our own replacement and `orig(p)` recurses
// straight into `kame_free` → infinite loop during libdispatch_init's
// `NXCreateHashTable` → free, observed via lldb backtrace.
//
// macOS fix: use the zone API directly.  `malloc_zone_from_ptr(p)`
// returns the zone that owns `p` (or NULL if `p` was not allocated by
// libsystem_malloc); `malloc_zone_free(zone, p)` calls the zone's free
// vtable entry, skipping the top-level `free()` symbol entirely.  This
// mirrors what libsystem_malloc's own `free()` implementation does
// (`free(p) = malloc_zone_free(malloc_zone_from_ptr(p), p)`), so
// behaviour is identical from libsystem's perspective.
//
// Linux/Windows: `dlsym(RTLD_NEXT, "free")` works (no `__interpose`
// section equivalent — our strong-symbol `free` shadows but doesn't
// retarget all images), but for symmetry and to avoid pulling in
// `<dlfcn.h>` we use `__libc_free` directly on glibc — it's a stable
// public ABI symbol exposed for malloc-replacement libraries.
//
// IMPORTANT: do NOT call `std::free()` / `::free()` here — those names
// resolve via the dylib's own export table, which the strong-symbol
// `free` shim defined below shadows.  Calling `free()` from this dylib
// recurses straight back into `kame_free`, producing an infinite tight
// loop (under -O3 + noinline the inner call is tail-jumped, so no stack
// overflow ever fires — the process just hangs).
__attribute__((noinline))
static void libsystem_free_for_pool(void *p);

inline void deallocate_pooled_or_free(void* p) throw() {
	// `PoolAllocatorBase::deallocate(p)` is safe to call pre-
	// `activateAllocator()`.  `deallocate_<0, ALLOC_MIN_CHUNK_SIZE>`
	// loads `s_mmapped_spaces[0]` which is zero-initialised (so
	// `nullptr` pre-activation); the subsequent
	// `(pdiff >= 0 && pdiff < CHUNK_SIZE * NUM_ALLOCATORS_IN_SPACE)`
	// range check trivially fails for any real pointer against a
	// `nullptr` base, the recursion through higher levels likewise
	// fails, and the call returns `false`.  We then drop to
	// `libsystem_free_for_pool(p)` — same outcome as an explicit
	// `!g_sys_image_loaded` early-out, which previously guarded this
	// path but was redundant.
	if(PoolAllocatorBase::deallocate(p))
		return;
	libsystem_free_for_pool(p);
}

#if defined(__linux__) && defined(__GLIBC__)
// glibc internal entry point — same address as libc's `free` but with
// a name our strong-symbol `free` shim does not shadow.  Declared with
// the same signature as `free` so the call is ABI-compatible.
extern "C" void __libc_free(void *) noexcept;
#endif

static void libsystem_free_for_pool(void *p) {
#if defined(__APPLE__)
	// Zone-API direct dispatch — skips the interposed `free` symbol.
	// `malloc_zone_from_ptr` may return NULL for pointers libsystem
	// doesn't recognise (e.g. mmap'd memory we hand back via munmap
	// elsewhere); in that case there's nothing to free at this layer.
	if(malloc_zone_t *zone = malloc_zone_from_ptr(p))
		malloc_zone_free(zone, p);
#elif defined(__linux__) && defined(__GLIBC__)
	// Bypass our strong-symbol `free` shim — `__libc_free` is the real
	// libc free under a name we don't override.  Without this the
	// `std::free` / `::free` lookup re-binds to `kame_free` and we
	// recurse forever (the call ends up tail-jumping under -O3).
	__libc_free(p);
#else
	// Other platforms (musl, Windows): fall back to `::free`.  On musl
	// the strong-symbol shadowing rule is the same as glibc, so this
	// will recurse if KAME's pool is active — Linux non-glibc builds
	// must add their own bypass before this branch is reachable.
	std::free(p);
#endif
}

// Pool-aware `free()` replacement.  Any call site that resolves
// `free` — whether from KAME code, libc++, libsystem (during thread
// teardown), or a libcxx-thread-exit destruction inlined under LTO
// — first checks if `p` belongs to our pool and routes through
// `PoolAllocatorBase::deallocate` if so; otherwise hands off to the
// real libsystem `free` via `libsystem_free_for_pool`.
//
// Motivation: under aggressive LTO + clang on macOS, some
// thread-local destruction paths (notably `_pthread_tsd_cleanup`'s
// per-key destructor invocations) end up calling `free(p)` directly
// on memory we originally returned via `::operator new` (overridden
// to come from our pool).  The resulting
// `___BUG_IN_CLIENT_OF_LIBMALLOC_POINTER_BEING_FREED_WAS_NOT_ALLOCATED`
// abort fires at thread exit.
//
// === macOS: DYLD_INTERPOSE from this dylib ===
//
// We rely on dyld processing the `__DATA,__interpose` section of
// this dylib at load time, redirecting every `free` import — across
// all dylibs (libc++, libsystem_pthread, etc.) — to `kame_free`.
// This is the same mechanism mimalloc uses.  Note dyld *only*
// processes interpose sections from `MH_DYLIB` images, not the main
// executable — which is why `kamepoolalloc` is factored out as a
// shared library.
//
// === Linux / Windows: strong-symbol `free` ===
//
// On non-Darwin we expose `kame_free` as the strong symbol `free`
// from this dylib.  Anything that links against us and resolves
// `free` via our dylib gets the pool-aware version.  Calls from
// other shared libraries that bound to libc's `free` at their own
// link time are not intercepted — equivalent to LD_PRELOAD-style
// interposing, which requires runtime cooperation.
//
// Inverse direction (libsystem/libc-malloc'd pointer reaching this
// override): `PoolAllocatorBase::deallocate(p)` returns `false` for
// any pointer outside our mmap regions and we fall through to
// libsystem free.  Safe.
__attribute__((noinline))
static void kame_free(void *p) {
	// `PoolAllocatorBase::deallocate(p)` is itself pre-activation-safe:
	// it early-returns false on null `p`, and the CCNT=0 lookup against
	// `s_mmapped_spaces[0] == nullptr` (zero-initialised pre-pool-use)
	// trivially fails its range check.  No outer `g_sys_image_loaded`
	// guard needed — the natural state of `s_mmapped_spaces[]` covers
	// the same fast-out.
	if(PoolAllocatorBase::deallocate(p))
		return;
	libsystem_free_for_pool(p);
}

#if defined(__APPLE__)
extern "C" void free(void *);  // libsystem prototype, for address-of

namespace {
struct kame_interpose_entry {
	const void *replacement;
	const void *replacee;
};
__attribute__((used))
kame_interpose_entry kame_interposers[]
    __attribute__((section("__DATA,__interpose"))) = {
        { reinterpret_cast<const void *>(&kame_free),
          reinterpret_cast<const void *>(&free) },
};
} // namespace
#else
// Non-Darwin (Linux / Windows): emit `free` as a strong symbol so
// our dylib's own consumers resolve to the pool-aware version.
extern "C" __attribute__((noinline)) void free(void *p) {
	kame_free(p);
}
#endif

// === calloc / realloc ============================================
//
// Pool-aware companions to `free`.  Same interpose / strong-symbol
// strategy as `free` above:
//   - macOS: `__DATA,__interpose` table extended so dyld rewrites
//     every `calloc` / `realloc` import across all dylibs to ours.
//   - Linux glibc: emit strong-symbol `calloc` / `realloc`; the
//     internal "forward to libc" path uses `__libc_calloc` /
//     `__libc_realloc` to bypass our own shadowing (same trick as
//     `__libc_free` for `free`).
//
// === Why interpose these too ===
//
// `realloc(p, n)` is the dangerous case.  If `p` came from our pool
// (via `::operator new` → `new_redirected`) and libsystem `realloc`
// gets the call, libsystem rejects the pointer with
// `pointer being realloc'd was not allocated`.  Symmetric to the
// `_pthread_tsd_cleanup → free` LTO crash we fixed for `free()`.
//
// `calloc` is safer: most consumers feed its result through to
// `free`, which is already interposed.  But intercepting calloc lets
// us serve `n * size ≤ ALLOC_MAX_BUCKETED_SIZE` allocations from the
// pool too — a 4-5 % chunk of the `alloc_stress` micro-bench
// distribution; on calloc-heavy workloads it can matter more.

#if defined(__linux__) && defined(__GLIBC__)
// glibc internal entries — same addresses as libc's `calloc` / `realloc`
// but under names our strong-symbol shims do not shadow.
extern "C" void *__libc_calloc(size_t, size_t) noexcept;
extern "C" void *__libc_realloc(void *, size_t) noexcept;
#endif

__attribute__((noinline))
static void *libsystem_realloc_for_pool(void *p, std::size_t n) {
#if defined(__APPLE__)
	// Zone-API direct dispatch — skips the interposed `realloc` symbol
	// for the same reason `libsystem_free_for_pool` uses
	// `malloc_zone_free`: `dlsym(RTLD_NEXT, "realloc")` would return
	// our own replacement under interposing.  `malloc_zone_from_ptr`
	// may return NULL for pointers libsystem doesn't recognise — in
	// that case we fall back to the default zone's `realloc` to honour
	// the "p == NULL  ⇒ malloc(n)" contract for calls that race a
	// chunk-release.
	malloc_zone_t *zone = p ? malloc_zone_from_ptr(p) : nullptr;
	if( !zone) zone = malloc_default_zone();
	return malloc_zone_realloc(zone, p, n);
#elif defined(__linux__) && defined(__GLIBC__)
	return __libc_realloc(p, n);
#else
	return std::realloc(p, n);
#endif
}

__attribute__((noinline))
static void *libsystem_calloc_for_pool(std::size_t n_elem, std::size_t sz) {
#if defined(__APPLE__)
	malloc_zone_t *zone = malloc_default_zone();
	return malloc_zone_calloc(zone, n_elem, sz);
#elif defined(__linux__) && defined(__GLIBC__)
	return __libc_calloc(n_elem, sz);
#else
	return std::calloc(n_elem, sz);
#endif
}

//! Pool-aware calloc.  Routes through `new_redirected` when the pool
//! is active and the total fits a bucket; otherwise falls through to
//! libsystem `calloc` (which sources zero-filled pages straight from
//! the OS, no manual memset).
__attribute__((noinline))
static void *kame_calloc(std::size_t n_elem, std::size_t sz) {
	// Overflow-checked multiply.  Mirrors libc's contract: return NULL
	// (no errno set) when the product would wrap.
	std::size_t total;
	if(__builtin_mul_overflow(n_elem, sz, &total))
		return nullptr;
	if( !total) total = 1;  // calloc(0, *) / calloc(*, 0): libc returns
	                        // a uniquely-freeable non-null pointer.
	if( !g_sys_image_loaded || s_alloc_tls_off)
		return libsystem_calloc_for_pool(n_elem, sz);
	// Pool path.  `new_redirected` may dispatch to libsystem itself for
	// over-bucket sizes — that branch returns libsystem-malloc'd memory
	// which `free()` / our interpose will route back to libsystem.
	void *p = new_redirected(total);
	if(p) std::memset(p, 0, total);
	return p;
}

//! Pool-aware realloc.  Three regimes by where `p` came from:
//!   1. `p == NULL`        → equivalent to malloc(n) (via new_redirected)
//!   2. `n == 0`           → equivalent to free(p), return NULL
//!   3. `p` is a pool slot → if new size fits the same slot, return p
//!                           unchanged (no copy).  Otherwise allocate
//!                           a fresh slot, memcpy min(old, n) bytes,
//!                           release the old slot.
//!   4. `p` is foreign     → defer to libsystem realloc.  (Cross-
//!                           allocator realloc would otherwise crash
//!                           libsystem with "pointer being realloc'd
//!                           was not allocated".)
__attribute__((noinline))
static void *kame_realloc(void *p, std::size_t n) {
	if( !p) {
		// p==NULL ⇒ malloc(n).  Pre-activate / post-cleanup must take
		// the libsystem path — keep the explicit guard here because
		// `new_redirected` would otherwise claim a fresh chunk on
		// first call from a pre-main static-init thread (qmake inline
		// mode); we want the chunk-claim deferred to `activateAllocator`.
		if( !g_sys_image_loaded || s_alloc_tls_off)
			return libsystem_realloc_for_pool(nullptr, n);
		return new_redirected(n);
	}
	if( !n) {
		// `realloc(p, 0)` is implementation-defined in C17 (DR 400):
		// glibc/libc++ tend to return NULL and free `p`; some
		// allocators return a unique freeable pointer.  We pick the
		// "free + return NULL" semantics — same as mimalloc.
		// `PoolAllocatorBase::deallocate` is pre-activate-safe (same
		// rationale as `kame_free`).
		if(PoolAllocatorBase::deallocate(p))
			return nullptr;
		libsystem_free_for_pool(p);
		return nullptr;
	}
	// `PoolAllocatorBase::size_of` is pre-activate-safe: it walks the
	// same `s_mmapped_spaces[]` ladder as `deallocate`, returns 0 for
	// any pointer outside our chunks (including the pre-activate case
	// where `s_mmapped_spaces[0] == nullptr`).  No outer
	// `g_sys_image_loaded` guard needed.
	std::size_t old = PoolAllocatorBase::size_of(p);
	if(old) {
		// In our pool.  Same-slot fit ⇒ return unchanged.
		if(n <= old) return p;
		void *q = new_redirected(n);
		if( !q) return nullptr;
		std::memcpy(q, p, old);  // old ≤ n, no overcopy
		PoolAllocatorBase::deallocate(p);
		return q;
	}
	// Foreign pointer — defer to libsystem.  Safe across the call,
	// since libsystem realloc operates on its own allocations.
	return libsystem_realloc_for_pool(p, n);
}

// === Why we interpose `realloc` but NOT `calloc` ===
//
// `realloc` is the correctness-critical one: pool pointers (from our
// `::operator new`) are routinely realloc'd by libcxx / glibc / libsystem
// (e.g. `std::vector` growth on a `vector<T>` whose elements were
// originally allocated via `new`).  Without our interpose, libsystem
// `realloc` rejects the pointer with "pointer being realloc'd was not
// allocated" — the realloc cousin of the `_pthread_tsd_cleanup → free`
// abort that motivated the `free` interpose.  Our `kame_realloc`
// checks `size_of(p)` (chunk-header `SizeOfFn` dispatch); pool pointers
// take the in-pool reshape path, foreign pointers fall through to
// `libsystem_realloc_for_pool`.
//
// `calloc` is NOT interposed.  ObjC's class realization (`_objc_init`
// → `realizeClassMaybeSwiftMaybeRelock`) calls `calloc()` to build
// the class table, then later checks the allocation via
// `malloc_size()` to detect dangling references.  If `calloc` is
// interposed, the class data sits in our pool and libsystem's
// `malloc_size()` returns 0 — ObjC reports "realized class has
// corrupt data pointer" and aborts.  Fixing this properly requires
// also interposing `malloc_size` / `malloc_zone_from_ptr` /
// `malloc_good_size` — the full mimalloc compat surface — which is
// out of scope for this work.
//
// `kame_calloc` stays available below as a non-interposed entry
// point: callers who want pool-backed zero-init can call it
// directly.  Default `calloc()` resolves to libsystem as before;
// because we DO interpose `free`, libsystem-calloc'd pointers
// returned by stdlib code still route back to libsystem free via
// our `kame_free` fallback (`PoolAllocatorBase::deallocate` returns
// false ⇒ `libsystem_free_for_pool` → `malloc_zone_free`).
__attribute__((used))
void *kame_pool_calloc(std::size_t n_elem, std::size_t sz) noexcept {
	return kame_calloc(n_elem, sz);
}

#if defined(__APPLE__)
extern "C" void *realloc(void *, std::size_t);

namespace {
__attribute__((used))
kame_interpose_entry kame_interposers_alloc[]
    __attribute__((section("__DATA,__interpose"))) = {
        { reinterpret_cast<const void *>(&kame_realloc),
          reinterpret_cast<const void *>(&realloc) },
};
} // namespace
#else
extern "C" __attribute__((noinline))
void *realloc(void *p, std::size_t n) {
	return kame_realloc(p, n);
}
#endif

// Phase 4b-final: `release_pools()` / `report_statistics()` / per-template
// `release_pools()` / `PoolAllocatorBase::release_chunks()` are all gone.
// They walked the per-template `s_chunks_of_type[]` registry — itself
// retired — and had no live callers anywhere in the tree (only a
// commented-out comment in `kame/main.cpp:345` and an unused declaration
// in `allocator.h`).  Memory is reclaimed naturally:
//   * Empty chunks released by `owner_release` (Phase 4a) /
//     `cross_release` (cross-thread last-slot) / `release_dll_chunks_for_thread`
//     (thread exit) call `deallocate_chunk` which mprotects PROT_NONE +
//     clears the region's claim bit, so the slot region is available
//     for a future chunk-claim.
//   * Process exit reclaims all mmap'd regions via OS teardown; no
//     `munmap` cleanup needed (see the comment in allocator.h about
//     "Why the destructor does NOT call release_pools()").

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

// Global `operator new` / `operator delete` MUST stay non-inline
// (C++ §17.6.4.6 replacement function rules).  An experiment to
// header-inline them tripped clang's `-Winline-new-delete` warning
// and crashed the STM tests with SIGTRAP at startup: `delete p`
// sites in libcxx headers (which don't include allocator.h) resolved
// to the libcxx default `operator delete` instead of our
// replacement, calling `free()` on a KAME pool pointer.  Cross-TU
// inlining of the alloc/dealloc fast paths is the job of LTO, not of
// header-only replacement operators.
//
// The inner work is still as inline-friendly as possible:
//   - `new_redirected` is header-inline (allocator_prv.h), so the
//     full alloc fast path (size→bucket + freelist pop) folds into
//     `operator new`'s single TU.
//   - `deallocate_pooled_or_free` is `inline` here in the same TU
//     so the recursive `deallocate_<>` ladder folds into
//     `operator delete`.
// The only remaining cross-TU boundary is `operator new` /
// `operator delete` itself — one direct branch per `new T` /
// `delete p`.

// `noinline` on every global `operator new` / `operator delete`:
// prevents LTO from inlining our replacement into other TUs.
// Without this, LTO can inline the pool path into library code
// that subsequently calls `free()` directly on the returned pointer
// (legal-but-fragile mixing of `new` with `free()`), and libsystem
// aborts with "pointer being freed was not allocated" at thread
// exit because the pool pointer never went through `malloc()`.
// Marking the replacements `noinline` forces every call to traverse
// the cross-TU boundary, which keeps the "all allocs go through one
// override" invariant the standard expects of replacement functions.
__attribute__((noinline))
void* operator new(std::size_t size) {
    KAME_HISTO_REC(size);
    return new_redirected(size);
}
__attribute__((noinline))
void* operator new[](std::size_t size) {
    KAME_HISTO_REC(size);
    return new_redirected(size);
}

__attribute__((noinline))
void operator delete(void* p) noexcept {
    deallocate_pooled_or_free(p);
}
__attribute__((noinline))
void operator delete[](void* p) noexcept {
    deallocate_pooled_or_free(p);
}

__attribute__((noinline))
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    KAME_HISTO_REC(size);
    return new_redirected(size);
}
__attribute__((noinline))
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    KAME_HISTO_REC(size);
    return new_redirected(size);
}
__attribute__((noinline))
void operator delete(void* p, const std::nothrow_t&) noexcept {
    deallocate_pooled_or_free(p);
}
__attribute__((noinline))
void operator delete[](void* p, const std::nothrow_t&) noexcept {
    deallocate_pooled_or_free(p);
}

// C++14 sized deallocation forms.  Without these overrides, libcxx's
// inline `operator delete(p, size)` defaults to `free(p)` directly
// (because the default sized form in libcxx is `inline void
// operator delete(void *p, size_t) { ::operator delete(p); }` and
// LTO can collapse the chain to a direct `free()`).  Any `new T[]` /
// `std::vector<T>::~vector()` call site that uses the sized form
// would then call `free()` on a KAME pool pointer → libsystem abort
// at thread/process exit.  Symptom under LTO -O3: SIGTRAP from
// `___BUG_IN_CLIENT_OF_LIBMALLOC_POINTER_BEING_FREED_WAS_NOT_ALLOCATED`
// inside `_pthread_tsd_cleanup` calling a thread_local destructor
// that frees a vector buffer.
//
// All sized / aligned forms route to the same `deallocate_pooled_or_free`
// (size is unused — the bitmap lookup determines slot identity).
__attribute__((noinline))
void operator delete(void* p, std::size_t /*size*/) noexcept {
    deallocate_pooled_or_free(p);
}
__attribute__((noinline))
void operator delete[](void* p, std::size_t /*size*/) noexcept {
    deallocate_pooled_or_free(p);
}

// C++17 aligned new — route to libsystem for over-aligned allocations.
// Our pool guarantees 16 B slot alignment (max_align_t on every
// supported arch), so under-16B aligned allocations come from us.
// Over-aligned (`new (std::align_val_t{64}) Foo`) goes to libsystem
// via posix_memalign and back via free — the aligned operator
// delete forms below route there.
__attribute__((noinline))
void* operator new(std::size_t size, std::align_val_t al) {
    if ((std::size_t)al <= ALLOC_ALIGNMENT)
        return new_redirected(size);
    void *p = nullptr;
    if (posix_memalign(&p, (std::size_t)al, size) != 0)
        throw std::bad_alloc();
    return p;
}
__attribute__((noinline))
void* operator new[](std::size_t size, std::align_val_t al) {
    return ::operator new(size, al);
}
__attribute__((noinline))
void* operator new(std::size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
    if ((std::size_t)al <= ALLOC_ALIGNMENT)
        return new_redirected(size);
    void *p = nullptr;
    posix_memalign(&p, (std::size_t)al, size);
    return p;
}
__attribute__((noinline))
void* operator new[](std::size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
    return ::operator new(size, al, std::nothrow);
}
__attribute__((noinline))
void operator delete(void* p, std::align_val_t al) noexcept {
    if ((std::size_t)al <= ALLOC_ALIGNMENT)
        deallocate_pooled_or_free(p);
    else
        std::free(p);  // came from posix_memalign
}
__attribute__((noinline))
void operator delete[](void* p, std::align_val_t al) noexcept {
    ::operator delete(p, al);
}
__attribute__((noinline))
void operator delete(void* p, std::size_t /*size*/, std::align_val_t al) noexcept {
    ::operator delete(p, al);
}
__attribute__((noinline))
void operator delete[](void* p, std::size_t /*size*/, std::align_val_t al) noexcept {
    ::operator delete(p, al);
}

char *PoolAllocatorBase::s_mmapped_spaces[ALLOC_MAX_MMAP_ENTRIES];
std::atomic<PoolAllocatorBase::BitmapWord>
    PoolAllocatorBase::s_claim_bitmap[ALLOC_MAX_MMAP_ENTRIES
                                       * PoolAllocatorBase::BITMAP_WORDS_PER_REGION];
uint8_t PoolAllocatorBase::s_back_offset[ALLOC_MAX_MMAP_ENTRIES
                                          * PoolAllocatorBase::NUM_ALLOCATORS_IN_SPACE];
std::atomic<uint64_t>
    PoolAllocatorBase::s_region_has_free[PoolAllocatorBase::REGION_BITMAP_WORDS];
template <unsigned int ALIGN, bool FS, bool DUMMY>
ALLOC_TLS PoolAllocator<ALIGN, DUMMY, DUMMY> *
    PoolAllocator<ALIGN, FS, DUMMY>::s_my_chunk;
// Per-thread DLL head/tail — sole source of truth for chunks this
// thread can allocate from.
template <unsigned int ALIGN, bool FS, bool DUMMY>
ALLOC_TLS PoolAllocator<ALIGN, DUMMY, DUMMY> *
    PoolAllocator<ALIGN, FS, DUMMY>::s_dll_head;
template <unsigned int ALIGN, bool FS, bool DUMMY>
ALLOC_TLS PoolAllocator<ALIGN, DUMMY, DUMMY> *
    PoolAllocator<ALIGN, FS, DUMMY>::s_dll_tail;
// Phase 5n: per-thread DLL walk cursor + exhaustion flag.
template <unsigned int ALIGN, bool FS, bool DUMMY>
ALLOC_TLS PoolAllocator<ALIGN, DUMMY, DUMMY> *
    PoolAllocator<ALIGN, FS, DUMMY>::s_dll_cursor;
template <unsigned int ALIGN, bool FS, bool DUMMY>
ALLOC_TLS bool
    PoolAllocator<ALIGN, FS, DUMMY>::s_dll_exhausted;

// (Per-template `thread_local TlsGuard s_tls_guard` removed.
//  AllocThreadExitCleanup::~AllocThreadExitCleanup — fired via the pthread_key dtor
//  registered by `XThreadLocal<AllocThreadExitCleanup>` on first allocate() —
//  is now the sole place that drains the per-thread AllocSlot
//  freelists, runs `clear_owner_tls`, and sets `s_alloc_tls_off =
//  true` at thread exit.  Eliminates the C++ thread_local init thunk
//  that macOS arm64 emits for `(void)&s_tls_guard` in the allocate()
//  hot path.)

// FS=false PoolAllocator instantiations.
//
// Phase 5d-4 layout uses three stages with explicit ALIGN values (64,
// 256, 1024).  ALLOC_ALIGN1 (= 32 on 64-bit) is retained as the ALIGN
// of the legacy FS=false buckets 6/8/10/12/14 (slot sizes 96/128/160/
// 192/224); bucket 16 (size 256) uses ALLOC_ALIGN(256) = ALLOC_ALIGN2
// = 256.  So we need ALIGN=32, 64, 256, 1024 — four total instantiations.
template class PoolAllocator<32u, false>;     // buckets 6, 8, 10, 12, 14
template class PoolAllocator<64u, false>;     // buckets 17..24 (Phase 5d-4 stage 1)
template class PoolAllocator<256u, false>;    // bucket 16 + buckets 25..32 (Phase 5d-4 stage 2)
template class PoolAllocator<1024u, false>;   // buckets 33..40 (Phase 5d-4 stage 3)

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
