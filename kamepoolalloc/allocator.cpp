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

//#define GUARDIAN 0xaaaaaaaauLL
//#define FILLING_AFTER_ALLOC 0x55555555uLL
// per-thread floor on `owner_release`.  Stop releasing
// when this thread's DLL has fewer than this many chunks for the given
// (ALIGN, FS) template — avoids release / re-mmap thrashing on bursty
// workloads.
//
// Value tuning history:
//   * 2 — fine for the original `s_tls.my_chunk` + DLL design.
//   * 16 — bumped as a workaround for the bucket34_repro
//     33.5 → 0.24 M/s Linux regression, on the (incorrect) theory
//     that aggressive release / re-mmap was the cause.
//   * REAL fix landed — `s_tls.dll_cursor` / `s_tls.dll_exhausted`
//     was the culprit, not the floor.  Three direct
//     `batch_return_to_bitmap` sites now reset the cursor so the
//     next walk finds the revived chunks.
//   * This commit: 16 → 2.  With an earlier change the floor=16 bloat is
//     unnecessary; bucket34_repro 1t actually IMPROVES at floor=2
//     (15-22 → 27 M/s) because empty chunks release sooner,
//     improving region locality and reducing post-workers RSS.
//     All other workloads parity.
#define LEAVE_VACANT_CHUNKS_PER_THREAD 2

#include "allocator.h"
#include "kame_pool.h"        // C-API stats struct + version macro
#if defined(__linux__)
#  include <dirent.h>          // /sys/devices/system/node walk (§14C)
#  include <sched.h>           // sched_getcpu                  (§14C)
#  include <sys/syscall.h>     // SYS_mbind                     (§14C)
#  include <unistd.h>          // syscall                       (§14C)
#endif

#ifndef USE_STD_ALLOCATOR

#include "atomic_mfence.h"   // readBarrier / writeBarrier / pause4spin
                             // (kamepoolalloc-internal, mirrors kame/atomic.h's
                             //  arch-select chain but drops atomic_shared_ptr et al)

#include <algorithm>
#include <assert.h>
#include <cerrno>
#include <chrono>           // (§28.1) lazy-drain wall clock for LRC_MMAP push
#include <cstdio>
#include <cstdlib>
#include <cstring>          // std::memset / std::memcpy
#include <limits>           // (§30) numeric_limits for the realtime-mode preset
#include <new>              // std::get_new_handler / std::bad_alloc (§18 OOM)
                            // (glibc's `<string.h>` puts them in the
                            //  global namespace only — libc++/Apple
                            //  pull them into `std::` transitively but
                            //  libstdc++ does not.  `<cstring>` is the
                            //  portable C++ way.)
#include <type_traits>
#if defined(__APPLE__)
    #include <malloc/malloc.h>   // for malloc_zone_from_ptr / malloc_zone_free
#elif defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
    #include <malloc.h>          // for _aligned_malloc / _aligned_free
                                 // (over-aligned alloc fallback when
                                 // alignment exceeds the pool's 16-B
                                 // guarantee)
    // VirtualAlloc / VirtualFree / MEM_COMMIT etc. for the radix L2 node
    // allocator (the Windows counterpart to mmap()).  WIN32_LEAN_AND_MEAN
    // keeps the symbol load small; we only need the memory + handle API.
    #ifndef WIN32_LEAN_AND_MEAN
    #  define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
    #  define NOMINMAX   // keep windows.h's min/max macros from clobbering
    #                    // std::numeric_limits<>::max() etc. (breaks MSVC)
    #endif
    #include <windows.h>
    #include <intrin.h>          // __readgsqword — PEB walk in the §31
                                 // free-redirect (loader-lock-safe module
                                 // enumeration; see kame_patch_all_modules)
#endif
#if KAME_FAST_TSD
    #include <pthread.h>
#endif

// (§32) Drop-in default: the standalone DYLIB build interposes the FULL libc
// malloc family by default on macOS, so kamepoolalloc is a real
// DYLD_INSERT_LIBRARIES / drop-in allocator (like mimalloc/jemalloc) out of the
// box.  Safe because the `malloc_size` co-interpose (see the macOS FULL block
// near the bottom) returns the true capacity of pool pointers — without it the
// Swift runtime (`__StringStorage`) and ObjC class realization corrupt.  Soak:
// Foundation, libswiftCore (CPython), QtCore, C++ STL, 2000-thread stress — all
// clean.  Opt out with -DKAMEPOOLALLOC_CONSERVATIVE_INTERCEPT (free+realloc only).
// Linux dylibs keep the explicit -DKAMEPOOLALLOC_FULL_INTERCEPT opt-in pending
// their own soak; Windows is default-on too now (soaked — see the §31 block).
// The inline kame.app (MH_EXECUTE) is unaffected — dyld
// honours __interpose only from MH_DYLIB, so its interpose set is inert either way.
#if defined(KAMEPOOLALLOC_DYLIB) && defined(__APPLE__) \
    && !defined(KAMEPOOLALLOC_FULL_INTERCEPT) \
    && !defined(KAMEPOOLALLOC_CONSERVATIVE_INTERCEPT)
#  define KAMEPOOLALLOC_FULL_INTERCEPT 1
#endif

// Windows: full malloc-family interception is the default too — matching
// the macOS dylib drop-in.  NOTE there is no `KAMEPOOLALLOC_DYLIB` gate
// here, unlike macOS: the §31 IAT redirect below patches imports of any
// loaded module from the *inline-compiled* kame.exe, so this covers the
// production executable (on macOS only an MH_DYLIB can interpose, so the
// inline kame.app stays conservative).  Ruby (msvcrt heap) is excluded by
// `kame_is_crt_dll` (ucrtbase / api-ms-win-crt-heap only), so only the
// UCRT-family (kame.exe, Qt, libc++) is pooled; the `_msize` co-redirect
// (the Windows analog of macOS `malloc_size`) keeps size-queries correct.
// Opt out with -DKAMEPOOLALLOC_CONSERVATIVE_INTERCEPT (free-family only).
#if (defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)) \
    && !defined(KAMEPOOLALLOC_FULL_INTERCEPT) \
    && !defined(KAMEPOOLALLOC_CONSERVATIVE_INTERCEPT)
#  define KAMEPOOLALLOC_FULL_INTERCEPT 1
#endif

#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
// ===================================================================
// (§31) Windows free-family redirect — genuine-CRT bypass pointers.
//
// On PE/COFF there is no cross-module interposition of the replaceable
// `operator new` / `operator delete` (or `free`) the way ELF strong
// symbols and Mach-O `__DATA,__interpose` provide.  Each prebuilt DLL
// (Qt6*.dll, libc++.dll) binds `free` / `operator delete` to the UCRT
// at *its* link time, so a Qt object that KAME pool-allocated and then
// hands back to Qt is freed by Qt via the UCRT's `free()` — heap
// corruption, because a pool pointer is not a CRT pointer.  (Confirmed
// on-target: the Create-Driver dialog's child widgets, pool-allocated
// in kame.exe, are deleted by QObjectPrivate::deleteChildren() inside
// Qt6Core.dll → libc++ free → int3 in ntdll's heap path.)
//
// We close the gap the way mimalloc-redirect.dll does.  By default we
// patch only the *deallocation* family (`free` / `realloc` / `_msize`),
// matching the production-conservative model: KAME allocs stay in the
// pool (kame.exe `operator new` override), Qt / Ruby / Python allocs
// stay in the CRT, and frees are reconciled wherever they happen via
// `kame_free`'s pool-or-foreign dispatcher.  See
// `kame_pool_win_install_redirect` near the bottom.
//
// FULL interception (`KAMEPOOLALLOC_FULL_INTERCEPT`, default-ON on Windows
// per the §32 block above) extends the IAT patch table to `malloc` and
// `calloc`, so EVERY UCRT-family alloc-family call (kame.exe, Qt, libc++)
// is routed through the pool — a true mimalloc-style drop-in, matching the
// macOS dylib default.  Ruby (msvcrt heap) stays excluded by
// `kame_is_crt_dll`, and `_msize` is co-redirected (the Windows analog of
// macOS `malloc_size`) so size-queries see pool pointers' true capacity.
// Soaked on-target (MinGW64 + lld): kame.exe — Qt + Ruby + Python +
// Create-Driver dialog teardown + load test — clean exit (176 slots / 58
// modules patched); alloc_stress_test 2000-thread / 42 M-op stress PASSES.
// Opt OUT with `-DKAMEPOOLALLOC_CONSERVATIVE_INTERCEPT` (free-family
// reconcile only — the prior 20-yr-stable Qt/Ruby/Python conservative model).
//
// These pointers hold the *genuine* UCRT entry points, resolved once
// from ucrtbase.dll.  The pool's own "forward a foreign pointer to the
// real heap" paths (`libsystem_*_for_pool`) and its region release
// (`free_munmap`) MUST call these — NOT `std::free` / `std::realloc` —
// because once the IAT redirect is installed, kame.exe's own `free`
// import is patched to route back into the pool; a plain `std::free`
// would recurse forever.  Until resolution/install happens these stay
// null and the call sites fall back to `std::*` (still genuine then,
// since nothing is patched yet).
typedef void        (*kame_real_free_fn)(void *);
typedef void       *(*kame_real_realloc_fn)(void *, std::size_t);
typedef void       *(*kame_real_calloc_fn)(std::size_t, std::size_t);
typedef std::size_t (*kame_real_msize_fn)(void *);
typedef void       *(*kame_real_malloc_fn)(std::size_t);
static kame_real_free_fn    g_real_free    = nullptr;
static kame_real_realloc_fn g_real_realloc = nullptr;
static kame_real_calloc_fn  g_real_calloc  = nullptr;
static kame_real_msize_fn   g_real_msize   = nullptr;
static kame_real_malloc_fn  g_real_malloc  = nullptr;

static void kame_resolve_real_crt() noexcept {
    if(g_real_free) return;  // idempotent
    HMODULE h = GetModuleHandleA("ucrtbase.dll");
    if( !h) h = LoadLibraryA("ucrtbase.dll");
    if( !h) h = GetModuleHandleA("msvcrt.dll");
    if( !h) return;
    g_real_free    = reinterpret_cast<kame_real_free_fn>   (GetProcAddress(h, "free"));
    g_real_realloc = reinterpret_cast<kame_real_realloc_fn>(GetProcAddress(h, "realloc"));
    g_real_calloc  = reinterpret_cast<kame_real_calloc_fn> (GetProcAddress(h, "calloc"));
    g_real_msize   = reinterpret_cast<kame_real_msize_fn>  (GetProcAddress(h, "_msize"));
    g_real_malloc  = reinterpret_cast<kame_real_malloc_fn> (GetProcAddress(h, "malloc"));
}
// Installed at pool activation (see `activateAllocator`).  Defined far
// below, after the pool dispatchers (`kame_free` / `kame_realloc`) it
// routes the patched imports to.
extern "C" void kame_pool_win_install_redirect() noexcept;
#endif // _WIN32 redirect bypass pointers

// Per-thread flag: set to true when AllocThreadExitCleanup has run, signalling
// that pool-allocator TLS (s_tls.my_chunk, freelists, pin counts) is no
// longer valid.  Trivially destructible (`ALLOC_TLS` = `__thread`) so it
// survives past all thread_local / pthread_key destructors.  Checked in
// `new_redirected()` to fall back to malloc for any heap operations
// that occur during later TLS cleanup phases (e.g. pthread_key dtors
// like RunnerCounterRegistration).
// (§23) IE-TLS: this flag is on the hot path of `new_redirected_large`
// (every large alloc reads it).  Under default global-dynamic TLS the
// access triggers `__tls_get_addr`, which perf-record measured at ~17 %
// of the 65 KiB tight-loop CPU.  IE-TLS bypasses the GOT round-trip and
// reads via fs:offset directly.  Same single-bool fits the IE budget
// easily; updates happen exactly once per thread (at TLS teardown).
ALLOC_TLS_IE bool s_alloc_tls_off = false;

// Per-thread owner id for the deallocate owner-check fast path.  A
// chunk stamps `m_owner_id = s_tls_owner_id` at allocate_chunk; a
// freeing thread compares its own id to decide owner-side
// (chunk-local freelist push, no atomics) vs cross-thread (batch).
// Assigned once per thread from a global counter on first use; 0 is
// reserved for "unassigned" so a never-allocated thread's frees never
// spuriously match a chunk (chunks always carry a non-zero id).
// (§hot-tls) `g_tls_page` (KameTlsPage: last_region_base + owner_id + m_slots[])
// is defined further down.  We use `kame_page()->owner_id` here.
// See allocator_prv.h for the rationale.
static std::atomic<uint32_t> s_owner_id_next{1};
static inline uint32_t kame_owner_id() noexcept {
    uint32_t id = kame_page()->owner_id;
    if(__builtin_expect(id == 0, 0)) {
        do { id = s_owner_id_next.fetch_add(1, std::memory_order_relaxed); }
        while(id == 0);   // skip the reserved 0 on 32-bit wrap
        kame_page()->owner_id = id;
    }
    return id;
}

// (§S7) The §36 orphan Treiber-stack packing (biased chunk ptr + 18-bit ABA
// tag in s_orphan_head, ORPHAN_PTR_BIAS / ORPHAN_TAG_MASK) is retired with the
// stack itself — the atomic_shared_ptr orphan chain needs no ABA tag (the
// smart-ptr refcount keeps a popped node alive across the CAS).
static_assert(((size_t)ALLOC_MIN_CHUNK_SIZE % ((size_t)1 << 18)) == 0,
              "ALLOC_MIN_CHUNK_SIZE must be a multiple of 2^18 so the unit "
              "boundary (= biased PoolAllocator ptr) has 18 low zero bits");

// (§28.2 / §28.4 / §28.5) Tier-attribution counters for kame_pool_get_stats.
//
// HISTORY:
//   §28.2 single global atomic per counter — 10x MT regression
//         (cache-line bouncing on every alloc/free).
//   §28.4 LRC_STATS_SHARDS=64 cache-line-aligned shards — fixed up to 64T,
//         but 128T re-introduces 2-way coherence collisions on the dedicated
//         tier's hot path (≈ 17 % drop at 64 KiB / 128T on Ohtaka).
//   §28.5 (this commit) DROP the running counters entirely — they were pure
//         telemetry for `kame_pool_get_stats()`, never used by allocator
//         logic.
//
//   * `dedicated_chunk_bytes` is recomputed on demand by walking the region
//     bitmap + back_offset table (already walked for `chunks_live`).  A
//     bit-7 dedicated marker on a base-unit's back_offset selects the
//     chunk; its DEDICATED_SIZE header field gives the size.  Includes
//     cache-parked dedicated chunks too — see header doc.
//   * `large_alloc_count` / `large_alloc_bytes` use 2 plain global atomics.
//     Large allocs (4..32 MiB) are rare (multi-MiB ⇒ ~kHz/thread at most),
//     so a single cache line is fine — no measurable contention.
// Pointer-width counters so i486 (no CMPXCHG8B) doesn't need libatomic.
// Live values are bounded by VA size on 32-bit (≤ 4 GiB), so size_t fits;
// the 32-bit "transiently negative" concern is replaced by unsigned-wrap
// semantics — a fetch_sub that briefly underflows produces ~SIZE_MAX, which
// the "> cap" pre-push gate naturally rejects (push refused), exactly the
// effect the prior signed `int64_t` clamp produced.
static std::atomic<size_t> g_large_alloc_count{0};
static std::atomic<size_t> g_large_alloc_bytes{0};

static inline void stats_inc_large(std::size_t mmap_size) noexcept {
    g_large_alloc_count.fetch_add(1, std::memory_order_relaxed);
    g_large_alloc_bytes.fetch_add(mmap_size, std::memory_order_relaxed);
}
static inline void stats_dec_large(std::size_t mmap_size) noexcept {
    g_large_alloc_count.fetch_sub(1, std::memory_order_relaxed);
    g_large_alloc_bytes.fetch_sub(mmap_size, std::memory_order_relaxed);
}

#if KAME_FAST_TSD
// Fast pthread-TSD bypass of macOS TLV thunk for KameTlsPage.
// See header for the design overview.  This global carries the discovered
// byte offset within the pthread struct (= `kame_thread_pointer()`) where
// our pthread_key's TSD slot lives.  Zero means "not yet initialised";
// the hot accessor (`kame_page()` in the header) falls to `tls_page_ie`
// or calls `kame_page_cold()` in that state.
std::size_t s_kame_page_tsd_offset = 0;

namespace {
pthread_key_t s_kame_page_key;

// Constructor priority 101: runs early but after libc/libpthread
// constructors at priorities <= 100.  If pthread_key_create or the
// sentinel scan fails, the offset stays 0 and the allocator stays on
// the TLV path with no further runtime overhead (degraded mode).
//
// Inter-TU ordering: other TUs' constructor(101)s may run before this
// one and call operator new; they hit the TLV/IE fallback (offset == 0),
// which is safe.  Once we run, subsequent allocations on the main
// thread go through fast TSD.  Other threads plant their own TSD slot
// lazily on their first allocation via `kame_page_cold` below.
__attribute__((constructor(101)))
void kame_tls_init_fast() noexcept {
    char *tp = kame_thread_pointer();
    if( !tp) {
#if defined(KAME_FIXED_TSD_SLOT) && (KAME_FIXED_TSD_SLOT)
        fprintf(stderr, "kamepoolalloc FATAL: KAME_FIXED_TSD_SLOT build but "
            "no thread pointer at init.\n");
        abort();
#else
        return;
#endif
    }

#if defined(KAME_FIXED_TSD_SLOT) && (KAME_FIXED_TSD_SLOT)
    // Fixed-slot build (opt-in; see kame_page()).  The hot path baked
    // KAME_FIXED_TSD_SLOT as the TSD byte offset (no runtime
    // `s_kame_page_tsd_offset` load, no offset guard — mimalloc-parity).
    // Force OUR OWN pthread key to land exactly at that slot: allocate
    // keys until the sentinel scan reports the baked offset.  Held probe
    // keys must NOT be deleted mid-spin — `pthread_key_create` hands out
    // the lowest free slot, so deleting one would let the next create
    // reuse it and never advance; delete them only AFTER the hit, to
    // return them to the PTHREAD_KEYS_MAX budget.  No runtime fallback
    // exists (a graceful fast/slow switch costs the hot path, and a
    // dlopen'd interposer does NOT retroactively rebind malloc — both
    // measured), so on overshoot / key exhaustion, fail loudly.
    {
        const std::size_t WANT = (std::size_t)(KAME_FIXED_TSD_SLOT);
        enum { MAX_SPIN = 480 };               // < PTHREAD_KEYS_MAX (512)
        pthread_key_t held[MAX_SPIN];
        int  nheld = 0;
        bool hit = false;
        for(int i = 0; i < (int)MAX_SPIN; ++i) {
            pthread_key_t k;
            if(pthread_key_create(&k, nullptr) != 0) break;   // key exhaustion
            // Unique sentinel per iteration so the scan can never match a
            // previously-held key's slot.
            const uintptr_t sent =
                (uintptr_t)0xDEAD600D11AA0000ull ^ (uintptr_t)(unsigned)i;
            pthread_setspecific(k, (void *)sent);
            std::size_t off = 0; bool got = false;
            for(std::size_t o = 0; o < 4096; o += 8)
                if(*reinterpret_cast<uintptr_t *>(tp + o) == sent) {
                    off = o; got = true; break;
                }
            if(got && off == WANT) { s_kame_page_key = k; hit = true; break; }
            held[nheld++] = k;                 // hold to advance the allocator
            if(got && off > WANT) break;        // overshot — cannot go back
        }
        for(int i = 0; i < nheld; ++i) pthread_key_delete(held[i]);
        if( !hit) {
            fprintf(stderr,
                "kamepoolalloc FATAL: built with -DKAME_FIXED_TSD_SLOT=%zu, but "
                "could not place a pthread TSD key at that slot (overshoot or "
                "key exhaustion) on this runtime. Rebuild with a reachable "
                "KAME_FIXED_TSD_SLOT (probe s_kame_page_tsd_offset for this "
                "OS), or drop the flag for the robust runtime-offset build.\n",
                WANT);
            abort();
        }
        s_kame_page_tsd_offset = WANT;          // cold-path readers (teardown) use it
        pthread_setspecific(s_kame_page_key, &g_tls_page);
        tls_page_ie = &g_tls_page;
    }
#else
    if(pthread_key_create(&s_kame_page_key, nullptr) != 0) return;

    // Sentinel scan: plant a magic value via the POSIX API, then walk
    // the pthread struct to find which byte offset received it.  POSIX
    // doesn't expose the layout, but the implementation must store the
    // value somewhere reachable from the thread pointer for
    // `pthread_getspecific` to be fast — we rely on it being a fixed
    // offset, true for both Apple's libc and glibc.
    const uintptr_t sent1 = (uintptr_t)0xDEAD600D11AA1234ull;
    pthread_setspecific(s_kame_page_key, (void *)sent1);

    std::size_t off1 = 0;
    // 4 KiB upper bound covers all libc TSD layouts we know about
    // (Apple reserves slots 0..N, then user keys start; offsets are
    // typically < 2 KiB).  Stride 8 — slot is a pointer.
    for(std::size_t off = 0; off < 4096 && !off1; off += 8) {
        uintptr_t v = *reinterpret_cast<uintptr_t *>(tp + off);
        if(v == sent1) off1 = off;
    }

    if(off1) {
        s_kame_page_tsd_offset = off1;
        // Plant THIS thread's (= typically the main thread's) TSD slot
        // now so the next allocation hits the fast path on the first try.
        // Touching the __thread struct triggers TLV lazy init; the
        // resulting address is stable for this thread's life.
        pthread_setspecific(s_kame_page_key, &g_tls_page);
        tls_page_ie = &g_tls_page;
    }
    else {
        // Scan failed — leave offset at 0 (degraded TLV-only mode).
        pthread_setspecific(s_kame_page_key, nullptr);
    }
#endif
}
} // anon namespace

// Cold path for the fast-TSD accessor in the header.  Called when
// either guard branch fails (offset == 0 → pre-init; or TSD slot
// null → first allocation on this thread, plant the pointer).
// `preserve_most` (matching the header decl) tells the caller that
// this call preserves nearly all caller-saved registers.
[[clang::preserve_most]]
__attribute__((cold, noinline))
KameTlsPage *kame_page_cold() noexcept {
    // (dylib TLV-bootstrap leak fix) Park the fast-TSD slot at the teardown
    // sentinel BEFORE the general-dynamic `&g_tls_page` access below.
    //
    // In a DYLIB build, that first thread_local touch makes dyld lazily
    // instantiate this image's per-thread TLV block (all our thread_locals:
    // g_tls_page + tls_cross_dealloc_batch (16 KiB) + the per-ALIGN s_tls +
    // tls_alloc_thread_exit_cleanup, ~32 KiB) via a single `malloc` — which the
    // dylib interposes.  Routed into the pool, that malloc claims a ~32 KiB
    // chunk to hold the process's OWN per-thread TLS; at thread exit dyld frees
    // the block off the pool's per-thread reclaim discipline, so the chunk is
    // never returned -> ~8 units leaked PER THREAD (unbounded across thread
    // churn).  Confirmed by lldb: deallocate -> kame_page() -> kame_page_cold ->
    // _tlv_get_addr -> dyld instantiateVariable -> malloc -> kame_pool_malloc.
    //
    // Parking the slot at g_teardown_page makes that re-entrant malloc observe
    // kame_thread_torn_down()==true in cold_first_access / new_redirected_large,
    // so it falls to libsystem_malloc_for_pool (the real heap) instead of the
    // pool.  The TLV block is then never pooled, and its eventual free passes
    // straight through (radix ABSENT -> libsystem free).  The slot is restored
    // to the real page before returning, so the outer caller is unaffected.
    //
    // Inline/static builds (production kame.app / kame.exe) reach g_tls_page via
    // initial-exec / static TLS — the block is allocated by the kernel at thread
    // creation, NOT via malloc — so no re-entry occurs and this parking is inert.
    char *tp = (s_kame_page_tsd_offset != 0) ? kame_thread_pointer() : nullptr;
    if(tp)
        *reinterpret_cast<KameTlsPage **>(tp + s_kame_page_tsd_offset) = &g_teardown_page;
    KameTlsPage *p = &g_tls_page;   // one GD TLV — cold, paid once per thread
    tls_page_ie = p;
    if(s_kame_page_tsd_offset != 0) {
        pthread_setspecific(s_kame_page_key, p);
        if(tp)
            *reinterpret_cast<KameTlsPage **>(tp + s_kame_page_tsd_offset) = p;
    }
    return p;
}
#endif // KAME_FAST_TSD

// Forward decl for `drain_thread_slot_freelists` — now a retained no-op
// stub (see its definition).  Owner-thread freelists are no longer in a
// global `g_thread_slots[]` array; each chunk's freelist is chunk-local
// (`m_freelist_head[]`) and is drained per-chunk by
// `release_dll_chunks_for_thread` before that chunk's BIT_OWNED clear.
// Kept as a symbol so the `~AllocThreadExitCleanup` call site stays valid.
namespace { void drain_thread_slot_freelists() noexcept; }

// (§22) Unified per-thread large-recycle cache, shared by BOTH large
// tiers so a tight alloc/free loop of 64 KiB–32 MiB reuses warm VA+pages
// instead of paying the per-cycle release every time:
//   - LRC_CHUNK : §15 dedicated multi-unit chunks (64 KiB–4 MiB).  The
//                 claim bits stay SET while cached (so no other thread can
//                 re-claim the units); reuse returns the payload directly
//                 with the chunk_header intact — NO claim_chunk, NO madvise.
//                 True release (on eviction / thread-exit) = the N-bit
//                 bitmap-CAS claim-clear + madvise inside deallocate_chunk.
//   - LRC_MMAP  : §19 single-mmap large allocs (4 MiB–32 MiB).  The VA
//                 stays mapped while cached (radix CLEARED for double-free
//                 routing); reuse re-registers the radix.  Release = munmap.
// The freeing thread wins the kind's single-winner clearing CAS (bitmap
// N-bit for chunk, radix for mmap) and is thus the unique owner, so the
// deferred release is race-free regardless of thread-exit ordering.  The
// `kind` tag selects the release backend on eviction.  Cache + helpers are
// defined far below (after deallocate_chunk / the mmap helpers); these are
// forward decls so the earlier §15 dedicated-chunk paths can reach them.
namespace {
enum { LRC_MMAP = 0, LRC_CHUNK = 1 };
char *large_recycle_pop(std::size_t need, unsigned kind) noexcept;
bool  large_recycle_push(char *base, std::size_t size, unsigned kind) noexcept;
}

// Per-thread cleanup at thread exit.  chunks are no longer
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
        // `drain_thread_slot_freelists()` is a retained no-op stub now
        // (see its definition); the per-chunk freelist drain has been
        // folded into the per-template DLL walk below
        // (`release_dll_chunks_for_thread`), which drains each chunk's
        // freelist right before clearing its BIT_OWNED.  Call kept for
        // call-site / ABI stability.
        drain_thread_slot_freelists();
        // Clear every per-thread bucket chunk pointer BEFORE the DLL
        // teardown walk.  Otherwise a later TLS destructor that
        // allocates could route through a chunk that's about to be
        // released.  After this loop the slow path's per-bucket
        // freelist-ptr slot reads as cleared, so
        // `new_redirected` falls to `cold_first_access`, which
        // observes `s_alloc_tls_off == true` (set a few lines below)
        // and returns `std::malloc(size)`.
        // (§12.3 / §hot-tls) Clear all per-bucket freelist-ptr slots in
        // the TLS page.  A null entry makes new_redirected take the cold
        // path on next access (which observes s_alloc_tls_off == true
        // and routes to std::malloc).
        // Use tls_page_ie for the drain path (TSD slot may already be
        // cleared at thread-exit; IE fallback is safe here).
        {
#if KAME_FAST_TSD
            KameTlsPage *pg = tls_page_ie ? tls_page_ie : &g_tls_page;
#else
            KameTlsPage *pg = &g_tls_page;
#endif
            for(int b = 0; b < ALLOC_NUM_BUCKETS; ++b)
                pg->m_slots[b].freelist_head = nullptr;
        }
        // Walk each registered template's per-thread DLL.  Each
        // callback wipes its own `s_tls.my_chunk` / `s_tls.dll_head` / `s_tls.dll_tail`
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
        // (§hot-tls teardown sentinel) Point this thread's fast-TSD page
        // slot at the static teardown sentinel.  After this, any later
        // pthread_key dtor (e.g. libc++ ~__thread_struct) that frees a pool
        // pointer reaches `deallocate` → owner-id mismatch (sentinel
        // owner_id == 0) → cold `deallocate_pooled`, which identity-compares
        // `kame_page() == &g_teardown_page` and takes a TLS-free path
        // WITHOUT re-touching `s_tls` / `&s_tls.dll_head` — whose TLV may
        // already be finalized, so a `_tlv_get_addr` re-instantiation would
        // `malloc` mid-teardown and trap.  This write is value-only (a
        // pthread TSD slot store, legal during cleanup) — no TLV deref.
        //
        // macOS-only: the sentinel exists solely to give the fast-TSD
        // `kame_page()` a teardown-safe value to return.  On Linux/Windows
        // `tls_page_ie` does not exist (the page is read directly as IE TLS)
        // and `kame_thread_torn_down()` uses the teardown-safe `s_alloc_tls_off`
        // flag set above — nothing to redirect here.
#if KAME_FAST_TSD
        tls_page_ie = &g_teardown_page;
        if(s_kame_page_tsd_offset != 0) {
            pthread_setspecific(s_kame_page_key, &g_teardown_page);
            char *tp = kame_thread_pointer();
            if(tp)
                *reinterpret_cast<KameTlsPage **>(tp + s_kame_page_tsd_offset)
                    = &g_teardown_page;
        }
#endif
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
    // FS=true-only small-slot batch (FS=false bypasses
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

    //! Direct/adaptive dispatch path — FS=true only (    //! FS=false bypasses cross-batch entirely in `deallocate_pooled`
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
        // (§20) Cache the dll-cursor-reset addresses BEFORE
        // batch_return_to_bitmap.  If this is `c`'s last slot AND
        // BIT_OWNED is clear (owner exited), batch_return releases the
        // chunk: the placement-new destructor runs, and `c` becomes a
        // stale pointer — accessing `c->m_owner_dll_head_addr` /
        // `c->m_owner_dll_force_walk_ptr` afterwards is UB by C++'s
        // object-lifetime rule (UBSAN's vptr check fires under
        // -fsanitize=undefined).  The fields are write-once at chunk
        // construction so the cached values are safe across the call.
        void *cached_dll_head_addr = c->m_owner_dll_head_addr;
        auto *cached_force_walk =
            c->m_owner_dll_force_walk_ptr.load(std::memory_order_acquire);
        c->batch_return_to_bitmap(tmp);
        // (§20) `c` may be destructed past this point — use cached values only.
        if(cached_dll_head_addr ==
           PoolAllocator<ALIGN, true, true>::dll_head_tls_addr())
            PoolAllocator<ALIGN, true, true>::reset_dll_walk_state();
        else if(cached_force_walk)
            // Acquire load (above) synchronises with owner-exit's
            // release-store of nullptr in
            // `release_dll_chunks_for_thread`.  Null after owner exit
            // → skip deref; non-null means owner's TLS storage is
            // still live (owner-exit nullifies BEFORE thread teardown).
            cached_force_walk->store(true, std::memory_order_relaxed);
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
        //
        // For each unique chunk the batch returned slots to: signal the
        // OWNER thread's "force walk from head" hint flag.  Without this
        // poke, the owner's `allocate_chunk_path` Phase 2 DLL walk stays
        // gated by its own `dll_exhausted` flag (set after the previous
        // walk found no space) and keeps mmap'ing fresh chunks instead
        // of reusing the slots we just returned.  The single-slot
        // `push_direct` path already does this; the batched flush path
        // skipped it — caught by `bench_xthread_pool -w2 -s64` where the
        // pool inflated +32 regions (1 GiB VA) over a 5-second run.
        //
        // Cache `m_owner_dll_force_walk_ptr` BEFORE
        // `batch_return_to_bitmap`: the call may release the chunk on
        // last-slot return + owner-exit, after which `chunk` is a stale
        // pointer.  The owner's TLS storage that the cached ptr targets
        // lives independently of the chunk; null after owner-exit's
        // release-store, so the post-call deref is safe-or-skipped.
        int i = 0;
        while(i < count) {
            PoolAllocatorBase *chunk = buf[i].chunk;
            std::atomic<bool> *cached_force_walk =
                chunk->m_owner_dll_force_walk_ptr.load(
                    std::memory_order_acquire);
            i += chunk->batch_return_to_bitmap(&buf[i]);
            if(cached_force_walk)
                cached_force_walk->store(true, std::memory_order_relaxed);
        }
        count = 0;
    }
    ~CrossDeallocBatch() noexcept { flush(); }
};
thread_local CrossDeallocBatch tls_cross_dealloc_batch;

// drain_thread_slot_freelists (defined below) is a retained no-op stub.
// Owner-thread freelists are chunk-local and drained per-chunk by
// `release_dll_chunks_for_thread` (per-template DLL walk) before each
// chunk's BIT_OWNED clear — see that function and the stub's own comment.
//
// each touched chunk still has `BIT_OWNER_EXITED == 0`
// at this point (the per-template DLL walk that sets it runs
// AFTER `drain_thread_slot_freelists` in `~AllocThreadExitCleanup`), so
// the cross_release inside batch_return_to_bitmap returns false
// — the owner thread (us) is still alive, no release allowed.
void drain_thread_slot_freelists() noexcept {
    // Single-slot scratch + trailing nullptr sentinel — satisfies
    // `batch_return_to_bitmap`'s `entries[k].chunk == this` walk
    // contract (one matching entry, then the sentinel terminates).
    //
    // freelists hold p_user pointers for BOTH FS=true and
    // FS=false (the "borrow scheme" puts FS=false's user pointer at
    // slot_start, same convention as FS=true).  `batch_return_to_bitmap`
    // and its MaskFn both work on `entries[k].slot == p_user` directly
    // — for FS=false they read the `{bucket, SIZE}` header from
    // `p_user - 8` (chunk_header pad for slot 0, predecessor's
    // reserved tail otherwise).  No per-FS conversion needed.
    // No-op since follow-up "(1)": owner-thread freelists are now
    // chunk-local (PoolAllocatorBase::m_freelist_head[]), not in the
    // global g_thread_slots[] array.  Each chunk's freelist is drained
    // by `release_dll_chunks_for_thread` (per-template DLL walk) right
    // before that chunk's BIT_OWNED clear — see there.  Kept as an
    // empty symbol so the ~AllocThreadExitCleanup call site and any
    // external references stay valid; the compiler elides it.
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
        // Genuine UCRT malloc — NOT the redirected `malloc`.  This IS the
        // pool's region-backing allocator; under KAMEPOOLALLOC_FULL_INTERCEPT
        // a plain `malloc` is IAT-patched to route back into the pool, which
        // would recurse infinitely here (pool → region claim → malloc_mmap →
        // malloc → pool ...).  Mirrors free_munmap's g_real_free.
        // g_real_malloc is resolved before the redirect installs, so the only
        // pre-resolution callers (none on the region path) fall to std::malloc.
        void *p = g_real_malloc ? g_real_malloc(size) : malloc(size);
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
        // Genuine UCRT free — NOT the redirected `free`.  This is pool-
        // backing region memory (malloc'd in mmap_new_region); routing it
        // through the pool dispatcher post-redirect would misclassify it.
        // g_real_free is resolved before the redirect is ever installed.
        if(g_real_free) g_real_free(p);
        else free(p);
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
//
// `[[gnu::used]]` keeps the symbol against `-fdata-sections / -ffunction-
// sections` + GC, and also against lld's static-internal-with-no-explicit-
// reference pruning.  Reported (MinGW64 + lld): the DLL would build
// fine, but `g_sys_image_loaded` stayed `false` at runtime — pool fell
// through to libc malloc and no `Reserve swap space` ever printed.  The
// backup static-init activator below also covers the case where lld
// emits the constructor record but Windows CRT init doesn't pick it up.
[[gnu::used]] __attribute__((constructor(101)))
static void kamepoolalloc_auto_activate() noexcept {
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
    // (§31) Hook the free-family across all modules BEFORE flipping the
    // flag, so the very first pool pointer the pool hands out is already
    // safe to free from any module (Qt / libc++ / kamestm).  Safe to call
    // from this constructor even when it runs in DllMain under the loader
    // lock: the module sweep walks the PEB list (no loader re-entry) — see
    // kame_patch_all_modules.  Idempotent + KAME_POOL_WIN_REDIRECT=0.
    kame_pool_win_install_redirect();
#endif
    g_sys_image_loaded = true;
}
// Backup: a file-scope global whose default constructor unconditionally
// flips the flag.  Static-init ordering is unspecified relative to other
// TUs, but `g_sys_image_loaded = true;` has no other-init dependency
// (just a plain bool store).  Static-init runs reliably on every linker
// we care about — including the Windows path where the priority-tagged
// `__attribute__((constructor))` record may be silently dropped.
namespace {
struct KamePoolAutoActivator {
    KamePoolAutoActivator() noexcept {
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
        kame_pool_win_install_redirect();  // see kamepoolalloc_auto_activate
#endif
        g_sys_image_loaded = true;
    }
};
[[gnu::used]] static KamePoolAutoActivator s_kamepool_auto_activator;
} // namespace
#else
// Inline-compiled mode (qmake): the kame app and each standalone test
// binary contain `allocator.cpp` as a TU of its own, and the activation
// flag flip stays an explicit step — `kame/main.cpp` does it via
// `KamePooledAllocGuard`, the standalone tests via the static-init
// shim in `tests/allocator.cpp`.  Both are no-ops once the dylib build
// path is selected (which is the case for the cmake test build that
// chases LTO interpose semantics).
void activateAllocator() {
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
    // (§31) Patch the free-family across all loaded modules BEFORE
    // flipping the flag, so by the time kame.exe's `operator new` hands
    // out its first pool pointer, every `free()` in the process already
    // routes through the pool dispatcher.  Idempotent + kill-switch
    // guarded (KAME_POOL_WIN_REDIRECT=0).  Later-loaded DLLs are caught
    // by the LdrRegisterDllNotification hook installed here.
    kame_pool_win_install_redirect();
#endif
    g_sys_image_loaded = true;
}
#endif

template <unsigned int ALIGN, bool FS, bool DUMMY>
inline PoolAllocator<ALIGN, FS, DUMMY>::PoolAllocator(int count, char *addr) :
	PoolAllocatorBase(),
	m_flags(reinterpret_cast<FUINT *>( &addr[((sizeof(PoolAllocator) + sizeof(FUINT) - 1) / sizeof(FUINT)) * sizeof(FUINT)])),
	m_idx(0),
	m_count(count) {
	// BIT_OWNED set at construction — the chunk has an
	// owner (the thread doing the chunk-claim and adding to its DLL).
	// MASK_CNT init depends on path:
	//   * FS=true real chunk (`FS && DUMMY`, pre-fill below): every
	//     m_flags bit gets set to 1 and every slot is linked into the
	//     chunk-local freelist.  MASK_CNT = count (all words non-empty).
	//   * FS=false base ctor pass (`FS && !DUMMY`) or the GUARDIAN
	//     debug path: bits stay 0; MASK_CNT = 0; allocate_pooled bumps
	//     it on the first bitmap-CAS claim.
	// BIT_OWNED is cleared by release_dll_chunks_for_thread /
	// owner_release via atomicFetchAnd, which doubles as the release-
	// rights check (newv == 0 ⇒ I'm the unique releaser).
	m_flags_packed = BIT_OWNED;
	m_flags_filled_cnt = 0;
	// capture this thread's `s_tls.dll_head` TLS address.  Used
	// by dealloc cursor-reset paths to identify same-thread frees and
	// skip wasted resets on cross-thread frees.  Note: each (ALIGN, FS)
	// template has its OWN s_tls.dll_head (TLS variable), so the captured
	// address is comparable only to `&s_tls.dll_head` taken in the same
	// template context — which is exactly what the dealloc paths do.
	this->m_owner_dll_head_addr = (void *)&s_tls.dll_head;
	// also capture the owner's "force walk from head" flag
	// pointer.  Cross-thread frees flip this so the owner's next
	// allocate_chunk_path force-restarts the DLL walk and visits
	// revived chunks (bitmap-cleared by cross-thread frees since the
	// last walk).
	// atomic publish (relaxed — chunk not visible to other
	// threads yet; bitmap-claim CAS that publishes the chunk has a
	// release fence which carries this store).
	this->m_owner_dll_force_walk_ptr.store(
	    &s_tls.dll_force_walk_from_head, std::memory_order_relaxed);
	// Owner id + chunk-local freelists for the dealloc fast path.
	// `kame_owner_id()` is non-zero, so a foreign / never-allocated
	// thread's `s_tls_owner_id == 0` never matches.  All heads start
	// empty (no slot handed out yet).
	this->m_owner_id = kame_owner_id();
	// Cache-line-1 copies of the chunk_header fast-path discriminators
	// (follow-up "(1b)") so the dealloc owner-free path reads ONLY this
	// line, never chunk_header (cache line 0):
	//   m_fs_flag    = (FS && DUMMY): true ONLY for a real FS=true chunk
	//                  (`<ALIGN,true,true>`); the FS=false partial spec
	//                  constructs through the `<ALIGN,true,false>` base,
	//                  so DUMMY=false there yields m_fs_flag=false.
	//   m_base_bucket= the single bucket an FS=true chunk serves
	//                  (slot size == ALIGN); constexpr-folds.  Unused for
	//                  FS=false (its bucket comes from the slot prefix).
	this->m_fs_flag = (FS && DUMMY);
	this->m_base_bucket = (FS && DUMMY)
	    ? static_cast<uint16_t>(bucket_for_size(ALIGN)) : 0;
	// (§16) "full-usable" m_sizes mode: enabled for FS=false chunks with
	// ALIGN >= 1024.  The FS=false partial spec constructs through the
	// `<ALIGN,true,false>` base ctor, so `FS && !DUMMY` uniquely selects an
	// FS=false chunk here (real FS=true is `<ALIGN,true,true>` → DUMMY=true).
	// m_sizes points just past m_flags[count]; m_align_shift = log2(ALIGN).
	// Borrow-mode chunks (FS=true, or FS=false ALIGN<1024) leave m_sizes
	// null so the dealloc fast path keeps reading the p-8 prefix.
	if constexpr (FS && !DUMMY && ALIGN >= 1024u) {
		this->m_sizes = reinterpret_cast<uint16_t *>(
		    reinterpret_cast<char *>(m_flags)
		    + static_cast<size_t>(count) * sizeof(FUINT));
		this->m_align_shift =
		    static_cast<uint8_t>(__builtin_ctz(ALIGN));
	}
	else {
		this->m_sizes = nullptr;
		this->m_align_shift = 0;
	}
	// (§29) Pre-fill runtime opt-out — set `KAME_POOL_DISABLE_PREFILL=1` to
	// fall back to the pre-§29 zero-init code path without rebuilding.
	// Diagnostic switch only — if you're hitting a fresh-chunk corruption
	// only on the Windows MinGW path, toggle this to confirm whether the
	// pre-fill is involved.  Constructed once on first chunk claim,
	// process-lifetime; the call to `std::getenv` here recurses into our
	// own `operator new`-overridden path only when the env value is
	// already cached (Windows MSVCRT) or via thread-safe local storage
	// (glibc), neither of which routes through the pool's hot path.
	static const bool s_prefill_enabled = []{
		const char *e = std::getenv("KAME_POOL_DISABLE_PREFILL");
		return !(e && e[0] != '\0' && e[0] != '0');
	}();
	bool prefilled = false;
#ifndef GUARDIAN
	if constexpr (FS && DUMMY) if(s_prefill_enabled) {
		prefilled = true;
		// (§29) FS=true freelist pre-fill at chunk-claim.
		//
		// Non-atomically link ALL slots into the chunk-local freelist
		// (LIFO; slot's first 8 B = next pointer) and set every m_flags
		// bit to 1.  Effect on the alloc paths:
		//   * fresh-chunk first alloc (cold path through
		//     `allocate_chunk_path` → `allocate_pooled`): the
		//     freelist_pop check at the head of allocate_pooled (§29)
		//     pops slot 0 in O(1) — no bitmap-CAS find-zero scan.
		//   * steady-state hot path (`new_redirected`): pops directly
		//     from `g_thread_freelist_ptr[bucket] = &m_freelist_head[0]`,
		//     same as the previous design (which populated the freelist
		//     incrementally as slots were dealloc'd).
		//
		// Lifecycle invariants (INV-6, INV-7, INV-15..18) preserved:
		// every pre-filled slot's m_flags bit IS set, so "bit set ⇔ slot
		// live in user OR in owner freelist" holds throughout.  Cross-
		// thread frees CAS-clear bits and dec MASK_CNT on word-empty
		// exactly as before; owner-exit drain in
		// release_dll_chunks_for_thread feeds each remaining freelist
		// entry through batch_return_to_bitmap, which clears the bit and
		// decs MASK_CNT on word-empty by the same path.
		//
		// Side benefit (automatic prewarm): the per-slot 8-byte next-
		// pointer write page-faults every slot's first page, so the
		// first user access is hot.  Matters for the cold-start path
		// of alloc-heavy workloads.
		//
		// Owner-exclusive at this point — the chunk is not yet in the
		// thread's DLL nor stamped into the radix — so non-atomic stores
		// on m_flags, m_flags_packed, m_freelist_head[0], and slot
		// next-pointer fields are safe.
		//
		// FS=true serves a single bucket per (ALIGN) instantiation, so
		// every slot's local-id is 0 (see `kBucketLocalId[]`).
		constexpr unsigned FUINT_BITS_PF = sizeof(FUINT) * 8;
		const size_t N_slots =
		    static_cast<size_t>(count) * FUINT_BITS_PF;
		char *base = this->mempool();
#if KAME_FS_TWOLIST
		// (§two-list) Range-init replaces the link-prefill: the virgin
		// inventory is the address range [2]=cur .. [3]=end, bump-served
		// through the K-capped window [4] — no 8-byte next-pointer store
		// per slot (and no prewarm side effect), no [0] mega-list.  The
		// m_flags/_packed/_filled init below is IDENTICAL to the linked
		// prefill, so every lifecycle invariant (INV-6/7/15..18) is
		// unchanged: a reserve-range slot is "owner-held", exactly like a
		// freelist-held one.
		{
			constexpr size_t K_slots =
			    (2048u / ALIGN) < 4u ? 4u : (2048u / ALIGN);
			for(int b = 0; b < KAME_LOCAL_BUCKETS; ++b)
				m_freelist_head[b] = nullptr;
			m_freelist_head[2] = base;                       // reserve cur
			m_freelist_head[3] = base + N_slots * ALIGN;     // reserve end
			size_t w = K_slots < N_slots ? K_slots : N_slots;
			m_freelist_head[4] = base + w * ALIGN;           // window end
			// FS=true ALIGN is NOT always a power of two (16..368 step
			// 16), so the bump increment can't be recovered from
			// m_align_shift — park the byte stride in the (equally
			// unused) cell [5].
			m_freelist_head[5] =
			    reinterpret_cast<char *>(static_cast<uintptr_t>(ALIGN));
		}
#else
		char *prev = nullptr;
		for(size_t i = N_slots; i-- > 0; ) {
			char *slot = base + i * ALIGN;
			*reinterpret_cast<char **>(slot) = prev;
			prev = slot;
		}
		m_freelist_head[0] = base;
		for(int b = 1; b < KAME_LOCAL_BUCKETS; ++b)
			m_freelist_head[b] = nullptr;
#endif
		for(int i = count - 1; i >= 0; --i)
			m_flags[i] = ~(FUINT)0u;
		m_flags_packed = BIT_OWNED | static_cast<std::uint32_t>(count);
		m_flags_filled_cnt = count;
	}
#endif
	if( !prefilled) {
		// FS=false base ctor pass (DUMMY=false), GUARDIAN debug path,
		// or §29 pre-fill disabled via KAME_POOL_DISABLE_PREFILL: keep
		// zero-init.  FS=false slots are variable-size (N bits per slot),
		// so a uniform "one slot per bit" freelist pre-fill doesn't apply.
		for(int b = 0; b < KAME_LOCAL_BUCKETS; ++b)
			m_freelist_head[b] = nullptr;
		for(int i = count - 1; i >= 0; --i)
			m_flags[i] = 0;
	}
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
		reinterpret_cast<uint64_t *>(this->mempool())[i] = GUARDIAN; //filling
#endif
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
inline PoolAllocator<ALIGN, FS, DUMMY> *PoolAllocator<ALIGN, FS, DUMMY>::create(size_t size, char *ppool) {
	// (§15) Forward-shift embed layout:
	//
	//   [chunk_base + 0..63]            chunk_header (caller-written)
	//   [chunk_base + 64 = ppool]       PoolAllocator object (placement new)
	//   [ppool + size_alloc]            m_flags[count]
	//   [padding]                        unused bytes up to `chunk_base + K_MAX`
	//   [chunk_base + K_MAX]            slot region (m_mempool) — UNIT-ALIGNED
	//   [...]                            slots × ALIGN bytes each
	//   [chunk_base + chunk_size - K_MAX] last K_MAX bytes reserved for the
	//                                     NEXT chunk's metadata (if any).
	//
	// The fixed-position slot region (`ppool + K_MAX - 64`) makes the slot
	// region start at chunk_base + K_MAX = unit_boundary — every chunk's
	// slot 0 is 256 KiB-aligned regardless of template.  Trade-off: K_MAX
	// bytes per chunk are reserved for metadata even if the actual
	// metadata is smaller (FS=false / dedicated paths use far less).
	//
	// `size` from the caller is `chunk_size - ALLOC_CHUNK_HEADER`.  The
	// usable slot region is `chunk_size - K_MAX` bytes.
	constexpr size_t size_alloc =
	    ((sizeof(PoolAllocator) + sizeof(FUINT) - 1) / sizeof(FUINT))
	    * sizeof(FUINT);
	constexpr unsigned FUINT_BITS = sizeof(FUINT) * 8;
	// Metadata budget within the K_MAX reservation (chunk_header occupies
	// the first ALLOC_CHUNK_HEADER bytes; PoolAllocator + m_flags + pad
	// must fit in the remainder).
	constexpr size_t kMetaBudget = ALLOC_CHUNK_K_MAX - ALLOC_CHUNK_HEADER;
	static_assert(size_alloc <= kMetaBudget,
	    "PoolAllocator + alignment must fit in ALLOC_CHUNK_K_MAX - 64.  "
	    "Increase K_MAX or shrink the struct.");
	// Slot region size = chunk_size - K_MAX = (size + ALLOC_CHUNK_HEADER)
	// - K_MAX = size - (K_MAX - ALLOC_CHUNK_HEADER).  The slot region
	// POINTER (`ppool + (K_MAX - HEADER)`) is no longer materialised here
	// — `PoolAllocatorBase::mempool()` derives it from `this` at the use
	// sites instead.
	size_t slot_region_size = size - kMetaBudget;
	// Page-bound the slot region: the slot region starts at the 256 KiB unit
	// boundary (mempool() = chunk_base + K_MAX), which is OS-page-aligned, so
	// rounding its SIZE down to a whole number of pages makes it END on a page
	// boundary too.  This keeps every handed-out slot inside a page this chunk
	// can fully reclaim on release.  Targets whose page size exceeds K_MAX
	// (macOS arm64: 16 KiB) otherwise leave a partial final page that §15
	// SHARES with the NEXT chunk's header (each header is the 4 KiB below its
	// unit boundary); a slot there could not be MADV_FREE'd without zeroing
	// the neighbour's live header (the straddle crash fixed in deallocate_chunk).
	// No-op on Linux (PAGE == K_MAX == 4 KiB, and chunk_size − K_MAX is already
	// page-aligned); costs PAGE − K_MAX (= 12 KiB) of slots per chunk on macOS.
	slot_region_size &= ~((size_t)ALLOC_PAGE_SIZE - 1u);
	// count must satisfy:
	//   (i)  size_alloc + count*sizeof(FUINT) <= kMetaBudget
	//          (m_flags fits between PoolAllocator and slot_region)
	//   (ii) count * ALIGN * FUINT_BITS <= slot_region_size
	//          (slots fit in the slot region)
	int count_meta = static_cast<int>(
	    (kMetaBudget - size_alloc) / sizeof(FUINT));
	int count_slots = static_cast<int>(
	    slot_region_size / ((size_t)ALIGN * FUINT_BITS));
	int count = count_meta < count_slots ? count_meta : count_slots;
	return new(ppool) PoolAllocator(count, ppool);
}
template <unsigned int ALIGN, bool DUMMY>
inline PoolAllocator<ALIGN, false, DUMMY>::PoolAllocator(int count, char *addr) :
	PoolAllocator<ALIGN, true, false>(count, addr) {
	// m_sizes and m_available_bits are gone.  Per-slot SIZE
	// is stored in the slot's own first ALIGN bytes (the "+1 prefix"
	// — see allocate_pooled below).  Nothing further to initialise
	// here: the base ctor zero-clears m_flags, and the prefix bytes
	// for each slot are written at allocate-time before the bitmap CAS
	// publishes the slot's ownership to other threads.
}
template <unsigned int ALIGN, bool DUMMY>
inline PoolAllocator<ALIGN, false, DUMMY> *PoolAllocator<ALIGN, false, DUMMY>::create(size_t size, char *ppool) {
	// (§15) Forward-shift embed layout — see the FS=true `create` above
	// for the full layout doc.  FS=false has a slightly different
	// PoolAllocator size; otherwise the same kMetaBudget / dual-count
	// constraint applies.
	constexpr size_t size_alloc =
	    ((sizeof(PoolAllocator) + sizeof(FUINT) - 1) / sizeof(FUINT))
	    * sizeof(FUINT);
	constexpr unsigned FUINT_BITS = sizeof(FUINT) * 8;
	constexpr size_t kMetaBudget = ALLOC_CHUNK_K_MAX - ALLOC_CHUNK_HEADER;
	static_assert(size_alloc <= kMetaBudget,
	    "PoolAllocator (FS=false) + alignment must fit in "
	    "ALLOC_CHUNK_K_MAX - 64");
	// Slot region size = size - (K_MAX - HEADER); pointer is derived from
	// `this` via `PoolAllocatorBase::mempool()` at use sites.
	size_t slot_region_size = size - kMetaBudget;
	// Page-bound the slot region — see the FS=true create() above for the
	// rationale (slots must never land in the chunk's final page, which on
	// PAGE > K_MAX targets is shared with the next chunk's header).  No-op on
	// Linux; −12 KiB of slots/chunk on macOS arm64.
	slot_region_size &= ~((size_t)ALLOC_PAGE_SIZE - 1u);
	// (§16) Per-word metadata cost.  Borrow mode: just the FUINT bitmap
	// word (8 B / 64 slots).  Full-usable mode (ALIGN>=1024): additionally
	// `m_sizes[64]` = 64 × uint16 = 128 B per bitmap word, packed right
	// after m_flags.  The count must satisfy both the metadata-fit and the
	// slot-region-fit; for the ALIGN>=1024 tiers count is slot-limited (the
	// 256 KiB / 1 MiB chunk holds few large slots), so the extra m_sizes
	// term never actually reduces count — but the budget math stays honest.
	constexpr size_t per_word_meta =
	    (ALIGN >= 1024u)
	        ? (sizeof(FUINT) + (size_t)FUINT_BITS * sizeof(uint16_t))
	        : sizeof(FUINT);
	int count_meta = static_cast<int>(
	    (kMetaBudget - size_alloc) / per_word_meta);
	int count_slots = static_cast<int>(
	    slot_region_size / ((size_t)ALIGN * FUINT_BITS));
	int count = count_meta < count_slots ? count_meta : count_slots;
	return new(ppool) PoolAllocator(count, ppool);
}
template <unsigned int ALIGN, bool FS, bool DUMMY>
inline void PoolAllocator<ALIGN, FS, DUMMY>::operator delete(void *p) throw() {
	free(p);
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
inline void *
PoolAllocator<ALIGN, FS, DUMMY>::allocate_pooled(unsigned int SIZE) {
#ifndef GUARDIAN
	if constexpr (FS && DUMMY) {
		// (§29) Freelist hit — pre-filled slots from chunk-claim, or
		// slots that came back to the chunk-local freelist via
		// owner-side `deallocate_pooled`.  Mainline FS=true cold-path:
		// fresh-chunk first alloc goes through here and pops slot 0 of
		// the pre-filled freelist; subsequent allocs on the same bucket
		// go through `new_redirected`'s freelist fast path which reads
		// `*g_thread_freelist_ptr[bucket]` directly.
		//
		// `freelist_pop` is non-atomic — only the owner thread touches
		// `m_freelist_head[0]`, and the cold path is itself owner-only
		// (allocate_pooled is reached via allocate_chunk_path /
		// slow_allocate, both single-threaded for this chunk).
		// Returns nullptr if the freelist is empty; we then fall through
		// to the existing bitmap-CAS scan, which finds bits cleared by
		// cross-thread frees (the only mechanism that opens bits with
		// the pre-fill design — slots returned to the OWNER stay in
		// `m_freelist_head[0]`, not the bitmap).
#if KAME_FS_TWOLIST
		// (§two-list) fresh-chunk / cold entry: alloc side first ([1]
		// segment, then refill = swap [0] / bump window).  The bitmap
		// scan below still serves cross-thread-cleared bits.
		if(void *p = this->freelist_pop(1))
			return p;
		if(void *p = this->fs_twolist_refill_take())
			return p;
#endif
		if(void *p = this->freelist_pop(0))
			return p;
	}
#endif
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
			// which the TLS s_tls.my_chunk fast path in allocate() removes —
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
		// (§94pct-gate) FS=true: bail when ≥ 93.75 % (= 1 − 1/16) of
		// words are already 100 % filled.  Reached only after the walk
		// hit a fully-filled word — at that point we already KNOW the
		// chunk is saturated enough that adjacent words are likely also
		// full.  Bailing here hands the alloc off to a less-saturated
		// chunk in the DLL (or a fresh mmap) instead of walking the
		// remaining 6 % of sparse-free-bit words.  Accepted trade-off:
		// ~6 % slot under-utilisation in exchange for cutting the
		// worst-case walk.  The check stays inside the loop (not at the
		// entry) so bench_loop's tight FS=true 64 hot path pays nothing.
		if(this->m_flags_filled_cnt >=
		   this->m_count - this->m_count / 16)
			return 0;
		idx++;
		if(idx == this->m_count) {
			idx = 0;
		}
	}

	int sidx = count_zeros_forward(one);

	this->m_idx = idx;

	void *p = &this->mempool()[(idx * sizeof(FUINT) * 8 + sidx) * ALIGN];
	return p;
}

template <unsigned int ALIGN, bool DUMMY>
inline void *
PoolAllocator<ALIGN, false, DUMMY>::allocate_pooled(unsigned int SIZE) {
	// Owner-side freelist hit is handled in `new_redirected` via the
	// per-thread per-bucket freelist — by the time we reach
	// `allocate_pooled` the freelist has missed.  This path
	// runs the bitmap CAS to claim N contiguous free bits (	// "borrow scheme" — the per-slot `{uint32_t bucket, uint32_t SIZE}`
	// header lives in the LAST 8 bytes of the PREVIOUS slot's ALIGN
	// area, or in `chunk_header[56..63]` for slot 0 at bit 0/word 0.
	// No separate "prefix bit" is claimed.  See allocator_prv.h's
	// chunk-header layout doc for the formal reservation.
	//
	// User pointer p = slot_start (ALIGN-aligned ✓).
	// Header at `p - 8`:
	//   * For slot at bit 0 of word 0: `mempool - 8` = `chunk_base +
	//     ALLOC_CHUNK_HEADER - 8` = `chunk_header[56..63]` —
	//     formally reserved by an earlier change (static_assert in
	//     allocator_prv.h confirms ≥ 8 B of pad before this region).
	//   * For slot at bit B > 0: byte position `B*ALIGN - 8` = LAST
	//     8 bytes of bit (B-1)'s ALIGN area.  Universal invariant:
	//     every allocated slot reserves its OWN last 8 bytes as
	//     storage for the next slot's header, so `user_area =
	//     N*ALIGN - 8 bytes` and the reservation is never trampled
	//     by user writes.
	//
	// (§16) Bit count N per slot.
	//   * Borrow mode (ALIGN<1024): N = ceil((SIZE+8)/ALIGN) — the slot's
	//     last 8 B are borrowed for the next slot's {local_id,SIZE} prefix,
	//     so the usable area is N*ALIGN-8.
	//   * Full-usable mode (ALIGN>=1024): N = ceil(SIZE/ALIGN) — no borrow,
	//     usable = N*ALIGN.  For the page-aligned bucket schedule
	//     (SIZE = N*ALIGN exactly) this is N, one fewer unit than borrow,
	//     eliminating the 50 % round-up at power-of-2 page sizes.
	const unsigned int N = (ALIGN >= 1024u)
	    ? ((SIZE + ALIGN - 1u) / ALIGN)
	    : ((SIZE + 8u + ALIGN - 1u) / ALIGN);
	// (§12.3) Per-chunk COMPACT local-id (kBucketLocalId resolved here on
	// the cold alloc path, so the dealloc fast path needs no bucket_for_size
	// / remap).  Borrow mode stores { uint32 local_id, uint32 SIZE } at
	// slot_start-8; full-usable mode stores (N<<8)|local_id in m_sizes[bit].
	//
	// (§17) For the FULL tier (ALIGN>=1024) we derive local_id directly
	// from N, not via `kBucketLocalId[bucket_for_size(SIZE)]`.  Reason:
	// bucket 49 (ALIGN=4096 N=2) and bucket 42 (ALIGN=1024 N=8) share the
	// same slot size (8192), so `bucket_for_size(8192)` is ambiguous — it
	// resolves to 42 (the denser plain-malloc route).  An ALIGN=4096 chunk
	// allocating a bucket-49 slot under that lookup would tag the slot
	// with bucket-42's local_id (2 instead of 1), routing its free to the
	// wrong chunk-local freelist.  Direct (ALIGN,N)→local_id avoids the
	// ambiguity and stays robust if more bucket entries land on the same
	// slot size in future.  Borrow tier's SIZE-to-bucket lookup is
	// bijective (user_max = N*ALIGN-8 is unique per (ALIGN,N)), so it
	// still uses the table.
	std::uint32_t local_id;
	std::uint32_t bucket_for_prefix = 0;
	if constexpr (ALIGN >= 1024u) {
		if constexpr (ALIGN == 1024u) {
			// N ∈ {6,7,8,9,11,13,15,17} → 0..7 (buckets 40..47).
			local_id = (N <= 9u) ? (N - 6u) : (4u + (N - 11u) / 2u);
		}
		else /* ALIGN == 4096 */ {
			// N ∈ {1,2,4,8} (power-of-2 page tier) → ctz(N) (buckets 48..51).
			local_id = static_cast<std::uint32_t>(__builtin_ctz(N));
		}
	}
	else {
		// (FS=false borrow) Also stash the bucket index in the slot prefix's
		// upper-low-32 bits so the dealloc fast path can re-aim
		// `g_thread_freelist_ptr[bucket]` at this chunk's freelist head
		// without a `bucket_for_size` lookup (12 % Ir on FS=false churn).
		// One `bucket_for_size` call here, on the cold alloc path.
		bucket_for_prefix = bucket_for_size(SIZE);
		local_id = kBucketLocalId[bucket_for_prefix];
	}
	// Slot prefix layout (FS=false borrow): bits 0..7 = local_id (max 8),
	// bits 16..23 = bucket (max 51), bits 32..63 = SIZE.  Bucket lives in
	// the upper byte of the low-32 half so the dealloc fast path can
	// extract it from the same `hdr` load already done for local_id.  For
	// full-usable mode (ALIGN>=1024) `bucket_for_prefix` stays 0 — the
	// fast-path freelist-follow currently skips that branch.
	const std::uint64_t hdr_word =
	    static_cast<std::uint64_t>(local_id)
	  | (static_cast<std::uint64_t>(bucket_for_prefix) << 16)
	  | (static_cast<std::uint64_t>(SIZE) << 32);
	const std::uint16_t msize_word = static_cast<std::uint16_t>(
	    (N << 8) | (local_id & 0xFFu));
	// dropped the an earlier change 80% fragmentation cutoff — it
	// walked all m_count FUINT words via count_bits on every
	// allocate_pooled call (catastrophic on high-level chunks where
	// m_count ≈ 4096 → ~4 µs/alloc).  The walk-once-and-bail logic
	// below is also bounded by m_count and only pays that cost when
	// the chunk is truly out of N-contiguous-zero runs (rare; only
	// at chunk-fill boundary).
	//
	// (§max-n-gate) m_flags_filled_cnt for FS=false counts words that
	// can no longer host a MAX_N-contiguous-zero run (see the CAS
	// success block below).  Only safe to bail here for max-N requests
	// — a smaller-N request may still find room in those words and
	// must walk normally.
	constexpr unsigned int MAX_N_HERE =
	    PoolAllocator<ALIGN, false, DUMMY>::MAX_N;
	if(this->m_flags_filled_cnt == this->m_count && N >= MAX_N_HERE)
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
			size_t bit_index = size_t(idx_cand) * sizeof(FUINT) * 8 + sidx;
			slot_start = &this->mempool()[bit_index * ALIGN];
			// Write the per-slot metadata BEFORE the CAS publishes the bit
			// (the CAS is the release that publishes it to a freeing
			// thread; the freer's acquire on the bitmap word synchronises
			// with it).
			if constexpr (ALIGN >= 1024u) {
				// (§16) Full-usable mode: store (N<<8)|local_id in the
				// chunk-header m_sizes[] array indexed by slot start bit.
				// The slot's own bytes are 100 % user-usable — no borrow.
				this->m_sizes[bit_index] = msize_word;
				(void)hdr_word;
			}
			else {
				// Borrow scheme — header at `slot_start - 8`:
				//   * Bit 0 of word 0 (slot_start == mempool()):
				//       slot_start - 8 = chunk_base + K_MAX - 8, the last
				//       8 B of the metadata region (reserved tail after
				//       m_flags[]; §15-shifted home — was chunk_base + 56
				//       pre-shift, when mempool sat at +ALLOC_CHUNK_HEADER).
				//   * Bit B > 0: slot_start - 8 lands in bit (B-1)'s last 8
				//       bytes (universal reservation invariant — the prior
				//       slot's user_area excludes its own last 8 B).
				*reinterpret_cast<std::uint64_t *>(slot_start - 8) = hdr_word;
				(void)msize_word;
			}
			// Always-CAS path (cf. FS=true sibling): TLS s_tls.my_chunk
			// fast path removes the bit0-lock around chunk access, so
			// a non-atomic store would torn-write under contention.
			FUINT newv = oldv | ones; //filling with N ones (all user bits).
			if(atomicCompareAndSet(oldv, newv, pflag)) {
				if(oldv == 0)
					atomicInc( &this->m_flags_packed);
				// (§max-n-gate) Widened m_flags_filled_cnt for FS=false:
				// count words that can no longer host a MAX_N-contiguous-
				// zero run (not just words at 0xFF...F).  Smaller-N
				// requests still walk fine; max-N requests get an O(1)
				// bail at the early-exit above when every word is in this
				// state.  Test the transition oldv-had-room → newv-no-room
				// so a CAS landing inside an already-saturated word
				// doesn't double-count.
				constexpr unsigned int MAX_N_HERE =
				    PoolAllocator<ALIGN, false, DUMMY>::MAX_N;
				if(find_training_zeros(MAX_N_HERE, newv) == 0 &&
				   find_training_zeros(MAX_N_HERE, oldv) != 0)
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
	// (§hot-tls teardown) COLD path only.  If THIS thread has run its
	// allocator cleanup, the fast-TSD page is the static teardown sentinel
	// (pure pointer compare — no `_tlv_get_addr`, no `g_tls_page` deref).
	// In that state `s_tls.my_chunk` (1540) and `&s_tls.dll_head` (the
	// cursor-reset below) may be torn down; route the slot straight to the
	// bitmap and return, touching no thread-local.
	if(__builtin_expect(kame_page() == &g_teardown_page, 0)) {
		CrossDeallocEntry tmp[2] = {{this, p}, {nullptr, nullptr}};
		this->batch_return_to_bitmap(tmp);
		return false;
	}
	// (§12.3) Per-slot prefix { uint32_t local_id, uint32_t SIZE } at
	// `p - 8` (LAST 8 bytes of the prefix bit's ALIGN area).  One 64-bit
	// load recovers the local-id directly — no bucket_for_size, no
	// kBucketLocalId remap on the hot path.
	//
	// Owner-side dealloc: push to this chunk's
	// `m_freelist_head[local_id]` — exactly the cell `new_redirected`
	// pops from via `g_thread_freelist_ptr[bucket]` on the next
	// allocation of that size.  Non-owner routes to the cross-thread
	// path which uses slot_start for batch dispatch.
	//
	// `s_tls.my_chunk` has declared type `PoolAllocator<ALIGN, false, false>*`
	// (from the base's `PoolAllocator<ALIGN, DUMMY, DUMMY>*` with
	// DUMMY=false), while `this` has type `PoolAllocator<ALIGN, false,
	// DUMMY>*` — different template instantiations referring to the
	// same chunk object.  Compare as void* to bypass the type mismatch.
	if(static_cast<void *>(PoolAllocator<ALIGN, true, false>::s_tls.my_chunk)
	    == static_cast<void *>(this)) {
		// (§16) local-id source: full-usable mode reads m_sizes[bit],
		// borrow mode reads the p-8 prefix.
		unsigned local;
		if constexpr (ALIGN >= 1024u) {
			size_t bit_index = static_cast<size_t>(p - this->mempool()) >> this->m_align_shift;
			local = this->m_sizes[bit_index] & 0xFFu;
		}
		else {
			std::uint64_t hdr = *reinterpret_cast<std::uint64_t *>(p - 8);
			// Mask: prefix's low byte holds local_id; bits 16..23 now hold
			// bucket (an earlier change introduced for the dealloc-fast-path
			// freelist-follow).  Without the mask `local` would include
			// the bucket bits and fail the KAME_LOCAL_BUCKETS bound check.
			local = static_cast<unsigned>(hdr) & 0xFFu;
		}
		// Defensive bound check (local-id must be valid; misroute on
		// stale data is detected and falls through to bitmap path).
		if(local < (unsigned)KAME_LOCAL_BUCKETS) {
			this->freelist_push(local, p);
			return false;
		}
	}
	// (§35) Orphan adoption is DISABLED on the dealloc path.  It was added
	// (§orphan-adopt) so a thread-respawn workload (larson: same thread
	// allocs+frees) could immediately reuse a freed orphan slot via its own
	// DLL/freelist.  But when the FREEING thread is a long-lived, non-
	// allocating, non-exiting consumer (e.g. KAME's main thread holding the
	// STM tree, while a per-run driver thread did the allocation), adoption
	// parks the freed slot on the consumer's freelist with the bitmap bit
	// still SET — never drained (no alloc, no thread-exit) → the now-empty
	// chunk is never recognised as empty, never released, and being owned
	// (BIT_OWNED) is no longer an orphan the next allocator can claim either.
	// Result: unbounded region growth across start/stop (see
	// tests/alloc_thread_churn_test.cpp scenario 2).
	//
	// Letting the free fall through to `batch_return_to_bitmap` below
	// instead CLEARS the bit and, when it brings the chunk to empty with
	// BIT_OWNED clear, RELEASES it (warm-recycled via §34 bucket_release_
	// chunk).  The only cost adoption used to avoid — a non-empty orphan's
	// freed slots being unreachable until the chunk fully drains and
	// recycles — is now cheap because that recycle path (§34/LRC) exists.
	// So: prefer guaranteed release over speculative reuse.
	//
	// (§hot-tls teardown) Thread-exit frees never reach here: the
	// teardown-sentinel check at the TOP of this function already routed
	// them to a TLS-free bitmap return.  So reaching this point implies the
	// freeing thread is alive and the `&s_tls.dll_head` cursor compare below
	// is safe.
	// FS=false never participates in cross-thread batch
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
	// (§20) Cache dll-cursor-reset addresses BEFORE batch_return_to_bitmap.
	// If this is the chunk's last live slot AND BIT_OWNED is clear
	// (owner exited), batch_return releases `this`: the placement-new
	// destructor runs and `this` becomes a stale pointer — accessing
	// `this->m_owner_dll_head_addr` / `this->m_owner_dll_force_walk_ptr`
	// afterwards is UB (UBSAN's vptr check fires).  The fields are
	// write-once at chunk construction so the cached values stay valid
	// across the call; the force-walk pointer's target lives in the
	// OWNER thread's TLS (independent of this chunk's lifetime).
	void *cached_dll_head_addr = this->m_owner_dll_head_addr;
	auto *cached_force_walk =
	    this->m_owner_dll_force_walk_ptr.load(std::memory_order_acquire);
	CrossDeallocEntry tmp[2] = {{this, p}, {nullptr, nullptr}};
	this->batch_return_to_bitmap(tmp);
	// (§20) `this` may be destructed past this point — use cached values
	// only.  Direct `batch_return_to_bitmap` cleared 1 bit on `this` chunk
	// → it may now have space for a future `allocate_pooled` (if not
	// released).  Two cases:
	//
	//   * Same-thread (we are the chunk's owner):
	//     cached_dll_head_addr == &s_tls.dll_head.  Reset OUR cursor
	//     directly — next allocate_chunk_path walks our DLL from
	//     head and finds the revival.
	//
	//   * Cross-thread (owner is some other thread):
	//     Bump the OWNER thread's "force walk from head" hint flag.
	//     The flag's storage lives in owner TLS (independent of this
	//     chunk); cached_force_walk is null after owner-exit's
	//     release-store, so we skip the deref.
	//
	// memory_order_relaxed on the store: hint flag, one-cycle false-
	// negative delay acceptable.
	if(cached_dll_head_addr ==
	   static_cast<void *>(&PoolAllocator<ALIGN, true, false>::s_tls.dll_head))
		PoolAllocator<ALIGN, true, false>::reset_dll_walk_state();
	else if(cached_force_walk)
		cached_force_walk->store(true, std::memory_order_relaxed);
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

// FS=false slot-size trampoline.  Used by `realloc()` /
// `malloc_usable_size` via `PoolAllocatorBase::size_of()`.
//   * Borrow mode (ALIGN<1024): SIZE = high 32 bits of the {local_id,SIZE}
//     header at `p - 8` (= the user-requested max the bucket serves).
//   * (§16) Full-usable mode (ALIGN>=1024): the slot's full N*ALIGN bytes
//     are usable; recover N from m_sizes[bit] and return N*ALIGN — the
//     true capacity (lets realloc grow in place across the full slot).
template <unsigned int ALIGN, bool DUMMY>
std::size_t
PoolAllocator<ALIGN, false, DUMMY>::size_of_static(
    PoolAllocatorBase *base, char *p) noexcept {
	if constexpr (ALIGN >= 1024u) {
		PoolAllocator *self = static_cast<PoolAllocator *>(base);
		size_t bit_index = static_cast<size_t>(p - self->mempool()) >> self->m_align_shift;
		unsigned N = static_cast<unsigned>(self->m_sizes[bit_index] >> 8);
		return static_cast<std::size_t>(N) * ALIGN;
	}
	else {
		(void)base;
		return static_cast<std::size_t>(
		    *reinterpret_cast<std::uint32_t *>(p - 4));
	}
}

// FS=false batch return — N-bit clear
// where N = ceil((SIZE + 8) / ALIGN).  Caller passes `p` = p_user
// (= slot_start in the borrow scheme).  The `{bucket, SIZE}` header
// lives at `p - 8` (= chunk_header pad for slot 0, or previous slot's
// reserved last-8 bytes otherwise).  Reuses the inherited
// batch_clear_impl skeleton with a borrow-scheme MaskFn and FS=false-
// specific OnClearFn (no filled_cnt; m_available_bits is gone since
// the earlier change's fragmentation cutoff in allocate_pooled replaces it).
template <unsigned int ALIGN, bool DUMMY>
int
PoolAllocator<ALIGN, false, DUMMY>::batch_return_to_bitmap(
    const CrossDeallocEntry *entries) noexcept {
	// Walk entries[k] while .chunk == this — terminates on the next
	// chunk's group OR the trailing {nullptr, nullptr} sentinel that
	// `CrossDeallocBatch::flush` plants at buf[count].  No `k < n_max`
	// test in the inner loop.  Drain / post-teardown single-slot paths
	// pass a stack-local {this, p_user} + sentinel pair.
	int n = this->batch_clear_impl(entries,
		// MaskFn: FS=false N-bit clear.  `p` is p_user (= slot_start);
		// `sidx` is the slot's own bit position within m_flags word `idx`.
		//   * Borrow mode (ALIGN<1024): N = ceil((SIZE+8)/ALIGN), SIZE from
		//     the {local_id,SIZE} header at p-8 (upper 32 bits at p-4).
		//   * (§16) Full-usable mode (ALIGN>=1024): N = m_sizes[bit] >> 8,
		//     where bit = idx*FUINT_BITS + sidx — no borrow header.
		[this](int idx, unsigned sidx, char *p) -> FUINT {
			unsigned N;
			if constexpr (ALIGN >= 1024u) {
				size_t bit_index =
				    static_cast<size_t>(idx) * (sizeof(FUINT) * 8u) + sidx;
				N = static_cast<unsigned>(this->m_sizes[bit_index] >> 8);
			}
			else {
				unsigned size_bytes = *reinterpret_cast<std::uint32_t *>(
				    p - 4);
				N = (size_bytes + 8u + ALIGN - 1u) / ALIGN;
			}
			FUINT slot_mask = (((FUINT(1) << N) - FUINT(1))) << sidx;
#ifdef GUARDIAN
			for(unsigned int j = 0;
			    j < N * ALIGN / sizeof(uint64_t); ++j)
				reinterpret_cast<uint64_t *>(p)[j] = GUARDIAN;
#endif
			return slot_mask;
		},
		// OnClearFn: FS=false.  Decrement MASK_CNT via atomicDecAndTest
		// when this slot's word goes 1 → 0 (maintains the live-word count
		// and supplies the full barrier).  We DO NOT release the chunk on the
		// dec-to-0-with-BIT_OWNED-clear case: such a chunk is an ORPHAN on the
		// atomic_shared_ptr chain (its chain-ref keeps it mapped), reclaimed by
		// orphan_chain_scrub (unlink → refcnt 0 → dispose) once drained — not
		// freed here.  Distinguished from the owner-exit empty release (separate
		// path in release_dll_chunks_for_thread, where BIT_OWNED is still SET
		// during the drain) and from the owner-alive case (BIT_OWNED set ⇒ dec
		// never reaches 0).  The return value is intentionally ignored.
		[this](FUINT oldv, FUINT newv) {
			if(newv == 0 && oldv != 0)
				(void)atomicDecAndTest(&this->m_flags_packed);
			// (§max-n-gate) Symmetric with the widened atomicInc in
			// allocate_pooled — decrement when a bit-clear restores
			// MAX_N-contiguous-zero room that the word didn't have
			// before.
			constexpr unsigned int MAX_N_HERE =
			    PoolAllocator<ALIGN, false, DUMMY>::MAX_N;
			if(find_training_zeros(MAX_N_HERE, oldv) == 0 &&
			   find_training_zeros(MAX_N_HERE, newv) != 0)
				atomicDec( &this->m_flags_filled_cnt);
		});
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
		int midx = (p - this->mempool()) / ALIGN;
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
			int midx_q = (q - this->mempool()) / ALIGN;
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
	// FS=false — an earlier change bypasses cross-batch entirely on the
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
// `CrossDeallocBatch::push` issues `&one, 1`, the per-chunk owner-exit
// drain `release_dll_chunks_for_thread` issues each chunk's freelist as
// one group, and the post-teardown bypass in `deallocate_pooled` issues
// `&one, 1`).  Single-chunk invariant lets us share one direct-map
// scratch.  Sole remaining consumer of `batch_clear_impl`
// (`drain_thread_slot_freelists` is now a retained no-op — see its
// definition).
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
		// OnClearFn: FS=true.  Decrement MASK_CNT via atomicDecAndTest
		// when this word goes 1 → 0.  The dec-to-0-with-BIT_OWNED-clear
		// case (a cross-thread free emptying an orphaned chunk) does NOT
		// release the chunk here: the orphan is on the atomic_shared_ptr
		// chain (chain-ref keeps it mapped), and orphan_chain_scrub reclaims
		// it (unlink → refcnt 0 → dispose) once drained.  Cross-free touches
		// MASK_CNT only — the dec return value is intentionally ignored, so
		// it never serialises against / races the chain reclaim.
		[this](FUINT oldv, FUINT newv) {
			if(oldv == ~(FUINT)0u)
				atomicDec( &this->m_flags_filled_cnt);
			if(newv == 0 && oldv != 0)
				(void)atomicDecAndTest(&this->m_flags_packed);
		});
	return n;
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
void
PoolAllocator<ALIGN, FS, DUMMY>::clear_owner_tls() noexcept {
	s_tls.my_chunk = nullptr;
}

// (§orphan-adopt) Claim an orphaned chunk (m_owner_id==0, BIT_OWNED==0)
// into this thread's DLL so the freed slot becomes reachable via
// scan_dll_freelist / Phase-2 allocate_pooled rather than silently
// accumulating in the bitmap of a chunk no thread can walk.
//
// Called from deallocate_pooled when s_tls.my_chunk != this AND
// m_owner_id == 0.  Thread-respawn workloads (e.g. larson) otherwise
// route all frees of the exited thread's objects through the cross-
// thread batch → bitmap path, leaving freed slots unreachable until
// the chunk is fully drained and recycled.
template <unsigned int ALIGN, bool FS, bool DUMMY>
bool
PoolAllocator<ALIGN, FS, DUMMY>::try_adopt_orphan(char *p, unsigned local) noexcept {
	// Need a valid (non-zero) owner-id so the fast-path check
	// `m_owner_id == kame_page()->owner_id` can fire after adoption.
	if(__builtin_expect(kame_page()->owner_id == 0, 0)) return false;
	// CAS m_owner_id: 0 → kame_page()->owner_id.  This CAS is the SOLE
	// arbitration point — only one thread can flip m_owner_id away from
	// 0, so concurrent adoption attempts are mutually exclusive and the
	// winner alone proceeds to set BIT_OWNED / wire the DLL below.  (The
	// CAS must come first for exactly this reason: setting BIT_OWNED
	// before knowing we won would leave a chunk with BIT_OWNED set by us
	// but m_owner_id claimed by another thread.)
	if(!atomicCompareAndSet((uint32_t)0u, kame_page()->owner_id,
	                        &this->m_owner_id))
		return false;
	// Set BIT_OWNED to mark the chunk as held by a live owner (this
	// thread) now that it joins our DLL.  Note the window between the CAS
	// above and this store is NOT protected by BIT_OWNED — it is
	// protected by p's still-set bitmap bit: p is the slot we are about
	// to free but have not yet returned, so its m_flags word is nonzero
	// and MASK_CNT ≥ 1, meaning no cross-thread dec-to-zero can release
	// the chunk regardless of BIT_OWNED.  p's protection lasts until the
	// slot is eventually drained back to the bitmap, by which time
	// BIT_OWNED has long been set — so the two protections chain with no
	// gap.  BIT_OWNED's real job is to prevent a premature cross-thread
	// release once the chunk lives in our DLL and is being walked.
	atomicFetchOr(&this->m_flags_packed, BIT_OWNED);
	// Wire force-walk pointer (cross-thread frees will now signal us).
	this->m_owner_dll_head_addr = &s_tls.dll_head;
	this->m_owner_dll_force_walk_ptr.store(
	    &s_tls.dll_force_walk_from_head, std::memory_order_release);
	// Append to this thread's DLL tail.
	auto *dll_self = static_cast<PoolAllocator<ALIGN, DUMMY, DUMMY>*>(this);
	dll_self->m_dll_prev = s_tls.dll_tail;
	dll_self->m_dll_next = nullptr;
	if(s_tls.dll_tail)
		s_tls.dll_tail->m_dll_next = dll_self;
	else
		s_tls.dll_head = dll_self;
	s_tls.dll_tail = dll_self;
	s_tls.dll_exhausted = false;
	// Ensure thread-exit cleanup walks this template's DLL (deduped).
	tls_alloc_thread_exit_cleanup.add(
	    &PoolAllocator<ALIGN, FS, DUMMY>::release_dll_chunks_for_thread);
	// Push freed slot to freelist — no atomics, we own the chunk.
	this->freelist_push(local, p);
	return true;
}

template <unsigned int ALIGN, bool FS, bool DUMMY>
bool
PoolAllocator<ALIGN, FS, DUMMY>::deallocate_pooled(char *p) {
	// (§hot-tls teardown) This is the COLD path — `deallocate`'s hot
	// owner-match returns before invoking the trampoline, so nothing here
	// affects the hot path.  If THIS thread has already run its allocator
	// cleanup, its fast-TSD page is the static teardown sentinel; detect it
	// with a pure pointer compare (NO `_tlv_get_addr`, NO deref of
	// `g_tls_page`'s possibly-finalized TLV storage).  In that state `s_tls`
	// and `&s_tls.dll_head` may be torn down, so we must touch NO thread-local:
	// route the single slot straight to the bitmap (TLS-free, scratch +
	// sentinel) and return.  Subsumes the former `s_alloc_tls_off` bypass.
	if(__builtin_expect(kame_page() == &g_teardown_page, 0)) {
		CrossDeallocEntry tmp[2] = {{this, p}, {nullptr, nullptr}};
		this->batch_return_to_bitmap(tmp);
		return false;
	}
	// Two-way dispatch:
	//
	//   owner               → push to per-thread AllocSlot freelist (no atomic)
	//   non-owner           → TLS cross-dealloc batch (batched bitmap CAS
	//                          per m_flags word at flush time)
	//
	// Owner check: per-template `s_tls.my_chunk` TLS only.  A former
	// secondary per-bucket "current chunk == this" check was dropped as
	// redundant — it always agreed with `s_tls.my_chunk` by construction —
	// saving one TLS read on every owner-side dealloc.
	if(static_cast<PoolAllocatorBase *>(s_tls.my_chunk) == this) {
		// Slot stays "allocated" in the bitmap until flushed back via
		// AllocThreadExitCleanup (thread exit) or the chunk's bitmap is
		// directly returned to (allocate_pooled goes there on freelist
		// miss).  An FS=true chunk serves exactly one size, so its
		// freelist is local-id 0 (§12.3); the owner's next alloc on this
		// bucket pops it back immediately via the TLS shortcut at
		// `g_thread_freelist_ptr[bucket]` -> `m_freelist_head[0]`.
		this->freelist_push(0, p);
		return false;
	}
	// (§35) Orphan adoption DISABLED here — see the FS=false sibling in the
	// other deallocate_pooled for the full rationale.  A non-allocating,
	// non-exiting consumer thread (KAME main) would otherwise strand the
	// adopted chunk (freed slot parked freelist-bit-set, never drained →
	// never released).  Falling through to the hold-and-batch path below
	// routes the free to batch_return_to_bitmap, which releases the chunk
	// (warm-recycled) once its last live slot is returned.
	//
	// (§hot-tls teardown) The former `if(s_alloc_tls_off)` post-teardown
	// bypass that lived here is gone: the teardown-sentinel check at the TOP
	// of this function already routed thread-exit frees to a TLS-free bitmap
	// return (it also subsumed that branch's `&s_tls.dll_head` cursor reset,
	// which is moot for a dying thread).  Reaching this point therefore
	// implies the freeing thread is alive, so the `tls_cross_dealloc_batch`
	// touch below is safe.
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
// path through this chunk's vtable.  ALIGN comes from the template
// instantiation (compile-time).  FS=true buckets are single-size
// (ALIGN == slot size), so `SIZE = ALIGN`; `bucket` selects the
// per-bucket fast-path freelist pointer to repoint at the newly-claimed
// chunk.
template <unsigned int ALIGN, bool FS, bool DUMMY>
__attribute__((cold, noinline))
void *
PoolAllocator<ALIGN, FS, DUMMY>::slow_allocate(unsigned bucket,
                                               std::size_t /*size*/) noexcept {
	// (§24) Before falling through to bitmap-claim, scan this thread's DLL
	// for any OTHER chunk holding freelist entries at the same local id.
	// `g_thread_freelist_ptr[bucket]` tracks only the most-recently-pinned
	// chunk's freelist[local]; when a workload's working set spans multiple
	// chunks (e.g. fifo:N where N > slots-per-chunk for non-power-of-2 N
	// values), pushes are distributed across chunks but pops only see ONE.
	// Without this scan the non-active chunks' freelist entries stay
	// unreachable through the fast path, slow_allocate hits the bitmap
	// (which can't find an N-zero run inside any one word for the wasted
	// bits at word ends) and re-mmaps a fresh chunk every cycle — caught
	// in the multi-allocator bench as a 100× regression at bucket 45
	// (ALIGN=1024 N=13) and similarly for other non-power-of-2 N tiers.
	if(char *head = scan_dll_freelist(/*local_id=*/0u)) {
		kame_page()->m_slots[bucket].freelist_head =
#if KAME_FS_TWOLIST
		    // (§two-list) the lean pops the [1] segment; aim there.
		    reinterpret_cast<char *>(&s_tls.my_chunk->m_freelist_head[1]);
#else
		    reinterpret_cast<char *>(&s_tls.my_chunk->m_freelist_head[0]);
#endif
		return head;
	}
	void *p = allocate_chunk_path(ALIGN);
	PoolAllocatorBase *new_chunk =
	    static_cast<PoolAllocatorBase *>(s_tls.my_chunk);
	// (§12.3 / §hot-tls) Update the KameTlsPage slot to store the pointer
	// to the new chunk's m_freelist_head[local-id-for-this-bucket].
	// `kBucketLocalId[]` is read HERE (cold path), so the hot path
	// (new_redirected) needs no remap.  chunk_from_freelist_ptr recovers
	// the chunk pointer from this stored value via a single mask.
	kame_page()->m_slots[bucket].freelist_head =
	    new_chunk ? reinterpret_cast<char *>(
#if KAME_FS_TWOLIST
	                    // (§two-list) FS=true lean pops the [1] segment.
	                    &new_chunk->m_freelist_head[1])
#else
	                    &new_chunk->m_freelist_head[kBucketLocalId[bucket]])
#endif
	              : nullptr;
	return p;
}

// FS=false slow_allocate override.  Multiple bucket indices share one
// PoolAllocator<ALIGN, false> instantiation, so the bucket's slot SIZE
// (= max user_size) differs from ALIGN and must be derived from
// `bucket` at runtime.  an earlier change 4-way exponential layout:
//   bucket 1..16  →  slot_size = bucket * 16        (sizes 16..256, FS=true mixed; this branch is reached
//                                                    via the FS=false specialisation only for buckets 6, 8,
//                                                    10, 12, 14, 16 — the FS=false-half of the mixed range)
//   bucket 17..24 →  4-way octave 8/9/10 sub 1..3/0..3/0  (ALIGN=64, user_size = total - 8)
//   bucket 25..32 →  4-way octave 10/11/12 sub 1..3/0..3/0 (ALIGN=256)
//   bucket 33..40 →  4-way octave 12/13/14 sub 1..3/0..3/0 (ALIGN=1024)
//
// The FS=false `allocate_pooled` expects SIZE = user size (max user
// bytes the bucket serves).  Internally it computes
// N = ceil((SIZE + 8) / ALIGN).
template <unsigned int ALIGN, bool DUMMY>
__attribute__((cold, noinline))
void *
PoolAllocator<ALIGN, false, DUMMY>::slow_allocate(unsigned bucket,
                                                  std::size_t /*size*/) noexcept {
	// Inverse of `bucket_for_size`: bucket index → max user_size to pass
	// to allocate_pooled (which derives the bit count N).
	//
	// Buckets 1..23 (FS=true + FS=false mixed): slot_size = bucket * 16.
	//
	// Buckets 24..51 (FS=false): use the `kBucketNewSlot[]` table directly
	// rather than the 4-way octave/sub formula.  The formula only covers
	// the monotonic ladder 24..47; the (§16) page-aligned tier 48..51
	// (slots 4096/8192/16384/32768) is out-of-order and the formula would
	// produce garbage for it (the latent bug that never fired because the
	// alloc_minimal_bench freelist never misses).  Per mode:
	//   * Borrow (ALIGN<1024): allocate_pooled needs SIZE = slot - 8 (the
	//     last 8 B are the borrow prefix), N = ceil((SIZE+8)/ALIGN).
	//   * Full-usable (ALIGN>=1024, §16): SIZE = slot, N = ceil(SIZE/ALIGN).
	unsigned int slot_size;
	if(bucket <= 23) {
		slot_size = bucket * 16u;
	}
	else {
		slot_size = kBucketNewSlot[bucket];
		if constexpr (ALIGN < 1024u)
			slot_size -= 8u;
	}
	// (§24) Scan this thread's DLL for an OTHER chunk holding freelist
	// entries at the SAME local id (FS=false: per-bucket local id).
	// `g_thread_freelist_ptr[bucket]` tracks only the most-recently-pinned
	// chunk; without this scan the non-active chunks' cached slots stay
	// unreachable from the fast path on multi-chunk working sets, and
	// every miss re-mmaps a fresh chunk (100× regression in the fifo:N
	// bench at bucket 45 / ALIGN=1024 N=13, etc.).  See FS=true sibling.
	using BaseTpl = PoolAllocator<ALIGN, true, false>;
	const unsigned local_id = kBucketLocalId[bucket];
	if(char *head = BaseTpl::scan_dll_freelist(local_id)) {
		kame_page()->m_slots[bucket].freelist_head =
		    reinterpret_cast<char *>(
		        &BaseTpl::s_tls.my_chunk->m_freelist_head[local_id]);
		return head;
	}
	// Inherited static; resolves to PoolAllocator<ALIGN, true, false>::
	// allocate_chunk_path, which uses the FS=false-instantiated
	// s_tls.my_chunk under the hood (the DUMMY=false template trick).
	void *p = BaseTpl::allocate_chunk_path(slot_size);
	PoolAllocatorBase *new_chunk = static_cast<PoolAllocatorBase *>(
	    PoolAllocator<ALIGN, true, false>::s_tls.my_chunk);
	// (§12.3 / §hot-tls) cf. FS=true sibling — update the KameTlsPage slot.
	kame_page()->m_slots[bucket].freelist_head =
	    new_chunk ? reinterpret_cast<char *>(
	                    &new_chunk->m_freelist_head[kBucketLocalId[bucket]])
	              : nullptr;
	return p;
}

template <class ALLOC>
inline ALLOC *
PoolAllocatorBase::allocate_chunk() {
	// Uniform 32 MiB regions carved into 128 × 256 KiB units.  A chunk =
	// `CHUNK_UNITS` (1, 2, or 4) contiguous units at a unit-aligned
	// position.  The per-region claim bitmap is 1 bit/unit — a multi-unit
	// chunk sets CHUNK_UNITS adjacent bits in a single CAS.  Per-unit
	// back-offset lives in `s_back_offset[]`; released/foreign is read
	// from `chunk_header.palloc == 0` (the former 2-bit `ready` and the
	// WIP recycle-epoch are both retired — DLL/lookup safety comes from
	// BIT_OWNED gating + the live-slot invariant).
	//
	// `s_region_has_free[]` skip-bitmap eliminates the O(N)
	// walk-past-full-regions cost.  Two passes:
	//   1. Walk set bits of `s_region_has_free` — try each region; on
	//      failure (region fully claimed), clear the bit and continue.
	//   2. If pass 1 exhausted, find an unallocated region, mmap it,
	//      set its bit, and claim there.
	constexpr unsigned int CHUNK_UNITS = ALLOC::CHUNK_UNITS;
	constexpr size_t CHUNK_SIZE = ALLOC::CHUNK_SIZE;

	// Build a PoolAllocator into a chunk whose CHUNK_UNITS units are
	// already claimed at `addr` (= chunk_base).  Shared by the
	// region-claim success path and the §34 LRC-pop path.  Mirrors the
	// dedicated path's header stamp; `ALLOC::create` placement-news the
	// PoolAllocator + m_flags + (§29) freelist pre-fill.
	auto construct_chunk_at = [&](char *addr) -> ALLOC * {
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
		return palloc;
	};

	// (§34) Unified recycle: try a warm cached chunk from the shared
	// LRC_CHUNK size-class slot before claiming a region unit or mmap'ing.
	// The block's CHUNK_UNITS units are still claimed (LRC keeps them so),
	// so we skip the bitmap-CAS claim AND the page refault entirely — same
	// win the dedicated tier already gets.  At bucket sizes (256 KiB /
	// 512 KiB / 1 MiB) the LRC idx slot holds exactly-this-size blocks
	// (each power-of-two unit count maps to a unique idx band whose only
	// quantized size is itself), so the popped block is always exactly
	// CHUNK_SIZE.  It may be DEDICATED-origin (back_off tagged 0x80) — so
	// re-stamp back_off to the bucket tag (0) before constructing.
	if(char *cached = large_recycle_pop(CHUNK_SIZE, LRC_CHUNK)) {
		restamp_back_offset(cached, CHUNK_SIZE, /*back_off_flag=*/0u);
		return construct_chunk_at(cached);
	}

	// (§74) Fresh claim via the SINGLE shared region-walk: claim_chunk runs
	// the two-pass NUMA region walk + the fresh aligned mmap + the unit-
	// aligned bitmap-CAS + the post-CAS back_offset publish — the runtime-
	// unit-count generalisation of the inline walk that used to live here
	// (now the one mmap+radix-claim site, shared with allocate_dedicated_chunk).
	// back_off_flag = 0 = the bucket tag (plain `u`), exactly what
	// construct_chunk_at expects; the CHUNK_OCC_MASK / CHUNK_STRIDE_BITS logic,
	// the d2e2c32b post-CAS publish, and the cap/swarm retry loop are all
	// inside claim_chunk and bit-for-bit identical for CHUNK_UNITS.
	char *addr = claim_chunk(CHUNK_UNITS, /*back_off_flag=*/0u);
	if( !addr)
		return nullptr;   // region cap / mmap refusal → caller falls to malloc
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
	// (§34) Shared construct (header stamp + create() + writeBarrier).  The
	// units' back_off was published as the bucket tag (plain `u`) by
	// claim_chunk, exactly what construct_chunk_at expects.
	return construct_chunk_at(addr);
}
// chunk-claim is purely mmap.  No global registry —
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
	// (§18) Pool OOM signals propagate as nullptr — top-level operator
	// new / C API wrappers handle the new_handler / errno=ENOMEM dance.
	// Throwing bad_alloc from inside a noexcept C wrapper (kame_pool_*
	// or operator new(nothrow)) terminates the process; returning null
	// preserves the noexcept contract.  Diagnostic stays on stderr so
	// the OOM is observable in test logs.
	if( !palloc) {
		fprintf(stderr,
		    "kamepoolalloc: OOM — chunk-claim failed for ALIGN=%d "
		    "(mmap region cap or kernel mmap refusal).\n", ALIGN);
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
	// Try the bitmap-CAS path on the current `s_tls.my_chunk` before
	// falling all the way through to the DLL scan + mmap-fresh path.
	// allocate_pooled() does its own per-flag atomic CAS so concurrent
	// allocations from the same chunk by other threads are safe.
	// `s_tls.my_chunk` is a DLL member of this thread — only
	// this thread can release it (via an earlier change `owner_release` or
	// thread-exit `release_dll_chunks_for_thread`), so no other-thread
	// guard is needed.
	if(PoolAllocator<ALIGN, DUMMY, DUMMY> *my = s_tls.my_chunk) {
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
	// only reset cursor / exhausted if the batch actually
	// had entries that we are about to flush back to chunk bitmaps.
	// In pure-alloc workloads (alloc_only, no cross-frees), the batch
	// stays empty, the flush is a no-op, and we keep the cursor's
	// O(N) → O(1) advantage.  Stress workloads see real cross-frees,
	// the count > 0 path fires, and the cursor rewinds so partial
	// revivals are visited.
	if(tls_cross_dealloc_batch.count != 0) {
		tls_cross_dealloc_batch.flush();
		s_tls.dll_cursor = nullptr;
		s_tls.dll_exhausted = false;
	}
	// own-empty-neighbour release.  Walk forward from
	// `s_tls.my_chunk` along the DLL: if the immediate next chunk is
	// empty ((m_flags_packed & MASK_CNT) == 0), release it; if its successor
	// is *also* empty, release that too — capped at two consecutive
	// releases per trigger so the cost stays bounded.  Steady-state
	// memory growth = mmap rate (one new chunk per chunk-fill); this
	// release path balances it at the same cadence.
	//
	// Safety: an earlier change `owner_release` CAS's `BIT_RELEASED` on the
	// chunk's `m_flags_packed` — this races safely against any
	// cross-thread `cross_release`, but cross_release additionally
	// requires `BIT_OWNER_EXITED == 1` which only the owner's
	// exit-path sets, so while we're alive only `owner_release` can
	// win the race.  Caller (us) handles the post-CAS DLL unlink +
	// `delete` + `deallocate_chunk`.
	if(s_tls.my_chunk) {
		auto *nx = s_tls.my_chunk->m_dll_next;
		for(int released = 0; nx && released < 2; ) {
			auto *nxnext = nx->m_dll_next;
			if((nx->m_flags_packed & PoolAllocator<ALIGN, DUMMY, DUMMY>::MASK_CNT) != 0)
				break;  // hit a non-empty
			// `nx` is `PoolAllocator<ALIGN, DUMMY, DUMMY> *` (same
			// type as `s_tls.my_chunk`).  Use the current template's
			// `owner_release` (FS=true and FS=false specialisations
			// share the static — `m_flags_packed` lives on the
			// FS=true base).
			if(PoolAllocator<ALIGN, FS, DUMMY>::owner_release(nx)) {
				// DLL unlink (single-writer = us).
				if(nx->m_dll_prev) nx->m_dll_prev->m_dll_next = nx->m_dll_next;
				else               s_tls.dll_head = nx->m_dll_next;
				if(nx->m_dll_next) nx->m_dll_next->m_dll_prev = nx->m_dll_prev;
				else               s_tls.dll_tail = nx->m_dll_prev;
				// PoolAllocator object now embedded inside chunk_base +
				// ALLOC_CHUNK_HEADER; chunk_base from `nx` directly.
				char *cbase = reinterpret_cast<char*>(nx) - ALLOC_CHUNK_HEADER;
				size_t csz = nx->m_chunk_size;
				// Phase-4a stale-cache invariant: multiple FS=false
				// buckets share one PoolAllocator<ALIGN,false>
				// template, so the released chunk `nx` may still be
				// referenced by `m_slots[b].freelist_head` for sibling
				// buckets that never triggered a chunk-switch.  Sweep all
				// per-thread bucket slots and clear matching pointers
				// BEFORE `delete nx`; the next `new_redirected_large`
				// on those buckets will route via `cold_first_access`
				// → `bucket_first_access<B>` and re-pin against the
				// current valid `s_tls.my_chunk` for this template.
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
				// (§12.3 / §hot-tls) Walk the KameTlsPage slots and clear
				// any entry whose stored pointer falls into the about-to-
				// be-released chunk.  `chunk_from_freelist_ptr` recovers
				// the chunk pointer from the stored char ** value.
				{
				KameTlsPage *pg = kame_page();
				for(int b = 0; b < ALLOC_NUM_BUCKETS; ++b) {
					char **fp = reinterpret_cast<char **>(pg->m_slots[b].freelist_head);
					if(fp && chunk_from_freelist_ptr(fp) == nx_pa)
						pg->m_slots[b].freelist_head = nullptr;
				}
				}
				// if the cursor was pointing at the released
				// chunk, advance past it.  Also clear the exhaustion
				// flag — DLL was just modified; an earlier chunk
				// (now closer to head via the unlink) might have had
				// cross-thread frees we didn't see during the previous
				// walk.  Conservative reset → next allocate_chunk_path
				// rewalks from head.
				if(s_tls.dll_cursor == nx)
					s_tls.dll_cursor = nxnext;
#if KAME_POOL_ONEBACK_SKIP
				// (§33) Clear dll_one_back if it pointed at the chunk
				// we're releasing — otherwise it dangles and a chunk
				// reused at the same address would be wrongly skipped
				// (benign ABA, but cheap to avoid).
				if(s_tls.dll_one_back == nx)
					s_tls.dll_one_back = nullptr;
#endif
				s_tls.dll_exhausted = false;
				// (1b) Clear m_owner_id BEFORE the destructor — see the
				// FS=true batch_return_to_bitmap sibling for the rationale.
				nx->m_owner_id = 0;
				if(nx->m_owner_self_ref) {
					// (Path B owner-ref) Adopted neighbour: it may still be
					// pinned by a concurrent scrubber.  DROP the self-ref —
					// owner_release already cleared BIT_OWNED, so the disposer
					// reclaims it when refcnt hits 0 (now, or at the last pin
					// drop).  Don't touch `nx` after this; it may be freed.
					nx->m_owner_self_ref.reset();
				} else {
					// Fresh chunk (never on the chain, no scrub pin) — direct.
					nx->~PoolAllocator();   // placement-new destructor
					PoolAllocatorBase::bucket_release_chunk(cbase, csz);  // (§34) park warm
				}
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
	// the DLL — an earlier change made this the SOLE chunk-reuse mechanism
	// (the per-template global chunk registry, retired in 4b-final,
	// no longer mediates cross-thread chunk reclaim).
	//
	// cursor-based DLL walk.  If `s_tls.dll_exhausted` is true,
	// the previous walk reached end without finding space — skip the
	// walk entirely (set false again when a new chunk is added below,
	// or when an owner_release clears it).  Otherwise resume from
	// `s_tls.dll_cursor` (set by the previous successful claim or end-of-
	// walk advance), or s_tls.dll_head on first walk after a reset.
	//
	// Forward sweep covers the full DLL from the cursor on: newer
	// chunks appear later (see the tail-append at the mmap-fresh path
	// below), so iterating from the cursor visits everything added
	// after the cursor — including the just-mmapped chunk if the
	// cursor was reset via mmap-fresh.  Skipping `s_tls.my_chunk` avoids
	// a redundant retry of the just-failed `allocate_pooled` call
	// above.
	// check the cross-thread revival hint.  If any cross-
	// thread free flipped our "force walk from head" flag since the
	// last walk, restart the walk from `s_tls.dll_head` so we visit
	// chunks that received bitmap clears we wouldn't see by
	// resuming from the (possibly past-end) cursor.  `exchange`
	// resets the flag in the same atomic; subsequent cross-thread
	// frees re-arm it.
	if(s_tls.dll_force_walk_from_head.exchange(false, std::memory_order_relaxed)) {
		s_tls.dll_cursor = nullptr;
		s_tls.dll_exhausted = false;
	}
	if( !s_tls.dll_exhausted) {
		auto *c = s_tls.dll_cursor ? s_tls.dll_cursor : s_tls.dll_head;
		while(c) {
			// (§33) Skip the pinned chunk always; skip "1-back" too when
			// `KAME_POOL_ONEBACK_SKIP` is enabled — see the macro doc.
#if KAME_POOL_ONEBACK_SKIP
			if(c != s_tls.my_chunk && c != s_tls.dll_one_back) {
#else
			if(c != s_tls.my_chunk) {
#endif
				if(void *p = c->allocate_pooled(SIZE)) {
#if KAME_POOL_ONEBACK_SKIP
					s_tls.dll_one_back = s_tls.my_chunk;  // pin shift
#endif
					s_tls.my_chunk = c;
					s_tls.dll_cursor = c;
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
		s_tls.dll_cursor = nullptr;
		s_tls.dll_exhausted = true;
	}
	// All own chunks full — mmap a fresh chunk.  an earlier change retired the
	// previous pin-CAS scan of the global registry; chunks owned by
	// other (live or exited) threads are no longer reclaimable here.
	// Exited-thread chunks drain naturally via cross-thread frees +
	// `BIT_OWNER_EXITED`; live-thread chunks are private DLL members
	// of their owner.
	//
	// The fresh chunk is appended to this thread's DLL tail and cached
	// in `s_tls.my_chunk`.  Also register the per-template DLL teardown
	// callback with `tls_alloc_thread_exit_cleanup` if not already registered
	// (deduped inside `add`), so thread-exit walks this template's
	// DLL.
	// (§36) Orphan reclaim: before mmap'ing a fresh chunk, try to re-own a
	// chunk orphaned by an exited thread (its free slots are otherwise
	// unreachable -> reserved-region growth, alloc_thread_churn scenario 2).
	// Pop one; CAS BIT_OWNED 0->1 to claim; re-arm owner metadata exactly as
	// create_allocator's fresh-chunk init does; splice into this thread's DLL
	// tail.  Then allocate from it if it has room, else leave it adopted in
	// the DLL and fall through to a fresh mmap.  After this the chunk is a
	// normal OWNED chunk again — if its owner later exits non-empty it is
	// pushed again (temporal reuse; never double-linked, one push at a time).
	// (Path B) Before mmap'ing fresh: (1) scrub — reclaim EMPTY orphans from
	// the chain (frees their region units); (2) ADOPT — pop ONE head orphan
	// and re-own it, reusing its free slots so SURVIVOR (non-empty) orphans
	// are not stranded.  HOLD oc_hold through the BIT_OWNED claim: that keeps
	// the chunk alive until BIT_OWNED is set; oc_hold is then MOVED into the
	// chunk's m_owner_self_ref (the owner-ref, see below) so a residual scrub
	// pin draining → refcnt 0 → atomic_intrusive_dispose no-ops while owned.
	// oc_hold is the FS=true-base type (the chain's node type); the downcast
	// reverses the upcast orphan_chain_push applied at owner-exit.
	orphan_chain_scrub();
	local_shared_ptr<PoolAllocator<ALIGN, true, DUMMY> > oc_hold = orphan_chain_pop();
	if(PoolAllocator<ALIGN, DUMMY, DUMMY> *oc =
	       static_cast<PoolAllocator<ALIGN, DUMMY, DUMMY> *>(oc_hold.get())) {
		// Claim: set BIT_OWNED, preserving the MASK_CNT survivors that the
		// holding thread's cross-thread frees may still be decrementing.
		// We are the SOLE owner of this popped pointer (it is now off the
		// stack and no other thread can see it), so no concurrent writer can
		// set BIT_OWNED — the only thing that races us on m_flags_packed is a
		// cross-thread free's MASK_CNT atomicDecAndTest.  The CAS therefore
		// retries on a MASK_CNT change (re-reading the value) and supplies
		// the acquire barrier; it can only OBSERVE BIT_OWNED already set in
		// the (invariant-violating) duplicate-push case, which we discard.
		// CRITICAL: a popped orphan must NOT be left stranded — looping
		// guarantees BIT_OWNED gets set (the chunk is re-owned) unless it
		// was already owned (impossible under single-push, hence the
		// defensive discard).  Read m_flags_packed non-atomically (same
		// access style as the neighbour walk above); the CAS is the
		// synchronisation point.
		bool claimed = false;
		for(;;) {
			uint32_t of = oc->m_flags_packed;
			if(of & PoolAllocator<ALIGN, DUMMY, DUMMY>::BIT_OWNED)
				break;  // duplicate-owned (should never happen) -> discard
			if(atomicCompareAndSet(of,
			       of | PoolAllocator<ALIGN, DUMMY, DUMMY>::BIT_OWNED,
			       &oc->m_flags_packed)) {
				claimed = true;
				break;  // BIT_OWNED set, MASK_CNT preserved
			}
			// CAS lost to a concurrent MASK_CNT dec — retry with fresh `of`.
		}
		if(claimed) {
			// Re-arm owner metadata to THIS thread, mirroring
			// create_allocator's fresh-chunk setup (PoolAllocator ctor):
			//   m_owner_id                = kame_owner_id()
			//   m_owner_dll_head_addr     = &s_tls.dll_head
			//   m_owner_dll_force_walk_ptr= &s_tls.dll_force_walk_from_head
			//                               (release store)
			oc->m_owner_id = kame_owner_id();
			oc->m_owner_dll_head_addr = static_cast<void *>(&s_tls.dll_head);
			oc->m_owner_dll_force_walk_ptr.store(
			    &s_tls.dll_force_walk_from_head, std::memory_order_release);
			// Splice at DLL tail (mirror create_allocator's append below).
			oc->m_dll_next = nullptr;
			oc->m_dll_prev = s_tls.dll_tail;
			if(s_tls.dll_tail) s_tls.dll_tail->m_dll_next = oc;
			else               s_tls.dll_head = oc;
			s_tls.dll_tail = oc;
#if KAME_POOL_ONEBACK_SKIP
			s_tls.dll_one_back = s_tls.my_chunk;  // (§33) pin shift
#endif
			s_tls.my_chunk = oc;
			s_tls.dll_cursor = oc;
			s_tls.dll_exhausted = false;
			tls_alloc_thread_exit_cleanup.add(
			    &PoolAllocator<ALIGN, FS, DUMMY>::release_dll_chunks_for_thread);
			// (Path B owner-ref) MOVE oc_hold into the chunk's self-ref instead
			// of dropping it: the chunk now holds a local_shared_ptr to ITSELF
			// (the owner-ref), so refcnt = self-ref + any residual scrub pins and
			// every owner-side free becomes a refcnt-mediated reset() rather than
			// a direct deallocate_chunk.  No refcnt change (the popped chain-ref
			// becomes the self-ref); oc_hold goes null.  Closes Inv_NoBadOwnerFree.
			oc->m_owner_self_ref = std::move(oc_hold);
			if(void *p = oc->allocate_pooled(SIZE)) {
#ifdef GUARDIAN
				for(unsigned int i = 0; i < SIZE / sizeof(uint64_t); ++i) {
					if(static_cast<uint64_t *>(p)[i] != GUARDIAN)
						fprintf(stderr, "Memory tainted in %p:64\n",
						        &static_cast<uint64_t *>(p)[i]);
				}
#endif
#ifdef FILLING_AFTER_ALLOC
				for(unsigned int i = 0; i < SIZE / sizeof(uint64_t); ++i)
					static_cast<uint64_t *>(p)[i] = FILLING_AFTER_ALLOC;
#endif
				return p;
			}
			// Adopted but full -> fall through to create a fresh chunk; oc
			// stays in our DLL and is reused once a survivor is freed into it.
		}
		// else: BIT_OWNED was already set (single-push invariant violated —
		// should never happen) -> discard `oc`, fall through to fresh mmap.
		// (No stranding from a lost CAS: the claim loop above retries until
		// it sets BIT_OWNED, so a popped orphan is always re-owned here unless
		// it was the impossible duplicate-owned case.)
	}
	PoolAllocator<ALIGN, DUMMY, DUMMY> *chunk = create_allocator();
	if( !chunk) {
		// (§18) OOM — caller (operator new / C wrapper) runs the
		// new_handler retry / errno=ENOMEM dance.  Leave s_tls.my_chunk
		// pointing at whatever exhausted chunk it had (or nullptr); the
		// next caller will retry the same path.
		return nullptr;
	}
	tls_alloc_thread_exit_cleanup.add(
	    &PoolAllocator<ALIGN, FS, DUMMY>::release_dll_chunks_for_thread);
#if KAME_POOL_ONEBACK_SKIP
	s_tls.dll_one_back = s_tls.my_chunk;  // (§33) pin shift on fresh-mmap
#endif
	s_tls.my_chunk = chunk;
	chunk->m_dll_prev = s_tls.dll_tail;
	chunk->m_dll_next = nullptr;
	if(s_tls.dll_tail)
		s_tls.dll_tail->m_dll_next = chunk;
	else
		s_tls.dll_head = chunk;
	s_tls.dll_tail = chunk;
	// fresh chunk has full capacity — clear the exhaustion
	// flag and point the cursor at it.  Next allocate_chunk_path that
	// finds `s_tls.my_chunk` full will resume the DLL walk from this
	// chunk; in alloc_only workloads where this chunk is the only
	// one with space, the walk does an O(1) skip-self-and-end instead
	// of the O(N) walk-all-prior-full-chunks.
	s_tls.dll_cursor = chunk;
	s_tls.dll_exhausted = false;
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
// chunk release is a single CAS on the chunk's
// `m_flags_packed` (BIT_RELEASED).  Whoever wins the CAS is the
// exclusive releaser; they then call `delete` + `deallocate_chunk()`.
// The pin field, the bit-0-lock CAS on the per-template chunk
// registry, and the registry itself are all gone — `BIT_RELEASED`
// on the packed word is the single serialisation point across:
//
//   1. `owner_release(palloc)` — an earlier change chunk-full neighbour
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
	// an earlier change release model:
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
	for(auto *c = s_tls.dll_head; c; c = c->m_dll_next) ++dll_len;
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
	void *ppool = palloc->mempool();
	for(unsigned int i = 0; i < palloc->m_chunk_size / sizeof(uint64_t); ++i) {
		if(static_cast<uint64_t *>(ppool)[i] != GUARDIAN) {
			fprintf(stderr, "Memory tainted in %p:64\n",
				&static_cast<uint64_t *>(ppool)[i]);
		}
	}
#endif
	return true;
}

// cross_release no longer needed as a separate path.
// Cross-thread releaser identification is inlined in
// batch_return_to_bitmap's OnClearFn via atomicDecAndTest — when
// dec brings m_flags_packed to 0 (= BIT_OWNED was clear AND MASK_CNT
// was 1), that thread is uniquely the releaser.  The function is
// kept declared in allocator_prv.h for ABI stability across template
// instantiations but defined as a stub here.
template <unsigned int ALIGN, bool FS, bool DUMMY>
bool
PoolAllocator<ALIGN, FS, DUMMY>::cross_release(PoolAllocator * /*palloc*/) {
	// Legacy entry — not used in an earlier change+.  See OnClearFn release
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
	// `s_tls.dll_head` / `s_tls.dll_tail` / `s_tls.my_chunk` are wiped FIRST so any
	// later TLS dtor that allocates (via `cold_first_access` →
	// `is_allocator_thread_active() == false` → libsystem fallback)
	// cannot route through a chunk we already released.
	// single atomicFetchAnd per DLL chunk.  Clears BIT_OWNED;
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
	auto *c = s_tls.dll_head;
	s_tls.dll_head = nullptr;
	s_tls.dll_tail = nullptr;
	s_tls.my_chunk = nullptr;
#if KAME_POOL_ONEBACK_SKIP
	s_tls.dll_one_back = nullptr;  // (§33) all chunks releasing — clear
#endif
	// thread exit → cursor and exhausted flag both moot.
	s_tls.dll_cursor = nullptr;
	s_tls.dll_exhausted = false;
	while(c) {
		auto *next = c->m_dll_next;
		c->m_dll_prev = nullptr;
		c->m_dll_next = nullptr;
		// Drain this chunk's owner-thread freelists back to the bitmap
		// BEFORE the BIT_OWNED clear below.  A parked freelist slot is
		// logically free but still bit-set in m_flags, so MASK_CNT
		// over-counts until we return it; draining first lets the
		// empty/non-empty decision (newv == 0) see the true live count.
		// Safe here because BIT_OWNED is still set, so
		// batch_return_to_bitmap's dec-to-zero releaser check
		// (atomicDecAndTest) never fires (word stays != 0) — it won't
		// delete `c` out from under us mid-drain.  Replaces the global
		// drain_thread_slot_freelists() (freelists are now chunk-local).
		{
			CrossDeallocEntry fdrain[2] = {};
			// (§L0-FIFO) FS=true: m_freelist_head[1..4] hold depth-4
			// ring ENTRIES — individual parked slots, NOT list heads.
			// Return them via the bitmap path and null the cells BEFORE
			// the generic per-local walk below, which would otherwise
			// misread an entry as a list head and walk user data.
#if KAME_FS_TWOLIST
			if(c->m_fs_flag) {
				// (§two-list) [2]..[3] hold the un-consumed virgin
				// RANGE (addresses, not list heads) and [5] the byte
				// stride — return every remaining range slot through
				// the bitmap path, then null the marker cells BEFORE
				// the generic per-local walk below (which would
				// misread them as list heads).  [0]/[1] are real
				// lists — the generic walk drains them as-is.
				char *cur = c->m_freelist_head[2];
				char *end = c->m_freelist_head[3];
				uintptr_t a = reinterpret_cast<uintptr_t>(
				    c->m_freelist_head[5]);
				for(; cur < end; cur += a) {
					fdrain[0].chunk = c;
					fdrain[0].slot = cur;
					c->batch_return_to_bitmap(fdrain);
				}
				for(int i = 2; i <= 5; ++i)
					c->m_freelist_head[i] = nullptr;
			}
#endif
#if KAME_FS_CHUNK_FIFO || KAME_FS_CHUNK_STASH
			if(c->m_fs_flag) {
				// Null-marking park cells: entries are exactly the
				// non-null cells [1..4] (ring) / [1] (depth-1 stash).
				// Return each as a SINGLE slot (not a list head!) and
				// null the cell before the generic per-local walk
				// below, which would otherwise misread an entry as a
				// list head and walk user data.
				constexpr int kParkCells = KAME_FS_CHUNK_FIFO ? 4 : 1;
				for(int i = 1; i <= kParkCells; ++i) {
					if(char *blk = c->m_freelist_head[i]) {
						c->m_freelist_head[i] = nullptr;
						fdrain[0].chunk = c;
						fdrain[0].slot = blk;
						c->batch_return_to_bitmap(fdrain);
					}
				}
#if KAME_FS_CHUNK_FIFO
				c->m_fifo.r = 0;
				c->m_fifo.w = 0;
#endif
			}
#endif /* KAME_FS_CHUNK_FIFO || KAME_FS_CHUNK_STASH */
			// (§12.3) freelists are compact LOCAL-id indexed
			// (KAME_LOCAL_BUCKETS = 9); walk that range, not 48.
			for(int b = 0; b < KAME_LOCAL_BUCKETS; ++b) {
				char *fh = c->m_freelist_head[b];
				c->m_freelist_head[b] = nullptr;
				while(fh) {
					char *fnext = *reinterpret_cast<char **>(fh);
					fdrain[0].chunk = c;
					fdrain[0].slot = fh;
					c->batch_return_to_bitmap(fdrain);
					fh = fnext;
				}
			}
		}
		// an earlier change/5x: nullify the owner-revival-hint pointer BEFORE
		// clearing BIT_OWNED.  Once BIT_OWNED is clear, cross-thread
		// frees may target this chunk; if our TLS storage gets
		// reclaimed in the meantime, their `store(true)` would
		// dereference a dangling pointer.  atomic
		// release-store synchronises-with cross-thread `acquire`
		// loads — a freer that observes nullptr is guaranteed to
		// have ALL of this thread's TLS-state-tied operations
		// happen-before its own (it skips the deref).  A freer that
		// observes the old non-null pointer must have loaded BEFORE
		// our release, in which case our TLS is still live.  This
		// fixes the Linux 1000-thread `alloc_stress` SEGV that
		// the earlier change's plain pointer access exhibited.
		c->m_owner_dll_force_walk_ptr.store(
		    nullptr, std::memory_order_release);
		uint32_t old = atomicFetchAnd(&c->m_flags_packed,
		                              static_cast<uint32_t>(~BIT_OWNED));
		uint32_t newv = old & ~BIT_OWNED;
		if(newv == 0) {
			// (1b) Clear m_owner_id BEFORE the destructor — see the
			// FS=true batch_return_to_bitmap sibling for the rationale.
			c->m_owner_id = 0;
			if(c->m_owner_self_ref) {
				// (Path B owner-ref) Re-owned (adopted) chunk: it may still be
				// pinned by a concurrent scrubber's load_shared.  DROP the
				// self-ref rather than freeing directly — the disposer reclaims
				// it when refcnt hits 0 (now, if no pin remains; else when the
				// last pin drops).  BIT_OWNED was already cleared by the
				// atomicFetchAnd above, so the disposer will not no-op.  `next`
				// was cached before this, so disposing `c` here is safe.
				c->m_owner_self_ref.reset();
			} else
			{
				// Fresh chunk (never on the chain, no scrub pin possible) —
				// direct free is safe.  PoolAllocator object embedded inside
				// chunk_base; recover chunk_base from `c` directly.
				char *cbase = reinterpret_cast<char*>(c) - ALLOC_CHUNK_HEADER;
				size_t csz = c->m_chunk_size;
				c->~PoolAllocator();   // placement-new destructor
				// (§21) madvise the slot pages on thread exit by default
				// (`s_thread_exit_reclaim`, default TRUE) so RSS is returned
				// promptly — the only release path that USED to skip madvise.
				// The skip (reclaim_pages=false) was a perf optimisation:
				// MADV_DONTNEED here was ~30 % of bench-style alloc_only
				// runtime (clear_page_erms + free_pages_and_swap_cache), and
				// pages would otherwise be reclaimed by the kernel at process
				// exit (exit_mmap, one batch) or recycled warm by the next
				// claimer.  Workloads that spawn/exit threads rapidly and
				// don't care about steady-state RSS can restore the skip via
				// `kame_pool_set_thread_exit_reclaim(0)`.
				PoolAllocatorBase::deallocate_chunk(cbase, csz,
				    /*reclaim_pages=*/
				    s_thread_exit_reclaim.load(std::memory_order_relaxed) != 0);
			}
		} else {
			// Non-empty orphaned chunk: clear m_owner_id so future threads
			// can adopt it via try_adopt_orphan (§orphan-adopt).
			// m_owner_dll_force_walk_ptr was already nulled (release) above;
			// atomicFetchAnd provides a full barrier ordering this store.
			c->m_owner_id = 0;
			// Push onto the atomic_shared_ptr orphan chain so an allocating
			// thread can re-own it and reuse its free slots, instead of
			// mmap'ing fresh (the scenario-2 stranding leak).  PUSH exactly
			// ONCE, here, AFTER BIT_OWNED was cleared (by the atomicFetchAnd
			// above) and m_owner_id == 0.  Empty chunks took the `newv == 0`
			// branch above and were released directly (self-ref reset); they
			// are never on the chain, so the chain-only no-release invariant
			// holds.
			orphan_chain_push(c);   // (Path B) push onto the atomic_shared_ptr chain
		}
		c = next;
	}
}
inline void
PoolAllocatorBase::deallocate_chunk(char *chunk_base, size_t chunk_size,
                                    bool reclaim_pages) {
	// Release sequence (multi-unit aware):
	//   1. chunk_header.palloc / size_info = 0 (plain).  palloc == 0 is
	//      the "released" signal a lookup-from-slot reads (foreign
	//      check).  Published by the claim-bit clear (step 4, release).
	//   2. Clear s_back_offset[] for ALL units (plain).
	//   3. madvise the SLOT region only.  The chunk_header's page stays
	//      resident so a concurrent lookup always reads a coherent
	//      palloc, never an madvise-zeroed transient.  Reclaims
	//      physical pages but leaves VA RW.  Gated by `reclaim_pages` —
	//      `false` from the thread-exit path skips it (perf: ~30 % of
	//      bench-style alloc_only time was spent here).
	//   4. Clear the claim bits of ALL units (release).  LAST — this is
	//      what makes the units recyclable: a re-allocator's claim CAS
	//      (acquire) synchronises with this release and so observes the
	//      cleared palloc / s_back_offset before overwriting them.
	//
	// chunk_size determines the unit count (CHUNK_UNITS = chunk_size /
	// ALLOC_MIN_CHUNK_SIZE).  Region size is uniform 32 MiB.
	unsigned int chunk_units =
	    static_cast<unsigned int>(chunk_size >> ALLOC_MIN_CHUNK_SHIFT);
	// (§13.3) Derive the owning region directly from chunk_base — regions
	// are ALLOC_MIN_MMAP_SIZE-aligned, so the former O(N) scan over
	// `s_mmapped_spaces[]` (now retired) is replaced by one mask.  The
	// inner block keeps the original body's indentation.
	{
		{
			{
				RegionMeta *rmeta = region_meta_of(chunk_base);
				// (§15) chunk_base = unit_boundary[base_unit] - K_MAX
				// (shifted forward).  Add K_MAX back to recover the
				// slot-region start, which is unit-aligned and yields
				// the correct base_unit_idx via the mmap-size mask.
				uintptr_t slot_region_start =
				    (uintptr_t)chunk_base + ALLOC_CHUNK_K_MAX;
				unsigned int base_unit_idx = static_cast<unsigned int>(
				    (slot_region_start
				     & ((uintptr_t)ALLOC_MIN_MMAP_SIZE - 1u))
				    >> ALLOC_MIN_CHUNK_SHIFT);
				int word = base_unit_idx / UNITS_PER_BITMAP_WORD;
				int base_in_word = base_unit_idx % UNITS_PER_BITMAP_WORD;
				int base_bit = base_in_word;
				std::atomic<BitmapWord> *bm = &rmeta->claim_bitmap[word];
				// Step 1: clear chunk_header.  palloc = 0 is the
				// "released" signal that lookup's foreign-check reads
				// (slow path); size_info = 0 too.  Plain stores — the
				// claim-bit clear at the end (release) publishes them, so
				// a re-claimer's CAS (acquire) observes palloc == 0
				// throughout its build window (no epoch needed).
				*reinterpret_cast<std::uint64_t *>(
				    chunk_base + ALLOC_CHUNK_HEADER_SIZE_INFO_OFFSET) = 0;
				*reinterpret_cast<PoolAllocatorBase **>(
				    chunk_base + ALLOC_CHUNK_HEADER_PALLOC_OFFSET) = nullptr;
				// (1b) The fast path's "released" signal — `m_owner_id` —
				// is cleared by each `deallocate_chunk` caller BEFORE
				// running the placement-new destructor (do it through the
				// LIVE typed object, not a typed pointer to a destructed
				// one — UBSAN's vptr check catches the latter).  Dedicated
				// single-slot chunks have no PoolAllocator at chunk_base +
				// ALLOC_CHUNK_HEADER, so they neither set nor clear it; the
				// dedicated free path is short-circuited by the back_off
				// bit-7 check in `PoolAllocatorBase::deallocate`, never
				// reaching the m_owner_id comparison.
				// Step 2: clear back_offset for ALL units of this chunk.
				for(unsigned u = 0; u < chunk_units; ++u)
					rmeta->back_offset[base_unit_idx + u] = 0;
				// Step 3: madvise reclaims physical pages (slot region only).
				//
				// gated by `reclaim_pages`.  Skipped from
				// `release_dll_chunks_for_thread` (thread-exit) —
				// perf showed `clear_page_erms` +
				// `free_pages_and_swap_cache` were eating ~30 % of
				// bench-style `alloc_only` time (2017 chunks ×
				// ~100 µs each per Linux measurement).  Mid-run
				// release paths (cross-thread last-slot, owner-side
				// empty, allocate-failure cleanup) still reclaim to
				// keep long-running-process RSS in check; thread
				// teardown leaves pages mapped for the next thread
				// (or for process exit, where the kernel reclaims
				// everything in one batch via `exit_mmap`).
				if(reclaim_pages) {
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
					(void)chunk_size;
#else
					// Reclaim only FULLY-OWNED slot pages — never any chunk's
					// header page.
					//
					// §15 places this chunk's ALLOC_CHUNK_K_MAX-byte header in
					// `[chunk_base, chunk_base + K_MAX)` = the 4 KiB *below* its
					// 256 KiB unit boundary; the slot region starts AT the unit
					// boundary (`chunk_base + K_MAX`), which is 256 KiB- (hence
					// page-) aligned.  On a target whose page size > K_MAX
					// (macOS arm64: 16 KiB), the NEXT chunk's header shares the
					// final page of this chunk's slot region, and this chunk's
					// header shares a page with the PREVIOUS chunk's slot tail.
					//
					// The previous `madvise(chunk_base + ALLOC_PAGE_SIZE,
					// chunk_size - ALLOC_PAGE_SIZE, ...)` has page-UNALIGNED
					// ends there; the kernel rounds advice ranges OUTWARD (XNU
					// truncates the start / rounds up the end), so MADV_FREE
					// bled into the adjacent chunk's header page — zeroing a
					// LIVE neighbour's embedded PoolAllocator (vtable +
					// m_flags), which then crashes its next virtual dispatch:
					// `release_dll_chunks_for_thread`'s `c->~PoolAllocator()`
					// (null vtable) or `CrossDeallocBatch::flush`'s
					// `chunk->batch_return_to_bitmap()`.  Only a page reclaim
					// can zero the +K_MAX-resident header object — deallocate
					// itself never touches it — which is why this manifested as
					// an `address=0x0` EXC_BAD_ACCESS at thread/process exit,
					// macOS-only, and only after reclaim-on-exit became the
					// default (cbd0462c).
					//
					// Fix: anchor at the slot-region start (already
					// page-aligned by construction) and round the range INWARD,
					// so it can never cover ANY chunk's header page.  Header
					// pages stay resident — which also keeps the prior
					// "concurrent lookup never reads an madvise-zeroed palloc"
					// guarantee.  On Linux (PAGE == K_MAX == 4 KiB) both ends
					// are already aligned, so the reclaimed range is byte-for-
					// byte identical to before; only macOS changes (it now
					// reclaims this chunk's lower slot pages that the old
					// unaligned start skipped, and leaves the single top page
					// shared with the next header resident).
					uintptr_t slot0 = reinterpret_cast<uintptr_t>(chunk_base)
					                  + ALLOC_CHUNK_K_MAX;
					uintptr_t cend  = reinterpret_cast<uintptr_t>(chunk_base)
					                  + chunk_size;
					uintptr_t ms = (slot0 + (uintptr_t)ALLOC_PAGE_SIZE - 1u)
					               & ~((uintptr_t)ALLOC_PAGE_SIZE - 1u);  // round UP
					uintptr_t me = cend & ~((uintptr_t)ALLOC_PAGE_SIZE - 1u); // round DOWN
					if(me > ms) {
#if defined(__APPLE__)
						// macOS: MADV_FREE — cheap per-page flag flip, fast
						// reuse (kernel zeros lazily on next access).
						madvise(reinterpret_cast<void *>(ms),
						        static_cast<size_t>(me - ms), MADV_FREE);
#else
						// Linux/others: MADV_DONTNEED — eager reclaim.
						// (MADV_FREE regressed reuse-heavy workloads
						// catastrophically: bucket34_repro 33.5 → 0.26 M/s,
						// alloc_stress RSS 9 → 698 MiB — Linux MADV_FREE's
						// LRU lazy-discard list + minor-fault-on-reuse costs
						// exceed the MADV_DONTNEED + zero-fault round-trip.)
						madvise(reinterpret_cast<void *>(ms),
						        static_cast<size_t>(me - ms), MADV_DONTNEED);
#endif
					}
#endif
				}
				else {
					(void)chunk_size;
				}
				// Step 5: clear claim bits for all units (release) — LAST.
				BitmapWord claim_mask = 0;
				for(unsigned u = 0; u < chunk_units; ++u)
					claim_mask |= BitmapWord(1) << (base_bit + u);
				bm->fetch_and(~claim_mask, std::memory_order_release);
				// Step 6: (§13.3) re-flag the region as maybe-having-free
				// space.  A concurrent claim that just cleared the hint
				// re-observes it; a stale set costs one wasted scan that
				// then re-clears it.  (Was a global s_region_has_free bit.)
				rmeta->has_free.store(1, std::memory_order_relaxed);
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
	// 1-bit encoding — every set bit is a claimed unit.  This counts
	// claimed UNITS, not chunks (a multi-unit chunk contributes
	// CHUNK_UNITS bits), which is sufficient as a leak probe: monotonic
	// growth across repeated alloc/free cycles still signals a leak in
	// the chunk-release path.
	// (§13.3) Walk the push-only region list and sum bits of each
	// region's embedded claim_bitmap.  Subtract bit 0 of word 0 (the
	// per-region metadata reservation) so this returns "units occupied
	// by actual chunks" — preserving the pre-§13.2 leak-probe semantics
	// (monotonic growth = leak).
	int n = 0;
	int total_nodes = s_num_numa_nodes.load(std::memory_order_relaxed);
	if(total_nodes <= 0) total_nodes = 1;
	for(int node = 0; node < total_nodes; ++node)
	for(RegionMeta *rmeta = s_region_dll_heads[node].load(
	        std::memory_order_acquire);
	    rmeta; rmeta = rmeta->dll_next.load(std::memory_order_acquire)) {
		for(int w = 0; w < BITMAP_WORDS_PER_REGION; ++w) {
			BitmapWord v =
			    rmeta->claim_bitmap[w].load(std::memory_order_relaxed);
			if(w == 0) v &= ~BitmapWord(1);  // skip permanent metadata bit
			n += int(count_bits(v));
		}
	}
	return n;
}

// Address → chunk resolution from a (presumed-live) slot pointer.
//
// NO seqlock, NO epoch.  Every caller (lookup_chunk, deallocate,
// size_of) is passed a pointer the application still owns — a LIVE
// slot.  A live slot keeps its bit set in m_flags, which keeps
// m_flags_packed != 0, which is the precondition for EVERY
// chunk-release path.  The chunk therefore cannot be released (let
// alone recycled into a different chunk) while this resolution runs,
// so the reclaim+recycle race cannot occur on this path.  (Protection
// would only matter for a DLL-walk caller holding a chunk POINTER
// without holding any slot in it — and those paths don't go through
// here; they rely on BIT_OWNED gating instead.)  back_off's
// correctness against cross-stride claim races is already secured by
// the post-CAS publish (commit
// d2e2c32b); the embedded-PoolAllocator layout (palloc identity ==
// chunk identity) closes the object-UAF.  So a single relaxed load
// of the back-offset table plus a plain palloc read suffice — the
// pre-WIP cost profile.
static inline PoolAllocatorBase *
resolve_chunk_from_slot(char *mp, size_t /*meta_base unused*/,
                        unsigned int unit_idx,
                        char **out_chunk_base) noexcept {
	// (§13.2) back_offset now lives inside the region (RegionMeta at
	// mp + 0).  `meta_base` is unused — kept in the signature so call
	// sites can keep their existing `ccnt * NUM_ALLOCATORS_IN_SPACE`
	// computation as a no-op until that arithmetic is removed.
	PoolAllocatorBase::RegionMeta *rmeta =
	    PoolAllocatorBase::region_meta(mp);
	unsigned int back_off = rmeta->back_offset[unit_idx] & 0x7Fu;  // mask dedicated bit7
	unsigned int base_idx = unit_idx - back_off;
	// (§15) chunk_base = unit_boundary[base_idx] - K_MAX (chunk's first
	// byte sits K_MAX bytes before its claimed-units' boundary, so the
	// slot region starts unit-aligned at chunk_base + K_MAX).
	char *chunk_base = mp + (size_t)base_idx * (size_t)ALLOC_MIN_CHUNK_SIZE
	                 - (size_t)ALLOC_CHUNK_K_MAX;
	PoolAllocatorBase *palloc =
	    *reinterpret_cast<PoolAllocatorBase * const *>(
	        chunk_base + ALLOC_CHUNK_HEADER_PALLOC_OFFSET);
	// palloc == 0 ⇒ released; <= 1 ⇒ in-creation sentinel or a
	// libsystem-malloc pointer that happens to land in our mmap range
	// (macOS interpose).  Either way: foreign, fall through to free.
	if((uintptr_t)palloc <= (uintptr_t)1u) return nullptr;
	*out_chunk_base = chunk_base;
	return palloc;
}

// Address → chunk lookup.  Two indexed atomic loads via the 2-level
// radix tree (§13) to find `p`'s region in O(1), then resolves the
// owning chunk via the (seqlock-free, live-slot) resolver above.
inline PoolAllocatorBase *
PoolAllocatorBase::lookup_chunk(void *p) noexcept {
	int ccnt = radix_lookup(p);
	if(ccnt < 0) return nullptr;
	// `mp` derived from `p` (region base is 32-MiB-aligned), avoiding
	// an `s_mmapped_spaces[ccnt]` load.  Note: when regions become
	// reclaimable (future §13.2), this fast path would still report
	// non-null for a recently-unmapped region until the radix slot is
	// cleared; the caller already validates via the chunk-header
	// `palloc` read inside `resolve_chunk_from_slot`.
	char *mp = reinterpret_cast<char *>(
	    (uintptr_t)p & ~((uintptr_t)ALLOC_MIN_MMAP_SIZE - 1u));
	ptrdiff_t pdiff = static_cast<char *>(p) - mp;
	unsigned int unit_idx = static_cast<unsigned int>(
	    (size_t)pdiff >> ALLOC_MIN_CHUNK_SHIFT);
	char *chunk_base;
	return resolve_chunk_from_slot(
	    mp, (size_t)ccnt * NUM_ALLOCATORS_IN_SPACE,
	    unit_idx, &chunk_base);
}

// LEAN hot path: the ONLY case inlined here is an FS=true owner-free
// (the small fixed-size buckets — e.g. the 64 B bench hot loop).  That
// case is pure pointer arithmetic + an inlined `freelist_push`, so the
// whole function is CALL-FREE: the compiler needs no callee-saved
// registers, and the 5–6×`stp`/`ldp` prologue/epilogue spill that the
// cold function-calls used to force onto every free is gone.  On Linux
// (IE-TLS `mov %fs:`, where the TSD read is already a single cheap
// instruction) the prologue was the dominant remaining 64 B free cost,
// so removing it is the lever there; on macOS it shaves the same spill.
// Every other case (region-cache miss → foreign/large/first-touch,
// FS=false owner-free, owner-mismatch → cross/released/dedicated)
// tail-calls `deallocate_cold` — out-of-line, so its calls never taint
// this frame.
//
// `always_inline`: every caller (free interpose, operator delete and its
// sized/aligned variants, kame_pool_free — all in this TU, all after this
// definition) expands the lean path directly.  Without forcing it the
// compiler's inline heuristic flip-flopped as the function's size changed
// (64 B throughput swung 568–662 Mops/s build-to-build); pinning the
// inline makes the hot free deterministic.  The cold off-ramps stay
// out-of-line (their own `noinline`), so this only duplicates the lean
// ~20-instruction hot path per call site.
KAME_ALWAYS_INLINE void
PoolAllocatorBase::deallocate(void *p) noexcept {
	// (hoist) Read this thread's KameTlsPage ONCE: the region-cache
	// check and the owner-id compare below share it.  `free(NULL)` needs
	// no `!p` pre-filter — `last_region_base` is RADIX_CACHE_EMPTY (~0),
	// unmatchable by base==0, so a null free misses the cache and routes
	// to `deallocate_cold` → radix_lookup_slow(0) → ABSENT → libc no-op.
	KameTlsPage *pg = kame_page();
	const uintptr_t up = (uintptr_t)p;
	const uintptr_t region_base = up & ~((uintptr_t)ALLOC_MIN_MMAP_SIZE - 1u);
	if(__builtin_expect(region_base != pg->last_region_base, 0))
		return deallocate_cold(p);  // miss: foreign / large / first-touch
	// Region-cache HIT ⇒ this is a populated POOL region.  Resolve the
	// owning chunk by pure arithmetic off the (§13.2) RegionMeta
	// back_offset — no chunk_header (cache-line 0) read, same discipline
	// as the original hot path.
	char *mp = reinterpret_cast<char *>(region_base);
	ptrdiff_t pdiff = static_cast<char *>(p) - mp;
	unsigned int unit_idx =
	    static_cast<unsigned int>((size_t)pdiff >> ALLOC_MIN_CHUNK_SHIFT);
	RegionMeta *rmeta = region_meta(mp);
	unsigned int back_off_raw = rmeta->back_offset[unit_idx];
	unsigned int base_idx = unit_idx - (back_off_raw & 0x7Fu);
	char *chunk_base = mp + (size_t)base_idx * (size_t)ALLOC_MIN_CHUNK_SIZE
	                 - (size_t)ALLOC_CHUNK_K_MAX;
	PoolAllocatorBase *chunk_obj = reinterpret_cast<PoolAllocatorBase *>(
	    chunk_base + ALLOC_CHUNK_HEADER);
	uint32_t page_owner_id = pg->owner_id;
	// Owner-free fast path: a live chunk THIS thread owns.  A released
	// chunk (m_owner_id==0), a foreign / post-teardown thread
	// (page_owner_id==0) and a dedicated chunk (m_owner_id zeroed → never
	// matches a non-zero owner) all FAIL this test and tail-call the cold
	// off-ramp.  BOTH FS=true (single-size, the 64 B hot loop) and
	// FS=false (borrow / full-usable size classes) are handled inline and
	// CALL-FREE — keeping FS=false here too means the 384..2048 B churn
	// pays neither the `bl deallocate_cold` nor cold's full re-derivation
	// of pg/region/chunk_base (which it already has in registers).  Only a
	// garbage local-id (corruption / coincidental owner match on a stray
	// pointer) tail-calls cold, which re-validates via palloc + the vtable
	// owner check.
	if(__builtin_expect(chunk_obj->m_owner_id == page_owner_id
	                    && page_owner_id != 0, 1)) {
		if(__builtin_expect(chunk_obj->m_fs_flag != 0, 1)) {  // FS=true — 64 B hot
#if KAME_FS_CHUNK_FIFO
			// (§L0-FIFO) Park into the chunk's depth-4 null-marking ring
			// instead of the freelist: no store into the block itself
			// (its lines stay untouched, still warm for the next user),
			// and the matching alloc-side take returns a slot parked
			// >= 2 frees ago.  The park side owns `w` exclusively;
			// "ring full" is read off the cell content (non-null), which
			// was nulled by a take >= 2 ops ago — branch-feeding only,
			// no data-chain stall.  Full -> plain freelist push.
			std::uint32_t fw = chunk_obj->m_fifo.w;
			char **rcell = &chunk_obj->m_freelist_head[1 + (fw & 3u)];
			if(*rcell == nullptr) {
				*rcell = static_cast<char *>(p);
				chunk_obj->m_fifo.w = fw + 1;
				return;
			}
#elif KAME_FS_CHUNK_STASH
			// (§L0-STASH) Depth-1 park: stash `p` in the single cell
			// m_freelist_head[1] when empty — one load + one store on
			// the already-hot chunk line, NO store into the block itself
			// (its lines stay warm for the next user), no counters.
			// Occupied -> plain freelist push.
			char **scell = &chunk_obj->m_freelist_head[1];
			if(*scell == nullptr) {
				*scell = static_cast<char *>(p);
				return;
			}
#endif /* KAME_FS_CHUNK_FIFO / KAME_FS_CHUNK_STASH */
			chunk_obj->freelist_push(0, p);
			return;
		}
		// FS=false owner free.  Routed to a dedicated noinline helper —
		// NOT inlined here, and NOT folded into the FS=true return above:
		// the original code (and a re-measurement) shows ~5–9 % bench_loop
		// regression on the 64 B FS=true path when the FS=false local-id /
		// bucket-re-aim code shares its function (bigger frame, worse
		// inlining into `operator delete`).  The helper still receives the
		// already-resolved `chunk_base`, so FS=false avoids `deallocate_cold`'s
		// full pg/region/chunk_base re-derivation (≈ the 1 KiB churn win).
		return deallocate_fs_false_owner(chunk_base, p);
	}
	return deallocate_cold(p);  // owner-mismatch / dedicated / cross / released
}

// FS=false owner-free helper — split out of the lean `deallocate` so the
// FS=true (64 B) hot path stays maximally lean / inlinable.  Reached only
// for a this-thread-owned FS=false chunk; `chunk_base` is already resolved
// by the caller (no re-derivation).  A garbage local-id (corruption /
// coincidental owner match) tail-calls the full cold resolver, which
// re-validates via palloc + the vtable owner check.
KAME_NOINLINE void
PoolAllocatorBase::deallocate_fs_false_owner(char *chunk_base, void *p) noexcept {
	PoolAllocatorBase *chunk_obj = reinterpret_cast<PoolAllocatorBase *>(
	    chunk_base + ALLOC_CHUNK_HEADER);
	unsigned local;
	unsigned bucket = 0;
	if(chunk_obj->m_sizes) {
		size_t bit_index =
		    static_cast<size_t>(static_cast<char *>(p) - chunk_obj->mempool())
		    >> chunk_obj->m_align_shift;
		local = chunk_obj->m_sizes[bit_index] & 0xFFu;
		if(local >= (unsigned)KAME_LOCAL_BUCKETS)
			return deallocate_cold(p);
	} else {
		std::uint64_t hdr = *reinterpret_cast<std::uint64_t *>(
		    static_cast<char *>(p) - 8);
		local  = static_cast<unsigned>(hdr) & 0xFFu;
		bucket = (static_cast<unsigned>(hdr) >> 16) & 0xFFu;
		if(local >= (unsigned)KAME_LOCAL_BUCKETS)
			return deallocate_cold(p);
	}
	chunk_obj->freelist_push(local, p);
	// (§freelist-follow) re-aim this thread's per-bucket alloc shortcut at
	// the slot we just freed (LIFO) — borrow tier only (bucket != 0).
	if(bucket != 0 && bucket < (unsigned)ALLOC_NUM_BUCKETS)
		kame_page()->m_slots[bucket].freelist_head =
		    reinterpret_cast<char *>(&chunk_obj->m_freelist_head[local]);
	return;
}

// Forward decl (real declaration with __attribute__((noinline)) sits next to
// the definition further down this TU) so `deallocate_cold` can call it.  The
// foreign-pointer fallback now lives INSIDE `deallocate_cold` (it is void +
// self-contained: a foreign / released pointer is libsystem-freed in place),
// so the lean `deallocate` TAIL-CALLS `deallocate_cold` frame-free — no
// separate wrapper, and no extra hop on the large/cold path (the wrapper hop
// previously cost ~10 % on the 64 KiB–256 KiB free band).
static void libsystem_free_for_pool(void *p) noexcept;

KAME_NOINLINE_COLD void
PoolAllocatorBase::deallocate_cold(void *p) noexcept {
	// COLD off-ramp (split out of `deallocate` so the hot path stays
	// call-free / prologue-free; see the lean `deallocate` just above).
	// This is the original full resolver: it re-derives everything from
	// `p` (region cache may have just been populated by the slow lookup)
	// and handles foreign/large, FS=false owner-free, and every
	// owner-mismatch case (cross-thread / released / dedicated /
	// post-teardown).  Reached on a region-cache MISS, or after the lean
	// path declines (not an FS=true owner-free).
	// `delete nullptr` / `free(NULL)` are well-defined no-ops.  No `!p`
	// pre-filter here: `s_tls_hot.last_region_base` is initialised to (and
	// only ever cleared to) `RADIX_CACHE_EMPTY` (= ~0), which is unmatchable
	// by `base = 0` from `p == NULL`, so a null free routes through
	// `radix_lookup_slow(0)` → ABSENT → caller's `libsystem_free(NULL)`
	// no-op.  Two instructions (`test %rdi,%rdi; je`) shaved off the hot
	// path's prologue.
	// §13: O(1) p -> radix kind via the 2-level radix tree + per-thread
	// 1-entry cache.  kind == 0 (KAME_RADIX_ABSENT) means the pointer
	// is outside every populated region — pass through to libsystem free.
	// kind == 2 (KAME_RADIX_LARGE) is the §19 large-alloc tier — single
	// mmap registered as one radix slot; dispatch to its free helper
	// which CAS-clears the slot then munmap's the region.
	// (hoist) Read this thread's KameTlsPage ONCE for the whole hot path:
	// the radix region-cache check, the owner-id compare below, and the
	// freelist push after an owner match all live in the same page.  The
	// design always intended a single kame_page() (see KameTlsPage doc),
	// but as separate `radix_lookup()` + `kame_page()->owner_id` calls the
	// compiler emitted TWO fast-TSD reads (offset load + mrs + indexed load
	// + guard, twice — confirmed by otool).  Inline radix_lookup's hot
	// region-cache check here against `pg` so the whole free shares one read.
	KameTlsPage *pg = kame_page();
	const uintptr_t up = (uintptr_t)p;
	const uintptr_t region_base = up & ~((uintptr_t)ALLOC_MIN_MMAP_SIZE - 1u);
	int kind = __builtin_expect(region_base == pg->last_region_base, 1)
	               ? (int)KAME_RADIX_POOL
	               : radix_lookup_slow(up);
	// Single hot-path branch: the overwhelmingly common case is a POOL
	// pointer (kind == 1).  Fold ABSENT (foreign → libc free) and LARGE
	// (§19 mmap tier) into one cold off-ramp so a normal small/dedicated
	// free pays ONE predicted-not-taken compare, not two (§19 originally
	// added a second `== LARGE` test in series on every free).
	if(__builtin_expect(kind != (int)KAME_RADIX_POOL, 0)) {
		if(kind == (int)KAME_RADIX_ABSENT) {               // foreign → libsystem free
			libsystem_free_for_pool(p);
			return;
		}
		PoolAllocatorBase::deallocate_large_va(p);         // KAME_RADIX_LARGE
		return;
	}
	// `mp` is the region base — already computed as `region_base` for the
	// radix cache check above (region is ALLOC_MIN_MMAP_SIZE-aligned by the
	// §13 alignment requirement), so reuse it (no `s_mmapped_spaces[ccnt]`
	// load, no recompute).
	char *mp = reinterpret_cast<char *>(region_base);
	ptrdiff_t pdiff = static_cast<char *>(p) - mp;
	{
		// `p` is a LIVE slot (the caller's contract: deallocating an
		// already-freed pointer is undefined behaviour).  A live slot
		// keeps its bit set in `m_flags`, which in turn keeps
		// `m_flags_packed != 0` and so prevents any path
		// (owner_release, cross-flush dec-to-zero, thread-exit) from
		// releasing this chunk.  No reclaim+recycle race can therefore
		// race this lookup -- the seqlock validation is unnecessary
		// here.  (The seqlock is meaningful only for DLL-walk paths
		// where a chunk POINTER is held without holding any slot in
		// that chunk; lookup_chunk-from-slot is not such a case.)
		unsigned int unit_idx =
		    static_cast<unsigned int>((size_t)pdiff >> ALLOC_MIN_CHUNK_SHIFT);
		// (§13.2) back_offset lives in `RegionMeta` at `mp + 0`, recovered
		// in O(1) from the mp we already derived.  The former
		// `meta_base = ccnt * NUM_ALLOCATORS_IN_SPACE` multiply is gone.
		RegionMeta *rmeta = region_meta(mp);
		unsigned int back_off_raw = rmeta->back_offset[unit_idx];
		unsigned int base_idx = unit_idx - (back_off_raw & 0x7Fu);
		// (§15) chunk_base = unit_boundary[base_idx] - K_MAX (see
		// resolve_chunk_from_slot for the layout rationale).
		char *chunk_base = mp + (size_t)base_idx * (size_t)ALLOC_MIN_CHUNK_SIZE
		                 - (size_t)ALLOC_CHUNK_K_MAX;
		// Dedicated single-slot large chunk?  bit7 of the (already-loaded)
		// back_off byte flags it — NO chunk_header read added to the hot
		// owner-free path (preserves the (1b) cache-line discipline).
		// Free the whole N-unit chunk (total bytes in chunk_header[32..39]).
		// (§hot-tls) bit-7 (dedicated-chunk flag) check moved BELOW the
		// owner-id compare.  `allocate_dedicated_chunk` zeros the would-be
		// m_owner_id slot in the K_MAX gap (see allocator_prv.h chunk
		// layout diagram), so a dedicated chunk's m_owner_id read always
		// returns 0 — guaranteed not to match any thread's non-zero
		// owner_id, so the chunk falls through to the cold path where the
		// bit-7 check selects the dedicated handler.  Net: TWO INSTRUCTIONS
		// (`test %dl,%dl; js`) gone from the hot-path prologue for the
		// common regular-chunk free.
		// (1b) Owner-free FAST PATH — touches ONLY chunk_obj's cache line
		// (chunk_base + ALLOC_CHUNK_HEADER), NEVER chunk_header (cache
		// line 0, where palloc lives at +8 and size_info at +0).  Under
		// the embed layout the PoolAllocator object sits AT
		// chunk_base + ALLOC_CHUNK_HEADER, so palloc (when live) always
		// equals that address — we reach the object by pure arithmetic
		// and skip the palloc load.  See tests/CHUNK_CLAIM_TLA_NOTES.md
		// §12/§12.1.
		PoolAllocatorBase *chunk_obj = reinterpret_cast<PoolAllocatorBase *>(
		    chunk_base + ALLOC_CHUNK_HEADER);
		// Owner-id compare via `kame_page()->owner_id` — same TLS page
		// as `last_region_base` (already touched at the radix-cache hit
		// above via kame_page()), so the compiler keeps the page pointer
		// in a register and this read is a single load with no second
		// TSD/TLV overhead.  Matches iff THIS thread owns the chunk AND
		// the chunk is live: a released chunk has m_owner_id == 0 (cleared
		// by deallocate_chunk), and a foreign / never-allocated thread
		// has `owner_id == 0` — neither matches a live non-zero owner
		// id.  So this one comparison subsumes the former `palloc <= 1`
		// released/foreign pre-filter for the fast path; palloc is read
		// only on the slow path below (cross-thread / released /
		// foreign / post-teardown).
		// (teardown) The "owner_id == 0 never matches a live owner" reasoning
		// above holds only while the FREEING thread is live.  During this
		// thread's own exit, AllocThreadExitCleanup orphans its chunks
		// (m_owner_id = 0) AND repoints kame_page() at g_teardown_page
		// (owner_id = 0); a later free (e.g. a pthread_key dtor releasing an
		// XThreadLocal buffer) would then match 0 == 0 here, take the fast
		// owner path, and freelist_push WITHOUT decrementing MASK_CNT — the
		// orphan never reaches MASK_CNT == 0 so orphan_chain_scrub never
		// reclaims it (unbounded thread-exit stranding; see
		// tests/alloc_thread_exit_free_test.cpp).  kame_owner_id() never
		// returns 0, so requiring a non-zero page owner routes every
		// post-teardown free to the cold cross-free path, which decrements
		// MASK_CNT and reclaims correctly.
		uint32_t page_owner_id = pg->owner_id;   // (hoist) reuse the page read at fn entry
		if(__builtin_expect(chunk_obj->m_owner_id == page_owner_id
		                    && page_owner_id != 0, 1)) {
			// (§12.3 / §16) Local-id from the cache-line-1 hot block:
			//   FS=true        : chunk serves one size -> local-id 0.
			//   FS=false borrow: per-slot prefix { uint32 local_id,
			//                    uint32 SIZE } at p-8 (slot's own line).
			//   FS=false full  : m_sizes != null (ALIGN>=1024) -> local-id
			//                    is the low byte of m_sizes[bit], bit =
			//                    (p - m_mempool) >> m_align_shift.  m_sizes
			//                    sits in this already-loaded hot line, so a
			//                    borrow chunk's `!m_sizes` check is free.
			//
			// FS=true is split out as its own early-return path: a
			// single bucket per chunk means the per-thread alloc
			// shortcut `g_thread_freelist_ptr[bucket]` is already
			// maintained correctly by chunk-claim / slow_allocate, so
			// no follow-update is needed.  Keeping the branch separate
			// stops the compiler from emitting the bucket-init + check
			// dead code on this hottest path (5 % bench_loop regression
			// when folded together).
			if(chunk_obj->m_fs_flag) {
				chunk_obj->freelist_push(0, p);
				return;
			}
			// FS=false owner free.  `bucket` is extracted (borrow tier
			// only) so we can re-aim `g_thread_freelist_ptr[bucket]` at
			// this chunk's freelist head — the next same-bucket alloc
			// pops the slot we just freed (LIFO) instead of running
			// `slow_allocate` → `scan_dll_freelist` on a multi-chunk
			// working set.  ~3–6× win on FS=false 384..2048 churn.
			// Full-usable (ALIGN>=1024) leaves bucket = 0 (skip
			// sentinel) since the bucket isn't encoded in m_sizes[].
			unsigned local;
			unsigned bucket = 0;
			if(chunk_obj->m_sizes) {
				size_t bit_index =
				    static_cast<size_t>(static_cast<char *>(p) - chunk_obj->mempool())
				    >> chunk_obj->m_align_shift;
				local = chunk_obj->m_sizes[bit_index] & 0xFFu;
				if(local >= (unsigned)KAME_LOCAL_BUCKETS)
					goto vtable_dispatch;
			} else {
				std::uint64_t hdr = *reinterpret_cast<std::uint64_t *>(
				    static_cast<char *>(p) - 8);
				local  = static_cast<unsigned>(hdr) & 0xFFu;
				// (FS=false borrow) bucket packed in bits 16..23 of the
				// prefix's low-32 half at allocate-time — free to extract
				// from the already-loaded `hdr`.
				bucket = (static_cast<unsigned>(hdr) >> 16) & 0xFFu;
				// Defensive: a stray pointer whose owner-id coincidentally
				// matched but whose prefix is garbage — fall to the slow
				// path (which re-validates via palloc + the vtable owner
				// check).
				if(local >= (unsigned)KAME_LOCAL_BUCKETS)
					goto vtable_dispatch;
			}
			chunk_obj->freelist_push(local, p);
			// (§freelist-follow) FS=false borrow only — see comment above.
			if(bucket != 0 && bucket < (unsigned)ALLOC_NUM_BUCKETS) {
				kame_page()->m_slots[bucket].freelist_head =
				    reinterpret_cast<char *>(&chunk_obj->m_freelist_head[local]);
			}
			return;
		}
		// Owner mismatch.  Either: dedicated chunk (m_owner_id == 0,
		// never matches), or regular chunk being freed cross-thread /
		// released / post-teardown.  Bit-7 of the already-loaded
		// `back_off_raw` flags the dedicated case; check before the
		// vtable dispatch below (which would mis-interpret a dedicated
		// chunk's stale chunk_header.fn).  Defensive `goto vtable_dispatch`
		// jumps from the owner-matched block above skip THIS bit-7 check
		// (they jump to the label directly), which is correct: those gotos
		// fire only when m_owner_id DID match, and m_owner_id never
		// matches for dedicated.
		if(__builtin_expect((back_off_raw & 0x80u) != 0u, 0)) {
			size_t bytes = (size_t)*reinterpret_cast<std::uint64_t *>(
			    chunk_base + ALLOC_CHUNK_HEADER_DEDICATED_SIZE_OFFSET);
			// (§28.5) dedicated_chunk_bytes is now recomputed on demand
			// inside `kame_pool_get_stats` via region+back_offset walk; the
			// hot-path running counter is gone.  (§28.4 sharded counters
			// still collided 2-way at ≥ 128T because LRC_STATS_SHARDS=64.)
			// (§22) Recycle into the per-thread cache, keeping the units
			// CLAIMED and the chunk_header intact for warm reuse (no
			// bitmap clear, no madvise here).  On overflow the cache
			// returns false and we do the real release now.  The unique
			// owner (this thread, or whichever later evicts it) performs
			// the N-bit bitmap-CAS clear inside deallocate_chunk.
			if( !large_recycle_push(chunk_base, bytes, LRC_CHUNK))
				deallocate_chunk(chunk_base, bytes);
			return;
		}
	vtable_dispatch:
		// Cold path: cross-thread / non-owner / released / post-teardown.
		// NOW read palloc (chunk_header[8]) — the only place the fast
		// path's missing cache-line-0 load reappears.  palloc == 0 ⇒
		// released; foreign (libsystem-malloc pointer in our mmap range,
		// macOS Apple Silicon early startup) reads 0 or garbage <= 1 —
		// fall through to libsystem free.  Otherwise dispatch the
		// per-template DeallocateFn at chunk_base + 16, which preserves
		// the adaptive-holding / cross-batch / chunk-release logic in
		// `deallocate_pooled`.
		{
		PoolAllocatorBase *palloc =
		    *reinterpret_cast<PoolAllocatorBase * const *>(
		        chunk_base + ALLOC_CHUNK_HEADER_PALLOC_OFFSET);
		if((uintptr_t)palloc <= (uintptr_t)1u) {   // released / foreign
			libsystem_free_for_pool(p);
			return;
		}
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
			// chunk_size is now per-template; read from palloc's
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
				// Current `deallocate_pooled` always returns false.
				// Chunk release happens elsewhere: the owner-side empty
				// release in `release_dll_chunks_for_thread` / `owner_release`
				// (BIT_OWNED clear → deallocate_chunk), and the neighbour
				// release in `allocate_chunk_path`.  A cross-thread free that
				// empties an ORPHANED chunk no longer releases it — the chunk is
				// on the atomic_shared_ptr chain and reclaimed by
				// orphan_chain_scrub, so `batch_return_to_bitmap` performs no
				// release at all now.  Kept as a defensive shim in case a future
				// trampoline opts to release at this site.
				deallocate_chunk(chunk_base, csz);
			}
		}
		return;
		}
	}
	libsystem_free_for_pool(p);  // unreachable (LIVE block returns); defensive foreign-free
}

// `size_of` — read-only sibling of `deallocate`.  Resolves the owning
// chunk via the same seqlock, then dispatches `SizeOfFn` at offset
// `ALLOC_CHUNK_HEADER_SIZEOF_FN_OFFSET` (= 24) and returns the slot size
// in bytes.  Used by `kame_realloc` to size copies.  Returns 0 for any
// pointer not inside our pool (libsystem-malloc'd, null, or released).
inline std::size_t
PoolAllocatorBase::size_of(void *p) {
	if( !p) return 0;
	// §13/§19: O(1) p -> radix kind via the 2-level radix tree.
	int kind = radix_lookup(p);
	if(kind <= (int)KAME_RADIX_ABSENT) return 0;
	if(kind == (int)KAME_RADIX_LARGE) {
		// (§19) Usable = the full slot past the meta page.  Returns
		// mmap_size - PAGE (not the user-requested alloc_size), matching
		// malloc_usable_size's libc convention of reporting the actually-
		// usable space — lets realloc-elision in client code grow in
		// place across the slack rounded up to PAGE.
		LargeAllocMeta *m = large_alloc_meta_of(p);
		return m->mmap_size - (std::size_t)ALLOC_PAGE_SIZE;
	}
	char *mp = reinterpret_cast<char *>(
	    (uintptr_t)p & ~((uintptr_t)ALLOC_MIN_MMAP_SIZE - 1u));
	ptrdiff_t pdiff = static_cast<char *>(p) - mp;
	unsigned int unit_idx =
	    static_cast<unsigned int>((size_t)pdiff >> ALLOC_MIN_CHUNK_SHIFT);
	char *chunk_base;
	PoolAllocatorBase *palloc = resolve_chunk_from_slot(
	    mp, (size_t)kind * NUM_ALLOCATORS_IN_SPACE, unit_idx, &chunk_base);
	if( !palloc) return 0;
	// Dedicated single-slot large chunk has no SizeOfFn — its size_info
	// is the DEDICATED sentinel; the total byte size lives at [32..39].
	// (§15) Payload starts at chunk_base + K_MAX (the unit boundary) and
	// ends K_MAX before the chunk's last unit boundary (reserved for the
	// next chunk's metadata), so usable payload = total − K_MAX.
	if(static_cast<std::uint32_t>(*reinterpret_cast<std::uint64_t *>(
	       chunk_base + ALLOC_CHUNK_HEADER_SIZE_INFO_OFFSET))
	   == ALLOC_CHUNK_DEDICATED_SIZEINFO) {
		size_t bytes = (size_t)*reinterpret_cast<std::uint64_t *>(
		    chunk_base + ALLOC_CHUNK_HEADER_DEDICATED_SIZE_OFFSET);
		return bytes - (size_t)ALLOC_CHUNK_K_MAX;
	}
	SizeOfFn fn = *reinterpret_cast<SizeOfFn *>(
	    chunk_base + ALLOC_CHUNK_HEADER_SIZEOF_FN_OFFSET);
	return fn(palloc, static_cast<char *>(p));
}
// Runtime N-unit chunk claim — see the header declaration.  The SINGLE
// region-walk + mmap + bitmap-claim site (§74): returns the raw chunk_base
// address (the caller writes the chunk_header).  back_off_flag is OR'd into
// every s_back_offset entry — 0 for the compile-time bucket templates
// (`allocate_chunk<ALLOC>()` calls with chunk_units = ALLOC::CHUNK_UNITS),
// 0x80 to tag a dedicated single-slot large chunk (`allocate_dedicated_chunk`).
char *
PoolAllocatorBase::claim_chunk(unsigned chunk_units,
                               std::uint8_t back_off_flag) noexcept {
	const BitmapWord occ_mask =
	    (chunk_units >= (unsigned)BITS_PER_BITMAP_WORD)
	        ? ~BitmapWord(0)
	        : ((BitmapWord(1) << chunk_units) - BitmapWord(1));
	auto try_claim_in_region = [&](RegionMeta *rmeta) -> char * {
		// (§13.2/§13.3) claim_bitmap + back_offset live in RegionMeta at
		// `region_base + 0`; the region base IS the RegionMeta pointer.
		char *mp = reinterpret_cast<char *>(rmeta);
		for(int word = 0; word < BITMAP_WORDS_PER_REGION; ++word) {
			std::atomic<BitmapWord> *bm = &rmeta->claim_bitmap[word];
			for(;;) {
				BitmapWord v = bm->load(std::memory_order_relaxed);
				int free_pos = -1;
				for(unsigned k = 0;
				    k + chunk_units <= (unsigned)BITS_PER_BITMAP_WORD;
				    k += chunk_units) {
					if(((v >> k) & occ_mask) == 0) { free_pos = (int)k; break; }
				}
				if(free_pos < 0) break;
				BitmapWord newv = v | (occ_mask << free_pos);
				int base_unit_idx = word * UNITS_PER_BITMAP_WORD + free_pos;
				if(bm->compare_exchange_strong(v, newv,
				                               std::memory_order_acquire,
				                               std::memory_order_relaxed)) {
					for(unsigned u = 0; u < chunk_units; ++u)
						rmeta->back_offset[base_unit_idx + (int)u] =
						    (std::uint8_t)u | back_off_flag;
					// (§15) chunk_base = unit_boundary - K_MAX, see
					// allocate_chunk's sibling site for rationale.
					return mp + (size_t)base_unit_idx
					          * (size_t)ALLOC_MIN_CHUNK_SIZE
					     - (size_t)ALLOC_CHUNK_K_MAX;
				}
			}
		}
		return nullptr;
	};
	// (§13.3) Retry loop — see allocate_chunk for the rationale (a fresh
	// region can be swarmed by other threads before our own claim).
	for(;;) {
		// (§14C) Pass 1: walk LOCAL NUMA-node's list first, fall back to
		// other nodes.
		int my_node = numa_node_for_this_thread();
		int total_nodes = s_num_numa_nodes.load(std::memory_order_relaxed);
		if(total_nodes <= 0) total_nodes = 1;
		for(int off = 0; off < total_nodes; ++off) {
			int n = (off == 0) ? my_node : ((my_node + off) % total_nodes);
			for(RegionMeta *rm = s_region_dll_heads[n].load(
			        std::memory_order_acquire);
			    rm; rm = rm->dll_next.load(std::memory_order_acquire)) {
				if( !rm->has_free.load(std::memory_order_relaxed)) continue;
				if(char *addr = try_claim_in_region(rm)) return addr;
				rm->has_free.store(0, std::memory_order_relaxed);
			}
		}
		// Pass 2: mmap a fresh region, then claim in it.
		RegionMeta *rm = mmap_new_region();
		if( !rm) return nullptr;
		if(char *addr = try_claim_in_region(rm)) return addr;
		// Swarmed before our claim — loop back to Pass 1.
	}
}

// (§22) Public forwarder to the protected `deallocate_chunk`, used by the
// large-recycle cache to truly release a recycled dedicated chunk on
// eviction / thread-exit.  reclaim_pages defaults to true (madvise), so a
// cached-then-evicted chunk returns its physical pages exactly as a normal
// dedicated free would.
void
PoolAllocatorBase::recycle_release_chunk(char *chunk_base,
                                         std::size_t chunk_size) noexcept {
	deallocate_chunk(chunk_base, chunk_size);
}

// (§34) Re-stamp back_offset[] for all units of the chunk to carry the
// consumer tier's tag.  See header doc.
void
PoolAllocatorBase::restamp_back_offset(char *chunk_base,
                                       std::size_t chunk_size,
                                       std::uint8_t back_off_flag) noexcept {
	RegionMeta *rmeta = region_meta_of(chunk_base);
	// chunk_base = unit_boundary - K_MAX; +K_MAX recovers the unit-aligned
	// slot-region start, mask to region offset, shift to unit index (same
	// derivation as deallocate_chunk).
	uintptr_t slot_region_start = (uintptr_t)chunk_base + ALLOC_CHUNK_K_MAX;
	unsigned int base_unit_idx = static_cast<unsigned int>(
	    (slot_region_start & ((uintptr_t)ALLOC_MIN_MMAP_SIZE - 1u))
	    >> ALLOC_MIN_CHUNK_SHIFT);
	unsigned chunk_units = static_cast<unsigned>(
	    chunk_size >> ALLOC_MIN_CHUNK_SHIFT);
	for(unsigned u = 0; u < chunk_units; ++u)
		rmeta->back_offset[base_unit_idx + u] =
		    static_cast<std::uint8_t>(u) | back_off_flag;
}

// (§34) Bucket-chunk release: park in LRC_CHUNK (units stay claimed, no
// madvise) when possible, else true-release via deallocate_chunk.  See
// header doc for the no-recurse contract.
void
PoolAllocatorBase::bucket_release_chunk(char *chunk_base,
                                        std::size_t chunk_size) noexcept {
	// lrc_block_size(LRC_CHUNK) reads the DEDICATED_SIZE header field, so
	// stamp it before pushing — a bucket chunk doesn't otherwise carry it.
	// Offset 32 lives in the [0,63] chunk-header region, separate from the
	// (already-destructed) PoolAllocator object at chunk_base + 64, so this
	// write is safe.
	*reinterpret_cast<std::uint64_t *>(
	    chunk_base + ALLOC_CHUNK_HEADER_DEDICATED_SIZE_OFFSET) =
	    (std::uint64_t)chunk_size;
	if(large_recycle_push(chunk_base, chunk_size, LRC_CHUNK))
		return;                         // parked warm — units stay claimed
	// Cache full / over cap → real release (bitmap-clear + madvise).
	deallocate_chunk(chunk_base, chunk_size);
}

void *
PoolAllocatorBase::allocate_dedicated_chunk(std::size_t size) noexcept {
	// (§15) Under the forward-shift layout, chunk_base sits K_MAX bytes
	// BEFORE its first claimed unit's boundary; the payload starts at
	// chunk_base + K_MAX = unit_boundary (256 KiB-aligned), mirroring the
	// regular slot region.  The metadata (chunk_header) occupies the K_MAX
	// region before the boundary, and the LAST K_MAX of the final claimed
	// unit is reserved for the NEXT chunk's metadata (exactly as a regular
	// chunk reserves chunk_size - K_MAX for its slots).  So the usable
	// payload of a chunk_units-unit chunk is `chunk_units*256K - K_MAX`,
	// and the units needed for `size` bytes is ceil((size + K_MAX)/256K).
	//
	// PRE-§15 used `size + ALLOC_CHUNK_HEADER` and returned
	// `chunk_base + ALLOC_CHUNK_HEADER` (header at the start of the
	// payload).  Returning `chunk_base + ALLOC_CHUNK_HEADER` under §15
	// hands back a pointer in the PREVIOUS unit, which `deallocate`
	// mis-resolves (its back_offset[unit-1] lookup lands on a neighbouring
	// chunk) — the source of the size > 17400 free-path SEGV.
	size_t total = size + (size_t)ALLOC_CHUNK_K_MAX;
	unsigned chunk_units = (unsigned)((total + (size_t)ALLOC_MIN_CHUNK_SIZE - 1)
	                                  >> ALLOC_MIN_CHUNK_SHIFT);
	if(chunk_units == 0) chunk_units = 1;
	if(chunk_units > (unsigned)ALLOC_MAX_CHUNK_UNITS)
		return nullptr;   // > 4 MiB payload — caller falls to std::malloc
	size_t chunk_size = (size_t)chunk_units * (size_t)ALLOC_MIN_CHUNK_SIZE;
	// (§22) Recycle a warm cached chunk of the same size class first: its
	// units are still claimed and its chunk_header (DEDICATED sentinel +
	// palloc + size) is intact, so we skip claim_chunk AND the page refault
	// entirely and hand back the payload directly.  pop_fit's [need, 2*need]
	// window means the cached chunk's actual size (kept in DEDICATED_SIZE)
	// is ≥ this request — size_of stays truthful via that field.
	if(char *cached = large_recycle_pop(chunk_size, LRC_CHUNK)) {
		// (§34) The shared LRC_CHUNK slot may now hand back a BUCKET-origin
		// block (units claimed, but back_off tagged plain `u` and the
		// header holding a stale PoolAllocator size_info/palloc instead of
		// the DEDICATED sentinel).  Pre-§34 this path returned `cached +
		// K_MAX` directly, trusting an intact dedicated header — no longer
		// safe.  Re-stamp unconditionally (idempotent for dedicated-origin
		// blocks, corrective for bucket-origin).
		//
		// Use the block's ACTUAL size (from the DEDICATED_SIZE header
		// field, which both tiers keep truthful: dedicated writes it at
		// alloc, bucket at release) rather than the request-derived
		// `chunk_size` — the dedicated fit window is [need, 2*need], so the
		// popped block can be larger than the request, and size_of must
		// stay truthful.  (Read inline: `lrc_block_size` is defined in the
		// anonymous namespace below this point; for LRC_CHUNK it is exactly
		// this field read.)
		std::size_t actual = (std::size_t)*reinterpret_cast<std::uint64_t *>(
		    cached + ALLOC_CHUNK_HEADER_DEDICATED_SIZE_OFFSET);
		restamp_back_offset(cached, actual, /*back_off_flag=*/0x80u);
		*reinterpret_cast<std::uint64_t *>(
		    cached + ALLOC_CHUNK_HEADER_SIZE_INFO_OFFSET) =
		    (std::uint64_t)ALLOC_CHUNK_DEDICATED_SIZEINFO;
		*reinterpret_cast<PoolAllocatorBase **>(
		    cached + ALLOC_CHUNK_HEADER_PALLOC_OFFSET) =
		    reinterpret_cast<PoolAllocatorBase *>(cached);
		*reinterpret_cast<std::uint64_t *>(
		    cached + ALLOC_CHUNK_HEADER_DEDICATED_SIZE_OFFSET) =
		    (std::uint64_t)actual;
		// (§hot-tls) Zero the would-be m_owner_id slot in the K_MAX gap so
		// the deallocate hot path's owner-id compare cannot match (real
		// owner_ids are non-zero).  Lets `deallocate` skip the bit-7 check
		// on the hot path and detect dedicated chunks via the natural
		// owner-id mismatch.  Bucket-origin recycled blocks may carry a
		// stale m_owner_id here; restamp it unconditionally.
		reinterpret_cast<PoolAllocatorBase *>(
		    cached + ALLOC_CHUNK_HEADER)->m_owner_id = 0;
		writeBarrier();
		// (§28.5) dedicated_chunk_bytes is now walked on demand; no
		// per-alloc counter to bump here.
		(void)actual;
		return cached + ALLOC_CHUNK_K_MAX;
	}
	// Claim N units, tagging back_off with bit7 so deallocate / size_of
	// detect the dedicated chunk from the already-loaded s_back_offset
	// byte (no chunk_header read on the hot free path — preserves the
	// (1b) cache-line discipline).
	char *chunk_base = claim_chunk(chunk_units, (std::uint8_t)0x80u);
	if( !chunk_base) return nullptr;
	// chunk_header: DEDICATED size_info sentinel + total bytes.  palloc =
	// chunk_base (non-null, > 1) so the foreign-pointer guard treats it as
	// ours; the dedicated free path never dereferences it as a PoolAllocator.
	*reinterpret_cast<std::uint64_t *>(
	    chunk_base + ALLOC_CHUNK_HEADER_SIZE_INFO_OFFSET) =
	    (std::uint64_t)ALLOC_CHUNK_DEDICATED_SIZEINFO;
	*reinterpret_cast<PoolAllocatorBase **>(
	    chunk_base + ALLOC_CHUNK_HEADER_PALLOC_OFFSET) =
	    reinterpret_cast<PoolAllocatorBase *>(chunk_base);
	*reinterpret_cast<std::uint64_t *>(
	    chunk_base + ALLOC_CHUNK_HEADER_DEDICATED_SIZE_OFFSET) =
	    (std::uint64_t)chunk_size;
	// (§hot-tls) Zero the would-be m_owner_id slot in the K_MAX gap so
	// the deallocate hot path's owner-id compare never matches for this
	// chunk.  `claim_chunk` returns mmap-fresh memory (zero-init) on the
	// truly-first claim, but the unit may have been previously held by a
	// bucket chunk that left a non-zero m_owner_id behind.  Stamp it
	// unconditionally — single uint32 store, negligible.
	reinterpret_cast<PoolAllocatorBase *>(
	    chunk_base + ALLOC_CHUNK_HEADER)->m_owner_id = 0;
	writeBarrier();
	// (§28.5) dedicated_chunk_bytes is now walked on demand; no per-alloc
	// counter to bump here.
	(void)chunk_size;
	// (§15) Payload starts at the unit boundary = chunk_base + K_MAX, so
	// `deallocate` resolves unit_idx = base_unit, back_off = 0, recovering
	// chunk_base via `unit_boundary - K_MAX`.
	return chunk_base + ALLOC_CHUNK_K_MAX;
}

// Forward declarations — `libsystem_malloc_for_pool` and `kame_calloc`
// are defined further down (alongside `libsystem_free_for_pool` /
// `libsystem_realloc_for_pool` and the rest of the libc-fallback set),
// but several earlier fall-through paths (allocate_large_size_or_malloc,
// cold_first_access, new_redirected_large, and our Linux strong-symbol
// `malloc` / `calloc` shims) need to call them.  Same pattern as the
// existing forward decl of `libsystem_free_for_pool` below.
__attribute__((noinline))
static void *libsystem_malloc_for_pool(std::size_t n);
__attribute__((noinline))
static void *kame_calloc(std::size_t n_elem, std::size_t sz);

void* allocate_large_size_or_malloc(size_t size) throw() {
	// Three-tier above-bucket dispatch:
	//
	//   1. size ≤ 4 MiB − K_MAX  →  `allocate_dedicated_chunk` (§15):
	//      a multi-unit chunk inside the existing 32-MiB region pool.
	//      Many such allocs share radix slots / NUMA hints / DLL state
	//      with regular bucket chunks — best locality for moderate-large
	//      sizes.
	//
	//   2. 4 MiB − K_MAX < size, mmap_size ≤ LRC_HI (= 256 MiB)  →  (§19)
	//      `allocate_large_va`: a 32-MiB-aligned mmap per alloc (a single
	//      32-MiB region for ≤ 32 MiB, a multi-region span up to 256 MiB),
	//      registered as a single KAME_RADIX_LARGE radix slot (the head
	//      32-MiB slot for spans), served from the §25/§26/§35 warm recycle
	//      cache.  Returns VA to the OS on free (munmap), unlike pool
	//      regions which are push-only.  Spans are safe in the radix: the
	//      alloc's sole valid user pointer is `base + PAGE`, which always
	//      resolves to the head slot; tail slots are never standalone
	//      `radix_lookup` targets (interior-pointer lookup into one alloc is
	//      UB caller-side), and the OS keeps the whole span mapped so no
	//      other alloc can claim a tail slot's VA.
	//
	//   3. (§27/§35) size so large that mmap_size > LRC_HI (= 256 MiB)  →
	//      ALSO `allocate_large_va` (same multi-region span), but the huge
	//      tier BYPASSES the warm cache (mmap fresh / munmap on free) — see
	//      allocate_large_va for why (the cache index tops out at LRC_HI;
	//      above it every size collapses to one unbounded top slot, so a
	//      cached huge block would over-satisfy smaller huge requests and
	//      pin RSS).
	//
	// libc malloc is reached ONLY if `allocate_large_va` returns nullptr
	// (the mmap itself failed) — no longer the routine path for huge sizes.
	//
	// Reached only when the allocator is active (new_redirected_large
	// gates on g_sys_image_loaded / s_alloc_tls_off before calling us).
	if(size <= (size_t)ALLOC_MAX_CHUNK_SIZE - (size_t)ALLOC_CHUNK_K_MAX) {
		if(void *p = PoolAllocatorBase::allocate_dedicated_chunk(size))
			return p;
	}
	if(void *p = PoolAllocatorBase::allocate_large_va(size))
		return p;
	// Last-resort libsystem fallback.  Use `libsystem_malloc_for_pool`
	// (not `std::malloc`) because our strong-symbol `malloc` override
	// would recurse otherwise — same pattern as libsystem_free_for_pool.
	return libsystem_malloc_for_pool(size);   // reached only if the mmap itself failed
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
//! `PunType` matches the `s_tls.my_chunk` declaration in the bucket's
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
// extend the FS=true 16-step ladder to buckets 17..23
// (sizes 272..368).  Each is FS=true with ALIGN = SIZE → zero
// internal frag (one slot per ALIGN-byte region).  Closes the
// 257..368 gap that the earlier change's bucket 17 (slot 384) absorbed at
// up to 32 % frag for the small end.
KAME_DECL_BUCKET(17, ALLOC_SIZE17,                 true,  ALLOC_SIZE17);  // 272
KAME_DECL_BUCKET(18, ALLOC_SIZE18,                 true,  ALLOC_SIZE18);  // 288
KAME_DECL_BUCKET(19, ALLOC_SIZE19,                 true,  ALLOC_SIZE19);  // 304
KAME_DECL_BUCKET(20, ALLOC_SIZE20,                 true,  ALLOC_SIZE20);  // 320
KAME_DECL_BUCKET(21, ALLOC_SIZE21,                 true,  ALLOC_SIZE21);  // 336
KAME_DECL_BUCKET(22, ALLOC_SIZE22,                 true,  ALLOC_SIZE22);  // 352
KAME_DECL_BUCKET(23, ALLOC_SIZE23,                 true,  ALLOC_SIZE23);  // 368

// Buckets 24..47: 4-way
// exponential FS=false ladder.  3 ALIGN stages × 8 (octave/sub) =
// 24 buckets.  The "borrow" header is the universal 8 B at p_user - 8
//.  Bucket `SIZE` is the MAX user_size the bucket serves
// (= slot total - 8); slow_allocate / bucket_first_access pass SIZE
// to allocate_pooled, which computes N = ceil((SIZE+8)/ALIGN).
// Slot total uniformly = (N+1)*ALIGN per an earlier change.

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

// Stage 3 — ALIGN=1024, (§16) FULL-USABLE mode.  No borrow theft: SIZE =
// slot = N × 1024.  allocate_pooled stores {N,local_id} in the chunk's
// m_sizes[] array, so the whole N*1024-byte slot is user-usable.
KAME_DECL_BUCKET(40, 1024u, false,  6144u);  // N=6,  slot= 6144
KAME_DECL_BUCKET(41, 1024u, false,  7168u);  // N=7,  slot= 7168
KAME_DECL_BUCKET(42, 1024u, false,  8192u);  // N=8,  slot= 8192
KAME_DECL_BUCKET(43, 1024u, false,  9216u);  // N=9,  slot= 9216
KAME_DECL_BUCKET(44, 1024u, false, 11264u);  // N=11, slot=11264
KAME_DECL_BUCKET(45, 1024u, false, 13312u);  // N=13, slot=13312
KAME_DECL_BUCKET(46, 1024u, false, 15360u);  // N=15, slot=15360
KAME_DECL_BUCKET(47, 1024u, false, 17408u);  // N=17, slot=17408

// Stage 4 — ALIGN=4096 page-aligned tier, (§16) FULL-USABLE.  Power-of-2
// slot sizes (4K, 8K, 16K, 32K) — every slot is a multiple of every
// common page size (4/16/32/64 KiB), keeping MADV_FREE granularity, TLB
// coverage and THP behaviour clean across platforms.  Full-usable: SIZE =
// slot = N × 4096, the entire slot is user-usable (the m_sizes mode kills
// the 50 % round-up that the borrow scheme caused at power-of-2 sizes).
// Routing (see `bucket_for_size`):
//   - bucket 48 catches 3833..4096   (vs borrow bucket 39 slot 4352)
//   - bucket 50 catches 15361..16384 (vs full bucket 47 slot 17408)
//   - bucket 51 extends the pool from 17408 to 32768 (was libc malloc)
//   - bucket 49 (8192) ties full bucket 42 (8192); plain malloc prefers
//     42 (denser ALIGN=1024 chunks), so 49 is reached only via a future
//     large-alignment posix_memalign/aligned_alloc path.
KAME_DECL_BUCKET(48, 4096u, false,  4096u);  // N=1, slot= 4096
KAME_DECL_BUCKET(49, 4096u, false,  8192u);  // N=2, slot= 8192
KAME_DECL_BUCKET(50, 4096u, false, 16384u);  // N=4, slot=16384
KAME_DECL_BUCKET(51, 4096u, false, 32768u);  // N=8, slot=32768
#undef KAME_DECL_BUCKET

//! First-access trampoline for bucket B.  Invoked from the
//! `cold_first_access` switch when bucket B's per-thread freelist-ptr
//! slot (`m_slots[B].freelist_head`) is unset.  Claims a chunk via the
//! existing `allocate<>()` slow path (which registers
//! AllocThreadExitCleanup) and wires that slot to the chunk's
//! `m_freelist_head[]` so subsequent freelist-miss calls go straight to
//! the chunk vtable path (`PoolAllocatorBase::slow_allocate`) and never
//! come back through `cold_first_access`.
template <int B>
__attribute__((noinline))
void *bucket_first_access(std::size_t /*size*/) noexcept {
    using BT = BucketTraits<B>;
    using PA = typename BT::PoolType;
    void *p = PA::template allocate<BT::SIZE>();
    PoolAllocatorBase *chunk = PA::get_pinned_chunk_base();
    if(chunk) {
        // (§12.3 / §hot-tls) Wire up the KameTlsPage slot so subsequent
        // allocs on this bucket hit the freelist directly.  kBucketLocalId
        // is constexpr-foldable here (B is a template parameter).
        // chunk_from_freelist_ptr recovers the chunk pointer from the
        // stored value via a single mask on the slow path.
#if KAME_FS_TWOLIST
        // (§two-list) First-touch activation runs BEFORE any slow_allocate
        // re-aim, so without this gate the slot stays on [0] forever (free
        // feeds [0], pops from [0] always succeed, the [1]/bump/refill
        // tiers never execute) and the gate measures bit-identical to OFF
        // — caught on Ohtaka when 4 benches matched gate-off to 0.1%.
        kame_page()->m_slots[B].freelist_head =
            reinterpret_cast<char *>(
                chunk->m_fs_flag ? &chunk->m_freelist_head[1]
                                 : &chunk->m_freelist_head[kBucketLocalId[B]]);
#else
        kame_page()->m_slots[B].freelist_head =
            reinterpret_cast<char *>(&chunk->m_freelist_head[kBucketLocalId[B]]);
#endif
    }
    return p;
}

} // anon namespace

// Cold path entry point used by `new_redirected` when bucket's
// per-thread freelist-ptr slot (`m_slots[bucket].freelist_head`) is
// unset.  Handles three states:
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
//      registers AllocThreadExitCleanup, and wires
//      `m_slots[B].freelist_head`.
//
// `__attribute__((cold))`: clang places this out-of-line so the
// freelist-miss path in `new_redirected` doesn't bloat its branch
// distance budget.  The switch lowers to a jump table on arm64.
__attribute__((cold, noinline))
void *cold_first_access(unsigned bucket, std::size_t size) noexcept {
    if( !g_sys_image_loaded || kame_thread_torn_down())
        return libsystem_malloc_for_pool(size);  // not std::malloc — would recurse under strong-symbol `malloc` override
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
        case 48:          return bucket_first_access<48>(size);
        case 49:          return bucket_first_access<49>(size);
        case 50:          return bucket_first_access<50>(size);
        case 51:          return bucket_first_access<51>(size);
    }
    return libsystem_malloc_for_pool(size);  // unreachable
}

// (§hot-tls) KameTlsPage — unified per-thread hot TLS page.
// Replaces the retired `g_thread_slots[]` (defunct since §12.3) and
// `g_thread_freelist_ptr[]` with a single struct accessed via one
// fast-TSD read (macOS) or one IE TLS reference (Linux).
//
// `g_tls_page` is:
//   macOS: ALLOC_TLS (global-dynamic) — too large for the IE surplus
//          budget (432 B > ~128 B typical slack); the address is cached
//          in the IE pointer `tls_page_ie` by kame_tls_init_fast and
//          kame_page_cold (paid once per thread on first alloc).
//   Linux: ALLOC_TLS_IE (initial-exec) — inlines to mov %fs:offset for
//          each field access, identically to the old per-variable IE TLS.
//
// `g_tls_page.last_region_base` is initialised to RADIX_CACHE_EMPTY (all-ones).
// `g_tls_page.owner_id` defaults to 0 (unassigned).
// `g_tls_page.m_slots[]` defaults to all-zeros (nullptr freelist heads).
#if KAME_FAST_TSD
ALLOC_TLS    KameTlsPage  g_tls_page  = {RADIX_CACHE_EMPTY, 0, 0, {}};
ALLOC_TLS_IE KameTlsPage *tls_page_ie = nullptr;
#else
ALLOC_TLS_IE KameTlsPage  g_tls_page  = {RADIX_CACHE_EMPTY, 0, 0, {}};
#endif

// (§hot-tls teardown sentinel) NOT thread-local: one process-global page,
// never freed, owner_id == 0.  See allocator_prv.h for the two-role rationale.
// owner_id 0 guarantees the hot owner-check never matches it; the cold dealloc
// path identity-compares against `&g_teardown_page` to take a TLS-free route.
KameTlsPage g_teardown_page = {RADIX_CACHE_EMPTY, 0, 0, {}};

// Cold off-ramp for the lean freelist-pop entries (`new_redirected` and
// `new_redirected_large`; declared in allocator_prv.h): an empty owner
// freelist (re-resolve the chunk and slow_allocate via its vtable), a
// not-yet-activated bucket (first access / post-cleanup), or a
// first-touch thread (kame_page_or_null() bailed — the kame_page() call
// below runs the full TSD cold-init).  `bucket` is pre-computed by the
// lean caller and may be any bucketed class (small formula range or the
// 369..32768 LUT range).  KAME_NOINLINE so the lean callers TAIL-CALL it —
// keeping their freelist-pop hot paths frame-free.
KAME_NOINLINE
void *new_redirected_cold(unsigned int bucket, std::size_t size) {
	if(char *cell_ptr_raw = kame_page()->m_slots[bucket].freelist_head) {
		char **head_ptr = reinterpret_cast<char **>(cell_ptr_raw);
		if(char *head = *head_ptr) {
			*head_ptr = *reinterpret_cast<char **>(head);
			return head;
		}
		PoolAllocatorBase *ck = chunk_from_freelist_ptr(head_ptr);
#if KAME_FS_TWOLIST
		// (§two-list) refill before the slow path: swap the free side in
		// as the next segment / extend the virgin window.
		if(ck->m_fs_flag)
			if(void *p = ck->fs_twolist_refill_take())
				return p;
#endif
		return ck->slow_allocate(bucket, size);
	}
	return cold_first_access(bucket, size);
}

// Cold tail for `new_redirected_large`: sizes above the bucketed pool
// ceiling (> 32 KiB) — activation gate + dedicated/large_va/libc routing.
// Out-of-line so the lean entry below tail-calls it.
KAME_NOINLINE_COLD
static void *new_redirected_large_tail(std::size_t size) noexcept {
    if( !g_sys_image_loaded || kame_thread_torn_down())
        return libsystem_malloc_for_pool(size);
    return allocate_large_size_or_malloc(size);
}

// Out-of-line FS=false-and-up dispatch.  Sizes > 368 B fall here from
// `new_redirected`.  LEAN body: the whole bucketed range (369..32768)
// resolves via bucket_for_size (formula / one LUT byte — no ladder call
// since the 32 KiB LUT extension) and pops the per-thread freelist.  Every
// off-ramp is a TAIL-CALL and the page read is the no-cold-init
// `kame_page_or_null()`, so the hit path is CALL-FREE — no callee-saved
// spill, no frame (the former body shared a 2×stp frame with its
// slow_allocate/cold_first_access calls on every steady-state hit).
//   • > 32768                  → new_redirected_large_tail (gate + large_va)
//   • first touch / slot empty → new_redirected_cold(bucket, size), which
//     runs the full kame_page() init, the activation-gated
//     cold_first_access, or the chunk-vtable slow_allocate.
void *new_redirected_large(std::size_t size) noexcept {
    if(__builtin_expect(size > ALLOC_MAX_BUCKETED_SIZE, 0))
        return new_redirected_large_tail(size);
    // (§12.3 / §hot-tls) Mirror new_redirected's direct-jump fast path
    // via the KameTlsPage slot.  m_slots[bucket].freelist_head stores the
    // char ** pointer (cast to char *) to the active chunk's freelist cell.
    unsigned int bucket = bucket_for_size(size);
    KameTlsPage *pg = kame_page_or_null();
    if(__builtin_expect(pg != nullptr, 1)) {
        if(char *cell_ptr_raw = pg->m_slots[bucket].freelist_head) {
            char **head_ptr = reinterpret_cast<char **>(cell_ptr_raw);
            if(char *head = *head_ptr) {
                *head_ptr = *reinterpret_cast<char **>(head);
                return head;
            }
        }
    }
    return new_redirected_cold(bucket, size);
}

// (§17) Aligned-allocation entry point — pool-or-libc dispatch on
// (alignment, size).  alignment is power-of-two, > ALLOC_ALIGNMENT.
//
// Routing:
//   1. bucket_for_aligned(A, S) finds the smallest pool bucket with
//      ALIGN ≥ A (A divides ALIGN) and usable ≥ S.  Hit → allocate via the
//      bucket's freelist-or-slow_allocate path (same as new_redirected_large
//      for that bucket), and the returned slot is ALIGN-aligned (hence
//      A-aligned because A divides ALIGN).
//   2. No bucket fits but A ≤ ALLOC_MIN_CHUNK_SIZE (256 KiB) and S fits in
//      a multi-unit chunk: route to allocate_dedicated_chunk.  Its payload
//      starts at a 256 KiB unit boundary (§15), which is A-aligned for
//      every A up to 256 KiB.
//   3. Otherwise → libc posix_memalign.
//
// Pre-activation / post-teardown: like new_redirected, fall through to
// posix_memalign so TLS dtor-time allocs etc. stay safe.
void *new_redirected_aligned(std::size_t alignment, std::size_t size) noexcept {
    // bucket_for_aligned returns ALLOC_NUM_BUCKETS when no bucket fits.
    unsigned int bucket = bucket_for_aligned(alignment, size);
    if(bucket < (unsigned)ALLOC_NUM_BUCKETS &&
       g_sys_image_loaded && !kame_thread_torn_down()) {
        // Pool bucket path — mirrors new_redirected_large's freelist /
        // slow_allocate / cold_first_access cascade via KameTlsPage.
        if(char *cell_ptr_raw = kame_page()->m_slots[bucket].freelist_head) {
            char **head_ptr = reinterpret_cast<char **>(cell_ptr_raw);
            if(char *head = *head_ptr) {
                *head_ptr = *reinterpret_cast<char **>(head);
                return head;
            }
            return chunk_from_freelist_ptr(head_ptr)->slow_allocate(bucket, size);
        }
        return cold_first_access(bucket, size);
    }
    // Pool can't serve this combo via a bucket (alignment > 4096, or size
    // too big for any matching bucket, or pre-activate / post-teardown).
    // Try the next step — a dedicated chunk — for alignment up to 256 KiB.
    if(g_sys_image_loaded && !kame_thread_torn_down() &&
       alignment <= (std::size_t)ALLOC_MIN_CHUNK_SIZE &&
       size <= (std::size_t)ALLOC_MAX_CHUNK_SIZE - (std::size_t)ALLOC_CHUNK_K_MAX) {
        // Dedicated-chunk path — §15 payload starts at a 256 KiB unit
        // boundary, satisfying any alignment up to 256 KiB.
        if(void *p = PoolAllocatorBase::allocate_dedicated_chunk(size))
            return p;
    }
    // Note: the large_va tier (4 MiB–32 MiB) and huge tier (> 32 MiB)
    // are NOT a useful next step for aligned alloc, even though every
    // large_va mmap *base* is 32 MiB-aligned.  Reason: `allocate_large_va`
    // returns `base + ALLOC_PAGE_SIZE` to the user — the first page of
    // every large_va block holds the in-band `LargeAllocMeta`.  The
    // exposed user pointer is therefore only PAGE-aligned, not 32 MiB-
    // aligned, so escalating here gives no alignment benefit beyond what
    // the bucket tier (ALIGN=4096) already provides.  Achieving 32 MiB
    // alignment on the user pointer would require moving LargeAllocMeta
    // to the tail and adjusting deallocate_large_va / radix lookup /
    // recycle cache — a non-trivial refactor that we defer until a
    // workload actually needs alignment > 256 KiB + pool management.
    //
    // Fall back to libc posix_memalign — handles alignment > 256 KiB
    // (the dedicated chunk's ceiling) and huge sizes the pool can't
    // serve.  Windows returns null here — `kame_pool_free` has no way to
    // pair with `_aligned_free` without per-pointer tagging, so we
    // surface that as `errno = ENOMEM` to the caller.  Use the
    // platform-native `_aligned_malloc` / `_aligned_free` directly for
    // those rare cases on Windows.
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
    (void)alignment;
    (void)size;
    return nullptr;
#else
    void *p = nullptr;
    if(posix_memalign(&p, alignment, size) != 0)
        return nullptr;
    return p;
#endif
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
static void libsystem_free_for_pool(void *p) noexcept;

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
	// `deallocate` is now void + self-contained: a foreign pointer is
	// libsystem-freed inside `deallocate_cold`, so there is no caller-side
	// fallback (and the tail-call keeps the hot path frame-free).
	PoolAllocatorBase::deallocate(p);
}

#if defined(__linux__) && defined(__GLIBC__)
// glibc internal entry point — same address as libc's `free` but with
// a name our strong-symbol `free` shim does not shadow.  Declared with
// the same signature as `free` so the call is ABI-compatible.
extern "C" void __libc_free(void *) noexcept;
#endif

static void libsystem_free_for_pool(void *p) noexcept {
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
	// Other platforms (musl, Windows).
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
	// Windows: the genuine UCRT free resolved by kame_resolve_real_crt —
	// NOT std::free, which the IAT redirect has repointed back into the
	// pool dispatcher (would recurse).  Falls back to std::free only in
	// the pre-install window, when nothing is patched yet.
	if(g_real_free) { g_real_free(p); return; }
#endif
	// musl: the strong-symbol shadowing rule is the same as glibc, so
	// this will recurse if KAME's pool is active — Linux non-glibc builds
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
static void kame_free(void *p) noexcept {
	// `PoolAllocatorBase::deallocate(p)` is itself pre-activation-safe:
	// it early-returns false on null `p`, and the CCNT=0 lookup against
	// `s_mmapped_spaces[0] == nullptr` (zero-initialised pre-pool-use)
	// trivially fails its range check.  No outer `g_sys_image_loaded`
	// guard needed — the natural state of `s_mmapped_spaces[]` covers
	// the same fast-out.  `deallocate` is void + self-contained (a foreign
	// pointer is libsystem-freed inside it), so no caller-side fallback.
	PoolAllocatorBase::deallocate(p);
}

// Shared body for the C `malloc` strong-symbol interpose and the
// `kame_pool_malloc` C-API entry — single source of truth for the
// pool-or-ENOMEM logic.  `new_redirected` self-gates for pre-activation /
// post-teardown via the null `g_thread_freelist_ptr[bucket]` path, so no
// `!g_sys_image_loaded || kame_thread_torn_down()` pre-filter is needed here (same
// as `operator new`).  `always_inline` so BOTH callers expand it directly:
// the hot `malloc` keeps its inlined freelist pop with NO extra call/PLT
// hop, while the source carries one copy.
// Cold continuation for `kame_malloc_impl`: the full size dispatch PLUS
// the C-malloc ENOMEM contract.  Folding the `errno` assignment in HERE —
// instead of null-checking in the caller — is what lets the lean malloc
// below tail-call on every miss: with no post-call work the hot path
// needs no frame at all (the VTune Zen 2 audit measured the former
// null-check + errno shape as a `push %rax`/`pop` pair plus a call that
// stayed on the hot side — ~2-3 of the ~10 glue insns/pair behind the
// 64 B gap vs mimalloc).
KAME_NOINLINE static void *kame_malloc_slow(std::size_t n) noexcept {
	void *p = (n > (std::size_t)ALLOC_SIZE23)
	              ? new_redirected_large(n)
	              : new_redirected_cold(
	                    static_cast<unsigned int>((n + 15u) >> 4), n);
	if(__builtin_expect(p == nullptr, 0))
		errno = ENOMEM;
	return p;
}

static __attribute__((always_inline)) inline
void *kame_malloc_impl(std::size_t n) noexcept {
	// LEAN: only the owner-freelist HIT is inlined (a leaf —
	// kame_page_or_null + pop, no call, no frame); EVERY miss (large
	// size, first touch, empty freelist, pre-activation / post-teardown
	// via the null slot or null page) tail-calls `kame_malloc_slow`,
	// which re-dispatches and owns the ENOMEM contract.  The large-size
	// branch hint mirrors `new_redirected`'s Apple-only policy (5e127eb5).
#if defined(__APPLE__)
	if(__builtin_expect(n > (std::size_t)ALLOC_SIZE23, 0))
#else
	if(n > (std::size_t)ALLOC_SIZE23)
#endif
		return kame_malloc_slow(n);
	unsigned int bucket = (static_cast<unsigned int>(n) + 15u) >> 4;
	KameTlsPage *pg = kame_page_or_null();
	if(__builtin_expect(pg != nullptr, 1)) {
		if(char *cell_ptr_raw = pg->m_slots[bucket].freelist_head) {
			char **head_ptr = reinterpret_cast<char **>(cell_ptr_raw);
			// (gate experiments) The 526e1819 lean split detached malloc
			// from `new_redirected`, which silently DROPPED the gated
			// ring/stash TAKE from the C-malloc path while the park side
			// (in `deallocate`) survived — parked slots were stranded
			// until the owner-exit drain (found on Ohtaka; 14c4f889
			// restored the STASH side, this adds the FIFO side so a ring
			// re-test needs no re-plumbing).  Prefer the parked slot: a
			// load on the already-hot chunk line, NO dereference of the
			// block itself.  Lean-path only — new_redirected / _cold /
			// _aligned still go straight to the plain pop (parked slots
			// there wait for the drain), acceptable for experiment gates.
			// With both gates OFF (the default) the preprocessor removes
			// this block and the body is bit-identical.
#if KAME_FS_CHUNK_FIFO
			PoolAllocatorBase *ck = chunk_from_freelist_ptr(head_ptr);
			if(ck->m_fs_flag) {
				std::uint32_t fr = ck->m_fifo.r;
				char *b0 = head_ptr[1 + (fr & 3u)];
				char *b1 = head_ptr[1 + ((fr + 1u) & 3u)];
				if(b0 && b1) {
					head_ptr[1 + (fr & 3u)] = nullptr;
					ck->m_fifo.r = fr + 1;
					return b0;
				}
			}
#elif KAME_FS_CHUNK_STASH
			if(chunk_from_freelist_ptr(head_ptr)->m_fs_flag) {
				if(char *b = head_ptr[1]) {
					head_ptr[1] = nullptr;
					return b;
				}
			}
#endif
			if(char *head = *head_ptr) {
				*head_ptr = *reinterpret_cast<char **>(head);
				return head;
			}
#if KAME_FS_TWOLIST
			// (§two-list) mirror of new_redirected's bump-window tier.
			{
				PoolAllocatorBase *ck = chunk_from_freelist_ptr(head_ptr);
				if(ck->m_fs_flag) {
					char *cur = ck->m_freelist_head[2];
					if(cur < ck->m_freelist_head[4]) {
						ck->m_freelist_head[2] = cur +
						    reinterpret_cast<uintptr_t>(ck->m_freelist_head[5]);
						return cur;
					}
				}
			}
#endif
		}
	}
	return kame_malloc_slow(n);
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
#elif defined(__linux__)
// Linux: emit `free` as a strong symbol so our dylib's own consumers
// resolve to the pool-aware version.  Also emit `malloc` / `calloc` here
// so a `-l kamepoolalloc` (or LD_PRELOAD) consumer gets a *complete* pool
// allocator — every libc malloc-family entry, including the ones inside
// libstdc++ / libc++ / Qt / Ruby / Python, routes through the pool.
// This matches mimalloc / jemalloc's Linux .so contract and is what
// makes head-to-head benchmarks (mimalloc-bench / LD_PRELOAD shootout)
// produce comparable numbers — without it, LD_PRELOAD'd consumers stay
// on libc malloc and the pool is bypassed entirely (only direct
// `kame_pool_*` calls land in the pool).
//
// Fall-through to libc: `libsystem_malloc_for_pool` uses `__libc_malloc`,
// which is the same address libc's malloc resolves to but under a name
// we do not shadow — preventing infinite recursion through our own
// override.
extern "C" __attribute__((noinline)) void free(void *p) noexcept {
	// noexcept end-to-end (free -> kame_free -> deallocate chain): no EH
	// barrier, so this compiles to a bare `jmp kame_free` tail-call — no
	// push/pop/ret, no __clang_call_terminate pad (the VTune Zen 2 audit
	// measured the former call-wrapper shape as ~5-6 of the ~10 glue
	// insns/pair behind the 64 B gap vs mimalloc, whose `vfree` is the
	// same tail-jump).
	kame_free(p);
}
extern "C" __attribute__((noinline)) void *malloc(std::size_t n) noexcept {
	// Strong-symbol interpose.  Body is the shared `kame_malloc_impl`,
	// `always_inline` so `new_redirected` expands directly here — no
	// call, no PLT hop on the alloc hot path.  See `kame_malloc_impl`
	// for why no activation-flag pre-filter is needed.
	return kame_malloc_impl(n);
}
extern "C" __attribute__((noinline)) void *calloc(std::size_t n_elem, std::size_t sz) noexcept {
	// `kame_calloc` is libc-spec compliant (overflow check, calloc(0,*)
	// returns unique freeable ptr, ENOMEM on fail) — just expose it.
	return kame_calloc(n_elem, sz);
}
#else
// Windows / others: no `free` interpose.
//
// Rationale: on PE/COFF (Windows), a DLL exporting `free` does NOT
// shadow other modules' bindings to msvcrt's `free` — each module
// has its own import table.  Worse, defining `free` in a static
// library would create a multiply-defined-symbol error against
// msvcrt's `free`.  The production Windows kame.exe inline-compiles
// `allocator.cpp` directly, so:
//   - C++ `operator new` / `operator delete` overrides apply (every
//     `new T()` / `delete p` in kame.exe is pool-routed);
//   - the C API (`kame_pool_*`) is available for explicit use;
//   - CRT `free` / `realloc` stay bound to msvcrt — what 3rd-party
//     DLLs expect.
//
// Cross-DLL risk: a kame.exe-allocated pool pointer handed to a
// 3rd-party DLL that calls CRT `free()` will crash with a heap-
// corruption check.  KAME's architecture does not do this; if a
// future call site needs it, the explicit `kame_pool_free` C API
// or `delete` operator should be used instead.
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
// glibc internal entries — same addresses as libc's `malloc` / `calloc` /
// `realloc` but under names our strong-symbol shims do not shadow.
extern "C" void *__libc_malloc(size_t) noexcept;
extern "C" void *__libc_calloc(size_t, size_t) noexcept;
extern "C" void *__libc_realloc(void *, size_t) noexcept;
#endif

__attribute__((noinline))
static void *libsystem_malloc_for_pool(std::size_t n) {
#if defined(__APPLE__)
	// Zone-API direct dispatch — same rationale as libsystem_free /
	// realloc / calloc above: dlsym(RTLD_NEXT, "malloc") would return
	// our interposed replacement under FULL_INTERCEPT.  malloc_zone_*
	// bypasses interposing and goes straight to the libsystem vtable.
	return malloc_zone_malloc(malloc_default_zone(), n);
#elif defined(__linux__) && defined(__GLIBC__)
	// Our strong-symbol `malloc` override would recurse; __libc_malloc is
	// the same address libc's malloc resolves to, under a name we do
	// not shadow.  Same trick as `__libc_free` / `__libc_realloc`.
	return __libc_malloc(n);
#else
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
	// Windows §31 by default doesn't redirect `malloc` — `std::malloc` is
	// safe to call directly.  Under KAMEPOOLALLOC_FULL_INTERCEPT it IS
	// patched, so use the genuine UCRT entry via `g_real_malloc` to avoid
	// recursing into our own override.  Pre-resolution falls back to
	// `std::malloc` (matches the §31 install-before-activation contract:
	// the patch is installed AFTER resolve_real_crt resolves these).
	if(g_real_malloc) return g_real_malloc(n);
#endif
	// Other Unix (musl, etc.): no strong-symbol malloc override is emitted
	// below, so std::malloc is safe.
	return std::malloc(n);
#endif
}

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
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
	// Genuine UCRT realloc — NOT the redirected one (would recurse).
	if(g_real_realloc) return g_real_realloc(p, n);
#endif
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
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
	// Genuine UCRT calloc — NOT the redirected path.
	if(g_real_calloc) return g_real_calloc(n_elem, sz);
#endif
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
	if( !g_sys_image_loaded || kame_thread_torn_down())
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
		if( !g_sys_image_loaded || kame_thread_torn_down())
			return libsystem_realloc_for_pool(nullptr, n);
		return new_redirected(n);
	}
	if( !n) {
		// `realloc(p, 0)` is implementation-defined in C17 (DR 400):
		// glibc/libc++ tend to return NULL and free `p`; some
		// allocators return a unique freeable pointer.  We pick the
		// "free + return NULL" semantics — same as mimalloc.
		// `PoolAllocatorBase::deallocate` is pre-activate-safe (same
		// rationale as `kame_free`).  `deallocate` is void + self-contained
		// (a foreign pointer is libsystem-freed inside it).
		PoolAllocatorBase::deallocate(p);
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
extern "C" __attribute__((used))
void *kame_pool_calloc(std::size_t n_elem, std::size_t sz) noexcept {
	return kame_calloc(n_elem, sz);
}

// =====================================================================
// an earlier change/follow-up: public C ABI (<kame_pool.h>).
//
// Thin extern-C wrappers over the existing internal entry points
// (`new_redirected` / `deallocate_pooled_or_free` / `kame_realloc` /
// `kame_calloc` / `PoolAllocatorBase::size_of`).  Each wrapper:
//   - is `extern "C"` so the symbol has no name mangling (`kame_pool_*`
//     mangle-free; usable from C, Rust, Go FFI, etc.);
//   - is `__attribute__((used))` so LTO does not strip it when no
//     in-binary consumer exists (the dylib is the consumer);
//   - sets `errno = ENOMEM` on allocation failure where the libc
//     contract calls for it (`malloc`/`calloc`/`realloc`/`aligned_alloc`);
//   - is fully reentrant — the underlying paths are lock-free per
//     thread plus single-CAS cross-thread frees.
//
// Pre-activation safety: `new_redirected` (via `cold_first_access` ->
// `new_redirected_large` fallback) and `PoolAllocatorBase::deallocate`
// / `size_of` are all safe to call before `activateAllocator()` has
// fired — they detect the inactive pool and route to libsystem
// malloc/free.  C API callers never need to coordinate with the C++
// activator.

// (§18) OOM helpers.  `new_redirected` / `new_redirected_large` /
// `new_redirected_aligned` return nullptr on failure (pool cap, mmap
// refusal, or libc fallback ENOMEM); they never throw.  The helpers
// below add the standard `std::get_new_handler()` retry loop on top.
//
//   `try_alloc_with_new_handler(fn)` — loop while a new_handler is
//   installed and returns control (it may free memory and let us
//   retry).  Returns nullptr only when no handler is installed.  If
//   the handler throws, the exception escapes — the throwing operator
//   new lets it propagate; the noexcept C wrappers and the
//   nothrow / non-throwing operator new variants wrap in try/catch.
//
// HOT/COLD split (mirrors the deallocate split — see KAME_ALWAYS_INLINE /
// KAME_NOINLINE_COLD in allocator_prv.h).  The previous shape was
//
//     inline void *try_alloc_with_new_handler(Fn fn) {
//         if(void *p = fn()) return p;
//         for(;;) { /* handler loop */ }
//     }
//     inline void *kame_alloc_with_handler(std::size_t s) {
//         return try_alloc_with_new_handler([=]{ return new_redirected(s); });
//     }
//
// which forced GCC to spill r12/r13/rbp across the cold handler loop and
// the lambda capture, fattening every `operator new` hot prologue from
// the `malloc` shim's `push %rbx` (2 inst) to push r13/r12/rbp/rbx + sub 8
// (5 inst).  Disassembly: operator new hot path 22 inst vs malloc 12 inst.
//
// New shape: the hot direct call stays in the (always-inlined) `..._hot`
// helper; the handler retry loop is split off to a `KAME_NOINLINE_COLD`
// function so the hot path's register pressure does not see it.  Each
// `operator new` variant calls hot first, hands off to cold only on the
// initial null.
namespace {
KAME_NOINLINE_COLD
void *try_alloc_with_new_handler_cold(void *(*alloc_fn)(std::size_t),
                                      std::size_t arg) noexcept(false) {
    for(;;) {
        std::new_handler h = std::get_new_handler();
        if( !h) return nullptr;
        h();  // may throw — propagates to caller
        if(void *p = alloc_fn(arg)) return p;
    }
}
KAME_NOINLINE_COLD
void *try_alloc_with_new_handler_aligned_cold(void *(*alloc_fn)(std::size_t,
                                                                 std::size_t),
                                              std::size_t a, std::size_t s)
                                              noexcept(false) {
    for(;;) {
        std::new_handler h = std::get_new_handler();
        if( !h) return nullptr;
        h();
        if(void *p = alloc_fn(a, s)) return p;
    }
}

// Hot wrappers — pure pass-through to the underlying alloc, then cold
// dispatch on null.  The plain function-pointer cold path means GCC
// no longer sees a lambda-captured `size` flowing across the cold call
// (the spills that bloated the hot prologue go away).
KAME_ALWAYS_INLINE
void *kame_alloc_with_handler(std::size_t size) {
    if(void *p = new_redirected(size)) return p;
    return try_alloc_with_new_handler_cold(&new_redirected, size);
}
KAME_ALWAYS_INLINE
void *kame_aligned_alloc_with_handler(std::size_t a, std::size_t s) {
    if(void *p = new_redirected_aligned(a, s)) return p;
    return try_alloc_with_new_handler_aligned_cold(&new_redirected_aligned,
                                                    a, s);
}
} // namespace

extern "C" __attribute__((noinline, used))
void *kame_pool_malloc(std::size_t size) noexcept {
	// (§18) C malloc semantics: no `std::get_new_handler()` (that's a
	// C++ operator-new concept); the shared `kame_malloc_impl` calls
	// `new_redirected` and sets `errno = ENOMEM` on null.  `create_allocator`
	// returns nullptr on pool OOM (no longer throws bad_alloc), so the
	// noexcept boundary is upheld without a try-block.
	return kame_malloc_impl(size);
}

extern "C" __attribute__((noinline, used))
void kame_pool_free(void *p) noexcept {
	deallocate_pooled_or_free(p);
}

extern "C" __attribute__((noinline, used))
void *kame_pool_realloc(void *p, std::size_t size) noexcept {
	void *q = kame_realloc(p, size);
	// `kame_realloc(NULL, 0)` returns NULL legitimately (no-op); the
	// errno-set is only for genuine ENOMEM (size > 0 path returned
	// NULL).  Mirror glibc behaviour: errno is set only when a
	// non-zero allocation failed.
	if( !q && size != 0u)
		errno = ENOMEM;
	return q;
}

extern "C" __attribute__((noinline, used))
std::size_t kame_pool_malloc_usable_size(const void *p) noexcept {
	if( !p) return 0;
	// `size_of` is read-only; cast away const safely (it does not
	// modify the pointee — only walks `s_mmapped_spaces` and chunk
	// headers).
	return PoolAllocatorBase::size_of(const_cast<void *>(p));
}

extern "C" __attribute__((noinline, used))
void *kame_pool_aligned_alloc(std::size_t alignment, std::size_t size) noexcept {
	// C17 §7.22.3.1: alignment must be power of two; size must be
	// integral multiple of alignment.  We accept any size (matches
	// glibc's lenient interpretation; the strict form is easy to
	// re-enable).
	if(alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
		errno = EINVAL;
		return nullptr;
	}
	if(alignment <= ALLOC_ALIGNMENT) {
		if(void *p = new_redirected(size))
			return p;
		errno = ENOMEM;
		return nullptr;
	}
	// Over-aligned (> 16 B): (§17) route via `new_redirected_aligned`,
	// which pulls the slot from a matching pool bucket
	// (ALIGN ∈ {32,64,256,1024,4096}) or from a dedicated chunk (payload
	// starts at a 256 KiB unit boundary — alignment up to 256 KiB) before
	// falling back to libc `posix_memalign`.  POOL pointers are freed via
	// the ordinary pool `free` path — no `_aligned_free` pairing.
	//
	// Windows: the bucket path and dedicated-chunk path use ONLY pool
	// memory (slots at `mempool + j*ALIGN` from a 256 KiB-aligned unit
	// boundary, OR an entire chunk whose payload starts at a 256 KiB
	// unit boundary) — neither path calls `_aligned_malloc`, so the
	// `_aligned_free` pairing concern that historically blocked this
	// API on Windows does not apply.  `new_redirected_aligned` itself
	// returns null when forced into the libc fallback on Windows
	// (alignment > 256 KiB OR size > a chunk's payload capacity), so
	// that one genuinely-unsupported case correctly surfaces as
	// `errno = ENOMEM` from the call below.  For Eigen / SIMD (Align
	// ∈ {32, 64}), AVX-512 (64), cacheline (64), page (4096), and any
	// alignment up to 256 KiB the Windows path now works.
	if(void *p = new_redirected_aligned(alignment, size))
		return p;
	errno = ENOMEM;
	return nullptr;
}

extern "C" __attribute__((noinline, used))
int kame_pool_posix_memalign(void **memptr, std::size_t alignment,
                             std::size_t size) noexcept {
	// POSIX: alignment must be a power of two AND a multiple of
	// sizeof(void*).  Returns the error code; does NOT set errno.
	if( !memptr) return EINVAL;
	if(alignment < sizeof(void *)
	   || (alignment & (alignment - 1u)) != 0u)
		return EINVAL;
	if(alignment <= ALLOC_ALIGNMENT) {
		void *p = new_redirected(size);
		if( !p) return ENOMEM;
		*memptr = p;
		return 0;
	}
	// (§17) Same pool-or-libc routing as `kame_pool_aligned_alloc` —
	// bucket / dedicated-chunk paths use pool memory only and work
	// identically on Windows; only the libc fallback (alignment > 256 KiB
	// OR oversize) returns null on Windows, which we propagate as ENOMEM.
	void *p = new_redirected_aligned(alignment, size);
	if( !p) return ENOMEM;
	*memptr = p;
	return 0;
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

#  if defined(KAMEPOOLALLOC_FULL_INTERCEPT)
// macOS opt-in: also interpose `malloc` so EVERY libc malloc-family call
// (libstdc++ / libc++ / Foundation / AppKit / Qt / Ruby / Python) routes
// through the pool — turns kamepoolalloc into a mimalloc-style "full
// LD_PRELOAD allocator" for head-to-head benchmarking.
//
// DEFAULT-OFF because:
//   - production kame.app (`kame/kame.pro`) is the macOS reference build
//     and has 20+ years of free-only-interpose stability with Qt + ObjC;
//   - Apple's ObjC runtime checks calloc'd class data via `malloc_size()`,
//     which would need to be co-interposed for full compat (see comment
//     above `kame_pool_calloc`) — out of scope for this opt-in.
//
// Enable for bench builds:  cmake -DKAMEPOOLALLOC_FULL_INTERCEPT=1 ...
// or for KAME desktop after on-target soak validation.
//
// `calloc` is deliberately NOT added here even under FULL_INTERCEPT —
// the documented ObjC class-realization issue makes calloc interpose
// risky without also overriding `malloc_size` / `malloc_zone_from_ptr`
// / `malloc_good_size`.  `free` and `realloc` interpose above stay
// unconditional (correctness-critical for the pool's own pointers).
extern "C" void *malloc(std::size_t);
extern "C" std::size_t malloc_size(const void *);

namespace {
// (FULL) Co-interpose malloc_size so macOS size-queries — Swift
// `__StringStorage` capacity, ObjC class realization — see the TRUE capacity
// of POOL pointers; foreign pointers fall through to their owning zone.
// Without this, malloc_size() returns 0 for a pooled allocation and the Swift
// runtime corrupts (verified: libswiftCore CommandLine.arguments init SEGV).
// This is the surface mimalloc/jemalloc co-interpose on macOS.
std::size_t kame_malloc_size(const void *p) noexcept {
    std::size_t s = PoolAllocatorBase::size_of(const_cast<void *>(p));
    if(s) return s;                                  // pool pointer
    if(malloc_zone_t *z = malloc_zone_from_ptr(p))   // foreign → its owning zone
        return z->size(z, p);
    return 0;
}
__attribute__((used))
kame_interpose_entry kame_interposers_full[]
    __attribute__((section("__DATA,__interpose"))) = {
        { reinterpret_cast<const void *>(&kame_pool_malloc),
          reinterpret_cast<const void *>(&malloc) },
        { reinterpret_cast<const void *>(&kame_malloc_size),
          reinterpret_cast<const void *>(&malloc_size) },
};
} // namespace
#  endif // KAMEPOOLALLOC_FULL_INTERCEPT

#elif defined(__linux__)
extern "C" __attribute__((noinline))
void *realloc(void *p, std::size_t n) {
	return kame_realloc(p, n);
}
#else
// Windows: `realloc` is not interposed via a strong symbol (PE/COFF has
// no cross-module symbol interposition).  Instead it is redirected at
// runtime together with `free` / `_msize` by the IAT patch engine below
// (see `kame_pool_win_install_redirect`).  Other non-Win/Linux/macOS
// targets: callers needing the pool path use `kame_pool_realloc()`.
#endif

#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
// ===================================================================
// (§31) Windows free-family IAT redirect engine.
//
// Patches the `free` / `realloc` / `_msize` import-address-table slots
// of every loaded module that imports them from a CRT DLL, repointing
// them at the pool dispatchers below.  Every `free()` in the process
// (kame.exe, Qt6*.dll, libc++.dll, modules) then funnels through
// `kame_free`, which frees pool pointers via the pool and forwards
// foreign (genuine-CRT) pointers to the real UCRT free.  This is the
// missing half of the pool's interposition story on Windows — the
// equivalent of the ELF strong-symbol `free` shim / Mach-O
// `__DATA,__interpose`, neither of which PE/COFF offers.
//
// We deliberately do NOT redirect `malloc`/`calloc`/`operator new`:
// crash-safety only requires that *frees* be reconciled.  KAME's own
// allocations still come from the pool (kame.exe's `operator new`);
// Qt's still come from the CRT; the redirect just makes either side's
// `free` land in the right allocator.
//
// Kill switch: set env `KAME_POOL_WIN_REDIRECT=0` to skip installation
// (falls back to the historical "pool + unreconciled CRT" behaviour).
// ===================================================================

// --- pool-side replacement functions (CRT-compatible signatures) ---
// On x64 every calling convention collapses to the MS x64 ABI, so these
// plain functions match `void free(void*)` / `void* realloc(void*,size_t)`
// / `size_t _msize(void*)` exactly when invoked through a patched slot.
static void kame_iat_free(void *p) noexcept {
    kame_free(p);  // pool-or-foreign dispatcher (foreign → g_real_free)
}
static void *kame_iat_realloc(void *p, std::size_t n) noexcept {
    return kame_realloc(p, n);  // pool reshape, or foreign → g_real_realloc
}
static std::size_t kame_iat_msize(void *p) noexcept {
    std::size_t s = PoolAllocatorBase::size_of(p);   // >0 ⇒ pool pointer
    if(s) return s;
    return g_real_msize ? g_real_msize(p) : 0;       // foreign ⇒ genuine
}
#if defined(KAMEPOOLALLOC_FULL_INTERCEPT)
// Opt-in: also intercept allocation entries.  Default-off because
// production kame.exe (Qt + Ruby + Python + module DLLs) has 20+ years
// of free-only-redirect stability — turning the malloc family ON means
// EVERY libc malloc / calloc across Qt6Core.dll, libc++.dll, ruby340.dll
// (already gated out by kame_is_crt_dll, but other UCRT-based modules
// stay in scope) routes to the pool.  Symmetric to macOS's
// KAMEPOOLALLOC_FULL_INTERCEPT.  Enable for bench / soak / measurement
// builds; leave off for production until on-target soak validation.
static void *kame_iat_malloc(std::size_t n) noexcept {
    // Pre-activation falls back via g_real_malloc inside
    // libsystem_malloc_for_pool (resolved before the patch installs).
    if(__builtin_expect(!g_sys_image_loaded || kame_thread_torn_down(), 0))
        return libsystem_malloc_for_pool(n);
    if(void *p = new_redirected(n)) return p;
    errno = ENOMEM;
    return nullptr;
}
static void *kame_iat_calloc(std::size_t n_elem, std::size_t sz) noexcept {
    return kame_calloc(n_elem, sz);  // libc-spec compliant (overflow check + zero-size handling)
}
#endif

// case-insensitive substring (avoid locale / _stricmp dependency)
static bool kame_ci_contains(const char *hay, const char *needle) noexcept {
    if( !hay || !needle) return false;
    for(; *hay; ++hay) {
        const char *h = hay, *n = needle;
        while(*n) {
            char a = *h, b = *n;
            if(a >= 'A' && a <= 'Z') a = char(a - 'A' + 'a');
            if(b >= 'A' && b <= 'Z') b = char(b - 'A' + 'a');
            if(a != b) break;
            ++h; ++n;
        }
        if( !*n) return true;
    }
    return false;
}
static bool kame_is_crt_dll(const char *name) noexcept {
    // Only the UCRT family that kame.exe / Qt / libc++ all share.  We
    // forward foreign pointers to ucrtbase's `free` (g_real_free), so we
    // must NOT patch modules on a *different* CRT heap — e.g. Ruby
    // (x64-msvcrt-ruby340.dll → legacy msvcrt.dll) or a VC++ redist
    // (msvcr120/140.dll).  Freeing an msvcrt-heap pointer with ucrtbase's
    // free corrupts the heap (observed: int3 in ntdll on the Ruby thread).
    // Those modules keep their own free; only pool pointers crossing into
    // a UCRT module's free() are the hazard we must reconcile.
    return kame_ci_contains(name, "ucrtbase")
        || kame_ci_contains(name, "api-ms-win-crt-heap");
}
static void *kame_iat_repl_for(const char *fn) noexcept {
    if(std::strcmp(fn, "free")    == 0) return reinterpret_cast<void *>(&kame_iat_free);
    if(std::strcmp(fn, "realloc") == 0) return reinterpret_cast<void *>(&kame_iat_realloc);
    if(std::strcmp(fn, "_msize")  == 0) return reinterpret_cast<void *>(&kame_iat_msize);
#if defined(KAMEPOOLALLOC_FULL_INTERCEPT)
    if(std::strcmp(fn, "malloc")  == 0) return reinterpret_cast<void *>(&kame_iat_malloc);
    if(std::strcmp(fn, "calloc")  == 0) return reinterpret_cast<void *>(&kame_iat_calloc);
#endif
    return nullptr;
}

// Diagnostics: how many import slots / modules the redirect patched.
static unsigned g_kame_redirect_slots = 0;
static unsigned g_kame_redirect_mods  = 0;

// Patch one already-mapped module's import table in place.
static void kame_patch_one_module(HMODULE hmod) noexcept {
    if( !hmod) return;
    BYTE *base = reinterpret_cast<BYTE *>(hmod);
    auto *dos = reinterpret_cast<IMAGE_DOS_HEADER *>(base);
    if(dos->e_magic != IMAGE_DOS_SIGNATURE) return;
    auto *nt = reinterpret_cast<IMAGE_NT_HEADERS *>(base + dos->e_lfanew);
    if(nt->Signature != IMAGE_NT_SIGNATURE) return;
    IMAGE_DATA_DIRECTORY &dir =
        nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if( !dir.VirtualAddress || !dir.Size) return;
    auto *desc = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR *>(base + dir.VirtualAddress);
    unsigned patched_here = 0;
    for(; desc->Name; ++desc) {
        const char *dll = reinterpret_cast<const char *>(base + desc->Name);
        if( !kame_is_crt_dll(dll)) continue;
        auto *iat = reinterpret_cast<IMAGE_THUNK_DATA *>(base + desc->FirstThunk);
        auto *oft = desc->OriginalFirstThunk
            ? reinterpret_cast<IMAGE_THUNK_DATA *>(base + desc->OriginalFirstThunk)
            : iat;  // bound-import edge case: names live in the IAT itself
        for(; oft->u1.AddressOfData; ++oft, ++iat) {
            if(IMAGE_SNAP_BY_ORDINAL(oft->u1.Ordinal)) continue;  // by ordinal
            auto *ibn = reinterpret_cast<IMAGE_IMPORT_BY_NAME *>(
                base + oft->u1.AddressOfData);
            void *repl = kame_iat_repl_for(reinterpret_cast<const char *>(ibn->Name));
            if( !repl) continue;
            void **slot = reinterpret_cast<void **>(&iat->u1.Function);
            if( *slot == repl) continue;  // idempotent
            DWORD oldProt = 0;
            if(VirtualProtect(slot, sizeof(void *), PAGE_READWRITE, &oldProt)) {
                *slot = repl;
                VirtualProtect(slot, sizeof(void *), oldProt, &oldProt);
                ++g_kame_redirect_slots;
                ++patched_here;
            }
        }
    }
    if(patched_here) ++g_kame_redirect_mods;
}

static void kame_patch_all_modules() noexcept {
    // Enumerate loaded modules by walking PEB->Ldr->InMemoryOrderModuleList
    // directly, NOT via EnumProcessModules.  EnumProcessModules re-enters
    // the loader and deadlocks when we run under the loader lock — which is
    // exactly the case when allocator.cpp is compiled into kamepoolalloc.dll
    // and its `__attribute__((constructor))` installs us from DllMain.
    // Reading the PEB list takes no lock; under the loader lock the list is
    // stable, and the patch is idempotent + the LdrRegisterDllNotification
    // hook covers anything loaded later, so a benign race outside the lock
    // can at worst miss/repeat an entry harmlessly.
    //
    // Struct offsets differ between 32-bit and 64-bit Windows:
    //
    //   x64: TEB[GS:0x60]=PEB; PEB+0x18=Ldr; Ldr+0x20=InMemoryOrderModuleList;
    //        InMemoryOrderLinks at LDR_DATA_TABLE_ENTRY+0x10;
    //        DllBase at entry+0x30 → Flink+0x20.
    //
    //   x86: TEB[FS:0x30]=PEB; PEB+0x0C=Ldr; Ldr+0x14=InMemoryOrderModuleList;
    //        InMemoryOrderLinks at LDR_DATA_TABLE_ENTRY+0x08;
    //        DllBase at entry+0x18 → Flink+0x10.
#ifdef _WIN64
    BYTE *peb   = reinterpret_cast<BYTE *>(__readgsqword(0x60));
    const std::ptrdiff_t off_ldr  = 0x18;
    const std::ptrdiff_t off_list = 0x20;
    const std::ptrdiff_t off_base = 0x20;
#else
    BYTE *peb   = reinterpret_cast<BYTE *>(__readfsdword(0x30));
    const std::ptrdiff_t off_ldr  = 0x0C;
    const std::ptrdiff_t off_list = 0x14;
    const std::ptrdiff_t off_base = 0x10;
#endif
    if( !peb) return;
    BYTE *ldr = *reinterpret_cast<BYTE **>(peb + off_ldr);
    if( !ldr) return;
    BYTE *head = ldr + off_list;
    for(BYTE *cur = *reinterpret_cast<BYTE **>(head);
        cur && cur != head;
        cur = *reinterpret_cast<BYTE **>(cur)) {
        void *dllBase = *reinterpret_cast<void **>(cur + off_base);
        if(dllBase) kame_patch_one_module(reinterpret_cast<HMODULE>(dllBase));
    }
}

// LdrRegisterDllNotification: ntdll calls us on every later DLL load so
// lazily/explicitly-loaded modules get patched too.  Struct + typedefs
// declared locally (not in mingw's winternl.h).
struct KAME_LDR_NOTIFICATION_DATA {  // matches LDR_DLL_NOTIFICATION_DATA (x64)
    ULONG       Flags;
    const void *FullDllName;   // PCUNICODE_STRING
    const void *BaseDllName;   // PCUNICODE_STRING
    PVOID       DllBase;
    ULONG       SizeOfImage;
};
typedef VOID (CALLBACK *kame_ldr_notify_fn)(ULONG, const void *, PVOID);
// Return type is NTSTATUS (== LONG); WIN32_LEAN_AND_MEAN doesn't pull in
// the NTSTATUS typedef, and we ignore the status anyway, so use LONG.
typedef LONG (NTAPI *kame_ldr_register_fn)(ULONG, kame_ldr_notify_fn, PVOID, PVOID *);

static VOID CALLBACK kame_dll_notification(ULONG reason, const void *data, PVOID) {
    // LDR_DLL_NOTIFICATION_REASON_LOADED == 1
    if(reason == 1 && data) {
        auto *d = reinterpret_cast<const KAME_LDR_NOTIFICATION_DATA *>(data);
        kame_patch_one_module(reinterpret_cast<HMODULE>(d->DllBase));
    }
}

extern "C" void kame_pool_win_install_redirect() noexcept {
    static bool s_done = false;          // single-threaded at activation
    if(s_done) return;
    s_done = true;
    const char *killswitch = std::getenv("KAME_POOL_WIN_REDIRECT");
    if(killswitch && killswitch[0] == '0')
        return;  // kill switch — leave the historical behaviour in place
    kame_resolve_real_crt();             // genuine CRT first — the forward
                                         // paths depend on it before any
                                         // patched free can run
    // Register for future loads BEFORE the initial sweep so a module
    // mapped concurrently with the sweep is still caught (patch is
    // idempotent, so double-patching the same slot is harmless).
    if(HMODULE ntdll = GetModuleHandleA("ntdll.dll")) {
        auto reg = reinterpret_cast<kame_ldr_register_fn>(
            GetProcAddress(ntdll, "LdrRegisterDllNotification"));
        if(reg) {
            static PVOID cookie = nullptr;
            reg(0, &kame_dll_notification, nullptr, &cookie);
        }
    }
    kame_patch_all_modules();
    // Quiet on success; opt in with KAME_POOL_VERBOSE for diagnostics.
    // Always warn if we patched nothing — that means no UCRT-family
    // module exposed a free import to redirect, i.e. cross-module frees
    // of pool pointers are NOT reconciled (a silent regression risk).
    if(std::getenv("KAME_POOL_VERBOSE") || g_kame_redirect_slots == 0)
        fprintf(stderr,
            "[kamepool] Windows free-redirect: patched %u import slot(s) "
            "across %u module(s)%s.\n",
            g_kame_redirect_slots, g_kame_redirect_mods,
            g_kame_redirect_slots == 0 ? " — WARNING: nothing patched" : "");
}
#endif // _WIN32 IAT redirect engine

// `release_pools()` / `report_statistics()` / per-template
// `release_pools()` / `PoolAllocatorBase::release_chunks()` are all gone.
// They walked the per-template `s_chunks_of_type[]` registry — itself
// retired — and had no live callers anywhere in the tree (only a
// commented-out comment in `kame/main.cpp:345` and an unused declaration
// in `allocator.h`).  Memory is reclaimed naturally:
//   * Empty chunks released by `owner_release` /
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
    if(void *p = kame_alloc_with_handler(size)) return p;
    throw std::bad_alloc();
}
__attribute__((noinline))
void* operator new[](std::size_t size) {
    return ::operator new(size);
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
    // (§18) nothrow variant: call the same new_handler loop as throwing
    // operator new, but catch a handler that throws (per [new.delete.single]
    // a nothrow new must not propagate the bad_alloc to the caller).
    KAME_HISTO_REC(size);
    try {
        return kame_alloc_with_handler(size);
    } catch(...) {
        return nullptr;
    }
}
__attribute__((noinline))
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return ::operator new(size, std::nothrow);
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
// Over-aligned (`new (std::align_val_t{64}) Foo`) goes to the
// platform-native aligned-alloc and back via the matching free —
// posix_memalign / free on POSIX, _aligned_malloc / _aligned_free on
// Windows.  The aligned operator delete forms below carry the
// alignment so dispatch is correct.
namespace {
// (§17) On POSIX, over-aligned new is routed via `new_redirected_aligned`
// — pool buckets up to ALIGN=4096, then dedicated chunk (256 KiB-aligned
// payload) up to alignment=256 KiB, then libc `posix_memalign`.  Pool
// pointers free through the ordinary pool path; libc pointers free
// through libc free; `deallocate_pooled_or_free` already resolves both,
// so `operator delete(align_val_t)` needs no separate aligned-free path.
//
// Windows keeps the platform-pair `_aligned_malloc`/`_aligned_free`
// because `kame_pool_free` lacks alignment info to dispatch correctly.
inline void *kame_overaligned_alloc(std::size_t alignment,
                                    std::size_t size) noexcept {
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
    return _aligned_malloc(size, alignment);
#else
    return new_redirected_aligned(alignment, size);
#endif
}
inline void kame_overaligned_free(void *p) noexcept {
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
    _aligned_free(p);
#else
    // POSIX path: pool slots + libc fallback both freed via the unified
    // pool-or-libc free.  Same as `deallocate_pooled_or_free`.
    deallocate_pooled_or_free(p);
#endif
}
} // namespace

__attribute__((noinline))
void* operator new(std::size_t size, std::align_val_t al) {
    // (§18) New-handler retry for the over-aligned path too.
    if ((std::size_t)al <= ALLOC_ALIGNMENT) {
        if(void *p = kame_alloc_with_handler(size)) return p;
        throw std::bad_alloc();
    }
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
    // Windows over-aligned path still pairs with _aligned_free; one shot
    // (no new_handler retry — the platform allocator handles its own
    // OOM semantics).
    void *p = kame_overaligned_alloc((std::size_t)al, size);
    if (!p) throw std::bad_alloc();
    return p;
#else
    if(void *p = kame_aligned_alloc_with_handler((std::size_t)al, size))
        return p;
    throw std::bad_alloc();
#endif
}
__attribute__((noinline))
void* operator new[](std::size_t size, std::align_val_t al) {
    return ::operator new(size, al);
}
__attribute__((noinline))
void* operator new(std::size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
    try {
        return ::operator new(size, al);
    } catch(...) {
        return nullptr;
    }
}
__attribute__((noinline))
void* operator new[](std::size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
    return ::operator new(size, al, std::nothrow);
}
__attribute__((noinline))
void operator delete(void* p, std::align_val_t al) noexcept {
    // (§17) POSIX: pool-or-libc unified through `kame_overaligned_free` =
    // `deallocate_pooled_or_free`.  Windows: keep the `_aligned_free`
    // pairing for the platform-native path.
    if ((std::size_t)al <= ALLOC_ALIGNMENT)
        deallocate_pooled_or_free(p);
    else
        kame_overaligned_free(p);
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

// runtime max-regions cap definition + public API.
std::atomic<int> PoolAllocatorBase::s_max_regions_cap{ALLOC_MAX_REGIONS};
// (§21) thread-exit madvise default ON.
std::atomic<int> PoolAllocatorBase::s_thread_exit_reclaim{1};

extern "C" void kame_pool_set_thread_exit_reclaim(int enable) noexcept {
    PoolAllocatorBase::s_thread_exit_reclaim.store(
        enable ? 1 : 0, std::memory_order_relaxed);
}

extern "C" void kame_pool_set_max_bytes(std::size_t max_bytes) noexcept {
    // 0 = disable cap → restore the compile-time ceiling.
    int regions;
    if(max_bytes == 0u) {
        regions = ALLOC_MAX_REGIONS;
    } else {
        // Round UP to multiple of ALLOC_MIN_MMAP_SIZE (= 32 MiB).
        std::size_t r =
            (max_bytes + ALLOC_MIN_MMAP_SIZE - 1u) / ALLOC_MIN_MMAP_SIZE;
        if(r > (std::size_t)ALLOC_MAX_REGIONS)
            r = (std::size_t)ALLOC_MAX_REGIONS;
        regions = static_cast<int>(r);
    }
    PoolAllocatorBase::s_max_regions_cap.store(
        regions, std::memory_order_relaxed);
}

extern "C" std::size_t kame_pool_get_max_bytes() noexcept {
    int regions = PoolAllocatorBase::s_max_regions_cap.load(
        std::memory_order_relaxed);
    if(regions >= ALLOC_MAX_REGIONS) return SIZE_MAX;
    return (std::size_t)regions * (std::size_t)ALLOC_MIN_MMAP_SIZE;
}

extern "C" std::size_t kame_pool_reserved_bytes() noexcept {
    return PoolAllocatorBase::populated_region_count()
         * (std::size_t)ALLOC_MIN_MMAP_SIZE;
}

// (§28.2) Forward decl of anonymous-namespace g_lrc_bytes (defined inside
// the §28 cache machinery much later in this file).  Re-opening the anon
// namespace with `extern` resolves to the same symbol in this TU; lets
// kame_pool_get_stats snapshot the cache bytes without rearranging the
// cache definitions.
namespace { extern std::atomic<size_t> g_lrc_bytes; }

// (§14) Diagnostic / tuning stats — walks the push-only region list
// (since §13.3, O(populated regions × BITMAP_WORDS_PER_REGION)) reading
// each region's embedded claim_bitmap + back_offset under relaxed loads.
// Counters are best-effort snapshots (concurrent alloc / free may make
// them slightly inconsistent with one another), suitable for tuning,
// not for assertions.
extern "C" void kame_pool_get_stats(kame_pool_stats_t *out) noexcept {
    if( !out) return;
    unsigned int req_ver = out->version;
    // We currently fill version 1 fields.  Older callers (req_ver == 0
    // from a memset, or req_ver == 1) get a v1 snapshot; future v2+
    // callers also get v1 fields plus whatever the library actually
    // fills (version_supported reports the cap).
    out->version_supported = KAME_POOL_STATS_VERSION;
    (void)req_ver;  // reserved for forward-compat gating

    std::size_t regions = 0;
    std::size_t units = 0;
    std::size_t chunks = 0;
    std::size_t ded_bytes = 0;
    using BitmapWord = PoolAllocatorBase::BitmapWord;
    int total_nodes = PoolAllocatorBase::s_num_numa_nodes.load(
        std::memory_order_relaxed);
    if(total_nodes <= 0) total_nodes = 1;
    for(int node = 0; node < total_nodes; ++node)
    for(auto *rm = PoolAllocatorBase::s_region_dll_heads[node].load(
                       std::memory_order_acquire);
        rm; rm = rm->dll_next.load(std::memory_order_acquire)) {
        ++regions;
        for(int w = 0; w < PoolAllocatorBase::BITMAP_WORDS_PER_REGION; ++w) {
            BitmapWord v =
                rm->claim_bitmap[w].load(std::memory_order_relaxed);
            // Mask the per-region metadata bit (bit 0 of word 0) so
            // `units_live` reports chunk-occupied units only.
            if(w == 0) v &= ~BitmapWord(1);
            units += (std::size_t)count_bits(v);
        }
        // "Chunks live" = base units = back_offset[u]==0 entries whose
        // claim bit is set.  Skip unit 0 (the metadata reservation).
        // (§28.5) Also accumulate dedicated_chunk_bytes here: a base unit
        // with bit-7 of its back_offset set is a dedicated chunk's first
        // unit, and its DEDICATED_SIZE header field gives the size.
        // base_unit u maps to chunk_base = mp + u*ALLOC_MIN_CHUNK_SIZE -
        // ALLOC_CHUNK_K_MAX (see §15).
        char *mp = reinterpret_cast<char *>(rm);
        for(int u = 1; u < PoolAllocatorBase::NUM_ALLOCATORS_IN_SPACE; ++u) {
            BitmapWord bw = rm->claim_bitmap[
                u / PoolAllocatorBase::BITS_PER_BITMAP_WORD]
                .load(std::memory_order_relaxed);
            bool claimed =
                (bw >> (u % PoolAllocatorBase::BITS_PER_BITMAP_WORD)) & 1u;
            if( !claimed) continue;
            std::uint8_t back_off = rm->back_offset[u];
            if((back_off & 0x7Fu) != 0u) continue;   // not a base unit
            ++chunks;
            if((back_off & 0x80u) != 0u) {
                char *chunk_base = mp + (std::size_t)u
                    * (std::size_t)ALLOC_MIN_CHUNK_SIZE
                    - (std::size_t)ALLOC_CHUNK_K_MAX;
                ded_bytes += (std::size_t)*reinterpret_cast<std::uint64_t *>(
                    chunk_base + ALLOC_CHUNK_HEADER_DEDICATED_SIZE_OFFSET);
            }
        }
    }
    out->regions_populated = regions;
    out->bytes_reserved    = regions * (std::size_t)ALLOC_MIN_MMAP_SIZE;
    out->chunks_live       = chunks;
    out->units_live        = units;

    // v2 — `cache_bytes` is the lock-free byte total maintained by the
    // recycle cache (§28); `large_*` are 2 plain global atomics (§28.5).
    // `dedicated_chunk_bytes` is the walk-derived total above and includes
    // cache-parked dedicated chunks (their units stay claimed + bit-7 set;
    // see header doc).
    // size_t-typed atomics (i486-clean: no CMPXCHG8B); unsigned-wrap
    // transients on a racing fetch_sub produce ~SIZE_MAX, which clamps
    // below to 0-style behavior is no longer needed — the public field
    // is already size_t and a true negative is unreachable.
    out->cache_bytes           = g_lrc_bytes.load(std::memory_order_relaxed);
    out->dedicated_chunk_bytes = ded_bytes;
    out->large_alloc_count     = g_large_alloc_count.load(std::memory_order_relaxed);
    out->large_alloc_bytes     = g_large_alloc_bytes.load(std::memory_order_relaxed);
}

std::atomic<RadixL2Node *> PoolAllocatorBase::s_radix_l1[RADIX_L1_SIZE];

// (§13.3 / §14C) Per-NUMA-node region lists + populated-region count.
std::atomic<PoolAllocatorBase::RegionMeta *>
    PoolAllocatorBase::s_region_dll_heads[PoolAllocatorBase::KAME_MAX_NUMA_NODES];
std::atomic<int> PoolAllocatorBase::s_region_count{0};
std::atomic<int> PoolAllocatorBase::s_num_numa_nodes{0};  // 0 = uninit

// (§14C) Per-thread preferred NUMA node, lazy-initialized.  -1 = uninit.
static ALLOC_TLS_IE int s_tls_numa_node = -1;

// (§14C) Read the system NUMA node count.  Linux:
// `/sys/devices/system/node/nodeN` directories; the highest N + 1 is the
// node count (clamped to KAME_MAX_NUMA_NODES).  Non-Linux returns 1.
// Called once via the lazy s_num_numa_nodes init in
// `numa_node_for_this_thread()`.
__attribute__((cold))
static int detect_num_numa_nodes() noexcept {
#if defined(__linux__)
	int max_n = 0;
	DIR *d = opendir("/sys/devices/system/node");
	if(d) {
		while(struct dirent *e = readdir(d)) {
			int n;
			if(sscanf(e->d_name, "node%d", &n) == 1 && n + 1 > max_n)
				max_n = n + 1;
		}
		closedir(d);
	}
	if(max_n < 1) max_n = 1;
	if(max_n > PoolAllocatorBase::KAME_MAX_NUMA_NODES)
		max_n = PoolAllocatorBase::KAME_MAX_NUMA_NODES;
	return max_n;
#else
	return 1;
#endif
}

// (§14C) NUMA node of a specific CPU id (Linux: read
// `/sys/devices/system/cpu/cpuN/node*/` — the only sibling node%d there
// is the CPU's home node).  Returns 0 on failure / non-Linux.
__attribute__((cold))
static int numa_node_for_cpu(int cpu) noexcept {
#if defined(__linux__)
	char path[64];
	snprintf(path, sizeof(path),
	         "/sys/devices/system/cpu/cpu%d", cpu);
	DIR *d = opendir(path);
	if( !d) return 0;
	int node = 0;
	while(struct dirent *e = readdir(d)) {
		int n;
		if(sscanf(e->d_name, "node%d", &n) == 1) { node = n; break; }
	}
	closedir(d);
	return node < PoolAllocatorBase::KAME_MAX_NUMA_NODES
	       ? node : PoolAllocatorBase::KAME_MAX_NUMA_NODES - 1;
#else
	(void)cpu;
	return 0;
#endif
}

__attribute__((cold, noinline))
int PoolAllocatorBase::numa_node_for_this_thread() noexcept {
	int n = s_tls_numa_node;
	if(__builtin_expect(n >= 0, 1)) return n;
	// REENTRANCY GUARD: the sysfs probes below (`opendir` in
	// detect_num_numa_nodes / numa_node_for_cpu) malloc a ~32 KiB DIR
	// buffer inside libc; under the malloc redirect that allocation
	// re-enters this allocator, reaches claim_chunk, and lands back
	// here — with the lazy init still in flight, an unbounded
	// opendir→malloc→claim_chunk→opendir recursion (stack overflow at
	// startup).  Pre-publish node 0 so the nested call short-circuits;
	// the real node overwrites it below.  The nested allocation may be
	// placed on node 0's region list once — harmless.
	s_tls_numa_node = 0;
	// Lazy init.  First call from any thread sets the global node count
	// (idempotent races OK — same value).
	int total = s_num_numa_nodes.load(std::memory_order_relaxed);
	if(total == 0) {
		total = detect_num_numa_nodes();
		s_num_numa_nodes.store(total, std::memory_order_relaxed);
	}
	if(total <= 1) return 0;
#if defined(__linux__)
	int cpu = sched_getcpu();
	n = cpu >= 0 ? numa_node_for_cpu(cpu) : 0;
#else
	n = 0;
#endif
	if(n >= total) n = total - 1;
	s_tls_numa_node = n;
	return n;
}

// (§13) Per-thread 1-entry region-lookup cache + owner-id stamp.
// Both fields now live in `KameTlsPage` (g_tls_page), accessed via
// `kame_page()->last_region_base` and `kame_page()->owner_id`.
// The KameTlsPage itself is defined and initialised near the top of
// this TU (see the ALLOC_TLS / ALLOC_TLS_IE g_tls_page declaration).
// The radix_lookup_slow and radix_remove functions below use kame_page().

// Out-of-line full radix walk + cache update.  Called on cache miss.
// Returns 0 if `up` is in a populated region, -1 otherwise.
__attribute__((noinline))
int PoolAllocatorBase::radix_lookup_slow(uintptr_t up) noexcept {
	// Defensive: pointers above our covered VA fall back to "not our
	// pointer".  On 32-bit hosts the radix already covers the full
	// uintptr_t range (region index = 7 bits ≤ 32 - 25), so the bound
	// check is vacuous and skipped (`up >> 48` is UB).
	constexpr int kBoundShift = RADIX_REGION_BITS + ALLOC_MIN_MMAP_SHIFT;
#if defined(_MSC_VER) && !defined(__GNUC__)
#pragma warning(push)
#pragma warning(disable: 4293) // MSVC warns on shift-by->=width inside discarded if constexpr
#endif
	if constexpr (kBoundShift < (int)(sizeof(uintptr_t) * 8)) {
		if(__builtin_expect((up >> kBoundShift) != 0u, 0))
			return (int)KAME_RADIX_ABSENT;
	}
#if defined(_MSC_VER) && !defined(__GNUC__)
#pragma warning(pop)
#endif
	unsigned region_idx = (unsigned)(up >> ALLOC_MIN_MMAP_SHIFT);
	unsigned l1 = region_idx >> RADIX_L2_BITS;
	unsigned l2 = region_idx & (RADIX_L2_SIZE - 1u);
	RadixL2Node *leaf = s_radix_l1[l1].load(std::memory_order_acquire);
	if(__builtin_expect(leaf == nullptr, 0)) return (int)KAME_RADIX_ABSENT;
	uint32_t v = leaf->entries[l2].load(std::memory_order_relaxed);
	if(__builtin_expect(v == 0u, 0)) return (int)KAME_RADIX_ABSENT;
	// (§19) Cache the region base for the next (locality-rich) call —
	// but ONLY for pool regions.  A §19 large-alloc base disappears on
	// munmap, and a stale cache entry on a different thread would
	// falsely report present without re-checking the slot.
	if(v == (uint32_t)KAME_RADIX_POOL)
		kame_page()->last_region_base = up & ~((uintptr_t)ALLOC_MIN_MMAP_SIZE - 1u);
	return (int)v;
}

// 2-level radix tree implementation (§13).  L2 nodes allocated lazily
// via mmap to avoid recursion through our own interposed libc malloc.
RadixL2Node *PoolAllocatorBase::radix_alloc_l2() noexcept {
#if defined(__WIN32__) || defined(WINDOWS) || defined(_WIN32)
	void *p = VirtualAlloc(nullptr, sizeof(RadixL2Node),
	                       MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
	if(!p) return nullptr;
	return static_cast<RadixL2Node *>(p);
#else
	void *p = mmap(nullptr, sizeof(RadixL2Node),
	               PROT_READ | PROT_WRITE,
	               MAP_ANON | MAP_PRIVATE, -1, 0);
	if(p == MAP_FAILED) return nullptr;
	// mmap zero-fills via the kernel — slots[] is fully empty (0).
	return static_cast<RadixL2Node *>(p);
#endif
}

void PoolAllocatorBase::radix_insert(char *mp, uint32_t kind) noexcept {
	uintptr_t up = (uintptr_t)mp;
	// Region must be ALLOC_MIN_MMAP_SIZE-aligned (mmap claim ensures this).
	unsigned region_idx = (unsigned)(up >> ALLOC_MIN_MMAP_SHIFT);
	unsigned l1 = region_idx >> RADIX_L2_BITS;
	unsigned l2 = region_idx & (RADIX_L2_SIZE - 1u);
	if(__builtin_expect(l1 >= RADIX_L1_SIZE, 0))
		return;  // Outside radix coverage; lookup will miss (returns 0).
	RadixL2Node *leaf = s_radix_l1[l1].load(std::memory_order_acquire);
	if(leaf == nullptr) {
		RadixL2Node *new_leaf = radix_alloc_l2();
		if(!new_leaf) return;  // OOM; lookup will miss this region.
		RadixL2Node *expected = nullptr;
		if(s_radix_l1[l1].compare_exchange_strong(
		       expected, new_leaf,
		       std::memory_order_release, std::memory_order_acquire)) {
			leaf = new_leaf;
		} else {
			// Concurrent installer won; release ours.
#if defined(__WIN32__) || defined(WINDOWS) || defined(_WIN32)
			VirtualFree(new_leaf, 0, MEM_RELEASE);
#else
			munmap(new_leaf, sizeof(RadixL2Node));
#endif
			leaf = expected;
		}
	}
	// Slot kind (KAME_RADIX_POOL or KAME_RADIX_LARGE).  Pool regions are
	// one-shot (never unmap), so the store is non-racing.  §19 large
	// allocs use the same path: a fresh base address never collides with
	// an existing slot because munmap-then-mmap of a different alloc at
	// the SAME base requires the prior `radix_clear` to have CAS'd the
	// slot to 0 first — see deallocate_large_va.
	// Release-paired with the reader's acquire load on the L1 entry, and
	// (for cross-thread frees) with the data handoff that passes the
	// pointer to the freeing thread.
	leaf->entries[l2].store(kind, std::memory_order_release);
}

// (§19) Clear the radix slot for a §19 large-alloc base prior to
// munmap.  Lock-free CAS-back-to-zero — concurrent readers either see
// the live slot (valid meta) or absent (fall through to libc free).
void PoolAllocatorBase::radix_clear(char *mp) noexcept {
	uintptr_t up = (uintptr_t)mp;
	unsigned region_idx = (unsigned)(up >> ALLOC_MIN_MMAP_SHIFT);
	unsigned l1 = region_idx >> RADIX_L2_BITS;
	unsigned l2 = region_idx & (RADIX_L2_SIZE - 1u);
	if(__builtin_expect(l1 >= RADIX_L1_SIZE, 0)) return;
	RadixL2Node *leaf = s_radix_l1[l1].load(std::memory_order_acquire);
	if(__builtin_expect(leaf == nullptr, 0)) return;  // never inserted
	// Plain release store: a racing lookup either reads the old non-zero
	// value (and then dereferences the meta — still valid because we
	// haven't munmap'd yet) or reads the new 0 (and falls through to
	// libc free).  munmap below is sequenced after this clear by
	// program order — no concurrent reader can be mid-deref past this
	// point because the caller (deallocate_large_va) was passed a
	// pointer into THIS alloc that no other thread should be holding
	// after free (same single-owner-on-free contract as libc free).
	leaf->entries[l2].store((uint32_t)KAME_RADIX_ABSENT,
	                      std::memory_order_release);
}

// (§13.3) mmap a fresh 32-MiB-aligned region, init its RegionMeta,
// register it in the radix, and push it on the region list.  Shared by
// both chunk-claim Pass-2 sites.  Returns the new region (== its base)
// or nullptr on cap-exceeded / mmap failure.
PoolAllocatorBase::RegionMeta *
PoolAllocatorBase::mmap_new_region() noexcept {
	// Runtime cap: reserve a slot first so a concurrent racer can't
	// overshoot.  Default cap is INT_MAX (VA-limited); a tighter value
	// comes from kame_pool_set_max_bytes.
	int c = s_region_count.fetch_add(1, std::memory_order_relaxed);
	if(c >= s_max_regions_cap.load(std::memory_order_relaxed)) {
		s_region_count.fetch_sub(1, std::memory_order_relaxed);
		return nullptr;
	}
	const size_t mmap_size = ALLOC_MIN_MMAP_SIZE;
	// MUST be ALLOC_MIN_MMAP_SIZE (32 MiB)-aligned: the §13 radix keys on
	// `p >> ALLOC_MIN_MMAP_SHIFT`, so a region's whole 32 MiB VA range
	// must sit in ONE radix slot, and `region_meta_of(p)` recovers the
	// base by masking off the low 25 bits.
	constexpr size_t kAlign = ALLOC_MIN_MMAP_SIZE;
#if defined __WIN32__ || defined WINDOWS || defined _WIN32
	char *p = static_cast<char *>(_aligned_malloc(mmap_size, kAlign));
	if( !p) {
		fprintf(stderr, "_aligned_malloc(%zu, %zu) failed.\n",
		        mmap_size, kAlign);
		s_region_count.fetch_sub(1, std::memory_order_relaxed);
		return nullptr;
	}
#else
	size_t total = mmap_size + kAlign;
	char *raw = static_cast<char *>(
	    mmap(0, total, PROT_READ | PROT_WRITE,
	         MAP_ANON | MAP_PRIVATE, -1, 0));
	if(raw == MAP_FAILED) {
		fprintf(stderr, "mmap() failed.\n");
		s_region_count.fetch_sub(1, std::memory_order_relaxed);
		return nullptr;
	}
	uintptr_t aligned =
	    ((uintptr_t)raw + kAlign - 1u) & ~(uintptr_t)(kAlign - 1u);
	char *p = reinterpret_cast<char *>(aligned);
	size_t prefix = p - raw;
	size_t suffix = total - prefix - mmap_size;
	if(prefix > 0) munmap(raw, prefix);
	if(suffix > 0) munmap(p + mmap_size, suffix);
	// (§35) Radix coverage guard (see large_va_raw_map): a region base the
	// radix can't index (≥ RADIX_VA_LIMIT = 2^48) would silently fail to
	// register, mis-routing its chunks' frees to libc.  Never fires under a
	// NULL-hint mmap on any known kernel; if it ever does, release the
	// region and fail the claim so the caller degrades to libc gracefully.
	if(__builtin_expect((uintptr_t)p >= RADIX_VA_LIMIT, 0)) {
		munmap(p, mmap_size);
		s_region_count.fetch_sub(1, std::memory_order_relaxed);
		return nullptr;
	}
	// (§14B) Opt-in transparent hugepages on the slot range (skip the
	// metadata page at offset 0).  The region is 32 MiB / 32 MiB-aligned
	// = 16 hugepages worth, ideal for the kernel's THP promoter on
	// TLB-bound HPC workloads with large working sets.
	//
	// Opt-in (env `KAME_POOL_HUGEPAGE=1`) because the microbenchmark
	// pattern (1-2 chunks per region, tight loop, ≤ 1 MiB working set)
	// REGRESSES under THP: a freshly faulted hugepage zero-fills 2 MiB
	// of physical pages even though only a few hundred KiB is touched,
	// so the upfront cost dominates without TLB-pressure payback.
	// Measured single-thread microbench impact (Linux x86-64, THP =
	// madvise; pre-B vs +THP, 12-iter interleaved):
	//     16..64 B  : neutral / +1 %
	//     128/256 B : +5..7 %
	//     512..4 KB : −4..−13 %   (regressed sizes)
	//     8 KB+     : neutral
	// Real HPC workloads typically populate many chunks per region;
	// once the region's pages are mostly touched, THP reduces TLB
	// misses on the application's data access — independent of (and
	// not modeled by) the alloc/free hot path.
	//
	// Read the env var ONCE (atomic ifd init, region claim is rare).
	// THP `/sys/kernel/mm/transparent_hugepage/enabled` must be
	// `always` or `madvise` for the advise to have effect; otherwise
	// the call is a no-op (we ignore the return regardless).
#  if defined(__linux__) && defined(MADV_HUGEPAGE)
	static const bool hugepage_enabled = [] {
		const char *e = std::getenv("KAME_POOL_HUGEPAGE");
		return e && e[0] != '\0' && e[0] != '0';
	}();
	if(hugepage_enabled)
		(void)madvise(p + ALLOC_PAGE_SIZE,
		              mmap_size - ALLOC_PAGE_SIZE, MADV_HUGEPAGE);
#  endif
#endif
	// (§14C) Bind the region to this thread's NUMA node — physical pages
	// touched later will land on that node (instead of whichever node
	// allocates first under the default MPOL_LOCAL).  Cross-node access
	// stays correct; performance just degrades to remote-memory latency.
	// Use the mbind syscall directly to avoid a libnuma dependency.
	int my_node = numa_node_for_this_thread();
#if defined(__linux__) && defined(SYS_mbind)
	if(s_num_numa_nodes.load(std::memory_order_relaxed) > 1) {
		constexpr int MPOL_BIND = 2;
		unsigned long mask = 1ul << my_node;
		(void)syscall(SYS_mbind, p, (unsigned long)mmap_size,
		              MPOL_BIND, &mask, (unsigned long)(sizeof(mask) * 8),
		              0u);
		// Best-effort: failure (older kernel, non-NUMA build) is benign.
	}
#endif
	// Init the embedded metadata block (mmap zero-filled the first page,
	// so back_offset[*]=0, claim_bitmap[*]=0, dll_next=0, has_free=0).
	// Reserve unit 0 (the metadata lives there) and flag has_free, BEFORE
	// publishing the region via the list push (whose release carries this
	// init to walkers' acquire).
	RegionMeta *rm = region_meta(p);
	rm->claim_bitmap[0].store(BitmapWord(1), std::memory_order_relaxed);
	rm->has_free.store(1, std::memory_order_relaxed);
	rm->numa_node = (std::uint16_t)my_node;
	// Publish in the radix (presence) BEFORE the list push so a free of a
	// chunk in this region can never miss the lookup.
	radix_insert(p, (uint32_t)KAME_RADIX_POOL);
	// Push on the per-node push-only list (Treiber).  Regions never
	// unmap, so no ABA / reclamation.
	std::atomic<RegionMeta *> *head = &s_region_dll_heads[my_node];
	RegionMeta *old = head->load(std::memory_order_relaxed);
	do {
		rm->dll_next.store(old, std::memory_order_relaxed);
	} while( !head->compare_exchange_weak(
	             old, rm, std::memory_order_release,
	             std::memory_order_relaxed));
	fprintf(stderr,
	    "Reserve swap space starting @ %p w/ len. of 0x%llxB (node %d).\n",
	    p, (unsigned long long)mmap_size, my_node);
	return rm;
}

// =====================================================================
// (§19/§21) Large-alloc tier — single-mmap, radix-registered, munmap-able,
//           with a per-thread LIFO recycle cache.
// =====================================================================
namespace {
// Raw 32-MiB-aligned mmap of `mmap_size` bytes.  Returns base or nullptr.
inline char *large_va_raw_map(std::size_t mmap_size) noexcept {
#if defined(__WIN32__) || defined(WINDOWS) || defined(_WIN32)
	return static_cast<char *>(_aligned_malloc(mmap_size, ALLOC_MIN_MMAP_SIZE));
#else
	std::size_t total = mmap_size + ALLOC_MIN_MMAP_SIZE;
	char *raw = static_cast<char *>(
	    mmap(0, total, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0));
	if(raw == MAP_FAILED) return nullptr;
	uintptr_t aligned =
	    ((uintptr_t)raw + ALLOC_MIN_MMAP_SIZE - 1u) &
	    ~(uintptr_t)(ALLOC_MIN_MMAP_SIZE - 1u);
	char *base = reinterpret_cast<char *>(aligned);
	std::size_t prefix = base - raw;
	std::size_t suffix = total - prefix - mmap_size;
	if(prefix > 0) munmap(raw, prefix);
	if(suffix > 0) munmap(base + mmap_size, suffix);
	// (§35) Radix coverage guard: if the kernel placed us at a base the
	// radix can't index (≥ RADIX_VA_LIMIT = 2^48), registration would
	// silently fail and a later free of this block would mis-route to
	// libc.  A NULL-hint mmap stays inside the kernel DEFAULT_MAP_WINDOW
	// (≤ 2^47 x86-64 / ≤ 2^48 arm64) on every known OS, so this never
	// fires; if it ever did, release and let the caller fall back to libc.
	// Only the head base matters — huge spans' tail slots are unregistered.
	if(__builtin_expect((uintptr_t)base >= RADIX_VA_LIMIT, 0)) {
		munmap(base, mmap_size);
		return nullptr;
	}
	return base;
#endif
}
inline void large_va_raw_unmap(char *base, std::size_t mmap_size) noexcept {
#if defined(__WIN32__) || defined(WINDOWS) || defined(_WIN32)
	(void)mmap_size;
	_aligned_free(base);
#else
	munmap(base, mmap_size);
#endif
}

// =====================================================================
// (§28) K-line log-slot large-recycle cache.  Supersedes the §25/§26
//       single-slot-per-band + ±10% band-scan layout.
// =====================================================================
// Layout: K_MAX independent K-arrays, each cache-line-aligned at the start.
// Within one array, all (N_MAX+1) idx slots are contiguous, indexed by idx.
//
//     g_lrc[k].slots[idx]   //  k ∈ [0, LRC_K_MAX), idx ∈ [0, LRC_N_MAX]
//                           //  alignas(CACHE_LINE) on each LrcKArray.
//
// Threads collide on the SAME (k, idx) pair only — never just from sharing a
// cache line via different k (different k-arrays start on different lines).
// push/pop walk K starting from `kame_owner_id() & (K-1)` so different
// threads probe DIFFERENT k first.
//
// 1:1 size-class rounding (NO band scan): a size maps to exactly one idx via
// `lrc_idx_natural`, then kind-clamped to `[0, LRC_CHUNK_BND]` for LRC_CHUNK
// and `(LRC_CHUNK_BND, LRC_N_MAX]` for LRC_MMAP, so the two tiers never
// share a slot — kind on pop is implied by idx alone.  The runtime
// `sz >= need` check absorbs the slight floor-rounding within a band.
//
// Indexing: 4 indices per octave starting at LRC_LO (= 256 KiB).
//   idx i ↔ nominal size 2^(i/4) × LRC_LO.
//   With LRC_N_MAX = 40, LRC_HI = LRC_LO × 2^10 = 256 MiB.
//   The chunk/mmap boundary is lrc_idx_natural(4 MiB) = LRC_CHUNK_BND = 16.
//
// Tunables (compile-time defaults; override via -D…):
//   LRC_N_MAX     = 40   ⇒ LRC_HI = 256 MiB (top size class).
//   LRC_K_MAX     = 256  ⇒ K slots per (idx, kind), must be a power of two.
//   LRC_N_MAX_L1  = 24   ⇒ per-thread L1 idx ceiling (≈ 16 MiB).
//   LRC_K_L1      = 32   ⇒ K slots per (idx, kind) in L1, power of two.
// Static memory: global ≈ 96 KiB (64-B cache lines) / 112 KiB (128-B lines);
// L1 ≈ 6.4 KiB / thread.  Caps above 256 MiB / per-thread > 16 MiB need a
// rebuild with larger LRC_*_MAX.
//
// (§35) LRC_N_MAX = 40 (was 32) so kame.app's 100 MB-class (and up to
// ~200 MB) image buffers stay in the warm recycle cache.  A 100 MB request
// maps to idx ≈ 33, a 200 MB one to idx ≈ 37 — BOUNDED bands well below the
// collapsing top slot (idx = LRC_N_MAX = 40, ≥ 256 MiB), so neither is
// subject to the unbounded-fit over-satisfaction that makes the very top slot
// pin RSS.  LRC_MMAP blocks in [32 MiB, 256 MiB] are multi-region spans (only
// the head 32-MiB radix slot registered); the 32–64 MiB sub-band has been
// span-cached since LRC_HI was first 64 MiB, so raising the ceiling to 256 MiB
// only widens an already-exercised path.  RSS is bounded solely by g_lrc_cap
// (default ~1 GiB ⇒ ~10 cached 100 MB blocks); raise it via
// kame_pool_set_max_bytes() for image-heavy runs.
//
// Block kinds (same as before):
//   - LRC_MMAP : region stays mapped, radix CLEARED on push (double-free
//                routes to libc), re-registered on reuse.  Release = munmap.
//   - LRC_CHUNK: units stay CLAIMED + chunk_header intact on push, payload
//                returned directly on reuse.  Release = deallocate_chunk.

#ifndef LRC_N_MAX
#define LRC_N_MAX 40
#endif
#ifndef LRC_K_MAX
#define LRC_K_MAX 256
#endif
#ifndef LRC_N_MAX_L1
#define LRC_N_MAX_L1 24
#endif
#ifndef LRC_K_L1
#define LRC_K_L1 32
#endif

static_assert(LRC_N_MAX % 4 == 0,
              "LRC_N_MAX must be a multiple of 4 (4 indices per octave)");
static_assert((LRC_K_MAX & (LRC_K_MAX - 1)) == 0,
              "LRC_K_MAX must be a power of two");
static_assert((LRC_K_L1 & (LRC_K_L1 - 1)) == 0,
              "LRC_K_L1 must be a power of two");
static_assert(LRC_N_MAX_L1 <= LRC_N_MAX,
              "L1 idx ceiling cannot exceed global");

constexpr int    LRC_LO_LOG2 = 18;                                    // log2(256 KiB)
constexpr std::size_t LRC_LO = (std::size_t)1 << LRC_LO_LOG2;         // = ALLOC_MIN_CHUNK_SIZE
static_assert(LRC_LO == (std::size_t)ALLOC_MIN_CHUNK_SIZE,
              "LRC_LO must equal ALLOC_MIN_CHUNK_SIZE");
constexpr std::size_t LRC_HI = LRC_LO << (LRC_N_MAX / 4);              // 256 MiB at LRC_N_MAX=40
// Chunk/mmap boundary in idx space.  lrc_idx_natural(ALLOC_MAX_CHUNK_SIZE).
constexpr int LRC_CHUNK_BND = 4 * (22 - LRC_LO_LOG2);                  // = 16 (log2(4 MiB)=22)
static_assert((std::size_t)1 << (LRC_LO_LOG2 + LRC_CHUNK_BND / 4)
              == (std::size_t)ALLOC_MAX_CHUNK_SIZE,
              "LRC_CHUNK_BND must equal lrc_idx_natural(ALLOC_MAX_CHUNK_SIZE)");

// Global L2: K_MAX independent K-arrays, each cache-line-aligned at start.
struct alignas(KAME_CACHE_LINE) LrcKArray {
    std::atomic<char *> slots[LRC_N_MAX + 1];
};
LrcKArray g_lrc[LRC_K_MAX];

// Pointer-width atomics so i486 (no CMPXCHG8B) doesn't need libatomic.
// On 32-bit, size_t = uint32_t = 4 GiB ceiling; the README's documented
// 32-bit cap is 3 GiB, so it fits with headroom.  64-bit unchanged.
std::atomic<size_t> g_lrc_bytes{0};
std::atomic<size_t> g_lrc_cap{(size_t)1 << 30};                       // ~1 GiB default

// idx = 4 * octave + 2-bit_mantissa, integer-only (no libm).
// Sizes [LRC_LO, LRC_HI) map to [0, LRC_N_MAX); LRC_HI and above clamp to LRC_N_MAX.
inline int lrc_idx_natural(std::size_t S) noexcept {
    if(S <= LRC_LO) return 0;
    if(S >= LRC_HI) return LRC_N_MAX;
    int msb = 63 - __builtin_clzll((unsigned long long)S);             // floor(log2 S)
    int octave = msb - LRC_LO_LOG2;                                    // ≥ 0
    int mantissa = (int)(((unsigned long long)S >> (msb - 2)) & 0x3u); // top 2 bits below MSB
    int i = 4 * octave + mantissa;
    if(i < 0) i = 0;
    if(i > LRC_N_MAX) i = LRC_N_MAX;
    return i;
}
// Kind-clamped idx: LRC_CHUNK fits in [0, LRC_CHUNK_BND]; LRC_MMAP in (LRC_CHUNK_BND, LRC_N_MAX].
// Guarantees a slot only ever holds one kind, so pop derives the meta layout
// from idx alone (`lrc_kind_from_idx`).
inline int lrc_idx(std::size_t S, unsigned kind) noexcept {
    int i = lrc_idx_natural(S);
    if(kind == (unsigned)LRC_CHUNK) { if(i > LRC_CHUNK_BND)     i = LRC_CHUNK_BND;     }
    else                            { if(i <= LRC_CHUNK_BND)    i = LRC_CHUNK_BND + 1; }
    return i;
}
// Kind implied by idx (for evict / drain — caller-passed kind absent).
inline unsigned lrc_kind_from_idx(int i) noexcept {
    return (i <= LRC_CHUNK_BND) ? (unsigned)LRC_CHUNK : (unsigned)LRC_MMAP;
}
// Real size of a cached block, read from its kind's meta.  Caller OWNS the
// block (taken via CAS) — this is own-then-read, NOT a peek, so it cannot
// race a concurrent release/munmap.
inline std::size_t lrc_block_size(char *base, unsigned kind) noexcept {
    if(kind == (unsigned)LRC_CHUNK)
        return (std::size_t)*reinterpret_cast<std::uint64_t *>(base + ALLOC_CHUNK_HEADER_DEDICATED_SIZE_OFFSET);
    return PoolAllocatorBase::large_alloc_meta_of(base)->mmap_size;
}
// Release a block per its kind (both touch only global state — bitmap+madvise
// for CHUNK, munmap for MMAP — never any allocator TLS).
inline void lrc_release(char *base, std::size_t size, unsigned kind) noexcept {
    if(kind == (unsigned)LRC_CHUNK) PoolAllocatorBase::recycle_release_chunk(base, size);
    else                            large_va_raw_unmap(base, size);
}
// kstart: thread-unique starting K offset for push/pop loops.  Reuses
// `kame_owner_id()` (already IE-TLS-cached, allocated on first allocator use
// from a global atomic counter — see top of file) so this introduces no new
// global state or TLS variable.
inline int lrc_kstart_g() noexcept {
    return (int)(kame_owner_id() & (LRC_K_MAX - 1));
}
inline int lrc_kstart_l1() noexcept {
    return (int)(kame_owner_id() & (LRC_K_L1 - 1));
}

// =====================================================================
// Per-thread L1 in front of the global L2 cache (§28).
// =====================================================================
// Same K-major shape as the global cache, but in TLS.  Per-thread: no
// atomics, no false-sharing concern, but the structure is symmetric for
// readability.  Both LRC_CHUNK and LRC_MMAP enter L1 as long as their idx is
// ≤ `tls_l1_max_idx` (the per-thread byte-budget cut, set at thread arm).
//
// TLS model (§23 lesson preserved): the L1 ARRAY is plain `__thread` (GD)
// because it is too big for the initial-exec surplus.  Its per-thread base
// is taken once and cached in an IE-TLS pointer — hot-path access is one
// `fs:offset` read, never `__tls_get_addr`.

struct L1KArray {
    char *slots[LRC_N_MAX_L1 + 1];     // 25 char* = 200 B (no alignas — TLS)
};
ALLOC_TLS    L1KArray  tls_l1_array[LRC_K_L1];   // 32 × 200 = 6.4 KiB / thread, GD
ALLOC_TLS_IE L1KArray *tls_l1         = nullptr; // cached &tls_l1_array[0] (IE)
ALLOC_TLS_IE int       tls_l1_max_idx = -1;      // per-thread idx ceiling (set lazily)
// (teardown) Set true by `l1_drain()` once this thread's L1 has been flushed
// to the global L2 at thread exit.  A large/dedicated free arriving AFTER the
// drain — e.g. from a pthread_key destructor freeing an XThreadLocal buffer,
// which glibc runs AFTER the C++ thread_local `l1_drain` dtor — must NOT
// repopulate the L1: nothing will flush it again, so the block's units stay
// claimed forever (unbounded thread-exit stranding, see
// tests/alloc_thread_exit_free_test.cpp scenario B).  When set, `l1_push`
// refuses and `recycle_push` falls through to `global_push` (global L2 —
// touches no TLS, safe at teardown) or, on refusal, a direct release.
ALLOC_TLS_IE bool      s_l1_drained   = false;

// Live count of threads that have armed an L1.  Sloppy by design — each
// thread keeps the cut it computed at arm time; the global L2 cap is the
// hard ceiling (plan B).  Decremented at thread exit by `l1_drain`.
std::atomic<int> g_lrc_l1_threads{0};

void l1_drain() noexcept;   // fwd (defined after global_push)
// Drain sentinel — empty thread_local whose dtor flushes L1 to global at
// thread exit AND decrements the armed-thread count.  Split-storage (§23):
// the DATA is IE-TLS; this carries only the C++ destructor.
struct L1Drain { void touch() noexcept {} ~L1Drain() noexcept { l1_drain(); } };
thread_local L1Drain tls_l1_drain;

// Materialise the L1 base (one-time per thread: cache the GD array address
// into the IE-TLS pointer, compute the per-thread index cut, arm drain).
//
// Cut derivation:
//   per_thread     = g_lrc_cap / live_threads
//   cut_size       = per_thread / LRC_K_L1   // K slots per idx; geom-sum dominated by top
//   tls_l1_max_idx = min(lrc_idx_natural(cut_size), LRC_N_MAX_L1)
// This bounds each thread's L1 RSS to ≈ per_thread, so the aggregate stays
// inside g_lrc_cap.  Kind-agnostic — both LRC_CHUNK and LRC_MMAP fight for
// the same idx budget.
inline L1KArray *l1_base() noexcept {
    L1KArray *l1 = tls_l1;
    if(__builtin_expect(l1 == nullptr, 0)) {
        l1 = tls_l1 = &tls_l1_array[0];
        int n = g_lrc_l1_threads.fetch_add(1, std::memory_order_relaxed) + 1;
        size_t per_thread = g_lrc_cap.load(std::memory_order_relaxed) / (size_t)n;
        std::size_t cut_size = per_thread / LRC_K_L1;
        int cut_idx = lrc_idx_natural(cut_size);
        if(cut_idx > LRC_N_MAX_L1) cut_idx = LRC_N_MAX_L1;
        tls_l1_max_idx = cut_idx;
        tls_l1_drain.touch();                                          // arm thread-exit drain
    }
    return l1;
}

// ---- L1 (per-thread, no atomics) ----
// Pop a fitting block from this thread's L1 at (kstart…kstart+K), or
// nullptr.  Single owner ⇒ plain loads/stores, no CAS.
inline char *l1_pop_fit(std::size_t need, unsigned kind) noexcept {
    // (teardown) Symmetric with l1_push: a torn-down thread must not re-arm its
    // L1 here either — l1_base() would bump g_lrc_l1_threads again (live-thread
    // counter drift → undersized L1 cuts for the threads that remain).  Fall to
    // the global L2 / fresh claim instead.
    if(__builtin_expect(kame_thread_torn_down(), 0)) return nullptr;
    L1KArray *l1 = l1_base();
    int idx = lrc_idx(need, kind);
    if(idx > tls_l1_max_idx) return nullptr;
    int kstart = lrc_kstart_l1();
    for(int kk = 0; kk < LRC_K_L1; kk++) {
        int k = (kstart + kk) & (LRC_K_L1 - 1);
        char *b = l1[k].slots[idx];
        if( !b) continue;
        std::size_t sz = lrc_block_size(b, kind);
        if(sz >= need) {                                               // VERIFY size (band approx)
            l1[k].slots[idx] = nullptr;
            return b;
        }
        // too small for this need — leave it; a smaller request reuses it.
    }
    return nullptr;
}
// Cache a block in this thread's L1 at the first empty (k, idx) starting
// from kstart.  Refused when idx > cut (→ global) or all K slots occupied
// (→ global).
inline bool l1_push(char *base, std::size_t size, unsigned kind) noexcept {
    // (teardown) Post-drain frees must not refill the L1 — see s_l1_drained.
    // ALSO refuse once this thread is torn down: a thread that NEVER armed its
    // L1 (consumes only sub-32 KiB, or is a pure non-allocating consumer) has
    // `s_l1_drained == false` because l1_drain()'s thread_local dtor never ran
    // (it is only armed by l1_base()).  A cross-thread-origin large/dedicated
    // (>32 KiB) block freed in such a thread's pthread_key destructor would
    // otherwise re-arm a fresh L1 here that nothing ever drains → +1 chunk/cycle
    // permanent stranding (a narrow re-opening of the 30ea1daa leak class).
    // kame_thread_torn_down() catches it for ALL threads, armed or not, and
    // mirrors the bucket-tier sentinel guard; recycle_push then falls to
    // global_push (L2, no allocator TLS) — safe at teardown.
    if(__builtin_expect(s_l1_drained || kame_thread_torn_down(), 0)) return false;
    L1KArray *l1 = l1_base();
    int idx = lrc_idx(size, kind);
    if(idx > tls_l1_max_idx) return false;
    int kstart = lrc_kstart_l1();
    for(int kk = 0; kk < LRC_K_L1; kk++) {
        int k = (kstart + kk) & (LRC_K_L1 - 1);
        if( !l1[k].slots[idx]) {
            l1[k].slots[idx] = base;
            return true;
        }
    }
    return false;
}

// ---- Global L2 (shared, lock-free) ----
// Push to the global cache only (no L1 step).  Used both by `recycle_push`'s
// L1-miss path and by `l1_drain` to flush survivors at thread exit.
inline bool global_push(char *base, std::size_t size, unsigned kind) noexcept {
    if(g_lrc_bytes.load(std::memory_order_relaxed) + size
       > g_lrc_cap.load(std::memory_order_relaxed))
        return false;                                                  // over cap → caller releases
    int idx = lrc_idx(size, kind);
    int kstart = lrc_kstart_g();
    for(int kk = 0; kk < LRC_K_MAX; kk++) {
        int k = (kstart + kk) & (LRC_K_MAX - 1);
        char *expected = nullptr;
        if(g_lrc[k].slots[idx].compare_exchange_weak(
               expected, base, std::memory_order_acq_rel)) {
            g_lrc_bytes.fetch_add(size, std::memory_order_relaxed);
            return true;
        }
        // weak: spurious / occupied → next k
    }
    return false;                                                      // all K full → caller releases
}
// Pop a fitting block from the global cache, weak-CAS each slot (own-then-
// read-size) until a fit; livelock-free (bounded K iterations, no inner retry).
inline char *global_pop_fit(std::size_t need, unsigned kind) noexcept {
    int idx = lrc_idx(need, kind);
    int kstart = lrc_kstart_g();
    for(int kk = 0; kk < LRC_K_MAX; kk++) {
        int k = (kstart + kk) & (LRC_K_MAX - 1);
        char *b = g_lrc[k].slots[idx].load(std::memory_order_acquire);
        if( !b) continue;
        if( !g_lrc[k].slots[idx].compare_exchange_weak(
                 b, nullptr, std::memory_order_acq_rel))
            continue;                                                  // weak: spurious / taken → next k
        std::size_t sz = lrc_block_size(b, kind);                      // own it now → safe meta read
        g_lrc_bytes.fetch_sub(sz, std::memory_order_relaxed);
        if(sz >= need) return b;                                       // VERIFY size
        // too small (sub-band rounding): one put-back, else release
        char *exp = nullptr;
        if(g_lrc[k].slots[idx].compare_exchange_weak(
               exp, b, std::memory_order_acq_rel))
            g_lrc_bytes.fetch_add(sz, std::memory_order_relaxed);
        else
            lrc_release(b, sz, kind);
    }
    return nullptr;
}

// Public entry points: L1 first, then global.
inline char *recycle_pop_fit(std::size_t need, unsigned kind) noexcept {
    if(char *b = l1_pop_fit(need, kind)) return b;
    return global_pop_fit(need, kind);
}

// (§28.1) Amortised lazy drain of the global MMAP-tier cache.  Called once
// per LRC_MMAP push (only; the chunk tier doesn't hold enough RSS per slot
// to need this).  If at least LRC_LAZY_INTERVAL_NS has passed since this
// thread's last tick, re-stamp the TLS clock and try ONE slot at the
// per-thread cursor: strong-CAS-take it and release the block if occupied,
// nothing if empty.  Cursor advances every tick whether or not a release
// happens, so the entire MMAP-slot space is swept over time.
//
// Rationale: a long-running workload with frequent multi-MiB allocs would
// otherwise pin large blocks in the cache until cap, even if they aren't
// being reused.  Constant drain at ≈ 100 ticks/sec/thread → 100·N
// releases/sec aggregate (where N = active threads) caps the steady-state
// residency without explicit `kame_pool_set_large_cache_cap` calls.  Hot
// push-pop pairs (same slot popped before the cursor reaches it) are
// unaffected; an unlucky cursor collision steals one block per 10 ms ×
// thread, which is millions of cycles between work — negligible.
//
// Cost on the LRC_MMAP push path:
//   - 10 ms NOT elapsed: 1 steady_clock::now() (~30 ns) + 1 compare.
//   - 10 ms elapsed: + 1 atomic load + maybe 1 CAS + munmap (~10 µs).
// LRC_MMAP push is rare (multi-MiB free, ~1–1000/sec/thread); the per-push
// cost is well under 1 % of normal work.
//
// Race safety: each tick attempts at most 1 strong CAS.  Concurrent push to
// the same slot fails its CAS and tries the next k; concurrent pop sees
// null after we take.  No retry loop, no chance of livelock.
// (§28.1 / §28.3) Lazy drain interval — now runtime-tunable + auto-calibrated.
//
// Default 10 ms (= the original constexpr); auto-tune on the first
// LRC_MMAP push measures the host's `munmap(32 MiB)` cost once and stores
// `clamp(20 × munmap_ns, 1 ms .. 1 s)` so the per-thread worst-case
// wallclock fraction spent inside lazy-tick munmaps stays ≤ 5 %.  Sites
// with abnormally slow munmap (containers, VMs, etc.) thus self-throttle;
// fast hosts (HPC nodes with cheap TLB shootdown) keep the responsive
// 10 ms.  Override via `kame_pool_set_lazy_drain_interval_ms()` (user
// takes over; auto-tune is locked out) or via env `KAME_POOL_AUTO_TUNE=0`
// (skip the calibration entirely, keep the 10 ms default).
constexpr std::int64_t LRC_LAZY_INTERVAL_DEFAULT_NS = 10LL * 1000 * 1000;
// Lazy-drain interval (ns) — plain int64 + volatile, NOT std::atomic.  ns
// magnitude needs 64 bits but the value is read once per LRC_MMAP push
// (millisecond-scale interval) and stored only at startup / set_realtime_mode
// — extreme cold path on both sides.  An atomic<int64_t> would force a
// CMPXCHG8B (i486: libatomic call) on every read; a torn read here is
// benign — at worst one tick's lazy-drain interval is misread for one cycle.
// Storing as 32-bit pieces in declaration order also keeps any natural
// 32-bit-aligned load-tearing window contained to the low/high half.
volatile std::int64_t g_lrc_lazy_interval_ns = LRC_LAZY_INTERVAL_DEFAULT_NS;
std::atomic<bool>         g_lazy_auto_tune_done{false};

ALLOC_TLS_IE std::int64_t tls_lazy_last_ns = 0;                     // 0 = epoch ⇒ first tick fires
ALLOC_TLS_IE int          tls_lazy_cursor  = 0;                     // position in the MMAP-slot sweep

// (§28.3) One-shot measurement of `munmap(32 MiB after first-touch)` cost
// on this host.  Uses raw mmap/munmap (NOT kamepoolalloc — recursion
// safety).  Skipped when KAME_POOL_AUTO_TUNE=0.  Roughly:
//   per-thread tick rate = 1 / interval
//   per-tick blocked time = munmap_ns
//   per-thread wallclock fraction = munmap_ns / interval
// Target ≤ 5 % ⇒ interval ≥ 20 × munmap_ns.  RAISE-ONLY: if the host's
// munmap is fast enough that the default 10 ms already satisfies the 5 %
// target, KEEP the default — never lower it, since a single measurement
// can underestimate the realistic munmap cost (cold-cache, under-load
// values typically differ by 3–5× on the same host) and an over-eager
// down-tune below default makes pressure WORSE.  Only clamp the upward
// direction (1 s ceiling).
static void lrc_auto_tune_lazy_interval() noexcept {
#if !(defined(__WIN32__) || defined(WINDOWS) || defined(_WIN32))
    const char *env = std::getenv("KAME_POOL_AUTO_TUNE");
    if(env && env[0] == '0' && env[1] == '\0') return;

    constexpr std::size_t SZ = (std::size_t)32 * 1024 * 1024;
    void *p = mmap(nullptr, SZ, PROT_READ | PROT_WRITE,
                   MAP_ANON | MAP_PRIVATE, -1, 0);
    if(p == MAP_FAILED) return;
    // ALLOC_PAGE_SIZE: 16 KiB on Apple arm64, 64 KiB on PPC64, 4 KiB else
    // (allocator_prv.h §172–§177).  Compile-time constant — no <unistd.h>
    // dependency, so the auto-tune compiles cleanly on the macOS path
    // where `_SC_PAGESIZE` isn't directly visible from this TU.
    for(std::size_t off = 0; off < SZ; off += (std::size_t)ALLOC_PAGE_SIZE)
        ((volatile char *)p)[off] = 1;
    auto t0 = std::chrono::steady_clock::now();
    munmap(p, SZ);
    std::int64_t munmap_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();
    if(munmap_ns <= 0) return;

    std::int64_t interval = 20 * munmap_ns;
    if(interval <= LRC_LAZY_INTERVAL_DEFAULT_NS) return;            // default is already fine — keep it
    if(interval > 1000000000LL) interval = 1000000000LL;            // 1 s ceiling
    g_lrc_lazy_interval_ns = interval;
#endif
}

inline void lrc_lazy_mmap_one() noexcept {
    // (§28.3) First call on this process: auto-tune the interval from the
    // measured munmap cost.  CAS the gate so exactly one thread does the
    // measurement; others fall through using whatever value is current
    // (default, then the tuned value after this completes).
    if( !g_lazy_auto_tune_done.load(std::memory_order_acquire)) {
        bool exp = false;
        if(g_lazy_auto_tune_done.compare_exchange_strong(
               exp, true, std::memory_order_acq_rel))
            lrc_auto_tune_lazy_interval();
    }

    auto now = std::chrono::steady_clock::now();
    std::int64_t now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    if(now_ns - tls_lazy_last_ns
       < g_lrc_lazy_interval_ns) return;
    tls_lazy_last_ns = now_ns;

    // MMAP-tier slots are at idx (LRC_CHUNK_BND, LRC_N_MAX] across all K.
    constexpr int LRC_MMAP_IDX_COUNT = LRC_N_MAX - LRC_CHUNK_BND;
    constexpr int LRC_MMAP_SLOT_TOTAL = LRC_MMAP_IDX_COUNT * LRC_K_MAX;
    static_assert((LRC_K_MAX & (LRC_K_MAX - 1)) == 0,
                  "LRC_K_MAX must be a power of two for the mask below");

    int pos = tls_lazy_cursor;
    if(pos < 0 || pos >= LRC_MMAP_SLOT_TOTAL) pos = 0;
    int idx = LRC_CHUNK_BND + 1 + (pos / LRC_K_MAX);
    int k   = pos & (LRC_K_MAX - 1);
    int nxt = pos + 1;
    if(nxt >= LRC_MMAP_SLOT_TOTAL) nxt = 0;
    tls_lazy_cursor = nxt;

    char *b = g_lrc[k].slots[idx].load(std::memory_order_acquire);
    if( !b) return;
    if( !g_lrc[k].slots[idx].compare_exchange_strong(
             b, nullptr, std::memory_order_acq_rel))
        return;                                                    // racing pop/push took it
    std::size_t sz = lrc_block_size(b, (unsigned)LRC_MMAP);
    g_lrc_bytes.fetch_sub(sz, std::memory_order_relaxed);
    lrc_release(b, sz, (unsigned)LRC_MMAP);
}

inline bool recycle_push(char *base, std::size_t size, unsigned kind) noexcept {
    if(l1_push(base, size, kind)) return true;
    bool ok = global_push(base, size, kind);
    if(kind == (unsigned)LRC_MMAP) lrc_lazy_mmap_one();              // §28.1 amortised drain
    return ok;
}

// Thread-exit drain: flush every L1 entry to the global cache (push;
// release on refusal).  Touches only global state + per-block release —
// no allocator TLS — so it is safe in any TLS destruction order.
void l1_drain() noexcept {
    L1KArray *l1 = tls_l1;
    if( !l1) return;                                                   // L1 never used on this thread
    for(int k = 0; k < LRC_K_L1; k++) {
        for(int idx = 0; idx <= LRC_N_MAX_L1; idx++) {
            char *b = l1[k].slots[idx];
            if( !b) continue;
            l1[k].slots[idx] = nullptr;
            unsigned kind = lrc_kind_from_idx(idx);
            std::size_t sz = lrc_block_size(b, kind);
            if( !global_push(b, sz, kind))
                lrc_release(b, sz, kind);
        }
    }
    // (teardown) Block any later l1_push from this thread (e.g. a pthread_key
    // dtor's large/dedicated free, which runs after this C++ thread_local
    // dtor): the L1 will not be drained again, so a refill would strand the
    // block.  Post-drain frees route to global_push / direct release.
    s_l1_drained = true;
    tls_l1 = nullptr;          // defensive: force l1_base re-arm if ever re-used
    // Track LIVE concurrency, not cumulative spawns — otherwise a long-
    // running process that churns short-lived threads would drive the
    // count (and every new thread's cut) toward zero.  Sloppy (a thread
    // keeps the cut it computed at arm time); the global L2 cap is the
    // hard ceiling (plan B).
    g_lrc_l1_threads.fetch_sub(1, std::memory_order_relaxed);
}
// (§22) Definitions of the forward-declared helpers used by the earlier
// §15 dedicated-chunk paths (allocate_dedicated_chunk / deallocate).
char *large_recycle_pop(std::size_t need, unsigned kind) noexcept {
	return recycle_pop_fit(need, kind);
}
bool large_recycle_push(char *base, std::size_t size, unsigned kind) noexcept {
	return recycle_push(base, size, kind);
}
} // namespace

// (§26 / §28) Large-recycle cache RSS cap API.  `total_bytes` is the cache's
// target total resident footprint; we store HALF in g_lrc_cap because the
// cache's resident bytes ≈ 2·g_lrc_cap (global L2 ≤ g_lrc_cap AND the
// aggregate per-thread L1 ≈ g_lrc_cap, since each thread's L1 cut derives
// from g_lrc_cap/concurrency).  So g_lrc_cap = total/2 ⇒ L1+L2 ≈ total.
//
// (§28) After lowering the cap, EVICT to bring g_lrc_bytes below it
// synchronously.  Two-phase priority — preserve SIZE COVERAGE while there
// is surplus, only drop SIZE CLASSES when forced:
//
//   threshold = hw_concurrency × Σ S_i
//             where S_i ≈ LRC_LO × 2^(i/4) is the nominal size at idx i,
//             so the sum is the "ideal" bytes if every idx held one block
//             per armed thread.  hw_concurrency uses g_lrc_l1_threads
//             (live armed count — same source the L1 per-thread cut uses;
//             no platform CPU-count API).
//
//   Phase 1 (g_lrc_bytes > threshold) — REDUCE K.
//     Walk k = K_MAX-1 → 0, for each k all idxs.  Drops redundant copies
//     uniformly while leaving every idx with at least the "1 per thread"
//     ideal coverage.  Stops at max(cap, threshold).
//
//   Phase 2 (g_lrc_bytes ≤ threshold, > cap) — REDUCE N.
//     Walk idx = N_MAX → 0, for each idx all k.  Drops the largest size
//     classes first.  Stops at cap.
//
// Strong CAS so each slot is examined at most once; bounded by total slot
// count (LRC_K_MAX × (LRC_N_MAX+1) per phase).  The sloppy g_lrc_bytes
// counter governs the stop condition, so concurrent push/pop can interleave
// without causing a retry loop.  Cap-LOWER is a heavy explicit user op
// (potentially many munmap/madvise syscalls); cap-RAISE or UNCHANGED is
// fast (the byte short-circuit returns immediately).
extern "C" void kame_pool_set_large_cache_cap(std::size_t total_bytes) noexcept {
    size_t cap = total_bytes / 2u;
    g_lrc_cap.store(cap, std::memory_order_relaxed);

    int hw = g_lrc_l1_threads.load(std::memory_order_relaxed);
    if(hw < 1) hw = 1;
    // Σ S_i with linear interpolation within each octave: S_i = 2^(i/4)·LRC_LO
    // ≈ (4 + i%4) / 4 · (LRC_LO << i/4).  ~5 % above the exact 2^(i/4) for
    // i mod 4 ∈ {1,2,3}; the threshold is a heuristic so this slack is fine.
    size_t sum_S = 0;
    for(int i = 0; i <= LRC_N_MAX; i++) {
        int oct = i / 4, frac = i % 4;
        sum_S += ((size_t)LRC_LO << oct) * (4 + frac) / 4;
    }
    size_t threshold = (size_t)hw * sum_S;
    size_t phase1_stop = (cap > threshold) ? cap : threshold;

    auto evict = [](int k, int idx) noexcept {
        char *b = g_lrc[k].slots[idx].load(std::memory_order_acquire);
        if( !b) return;
        if( !g_lrc[k].slots[idx].compare_exchange_strong(
                 b, nullptr, std::memory_order_acq_rel))
            return;                                                    // racing pop took it; move on
        unsigned kind = lrc_kind_from_idx(idx);
        std::size_t sz = lrc_block_size(b, kind);
        g_lrc_bytes.fetch_sub(sz, std::memory_order_relaxed);
        lrc_release(b, sz, kind);
    };

    // Phase 1: reduce K (drop redundant duplicates uniformly across idxs).
    for(int k = LRC_K_MAX - 1; k >= 0; k--) {
        if(g_lrc_bytes.load(std::memory_order_relaxed) <= phase1_stop) break;
        for(int idx = 0; idx <= LRC_N_MAX; idx++) {
            if(g_lrc_bytes.load(std::memory_order_relaxed) <= phase1_stop) break;
            evict(k, idx);
        }
    }
    if(g_lrc_bytes.load(std::memory_order_relaxed) <= cap) return;
    // Phase 2: reduce N (drop largest size classes first).  Only reached
    // when cap < threshold and Phase 1 left us above cap.
    for(int idx = LRC_N_MAX; idx >= 0; idx--) {
        if(g_lrc_bytes.load(std::memory_order_relaxed) <= cap) break;
        for(int k = 0; k < LRC_K_MAX; k++) {
            if(g_lrc_bytes.load(std::memory_order_relaxed) <= cap) break;
            evict(k, idx);
        }
    }
}
extern "C" std::size_t kame_pool_get_large_cache_cap(void) noexcept {
	std::int64_t h = g_lrc_cap.load(std::memory_order_relaxed);
	return (std::size_t)((h < 0 ? 0 : h) * 2);
}

// (§28.3) Lazy-drain interval runtime API.  Default is 10 ms; on first
// LRC_MMAP push the library calibrates it from a single `munmap(32 MiB)`
// measurement.  Calling `set` here locks out the auto-tune (user wins)
// and stores the supplied value verbatim.  `ms == 0` is silently rejected
// (avoids divide-by-zero / hot ticking).  `get` returns the currently
// effective value, which may be the default, the auto-tuned, or the
// user-set value.
extern "C" void kame_pool_set_lazy_drain_interval_ms(unsigned int ms) noexcept {
	if(ms == 0) return;
	std::int64_t ns = (std::int64_t)ms * 1000000LL;
	g_lrc_lazy_interval_ns = ns;
	g_lazy_auto_tune_done.store(true, std::memory_order_release);   // lock out auto-tune
}
extern "C" unsigned int kame_pool_get_lazy_drain_interval_ms(void) noexcept {
	std::int64_t ns = g_lrc_lazy_interval_ns;
	if(ns <= 0) return 0;
	return (unsigned int)(ns / 1000000LL);
}

// (§30) Realtime-mode preset — silences the three background maintenance
// paths that can inject munmap / madvise latency into a measurement loop:
//   1. §28.1 lazy drain (per-LRC_MMAP-push tick that munmaps one cached
//      mmap-tier block — the only path that mmaps/munmaps OUTSIDE of an
//      explicit alloc/free call).  Set the interval to ~146 years so the
//      `now_ns - tls_lazy_last_ns < interval` guard is always true.
//   2. §28.3 auto-tune startup probe.  `g_lazy_auto_tune_done = true`
//      makes the first LRC_MMAP push skip the one-shot `munmap(32 MiB)`
//      measurement — the user has explicitly opted out of background
//      tuning, so we trust them.  (`set_lazy_drain_interval_ms` already
//      sets this flag for the same reason; we mirror it here for the
//      case where the user calls realtime_mode WITHOUT also pinning the
//      interval — keeping all three lazy/auto knobs in one place.)
//   3. §21 thread-exit reclaim — the `madvise(MADV_DONTNEED)` issued by
//      `release_dll_chunks_for_thread` / `deallocate_chunk` when a
//      worker thread exits.  Real measurement programs usually keep
//      their worker pool alive for the whole run, but if they don't, a
//      thread teardown during the realtime section would block on per-
//      chunk madvise calls — turning this off makes thread exit
//      essentially free at the cost of holding RSS until process exit.
//
// `enable == 0` reverts to the documented defaults (10 ms / auto-tune
// re-armed / thread-exit reclaim on), so test programs and toggling
// callers don't need to remember the prior values.
extern "C" void kame_pool_set_realtime_mode(int enable) noexcept {
	if(enable) {
		// (1) Lazy drain interval → effectively infinite.  INT64_MAX/2
		// avoids any overflow in the `now - last < interval` compare
		// even if `last` is 0 (epoch) and `now` is a large monotonic
		// time_since_epoch tick.
		g_lrc_lazy_interval_ns =
		    std::numeric_limits<std::int64_t>::max() / 2;
		// (2) Auto-tune locked out (would otherwise overwrite the
		// interval on the first LRC_MMAP push).
		g_lazy_auto_tune_done.store(true, std::memory_order_release);
		// (3) Thread-exit madvise off.
		PoolAllocatorBase::s_thread_exit_reclaim.store(
		    0, std::memory_order_relaxed);
	}
	else {
		g_lrc_lazy_interval_ns = LRC_LAZY_INTERVAL_DEFAULT_NS;
		g_lazy_auto_tune_done.store(false, std::memory_order_release);
		PoolAllocatorBase::s_thread_exit_reclaim.store(
		    1, std::memory_order_relaxed);
	}
}

// =====================================================================
// (§19) Large-alloc tier — single-mmap, radix-registered, munmap-able.
// =====================================================================
// Each large_va allocation is its own 32-MiB-aligned mmap of size
// `round_up(user_size + PAGE, PAGE)`.  The first page holds a
// LargeAllocMeta; the user receives `base + PAGE`.  One radix slot
// (KAME_RADIX_LARGE) covers the alloc.  On free, the slot is CAS-cleared,
// then the entire mmap is unmap'd — VA returns to the kernel, unlike
// pool regions which are push-only.
//
// Size bounds:
//   - lower: enforced by the caller (allocate_large_size_or_malloc) —
//     sizes that fit `allocate_dedicated_chunk` (≤ 4 MiB - K_MAX) stay
//     in the pool to avoid radix-slot overhead for many small-large allocs.
//   - upper: one radix slot covers 32 MiB of VA, and the meta consumes
//     one page, so `size <= ALLOC_MIN_MMAP_SIZE - ALLOC_PAGE_SIZE`.
//     Beyond that the request falls through to libc.
//
// Concurrency: lock-free.  Insert and clear are atomic CAS / release
// stores on a single L2 slot.  A racing reader either observes the live
// kind (and dereferences valid meta — alloc/free both keep meta intact
// until after the slot is cleared) or KAME_RADIX_ABSENT (falls through
// to libc free, matching libc's behaviour for foreign pointers).
//
// The `s_last_region_base` cache in `radix_lookup_slow` skips
// KAME_RADIX_LARGE entries so a §19 base never lingers in another
// thread's TLS after its munmap.
void *
PoolAllocatorBase::allocate_large_va(std::size_t size) noexcept {
	std::size_t mmap_size =
	    (size + ALLOC_PAGE_SIZE + ALLOC_PAGE_SIZE - 1u) &
	    ~(std::size_t)(ALLOC_PAGE_SIZE - 1u);
	// (§21) Recycle a cached region first (warm VA + pages, no syscalls).
	// A cache hit returns a radix-CLEARED but still-mapped region; we
	// re-register it below.  The cached region's real size (meta->mmap_size,
	// preserved) is ≥ mmap_size and ≤ 2× it (pop_fit's fit window).
	//
	// (§27/§35) BUT only for the ≤ LRC_HI (= 256 MiB) tier.  The recycle
	// cache's log-index space (lrc_idx) tops out at LRC_HI: every size ≥
	// LRC_HI collapses to the single top slot, where the only pop gate is
	// `cached_size ≥ need` with NO upper bound — so a cached huge block
	// could satisfy (and pin the full RSS of) a much smaller huge request.
	// Allocs ≥ LRC_HI therefore skip the cache entirely: mmap fresh here,
	// munmap on free (deallocate_large_va mirrors this gate).  This matches
	// how libc / jemalloc / mimalloc treat their huge class.  Sizes BELOW
	// LRC_HI (incl. 100 MB-class image buffers ⇒ idx ≈ 33) land in a bounded
	// band and are cached normally as multi-region spans.
	bool cacheable = (mmap_size <= LRC_HI);
	char *base = cacheable ? recycle_pop_fit(mmap_size, LRC_MMAP) : nullptr;
	bool recycled = (base != nullptr);
	if( !recycled) {
		base = large_va_raw_map(mmap_size);
		if( !base) return nullptr;
	}
	// Write/refresh the meta.  On recycle, mmap_size keeps the cached
	// region's ACTUAL (larger-or-equal) size so a later free re-caches it
	// at the right size and malloc_usable_size stays truthful.
	LargeAllocMeta *meta = reinterpret_cast<LargeAllocMeta *>(base);
	meta->magic      = KAME_LARGE_ALLOC_MAGIC;
	meta->alloc_size = size;
	if( !recycled) meta->mmap_size = mmap_size;
	meta->numa_node  = 0;  // (§14C-style NUMA bind could be added here)
	writeBarrier();
	// Publish in the radix LAST.  Any concurrent reader's `radix_lookup`
	// either sees KAME_RADIX_ABSENT (and falls through to libc) or
	// KAME_RADIX_LARGE (and reads the meta we just published,
	// release-paired with this store).
	radix_insert(base, (uint32_t)KAME_RADIX_LARGE);
	// (§28.2) Track live large-alloc count + bytes for kame_pool_get_stats.
	// On a recycled hit we've just transferred the block out of the cache
	// (recycle_pop_fit already did `g_lrc_bytes -= sz`) so it correctly
	// enters the "live in program" bucket here either way.
	stats_inc_large(mmap_size);
	return base + ALLOC_PAGE_SIZE;
}

void
PoolAllocatorBase::deallocate_large_va(void *p) noexcept {
	char *base = reinterpret_cast<char *>(
	    (uintptr_t)p & ~((uintptr_t)ALLOC_MIN_MMAP_SIZE - 1u));
	LargeAllocMeta *meta = reinterpret_cast<LargeAllocMeta *>(base);
	std::size_t mmap_size = meta->mmap_size;
	// (§28.2) Leaving the "live in program" bucket — whether we cache-park
	// it or release outright, the program no longer holds this block.
	stats_dec_large(mmap_size);
	// Clear radix slot FIRST — any racing reader now sees absent and
	// routes to libc free (and a double-free of this pointer lands in
	// libc, not the cache); once cleared, the region is invisible to the
	// radix.  Done BEFORE both the cache push and any munmap.
	radix_clear(base);
	// Invalidate the per-thread region cache so a same-thread lookup of
	// this base doesn't false-hit (the slot is cleared, but the fast-path
	// cache shortcuts the slot read).
	if((uintptr_t)base == kame_page()->last_region_base)
		kame_page()->last_region_base = RADIX_CACHE_EMPTY;
	// (§21) Try to recycle into the warm cache (still mapped, warm).  On
	// overflow / over-cap the cache returns false and we munmap now.
	// (§27) Huge allocs (mmap_size > LRC_HI) bypass the cache — see
	// allocate_large_va.  The `||` short-circuits so recycle_push (hence
	// lrc_idx) is NEVER called with a > 32 MiB size.
	if(mmap_size > LRC_HI || !recycle_push(base, mmap_size, LRC_MMAP))
		large_va_raw_unmap(base, mmap_size);
}

// single consolidated TLS struct holds all per-thread state
// for each (ALIGN, FS, DUMMY) instantiation.
template <unsigned int ALIGN, bool FS, bool DUMMY>
ALLOC_TLS typename PoolAllocator<ALIGN, FS, DUMMY>::ThreadLocalState
    PoolAllocator<ALIGN, FS, DUMMY>::s_tls;

// (§S7) The §36 s_orphan_head Treiber stack is retired — the
// atomic_shared_ptr orphan chain below is the sole orphan mechanism.
//
// NEVER-DESTROYED accessor (not a plain static member).  The head holds a
// chain-ref on the first orphan node; a static member's process-exit
// destructor would drop that ref outside the scrub gate and dispose a
// still-non-empty orphan, destructing a live chunk's PoolAllocator (then an
// atexit free() of one of its slots hits the now-pure-virtual
// `deallocate_pooled` — the 12/14 Linux ctest "pure virtual method called"
// abort after the chain flip).  Placement-new into a function-local static
// byte buffer: the atomic_shared_ptr is constructed once (thread-safe init
// guard) and its destructor is NEVER registered, so it leaks at exit —
// exactly as every region mmap does (`mmap_new_region`: "Regions never
// unmap").  The two backing statics (`buf`, `head`) are trivially
// destructible, so they add no atexit work either.
template <unsigned int ALIGN, bool FS, bool DUMMY>
atomic_shared_ptr<PoolAllocator<ALIGN, FS, DUMMY> > &
PoolAllocator<ALIGN, FS, DUMMY>::s_orphan_chain_head() noexcept {
	alignas(atomic_shared_ptr<PoolAllocator>)
	    static unsigned char buf[sizeof(atomic_shared_ptr<PoolAllocator>)];
	static atomic_shared_ptr<PoolAllocator> *head =
	    ::new (static_cast<void *>(buf)) atomic_shared_ptr<PoolAllocator>();
	return *head;
}

//! (Path B Stage 1) Treiber push onto the atomic_shared_ptr orphan chain.
//! `refcnt` is established to 1 BEFORE publish — at owner-exit the chunk is
//! still owner-private (off every per-thread DLL; not yet on the chain), so
//! no concurrent `load_shared` can have pinned it, making the plain store
//! race-free.  Adopt into a local_shared_ptr (which takes that 1) and CAS it
//! onto the head (mirrors atomic_intrusive_chain_test.cpp's push_head); head +
//! the chunk's m_orphan_next hold the chain-ref.
template <unsigned int ALIGN, bool FS, bool DUMMY>
void PoolAllocator<ALIGN, FS, DUMMY>::orphan_chain_push(
    PoolAllocator<ALIGN, DUMMY, DUMMY> *craw) noexcept {
	PoolAllocator *c = static_cast<PoolAllocator *>(craw);  // upcast to FS=true base (chain node type)
	local_shared_ptr<PoolAllocator> n;
	if(c->m_owner_self_ref) {
		// (Path B owner-ref) Re-owned chunk being re-orphaned at owner-exit:
		// MOVE its self-ref onto the chain (self-ref → chain-ref), preserving
		// refcnt — INCLUDING any residual scrub pin.  A `refcnt.store(1)` here
		// would CLOBBER that pin's count, so a later pin-drop would dispose the
		// chunk while it is back on the chain (premature free).  Move keeps the
		// true count; oc's self-ref goes null.
		n = std::move(c->m_owner_self_ref);
	}
	else {
		// Fresh chunk's FIRST orphaning: refcnt is 0 (never refcounted) and no
		// scrub pin can exist (it was never on the chain) — establish 1 and
		// adopt.  (Owner-private off every DLL here, so the plain store is safe.)
		c->refcnt.store(1, std::memory_order_relaxed);
		n = local_shared_ptr<PoolAllocator>(c);
	}
	local_shared_ptr<PoolAllocator> old(s_orphan_chain_head());
	for(;;) {
		n->m_orphan_next = old;
		if(s_orphan_chain_head().compareAndSwap(old, n)) break;
	}
}

//! (Path B step 4) Reclaim pass — see allocator_prv.h.  Walks the orphan chain
//! holding local_shared_ptr pins; CAS-unlinks each DEAD (empty, MASK_CNT==0)
//! node from its predecessor (or the head).  Unlinking drops the chain-ref;
//! when this pass releases its own pin on the node (next iteration) refcnt
//! hits 0 → atomic_intrusive_dispose → bucket_release_chunk frees the region.
//! Orphans never refill (adopt deferred) so MASK_CNT==0 is stable; a plain
//! read of m_flags_packed can only be stale-HIGH (skip now, reclaim next
//! pass) = safe-side.  Dead-only + reachability-preserving relink; a lost CAS
//! restarts from head — multiple scrubbers are safe-side.
template <unsigned int ALIGN, bool FS, bool DUMMY>
void PoolAllocator<ALIGN, FS, DUMMY>::orphan_chain_scrub() noexcept {
	local_shared_ptr<PoolAllocator> pred;                      // empty ⇒ pred is head
	local_shared_ptr<PoolAllocator> cur(s_orphan_chain_head());
	while(cur) {
		local_shared_ptr<PoolAllocator> nxt(cur->m_orphan_next);
		if((cur->m_flags_packed & MASK_CNT) == 0u) {           // dead orphan → unlink
			bool ok = pred ? pred->m_orphan_next.compareAndSet(cur, nxt)
			               : s_orphan_chain_head().compareAndSet(cur, nxt);
			if(ok) { cur = nxt; continue; }                    // unlinked; pred unchanged
			pred.reset();                                      // CAS lost → restart from head
			cur = local_shared_ptr<PoolAllocator>(s_orphan_chain_head());
			continue;
		}
		pred = cur;                                            // live orphan → keep, advance
		cur = nxt;
	}
}

//! (Path B adopt) Treiber-pop the head node off the orphan chain — see
//! allocator_prv.h.  Returns the held local_shared_ptr (the caller keeps it
//! through the BIT_OWNED claim); clears the popped node's m_orphan_next.
template <unsigned int ALIGN, bool FS, bool DUMMY>
local_shared_ptr<PoolAllocator<ALIGN, FS, DUMMY> >
PoolAllocator<ALIGN, FS, DUMMY>::orphan_chain_pop() noexcept {
	local_shared_ptr<PoolAllocator> old(s_orphan_chain_head());
	for(;;) {
		if( !old) return local_shared_ptr<PoolAllocator>();   // chain empty
		local_shared_ptr<PoolAllocator> nxt(old->m_orphan_next);
		if(s_orphan_chain_head().compareAndSwap(old, nxt)) {
			old->m_orphan_next = local_shared_ptr<PoolAllocator>();  // off the chain
			return old;
		}
		// old reloaded by compareAndSwap on failure → retry
	}
}

// (§S7) §36 orphan_push / orphan_pop (the raw Treiber-stack helpers reusing
// m_dll_next + the 18-bit ABA tag) retired — owner-exit pushes the
// atomic_shared_ptr chain (orphan_chain_push) and adopt pops it
// (orphan_chain_pop), both refcount-safe.

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
// an earlier change layout uses three stages with explicit ALIGN values (64,
// 256, 1024).  ALLOC_ALIGN1 (= 32 on 64-bit) is retained as the ALIGN
// of the legacy FS=false buckets 6/8/10/12/14 (slot sizes 96/128/160/
// 192/224); bucket 16 (size 256) uses ALLOC_ALIGN(256) = ALLOC_ALIGN2
// = 256.  So we need ALIGN=32, 64, 256, 1024 — four total instantiations.
template class PoolAllocator<32u, false>;     // buckets 6, 8, 10, 12, 14
template class PoolAllocator<64u, false>;     // buckets 17..24
template class PoolAllocator<256u, false>;    // bucket 16 + buckets 25..32
template class PoolAllocator<1024u, false>;   // buckets 33..40

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
