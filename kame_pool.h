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

/*
 * kame_pool.h — pure-C interface to the kamepoolalloc pool allocator.
 *
 * Two ways to use this allocator:
 *
 *   (1) Process-wide drop-in malloc replacement.  Link against
 *       libkamepoolalloc and the dylib's `__DATA,__interpose`
 *       section (macOS) / strong-symbol `free`/`realloc` (Linux glibc)
 *       intercept every libc malloc/free/realloc call automatically.
 *       Nothing else to do — your existing `malloc()` / `free()` /
 *       `realloc()` calls are now pool-backed where appropriate.
 *
 *   (2) Explicit pool API.  Include <kame_pool.h>, call
 *       `kame_pool_malloc()` / `kame_pool_free()` / etc. directly.
 *       Useful when:
 *         - you cannot LD_PRELOAD / interpose (static link, restricted
 *           runtime, sandbox);
 *         - you want a specific subset of code to use the pool while
 *           the rest stays on libc malloc;
 *         - you are writing C/Rust/Go bindings that need a stable C ABI.
 *
 * Path (2) does NOT require the C++ side `activateAllocator()` to have
 * been called.  Pre-activation calls (or post-thread-teardown calls)
 * transparently fall through to libsystem `malloc()` / `free()`, so the
 * functions are safe to call from any context — including library
 * static initialisers, signal-safe paths (after a longjmp, etc.), and
 * thread destructors.  The price is that very-early-init allocations
 * never see the pool's fast path; explicitly call `activateAllocator()`
 * (C++ side, from <allocator.h>) or instantiate `KamePooledAllocGuard`
 * to enable the pool path.
 *
 * Thread-safety: every function below is fully reentrant and safe to
 * call concurrently from any thread.  The underlying pool uses
 * lock-free per-thread bucketed allocation; the C wrappers add only a
 * trivial null check and forwarding call.
 *
 * Error reporting follows libc malloc conventions:
 *   - allocation failure returns NULL and sets `errno = ENOMEM`,
 *   - posix_memalign returns the error code as its return value (no
 *     `errno` set), per POSIX,
 *   - all other functions never fail in ways requiring errno.
 *
 * Alignment: the pool guarantees `max_align_t` alignment (16 B on every
 * supported arch) for all sizes through the standard malloc/realloc
 * path.  Use `kame_pool_aligned_alloc()` / `kame_pool_posix_memalign()`
 * for stricter alignment; alignments up to 16 B are served by the pool,
 * larger alignments fall back to the system `posix_memalign()`.
 */

#ifndef KAMEPOOLALLOC_KAME_POOL_H_
#define KAMEPOOLALLOC_KAME_POOL_H_

#include <stddef.h>  /* size_t */

#ifdef __cplusplus
extern "C" {
#  define KAMEPOOLALLOC_NOEXCEPT noexcept
#else
#  define KAMEPOOLALLOC_NOEXCEPT
#endif

/*
 * Standard malloc/free family.  Identical semantics to ISO C
 * stdlib counterparts; only the backing implementation differs.
 *
 *   kame_pool_malloc(0) returns a unique freeable pointer (one of two
 *     standard-compliant choices).
 *   kame_pool_calloc(n, sz) returns NULL with errno=ENOMEM on overflow.
 *   kame_pool_realloc(NULL, n) == kame_pool_malloc(n).
 *   kame_pool_realloc(p, 0)    frees p and returns NULL.
 *   kame_pool_free(NULL) is a no-op.
 */
void  *kame_pool_malloc(size_t size) KAMEPOOLALLOC_NOEXCEPT;
void  *kame_pool_calloc(size_t nmemb, size_t size) KAMEPOOLALLOC_NOEXCEPT;
void  *kame_pool_realloc(void *p, size_t size) KAMEPOOLALLOC_NOEXCEPT;
void   kame_pool_free(void *p) KAMEPOOLALLOC_NOEXCEPT;

/*
 * Aligned allocation.  `alignment` must be a power of two; for
 * kame_pool_posix_memalign() it must additionally be a multiple of
 * sizeof(void*).  Alignments <= 16 B are served by the pool;
 * larger alignments fall back to the system allocator.
 *
 * Alignment ceiling: the pool serves alignments up to and INCLUDING
 * `ALLOC_MIN_CHUNK_SIZE` (256 KiB) on every supported OS, including
 * Windows — the bucket tier (alignment ≤ 4096) hands out slots from
 * the chunk's mempool which itself sits on a 256 KiB unit boundary, and
 * the dedicated-chunk tier (4096 < alignment ≤ 256 KiB) returns a chunk
 * whose payload starts at that same unit boundary.  Both paths are pure
 * pool memory and free via the ordinary `kame_pool_free` — no
 * `_aligned_free` lifetime concern.
 *
 * Alignments > 256 KiB return ENOMEM (POSIX falls through to
 * `posix_memalign` for libc-backed satisfaction; Windows returns null
 * because `kame_pool_free` has no way to pair with `_aligned_free` for
 * libc-aligned pointers).  Use the platform-native `_aligned_malloc` /
 * `_aligned_free` directly for those rare cases on Windows, or C++
 * `operator new(size, std::align_val_t{N})` which carries the
 * alignment into the matching `operator delete`.
 */
void  *kame_pool_aligned_alloc(size_t alignment, size_t size) KAMEPOOLALLOC_NOEXCEPT;
int    kame_pool_posix_memalign(void **memptr, size_t alignment, size_t size) KAMEPOOLALLOC_NOEXCEPT;

/*
 * Returns the actual allocated size of `p` (which may exceed the
 * size requested at allocation, due to bucket rounding).  Returns 0
 * for foreign pointers (those not allocated through this pool).
 * Safe on NULL.
 *
 * Use case: realloc-elision in client code that maintains its own
 * vector-growth heuristic; pass the actual size to vector::reserve()
 * to avoid spurious reallocations.
 */
size_t kame_pool_malloc_usable_size(const void *p) KAMEPOOLALLOC_NOEXCEPT;

/*
 * Runtime memory cap.  Same semantics as the C++ declarations in
 * allocator.h, but with C-linkage so pure-C consumers can call them.
 *
 *   kame_pool_set_max_bytes(0)       — disable cap (default).
 *   kame_pool_set_max_bytes(N)       — round N up to a multiple of
 *                                      32 MiB, refuse mmap of fresh
 *                                      regions once that limit is hit.
 *   kame_pool_get_max_bytes()        — current cap (`SIZE_MAX` if unset).
 *   kame_pool_reserved_bytes()       — total bytes the pool has mmap'd.
 */
void   kame_pool_set_max_bytes(size_t max_bytes) KAMEPOOLALLOC_NOEXCEPT;
size_t kame_pool_get_max_bytes(void) KAMEPOOLALLOC_NOEXCEPT;
size_t kame_pool_reserved_bytes(void) KAMEPOOLALLOC_NOEXCEPT;

/*
 * Large-recycle cache RSS cap (§25/§26 — the warm-reuse cache for large
 * allocations, 256 KiB .. 32 MiB).  Distinct from kame_pool_set_max_bytes
 * (which caps fresh region mmaps).  `total_bytes` is the cache's target
 * total resident footprint; it is split internally — ~half to the shared
 * global L2 and ~half to the aggregate per-thread L1 — so L1+L2 ≈
 * total_bytes.  Default ≈ 2 GiB.  0 effectively disables the cache (every
 * large free releases immediately).
 *
 * Cheap: a single atomic store.  The global L2 honours the new cap on the
 * very next op.  Per-thread L1 sizing is derived once when each thread
 * first uses the cache, so a mid-run change applies to the L2 and to
 * threads that arm their L1 afterwards, but NOT retroactively to
 * already-armed threads — set at startup for an exact bound.
 */
void   kame_pool_set_large_cache_cap(size_t total_bytes) KAMEPOOLALLOC_NOEXCEPT;
size_t kame_pool_get_large_cache_cap(void) KAMEPOOLALLOC_NOEXCEPT;

/*
 * Amortised lazy drain of the global MMAP-tier cache (§28.1).  On every
 * LRC_MMAP `free` (4..64 MiB), if at least `interval_ms` have passed since
 * this thread's last tick, one slot is examined and (if occupied) released.
 * Keeps the steady-state cache from growing unboundedly under sustained
 * mmap-tier allocation, without explicit `set_large_cache_cap` calls.
 *
 *   kame_pool_set_lazy_drain_interval_ms(N)
 *     Set the interval (locks out auto-tune; user wins).  N==0 is rejected
 *     to avoid hot-ticking.  Pass a large value (e.g. 3600000 = 1 hour) to
 *     effectively disable the lazy drain.
 *   kame_pool_get_lazy_drain_interval_ms()
 *     Current effective interval in ms.
 *
 * Default 10 ms.  On the first LRC_MMAP push the library auto-calibrates
 * the interval from a single `munmap(32 MiB)` measurement so that the
 * per-thread worst-case wallclock fraction spent inside lazy-tick munmaps
 * stays ≤ 5 %.  Sites with abnormally slow munmap (containers, VMs)
 * self-throttle to e.g. 100 ms; HPC nodes keep the responsive 10 ms.
 * Override the calibration via env `KAME_POOL_AUTO_TUNE=0` (skip; keep
 * the 10 ms default) or by calling `set_lazy_drain_interval_ms()` early
 * in process startup.
 */
void         kame_pool_set_lazy_drain_interval_ms(unsigned int ms) KAMEPOOLALLOC_NOEXCEPT;
unsigned int kame_pool_get_lazy_drain_interval_ms(void) KAMEPOOLALLOC_NOEXCEPT;

/*
 * Thread-exit page reclamation toggle.  Default ENABLED: when a thread
 * exits, the pool madvise(MADV_DONTNEED)'s the slot pages of the chunks
 * it releases, returning RSS promptly.  Pass 0 to disable (skip the
 * thread-exit madvise — ~30% faster thread teardown, at the cost of
 * holding the freed pages resident until the kernel reclaims them at
 * process exit or the pages are recycled by a later allocation).  Tune
 * for workloads that rapidly spawn/exit threads and don't track
 * steady-state RSS.  Mid-run frees always reclaim regardless of this
 * setting.
 */
void   kame_pool_set_thread_exit_reclaim(int enable) KAMEPOOLALLOC_NOEXCEPT;

/*
 * Realtime-mode preset — silence all background maintenance that could
 * inject unpredictable munmap/madvise latency into a tight measurement
 * loop.  Call ONCE at process startup, before the realtime section
 * begins.  Equivalent to:
 *
 *   kame_pool_set_lazy_drain_interval_ms(huge)   // §28.1 lazy drain
 *                                                // off (the only path
 *                                                // that munmaps mid-run)
 *   (auto-tune is locked-out as a side-effect    // §28.3 — no startup
 *    of the set above)                           //  munmap probe either
 *   kame_pool_set_thread_exit_reclaim(0)         // §21 thread teardown
 *                                                // madvise off
 *
 * Pass 0 to restore the defaults (10 ms lazy interval / auto-tune armed /
 * thread-exit reclaim on).  Realtime mode does NOT cap allocation or
 * change the freelist / chunk-claim hot paths; it only silences
 * background maintenance.  Mid-run allocations still mmap fresh chunks
 * if the pool is starved; if zero mmap latency is required, pre-allocate
 * enough chunks at startup (e.g. by deliberately allocating + freeing a
 * representative working-set before the realtime section).
 */
void   kame_pool_set_realtime_mode(int enable) KAMEPOOLALLOC_NOEXCEPT;

/* ===================================================================
 * (§75) Realtime: per-thread gating, OS policy, drain, prewarm
 * ===================================================================
 * `kame_pool_set_realtime_mode` above is PROCESS-wide and silences
 * background maintenance.  The calls below are the complementary
 * per-thread half and are ORTHOGONAL to it — a realtime program calls
 * both: set_realtime_mode(1) once at startup, then mark each
 * time-critical thread (or use the kame::rt_section RAII guard below).
 *
 * A thread marked realtime never enters the kernel to RETURN memory
 * from its own free() calls: the chunk madvise is skipped (pages stay
 * warm, the chunk is still immediately recyclable) and a large-tier
 * munmap is parked on a pending list.  Both are settled later by
 * kame_pool_rt_drain().  Gating is per-thread because the reclaim
 * syscall runs on the freeing thread, so this confines the extra RSS
 * to the realtime thread's own working set.
 *
 * The ALLOCATION side cannot be deferred, so it takes a policy.
 */
void   kame_pool_set_realtime_thread(int enable) KAMEPOOLALLOC_NOEXCEPT;
int    kame_pool_get_realtime_thread(void) KAMEPOOLALLOC_NOEXCEPT;

/* What to do when a realtime thread's allocation needs a NEW mapping. */
enum {
    KAME_RT_OS_ALLOW = 0,  /* map anyway, do not even count (default) */
    KAME_RT_OS_COUNT = 1,  /* map anyway, count it (observability)    */
    KAME_RT_OS_FAIL  = 2,  /* refuse: the allocation degrades to libc */
    KAME_RT_OS_ABORT = 3   /* count, print the site, abort (CI/debug) */
};
/*
 * NOTE on coverage: FAIL/ABORT are honoured at the two sites where a
 * refusal degrades safely — the 32 MiB region mmap and the large-tier
 * mmap, both of which already have a "fall back to libc" nullptr path.
 * A radix-leaf mapping is COUNTED but never refused: without its leaf a
 * region goes unregistered and later frees of pointers inside it would
 * be routed to libc free() — corruption rather than degradation.
 * Prewarming is what actually keeps all three off the realtime path.
 */
void   kame_pool_set_rt_os_policy(int policy) KAMEPOOLALLOC_NOEXCEPT;
int    kame_pool_get_rt_os_policy(void) KAMEPOOLALLOC_NOEXCEPT;

/*
 * Violation counter: times a realtime thread actually entered the kernel
 * for a NEW mapping.  This is the number a realtime test asserts stays
 * zero across its critical section after prewarming.  The deferred_*
 * counters are informational (realtime mode working as designed), and
 * pending_bytes is the VA still held by deferred unmaps — the growth
 * traded for determinism, visible rather than silent.
 */
unsigned long long kame_pool_rt_violations(void) KAMEPOOLALLOC_NOEXCEPT;
unsigned long long kame_pool_rt_deferred_reclaims(void) KAMEPOOLALLOC_NOEXCEPT;
unsigned long long kame_pool_rt_deferred_unmaps(void) KAMEPOOLALLOC_NOEXCEPT;
size_t kame_pool_rt_pending_bytes(void) KAMEPOOLALLOC_NOEXCEPT;

/*
 * (G5) Ceiling on the VA the realtime munmap deferral may park.  An
 * UNBOUNDED deferral queue would trade a bounded free() tail for
 * unbounded memory — measured, 40 deferred 300 MiB frees park 12.6 GB.
 * Past the cap a realtime free releases INLINE instead of parking more
 * (bounded memory beats a bounded tail) and kame_pool_rt_forced_releases()
 * counts it, so the trade is never silent.  Default 1 GiB, matching the
 * large-recycle cache's own default appetite; lower it for tighter RSS, or
 * simply call kame_pool_rt_drain() once per control cycle.
 *
 * A program with both realtime and ordinary threads also settles the
 * backlog by itself: each non-realtime large free releases at most ONE
 * parked block (bounded, so no single free inherits the queue).
 *
 * A strict realtime check asserts BOTH rt_violations() == 0 AND
 * rt_forced_releases() == 0 over the critical section.
 */
void   kame_pool_set_rt_pending_cap(size_t bytes) KAMEPOOLALLOC_NOEXCEPT;
size_t kame_pool_get_rt_pending_cap(void) KAMEPOOLALLOC_NOEXCEPT;
unsigned long long kame_pool_rt_forced_releases(void) KAMEPOOLALLOC_NOEXCEPT;
void   kame_pool_rt_reset_counters(void) KAMEPOOLALLOC_NOEXCEPT;

/* One-line "this realtime section mapped memory N times" warning on
 * stderr.  Exists as a C entry point so the kame::rt_section guard below
 * can report without pulling <cstdio> into this header; callable
 * directly as well. */
void   kame_pool_rt_report_violations(unsigned long long count)
                                                 KAMEPOOLALLOC_NOEXCEPT;

/*
 * Perform all reclaim work the realtime paths deferred (the mi_collect /
 * malloc_trim / arena.N.purge analogue): pending large-tier unmaps, this
 * thread's L1 recycle cache, and the global L2 cache.  Call from a
 * non-critical phase — never inside the time-critical section, since
 * this is precisely the syscall batch realtime mode exists to keep out.
 */
void   kame_pool_rt_drain(void) KAMEPOOLALLOC_NOEXCEPT;

/*
 * Prewarm: allocate + page-TOUCH + free `counts[i]` blocks of `sizes[i]`
 * so the chunks, their regions, the radix leaves and this thread's
 * allocator TLS all exist before the time-critical section.  Touching is
 * the point: allocate/free alone leaves pages mapped-but-unfaulted, so
 * the first realtime write would still take a minor fault.  Call from
 * EACH realtime thread (the allocator TLS is per-thread).
 * Returns 0 on success, -1 if the working set does not fit.
 */
int    kame_pool_prewarm(const size_t *sizes, const unsigned *counts,
                         unsigned n) KAMEPOOLALLOC_NOEXCEPT;

/*
 * Pre-reserve `n_regions` 32 MiB regions up front (prefault != 0 also
 * touches their slot pages).  Regions never unmap, so this is permanent.
 * Belt-and-braces companion to prewarm for when the working-set SIZE is
 * known but its size classes are not.  Returns how many were created.
 */
unsigned kame_pool_reserve_regions(unsigned n_regions,
                                   int prefault) KAMEPOOLALLOC_NOEXCEPT;

/*
 * Observability — snapshot of pool counters at the moment of the call.
 *
 * ABI versioning: callers set `version = KAME_POOL_STATS_VERSION` before
 * calling.  Future fields are appended; older callers using an older
 * version receive only the prefix.  `version_supported` reports the
 * highest version the loaded library actually fills in.
 *
 * Cost: O(populated regions × BITMAP_WORDS_PER_REGION) (≤ ~4 K
 * cache-cold loads at the 128 TiB ceiling); a relaxed-load walk of
 * each region's embedded claim_bitmap.  Intended for diagnostic /
 * tuning use, NOT the hot path.
 */
#define KAME_POOL_STATS_VERSION 2u

typedef struct kame_pool_stats {
    /* IN: caller-set version.  OUT: highest version this library fills. */
    unsigned int version;
    unsigned int version_supported;

    /* --- version 1 fields --- */

    /* Number of 32 MiB regions the pool has mmap'd (never decreases —
     * regions are permanent in the current design).  Same as
     * `kame_pool_reserved_bytes() / (32 MiB)`. */
    size_t regions_populated;

    /* Total VA reserved (= regions_populated × 32 MiB).  Note that
     * actual RSS is far smaller: free chunks have been MADV_DONTNEED'd
     * and the radix L2 nodes are 8 KiB each. */
    size_t bytes_reserved;

    /* Currently-live ALLOCATIONS — chunks that have at least one
     * outstanding slot.  Each "live chunk" = one chunk_base unit (one
     * back_offset==0 entry whose claim bit is set).  Counted by walking
     * every region's embedded claim_bitmap + back_offset table. */
    size_t chunks_live;

    /* Currently-claimed UNITS (256 KiB each, EXCLUDING the per-region
     * metadata unit 0 reservation).  For multi-unit chunks
     * (CHUNK_UNITS = 2 or 4), one chunk contributes CHUNK_UNITS units.
     * Sum of set bits across all regions' claim_bitmap, with bit 0 of
     * word 0 of each region masked off.  Indicator of internal
     * fragmentation: `units_live / chunks_live` ≈ average chunk size
     * (in units).  `units_live × 256 KiB` ≈ pool's "claimed" footprint. */
    size_t units_live;

    /* --- version 2 fields (24/7 leak / RSS-attribution diagnostics) --- */

    /* Total bytes currently held in the global L2 large-recycle cache
     * (`g_lrc_bytes` snapshot).  Includes both LRC_CHUNK (256 KiB..4 MiB
     * dedicated chunks) and LRC_MMAP (4..64 MiB large_va, including
     * §27 huge multi-region) entries.  These bytes ARE resident but
     * available for warm reuse — subtract from "total RSS" to attribute
     * to actual program live data.  Drops on `kame_pool_set_large_cache_cap`
     * tighten and on §28.1 lazy ticks. */
    size_t cache_bytes;

    /* §15 dedicated chunks (256 KiB..4 MiB), sum of their DEDICATED_SIZE
     * values.  INCLUDES chunks currently parked in the LRC_CHUNK recycle
     * cache (their units stay claimed with bit-7 of back_offset set, so the
     * region walk that computes this counter cannot distinguish parked from
     * live in O(1); use `cache_bytes` for the parked subtotal across both
     * tiers).  Monotone growth in `(dedicated_chunk_bytes - cache_bytes)`
     * signals a leak.  Computed via region+back_offset walk inside
     * `kame_pool_get_stats` — O(regions × NUM_ALLOCATORS_IN_SPACE), same
     * walk that produces `chunks_live` / `units_live`; no hot-path running
     * counter (§28.5: that ran into cache-line contention at ≥ 128T). */
    size_t dedicated_chunk_bytes;

    /* §19/§27 large_va allocations (4..64 MiB single-region + > 32 MiB
     * multi-region huge) currently held by the program — likewise excludes
     * cache-parked entries.  `large_alloc_count` is the number of distinct
     * allocations; `large_alloc_bytes` is the sum of their mmap_size's.
     * Two plain global atomics; large allocs are rare (kHz/thread at most)
     * so a single cache line has no measurable contention.  O(1) snapshot. */
    size_t large_alloc_count;
    size_t large_alloc_bytes;
} kame_pool_stats_t;

void kame_pool_get_stats(kame_pool_stats_t *out) KAMEPOOLALLOC_NOEXCEPT;

#ifdef __cplusplus
}  /* extern "C" */

/* ===================================================================
 * (§75) C++ RAII guard for a realtime section
 * ===================================================================
 * Marks the calling thread realtime for the scope, and — in a debug
 * build, or whenever `check` is passed explicitly — verifies on exit
 * that the section entered the kernel for no new mapping.  The check is
 * a counter comparison, so it costs nothing inside the section itself:
 *
 *     kame_pool_set_realtime_mode(1);                 // once, startup
 *     const size_t  sz[] = { 64, 4096, 1u << 20 };
 *     const unsigned ct[] = { 4096, 256, 8 };
 *     kame_pool_prewarm(sz, ct, 3);                   // per RT thread
 *     for(;;) {
 *         {
 *             kame::rt_section rt;                    // critical section
 *             ...                                     // no mmap/madvise
 *         }                                           // violations checked
 *         kame_pool_rt_drain();                       // in the trough
 *     }
 *
 * The guard nests correctly (it restores the previous flag, it does not
 * blindly clear it), so a helper that opens its own section inside an
 * already-realtime thread behaves.
 */
namespace kame {

class rt_section {
public:
    /* `check`: report to stderr if the section caused a new mapping.
     * Defaults to on in debug builds only.  `violations()` is always
     * available regardless, for a test to assert on. */
    explicit rt_section(bool check =
#ifdef NDEBUG
                        false
#else
                        true
#endif
                       ) KAMEPOOLALLOC_NOEXCEPT
        : m_prev(kame_pool_get_realtime_thread() != 0),
          m_base(kame_pool_rt_violations()),
          m_check(check) {
        kame_pool_set_realtime_thread(1);
    }
    ~rt_section() KAMEPOOLALLOC_NOEXCEPT {
        unsigned long long v = violations();
        kame_pool_set_realtime_thread(m_prev ? 1 : 0);
        if(m_check && v != 0ull)
            kame_pool_rt_report_violations(v);
    }
    /* New mappings made since this guard was entered. */
    unsigned long long violations() const KAMEPOOLALLOC_NOEXCEPT {
        return kame_pool_rt_violations() - m_base;
    }

    rt_section(const rt_section &) = delete;
    rt_section &operator=(const rt_section &) = delete;

private:
    bool               m_prev;
    unsigned long long m_base;
    bool               m_check;
};

}  /* namespace kame */
#endif

#endif  /* KAMEPOOLALLOC_KAME_POOL_H_ */
