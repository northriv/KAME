/*
 * (§75) Realtime per-thread gating test.
 *
 * Verifies what the RT half actually promises, each by
 * OBSERVING a counter change rather than by trusting that a call was made:
 *
 *   1. prewarm + a realtime section makes NO new mapping (the headline
 *      claim: kame_pool_rt_violations() does not move).
 *   2. free() on a realtime thread DEFERS page reclaim (deferred_reclaims
 *      moves) and rt_drain() then settles it.
 *   3. KAME_RT_OS_FAIL is honoured where it is safe, and — the part worth
 *      testing because it is a deliberate exception — a realtime thread
 *      can still allocate: refused pool mappings degrade to libc, so the
 *      pointer is valid and freeable either way.
 *   4. rt_section nests (restores the previous flag, does not clear it)
 *      and reports its own violation delta.
 *   5. (G6a) the THP policy round-trips, its re-advise walk really reaches
 *      already-mapped regions, and KAME_THP_NEVER measurably holds
 *      AnonHugePages at zero — skipped, not faked, on a host that backs no
 *      transparent hugepages in the first place.
 *
 * Licensed under Apache-2.0 OR GPL-2.0-or-later, as the rest of the tree.
 */
#include "../kame_pool.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

static int g_failures = 0;

static void check(bool ok, const char *what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if( !ok) g_failures++;
}

// The size classes the "realtime loop" below touches: one bucket size, one
// page-ish size, one dedicated-chunk size.  Deliberately spans the tiers so
// prewarm has to reach chunks, regions and the radix.
static const size_t   kSizes[]  = { 64, 4096, 256u * 1024u };
static const unsigned kCounts[] = { 2048, 128, 4 };
static const unsigned kNClasses = 3;

// One iteration of a "control loop" body: churn each size class.  Returns a
// checksum so nothing is optimised away.
static unsigned long rt_loop_body() {
    unsigned long sum = 0;
    for(unsigned c = 0; c < kNClasses; c++) {
        void *p[8];
        unsigned n = kCounts[c] < 8 ? kCounts[c] : 8;
        for(unsigned i = 0; i < n; i++) {
            p[i] = kame_pool_malloc(kSizes[c]);
            if(p[i]) {
                std::memset(p[i], (int)(i & 0xff), kSizes[c]);
                sum += *static_cast<unsigned char *>(p[i]);
            }
        }
        for(unsigned i = 0; i < n; i++) kame_pool_free(p[i]);
    }
    return sum;
}

// ---------------------------------------------------------------- 1 + 2
static void test_prewarm_then_no_mappings() {
    std::printf("(1) prewarm -> realtime section makes no new mapping\n");

    int pw = kame_pool_prewarm(kSizes, kCounts, kNClasses);
    check(pw == 0, "kame_pool_prewarm succeeded");

    // Run the loop once OUTSIDE the section too: this pulls in anything
    // prewarm's exact size rounding missed, so the measured section below
    // isolates steady-state behaviour rather than first-touch effects.
    volatile unsigned long warm = rt_loop_body();
    (void)warm;

    kame_pool_rt_reset_counters();
    unsigned long long defer0 = kame_pool_rt_deferred_reclaims();
    {
        kame::rt_section rt(/*check=*/false);
        for(int iter = 0; iter < 200; iter++) {
            volatile unsigned long s = rt_loop_body();
            (void)s;
        }
        check(rt.violations() == 0ull,
              "no new mapping inside the realtime section");
    }
    check(kame_pool_rt_violations() == 0ull,
          "global violation counter still zero");

    (void)defer0;
}

// Churn `n` dedicated-tier blocks; optionally as a realtime thread.
static void dedicated_churn(bool realtime, int n) {
    if(realtime) kame_pool_set_realtime_thread(1);
    for(int i = 0; i < n; i++) {
        void *p = kame_pool_malloc(300u * 1024u);   // > 32 KiB ⇒ dedicated
        if(p) { std::memset(p, 7, 4096); kame_pool_free(p); }
    }
    if(realtime) kame_pool_set_realtime_thread(0);
}

// ------------------------------------------------------------------ 2
static void test_deferred_reclaim() {
    std::printf("(2) realtime free() defers reclaim; a non-RT thread does not\n");

    // Reaching the madvise gate takes cache PRESSURE, worth stating
    // explicitly: in steady state a freed chunk is parked warm in the
    // large-recycle cache (§34) and `deallocate_chunk` — the only site that
    // madvise's — is never reached at all, so there is nothing to defer.
    // The gate matters exactly when the cache cannot absorb the chunk
    // (over cap / evicted / thread exit).
    //
    // Squeezing the cap alone is NOT enough: a thread's L1 keeps the byte
    // cut it computed when it armed, so an already-armed thread ignores a
    // later cap change.  Run the churn on FRESH threads, which arm under
    // the zero cap and therefore refuse every push.
    const size_t saved_cap = kame_pool_get_large_cache_cap();
    kame_pool_set_large_cache_cap(0);

    unsigned long long d0 = kame_pool_rt_deferred_reclaims();
    std::thread([] { dedicated_churn(/*realtime=*/true, 32); }).join();
    unsigned long long d1 = kame_pool_rt_deferred_reclaims();
    check(d1 > d0, "realtime churn deferred page reclaim (madvise skipped)");

    std::thread([] { dedicated_churn(/*realtime=*/false, 32); }).join();
    check(kame_pool_rt_deferred_reclaims() == d1,
          "identical churn off the realtime thread reclaimed immediately");

    kame_pool_set_large_cache_cap(saved_cap);
}

// ------------------------------------------------------------------ 2b
static void test_deferred_unmap() {
    std::printf("(2b) realtime free() defers munmap; rt_drain settles it\n");

    // Blocks above LRC_HI (256 MiB) bypass the recycle cache entirely, so
    // their free always reaches the tier's munmap — a deterministic way to
    // exercise the deferral without depending on cache state.  VA only:
    // just two pages are touched.
    const size_t kHuge = 300u * 1024u * 1024u;

    unsigned long long u0 = kame_pool_rt_deferred_unmaps();
    {
        kame::rt_section rt(/*check=*/false);
        void *p = kame_pool_malloc(kHuge);
        if(p) {
            std::memset(p, 0x3c, 4096);
            std::memset(static_cast<char *>(p) + kHuge - 4096, 0xc3, 4096);
            kame_pool_free(p);
        }
        check(p != nullptr, "huge allocation succeeded on a realtime thread");
    }
    check(kame_pool_rt_deferred_unmaps() > u0,
          "munmap was deferred, not performed");
    check(kame_pool_rt_pending_bytes() >= kHuge,
          "the deferred VA is reported by rt_pending_bytes");

    kame_pool_rt_drain();
    check(kame_pool_rt_pending_bytes() == 0,
          "rt_drain unmapped everything it had parked");

}

// ------------------------------------------------------------------ 2c
static void test_pending_is_bounded() {
    std::printf("(2c) the deferral backlog is BOUNDED and self-settling\n");

    // Before the G5 cap this loop parked 12.6 GB of VA for 40 frees and only
    // an explicit rt_drain() ever gave it back — deferring without a ceiling
    // trades a bounded free() tail for unbounded memory.
    const size_t kHuge = 300u * 1024u * 1024u;   // > LRC_HI: cache bypassed
    const size_t kCap  = 600u * 1024u * 1024u;   // room for ~1 block

    const size_t saved = kame_pool_get_rt_pending_cap();
    kame_pool_set_rt_pending_cap(kCap);
    kame_pool_rt_reset_counters();

    for(int i = 0; i < 40; i++) {
        kame::rt_section rt(/*check=*/false);
        void *p = kame_pool_malloc(kHuge);
        if(p) { std::memset(p, 5, 4096); kame_pool_free(p); }
    }
    size_t pending = kame_pool_rt_pending_bytes();
    check(pending <= kCap,
          "parked VA never exceeds the cap (was unbounded before G5)");
    check(kame_pool_rt_forced_releases() > 0,
          "over-cap frees released inline, and said so");

    // A non-realtime large free settles one parked block per call, so a
    // mixed-thread program drains itself without an explicit rt_drain().
    size_t before = kame_pool_rt_pending_bytes();
    if(before) {
        for(int i = 0; i < 4; i++) {
            void *p = kame_pool_malloc(kHuge);
            if(p) { std::memset(p, 6, 4096); kame_pool_free(p); }
        }
        check(kame_pool_rt_pending_bytes() < before,
              "non-realtime frees settled part of the backlog on their own");
    }

    kame_pool_rt_drain();
    check(kame_pool_rt_pending_bytes() == 0, "rt_drain settles the remainder");
    kame_pool_set_rt_pending_cap(saved);
}

// ------------------------------------------------------------------ 2d
static void test_mlock_regions() {
    std::printf("(2d) region-scoped mlock pins (and populates) pool memory\n");

    // Deliberately tolerant of a low RLIMIT_MEMLOCK: CI containers often cap
    // it at a few MiB (or 64 KiB), and a partial lock is a reported outcome,
    // not a failure.  What we assert is CONSISTENCY, not success.
    kame_pool_reserve_regions(2, /*prefault=*/0);
    const size_t reserved = kame_pool_reserved_bytes();

    const size_t locked = kame_pool_mlock_regions();
    if(locked == 0) {
        std::printf("  [ok] skipped: RLIMIT_MEMLOCK / working-set quota "
                    "allows nothing\n");
        return;
    }
    check(locked <= reserved,
          "locked bytes never exceed the pool's mapped regions");
    check(locked % (32u * 1024u * 1024u) == 0,
          "locked in whole 32 MiB regions");

    // Allocation must still work while pinned — pinning is not a lock in the
    // mutual-exclusion sense, but it is worth proving it did not wedge
    // anything.
    void *p = kame_pool_malloc(4096);
    check(p != nullptr, "allocation still works while regions are pinned");
    if(p) { std::memset(p, 3, 4096); kame_pool_free(p); }

    const size_t unlocked = kame_pool_munlock_regions();
    check(unlocked == locked, "munlock releases exactly what mlock took");
}

// ------------------------------------------------------------------ 2e
#if defined(__linux__)
// Sum AnonHugePages over the smaps VMAs OVERLAPPING [lo, hi).  -1 if
// unreadable.  Overlap, not containment: the allocation we are asked about
// starts a header inside its mapping, so the VMA begins below `lo`.
static long anon_hugepages_in(uintptr_t lo, uintptr_t hi) {
    std::FILE *f = std::fopen("/proc/self/smaps", "r");
    if( !f) return -1;
    char line[512];
    long huge = 0;
    bool inside = false;
    while(std::fgets(line, sizeof line, f)) {
        // Scan into unsigned long long, not uintptr_t: on ILP32 `uintptr_t`
        // is `unsigned int` and `%lx` would be a type mismatch (harmless in
        // practice where long is also 32-bit, but a real one where it is not).
        unsigned long long a, b;
        if(std::sscanf(line, "%llx-%llx", &a, &b) == 2 && !std::strstr(line, "kB")) {
            inside = ((uintptr_t)a < hi && (uintptr_t)b > lo);
            continue;
        }
        long v;
        if(inside && std::sscanf(line, "AnonHugePages: %ld kB", &v) == 1) huge += v;
    }
    std::fclose(f);
    return huge;
}
#endif

static void test_thp_policy() {
    std::printf("(2e) THP policy round-trips, reaches existing regions, and "
                "actually suppresses hugepages\n");

    const int saved = kame_pool_get_thp_policy();

    // The API half — true on every platform.  An out-of-range value must be
    // ignored rather than stored, or a typo would silently disable the pool's
    // hugepage handling.
    kame_pool_set_thp_policy(KAME_THP_NEVER);
    check(kame_pool_get_thp_policy() == KAME_THP_NEVER, "policy round-trips");
    kame_pool_set_thp_policy(99);
    check(kame_pool_get_thp_policy() == KAME_THP_NEVER,
          "out-of-range policy is ignored, not stored");

    // The re-advise walk is the reason this is a policy and not a flag read
    // at region-claim time: regions are created LAZILY, so a policy set after
    // the first allocation has to reach what is already mapped.  Assert it
    // covered whole regions minus their metadata page — i.e. that it really
    // walked the region lists rather than returning a plausible constant.
    kame_pool_reserve_regions(2, /*prefault=*/0);
    const size_t re = kame_pool_set_thp_policy(KAME_THP_NEVER);
#if defined(__linux__)
    const size_t kRegion = 32u * 1024u * 1024u, kPage = 4096u;
    check(re >= kRegion - kPage, "re-advise reached existing regions");
    check(re % (kRegion - kPage) == 0, "re-advised in whole regions");
    check(kame_pool_set_thp_policy(KAME_THP_SYSTEM) == 0,
          "KAME_THP_SYSTEM re-advises nothing (Linux has no 'clear' advice)");
#else
    check(re == 0, "non-Linux: THP policy is a no-op returning 0");
#endif

#if defined(__linux__)
    // The behavioural half.  A block above LRC_HI is a fresh mmap every time
    // (the recycle cache is bypassed there), so each arm below gets memory
    // that has never been faulted — which is what makes an in-process A/B
    // valid at all: MADV_NOHUGEPAGE does not split hugepages that already
    // exist, so the two arms MUST NOT share pages.
    const size_t kHuge = 300u * 1024u * 1024u, kTouch = 8u * 1024u * 1024u;

    auto touch_and_measure = [&](int policy) -> long {
        kame_pool_set_thp_policy(policy);
        char *p = static_cast<char *>(kame_pool_malloc(kHuge));
        if( !p) return -1;
        for(size_t off = 0; off < kTouch; off += 4096u) p[off] = 0x5a;
        long huge = anon_hugepages_in((uintptr_t)p, (uintptr_t)p + kTouch);
        kame_pool_free(p);
        kame_pool_rt_drain();
        return huge;
    };

    // Baseline first: if this host cannot produce transparent hugepages at
    // all (THP=never, a THP-less kernel, or PR_SET_THP_DISABLE set on the
    // process tree — containers do this), then "0 kB under NEVER" would pass
    // for the wrong reason.  Skip rather than claim a result.
    //
    // The baseline arm is KAME_THP_ALWAYS, not KAME_THP_SYSTEM.  An unadvised
    // range is only opportunistically backed by hugepages: under the common
    // `defrag = madvise` setting the kernel will NOT compact to find a 2 MiB
    // block for it, so a SYSTEM baseline reads 0 whenever memory is even
    // mildly fragmented and the whole check skips itself for no good reason
    // (observed: passes 64-bit, skips 32-bit, same host, same kernel).
    // MADV_HUGEPAGE does earn the compaction effort, so ALWAYS-vs-NEVER is
    // both deterministic and the sharper test — it exercises each direction
    // of the policy rather than one direction against the kernel's whim.
    const long base = touch_and_measure(KAME_THP_ALWAYS);
    if(base <= 0) {
        std::printf("  [ok] skipped: this host backs no transparent hugepages "
                    "(baseline AnonHugePages=%ld kB) — nothing to suppress\n",
                    base);
    }
    else {
        const long never = touch_and_measure(KAME_THP_NEVER);
        check(never == 0,
              "KAME_THP_NEVER holds AnonHugePages at 0 on fresh pool memory");
        std::printf("      (KAME_THP_ALWAYS %ld kB over %zu MiB touched, "
                    "KAME_THP_NEVER %ld kB)\n",
                    base, kTouch >> 20, never);
    }
#endif

    kame_pool_set_thp_policy(saved);
}

// -------------------------------------------------------------------- 3
static void test_os_fail_policy() {
    std::printf("(3) KAME_RT_OS_FAIL still yields usable memory\n");

    kame_pool_set_rt_os_policy(KAME_RT_OS_FAIL);
    check(kame_pool_get_rt_os_policy() == KAME_RT_OS_FAIL,
          "policy round-trips");

    // Ask for sizes far outside anything prewarmed, on a realtime thread,
    // under FAIL.  Whether the pool refuses (degrading to libc) or serves
    // it from cache, the contract is the same: a valid, freeable pointer.
    bool all_usable = true;
    {
        kame::rt_section rt(/*check=*/false);
        for(int i = 0; i < 16; i++) {
            size_t sz = (size_t)(1u << 20) * (size_t)(3 + i);   // 3..18 MiB
            void *p = kame_pool_malloc(sz);
            if( !p) { all_usable = false; break; }
            std::memset(p, 0xa5, 4096);
            std::memset(static_cast<char *>(p) + sz - 4096, 0x5a, 4096);
            kame_pool_free(p);
        }
    }
    check(all_usable,
          "allocation under FAIL degrades gracefully (never returns null)");

    kame_pool_set_rt_os_policy(KAME_RT_OS_ALLOW);
    kame_pool_rt_drain();
}

// -------------------------------------------------------------------- 4
static void test_rt_section_nesting() {
    std::printf("(4) rt_section nests and restores\n");

    check(kame_pool_get_realtime_thread() == 0, "thread starts non-realtime");
    {
        kame::rt_section outer(/*check=*/false);
        check(kame_pool_get_realtime_thread() == 1, "outer marks realtime");
        {
            kame::rt_section inner(/*check=*/false);
            check(kame_pool_get_realtime_thread() == 1, "inner keeps it set");
        }
        check(kame_pool_get_realtime_thread() == 1,
              "inner's exit RESTORES realtime (does not clear it)");
    }
    check(kame_pool_get_realtime_thread() == 0, "outer's exit clears it");
}

// Per-thread-ness: a realtime worker must not make the OTHER thread defer.
static void test_flag_is_per_thread() {
    std::printf("(5) the realtime flag is per-thread\n");

    int seen_in_worker = -1;
    {
        kame::rt_section rt(/*check=*/false);
        std::thread t([&] { seen_in_worker = kame_pool_get_realtime_thread(); });
        t.join();
    }
    check(seen_in_worker == 0,
          "a fresh thread is NOT realtime while the parent is");
}

int main() {
    std::printf("== alloc_rt_thread_test (§75) ==\n");
    kame_pool_set_realtime_mode(1);      // process-wide: silence background

    test_prewarm_then_no_mappings();
    test_deferred_reclaim();
    test_deferred_unmap();
    test_pending_is_bounded();
    test_mlock_regions();
    test_thp_policy();
    test_os_fail_policy();
    test_rt_section_nesting();
    test_flag_is_per_thread();

    kame_pool_set_realtime_mode(0);
    std::printf("== %s (%d failure%s) ==\n",
                g_failures ? "FAILED" : "PASSED",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
