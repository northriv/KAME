// alloc_tune_report.cpp
//
// Self-contained tuning-diagnostic for kamepoolalloc on the host machine.
//
// What it does:
//   1. Detects the environment (CPU count, NUMA nodes, page size, THP).
//   2. Microbenchmarks raw `mmap` / `munmap` / `madvise(MADV_DONTNEED)`
//      single-threaded, with and without first-touch.
//   3. Sweeps multi-thread `mmap` + `munmap` over { 1, 4, 16, 64, hw } to
//      show how TLB shootdown cost scales — the dominant factor on
//      many-core NUMA targets (e.g. Ohtaka-class 256-CPU 1-node).
//   4. Sweeps kamepoolalloc throughput at every tier (bucket / dedicated
//      chunk / large_va / huge) single- and multi-threaded.
//   5. Emits a recommendation block: which `-DLRC_*` build knobs (or
//      runtime `kame_pool_set_*` calls) match the measured environment.
//
// Pool-only (uses the kame_pool_* C API + raw mmap).  Build-time linked
// against libkamepoolalloc; the report compares "raw mmap" cost against
// "kamepoolalloc cached" cost on the same machine to make the warm-reuse
// win quantifiable.
//
// Runtime: ~20-40 s on a typical multi-core server; mostly the MT mmap
// sweep on high core counts.  Output is plain text — pipe to a file and
// attach to bug reports / tuning discussions.
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "kame_pool.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(__linux__) || defined(__APPLE__)
#  include <sys/mman.h>
#  include <unistd.h>
#endif
#if defined(__linux__)
#  include <dirent.h>
#  include <sys/syscall.h>
#endif

using clk = std::chrono::steady_clock;
static double ns_since(clk::time_point t0) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        clk::now() - t0).count();
}

// ----- 1. environment detection -----------------------------------------

struct Env {
    int hw_concurrency;
    int numa_nodes;
    size_t page_size;
    const char *thp_state;     // "always" / "madvise" / "never" / "n/a"
    const char *platform;
};

#if defined(__linux__)
static int detect_numa_nodes() {
    int max_n = 0;
    DIR *d = opendir("/sys/devices/system/node");
    if(!d) return 1;
    while(struct dirent *e = readdir(d)) {
        int n;
        if(sscanf(e->d_name, "node%d", &n) == 1 && n + 1 > max_n) max_n = n + 1;
    }
    closedir(d);
    return max_n > 0 ? max_n : 1;
}
static const char *detect_thp() {
    FILE *f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
    if(!f) return "n/a";
    static char buf[128];
    if(!fgets(buf, sizeof(buf), f)) { fclose(f); return "n/a"; }
    fclose(f);
    if(strstr(buf, "[always]"))  return "always";
    if(strstr(buf, "[madvise]")) return "madvise";
    if(strstr(buf, "[never]"))   return "never";
    return "?";
}
#else
static int detect_numa_nodes() { return 1; }
static const char *detect_thp() { return "n/a"; }
#endif

static Env probe_env() {
    Env e;
    e.hw_concurrency = std::max(1u, std::thread::hardware_concurrency());
    e.numa_nodes = detect_numa_nodes();
#if defined(__linux__) || defined(__APPLE__)
    e.page_size = (size_t)sysconf(_SC_PAGESIZE);
#else
    e.page_size = 4096;
#endif
    e.thp_state = detect_thp();
#if defined(__linux__)
    e.platform = "Linux";
#elif defined(__APPLE__)
    e.platform = "macOS";
#elif defined(_WIN32)
    e.platform = "Windows";
#else
    e.platform = "?";
#endif
    return e;
}

// ----- 2. raw mmap / munmap / madvise microbench -----------------------

#if defined(__linux__) || defined(__APPLE__)
static const size_t kProbeSize = (size_t)32 * 1024 * 1024;   // 32 MiB

static double bench_mmap_no_touch_ns() {
    constexpr int N = 32;
    double total = 0;
    for(int i = 0; i < N; i++) {
        auto t0 = clk::now();
        void *p = mmap(nullptr, kProbeSize, PROT_READ | PROT_WRITE,
                       MAP_ANON | MAP_PRIVATE, -1, 0);
        total += ns_since(t0);
        munmap(p, kProbeSize);
    }
    return total / N;
}
static double bench_munmap_ns() {
    constexpr int N = 32;
    double total = 0;
    for(int i = 0; i < N; i++) {
        void *p = mmap(nullptr, kProbeSize, PROT_READ | PROT_WRITE,
                       MAP_ANON | MAP_PRIVATE, -1, 0);
        // First-touch a few pages so the kernel has page tables to tear down.
        // (Untouched pages are nearly free to munmap.)
        for(size_t off = 0; off < kProbeSize; off += 4096)
            ((volatile char *)p)[off] = 1;
        auto t0 = clk::now();
        munmap(p, kProbeSize);
        total += ns_since(t0);
    }
    return total / N;
}
static double bench_mmap_first_touch_ns_per_page() {
    constexpr int N = 8;
    double total = 0;
    size_t pages = kProbeSize / 4096;
    for(int i = 0; i < N; i++) {
        void *p = mmap(nullptr, kProbeSize, PROT_READ | PROT_WRITE,
                       MAP_ANON | MAP_PRIVATE, -1, 0);
        auto t0 = clk::now();
        for(size_t off = 0; off < kProbeSize; off += 4096)
            ((volatile char *)p)[off] = 1;
        total += ns_since(t0);
        munmap(p, kProbeSize);
    }
    return total / (N * pages);
}
static double bench_madvise_dontneed_ns() {
    constexpr int N = 16;
    double total = 0;
    for(int i = 0; i < N; i++) {
        void *p = mmap(nullptr, kProbeSize, PROT_READ | PROT_WRITE,
                       MAP_ANON | MAP_PRIVATE, -1, 0);
        for(size_t off = 0; off < kProbeSize; off += 4096)
            ((volatile char *)p)[off] = 1;
        auto t0 = clk::now();
#if defined(__APPLE__)
        madvise(p, kProbeSize, MADV_FREE);
#else
        madvise(p, kProbeSize, MADV_DONTNEED);
#endif
        total += ns_since(t0);
        munmap(p, kProbeSize);
    }
    return total / N;
}
static double bench_madvise_hugepage_ns() {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    constexpr int N = 8;
    double total = 0;
    size_t pages = kProbeSize / 4096;
    for(int i = 0; i < N; i++) {
        void *p = mmap(nullptr, kProbeSize, PROT_READ | PROT_WRITE,
                       MAP_ANON | MAP_PRIVATE, -1, 0);
        madvise(p, kProbeSize, MADV_HUGEPAGE);
        auto t0 = clk::now();
        for(size_t off = 0; off < kProbeSize; off += 4096)
            ((volatile char *)p)[off] = 1;
        total += ns_since(t0);
        munmap(p, kProbeSize);
    }
    return total / (N * pages);
#else
    return -1;
#endif
}
#endif

// ----- 3. MT munmap contention (TLB shootdown) --------------------------

#if defined(__linux__) || defined(__APPLE__)
static double bench_mt_munmap_ns(int nthreads, int iters_per_thread = 4) {
    std::atomic<bool> go{false};
    std::atomic<double> total_ns{0};
    std::vector<std::thread> ts;
    for(int t = 0; t < nthreads; t++) {
        ts.emplace_back([&, t] {
            while(!go.load(std::memory_order_acquire)) ;
            double sum = 0;
            for(int i = 0; i < iters_per_thread; i++) {
                void *p = mmap(nullptr, kProbeSize, PROT_READ | PROT_WRITE,
                               MAP_ANON | MAP_PRIVATE, -1, 0);
                for(size_t off = 0; off < kProbeSize; off += 4096)
                    ((volatile char *)p)[off] = 1;
                auto t0 = clk::now();
                munmap(p, kProbeSize);
                sum += ns_since(t0);
            }
            double exp = total_ns.load(std::memory_order_relaxed);
            while(!total_ns.compare_exchange_weak(exp, exp + sum)) ;
        });
    }
    go.store(true, std::memory_order_release);
    for(auto &t : ts) t.join();
    return total_ns.load() / (nthreads * iters_per_thread);
}
#endif

// ----- 4. kamepoolalloc throughput --------------------------------------

struct ThroughputResult {
    const char *label;
    size_t size;
    int nthreads;
    double ns_per_op;   // alloc + free
    double mops_per_sec;
};

static ThroughputResult bench_alloc_free(const char *label, size_t size,
                                         int nthreads, int seconds) {
    std::atomic<bool> go{false};
    std::atomic<bool> stop{false};
    std::atomic<int64_t> total_ops{0};
    std::vector<std::thread> ts;
    for(int t = 0; t < nthreads; t++) {
        ts.emplace_back([&] {
            while(!go.load(std::memory_order_acquire)) ;
            int64_t my_ops = 0;
            while(!stop.load(std::memory_order_relaxed)) {
                void *p = kame_pool_malloc(size);
                if(p) {
                    ((volatile char *)p)[0] = 1;       // first-touch
                    kame_pool_free(p);
                    my_ops++;
                }
            }
            total_ops.fetch_add(my_ops, std::memory_order_relaxed);
        });
    }
    auto t0 = clk::now();
    go.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    stop.store(true, std::memory_order_relaxed);
    for(auto &t : ts) t.join();
    double elapsed = ns_since(t0);
    int64_t ops = total_ops.load();
    ThroughputResult r;
    r.label = label;
    r.size = size;
    r.nthreads = nthreads;
    r.ns_per_op = ops ? elapsed / ops : 0;
    r.mops_per_sec = ops ? (double)ops * 1e9 / elapsed / 1e6 : 0;
    return r;
}

// ----- main -------------------------------------------------------------

int main(int argc, char **argv) {
    int sec_per_bench = (argc > 1) ? atoi(argv[1]) : 2;
    if(sec_per_bench < 1) sec_per_bench = 1;

    Env e = probe_env();
    std::printf("=== kamepoolalloc tuning report ===\n");
    std::printf("Platform : %s\n", e.platform);
    std::printf("CPUs     : %d\n", e.hw_concurrency);
    std::printf("NUMA     : %d node%s\n", e.numa_nodes, e.numa_nodes == 1 ? "" : "s");
    std::printf("Page sz  : %zu B\n", e.page_size);
    std::printf("THP      : %s\n", e.thp_state);
    std::printf("Build    : LRC_K_MAX=256  LRC_N_MAX=32  LRC_HI=64 MiB\n");
    std::printf("           (defaults — override with -DLRC_K_MAX=... etc.)\n");
    std::printf("\n");

#if defined(__linux__) || defined(__APPLE__)
    // Microbench
    std::printf("=== Raw VMM cost (32 MiB regions) ===\n");
    double mmap_ns       = bench_mmap_no_touch_ns();
    double munmap_ns     = bench_munmap_ns();
    double touch_per_pg  = bench_mmap_first_touch_ns_per_page();
    double madvise_ns    = bench_madvise_dontneed_ns();
    double thp_per_pg    = bench_madvise_hugepage_ns();
    std::printf("  mmap (no touch)            %8.1f us\n", mmap_ns / 1000.0);
    std::printf("  first-touch fault          %8.2f us/page  (%.2f ms / 32 MiB)\n",
                touch_per_pg / 1000.0,
                touch_per_pg * (kProbeSize / e.page_size) / 1e6);
    std::printf("  munmap (after touch)       %8.1f us\n", munmap_ns / 1000.0);
#if defined(__APPLE__)
    std::printf("  madvise(MADV_FREE)         %8.1f us\n", madvise_ns / 1000.0);
#else
    std::printf("  madvise(MADV_DONTNEED)     %8.1f us\n", madvise_ns / 1000.0);
#endif
    if(thp_per_pg > 0)
        std::printf("  mmap + MADV_HUGEPAGE touch %8.2f us/page  (%.2f ms / 32 MiB)\n",
                    thp_per_pg / 1000.0,
                    thp_per_pg * (kProbeSize / e.page_size) / 1e6);
    else
        std::printf("  mmap + MADV_HUGEPAGE       n/a\n");
    std::printf("\n");

    // MT munmap contention
    std::printf("=== MT munmap contention (32 MiB, TLB shootdown effect) ===\n");
    double t1 = bench_mt_munmap_ns(1);
    std::printf("  %3d threads  %7.1f us  (1.00x baseline)\n", 1, t1 / 1000.0);
    int seen[8] = {1, 0, 0, 0, 0, 0, 0, 0}; int n_seen = 1;
    for(int n : {4, 16, 64, e.hw_concurrency}) {
        if(n <= 1 || n > e.hw_concurrency) continue;
        bool dup = false;
        for(int i = 0; i < n_seen; i++) if(seen[i] == n) { dup = true; break; }
        if(dup) continue;
        seen[n_seen++] = n;
        double tn = bench_mt_munmap_ns(n);
        std::printf("  %3d threads  %7.1f us  (%.2fx)\n", n, tn / 1000.0, tn / t1);
    }
    std::printf("\n");
#endif

    // kamepoolalloc throughput
    std::printf("=== kamepoolalloc throughput (alloc+free, first-touch byte 0) ===\n");
    std::printf("%-22s %5s %12s %14s\n", "size,tier", "T", "ns/op", "M ops/s");
    auto run = [&](const char *label, size_t size, int n) {
        auto r = bench_alloc_free(label, size, n, sec_per_bench);
        std::printf("%-22s %5d %12.0f %14.2f\n", r.label, r.nthreads,
                    r.ns_per_op, r.mops_per_sec);
    };
    // Label for the 40 MiB tier depends on whether §28's raised LRC_HI
    // catches it.  With LRC_HI=64 MiB (the default since §28), 40 MiB is
    // cacheable through allocate_large_va; §27 huge bypass only kicks in
    // above LRC_HI.  Adjust the label to match.
    const char *label_40m = "40 MiB (large_va, cached)";
    int hw = e.hw_concurrency;
    int seen_t[8] = {0}; int n_seen_t = 0;
    for(int n : {1, 4, std::min(hw, 16), std::min(hw, 64), hw}) {
        if(n <= 0 || n > hw) continue;
        bool dup = false;
        for(int i = 0; i < n_seen_t; i++) if(seen_t[i] == n) { dup = true; break; }
        if(dup) continue;
        seen_t[n_seen_t++] = n;
        run("64 B (bucket)",       64,            n);
        run("1 KiB (bucket)",      1024,          n);
        run("64 KiB (chunk)",      64u * 1024,    n);
        run("1 MiB (chunk)",       1u << 20,      n);
        run("8 MiB (large_va)",    8u << 20,      n);
        run(label_40m,             40u << 20,     n);
        std::printf("\n");
    }

    // ----- 5. Recommendations -------------------------------------------
    std::printf("=== Recommendations ===\n");
#if defined(__linux__) || defined(__APPLE__)
    // §28.1 fires up to ONE munmap per 10 ms per thread (only on the
    // mmap-tier push path — rare relative to alloc rate).  Worst-case
    // per-thread fraction of wallclock spent inside that munmap is
    // (1/interval) × munmap_ns × 1e-9 — IF the workload is push-saturated.
    // TLB-shootdown side-effects on OTHER cores are folded into
    // `munmap_ns` already (it was measured under the current MT load,
    // see "MT munmap contention" above).  Target: keep per-thread
    // worst case ≤ 5 % of wallclock.
    //
    // (§28.3) Auto-tune: trigger one LRC_MMAP push so the library's
    // first-use calibration runs, then read back the library's picked
    // value.  Compare against this report's own measurement.
    void *probe = kame_pool_malloc(8u << 20);
    if(probe) kame_pool_free(probe);
    unsigned int auto_tuned_ms = kame_pool_get_lazy_drain_interval_ms();

    double per_thread_at_default = (100.0 * munmap_ns) / 1e9;            // at 10 ms
    double per_thread_at_tuned   = auto_tuned_ms > 0
        ? (1000.0 / auto_tuned_ms) * munmap_ns / 1e9 : 0;
    std::printf("\n* §28.1 lazy drain:\n");
    std::printf("    library auto-tuned interval   : %u ms\n", auto_tuned_ms);
    std::printf("    worst-case @ default 10 ms    : %.2f %% per-thread wallclock\n",
                per_thread_at_default * 100);
    std::printf("    worst-case @ auto-tuned %u ms : %.2f %% per-thread wallclock\n",
                auto_tuned_ms, per_thread_at_tuned * 100);
    if(per_thread_at_default <= 0.05) {
        std::printf("  → default 10 ms is fine on this host; auto-tune kept it (raise-only).\n");
    } else if(per_thread_at_tuned <= 0.05) {
        std::printf("  → auto-tune raised it; pressure now under 5 %% — no action needed.\n");
    } else {
        std::printf("  → auto-tune hit the 1 s ceiling and still over budget;\n");
        std::printf("    consider kame_pool_set_lazy_drain_interval_ms(N) manually.\n");
    }
    if(thp_per_pg > 0 && touch_per_pg / thp_per_pg > 1.5) {
        std::printf("\n* MADV_HUGEPAGE speeds up first-touch by %.1fx on this kernel\n",
                    touch_per_pg / thp_per_pg);
        std::printf("  → consider adding MADV_HUGEPAGE in allocate_chunk / large_va_raw_map\n");
        std::printf("    (not currently requested; THP state = '%s')\n", e.thp_state);
    }
    if(!strcmp(e.thp_state, "never")) {
        std::printf("\n* THP is 'never' on this system — even explicit MADV_HUGEPAGE is a no-op.\n");
        std::printf("  → ask sysadmin about enabling 'madvise' mode if first-touch is hot.\n");
    }
#endif
    // K_MAX
    std::printf("\n* LRC_K_MAX guideline (current default 256):\n");
    if(hw <= 16)
        std::printf("  → %d cores: K_MAX=32 is plenty (rebuild -DLRC_K_MAX=32 for 10 KiB array)\n", hw);
    else if(hw <= 64)
        std::printf("  → %d cores: K_MAX=64-128 is plenty\n", hw);
    else if(hw <= 256)
        std::printf("  → %d cores: K_MAX=256 (default) is appropriate\n", hw);
    else
        std::printf("  → %d cores: consider K_MAX=512 (-DLRC_K_MAX=512)\n", hw);

    if(e.numa_nodes > 1) {
        std::printf("\n* NUMA: %d nodes detected.  The slot array lives at one NUMA domain;\n",
                    e.numa_nodes);
        std::printf("  cross-domain accesses pay remote-line latency.  Per-NUMA arenas are\n");
        std::printf("  not implemented — workloads with strong NUMA locality may benefit.\n");
    }
    std::printf("\n=== end of report ===\n");
    return 0;
}
