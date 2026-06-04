// Thread-churn / orphan-reuse repro for the "driver start/stop → reserved
// keeps growing" symptom.
//
// A worker thread is spawned and JOINED each cycle, mimicking a per-run
// instrument-driver thread (KAME's TestDriver start/stop).  It allocates a
// pool working set; what survives the thread's exit models objects the
// persistent STM tree / graphs still reference.
//
// Two scenarios:
//
//   (1) PURE CHURN — the worker frees everything before it exits.  Across
//       repeated spawn/exit, regions MUST be recycled: `bytes_reserved`
//       (= regions × 32 MiB, monotone non-decreasing by design) MUST PLATEAU
//       after a short warm-up.  This proves chunk recycling works across
//       thread teardown at all.
//
//   (2) SURVIVOR CHURN — each cycle leaves K SCATTERED survivors (1 per
//       KEEP_STRIDE allocations, so they touch many distinct chunks) held by
//       the main thread.  The worker exits with those slots still live, so
//       their chunks are ORPHANED (m_owner_id == 0).  The main thread then
//       frees the PREVIOUS cycle's survivors, so net-live data is BOUNDED.
//       `bytes_reserved` MUST STILL PLATEAU — if it grows monotonically, the
//       orphaned partially-used chunks are being STRANDED (not reused by the
//       next thread), which is the suspected start/stop region-growth cause.
//
// Pool-only (uses the kame_pool_* C API directly); built when
// USE_KAME_ALLOCATOR is ON.  Prints the per-cycle `bytes_reserved` /
// `chunks_live` trajectory so the plateau-vs-ramp shape is visible even when
// the hard assertions pass.
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "kame_pool.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static int g_fails = 0;
#define CHECK(c, ...) do { if(!(c)) { ++g_fails; \
    std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while(0)

static const size_t MiB = (size_t)1u << 20;

// Tunables.  Overridable from argv so the thread spawn/exit count can be
// cranked to distinguish a TRUE per-thread leak (keeps ramping linearly) from
// a bounded one-time overhead that only LOOKS like a ramp over a short window
// (would flatten at higher counts).  Usage:
//     alloc_thread_churn_test [cycles] [M] [keep_stride]
// e.g. `alloc_thread_churn_test 1000` = 1000 thread spawn/exits per scenario.
static int    CYCLES      = 120;      // spawn/join iterations (= thread births)
static int    WARMUP      = 10;       // cycles to reach steady-state regions
static int    M           = 250000;   // allocations per cycle (worker working set)
static int    KEEP_STRIDE = 64;       // scenario 2: keep 1 survivor per this many

// Small pool-tier sizes spanning several buckets (FS=true + FS=false).
static const size_t kSizes[] = {32, 48, 64, 96, 128, 200};
static inline size_t pick_size(int i) { return kSizes[(unsigned)i % 6u]; }

static size_t reserved_mib() { return kame_pool_reserved_bytes() / MiB; }
static size_t chunks_live() {
    kame_pool_stats_t s; std::memset(&s, 0, sizeof s); s.version = 2;
    kame_pool_get_stats(&s);
    return s.chunks_live;
}

// Platform-robust plateau check.  Compares the SECOND-HALF growth (mid ->
// end), NOT an absolute level.  A healthy allocator is flat once the bounded
// working set reaches steady state; the stranding bug ramps LINEARLY, so its
// tail growth is enormous (hundreds of MiB / thousands of chunks) on EVERY
// platform, while a healthy tail is ~0.  This sidesteps macOS-specific
// absolute-level differences (16 KiB pages, lazy MADV_FREE, thread scheduling)
// that would make an absolute-MiB threshold flaky on Mac but pass on Linux.
// Both metrics are computed identically on all platforms and are immune to
// MADV_FREE laziness (they count VA regions / claimed units, not RSS).
static void check_plateau(const char *name,
                          size_t mid_r, size_t end_r,    // reserved MiB
                          size_t mid_c, size_t end_c) {  // chunks_live
    // reserved (32-MiB regions, monotone non-decreasing): allow 1 region jitter.
    bool r_ramp = end_r > mid_r + 32u;
    // chunks_live: allow 20% + a small absolute margin for steady-state churn.
    bool c_ramp = end_c > mid_c + (mid_c / 5u) + 64u;
    CHECK(!r_ramp && !c_ramp,
          "%s: NOT a plateau — back-half growth reserved %zu->%zu MiB, "
          "chunks_live %zu->%zu (stranding/leak: the tail is still ramping)",
          name, mid_r, end_r, mid_c, end_c);
    std::printf("  [%s] %s tail (mid->end): reserved %zu->%zu MiB, "
                "chunks_live %zu->%zu\n",
                (r_ramp || c_ramp) ? "BAD" : "ok", name,
                mid_r, end_r, mid_c, end_c);
}

// ---- Scenario 1: pure churn (worker frees everything) ----
static void scenario_pure_churn() {
    std::printf("[scenario 1] pure churn (worker frees all before exit)\n");
    size_t mid_r = 0, mid_c = 0;
    for(int cyc = 0; cyc < CYCLES; cyc++) {
        std::thread([&] {
            std::vector<void *> v;
            v.reserve(M);
            for(int i = 0; i < M; i++) {
                void *p = kame_pool_malloc(pick_size(i));
                if(p) { *(volatile char *)p = (char)i; v.push_back(p); }
            }
            for(void *p : v) kame_pool_free(p);
        }).join();

        if(cyc == CYCLES / 2) { mid_r = reserved_mib(); mid_c = chunks_live(); }
        if(cyc >= WARMUP && (cyc % (CYCLES / 10 > 0 ? CYCLES / 10 : 1) == 0 || cyc == CYCLES - 1))
            std::printf("  cyc %3d: reserved=%4zu MiB  chunks_live=%zu\n",
                        cyc, reserved_mib(), chunks_live());
    }
    check_plateau("pure churn", mid_r, reserved_mib(), mid_c, chunks_live());
}

// ---- Scenario 2: survivor churn (KAME-like) ----
static void scenario_survivor_churn() {
    std::printf("[scenario 2] survivor churn (K scattered survivors held by main)\n");
    std::vector<void *> survivors_prev;   // held across the join boundary by main
    size_t mid_r = 0, mid_c = 0;
    for(int cyc = 0; cyc < CYCLES; cyc++) {
        std::vector<void *> survivors_cur;  // filled by worker, read by main after join
        std::thread([&] {
            std::vector<void *> v;
            v.reserve(M);
            for(int i = 0; i < M; i++) {
                void *p = kame_pool_malloc(pick_size(i));
                if(p) { *(volatile char *)p = (char)i; v.push_back(p); }
            }
            for(size_t i = 0; i < v.size(); i++) {
                if((i % (size_t)KEEP_STRIDE) == 0) survivors_cur.push_back(v[i]); // strand
                else                               kame_pool_free(v[i]);
            }
            // worker exits: survivor slots still live -> their chunks orphan.
        }).join();

        // main frees the PREVIOUS cycle's survivors (cross-thread free into the
        // now-orphaned chunks -> should reuse/release & make them reusable).
        for(void *p : survivors_prev) kame_pool_free(p);
        survivors_prev.swap(survivors_cur);

        if(cyc == CYCLES / 2) { mid_r = reserved_mib(); mid_c = chunks_live(); }
        if(cyc >= WARMUP && (cyc % (CYCLES / 10 > 0 ? CYCLES / 10 : 1) == 0 || cyc == CYCLES - 1))
            std::printf("  cyc %3d: reserved=%4zu MiB  chunks_live=%zu  survivors=%zu\n",
                        cyc, reserved_mib(), chunks_live(), survivors_prev.size());
    }
    // Measure the steady-state tail BEFORE draining the last cycle's survivors
    // (net-live is bounded throughout, so this is the true plateau sample).
    check_plateau("survivor churn", mid_r, reserved_mib(), mid_c, chunks_live());
    for(void *p : survivors_prev) kame_pool_free(p);   // drain the last cycle
}

int main(int argc, char **argv) {
    if(argc > 1) CYCLES      = std::atoi(argv[1]);
    if(argc > 2) M           = std::atoi(argv[2]);
    if(argc > 3) KEEP_STRIDE = std::atoi(argv[3]);
    if(CYCLES < 4) CYCLES = 4;
    if(M < 1) M = 1;
    if(KEEP_STRIDE < 1) KEEP_STRIDE = 1;
    if(WARMUP >= CYCLES / 2) WARMUP = CYCLES / 4;
    std::printf("config: CYCLES=%d (thread spawn/exits) M=%d KEEP_STRIDE=%d\n",
                CYCLES, M, KEEP_STRIDE);
    scenario_pure_churn();
    scenario_survivor_churn();
    std::printf(g_fails == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fails);
    return g_fails ? 1 : 0;
}
