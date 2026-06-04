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
#include <cstring>
#include <thread>
#include <vector>

static int g_fails = 0;
#define CHECK(c, ...) do { if(!(c)) { ++g_fails; \
    std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while(0)

static const size_t MiB = (size_t)1u << 20;

// Tunables (kept modest so the whole test runs in a few seconds).
static const int    CYCLES      = 80;       // spawn/join iterations
static const int    WARMUP      = 10;       // cycles to reach steady-state regions
static const int    M           = 250000;   // allocations per cycle (worker working set)
static const int    KEEP_STRIDE = 64;       // scenario 2: keep 1 survivor per this many

// Small pool-tier sizes spanning several buckets (FS=true + FS=false).
static const size_t kSizes[] = {32, 48, 64, 96, 128, 200};
static inline size_t pick_size(int i) { return kSizes[(unsigned)i % 6u]; }

static size_t reserved_mib() { return kame_pool_reserved_bytes() / MiB; }
static size_t chunks_live() {
    kame_pool_stats_t s; std::memset(&s, 0, sizeof s); s.version = 2;
    kame_pool_get_stats(&s);
    return s.chunks_live;
}

// True if `cur` exceeds `base` by more than `slack_regions` 32-MiB regions —
// i.e. the high-water mark is still climbing well past the warm-up plateau.
static bool grew(size_t base_mib, size_t cur_mib, int slack_regions) {
    return cur_mib > base_mib + (size_t)slack_regions * 32u;
}

// ---- Scenario 1: pure churn (worker frees everything) ----
static void scenario_pure_churn() {
    std::printf("[scenario 1] pure churn (worker frees all before exit)\n");
    size_t base = 0;
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

        if(cyc == WARMUP) base = reserved_mib();
        if(cyc >= WARMUP && (cyc % 10 == 0 || cyc == CYCLES - 1))
            std::printf("  cyc %3d: reserved=%4zu MiB  chunks_live=%zu\n",
                        cyc, reserved_mib(), chunks_live());
    }
    size_t end = reserved_mib();
    CHECK(!grew(base, end, 1),
          "pure churn: reserved grew %zu -> %zu MiB across %d cycles "
          "(regions NOT recycled across thread exit)", base, end, CYCLES - WARMUP);
    std::printf("  [%s] pure churn plateau: %zu -> %zu MiB\n",
                grew(base, end, 1) ? "BAD" : "ok", base, end);
}

// ---- Scenario 2: survivor churn (KAME-like) ----
static void scenario_survivor_churn() {
    std::printf("[scenario 2] survivor churn (K scattered survivors held by main)\n");
    std::vector<void *> survivors_prev;   // held across the join boundary by main
    size_t base = 0;
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
        // now-orphaned chunks -> should adopt/release & make them reusable).
        for(void *p : survivors_prev) kame_pool_free(p);
        survivors_prev.swap(survivors_cur);

        if(cyc == WARMUP) base = reserved_mib();
        if(cyc >= WARMUP && (cyc % 10 == 0 || cyc == CYCLES - 1))
            std::printf("  cyc %3d: reserved=%4zu MiB  chunks_live=%zu  survivors=%zu\n",
                        cyc, reserved_mib(), chunks_live(), survivors_prev.size());
    }
    for(void *p : survivors_prev) kame_pool_free(p);   // drain the last cycle

    size_t end = reserved_mib();
    // Allow a little slack (2 regions) for fragmentation jitter; a real
    // stranding bug shows steady multi-region growth far beyond that.
    CHECK(!grew(base, end, 2),
          "survivor churn: reserved grew %zu -> %zu MiB across %d cycles "
          "(orphaned partially-used chunks STRANDED, not reused)",
          base, end, CYCLES - WARMUP);
    std::printf("  [%s] survivor churn plateau: %zu -> %zu MiB\n",
                grew(base, end, 2) ? "BAD" : "ok", base, end);
}

int main() {
    scenario_pure_churn();
    scenario_survivor_churn();
    std::printf(g_fails == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fails);
    return g_fails ? 1 : 0;
}
