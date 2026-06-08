// Thread-exit free leak repro: freeing pool memory from a pthread_key
// destructor (i.e. DURING thread teardown, after the allocator's own
// per-thread cleanup has run) must still reclaim the thread's chunks.
//
// This models KAME's real pattern: a per-thread receive buffer held in an
// `XThreadLocal<std::vector<unsigned char>>` (libcharinterface) is freed via a
// pthread_key destructor (`Transactional::detail::pthread_destroy`) when a
// driver thread stops.  KAME starts/stops such threads constantly (driver
// start/stop), so any per-thread-exit stranding accumulates without bound.
//
// Two scenarios, IDENTICAL except for WHERE the free happens:
//
//   (A) FREE-IN-BODY  — the worker frees everything before it returns, while it
//       still owns its chunks.  Baseline: MUST plateau (proves the harness is
//       not trivially always-growing).
//
//   (B) FREE-AT-EXIT  — the worker strands its pointers in a pthread_key TSD;
//       the key's destructor frees them during _pthread_tsd_cleanup, AFTER the
//       allocator's C++ thread_local teardown has orphaned the chunks.  This is
//       the leaking path: on the buggy allocator the orphaned chunk's
//       m_owner_id (0) collides with the teardown sentinel page's owner_id (0),
//       so deallocate takes the fast owner-free path (freelist_push) WITHOUT
//       decrementing MASK_CNT — the orphan never looks empty to
//       orphan_chain_scrub and is never reclaimed.
//
// Assertions use `units_live` / `chunks_live` (claim-bitmap walk, madvise
// INDEPENDENT), NOT `bytes_reserved` (which is a monotone high-water mark and
// on macOS is clouded by lazy MADV_FREE).  That makes this test a valid leak
// gate on macOS, unlike the reserved-based alloc_thread_churn_test.
//
// Pool-only (kame_pool_* C API); built when USE_KAME_ALLOCATOR is ON.
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "kame_pool.h"

#include <pthread.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static int g_fails = 0;
#define CHECK(c, ...) do { if(!(c)) { ++g_fails; \
    std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while(0)

// Tunables — modest so the whole test runs in a few seconds.
static const int CYCLES = 120;   // spawn/join iterations
static const int WARMUP = 20;    // cycles to reach steady-state before sampling
static const int M      = 6000;  // allocations per worker

// Sizes spanning FS=true small buckets, FS=false borrow buckets, and the
// dedicated (>512 B) tier, so many distinct (ALIGN,FS) templates are exercised.
static inline size_t pick_size(int i) { return (size_t)32 + (size_t)((i * 7) % 2200); }

static size_t units_live() {
    kame_pool_stats_t s; std::memset(&s, 0, sizeof s); s.version = 2;
    kame_pool_get_stats(&s);
    return s.units_live;
}
static size_t chunks_live() {
    kame_pool_stats_t s; std::memset(&s, 0, sizeof s); s.version = 2;
    kame_pool_get_stats(&s);
    return s.chunks_live;
}

// pthread_key whose destructor frees pooled pointers during thread teardown.
static pthread_key_t g_tsd_key;
static void tsd_free_dtor(void *v) {
    auto *vec = static_cast<std::vector<void *> *>(v);
    for(void *p : *vec) kame_pool_free(p);
    delete vec;
}

static void fill(std::vector<void *> *vec) {
    vec->reserve(M);
    for(int i = 0; i < M; i++) {
        size_t sz = pick_size(i);
        void *p = kame_pool_malloc(sz);
        if(p) { std::memset(p, 0xAB, sz > 64 ? 64 : sz); vec->push_back(p); }
    }
}

// True if `cur` exceeds `base` by more than `slack` units (256 KiB each).  A
// real stranding bug shows steady growth far beyond a small fragmentation slack.
static bool grew(size_t base, size_t cur, size_t slack) { return cur > base + slack; }

static void scenario(bool free_at_exit, const char *name) {
    std::printf("[%s] %s\n", name,
                free_at_exit ? "free via pthread_key dtor (at thread exit)"
                             : "free in worker body (before exit)");
    size_t base_u = 0, base_c = 0;
    for(int cyc = 0; cyc < CYCLES; cyc++) {
        std::thread([free_at_exit] {
            auto *vec = new std::vector<void *>();
            fill(vec);
            if(free_at_exit) {
                pthread_setspecific(g_tsd_key, vec);   // dtor frees at thread exit
            } else {
                for(void *p : *vec) kame_pool_free(p);
                delete vec;
            }
        }).join();   // join => the worker (incl. its TSD cleanup) has fully exited

        if(cyc == WARMUP) { base_u = units_live(); base_c = chunks_live(); }
        if(cyc >= WARMUP && (cyc % 20 == 0 || cyc == CYCLES - 1))
            std::printf("  cyc %3d: units_live=%6zu chunks_live=%6zu\n",
                        cyc, units_live(), chunks_live());
    }
    size_t end_u = units_live(), end_c = chunks_live();
    // Allow a few units of fragmentation jitter; a stranding bug shows
    // hundreds-to-thousands of units of monotone growth over (CYCLES-WARMUP).
    CHECK(!grew(base_u, end_u, 16),
          "%s: units_live grew %zu -> %zu over %d cycles "
          "(thread-exit frees STRANDED, chunks never reclaimed)",
          name, base_u, end_u, CYCLES - WARMUP);
    CHECK(!grew(base_c, end_c, 16),
          "%s: chunks_live grew %zu -> %zu over %d cycles",
          name, base_c, end_c, CYCLES - WARMUP);
    std::printf("  [%s] %s plateau: units %zu -> %zu, chunks %zu -> %zu\n",
                grew(base_u, end_u, 16) ? "BAD" : "ok", name,
                base_u, end_u, base_c, end_c);
}

int main() {
    pthread_key_create(&g_tsd_key, tsd_free_dtor);
    scenario(false, "scenario A (control)");
    scenario(true,  "scenario B (thread-exit free)");
    std::printf(g_fails == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fails);
    return g_fails ? 1 : 0;
}
