// Allocator stress + bench: rapid thread churn + mixed-size alloc/dealloc +
// cross-thread free, with sentinel-paint correctness checks.
//
// Goals
//   1. Exercise `~AllocPinCleanup` / `~CrossDeallocBatch` thread-teardown
//      paths under continuous churn — the recent TLS-dtor-order /
//      dangling-`s_chunks[cidx]` bugs were teardown-only.
//   2. Cover small (FS=true bucketed ≤256), medium (FS=false 257..512),
//      and large (>512 fallback to malloc) size classes in one run.
//   3. Exercise the cross-thread dealloc batch path (a slot allocated
//      by thread A and freed by thread B).
//   4. Detect memory corruption via per-allocation sentinel paint +
//      verify-on-free.
//   5. Surface leaks: allocs == frees at end.
//
// Usage:
//   alloc_stress_test [total_threads] [concurrent_threads]
//                     [ops_per_thread] [cross_thread_pct]
//
// Defaults are tuned for ~10 s wall-clock on a 4-core VM.
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "support_standalone.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

namespace {

struct Config {
    int total_threads;
    int concurrent_threads;
    int ops_per_thread;
    int cross_thread_pct; // 0..100
};

// Each tracked allocation records its size and the byte the buffer was
// painted with — verified before free.  `tid_alloc` lets cross-thread
// frees report which producer's data tripped.
struct AllocBlock {
    void *p;
    size_t size;
    uint8_t sentinel;
    int tid_alloc;
};

// Pseudo-realistic mixed size distribution:
//   60% ≤  64  B   (FS=true tight buckets)
//   25%  65..256 B (FS=true wider buckets)
//   10% 257..512 B (FS=false partial spec)
//    5% 513..8192 B (large fallback → malloc)
size_t pick_size(std::mt19937 &rng) {
    std::uniform_int_distribution<int> p(0, 99);
    int r = p(rng);
    if(r < 60) return 8 + (rng() % 57);          //   8..64
    if(r < 85) return 65 + (rng() % 192);        //  65..256
    if(r < 95) return 257 + (rng() % 256);       // 257..512
    return 513 + (rng() % 7680);                 // 513..8192
}

// Global counters — atomic, no allocation.
std::atomic<uint64_t> g_total_allocs{0};
std::atomic<uint64_t> g_total_frees{0};
std::atomic<uint64_t> g_sentinel_fails{0};

// Cross-thread free queue.  std::queue<>'s std::deque does small
// allocations of its own (which themselves go through the pool — that's
// part of the workload).  We protect with a plain mutex to keep the
// test code uncomplicated; throughput is dominated by per-thread work.
std::mutex g_cross_mu;
std::queue<AllocBlock> g_cross_q;

void check_sentinel(const AllocBlock &b, const char *where) {
    const uint8_t *q = static_cast<const uint8_t *>(b.p);
    for(size_t i = 0; i < b.size; ++i) {
        if(q[i] != b.sentinel) {
            fprintf(stderr,
                "FAIL sentinel mismatch %s p=%p size=%zu sentinel=0x%02x "
                "tid_alloc=%d offset=%zu got=0x%02x\n",
                where, b.p, b.size, b.sentinel, b.tid_alloc, i, q[i]);
            g_sentinel_fails.fetch_add(1, std::memory_order_relaxed);
            std::abort();
        }
    }
}

void worker(int tid, const Config &cfg) {
    std::mt19937 rng(uint32_t(tid * 2654435761u));
    std::uniform_int_distribution<int> pct(0, 99);

    // Local live set — pop a random index to free.
    std::vector<AllocBlock> live;
    live.reserve(cfg.ops_per_thread / 4);

    for(int i = 0; i < cfg.ops_per_thread; ++i) {
        // 60% alloc / 40% free when live is non-empty.
        bool do_alloc = live.empty() || pct(rng) < 60;
        if(do_alloc) {
            size_t s = pick_size(rng);
            auto *p = new char[s];
            uint8_t sent = uint8_t((tid * 131 + i * 7 + 1) & 0xff);
            std::memset(p, sent, s);
            AllocBlock b{p, s, sent, tid};

            if(pct(rng) < cfg.cross_thread_pct) {
                std::lock_guard<std::mutex> lk(g_cross_mu);
                g_cross_q.push(b);
            }
            else {
                live.push_back(b);
            }
            g_total_allocs.fetch_add(1, std::memory_order_relaxed);
        }
        else {
            std::uniform_int_distribution<size_t> idx(0, live.size() - 1);
            size_t k = idx(rng);
            AllocBlock b = live[k];
            live[k] = live.back();
            live.pop_back();
            check_sentinel(b, "local");
            delete[] static_cast<char *>(b.p);
            g_total_frees.fetch_add(1, std::memory_order_relaxed);
        }

        // Drain a small batch from the cross-thread queue every so
        // often — this is the path that exercises non-owner dealloc →
        // `tls_cross_dealloc_batch` push.
        if((i & 31) == 7) {
            for(int j = 0; j < 4; ++j) {
                AllocBlock b;
                {
                    std::lock_guard<std::mutex> lk(g_cross_mu);
                    if(g_cross_q.empty()) break;
                    b = g_cross_q.front();
                    g_cross_q.pop();
                }
                check_sentinel(b, "cross");
                delete[] static_cast<char *>(b.p);
                g_total_frees.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // Drop the rest of the local set before the thread exits — this
    // hammers the owner-side freelist push into `g_thread_slots[]`.
    for(auto &b : live) {
        check_sentinel(b, "drain");
        delete[] static_cast<char *>(b.p);
        g_total_frees.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace

int main(int argc, char **argv) {
    Config cfg;
    cfg.total_threads      = argc > 1 ? std::atoi(argv[1]) : 2000;
    cfg.concurrent_threads = argc > 2 ? std::atoi(argv[2]) : 32;
    cfg.ops_per_thread     = argc > 3 ? std::atoi(argv[3]) : 20000;
    cfg.cross_thread_pct   = argc > 4 ? std::atoi(argv[4]) : 10;

    fprintf(stderr,
        "[alloc_stress] total_threads=%d concurrent=%d ops/thread=%d "
        "cross_pct=%d%%\n",
        cfg.total_threads, cfg.concurrent_threads, cfg.ops_per_thread,
        cfg.cross_thread_pct);

    auto t0 = std::chrono::steady_clock::now();

    // Rolling pool: keep up to `concurrent_threads` workers alive at a
    // time.  Joining the FIRST element each cycle keeps thread teardown
    // continuous (= our target stress).
    std::vector<std::thread> active;
    active.reserve(cfg.concurrent_threads);
    int spawned = 0;
    int next_tid = 1;
    while(spawned < cfg.total_threads || !active.empty()) {
        while((int)active.size() < cfg.concurrent_threads
              && spawned < cfg.total_threads) {
            active.emplace_back(worker, next_tid++, std::cref(cfg));
            ++spawned;
        }
        if( !active.empty()) {
            active.front().join();
            active.erase(active.begin());
        }
    }

    // Main thread drains anything stuck in the cross-thread queue.
    while( !g_cross_q.empty()) {
        AllocBlock b = g_cross_q.front();
        g_cross_q.pop();
        check_sentinel(b, "main-drain");
        delete[] static_cast<char *>(b.p);
        g_total_frees.fetch_add(1, std::memory_order_relaxed);
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    uint64_t allocs = g_total_allocs.load();
    uint64_t frees  = g_total_frees.load();
    uint64_t fails  = g_sentinel_fails.load();
    uint64_t ops    = allocs + frees;

    bool leak = (allocs != frees);
    bool ok   = (fails == 0) && !leak;

    printf("%s\n", ok ? "PASS" : "FAIL");
    printf("[alloc_stress] threads=%d ops=%llu time=%.2fs "
           "rate=%.2fM ops/s (alloc+free)\n",
        spawned,
        (unsigned long long)ops,
        secs,
        ops / 1e6 / secs);
    printf("  allocs=%llu frees=%llu diff=%lld sentinel_fails=%llu\n",
        (unsigned long long)allocs,
        (unsigned long long)frees,
        (long long)(allocs - frees),
        (unsigned long long)fails);

    return ok ? 0 : 1;
}
