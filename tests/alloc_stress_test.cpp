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
#include "atomic_smart_ptr.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Cross-thread handoff via single-slot atomic swap — NO queue, NO CAS
// loop on the global, NO refcounting.  Each thread builds its outgoing
// chain LOCALLY (single-writer, plain pointers, zero atomic ops); when
// the chain fills its threshold (or the thread runs dry of pending
// work), the thread does ONE `std::atomic::exchange` against a global
// slot, trading its chain for whatever the slot held.  The acquired
// chain becomes the thread's new local chain — items to free / process
// locally.
//
// Why this beats the previous Treiber stacks:
//   * No CAS-loop: a single `exchange` per N pushes (here N = 32).
//     32× fewer atomic ops on the global cache line than per-op CAS.
//   * No refcounting: plain `CrossNode *`, no gref_ control block.
//   * No UAF concerns: nodes are owned exactly once at any moment —
//     by a thread's `tl_chain`, or by `g_swap_slot`, never both.
//     Pops touch only the local chain (never deref nodes that may be
//     in flight).
//   * No queue infrastructure: the "queue" is just one pointer.  No
//     ring, no allocator-shaped block traffic from std::deque.
//
// The trade-off: items are processed in a thread-shuffled order with
// up to ~threshold latency between push and the first consumer
// touching them.  Fine for a stress test — order doesn't matter, the
// sentinel check verifies the data was preserved across the handoff.
struct CrossNode {
    AllocBlock block;
    CrossNode *next;
};

std::atomic<CrossNode *> g_swap_slot{nullptr};

// Per-thread outgoing/incoming chain.  Single-writer (the thread
// itself) so no atomic ops needed on the chain itself.  Items arrive
// via the global swap and depart via the global swap or via local
// drains in `pop_cross`.
thread_local CrossNode *tl_chain = nullptr;
thread_local int tl_chain_count = 0;
constexpr int CROSS_SWAP_THRESHOLD = 32;

void push_cross(const AllocBlock &b) {
    auto *node = new CrossNode{b, tl_chain};
    tl_chain = node;
    if(++tl_chain_count >= CROSS_SWAP_THRESHOLD) {
        // Hand off our accumulated chain to the global slot, taking
        // whatever was there as our new local backlog.  One atomic
        // exchange, no loops.  Memory ordering: release to publish
        // the chain we built (so other threads see the linked-list
        // writes), acquire to see the chain we get back.
        CrossNode *mine = tl_chain;
        tl_chain = g_swap_slot.exchange(mine, std::memory_order_acq_rel);
        // Reset count.  We don't bother counting the chain we just
        // acquired — it'll drain back to 0 through pop_cross calls
        // and the next swap fires when it next exceeds the threshold.
        tl_chain_count = 0;
    }
}
bool pop_cross(AllocBlock &out) {
    if( !tl_chain) {
        // Local empty — try to acquire whatever the global slot holds.
        tl_chain = g_swap_slot.exchange(nullptr, std::memory_order_acquire);
        if( !tl_chain) return false;
    }
    CrossNode *n = tl_chain;
    tl_chain = n->next;
    out = n->block;
    delete n;
    return true;
}
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

// Called at end of each worker (and at the end of `main`) to drain
// the thread-local chain so no items are stuck in retired threads'
// `tl_chain`s.
void drain_local_cross() {
    while(tl_chain) {
        CrossNode *n = tl_chain;
        tl_chain = n->next;
        check_sentinel(n->block, "thread-exit");
        delete[] static_cast<char *>(n->block.p);
        delete n;
        g_total_frees.fetch_add(1, std::memory_order_relaxed);
    }
    tl_chain_count = 0;
}

void worker(int tid, const Config &cfg) {
    std::mt19937 rng(uint32_t(tid * 2654435761u));
    std::uniform_int_distribution<int> pct(0, 99);

    // Local live set — pop a random index to free.  Capped to keep
    // the bookkeeping vector itself out of the >16 KiB
    // `allocate_large_size_or_malloc` fallback (which routes to glibc
    // `std::malloc` on KAME and would distort the allocator
    // comparison).  When `live` reaches the cap we force-free instead
    // of allocating.
    std::vector<AllocBlock> live;
    constexpr size_t LIVE_CAP = 256;  // 256 * 32 B = 8 KiB ≤ pool max
    live.reserve(LIVE_CAP);

    for(int i = 0; i < cfg.ops_per_thread; ++i) {
        // 60% alloc / 40% free when live is non-empty (and below cap).
        // When at cap, force a free to keep `live` from growing the
        // backing vector past `LIVE_CAP * 32 B`.
        bool do_alloc =
            (live.empty() || pct(rng) < 60) && live.size() < LIVE_CAP;
        if(do_alloc) {
            size_t s = pick_size(rng);
            auto *p = new char[s];
            uint8_t sent = uint8_t((tid * 131 + i * 7 + 1) & 0xff);
            std::memset(p, sent, s);
            AllocBlock b{p, s, sent, tid};

            if(pct(rng) < cfg.cross_thread_pct) {
                push_cross(b);
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
                if( !pop_cross(b)) break;
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
    // Drain whatever cross-thread items this worker still holds in
    // `tl_chain` (received via global swap and not yet popped, plus
    // own items added since the last swap).  Otherwise they'd be lost
    // when the thread_local dtor wipes `tl_chain`.
    drain_local_cross();
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
    for(;;) {
        AllocBlock b;
        if( !pop_cross(b)) break;
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

    // Report peak virtual / resident set so allocators can be compared
    // on address-space cost.  VmPeak = peak virtual size (all mappings
    // ever reserved, even if since unmapped); VmHWM = peak resident
    // set (physically backed pages).
    {
        long vm_peak_kb = -1, vm_hwm_kb = -1, vm_size_kb = -1, vm_rss_kb = -1;
        if(FILE *f = std::fopen("/proc/self/status", "r")) {
            char line[256];
            while(std::fgets(line, sizeof(line), f)) {
                long kb;
                if(std::sscanf(line, "VmPeak: %ld kB", &kb) == 1) vm_peak_kb = kb;
                else if(std::sscanf(line, "VmHWM: %ld kB", &kb) == 1) vm_hwm_kb = kb;
                else if(std::sscanf(line, "VmSize: %ld kB", &kb) == 1) vm_size_kb = kb;
                else if(std::sscanf(line, "VmRSS: %ld kB", &kb) == 1) vm_rss_kb = kb;
            }
            std::fclose(f);
        }
        printf("[mem] VmPeak=%ld MiB  VmHWM=%ld MiB  VmSize=%ld MiB  VmRSS=%ld MiB\n",
               vm_peak_kb / 1024, vm_hwm_kb / 1024,
               vm_size_kb / 1024, vm_rss_kb / 1024);
    }

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
