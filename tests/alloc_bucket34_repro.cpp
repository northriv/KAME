// Targeted reproducer for the 128 KiB ALLOC_MIN_CHUNK_SIZE crash on
// Linux x86_64.  Hits the bucket 31..36 tier (sizes 3072 .. 8192,
// ALIGN3 = 1024, FS=false) — at 128 KiB chunks this template runs at
// `m_count = 1` (single FUINT word) which the crash report points at.
//
// Workload: N worker threads, each maintains a small LIVE_PER_THREAD
// rolling set of slots.  Each iteration:
//   1. Allocate `pick_size()` bytes from the large-slot tier
//      (between 2049 and 8192 — guarantees bucket 31..36).
//   2. Paint the slot with a per-slot sentinel byte across every byte
//      so any pre-existing chunk-header overwrite would have flipped
//      to the sentinel — slot data writers leave their signature.
//   3. Verify the sentinel of a randomly chosen earlier slot (catches
//      slot↔slot stomps where the under-budgeted m_count=1 layout
//      may have placed slots overlapping the next chunk's header).
//   4. Free.  If a debug build catches a corrupted chunk header here,
//      `KAME_DEBUG_CHUNK_HEADER` in allocator.cpp will dump the
//      offending header + slot and abort.
//
// Run with e.g. `./alloc_bucket34_repro 8 4096` for 8 threads × 4096
// ops/thread.  Defaults: nproc threads, 200000 ops each.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {

constexpr int    LIVE_PER_THREAD = 1024;
std::atomic<uint64_t> g_total_allocs{0};
std::atomic<uint64_t> g_sentinel_fails{0};

struct AllocBlock {
    char    *p;
    size_t   sz;
    uint8_t  sent;
    int      tid;
};

// Bucket 31..36 cover sizes 2049..8192 in 1024-byte steps:
//   slot_size = 2048 + (bucket-30)*1024.
// pick_size returns user-requested sizes that land in those buckets.
// Skewed toward the smaller end of the tier (sizes 3 .. 5 KiB) to
// keep allocation rate high while still exercising every bucket.
size_t pick_size(std::mt19937 &rng) {
    static const size_t sizes[] = {
        2050, 2500, 3000, 3072, 3500, 4000, 4096,
        4500, 5000, 5120, 5500, 6016, 6144, 7000, 7168, 8000, 8192
    };
    std::uniform_int_distribution<int> d(0, (int)(sizeof(sizes)/sizeof(sizes[0])) - 1);
    return sizes[d(rng)];
}

bool check_sentinel(const AllocBlock &b, const char *where) {
    for(size_t i = 0; i < b.sz; ++i) {
        if((uint8_t)b.p[i] != b.sent) {
            std::fprintf(stderr,
                "[%s] sentinel mismatch tid=%d slot=%p size=%zu off=%zu "
                "expected=0x%02x actual=0x%02x\n",
                where, b.tid, b.p, b.sz, i, b.sent, (uint8_t)b.p[i]);
            return false;
        }
    }
    return true;
}

struct Config {
    int    threads;
    int    ops_per_thread;
};

void worker(int tid, Config cfg) {
    std::mt19937 rng((uint32_t)(tid * 2654435761u + 1));
    std::uniform_int_distribution<int> action(0, 99);
    std::vector<AllocBlock> live;
    live.reserve(LIVE_PER_THREAD);

    for(int i = 0; i < cfg.ops_per_thread; ++i) {
        bool do_alloc =
            (live.empty() || action(rng) < 60) && (int)live.size() < LIVE_PER_THREAD;
        if(do_alloc) {
            size_t s = pick_size(rng);
            auto *p = new char[s];
            uint8_t sent = uint8_t((tid * 131 + i * 7 + 1) & 0xff);
            std::memset(p, sent, s);
            live.push_back(AllocBlock{p, s, sent, tid});
            g_total_allocs.fetch_add(1, std::memory_order_relaxed);
            // Spot-check an earlier slot: catches slot↔slot stomps
            // (e.g. an m_count=1 layout overrunning into a neighbour).
            if(live.size() > 4) {
                std::uniform_int_distribution<size_t> pick(0, live.size() - 2);
                auto &q = live[pick(rng)];
                if( !check_sentinel(q, "spot")) {
                    g_sentinel_fails.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        else {
            std::uniform_int_distribution<size_t> pick(0, live.size() - 1);
            size_t idx = pick(rng);
            auto b = live[idx];
            if( !check_sentinel(b, "pre-free")) {
                g_sentinel_fails.fetch_add(1, std::memory_order_relaxed);
            }
            delete[] b.p;
            live[idx] = live.back();
            live.pop_back();
        }
    }
    // Drain leftover.
    for(auto &b : live) {
        if( !check_sentinel(b, "drain")) g_sentinel_fails.fetch_add(1, std::memory_order_relaxed);
        delete[] b.p;
    }
}

} // namespace

int main(int argc, char **argv) {
    Config cfg;
    cfg.threads        = (argc > 1) ? std::atoi(argv[1]) : (int)std::thread::hardware_concurrency();
    cfg.ops_per_thread = (argc > 2) ? std::atoi(argv[2]) : 200000;
    if(cfg.threads <= 0)         cfg.threads = 4;
    if(cfg.ops_per_thread <= 0)  cfg.ops_per_thread = 200000;

    std::fprintf(stderr, "[alloc_bucket34_repro] threads=%d ops=%d live=%d\n",
        cfg.threads, cfg.ops_per_thread, LIVE_PER_THREAD);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ths;
    ths.reserve(cfg.threads);
    for(int t = 0; t < cfg.threads; ++t) ths.emplace_back(worker, t, cfg);
    for(auto &t : ths) t.join();
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    std::fprintf(stderr,
        "[alloc_bucket34_repro] done, allocs=%llu sentinel_fails=%llu time=%.2fs\n",
        (unsigned long long)g_total_allocs.load(),
        (unsigned long long)g_sentinel_fails.load(),
        sec);
    return g_sentinel_fails.load() ? 1 : 0;
}
