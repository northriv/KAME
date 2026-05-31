// Targeted repro for the bucket-34 (size 6144, ALIGN=1024, FS=false)
// SEGV reported under chunk sizes that yield m_count=1 (128 KiB on
// 64-bit).  The user observed:
//   "全ての alloc_stress 起動で SEGV → 0x7fffXX0000a0 (chunk_base + 0xa0)"
//   "chunk header (chunk_base + 0..15) のどこかが slot data で上書きされた
//    状態で trampoline call → 不正アドレスへジャンプ"
//
// Strategy: many threads, each allocates random-size mix from buckets
// 31..36 (which all share `PoolAllocator<ALLOC_ALIGN3, false>` chunks
// — ALLOC_ALIGN3=1024 on 64-bit / 512 on 32-bit).  Cross-thread free
// every iteration (slot allocated by thread A, freed by thread B's
// drain).  Sentinel paint + verify exposes any slot-overrun corruption
// before SEGV.
//
// Usage:
//   alloc_bucket34_repro [threads] [iters_per_thread]
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#ifndef DISABLE_POOL_ALLOCATOR
#  include "allocator_prv.h"  // for PoolAllocatorBase::count_live_chunks
#endif

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

constexpr int kAlive = 1024;
// Sizes covering buckets 31..36 — all FS=false ALIGN=ALLOC_ALIGN3 = 1024.
// The reported SEGV is at bucket 34 (size 6144) specifically.
constexpr std::size_t kSizes[] = {3072, 4096, 5120, 6144, 7168, 8192};
constexpr int kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);

struct Block {
    char *p;
    int size;
    int paint;
};

std::atomic<uint64_t> g_ops{0};
std::atomic<uint64_t> g_fails{0};

void touch_and_paint(Block &b, int paint) {
    b.paint = paint;
    // Paint first and last bytes — any slot-overrun corruption will
    // mismatch on verify.
    b.p[0] = (char)(paint & 0xFF);
    b.p[b.size - 1] = (char)(~paint & 0xFF);
}

bool verify(const Block &b) {
    char f = b.p[0], l = b.p[b.size - 1];
    if(f != (char)(b.paint & 0xFF) || l != (char)(~b.paint & 0xFF)) {
        fprintf(stderr,
            "[bucket34] sentinel fail: p=%p size=%d paint=%d "
            "first=%02x (expected %02x) last=%02x (expected %02x)\n",
            b.p, b.size, b.paint,
            (unsigned)(unsigned char)f, (unsigned)(unsigned char)(b.paint & 0xFF),
            (unsigned)(unsigned char)l, (unsigned)(unsigned char)(~b.paint & 0xFF));
        return false;
    }
    return true;
}

void run_worker(int tid, int iters) {
    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL ^ (uint64_t)tid);
    std::vector<Block> live(kAlive);
    // Initial fill.
    for(int i = 0; i < kAlive; ++i) {
        int sz = (int)kSizes[rng() % kNumSizes];
        live[i].p = new char[sz];
        live[i].size = sz;
        touch_and_paint(live[i], (tid << 16) | i);
    }
    for(int it = 0; it < iters; ++it) {
        for(int k = 0; k < kAlive; ++k) {
            int slot = (int)(rng() % kAlive);
            if( !verify(live[slot])) g_fails.fetch_add(1);
            delete[] live[slot].p;
            int sz = (int)kSizes[rng() % kNumSizes];
            live[slot].p = new char[sz];
            live[slot].size = sz;
            touch_and_paint(live[slot], (tid << 16) | (slot ^ it));
        }
        g_ops.fetch_add(2ULL * kAlive);
    }
    for(auto &b : live) { verify(b); delete[] b.p; }
}

}  // namespace

int main(int argc, char **argv) {
    int nthreads = argc > 1 ? std::atoi(argv[1]) : 8;
    int iters    = argc > 2 ? std::atoi(argv[2]) : 64;

    fprintf(stderr, "[bucket34_repro] threads=%d iters=%d\n", nthreads, iters);
#ifndef DISABLE_POOL_ALLOCATOR
    int chunks_initial = PoolAllocatorBase::count_live_chunks();
#endif

    std::vector<std::thread> ths;
    ths.reserve(nthreads);
    auto t0 = std::chrono::steady_clock::now();
    for(int i = 0; i < nthreads; ++i)
        ths.emplace_back(run_worker, i, iters);
    for(auto &t : ths) t.join();
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

#ifndef DISABLE_POOL_ALLOCATOR
    int chunks_final = PoolAllocatorBase::count_live_chunks();
#else
    int chunks_initial = 0, chunks_final = 0;
#endif

    uint64_t ops = g_ops.load();
    uint64_t fails = g_fails.load();
    printf("[bucket34_repro] threads=%d iters=%d ops=%llu time=%.2fs "
           "rate=%.2fM ops/s sentinel_fails=%llu chunks=%d→%d\n",
        nthreads, iters,
        (unsigned long long)ops, secs, ops / 1e6 / secs,
        (unsigned long long)fails,
        chunks_initial, chunks_final);
    return fails ? 1 : 0;
}
