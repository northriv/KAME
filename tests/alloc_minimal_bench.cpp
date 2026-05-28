// Minimal 1-thread alloc/dealloc bench — pure fast-path cost vs object size.
//
// Two patterns:
//   "hot"    — alloc-then-free in tight loop.  Each iteration: pop one slot
//              from the per-thread freelist, push one slot back.  Same slot
//              recycles → 1-deep freelist, fits in L1d.
//   "fifo"   — fill-then-drain.  Alloc N, then free N (FIFO order, oldest
//              first).  Stretches the freelist to N entries, then drains
//              in the same order (or LIFO depending on -DLIFO).
//
// No threading, no sentinel checking, no cross-thread frees.  Just
//   `new char[size]` / `delete[]` calls inlined into a `for` loop.
//
// Usage:
//   alloc_minimal_bench [size_bytes] [iterations] [pattern]
//     pattern: "hot" (default) or "fifo:N" (N = stash depth)
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "support_standalone.h"
#ifndef DISABLE_POOL_ALLOCATOR
#  include "allocator_prv.h"   // for PoolAllocatorBase::count_live_chunks
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    int size  = argc > 1 ? std::atoi(argv[1]) : 64;
    int iters = argc > 2 ? std::atoi(argv[2]) : 10'000'000;
    const char *pat = argc > 3 ? argv[3] : "hot";

    if(std::strncmp(pat, "fifo:", 5) == 0) {
        int depth = std::atoi(pat + 5);
        if(depth <= 0) depth = 1024;
        // Pre-allocate the stash so we time only the alloc/free body.
        std::vector<char *> stash(depth, nullptr);
        // Round iters to a multiple of depth for clean accounting.
        int rounds = iters / depth;
        if(rounds < 1) rounds = 1;
        int total_ops = rounds * depth * 2;

        auto t0 = std::chrono::steady_clock::now();
        for(int r = 0; r < rounds; ++r) {
            for(int i = 0; i < depth; ++i) stash[i] = new char[size];
            for(int i = 0; i < depth; ++i) delete[] stash[i];
        }
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::printf("[minimal] size=%5d B  fifo depth=%d  rounds=%d  "
            "ops=%d  time=%.3fs  rate=%.2fM ops/s\n",
            size, depth, rounds, total_ops, secs,
            total_ops / 1e6 / secs);
    }
    else if(std::strcmp(pat, "alloc_only") == 0) {
        // Pure alloc, then bulk free at end — measures only `operator new`
        // cost (one alloc per iteration, no delete in the loop).
        std::vector<char *> bag(iters, nullptr);
        auto t0 = std::chrono::steady_clock::now();
        for(int i = 0; i < iters; ++i) {
            bag[i] = new char[size];
            asm volatile("" : : "r"(bag[i]) : "memory");
        }
        auto t1 = std::chrono::steady_clock::now();
        for(auto *p : bag) delete[] p;
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::printf("[minimal] size=%5d B  alloc_only  iters=%d  "
            "time=%.3fs  rate=%.2fM ops/s  (alloc only)\n",
            size, iters, secs, iters / 1e6 / secs);
    }
    else if(std::strcmp(pat, "free_only") == 0) {
        // Pre-allocate, then time only the bulk free loop.  Isolates the
        // `operator delete` cost.
        std::vector<char *> bag(iters, nullptr);
        for(int i = 0; i < iters; ++i) bag[i] = new char[size];
        auto t0 = std::chrono::steady_clock::now();
        for(auto *p : bag) {
            asm volatile("" : : "r"(p) : "memory");
            delete[] p;
        }
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::printf("[minimal] size=%5d B  free_only  iters=%d  "
            "time=%.3fs  rate=%.2fM ops/s  (free only)\n",
            size, iters, secs, iters / 1e6 / secs);
    }
    else {
        // Defeat compiler elision of `new` + `delete[]` (the write to
        // `p[0]` becomes a dead store the compiler can remove, and then
        // the whole alloc/free pair becomes a no-op).  An inline-asm
        // memory clobber + register-clobber on the pointer forces the
        // compiler to materialise the pointer and treat the heap as
        // observable.  Zero extra instructions in the loop body.
#ifndef DISABLE_POOL_ALLOCATOR
        int chunks_before = PoolAllocatorBase::count_live_chunks();
#endif
        // Warm-up: alloc/free 1 slot to trigger chunk claim BEFORE
        // we start counting chunks_mid.
        { char *w = new char[size]; w[0] = 0; asm volatile("":::"memory"); delete[] w; }
#ifndef DISABLE_POOL_ALLOCATOR
        int chunks_warm = PoolAllocatorBase::count_live_chunks();
        int chunks_at_1   = -1;
        int chunks_at_100 = -1;
        int chunks_at_10k = -1;
#endif
        auto t0 = std::chrono::steady_clock::now();
        for(int i = 0; i < iters; ++i) {
            char *p = new char[size];
            p[0] = (char)i;
            asm volatile("" : : "r"(p) : "memory");
            delete[] p;
#ifndef DISABLE_POOL_ALLOCATOR
            if(i == 0)      chunks_at_1   = PoolAllocatorBase::count_live_chunks();
            if(i == 100)    chunks_at_100 = PoolAllocatorBase::count_live_chunks();
            if(i == 10000)  chunks_at_10k = PoolAllocatorBase::count_live_chunks();
#endif
        }
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
#ifndef DISABLE_POOL_ALLOCATOR
        int chunks_after = PoolAllocatorBase::count_live_chunks();
        std::printf("[minimal] size=%5d B  hot  iters=%d  ops=%d  "
            "time=%.3fs  rate=%.2fM ops/s  chunks=%d→%d  [i=1:%d, 100:%d, 10K:%d, end:%d]\n",
            size, iters, iters * 2,
            secs, iters * 2.0 / 1e6 / secs,
            chunks_before, chunks_warm,
            chunks_at_1, chunks_at_100, chunks_at_10k, chunks_after);
#else
        std::printf("[minimal] size=%5d B  hot  iters=%d  ops=%d  "
            "time=%.3fs  rate=%.2fM ops/s\n",
            size, iters, iters * 2,
            secs, iters * 2.0 / 1e6 / secs);
#endif
    }
    return 0;
}
