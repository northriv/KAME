// bench_xthread — cross-thread producer/consumer free benchmark.
//
// Independent reimplementation of the well-known "xmalloc-test" workload
// (Lever & Boreham, "malloc() Performance in a Multithreaded Linux
// Environment", USENIX 2000; reproduced in mimalloc-bench under GPL).  N
// producer threads allocate fixed-size objects in batches and hand them to
// a queue; N consumer threads dequeue batches and free.  Each free is
// therefore cross-thread (allocated in producer i, released in consumer
// j ≠ i), which is the canonical worst case for per-thread freelist /
// chunked-bitmap allocator designs.
//
// Build variants (selected via -DXTHREAD_USE_KAME_POOL):
//   * default                 — uses `malloc` / `free`.  Sees whatever
//                               allocator the dynamic linker resolved
//                               (LD_PRELOAD = libkamepoolalloc.so /
//                               libmimalloc.so / libjemalloc.so.2 / libc).
//   * -DXTHREAD_USE_KAME_POOL — uses `kame_pool_malloc` / `kame_pool_free`
//                               directly.  Bypasses the strong-symbol
//                               override layer to isolate "kame's
//                               allocator" from "kame's malloc override".
//
// Usage:
//   ./bench_xthread       -w 2 -s 64 -t 5
//   ./bench_xthread_pool  -w 2 -s 64 -t 5
//   LD_PRELOAD=.../libkamepoolalloc.so ./bench_xthread -w 2 -s 64 -t 5
//
// Output line (mimalloc-bench format):
//   rtime: 6.541, free/sec: 15.288 M
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef XTHREAD_USE_KAME_POOL
#  include "kame_pool.h"
#  define XT_MALLOC(sz) kame_pool_malloc((sz))
#  define XT_FREE(p)    kame_pool_free((p))
static const char *kBenchName = "bench_xthread_pool";
#else
#  define XT_MALLOC(sz) std::malloc((sz))
#  define XT_FREE(p)    std::free((p))
static const char *kBenchName = "bench_xthread";
#endif

// Lehmer-style PRNG with random.h-compatible seed/step semantics.
struct lran2_st { long x, y, v[32]; };
static void lran2_init(struct lran2_st *d, long seed) {
    d->x = (seed + 1) & 0x7fffffff;
    d->y = 1;
    for(int i = 0; i < 32; ++i) {
        d->x = (1103515245L * d->x + 12345L) & 0x7fffffff;
        d->v[i] = d->x;
    }
    d->x = (1103515245L * d->x + 12345L) & 0x7fffffff;
}
static long lran2(struct lran2_st *d) {
    int i = (d->y >> 26) & 0x1f;
    d->y = d->v[i];
    d->x = (1103515245L * d->x + 12345L) & 0x7fffffff;
    d->v[i] = d->x;
    return d->y;
}

#define DEFAULT_OBJECT_SIZE 1024
#define OBJECTS_PER_BATCH 4096

static int num_workers = 4;
static double run_time = 5.0;
static int object_size = DEFAULT_OBJECT_SIZE;
static pthread_t *thread_ids;
static struct counter { long c __attribute__((aligned(64))); } *counters;
static std::atomic<int> done_flag{0};
static struct timeval begin;

struct batch {
    struct batch *next_batch;
    void *objects[OBJECTS_PER_BATCH];
};
static volatile struct batch *batches = nullptr;
static volatile int batch_count = 0;
static const int batch_count_limit = 100;
static pthread_cond_t empty_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t full_cv  = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static const long possible_sizes[] = {
    8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048
};
static const int n_sizes = sizeof(possible_sizes) / sizeof(long);

static double elapsed_time(struct timeval *t0) {
    struct timeval now, td;
    gettimeofday(&now, nullptr);
    td.tv_sec  = now.tv_sec - t0->tv_sec;
    td.tv_usec = now.tv_usec - t0->tv_usec;
    if(td.tv_usec < 0) { td.tv_sec--; td.tv_usec += 1000000; }
    return td.tv_sec + td.tv_usec / 1e6;
}

static void enqueue_batch(struct batch *b) {
    pthread_mutex_lock(&lock);
    while(batch_count >= batch_count_limit && !done_flag.load(std::memory_order_acquire))
        pthread_cond_wait(&full_cv, &lock);
    b->next_batch = (struct batch *)batches;
    batches = b;
    batch_count++;
    pthread_cond_signal(&empty_cv);
    pthread_mutex_unlock(&lock);
}
static struct batch *dequeue_batch() {
    pthread_mutex_lock(&lock);
    while(batches == nullptr && !done_flag.load(std::memory_order_acquire))
        pthread_cond_wait(&empty_cv, &lock);
    struct batch *r = (struct batch *)batches;
    if(r) { batches = r->next_batch; batch_count--; pthread_cond_signal(&full_cv); }
    pthread_mutex_unlock(&lock);
    return r;
}

static void *mem_allocator(void *arg) {
    int tid = *(int *)arg;
    struct lran2_st lr; lran2_init(&lr, tid);
    while(!done_flag.load(std::memory_order_acquire)) {
        struct batch *b = (struct batch *)XT_MALLOC(sizeof(*b));
        for(int i = 0; i < OBJECTS_PER_BATCH; ++i) {
            std::size_t siz = object_size > 0 ?
                              (std::size_t)object_size :
                              (std::size_t)possible_sizes[lran2(&lr) % n_sizes];
            b->objects[i] = XT_MALLOC(siz);
            std::memset(b->objects[i], i % 256, siz > 128 ? 128 : siz);
        }
        enqueue_batch(b);
    }
    return nullptr;
}
static void *mem_releaser(void *arg) {
    int tid = *(int *)arg;
    while(!done_flag.load(std::memory_order_acquire)) {
        struct batch *b = dequeue_batch();
        if(b) {
            for(int i = 0; i < OBJECTS_PER_BATCH; ++i) XT_FREE(b->objects[i]);
            XT_FREE(b);
            counters[tid].c += OBJECTS_PER_BATCH;
        }
    }
    return nullptr;
}

static void usage(const char *prog) {
    std::fprintf(stderr,
        "%s [-w workers] [-t seconds] [-s size]\n"
        "  workers : N producer threads + N consumer threads (default 4)\n"
        "  seconds : wall-clock run time (default 5.0)\n"
        "  size    : fixed object size in bytes (default 1024; -1 = mix)\n",
        prog);
    std::exit(1);
}

int main(int argc, char **argv) {
    int c;
    while((c = getopt(argc, argv, "w:t:s:vh")) != -1) {
        switch(c) {
            case 'w': num_workers = std::atoi(optarg); break;
            case 't': run_time    = std::atof(optarg); break;
            case 's': object_size = std::atoi(optarg); break;
            case 'v': break;
            case 'h': default: usage(argv[0]);
        }
    }
    if(num_workers < 1) usage(argv[0]);

    thread_ids = (pthread_t *)XT_MALLOC(sizeof(pthread_t) * num_workers * 2);
    counters   = (struct counter *)XT_MALLOC(sizeof(*counters) * num_workers);
    int *ids   = (int *)XT_MALLOC(sizeof(int) * num_workers);
    for(int i = 0; i < num_workers; ++i) counters[i].c = 0;
#ifdef XTHREAD_USE_KAME_POOL
    kame_pool_stats_t st_before{};
    st_before.version = KAME_POOL_STATS_VERSION;
    kame_pool_get_stats(&st_before);
#endif
    gettimeofday(&begin, nullptr);
    for(int i = 0; i < num_workers; ++i) {
        ids[i] = i;
        pthread_create(&thread_ids[i * 2    ], nullptr, mem_releaser,  &ids[i]);
        pthread_create(&thread_ids[i * 2 + 1], nullptr, mem_allocator, &ids[i]);
    }
    while(elapsed_time(&begin) <= run_time) usleep(1000);
    done_flag.store(1, std::memory_order_release);
    pthread_cond_broadcast(&empty_cv);
    pthread_cond_broadcast(&full_cv);
    for(int i = 0; i < num_workers * 2; ++i) pthread_join(thread_ids[i], nullptr);
    double et = elapsed_time(&begin);
    long total = 0;
    for(int i = 0; i < num_workers; ++i) total += counters[i].c;
    double mfree = (double)total / et * 1e-6;
    double rtime = 100.0 / mfree;
    std::printf("[%s] workers=%d size=%d run=%.1fs  "
                "rtime: %.3f, free/sec: %.3f M\n",
                kBenchName, num_workers, object_size, run_time, rtime, mfree);
#ifdef XTHREAD_USE_KAME_POOL
    kame_pool_stats_t st_after{};
    st_after.version = KAME_POOL_STATS_VERSION;
    kame_pool_get_stats(&st_after);
    std::printf("[%s] pool: regions %zu→%zu (+%zu), units_live %zu→%zu, "
                "chunks_live %zu→%zu, large_alloc %zu→%zu, cache %zuMiB\n",
                kBenchName,
                st_before.regions_populated, st_after.regions_populated,
                st_after.regions_populated - st_before.regions_populated,
                st_before.units_live, st_after.units_live,
                st_before.chunks_live, st_after.chunks_live,
                st_before.large_alloc_count, st_after.large_alloc_count,
                st_after.cache_bytes >> 20);
#endif
    // Drain residual batches so leak checkers don't flag the workload.
    while(batches) {
        struct batch *b = (struct batch *)batches;
        batches = b->next_batch;
        for(int i = 0; i < OBJECTS_PER_BATCH; ++i) XT_FREE(b->objects[i]);
        XT_FREE(b);
    }
    XT_FREE(thread_ids);
    XT_FREE(counters);
    XT_FREE(ids);
    return 0;
}
