// bench_xlatency — producer allocation latency while a consumer frees at
// random across a large live set.
//
// The shape kamepoolalloc v1.2.0 (§revive) was measured on.  A producer
// allocates SIZE-byte blocks and hands each to a consumer over a
// single-producer / single-consumer ring.  The consumer holds LIVE blocks;
// for every block it receives it frees a randomly chosen live one and keeps
// the new block in its place.  So the live set stays at LIVE, every free is
// cross-thread, and the room each free makes lands on a random one of the
// producer's chunks.  Each producer malloc is timed on its own, and the
// report is the distribution: median, p99, p99.9, max.
//
// An allocator that has to search its chunks for the room those frees made
// shows a median that grows with LIVE; one that is told where the room is
// does not.  The throughput benches (bench_xthread, xmalloc-test) average
// this away.
//
// Routed through plain malloc/free and NOT linked against kamepoolalloc, so
// the allocator under test is whatever LD_PRELOAD / DYLD_INSERT_LIBRARIES
// supplies, and one binary measures every arm of an A/B.  When the preloaded
// library exports kame_pool_set_realtime_thread (kamepoolalloc v1.1.0 and
// later) the producer is marked KAME_RT_DEFER, as KAME marks its realtime
// threads; --no-rt leaves it unmarked.  What the harness itself needs (the
// sample buffer, the live array, the ring) comes from mmap or static
// storage, so the allocator under test sees only the workload.
//
// Usage:
//   bench_xlatency [-s size] [-l live] [-n samples] [-t seconds]
//                  [-f fill-seconds] [--no-rt] [--cpu P,C]
//     size     block size in bytes                        (default 1024)
//     live     blocks the consumer keeps                  (default 200000)
//     samples  timed allocations, at most                 (default 2000000)
//     seconds  time limit of the timed phase              (default 10)
//     fill     time limit of filling the live set         (default 120)
//     --cpu    pin the producer and the consumer (Linux)
//
// Output, one line, latencies in ns.  A percentile the sample count cannot
// support (fewer than 10 samples beyond it) prints as "na".  clk_ns is the
// median cost of the timing itself, which every sample includes, and tick_ns
// the clock's step — 41.7 ns on Apple silicon, whose counter runs at 24 MHz,
// so short latencies there are multiples of it:
//   [bench_xlatency] size=1024 live=200000 rt=defer n=2000000 secs=1.82
//   fill_secs=0.21 rate_M=1.10 clk_ns=21 tick_ns=1 med_ns=130 p99_ns=260
//   p999_ns=310 max_ns=14822 status=ok
// Exit status: 0 ok, 2 usage, 3 the fill hit its time limit, 4 malloc failed.
//
// Licensed under Apache-2.0 OR GPL-2.0-or-later, as the rest of the tree.

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__APPLE__)
#  include <sys/qos.h>
#elif defined(__linux__)
#  include <sched.h>
#endif

static const char *kBenchName = "bench_xlatency";

// ---------------------------------------------------------------- params
static std::size_t g_size     = 1024;
static std::size_t g_live     = 200000;
static std::size_t g_nsamples = 2000000;
static double      g_seconds  = 10.0;
static double      g_fill_s   = 120.0;
static bool        g_rt       = true;
static int         g_cpu_p    = -1, g_cpu_c = -1;

// ---------------------------------------------------------------- clock
using clk = std::chrono::steady_clock;
static inline std::uint64_t now_ns() {
    return (std::uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               clk::now().time_since_epoch()).count();
}

static inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#endif
}

// ---------------------------------------------------------------- ring
// SPSC, static storage.  A full ring stalls the producer outside the timed
// window, so a slow consumer lowers the rate but not the latencies.
enum : std::size_t { RING = 4096 };
struct Ring {
    alignas(64) std::atomic<std::size_t> head{0};   // producer's
    alignas(64) std::atomic<std::size_t> tail{0};   // consumer's
    alignas(64) void *slot[RING];
};
static Ring g_ring;

static void push(void *p) {
    std::size_t h = g_ring.head.load(std::memory_order_relaxed);
    while(h - g_ring.tail.load(std::memory_order_acquire) >= RING)
        cpu_relax();
    g_ring.slot[h & (RING - 1)] = p;
    g_ring.head.store(h + 1, std::memory_order_release);
}

static void *pop() {
    std::size_t t = g_ring.tail.load(std::memory_order_relaxed);
    while(g_ring.head.load(std::memory_order_acquire) == t)
        cpu_relax();
    void *p = g_ring.slot[t & (RING - 1)];
    g_ring.tail.store(t + 1, std::memory_order_release);
    return p;
}

// ---------------------------------------------------------------- state
static std::uint32_t *g_samples;          // mmap'd, g_nsamples entries
static void         **g_live_set;         // mmap'd, g_live entries
static std::size_t    g_n;                // samples taken
static std::uint64_t  g_t0, g_t1, g_fill_ns;
static const char    *g_status = "ok";
static const char    *g_rt_state = "off";
static std::atomic<bool> g_filled{false}, g_consumer_done{false};

static void *map_zeroed(std::size_t bytes) {
    void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
    if(p == MAP_FAILED) {
        std::perror("mmap");
        std::exit(4);
    }
    std::memset(p, 0, bytes);     // fault it in before anything is timed
    return p;
}

//! Keep both threads on performance cores (macOS has no pinning) or pin
//! them when asked (Linux).
static void place_this_thread(int cpu) {
#if defined(__APPLE__)
    (void)cpu;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#elif defined(__linux__)
    if(cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO( &set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#else
    (void)cpu;
#endif
}

//! Mark this thread realtime if the preloaded allocator knows how.
static const char *mark_realtime() {
    if( !g_rt) return "off";
    void *sym = dlsym(RTLD_DEFAULT, "kame_pool_set_realtime_thread");
    if( !sym) return "na";
    reinterpret_cast<void (*)(int)>(sym)(1);    // KAME_RT_DEFER
    return "defer";
}

// ---------------------------------------------------------------- threads
static void *producer(void *) {
    place_this_thread(g_cpu_p);
    g_rt_state = mark_realtime();

    const std::uint64_t fill_limit = (std::uint64_t)(g_fill_s * 1e9);
    const std::uint64_t f0 = now_ns();
    for(std::size_t i = 0; i < g_live; ++i) {
        char *p = static_cast<char *>(std::malloc(g_size));
        if( !p) { g_status = "malloc_failed"; goto done; }
        p[0] = (char)i;
        push(p);
        if(((i & 1023) == 0) && (now_ns() - f0 > fill_limit)) {
            g_status = "fill_timeout";
            goto done;
        }
    }
    while( !g_filled.load(std::memory_order_acquire))
        cpu_relax();
    g_fill_ns = now_ns() - f0;

    {
        const std::uint64_t limit = (std::uint64_t)(g_seconds * 1e9);
        std::size_t n = 0;
        g_t0 = now_ns();
        while(n < g_nsamples) {
            const std::uint64_t a = now_ns();
            char *p = static_cast<char *>(std::malloc(g_size));
            const std::uint64_t b = now_ns();
            if( !p) { g_status = "malloc_failed"; break; }
            p[0] = (char)n;
            const std::uint64_t d = b - a;
            g_samples[n++] = d > 0xffffffffull ? 0xffffffffu : (std::uint32_t)d;
            push(p);
            if(b - g_t0 > limit) break;
        }
        g_t1 = now_ns();
        g_n = n;
    }
done:
    push(nullptr);
    // Stay alive until the consumer has freed everything: a free racing
    // its owner's exit is a different test (alloc_tsd_exclusivity_test).
    while( !g_consumer_done.load(std::memory_order_acquire))
        usleep(100);
    return nullptr;
}

static void *consumer(void *) {
    place_this_thread(g_cpu_c);
    std::size_t held = 0;
    for(; held < g_live; ++held) {
        void *p = pop();
        if( !p) goto cleanup;
        g_live_set[held] = p;
    }
    g_filled.store(true, std::memory_order_release);
    {
        std::uint64_t x = 0x9e3779b97f4a7c15ull;    // xorshift64*
        for(;;) {
            void *p = pop();
            if( !p) break;
            x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
            const std::size_t r =
                (std::size_t)((x * 0x2545f4914f6cdd1dull) >> 11) % g_live;
            std::free(g_live_set[r]);
            g_live_set[r] = p;
        }
    }
cleanup:
    for(std::size_t i = 0; i < held; ++i) std::free(g_live_set[i]);
    g_consumer_done.store(true, std::memory_order_release);
    return nullptr;
}

// ---------------------------------------------------------------- report
//! Nearest-rank percentile of the sorted samples, or "na" when fewer than
//! 10 samples lie beyond it.
static void pct(char *buf, std::size_t len, double p) {
    if(g_n == 0 || (double)g_n * (1.0 - p) < 10.0) {
        std::snprintf(buf, len, "na");
        return;
    }
    std::size_t k = (std::size_t)((double)g_n * p + 0.999999);
    if(k == 0) k = 1;
    std::snprintf(buf, len, "%u", g_samples[k - 1]);
}

//! Median cost of a timing pair, and the clock's step: the smallest advance
//! seen when spinning until the value changes.
static void clock_cost(unsigned &med, unsigned &tick) {
    std::uint32_t v[1001];
    for(auto &e : v) {
        const std::uint64_t a = now_ns();
        const std::uint64_t b = now_ns();
        e = (std::uint32_t)(b - a);
    }
    std::sort(v, v + 1001);
    med = v[500];
    tick = 0xffffffffu;
    for(int i = 0; i < 1000; ++i) {
        const std::uint64_t a = now_ns();
        std::uint64_t b;
        while((b = now_ns()) == a) {}
        if(b - a < tick) tick = (unsigned)(b - a);
    }
}

static void usage(const char *prog) {
    std::fprintf(stderr,
        "%s [-s size] [-l live] [-n samples] [-t seconds] [-f fill-seconds]"
        " [--no-rt] [--cpu P,C]\n", prog);
    std::exit(2);
}

int main(int argc, char **argv) {
    for(int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const bool more = i + 1 < argc;
        if( !std::strcmp(a, "-s") && more) g_size = std::strtoull(argv[++i], nullptr, 10);
        else if( !std::strcmp(a, "-l") && more) g_live = std::strtoull(argv[++i], nullptr, 10);
        else if( !std::strcmp(a, "-n") && more) g_nsamples = std::strtoull(argv[++i], nullptr, 10);
        else if( !std::strcmp(a, "-t") && more) g_seconds = std::atof(argv[++i]);
        else if( !std::strcmp(a, "-f") && more) g_fill_s = std::atof(argv[++i]);
        else if( !std::strcmp(a, "--no-rt")) g_rt = false;
        else if( !std::strcmp(a, "--cpu") && more) {
            if(std::sscanf(argv[++i], "%d,%d", &g_cpu_p, &g_cpu_c) != 2) usage(argv[0]);
        }
        else usage(argv[0]);
    }
    if(g_size == 0 || g_live == 0 || g_nsamples == 0) usage(argv[0]);

    g_samples  = static_cast<std::uint32_t *>(map_zeroed(g_nsamples * sizeof(std::uint32_t)));
    g_live_set = static_cast<void **>(map_zeroed(g_live * sizeof(void *)));
    unsigned clk_ns, tick_ns;
    clock_cost(clk_ns, tick_ns);

    pthread_t tp, tc;
    pthread_create( &tc, nullptr, consumer, nullptr);
    pthread_create( &tp, nullptr, producer, nullptr);
    pthread_join(tp, nullptr);
    pthread_join(tc, nullptr);

    std::sort(g_samples, g_samples + g_n);
    char p99[24], p999[24], med[24], mx[24];
    if(g_n) {
        std::snprintf(med, sizeof(med), "%u", g_samples[(g_n - 1) / 2]);
        std::snprintf(mx, sizeof(mx), "%u", g_samples[g_n - 1]);
    }
    else {
        std::snprintf(med, sizeof(med), "na");
        std::snprintf(mx, sizeof(mx), "na");
    }
    pct(p99, sizeof(p99), 0.99);
    pct(p999, sizeof(p999), 0.999);
    const double secs = g_n ? (double)(g_t1 - g_t0) * 1e-9 : 0.0;
    std::printf("[%s] size=%zu live=%zu rt=%s n=%zu secs=%.2f fill_secs=%.2f "
                "rate_M=%.3f clk_ns=%u tick_ns=%u med_ns=%s p99_ns=%s "
                "p999_ns=%s max_ns=%s status=%s\n",
                kBenchName, g_size, g_live, g_rt_state, g_n, secs,
                (double)g_fill_ns * 1e-9, secs > 0 ? (double)g_n / secs * 1e-6 : 0.0,
                clk_ns, tick_ns, med, p99, p999, mx, g_status);
    if( !std::strcmp(g_status, "fill_timeout")) return 3;
    if( !std::strcmp(g_status, "malloc_failed")) return 4;
    return 0;
}
