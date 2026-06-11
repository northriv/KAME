/* bench_loop — single-thread tight malloc(size)+free(p) loop.
 *
 * Same workload shape as `alloc_minimal_bench` "hot" mode, but routed
 * through `malloc` / `free` instead of C++ `new`/`delete[]`.  Lets you
 * compare the kame strong-symbol override path against libc / mimalloc /
 * jemalloc by switching `LD_PRELOAD`.
 *
 *   * default                — uses `malloc` / `free`.  On Linux/macOS this
 *                              sees whatever allocator is preloaded
 *                              (LD_PRELOAD / DYLD_INSERT_LIBRARIES).
 *   * -DLOOP_USE_KAME_POOL    — uses `kame_pool_malloc` / `kame_pool_free`
 *                              directly (built as `bench_loop_pool`).  This
 *                              is the Windows "kame" route: PE/COFF has no
 *                              LD_PRELOAD, and llvm-mingw resolves `malloc`
 *                              statically (no IAT entry for the §31 redirect
 *                              to patch), so the plain `malloc` build stays
 *                              at libc speed there.  Mirrors the
 *                              bench_xthread / bench_xthread_pool split.
 *
 * Build (with parent CMakeLists.txt under tests/):
 *   cmake --build . --target bench_loop        # malloc/free
 *   cmake --build . --target bench_loop_pool   # kame_pool_malloc/free
 *
 * Usage:
 *   ./bench_loop [size_bytes=64] [iters=30000000]
 *   LD_PRELOAD=.../libkamepoolalloc.so ./bench_loop 64 30000000
 *   LD_PRELOAD=.../libmimalloc.so      ./bench_loop 64 30000000
 *   ./bench_loop_pool 64 30000000              # direct pool, any OS
 *
 * Output line:
 *   [bench_loop] size=   64 B  iters=30000000  ops=60000000  time=0.250s
 *               rate=240.00M ops/s
 *
 * Co-Authored-By: Claude <noreply@anthropic.com>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef LOOP_USE_KAME_POOL
#  include "kame_pool.h"
#  define BL_MALLOC(sz) kame_pool_malloc((sz))
#  define BL_FREE(p)    kame_pool_free((p))
static const char *kBenchName = "bench_loop_pool";
#else
#  define BL_MALLOC(sz) malloc((sz))
#  define BL_FREE(p)    free((p))
static const char *kBenchName = "bench_loop";
#endif

/* Monotonic wall clock.  clock_gettime(CLOCK_MONOTONIC) on POSIX
 * (incl. llvm-mingw); QueryPerformanceCounter under genuine MSVC,
 * whose UCRT has neither clock_gettime nor CLOCK_MONOTONIC. */
#if defined(_MSC_VER) && !defined(__GNUC__)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
static double now_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/* Compiler barrier so the alloc/free pair can't be elided.  GNU inline
 * asm on GCC/clang; _ReadWriteBarrier (MSVC intrinsic) under cl. */
#if defined(_MSC_VER) && !defined(__GNUC__)
#  include <intrin.h>
#  define BL_CLOBBER(p) do { (void)(p); _ReadWriteBarrier(); } while(0)
#else
#  define BL_CLOBBER(p) __asm__ __volatile__("" : : "r"(p) : "memory")
#endif

int main(int argc, char **argv) {
    size_t size  = (argc > 1) ? (size_t)atoi(argv[1]) : 64;
    long   iters = (argc > 2) ? atol(argv[2])         : 30000000L;

    /* Warm-up: pull the first chunk so its claim cost is excluded. */
    void *w = BL_MALLOC(size);
    if(!w) { fprintf(stderr, "warmup malloc failed\n"); return 2; }
    ((char *)w)[0] = 0;
    BL_FREE(w);

    double t0 = now_s();
    for(long i = 0; i < iters; ++i) {
        char *p = (char *)BL_MALLOC(size);
        p[0] = (char)i;
        /* Memory clobber so the compiler can't elide the alloc/free pair. */
        BL_CLOBBER(p);
        BL_FREE(p);
    }
    double t1 = now_s();
    double secs = t1 - t0;
    double mops = (double)iters * 2.0 / 1e6 / secs;
    printf("[%s] size=%5zu B  iters=%ld  ops=%ld  time=%.3fs  "
           "rate=%.2fM ops/s\n",
           kBenchName, size, iters, iters * 2, secs, mops);
    return 0;
}
