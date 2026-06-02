/* bench_loop — single-thread tight malloc(size)+free(p) loop.
 *
 * Same workload shape as `alloc_minimal_bench` "hot" mode, but routed
 * through `malloc` / `free` instead of C++ `new`/`delete[]`.  Lets you
 * compare the kame strong-symbol override path against libc / mimalloc /
 * jemalloc by switching `LD_PRELOAD`.
 *
 * Build (with parent CMakeLists.txt under tests/):
 *   cmake --build . --target bench_loop
 *
 * Usage:
 *   ./bench_loop [size_bytes=64] [iters=30000000]
 *   LD_PRELOAD=.../libkamepoolalloc.so ./bench_loop 64 30000000
 *   LD_PRELOAD=.../libmimalloc.so      ./bench_loop 64 30000000
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

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    size_t size  = (argc > 1) ? (size_t)atoi(argv[1]) : 64;
    long   iters = (argc > 2) ? atol(argv[2])         : 30000000L;

    /* Warm-up: pull the first chunk so its claim cost is excluded. */
    void *w = malloc(size);
    if(!w) { fprintf(stderr, "warmup malloc failed\n"); return 2; }
    ((char *)w)[0] = 0;
    free(w);

    double t0 = now_s();
    for(long i = 0; i < iters; ++i) {
        char *p = (char *)malloc(size);
        p[0] = (char)i;
        /* Memory clobber so the compiler can't elide the alloc/free pair. */
        __asm__ __volatile__("" : : "r"(p) : "memory");
        free(p);
    }
    double t1 = now_s();
    double secs = t1 - t0;
    double mops = (double)iters * 2.0 / 1e6 / secs;
    printf("[bench_loop] size=%5zu B  iters=%ld  ops=%ld  time=%.3fs  "
           "rate=%.2fM ops/s\n",
           size, iters, iters * 2, secs, mops);
    return 0;
}
