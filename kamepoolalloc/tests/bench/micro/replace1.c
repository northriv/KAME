/* larson's steady state without its threads: one thread, LIVE blocks of
 * random size in [MIN, MAX), each iteration frees a random one and
 * allocates a replacement. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/resource.h>
#include <dlfcn.h>
#include "kame_pool.h"
static uint64_t x = 88172645463325252ull;
static inline uint64_t rnd(void) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    long live = argc > 1 ? atol(argv[1]) : 5000;
    long iters = argc > 2 ? atol(argv[2]) : 20000000;
    long mn = argc > 3 ? atol(argv[3]) : 8, mx = argc > 4 ? atol(argv[4]) : 1000;
    char **a = malloc(sizeof(char *) * live);
    for (long i = 0; i < live; i++) { a[i] = malloc(mn + rnd() % (mx - mn)); a[i][0] = 1; }
    double t0 = now();
    for (long i = 0; i < iters; i++) {
        long v = rnd() % live;
        free(a[v]);
        a[v] = malloc(mn + rnd() % (mx - mn));
        a[v][0] = 1;
    }
    double t1 = now();
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    int (*gs)(kame_pool_stats_t *) = (int (*)(kame_pool_stats_t *))dlsym(RTLD_DEFAULT, "kame_pool_get_stats");
    if (gs) { kame_pool_stats_t st; memset(&st, 0, sizeof st); st.version = 1; gs(&st);
              printf("  chunks_live=%zu units_live=%zu ", st.chunks_live, st.units_live); }
    printf("replace1 live=%ld size=[%ld,%ld) rate=%.2fM ops/s maxrss=%.1fMiB\n", live, mn, mx, 2.0 * iters / (t1 - t0) * 1e-6,
#ifdef __APPLE__
           ru.ru_maxrss / 1048576.0
#else
           ru.ru_maxrss / 1024.0
#endif
           );
    for (long i = 0; i < live; i++) free(a[i]);
    free(a);
    return 0;
}
