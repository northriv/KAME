// heavy: allocation-heavy with sparse frees -- the heap keeps growing and
// only every EVERY-th allocation is followed by a free of a random old block.
// A miss that walks the chunks for freelist room must go round a large heap
// again after each such free.  usage: heavy LIVE0 ALLOCS EVERY MIN MAX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <dlfcn.h>
typedef struct { unsigned version, version_supported; size_t regions_populated, bytes_reserved,
                 chunks_live, units_live; } st_t;
static unsigned long long x = 88172645463325252ull;
static unsigned long long rnd(void) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; }
int main(int argc, char **argv) {
	long live0 = argc > 1 ? atol(argv[1]) : 100000;
	long allocs = argc > 2 ? atol(argv[2]) : 1000000;
	long every = argc > 3 ? atol(argv[3]) : 32;
	long mn = argc > 4 ? atol(argv[4]) : 32, mx = argc > 5 ? atol(argv[5]) : 512;
	long cap = live0 + allocs + 16;
	void **a = mmap(0, cap * sizeof(void *), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	long n = 0;
	for(; n < live0; ++n) {
		size_t s = mn + rnd() % (mx - mn);
		a[n] = malloc(s); ((char *)a[n])[0] = 1;
	}
	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	for(long i = 0; i < allocs; ++i) {
		size_t s = mn + rnd() % (mx - mn);
		void *p = malloc(s); ((char *)p)[0] = 1;
		a[n++] = p;
		if(i % every == every - 1) {
			long k = (long)(rnd() % (unsigned long long)n);
			free(a[k]); a[k] = a[--n];
		}
	}
	clock_gettime(CLOCK_MONOTONIC, &t1);
	double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
	struct rusage ru; getrusage(RUSAGE_SELF, &ru);
	char chunks[64] = "";
	void (*gs)(st_t *) = (void (*)(st_t *))dlsym(RTLD_DEFAULT, "kame_pool_get_stats");
	if(gs) { st_t st = {1}; gs(&st); snprintf(chunks, sizeof chunks, " chunks_live=%zu", st.chunks_live); }
	printf("heavy live0=%ld allocs=%ld every=%ld size=[%ld,%ld) rate=%.2fM allocs/s ns_per_alloc=%.1f maxrss=%.0fMiB%s\n",
	       live0, allocs, every, mn, mx, allocs / secs * 1e-6, secs * 1e9 / allocs,
	       ru.ru_maxrss / 1048576.0, chunks);
	return 0;
}
