// freelat: the latency of every free while one thread frees MB megabytes of
// SIZE-byte blocks it allocated, in allocation order or, with SHUFFLE set in
// the environment, in a random order -- the case where a free that returns
// slots freed long ago pays a dependent miss for each.  RT set: the frees run
// with kame_pool_set_realtime_thread(1), if the preloaded library has it.
// usage: freelat SIZE MB   -> mean, p99.99, max, and the frees over 1 / 10 / 100 us
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#if defined(__APPLE__)
#include <mach/mach_time.h>
static uint64_t ticks(void) { return mach_absolute_time(); }
static double ns_per_tick(void) {
	mach_timebase_info_data_t tb; mach_timebase_info(&tb);
	return (double)tb.numer / tb.denom;
}
#else
#include <time.h>
static uint64_t ticks(void) {
	struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}
static double ns_per_tick(void) { return 1.0; }
#endif
static int cmp(const void *a, const void *b) {
	const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
	return x < y ? -1 : x > y;
}
int main(int argc, char **argv) {
	if(argc < 3) { fprintf(stderr, "usage: freelat SIZE MB\n"); return 2; }
	size_t sz = strtoul(argv[1], 0, 10), mb = strtoul(argv[2], 0, 10);
	size_t n = mb * 1048576 / sz;
	void **v = malloc(n * sizeof(void *));
	uint64_t *d = malloc(n * sizeof(uint64_t));
	for(size_t i = 0; i < n; ++i) { v[i] = malloc(sz); memset(v[i], 1, sz); }
	if(getenv("SHUFFLE")) {
		uint64_t r = 88172645463325252ull;
		for(size_t i = n; i > 1; --i) {
			r ^= r << 13; r ^= r >> 7; r ^= r << 17;
			size_t j = (size_t)(r % i); void *t = v[i - 1]; v[i - 1] = v[j]; v[j] = t;
		}
	}
	if(getenv("RT")) {
		void (*f)(int) = (void (*)(int))dlsym(RTLD_DEFAULT, "kame_pool_set_realtime_thread");
		if( !f) { fprintf(stderr, "freelat: no kame_pool_set_realtime_thread\n"); return 1; }
		f(1);
	}
	const double k = ns_per_tick();
	for(size_t i = 0; i < n; ++i) {
		const uint64_t t0 = ticks();
		free(v[i]);
		d[i] = (uint64_t)((ticks() - t0) * k);
	}
	size_t over1 = 0, over10 = 0, over100 = 0;
	double total = 0;
	for(size_t i = 0; i < n; ++i) {
		total += (double)d[i];
		over1 += d[i] > 1000; over10 += d[i] > 10000; over100 += d[i] > 100000;
	}
	qsort(d, n, sizeof *d, cmp);
	printf("%s%s %zu B x %zu: mean %.1f ns, p99.99 %.2f us, max %.1f us, "
	       "frees over 1 us %zu, 10 us %zu, 100 us %zu\n",
	       getenv("SHUFFLE") ? "random" : "in order", getenv("RT") ? " RT" : "", sz, n,
	       total / n, d[(size_t)(n * 0.9999)] / 1000.0, d[n - 1] / 1000.0, over1, over10, over100);
	free(d);
	free(v);
	return 0;
}
