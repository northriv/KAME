// SPSC ring: a producer mallocs SIZE-byte blocks into a ring of S slots, a
// consumer frees them -- every free cross-thread.  M ops/s (alloc+free pairs).
// usage: ring SIZE S MILLION_OPS (S a power of two)
#include <stdatomic.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static size_t SIZE; static unsigned S; static long N;
static void **ring; static _Atomic unsigned long head, tail;
static void *consumer(void *a) {
	(void)a;
	unsigned long t = 0;
	for(long i = 0; i < N; ++i) {
		while(atomic_load_explicit(&head, memory_order_acquire) == t) ;
		void *p = ring[t & (S - 1)];
		atomic_store_explicit(&tail, ++t, memory_order_release);
		free(p);
	}
	return 0;
}
int main(int argc, char **argv) {
	SIZE = argc > 1 ? strtoul(argv[1], 0, 10) : 64;
	S = argc > 2 ? (unsigned)strtoul(argv[2], 0, 10) : 32;
	N = (argc > 3 ? atol(argv[3]) : 20) * 1000000L;
	ring = calloc(S, sizeof(void *));
	pthread_t th; pthread_create(&th, 0, consumer, 0);
	struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
	unsigned long h = 0;
	for(long i = 0; i < N; ++i) {
		char *p = malloc(SIZE);
		p[0] = (char)i;
		while(h - atomic_load_explicit(&tail, memory_order_acquire) >= S) ;
		ring[h & (S - 1)] = p;
		atomic_store_explicit(&head, ++h, memory_order_release);
	}
	pthread_join(th, 0);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
	printf("ring size=%zu S=%u rate=%.1fM\n", SIZE, S, N / s / 1e6);
	return 0;
}
