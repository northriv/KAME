#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static uint64_t x = 88172645463325252ull;
static inline uint64_t rnd(void) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; }
__attribute__((noinline)) void *chase(void **p, long n) {
    for (long i = 0; i < n; i++) p = (void **)*p;
    return p;
}
int main(void) {
    size_t n = (size_t)1 << 25;                 /* 256 MiB of pointers */
    void **a = malloc(n * sizeof(void *));
    size_t *perm = malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) perm[i] = i;
    for (size_t i = n - 1; i > 0; i--) { size_t j = rnd() % (i + 1); size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    for (size_t i = 0; i < n; i++) a[perm[i]] = &a[perm[(i + 1) % n]];
    free(perm);
    void *r = chase((void **)a[0], 60000000L);
    printf("%p\n", r);
    return 0;
}
