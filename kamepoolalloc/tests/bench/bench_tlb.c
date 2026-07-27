/* bench_tlb — TLB-reach cost of the pool's transparent-hugepage policy.
 *
 * (§75 / RT_READINESS G6a)  `kame_pool_set_thp_policy(KAME_THP_NEVER)` buys
 * a bounded first-touch fault (4 KiB instead of 2 MiB) and immunity from
 * khugepaged collapses.  It is paid for in TLB reach: a working set that fit
 * in the TLB as 2 MiB pages may not fit as 4 KiB ones.  Nothing else in
 * tests/bench/ can see that — bench_loop keeps ONE block live and
 * bench_rt_wcet measures the allocator, not the application's access to what
 * it allocated — so this exists to put a number on the trade the README has
 * to state.
 *
 * Method: allocate a working set from the pool, lay a random permutation
 * cycle over it one cache line at a time, then chase it.  Every load depends
 * on the previous one, so there is no memory-level parallelism to hide a
 * page-walk behind: the result is (DRAM latency + TLB-walk cost) per hop,
 * and the policy moves only the second term.
 *
 * Usage:
 *   bench_tlb [working_set_MiB=512] [block_KiB=1024] [hops=20000000]
 *
 * The policy is NOT set here — set it from the environment so the same
 * binary measures both arms across processes (which is the only honest way:
 * MADV_NOHUGEPAGE does not split hugepages that already exist):
 *   KAME_POOL_NOHUGEPAGE=1 ./bench_tlb
 *   ./bench_tlb
 *
 * Licensed under Apache-2.0 OR GPL-2.0-or-later, as the rest of the tree.
 *
 * Co-Authored-By: Claude <noreply@anthropic.com>
 */

#include "../../kame_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define LINE 64u                       /* one hop per cache line */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* xorshift64* — deterministic, so both arms chase the same shape. */
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) {
    rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27;
    return rs * 0x2545F4914F6CDD1Dull;
}

int main(int argc, char **argv) {
    size_t ws_mib  = (argc > 1) ? (size_t)atol(argv[1]) : 512;
    size_t blk_kib = (argc > 2) ? (size_t)atol(argv[2]) : 1024;
    long   hops    = (argc > 3) ? atol(argv[3])         : 20000000L;

    const size_t blk    = blk_kib * 1024u;
    const size_t nblk   = (ws_mib * 1024u * 1024u) / blk;
    const size_t nlines = nblk * (blk / LINE);
    if(nblk == 0 || nlines < 2) { fprintf(stderr, "working set too small\n"); return 2; }

    void **blocks = (void **)malloc(nblk * sizeof(void *));
    if( !blocks) { fprintf(stderr, "OOM\n"); return 2; }
    for(size_t i = 0; i < nblk; i++) {
        blocks[i] = kame_pool_malloc(blk);
        if( !blocks[i]) { fprintf(stderr, "pool OOM at block %zu\n", i); return 2; }
        memset(blocks[i], 0, blk);            /* first-touch the whole set */
    }

    /* Address of hop i, spread across blocks so the chain crosses regions. */
    const size_t lines_per_blk = blk / LINE;
    #define HOP_ADDR(i) ((char *)blocks[(i) / lines_per_blk] \
                         + ((i) % lines_per_blk) * LINE)

    /* Build one random cycle with a Sattolo shuffle over the index space,
     * then write the chain.  The index array is libc memory on purpose — it
     * is scaffolding, not part of the measured working set. */
    size_t *perm = (size_t *)malloc(nlines * sizeof(size_t));
    if( !perm) { fprintf(stderr, "OOM (perm)\n"); return 2; }
    for(size_t i = 0; i < nlines; i++) perm[i] = i;
    for(size_t i = nlines - 1; i > 0; i--) {          /* Sattolo -> single cycle */
        size_t j = (size_t)(rnd() % i);
        size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for(size_t i = 0; i < nlines; i++)
        *(void **)HOP_ADDR(perm[i]) = (void *)HOP_ADDR(perm[(i + 1) % nlines]);
    free(perm);

    void *p = HOP_ADDR(0);
    /* Warm the chain once so the measurement is steady-state, not faults. */
    for(size_t i = 0; i < nlines; i++) p = *(void **)p;

    double t0 = now_s();
    for(long i = 0; i < hops; i++) p = *(void **)p;
    double t1 = now_s();

    /* Keep the chase alive for the optimiser. */
    if(p == (void *)1) fprintf(stderr, "unreachable %p\n", p);

    double ns = (t1 - t0) * 1e9 / (double)hops;
    printf("[bench_tlb] ws=%zu MiB block=%zu KiB lines=%zu hops=%ld "
           "thp_policy=%d  %.2f ns/hop\n",
           ws_mib, blk_kib, nlines, hops, kame_pool_get_thp_policy(), ns);

    for(size_t i = 0; i < nblk; i++) kame_pool_free(blocks[i]);
    free(blocks);
    return 0;
}
