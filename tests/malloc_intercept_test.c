/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            kamepoolalloc/LICENSE-APACHE-2.0)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (see kamepoolalloc/LICENSE-GPL-2.0).

        Pick whichever license suits your project.
***************************************************************************/

/*
 * malloc_intercept_test.c — validate that the §31 IAT redirect on Windows
 * (KAMEPOOLALLOC_FULL_INTERCEPT, default-ON) routes plain stdlib
 * malloc / calloc / realloc / free calls through the pool.
 *
 * On macOS (KAMEPOOLALLOC_DYLIB built as MH_DYLIB): __interpose replaces
 * malloc/free/realloc transparently — same probe exercises that path.
 * On Linux: strong-symbol override takes the same path via link order.
 *
 * Probe: kame_pool_malloc_usable_size(p) returns 0 for foreign pointers
 * and >= the requested size for pool-owned pointers.  If intercept works,
 * every malloc/calloc/realloc result is pool-owned (usable_size >= size).
 *
 * Build: link alongside allocator.cpp with KAMEPOOLALLOC_DYLIB so the
 * auto-activator fires (installs the IAT redirect) before main() is entered.
 */

#include "kame_pool.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define EXPECT(cond, msg)                                          \
    do {                                                           \
        if(!(cond)) {                                              \
            fprintf(stderr, "FAIL [%s:%d] %s\n",                  \
                    __FILE__, __LINE__, msg);                      \
            ++failures;                                            \
        }                                                          \
    } while(0)

/* ---------- test helpers ---------- */

static void test_malloc_intercepted(void) {
    void *p = malloc(64);
    EXPECT(p != NULL, "malloc(64) returned NULL");
    if(p) {
        /* kame_pool_malloc_usable_size returns 0 for foreign pointers
         * and >= requested_size for pool-owned pointers. */
        size_t us = kame_pool_malloc_usable_size(p);
        EXPECT(us >= 64,
               "malloc(64) NOT intercepted — usable_size==0 (foreign pointer)");
        memset(p, 0xaa, 64);
        EXPECT(((unsigned char *)p)[0]  == 0xaa, "malloc: mem readback [0]");
        EXPECT(((unsigned char *)p)[63] == 0xaa, "malloc: mem readback [63]");
        free(p);
    }
}

static void test_calloc_intercepted(void) {
    void *p = calloc(4, 32); /* 128 bytes, should be zero */
    EXPECT(p != NULL, "calloc(4,32) returned NULL");
    if(p) {
        size_t us = kame_pool_malloc_usable_size(p);
        EXPECT(us >= 128,
               "calloc(4,32) NOT intercepted — usable_size==0");
        /* calloc guarantee: zero-filled */
        int all_zero = 1;
        for(int i = 0; i < 128; ++i)
            if(((unsigned char *)p)[i]) { all_zero = 0; break; }
        EXPECT(all_zero, "calloc: memory not zero-initialised");
        free(p);
    }
}

static void test_realloc_intercepted(void) {
    void *p = malloc(32);
    EXPECT(p != NULL, "malloc(32) for realloc test returned NULL");
    if(p) {
        memset(p, 0xbb, 32);
        /* grow: 32 -> 128 */
        void *q = realloc(p, 128);
        EXPECT(q != NULL, "realloc(p,128) returned NULL");
        if(q) {
            size_t us = kame_pool_malloc_usable_size(q);
            EXPECT(us >= 128,
                   "realloc(p,128) result NOT intercepted — usable_size==0");
            /* original bytes must be preserved */
            EXPECT(((unsigned char *)q)[0]  == 0xbb, "realloc: preserve[0]");
            EXPECT(((unsigned char *)q)[31] == 0xbb, "realloc: preserve[31]");
            free(q);
        }
    }
}

static void test_free_null(void) {
    /* free(NULL) must be a no-op regardless of intercept state */
    free(NULL);
    /* reaching here == no crash */
    EXPECT(1, "free(NULL) survived");
}

static void test_repeated_malloc_free(void) {
    /* Tight loop: confirm pool handles repeated alloc/free via the intercept. */
    enum { N = 200 };
    void *ptrs[N];
    for(int i = 0; i < N; ++i) {
        ptrs[i] = malloc((size_t)(i + 1) * 8);
        EXPECT(ptrs[i] != NULL, "repeated malloc returned NULL");
        if(ptrs[i]) {
            size_t us = kame_pool_malloc_usable_size(ptrs[i]);
            EXPECT(us >= (size_t)(i + 1) * 8,
                   "repeated malloc NOT intercepted");
        }
    }
    for(int i = 0; i < N; ++i)
        free(ptrs[i]);
}

/* ---------- main ---------- */

int main(void) {
    test_malloc_intercepted();
    test_calloc_intercepted();
    test_realloc_intercepted();
    test_free_null();
    test_repeated_malloc_free();

    if(failures == 0) {
        printf("PASS (malloc_intercept_test: stdlib malloc/calloc/realloc/free are pool-backed)\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d check(s) failed\n", failures);
    return 1;
}
