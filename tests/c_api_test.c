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
 * c_api_test.c — smoke test for the kamepoolalloc C ABI.
 *
 * Compiles as pure C (no C++ headers).  Exercises every function in
 * <kame_pool.h>: alloc / calloc / realloc / free family, aligned
 * variants, usable_size introspection, and the runtime cap controls.
 *
 * Failure mode: prints "FAIL <reason>" to stderr and returns non-zero.
 * Success: prints "PASS" and returns 0.
 *
 * Note: pre-activation safety means these calls may transparently
 * route to libsystem malloc/free.  We don't assume the pool is active;
 * we only require correctness of the public API contract.
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
            fprintf(stderr, "FAIL [%s:%d] %s\n",                   \
                    __FILE__, __LINE__, msg);                      \
            ++failures;                                            \
        }                                                          \
    } while(0)

static void test_malloc_free(void) {
    void *p = kame_pool_malloc(64);
    EXPECT(p != NULL, "malloc(64) returned NULL");
    if(p) {
        /* Write + read to confirm the memory is usable */
        memset(p, 0xab, 64);
        EXPECT(((unsigned char *)p)[0] == 0xab, "memset[0] readback");
        EXPECT(((unsigned char *)p)[63] == 0xab, "memset[63] readback");
        kame_pool_free(p);
    }
    /* free(NULL) must be a no-op */
    kame_pool_free(NULL);
}

static void test_calloc(void) {
    /* calloc(8, 16) = 128 bytes, all zero */
    void *p = kame_pool_calloc(8, 16);
    EXPECT(p != NULL, "calloc(8,16) returned NULL");
    if(p) {
        int all_zero = 1;
        for(size_t i = 0; i < 128; ++i)
            if(((unsigned char *)p)[i] != 0) { all_zero = 0; break; }
        EXPECT(all_zero, "calloc memory not zero-initialised");
        kame_pool_free(p);
    }
    /* calloc overflow: SIZE_MAX * 2 wraps */
    errno = 0;
    void *q = kame_pool_calloc(SIZE_MAX, 2);
    EXPECT(q == NULL, "calloc(SIZE_MAX,2) did not return NULL on overflow");
}

static void test_realloc(void) {
    /* realloc(NULL, n) == malloc(n) */
    void *p = kame_pool_realloc(NULL, 32);
    EXPECT(p != NULL, "realloc(NULL,32) returned NULL");
    if(p) {
        memset(p, 0xcd, 32);
        /* Grow */
        void *q = kame_pool_realloc(p, 256);
        EXPECT(q != NULL, "realloc grow returned NULL");
        if(q) {
            /* Original bytes preserved */
            EXPECT(((unsigned char *)q)[0] == 0xcd, "realloc preserve[0]");
            EXPECT(((unsigned char *)q)[31] == 0xcd, "realloc preserve[31]");
            /* Shrink */
            void *r = kame_pool_realloc(q, 16);
            EXPECT(r != NULL, "realloc shrink returned NULL");
            if(r) {
                EXPECT(((unsigned char *)r)[0] == 0xcd, "realloc shrink preserve");
                /* realloc(p, 0) frees and returns NULL */
                void *t = kame_pool_realloc(r, 0);
                EXPECT(t == NULL, "realloc(p,0) did not return NULL");
            }
        }
    }
}

static void test_aligned(void) {
    /* 16-byte alignment (pool guarantee) */
    void *p = kame_pool_aligned_alloc(16, 128);
    EXPECT(p != NULL, "aligned_alloc(16,128) returned NULL");
    EXPECT(((uintptr_t)p & 15u) == 0u, "aligned_alloc(16) not 16-aligned");
    kame_pool_free(p);

    /* 8-byte alignment */
    p = kame_pool_aligned_alloc(8, 64);
    EXPECT(p != NULL, "aligned_alloc(8,64) returned NULL");
    EXPECT(((uintptr_t)p & 7u) == 0u, "aligned_alloc(8) not 8-aligned");
    kame_pool_free(p);

    /* 64-byte alignment — over-aligned, falls back to posix_memalign.
     * Windows-restricted: the C API does not support alignment > 16 B
     * (see kame_pool.h for rationale).  Skip the assertion on Windows. */
#if !(defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS))
    p = kame_pool_aligned_alloc(64, 256);
    EXPECT(p != NULL, "aligned_alloc(64,256) returned NULL");
    EXPECT(((uintptr_t)p & 63u) == 0u, "aligned_alloc(64) not 64-aligned");
    kame_pool_free(p);
#endif

    /* Invalid alignment (not power of two) */
    errno = 0;
    void *q = kame_pool_aligned_alloc(7, 64);
    EXPECT(q == NULL, "aligned_alloc(7,*) accepted non-power-of-2");
    EXPECT(errno == EINVAL, "aligned_alloc EINVAL not set");
}

static void test_posix_memalign(void) {
    void *p = NULL;
    int rc = kame_pool_posix_memalign(&p, 16, 128);
    EXPECT(rc == 0, "posix_memalign(16,128) failed");
    EXPECT(p != NULL, "posix_memalign(16,128) gave NULL");
    EXPECT(((uintptr_t)p & 15u) == 0u, "posix_memalign(16) not aligned");
    kame_pool_free(p);

    /* Over-aligned: falls back */
    p = NULL;
    rc = kame_pool_posix_memalign(&p, 128, 512);
    EXPECT(rc == 0, "posix_memalign(128,512) failed");
    EXPECT(p != NULL, "posix_memalign(128,512) gave NULL");
    EXPECT(((uintptr_t)p & 127u) == 0u, "posix_memalign(128) not aligned");
    kame_pool_free(p);

    /* Invalid alignment (< sizeof(void*)).  Use 2 — it is < sizeof(void*)
     * on both 32-bit (sizeof(void*)==4) and 64-bit (==8); 4 would be a
     * valid alignment on ILP32 and produce a spurious failure there. */
    p = NULL;
    rc = kame_pool_posix_memalign(&p, 2, 64);
    EXPECT(rc == EINVAL, "posix_memalign(2,*) did not return EINVAL");

    /* NULL memptr */
    rc = kame_pool_posix_memalign(NULL, 16, 64);
    EXPECT(rc == EINVAL, "posix_memalign(NULL,*) did not return EINVAL");
}

static void test_usable_size(void) {
    /* usable_size on NULL = 0 */
    EXPECT(kame_pool_malloc_usable_size(NULL) == 0,
           "usable_size(NULL) != 0");

    void *p = kame_pool_malloc(40);
    EXPECT(p != NULL, "malloc(40) for usable_size returned NULL");
    if(p) {
        size_t us = kame_pool_malloc_usable_size(p);
        /* Pool bucket rounds 40 up to 48 (16-byte aligned bucket).  We
         * verify only the lower bound — the pool's bucket schedule may
         * change.  If the pointer routes to libsystem (e.g. very early
         * init before activator), usable_size returns 0; accept that.
         */
        EXPECT(us == 0 || us >= 40, "usable_size lower bound");
        kame_pool_free(p);
    }
}

static void test_cap(void) {
    /* Save current cap */
    size_t orig = kame_pool_get_max_bytes();

    /* Set to 256 MiB and read back (rounded up to 32 MiB granularity) */
    kame_pool_set_max_bytes(256u * 1024u * 1024u);
    size_t got = kame_pool_get_max_bytes();
    EXPECT(got == 256u * 1024u * 1024u,
           "set_max_bytes(256MiB) did not roundtrip");

    /* reserved_bytes is monotonic, never exceeds the cap */
    size_t reserved = kame_pool_reserved_bytes();
    EXPECT(reserved <= got, "reserved_bytes > cap");

    /* Restore */
    if(orig == SIZE_MAX) kame_pool_set_max_bytes(0);
    else kame_pool_set_max_bytes(orig);
}

int main(void) {
    test_malloc_free();
    test_calloc();
    test_realloc();
    test_aligned();
    test_posix_memalign();
    test_usable_size();
    test_cap();

    if(failures == 0) {
        printf("PASS (c_api_test: all checks)\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d check(s) failed\n", failures);
    return 1;
}
