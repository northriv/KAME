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

// malloc_intercept_test — verifies the strong-symbol / interpose override
// layer actually routes the bare libc malloc-family (and C++ operator new)
// through the pool.
//
// Everything else in this directory drives the pool via the explicit
// `kame_pool_*` C API or `new`; nothing asserted that a *plain* `malloc()`
// — the call libstdc++ / Qt / Ruby / Python make internally — lands in the
// pool.  That interception is the whole premise of the LD_PRELOAD /
// DYLD_INSERT_LIBRARIES drop-in contract and the bench/README head-to-head
// numbers, yet it was previously exercised only by benchmarks (no
// correctness gate).  This test fills that gap.
//
// Oracle: `kame_pool_malloc_usable_size(p)` returns the true pool capacity
// for a pool pointer and 0 for a foreign one (documented in kame_pool.h).
// So `usable_size(malloc(N)) >= N` proves the malloc call was intercepted.
//
// Platform matrix (allocator.cpp §32 + the §31 Windows block):
//   - Linux glibc : strong-symbol malloc/free/calloc/realloc always emitted
//                   → plain malloc IS intercepted when the lib is linked.
//   - macOS dylib : __DATA,__interpose, FULL by default → intercepted.
//   - musl / other: NO strong-symbol malloc → plain malloc is NOT pooled
//                   (only operator new and the kame_pool_* API are).
//   - Windows     : USE_KAME_ALLOCATOR is OFF in this CMake, so the test is
//                   not built there at all.
// The C-malloc assertions are therefore made conditional on a runtime probe
// (`g_malloc_intercepted`); the operator-new and C-API assertions run
// unconditionally (those routes are active wherever the pool is linked).
//
// Co-Authored-By: Claude <noreply@anthropic.com>

#include "kame_pool.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <new>

static int g_fails = 0;
#define CHECK(c, ...) do { if(!(c)) { ++g_fails; \
    std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while(0)

// A pointer guaranteed NOT to come from the pool, to validate the oracle's
// negative case (foreign → usable_size == 0).
static char g_static_buf[256];

int main() {
    // --- (0) Oracle sanity: foreign pointers report usable_size 0. --------
    int stackvar = 0;
    CHECK(kame_pool_malloc_usable_size(&stackvar) == 0,
          "oracle broken: stack pointer reported as pool-owned");
    CHECK(kame_pool_malloc_usable_size(g_static_buf) == 0,
          "oracle broken: static buffer reported as pool-owned");
    CHECK(kame_pool_malloc_usable_size(nullptr) == 0,
          "oracle broken: NULL reported as pool-owned");

    // --- (1) Runtime probe: is plain malloc intercepted on this build? ----
    // (Linux glibc / macOS: yes.  musl: no.)  We branch the C-malloc-family
    // assertions on this so the test is meaningful everywhere it compiles.
    void *probe = malloc(64);
    CHECK(probe != nullptr, "malloc(64) returned NULL");
    bool g_malloc_intercepted =
        probe && kame_pool_malloc_usable_size(probe) >= 64;
    free(probe);

    if(g_malloc_intercepted)
        std::printf("  [info] plain malloc IS intercepted (strong-symbol/interpose)\n");
    else
        std::printf("  [info] plain malloc NOT intercepted on this platform "
                    "(musl-style); testing operator new + C API only\n");

    // --- (2) C malloc-family interception (conditional). ------------------
    if(g_malloc_intercepted) {
        // malloc
        {
            const size_t N = 12345;
            void *p = malloc(N);
            CHECK(p != nullptr, "malloc(%zu) returned NULL", N);
            size_t us = kame_pool_malloc_usable_size(p);
            CHECK(us >= N, "malloc(%zu): usable_size=%zu < requested "
                  "(not pool-routed)", N, us);
            // Writable across the whole reported capacity.
            std::memset(p, 0xAB, us ? us : N);
            free(p);
        }
        // calloc — pool-routed AND zero-initialised.
        {
            const size_t NMEMB = 1000, SZ = 8;
            unsigned char *p =
                static_cast<unsigned char *>(calloc(NMEMB, SZ));
            CHECK(p != nullptr, "calloc returned NULL");
            CHECK(kame_pool_malloc_usable_size(p) >= NMEMB * SZ,
                  "calloc: not pool-routed");
            bool allzero = true;
            for(size_t i = 0; i < NMEMB * SZ; i++)
                if(p[i] != 0) { allzero = false; break; }
            CHECK(allzero, "calloc did not zero-initialise");
            free(p);
        }
        // realloc — grow keeps pool routing and preserves bytes.
        {
            const size_t N0 = 100, N1 = 4000;
            unsigned char *p = static_cast<unsigned char *>(malloc(N0));
            CHECK(p != nullptr, "malloc for realloc returned NULL");
            for(size_t i = 0; i < N0; i++) p[i] = (unsigned char)(i & 0xFF);
            unsigned char *q = static_cast<unsigned char *>(realloc(p, N1));
            CHECK(q != nullptr, "realloc returned NULL");
            CHECK(kame_pool_malloc_usable_size(q) >= N1,
                  "realloc: grown block not pool-routed");
            bool preserved = true;
            for(size_t i = 0; i < N0; i++)
                if(q[i] != (unsigned char)(i & 0xFF)) { preserved = false; break; }
            CHECK(preserved, "realloc did not preserve original bytes");
            free(q);
        }
        // realloc(NULL, n) == malloc(n); realloc(p, 0) frees.
        {
            void *p = realloc(nullptr, 512);
            CHECK(p != nullptr && kame_pool_malloc_usable_size(p) >= 512,
                  "realloc(NULL,512) did not behave as pool malloc");
            void *r = realloc(p, 0);
            CHECK(r == nullptr, "realloc(p,0) should return NULL");
        }
        // A large size that escapes the bucket tier (dedicated/large_va) —
        // still must be pool-routed.
        {
            const size_t N = 2u << 20;   // 2 MiB → dedicated chunk tier
            void *p = malloc(N);
            CHECK(p != nullptr, "malloc(2MiB) returned NULL");
            CHECK(kame_pool_malloc_usable_size(p) >= N,
                  "large malloc not pool-routed (dedicated/large tier)");
            free(p);
        }
    }

    // --- (3) C++ operator new / delete interception (unconditional). ------
    // operator new is overridden in EVERY config this test is built in: the
    // strong-symbol override (Linux / macOS), the inline-compiled exe path
    // (MinGW monorepo), and the live MSVC pool (`_MSC_VER` shim in
    // allocator_prv.h).  So unlike plain malloc — whose libc-symbol takeover
    // is platform-conditional — operator new is asserted hard: a regression
    // in the new/delete override must fail this test, not silently skip.
    {
        int *q = new int[2000];
        CHECK(kame_pool_malloc_usable_size(q) >= 2000 * sizeof(int),
              "operator new[] not pool-routed");
        for(int i = 0; i < 2000; i++) q[i] = i;
        CHECK(q[1999] == 1999, "new[] memory not writable/consistent");
        delete[] q;

        // Single-object new and nothrow new.
        double *d = new double(3.14);
        CHECK(kame_pool_malloc_usable_size(d) >= sizeof(double),
              "operator new not pool-routed");
        delete d;

        void *n = ::operator new(777, std::nothrow);
        CHECK(n != nullptr && kame_pool_malloc_usable_size(n) >= 777,
              "nothrow operator new not pool-routed");
        ::operator delete(n, std::nothrow);
    }

    // --- (4) Cross-route free reconciliation. -----------------------------
    // The pool's free/delete dispatch must accept a pool pointer regardless
    // of which front door allocated it.  Free a new[]'d pointer via free()
    // (the §31/interpose reconciliation case that production relies on when
    // Qt/CRT hands a pool pointer to libc free).  Must not crash; pointer
    // must be recognised as pool-owned first.
    if(g_malloc_intercepted) {
        char *fromnew = new char[321];
        CHECK(kame_pool_malloc_usable_size(fromnew) >= 321,
              "new[] pointer not pool-owned before cross-free");
        free(fromnew);            // free() reconciles a new[]'d pool pointer
    }

    // --- (5) The kame_pool_* C API is always-on regardless of override. ---
    {
        void *p = kame_pool_malloc(4096);
        CHECK(p != nullptr && kame_pool_malloc_usable_size(p) >= 4096,
              "kame_pool_malloc direct API not pool-routed");
        kame_pool_free(p);

        int rc;
        void *ap = nullptr;
        rc = kame_pool_posix_memalign(&ap, 64, 1000);
        CHECK(rc == 0 && ap != nullptr, "kame_pool_posix_memalign failed");
        CHECK(((uintptr_t)ap & 63u) == 0, "posix_memalign result not aligned");
        CHECK(kame_pool_malloc_usable_size(ap) >= 1000,
              "posix_memalign result not pool-routed");
        kame_pool_free(ap);
    }

    std::printf(g_fails == 0 ? "\nPASS\n" : "\nFAIL (%d)\n", g_fails);
    return g_fails ? 1 : 0;
}
