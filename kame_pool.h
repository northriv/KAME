/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
        implied.  See the License for the specific language governing
        permissions and limitations under the License.

        SPDX-License-Identifier: Apache-2.0
***************************************************************************/

/*
 * kame_pool.h — pure-C interface to the kamepoolalloc pool allocator.
 *
 * Two ways to use this allocator:
 *
 *   (1) Process-wide drop-in malloc replacement.  Link against
 *       libkamepoolalloc and the dylib's `__DATA,__interpose`
 *       section (macOS) / strong-symbol `free`/`realloc` (Linux glibc)
 *       intercept every libc malloc/free/realloc call automatically.
 *       Nothing else to do — your existing `malloc()` / `free()` /
 *       `realloc()` calls are now pool-backed where appropriate.
 *
 *   (2) Explicit pool API.  Include <kame_pool.h>, call
 *       `kame_pool_malloc()` / `kame_pool_free()` / etc. directly.
 *       Useful when:
 *         - you cannot LD_PRELOAD / interpose (static link, restricted
 *           runtime, sandbox);
 *         - you want a specific subset of code to use the pool while
 *           the rest stays on libc malloc;
 *         - you are writing C/Rust/Go bindings that need a stable C ABI.
 *
 * Path (2) does NOT require the C++ side `activateAllocator()` to have
 * been called.  Pre-activation calls (or post-thread-teardown calls)
 * transparently fall through to libsystem `malloc()` / `free()`, so the
 * functions are safe to call from any context — including library
 * static initialisers, signal-safe paths (after a longjmp, etc.), and
 * thread destructors.  The price is that very-early-init allocations
 * never see the pool's fast path; explicitly call `activateAllocator()`
 * (C++ side, from <allocator.h>) or instantiate `KamePooledAllocGuard`
 * to enable the pool path.
 *
 * Thread-safety: every function below is fully reentrant and safe to
 * call concurrently from any thread.  The underlying pool uses
 * lock-free per-thread bucketed allocation; the C wrappers add only a
 * trivial null check and forwarding call.
 *
 * Error reporting follows libc malloc conventions:
 *   - allocation failure returns NULL and sets `errno = ENOMEM`,
 *   - posix_memalign returns the error code as its return value (no
 *     `errno` set), per POSIX,
 *   - all other functions never fail in ways requiring errno.
 *
 * Alignment: the pool guarantees `max_align_t` alignment (16 B on every
 * supported arch) for all sizes through the standard malloc/realloc
 * path.  Use `kame_pool_aligned_alloc()` / `kame_pool_posix_memalign()`
 * for stricter alignment; alignments up to 16 B are served by the pool,
 * larger alignments fall back to the system `posix_memalign()`.
 */

#ifndef KAMEPOOLALLOC_KAME_POOL_H_
#define KAMEPOOLALLOC_KAME_POOL_H_

#include <stddef.h>  /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Standard malloc/free family.  Identical semantics to ISO C
 * stdlib counterparts; only the backing implementation differs.
 *
 *   kame_pool_malloc(0) returns a unique freeable pointer (one of two
 *     standard-compliant choices).
 *   kame_pool_calloc(n, sz) returns NULL with errno=ENOMEM on overflow.
 *   kame_pool_realloc(NULL, n) == kame_pool_malloc(n).
 *   kame_pool_realloc(p, 0)    frees p and returns NULL.
 *   kame_pool_free(NULL) is a no-op.
 */
void  *kame_pool_malloc(size_t size);
void  *kame_pool_calloc(size_t nmemb, size_t size);
void  *kame_pool_realloc(void *p, size_t size);
void   kame_pool_free(void *p);

/*
 * Aligned allocation.  `alignment` must be a power of two; for
 * kame_pool_posix_memalign() it must additionally be a multiple of
 * sizeof(void*).  Alignments <= 16 B are served by the pool;
 * larger alignments fall back to the system allocator.
 *
 * Windows restriction: alignments > 16 B return EINVAL.  The reason is
 * the platform `_aligned_malloc` / `_aligned_free` pairing — pointers
 * from `_aligned_malloc` cannot be passed to CRT `free()`, and
 * `kame_pool_free()` does not carry alignment info to dispatch
 * `_aligned_free` correctly.  Callers needing alignment > 16 B on
 * Windows should use `_aligned_malloc` / `_aligned_free` directly, or
 * use C++ `operator new(size, std::align_val_t{N})` which carries
 * alignment into the matching `operator delete`.
 */
void  *kame_pool_aligned_alloc(size_t alignment, size_t size);
int    kame_pool_posix_memalign(void **memptr, size_t alignment, size_t size);

/*
 * Returns the actual allocated size of `p` (which may exceed the
 * size requested at allocation, due to bucket rounding).  Returns 0
 * for foreign pointers (those not allocated through this pool).
 * Safe on NULL.
 *
 * Use case: realloc-elision in client code that maintains its own
 * vector-growth heuristic; pass the actual size to vector::reserve()
 * to avoid spurious reallocations.
 */
size_t kame_pool_malloc_usable_size(const void *p);

/*
 * Runtime memory cap.  Same semantics as the C++ declarations in
 * allocator.h, but with C-linkage so pure-C consumers can call them.
 *
 *   kame_pool_set_max_bytes(0)       — disable cap (default).
 *   kame_pool_set_max_bytes(N)       — round N up to a multiple of
 *                                      32 MiB, refuse mmap of fresh
 *                                      regions once that limit is hit.
 *   kame_pool_get_max_bytes()        — current cap (`SIZE_MAX` if unset).
 *   kame_pool_reserved_bytes()       — total bytes the pool has mmap'd.
 */
void   kame_pool_set_max_bytes(size_t max_bytes);
size_t kame_pool_get_max_bytes(void);
size_t kame_pool_reserved_bytes(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* KAMEPOOLALLOC_KAME_POOL_H_ */
