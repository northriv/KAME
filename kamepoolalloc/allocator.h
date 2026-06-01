/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            LICENSE-APACHE-2.0 in this directory)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see LICENSE-GPL-2.0 in this directory).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
***************************************************************************/

// =====================================================================
//                       *** READ THIS FIRST ***
//
//  The KAME pool allocator is INACTIVE BY DEFAULT.  Until you call
//  `activateAllocator()` (or instantiate `KamePooledAllocGuard` in your
//  `main()`), every `operator new` / `operator delete` falls through to
//  `std::malloc` / `std::free`.  This is intentional — dyld, static
//  constructors, libc++ ICU/Foundation, and anything that runs before
//  `main()` must use the system allocator (we cannot accept pool
//  pointers before our mmap regions exist).
//
//  --- For application code ---
//
//    int main(int argc, char **argv) {
//        KamePooledAllocGuard pool_guard;       // <-- activates here
//        // ... rest of main() ...
//    }
//
//  See `KamePooledAllocGuard` at the bottom of this file for the
//  rationale on why the guard's destructor does NOT tear down pools.
//
//  --- For test binaries / benchmarks ---
//
//  If you build a custom standalone TU that links `kame/allocator.cpp`
//  directly, ALSO link `tests/allocator.cpp` (or copy its
//  `KamePoolActivator` static-init wrapper) so the pool is activated
//  during dyld image load.  Without it, your "KAME bench" is silently
//  measuring `std::malloc`.  Symptom: profile shows samples in
//  `_xzm_xzone_malloc_*` instead of `PoolAllocator<...>::deallocate_*`.
//
//  --- Sanity check ---
//
//  When the pool is active, the first allocation prints
//      "Reserve swap space starting @ 0x... w/ len. of 0x......B."
//  to stderr (from `allocate_chunk` after the first mmap).  No such
//  message ⇒ pool is inactive ⇒ check your activator wiring.
//
// =====================================================================

#ifndef ALLOCATOR_H_
#define ALLOCATOR_H_

// Arches for which the lock-free pool allocator is enabled.
//   x86 / x86_64 — original target, inline-asm path.
//   ARM64 (Apple Silicon, Linux aarch64) — uses __builtin_ctzll for
//      bit-scan and the ARM8 dmb/yield barriers from atomic_prv_mfence_arm8.h.
//   Windows (x86_64 / ARM64) — MinGW-only.  MinGW provides
//      `__sync_*` atomics, `__attribute__((constructor))`, and a
//      `thread_local` runtime that drives `AllocThreadExitCleanup`'s
//      destructor at thread exit.  MSVC support requires an
//      `_Interlocked*` atomic shim and `__declspec(allocate(".CRT$XCB"))`
//      for the static-init hook — out of scope for an earlier change.
//      Production kame.exe inline-compiles `allocator.cpp` (no DLL
//      boundary), so the strong-symbol `free`/`realloc` interpose
//      (Linux-glibc strategy) is NOT used on Windows — `operator new`
//      / `operator delete` overrides plus the explicit `kame_pool_*`
//      C API cover every legitimate code path.  CRT `free` /
//      `realloc` stay bound to msvcrt, which is what 3rd-party DLLs
//      expect.
// Anything else falls back to std::allocator via USE_STD_ALLOCATOR.
#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__\
    || defined __x86_64__ || defined _M_IX86 || defined _M_X64\
    || defined __arm64__ || defined __aarch64__ || defined _M_ARM64\
    || defined __WIN32__ || defined WINDOWS || defined _WIN32
#else
    #define USE_STD_ALLOCATOR
#endif

// MSVC carve-out: the pool's atomic primitives are GCC __sync builtins
// and the constructor hook uses `__attribute__((constructor))`.  Both
// are MinGW-supported on Windows but MSVC requires intrinsics-based
// replacements.  Until that shim lands, MSVC-built Windows binaries
// stay on USE_STD_ALLOCATOR.  MinGW continues onto the active path.
#if (defined(_WIN32) || defined(WINDOWS)) && defined(_MSC_VER) && !defined(__GNUC__) && !defined(KAME_ENABLE_POOL_MSVC)
    // Define KAME_ENABLE_POOL_MSVC to opt the MSVC build INTO the live pool
    // (WIP: requires the _Interlocked* / __declspec MSVC shims).  Without
    // it, MSVC stays on std::allocator as before.
    #define USE_STD_ALLOCATOR
#endif

#if defined USE_STD_ALLOCATOR
    #include <cstddef>   // std::size_t for the pool-API stubs below — on
                         // MSVC <vector>/<limits> aren't pulled in until
                         // far later in this header.
    inline void activateAllocator() {}

    //! \return always true on USE_STD_ALLOCATOR builds — no per-thread
    //! pool state to worry about.
    inline bool is_allocator_thread_active() noexcept { return true; }

    //! pool API stubs for USE_STD_ALLOCATOR builds (Windows
    //! by default).  No pool → cap is meaningless; the functions
    //! exist so consumers can call them unconditionally without
    //! `#ifdef`.  These MUST use C linkage to match the `extern "C"`
    //! declarations in kame_pool.h — otherwise MSVC rejects the later
    //! kame_pool.h declarations with C2732 (linkage-spec mismatch).
    extern "C" {
    inline void kame_pool_set_max_bytes(std::size_t /*max_bytes*/) noexcept {}
    inline std::size_t kame_pool_get_max_bytes() noexcept { return ~std::size_t(0); }
    inline std::size_t kame_pool_reserved_bytes() noexcept { return 0; }
    //! (§30) Realtime-mode toggle stub.  Pool background maintenance
    //! doesn't exist in this build, so silencing it is a no-op.
    inline void kame_pool_set_realtime_mode(int /*enable*/) noexcept {}
    } // extern "C"
#else
    #include "allocator_prv.h"

    //! Fast lock-free allocators for small objects: new(), new[](),
    //! delete(), delete[]() operators.  Memory blocks in a unit of
    //! double-quad word less than 8 KiB can be allocated from
    //! fixed-size or variable-size memory pools.  Larger memory is
    //! provided by standard malloc().  \sa PoolAllocator,
    //! allocator_test.cpp.
    //!
    //! These globals stay non-inline in `allocator.cpp` per C++ §17.6.4.6
    //! (replacement allocation functions must not be `inline`).  An
    //! earlier experiment to header-inline them tripped the
    //! `-Winline-new-delete` warning and produced SIGTRAP on the STM
    //! tests at link time — symptom of the linker resolving some
    //! `delete p` sites to the libcxx default (which calls `free()` on
    //! a KAME pool pointer) instead of to our replacement.  Cross-TU
    //! inlining of the alloc/dealloc fast paths is a job for LTO, not
    //! for header-only replacement operators.

    #if defined(KAMEPOOLALLOC_DYLIB)
        //! Dylib mode (cmake test build): the pool is activated at dylib
        //! load via a `__attribute__((constructor))` inside
        //! libkamepoolalloc itself, before any consumer image's static
        //! init runs.  Consumers therefore never call this — kept as an
        //! inline no-op only so existing `KamePooledAllocGuard`
        //! references compile uniformly across modes.
        inline void activateAllocator() noexcept {}
    #else
        //! Inline-compiled mode (qmake production build): caller must
        //! flip the activation switch explicitly, normally via
        //! `KamePooledAllocGuard` in `main()`.
        extern void activateAllocator();
    #endif

    //! Runtime memory cap.  When the pool has mmap'd at
    //! least this many bytes (counted by region count × 32 MiB region
    //! size), `allocate_chunk` refuses to mmap fresh regions and
    //! returns nullptr from the chunk-claim path — `allocate_pooled`
    //! propagates `0`, and `operator new` falls back to libsystem
    //! `std::malloc()` for the offending allocation.
    //!
    //! Pass `0` to disable the cap (default).  The implementation cap
    //! `ALLOC_MAX_MMAP_ENTRIES × 32 MiB` (= 100 GiB on 64-bit, 3 GiB
    //! on 32-bit) is still enforced — `kame_pool_set_max_bytes(N)`
    //! lowers the effective cap further when N is smaller.
    //!
    //! Granularity: 32 MiB (one mmap region).  N is rounded UP to the
    //! nearest multiple of 32 MiB internally; e.g. setting 100 MiB
    //! actually caps at 128 MiB (4 regions).
    //!
    //! Thread-safety: relaxed atomic store / load.  Safe to call from
    //! any thread at any time; changes take effect for subsequent
    //! `allocate_chunk` calls.  Typical usage: set once early in
    //! `main()` before allocating, then never touch.
    //!
    //! Use case: embedded / RTOS deployments where you want a hard
    //! ceiling on the pool's RSS+VA reservation regardless of
    //! workload.  Combined with `is_allocator_thread_active()`
    //! returning true, allocations beyond the cap still succeed via
    //! libsystem malloc — applications that need a true OOM (no
    //! malloc fallback) should additionally trip on the OOM via
    //! `std::set_new_handler` or by wrapping `operator new`.
    //!
    //! Declared with C linkage so the same symbol is usable from
    //! both C and C++ via `<kame_pool.h>` / `<allocator.h>`.
    extern "C" void kame_pool_set_max_bytes(std::size_t max_bytes) noexcept;

    //! \return current effective cap in bytes (`SIZE_MAX` if unset).
    extern "C" std::size_t kame_pool_get_max_bytes() noexcept;

    //! \return total bytes currently mmap'd by the pool (= region
    //! count × 32 MiB).  Excludes external `std::malloc` fallbacks
    //! for huge allocations.  Monotonically non-decreasing during
    //! steady-state execution; reflects VA reservation, not RSS
    //! (use `getrusage(RUSAGE_SELF)` for RSS).
    extern "C" std::size_t kame_pool_reserved_bytes() noexcept;

    //! \return true while this thread's pool allocator state is fully
    //! live (`g_sys_image_loaded && !s_alloc_tls_off`).  Returns false
    //! once `AllocThreadExitCleanup::~dtor` has fired on this thread (which
    //! is the single point that sets `s_alloc_tls_off`).
    //!
    //! Allocator-using TLS destructors / pthread_key cleanups / atexit
    //! hooks should check this before doing CAS-retry loops, COW
    //! vector rebuilds, or anything that depends on a steady pool /
    //! shared global atomic_shared_ptr state.  A bare `operator new`
    //! (which has its own malloc fallback via `new_redirected`) is
    //! safe regardless and need not check.
    inline bool is_allocator_thread_active() noexcept {
        return g_sys_image_loaded && !s_alloc_tls_off;
    }
#endif

//! RAII guard: enables the KAME pool allocator on construction.
//! Declare one in `main()` (or any function whose scope brackets all
//! pool-allocated lifetimes).  Until the guard is alive, `operator
//! new` falls back to `malloc` — so dyld and static-constructor
//! allocations stay out of the pool.
//!
//! ## Why the destructor does NOT tear pools down
//!
//! `main()` has stack-local objects whose destructors run in LIFO
//! order at function return: the pool guard is constructed last in
//! `main()`, so it would be destructed FIRST.  Any object
//! constructed earlier (e.g. `QTranslator`) is destructed AFTER the
//! guard.
//!
//! `~QTranslator` calls `QCoreApplication::removeTranslator`, which
//! sends a `LanguageChange` event synchronously to every registered
//! widget; the event handlers go through Qt's normal allocator,
//! which is hooked into our pool via `operator new` / `delete`.
//! Tearing the pool's mmap regions down before those allocations
//! would cause `operator new` to dereference unmapped memory →
//! SIGSEGV.  (Observed during in-house crash debugging: faulting
//! address falls in an unmapped gap left by a torn-down chunk;
//! stack is `main -> ~QTranslator -> removeTranslator -> sendEvent
//! -> QApplication::event`.)
//!
//! Letting pool chunks live until process exit is harmless: the
//! mmap'd regions are reclaimed by the kernel.  an earlier change
//! removed the `release_pools()` diagnostic API entirely (no
//! callers anywhere in the tree), so the destructor is a true
//! no-op.
//!
//! On `USE_STD_ALLOCATOR` builds (Windows by default) the guard is a
//! no-op.  Idempotent — multiple guards in nested scopes are harmless.
class KamePooledAllocGuard {
public:
#if defined(KAMEPOOLALLOC_DYLIB)
    //! Dylib mode: pool is auto-activated at dylib load (see
    //! allocator.cpp's `__attribute__((constructor))`); the guard is
    //! a true no-op kept only for source-level uniformity with the
    //! inline-compiled build.
    KamePooledAllocGuard() noexcept = default;
#else
    //! Inline-compiled mode (USE_STD_ALLOCATOR or qmake non-dylib):
    //! activate on construction.
    KamePooledAllocGuard() noexcept { activateAllocator(); }
#endif
    ~KamePooledAllocGuard() noexcept = default;
    KamePooledAllocGuard(const KamePooledAllocGuard &) = delete;
    KamePooledAllocGuard &operator=(const KamePooledAllocGuard &) = delete;
};


#include <vector>
#include <limits>

namespace Transactional {

template<typename T>
class allocator {
public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    template<class Y>
    struct rebind {
        typedef allocator<Y> other;
    };

    allocator() throw () { }
    allocator(const allocator&) throw () { }
    template<typename Y> allocator(const allocator<Y> &) throw () {}
    ~allocator() throw () {}

    pointer allocate(size_type num, const void * /*hint*/ = 0) {
        return (pointer) (operator new(num * sizeof(T)));
    }
    void construct(pointer p, const T& value) {
        new((void*) p) T(value);
    }

    void deallocate(pointer p, size_type /*num*/) {
        operator delete((void *) p);
    }
    void destroy(pointer p) {
        p->~T();
    }

    pointer address(reference value) const {
        return &value;
    }
    const_pointer address(const_reference value) const {
        return &value;
    }

    size_type max_size() const throw () {
        return std::numeric_limits<size_t>::max() / sizeof(T);
    }
};

template <class T1, class T2>
bool operator==(const allocator<T1>&, const allocator<T2>&) throw() { return true; }

template <class T1, class T2>
bool operator!=(const allocator<T1>&, const allocator<T2>&) throw() { return false; }

}
#endif /* ALLOCATOR_H_ */
