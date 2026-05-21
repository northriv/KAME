/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef THREADLOCAL_H_
#define THREADLOCAL_H_

#include "support.h"
#include <type_traits>
#include <cassert>

#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
// Win32 Fiber-Local Storage API (Vista+) — used by the non-trivial-
// dtor specialization below for OS-managed per-thread cleanup.
// Forward-declared rather than pulling in <windows.h> (whose macros
// `min`, `max`, `near`, `far`, ... pollute the global namespace).
// `WINAPI` is `__stdcall` on x86 and effectively a no-op on x64 /
// ARM64; declaring with `__stdcall` unconditionally is safe across
// all Windows architectures Microsoft supports.
extern "C" {
    __declspec(dllimport) unsigned long __stdcall
        FlsAlloc(void (__stdcall *)(void *));
    __declspec(dllimport) void * __stdcall FlsGetValue(unsigned long);
    __declspec(dllimport) int __stdcall FlsSetValue(unsigned long, void *);
    __declspec(dllimport) int __stdcall FlsFree(unsigned long);
}
constexpr unsigned long KAME_FLS_OUT_OF_INDEXES = 0xFFFFFFFFu;
#endif

//! Native TLS storage qualifier — **applied to the per-TU cached
//! `T*` pointer** in `operator*()` below, NOT to the central storage
//! variable in `libkame_storage()`.  The cache is the hot-path slot
//! that every TLS access touches; the central storage is reached
//! exactly once per thread (then latched into the cache).
//!
//! `__thread` on POSIX GCC/Clang lets the linker pick **initial-exec**
//! TLS model for libkame (loaded at process startup) — a single
//! `FS:`/`GS:` register-offset load on Linux ELF.  Plugins (dlopen'd
//! after startup) fall back to general-dynamic automatically.
//! macOS Mach-O has no TLS register, so both `__thread` and
//! `thread_local` compile to `_tlv_get_addr`; the macro choice is a
//! no-op there.  Windows MSVC requires `thread_local`.
//!
//! Since the cached `T*` is a plain pointer (trivially default-
//! constructible AND trivially destructible by construction), the
//! constant-init-only restriction of `__thread` is never an issue
//! here — it applies universally regardless of `T`'s traits.
#if (defined(__GNUC__) || defined(__clang__)) \
    && !defined(_WIN32) && !defined(__WIN32__) && !defined(WINDOWS)
    #define KAME_TLS_QUAL __thread
#else
    #define KAME_TLS_QUAL thread_local
#endif

//! Thread Local Storage template — single point of abstraction for
//! all per-thread storage in KAME.  No SFINAE on the user-facing
//! interface; an internal `if constexpr` inside `libkame_storage()`
//! picks the safest mechanism for `T`'s destructor characteristics.
//!
//! ## Design
//!
//!   * **`operator*()` (hot path)** keeps a per-TU per-thread cached
//!     `T*` qualified `KAME_TLS_QUAL` (= `__thread` on POSIX GCC/
//!     Clang, `thread_local` elsewhere).  First access from each
//!     thread calls `libkame_storage()` once and latches the
//!     returned address; subsequent accesses are a single cached-
//!     pointer deref — same speed as a raw `__thread T` load on
//!     Linux, no DLL boundary cost from plugins.
//!
//!   * **`libkame_storage()` (cold path)** holds the actual T storage.
//!     Speed doesn't matter (called once per thread, amortised by
//!     the cache above), so we optimise for correctness:
//!       - Trivial-dtor T → plain `static thread_local T v{};`.
//!         Safe on every compiler / runtime that supports
//!         thread_local at all (no historical bugs for trivial-dtor).
//!       - Non-trivial-dtor T → native OS TLS slot with explicit
//!         cleanup callback:
//!           * Windows: **FlsAlloc / FlsGetValue / FlsSetValue**
//!             (Vista+, native API in kernel32 — no Qt dependency).
//!             Runs the callback at thread exit for **any** thread,
//!             including raw `CreateThread` / `std::thread` / OS
//!             thread-pool workers that the Qt-based
//!             `QThreadStorage` historically couldn't reach.
//!           * POSIX: **pthread_key_create / pthread_getspecific /
//!             pthread_setspecific** — same semantics, POSIX
//!             guarantees the destructor runs at thread exit.
//!       This works around g++ pre-10 + shared-library
//!       thread_local-dtor bugs and the historical
//!       `__cxa_thread_atexit_impl` reliability issues across DSO
//!       boundaries.  libc++ and modern libstdc++ would handle a
//!       plain `thread_local` correctly too, but routing through
//!       the OS-level cleanup is universally safe — and the cached
//!       pointer in `operator*()` makes the indirect-call overhead
//!       irrelevant.
//!
//! \tparam T   The per-thread datum.  Default-constructed on first
//!             access from each thread; for non-trivial-dtor T,
//!             destroyed at thread exit by the explicit pthread/Qt
//!             cleanup callback.
//! \tparam Tag Disambiguator (defaulted to `void`).  Two
//!             `XThreadLocal<T, Tag>` with the same `T` AND `Tag`
//!             share the function-local storage of
//!             `libkame_storage()` and therefore alias; supply a
//!             distinct empty struct as `Tag` whenever shared `T`
//!             needs distinct slots.
//!
//! ## Cross-DLL singleton via explicit member-template instantiation
//!
//! For variables that must be a process-wide singleton across
//! libkame and plugin DLLs (e.g. `s_tx_nest`), explicit-instantiate
//! `libkame_storage()` in libkame and declare it `extern` in the
//! header.  Per-member `extern template` (C++17) lets `operator*()`
//! still be inlined per-TU — required for the cached pointer to
//! live in each consumer's TLS section.
//!
//! ```cpp
//! struct STxNestTag;
//! extern template int&
//!     XThreadLocal<int, STxNestTag>::libkame_storage();  // in header
//! DECLSPEC_KAME extern XThreadLocal<int, STxNestTag> s_tx_nest;
//!
//! template DECLSPEC_KAME int&
//!     XThreadLocal<int, STxNestTag>::libkame_storage();  // in libkame TU
//! DECLSPEC_KAME XThreadLocal<int, STxNestTag> s_tx_nest;
//! ```
//!
//! Plugins (compiled with `BUILDING_PLUGIN` set in
//! `modules/modules.pri`) inherit the same operator*() — the macro
//! is currently informational only; the cache always runs, in both
//! libkame and plugins.
template <typename T, typename Tag = void>
class XThreadLocal {
public:
    template <typename ...Arg>
    XThreadLocal(Arg&& ...) noexcept {}
    XThreadLocal(const XThreadLocal &) = delete;
    XThreadLocal &operator=(const XThreadLocal &) = delete;

    T &operator*() const noexcept {
        // Per-TU per-thread cached pointer — `T*` is trivial so
        // `__thread` (KAME_TLS_QUAL on POSIX GCC/Clang) applies
        // universally regardless of T's traits.  Cold storage in
        // `libkame_storage()` returns a stable address for the
        // lifetime of the thread, so latching is always safe.
        static KAME_TLS_QUAL T *cached = nullptr;
        if( !cached) [[unlikely]]
            cached = &libkame_storage();
        return *cached;
    }
    T *operator->() const noexcept { return &( **this); }

    DECLSPEC_KAME static T &libkame_storage();

private:
    // pthread_key cleanup callback signature: `void (*)(void *)`.
    static void delete_tls(void *p) noexcept { delete static_cast<T *>(p); }
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
    // Win32 FlsAlloc cleanup callback signature: WINAPI (= __stdcall
    // on x86, unified default on x64/ARM64).  Same body as
    // `delete_tls` but distinct symbol with the right ABI.
    static void __stdcall delete_tls_winapi(void *p) noexcept {
        delete static_cast<T *>(p);
    }
#endif
};

template <typename T, typename Tag>
inline T &XThreadLocal<T, Tag>::libkame_storage() {
    // Cold path — called once per thread per (T, Tag).  Speed doesn't
    // matter, so we optimise for correctness across toolchains.
    if constexpr (std::is_trivially_destructible<T>::value) {
        // No destructor needed — `static thread_local T v;` is safe
        // on every conformant C++11 runtime.  No historical
        // compiler bugs for trivial-dtor T.
        static thread_local T v{};
        return v;
    }
    else {
        // Non-trivial dtor — register an explicit per-thread cleanup
        // callback rather than rely on the runtime's `thread_local`
        // dtor handling (which had bugs on g++ pre-10 in shared
        // libraries and historically required `__cxa_thread_atexit_impl`
        // support across DSO boundaries).  The function-local static
        // owns one slot per (T, Tag) instantiation; magic-statics
        // give us thread-safe lazy init for free.
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
        // Windows: native FLS API (Vista+).  Has a built-in cleanup
        // callback like `pthread_key_create` and runs unconditionally
        // at thread exit — including for non-Qt threads (raw
        // `CreateThread`, `std::thread`, OS thread pool).  Replaces
        // the historical QThreadStorage path: drops the Qt dependency
        // from this header AND tightens cleanup for non-Qt threads
        // (`QThreadStorage` relies on Qt's QThreadData mechanism
        // which is brittle for adopted/raw threads).
        struct Key {
            unsigned long k;
            Key() {
                k = FlsAlloc(&delete_tls_winapi);
                assert(k != KAME_FLS_OUT_OF_INDEXES); (void)k;
            }
            ~Key() { FlsFree(k); }
        };
        static Key key;
        T *p = static_cast<T *>(FlsGetValue(key.k));
        if( !p) [[unlikely]] {
            p = new T;
            int ok = FlsSetValue(key.k, p);
            assert(ok); (void)ok;
        }
        return *p;
#elif defined(USE_PTHREAD)
        struct Key {
            pthread_key_t k;
            Key() {
                int ret = pthread_key_create( &k, &delete_tls);
                assert( !ret); (void)ret;
            }
            ~Key() {
                int ret = pthread_key_delete(k);
                assert( !ret); (void)ret;
            }
        };
        static Key key;
        T *p = static_cast<T *>(pthread_getspecific(key.k));
        if( !p) [[unlikely]] {
            p = new T;
            int ret = pthread_setspecific(key.k, p);
            assert( !ret); (void)ret;
        }
        return *p;
#else
        // Minimal-runtime fallback (e.g. a unit test built without
        // explicit USE_PTHREAD).  Native `thread_local` — modern
        // libc++ and libstdc++ run T's dtor at thread exit; older
        // libstdc++ in shared libraries may not, but this fallback
        // branch is only reached in trivial single-TU contexts where
        // that's acceptable.
        static thread_local T v{};
        return v;
#endif
    }
}

#endif /*THREADLOCAL_H_*/
