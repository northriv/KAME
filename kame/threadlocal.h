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
// Win32 Fiber-Local Storage (Vista+) — forward-declared to avoid
// pulling in <windows.h>.  WINAPI = __stdcall on x86, no-op on
// x64/ARM64; declaring __stdcall is safe on all three.
extern "C" {
    __declspec(dllimport) unsigned long __stdcall
        FlsAlloc(void (__stdcall *)(void *));
    __declspec(dllimport) void * __stdcall FlsGetValue(unsigned long);
    __declspec(dllimport) int __stdcall FlsSetValue(unsigned long, void *);
    __declspec(dllimport) int __stdcall FlsFree(unsigned long);
}
constexpr unsigned long KAME_FLS_OUT_OF_INDEXES = 0xFFFFFFFFu;
#endif

//! Applied to the cached `T*` in `operator*()` below.  `__thread` lets
//! the linker pick initial-exec TLS for libkame (loaded at startup) →
//! one register-offset load on Linux ELF.  Plugins (dlopen'd) get
//! general-dynamic automatically.  macOS goes through `_tlv_get_addr`
//! either way; MSVC requires `thread_local`.
#if (defined(__GNUC__) || defined(__clang__)) \
    && !defined(_WIN32) && !defined(__WIN32__) && !defined(WINDOWS)
    #define KAME_TLS_QUAL __thread
#else
    #define KAME_TLS_QUAL thread_local
#endif

//! Thread Local Storage template — single abstraction for per-thread
//! storage in KAME.
//!
//! Two paths:
//!
//!   * `operator*()` (hot) — per-TU per-thread `KAME_TLS_QUAL T*`
//!     cache.  First access calls `libkame_storage()` and latches
//!     the returned address; subsequent accesses are a single
//!     cached-pointer deref.  `T*` is trivial so `__thread` applies
//!     regardless of `T`'s traits.
//!
//!   * `libkame_storage()` (cold) — actual storage.  Speed doesn't
//!     matter (amortised by the cache).  Internal `if constexpr`:
//!       - trivial-dtor T → `static thread_local T v{}`.
//!       - non-trivial-dtor T → OS TLS slot with explicit cleanup:
//!         Windows `FlsAlloc`, POSIX `pthread_key_create`.  Works
//!         around g++ pre-10 thread_local-dtor bugs in shared
//!         libraries; also drops the Qt dependency that the
//!         previous `QThreadStorage` path needed.
//!
//! \tparam T   Per-thread datum; default-constructed on first access.
//! \tparam Tag Disambiguator (defaulted to `void`).  Same `T` + same
//!             `Tag` alias the function-local storage of
//!             `libkame_storage()`; supply a distinct empty struct
//!             when shared `T` needs distinct slots.
//!
//! Cross-DLL singleton via explicit member-template instantiation:
//!
//! ```cpp
//! struct STxNestTag;
//! extern template int&
//!     XThreadLocal<int, STxNestTag>::libkame_storage();   // header
//! DECLSPEC_KAME extern XThreadLocal<int, STxNestTag> s_tx_nest;
//!
//! template DECLSPEC_KAME int&
//!     XThreadLocal<int, STxNestTag>::libkame_storage();   // libkame TU
//! DECLSPEC_KAME XThreadLocal<int, STxNestTag> s_tx_nest;
//! ```
template <typename T, typename Tag = void>
class XThreadLocal {
public:
    template <typename ...Arg>
    XThreadLocal(Arg&& ...) noexcept {}
    XThreadLocal(const XThreadLocal &) = delete;
    XThreadLocal &operator=(const XThreadLocal &) = delete;

    T &operator*() const noexcept {
        static KAME_TLS_QUAL T *cached = nullptr;
        if( !cached) [[unlikely]]
            cached = &libkame_storage();
        return *cached;
    }
    T *operator->() const noexcept { return &( **this); }

    DECLSPEC_KAME static T &libkame_storage();

private:
    static void delete_tls(void *p) noexcept { delete static_cast<T *>(p); }
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
    static void __stdcall delete_tls_winapi(void *p) noexcept {
        delete static_cast<T *>(p);
    }
#endif
};

template <typename T, typename Tag>
inline T &XThreadLocal<T, Tag>::libkame_storage() {
    if constexpr (std::is_trivially_destructible<T>::value) {
        static thread_local T v{};
        return v;
    }
    else {
#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
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
        // Fallback for minimal TUs with neither USE_PTHREAD nor Windows.
        static thread_local T v{};
        return v;
#endif
    }
}

#endif /*THREADLOCAL_H_*/
