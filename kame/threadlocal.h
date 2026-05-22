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
#include <cassert>

//! Applied to the per-DLL cached `T*` in `operator*()` below.  `__thread`
//! lets the linker pick initial-exec TLS for libkame (loaded at startup)
//! → one register-offset load on Linux ELF.  Plugins (dlopen'd) get
//! general-dynamic automatically.  macOS goes through `_tlv_get_addr`
//! either way; MSVC requires `thread_local`.
#if (defined(__GNUC__) || defined(__clang__)) \
    && !defined(_WIN32) && !defined(__WIN32__) && !defined(WINDOWS)
    #define KAME_TLS_QUAL __thread
#else
    #define KAME_TLS_QUAL thread_local
#endif

namespace Transactional { namespace detail {

//! \brief Type-erased TLS slot acquisition — defined once in libkame.dll,
//! dllimport from all plugin DLLs.  Identifies a slot by the address of
//! the caller's `XThreadLocal<T, Tag>` object: when that object is
//! declared `DECLSPEC_KAME extern` in a header and defined in libkame's
//! TU, every DLL sees the same address, so every DLL gets the same
//! per-thread storage.  Module-internal XThreadLocal (not DECLSPEC_KAME)
//! has a per-DLL address — also correct: each module gets its own slot.
//!
//! \param key   Identity of the slot (one per XThreadLocal object).
//! \param ctor  Called once per thread on first access; returns a heap
//!              allocation of T.  Function pointer is captured at first
//!              access and reused; pointer must remain valid for the
//!              thread's lifetime (it is in practice — DLL unload after
//!              all threads exited is the only way it could break).
//! \param dtor  Called on thread exit with the ctor's result.
//!
//! Lookup is a short linked-list walk per first-access; subsequent
//! accesses are served by `XThreadLocal::operator*()`'s `KAME_TLS_QUAL`
//! cached pointer (one load, no function call).
DECLSPEC_KAME void *tls_storage(const void *key,
                                 void *(*ctor)(),
                                 void (*dtor)(void *)) noexcept;

}} // namespace Transactional::detail

//! Thread Local Storage template — single abstraction for per-thread
//! storage in KAME.
//!
//! Two paths:
//!
//!   * `operator*()` (hot) — per-TU per-thread `KAME_TLS_QUAL T*`
//!     cache.  First access calls `detail::tls_storage()` (in libkame.dll)
//!     and latches the returned address; subsequent accesses are a
//!     single cached-pointer deref.
//!
//!   * `detail::tls_storage()` (cold) — single libkame-side
//!     dispatcher.  Walks a per-thread linked list keyed by the
//!     `XThreadLocal` object's address; on miss, calls `ctor_()` to
//!     create the T and registers `dtor_()` for thread-exit cleanup.
//!
//! Cross-DLL singleton is automatic when the `XThreadLocal<T, Tag>`
//! object itself is declared `DECLSPEC_KAME extern` in a shared header
//! and defined once in libkame.dll — all DLLs then reference the same
//! object address, which is the key for `tls_storage()`.  Module-internal
//! `XThreadLocal` (no DECLSPEC_KAME) has a per-DLL address and gets its
//! own per-DLL slot — also correct.
//!
//! No explicit instantiation or `extern template` declaration is
//! required anywhere — the compile-time identity of `T` and `Tag` is
//! used only to wire the type-specific `ctor_()` / `dtor_()` callbacks.
//!
//! \tparam T   Per-thread datum; default-constructed on first access.
//! \tparam Tag Disambiguator (defaulted to `void`).  Same `T` + same
//!             `Tag` with the same enclosing object address share a slot;
//!             supply a distinct empty struct when you need distinct
//!             slots for the same `T`.
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
            cached = static_cast<T *>(
                Transactional::detail::tls_storage(
                    static_cast<const void *>(this), &ctor_, &dtor_));
        return *cached;
    }
    T *operator->() const noexcept { return &( **this); }

private:
    static void *ctor_() { return new T(); }
    static void dtor_(void *p) noexcept { delete static_cast<T *>(p); }
};

#endif /*THREADLOCAL_H_*/
