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
#include "threadlocal.h"

#include <cassert>
#include <cstdlib>
#include <new>

#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)
// Win32 Fiber-Local Storage (Vista+).  Forward-declare to avoid pulling
// in <windows.h>; declaring `__stdcall` is correct on x86 and a no-op
// on x64 / ARM64.  Fiber-local storage cleans up correctly on thread
// exit (FlsAlloc registers a per-key destructor), and avoids the MSVC
// `thread_local` cross-DLL pitfalls.
extern "C" {
    __declspec(dllimport) unsigned long __stdcall
        FlsAlloc(void (__stdcall *)(void *));
    __declspec(dllimport) void * __stdcall FlsGetValue(unsigned long);
    __declspec(dllimport) int __stdcall FlsSetValue(unsigned long, void *);
    __declspec(dllimport) int __stdcall FlsFree(unsigned long);
}
constexpr unsigned long KAME_FLS_OUT_OF_INDEXES = 0xFFFFFFFFu;
#else
#include <pthread.h>
#endif

namespace Transactional { namespace detail {

namespace {

//! One entry per (thread × XThreadLocal-instance) seen.  Linked-list
//! lookup is fine: each access in `XThreadLocal::operator*()` is
//! amortised by a per-DLL `KAME_TLS_QUAL` cached pointer (one hit per
//! (thread, DLL, instance) over the thread's lifetime).
struct TLSEntry {
    const void *key;
    void *data;
    void (*dtor)(void *);
    TLSEntry *next;
};

//! Per-thread root of the linked list.  Owned via the OS TLS key below;
//! its destructor walks the list LIFO and invokes each registered
//! `dtor()`, then frees the entries.
//!
//! All allocations / deallocations on this path use `std::malloc` /
//! `std::free` directly, bypassing the global `operator new` override
//! (which would route through `PoolAllocator::allocate()`).  This
//! keeps `XThreadLocal` lazy-init reentrancy-safe even when called
//! from inside the allocator bootstrap path (e.g.
//! `tls_alloc_pin_cleanup.add(...)` fires before `s_my_chunk` is set).
struct PerThread {
    TLSEntry *head;
    PerThread() noexcept : head(nullptr) {}
    ~PerThread() noexcept {
        TLSEntry *e = head;
        while(e) {
            TLSEntry *next = e->next;
            e->dtor(e->data);
            //! TLSEntry is trivially destructible; just free the malloc'd block.
            std::free(e);
            e = next;
        }
    }

    PerThread(const PerThread &) = delete;
    PerThread &operator=(const PerThread &) = delete;
};

#if defined(_WIN32) || defined(__WIN32__) || defined(WINDOWS)

void __stdcall fls_destroy(void *p) noexcept {
    auto *pt = static_cast<PerThread *>(p);
    pt->~PerThread();
    std::free(pt);
}

unsigned long get_fls_key() noexcept {
    //! C++11+ thread-safe static initialisation.  Called from the
    //! libkame TU only — single key, allocated once for the whole
    //! process.  No teardown destructor on the key holder: process
    //! exit runs each thread's FLS dtor via the OS.
    static const unsigned long key = []() noexcept {
        unsigned long k = FlsAlloc(&fls_destroy);
        assert(k != KAME_FLS_OUT_OF_INDEXES);
        return k;
    }();
    return key;
}

PerThread &get_per_thread() noexcept {
    const unsigned long key = get_fls_key();
    PerThread *p = static_cast<PerThread *>(FlsGetValue(key));
    if( !p) {
        void *mem = std::malloc(sizeof(PerThread));
        if( !mem) std::abort();
        p = new(mem) PerThread();
        int ok = FlsSetValue(key, p);
        assert(ok); (void)ok;
    }
    return *p;
}

#else // POSIX

void pthread_destroy(void *p) noexcept {
    auto *pt = static_cast<PerThread *>(p);
    pt->~PerThread();
    std::free(pt);
}

pthread_key_t get_pthread_key() noexcept {
    //! Single libkame-side pthread_key for the whole process.
    static const pthread_key_t key = []() noexcept {
        pthread_key_t k;
        int ret = pthread_key_create( &k, &pthread_destroy);
        assert( !ret); (void)ret;
        return k;
    }();
    return key;
}

PerThread &get_per_thread() noexcept {
    const pthread_key_t key = get_pthread_key();
    PerThread *p = static_cast<PerThread *>(pthread_getspecific(key));
    if( !p) {
        void *mem = std::malloc(sizeof(PerThread));
        if( !mem) std::abort();
        p = new(mem) PerThread();
        int ret = pthread_setspecific(key, p);
        assert( !ret); (void)ret;
    }
    return *p;
}

#endif

} // anonymous namespace

DECLSPEC_KAME void *
tls_storage(const void *key, void *(*ctor)(), void (*dtor)(void *)) noexcept {
    PerThread &pt = get_per_thread();
    //! Linear scan — cheap in practice: short list (~tens of entries),
    //! and each lookup is hit at most once per (thread, DLL, instance)
    //! before `XThreadLocal::operator*()`'s cached pointer takes over.
    for(TLSEntry *e = pt.head; e; e = e->next) {
        if(e->key == key) return e->data;
    }
    //! First access on this thread for this key.  Allocate the
    //! entry via malloc (NOT operator new — that would route through
    //! the pool allocator) and chain-prepend.  `ctor()` is also
    //! malloc-based (see `XThreadLocal::ctor_` in threadlocal.h).
    void *mem = std::malloc(sizeof(TLSEntry));
    if( !mem) std::abort();
    TLSEntry *e = new(mem) TLSEntry{key, ctor(), dtor, pt.head};
    pt.head = e;
    return e->data;
}

}} // namespace Transactional::detail
