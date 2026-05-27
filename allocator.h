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
// Anything else falls back to std::allocator via USE_STD_ALLOCATOR.
#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__\
    || defined __x86_64__ || defined _M_IX86 || defined _M_X64\
    || defined __arm64__ || defined __aarch64__ || defined _M_ARM64\
    || defined __WIN32__ || defined WINDOWS || defined _WIN32
#else
    #define USE_STD_ALLOCATOR
#endif
#if defined WINDOWS || defined _WIN32
    #define USE_STD_ALLOCATOR
#endif

#if defined USE_STD_ALLOCATOR
    inline void activateAllocator() {}
    inline void release_pools() {}

    //! \return always true on USE_STD_ALLOCATOR builds — no per-thread
    //! pool state to worry about.
    inline bool is_allocator_thread_active() noexcept { return true; }
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

    extern void activateAllocator();
    extern void release_pools();

    //! \return true while this thread's pool allocator state is fully
    //! live (`g_sys_image_loaded && !s_alloc_tls_off`).  Returns false
    //! once `AllocPinCleanup::~dtor` has fired on this thread (which
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
//! ## Why the destructor does NOT call `release_pools()`
//!
//! `main()` has stack-local objects whose destructors run in LIFO
//! order at function return: the pool guard is constructed last in
//! `main()`, so it is destructed FIRST.  Any object constructed
//! earlier (e.g. `QTranslator`) is destructed AFTER the guard.
//!
//! `~QTranslator` calls `QCoreApplication::removeTranslator`, which
//! sends a `LanguageChange` event synchronously to every registered
//! widget; the event handlers go through Qt's normal allocator,
//! which is hooked into our pool via `operator new` / `delete`.
//! If `~KamePooledAllocGuard` had already `munmap`'d the pool
//! chunks, those allocations dereference unmapped memory → SIGSEGV.
//! (Observed in kame-2026-05-24-193408.ips: faulting address
//! 0x13ec60028 falls in an unmapped gap left by a torn-down chunk;
//! stack is `main -> ~QTranslator -> removeTranslator -> sendEvent
//! -> QApplication::event`.)
//!
//! Letting pool chunks live until process exit is harmless: the
//! mmap'd regions are reclaimed by the kernel.  Callers who genuinely
//! want to tear pools down at a specific point (e.g. unit-test
//! harnesses that recreate allocator state) may still invoke
//! `release_pools()` directly.
//!
//! On `USE_STD_ALLOCATOR` builds (Windows by default) the guard is a
//! no-op.  Idempotent — multiple guards in nested scopes are harmless.
class KamePooledAllocGuard {
public:
    KamePooledAllocGuard() noexcept { activateAllocator(); }
    ~KamePooledAllocGuard() noexcept = default;
    KamePooledAllocGuard(const KamePooledAllocGuard &) = delete;
    KamePooledAllocGuard &operator=(const KamePooledAllocGuard &) = delete;
};


#include <array>
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


template <typename T, size_t SIZE_HINT = 1>
class fast_vector {
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using pointer = T*;
    using const_pointer = const T*;
    static constexpr size_t max_fixed_size = (8 * sizeof(pointer) <= sizeof(T) * SIZE_HINT) ? SIZE_HINT : (8 * sizeof(pointer) / sizeof(T));
public:
    using iterator = pointer;
    using const_iterator = const_pointer;
    fast_vector() : m_size(0) {}
    ~fast_vector() { destroy(); }
    fast_vector(size_type size) {
        if(size > max_fixed_size) {
            new (&m_vector) std::vector<T>(size);
            m_size = HAS_STD_VECTOR;
        }
        else {
            for(size_type i = 0; i < size; ++i)
                new (m_array + i) T();
            m_size = size;
        }
    }
    fast_vector(const fast_vector &r) : m_size(0) {
        this->operator=(r);
    }
    fast_vector(fast_vector &&r) : m_size(0) {
        this->operator=(std::move(r));
    }
    fast_vector& operator=(const fast_vector &r) {
        destroy();
        if(r.is_fixed()) {
            m_size = r.m_size;
            for(size_type i = 0; i < m_size; ++i) {
                new (m_array + i) T(r.m_array[i]);
            }
        }
        else if(r.m_vector.size() <= max_fixed_size) {
            m_size = r.m_vector.size();
            for(size_type i = 0; i < m_size; ++i) {
                new (m_array + i) T(r.m_vector[i]);
            }
        }
        else {
            m_size = HAS_STD_VECTOR;
            new (&m_vector) std::vector<T>(r.m_vector);
        }
        return *this;
    }
    fast_vector& operator=(fast_vector &&r) {
        destroy();
        if(r.is_fixed()) {
            m_size = r.m_size;
            for(size_type i = 0; i < m_size; ++i) {
                new (m_array + i) T(std::move(r.m_array[i]));
            }
            r.clear_fixed();
        }
        else {
            m_size = HAS_STD_VECTOR;
            new (&m_vector) std::vector<T>(std::move(r.m_vector));
        }
        return *this;
    }
    iterator begin() noexcept {return is_fixed() ? &m_array[0] : &m_vector[0];}
    const_iterator begin() const noexcept {return is_fixed() ? &m_array[0] : &m_vector[0];}
    iterator end() noexcept {return is_fixed() ? &m_array[m_size] : &m_vector[m_vector.size()];}
    const_iterator end() const noexcept {return is_fixed() ? &m_array[m_size] : &m_vector[m_vector.size()];}
    size_type size() const noexcept {return is_fixed() ? m_size : m_vector.size();}
    bool empty() const noexcept {return !size();}
    reference operator[](size_type n) {return is_fixed() ? m_array[n] : m_vector[n];}
    const_reference operator[](size_type n) const {return is_fixed() ? m_array[n] : m_vector[n];}
    const_reference at(size_type n) const {if(n >= size()) throw std::out_of_range(""); return (*this)[n];}
    reference at(size_type n) {if(n >= size()) throw std::out_of_range(""); return (*this)[n];}
    reference front() {return (*this)[0];}
    const_reference front() const {return (*this)[0];}
    reference back() {return (*this)[this->size() - 1];}
    const_reference back() const {return (*this)[this->size() - 1];}
    void push_back(const T& x) {
        emplace_back(x);
    }
    void push_back(T&& x) {
        emplace_back(std::move(x));
    }
    template <class... Args>
    void emplace_back(Args&&... args) {
        if(m_size < max_fixed_size) {
            new (m_array + m_size) T(std::forward<Args>(args)...);
            ++m_size;
        }
        else {
            if(m_size == max_fixed_size) {
                move_fixed_to_var(m_size);
            }
            m_vector.emplace_back(std::forward<Args>(args)...);
        }
    }
    iterator erase(const_iterator position) {
        if(is_fixed()) {
            for(auto it = const_cast<iterator>(position);;) {
                 auto nit = it + 1;
                 if(nit == end()) {
                     it->~T();
                     break;
                 }
                 else
                     *it = std::move(*nit);
                 it = nit;
            }
            --m_size;
            return const_cast<iterator>(position);
        }
        else {
            auto it = m_vector.erase(m_vector.begin() + (position - begin()));
            return &*it;
        }
    }
//    iterator erase(const_iterator first, const_iterator last);
    void clear() {
        if(is_fixed()) {
            clear_fixed();
        }
        else {
            m_vector.clear();
        }
    }
    void resize(size_type sz) {
        if(is_fixed()) {
            if(sz > max_fixed_size) {
                move_fixed_to_var(sz);
                m_vector.resize(sz);
            }
            else {
                for(size_type i = m_size; i < sz; ++i)
                    new (m_array + i) T();
                for(size_type i = sz; i < m_size; ++i)
                    m_array[i].~T();
                m_size = sz;
            }
        }
        else {
            m_vector.resize(sz);
//            shrink_to_fit();
        }
    }
    void shrink_to_fit() {
        if( !is_fixed()) return;
        if(m_vector.capacity() - m_vector.size() > max_fixed_size) {
            m_vector.shrink_to_fit();
        }
    }
private:
    void destroy() {
        clear();
        if(!is_fixed())
            m_vector.~vector();
    }
    bool is_fixed() const noexcept {return m_size != (size_type)HAS_STD_VECTOR;}
    void clear_fixed() noexcept {
        assert(is_fixed());
        for(size_type i = 0; i < m_size; ++i)
            m_array[i].~T();
        m_size = 0;
    }
    void move_fixed_to_var(size_type reserve_size) {
        std::vector<T> tmp;
        tmp.reserve(m_size);
        for(size_type i = 0; i < m_size; ++i) {
            tmp.emplace_back(std::move(m_array[i]));
            m_array[i].~T();
        }
        new (&m_vector) std::vector<T>();
        m_vector.reserve(std::max(reserve_size, (size_type)(max_fixed_size * 2)));
        assert(reserve_size >= m_size);
        for(auto &&x: tmp)
            m_vector.emplace_back(std::move(x));
        m_size = HAS_STD_VECTOR;
    }
    size_type m_size;
    static constexpr size_type HAS_STD_VECTOR = (size_type)-1;
    union {
        T m_array[max_fixed_size];
        std::vector<T> m_vector;
    };
};

}
#endif /* ALLOCATOR_H_ */
