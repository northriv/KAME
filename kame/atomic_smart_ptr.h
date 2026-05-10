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
#ifndef ATOMIC_SMART_PTR_H_
#define ATOMIC_SMART_PTR_H_

#include "atomic_prv_basic.h"
#include <functional>
#include <utility>
#include <type_traits>
#include <assert.h>

//! Trait to disable load_shared_() for specific types at compile time.
//! To disable for type T, add `using load_shared_disabled_tag = void;` to T.
//! Detected via SFINAE — no template specialization required.
namespace detail_asp {
    template <typename T, typename = void>
    struct load_shared_enabled_impl : std::true_type {};
    template <typename T>
    struct load_shared_enabled_impl<T, typename std::conditional<true, void,
        typename T::load_shared_disabled_tag>::type> : std::false_type {};
}
template <typename T>
struct load_shared_enabled : detail_asp::load_shared_enabled_impl<T> {};

#ifndef BACKOFF_IN_ATOMIC_SMART_PTR
//if defined as 0, backoff by pause4spin()[=__mm_spin/yield] will be completely killed.
//if > 0, 2^(retry) spin count will by divided by BACKOFF_IN_ATOMIC_SMART_PTR.
    #define BACKOFF_IN_ATOMIC_SMART_PTR 0 //disabled by default, in accord with our tests for AppleSilicon M4 and EPYC 128CPU.
#endif

//! \brief This is an atomic variant of \a std::unique_ptr.
//! An instance of atomic_unique_ptr can be shared among threads by the use of \a swap(\a _shared_target_).\n
//! Namely, it is destructive reading.
//! Use atomic_shared_ptr when the pointer is required to be shared among scopes and threads.\n
//! This implementation relies on an atomic-swap machine code, e.g. lock xchg, or std::atomic.
//! \sa atomic_shared_ptr, atomic_unique_ptr_test.cpp
template <typename T>
class atomic_unique_ptr {
    typedef T* t_ptr;
public:
    atomic_unique_ptr() noexcept : m_ptr(nullptr) {}

    explicit atomic_unique_ptr(t_ptr t) noexcept : m_ptr(t) {}

    ~atomic_unique_ptr() {delete (t_ptr)m_ptr;}

    void reset(t_ptr t = nullptr) noexcept {
        t_ptr old = m_ptr.exchange(t);
        delete old;
    }
    //! \param[in,out] x \p x is atomically swapped.
    //! Nevertheless, this object is not atomically replaced.
    //! That is, the object pointed by "this" must not be shared among threads.
    void swap(atomic_unique_ptr &x) noexcept {
        m_ptr = x.m_ptr.exchange(m_ptr);
    }

    bool operator!() const noexcept {return !(t_ptr)m_ptr;}
    operator bool() const noexcept {return (t_ptr)m_ptr;}

    //! This function lacks thread-safety.
    T &operator*() const noexcept { assert((t_ptr)m_ptr); return (T &) *(t_ptr)m_ptr;}

    //! This function lacks thread-safety.
    t_ptr operator->() const noexcept { assert((t_ptr)m_ptr); return (t_ptr)m_ptr;}

    //! This function lacks thread-safety.
    t_ptr get() const noexcept { return (t_ptr )m_ptr;}

    atomic_unique_ptr(const atomic_unique_ptr &) = delete;
    atomic_unique_ptr& operator=(const atomic_unique_ptr &) = delete;

private:
    atomic<t_ptr> m_ptr;
};

//! This is an internal class holding a global reference counter and a pointer to the object.
//! \sa atomic_shared_ptr
template <typename T>
struct atomic_shared_ptr_gref_ {
    template <typename D>
    atomic_shared_ptr_gref_(T *p, D d) noexcept : ptr(p), refcnt(1), deleter(d) {}
    ~atomic_shared_ptr_gref_() noexcept { assert(refcnt == 0); deleter(); }
    //! The pointer to the object.
    T *ptr;
    typedef uintptr_t Refcnt;
    //! The global reference counter.
    atomic<Refcnt> refcnt;
    std::function<void()> deleter;

    atomic_shared_ptr_gref_(const atomic_shared_ptr_gref_ &) = delete;
};

template <typename X, typename Y, typename Z, typename E> struct atomic_shared_ptr_base;
template <typename X> class atomic_shared_ptr;
template <typename X, typename Y> class local_shared_ptr;
template <typename X> class scoped_atomic_view;

//! Use subclass of this to be storaged in atomic_shared_ptr with
//! intrusive counting to obtain better performance.
// Fwd decl for friend declaration below.
template <typename T> struct atomic_countable_deleter;
template <typename T>
using local_unique_ptr = std::unique_ptr<T, atomic_countable_deleter<T>>;
template <typename T, typename... Args>
local_unique_ptr<T> make_local_unique(Args&&...);

struct atomic_countable {
    atomic_countable() noexcept : refcnt(1) {}
    atomic_countable(const atomic_countable &) noexcept : refcnt(1) {}
    ~atomic_countable() { assert(refcnt == 0); }

    atomic_countable& operator=(const atomic_countable &) = delete;
private:
    template <typename X, typename Y, typename Z, typename E> friend struct atomic_shared_ptr_base;
    template <typename X> friend class atomic_shared_ptr;
    template <typename X, typename Y> friend class local_shared_ptr;
    template <typename X> friend class scoped_atomic_view;
    template <typename X> friend struct atomic_countable_deleter;
    template <typename X, typename... A> friend local_unique_ptr<X> make_local_unique(A&&...);
    typedef uintptr_t Refcnt;
    //! Global reference counter.
    atomic<Refcnt> refcnt;

    void set_deleter(const std::function<void()> &d) {m_deleter = d;}
    std::function<void()> get_deleter() const {return m_deleter;}
    std::function<void()> m_deleter;
};

//! Type trait: true when T uses intrusive reference counting.
template <typename T>
using is_intrusive = std::is_base_of<atomic_countable, T>;

//! Deleter for std::unique_ptr<T> when T is intrusive (subclass of
//! atomic_countable).  T's refcnt starts at 1 (atomic_countable's ctor
//! initialises to 1), so the unique_ptr's destructor must fetch_sub(1)
//! and conditionally invoke the underlying deleter.  The "unique"
//! ownership semantics matches refcnt == 1.
//!
//! Use cases:
//!   - Hold a freshly-allocated `new T(...)` until ownership is
//!     transferred (e.g., into atomic_shared_ptr's m_ref via the
//!     unique_ptr-aware compareAndSet variant).  Saves 2 atomic ops
//!     vs the local_shared_ptr<T> path (no fetch_add at CAS start, no
//!     fetch_sub at caller's dtor).
//!   - RAII safety on CAS failure: if release() is not called, the
//!     destructor cleans up correctly.
template <typename T>
struct atomic_countable_deleter {
    void operator()(T *p) const noexcept {
        // Direct delete: refcnt fetch_sub to 0, then delete (which
        // runs T's dtor including ~atomic_countable's
        // assert(refcnt == 0) — passes since we just zeroed it).
        // No m_deleter lambda dispatch (used only when local_shared_ptr
        // installed a custom allocator deleter; not applicable for
        // local_unique_ptr's transient new-then-transfer pattern).
        if(p->refcnt.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
            delete p;
        }
    }
};

//! `std::unique_ptr<T, atomic_countable_deleter<T>>` shorthand.  T must
//! be intrusive (subclass of atomic_countable).
template <typename T>
using local_unique_ptr = std::unique_ptr<T, atomic_countable_deleter<T>>;

//! Construct a fresh local_unique_ptr<T> from `new T(args...)`.
//! T must be intrusive.  refcnt starts at 1 (atomic_countable's ctor).
//! Sets m_deleter lambda so that when m_ref eventually releases the
//! wrapper (via atomic_shared_ptr_base::deleter), the underlying
//! `delete` fires correctly.  Mirrors local_shared_ptr<T>(new T(...))'s
//! reset_unsafe pattern.
template <typename T, typename... Args>
local_unique_ptr<T> make_local_unique(Args&&... args) {
    T *p = new T(std::forward<Args>(args)...);
    p->set_deleter([p]() { delete p; });
    return local_unique_ptr<T>(p);
}

//! \brief Base class for atomic_shared_ptr without intrusive counting, so-called "simple counted".\n
//! A global referece counter (an instance of atomic_shared_ptr_gref_) will be created.
template <typename T, typename reflocal_t, typename reflocal_var_t, typename Enable = void>
struct atomic_shared_ptr_base {
protected:
    typedef atomic_shared_ptr_gref_<T> Ref;
    typedef typename Ref::Refcnt Refcnt;

    static int deleter(Ref *p) noexcept { delete p; return 1; }

    //! can be used to initialize the internal pointer \a m_ref.
    //! \sa reset()
    template<typename Y, typename D> void reset_unsafe(Y *y, D deleter) {
        m_ref = (reflocal_t)new Ref(y, deleter);
    }
    T *get() noexcept { return this->m_ref ? ((Ref*)(reflocal_t)this->m_ref)->ptr : NULL; }
    const T *get() const noexcept { return this->m_ref ? ((const Ref*)(reflocal_t)this->m_ref)->ptr : NULL; }

    int _use_count_() const noexcept {return ((const Ref*)(reflocal_t)this->m_ref)->refcnt;}

    reflocal_var_t m_ref;
    // LOCAL_REF_CAPACITY defines how many low pointer bits are
    // available for the local refcount. Equal to the minimum alignment
    // guaranteed by the allocator. For non-intrusive mode: sizeof(intptr_t)
    // (Ref is heap-allocated via new). For intrusive mode: sizeof(double)
    // (Ref IS the object, whose alignment may differ). Both are 8 on 64-bit,
    // giving 3 usable bits (max local refcount = 7).
    // Do NOT use alignas/alignof — see CLAUDE.md for details.
    //
    // KAME_LOCAL_REF_CAPACITY_OVERRIDE: force a smaller capacity for
    // stress-testing the back-pressure relief design under simulated
    // high-CPU contention without changing actual allocator alignment.
    // E.g. -DKAME_LOCAL_REF_CAPACITY_OVERRIDE=4 makes 128-thread
    // benchmarks behave like 1024 threads vs. CAPACITY=32 systems.
    // Must be a power of two and <= sizeof(intptr_t) so the pointer
    // mask still strips only zero bits.
#ifdef KAME_LOCAL_REF_CAPACITY_OVERRIDE
    enum {LOCAL_REF_CAPACITY = KAME_LOCAL_REF_CAPACITY_OVERRIDE};
#else
    enum {LOCAL_REF_CAPACITY = (sizeof(intptr_t))};
#endif
};
//! \brief Base class for atomic_shared_ptr with intrusive counting.
template <typename T, typename reflocal_t, typename reflocal_var_t>
struct atomic_shared_ptr_base<T, reflocal_t, reflocal_var_t, typename std::enable_if<is_intrusive<T>::value>::type > {
protected:
    typedef T Ref;
    typedef typename atomic_countable::Refcnt Refcnt;

    int deleter(T *p) noexcept { auto d = p->get_deleter(); d(); return 1;}

    //! can be used to initialize the internal pointer \a m_ref.
    template<typename Y, typename D> void reset_unsafe(Y *y, D d) noexcept {
        m_ref = (reflocal_t)static_cast<T*>(y);
        get()->set_deleter(d);
    }
    T *get() noexcept { return (T*)(reflocal_t)this->m_ref; }
    const T *get() const noexcept { return (const T*)(reflocal_t)this->m_ref; }

    int _use_count_() const noexcept {return ((const T*)(reflocal_t)this->m_ref)->refcnt;}

    reflocal_var_t m_ref;
    // See LOCAL_REF_CAPACITY notes in the non-intrusive base above.
#ifdef KAME_LOCAL_REF_CAPACITY_OVERRIDE
    enum {LOCAL_REF_CAPACITY = KAME_LOCAL_REF_CAPACITY_OVERRIDE};
#else
    enum {LOCAL_REF_CAPACITY = (sizeof(double))};
#endif
};

//! \brief This class provides non-reentrant interfaces for atomic_shared_ptr: operator->(), operator*() and so on.\n
//! Use this class in non-reentrant scopes instead of costly atomic_shared_ptr.
//! \sa atomic_shared_ptr, atomic_unique_ptr, atomic_shared_ptr_test.cpp.
template <typename T, typename reflocal_var_t = uintptr_t>
class local_shared_ptr : protected atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t> {
public:
    local_shared_ptr() noexcept { this->m_ref = (TaggedPtr)nullptr; }

    template<typename Y> explicit local_shared_ptr(Y *y) { this->reset_unsafe(y, [y](){delete y;}); }
    template<typename Y, typename D> local_shared_ptr(Y *y, D deleter) { this->reset_unsafe(y, deleter); }

    explicit local_shared_ptr(const atomic_shared_ptr<T> &t) noexcept { this->m_ref = reinterpret_cast<TaggedPtr>(t.load_shared_()); }
    template<typename Y> local_shared_ptr(const atomic_shared_ptr<Y> &y) {
        static_assert(sizeof(static_cast<const T*>(y.get())), "");
        this->m_ref = reinterpret_cast<TaggedPtr>(y.load_shared_());
    }
    inline local_shared_ptr(const local_shared_ptr<T, reflocal_var_t> &t) noexcept;
    template<typename Y, typename Z> inline local_shared_ptr(const local_shared_ptr<Y, Z> &y) noexcept;
    local_shared_ptr(local_shared_ptr<T, reflocal_var_t> &&t) noexcept {
        this->m_ref = t.m_ref;
        t.m_ref = (TaggedPtr)nullptr;
    }
    template<typename Y, typename Z> local_shared_ptr(local_shared_ptr<Y, Z> &&y) noexcept {
        this->m_ref = y.m_ref;
        y.m_ref = (TaggedPtr)nullptr;
    }
    inline ~local_shared_ptr();

    local_shared_ptr &operator=(const local_shared_ptr &t) noexcept {
        local_shared_ptr(t).swap( *this);
        return *this;
    }
    template<typename Y, typename Z> local_shared_ptr &operator=(const local_shared_ptr<Y, Z> &y) noexcept {
        local_shared_ptr(y).swap( *this);
        return *this;
    }
    local_shared_ptr &operator=(local_shared_ptr &&t) noexcept {
        t.swap( *this);
        t.reset();
        return *this;
    }
    template<typename Y, typename Z> local_shared_ptr &operator=(local_shared_ptr<Y, Z> &&y) noexcept {
        y.swap( *this);
        y.reset();
        return *this;
    }
    //! \param[in] t The pointer held by this instance is replaced with that of \a t.
    local_shared_ptr &operator=(const atomic_shared_ptr<T> &t) noexcept {
        this->reset();
        this->m_ref = reinterpret_cast<TaggedPtr>(t.load_shared_());
        return *this;
    }
    //! \param[in] y The pointer held by this instance is replaced with that of \a y.
    template<typename Y> local_shared_ptr &operator=(const atomic_shared_ptr<Y> &y) noexcept {
        static_assert(sizeof(static_cast<const T*>(y.get())), "");
        this->reset();
        this->m_ref = reinterpret_cast<TaggedPtr>(y.load_shared_());
        return *this;
    }

    //! \param[in,out] x \p The pointer held by \a x is swapped with that of this instance.
    inline void swap(local_shared_ptr &x) noexcept;
    //! \param[in,out] x \p The pointer held by \a x is atomically swapped with that of this instance.
    void swap(atomic_shared_ptr<T> &x) noexcept;

    //! The pointer held by this instance is reset to null pointer.
    inline void reset() noexcept;
    //! The pointer held by this instance is reset with a pointer \a y.
    template<typename Y> void reset(Y *y) { reset(); this->reset_unsafe(y, [y](){ delete y;}); }
    template<typename Y, typename D> void reset(Y *y, D deleter) { reset(); this->reset_unsafe(y, deleter); }

    T *get() noexcept { return atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t>::get(); }
    const T *get() const noexcept { return atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t>::get(); }

    T &operator*() noexcept { assert( *this); return *get();}
    const T &operator*() const noexcept { assert( *this); return *get();}

    T *operator->() noexcept { assert( *this); return get();}
    const T *operator->() const noexcept { assert( *this); return get();}

    bool operator!() const noexcept {return !this->m_ref;}
    operator bool() const noexcept {return this->m_ref;}

    template<typename Y, typename Z> bool operator==(const local_shared_ptr<Y, Z> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (this->ref_ptr_() == (const Ref *)x.ref_ptr_());}
    template<typename Y> bool operator==(const atomic_shared_ptr<Y> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (this->ref_ptr_() == (const Ref *)x.ref_ptr_());}
    template<typename Y, typename Z> bool operator!=(const local_shared_ptr<Y, Z> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (this->ref_ptr_() != (const Ref *)x.ref_ptr_());}
    template<typename Y> bool operator!=(const atomic_shared_ptr<Y> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (this->ref_ptr_() != (const Ref *)x.ref_ptr_());}

    int use_count() const noexcept { return this->_use_count_();}
    bool unique() const noexcept {return use_count() == 1;}
protected:
    template <typename Y, typename Z> friend class local_shared_ptr;
    template <typename Y> friend class atomic_shared_ptr;
    template <typename Y> friend class scoped_atomic_view;  // operator local_shared_ptr<T>() needs m_ref
    typedef typename atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t>::Ref Ref;
    typedef typename atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t>::Refcnt Refcnt;
    typedef uintptr_t TaggedPtr;

    //! A pointer to global reference struct.
    Ref* ref_ptr_() const noexcept {return (Ref *)(TaggedPtr)(this->m_ref);}
};

/*! \brief This is an atomic variant of \a std::shared_ptr, and can be shared by atomic and lock-free means.\n
 *
* \a atomic_shared_ptr can be shared among threads by the use of \a operator=(_target_), \a swap(_target_).
* An instance of \a atomic_shared_ptr<T> holds:\n
* 	a) a pointer to \a atomic_shared_ptr_gref_<T>, which is a struct. consisting of a pointer to a T-type object and a global reference counter.\n
* 	b) a local (temporary) reference counter, which is embedded in the above pointer by using several LSBs that should be usually zero.\n
* The values of a) and b), \a m_ref, are atomically handled with CAS machine codes.
* The purpose of b) the local reference counter is to tell the "observation" to the shared target before increasing the global reference counter.
* This process is implemented in \a acquire_tag_ref_().\n
* A function \a release_tag_ref_() tries to decrease the local counter first. When it fails, the global counter is decreased.\n
* To swap the pointer and local reference counter (which will be reset to zero), the setter must adds the local counting to the global counter before swapping.
* \sa atomic_unique_ptr, local_shared_ptr, atomic_shared_ptr_test.cpp.
 */
template <typename T>
class atomic_shared_ptr : protected local_shared_ptr<T, atomic<uintptr_t>> {
public:
    atomic_shared_ptr() noexcept : local_shared_ptr<T, atomic<uintptr_t>>() {}

    template<typename Y> explicit atomic_shared_ptr(Y *y) : local_shared_ptr<T, atomic<uintptr_t>>(y) {}
    atomic_shared_ptr(const atomic_shared_ptr<T> &t) noexcept : local_shared_ptr<T, atomic<uintptr_t>>(t) {}
    template<typename Y> atomic_shared_ptr(const atomic_shared_ptr<Y> &y) noexcept : local_shared_ptr<T, atomic<uintptr_t>>(y) {}
    atomic_shared_ptr(const local_shared_ptr<T> &t) noexcept : local_shared_ptr<T, atomic<uintptr_t>>(t) {}
    template<typename Y> atomic_shared_ptr(const local_shared_ptr<Y> &y) noexcept : local_shared_ptr<T, atomic<uintptr_t>>(y) {}
    atomic_shared_ptr(atomic_shared_ptr<T> &&t) noexcept {
        operator=(std::move(t));
    }
    template<typename Y> atomic_shared_ptr(atomic_shared_ptr<Y> &&y) noexcept {
        operator=(std::move(y));
    }

    ~atomic_shared_ptr() {}

    //! \param[in] t The pointer held by this instance is atomically replaced with that of \a t.
    atomic_shared_ptr &operator=(const atomic_shared_ptr &t) noexcept {
        local_shared_ptr<T>(t).swap( *this);
        return *this;
    }
    //! \param[in] y The pointer held by this instance is atomically replaced with that of \a y.
    template<typename Y> atomic_shared_ptr &operator=(const local_shared_ptr<Y> &y) noexcept {
        local_shared_ptr<T>(y).swap( *this);
        return *this;
    }
    atomic_shared_ptr &operator=(local_shared_ptr<T> &&t) noexcept {
        t.swap( *this);
        t.reset();
        return *this;
    }
    template<typename Y> atomic_shared_ptr &operator=(local_shared_ptr<Y> &&y) noexcept {
        y.swap( *this);
        y.reset();
        return *this;
    }
    //! The pointer held by this instance is atomically reset to null pointer.
    void reset() noexcept {
        local_shared_ptr<T>().swap( *this);
    }
    //! The pointer held by this instance is atomically reset with a pointer \a y.
    template<typename Y> void reset(Y *y) {
        local_shared_ptr<T>(y).swap( *this);
    }

    //! \return true if succeeded.
    //! \sa compareAndSwap()
    bool compareAndSet(const local_shared_ptr<T> &oldvalue, const local_shared_ptr<T> &newvalue) noexcept;
    //! \return true if succeeded.
    //! \sa compareAndSet()
    bool compareAndSwap(local_shared_ptr<T> &oldvalue, const local_shared_ptr<T> &newvalue) noexcept;
    //! \return true if succeeded.
    //! \sa compareAndSet()
    bool compareAndSetWeak(const local_shared_ptr<T> &oldvalue, const local_shared_ptr<T> &newvalue) noexcept;

    //! local_unique_ptr<T> overloads — newr ownership transfers to
    //! m_ref on success.  Saves 2 atomic ops per CAS vs the
    //! local_shared_ptr version.  newr is in/out: released on
    //! success, retained on failure.
    bool compareAndSet(const local_shared_ptr<T> &oldvalue, local_unique_ptr<T> &newvalue) noexcept;
    bool compareAndSetWeak(const local_shared_ptr<T> &oldvalue, local_unique_ptr<T> &newvalue) noexcept;
    bool compareAndSetWeak(scoped_atomic_view<T> &scoped, local_unique_ptr<T> &newvalue) noexcept;
    //! \return true if succeeded.
    //! \brief Weakly version using a pre-acquired \a scoped_atomic_view.
    //!   On success, \a scoped is reset to Empty (tag consumed by CAS). On
    //!   weak failure (CAS contention), \a scoped remains TagHeld for retry.
    //!   On pointer change since acquire, \a scoped is eagerly cleaned up
    //!   to Empty so the caller can detect it via \a scoped.operator bool().
    inline bool compareAndSetWeak(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newvalue) noexcept;

    //! \brief Like compareAndSetWeak(scoped, newr) but on success \a scoped
    //!   transitions to Owned(newr) instead of Empty.  Saves a reload when
    //!   the caller needs to keep tracking the new value after CAS success.
    //!   Entry does fetch_add(2) instead of (1); failure undo is fetch_sub(2).
    inline bool compareAndSetWeakRetain(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newvalue) noexcept;

    //! \brief STRONG (spinning) version of compareAndSetWeak(scoped, newr).
    //!   Internal CAS retry loop on spurious weak failure; returns false
    //!   only on pointer mismatch (real contention). Intended for use by
    //!   the privileged thread (s_privileged_tidstamp holder), where
    //!   fair_mode blocks all other CAS — no peer to contend with.
    inline bool compareAndSetStrong(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newvalue) noexcept;
    //! \brief STRONG + RETAIN_NEWR variant — see compareAndSetStrong and
    //!   compareAndSetWeakRetain.
    inline bool compareAndSetStrongRetain(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newvalue) noexcept;

    bool operator!() const noexcept {return !this->m_ref;}
    operator bool() const noexcept {return this->m_ref;}

    template<typename Y> bool operator==(const local_shared_ptr<Y> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (ref_ptr_() == (const Ref*)x.ref_ptr_());}
    template<typename Y> bool operator==(const atomic_shared_ptr<Y> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (ref_ptr_() == (const Ref*)x.ref_ptr_());}
    template<typename Y> bool operator!=(const local_shared_ptr<Y> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (ref_ptr_() != (const Ref*)x.ref_ptr_());}
    template<typename Y> bool operator!=(const atomic_shared_ptr<Y> &x) const noexcept {
        static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (ref_ptr_() != (const Ref*)x.ref_ptr_());}
    //! Comparison with scoped_atomic_view (staleness check etc.).
    bool operator==(const scoped_atomic_view<T> &x) const noexcept {
        return (ref_ptr_() == x.ref_ptr_());}
    bool operator!=(const scoped_atomic_view<T> &x) const noexcept {
        return (ref_ptr_() != x.ref_ptr_());}
protected:
    template <typename Y, typename Z> friend class local_shared_ptr;
    template <typename Y> friend class atomic_shared_ptr;
    template <typename Y> friend class scoped_atomic_view;
    typedef typename atomic_shared_ptr_base<T, uintptr_t, atomic<uintptr_t>>::Ref Ref;
    typedef typename atomic_shared_ptr_base<T, uintptr_t, atomic<uintptr_t>>::Refcnt Refcnt;
    typedef atomic<uintptr_t> TaggedPtr;
    //! A pointer to global reference struct.
    Ref* ref_ptr_() const noexcept {
        auto ref = this->m_ref.load(std::memory_order_relaxed);
        return (Ref*)(ref & (~(uintptr_t)(this->LOCAL_REF_CAPACITY - 1)));
    }
    //! Single atomic load returning both the pointer and the local refcount.
    std::pair<Ref*, Refcnt> load_tagged_() const noexcept {
        auto ref = this->m_ref.load(std::memory_order_relaxed);
        return {(Ref*)(ref & (~(uintptr_t)(this->LOCAL_REF_CAPACITY - 1))),
                (Refcnt)(ref & (uintptr_t)(this->LOCAL_REF_CAPACITY - 1))};
    }

    //internal functions below.
    //! Atomically scans \a m_ref and increases the global reference counter.
    //! \a load_shared_() is used for atomically coping the pointer.
    inline Ref *load_shared_() const noexcept;
    //! Atomically scans \a m_ref and increases the  local (temporary) reference counter.
    //! use \a release_tag_ref_() to release the temporary reference.
    inline std::pair<Ref*, bool> acquire_tag_ref_(Refcnt *, bool weakly = false) const noexcept;
    //! Tries to decrease local (temporary) reference counter.
    //! In case the reference is lost, \a release_tag_ref_() releases the global reference counter instead.
    //! When \a left_global_rcnt > 0, undoes step 4's
    //! excess (left_global_rcnt - 1) on tag-success, or combines undo+release on pointer-changed.
    //! \param[in] single_attempt  If true, drain CAS is single-shot;
    //!   on CAS-loss, returns false WITHOUT global fetch_sub.
    inline bool release_tag_ref_(Ref *, Refcnt added_global_rcnt,
                                  bool single_attempt = false) const noexcept;

    //! Unified CAS template covering all flavours of compareAndSet/Swap.
    //!
    //! The OldrT parameter is deduced from the call site and selects the
    //! variant via constexpr if branches:
    //!   - OldrT = const local_shared_ptr<T> (Set):
    //!       no acquire_tag_ref_ (oldr keeps pref alive); step4 = +T;
    //!       failure undo via plain fetch_sub(T, relaxed).
    //!   - OldrT = local_shared_ptr<T> (Swap):
    //!       acquire_tag_ref_ required (will update oldr on mismatch);
    //!       step4 = +(T-1); failure undo via release_tag_ref_(pref, T).
    //!   - OldrT = scoped_atomic_view<T> (SetScoped, weak only):
    //!       scoped already holds tag (no acquire); step4 = +(T-1);
    //!       failure undo via plain fetch_sub(T-1, relaxed); on success,
    //!       scoped's tag is consumed by CAS (m_pref reset to nullptr).
    //! NewrT controls how the desired ref's refcnt is managed:
    //!   - NewrT = const local_shared_ptr<T> (caller retains ownership):
    //!       fetch_add(1) at start, fetch_sub(1) at WEAK-failure undo.
    //!       m_ref takes its own +1 via the fetch_add; caller's local
    //!       var keeps its +1 separately.
    //!   - NewrT = local_unique_ptr<T> (caller transfers ownership):
    //!       NO fetch_add at start.  On CAS success: newr.release()
    //!       transfers the existing refcnt=1 to m_ref's implicit ref.
    //!       On CAS failure (weak): caller's unique_ptr keeps the
    //!       wrapper; its destructor handles cleanup.
    //!       Saves 2 atomic ops per CAS (success or failure path).
    template<typename OldrT, typename NewrT, bool WEAK = false, bool RETAIN_NEWR = false>
    inline bool compareAndSet_impl_(OldrT &oldr,
        NewrT &newr) noexcept;
private:
};

//! \brief RAII scoped tag holder on \a atomic_shared_ptr<T>.
//!
//! Acquires a tag ref on the supplied atomic_shared_ptr's m_ref in the
//! constructor (1 CAS). Releases on destruction. Move-only.
//!
//! Three logical states:
//!   - Empty (m_pref == nullptr):
//!       - asp was nullptr, or
//!       - weakly acquire failed (\a acquire_succeeded() == false), or
//!       - tag was consumed by a successful compareAndSetWeak.
//!   - TagHeld (m_pref != nullptr, m_tag_held == true):
//!       Tag still held in m_ref's tag count. Destructor calls
//!       release_tag_ref_(pref, 1u). compareAndSetWeak uses scoped path.
//!   - Owned (m_pref != nullptr, m_tag_held == false):
//!       Tag was promoted to refcnt at construction time
//!       (fetch_add(rcnt) + release_tag_ref_). Destructor does plain
//!       fetch_sub(1, acq_rel) + delete check (like local_shared_ptr).
//!       compareAndSetWeak uses Set path (regular const local_shared_ptr
//!       semantics).
//!
//! Adaptive promotion:
//!   The optional \a promote_threshold parameter selects between TagHeld and
//!   Owned at construction. If the tag count after acquire (rcnt_new) is >=
//!   threshold, the scoped is promoted to Owned (frees a tag slot). Useful
//!   when many concurrent readers risk filling LOCAL_REF_CAPACITY.
//!     - threshold = 1: always promote → equivalent to load_shared_().
//!     - threshold = LOCAL_REF_CAPACITY-1 (DEFER, default): promote only at
//!       the LAST slot. Reserved for the privileged thread (single contender
//!       guarantee, see ScopedNegotiateLinkage); keeps TagHeld cheap.
//!     - threshold = LOCAL_REF_CAPACITY-2 (ADAPTIVE): promote one slot earlier
//!       — the thread that lands at the second-to-last slot pre-emptively
//!       drains tag bits, leaving room (the LAST slot) for the privileged
//!       thread.  Used by all non-privileged acquires.
//!
//!   A "never-promote" mode (formerly threshold = LOCAL_REF_CAPACITY) was
//!   removed — without promote, peer-thread TagHeld views accumulate and
//!   block zero-reset CAS indefinitely (see livelock fixed in c363629a).
template <typename T>
class scoped_atomic_view {
public:
    typedef typename atomic_shared_ptr<T>::Ref Ref;
    typedef typename atomic_shared_ptr<T>::Refcnt Refcnt;

    enum {
        LOCAL_REF_CAPACITY = atomic_shared_ptr<T>::LOCAL_REF_CAPACITY,
        DEFER_THRESHOLD = LOCAL_REF_CAPACITY - 1,        //!< promote at last slot (privileged-only)
        ADAPTIVE_THRESHOLD = LOCAL_REF_CAPACITY - 2,     //!< promote one slot early (non-privileged)
    };

    scoped_atomic_view() noexcept
        : m_asp(nullptr), m_pref(nullptr), m_tag_held(false),
          m_acquire_succeeded(true) {}

    //! Acquires a tag ref on \a asp.m_ref.
    //! \param[in] promote_threshold Tag-count threshold for adaptive
    //!   promotion. After acquire bumps tag to rcnt_new, if rcnt_new >=
    //!   promote_threshold the tag is promoted to refcnt (Owned mode).
    //!   Default \a ADAPTIVE_THRESHOLD promotes one slot before the cap
    //!   (= reserve last slot for the privileged thread).  Use
    //!   \a DEFER_THRESHOLD only on the privileged path.
    //! \param[in] weakly If true, the acquire CAS is single-shot — on
    //!   contention, this constructs an Empty instance with
    //!   \a acquire_succeeded() == false. Strong (default) loops until success.
    //! \note On Empty after construction, distinguish:
    //!   - \a acquire_succeeded() == true  AND \a m_pref == nullptr →
    //!       asp held nullptr (genuinely empty atomic_shared_ptr).
    //!   - \a acquire_succeeded() == false AND \a m_pref == nullptr →
    //!       weakly = true and the acquire CAS lost (caller should retry).
    explicit scoped_atomic_view(atomic_shared_ptr<T> &asp,
                                     Refcnt promote_threshold = ADAPTIVE_THRESHOLD,
                                     bool weakly = false) noexcept
        : m_asp(&asp), m_pref(nullptr), m_tag_held(false),
          m_acquire_succeeded(true) {
        Refcnt rcnt;
        auto [p, ok] = asp.acquire_tag_ref_( &rcnt, weakly);
        if(p && ok) {
            if(rcnt >= promote_threshold) {
                // Promote: tag → refcnt. Bit-identical to load_shared_.
                p->refcnt.fetch_add(rcnt, std::memory_order_relaxed);
                asp.release_tag_ref_(p, rcnt);
                m_pref = p;
                m_tag_held = false;  // sentinel for Owned
            } else {
                m_pref = p;
                m_tag_held = true;  // TagHeld
            }
        } else {
            m_acquire_succeeded = ok;  // false on weak fail; true on null asp
        }
    }

    //! Move-construct from a `local_shared_ptr<T>&&`.  Takes ownership of
    //! `from`'s +1 refcount with ZERO atomic ops — the new view starts in
    //! Owned mode (`m_tag_held == 0`) reusing `from`'s refcount.
    //! `asp` is the atomic_shared_ptr the view is bound to (used for the
    //! weak-CAS scoped path and for release on dtor).  Caller is
    //! responsible that the moved-from `local_shared_ptr` was a valid
    //! reference to `asp`'s current value at construction time (we do
    //! NOT verify; standard move-semantics caveat).
    scoped_atomic_view(atomic_shared_ptr<T> &asp, local_shared_ptr<T> &&from) noexcept
        : m_asp(&asp), m_pref(nullptr), m_tag_held(false),
          m_acquire_succeeded(true) {
        if(from.m_ref) {
            m_pref = (Ref *)from.m_ref;
            // m_tag_held stays 0 → Owned mode.
            from.m_ref = 0;  // empty out the source
        }
    }

    //! Replace the current view with a value from `from`, taking
    //! ownership of `from`'s +1 refcount (zero atomic ops on the
    //! transfer).  The previous view is released first.
    //!
    //! Use this after a successful CAS to update the view to the new
    //! linkage value without paying a load_shared_ — e.g. a multi-phase
    //! CAS protocol where each phase advances the linkage and we want
    //! the view to track without re-reading the atomic_shared_ptr.
    void assign_from_local(local_shared_ptr<T> &&from) noexcept {
        release_();
        if(from.m_ref) {
            m_pref = (Ref *)from.m_ref;
            m_tag_held = false;  // Owned mode
            from.m_ref = 0;  // empty out the source
        } else {
            m_pref = nullptr;
            m_tag_held = false;
        }
        m_acquire_succeeded = true;
    }

    scoped_atomic_view(scoped_atomic_view &&other) noexcept
        : m_asp(other.m_asp), m_pref(other.m_pref),
          m_tag_held(other.m_tag_held),
          m_acquire_succeeded(other.m_acquire_succeeded) {
        other.m_pref = nullptr;
        other.m_tag_held = false;
        other.m_acquire_succeeded = true;
    }
    scoped_atomic_view &operator=(scoped_atomic_view &&other) noexcept {
        if(this != &other) {
            release_();
            m_asp = other.m_asp;
            m_pref = other.m_pref;
            m_tag_held = other.m_tag_held;
            m_acquire_succeeded = other.m_acquire_succeeded;
            other.m_pref = nullptr;
            other.m_tag_held = false;
            other.m_acquire_succeeded = true;
        }
        return *this;
    }
    scoped_atomic_view(const scoped_atomic_view &) = delete;
    scoped_atomic_view &operator=(const scoped_atomic_view &) = delete;

    //! Exchange internal state with another view.  Useful for
    //! "transferring ownership through a sub-routine":
    //!   void f(scoped_atomic_view<T> &out) {
    //!       scoped_atomic_view<T> local(*some_asp, ADAPTIVE_THRESHOLD);
    //!       ... use local for CAS / read ...
    //!       out.swap(local);  // hand it back to caller; local goes
    //!                         // out of scope releasing the *old* out.
    //!   }
    //! No tag refcount op happens — just a stateless rearrangement.
    void swap(scoped_atomic_view &other) noexcept {
        std::swap(m_asp, other.m_asp);
        std::swap(m_pref, other.m_pref);
        std::swap(m_tag_held, other.m_tag_held);
        std::swap(m_acquire_succeeded, other.m_acquire_succeeded);
    }

    ~scoped_atomic_view() noexcept { release_(); }

    //! \brief Convert to local_shared_ptr<T>. Internally promotes if needed.
    //!   - From TagHeld: promote (tag → refcnt; bit-identical to load_shared_),
    //!     then fetch_add(1) for the new ref. Scoped transitions to Owned.
    //!   - From Owned: just fetch_add(1) for the new ref. Scoped retains.
    //!   - From Empty: returns empty.
    //! \note lvalue conversion — scoped remains usable after the call.
    //!   For scoped that holds a long-lived ref, consider whether
    //!   the extra fetch_add(1) is worth it vs. holding the scoped directly.
    operator local_shared_ptr<T>() & noexcept {
        local_shared_ptr<T> ret;
        if(m_pref) {
            if(m_tag_held) {
                // TagHeld → Promote (zero-reset): load CURRENT tag count
                // and drain it all in one shot, transferring rcnt_now
                // refs to global.  This absorbs all current tag holders'
                // IOUs (not just our acquire-time snapshot), helping
                // keep tag bits low for other threads' acquires.
                promote_tagheld_();
                m_tag_held = false;  // mode flips to Owned
            }
            // Owned: scoped already has +1 in refcnt. Add another +1 for the
            //   new local_shared_ptr's own ownership.
            m_pref->refcnt.fetch_add(1, std::memory_order_relaxed);
            ret.m_ref = (uintptr_t)m_pref;
        }
        return ret;
    }

    //! \brief rvalue (move) conversion — transfer ownership to a
    //! `local_shared_ptr<T>`, leaving this view empty.
    //!   - From Owned: ZERO atomic ops — the +1 refcnt is just transferred.
    //!   - From TagHeld: promote (zero-reset, current rcnt) but skip the
    //!     fetch_add(1) for the new local_shared_ptr's ownership (the
    //!     promoted ref IS the new ownership).
    //!   - From Empty: returns empty.
    //! Use `std::move(scoped)` to explicitly opt in.  Saves 1 atomic op
    //! vs the lvalue conversion when the view will not be used again.
    operator local_shared_ptr<T>() && noexcept {
        local_shared_ptr<T> ret;
        if(m_pref) {
            if(m_tag_held) {
                promote_tagheld_();
            }
            // Transfer m_pref to ret.  Empty out self so dtor is a no-op.
            ret.m_ref = (uintptr_t)m_pref;
            m_pref = nullptr;
            m_tag_held = false;
        }
        return ret;
    }

    bool operator!() const noexcept { return m_pref == nullptr; }
    explicit operator bool() const noexcept { return m_pref != nullptr; }

    //! \return false only when weakly acquire CAS lost; true otherwise (incl. null asp).
    bool acquire_succeeded() const noexcept { return m_acquire_succeeded; }

    //! \return true if currently in TagHeld mode (vs Owned or Empty).
    bool is_tag_held() const noexcept { return m_pref && m_tag_held; }
    //! \return true if currently in Owned mode (promoted at construction).
    bool is_owned() const noexcept { return m_pref && !m_tag_held; }

    //! Smart-pointer accessors (return T*).
    T *get() const noexcept {
        if( !m_pref) return nullptr;
        if constexpr (is_intrusive<T>::value) {
            return reinterpret_cast<T*>(m_pref);
        } else {
            return m_pref->ptr;
        }
    }
    T *operator->() const noexcept { assert(m_pref); return get(); }
    T &operator*() const noexcept { assert(m_pref); return *get(); }

    Ref *ref_ptr_() const noexcept { return m_pref; }

    //! Identity comparison against atomic_shared_ptr (e.g. Linkage).
    //! Pure relaxed load + pointer comparison — no load_shared_,
    //! no refcount manipulation.  scoped_atomic_view is a friend of
    //! atomic_shared_ptr, so ref_ptr_() access is valid.
    //! Reverse direction (asp != view) uses inherited
    //! local_shared_ptr::operator!=(scoped_atomic_view) — no friend
    //! needed (adding one would create ambiguity with the inherited
    //! member via Linkage's inheritance chain).
    bool operator==(const atomic_shared_ptr<T> &rhs) const noexcept {
        return m_pref == rhs.ref_ptr_();
    }
    bool operator!=(const atomic_shared_ptr<T> &rhs) const noexcept {
        return m_pref != rhs.ref_ptr_();
    }

private:
    //! Promote TagHeld → Owned via "zero-reset": load current tag
    //! count and drain ALL of them, transferring rcnt_now refs to
    //! global in a single fetch_add + drain CAS.
    //!
    //! Compared to the previous "use rcnt_at_acquire" pattern:
    //!   - Same atomic op count on the success path
    //!     (1 fetch_add + 1 CAS via release_tag_ref_, drains all,
    //!     sub_amount = 0, no extra fetch_sub)
    //!   - Captures CURRENT state (others' tags acquired AFTER our
    //!     acquire are also drained), helping keep tag bits at 0 for
    //!     subsequent acquires
    //!   - On ptr-change (swapper absorbed our tag) or rcnt_now == 0
    //!     (some drainer already absorbed us), fall back to plain
    //!     fetch_add(1) — our tag is already accounted in global.
    //!
    //! Caller is responsible for setting m_tag_held = false after,
    //! since this function only handles the atomic-state transition.
    void promote_tagheld_() noexcept {
        auto [cur_ptr, rcnt_now] = m_asp->load_tagged_();
        if(cur_ptr == m_pref && rcnt_now > 0) {
            // Pre-pay rcnt_now to global: covers all rcnt_now tag
            // holders (us + others present at this moment).  Drain
            // CAS in release_tag_ref_ tries to remove rcnt_now tags;
            // sub_amount = rcnt_now - drained, so net global change
            // = drained.  Our own +1 is part of those drained refs.
            m_pref->refcnt.fetch_add(rcnt_now, std::memory_order_relaxed);
            m_asp->release_tag_ref_(m_pref, rcnt_now);
        } else {
            // ptr changed (swapper absorbed) or tag already drained
            // (another load_shared_ / promote took our tag and
            // converted it to global).  Either way our +1 is in
            // global; just add 1 more for the Owned ref we want.
            // Wait — we already had +1 absorbed; we need to gain
            // a +1 for "Owned" mode.  Since absorption transferred
            // our tag's implicit ref to global, we already have it.
            // No fetch_add needed.
        }
    }

    //! Release TagHeld via "zero-reset": load current tag count and
    //! drain ALL tags, paying others' IOUs to global.  After this
    //! call tag bits are 0 (assuming no race), letting other threads'
    //! acquires succeed without weakly-failing on capacity.
    //!
    //! Atomic op count: 1 fetch_add + 1 CAS (via release_tag_ref_).
    //! Compared to the simple release_tag_ref_(pref, 1) (1 CAS),
    //! this costs +1 op per release but bounds tag accumulation —
    //! crucial under high-contender:capacity ratios.
    //!
    //! Math (state at call: tag = T including our +1, global = G):
    //!   - Pre-pay (T-1) to global: G' = G + T - 1
    //!   - release_tag_ref_(pref, T) drains drained tags;
    //!     sub_amount = T - drained, fetch_sub.
    //!   - Net global: G + T - 1 - (T - drained) = G + drained - 1
    //!     - Full drain (drained = T): G + T - 1, tag = 0.  ✓
    //!     - Partial drain: G + drained - 1, tag = T - drained.  ✓
    //!     - Fall-through (ptr changed): drained = 0, sub_amount = T,
    //!       net global = G - 1, tag wherever.  ✓ (our +1 was absorbed
    //!       by swapper into G already; -1 releases our share)
    //! All cases: net true ref change = -1.  Verified by induction
    //! on the standard refcnt invariant (true_refs = global + tag
    //! when m_ref still points to pref).
    bool release_tagheld_zeroreset_(bool single_attempt) noexcept {
        auto [cur_ptr, rcnt_now] = m_asp->load_tagged_();
        if(cur_ptr == m_pref && rcnt_now > 0) {
            if(rcnt_now > 1) {
                m_pref->refcnt.fetch_add(rcnt_now - 1,
                    std::memory_order_relaxed);
            }
            return m_asp->release_tag_ref_(m_pref, rcnt_now, single_attempt);
        }
        // ptr changed or tag already drained — our +1 is in global.
        if(m_pref->refcnt.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            m_asp->deleter(m_pref);
        }
        return true;
    }

public:
    //! Single-attempt release.  Returns true on success (view becomes
    //! empty); false if drain CAS lost (view stays valid; caller must
    //! retry).  Tracks rcnt_added across iterations so we only
    //! fetch_add / fetch_sub the DIFF when observed tag count changes.
    //! All fetch_sub use acq_rel + delete check (memory ordering
    //! correctness even though the typical case won't drop refcnt to 0).
    //! Caller initialises rcnt_added=0 before first call.
    bool try_release_single_attempt(uintptr_t &rcnt_added) noexcept {
        if( !m_pref) return true;
        // Helper: skip fetch_sub when sub == 0 (a fetch_sub(0, acq_rel)
        // is not a no-op — its delete check fires if OLD refcnt == 0,
        // a race that can occur if m_ref was reset between our CAS and
        // this fetch_sub).  Captures m_pref and m_asp for deleter.
        auto sub_with_delete_check = [this](uintptr_t sub) {
            if(sub) {
                if(m_pref->refcnt.fetch_sub(sub,
                        std::memory_order_acq_rel) == sub) {
                    m_asp->deleter(m_pref);
                }
            }
        };
        if( !m_tag_held) {
            // Owned: plain fetch_sub(1).  Plus undo any pre-pay (Owned
            // mode shouldn't have pre-pay normally; safety).
            sub_with_delete_check(rcnt_added + 1);
            m_pref = nullptr;
            rcnt_added = 0;
            return true;
        }
        auto [cur_ptr, rcnt_now] = m_asp->load_tagged_();
        if(cur_ptr != m_pref || rcnt_now == 0) {
            // ptr changed (swapper absorbed our +1) or tag drained.
            // Release our +1 (now in refcnt) + undo our pre-pay.
            sub_with_delete_check(rcnt_added + 1);
            m_pref = nullptr;
            m_tag_held = false;
            rcnt_added = 0;
            return true;
        }
        // TagHeld + ptr unchanged.  One-shot pre-pay LOCAL_REF_CAPACITY
        // (an upper bound — rcnt_now never exceeds CAPACITY-1, so
        // CAP-1 + slack always covers needed).  After the first entry,
        // rcnt_added stays at CAP and the branch is never re-entered.
        uintptr_t needed = rcnt_now - 1;
        if(needed > rcnt_added) {
            // This route fires at most once per loop.
            m_pref->refcnt.fetch_add(LOCAL_REF_CAPACITY,
                std::memory_order_relaxed);
            rcnt_added = LOCAL_REF_CAPACITY;
        }
        // Single-shot drain CAS: tag rcnt_now → 0.
        if(const_cast<atomic_shared_ptr<T> *>(m_asp)->m_ref.compare_set_weak(
            (uintptr_t)m_pref + rcnt_now,
            (uintptr_t)m_pref + 0)) {
            // Adjust for excess pre-pay.
            sub_with_delete_check(rcnt_added - needed);
            rcnt_added = 0;
            m_pref = nullptr;
            m_tag_held = false;
            return true;
        }
        return false;
    }

private:
    void release_() noexcept {
        if(m_pref) {
            if(m_tag_held) {
                (void)release_tagheld_zeroreset_(/*single_attempt=*/false);
            } else {
                // Owned → plain fetch_sub(1) + delete check.
                if(m_pref->refcnt.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    m_asp->deleter(m_pref);
                }
            }
            m_pref = nullptr;
            m_tag_held = false;
        }
    }

    atomic_shared_ptr<T> *m_asp;
    Ref *m_pref;
    bool m_tag_held;
    bool m_acquire_succeeded;

    template <typename Y> friend class atomic_shared_ptr;
};

template <typename T, class... Args>
local_shared_ptr<T> make_local_shared(Args&&... args) {
    return local_shared_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename T, class Alloc, class... Args>
local_shared_ptr<T> allocate_local_shared(Alloc &base_alloc, Args&&... args) {
    typename Alloc::template rebind<T>::other alloc(base_alloc);
    auto p = alloc.allocate(1);
    alloc.construct(p, std::forward<Args>(args)...);
    auto deleter = [alloc, p]() mutable {
        alloc.destroy(p);
        alloc.deallocate(p, 1);
    };
    return local_shared_ptr<T>(p, deleter);
}

template <typename T, typename reflocal_var_t>
inline local_shared_ptr<T, reflocal_var_t>::local_shared_ptr(const local_shared_ptr &y) noexcept {
    static_assert(sizeof(static_cast<const T*>(y.get())), "");
    this->m_ref = (TaggedPtr)y.m_ref;
    if(ref_ptr_())
        ref_ptr_()->refcnt.fetch_add(1, std::memory_order_relaxed);
}

template <typename T, typename reflocal_var_t>
template<typename Y, typename Z>
inline local_shared_ptr<T, reflocal_var_t>::local_shared_ptr(const local_shared_ptr<Y, Z> &y) noexcept {
    static_assert(sizeof(static_cast<const T*>(y.get())), "");
    this->m_ref = (TaggedPtr)y.m_ref;
    if(ref_ptr_())
        ref_ptr_()->refcnt.fetch_add(1, std::memory_order_relaxed);
}

template <typename T, typename reflocal_var_t>
inline local_shared_ptr<T, reflocal_var_t>::~local_shared_ptr() {
    reset();
}

template <typename T, typename reflocal_var_t>
inline void
local_shared_ptr<T, reflocal_var_t>::reset() noexcept {
    Ref *pref = ref_ptr_();
    if( !pref) return;
    // decreases global reference counter.
    if(unique()) {
        pref->refcnt.store(0, std::memory_order_relaxed);
        this->deleter(pref);
    }
    else if(pref->refcnt.decAndTest()) {
        this->deleter(pref);
    }
    this->m_ref = (TaggedPtr)nullptr;
}
//=============================================================================
// acquire_tag_ref_() — atomically read m_ref and increment the local refcount
//   (Comments by Claude Opus — based on source code analysis)
//
// The local refcount is embedded in the lower bits of the pointer (the bits
// guaranteed zero by allocator alignment). Incrementing it via CAS tells
// other threads "someone is observing this pointer — don't free the Ref yet"
// without touching the global (heap-allocated) reference counter.
//
// Invariant: local refcount < LOCAL_REF_CAPACITY. If the counter
// would overflow (extremely unlikely — requires that many concurrent readers),
// spin-wait until a slot opens.
//
// The caller MUST call release_tag_ref_() after it is done with the pointer.
//=============================================================================
template <typename T>
inline std::pair<typename atomic_shared_ptr<T>::Ref *, bool>
atomic_shared_ptr<T>::acquire_tag_ref_(Refcnt *rcnt, bool weakly) const noexcept {
    Ref *pref;
    Refcnt rcnt_new;
    for(int spins = 1;; spins *= 2) {
        auto [p, rcnt_old] = load_tagged_();
        pref = p;
        if( !pref) {
            // target is null.
            *rcnt = rcnt_old;
            return {(Ref*)nullptr, true};
        }
        rcnt_new = rcnt_old + 1u;
        /*
        static int rcnt_max = 0;
        if(rcnt_new > (int)rcnt_max) {
            rcnt_max = rcnt_new;
            fprintf(stderr, "max_rcnt=%d\n", rcnt_max);
        }
        */
        // Weak callers fail-fast on either overflow OR CAS-loss with
        // no pause (caller retries at a higher level).
        if(weakly) {
            if(rcnt_new < this->LOCAL_REF_CAPACITY
               && const_cast<atomic_shared_ptr<T> *>(this)->m_ref.compare_set_weak(
                   TaggedPtr((uintptr_t)pref + rcnt_old),
                   TaggedPtr((uintptr_t)pref + rcnt_new)))
                break;
            return {(Ref*)nullptr, false};
        }
        // Strong path: pause on overflow, exponential backoff on CAS loss.
        if(rcnt_new < this->LOCAL_REF_CAPACITY) {
            if(const_cast<atomic_shared_ptr<T> *>(this)->m_ref.compare_set_weak(
                TaggedPtr((uintptr_t)pref + rcnt_old),
                TaggedPtr((uintptr_t)pref + rcnt_new)))
                break;
        }
        else {
            pause4spin();
        }
#ifndef BACKOFF_IN_ATOMIC_SMART_PTR
        for(int i = 0; i < spins / BACKOFF_IN_ATOMIC_SMART_PTR; ++i)
            pause4spin(); //exponential backoff.
#else
        (void)spins;
#endif
    }
    assert(rcnt_new);
    *rcnt = rcnt_new;
    return {pref, true};
}
template <typename T>
inline typename atomic_shared_ptr<T>::Ref *
atomic_shared_ptr<T>::load_shared_() const noexcept {
    static_assert(load_shared_enabled<T>::value,
        "load_shared_ is disabled for this type; use scoped_atomic_view instead");
    Refcnt rcnt;
    auto [pref, success] = acquire_tag_ref_( &rcnt);
    if( !pref) return (Ref*)nullptr;
    // Transfer all rcnt tag refs to global at once (instead of just +1). The
    // matching release_tag_ref_(pref, rcnt) attempts to drain rcnt tag refs in
    // a single CAS, stealing from other threads' tag refs. Those threads then
    // fall back to fetch_sub(1) on refcnt on their own release_tag_ref_, which
    // shifts contention from m_ref (the hot tagged pointer) to refcnt (per
    // object). When rcnt=1 this is identical to the previous +1 / K=1 pattern.
    pref->refcnt.fetch_add(rcnt, std::memory_order_relaxed);
    release_tag_ref_(pref, rcnt);
    return pref;
}

// release_tag_ref_() — release the local reference acquired by acquire_tag_ref_().
// Tries to decrement the local refcount via CAS. If the pointer has been
// swapped out by another thread since acquire_tag_ref_(), the local counter
// is gone — fall back to decrementing the global refcount instead (the
// swapper transferred local counts to global before swapping).
//
// added_global_rcnt (default 1): total number of global refcount units that
//   the caller has pre-added to pref->refcnt on top of the 1 local ref being
//   released. Callers use this to batch-release excess refs in one shot:
//
//   compareAndSwap_ / compareAndSwapWeak_ / swap (CAS failure path):
//     Step 4 pre-added (rcnt_old - 1) to global. Pass added_global_rcnt=rcnt_old
//     to release that excess and our own local ref in a single operation.
//
//   load_shared_ (rcnt-bulk transfer):
//     fetch_add(rcnt) pre-adds rcnt to global (instead of the usual +1). Pass
//     added_global_rcnt=rcnt so the tag drain and global undo are consistent.
//
// Same-pointer CAS success path:
//   Drains local_release = min(rcnt_old, added_global_rcnt) from the tag.
//   Remaining excess = (added_global_rcnt - local_release) is undone from global
//   via fetch_sub(excess, acq_rel) with a delete check. MUST be acq_rel (not
//   relaxed): a concurrent local_reset can drop refcnt to exactly the excess
//   amount; our fetch_sub would then reach 0 and the deleter must fire.
//
// Pointer-changed path:
//   Combines excess undo + our own 1 ref into a single fetch_sub(added_global_rcnt,
//   acq_rel) with delete check — one fewer atomic op vs. the two-step old code.
template <typename T>
inline bool atomic_shared_ptr<T>::release_tag_ref_(Ref *pref, Refcnt added_global_rcnt,
                                                    bool single_attempt) const noexcept {
    Refcnt sub_amount = added_global_rcnt;
    for(int spins = 1;; spins *= 2) {
        auto [cur_ptr, rcnt_old] = load_tagged_();
        if(rcnt_old && (cur_ptr == pref)) {
            Refcnt local_release = std::min(rcnt_old, added_global_rcnt); //1 by default.
            Refcnt rcnt_new = rcnt_old - local_release;
            // trying to dec. reference counter if stored pointer is unchanged.
            if(const_cast<atomic_shared_ptr<T> *>(this)->m_ref.compare_set_weak(
                TaggedPtr((uintptr_t)pref + rcnt_old),
                TaggedPtr((uintptr_t)pref + rcnt_new))) {
                //decreases the rest of global counting.
                // CRITICAL: must be acq_rel + delete check, NOT relaxed.
                // Concurrent local_reset() can drop refcnt to (added_global_rcnt -
                // local_release), and our fetch_sub then takes it to 0. Without the
                // delete check the object leaks. Discovered via GenMC test 7
                sub_amount = added_global_rcnt - local_release;
                break;
            }
            // CAS lost.  single_attempt callers return false WITHOUT
            // doing the global fetch_sub — caller's pre-pay IOU stays
            // in pref->refcnt and is balanced by a later call.
            if(single_attempt)
                return false;
            auto [cur_ptr, rcnt_old] = load_tagged_();
            if((cur_ptr == pref) && rcnt_old) {
#ifndef BACKOFF_IN_ATOMIC_SMART_PTR
                for(int i = 0; i < spins / BACKOFF_IN_ATOMIC_SMART_PTR; ++i)
                    pause4spin(); //exponential backoff.
#else
                (void)spins;
#endif
                continue; // pointer unchanged, retry.
            }
        }
        // local reference has released by other processes.
        break;
    }
    if(sub_amount) {
        if(pref->refcnt.fetch_sub(sub_amount, std::memory_order_acq_rel) == sub_amount) {
            const_cast<atomic_shared_ptr*>(this)->deleter(pref);
        }
    }
    return true;
}

//=============================================================================
// compareAndSet_impl_<OldrT, WEAK, RETAIN_NEWR>() — unified atomic CAS on the shared pointer
//
// OldrT-driven dispatch via constexpr if:
//   - OldrT = const local_shared_ptr<T> (Set):
//       no acquire (oldr keeps pref alive); step4 = +T;
//       failure undo via plain fetch_sub(T, relaxed).
//   - OldrT = local_shared_ptr<T> (Swap, ACQUIRE):
//       acquire_tag_ref_() to pin pref while updating oldr on mismatch;
//       step4 = +(T-1); failure undo via release_tag_ref_(pref, T).
//   - OldrT = scoped_atomic_view<T> (SetScoped, WEAK only):
//       scoped already holds tag; step4 = +(T-1);
//       failure undo via plain fetch_sub(T-1, relaxed) (eager); on success,
//       scoped's tag is consumed by CAS (m_pref reset to nullptr).
//
// RETAIN_NEWR (SCOPED+WEAK only): on CAS success, scoped transitions to
//   Owned mode on newr instead of going Empty.  Entry does fetch_add(2)
//   instead of (1) — one for m_ref, one for scoped's Owned ref.  Failure
//   undo is fetch_sub(2) (same op count, different amount).  Useful when
//   the caller needs to keep tracking the new value after CAS (e.g.
//   bundle Phase 2 → Phase 4: scoped retains newr, eliminating reload).
//
// Common steps:
//   1. Pre-increment newr's global refcount (optimistic).
//   2. Read the current pointer (acquire_tag_ref_ or load_tagged_).
//   3. Pointer mismatch → undo + return false.
//   4. Pre-pay other tag holders via fetch_add(step4_amount).
//   5. CAS m_ref: (pref + rcnt_old) → (newr + 0).
//      On failure, undo step 4; retry (or return false if WEAK).
//   6. On success, decrement pref's global refcount (m_ref no longer owns it).
//=============================================================================
template <typename T>
template<typename OldrT, typename NewrT, bool WEAK, bool RETAIN_NEWR>
inline bool
atomic_shared_ptr<T>::compareAndSet_impl_(
    OldrT &oldr,
    NewrT &newr) noexcept {

    using OldrPlain = typename std::remove_cv<OldrT>::type;
    using NewrPlain = typename std::remove_cv<NewrT>::type;
    constexpr bool SCOPED = std::is_same<OldrPlain, scoped_atomic_view<T>>::value;
    constexpr bool ACQUIRE = !std::is_const<OldrT>::value && !SCOPED;
    constexpr bool UNIQUE = std::is_same<NewrPlain, local_unique_ptr<T>>::value;
    // SCOPED + STRONG: enabled for the privileged-thread fast path.
    // Privilege is exclusive (s_privileged_tidstamp slot) and fair_mode
    // blocks all other threads' CAS on this linkage, so the strong-spin
    // has no peer to contend with — guaranteed forward progress, no
    // livelock. Caller is responsible for invoking strong-mode only
    // when privileged.
    static_assert( !(RETAIN_NEWR && !SCOPED),
        "RETAIN_NEWR requires SCOPED (scoped_atomic_view oldr)");
    static_assert( !(RETAIN_NEWR && UNIQUE),
        "RETAIN_NEWR not supported with local_unique_ptr newr");

    auto oldr_pref = [&]() -> Ref* {
        if constexpr (SCOPED) return oldr.m_pref;
        else return oldr.ref_ptr_();
    };
    auto newr_pref = [&]() -> Ref* {
        // For local_unique_ptr<T> (intrusive only), Ref == T so
        // get() returns the same pointer ref_ptr_() would.
        if constexpr (UNIQUE) return (Ref *)newr.get();
        else return newr.ref_ptr_();
    };
    // RETAIN_NEWR adds +1 for scoped's Owned ref on newr after CAS success.
    constexpr Refcnt NEWR_ADD = RETAIN_NEWR ? 2u : 1u;
    auto new_refcnt_undo = [&newr]() {
        if constexpr ( !UNIQUE) {
            if(newr.ref_ptr_()) {
                if constexpr ( !RETAIN_NEWR) {
                    if(newr.use_count() == 2) //unique at start pt., and was +1.
                        { newr.ref_ptr_()->refcnt--; return; }
                }
                newr.ref_ptr_()->refcnt.fetch_sub(NEWR_ADD, std::memory_order_relaxed);
            }
        }
    };

    if constexpr ( !UNIQUE) {
        // Optimistic +NEWR_ADD for m_ref's implicit ref (+ scoped's Owned
        // ref when RETAIN_NEWR) on success; will undo on WEAK-failure or
        // pointer-mismatch.  UNIQUE skips this: newr's existing refcnt=1
        // transfers directly to m_ref on success.
        if(newr_pref()) {
            if constexpr ( !RETAIN_NEWR) {
                if(newr.unique())
                    { newr_pref()->refcnt++; }
                else
                    newr_pref()->refcnt.fetch_add(1, std::memory_order_relaxed);
            }
            else {
                newr_pref()->refcnt.fetch_add(NEWR_ADD, std::memory_order_relaxed);
            }
        }
    }
    for(int spins = 1;; spins *= 2) {
        Ref *pref;
        Refcnt rcnt_old;

        if constexpr (ACQUIRE) {
            auto [p, success] = acquire_tag_ref_( &rcnt_old, WEAK);
            if constexpr (WEAK) {
                if( !success) {
                    new_refcnt_undo();
                    return false;
                }
            }
            pref = p;
        } else {
            std::tie(pref, rcnt_old) = load_tagged_();
        }

        if(pref != oldr_pref()) {
            // pointer mismatch
            if constexpr (ACQUIRE) {
                if(pref) {
                    pref->refcnt.fetch_add(1, std::memory_order_relaxed);
                    release_tag_ref_(pref, 1u);
                }
            }
            new_refcnt_undo();
            if constexpr (ACQUIRE) {
                if(oldr.ref_ptr_()) {
                    // decreasing global reference counter.
                    if(oldr.ref_ptr_()->refcnt.decAndTest()) {
                        this->deleter(oldr.ref_ptr_());
                    }
                }
                oldr.m_ref = (uintptr_t)pref;
            } else if constexpr (SCOPED) {
                // For TagHeld: pointer changed since acquire; our tag was
                //   absorbed by the swapper (their step 4 pre-paid us +1).
                //   Eagerly clean up so scoped becomes Empty and the caller
                //   can detect "tag gone" via scoped.operator bool() == false.
                // For Owned: scoped still holds its +1 in OLD pref's refcnt.
                //   No special cleanup; caller sees scoped as still valid
                //   but the CAS returned false (caller may retry or destruct).
                if(oldr.m_tag_held) {
                    // TagHeld
                    release_tag_ref_(oldr.m_pref, 1u);
                    oldr.m_pref = nullptr;
                    oldr.m_tag_held = false;
                }
            }
            return false;
        }

        // step 4: pre-pay other tag holders.
        //
        //   - Swap (ACQUIRE): step4 = +(T-1) — own acquired tag is consumed
        //     by the CAS (no pre-pay needed for self).
        //   - Set: step4 = +T — caller's oldr keeps pref alive separately
        //     via its +1 in refcnt, so all T tag holders are external.
        //   - SCOPED: step4 = +T — treat scoped's tag as if it were already
        //     +1 in refcnt (because: in the ABSORBED case our CAS will
        //     consume scoped's tag along with others, and we owe -1 in the
        //     success path; in the DRAINED case some drainer already
        //     pre-paid scoped +1, and we likewise owe -1). Either way,
        //     a fetch_sub(2) at success consumes both m_ref's release and
        //     scoped's tag-share uniformly. This avoids needing to detect
        //     ABSORBED vs DRAINED at runtime.
        Refcnt step4_amount;
        if constexpr (ACQUIRE) {
            step4_amount = (rcnt_old > 1u) ? rcnt_old - 1u : 0u;
        } else {
            step4_amount = rcnt_old;
        }
        if(pref && step4_amount) {
            pref->refcnt.fetch_add(step4_amount, std::memory_order_relaxed);
        }

        // CAS m_ref: pref + rcnt_old → newr + 0
        Refcnt rcnt_new = 0;
        if(this->m_ref.compare_set_weak(
                TaggedPtr((uintptr_t)pref + rcnt_old),
                TaggedPtr((uintptr_t)newr_pref() + rcnt_new))) {
            if constexpr (UNIQUE) {
                // Transfer ownership: m_ref now holds the wrapper
                // with its existing refcnt=1.  Release the unique_ptr
                // so its destructor doesn't fetch_sub on it.
                newr.release();
            }
            if(pref) {
                // Release m_ref's implicit ownership.
                // For SCOPED in TagHeld mode, additionally consume scoped's
                // tag-share in the same fetch_sub (sub = 2). For Owned mode
                // and non-SCOPED, scoped's/oldr's +1 stays — sub = 1.
                //   TagHeld (refcnt >= 2 always at this point):
                //     ABSORBED: step4=+T pre-paid T including scoped →
                //       refcnt >= R_init + T - (T-1) = R_init + 1 >= 2.
                //     DRAINED:  drainer pre-paid +1 → R_init >= 2;
                //       step4=+T (T>=0) → refcnt >= 2.
                // RETAIN_NEWR + Owned: scoped is reassigned to newr below,
                //   so its Owned +1 on OLD pref must also be released here
                //   (sub = 2).  Without this, OLD pref's refcnt leaks +1
                //   per Owned-RETAIN call — the bug appears at low
                //   LOCAL_REF_CAPACITY where Owned mode is hit frequently
                //   (e.g., CAP=4 ADAPTIVE=2 → any rcnt>=2 acquire promotes).
                Refcnt sub = 1u;
                if constexpr (SCOPED) {
                    if(oldr.m_tag_held) sub = 2u;  // TagHeld
                    else if constexpr (RETAIN_NEWR) sub = 2u;  // Owned + RETAIN
                    // else Owned non-RETAIN: sub = 1 (scoped keeps OLD)
                }
                if(pref->refcnt.fetch_sub(sub, std::memory_order_acq_rel) == sub) {
                    const_cast<atomic_shared_ptr*>(this)->deleter(pref);
                }
            }
            if constexpr (SCOPED) {
                if constexpr (RETAIN_NEWR) {
                    // Transition scoped to Owned(newr).  The extra +1
                    // from NEWR_ADD at entry provides the Owned ref.
                    if(oldr.m_tag_held) oldr.m_tag_held = false;
                    oldr.m_pref = newr_pref();
                    // oldr is now Owned on newr.  Destructor will
                    // release_tag_ref_(newr, 1u) → fetch_sub(1).
                } else {
                    if(oldr.m_tag_held) {
                        // TagHeld: tag-share consumed via fetch_sub(2). Mark Empty.
                        oldr.m_pref = nullptr;
                        oldr.m_tag_held = false;
                    }
                    // Owned: scoped retains its +1 in pref's refcnt — but pref is
                    //   no longer in m_ref. Caller may still hold it as if from
                    //   compareAndSet on a const local_shared_ptr; destructor
                    //   eventually releases via fetch_sub(1).
                }
            }
            return true;
        }

        // CAS failure — undo step 4.
        if constexpr (ACQUIRE) {
            // Swap: batch undo via release_tag_ref_(pref, rcnt_old)
            //   = drain CAS for tag (rcnt_old refs) + global undo combined.
            if(pref) {
                assert(rcnt_old);
                release_tag_ref_(pref, rcnt_old);
            }
        } else {
            // Set / SCOPED: a held ref keeps pref alive (refcnt >= 2),
            //   so plain relaxed fetch_sub is safe.
            //   - Set:    caller's oldr provides the +1.
            //   - SCOPED: scoped's tag is still held (in m_ref's tag count
            //             OR pre-paid by an external drainer); destructor's
            //             release_tag_ref_(pref, 1u) handles cleanup.
            if(pref && step4_amount) {
                pref->refcnt.fetch_sub(step4_amount, std::memory_order_relaxed);
            }
        }
        if constexpr (WEAK) {
            // Roll back the optimistic newr fetch_add(1) from the entry of
            // this function — STRONG mode keeps it across retries, but WEAK
            // returns false without retry, so the +1 must be undone.
            // UNIQUE skips this: caller's unique_ptr keeps the wrapper;
            // its destructor handles cleanup if not retried.
            new_refcnt_undo();
            return false;
        }
#ifndef BACKOFF_IN_ATOMIC_SMART_PTR
        for(int i = 0; i < spins / BACKOFF_IN_ATOMIC_SMART_PTR; ++i)
            pause4spin(); //exponential backoff.
#else
        (void)spins;
#endif
    }
}

template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSwap(local_shared_ptr<T> &oldr, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<local_shared_ptr<T>, const local_shared_ptr<T>, false>(oldr, newr);
}
template <typename T>
bool
atomic_shared_ptr<T>::compareAndSet(const local_shared_ptr<T> &oldr, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<const local_shared_ptr<T>, const local_shared_ptr<T>, false>(oldr, newr);
}
template <typename T>
bool
atomic_shared_ptr<T>::compareAndSetWeak(const local_shared_ptr<T> &oldr, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<const local_shared_ptr<T>, const local_shared_ptr<T>, true>(oldr, newr);
}
template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSetWeak(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<scoped_atomic_view<T>, const local_shared_ptr<T>, true>(scoped, newr);
}
template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSetWeakRetain(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<scoped_atomic_view<T>, const local_shared_ptr<T>, true, true>(scoped, newr);
}
template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSetStrong(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newr) noexcept {
    // STRONG (WEAK=false): the impl_'s outer for-loop spins on weak CAS
    // failure.  Pointer mismatch (oldr.m_pref != m_ref's load) returns
    // false unconditionally — the only "real contention" exit.
    return compareAndSet_impl_<scoped_atomic_view<T>, const local_shared_ptr<T>, false, false>(scoped, newr);
}
template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSetStrongRetain(scoped_atomic_view<T> &scoped, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<scoped_atomic_view<T>, const local_shared_ptr<T>, false, true>(scoped, newr);
}

//! ----- local_unique_ptr<T> CAS variants (newr ownership transfer) -----
//! Save 2 atomic ops vs the local_shared_ptr<T> version: no fetch_add
//! at start, no fetch_sub on WEAK-failure undo or success.  newr is
//! in/out: on success it's released (m_ref takes ownership); on
//! failure it retains the wrapper (caller's destructor cleans up).
template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSet(const local_shared_ptr<T> &oldr, local_unique_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<const local_shared_ptr<T>, local_unique_ptr<T>, false>(oldr, newr);
}
template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSetWeak(const local_shared_ptr<T> &oldr, local_unique_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<const local_shared_ptr<T>, local_unique_ptr<T>, true>(oldr, newr);
}
template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSetWeak(scoped_atomic_view<T> &scoped, local_unique_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<scoped_atomic_view<T>, local_unique_ptr<T>, true>(scoped, newr);
}
template <typename T, typename reflocal_var_t>
inline void
local_shared_ptr<T, reflocal_var_t>::swap(local_shared_ptr &r) noexcept {
    TaggedPtr x = this->m_ref;
    this->m_ref = (TaggedPtr)r.m_ref;
    r.m_ref = x;
}

template <typename T, typename reflocal_var_t>
void
local_shared_ptr<T, reflocal_var_t>::swap(atomic_shared_ptr<T> &r) noexcept {
    for(int spins = 1;; spins *= 2) {
        Refcnt rcnt_old, rcnt_new;
        auto [pref, success] = r.acquire_tag_ref_( &rcnt_old);
        if(pref && (rcnt_old != 1u)) {
            pref->refcnt.fetch_add(rcnt_old - 1u, std::memory_order_relaxed);
        }
        rcnt_new = 0;
        if(r.m_ref.compare_set_weak(
            TaggedPtr((uintptr_t)pref + rcnt_old),
            TaggedPtr((uintptr_t)this->m_ref + rcnt_new))) {
            this->m_ref = (TaggedPtr)pref;
            return;
        }
        if(pref) {
            assert(rcnt_old);
            r.release_tag_ref_(pref, rcnt_old);
        }
#ifndef BACKOFF_IN_ATOMIC_SMART_PTR
        for(int i = 0; i < spins / BACKOFF_IN_ATOMIC_SMART_PTR; ++i)
            pause4spin(); //exponential backoff.
#else
        (void)spins;
#endif
    }
}

#endif /*ATOMIC_SMART_PTR_H_*/
