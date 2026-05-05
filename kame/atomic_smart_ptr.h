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
#include <assert.h>

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
template <typename X> class scoped_local_shared_ptr;

//! Use subclass of this to be storaged in atomic_shared_ptr with
//! intrusive counting to obtain better performance.
struct atomic_countable {
    atomic_countable() noexcept : refcnt(1) {}
    atomic_countable(const atomic_countable &) noexcept : refcnt(1) {}
    ~atomic_countable() { assert(refcnt == 0); }

    atomic_countable& operator=(const atomic_countable &) = delete;
private:
    template <typename X, typename Y, typename Z, typename E> friend struct atomic_shared_ptr_base;
    template <typename X> friend class atomic_shared_ptr;
    template <typename X, typename Y> friend class local_shared_ptr;
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
    enum {LOCAL_REF_CAPACITY = (sizeof(intptr_t))};
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
    enum {LOCAL_REF_CAPACITY = (sizeof(double))};
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
protected:
    template <typename Y, typename Z> friend class local_shared_ptr;
    template <typename Y> friend class atomic_shared_ptr;
    template <typename Y> friend class scoped_local_shared_ptr;
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
    inline void release_tag_ref_(Ref *, Refcnt added_global_rcnt) const noexcept;

    //! Unified CAS template covering compareAndSet / compareAndSetWeak
    //! (ACQUIRE = false: caller's oldr keeps pref alive, no acquire_tag_ref_
    //!   needed; step4 = +T; failure undo via plain fetch_sub) and
    //! compareAndSwap (ACQUIRE = true: oldr may be updated on mismatch,
    //!   acquire_tag_ref_ required to pin pref; step4 = +(T-1); failure undo
    //!   via release_tag_ref_(pref, rcnt_old)).
    template<bool ACQUIRE, bool WEAK = false>
    inline bool compareAndSet_impl_(
        typename std::conditional<ACQUIRE,
            local_shared_ptr<T>&,
            const local_shared_ptr<T>&>::type oldr,
        const local_shared_ptr<T> &newr) noexcept;
private:
};

//! \brief RAII scoped tag holder on \a atomic_shared_ptr<T>.
//!
//! Acquires a tag ref on the supplied atomic_shared_ptr's m_ref in the
//! constructor (1 CAS) and releases it on destruction. Move-only.
//!
//! Two states: Empty (m_pref == nullptr) and TagHeld (m_pref != nullptr).
//!
//! Usage:
//!   - Promote to local_shared_ptr<T> via the rvalue-only conversion
//!     \a operator local_shared_ptr<T>() &&. The conversion is bit-identical
//!     to \a atomic_shared_ptr::load_shared_(): fetch_add(rcnt) on
//!     pref->refcnt + release_tag_ref_(pref, rcnt). After conversion the
//!     scoped is Empty (destructor no-op).
//!   - Phase 2 will add \a compareAndSetWeak(newr) using the held tag
//!     directly (no fresh acquire). On success, m_pref is cleared (Consumed
//!     state encoded as Empty). On failure, the scoped remains TagHeld and
//!     can be reused for retry.
template <typename T>
class scoped_local_shared_ptr {
public:
    typedef typename atomic_shared_ptr<T>::Ref Ref;
    typedef typename atomic_shared_ptr<T>::Refcnt Refcnt;

    scoped_local_shared_ptr() noexcept
        : m_asp(nullptr), m_pref(nullptr), m_rcnt_at_acquire(0) {}

    //! Acquires a tag ref on \a asp.m_ref. On null target / weakly-fail,
    //! constructs an Empty instance.
    explicit scoped_local_shared_ptr(atomic_shared_ptr<T> &asp) noexcept
        : m_asp(&asp), m_pref(nullptr), m_rcnt_at_acquire(0) {
        Refcnt rcnt;
        auto [p, ok] = asp.acquire_tag_ref_( &rcnt);
        (void)ok;
        if(p) {
            m_pref = p;
            m_rcnt_at_acquire = rcnt;
        }
    }

    scoped_local_shared_ptr(scoped_local_shared_ptr &&other) noexcept
        : m_asp(other.m_asp), m_pref(other.m_pref),
          m_rcnt_at_acquire(other.m_rcnt_at_acquire) {
        other.m_pref = nullptr;
        other.m_rcnt_at_acquire = 0;
    }
    scoped_local_shared_ptr &operator=(scoped_local_shared_ptr &&other) noexcept {
        if(this != &other) {
            release_();
            m_asp = other.m_asp;
            m_pref = other.m_pref;
            m_rcnt_at_acquire = other.m_rcnt_at_acquire;
            other.m_pref = nullptr;
            other.m_rcnt_at_acquire = 0;
        }
        return *this;
    }
    scoped_local_shared_ptr(const scoped_local_shared_ptr &) = delete;
    scoped_local_shared_ptr &operator=(const scoped_local_shared_ptr &) = delete;

    ~scoped_local_shared_ptr() noexcept { release_(); }

    //! \brief Promote to local_shared_ptr<T> by transferring the tag ref to
    //!   the global refcount. Bit-identical to load_shared_().
    //! \note rvalue-only — the scoped instance becomes Empty after this.
    operator local_shared_ptr<T>() && noexcept {
        local_shared_ptr<T> ret;
        if(m_pref) {
            // Promote: rcnt tag refs → global, then drain tag.
            m_pref->refcnt.fetch_add(m_rcnt_at_acquire, std::memory_order_relaxed);
            m_asp->release_tag_ref_(m_pref, m_rcnt_at_acquire);
            ret.m_ref = (uintptr_t)m_pref;
            m_pref = nullptr;
            m_rcnt_at_acquire = 0;
        }
        return ret;
    }

    bool operator!() const noexcept { return m_pref == nullptr; }
    explicit operator bool() const noexcept { return m_pref != nullptr; }

    Ref *ref_ptr_() const noexcept { return m_pref; }
    Refcnt rcnt_at_acquire_() const noexcept { return m_rcnt_at_acquire; }

private:
    void release_() noexcept {
        if(m_pref) {
            // TagHeld → release_tag_ref_ (drain CAS or fetch_sub on pointer-changed).
            m_asp->release_tag_ref_(m_pref, 1u);
            m_pref = nullptr;
            m_rcnt_at_acquire = 0;
        }
    }

    atomic_shared_ptr<T> *m_asp;
    Ref *m_pref;
    Refcnt m_rcnt_at_acquire;

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
        if(rcnt_new < this->LOCAL_REF_CAPACITY) { // This would never happen in typical machines.
            // trying to increase local reference counter w/ same serial.
            if(const_cast<atomic_shared_ptr<T> *>(this)->m_ref.compare_set_weak(
                TaggedPtr((uintptr_t)pref + rcnt_old),
                TaggedPtr((uintptr_t)pref + rcnt_new)))
                break;
        }
        else {
            pause4spin();
        }
        if(weakly)
            return {(Ref*)nullptr, false};
#ifndef BACKOFF_IN_ATOMIC_SMART_PTR
        for(int i = 0; i < spins / BACKOFF_IN_ATOMIC_SMART_PTR; ++i)
            pause4spin(); //exponential backoff.
#else
        spins;
#endif
    }
    assert(rcnt_new);
    *rcnt = rcnt_new;
    return {pref, true};
}
template <typename T>
inline typename atomic_shared_ptr<T>::Ref *
atomic_shared_ptr<T>::load_shared_() const noexcept {
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
inline void atomic_shared_ptr<T>::release_tag_ref_(Ref *pref, Refcnt added_global_rcnt) const noexcept {
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
            auto [cur_ptr, rcnt_old] = load_tagged_();
            if((cur_ptr == pref) && rcnt_old) {
#ifndef BACKOFF_IN_ATOMIC_SMART_PTR
                for(int i = 0; i < spins / BACKOFF_IN_ATOMIC_SMART_PTR; ++i)
                    pause4spin(); //exponential backoff.
#else
                spins;
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
}

//=============================================================================
// compareAndSet_impl_<MODE, WEAK>() — unified atomic CAS on the shared pointer
//
// Steps (common to Set and Swap):
//   1. Pre-increment newr's global refcount (optimistic).
//   2. Read the current pointer:
//      - Set:  load_tagged_() (no acquire — oldr keeps pref alive).
//      - Swap: acquire_tag_ref_() (must pin pref since we may update oldr).
//   3. If current != oldr: CAS fails.
//      - Set:  just undo newr's pre-increment; return false.
//      - Swap: release the acquired tag, transfer ref to oldr, release
//              oldr's old pref, update oldr to current.
//   4. Pre-pay other tag holders:
//      - Set:  step4 = +T (no implicit acquire ref consumed).
//      - Swap: step4 = +(T-1) (own acquired tag is consumed by CAS).
//   5. CAS m_ref: replace (pref + rcnt_old) with (newr + 0).
//      On failure, undo step 4 and retry (or return false if WEAK).
//   6. On success, decrement pref's global refcount (m_ref no longer owns it).
//=============================================================================
template <typename T>
template<bool ACQUIRE, bool WEAK>
inline bool
atomic_shared_ptr<T>::compareAndSet_impl_(
    typename std::conditional<ACQUIRE,
        local_shared_ptr<T>&,
        const local_shared_ptr<T>&>::type oldr,
    const local_shared_ptr<T> &newr) noexcept {

    if(newr.ref_ptr_()) {
        newr.ref_ptr_()->refcnt.fetch_add(1, std::memory_order_relaxed);
    }
    for(int spins = 1;; spins *= 2) {
        Ref *pref;
        Refcnt rcnt_old;

        if constexpr (ACQUIRE) {
            auto [p, success] = acquire_tag_ref_( &rcnt_old, WEAK);
            if constexpr (WEAK) {
                if( !success) {
                    if(newr.ref_ptr_())
                        newr.ref_ptr_()->refcnt.fetch_sub(1, std::memory_order_relaxed);
                    return false;
                }
            }
            pref = p;
        } else {
            std::tie(pref, rcnt_old) = load_tagged_();
        }

        if(pref != oldr.ref_ptr_()) {
            // pointer mismatch
            if constexpr (ACQUIRE) {
                if(pref) {
                    pref->refcnt.fetch_add(1, std::memory_order_relaxed);
                    release_tag_ref_(pref, 1u);
                }
            }
            if(newr.ref_ptr_())
                newr.ref_ptr_()->refcnt.fetch_sub(1, std::memory_order_relaxed);
            if constexpr (ACQUIRE) {
                if(oldr.ref_ptr_()) {
                    // decreasing global reference counter.
                    if(oldr.ref_ptr_()->refcnt.decAndTest()) {
                        this->deleter(oldr.ref_ptr_());
                    }
                }
                oldr.m_ref = (uintptr_t)pref;
            }
            return false;
        }

        // step 4: pre-pay other tag holders.
        // Set:  +T  (no implicit ref consumed by CAS — caller's oldr keeps pref alive)
        // Swap: +(T-1) (our own acquired tag is consumed by CAS)
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
                TaggedPtr((uintptr_t)newr.ref_ptr_() + rcnt_new))) {
            if(pref) {
                pref->refcnt.fetch_sub(1, std::memory_order_acq_rel); //atomic: release m_ref's ownership
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
            // Set: caller's oldr keeps pref alive (refcnt >= 2 always),
            //   so plain relaxed fetch_sub is safe (cannot underflow nor reach 0).
            if(pref && step4_amount) {
                pref->refcnt.fetch_sub(step4_amount, std::memory_order_relaxed);
            }
        }
        if constexpr (WEAK) return false;
#ifndef BACKOFF_IN_ATOMIC_SMART_PTR
        for(int i = 0; i < spins / BACKOFF_IN_ATOMIC_SMART_PTR; ++i)
            pause4spin(); //exponential backoff.
#else
        spins;
#endif
    }
}

template <typename T>
inline bool
atomic_shared_ptr<T>::compareAndSwap(local_shared_ptr<T> &oldr, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<true, false>(oldr, newr);
}
template <typename T>
bool
atomic_shared_ptr<T>::compareAndSet(const local_shared_ptr<T> &oldr, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<false, false>(oldr, newr);
}
template <typename T>
bool
atomic_shared_ptr<T>::compareAndSetWeak(const local_shared_ptr<T> &oldr, const local_shared_ptr<T> &newr) noexcept {
    return compareAndSet_impl_<false, true>(oldr, newr);
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
        spins;
#endif
    }
}

#endif /*ATOMIC_SMART_PTR_H_*/
