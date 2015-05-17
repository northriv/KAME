/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef ATOMIC_SMART_PTR_H_
#define ATOMIC_SMART_PTR_H_

#include "atomic_prv_basic.h"

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
    atomic_unique_ptr() : m_ptr(0) {}

    explicit atomic_unique_ptr(t_ptr t) : m_ptr(t) {}

    ~atomic_unique_ptr() {delete (t_ptr)m_ptr;}

	void reset(t_ptr t = 0) {
        t_ptr old = m_ptr.exchange(t);
		delete old;
	}
	//! \param[in,out] x \p x is atomically swapped.
	//! Nevertheless, this object is not atomically replaced.
	//! That is, the object pointed by "this" must not be shared among threads.
	void swap(atomic_unique_ptr &x) {
        m_ptr = x.m_ptr.exchange(m_ptr);
	}

    bool operator!() const {return !(t_ptr)m_ptr;}
    operator bool() const {return (t_ptr)m_ptr;}

	//! This function lacks thread-safety.
    T &operator*() const { assert((t_ptr)m_ptr); return (T &) *(t_ptr)m_ptr;}

	//! This function lacks thread-safety.
    t_ptr operator->() const { assert((t_ptr)m_ptr); return (t_ptr)m_ptr;}

	//! This function lacks thread-safety.
	t_ptr get() const { return (t_ptr )m_ptr;}
private:
	atomic_unique_ptr(const atomic_unique_ptr &) = delete;
	atomic_unique_ptr& operator=(const atomic_unique_ptr &) = delete;

    atomic<t_ptr> m_ptr;
};

//! This is an internal class holding a global reference counter and a pointer to the object.
//! \sa atomic_shared_ptr
template <typename T>
struct atomic_shared_ptr_gref_ {
    atomic_shared_ptr_gref_(T *p) : ptr(p), refcnt(1) {}
	~atomic_shared_ptr_gref_() { assert(refcnt == 0); delete ptr; }
	//! The pointer to the object.
	T *ptr;
	typedef uintptr_t Refcnt;
	//! The global reference counter.
    atomic<Refcnt> refcnt;
private:
	atomic_shared_ptr_gref_(const atomic_shared_ptr_gref_ &) = delete;
};

template <typename X, typename Y, typename Z, typename E> struct atomic_shared_ptr_base;
template <typename X> class atomic_shared_ptr;
template <typename X, typename Y> class local_shared_ptr;

//! Use subclass of this to be storaged in atomic_shared_ptr with
//! intrusive counting to obtain better performance.
struct atomic_countable {
    atomic_countable() : refcnt(1) {}
    atomic_countable(const atomic_countable &) : refcnt(1) {}
	~atomic_countable() { assert(refcnt == 0); }
private:
    template <typename X, typename Y, typename Z, typename E> friend struct atomic_shared_ptr_base;
	template <typename X> friend class atomic_shared_ptr;
    template <typename X, typename Y> friend class local_shared_ptr;
	atomic_countable& operator=(const atomic_countable &); //inhibited.
	typedef uintptr_t Refcnt;
	//! Global reference counter.
    atomic<Refcnt> refcnt;
};

//! \brief Base class for atomic_shared_ptr without intrusive counting, so-called "simple counted".\n
//! A global referece counter (an instance of atomic_shared_ptr_gref_) will be created.
template <typename T, typename reflocal_t, typename reflocal_var_t, typename Enable = void>
struct atomic_shared_ptr_base {
protected:
	typedef atomic_shared_ptr_gref_<T> Ref;
	typedef typename Ref::Refcnt Refcnt;

	static int deleter(Ref *p) { delete p; return 1; }

	//! can be used to initialize the internal pointer \a m_ref.
	//! \sa reset()
	template<typename Y> void reset_unsafe(Y *y) {
        m_ref = (reflocal_t)new Ref(static_cast<T*>(y));
	}
    T *get() { return this->m_ref ? ((Ref*)(reflocal_t)this->m_ref)->ptr : NULL; }
    const T *get() const { return this->m_ref ? ((const Ref*)(reflocal_t)this->m_ref)->ptr : NULL; }

    int _use_count_() const {return ((const Ref*)(reflocal_t)this->m_ref)->refcnt;}

    reflocal_var_t m_ref;
    enum {ATOMIC_SHARED_REF_ALIGNMENT = (sizeof(intptr_t))};
};
//! \brief Base class for atomic_shared_ptr with intrusive counting.
template <typename T, typename reflocal_t, typename reflocal_var_t>
struct atomic_shared_ptr_base<T, reflocal_t, reflocal_var_t, typename std::enable_if<std::is_base_of<atomic_countable, T>::value>::type > {
protected:
	typedef T Ref;
	typedef typename atomic_countable::Refcnt Refcnt;

	static int deleter(T *p) { delete p; return 1;}

	//! can be used to initialize the internal pointer \a m_ref.
	template<typename Y> void reset_unsafe(Y *y) {
        m_ref = (reflocal_t)static_cast<T*>(y);
	}
    T *get() { return (T*)(reflocal_t)this->m_ref; }
    const T *get() const { return (const T*)(reflocal_t)this->m_ref; }

    int _use_count_() const {return ((const T*)(reflocal_t)this->m_ref)->refcnt;}

    reflocal_var_t m_ref;
    enum {ATOMIC_SHARED_REF_ALIGNMENT = (sizeof(double))};
};

//! \brief This class provides non-reentrant interfaces for atomic_shared_ptr: operator->(), operator*() and so on.\n
//! Use this class in non-reentrant scopes instead of costly atomic_shared_ptr.
//! \sa atomic_shared_ptr, atomic_unique_ptr, atomic_shared_ptr_test.cpp.
template <typename T, typename reflocal_var_t = uintptr_t>
class local_shared_ptr : protected atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t> {
public:
	local_shared_ptr() { this->m_ref = 0; }

	template<typename Y> explicit local_shared_ptr(Y *y) { this->reset_unsafe(y); }

	local_shared_ptr(const atomic_shared_ptr<T> &t) { this->m_ref = reinterpret_cast<RefLocal_>(t.scan_()); }
	template<typename Y> local_shared_ptr(const atomic_shared_ptr<Y> &y) {
		static_assert(sizeof(static_cast<const T*>(y.get())), "");
		this->m_ref = reinterpret_cast<RefLocal_>(y.scan_());
	}
    inline local_shared_ptr(const local_shared_ptr<T, reflocal_var_t> &t);
    template<typename Y, typename Z> inline local_shared_ptr(const local_shared_ptr<Y, Z> &y);
	inline ~local_shared_ptr();

	local_shared_ptr &operator=(const local_shared_ptr &t) {
		local_shared_ptr(t).swap( *this);
		return *this;
	}
    template<typename Y, typename Z> local_shared_ptr &operator=(const local_shared_ptr<Y, Z> &y) {
		local_shared_ptr(y).swap( *this);
		return *this;
	}
	//! \param[in] t The pointer holded by this instance is replaced with that of \a t.
	local_shared_ptr &operator=(const atomic_shared_ptr<T> &t) {
		this->reset();
		this->m_ref = reinterpret_cast<RefLocal_>(t.scan_());
		return *this;
	}
	//! \param[in] y The pointer holded by this instance is replaced with that of \a y.
	template<typename Y> local_shared_ptr &operator=(const atomic_shared_ptr<Y> &y) {
		static_assert(sizeof(static_cast<const T*>(y.get())), "");
		this->reset();
		this->m_ref = reinterpret_cast<RefLocal_>(y.scan_());
		return *this;
	}

	//! \param[in,out] x \p The pointer holded by \a x is swapped with that of this instance.
	inline void swap(local_shared_ptr &x);
	//! \param[in,out] x \p The pointer holded by \a x is atomically swapped with that of this instance.
	void swap(atomic_shared_ptr<T> &x);

	//! The pointer holded by this instance is reset to null pointer.
	inline void reset();
	//! The pointer holded by this instance is reset with a pointer \a y.
	template<typename Y> void reset(Y *y) { reset(); this->reset_unsafe(y); }

    T *get() { return atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t>::get(); }
    const T *get() const { return atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t>::get(); }

	T &operator*() { assert( *this); return *get();}
	const T &operator*() const { assert( *this); return *get();}

	T *operator->() { assert( *this); return get();}
	const T *operator->() const { assert( *this); return get();}

	bool operator!() const {return !this->m_ref;}
	operator bool() const {return this->m_ref;}

    template<typename Y, typename Z> bool operator==(const local_shared_ptr<Y, Z> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
		return (this->pref_() == (const Ref *)x.pref_());}
    template<typename Y> bool operator==(const atomic_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (this->pref_() == (const Ref *)x.pref_());}
    template<typename Y, typename Z> bool operator!=(const local_shared_ptr<Y, Z> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
		return (this->pref_() != (const Ref *)x.pref_());}
    template<typename Y> bool operator!=(const atomic_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (this->pref_() != (const Ref *)x.pref_());}

	int use_count() const { return this->_use_count_();}
	bool unique() const {return use_count() == 1;}
protected:
    template <typename Y, typename Z> friend class local_shared_ptr;
	template <typename Y> friend class atomic_shared_ptr;
    typedef typename atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t>::Ref Ref;
    typedef typename atomic_shared_ptr_base<T, uintptr_t, reflocal_var_t>::Refcnt Refcnt;
    typedef uintptr_t RefLocal_;

	//! A pointer to global reference struct.
    Ref* pref_() const {return (Ref *)(RefLocal_)(this->m_ref);}
};

/*! \brief This is an atomic variant of \a std::shared_ptr, and can be shared by atomic and lock-free means.\n
 *
* \a atomic_shared_ptr can be shared among threads by the use of \a operator=(_target_), \a swap(_target_).
* An instance of \a atomic_shared_ptr<T> holds:\n
* 	a) a pointer to \a atomic_shared_ptr_gref_<T>, which is a struct. consisting of a pointer to a T-type object and a global reference counter.\n
* 	b) a local (temporary) reference counter, which is embedded in the above pointer by using several LSBs that should be usually zero.\n
* The values of a) and b), \a m_ref, are atomically handled with CAS machine codes.
* The purpose of b) the local reference counter is to tell the "observation" to the shared target before increasing the global reference counter.
* This process is implemented in \a reserve_scan_().\n
* A function \a leave_scan_() tries to decrease the local counter first. When it fails, the global counter is decreased.\n
* To swap the pointer and local reference counter (which will be reset to zero), the setter must adds the local counting to the global counter before swapping.
* \sa atomic_unique_ptr, local_shared_ptr, atomic_shared_ptr_test.cpp.
 */
template <typename T>
class atomic_shared_ptr : protected local_shared_ptr<T, atomic<uintptr_t>> {
public:
    atomic_shared_ptr() : local_shared_ptr<T, atomic<uintptr_t>>() {}

    template<typename Y> explicit atomic_shared_ptr(Y *y) : local_shared_ptr<T, atomic<uintptr_t>>(y) {}
    atomic_shared_ptr(const atomic_shared_ptr<T> &t) : local_shared_ptr<T, atomic<uintptr_t>>(t) {}
    template<typename Y> atomic_shared_ptr(const atomic_shared_ptr<Y> &y) : local_shared_ptr<T, atomic<uintptr_t>>(y) {}
    atomic_shared_ptr(const local_shared_ptr<T> &t) : local_shared_ptr<T, atomic<uintptr_t>>(t) {}
    template<typename Y> atomic_shared_ptr(const local_shared_ptr<Y> &y) : local_shared_ptr<T, atomic<uintptr_t>>(y) {}

    ~atomic_shared_ptr() {}

	//! \param[in] t The pointer holded by this instance is atomically replaced with that of \a t.
	atomic_shared_ptr &operator=(const atomic_shared_ptr &t) {
		local_shared_ptr<T>(t).swap( *this);
		return *this;
	}
	//! \param[in] y The pointer holded by this instance is atomically replaced with that of \a y.
	template<typename Y> atomic_shared_ptr &operator=(const local_shared_ptr<Y> &y) {
		local_shared_ptr<T>(y).swap( *this);
		return *this;
	}
	//! \param[in] y The pointer holded by this instance is atomically replaced with that of \a y.
	template<typename Y> atomic_shared_ptr &operator=(const atomic_shared_ptr<Y> &y) {
		local_shared_ptr<T>(y).swap( *this);
		return *this;
	}
	//! The pointer holded by this instance is atomically reset to null pointer.
	void reset() {
		local_shared_ptr<T>().swap( *this);
	}
	//! The pointer holded by this instance is atomically reset with a pointer \a y.
	template<typename Y> void reset(Y *y) {
		local_shared_ptr<T>(y).swap( *this);
	}

	//! \return true if succeeded.
	//! \sa compareAndSwap()
	bool compareAndSet(const local_shared_ptr<T> &oldvalue, const local_shared_ptr<T> &newvalue);
	//! \return true if succeeded.
	//! \sa compareAndSet()
	bool compareAndSwap(local_shared_ptr<T> &oldvalue, const local_shared_ptr<T> &newvalue);

    bool operator!() const {return !this->m_ref;}
    operator bool() const {return this->m_ref;}

	template<typename Y> bool operator==(const local_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (pref_() == (const Ref*)x.pref_());}
	template<typename Y> bool operator==(const atomic_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (pref_() == (const Ref*)x.pref_());}
	template<typename Y> bool operator!=(const local_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (pref_() != (const Ref*)x.pref_());}
	template<typename Y> bool operator!=(const atomic_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
        return (pref_() != (const Ref*)x.pref_());}
protected:
    template <typename Y, typename Z> friend class local_shared_ptr;
	template <typename Y> friend class atomic_shared_ptr;
    typedef typename atomic_shared_ptr_base<T, uintptr_t, atomic<uintptr_t>>::Ref Ref;
    typedef typename atomic_shared_ptr_base<T, uintptr_t, atomic<uintptr_t>>::Refcnt Refcnt;
    typedef atomic<uintptr_t> RefLocal_;
	//! A pointer to global reference struct.
	Ref* pref_() const {return (Ref*)(this->m_ref & (~(uintptr_t)(this->ATOMIC_SHARED_REF_ALIGNMENT - 1)));}
	//! Local (temporary) reference counter.
	//! Local reference counter is a trick to tell the observation to other threads.
	Refcnt refcnt_() const {return (Refcnt)(this->m_ref & (uintptr_t)(this->ATOMIC_SHARED_REF_ALIGNMENT - 1));}

	//internal functions below.
	//! Atomically scans \a m_ref and increases the global reference counter.
	//! \a scan_() is used for atomically coping the pointer.
	inline Ref *scan_() const;
	//! Atomically scans \a m_ref and increases the  local (temporary) reference counter.
	//! use \a leave_scan_() to release the temporary reference.
	inline Ref *reserve_scan_(Refcnt *) const;
	//! Tries to decrease local (temporary) reference counter.
	//! In case the reference is lost, \a leave_scan_() releases the global reference counter instead.
	inline void leave_scan_(Ref *) const;

	template <bool NOSWAP>
	inline bool compareAndSwap_(local_shared_ptr<T> &oldvalue, const local_shared_ptr<T> &newvalue);
private:
};

template <typename T, typename reflocal_var_t>
inline local_shared_ptr<T, reflocal_var_t>::local_shared_ptr(const local_shared_ptr &y) {
	static_assert(sizeof(static_cast<const T*>(y.get())), "");
    this->m_ref = (RefLocal_)y.m_ref;
	if(pref_())
        ++(pref_()->refcnt); //atomic
}

template <typename T, typename reflocal_var_t>
template<typename Y, typename Z>
inline local_shared_ptr<T, reflocal_var_t>::local_shared_ptr(const local_shared_ptr<Y, Z> &y) {
	static_assert(sizeof(static_cast<const T*>(y.get())), "");
    this->m_ref = (RefLocal_)y.m_ref;
	if(pref_())
        ++(pref_()->refcnt); //atomic
}

template <typename T, typename reflocal_var_t>
inline local_shared_ptr<T, reflocal_var_t>::~local_shared_ptr() {
	reset();
}

template <typename T, typename reflocal_var_t>
inline void
local_shared_ptr<T, reflocal_var_t>::reset() {
	Ref *pref = pref_();
	if( !pref) return;
    // decreases global reference counter.
	if(unique()) {
        pref->refcnt = 0;
		this->deleter(pref);
	}
    else if(pref->refcnt.decAndTest()) {
		this->deleter(pref);
	}
	this->m_ref = 0;
}
template <typename T>
inline typename atomic_shared_ptr<T>::Ref *
atomic_shared_ptr<T>::reserve_scan_(Refcnt *rcnt) const {
	Ref *pref;
	Refcnt rcnt_new;
	for(;;) {
		pref = pref_();
		Refcnt rcnt_old;
		rcnt_old = refcnt_();
		if( !pref) {
			// target is null.
			*rcnt = rcnt_old;
			return 0;
		}
		rcnt_new = rcnt_old + 1u;
		/*
		static int rcnt_max = 0;
		if(rcnt_new > (int)rcnt_max) {
			rcnt_max = rcnt_new;
			fprintf(stderr, "max_rcnt=%d\n", rcnt_max);
		}
		*/
		if(rcnt_new >= this->ATOMIC_SHARED_REF_ALIGNMENT) {
			// This would never happen.
            pause4spin();
			continue;
		}
		// trying to increase local reference counter w/ same serial.
        if(const_cast<atomic_shared_ptr<T> *>(this)->m_ref.compare_set_strong(
            RefLocal_((uintptr_t)pref + rcnt_old),
            RefLocal_((uintptr_t)pref + rcnt_new)))
			break;
	}
	assert(rcnt_new);
	*rcnt = rcnt_new;
	return pref;
}
template <typename T>
inline typename atomic_shared_ptr<T>::Ref *
atomic_shared_ptr<T>::scan_() const {
	Refcnt rcnt;
	Ref *pref = reserve_scan_( &rcnt);
	if( !pref) return 0;
    ++(pref->refcnt); //atomic
	leave_scan_(pref);
	return pref;
}

template <typename T>
inline void
atomic_shared_ptr<T>::leave_scan_(Ref *pref) const {
	for(;;) {
		Refcnt rcnt_old;
		rcnt_old = refcnt_();
		if(rcnt_old) {
			Refcnt rcnt_new = rcnt_old - 1;
			// trying to dec. reference counter if stored pointer is unchanged.
            if(const_cast<atomic_shared_ptr<T> *>(this)->m_ref.compare_set_strong(
				RefLocal_((uintptr_t)pref + rcnt_old),
                RefLocal_((uintptr_t)pref + rcnt_new)))
				break;
			if(pref == pref_())
				continue; // try again.
		}
		// local reference has released by other processes.
        if(pref->refcnt.decAndTest()) {
			this->deleter(pref);
		}
		break;
	}
}

template <typename T>
template <bool NOSWAP>
inline bool
atomic_shared_ptr<T>::compareAndSwap_(local_shared_ptr<T> &oldr, const local_shared_ptr<T> &newr) {
	Ref *pref;
	if(newr.pref_()) {
        ++(newr.pref_()->refcnt); //atomic
	}
	for(;;) {
		Refcnt rcnt_old, rcnt_new;
		pref = reserve_scan_( &rcnt_old);
		if(pref != oldr.pref_()) {
			if(pref) {
				if( !NOSWAP) {
                    ++(pref->refcnt);//atomic
				}
				leave_scan_(pref);
			}
			if(newr.pref_())
                --(newr.pref_()->refcnt); //atomic
			if( !NOSWAP) {
				if(oldr.pref_()) {
					// decreasing global reference counter.
                    if(oldr.pref_()->refcnt.decAndTest()) {
						this->deleter(oldr.pref_());
					}
				}
                oldr.m_ref = (uintptr_t)pref;
			}
			return false;
		}
		if(pref && (rcnt_old != 1u)) {
            pref->refcnt += rcnt_old - 1u; //atomic
		}
		rcnt_new = 0;
        if(this->m_ref.compare_set_strong(
			RefLocal_((uintptr_t)pref + rcnt_old),
            RefLocal_((uintptr_t)newr.pref_() + rcnt_new)))
			break;
		if(pref) {
			assert(rcnt_old);
			if(rcnt_old != 1u)
                pref->refcnt += (Refcnt)( -(int)(rcnt_old - 1u)); //atomic
			leave_scan_(pref);
		}
	}
	if(pref) {
        --(pref->refcnt); //atomic
	}
	return true;
}
template <typename T>
bool
atomic_shared_ptr<T>::compareAndSet(const local_shared_ptr<T> &oldr, const local_shared_ptr<T> &newr) {
	return compareAndSwap_<true>(const_cast<local_shared_ptr<T> &>(oldr), newr);
}
template <typename T>
bool
atomic_shared_ptr<T>::compareAndSwap(local_shared_ptr<T> &oldr, const local_shared_ptr<T> &newr) {
	return compareAndSwap_<false>(oldr, newr);
}
template <typename T, typename reflocal_var_t>
inline void
local_shared_ptr<T, reflocal_var_t>::swap(local_shared_ptr &r) {
	RefLocal_ x = this->m_ref;
    this->m_ref = (RefLocal_)r.m_ref;
	r.m_ref = x;
}

template <typename T, typename reflocal_var_t>
void
local_shared_ptr<T, reflocal_var_t>::swap(atomic_shared_ptr<T> &r) {
	Ref *pref;
	for(;;) {
		Refcnt rcnt_old, rcnt_new;
		pref = r.reserve_scan_( &rcnt_old);
		if(pref && (rcnt_old != 1u)) {
            pref->refcnt += rcnt_old - 1u; //atomic
		}
		rcnt_new = 0;
        if(r.m_ref.compare_set_strong(
			RefLocal_((uintptr_t)pref + rcnt_old),
            RefLocal_((uintptr_t)this->m_ref + rcnt_new)))
			break;
		if(pref) {
			assert(rcnt_old);
			if(rcnt_old != 1u)
                pref->refcnt += (Refcnt)( -(int)(rcnt_old - 1u)); //atomic
			r.leave_scan_(pref);
		}
	}
	this->m_ref = (RefLocal_)pref;
}

#endif /*ATOMIC_SMART_PTR_H_*/
