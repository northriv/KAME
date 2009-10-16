/***************************************************************************
		Copyright (C) 2002-2009 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

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

#include <atomic.h>


//! This is an atomic variant of boost::scoped_ptr<>.
//! atomic_scoped_ptr<> can be shared among threads by the use of swap() as the argument.
//! That is, a destructive read. Use atomic_shared_ptr<> for non-destructive reading.
//! The implementation relies on an atimic-swap machine code, e.g. lock xchg.
//! \sa atomic_shared_ptr
template <typename T>
class atomic_scoped_ptr
{
	typedef T* t_ptr;
public:
	atomic_scoped_ptr() : m_ptr(0) {
	}

	explicit atomic_scoped_ptr(t_ptr t) : m_ptr(t) {
		writeBarrier();
	}

	~atomic_scoped_ptr() { readBarrier(); delete m_ptr;}

	void reset(t_ptr t = 0) {
		writeBarrier();
		t_ptr old = atomicSwap(t, &m_ptr);
		readBarrier();
		delete old;
	}
	//! \param x \p x is atomically swapped.
	//! Nevertheless, this object is not atomically replaced.
	//! That is, the object pointed by "this" must not be shared among threads.
	void swap(atomic_scoped_ptr &x) {
		writeBarrier();
		m_ptr = atomicSwap(m_ptr, &x.m_ptr);
		readBarrier();
	}

	//! These functions must be called while writing is blocked.
	T &operator*() const { ASSERT(m_ptr); return (T&)*m_ptr;}

	t_ptr operator->() const { ASSERT(m_ptr); return (t_ptr)m_ptr;}

	t_ptr get() const {
		return (t_ptr )m_ptr;
	}
private:
	atomic_scoped_ptr(const atomic_scoped_ptr &) {}
	atomic_scoped_ptr& operator=(const atomic_scoped_ptr &) {return *this;}

	t_ptr m_ptr;
};

#define ATOMIC_SHARED_REF_ALIGNMENT 8

//! This class holds a global reference counter and a pointer to the object.
template <typename T>
struct atomic_shared_ptr_ref {
	template <class Y>
	atomic_shared_ptr_ref(Y *p) : ptr(p), refcnt(1) {}
	~atomic_shared_ptr_ref() { ASSERT(refcnt == 0); delete ptr; }
	T *ptr;
	//! Global reference counter.
	typedef uintptr_t Refcnt;
	Refcnt refcnt;
} __attribute__((aligned(ATOMIC_SHARED_REF_ALIGNMENT)));;


//! This is an atomic variant of boost::shared_ptr<>.
//! atomic_shared_ptr<> can be shared among threads by the use of operator=(), swap() as the argument.
//! \sa atomic_scoped_ptr
template <typename T>
class atomic_shared_ptr
{
public:
	typedef atomic_shared_ptr_ref<T> Ref;

	atomic_shared_ptr() {
		m_ref = 0;
		writeBarrier();
	}

	template<typename Y> explicit atomic_shared_ptr(Y *y) {
		m_ref = (_RefLocal)new Ref(y);
		writeBarrier();
	}

	atomic_shared_ptr(const atomic_shared_ptr &t) {
		m_ref = (_RefLocal)t._scan_();
		writeBarrier();
	}
	template<typename Y> atomic_shared_ptr(const atomic_shared_ptr<Y> &y) {
		m_ref = (_RefLocal)(typename atomic_shared_ptr::Ref*)y._scan_();
		writeBarrier();
	}

	~atomic_shared_ptr();

	//! \param t This object is atomically replaced with \a t.
	atomic_shared_ptr &operator=(const atomic_shared_ptr &t) {
		atomic_shared_ptr(t).swap(*this);
		return *this;
	}
	//! \param y This object is atomically replaced with \a t.
	template<typename Y> atomic_shared_ptr &operator=(const atomic_shared_ptr<Y> &y) {
		atomic_shared_ptr(y).swap(*this);
		return *this;
	}
	//! This object is atomically reset.
	void reset() {
		atomic_shared_ptr().swap(*this);
	}
	//! \param y This object is atomically reset with a pointer \a y.
	template<typename Y> void reset(Y *y) {
		atomic_shared_ptr(y).swap(*this);
	}

	//! \param x \p x is atomically swapped.
	//! Nevertheless, this object is not atomically replaced.
	//! That is, the object pointed by "this" must not be shared among threads.
	void swap(atomic_shared_ptr &);

	//! \return true if succeeded.
	bool compareAndSwap(const atomic_shared_ptr &oldr, atomic_shared_ptr &r);

	//! These functions must be called while writing is blocked.
	T *get() const { Ref *pref = _pref(); return pref ? pref->ptr : 0L; }

	T &operator*() const { ASSERT(*this); return *get();}

	T *operator->() const { ASSERT(*this); return get();}

	bool operator!() const {return !m_ref;}
	operator bool() const {return m_ref;}

public:
	typedef uintptr_t Refcnt;
	//! internal functions below.
	//! atomically scan \a m_ref and increase global reference counter.
	//! \a _scan_ is used for atomically coping the pointer.
	Ref *_scan_() const;
	//! atomically scan \a m_ref and increase local (temporary) reference counter.
	//! use \a _leave_scan_ to release the temporary reference.
	Ref *_reserve_scan_(Refcnt *) const;
	//! try to decrease local (temporary) reference counter.
	//! In case the reference is lost, \a _leave_scan_ releases the global reference counter instead.
	void _leave_scan_(Ref *) const;
private:
	typedef uintptr_t _RefLocal;
	mutable _RefLocal m_ref;
	//! A pointer to global reference struct.
	Ref* _pref() const {return (Ref*)(m_ref & (~(uintptr_t)(ATOMIC_SHARED_REF_ALIGNMENT - 1)));}
	//! Local (temporary) reference counter.
	//! Local reference counter is a trick to tell the observation to other threads.
	Refcnt _refcnt() const {return (Refcnt)(m_ref & (uintptr_t)(ATOMIC_SHARED_REF_ALIGNMENT - 1));}
};

template <typename T>
atomic_shared_ptr<T>::~atomic_shared_ptr() {
	readBarrier();
	ASSERT(_refcnt() == 0);
	Ref *pref = _pref();
	if(!pref) return;
	// decrease global reference counter.
	readBarrier();
	if(atomicDecAndTest(&pref->refcnt)) {
		delete pref;
	}
}
template <typename T>
typename atomic_shared_ptr<T>::Ref *
atomic_shared_ptr<T>::_reserve_scan_(Refcnt *rcnt) const {
	Ref *pref;
	Refcnt rcnt_new;
	for(;;) {
		pref = _pref();
		Refcnt rcnt_old;
		rcnt_old = _refcnt();
		if(!pref) {
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
		if(rcnt_new >= ATOMIC_SHARED_REF_ALIGNMENT) {
			usleep(1);
			continue;
		}
		// try to increase local reference counter w/ same serial.
		if(atomicCompareAndSet(
			_RefLocal((uintptr_t)pref + rcnt_old),
			_RefLocal((uintptr_t)pref + rcnt_new),
			&m_ref))
			break;
	}
	ASSERT(rcnt_new);
	*rcnt = rcnt_new;
	return pref;
}
template <typename T>
typename atomic_shared_ptr<T>::Ref *atomic_shared_ptr<T>::_scan_() const {
	Refcnt rcnt;
	Ref *pref = _reserve_scan_(&rcnt);
	if(!pref) return 0;
	atomicInc(&pref->refcnt);
	writeBarrier();
	_leave_scan_(pref);
	return pref;
}

template <typename T>
void
atomic_shared_ptr<T>::_leave_scan_(Ref *pref) const {
	for(;;) {
		Refcnt rcnt_old;
		rcnt_old = _refcnt();
		if(rcnt_old) {
			Refcnt rcnt_new = rcnt_old - 1;
			// try to dec. reference counter if stored pointer is unchanged.
			if(atomicCompareAndSet(
				_RefLocal((uintptr_t)pref + rcnt_old),
				_RefLocal((uintptr_t)pref + rcnt_new),
				&m_ref))
				break;
			if((pref == _pref()))
				continue; // try again.
		}
		// local reference of this context has released by other processes.
		readBarrier();
		if(atomicDecAndTest(&pref->refcnt)) {
			delete pref;
		}
		break;
	}
}

template <typename T>
void
atomic_shared_ptr<T>::swap(atomic_shared_ptr<T> &r) {
	Ref *pref;
	ASSERT(_refcnt() == 0);
	for(;;) {
		Refcnt rcnt_old, rcnt_new;
		pref = r._reserve_scan_(&rcnt_old);
		if(pref) {
			atomicAdd(&pref->refcnt, rcnt_old - 1u);
			writeBarrier();
		}
		rcnt_new = 0;
		if(atomicCompareAndSet(
			_RefLocal((uintptr_t)pref + rcnt_old),
			_RefLocal((uintptr_t)_pref() + rcnt_new),
			&r.m_ref))
			break;
		if(pref) {
			ASSERT(rcnt_old);
			atomicAdd(&pref->refcnt, (Refcnt)( -(int)(rcnt_old - 1u)));
			r._leave_scan_(pref);
		}
	}
	m_ref = (_RefLocal)pref;
}

template <typename T>
bool
atomic_shared_ptr<T>::compareAndSwap(const atomic_shared_ptr<T> &oldr, atomic_shared_ptr<T> &r) {
	Ref *pref;
	ASSERT(_refcnt() == 0);
	for(;;) {
		Refcnt rcnt_old, rcnt_new;
		pref = r._reserve_scan_(&rcnt_old);
		if(pref != oldr._pref()) {
			if(pref)
				r._leave_scan_(pref);
			return false;
		}
		if(pref) {
			atomicAdd(&pref->refcnt, rcnt_old - 1u);
			writeBarrier();
		}
		rcnt_new = 0;
		if(atomicCompareAndSet(
			_RefLocal((uintptr_t)pref + rcnt_old),
			_RefLocal((uintptr_t)_pref() + rcnt_new),
			&r.m_ref))
			break;
		if(pref) {
			ASSERT(rcnt_old);
			atomicAdd(&pref->refcnt, (Refcnt)( -(int)(rcnt_old - 1u)));
			r._leave_scan_(pref);
		}
	}
	m_ref = (_RefLocal)pref;
	return true;
}

template <typename T, class Enable>
class atomic
{
public:
	atomic() : m_var(new T) {}
	atomic(T t) : m_var(new T(t)) {}
	atomic(const atomic &t) : m_var(t) {}
	~atomic() {}
	operator T() const {
		atomic_shared_ptr<T> x = m_var;
		return *x;
	}
	atomic &operator=(T t) {
		m_var.reset(new T(t));
		return *this;
	}
	T swap(T newv) {
		atomic_shared_ptr<T> x(newv);
		x.swap(m_var);
		return *x;
	}
protected:
	atomic_shared_ptr<T> m_var;
};

#endif /*ATOMIC_SMART_PTR_H_*/
