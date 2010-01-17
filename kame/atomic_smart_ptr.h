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

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__ || defined __x86_64__
#include "atomic_prv_x86.h"
#else
#if defined __ppc__ || defined __POWERPC__ || defined __powerpc__
#include "atomic_prv_ppc.h"
#else
#error Unsupported processor
#endif // __ppc__
#endif // __i386__

//! This is an atomic variant of \a boost::scoped_ptr<>.
//! atomic_scoped_ptr<> can be shared among threads by the use of \a swap(_shared_target_).
//! Namely, a destructive read. Use atomic_shared_ptr<> for non-destructive reading.\n
//! The implementation relies on an atomic-swap machine code, e.g. lock xchg.
//! \sa atomic_shared_ptr
template <typename T>
class atomic_scoped_ptr
{
	typedef T* t_ptr;
public:
	atomic_scoped_ptr() : m_ptr(0) {}

	explicit atomic_scoped_ptr(t_ptr t) : m_ptr(t) {}

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

//! This is an internal class holding a global reference counter and a pointer to the object.
//! \sa atomic_shared_ptr
template <typename T>
struct _atomic_shared_ptr_gref {
	template <class Y>
	_atomic_shared_ptr_gref(Y *p) : ptr(p), refcnt(1) {}
	~_atomic_shared_ptr_gref() { ASSERT(refcnt == 0); delete ptr; }
	//! The pointer to the object.
	T *ptr;
	typedef uintptr_t Refcnt;
	//! Global reference counter.
	Refcnt refcnt;
private:
	_atomic_shared_ptr_gref(const _atomic_shared_ptr_gref &);
} __attribute__((aligned(ATOMIC_SHARED_REF_ALIGNMENT)));;

template <typename T>
class atomic_shared_ptr;

template <typename T>
class local_shared_ptr {
public:
	local_shared_ptr() : m_ref(0) {}

	template<typename Y> explicit local_shared_ptr(Y *y) {
		reset_unsafe(y);
	}

	local_shared_ptr(const local_shared_ptr &t) : m_ref(t.m_ref) {
		if(_pref())
			atomicInc(&_pref()->refcnt);
	}
	template<typename Y> explicit local_shared_ptr(const local_shared_ptr<Y> &y) : m_ref(y.m_ref) {
		ASSERT(_refcnt() == 0);
		if(_pref())
			atomicInc(&_pref()->refcnt);
	}
	template<typename Y> explicit local_shared_ptr(const atomic_shared_ptr<Y> &y) {
		m_ref = (_RefLocal)(typename local_shared_ptr::Ref*)y._scan_();
		readBarrier();
	}

	~local_shared_ptr();

	local_shared_ptr &operator=(const local_shared_ptr &t) {
		local_shared_ptr(t).swap(*this);
		return *this;
	}
	template<typename Y> local_shared_ptr &operator=(const local_shared_ptr<Y> &y) {
		local_shared_ptr(y).swap(*this);
		return *this;
	}
	//! \param y This instance is atomically replaced with \a t.
	template<typename Y> local_shared_ptr &operator=(const atomic_shared_ptr<Y> &y) {
		atomic_shared_ptr<T>(y).swap(*this);
		return *this;
	}
	void reset() {
		local_shared_ptr().swap(*this);
	}
	template<typename Y> void reset(Y *y) {
		local_shared_ptr(y).swap(*this);
	}
	//! Non-atomic access to the internal pointer.
	//! Never use this function for a shared instance.
	//! \sa reset()
	template<typename Y> void reset_unsafe(Y *y) {
		m_ref = (_RefLocal)new Ref(y);
	}

	//! \param x \p x is swapped with this instance.
	void swap(local_shared_ptr &x);
	//! \param x \p x is atomically swapped with this instance.
	void swap(atomic_shared_ptr<T> &x);

	T *get() { return m_ref ? ((Ref*)m_ref)->ptr : NULL; }
	const T *get() const { return m_ref ? ((Ref*)m_ref)->ptr : NULL; }

	T &operator*() { ASSERT(*this); return *get();}
	const T &operator*() const { ASSERT(*this); return *get();}

	T *operator->() { ASSERT(*this); return get();}
	const T *operator->() const { ASSERT(*this); return get();}

	bool operator!() const {return !m_ref;}

	operator bool() const {return m_ref;}

	template<typename Y> bool operator==(const local_shared_ptr<Y> &x) const {readBarrier(); return (_pref() == x._pref());}
	template<typename Y> bool operator==(const atomic_shared_ptr<Y> &x) const {readBarrier(); return (_pref() == x._pref());}
	template<typename Y> bool operator!=(const local_shared_ptr<Y> &x) const {readBarrier(); return (_pref() != x._pref());}
	template<typename Y> bool operator!=(const atomic_shared_ptr<Y> &x) const {readBarrier(); return (_pref() != _pref());}
protected:
	template <typename Y> friend class local_shared_ptr;
	template <typename Y> friend class atomic_shared_ptr;
	typedef _atomic_shared_ptr_gref<T> Ref;
	typedef typename Ref::Refcnt Refcnt;
	typedef uintptr_t _RefLocal;
	_RefLocal m_ref;
	//! A pointer to global reference struct.
	Ref* _pref() const {return (Ref*)(m_ref);}
	//! Local (temporary) reference counter.
	//! Local reference counter is a trick to tell the observation to other threads.
	Refcnt _refcnt() const {return (Refcnt)(m_ref & (uintptr_t)(ATOMIC_SHARED_REF_ALIGNMENT - 1));}
};

/*! This is an atomic variant of \a boost::shared_ptr<>.\n
* \a atomic_shared_ptr<> can be shared among threads by the use of \a operator=(_target_), \a swap(_target_).
* An instance of \a atomic_shared_ptr<T> holds:\n
* 	a) a pointer to \a _atomic_shared_ptr_gref<T>, which is a struct consisting of a pointer to the T-type object, and a global reference counter.\n
* 	b) a local (temporary) reference counter, which is embedded in the above pointer by using the least significant bits that should be usually zero.\n
* The values of a) and b), \a m_ref, are atomically handled with CAS machine codes.
* The purpose of b) the local reference counter is to tell the "observation" to the shared target before increasing the global reference counter.
* This process is implemented in \a _reserve_scan_().\n
* A function \a _leave_scan_() tries to decrease the local counter first. When it fails, the global counter is decreased.\n
* To swap the pointer and local reference counter (which will be reset to zero), the setter must adds the local counting to the global counter before swapping.
* \sa atomic_scoped_ptr
 */
template <typename T>
class atomic_shared_ptr : protected local_shared_ptr<T> {
public:
	atomic_shared_ptr() : local_shared_ptr<T>() {}

	template<typename Y> explicit atomic_shared_ptr(Y *y) : local_shared_ptr<T>(y) {}

	atomic_shared_ptr(const atomic_shared_ptr &t) : local_shared_ptr<T>(t) {}
	template<typename Y> explicit atomic_shared_ptr(const local_shared_ptr<Y> &y) : local_shared_ptr<T>(y) {}
	template<typename Y> explicit atomic_shared_ptr(const atomic_shared_ptr<Y> &y) : local_shared_ptr<T>(y) {}

	~atomic_shared_ptr() {}

	//! \param t This instance is atomically replaced with \a t.
	atomic_shared_ptr &operator=(const atomic_shared_ptr &t) {
		atomic_shared_ptr(t).swap(*this);
		return *this;
	}
	//! \param y This instance is atomically replaced with \a t.
	template<typename Y> atomic_shared_ptr &operator=(const local_shared_ptr<Y> &y) {
		local_shared_ptr<T>(y).swap(*this);
		return *this;
	}
	//! \param y This instance is atomically replaced with \a t.
	template<typename Y> atomic_shared_ptr &operator=(const atomic_shared_ptr<Y> &y) {
		atomic_shared_ptr(y).swap(*this);
		return *this;
	}
	//! This instance is atomically reset to null pointer.
	void reset() {
		atomic_shared_ptr().swap(*this);
	}
	//! This instance is atomically reset with a pointer \a y.
	template<typename Y> void reset(Y *y) {
		atomic_shared_ptr(y).swap(*this);
	}

	//! \param x \p x is atomically swapped with this instance.
	//! Nevertheless, this instance is not atomically replaced.
	//! That is, "this" must not be shared among threads.
	void swap(atomic_shared_ptr &x) {local_shared_ptr<T>::swap(x);}

	//! \return true if succeeded.
	//! \sa compareAndSwap()
	bool compareAndSet(const local_shared_ptr<T> &oldvalue, const local_shared_ptr<T> &newvalue) {
		return compareAndSwap(const_cast<local_shared_ptr<T> &>(oldvalue), newvalue, true);
	}
	//! \return true if succeeded.
	//! \sa compareAndSet()
	bool compareAndSwap(local_shared_ptr<T> &oldvalue, const local_shared_ptr<T> &newvalue, bool noswap = false);

	bool operator!() const {return !this->m_ref;}
	operator bool() const {return this->m_ref;}

	template<typename Y> bool operator==(const local_shared_ptr<Y> &x) const {readBarrier(); return (_pref() == x._pref());}
	template<typename Y> bool operator==(const atomic_shared_ptr<Y> &x) const {readBarrier(); return (_pref() == x._pref());}
	template<typename Y> bool operator!=(const local_shared_ptr<Y> &x) const {readBarrier(); return (_pref() != x._pref());}
	template<typename Y> bool operator!=(const atomic_shared_ptr<Y> &x) const {readBarrier(); return (_pref() != x._pref());}
protected:
	template <typename Y> friend class local_shared_ptr;
	template <typename Y> friend class atomic_shared_ptr;

	typedef typename local_shared_ptr<T>::_RefLocal _RefLocal;
	typedef typename local_shared_ptr<T>::Refcnt Refcnt;
	typedef typename local_shared_ptr<T>::Ref Ref;

	//internal functions below.
	//! atomically scans \a m_ref and increases the global reference counter.
	//! \a _scan_ is used for atomically coping the pointer.
	Ref *_scan_() const;
	//! atomically scans \a m_ref and increases the  local (temporary) reference counter.
	//! use \a _leave_scan_ to release the temporary reference.
	Ref *_reserve_scan_(Refcnt *) const;
	//! tries to decrease local (temporary) reference counter.
	//! In case the reference is lost, \a _leave_scan_ releases the global reference counter instead.
	void _leave_scan_(Ref *) const;

	//! A pointer to global reference struct.
	Ref* _pref() const {return (Ref*)(this->m_ref & (~(uintptr_t)(ATOMIC_SHARED_REF_ALIGNMENT - 1)));}
	//! Local (temporary) reference counter.
	//! Local reference counter is a trick to tell the observation to other threads.
	Refcnt _refcnt() const {return local_shared_ptr<T>::_refcnt();}
};

template <typename T>
local_shared_ptr<T>::~local_shared_ptr() {
	ASSERT(_refcnt() == 0);
	Ref *pref = _pref();
	if(!pref) return;
	// decreasing global reference counter.
	if(atomicDecAndTest(&pref->refcnt)) {
		readBarrier();
		delete pref;
	}
}

template <typename T>
void
local_shared_ptr<T>::swap(local_shared_ptr &r) {
	_RefLocal x = m_ref;
	m_ref = r.m_ref;
	r.m_ref = x;
}

template <typename T>
void
local_shared_ptr<T>::swap(atomic_shared_ptr<T> &r) {
	Ref *pref;
	ASSERT(_refcnt() == 0);
	writeBarrier();
	for(;;) {
		Refcnt rcnt_old, rcnt_new;
		pref = r._reserve_scan_(&rcnt_old);
		if(pref && (rcnt_old != 1u)) {
			atomicAdd(&pref->refcnt, rcnt_old - 1u);
		}
		rcnt_new = 0;
		if(atomicCompareAndSet(
			_RefLocal((uintptr_t)pref + rcnt_old),
			_RefLocal((uintptr_t)_pref() + rcnt_new),
			&r.m_ref))
			break;
		if(pref) {
			ASSERT(rcnt_old);
			if(rcnt_old != 1u)
				atomicAdd(&pref->refcnt, (Refcnt)( -(int)(rcnt_old - 1u)));
			r._leave_scan_(pref);
		}
	}
	m_ref = (_RefLocal)pref;
	readBarrier();
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
			// This would be never happen.
			usleep(1);
			continue;
		}
		// trying to increase local reference counter w/ same serial.
		if(atomicCompareAndSet(
			_RefLocal((uintptr_t)pref + rcnt_old),
			_RefLocal((uintptr_t)pref + rcnt_new),
			&const_cast<atomic_shared_ptr<T> *>(this)->m_ref))
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
	_leave_scan_(pref);
	readBarrier();
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
			// trying to dec. reference counter if stored pointer is unchanged.
			if(atomicCompareAndSet(
				_RefLocal((uintptr_t)pref + rcnt_old),
				_RefLocal((uintptr_t)pref + rcnt_new),
				&const_cast<atomic_shared_ptr<T> *>(this)->m_ref))
				break;
			if((pref == _pref()))
				continue; // try again.
		}
		// local reference has released by other processes.
		if(atomicDecAndTest(&pref->refcnt)) {
			readBarrier();
			delete pref;
		}
		break;
	}
}

template <typename T>
bool
atomic_shared_ptr<T>::compareAndSwap(local_shared_ptr<T> &oldr, const local_shared_ptr<T> &newr, bool noswap) {
	Ref *pref;
	ASSERT(newr._refcnt() == 0);
	ASSERT(oldr._refcnt() == 0);
	if(newr._pref()) {
		atomicInc(&newr._pref()->refcnt);
		writeBarrier();
	}
	for(;;) {
		Refcnt rcnt_old, rcnt_new;
		pref = _reserve_scan_(&rcnt_old);
		if(pref != oldr._pref()) {
			if(pref) {
				if(!noswap) {
					atomicInc(&pref->refcnt);
				}
				_leave_scan_(pref);
			}
			if(newr._pref())
				atomicDec(&newr._pref()->refcnt);
			if(!noswap) {
				readBarrier();
				if(oldr._pref()) {
					// decreasing global reference counter.
					if(atomicDecAndTest(&oldr._pref()->refcnt)) {
						delete oldr._pref();
					}
				}
				oldr.m_ref = (_RefLocal)pref;
			}
			return false;
		}
		if(pref && (rcnt_old != 1u)) {
			atomicAdd(&pref->refcnt, rcnt_old - 1u);
		}
		rcnt_new = 0;
		if(atomicCompareAndSet(
			_RefLocal((uintptr_t)pref + rcnt_old),
			_RefLocal((uintptr_t)newr._pref() + rcnt_new),
			&this->m_ref))
			break;
		if(pref) {
			ASSERT(rcnt_old);
			if(rcnt_old != 1u)
				atomicAdd(&pref->refcnt, (Refcnt)( -(int)(rcnt_old - 1u)));
			_leave_scan_(pref);
		}
	}
	if(pref) {
		atomicDec(&pref->refcnt);
	}
	return true;
}

#endif /*ATOMIC_SMART_PTR_H_*/
