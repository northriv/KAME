/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
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

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>

//! This is an atomic variant of \a boost::scoped_ptr.
//! An instance of atomic_scoped_ptr can be shared among threads by the use of \a swap(\a _shared_target_).\n
//! Namely, it is destructive reading.
//! Use atomic_shared_ptr when the pointer is required to be shared among scopes and threads.\n
//! This implementation relies on an atomic-swap machine code, e.g. lock xchg.
//! \sa atomic_shared_ptr, atomic_scoped_ptr_test.cpp
template <typename T>
class atomic_scoped_ptr {
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
	//! \param[in,out] x \p x is atomically swapped.
	//! Nevertheless, this object is not atomically replaced.
	//! That is, the object pointed by "this" must not be shared among threads.
	void swap(atomic_scoped_ptr &x) {
		writeBarrier();
		m_ptr = atomicSwap(m_ptr, &x.m_ptr);
		readBarrier();
	}

	bool operator!() const {readBarrier(); return !m_ptr;}
	operator bool() const {readBarrier(); return m_ptr;}

	//! This function lacks thread-safety.
	T &operator*() const { ASSERT(m_ptr); return (T &) *m_ptr;}

	//! This function lacks thread-safety.
	t_ptr operator->() const { ASSERT(m_ptr); return (t_ptr)m_ptr;}

	//! This function lacks thread-safety.
	t_ptr get() const { return (t_ptr )m_ptr;}
private:
	atomic_scoped_ptr(const atomic_scoped_ptr &);
	atomic_scoped_ptr& operator=(const atomic_scoped_ptr &);

	t_ptr m_ptr;
};

//! This is an internal class holding a global reference counter and a pointer to the object.
//! \sa atomic_shared_ptr
template <typename T>
struct _atomic_shared_ptr_gref {
	_atomic_shared_ptr_gref(T *p) : ptr(p), refcnt(1) {}
	~_atomic_shared_ptr_gref() { ASSERT(refcnt == 0); delete ptr; }
	//! The pointer to the object.
	T *ptr;
	typedef uintptr_t Refcnt;
	//! The global reference counter.
	Refcnt refcnt;
private:
	_atomic_shared_ptr_gref(const _atomic_shared_ptr_gref &);
};

template <typename T, typename E> class atomic_shared_ptr_base;
template <typename T> class atomic_shared_ptr;
template <typename T> class local_shared_ptr;

//! Use subclass of this to be storaged in atomic_shared_ptr with
//! intrusive counting to obtain better performance.
struct atomic_countable {
	atomic_countable() : refcnt(1) {}
	atomic_countable(const atomic_countable &x) : refcnt(1) {}
	~atomic_countable() { ASSERT(refcnt == 0); }
private:
	template <typename X, typename E> friend class atomic_shared_ptr_base;
	template <typename X> friend class atomic_shared_ptr;
	template <typename X> friend class local_shared_ptr;
	atomic_countable& operator=(const atomic_countable &); //inhibited.
	typedef uintptr_t Refcnt;
	//! Global reference counter.
	Refcnt refcnt;
};

#define ATOMIC_SHARED_REF_ALIGNMENT (sizeof(uintptr_t))

//! Base class for atomic_shared_ptr without intrusive counting, so-called "simple counted".\n
//! A global referece counter (an instance of _atomic_shared_ptr_gref) will be created.
template <typename T, typename Enable = void>
struct atomic_shared_ptr_base {
protected:
	//! Non-atomic access to the internal pointer.
	//! Never use this function for a shared instance.
	//! \sa reset()

	typedef _atomic_shared_ptr_gref<T> Ref;
	typedef typename Ref::Refcnt Refcnt;
	typedef uintptr_t _RefLocal;

	static int deleter(Ref *p) { delete p; return 1; }

	template<typename Y> void reset_unsafe(Y *y) {
		m_ref = (_RefLocal)new Ref(static_cast<T*>(y));
	}
	T *get() { return this->m_ref ? ((Ref*)this->m_ref)->ptr : NULL; }
	const T *get() const { return this->m_ref ? ((const Ref*)this->m_ref)->ptr : NULL; }

	int _use_count() const {readBarrier(); return ((const Ref*)this->m_ref)->refcnt;}

	_RefLocal m_ref;
};
//! Base class for atomic_shared_ptr with intrusive counting.
template <typename T>
struct atomic_shared_ptr_base<T, typename boost::enable_if<boost::is_base_of<atomic_countable, T> >::type > {
protected:
	//! Non-atomic access to the internal pointer.
	//! Never use this function for a shared instance.
	//! \sa reset().
	typedef T Ref;
	typedef typename atomic_countable::Refcnt Refcnt;
	typedef uintptr_t _RefLocal;

	static int deleter(T *p) { delete p; return 1;}

	template<typename Y> void reset_unsafe(Y *y) {
		m_ref = (_RefLocal)static_cast<T*>(y);
	}
	T *get() { return (T*)this->m_ref; }
	const T *get() const { return (const T*)this->m_ref; }

	int _use_count() const {readBarrier(); return ((const T*)this->m_ref)->refcnt;}

	_RefLocal m_ref;
};
/*! This is an atomic variant of \a boost::shared_ptr<>.\n
* \a atomic_shared_ptr can be shared among threads by the use of \a operator=(_target_), \a swap(_target_).
* An instance of \a atomic_shared_ptr<T> holds:\n
* 	a) a pointer to \a _atomic_shared_ptr_gref<T>, which is a struct consisting of a pointer to the T-type object, and a global reference counter.\n
* 	b) a local (temporary) reference counter, which is embedded in the above pointer by using the least significant bits that should be usually zero.\n
* The values of a) and b), \a m_ref, are atomically handled with CAS machine codes.
* The purpose of b) the local reference counter is to tell the "observation" to the shared target before increasing the global reference counter.
* This process is implemented in \a _reserve_scan_().\n
* A function \a _leave_scan_() tries to decrease the local counter first. When it fails, the global counter is decreased.\n
* To swap the pointer and local reference counter (which will be reset to zero), the setter must adds the local counting to the global counter before swapping.
* \sa atomic_scoped_ptr, local_shared_ptr, atomic_shared_ptr_test.cpp.
 */
template <typename T>
class atomic_shared_ptr : public atomic_shared_ptr_base<T> {
public:
	atomic_shared_ptr() { this->m_ref = 0; }

	template<typename Y> explicit atomic_shared_ptr(Y *y) {
		reset_unsafe(y);
	}

	atomic_shared_ptr(const atomic_shared_ptr<T> &t) {
		this->m_ref = (_RefLocal)(typename atomic_shared_ptr::Ref*)t._scan_();
		readBarrier();
	}
	template<typename Y> atomic_shared_ptr(const atomic_shared_ptr<Y> &y) {
		C_ASSERT(sizeof(static_cast<const T*>(y.get())));
		this->m_ref = (_RefLocal)(typename atomic_shared_ptr::Ref*)y._scan_();
		readBarrier();
	}
	atomic_shared_ptr(const local_shared_ptr<T> &t) {
		this->m_ref = t.m_ref;
		if(_pref())
			atomicInc(&_pref()->refcnt);
	}
	template<typename Y> atomic_shared_ptr(const local_shared_ptr<Y> &y) {
		C_ASSERT(sizeof(static_cast<const T*>(y.get())));
		this->m_ref = y.m_ref;
		ASSERT(_refcnt() == 0);
		if(_pref())
			atomicInc(&_pref()->refcnt);
	}

	~atomic_shared_ptr();

	//! \param[in] t The pointer holded by this instance is atomically replaced with that of \a t.
	atomic_shared_ptr &operator=(const atomic_shared_ptr &t) {
		atomic_shared_ptr<T>(t).swap(*this);
		return *this;
	}
	//! \param[in] y The pointer holded by this instance is atomically replaced with that of \a y.
	template<typename Y> atomic_shared_ptr &operator=(const local_shared_ptr<Y> &y) {
		atomic_shared_ptr<T>(y).swap( *this);
		return *this;
	}
	//! \param[in] y The pointer holded by this instance is atomically replaced with that of \a y.
	template<typename Y> atomic_shared_ptr &operator=(const atomic_shared_ptr<Y> &y) {
		atomic_shared_ptr<T>(y).swap( *this);
		return *this;
	}
	//! The pointer holded by this instance is atomically reset to null pointer.
	void reset() {
		atomic_shared_ptr<T>().swap( *this);
	}
	//! The pointer holded by this instance is atomically reset with a pointer \a y.
	template<typename Y> void reset(Y *y) {
		atomic_shared_ptr<T>(y).swap( *this);
	}
	//! \param[in,out] x \p The pointer holded by \a x is atomically swapped with that of this instance.
	void swap(atomic_shared_ptr<T> &x);

	//! \return true if succeeded.
	//! \sa compareAndSwap()
	bool compareAndSet(const local_shared_ptr<T> &oldvalue, const local_shared_ptr<T> &newvalue) {
		return compareAndSwap(const_cast<local_shared_ptr<T> &>(oldvalue), newvalue, true);
	}
	//! \return true if succeeded.
	//! \sa compareAndSet()
	bool compareAndSwap(local_shared_ptr<T> &oldvalue, const local_shared_ptr<T> &newvalue, bool noswap = false);

	bool operator!() const {readBarrier(); return !this->m_ref;}
	operator bool() const {readBarrier(); return this->m_ref;}

	template<typename Y> bool operator==(const local_shared_ptr<Y> &x) const {
		C_ASSERT(sizeof(static_cast<const T*>(x.get())));
		readBarrier(); return (_pref() == (const Ref*)x._pref());}
	template<typename Y> bool operator==(const atomic_shared_ptr<Y> &x) const {
		C_ASSERT(sizeof(static_cast<const T*>(x.get())));
		readBarrier(); return (_pref() == (const Ref*)x._pref());}
	template<typename Y> bool operator!=(const local_shared_ptr<Y> &x) const {
		C_ASSERT(sizeof(static_cast<const T*>(x.get())));
		readBarrier(); return (_pref() != (const Ref*)x._pref());}
	template<typename Y> bool operator!=(const atomic_shared_ptr<Y> &x) const {
		C_ASSERT(sizeof(static_cast<const T*>(x.get())));
		readBarrier(); return (_pref() != (const Ref*)x._pref());}
protected:
	template <typename Y> friend class local_shared_ptr;
	template <typename Y> friend class atomic_shared_ptr;
	typedef typename atomic_shared_ptr_base<T>::Ref Ref;
	typedef typename atomic_shared_ptr_base<T>::Refcnt Refcnt;
	typedef typename atomic_shared_ptr_base<T>::_RefLocal _RefLocal;
	//! A pointer to global reference struct.
	Ref* _pref() const {return (Ref*)(this->m_ref & (~(uintptr_t)(ATOMIC_SHARED_REF_ALIGNMENT - 1)));}
	//! Local (temporary) reference counter.
	//! Local reference counter is a trick to tell the observation to other threads.
	Refcnt _refcnt() const {return (Refcnt)(this->m_ref & (uintptr_t)(ATOMIC_SHARED_REF_ALIGNMENT - 1));}

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
};

//! This class provides non-reentrant interfaces on atomic_shared_ptr: operator->(), operator*() and so on.\n
//! Use this class in non-reentrant scopes instead of costly atomic_shared_ptr.
//! \sa atomic_shared_ptr, atomic_scoped_ptr, atomic_shared_ptr_test.cpp.
template <typename T>
class local_shared_ptr : protected atomic_shared_ptr<T> {
public:
	local_shared_ptr() : atomic_shared_ptr<T>() {}

	template<typename Y> explicit local_shared_ptr(Y *y) : atomic_shared_ptr<T>(y) {}

	local_shared_ptr(const local_shared_ptr &t) : atomic_shared_ptr<T>(t) {}
	template<typename Y> local_shared_ptr(const local_shared_ptr<Y> &y) : atomic_shared_ptr<T>(y) {}
	template<typename Y> local_shared_ptr(const atomic_shared_ptr<Y> &y) : atomic_shared_ptr<T>(y) {}

	~local_shared_ptr() {}

	local_shared_ptr &operator=(const local_shared_ptr &t) {
		local_shared_ptr(t).swap( *this);
		return *this;
	}
	template<typename Y> local_shared_ptr &operator=(const local_shared_ptr<Y> &y) {
		local_shared_ptr(y).swap( *this);
		return *this;
	}
	//! \param[in] t The pointer holded by this instance is replaced with that of \a t.
	local_shared_ptr &operator=(const atomic_shared_ptr<T> &t) {
		this->reset();
		this->m_ref = (_RefLocal)(typename local_shared_ptr::Ref*)t._scan_();
		readBarrier();
		return *this;
	}
	//! \param[in] y The pointer holded by this instance is replaced with that of \a y.
	template<typename Y> local_shared_ptr &operator=(const atomic_shared_ptr<Y> &y) {
		C_ASSERT(sizeof(static_cast<const T*>(y.get())));
		this->reset();
		this->m_ref = (_RefLocal)(typename local_shared_ptr::Ref*)y._scan_();
		readBarrier();
		return *this;
	}

	//! \param[in,out] x \p The pointer holded by \a x is swapped with that of this instance.
	void swap(local_shared_ptr &x);
	//! \param[in,out] x \p The pointer holded by \a x is atomically swapped with that of this instance.
	void swap(atomic_shared_ptr<T> &x) {atomic_shared_ptr<T>::swap(x);}

	//! The pointer holded by this instance is atomically reset to null pointer.
	void reset() {
		atomic_shared_ptr<T>::reset();
	}
	//! The pointer holded by this instance is atomically reset with a pointer \a y.
	template<typename Y> void reset(Y *y) {
		atomic_shared_ptr<T>::reset(y);
	}

	T *get() { return atomic_shared_ptr<T>::get(); }
	const T *get() const { return atomic_shared_ptr<T>::get(); }

	T &operator*() { ASSERT(*this); return *get();}
	const T &operator*() const { ASSERT(*this); return *get();}

	T *operator->() { ASSERT(*this); return get();}
	const T *operator->() const { ASSERT(*this); return get();}

	bool operator!() const {return !this->m_ref;}
	operator bool() const {return this->m_ref;}

	template<typename Y> bool operator==(const local_shared_ptr<Y> &x) const {
		C_ASSERT(sizeof(static_cast<const T*>(x.get())));
		return (this->_pref() == (const Ref *)x._pref());}
	template<typename Y> bool operator==(const atomic_shared_ptr<Y> &x) const {
		C_ASSERT(sizeof(static_cast<const T*>(x.get())));
		readBarrier(); return (this->_pref() == (const Ref *)x._pref());}
	template<typename Y> bool operator!=(const local_shared_ptr<Y> &x) const {
		C_ASSERT(sizeof(static_cast<const T*>(x.get())));
		return (this->_pref() != (const Ref *)x._pref());}
	template<typename Y> bool operator!=(const atomic_shared_ptr<Y> &x) const {
		C_ASSERT(sizeof(static_cast<const T*>(x.get())));
		readBarrier(); return (this->_pref() != (const Ref *)x._pref());}

	int use_count() const {readBarrier(); return this->_use_count();}
	bool unique() const {return use_count() == 1;}
protected:
	template <typename Y> friend class local_shared_ptr;
	template <typename Y> friend class atomic_shared_ptr;

	typedef typename atomic_shared_ptr<T>::_RefLocal _RefLocal;
	typedef typename atomic_shared_ptr<T>::Refcnt Refcnt;
	typedef typename atomic_shared_ptr<T>::Ref Ref;
	//! A pointer to global reference struct.
	Ref* _pref() const {return (Ref *)(this->m_ref);}
};

template <typename T>
atomic_shared_ptr<T>::~atomic_shared_ptr() {
	ASSERT(_refcnt() == 0);
	Ref *pref = _pref();
	if( !pref) return;
	// decreasing global reference counter.
	if(atomicDecAndTest(&pref->refcnt)) {
		readBarrier();
		deleter(pref);
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
		if(rcnt_new >= ATOMIC_SHARED_REF_ALIGNMENT) {
			// This would never happen.
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
			deleter(pref);
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
			if( !noswap) {
				readBarrier();
				if(oldr._pref()) {
					// decreasing global reference counter.
					if(atomicDecAndTest(&oldr._pref()->refcnt)) {
						deleter(oldr._pref());
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

template <typename T>
void
local_shared_ptr<T>::swap(local_shared_ptr &r) {
	_RefLocal x = this->m_ref;
	this->m_ref = r.m_ref;
	r.m_ref = x;
}

template <typename T>
void
atomic_shared_ptr<T>::swap(atomic_shared_ptr<T> &r) {
	Ref *pref;
	ASSERT(this->_refcnt() == 0);
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
			_RefLocal((uintptr_t)this->_pref() + rcnt_new),
			&r.m_ref))
			break;
		if(pref) {
			ASSERT(rcnt_old);
			if(rcnt_old != 1u)
				atomicAdd(&pref->refcnt, (Refcnt)( -(int)(rcnt_old - 1u)));
			r._leave_scan_(pref);
		}
	}
	this->m_ref = (_RefLocal)pref;
	readBarrier();
}

#endif /*ATOMIC_SMART_PTR_H_*/
