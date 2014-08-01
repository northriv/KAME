/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp

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

#include <type_traits>
#include <unistd.h>

//! \brief This is an atomic variant of \a std::unique_ptr.
//! An instance of atomic_unique_ptr can be shared among threads by the use of \a swap(\a _shared_target_).\n
//! Namely, it is destructive reading.
//! Use atomic_shared_ptr when the pointer is required to be shared among scopes and threads.\n
//! This implementation relies on an atomic-swap machine code, e.g. lock xchg.
//! \sa atomic_shared_ptr, atomic_unique_ptr_test.cpp
template <typename T>
class atomic_unique_ptr {
	typedef T* t_ptr;
public:
	atomic_unique_ptr() : m_ptr(0) {}

	explicit atomic_unique_ptr(t_ptr t) : m_ptr(t) {}

	~atomic_unique_ptr() { delete m_ptr;}

	void reset(t_ptr t = 0) {
		if(t) writeBarrier(); //for *t.
		t_ptr old = atomicSwap(t, &m_ptr);
		if(old) readBarrier(); //for *old.
		delete old;
	}
	//! \param[in,out] x \p x is atomically swapped.
	//! Nevertheless, this object is not atomically replaced.
	//! That is, the object pointed by "this" must not be shared among threads.
	void swap(atomic_unique_ptr &x) {
		if(m_ptr) writeBarrier(); //for the contents of this.
		m_ptr = atomicSwap(m_ptr, &x.m_ptr);
		if(m_ptr) readBarrier(); //for the contents.
	}

	bool operator!() const {readBarrier(); return !m_ptr;}
	operator bool() const {readBarrier(); return m_ptr;}

	//! This function lacks thread-safety.
	T &operator*() const { assert(m_ptr); return (T &) *m_ptr;}

	//! This function lacks thread-safety.
	t_ptr operator->() const { assert(m_ptr); return (t_ptr)m_ptr;}

	//! This function lacks thread-safety.
	t_ptr get() const { return (t_ptr )m_ptr;}
private:
	atomic_unique_ptr(const atomic_unique_ptr &) = delete;
	atomic_unique_ptr& operator=(const atomic_unique_ptr &) = delete;

	t_ptr m_ptr;
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
	Refcnt refcnt;
private:
	atomic_shared_ptr_gref_(const atomic_shared_ptr_gref_ &) = delete;
};

template <typename T, typename E> struct atomic_shared_ptr_base;
template <typename T> class atomic_shared_ptr;
template <typename T> class local_shared_ptr;

//! Use subclass of this to be storaged in atomic_shared_ptr with
//! intrusive counting to obtain better performance.
struct atomic_countable {
	atomic_countable() : refcnt(1) {}
    atomic_countable(const atomic_countable &) : refcnt(1) {}
	~atomic_countable() { assert(refcnt == 0); }
private:
	template <typename X, typename E> friend struct atomic_shared_ptr_base;
	template <typename X> friend class atomic_shared_ptr;
	template <typename X> friend class local_shared_ptr;
	atomic_countable& operator=(const atomic_countable &); //inhibited.
	typedef uintptr_t Refcnt;
	//! Global reference counter.
	Refcnt refcnt;
};

//! \brief Base class for atomic_shared_ptr without intrusive counting, so-called "simple counted".\n
//! A global referece counter (an instance of atomic_shared_ptr_gref_) will be created.
template <typename T, typename Enable = void>
struct atomic_shared_ptr_base {
protected:
	typedef atomic_shared_ptr_gref_<T> Ref;
	typedef typename Ref::Refcnt Refcnt;
	typedef uintptr_t RefLocal_;

	static int deleter(Ref *p) { delete p; return 1; }

	//! can be used to initialize the internal pointer \a m_ref.
	//! \sa reset()
	template<typename Y> void reset_unsafe(Y *y) {
		m_ref = (RefLocal_)new Ref(static_cast<T*>(y));
	}
	T *get() { return this->m_ref ? ((Ref*)this->m_ref)->ptr : NULL; }
	const T *get() const { return this->m_ref ? ((const Ref*)this->m_ref)->ptr : NULL; }

	int _use_count_() const {return ((const Ref*)this->m_ref)->refcnt;}

	union {
		RefLocal_ m_ref;
		int64_t _for_alignment_;
	}; // __attribute__((aligned(8)));
	enum {ATOMIC_SHARED_REF_ALIGNMENT = (sizeof(intptr_t))};
};
//! \brief Base class for atomic_shared_ptr with intrusive counting.
template <typename T>
struct atomic_shared_ptr_base<T, typename std::enable_if<std::is_base_of<atomic_countable, T>::value>::type > {
protected:
	typedef T Ref;
	typedef typename atomic_countable::Refcnt Refcnt;
	typedef uintptr_t RefLocal_;

	static int deleter(T *p) { delete p; return 1;}

	//! can be used to initialize the internal pointer \a m_ref.
	template<typename Y> void reset_unsafe(Y *y) {
		m_ref = (RefLocal_)static_cast<T*>(y);
	}
	T *get() { return (T*)this->m_ref; }
	const T *get() const { return (const T*)this->m_ref; }

	int _use_count_() const {return ((const T*)this->m_ref)->refcnt;}

	RefLocal_ m_ref;
	enum {ATOMIC_SHARED_REF_ALIGNMENT = (sizeof(double))};
};

//! \brief This class provides non-reentrant interfaces for atomic_shared_ptr: operator->(), operator*() and so on.\n
//! Use this class in non-reentrant scopes instead of costly atomic_shared_ptr.
//! \sa atomic_shared_ptr, atomic_unique_ptr, atomic_shared_ptr_test.cpp.
template <typename T>
class local_shared_ptr : protected atomic_shared_ptr_base<T> {
public:
	local_shared_ptr() { this->m_ref = 0; }

	template<typename Y> explicit local_shared_ptr(Y *y) { this->reset_unsafe(y); }

	local_shared_ptr(const atomic_shared_ptr<T> &t) { this->m_ref = reinterpret_cast<RefLocal_>(t.scan_()); }
	template<typename Y> local_shared_ptr(const atomic_shared_ptr<Y> &y) {
		static_assert(sizeof(static_cast<const T*>(y.get())), "");
		this->m_ref = reinterpret_cast<RefLocal_>(y.scan_());
	}
	inline local_shared_ptr(const local_shared_ptr<T> &t);
	template<typename Y> inline local_shared_ptr(const local_shared_ptr<Y> &y);
	inline ~local_shared_ptr();

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

	T *get() { return atomic_shared_ptr_base<T>::get(); }
	const T *get() const { return atomic_shared_ptr_base<T>::get(); }

	T &operator*() { assert( *this); return *get();}
	const T &operator*() const { assert( *this); return *get();}

	T *operator->() { assert( *this); return get();}
	const T *operator->() const { assert( *this); return get();}

	bool operator!() const {return !this->m_ref;}
	operator bool() const {return this->m_ref;}

	template<typename Y> bool operator==(const local_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
		return (this->pref_() == (const Ref *)x.pref_());}
	template<typename Y> bool operator==(const atomic_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
		readBarrier(); return (this->pref_() == (const Ref *)x.pref_());}
	template<typename Y> bool operator!=(const local_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
		return (this->pref_() != (const Ref *)x.pref_());}
	template<typename Y> bool operator!=(const atomic_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
		readBarrier(); return (this->pref_() != (const Ref *)x.pref_());}

	int use_count() const { return this->_use_count_();}
	bool unique() const {return use_count() == 1;}
protected:
	template <typename Y> friend class local_shared_ptr;
	template <typename Y> friend class atomic_shared_ptr;
	typedef typename atomic_shared_ptr_base<T>::Ref Ref;
	typedef typename atomic_shared_ptr_base<T>::Refcnt Refcnt;
	typedef typename atomic_shared_ptr_base<T>::RefLocal_ RefLocal_;

	//! A pointer to global reference struct.
	Ref* pref_() const {return (Ref *)(this->m_ref);}
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
class atomic_shared_ptr : protected local_shared_ptr<T> {
public:
	atomic_shared_ptr() : local_shared_ptr<T>() {}

	template<typename Y> explicit atomic_shared_ptr(Y *y) : local_shared_ptr<T>(y) {}
	atomic_shared_ptr(const atomic_shared_ptr<T> &t) : local_shared_ptr<T>(t) {}
	template<typename Y> atomic_shared_ptr(const atomic_shared_ptr<Y> &y) : local_shared_ptr<T>(y) {}
	atomic_shared_ptr(const local_shared_ptr<T> &t) : local_shared_ptr<T>(t) {}
	template<typename Y> atomic_shared_ptr(const local_shared_ptr<Y> &y) : local_shared_ptr<T>(y) {}

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

	bool operator!() const {readBarrier(); return !this->m_ref;}
	operator bool() const {readBarrier(); return this->m_ref;}

	template<typename Y> bool operator==(const local_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
		readBarrier(); return (pref_() == (const Ref*)x.pref_());}
	template<typename Y> bool operator==(const atomic_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
		readBarrier(); return (pref_() == (const Ref*)x.pref_());}
	template<typename Y> bool operator!=(const local_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
		readBarrier(); return (pref_() != (const Ref*)x.pref_());}
	template<typename Y> bool operator!=(const atomic_shared_ptr<Y> &x) const {
		static_assert(sizeof(static_cast<const T*>(x.get())), "");
		readBarrier(); return (pref_() != (const Ref*)x.pref_());}
protected:
	template <typename Y> friend class local_shared_ptr;
	template <typename Y> friend class atomic_shared_ptr;
	typedef typename atomic_shared_ptr_base<T>::Ref Ref;
	typedef typename atomic_shared_ptr_base<T>::Refcnt Refcnt;
	typedef typename atomic_shared_ptr_base<T>::RefLocal_ RefLocal_;
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

template <typename T>
inline local_shared_ptr<T>::local_shared_ptr(const local_shared_ptr &y) {
	static_assert(sizeof(static_cast<const T*>(y.get())), "");
	this->m_ref = y.m_ref;
	if(pref_())
		atomicInc( &pref_()->refcnt);
}

template <typename T>
template<typename Y>
inline local_shared_ptr<T>::local_shared_ptr(const local_shared_ptr<Y> &y) {
	static_assert(sizeof(static_cast<const T*>(y.get())), "");
	this->m_ref = y.m_ref;
	if(pref_())
		atomicInc( &pref_()->refcnt);
}

template <typename T>
inline local_shared_ptr<T>::~local_shared_ptr() {
	reset();
}

template <typename T>
inline void
local_shared_ptr<T>::reset() {
	Ref *pref = pref_();
	if( !pref) return;
	// decreasing global reference counter.
	if(unique()) {
		pref->refcnt = 0;
		this->deleter(pref);
	}
	else if(atomicDecAndTest( &pref->refcnt)) {
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
			usleep(1);
			continue;
		}
		// trying to increase local reference counter w/ same serial.
		if(atomicCompareAndSet(
			RefLocal_((uintptr_t)pref + rcnt_old),
			RefLocal_((uintptr_t)pref + rcnt_new),
			&const_cast<atomic_shared_ptr<T> *>(this)->m_ref))
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
	atomicInc( &pref->refcnt);
	leave_scan_(pref);
	readBarrier(); //for *pref..
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
			if(atomicCompareAndSet(
				RefLocal_((uintptr_t)pref + rcnt_old),
				RefLocal_((uintptr_t)pref + rcnt_new),
				&const_cast<atomic_shared_ptr<T> *>(this)->m_ref))
				break;
			if(pref == pref_())
				continue; // try again.
		}
		// local reference has released by other processes.
		if(atomicDecAndTest( &pref->refcnt)) {
			readBarrier(); //for *pref.
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
		atomicInc( &newr.pref_()->refcnt);
		writeBarrier(); //for *newr.pref_().
	}
	for(;;) {
		Refcnt rcnt_old, rcnt_new;
		pref = reserve_scan_( &rcnt_old);
		if(pref != oldr.pref_()) {
			if(pref) {
				if( !NOSWAP) {
					atomicInc( &pref->refcnt);
				}
				leave_scan_(pref);
			}
			if(newr.pref_())
				atomicDec( &newr.pref_()->refcnt);
			if( !NOSWAP) {
				readBarrier(); //for *pref and *oldr.pref_();
				if(oldr.pref_()) {
					// decreasing global reference counter.
					if(atomicDecAndTest(&oldr.pref_()->refcnt)) {
						this->deleter(oldr.pref_());
					}
				}
				oldr.m_ref = (RefLocal_)pref;
			}
			return false;
		}
		if(pref && (rcnt_old != 1u)) {
			atomicAdd(&pref->refcnt, rcnt_old - 1u);
		}
		rcnt_new = 0;
		if(atomicCompareAndSet(
			RefLocal_((uintptr_t)pref + rcnt_old),
			RefLocal_((uintptr_t)newr.pref_() + rcnt_new),
			&this->m_ref))
			break;
		if(pref) {
			assert(rcnt_old);
			if(rcnt_old != 1u)
				atomicAdd( &pref->refcnt, (Refcnt)( -(int)(rcnt_old - 1u)));
			leave_scan_(pref);
		}
	}
	if(pref) {
		atomicDec( &pref->refcnt);
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
template <typename T>
inline void
local_shared_ptr<T>::swap(local_shared_ptr &r) {
	RefLocal_ x = this->m_ref;
	this->m_ref = r.m_ref;
	r.m_ref = x;
}

template <typename T>
void
local_shared_ptr<T>::swap(atomic_shared_ptr<T> &r) {
	Ref *pref;
	if(this->m_ref) writeBarrier(); //for the contents held by this.
	for(;;) {
		Refcnt rcnt_old, rcnt_new;
		pref = r.reserve_scan_( &rcnt_old);
		if(pref && (rcnt_old != 1u)) {
			atomicAdd( &pref->refcnt, rcnt_old - 1u);
		}
		rcnt_new = 0;
		if(atomicCompareAndSet(
			RefLocal_((uintptr_t)pref + rcnt_old),
			RefLocal_((uintptr_t)this->m_ref + rcnt_new),
			&r.m_ref))
			break;
		if(pref) {
			assert(rcnt_old);
			if(rcnt_old != 1u)
				atomicAdd( &pref->refcnt, (Refcnt)( -(int)(rcnt_old - 1u)));
			r.leave_scan_(pref);
		}
	}
	this->m_ref = (RefLocal_)pref;
	if(pref) readBarrier(); //for *pref.
}

#endif /*ATOMIC_SMART_PTR_H_*/
