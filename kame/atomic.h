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
#ifndef ATOMIC_H_
#define ATOMIC_H_

#include <stdint.h>

#include "atomic_smart_ptr.h"

template <typename T, class Enable = void > class atomic;

//! atomic access to POD type capable of CAS.
template <typename T>
class atomic_pod_cas {
public:
	atomic_pod_cas() {}
	atomic_pod_cas(T t) : m_var(t) {}
	atomic_pod_cas(const atomic_pod_cas &t) : m_var(t) {}
	~atomic_pod_cas() {}
	operator T() const { T x = m_var; readBarrier(); return x;}
	atomic_pod_cas &operator=(T t) {
		writeBarrier(); m_var = t; return *this;
	}
	atomic_pod_cas &operator=(const atomic_pod_cas &x) {
		writeBarrier(); m_var = x.m_var; return *this;
	}
	T swap(T newv) {
		writeBarrier();
		T old = atomicSwap(newv, &m_var);
		return old;
	}
	bool compareAndSet(T oldv, T newv) {
		writeBarrier();
		bool ret = atomicCompareAndSet(oldv, newv, &m_var);
		return ret;
	}
protected:
	T m_var;
};

//! atomic access to POD type capable of CAS2.
template <typename T>
class atomic_pod_cas2 {
public:
	atomic_pod_cas2() {}
	atomic_pod_cas2(T t) : m_var(t) {}
	atomic_pod_cas2(const atomic_pod_cas2 &t) : m_var(t) {}
	~atomic_pod_cas2() {}
	operator T() const {
		for(;;) {
			T oldv = m_var;
			if(atomicCompareAndSet(oldv, oldv, &m_var)) {
				readBarrier();
				return oldv;
			}
		}
	}
	atomic_pod_cas2 &operator=(T t) {
		writeBarrier();
		for(;;) {
			T oldv = m_var;
			if(atomicCompareAndSet(oldv, t, &m_var))
				break;
		}
		return *this;
	}
	atomic_pod_cas2 &operator=(const atomic_pod_cas2 &x) {
		*this = (T)x;
		return *this;
	}
	T swap(T newv) {
		for(;;) {
			writeBarrier();
			T oldv = m_var;
			if(atomicCompareAndSet(oldv, newv, &m_var)) {
				return oldv;
			}
		}
	}
	bool compareAndSet(T oldv, T newv) {
		writeBarrier();
		bool ret = atomicCompareAndSet(oldv, newv, &m_var);
		return ret;
	}
protected:
	union {
		T m_var;
		int64_t for_alignment;
	};
};

//! atomic access to POD type capable of CAS2.
template <typename T>
class atomic<T, typename std::enable_if<
(sizeof(int_cas2) * 2 == sizeof(T)) && std::is_pod<T>::value>::type>
: public atomic_pod_cas2<T> {
public:
	atomic() {}
	atomic(T t) : atomic_pod_cas2<T>(t) {}
	atomic(const atomic &t) : atomic_pod_cas2<T>(t) {}
};

//! atomic access to POD type capable of CAS.
template <typename T>
class atomic<T, typename std::enable_if<
(sizeof(int_cas_max) >= sizeof(T)) && std::is_pod<T>::value &&
!std::is_integral<T>::value>::type>
: public atomic_pod_cas<T> {
public:
	atomic() {}
	atomic(T t) : atomic_pod_cas<T>(t) {}
	atomic(const atomic &t) : atomic_pod_cas<T>(t) {}
};

//! atomic access to integer-POD-type capable of CAS.
template <typename T>
class atomic<T, typename std::enable_if<
(sizeof(int_cas_max) >= sizeof(T)) && std::is_integral<T>::value>::type >
: public atomic_pod_cas<T> {
public:
	atomic() : atomic_pod_cas<T>((T)0) {}
	atomic(T t) : atomic_pod_cas<T>(t) {}
	atomic(const atomic &t) : atomic_pod_cas<T>(t) {}
	~atomic() {}
	//! Note that the return value is atomically given.
	atomic &operator++() {atomicInc( &this->m_var); writeBarrier(); return *this;}
	//! Note that the return value is atomically given.
	atomic &operator--() {atomicDecAndTest( &this->m_var); writeBarrier(); return *this;}
	//! Note that the return value is atomically given.
	atomic &operator+=(T t) {atomicAdd( &this->m_var, t); writeBarrier(); return *this;}
	//! Note that the return value is atomically given.
	atomic &operator-=(T t) {atomicAdd( &this->m_var, -t); writeBarrier(); return *this;}
	bool decAndTest() {
		writeBarrier();
		bool ret = atomicDecAndTest( &this->m_var);
		readBarrier();
		return ret;
	}
	bool addAndTest(T t) {
		writeBarrier();
		bool ret = atomicAddAndTest( &this->m_var, t);
		readBarrier();
		return ret;
	}
};

//! Atomic access for a copy-able class which does not require transactional writing.
template <typename T, class Enable>
class atomic {
public:
	atomic() : m_var(new T) {}
	atomic(T t) : m_var(new T(t)) {}
	atomic(const atomic &t) : m_var(t.m_var) {}
	~atomic() {}
	operator T() const {
		local_shared_ptr<T> x = m_var;
		return *x;
	}
	atomic &operator=(T t) {
		m_var.reset(new T(t));
		return *this;
	}
	atomic &operator=(const atomic &x) {
		m_var = x.m_var;
		return *this;
	}
	bool compareAndSet(const T &oldv, const T &newv) {
		local_shared_ptr<T> oldx(m_var);
		if( *oldx != oldv)
			return false;
		local_shared_ptr<T> newx(new T(newv));
		bool ret = m_var.compareAndSet(oldx, newx);
		return ret;
	}
protected:
	atomic_shared_ptr<T> m_var;
};

#endif /*ATOMIC_H_*/
