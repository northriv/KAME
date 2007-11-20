/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

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
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_pod.hpp>
#include <boost/type_traits/is_integral.hpp>

//! Lock-free synchronizations.

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__
#include <atomic_prv_x86.h>
#else
#if defined __ppc__ || defined __POWERPC__ || defined __powerpc__
#include <atomic_prv_ppc.h>
#else
#error Unsupported processor
#endif // __ppc__
#endif // __i386__

template <typename T, class Enable = void > class atomic;

//! atomic access to POD type capable of CAS.
template <typename T>
class atomic_pod_cas
{
public:
	atomic_pod_cas() {}
	atomic_pod_cas(T t) : m_var(t) {}
	atomic_pod_cas(const atomic_pod_cas &t) : m_var(t) {}
	~atomic_pod_cas() {}
	operator T() const {readBarrier(); return m_var;}
	atomic_pod_cas &operator=(T t) {m_var = t; writeBarrier(); return *this;}
	T swap(T newv) {
		T old = atomicSwap(newv, &m_var);
		writeBarrier();
		return old;
	}
	bool compareAndSet(T oldv, T newv) {
		bool ret = atomicCompareAndSet(oldv, newv, &m_var);
		if(ret) writeBarrier();
		return ret;
	}
protected:
	T m_var;
};

//! atomic access to POD type capable of CAS2.
template <typename T>
class atomic_pod_cas2
{
public:
	atomic_pod_cas2() {}
	atomic_pod_cas2(T t) : m_var(t) {}
	atomic_pod_cas2(const atomic_pod_cas2 &t) : m_var(t) {}
	~atomic_pod_cas2() {}
	operator T() const {
		readBarrier();
#ifdef HAVE_ATOMIC_RW64
		C_ASSERT(sizeof(T) == 8);
		T x __attribute__((aligned(8)));
		atomicRead64(&x, m_var);
		return x;
#else
		for(;;) {
			T oldv = m_var;
			if(atomicCompareAndSet(oldv, oldv, &m_var)) {
				return oldv;
			}
		}
#endif
	}
	atomic_pod_cas2 &operator=(T t) {
#ifdef HAVE_ATOMIC_RW64
		C_ASSERT(sizeof(T) == 8);
		volatile T x __attribute__((aligned(8))) = t;
		atomicWrite64(x, &m_var);
#else
		for(;;) {
			T oldv = m_var;
			if(atomicCompareAndSet(oldv, t, &m_var))
				break;
		}
#endif
		writeBarrier();
		return *this;
	}
	T swap(T newv) {
		for(;;) {
			T oldv = m_var;
			if(atomicCompareAndSet(oldv, newv, &m_var)) {
				writeBarrier();
				return oldv;
			}
		}
	}
	bool compareAndSet(T oldv, T newv) {
		bool ret = atomicCompareAndSet(oldv, newv, &m_var);
		if(ret) writeBarrier();
		return ret;
	}
protected:
#ifdef HAVE_ATOMIC_RW64
	T m_var __attribute__((aligned(8)));
#else
	mutable T m_var;
#endif
};

//! atomic access to POD type capable of CAS2.
template <typename T>
class atomic<T, typename boost::enable_if_c<
(sizeof(int_cas2_both) == sizeof(T)) && boost::is_pod<T>::value>::type>
: public atomic_pod_cas2<T>
{
public:
	atomic() {}
	atomic(T t) : atomic_pod_cas2<T>(t) {}
	atomic(const atomic &t) : atomic_pod_cas2<T>(t) {}
};

//! atomic access to POD type capable of CAS.
template <typename T>
class atomic<T, typename boost::enable_if_c<
(sizeof(int_cas_max) >= sizeof(T)) && boost::is_pod<T>::value && 
!boost::is_integral<T>::value>::type>
: public atomic_pod_cas<T>
{
public:
	atomic() {}
	atomic(T t) : atomic_pod_cas<T>(t) {}
	atomic(const atomic &t) : atomic_pod_cas<T>(t) {}
};

//! atomic access to integer-POD-type capable of CAS.
template <typename T>
class atomic<T, typename boost::enable_if_c<
(sizeof(int_cas_max) >= sizeof(T)) && boost::is_integral<T>::value>::type >
: public atomic_pod_cas<T>
{
public:
	atomic() : atomic_pod_cas<T>((T)0) {}
	atomic(T t) : atomic_pod_cas<T>(t) {}
	atomic(const atomic &t) : atomic_pod_cas<T>(t) {}
	~atomic() {}
	atomic &operator++() {atomicInc(&this->m_var); writeBarrier(); return *this;}
	atomic &operator--() {atomicDecAndTest(&this->m_var); writeBarrier(); return *this;}
	atomic &operator+=(T t) {atomicAdd(&this->m_var, t); writeBarrier(); return *this;}
	atomic &operator-=(T t) {atomicAdd(&this->m_var, -t); writeBarrier(); return *this;}
	bool decAndTest() {
		bool ret = atomicDecAndTest(&this->m_var);
		writeBarrier();
		return ret;
	}
	bool addAndTest(T t) {
		bool ret = atomicAddAndTest(&this->m_var, t);
		writeBarrier();
		return ret;
	}
};


#endif /*ATOMIC_H_*/
