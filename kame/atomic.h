/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
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

//! Lock-free synchronizations.

#if defined __i386__ || defined __i486__ || defined __i586__ || defined __i686__
#include "atomic_prv_x86.h"
#else
#if defined __ppc__ || defined __POWERPC__ || defined __powerpc__
#include "atomic_prv_ppc.h"
#else
#error Unsupported processor
#endif // __ppc__
#endif // __i386__

template <typename T>
class atomic
{
public:
	atomic() : m_var(0) {}
	atomic(T t) : m_var(t) {}
	atomic(const atomic &t) : m_var(t) {}
	~atomic() {}
	operator T() const {readBarrier(); return m_var;}
	atomic &operator=(T t) {m_var = t; writeBarrier(); return *this;}
	atomic &operator++() {atomicInc(&m_var); writeBarrier(); return *this;}
	atomic &operator--() {atomicDecAndTest(&m_var); writeBarrier(); return *this;}
	atomic &operator+=(T t) {atomicAdd(&m_var, t); writeBarrier(); return *this;}
	atomic &operator-=(T t) {atomicAdd(&m_var, -t); writeBarrier(); return *this;}
	T swap(T newv) {
		T old = atomicSwap(newv, &m_var);
		writeBarrier();
		return old;
	}
	bool decAndTest() {
		bool ret = atomicDecAndTest(&m_var);
		writeBarrier();
		return ret;
	}
	bool addAndTest(T t) {
		bool ret = atomicAddAndTest(&m_var, t);
		writeBarrier();
		return ret;
	}
	bool compareAndSet(T oldv, T newv) {
		bool ret = atomicCompareAndSet(oldv, newv, &m_var);
		writeBarrier();
		return ret;
	}
private:
	T m_var;
};

#endif /*ATOMIC_H_*/
