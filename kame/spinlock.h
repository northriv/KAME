/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef SPINLOCK_H_
#define SPINLOCK_H_

#include <atomic.h>
#include <thread.h>

class XSpinLock
{
public:
	XSpinLock() : m_flag(1) {
		writeBarrier();
	}
	~XSpinLock() {
		readBarrier();
		ASSERT(m_flag == 1);
	}

	void unlock() {
		writeBarrier();
		m_flag = 1;
		writeBarrier();
	}
	//! \return true if locked.
	bool trylock() {
		if(!m_flag) return false;
		if(atomicSwap(0, &m_flag)) {
//		if(atomicCompareAndSet(1, 0, &m_flag)) {
			readBarrier();
			return true;
		}
		return false;
	}
	void lock() {
		for(;;) {
			if(trylock())
				return;
			pause4spin(); //for HTT.
			readBarrier(); //redundant.
			continue;
		}
	}
protected:
	int m_flag;
};

class XAdaptiveSpinLock : public XSpinLock
{
public:
	XAdaptiveSpinLock() : m_waitcnt(0) {}
	~XAdaptiveSpinLock() {}
	void lock() {
		for(unsigned int lp = 0;; lp++) {
			if(trylock())
				return;
			if(lp < 4096) {
				pause4spin();
				readBarrier();
				continue;
			}
			XScopedLock<XPthreadCondition> lock(m_cond);
			m_waitcnt++;
			m_cond.wait(2);
			m_waitcnt--;
		}
	}
	void unlock() {
		XSpinLock::unlock();
		if(m_waitcnt) {
			m_cond.signal();
		}
	}
protected:
	XPthreadCondition m_cond;
	int m_waitcnt;
};


#endif /*SPINLOCK_H_*/
