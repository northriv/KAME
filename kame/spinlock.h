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
		if(atomicSwap(0, &m_flag)) {
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
			XScopedLock<XCondition> lock(m_cond);
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
	XCondition m_cond;
	int m_waitcnt;
};

#endif /*SPINLOCK_H_*/
