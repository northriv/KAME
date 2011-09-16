/***************************************************************************
		Copyright (C) 2002-2011 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef threadH
#define threadH
//---------------------------------------------------------------------------
#include "support.h"
#include "atomic.h"

#if defined __WIN32__ || defined WINDOWS
#define threadID() GetCurrentThreadId()
#endif

#define threadid_t pthread_t
#define threadID() pthread_self()

#include "threadlocal.h"
#include <sys/mman.h>

//! Lock mutex during its life time.
template <class Mutex>
struct XScopedLock {
    explicit XScopedLock(Mutex &mutex) : m_mutex(mutex) {
        m_mutex.lock();
    }
    ~XScopedLock() {
        m_mutex.unlock();
    }
private:
    Mutex &m_mutex;
};

//! Lock mutex during its life time.
template <class Mutex>
struct XScopedTryLock {
    explicit XScopedTryLock(Mutex &mutex) : m_mutex(mutex) {
		m_bLocking = m_mutex.trylock();
    }
    ~XScopedTryLock() {
		if(m_bLocking) m_mutex.unlock();
    }
    bool operator!() const {
        return !m_bLocking;
    }
    operator bool() const {
        return m_bLocking;
    }
private:
    Mutex &m_mutex;
    bool m_bLocking;
};

#include "pthreadlock.h"

typedef XPthreadMutex XMutex;
typedef XPthreadCondition XCondition;

//! recursive mutex.
class XRecursiveMutex {
public:
	XRecursiveMutex() {
		m_lockingthread = (threadid_t)-1;
	}
	~XRecursiveMutex() {}

	void lock() {
		if(!pthread_equal(m_lockingthread, threadID())) {
			m_mutex.lock();
			m_lockcount = 1;
			m_lockingthread = threadID();
		}
		else
			m_lockcount++;
	}
	//! unlock me with locking thread.
	void unlock() {
		m_lockcount--;
		if(m_lockcount == 0) {
			m_lockingthread = (threadid_t)-1;
			m_mutex.unlock();
		}
	}
	//! \return true if locked.
	bool trylock() {
		if(!pthread_equal(m_lockingthread, threadID())) {
			if(m_mutex.trylock()) {
				m_lockcount = 1;
				m_lockingthread = threadID();
			}
			else
				return false;
		}
		else
			m_lockcount++;
		return true;
	}
	//! \return true if the current thread is locking mutex.
	bool isLockedByCurrentThread() const {
	    return pthread_equal(m_lockingthread, threadID());
	}
private:
	XMutex m_mutex;
	threadid_t m_lockingthread;
	int m_lockcount;
};

//! create a new thread.
template <class T>
class XThread {
public:
	/*! use resume() to start a thread.
	 * \p X must be super class of \p T.
	 */
	template <class X>
	XThread(const shared_ptr<X> &t, void *(T::*func)(const atomic<bool> &));
	~XThread() {terminate();}
	//! resume a new thread.
	void resume();
	/*! join a running thread.
	 * should be called before destruction.
	 * \param retval a pointer to return value from a thread.
	 */
	void waitFor(void **retval = 0L);
	//! set termination flag.
	void terminate();
	//! fetch termination flag.
	bool isTerminated() const {return m_startarg->is_terminated;}
private:
	pthread_t m_threadid;
	struct targ{
		shared_ptr<targ> this_ptr;
		shared_ptr<T> obj;
		void *(T::*func)(const atomic<bool> &);
		atomic<bool> is_terminated;
	};
	shared_ptr<targ> m_startarg;
	static void * xthread_start_routine(void *);
};

template <class T>
template <class X>
XThread<T>::XThread(const shared_ptr<X> &t, void *(T::*func)(const atomic<bool> &))
	: m_startarg(new targ) {
	m_startarg->obj = dynamic_pointer_cast<T>(t);
	assert(m_startarg->obj);
	m_startarg->func = func;
	m_startarg->is_terminated = false;
}

template <class T>
void
XThread<T>::resume() {
	m_startarg->this_ptr = m_startarg;
	int ret =
		pthread_create((pthread_t*)&m_threadid, NULL,
					   &XThread<T>::xthread_start_routine , m_startarg.get());
	assert( !ret);
}
template <class T>
void *
XThread<T>::xthread_start_routine(void *x) {
	shared_ptr<targ> arg = ((targ *)x)->this_ptr;
	if(g_bMLockAlways) {
		if(( mlockall(MCL_CURRENT | MCL_FUTURE ) == 0)) {
			dbgPrint("MLOCKALL succeeded.");
		}
		else{
			dbgPrint("MLOCKALL failed.");
		}
	}
	if(g_bUseMLock)
		mlock(&arg, 8192uL); //reserve stack.

	arg->this_ptr.reset();
	void *p = ((arg->obj.get())->*(arg->func))(arg->is_terminated);
	arg->obj.reset();

	return p;
}
template <class T>
void 
XThread<T>::waitFor(void **retval) {
	pthread_join(m_threadid, retval);
//  assert(!ret);
}
template <class T>
void 
XThread<T>::terminate() {
    m_startarg->is_terminated = true;
}

#endif
