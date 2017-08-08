/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
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

#ifdef USE_QTHREAD
    #include <QThread>
    #define threadid_t Qt::HANDLE
    inline threadid_t threadID() {return QThread::currentThreadId();}
    #define is_thread_equal(x,y) ((x) == (y))
    #include <QMutex>
    #include <QWaitCondition>
    #include <QThread>
#else
    #include <thread>
    using threadid_t = std::thread::id;
    inline threadid_t threadID() noexcept {return std::this_thread::get_id();}
    inline bool is_thread_equal(threadid_t x, threadid_t y) noexcept {return x == y;}
    #include <sys/mman.h>
#endif

#include "threadlocal.h"

//! Lock mutex during its life time.
template <class Mutex>
struct XScopedLock {
    explicit XScopedLock(Mutex &mutex) : m_mutex(mutex) {
        m_mutex.lock();
    }
    ~XScopedLock() {
        m_mutex.unlock();
    }
    XScopedLock(const XScopedLock &) = delete;
    XScopedLock& operator=(const XScopedLock &) = delete;
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
    XScopedTryLock(const XScopedTryLock &) = delete;
    XScopedTryLock& operator=(const XScopedTryLock &) = delete;
private:
    Mutex &m_mutex;
    bool m_bLocking;
};

/*! non-recursive mutex.
 * double lock is inhibited.
 * \sa XRecursiveMutex.
 */
class DECLSPEC_KAME XMutex {
public:
    XMutex();
    ~XMutex();
    XMutex(const XMutex &) = delete;
    XMutex& operator=(const XMutex &) = delete;

    void lock();
    void unlock();
    //! \return true if locked.
    bool trylock();
protected:
#ifdef USE_QTHREAD
    QMutex m_mutex;
#elif defined USE_PTHREAD
    pthread_mutex_t m_mutex;
#endif
};

//! condition class.
class DECLSPEC_KAME XCondition : public XMutex
{
public:
    XCondition();
    ~XCondition();
    XCondition(const XCondition &) = delete;
    XCondition& operator=(const XCondition &) = delete;
    //! Lock me before calling me.
    //! go asleep until signal is emitted.
    //! \param usec if non-zero, timeout occurs after \a usec.
    //! \return zero if locked thread is waked up.
    int wait(int usec = 0);
    //! wake-up at most one thread.
    //! \sa broadcast()
    void signal();
    //! wake-up all waiting threads.
    //! \sa signal()
    void broadcast();
private:
#ifdef USE_QTHREAD
    QWaitCondition m_cond;
#elif defined USE_PTHREAD
    pthread_cond_t m_cond;
#endif
};

//! recursive mutex.
class DECLSPEC_KAME XRecursiveMutex {
public:
    XRecursiveMutex() {}
	~XRecursiveMutex() {}
    XRecursiveMutex(const XRecursiveMutex &) = delete;
    XRecursiveMutex &operator=(const XRecursiveMutex &) = delete;

	void lock() {
        if(!is_thread_equal(m_lockingthread, threadID())) {
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
            m_lockingthread = threadid_t{};
			m_mutex.unlock();
		}
	}
	//! \return true if locked.
	bool trylock() {
        if(!is_thread_equal(m_lockingthread, threadID())) {
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
        return is_thread_equal(m_lockingthread, threadID());
	}
private:
	XMutex m_mutex;
    threadid_t m_lockingthread = {};
	int m_lockcount;
};


//! Guard pattern for a thread object.
class XThread {
public:
    //! Starts up a new thread.
    //! \param f a member function of an object \p r, r->f(const atomic<bool>& terminated, args...)
    template <class X, class T, class R, class... Args>
    XThread(const shared_ptr<X> &r, R(T::*func)(const atomic<bool> &, Args...), Args&&...args);
    template <class X, class Function, class... Args>
    XThread(const shared_ptr<X> &r, Function &&func, Args&&...args);
    //! Joins a thread here if it is still un-joined (joinable).
    ~XThread();
    XThread(const XThread &) = delete;
    XThread& operator=(const XThread &) = delete;
    XThread(XThread &&) noexcept = delete;

    //! joins a running thread.
    void join() {m_thread.join();}
    //! sets termination flag.
    void terminate() { m_isTerminated = true;}
    //! fetches termination flag.
    bool isTerminated() const noexcept {return m_isTerminated;}
private:
    atomic<bool> m_isTerminated = false;
    std::thread m_thread;
};

template <class X, class T, class R, class... Args>
XThread::XThread(const shared_ptr<X> &r, R(T::*func)(const atomic<bool> &, Args...), Args&&...args) :
    m_thread(
        [r, func, this](Args&&...args) {
            auto obj = dynamic_pointer_cast<T>(r);
            (obj.get()->*func)(std::ref(m_isTerminated), std::forward<Args>(args)...);
        }, std::forward<Args>(args)...) {
}

template <class X, class Function, class... Args>
XThread::XThread(const shared_ptr<X> &r, Function &&func, Args&&...args) :
    m_thread(
        [r, this](Function &&func, Args&&...args) {
            func(std::ref(m_isTerminated), std::forward<Args>(args)...);
        }, std::forward<Function>(func), std::forward<Args>(args)...) {
}

#endif
