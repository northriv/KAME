/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            LICENSE-APACHE-2.0 in this directory)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see LICENSE-GPL-2.0 in this directory).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
***************************************************************************/
//---------------------------------------------------------------------------

#include "xthread.h"

//---------------------------------------------------------------------------

#ifdef USE_QTHREAD

XMutex::XMutex() {}
XMutex::~XMutex() {}

void XMutex::lock() {m_mutex.lock();}

void XMutex::unlock() {m_mutex.unlock();}

bool XMutex::trylock() {return m_mutex.tryLock();}

XCondition::XCondition() : XMutex() {
}
XCondition::~XCondition() {
}

int XCondition::wait(int usec) {
    bool ret;
    if(usec)
        ret = m_cond.wait( &m_mutex, usec / 1000);
    else
        ret = m_cond.wait( &m_mutex);
    return ret ? 0 : 1;
}

void XCondition::signal() {
    m_cond.wakeOne();
}

void XCondition::broadcast() {
    m_cond.wakeAll();
}


#else
    #ifdef USE_PTHREAD

    #include <assert.h>
    #include <errno.h>
    #include <algorithm>
    #include <sys/time.h>

    XMutex::XMutex() {
        pthread_mutexattr_t attr;
        int ret;
        ret = pthread_mutexattr_init( &attr);
        if(DEBUG_XTHREAD) assert( !ret);

        ret = pthread_mutex_init( &m_mutex, &attr);
        if(DEBUG_XTHREAD) assert( !ret);

        ret = pthread_mutexattr_destroy( &attr);
        if(DEBUG_XTHREAD) assert( !ret);
    }

    XMutex::~XMutex() {
        int ret = pthread_mutex_destroy( &m_mutex);
        if(DEBUG_XTHREAD) assert( !ret);
    }
    void
    XMutex::lock() {
        int ret = pthread_mutex_lock( &m_mutex);
        if(DEBUG_XTHREAD) assert( !ret);
    }
    bool
    XMutex::trylock() {
        int ret = pthread_mutex_trylock(&m_mutex);
        if(DEBUG_XTHREAD) assert(ret != EINVAL);
        return (ret == 0);
    }
    void
    XMutex::unlock() {
        int ret = pthread_mutex_unlock( &m_mutex);
        if(DEBUG_XTHREAD) assert( !ret);
    }

    XCondition::XCondition() : XMutex() {
        int ret = pthread_cond_init( &m_cond, NULL);
        if(DEBUG_XTHREAD) assert( !ret);
    }
    XCondition::~XCondition() {
        int ret = pthread_cond_destroy( &m_cond);
        if(DEBUG_XTHREAD) assert( !ret);
    }
    int
    XCondition::wait(int usec) {
        int ret;
        if(usec > 0) {
            struct timespec abstime;
            timeval tv;
            long nsec;
            gettimeofday(&tv, NULL);
            abstime.tv_sec = tv.tv_sec;
            nsec = (tv.tv_usec + usec) * 1000;
            if(nsec >= 1000000000) {
                nsec -= 1000000000; abstime.tv_sec++;
            }
            abstime.tv_nsec = nsec;
            ret = pthread_cond_timedwait(&m_cond, &m_mutex, &abstime);
        }
        else {
            ret = pthread_cond_wait(&m_cond, &m_mutex);
        }
        return ret;
    }
    void
    XCondition::signal() {
        int ret = pthread_cond_signal( &m_cond);
        if(DEBUG_XTHREAD) assert( !ret);
    }
    void
    XCondition::broadcast() {
        int ret = pthread_cond_broadcast( &m_cond);
        if(DEBUG_XTHREAD) assert( !ret);
    }
    #else // !USE_PTHREAD — portable C++11 fallback (Windows tests etc.)

    #include <chrono>

    XMutex::XMutex() = default;
    XMutex::~XMutex() = default;
    void XMutex::lock()    { m_mutex.lock();    }
    void XMutex::unlock()  { m_mutex.unlock();  }
    bool XMutex::trylock() { return m_mutex.try_lock(); }

    XCondition::XCondition() : XMutex() {}
    XCondition::~XCondition() {}
    int XCondition::wait(int usec) {
        // The mutex is already locked by the caller (per the
        // pthread/Qt contract). std::condition_variable::wait_for
        // requires a unique_lock, so adopt the held lock without
        // re-locking, wait, then release ownership so the caller's
        // unlock path is unchanged.
        std::unique_lock<std::mutex> lk(m_mutex, std::adopt_lock);
        std::cv_status st = (usec > 0)
            ? m_cond.wait_for(lk, std::chrono::microseconds(usec))
            : (m_cond.wait(lk), std::cv_status::no_timeout);
        lk.release();
        return (st == std::cv_status::timeout) ? 1 : 0;
    }
    void XCondition::signal()    { m_cond.notify_one(); }
    void XCondition::broadcast() { m_cond.notify_all(); }

    #endif // USE_PTHREAD
#endif // USE_QTHREAD

XThread::~XThread() {
    terminate();
    if(m_thread.joinable()) {
        if(m_thread.get_id() == std::this_thread::get_id())
            m_thread.detach();
        else
            m_thread.join();
    }
}

