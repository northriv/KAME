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

//! Lock mutex during its life time.
template <class Mutex>
struct XScopedLock
{
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
struct XScopedTryLock
{
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

/*! non-recursive mutex.
 * double lock is inhibited.
 * \sa XRecursiveMutex.
 */
class XMutex
{
 public:
  XMutex();
  ~XMutex();

  void lock();
  void unlock();
  //! \return true if locked.
  bool trylock();
 protected:
  pthread_mutex_t m_mutex;
};

//! recursive mutex.
class XRecursiveMutex
{
 public:
  XRecursiveMutex();
  ~XRecursiveMutex();

  void lock();
  //! unlock me with locking thread.
  void unlock();
  //! \return true if locked.
  bool trylock();
  //! \return true if the current thread is locking mutex.
  bool isLockedByCurrentThread() const;
 private:
  XMutex m_mutex;
  threadid_t m_lockingthread;
  int m_lockcount;
};

//! condition class.
class XCondition : public XMutex
{
 public:
  XCondition();
  ~XCondition();
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
  pthread_cond_t m_cond;
};

//! create a new thread.
template <class T>
class XThread
{
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
: m_startarg(new targ)
{
  m_startarg->obj = dynamic_pointer_cast<T>(t);
  ASSERT(m_startarg->obj);
  m_startarg->func = func;
  m_startarg->is_terminated = false;
}

template <class T>
void
XThread<T>::resume()
{
  m_startarg->this_ptr = m_startarg;
  int ret =
    pthread_create((pthread_t*)&m_threadid, NULL,
           &XThread<T>::xthread_start_routine , m_startarg.get());
  dbgPrint(QString("New Thread 0x%1.").arg((unsigned int)m_threadid, 0, 16));
  ASSERT(!ret);
}
template <class T>
void *
XThread<T>::xthread_start_routine(void *x)
{
  shared_ptr<targ> arg = ((targ *)x)->this_ptr;
  arg->this_ptr.reset();
  void *p = ((arg->obj.get())->*(arg->func))(arg->is_terminated);
  arg->obj.reset();
  return p;
}
template <class T>
void 
XThread<T>::waitFor(void **retval)
{
  int ret = pthread_join(m_threadid, retval);
  ASSERT(!ret);
}
template <class T>
void 
XThread<T>::terminate()
{
    m_startarg->is_terminated = true;
}

#endif
