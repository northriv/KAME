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
#ifndef signalH
#define signalH

/*!
* M/M Signal & Slot system
* Listener is called by the current thread or will be called by the main thread.
*/

#include "support.h"
#include "xtime.h"
#include "thread.h"
#include "atomic_smart_ptr.h"
#include <deque>

//! Detect whether the current thread is the main thread.
bool isMainThread();

template <class tArg>
class XTalker;

//! Base class of listener, which holds pointers to object and function.
//! Hold me by shared_ptr.
class XListener
{
 public:
  virtual ~XListener();
  //! block emission of signals to this listener.
  //! use this with owner's responsibility.
  //! \sa unmask(), _XTalkerBase::mask()
  void mask();
  //! un-block emission of signals to this listener.
  //! use this with owner's responsibility.
  //! \sa mask(), _XTalkerBase::unmask()
  void unmask();
  
  unsigned int delay_ms() const {return m_delay_ms;}
 protected:
  template <class tArg>
  friend class XTalker;
  XListener(bool mainthreadcall, bool avoid_dup, unsigned int delay_ms);
  const bool m_bMainThreadCall;
  const bool m_bAvoidDup;
  const unsigned int m_delay_ms;
  atomic<bool> m_bMasked;
};

#include "xsignal_prv.h"

class _XTalkerBase
{
 protected:
  _XTalkerBase();

 public:
  virtual ~_XTalkerBase();

  //! block emission of signals from this talker.
  //! use this with owner's responsibility.
  //! \sa unmask()
  void mask();
  //! un-block emission of signals from this talker.
  //! use this with owner's responsibility.
  //! \sa mask()
  void unmask();
 protected:
  atomic<bool> m_bMasked;
};

struct _XTransaction
{   
    _XTransaction() : registered_time(timeStamp()) {}
    virtual ~_XTransaction() {}
    const unsigned long registered_time;
    virtual bool talkBuffered() = 0;
};

//! M/M Listener and Talker model
//! \sa XListener, XSignalStore
//! \p tArg: value which will be derivered
//! \p tArgWrapper: copied argument, will be released by GC someday
template <class tArg>
class XTalker : public _XTalkerBase
{
 public:
  XTalker();
  virtual ~XTalker();
  
  //! Associate XTalker to XListener
  //! Talker will call user member function of \a listener.
  //! This function can be called over once.
  //! \sa disconnect(), mask(), unmask()
  //! XListener holds a pointer of a static function
  //! \param mainthreadcall if true, listener is called by the main thread.
  //!        this is useful to handle with GUI.
  //! \param func a pointer to static function
  //! \param avoid_dup If \a mainthreadcall is true,
  //!        when stored signals to the listener are duplicated,
  //!        this signal is copied to the first request.
  //! \param delay_ms If non-zero, this signal is buffered at least this \a delay_ms.
  //!        ignored if avoid_dup is false.
  //! \return shared_ptr to \p XListener.
  shared_ptr<XListener> connectStatic(bool mainthreadcall, 
    void (*func)(const tArg &),
    bool avoid_dup = false, unsigned int delay_ms = 0);
  //! Associate XTalker to XListener
  //! Talker will call user member function of \a listener.
  //! This function can be called over once.
  //! \sa disconnect(), mask(), unmask()
  //! XListener holds weak_ptr() of the instance
  //! \param mainthreadcall if true, listener is called by the main thread.
  //!        this is useful to handle with GUI.
  //! \param obj a pointer to object
  //! \param func a pointer to member function
  //! \param avoid_dup If \a mainthreadcall is true,
  //!        when stored signals to the listener are duplicated,
  //!        this signal is copied to the first request.
  //! \param delay_ms If non-zero, this signal is buffered at least this \a delay_ms.
  //!        ignored if avoid_dup is false.
  //! \return shared_ptr to \p XListener.
  template <class tObj, class tClass>
  shared_ptr<XListener> connectWeak(bool mainthreadcall, const shared_ptr<tObj> &obj,
    void (tClass::*func)(const tArg &),
    bool avoid_dup = false, unsigned int delay_ms = 0);
  //! Associate XTalker to XListener
  //! Talker will call user member function of \a listener.
  //! This function can be called over once.
  //! \sa disconnect(), mask(), unmask()
  //! XListener holds shared_ptr() of the instance
  //! \param mainthreadcall if true, listener is called by the main thread.
  //!        this is useful to handle with GUI.
  //! \param obj a pointer to object
  //! \param func a pointer to member function
  //! \param avoid_dup If \a mainthreadcall is true,
  //!        when stored signals to the listener are duplicated,
  //!        this signal is copied to the first request.
  //! \param delay_ms If non-zero, this signal is buffered at least this \a delay_ms.
  //!        ignored if avoid_dup is false.
  //! \return shared_ptr to \p XListener.
  template <class tObj, class tClass>
  shared_ptr<XListener> connectShared(bool mainthreadcall, const shared_ptr<tObj> &obj,
    void (tClass::*func)(const tArg &),
    bool avoid_dup = false, unsigned int delay_ms = 0);
    
  void connect(const shared_ptr<XListener> &);
  void disconnect(const shared_ptr<XListener> &);

  //! Request a talk to connected listeners.
  //! If a listener is not mainthread model, the listener will be called later.
  //! \param arg passing argument to all listeners
  //! If listener avoids duplication, lock won't be passed to listener.
  void talk(const tArg &arg);

  bool empty() const {readBarrier(); return !m_listeners;}
 private:
  typedef _XListenerImpl<tArg> Listener;
  typedef std::deque<weak_ptr<Listener> > ListenerList;
  typedef typename ListenerList::iterator ListenerList_it;
  typedef typename ListenerList::const_iterator ListenerList_const_it;
  //! listener list is atomically substituted by new one. i.e. Read-Copy-Update.
  atomic_shared_ptr<ListenerList> m_listeners;
  void connect(const shared_ptr<Listener> &);
  
  struct Transaction : public _XTransaction
  {   
        Transaction(const shared_ptr<Listener> &l) :
             _XTransaction(), listener(l) {}
        const shared_ptr<Listener> listener;
        virtual bool talkBuffered() = 0;
  };
  struct TransactionAllowDup : public XTalker<tArg>::Transaction
  {   
        TransactionAllowDup(const shared_ptr<Listener> &l, const tArg &a) :
             XTalker<tArg>::Transaction(l), arg(a) {}
        const tArg arg;
        virtual bool talkBuffered() {
            (*XTalker<tArg>::Transaction::listener)(arg);
            return false;
        }
  };
  struct TransactionAvoidDup : public XTalker<tArg>::Transaction
  {   
        TransactionAvoidDup(const shared_ptr<Listener> &l) :
             XTalker<tArg>::Transaction(l) {}
        virtual bool talkBuffered() {
            bool skip = false;
            if(XTalker<tArg>::Transaction::listener->delay_ms()) {
                long elapsed_ms = (timeStamp() - 
                        XTalker<tArg>::Transaction::registered_time) / 1000uL;
                skip = ((long)XTalker<tArg>::Transaction::listener->delay_ms() > elapsed_ms);
            }
            if(!skip) {
                atomic_scoped_ptr<tArg> arg;
                arg.swap(XTalker<tArg>::Transaction::listener->arg);
                ASSERT(arg.get());
                (*XTalker<tArg>::Transaction::listener)(*arg);
            }
            return skip;
        }            
  };  
};

template <typename T, unsigned int SIZE> class atomic_pointer_queue;

//! Synchronize requests in talkers with main-thread
//! \sa XTalker, XListener
class XSignalBuffer
{
 public:
  XSignalBuffer();
  ~XSignalBuffer();
  //! Called by XTalker
  void registerTransactionList(_XTransaction *);
  //! be called by thread pool
  bool synchronize(); //return true if not busy
  
 private:
  typedef atomic_pointer_queue<_XTransaction, 10000> Queue;
  const scoped_ptr<Queue> m_queue;
  atomic<unsigned long> m_queue_oldest_timestamp;
};

extern shared_ptr<XSignalBuffer> g_signalBuffer;

// template definitions below.

template <class tArg>
XTalker<tArg>::XTalker()
 : m_listeners() {  
}
template <class tArg>
XTalker<tArg>::~XTalker() {
}
template <class tArg>
shared_ptr<XListener>
XTalker<tArg>::connectStatic(bool mainthreadcall, 
    void (*func)(const tArg &),
    bool avoid_dup, unsigned int delay_ms)
{
    shared_ptr<Listener> listener(
        new _XListenerStatic<tArg>(func, mainthreadcall, avoid_dup, delay_ms) );
    connect(listener);
    return listener;
}
template <class tArg>
template <class tObj, class tClass>
shared_ptr<XListener>
XTalker<tArg>::connectShared(bool mainthreadcall, const shared_ptr<tObj> &obj,
    void (tClass::*func)(const tArg &),
    bool avoid_dup, unsigned int delay_ms)
{
    shared_ptr<Listener> listener(
        new _XListenerShared<tClass, tArg>(
        dynamic_pointer_cast<tClass>(obj), func, mainthreadcall, avoid_dup, delay_ms) );
    connect(listener);
    return listener;
}
template <class tArg>
template <class tObj, class tClass>
shared_ptr<XListener>
XTalker<tArg>::connectWeak(bool mainthreadcall, const shared_ptr<tObj> &obj,
    void (tClass::*func)(const tArg &),
    bool avoid_dup, unsigned int delay_ms)
{
    shared_ptr<Listener> listener(
        new _XListenerWeak<tClass, tArg>(
        dynamic_pointer_cast<tClass>(obj), func, mainthreadcall, avoid_dup, delay_ms) );
    connect(listener);
    return listener;
}
template <class tArg>
void
XTalker<tArg>::connect(const shared_ptr<XListener> &lx)
{
    shared_ptr<Listener> listener = dynamic_pointer_cast<Listener>(lx);
    connect(listener);
}
template <class tArg>
void
XTalker<tArg>::connect(const shared_ptr<Listener> &lx)
{
    for(;;) {
        atomic_shared_ptr<ListenerList> old_list(m_listeners);
        atomic_shared_ptr<ListenerList> new_list(
            old_list ? (new ListenerList(*old_list)) : (new ListenerList));
        // clean-up dead listeners.
        for(ListenerList_it it = new_list->begin(); it != new_list->end();) {
            if(!it->lock())
                it = new_list->erase(it);
            else
                it++;
        }
        new_list->push_back(lx);
        if(new_list.compareAndSwap(old_list, m_listeners)) break;
    }
}
template <class tArg>
void
XTalker<tArg>::disconnect(const shared_ptr<XListener> &lx)
{
    for(;;) {
        atomic_shared_ptr<ListenerList> old_list(m_listeners);
        atomic_shared_ptr<ListenerList> new_list(
            old_list ? (new ListenerList(*old_list)) : (new ListenerList));
        for(ListenerList_it it = new_list->begin(); it != new_list->end();) {
            if(shared_ptr<Listener> listener = it->lock()) {
                // clean dead listeners and matching one.
                if(!listener || (lx == listener)) {
                    it = new_list->erase(it);
                    continue;
                }
            }
            it++;
        }
        if(new_list->empty()) new_list.reset();
        if(new_list.compareAndSwap(old_list, m_listeners)) break;
    }
}

template <class tArg>
void
XTalker<tArg>::talk(const tArg &arg)
{
  if(m_bMasked) return;  
  
  if(empty()) return;
  atomic_shared_ptr<ListenerList> list(m_listeners);
  if(!list) return;

  for(ListenerList_it it = list->begin(); it != list->end(); it++)
  {
    if(shared_ptr<Listener> listener = it->lock()) {
        if(!listener->m_bMasked) {
            if(isMainThread() || !listener->m_bMainThreadCall) {
                try {
                  (*listener)(arg);
                }
                catch (XKameError &e) {
                    e.print();
                }
                catch (std::bad_alloc &) {
                    gErrPrint("Memory Allocation Failed!");
                }
                catch (...) {
                    gErrPrint("Unhandled Exception Occurs!");
                }
            }
            else {
        		   if(listener->m_bAvoidDup) {
                    atomic_scoped_ptr<tArg> newarg(new tArg(arg) );
                    newarg.swap(listener->arg);
                    if(!newarg.get()) {
                         g_signalBuffer->registerTransactionList(
                            new TransactionAvoidDup(listener));
                    }
        		    }
                else {
                     g_signalBuffer->registerTransactionList(
                            new TransactionAllowDup(listener, arg));
          		}
            }
        }
     }
  }
}

#endif
