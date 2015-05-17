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
#ifndef signalH
#define signalH

#include "support.h"
#include "xtime.h"
#include "xthread.h"
#include "atomic_smart_ptr.h"
#include <deque>

//! Detect whether the current thread is the main thread.
DECLSPEC_KAME bool isMainThread();

template <class tArg>
class XTalker;
namespace Transactional {
template <class XN, typename tArg, typename tArgRef>
class Talker;}

//! Base class of listener, which holds pointers to object and function.
//! Hold instances by shared_ptr.
class DECLSPEC_KAME XListener {
public:
	virtual ~XListener();
	//! \return an appropriate delay for delayed transactions.
	unsigned int delay_ms() const;

	enum FLAGS {
		FLAG_MAIN_THREAD_CALL = 0x01, FLAG_AVOID_DUP = 0x02,
		FLAG_DELAY_SHORT = 0x100, FLAG_DELAY_ADAPTIVE = 0x200
	};

	FLAGS flags() const {return m_flags;}
protected:
	template <class tArg>
	friend class XTalker;
	template <class XN, typename tArg, typename tArgRef>
	friend class Transactional::Talker;
	XListener(FLAGS flags);
	atomic<FLAGS> m_flags;
};

#include "xsignal_prv.h"

class DECLSPEC_KAME XTalkerBase_ {
protected:
	XTalkerBase_() {}
public:
	virtual ~XTalkerBase_() {}
protected:
};

struct XTransaction_ {
	XTransaction_() : registered_time(timeStamp()) {}
	virtual ~XTransaction_() {}
	const unsigned long registered_time;
	virtual bool talkBuffered() = 0;
};

DECLSPEC_KAME void registerTransactionList(XTransaction_ *);

//! M/M Listener and Talker model
//! \sa XListener, XSignalStore
//! \p tArg: value which will be derivered
//! \p tArgWrapper: copied argument, will be released by GC someday
template <class tArg>
class XTalker : public XTalkerBase_ {
public:
	XTalker() {}
	virtual ~XTalker() {}

	//! Associate XTalker to XListener
	//! Talker will call user member function of \a listener.
	//! This function can be called over once.
	//! \sa disconnect()
	//! XListener holds a pointer of a static function
	//! \param func a pointer to static function
	//! \param flags \sa XListener::FLAGS
	//! \return shared_ptr to \p XListener.
	shared_ptr<XListener> connectStatic(void (*func)(const tArg &), int flags = 0);
	//! Associate XTalker to XListener
	//! Talker will call user member function of \a listener.
	//! This function can be called over once.
	//! \sa disconnect()
	//! XListener holds weak_ptr() of the instance
	//! \param obj a pointer to object
	//! \param func a pointer to member function
	//! \param flags \sa XListener::FLAGS
	//! \return shared_ptr to \p XListener.
	template <class tObj, class tClass>
	shared_ptr<XListener> connectWeak(const shared_ptr<tObj> &obj,
		void (tClass::*func)(const tArg &), int flags = 0);
	//! Associate XTalker to XListener
	//! Talker will call user member function of \a listener.
	//! This function can be called over once.
	//! \sa disconnect()
	//! XListener holds shared_ptr() of the instance
	//! \param obj a pointer to object
	//! \param func a pointer to member function
	//! \param flags \sa XListener::FLAGS
	//! \return shared_ptr to \p XListener.
	template <class tObj, class tClass>
	shared_ptr<XListener> connectShared(const shared_ptr<tObj> &obj,
		void (tClass::*func)(const tArg &), int flags = 0);

	void connect(const shared_ptr<XListener> &);
	void disconnect(const shared_ptr<XListener> &);

	//! Request a talk to connected listeners.
	//! If a listener is not mainthread model, the listener will be called later.
	//! \param arg passing argument to all listeners
	//! If listener avoids duplication, lock won't be passed to listener.
	void talk(const tArg &arg);

	bool empty() const {readBarrier(); return !m_listeners;}
private:
	typedef XListenerImpl_<tArg> Listener;
	typedef std::deque<weak_ptr<Listener> > ListenerList;
	typedef typename ListenerList::iterator ListenerList_it;
	typedef typename ListenerList::const_iterator ListenerList_const_it;
	//! listener list is atomically substituted by new one. i.e. Read-Copy-Update.
	atomic_shared_ptr<ListenerList> m_listeners;
	void connect(const shared_ptr<Listener> &);

	struct Transaction : public XTransaction_ {
		Transaction(const shared_ptr<Listener> &l) :
			XTransaction_(), listener(l) {}
		const shared_ptr<Listener> listener;
		virtual bool talkBuffered() = 0;
	};
	struct TransactionAllowDup : public XTalker<tArg>::Transaction {   
		TransactionAllowDup(const shared_ptr<Listener> &l, const tArg &a) :
			XTalker<tArg>::Transaction(l), arg(a) {}
			const tArg arg;
			virtual bool talkBuffered() {
				(*XTalker<tArg>::Transaction::listener)(arg);
				return false;
			}
	};
	struct TransactionAvoidDup : public XTalker<tArg>::Transaction {   
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
					atomic_unique_ptr<tArg> arg;
					arg.swap(XTalker<tArg>::Transaction::listener->arg);
					assert(arg.get());
					( *XTalker<tArg>::Transaction::listener)( *arg);
				}
				return skip;
			}            
	};  
};

// template definitions below.
template <class tArg>
shared_ptr<XListener>
XTalker<tArg>::connectStatic(
	void (*func)(const tArg &),
	int flags) {
	shared_ptr<Listener> listener(
		new XListenerStatic_<tArg>(func, (XListener::FLAGS)flags) );
	connect(listener);
	return listener;
}
template <class tArg>
template <class tObj, class tClass>
shared_ptr<XListener>
XTalker<tArg>::connectShared(const shared_ptr<tObj> &obj,
	void (tClass::*func)(const tArg &),
	int flags) {
	shared_ptr<Listener> listener(
		new XListenerShared_<tClass, tArg>(
			dynamic_pointer_cast<tClass>(obj), func, (XListener::FLAGS)flags) );
	connect(listener);
	return listener;
}
template <class tArg>
template <class tObj, class tClass>
shared_ptr<XListener>
XTalker<tArg>::connectWeak(const shared_ptr<tObj> &obj,
	void (tClass::*func)(const tArg &),
	int flags) {
	shared_ptr<Listener> listener(
		new XListenerWeak_<tClass, tArg>(
			dynamic_pointer_cast<tClass>(obj), func, (XListener::FLAGS)flags) );
	connect(listener);
	return listener;
}
template <class tArg>
void
XTalker<tArg>::connect(const shared_ptr<XListener> &lx) {
	shared_ptr<Listener> listener = dynamic_pointer_cast<Listener>(lx);
	connect(listener);
}
template <class tArg>
void
XTalker<tArg>::connect(const shared_ptr<Listener> &lx) {
	for(local_shared_ptr<ListenerList> old_list(m_listeners);;) {
		local_shared_ptr<ListenerList> new_list(
			old_list ? (new ListenerList( *old_list)) : (new ListenerList));
		// clean-up dead listeners.
		for(ListenerList_it it = new_list->begin(); it != new_list->end();) {
			if(!it->lock())
				it = new_list->erase(it);
			else
				it++;
		}
		new_list->push_back(lx);
		if(m_listeners.compareAndSwap(old_list, new_list)) break;
	}
}
template <class tArg>
void
XTalker<tArg>::disconnect(const shared_ptr<XListener> &lx) {
	for(local_shared_ptr<ListenerList> old_list(m_listeners);;) {
		local_shared_ptr<ListenerList> new_list(
			old_list ? (new ListenerList( *old_list)) : (new ListenerList));
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
		if(m_listeners.compareAndSwap(old_list, new_list)) break;
	}
}

template <class tArg>
void
XTalker<tArg>::talk(const tArg &arg) {
	if(empty()) return;
	local_shared_ptr<ListenerList> list(m_listeners);
	if( !list) return;
	for(ListenerList_it it = list->begin(); it != list->end(); it++) {
		if(shared_ptr<Listener> listener = it->lock()) {
			if(isMainThread() || ((listener->m_flags & XListener::FLAG_MAIN_THREAD_CALL) == 0)) {
				try {
					( *listener)(arg);
				}
				catch (XKameError &e) {
					e.print();
				}
				catch (std::bad_alloc &) {
					gErrPrint("Memory Allocation Failed!");
				}
			}
			else {
				if(listener->m_flags & XListener::FLAG_AVOID_DUP) {
					atomic_unique_ptr<tArg> newarg(new tArg(arg) );
					newarg.swap(listener->arg);
					if( !newarg.get()) {
						registerTransactionList(new TransactionAvoidDup(listener));
					}
				}
				else {
					registerTransactionList(new TransactionAllowDup(listener, arg));
				}
			}
		}
	}
}

#endif
