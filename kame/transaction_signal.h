/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef TRANSACTION_SIGNAL_H
#define TRANSACTION_SIGNAL_H

#include "transaction.h"
#include <deque>
#include "xsignal.h"

namespace Transactional {

template <class XN, typename tArg>
struct Event {
	Event(const Snapshot<XN> &b, const Snapshot<XN> &s, tArg a) :
		before(b), shot(s), arg(a) {}
	Snapshot<XN> before;
	Snapshot<XN> shot;
	tArg arg;
};

template <class XN, class tClass, typename Arg>
struct _ListenerRef : public _XListenerImpl<Event<XN, Arg> > {
	_ListenerRef(tClass &obj,
		void (tClass::*func)(const Snapshot<XN> &shot, Arg),
				   XListener::FLAGS flags) :
		_XListenerImpl<Event<XN, Arg> >(flags), m_func(func), m_obj(obj) {
    }
	virtual void operator() (const Event<XN, Arg> &x) const {
		(m_obj.*m_func)(x.shot, x.arg);
	}
private:
	void (tClass::*const m_func)(const Snapshot<XN> &shot, Arg);
	tClass &m_obj;
};

template <class XN, class tClass, typename Arg>
struct _ListenerRefWBefore : public _XListenerImpl<Event<XN, Arg> > {
public:
	_ListenerRefWBefore(tClass &obj,
		void (tClass::*func)(const Snapshot<XN> &before, const Snapshot<XN> &shot, Arg),
				   XListener::FLAGS flags) :
		_XListenerImpl<Event<XN, Arg> >(flags), m_func(func), m_obj(obj) {
    }
	virtual void operator() (const Event<XN, Arg> &x) const {
		(m_obj.*m_func)(x.before, x.shot, x.arg);
	}
private:
	void (tClass::*const m_func)(const Snapshot<XN> &before,
		const Snapshot<XN> &shot, Arg);
	tClass &m_obj;
};

template <class XN>
class _TalkerBase : public _XTalkerBase {
public:
	_TalkerBase() : _XTalkerBase() {}
	virtual ~_TalkerBase() {}
	virtual void talk(const Snapshot<XN> &before, const Snapshot<XN> &shot) = 0;
};

//! M/M Listener and Talker model
//! \sa XListener, XSignalStore
//! \p tArg: value which will be derivered
//! \p tArgWrapper: copied argument, will be released by GC someday
template <class XN, typename tArg>
class Talker : public _TalkerBase<XN> {
public:
	Talker() : _TalkerBase<XN>() {}
	Talker(const Talker &x) : _TalkerBase<XN>(x), m_listeners(x.m_listeners), m_context() {}
	virtual ~Talker() {}

	template <class tObj, class tClass>
	shared_ptr<XListener> connect(tObj &obj, void (tClass::*func)(
		const Snapshot<XN> &shot, tArg), int flags = 0);
	template <class tObj, class tClass>
	shared_ptr<XListener> connect(tObj &obj, void (tClass::*func)(
		const Snapshot<XN> &before, const Snapshot<XN> &shot, tArg), int flags = 0);

	void connect(const shared_ptr<XListener> &);
	void disconnect(const shared_ptr<XListener> &);

	//! Request a talk to connected listeners.
	//! If a listener is not mainthread model, the listener will be called later.
	//! \param arg passing argument to all listeners
	//! If listener avoids duplication, lock won't be passed to listener.
	void mark(tArg arg) { m_context.reset(new tArg(arg));}
	virtual void talk(const Snapshot<XN> &before, const Snapshot<XN> &shot);

	bool empty() const {readBarrier(); return !m_listeners;}
private:
	typedef Event<XN, tArg> _Event;
	typedef _XListenerImpl<Event<XN, tArg> > _Listener;
	typedef std::deque<weak_ptr<_Listener> > ListenerList;
	typedef typename ListenerList::iterator ListenerList_it;
	typedef typename ListenerList::const_iterator ListenerList_const_it;
	shared_ptr<ListenerList> m_listeners;
	scoped_ptr<tArg> m_context;

	void connect(const shared_ptr<_Listener> &);

	struct EventWrapper : public _XTransaction {
		EventWrapper(const shared_ptr<_Listener> &l) :
			_XTransaction(), listener(l) {}
		const shared_ptr<_Listener> listener;
		virtual bool talkBuffered() = 0;
	};
	struct EventWrapperAllowDup : public EventWrapper {
		EventWrapperAllowDup(const shared_ptr<_Listener> &l, const _Event &e) :
			EventWrapper(l), event(e) {}
		const _Event event;
		virtual bool talkBuffered() {
			(*this->listener)(event);
			return false;
		}
	};
	struct EventWrapperAvoidDup : public EventWrapper {
		EventWrapperAvoidDup(const shared_ptr<_Listener> &l) : EventWrapper(l) {}
			virtual bool talkBuffered() {
				bool skip = false;
				if(this->listener->delay_ms()) {
					long elapsed_ms = (timeStamp() - this->registered_time) / 1000uL;
					skip = ((long)this->listener->delay_ms() > elapsed_ms);
				}
				if(!skip) {
					atomic_scoped_ptr<_Event> e;
					e.swap(this->listener->arg);
					ASSERT(e.get());
					(*this->listener)(*e);
				}
				return skip;
			}
	};
};

template <class XN, typename tArg>
template <class tObj, class tClass>
shared_ptr<XListener>
Talker<XN, tArg>::connect(tObj &obj, void (tClass::*func)(
	const Snapshot<XN> &shot, tArg), int flags) {
	shared_ptr<_Listener> listener(
		new _ListenerRef<XN, tClass, tArg>(
			static_cast<tClass&>(obj), func, (XListener::FLAGS)flags) );
	connect(listener);
	return listener;
}

template <class XN, typename tArg>
template <class tObj, class tClass>
shared_ptr<XListener>
Talker<XN, tArg>::connect(tObj &obj, void (tClass::*func)(
	const Snapshot<XN> &before, const Snapshot<XN> &shot, tArg), int flags) {
	shared_ptr<_Listener> listener(
		new _ListenerRefWBefore<XN, tClass, tArg>(
			static_cast<tClass&>(obj), func, (XListener::FLAGS)flags) );
	connect(listener);
	return listener;
}

template <class XN, typename tArg>
void
Talker<XN, tArg>::connect(const shared_ptr<XListener> &lx) {
	shared_ptr<_Listener> listener = dynamic_pointer_cast<_Listener>(lx);
	connect(listener);
}
template <class XN, typename tArg>
void
Talker<XN, tArg>::connect(const shared_ptr<_Listener> &lx) {
	shared_ptr<ListenerList> new_list(
		m_listeners ? (new ListenerList(*m_listeners)) : (new ListenerList));
	// clean-up dead listeners.
	for(ListenerList_it it = new_list->begin(); it != new_list->end();) {
		if(!it->lock())
			it = new_list->erase(it);
		else
			it++;
	}
	new_list->push_back(lx);
	m_listeners = new_list;
}
template <class XN, typename tArg>
void
Talker<XN, tArg>::disconnect(const shared_ptr<XListener> &lx) {
	shared_ptr<ListenerList> new_list(
		m_listeners ? (new ListenerList(*m_listeners)) : (new ListenerList));
	for(ListenerList_it it = new_list->begin(); it != new_list->end();) {
		if(shared_ptr<XListener> listener = it->lock()) {
			// clean dead listeners and matching one.
			if(!listener || (lx == listener)) {
				it = new_list->erase(it);
				continue;
			}
		}
		it++;
	}
	if(new_list->empty()) new_list.reset();
	m_listeners = new_list;
}

template <class XN, typename tArg>
void
Talker<XN, tArg>::talk(const Snapshot<XN> &before, const Snapshot<XN> &shot) {
	if(this->m_bMasked) return;
	if(empty()) return;
	shared_ptr<ListenerList> &list(m_listeners);
	if(!list) return;
	ASSERT(m_context);
	_Event event(before, shot, *m_context);
	for(ListenerList_it it = list->begin(); it != list->end(); it++) {
		if(shared_ptr<_Listener> listener = it->lock()) {
			if((listener->m_flags & XListener::FLAG_MASKED) == 0) {
				if(isMainThread() || ((listener->m_flags & XListener::FLAG_MAIN_THREAD_CALL) == 0)) {
					try {
						(*listener)(event);
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
						atomic_scoped_ptr<_Event> newevent(new _Event(event) );
						newevent.swap(listener->arg);
						if( !newevent.get()) {
							registerTransactionList(new EventWrapperAvoidDup(listener));
						}
					}
					else {
						registerTransactionList(new EventWrapperAllowDup(listener, event));
					}
				}
			}
		}
	}
}


} //namespace Transactional

#endif /*TRANSACTION_SIGNAL_H*/
