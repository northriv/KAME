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

template <class XN, typename tArg, typename tArgRef = const tArg &>
struct Event {
	Event(const Snapshot<XN> &s, tArgRef a) :
		shot(s), arg(a) {}
	Snapshot<XN> shot;
	tArg arg;
};

template <class XN, class tClass, typename tArg, typename tArgRef = const tArg &>
struct _ListenerRef : public _XListenerImpl<Event<XN, tArg, tArgRef> > {
	_ListenerRef(tClass &obj,
		void (tClass::*func)(const Snapshot<XN> &shot, tArgRef),
		XListener::FLAGS flags) :
		_XListenerImpl<Event<XN, tArg, tArgRef> >(flags), m_func(func), m_obj(obj) { }
	virtual void operator() (const Event<XN, tArg, tArgRef> &x) const {
		(m_obj.*m_func)(x.shot, x.arg);
	}
private:
	void (tClass::*const m_func)(const Snapshot<XN> &shot, tArgRef);
	tClass &m_obj;
};
template <class XN, class tClass, typename tArg, typename tArgRef = const tArg &>
struct _ListenerWeak : public _XListenerImpl<Event<XN, tArg, tArgRef> > {
	_ListenerWeak(const shared_ptr<tClass> &obj,
		void (tClass::*func)(const Snapshot<XN> &shot, tArgRef),
		 XListener::FLAGS flags) :
		 _XListenerImpl<Event<XN, tArg, tArgRef> >(flags), m_func(func), m_obj(obj) { }
	virtual void operator() (const Event<XN, tArg, tArgRef> &x) const {
		if(shared_ptr<tClass> p = m_obj.lock() )
			(p.get()->*m_func)(x.shot, x.arg);
	}
private:
	void (tClass::*const m_func)(const Snapshot<XN> &shot, tArgRef);
	const weak_ptr<tClass> m_obj;
};

template <class XN>
struct _Message {
	virtual ~_Message() {}
	virtual void talk(const Snapshot<XN> &shot) = 0;
	virtual void unmark(const shared_ptr<XListener> &x) = 0;
};

//! M/M Listener and Talker model
//! \sa XListener, XSignalStore
//! \p tArg: value which will be derivered
//! \p tArgWrapper: copied argument, will be released by GC someday
//!\todo abandon tArgRef, use const tArg&.
template <class XN, typename tArg, typename tArgRef = const tArg &>
class Talker {
public:
	Talker() {}
	Talker(const Talker &x) : m_listeners(x.m_listeners) {}
	virtual ~Talker() {}

	template <class tObj, class tClass>
	shared_ptr<XListener> connect(tObj &obj, void (tClass::*func)(
		const Snapshot<XN> &shot, tArgRef), int flags = 0);
	template <class tObj, class tClass>
	shared_ptr<XListener> connectWeakly(const shared_ptr<tObj> &obj, void (tClass::*func)(
		const Snapshot<XN> &shot, tArgRef), int flags = 0);

	void connect(const shared_ptr<XListener> &);
	void disconnect(const shared_ptr<XListener> &);

	//! Requests a talk to connected listeners.
	//! If a listener is not mainthread model, the listener will be called later.
	//! \param arg passing argument to all listeners
	//! If listener avoids duplication, lock won't be passed to listener.
	virtual _Message<XN>* createMessage(tArgRef arg);
	void talk(const Snapshot<XN> &shot, tArgRef arg) const {
		_talk(shot, m_listeners, shared_ptr<UnmarkedListenerList>(), arg);
	}

	bool empty() const {return !m_listeners;}
private:
	typedef Event<XN, tArg, tArgRef> _Event;
	typedef _XListenerImpl<Event<XN, tArg, tArgRef> > _Listener;
	typedef std::deque<weak_ptr<_Listener> > ListenerList;
	typedef std::deque<shared_ptr<XListener> > UnmarkedListenerList;
	shared_ptr<ListenerList> m_listeners;

	void connect(const shared_ptr<_Listener> &);

	struct EventWrapper : public _XTransaction {
		EventWrapper(const shared_ptr<_Listener> &l) :
			_XTransaction(), listener(l) {}
		virtual ~EventWrapper() {}
		const shared_ptr<_Listener> listener;
		virtual bool talkBuffered() = 0;
	};
	struct EventWrapperAllowDup : public EventWrapper {
		EventWrapperAllowDup(const shared_ptr<_Listener> &l, const _Event &e) :
			EventWrapper(l), event(e) {}
		const _Event event;
		virtual bool talkBuffered() {
			( *this->listener)(event);
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
				if( !skip) {
					atomic_scoped_ptr<_Event> e;
					e.swap(this->listener->arg);
					ASSERT(e.get());
					( *this->listener)( *e);
				}
				return skip;
			}
	};
	inline static void _talk(const Snapshot<XN> &shot,
		const shared_ptr<ListenerList> &list,
		const shared_ptr<UnmarkedListenerList> &unmarked, tArgRef arg);
protected:
	struct Message : public _Message<XN> {
		Message(tArgRef a, const shared_ptr<ListenerList> &l) : _Message<XN>(), arg(a), listeners(l) {}
		tArg arg;
		shared_ptr<ListenerList> listeners;
		shared_ptr<UnmarkedListenerList> listeners_unmarked;
		virtual void talk(const Snapshot<XN> &shot) {
			_talk(shot, listeners, listeners_unmarked, arg);
		}
		virtual void unmark(const shared_ptr<XListener> &x) {
			if( !listeners)
				return;
			for(typename ListenerList::const_iterator it = listeners->begin(); it != listeners->end(); it++) {
				if(shared_ptr<_Listener> listener = it->lock()) {
					if(listener == x) {
						if( !listeners_unmarked)
							listeners_unmarked.reset(new UnmarkedListenerList);
						listeners_unmarked->push_back(x);
					}
				}
			}
		}
	};
};

template <class XN, typename tArg, typename tArgRef = const tArg &>
class TalkerSingleton : public Talker<XN, tArg, tArgRef> {
public:
	TalkerSingleton() : Talker<XN, tArg, tArgRef>(), m_marked(0) {}
	TalkerSingleton(const TalkerSingleton &x) : Talker<XN, tArg, tArgRef>(x), m_marked(0) {}
	virtual _Message<XN>* createMessage(tArgRef arg) {
		if(m_marked) {
			static_cast<typename Talker<XN, tArg, tArgRef>::Message *>(m_marked)->arg = arg;
			return 0;
		}
		m_marked = Talker<XN, tArg, tArgRef>::createMessage(arg);
		return m_marked;
	}
private:
	_Message<XN> *m_marked;
};

template <class XN, typename tArg, typename tArgRef>
_Message<XN>*
Talker<XN, tArg, tArgRef>::createMessage(tArgRef arg) {
	if( !m_listeners)
		return 0;
	return new Message(arg, m_listeners);
}

template <class XN, typename tArg, typename tArgRef>
template <class tObj, class tClass>
shared_ptr<XListener>
Talker<XN, tArg, tArgRef>::connect(tObj &obj, void (tClass::*func)(
	const Snapshot<XN> &shot, tArgRef), int flags) {
	shared_ptr<_Listener> listener(
		new _ListenerRef<XN, tClass, tArg, tArgRef>(
			static_cast<tClass&>(obj), func, (XListener::FLAGS)flags) );
	connect(listener);
	return listener;
}

template <class XN, typename tArg, typename tArgRef>
template <class tObj, class tClass>
shared_ptr<XListener>
Talker<XN, tArg, tArgRef>::connectWeakly(const shared_ptr<tObj> &obj,
	void (tClass::*func)(const Snapshot<XN> &shot, tArgRef), int flags) {
	shared_ptr<_Listener> listener(
		new _ListenerWeak<XN, tClass, tArg, tArgRef>(
			static_pointer_cast<tClass>(obj), func, (XListener::FLAGS)flags) );
	connect(listener);
	return listener;
}
template <class XN, typename tArg, typename tArgRef>
void
Talker<XN, tArg, tArgRef>::connect(const shared_ptr<XListener> &lx) {
	shared_ptr<_Listener> listener = dynamic_pointer_cast<_Listener>(lx);
	connect(listener);
}
template <class XN, typename tArg, typename tArgRef>
void
Talker<XN, tArg, tArgRef>::connect(const shared_ptr<_Listener> &lx) {
	shared_ptr<ListenerList> new_list(
		m_listeners ? (new ListenerList( *m_listeners)) : (new ListenerList));
	// clean-up dead listeners.
	for(typename ListenerList::iterator it = new_list->begin(); it != new_list->end();) {
		if( !it->lock())
			it = new_list->erase(it);
		else
			++it;
	}
	new_list->push_back(lx);
	m_listeners = new_list;
}
template <class XN, typename tArg, typename tArgRef>
void
Talker<XN, tArg, tArgRef>::disconnect(const shared_ptr<XListener> &lx) {
	shared_ptr<ListenerList> new_list(
		m_listeners ? (new ListenerList( *m_listeners)) : (new ListenerList));
	for(typename ListenerList::iterator it = new_list->begin(); it != new_list->end();) {
		if(shared_ptr<XListener> listener = it->lock()) {
			// clean dead listeners and matching one.
			if( !listener || (lx == listener)) {
				it = new_list->erase(it);
				continue;
			}
		}
		++it;
	}
	if(new_list->empty()) new_list.reset();
	m_listeners = new_list;
}

template <class XN, typename tArg, typename tArgRef>
void
Talker<XN, tArg, tArgRef>::_talk(const Snapshot<XN> &shot,
	const shared_ptr<ListenerList> &listeners,
	const shared_ptr<UnmarkedListenerList> &unmarked, tArgRef arg) {
	if( !listeners) return;
	_Event event(shot, arg);
	for(typename ListenerList::const_iterator it = listeners->begin(); it != listeners->end(); it++) {
		if(shared_ptr<_Listener> listener = it->lock()) {
			if(unmarked &&
				(std::find(unmarked->begin(), unmarked->end(), listener) != unmarked->end()))
				continue;
			if(isMainThread() || ((listener->m_flags & XListener::FLAG_MAIN_THREAD_CALL) == 0)) {
				try {
					( *listener)(event);
				}
				catch (XKameError &e) {
					e.print();
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

} //namespace Transactional

#endif /*TRANSACTION_SIGNAL_H*/
