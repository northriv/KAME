/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef PRIMARYDRIVERWITHTHREAD_H_
#define PRIMARYDRIVERWITHTHREAD_H_

#include "primarydriver.h"

class XPrimaryDriverWithThread : public XPrimaryDriver {
public:
	XPrimaryDriverWithThread(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
		XPrimaryDriver(name, runtime, ref(tr_meas), meas) {}
	virtual ~XPrimaryDriverWithThread() {}
  
	//! Shuts down your threads, unconnects GUI, and deactivates signals.\n
	//! This function may be called even if driver has already stopped.
	//! This should not cause an exception.
	virtual void stop();
private:
protected:
	//! Starts up your threads, connects GUI, and activates signals.
	//! This function should not cause an exception.
	virtual void start();
	//! This function should not cause an exception.
	virtual void closeInterface() = 0;

	virtual void *execute(const atomic<bool> &terminated) = 0;
private:
	shared_ptr<XThread<XPrimaryDriverWithThread> > m_thread;
	void *execute_internal(const atomic<bool> &terminated) {
		void *ret = NULL;
		try {
			ret = execute(terminated);
		}
		catch(XKameError &e) {
			e.print(getLabel() + i18n(" Error: "));
		}
		closeInterface(); //closes interface if any.
		return ret;
	}
};

inline void
XPrimaryDriverWithThread::start() {
	m_thread.reset(new XThread<XPrimaryDriverWithThread>(shared_from_this(),
		&XPrimaryDriverWithThread::execute_internal));
	m_thread->resume();
}

inline void
XPrimaryDriverWithThread::stop() {
	if(m_thread)
		m_thread->terminate();
	else
		closeInterface(); //closes interface if any.
}

#endif /*PRIMARYDRIVERWITHTHREAD_H_*/
