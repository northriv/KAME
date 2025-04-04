/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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

class DECLSPEC_KAME XPrimaryDriverWithThread : public XPrimaryDriver {
public:
    using XPrimaryDriver::XPrimaryDriver; //inherits constructors.
  
	//! Shuts down your threads, unconnects GUI, and deactivates signals.\n
	//! This function may be called even if driver has already stopped.
	//! This should not cause an exception.
    virtual void stop() override;

    class DECLSPEC_KAME Payload : public XPrimaryDriver::Payload {};
protected:
	//! Starts up your threads, connects GUI, and activates signals.
	//! This function should not cause an exception.
    virtual void start() override;

	virtual void *execute(const atomic<bool> &terminated) = 0;
private:
    unique_ptr<XThread> m_thread;
	void *execute_internal(const atomic<bool> &terminated) {
        Transactional::setCurrentPriorityMode(Priority::NORMAL);

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
    m_thread.reset(new XThread(shared_from_this(),
        &XPrimaryDriverWithThread::execute_internal));
}

inline void
XPrimaryDriverWithThread::stop() {
    unique_ptr<XThread> thread = std::move(m_thread);
    if(thread && !thread->isTerminated()) {
        thread->terminate();
        m_thread = std::move(thread);
    }
	else
		closeInterface(); //closes interface if any.
}

#endif /*PRIMARYDRIVERWITHTHREAD_H_*/
