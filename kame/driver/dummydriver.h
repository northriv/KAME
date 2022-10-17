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
#ifndef DUMMYDRIVER_H_
#define DUMMYDRIVER_H_

#include "driver.h"
#include "interface.h"

class XDummyInterface : public XInterface {
public:
	XDummyInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
		: XInterface(name, runtime, driver), m_bOpened(false)
	{}
	virtual ~XDummyInterface() {}

    virtual void open() {
		m_bOpened = true;
	}
	//! This can be called even if has already closed.
    virtual void close() {
		m_bOpened = false;
	}

	virtual bool isOpened() const {return m_bOpened;}
private:
	bool m_bOpened;
};
template<class tDriver>
class XDummyDriver : public tDriver {
public:
	XDummyDriver(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XDummyDriver() {}
protected:
	virtual void closeInterface() {interface()->stop();}
	const shared_ptr<XDummyInterface> &interface() const {return m_interface;}
private:
    shared_ptr<Listener> m_lsnOnOpen, m_lsnOnClose;
	void onOpen(const Snapshot &shot, XInterface *);
	void onClose(const Snapshot &shot, XInterface *);
	const shared_ptr<XDummyInterface> m_interface;
};

template<class tDriver>
XDummyDriver<tDriver>::XDummyDriver(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    tDriver(name, runtime, ref(tr_meas), meas),
	m_interface(XNode::create<XDummyInterface>("Interface", false,
											   dynamic_pointer_cast<XDriver>(this->shared_from_this()))) {
    meas->interfaces()->insert(tr_meas, m_interface);
    this->iterate_commit([=](Transaction &tr){
	    m_lsnOnOpen = tr[ *interface()].onOpen().connectWeakly(
	    	this->shared_from_this(), &XDummyDriver<tDriver>::onOpen);
	    m_lsnOnClose = tr[ *interface()].onClose().connectWeakly(
	    	this->shared_from_this(), &XDummyDriver<tDriver>::onClose);
    });
}
template<class tDriver>
void
XDummyDriver<tDriver>::onOpen(const Snapshot &shot, XInterface *) {
	try {
		this->start();
	}
	catch (XKameError& e) {
		e.print(this->getLabel() + i18n(": Starting driver failed, because "));
		interface()->stop();
	}
}
template<class tDriver>
void
XDummyDriver<tDriver>::onClose(const Snapshot &shot, XInterface *) {
	try {
		this->stop();
	}
	catch (XKameError& e) {
		e.print(this->getLabel() + i18n(": Stopping driver failed, because "));
	}
}

#endif /*DUMMYDRIVER_H_*/
