/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef CHARDEVICEDRIVER_H_
#define CHARDEVICEDRIVER_H_

#include "driver.h"
#include "interface.h"
class XCharInterface;

template<class tDriver, class tInterface = XCharInterface>
class XCharDeviceDriver : public tDriver {
public:
    template <typename... Args>
    XCharDeviceDriver(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas, Args&&... args);
    virtual ~XCharDeviceDriver() = default;

    struct DECLSPEC_KAME Payload : public tDriver::Payload {};
protected:
	const shared_ptr<tInterface> &interface() const {return m_interface;}
	//! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() {this->start();}
    //! Be called during stopping driver. Call interface()->stop() inside this routine.
    virtual void close() {interface()->stop();}
	void onOpen(const Snapshot &shot, XInterface *);
	void onClose(const Snapshot &shot, XInterface *);
	//! This should not cause an exception.
	//! This function should be called before leaving a measurement thread to terminate the interface.
	virtual void closeInterface() {close();}
private:
	shared_ptr<Listener> m_lsnOnOpen, m_lsnOnClose;
  
	const shared_ptr<tInterface> m_interface;
};

template<class tDriver, class tInterface>
template <typename... Args>
XCharDeviceDriver<tDriver, tInterface>::XCharDeviceDriver(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas, Args&&... args) :
    tDriver(name, runtime, ref(tr_meas), meas, std::forward<Args>(args)...),
	m_interface(XNode::create<tInterface>("Interface", false,
										  dynamic_pointer_cast<XDriver>(this->shared_from_this()))) {

    meas->interfaces()->insert(tr_meas, m_interface);
    this->iterate_commit([=](Transaction &tr){
	    m_lsnOnOpen = tr[ *interface()].onOpen().connectWeakly(
			this->shared_from_this(), &XCharDeviceDriver<tDriver, tInterface>::onOpen);
	    m_lsnOnClose = tr[ *interface()].onClose().connectWeakly(
	    	this->shared_from_this(), &XCharDeviceDriver<tDriver, tInterface>::onClose);
    });
}
template<class tDriver, class tInterface>
void
XCharDeviceDriver<tDriver, tInterface>::onOpen(const Snapshot &shot, XInterface *) {
	try {
		open();
	}
	catch (XKameError& e) {
		e.print(this->getLabel() + i18n(": Opening driver failed, because "));
		onClose(shot, NULL);
	}
}
template<class tDriver, class tInterface>
void
XCharDeviceDriver<tDriver, tInterface>::onClose(const Snapshot &, XInterface *) {
	try {
		this->stop();
	}
	catch (XKameError& e) {
		e.print(this->getLabel() + i18n(": Stopping driver failed, because "));
		close();
	}
}
#endif /*CHARDEVICEDRIVER_H_*/
