/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
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
class XCharDeviceDriver : public tDriver
{
	XNODE_OBJECT
protected:
	XCharDeviceDriver(const char *name, bool runtime, 
					  const shared_ptr<XScalarEntryList> &scalarentries,
					  const shared_ptr<XInterfaceList> &interfaces,
					  const shared_ptr<XThermometerList> &thermometers,
					  const shared_ptr<XDriverList> &drivers);
public:
	virtual ~XCharDeviceDriver() {}
protected:
	const shared_ptr<tInterface> &interface() const {return m_interface;}
	//! Be called just after opening interface. Call start() inside this routine appropriately.
	virtual void open() throw (XInterface::XInterfaceError &) {this->start();}
	//! Be called during stopping driver. Call interface()->stop() inside this routine.
	virtual void close() throw (XInterface::XInterfaceError &) {interface()->stop();}
	void onOpen(const shared_ptr<XInterface> &);
	void onClose(const shared_ptr<XInterface> &);
	//! This should not cause an exception.
	virtual void afterStop() {close();}
private:
	shared_ptr<XListener> m_lsnOnOpen, m_lsnOnClose;
  
	const shared_ptr<tInterface> m_interface;
};

template<class tDriver, class tInterface>
XCharDeviceDriver<tDriver, tInterface>::XCharDeviceDriver(const char *name, bool runtime, 
														  const shared_ptr<XScalarEntryList> &scalarentries,
														  const shared_ptr<XInterfaceList> &interfaces,
														  const shared_ptr<XThermometerList> &thermometers,
														  const shared_ptr<XDriverList> &drivers) :
    tDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_interface(XNode::create<tInterface>("Interface", false,
										  dynamic_pointer_cast<XDriver>(this->shared_from_this())))
{
    interfaces->insert(m_interface);
    m_lsnOnOpen = interface()->onOpen().connectWeak(
		this->shared_from_this(), &XCharDeviceDriver<tDriver, tInterface>::onOpen);
    m_lsnOnClose = interface()->onClose().connectWeak( 
    	this->shared_from_this(), &XCharDeviceDriver<tDriver, tInterface>::onClose);
}
template<class tDriver, class tInterface>
void
XCharDeviceDriver<tDriver, tInterface>::onOpen(const shared_ptr<XInterface> &)
{
	try {
		open();
	}
	catch (XInterface::XInterfaceError& e) {
		e.print(this->getLabel() + KAME::i18n(": Opening interface failed, because "));
		close();
	}
}
template<class tDriver, class tInterface>
void
XCharDeviceDriver<tDriver, tInterface>::onClose(const shared_ptr<XInterface> &)
{
	try {
		this->stop();
	}
	catch (XInterface::XInterfaceError& e) {
		e.print(this->getLabel() + KAME::i18n(": Stopping driver failed, because "));
	}
}
#endif /*CHARDEVICEDRIVER_H_*/
