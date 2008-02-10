/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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

class XDummyInterface : public XInterface
{
	XNODE_OBJECT
	XDummyInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
		: XInterface(name, runtime, driver), m_bOpened(false)
	{}
public:
	virtual ~XDummyInterface() {}

	virtual void open() throw (XInterfaceError &) {
		m_bOpened = true;
	}
	//! This can be called even if has already closed.
	virtual void close() throw (XInterfaceError &) {
		m_bOpened = false;
	}

	virtual bool isOpened() const {return m_bOpened;}
private:
	bool m_bOpened;
};
template<class tDriver>
class XDummyDriver : public tDriver
{
	XNODE_OBJECT
protected:
	XDummyDriver(const char *name, bool runtime, 
				 const shared_ptr<XScalarEntryList> &scalarentries,
				 const shared_ptr<XInterfaceList> &interfaces,
				 const shared_ptr<XThermometerList> &thermometers,
				 const shared_ptr<XDriverList> &drivers);
public:
	virtual ~XDummyDriver() {}
protected:
	virtual void afterStop() {interface()->stop();}
	const shared_ptr<XDummyInterface> &interface() const {return m_interface;}
private:
	shared_ptr<XListener> m_lsnOnOpen, m_lsnOnClose;
	void onOpen(const shared_ptr<XInterface> &);
	void onClose(const shared_ptr<XInterface> &);
	const shared_ptr<XDummyInterface> m_interface;
};

template<class tDriver>
XDummyDriver<tDriver>::XDummyDriver(const char *name, bool runtime, 
									const shared_ptr<XScalarEntryList> &scalarentries,
									const shared_ptr<XInterfaceList> &interfaces,
									const shared_ptr<XThermometerList> &thermometers,
									const shared_ptr<XDriverList> &drivers) :
    tDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_interface(XNode::create<XDummyInterface>("Interface", false,
											   dynamic_pointer_cast<XDriver>(this->shared_from_this())))
{
    interfaces->insert(m_interface);
    m_lsnOnOpen = interface()->onOpen().connectWeak(
    	this->shared_from_this(), &XDummyDriver<tDriver>::onOpen);
    m_lsnOnClose = interface()->onClose().connectWeak(
    	this->shared_from_this(), &XDummyDriver<tDriver>::onClose);
}
template<class tDriver>
void
XDummyDriver<tDriver>::onOpen(const shared_ptr<XInterface> &)
{
	try {
		this->start();
	}
	catch (XInterface::XInterfaceError& e) {
		e.print(this->getLabel() + KAME::i18n(": Starting driver failed, because "));
		interface()->stop();
	}
}
template<class tDriver>
void
XDummyDriver<tDriver>::onClose(const shared_ptr<XInterface> &)
{
	try {
		this->stop();
	}
	catch (XInterface::XInterfaceError& e) {
		e.print(this->getLabel() + KAME::i18n(": Stopping driver failed, because "));
	}
}

#endif /*DUMMYDRIVER_H_*/
