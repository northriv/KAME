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
//---------------------------------------------------------------------------
#include "measure.h"
#include "interface.h"
#include "xnodeconnector.h"
#include <string>
#include "driver.h"

//---------------------------------------------------------------------------

XInterface::XInterfaceError::XInterfaceError(const XString &msg, const char *file, int line)
	: XKameError(msg, file, line) {}
XInterface::XConvError::XConvError(const char *file, int line)
	: XInterfaceError(i18n("Conversion Error"), file, line) {}
XInterface::XCommError::XCommError(const XString &msg, const char *file, int line)
	:  XInterfaceError(i18n("Communication Error") + ", " + msg, file, line) {}
XInterface::XOpenInterfaceError::XOpenInterfaceError(const char *file, int line)
	:  XInterfaceError(i18n("Open Interface Error"), file, line) {}


XInterface::XInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) : 
    XNode(name, runtime), 
    m_driver(driver),
    m_device(create<XComboNode>("Device", false, true)),
    m_port(create<XStringNode>("Port", false)),
    m_address(create<XUIntNode>("Address", false)),
    m_control(create<XBoolNode>("Control", true)) {

	iterate_commit([=](Transaction &tr){
		lsnOnControlChanged = tr[ *control()].onValueChanged().connectWeakly(
	        shared_from_this(), &XInterface::onControlChanged);
    });
}

XString
XInterface::getLabel() const {
	if(m_label.empty())
		return driver()->getLabel();
	else
		return driver()->getLabel() + ":" + m_label;
}

void
XInterface::onControlChanged(const Snapshot &shot, XValueNodeBase *) {
	if(shot[ *control()]) {
	    g_statusPrinter->printMessage(driver()->getLabel() + i18n(": Starting..."));
		start();
	}
	else {
        control()->setUIEnabled(false);
        Snapshot shot( *this);
		shot.talk(shot[ *this].onClose(), this);
	}
}

void
XInterface::start() {
	XScopedLock<XInterface> lock( *this);
	try {
		if(isOpened()) {
			gErrPrint(getLabel() + i18n("Port has already opened"));
			return;
		}
		open();
	}
	catch (XInterfaceError &e) {
        e.print(getLabel() + i18n(": Opening interface failed, because "));
    	iterate_commit([=](Transaction &tr){
    		tr[ *control()] = false;
    		tr.unmark(lsnOnControlChanged);
        });
		return;
	}

    Transaction tr = iterate_commit([=](Transaction &tr){
		tr[ *device()].setUIEnabled(false);
		tr[ *port()].setUIEnabled(false);
		tr[ *address()].setUIEnabled(false);

		tr[ *control()] = true;
		tr.unmark(lsnOnControlChanged);
    });
    tr.talk(tr[ *this].onOpen(), this);
}
void
XInterface::stop() {
	XScopedLock<XInterface> lock( *this);
  
	try {
		close();
	}
	catch (XInterfaceError &e) {
        e.print(getLabel() + i18n(": Closing interface failed, because"));
	}

	iterate_commit([=](Transaction &tr){
		tr[ *device()].setUIEnabled(true);
		tr[ *port()].setUIEnabled(true);
		tr[ *address()].setUIEnabled(true);
        tr[ *control()].setUIEnabled(true);

		tr[ *control()] = false;
		tr.unmark(lsnOnControlChanged);
    });
	//g_statusPrinter->clear();
}
