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
XInterface::XUnsupportedFeatureError::XUnsupportedFeatureError(const char *file, int line)
    :  XInterfaceError(i18n("Unsupported Feature Error"), file, line) {}

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
    control()->setUIEnabled(false);
    if(shot[ *control()]) {
        g_statusPrinter->printMessage(driver()->getLabel() + i18n(": Starting..."));
        m_threadStart.reset(new XThread{shared_from_this(), [this](const atomic<bool>&) {
            start();
            control()->setUIEnabled(true);
            }});
	}
	else {
        Snapshot shot( *this);
        shot.talk(shot[ *this].onClose(), this); //stop() will be called here.
        control()->setUIEnabled(true);
    }
}

void
XInterface::start() {
    XScopedLock<XInterface> lock( *this);
    Transactional::setCurrentPriorityMode(Priority::NORMAL);
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

    Snapshot shot = iterate_commit([=](Transaction &tr){
        tr[ *device()].setUIEnabled(false);
        tr[ *port()].setUIEnabled(false);
        tr[ *address()].setUIEnabled(false);
        tr[ *control()] = true; //to set a proper icon before enabling UI.
        tr.unmark(lsnOnControlChanged);
    });
    shot.talk(shot[ *this].onOpen(), this);
}
void
XInterface::stop() {
    m_threadStart.reset();
	XScopedLock<XInterface> lock( *this);
    Transactional::setCurrentPriorityMode(Priority::NORMAL);
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
		tr[ *control()] = false;
		tr.unmark(lsnOnControlChanged);
    });
    //g_statusPrinter->clear();
}
