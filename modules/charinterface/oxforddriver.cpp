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
#include "oxforddriver.h"

XOxfordInterface::XOxfordInterface
(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
	: XCharInterface(name, runtime, driver) {
    setEOS("\r\n");
    setGPIBWaitBeforeSPoll(10);
}
void
XOxfordInterface::send(const XString &str) {
    this->send(str.c_str());
}
void
XOxfordInterface::send(const char *str) {
	assert(strlen(str));
	if(str[0] == '$') {
		XCharInterface::send(str);
	}
	else {
		//Oxfords always send back echo
		query(str);
	}
}
void
XOxfordInterface::query(const XString &str) {
    query(str.c_str());
}
void
XOxfordInterface::query(const char *str) {
	lock();
	try {
		for(int i = 0; i < 30; i++) {
			XCharInterface::send(str);
			XCharInterface::receive();
			if(buffer().size() >= 1)
				if(buffer()[0] == str[0]) {
					unlock();
					return;
				}
			msecsleep(100);
		}
	}
	catch (XCommError &e) {
		unlock();
		throw e;
	}
	unlock();
	throw XCommError(i18n("Oxford Query Error, Initial doesn't match"), __FILE__, __LINE__);
}

void
XOxfordInterface::open() {
    XCharInterface::open();
	//    XDriver::Send("@0");
	send("$Q2");
	//    msecsleep(100);
	//remote & unlocked
	send("C3");
}

void
XOxfordInterface::close() {
    if(isOpened()) {
        try {
            send("C0"); //local
        }
        catch (XInterfaceError &e) {
            e.print(getLabel());
        }
    }
	XCharInterface::close();
}
void
XOxfordInterface::receive() {
    XCharInterface::receive();
}
void
XOxfordInterface::receive(unsigned int length) {
    XCharInterface::receive(length);
}


