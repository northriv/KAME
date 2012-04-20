/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef OXFORDDRIVER_H_
#define OXFORDDRIVER_H_

#include "charinterface.h"
#include "chardevicedriver.h"
#include "primarydriver.h"

class XOxfordInterface : public XCharInterface {
public:
	XOxfordInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);

	virtual void open() throw (XInterfaceError &);
	virtual void close() throw (XInterfaceError &);
  
	void send(const XString &str) throw (XCommError &);
	virtual void send(const char *str) throw (XCommError &);
	//! don't use me
	virtual void write(const char *, int) throw (XCommError &) {
		assert(false);
	}
	virtual void receive() throw (XCommError &);
	virtual void receive(unsigned int length) throw (XCommError &);
	void query(const XString &str) throw (XCommError &);
	virtual void query(const char *str) throw (XCommError &);
};

template <class tDriver>
class XOxfordDriver : public XCharDeviceDriver<tDriver, XOxfordInterface> {
public:
	XOxfordDriver(const char *name, bool runtime, 
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
		: XCharDeviceDriver<tDriver, XOxfordInterface>(name, runtime, ref(tr_meas), meas) {}
	double read(int arg) throw (XInterface::XInterfaceError &);
};

template<class tDriver>
double
XOxfordDriver<tDriver>::read(int arg) throw (XInterface::XInterfaceError &) {
	double x;
	this->interface()->queryf("R%d", arg);
	int ret = this->interface()->scanf("R%lf", &x);
	if(ret != 1) throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}

#endif /*OXFORDDRIVER_H_*/
