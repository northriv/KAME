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
#ifndef OXFORDDRIVER_H_
#define OXFORDDRIVER_H_

#include "charinterface.h"
#include "chardevicedriver.h"
#include "primarydriver.h"

class DECLSPEC_SHARED XOxfordInterface : public XCharInterface {
public:
	XOxfordInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);

    virtual void open() throw (XInterfaceError &);
	virtual void close() throw (XInterfaceError &);
  
	void send(const XString &str);
	virtual void send(const char *str);
	//! don't use me
	virtual void write(const char *, int) {
		assert(false);
	}
	virtual void receive();
	virtual void receive(unsigned int length);
	void query(const XString &str);
	virtual void query(const char *str);
};

template <class tDriver>
class XOxfordDriver : public XCharDeviceDriver<tDriver, XOxfordInterface> {
public:
	XOxfordDriver(const char *name, bool runtime, 
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
		: XCharDeviceDriver<tDriver, XOxfordInterface>(name, runtime, ref(tr_meas), meas) {}
	double read(int arg);
};

template<class tDriver>
double
XOxfordDriver<tDriver>::read(int arg) {
	double x;
	this->interface()->queryf("R%d", arg);
	int ret = this->interface()->scanf("R%lf", &x);
	if(ret != 1) throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}

#endif /*OXFORDDRIVER_H_*/
