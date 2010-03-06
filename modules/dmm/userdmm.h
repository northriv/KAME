/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#ifndef userdmmH
#define userdmmH

#include "chardevicedriver.h"
#include "charinterface.h"
#include "dmm.h"
//---------------------------------------------------------------------------
//! Base class for SCPI DMMs.
class XDMMSCPI : public XCharDeviceDriver<XDMM> {
public:
	XDMMSCPI(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
		XCharDeviceDriver<XDMM>(name, runtime, ref(tr_meas), meas) {}
	virtual ~XDMMSCPI() {}

	//! requests the latest reading
	virtual double fetch();
	//! one-shot reading
	virtual double oneShotRead();
protected:
	//! called when m_function is changed
	virtual void changeFunction();
};


//! Keithley 2182 nanovolt meter
//! One must setup 2182 for SCPI mode
class XKE2182:public XDMMSCPI {
public:
	XKE2182(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
		XDMMSCPI(name, runtime, ref(tr_meas), meas) {
		function()->add("VOLT");
		function()->add("TEMP");
	}
};

//! Keithley 2000 Multimeter
//! One must setup 2000 for SCPI mode
class XKE2000:public XDMMSCPI {
public:
	XKE2000(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
		XDMMSCPI(name, runtime, ref(tr_meas), meas) {
		function()->add("VOLT:DC");
		function()->add("VOLT:AC");
		function()->add("CURR:DC");
		function()->add("CURR:AC");
		function()->add("RES");
		function()->add("FRES");
		function()->add("FREQ");
		function()->add("TEMP");
		function()->add("PER");
		function()->add("DIOD");
		function()->add("CONT");

		interface()->setGPIBWaitBeforeRead(20);
	}
};

//! Agilent(Hewlett-Packard) 34420A nanovolt meter
class XHP34420A:public XDMMSCPI {
public:
	XHP34420A(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
		XDMMSCPI(name, runtime, ref(tr_meas), meas)
	{
		function()->add("VOLT");
		function()->add("CURR");
		function()->add("RES");
		function()->add("FRES");
	}
};

//! Agilent(Hewlett-Packard) 3458A DMM.
class XHP3458A : public XCharDeviceDriver<XDMM> {
public:
	XHP3458A(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XHP3458A() {}

	//requests the latest reading
	virtual double fetch();
	//one-shot reading
	virtual double oneShotRead();
protected:
	//! called when m_function is changed
	virtual void changeFunction();
};

//! Agilent(Hewlett-Packard) 3478A DMM.
class XHP3478A : public XCharDeviceDriver<XDMM> {
public:
	XHP3478A(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XHP3478A() {}

	//requests the latest reading
	virtual double fetch();
	//one-shot reading
	virtual double oneShotRead();
protected:
	//! called when m_function is changed
	virtual void changeFunction();
};


//! SANWA PC500/510/520M DMM.
class XSanwaPC500 : public XCharDeviceDriver<XDMM> {
public:
	XSanwaPC500(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XSanwaPC500() {}

	//requests the latest reading
	virtual double fetch();
	//one-shot reading
	virtual double oneShotRead();
protected:
	//! called when m_function is changed
	virtual void changeFunction();
	//! send command.
	virtual void requestData();
};
//! SANWA PC5000 DMM.
class XSanwaPC5000 : public XSanwaPC500 {
public:
	XSanwaPC5000(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
protected:
	//! send command.
	virtual void requestData();
};

#endif
