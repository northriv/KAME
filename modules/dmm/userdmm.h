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
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas, unsigned int num_channels = 1) :
        XCharDeviceDriver<XDMM>(name, runtime, ref(tr_meas), meas, num_channels) {}
	virtual ~XDMMSCPI() {}

	//! requests the latest reading
	virtual double fetch();
	//! one-shot reading
    virtual double oneShotRead() override;
    //! one-shot multi-channel reading
    virtual std::deque<double> oneShotMultiRead() override;
protected:
	//! called when m_function is changed
    virtual void changeFunction() override;
};


//! Keithley 2182 nanovolt meter
//! One must setup 2182 for SCPI mode
class XKE2182:public XDMMSCPI {
public:
	XKE2182(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
        XDMMSCPI(name, runtime, ref(tr_meas), meas, 2) { //2channels
		iterate_commit([=](Transaction &tr){
			tr[ *function()].add("VOLT");
			tr[ *function()].add("TEMP");
        });
	}
};

//! Keithley 2000 Multimeter
//! One must setup 2000 for SCPI mode
class XKE2000:public XDMMSCPI {
public:
	XKE2000(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
		XDMMSCPI(name, runtime, ref(tr_meas), meas) {
		iterate_commit([=](Transaction &tr){
			tr[ *function()].add("VOLT:DC");
			tr[ *function()].add("VOLT:AC");
			tr[ *function()].add("CURR:DC");
			tr[ *function()].add("CURR:AC");
			tr[ *function()].add("RES");
			tr[ *function()].add("FRES");
			tr[ *function()].add("FREQ");
			tr[ *function()].add("TEMP");
			tr[ *function()].add("PER");
			tr[ *function()].add("DIOD");
			tr[ *function()].add("CONT");
        });

		interface()->setGPIBWaitBeforeRead(20);
	}
};


//! Keithley Integra 2700 w/ 7700 switching module.
class XKE2700_7700 : public XCharDeviceDriver<XDMM> {
public:
    XKE2700_7700(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XKE2700_7700() {}

protected:
    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;
    //! one-shot multi-channel reading
    virtual std::deque<double> oneShotMultiRead() override;
    //! one-shot reading
    virtual double oneShotRead() override {return 0.0;}
    //! called when m_function is changed
    virtual void changeFunction() override {}
private:
};

//! Agilent(Hewlett-Packard) 34420A nanovolt meter
class XHP34420A:public XDMMSCPI {
public:
	XHP34420A(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
		XDMMSCPI(name, runtime, ref(tr_meas), meas)
	{
		iterate_commit([=](Transaction &tr){
			tr[ *function()].add("VOLT");
			tr[ *function()].add("CURR");
			tr[ *function()].add("RES");
			tr[ *function()].add("FRES");
        });
	}
};

//! Keithley 6482 picoam meter
//! One must setup instruments for SCPI mode
class XKE6482:public XDMMSCPI {
public:
    XKE6482(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
        XDMMSCPI(name, runtime, ref(tr_meas), meas) {
        iterate_commit([=](Transaction &tr){
            tr[ *function()].add("CURR1");
            tr[ *function()].add("CURR2");
        });
    }
protected:
    //! called when m_function is changed
    virtual void changeFunction();
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
