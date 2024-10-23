/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef userlockinampH
#define userlockinampH

#include "lockinamp.h"
#include "chardevicedriver.h"
//---------------------------------------------------------------------------
//! Stanford Research SR830 Lock-in Amplifier
class XSR830 : public XCharDeviceDriver<XLIA> {
public:
	XSR830(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
protected:
	virtual void get(double *cos, double *sin);
	virtual void changeOutput(double volt);
	virtual void changeFreq(double freq);
	virtual void changeSensitivity(int);
	virtual void changeTimeConst(int);

	//! Be called just after opening interface. Call start() inside this routine appropriately.
	virtual void open();
	//! Be called for closing interfaces.
	virtual void closeInterface();

	int m_cCount;
};

//! Lakeshore M81-SSM LIA module
class XLakeshoreM81LIA : public XCharDeviceDriver<XLIA> {
public:
    XLakeshoreM81LIA(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
protected:
    virtual void get(double *cos, double *sin);
    virtual void changeOutput(double volt);
    virtual void changeFreq(double freq);
    virtual void changeSensitivity(int);
    virtual void changeTimeConst(int);

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open();
    //! Be called for closing interfaces.
    virtual void closeInterface();
private:
    int channel();
};

//! NF LI 5640 Lock-in Amplifier
class XLI5640 : public XCharDeviceDriver<XLIA> {
public:
	XLI5640(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
protected:
	virtual void get(double *cos, double *sin);
	virtual void changeOutput(double volt);
	virtual void changeFreq(double freq);
	virtual void changeSensitivity(int);
	virtual void changeTimeConst(int);

	//! Be called just after opening interface. Call start() inside this routine appropriately.
	virtual void open();
	//! Be called for closing interfaces.
	virtual void closeInterface();

	int m_cCount;
	bool m_currMode;
};

//! Signal Recovery Model7265 Lock-in Amplifier
class XSignalRecovery7265 : public XCharDeviceDriver<XLIA> {
public:
    XSignalRecovery7265(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
protected:
    virtual void get(double *cos, double *sin);
    virtual void changeOutput(double volt);
    virtual void changeFreq(double freq);
    virtual void changeSensitivity(int);
    virtual void changeTimeConst(int);

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open();
    //! Be called for closing interfaces.
    virtual void closeInterface();
};

//! Agilent/HP4284A Precision LCR Meter
class XHP4284A : public XCharDeviceDriver<XLIA> {
public:
    XHP4284A(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
protected:
    virtual void get(double *cos, double *sin);
    virtual void changeOutput(double volt);
    virtual void changeFreq(double freq);
    virtual void changeSensitivity(int);
    virtual void changeTimeConst(int);
    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open();
    //! Be called for closing interfaces.
    virtual void closeInterface();
};

//! ANDEEN HAGERLING 2500A 1kHz Ultra-Precision Capcitance Bridge
class XAH2500A : public XCharDeviceDriver<XLIA> {
public:
	XAH2500A(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
protected:
	virtual void get(double *cos, double *sin);
	virtual void changeOutput(double volt);
	virtual void changeFreq(double freq);
	virtual void changeSensitivity(int);
	virtual void changeTimeConst(int);
	//! Be called just after opening interface. Call start() inside this routine appropriately.
	virtual void open();
	//! Be called for closing interfaces.
	virtual void closeInterface();
};

#endif
