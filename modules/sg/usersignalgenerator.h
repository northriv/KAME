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
#ifndef usersignalgeneratorH
#define usersignalgeneratorH

#include "chardevicedriver.h"
#include "signalgenerator.h"

//! KENWOOD SG-7200
class XSG7200 : public XCharDeviceDriver<XSG> {
public:
	XSG7200(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XSG7200() {}

protected:
    virtual double getFreq() override {throw XInterface::XUnsupportedFeatureError(__FILE__, __LINE__);} //!< [MHz]
    virtual void changeFreq(double mhz) override;
    virtual void onRFONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onFMONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAMONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAMDepthChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onFMDevChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onAMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onFMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onSweepCondChanged(const Snapshot &shot, XValueNodeBase *) override {}
private:
};

//! KENWOOD SG-7130
class XSG7130 : public XSG7200 {
public:
	XSG7130(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XSG7130() {}
};

//! Agilent 8643A, 8644A
class XHP8643 : public XCharDeviceDriver<XSG> {
public:
	XHP8643(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XHP8643() {}
protected:
    virtual double getFreq(); //!< [MHz]
    virtual void changeFreq(double mhz) override;
    virtual void onRFONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onFMONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAMONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAMDepthChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onFMDevChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onAMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onFMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onSweepCondChanged(const Snapshot &shot, XValueNodeBase *) override {}
private:
};

//! Agilent 8648
class XHP8648 : public XHP8643 {
public:
	XHP8648(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XHP8648() {}
protected:
    virtual void onRFONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *) override;
private:
};

//! Agilent E44*1B SCPI
class XAgilentSGSCPI : public XCharDeviceDriver<XSG> {
public:
    XAgilentSGSCPI(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XAgilentSGSCPI() {}
protected:
    virtual double getFreq() override; //!< [MHz]
    virtual void changeFreq(double mhz) override;
    virtual void onRFONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onFMONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAMONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAMDepthChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onFMDevChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onFMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onSweepCondChanged(const Snapshot &shot, XValueNodeBase *) override;
private:
};

//! Agilent 8664A, 8665A
class XHP8664 : public XAgilentSGSCPI {
public:
    XHP8664(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XHP8664() {}
protected:
    virtual void onRFONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *) override;
private:
};

class XLibreVNASGSCPI : public XCharDeviceDriver<XSG> {
public:
    XLibreVNASGSCPI(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XLibreVNASGSCPI() {}
protected:
    virtual double getFreq() override; //!< [MHz]
    virtual void changeFreq(double mhz) override;
    virtual void onRFONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onFMONChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onAMONChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onAMDepthChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onFMDevChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onAMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onFMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onSweepCondChanged(const Snapshot &shot, XValueNodeBase *) override {}
private:
};

//! DS Technology DPL-3.2XGF
class XDPL32XGF : public XCharDeviceDriver<XSG> {
public:
	XDPL32XGF(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XDPL32XGF() {}
protected:
    virtual double getFreq() {throw XInterface::XUnsupportedFeatureError(__FILE__, __LINE__);} //!< [MHz]
    virtual void changeFreq(double mhz);
    virtual void onRFONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onFMONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAMONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAMDepthChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onFMDevChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onAMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onFMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onSweepCondChanged(const Snapshot &shot, XValueNodeBase *) override {}
private:
};

//! Rhode-Schwartz SML01/02/03 SMV03
class XRhodeSchwartzSMLSMV : public XCharDeviceDriver<XSG> {
public:
	XRhodeSchwartzSMLSMV(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XRhodeSchwartzSMLSMV() {}
protected:
    virtual double getFreq() override {throw XInterface::XUnsupportedFeatureError(__FILE__, __LINE__);} //!< [MHz]
    virtual void changeFreq(double mhz) override;
    virtual void onRFONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onFMONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAMONChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAMDepthChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onFMDevChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onAMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onFMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onSweepCondChanged(const Snapshot &shot, XValueNodeBase *) override {}
private:
};
#endif
