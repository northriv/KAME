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
	virtual void changeFreq(double mhz);
	virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onFMONChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onAMONChanged(const Snapshot &shot, XValueNodeBase *);
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
	virtual void changeFreq(double mhz);
	virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onFMONChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onAMONChanged(const Snapshot &shot, XValueNodeBase *);
private:
};

//! Agilent 8648
class XHP8648 : public XHP8643 {
public:
	XHP8648(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XHP8648() {}
protected:
	virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *);
private:
};

//! Agilent 8664A, 8665A
class XHP8664 : public XCharDeviceDriver<XSG> {
public:
	XHP8664(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XHP8664() {}
protected:
	virtual void changeFreq(double mhz);
	virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onFMONChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onAMONChanged(const Snapshot &shot, XValueNodeBase *);
private:
};

#endif
