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

#ifndef userdcsourceH
#define userdcsourceH

#include "chardevicedriver.h"
#include "dcsource.h"

//!YOKOGAWA 7551 DC V/DC A source
class XYK7651:public XCharDeviceDriver<XDCSource> {
public:
	XYK7651(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual void changeFunction(int ch, int x);
	virtual void changeOutput(int ch, bool x);
	virtual void changeValue(int ch, double x, bool autorange);
	virtual void changeRange(int, int);
	virtual double max(int ch, bool autorange) const;
	virtual void queryStatus(Transaction &, int) {}
protected:
	virtual void open() throw (XInterface::XInterfaceError &);
};

//!ADVANTEST TR6142/R6142/R6144 DC V/DC A source
class XADVR6142:public XCharDeviceDriver<XDCSource> {
public:
	XADVR6142(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual void changeFunction(int ch, int x);
	virtual void changeOutput(int ch, bool x);
	virtual void changeValue(int ch, double x, bool autorange);
	virtual void changeRange(int, int);
	virtual double max(int ch, bool autorange) const;
	virtual void queryStatus(Transaction &, int) {}
protected:
	virtual void open() throw (XInterface::XInterfaceError &);
};

//!MicroTask/Leiden Triple Current Source.
class XMicroTaskTCS:public XCharDeviceDriver<XDCSource> {
public:
	XMicroTaskTCS(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual void changeFunction(int, int) {}
	virtual void changeOutput(int ch, bool x);
	virtual void changeValue(int ch, double x, bool autorange);
	virtual void changeRange(int ch, int x);
	virtual double max(int ch, bool autorange) const;
	virtual void queryStatus(Transaction &tr, int ch);
protected:
	virtual void open() throw (XInterface::XInterfaceError &);
};
#endif

