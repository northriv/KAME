/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
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
class XYK7651:public XCharDeviceDriver<XDCSource>
{
public:
	XYK7651(const char *name, bool runtime,
			const shared_ptr<XScalarEntryList> &scalarentries,
			const shared_ptr<XInterfaceList> &interfaces,
			const shared_ptr<XThermometerList> &thermometers,
			const shared_ptr<XDriverList> &drivers);
protected:
	virtual void changeFunction(int ch, int x);
	virtual void changeOutput(int ch, bool x);
	virtual void changeValue(int ch, double x);
	virtual void queryStatus(int) {}
};

//!MicroTask/Leiden Triple Current Source.
class XMicroTaskTCS:public XCharDeviceDriver<XDCSource>
{
public:
	XMicroTaskTCS(const char *name, bool runtime,
			const shared_ptr<XScalarEntryList> &scalarentries,
			const shared_ptr<XInterfaceList> &interfaces,
			const shared_ptr<XThermometerList> &thermometers,
			const shared_ptr<XDriverList> &drivers);
protected:
	virtual void open() throw (XInterface::XInterfaceError &);
	virtual void changeFunction(int, int) {}
	virtual void changeOutput(int ch, bool x);
	virtual void changeValue(int ch, double x);
	virtual void queryStatus(int ch);
};
#endif

