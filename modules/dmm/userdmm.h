/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
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

class XDMMSCPI : public XCharDeviceDriver<XDMM>
{
	XNODE_OBJECT
protected:
	XDMMSCPI(const char *name, bool runtime,
			 const shared_ptr<XScalarEntryList> &scalarentries,
			 const shared_ptr<XInterfaceList> &interfaces,
			 const shared_ptr<XThermometerList> &thermometers,
			 const shared_ptr<XDriverList> &drivers) :
		XCharDeviceDriver<XDMM>(name, runtime, scalarentries, interfaces, thermometers, drivers) {}
public:
	virtual ~XDMMSCPI() {}

	//requests the latest reading
	virtual double fetch();
	//one-shot reading
	virtual double oneShotRead();
protected:
	//! called when m_function is changed
	virtual void changeFunction();
};


//Keithley 2182 nanovolt meter
//You must setup 2182 for SCPI mode
class XKE2182:public XDMMSCPI
{
	XNODE_OBJECT
protected:
	XKE2182(const char *name, bool runtime,
			const shared_ptr<XScalarEntryList> &scalarentries,
			const shared_ptr<XInterfaceList> &interfaces,
			const shared_ptr<XThermometerList> &thermometers,
			const shared_ptr<XDriverList> &drivers) :
		XDMMSCPI(name, runtime, scalarentries, interfaces, thermometers, drivers)
	{
		function()->add("VOLT");
		function()->add("TEMP");
	}
};

//Keithley 2000 Multimeter
//You must setup 2000 for SCPI mode
class XKE2000:public XDMMSCPI
{
	XNODE_OBJECT
protected:
	XKE2000(const char *name, bool runtime,
			const shared_ptr<XScalarEntryList> &scalarentries,
			const shared_ptr<XInterfaceList> &interfaces,
			const shared_ptr<XThermometerList> &thermometers,
			const shared_ptr<XDriverList> &drivers) :
		XDMMSCPI(name, runtime, scalarentries, interfaces, thermometers, drivers)
	{
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

//Agilent(Hewlett-Packard) 34420A nanovolt meter
class XHP34420A:public XDMMSCPI
{
	XNODE_OBJECT
protected:
	XHP34420A(const char *name, bool runtime,
			  const shared_ptr<XScalarEntryList> &scalarentries,
			  const shared_ptr<XInterfaceList> &interfaces,
			  const shared_ptr<XThermometerList> &thermometers,
			  const shared_ptr<XDriverList> &drivers) :
		XDMMSCPI(name, runtime, scalarentries, interfaces, thermometers, drivers)
	{
		function()->add("VOLT");
		function()->add("CURR");
		function()->add("RES");
		function()->add("FRES");
	}
};

class XHP3478A : public XCharDeviceDriver<XDMM>
{
	XNODE_OBJECT
protected:
	XHP3478A(const char *name, bool runtime,
			 const shared_ptr<XScalarEntryList> &scalarentries,
			 const shared_ptr<XInterfaceList> &interfaces,
			 const shared_ptr<XThermometerList> &thermometers,
			 const shared_ptr<XDriverList> &drivers);
public:
	virtual ~XHP3478A() {}

	//requests the latest reading
	virtual double fetch();
	//one-shot reading
	virtual double oneShotRead();
protected:
	//! called when m_function is changed
	virtual void changeFunction();
};

#endif
