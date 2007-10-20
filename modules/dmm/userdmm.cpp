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

#include "userdmm.h"
#include "charinterface.h"
//---------------------------------------------------------------------------

REGISTER_TYPE(XDriverList, KE2000, "Keithley 2000/2001 DMM");
REGISTER_TYPE(XDriverList, KE2182, "Keithley 2182 nanovolt meter");
REGISTER_TYPE(XDriverList, HP34420A, "Agilent 34420A nanovolt meter");
REGISTER_TYPE(XDriverList, HP3458A, "Agilent 3458A DMM");
REGISTER_TYPE(XDriverList, HP3478A, "Agilent 3478A DMM");

void
XDMMSCPI::changeFunction()
{
    std::string func = function()->to_str();
    if(!func.empty())
        interface()->sendf(":CONF:%s", func.c_str());
}
double
XDMMSCPI::fetch()
{
    interface()->query(":FETC?");
    return interface()->toDouble();
}
double
XDMMSCPI::oneShotRead()
{
    interface()->query(":READ?");
    return interface()->toDouble();
}
/*
double
XDMMSCPI::measure(const std::string &func)
{
    interface()->queryf(":MEAS:%s?", func.c_str());
    return interface()->toDouble();
}
*/

XHP3458A::XHP3458A(const char *name, bool runtime,
		 const shared_ptr<XScalarEntryList> &scalarentries,
		 const shared_ptr<XInterfaceList> &interfaces,
		 const shared_ptr<XThermometerList> &thermometers,
		 const shared_ptr<XDriverList> &drivers) :
	XCharDeviceDriver<XDMM>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
	const char *funcs[] = {
		"DCV", "ACV", "ACDCV", "OHM", "OHMF", "DCI", "ACI", "ACDCI", "FREQ", "PER", "DSAC", "DSDC", "SSAC", "SSDC", ""
	};
	for(const char **func = funcs; strlen(*func); func++) {
		function()->add(*func);
	}
}
void
XHP3458A::changeFunction()
{
    std::string func = function()->to_str();
    if(!func.empty())
        interface()->sendf("FUNC %s;ARANGE ON", func.c_str());
}
double
XHP3458A::fetch()
{
    interface()->receive();
    return interface()->toDouble();
}
double
XHP3458A::oneShotRead()
{
    interface()->query("END ALWAYS;OFORMAT ASCII;QFORMAT NUM;TARM SGL;NRDGS 1;TRIG SGL");
    return interface()->toDouble();
}


XHP3478A::XHP3478A(const char *name, bool runtime,
		 const shared_ptr<XScalarEntryList> &scalarentries,
		 const shared_ptr<XInterfaceList> &interfaces,
		 const shared_ptr<XThermometerList> &thermometers,
		 const shared_ptr<XDriverList> &drivers) :
	XCharDeviceDriver<XDMM>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
	interface()->setGPIBMAVbit(0x01);
//	setEOS("\r\n");
	const char *funcs[] = {
		"DCV", "ACV", "OHM", "OHMF", "DCI", "ACI", ""
	};
	for(const char **func = funcs; strlen(*func); func++) {
		function()->add(*func);
	}
}
void
XHP3478A::changeFunction()
{
    int func = *function();
    if(func < 0)
		return;
//    		throw XInterface::XInterfaceError(KAME::i18n("Select function!"), __FILE__, __LINE__);
    interface()->sendf("F%dRAZ1", func + 1);
}
double
XHP3478A::fetch()
{
    interface()->receive();
    return interface()->toDouble();
}
double
XHP3478A::oneShotRead()
{
    interface()->query("T3");
    return interface()->toDouble();
}
