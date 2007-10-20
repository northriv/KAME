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
REGISTER_TYPE(XDriverList, HP3478A, "HP3478A DMM");

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
XHP3478A::XHP3478A(const char *name, bool runtime,
		 const shared_ptr<XScalarEntryList> &scalarentries,
		 const shared_ptr<XInterfaceList> &interfaces,
		 const shared_ptr<XThermometerList> &thermometers,
		 const shared_ptr<XDriverList> &drivers) :
	XCharDeviceDriver<XDMM>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
	interface()->setGPIBMAVbit(0x01);
//	setEOS("\r\n");
	function()->add("VOLT:DC");
	function()->add("VOLT:AC");
	function()->add("RES");
	function()->add("FRES");
	function()->add("CURR:DC");
	function()->add("CURR:AC");
	function()->add("EXTRES");
}
void
XHP3478A::changeFunction()
{
    int func = *function();
    if(func < 0)
    		throw XInterface::XInterfaceError(KAME::i18n("Select function!"), __FILE__, __LINE__);
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
