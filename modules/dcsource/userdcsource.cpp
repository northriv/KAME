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
#include "charinterface.h"
#include "userdcsource.h"

REGISTER_TYPE(XDriverList, YK7651, "YOKOGAWA 7651 dc source");
REGISTER_TYPE(XDriverList, MicroTaskTCS, "MicroTask/Leiden Triple Current Source");

XYK7651::XYK7651(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) 
   : XCharDeviceDriver<XDCSource>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
  function()->add("F1");
  function()->add("F5");
}
void
XYK7651::changeFunction(int )
{
  interface()->send(function()->to_str() + "E");
}
void
XYK7651::changeOutput(bool x)
{
  interface()->sendf("O%uE", x ? 1 : 0);
}
void
XYK7651::changeValue(double x)
{
  interface()->sendf("SA%.10fE", x);
}
void
XYK7651::open() throw (XInterface::XInterfaceError &)
{
	this->start();
	channel()->setUIEnabled(false);
}

XMicroTaskTCS::XMicroTaskTCS(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) 
   : XCharDeviceDriver<XDCSource>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
	interface()->setEOS("\n");
	interface()->baudrate()->value(9600);
	channel()->value(1);
}
void
XMicroTaskTCS::changeFunction(int )
{
}
void
XMicroTaskTCS::changeOutput(bool x)
{
	unsigned int ch = *channel();
	if((ch < 1) || (ch > 3))
		throw XInterface::XInterfaceError(KAME::i18n("Value is out of range."), __FILE__, __LINE__);
	unsigned int v[3];
	v[ch - 1] = x ? 1 : 0;
	interface()->sendf("SETUP 0,0,%u,0,0,0,%u,0,0,0,%u,0", v[0], v[1], v[2]);
	interface()->receive(2);
}
void
XMicroTaskTCS::changeValue(double x)
{
	unsigned int ch = *channel();
	if((ch < 1) || (ch > 3) ||
	 (x > 0.1) || (x < 0))
		throw XInterface::XInterfaceError(KAME::i18n("Value is out of range."), __FILE__, __LINE__);
	interface()->sendf("SETDAC %u 0 %u", ch, (unsigned int)lrint(x * 1e6));
	interface()->receive(1);
}
void
XMicroTaskTCS::open() throw (XInterface::XInterfaceError &)
{
	this->start();
	function()->setUIEnabled(false);
}
