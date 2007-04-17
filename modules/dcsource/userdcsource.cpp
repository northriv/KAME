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
REGISTER_TYPE(XDriverList, MicroTaskTCS, "MICROTASK/Leiden Triple Current Source");

XYK7651::XYK7651(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) 
   : XCharDeviceDriver<XDCSource>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
  function()->add("F1");
  function()->add("F5");
  channel()->disable();
  range()->disable();
}
void
XYK7651::changeFunction(int /*ch*/, int )
{
	XScopedLock<XInterface> lock(*interface());
	if(!interface()->isOpened()) return;
	interface()->send(function()->to_str() + "E");
}
void
XYK7651::changeOutput(int /*ch*/, bool x)
{
	XScopedLock<XInterface> lock(*interface());
	if(!interface()->isOpened()) return;
	interface()->sendf("O%uE", x ? 1 : 0);
}
void
XYK7651::changeValue(int /*ch*/, double x, bool /*autorange*/)
{
	XScopedLock<XInterface> lock(*interface());
	if(!interface()->isOpened()) return;
	interface()->sendf("SA%.10fE", x);
}
double
XYK7651::max(bool /*autorange*/) const
{
	if(*function() == 0)
		return 30;
	return 0.12;
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
	channel()->add("1");
	channel()->add("2");
	channel()->add("3");
	function()->disable();
	range()->add("99uA");
	range()->add("990uA");
	range()->add("0.99mA");
	range()->add("9.9mA");
}
void
XMicroTaskTCS::queryStatus(int ch)
{
	XScopedLock<XInterface> lock(*interface());
	if(!interface()->isOpened()) return;
	unsigned int ran[3];
	unsigned int v[3];
	unsigned int o[3];
	interface()->query("STATUS?");
	if(interface()->scanf("%*u%*u,%u,%u,%u,%*u,%u,%u,%u,%*u,%u,%u,%u,%*u",
		&ran[0], &v[0], &o[0],
		&ran[1], &v[1], &o[1],
		&ran[2], &v[2], &o[2]) != 9)
		throw XInterface::XConvError(__FILE__, __LINE__);
	value()->value(pow(10.0, (double)ran[ch] - 1) * 1e-6 * v[ch]);
	output()->value(o[ch]);
	range()->value(ran[ch] - 1);
}
void
XMicroTaskTCS::changeOutput(int ch, bool x)
{
	XScopedLock<XInterface> lock(*interface());
	if(!interface()->isOpened()) return;
	unsigned int v[3];
	interface()->query("STATUS?");
	if(interface()->scanf("%*u%*u,%*u,%*u,%u,%*u,%*u,%*u,%u,%*u,%*u,%*u,%u,%*u", &v[0], &v[1], &v[2])
		!= 3)
		throw XInterface::XConvError(__FILE__, __LINE__);
	for(int i = 0; i < 3; i++) {
		if(ch != i)
			v[i] = 0;
		else
			v[i] ^= x ? 1 : 0;
	}
	interface()->sendf("SETUP 0,0,%u,0,0,0,%u,0,0,0,%u,0", v[0], v[1], v[2]);
	interface()->receive(2);
	updateStatus();
}
void
XMicroTaskTCS::changeValue(int ch, double x, bool autorange)
{
	XScopedLock<XInterface> lock(*interface());
	if(!interface()->isOpened()) return;
	if((x >= 0.0099) || (x < 0))
		throw XInterface::XInterfaceError(KAME::i18n("Value is out of range."), __FILE__, __LINE__);
	if(autorange) {
		interface()->sendf("SETDAC %u 0 %u", (unsigned int)(ch + 1), (unsigned int)lrint(x * 1e6));
		interface()->receive(1);
	}
	else {
		int ran = (int)*range();
		int v = lrint(x / (pow(10.0, (double)ran) * 1e-6));
		v = std::max(std::min(v, 99), 0);
		interface()->sendf("SETDAC %u %u %u", (unsigned int)(ch + 1), (unsigned int)(ran + 1), (unsigned int)v);
		interface()->receive(1);
	}
	updateStatus();
}
void
XMicroTaskTCS::changeRange(int ch, int /*ran*/)
{
	changeValue(ch, *value(), false);
}
double
XMicroTaskTCS::max(bool autorange) const
{
	if(autorange) return 0.0099;
	return pow(10.0, (double)*range()) * 99e-6;
}
void
XMicroTaskTCS::open() throw (XInterface::XInterfaceError &)
{
	this->start();
	interface()->query("ID?");
	fprintf(stderr, "%s\n", (const char*)&interface()->buffer()[0]);
}
