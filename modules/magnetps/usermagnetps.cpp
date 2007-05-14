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
#include "usermagnetps.h"
#include <klocale.h>
//---------------------------------------------------------------------------

REGISTER_TYPE(XDriverList, PS120, "Oxford PS-120 magnet power supply");
REGISTER_TYPE(XDriverList, IPS120, "Oxford IPS-120 magnet power supply");

XPS120::XPS120(const char *name, bool runtime,
			   const shared_ptr<XScalarEntryList> &scalarentries,
			   const shared_ptr<XInterfaceList> &interfaces,
			   const shared_ptr<XThermometerList> &thermometers,
			   const shared_ptr<XDriverList> &drivers) :
    XOxfordDriver<XMagnetPS>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
}

void
XPS120::setActivity(int val) throw (XInterface::XInterfaceError&)
{
	int ret;
	interface()->lock();
	try {
		for(int i = 0; i < 3; i++)
		{
			//query Activity
			interface()->query("X");
			if(interface()->scanf("X%*2dA%1dC%*1dH%*1dM%*2dP%*2d", &ret) != 1)
				throw XInterface::XConvError(__FILE__, __LINE__);
			if(ret == val) break;
			interface()->sendf("A%u", val);
			msecsleep(i * 100);
		}
	}
	catch (XKameError &e) {
        interface()->unlock();
        throw e;
	}
	interface()->unlock();
}

void
XPS120::toPersistent()
{
	interface()->lock();
	try {
		//Set to HOLD
		interface()->send("A0");
		msecsleep(100);

		setPCSHeater(false);
	}
	catch (XKameError& e) {
        interface()->unlock();
        throw e;
	}
	interface()->unlock();
}

void
XPS120::toZero()
{
	interface()->lock();
	try {
		int ret;
		//query Activity
		interface()->query("X");
		if(interface()->scanf("X%*2dA%1dC%*1dH%*1dM%*2dP%*2d", &ret) != 1)
            throw XInterface::XConvError(__FILE__, __LINE__);
		//CLAMPED
		if(ret == 4)
		{
			//Set to HOLD
			setActivity(0);
			msecsleep(100);
		}
		//Set to TO_ZERO
		setActivity(2);
	}
	catch (XInterface::XInterfaceError&e) {
        interface()->unlock();
        throw e;
	}
	interface()->unlock();
}
void
XPS120::toNonPersistent()
{
	interface()->lock();
	try {
		int ret;
		for(int i = 0; i < 3; i++)
		{
			msecsleep(100);
			//query MODE
			interface()->query("X");
			if(interface()->scanf("X%*2dA%*1dC%*1dH%*1dM%*1d%1dP%*2d", &ret) != 1)
				throw XInterface::XConvError(__FILE__, __LINE__);
			if(ret == 0) break; //At rest
		}
		if(ret != 0)
            throw XInterface::XInterfaceError(
				KAME::i18n("Cannot enter non-persistent mode. Output is busy."), __FILE__, __LINE__);
    
		//Set to HOLD
		setActivity(0);
    
		setPCSHeater(true);
	}
	catch (XKameError& e) {
        interface()->unlock();
        throw e;
	}
	interface()->unlock();
}
void
XPS120::toSetPoint()
{
	interface()->lock();
	try {
		int ret;
		//query Activity
		interface()->query("X");
		if(interface()->scanf("X%*2dA%1dC%*1dH%*1dM%*2dP%*2d", &ret) != 1)
            throw XInterface::XConvError(__FILE__, __LINE__);
		//CLAMPED
		if(ret == 4)
		{
			//Set to HOLD
			setActivity(0);
			msecsleep(300);
		}
        setActivity(1);
	}
	catch (XKameError& e) {
        interface()->unlock();
        throw e;
	}
	interface()->unlock();
}

void
XPS120::setPoint(double field)
{
	for(int i = 0; i < 2; i++)
	{
		int df;
		if(fabs(getTargetField() - field) < fieldResolution()) break;
		msecsleep(100);
		interface()->sendf("P%d", ((field >= 0) ? 1 : 2));
		df = lrint(fabs(field) / fieldResolution());
		interface()->sendf("J%d", df);
	}
}
void
XIPS120::setPoint(double field)
{
	for(int i = 0; i < 2; i++)
	{
		if(fabs(getTargetField() - field) < fieldResolution()) break;
		msecsleep(100);
		interface()->sendf("J%f", field);
	}
}

double
XPS120::getMagnetField()
{
	if(isPCSHeaterOn())
	{
		return getOutputField();
	}
	else
	{
		return getPersistentField();
	}
}
double
XIPS120::getSweepRate()
{
	return read(9);
}
double
XPS120::getSweepRate()
{
	return read(9) * fieldResolution();
}
double
XIPS120::getTargetField()
{
	return read(8);
}
double
XPS120::getTargetField()
{
	int ret;
	interface()->query("X");
	if(interface()->scanf("X%*2dA%*1dC%*1dH%*1dM%*2dP%1d%*1d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return ((ret & 4) ? -1 : 1) * fabs(read(8) * fieldResolution());
}
double
XIPS120::getPersistentField()
{
	return read(18);
}
double
XPS120::getPersistentField()
{
	int ret;
	interface()->query("X");
	if(interface()->scanf("X%*2dA%*1dC%*1dH%*1dM%*2dP%1d%*1d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return ((ret & 2) ? -1 : 1) * fabs(read(18) * fieldResolution());
}
double
XIPS120::getOutputField()
{
	return read(7);
}
double
XPS120::getOutputField()
{
	int ret;
	interface()->query("X");
	if(interface()->scanf("X%*2dA%*1dC%*1dH%*1dM%*2dP%1d%*1d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return ((ret & 1) ? -1 : 1) * fabs(read(7) * fieldResolution());
}
double
XIPS120::getOutputVolt()
{
	return read(1);
}
double
XPS120::getOutputVolt()
{
	return read(1) * voltageResolution();
}
double
XIPS120::getOutputCurrent()
{
	return read(0);
}
double
XPS120::getOutputCurrent()
{
	int ret;
	interface()->query("X");
	if(interface()->scanf("X%*2dA%*1dC%*1dH%*1dM%*2dP%1d%*1d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return ((ret & 1) ? -1 : 1) * fabs(read(0) * currentResolution());
}
bool
XPS120::isPCSHeaterOn()
{
	int ret;
	interface()->query("X");
	if(interface()->scanf("X%*2dA%*1dC%*1dH%1dM%*2dP%*2d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return (ret == 1) || (ret == 8) || (ret == 5); //On or Fault or NOPCS
}
bool
XPS120::isPCSFitted()
{
	int ret;
	interface()->query("X");
	if(interface()->scanf("X%*2dA%*1dC%*1dH%1dM%*2dP%*2d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return (ret != 8);
}
void
XPS120::setPCSHeater(bool val) throw (XInterface::XInterfaceError&)
{
	interface()->sendf("H%u", (unsigned int)(val ? 1 : 0));
	msecsleep(200);
	if(isPCSHeaterOn() != val)
		throw XInterface::XInterfaceError(
			KAME::i18n("Persistent Switch Heater not responding"), __FILE__, __LINE__);
}
void
XIPS120::setRate(double hpm)
{
	for(int i = 0; i < 2; i++)
	{
		if(fabs(getSweepRate() - hpm) < fieldResolution()) break;
		interface()->sendf("T%f", hpm);
		msecsleep(100);
	}
}

void
XPS120::setRate(double hpm)
{
	int ihpm = lrint(hpm / fieldResolution());
	for(int i = 0; i < 2; i++)
	{
		if(fabs(getSweepRate() - hpm) < fieldResolution()) break;
		interface()->sendf("T%d", ihpm);
		msecsleep(100);
	}
}

void
XIPS120::open() throw (XInterface::XInterfaceError &)
{
	interface()->send("$Q6");
	start();
}
