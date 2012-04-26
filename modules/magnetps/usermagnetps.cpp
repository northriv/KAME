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
//---------------------------------------------------------------------------
#include "usermagnetps.h"
//---------------------------------------------------------------------------

REGISTER_TYPE(XDriverList, PS120, "Oxford PS-120 magnet power supply");
REGISTER_TYPE(XDriverList, IPS120, "Oxford IPS-120 magnet power supply");
REGISTER_TYPE(XDriverList, CryogenicSMS, "Cryogenic SMS10/30/120C magnet power supply");

XPS120::XPS120(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XOxfordDriver<XMagnetPS>(name, runtime, ref(tr_meas), meas) {
}

void
XPS120::setActivity(int val) throw (XInterface::XInterfaceError&) {
	int ret;
	XScopedLock<XInterface> lock( *interface());
	for(int i = 0; i < 3; i++) {
		//query Activity
		interface()->query("X");
		if(interface()->scanf("X%*2dA%1dC%*1dH%*1dM%*2dP%*2d", &ret) != 1)
			throw XInterface::XConvError(__FILE__, __LINE__);
		if(ret == val) break;
		interface()->sendf("A%u", val);
		msecsleep(i * 100);
	}
}

void
XPS120::toPersistent() {
	XScopedLock<XInterface> lock( *interface());
	//Set to HOLD
	interface()->send("A0");
	msecsleep(100);

	setPCSHeater(false);
}

void
XPS120::toZero() {
	XScopedLock<XInterface> lock( *interface());
	int ret;
	//query Activity
	interface()->query("X");
	if(interface()->scanf("X%*2dA%1dC%*1dH%*1dM%*2dP%*2d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	//CLAMPED
	if(ret == 4) {
		//Set to HOLD
		setActivity(0);
		msecsleep(100);
	}
	//Set to TO_ZERO
	setActivity(2);
}
void
XPS120::toNonPersistent() {
	XScopedLock<XInterface> lock( *interface());
	int ret;
	for(int i = 0; i < 3; i++) {
		msecsleep(100);
		//query MODE
		interface()->query("X");
		if(interface()->scanf("X%*2dA%*1dC%*1dH%*1dM%*1d%1dP%*2d", &ret) != 1)
			throw XInterface::XConvError(__FILE__, __LINE__);
		if(ret == 0) break; //At rest
	}
	if(ret != 0)
		throw XInterface::XInterfaceError(
			i18n("Cannot enter non-persistent mode. Output is busy."), __FILE__, __LINE__);

	//Set to HOLD
	setActivity(0);

	setPCSHeater(true);
}
void
XPS120::toSetPoint() {
	XScopedLock<XInterface> lock( *interface());
	int ret;
	//query Activity
	interface()->query("X");
	if(interface()->scanf("X%*2dA%1dC%*1dH%*1dM%*2dP%*2d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	//CLAMPED
	if(ret == 4) {
		//Set to HOLD
		setActivity(0);
		msecsleep(300);
	}
	setActivity(1);
}

void
XPS120::setPoint(double field) {
	for(int i = 0; i < 2; i++) {
		int df;
		if(fabs(getTargetField() - field) < fieldResolution()) break;
		msecsleep(100);
		interface()->sendf("P%d", ((field >= 0) ? 1 : 2));
		df = lrint(fabs(field) / fieldResolution());
		interface()->sendf("J%d", df);
	}
}
void
XIPS120::setPoint(double field) {
	for(int i = 0; i < 2; i++) {
		if(fabs(getTargetField() - field) < fieldResolution()) break;
		msecsleep(100);
		interface()->sendf("J%f", field);
	}
}

double
XPS120::getMagnetField() {
	if(isPCSHeaterOn()) {
		return getOutputField();
	}
	else {
		return getPersistentField();
	}
}
double
XIPS120::getSweepRate() {
	return read(9);
}
double
XPS120::getSweepRate() {
	return read(9) * fieldResolution();
}
double
XIPS120::getTargetField() {
	return read(8);
}
double
XPS120::getTargetField() {
	int ret;
	interface()->query("X");
	if(interface()->scanf("X%*2dA%*1dC%*1dH%*1dM%*2dP%1d%*1d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return ((ret & 4) ? -1 : 1) * fabs(read(8) * fieldResolution());
}
double
XIPS120::getPersistentField() {
	return read(18);
}
double
XPS120::getPersistentField() {
	int ret;
	interface()->query("X");
	if(interface()->scanf("X%*2dA%*1dC%*1dH%*1dM%*2dP%1d%*1d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return ((ret & 2) ? -1 : 1) * fabs(read(18) * fieldResolution());
}
double
XIPS120::getOutputField() {
	return read(7);
}
double
XPS120::getOutputField() {
	int ret;
	interface()->query("X");
	if(interface()->scanf("X%*2dA%*1dC%*1dH%*1dM%*2dP%1d%*1d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return ((ret & 1) ? -1 : 1) * fabs(read(7) * fieldResolution());
}
double
XIPS120::getOutputVolt() {
	return read(1);
}
double
XPS120::getOutputVolt() {
	return read(1) * voltageResolution();
}
double
XIPS120::getOutputCurrent() {
	return read(0);
}
double
XPS120::getOutputCurrent() {
	int ret;
	interface()->query("X");
	if(interface()->scanf("X%*2dA%*1dC%*1dH%*1dM%*2dP%1d%*1d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return ((ret & 1) ? -1 : 1) * fabs(read(0) * currentResolution());
}
bool
XPS120::isPCSHeaterOn() {
	int ret;
	interface()->query("X");
	if(interface()->scanf("X%*2dA%*1dC%*1dH%1dM%*2dP%*2d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return (ret == 1) || (ret == 8) || (ret == 5); //On or Fault or NOPCS
}
bool
XPS120::isPCSFitted() {
	int ret;
	interface()->query("X");
	if(interface()->scanf("X%*2dA%*1dC%*1dH%1dM%*2dP%*2d", &ret) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return (ret != 8);
}
void
XPS120::setPCSHeater(bool val) throw (XInterface::XInterfaceError&) {
	interface()->sendf("H%u", (unsigned int)(val ? 1 : 0));
	msecsleep(200);
	if(isPCSHeaterOn() != val)
		throw XInterface::XInterfaceError(
			i18n("Persistent Switch Heater not responding"), __FILE__, __LINE__);
}
void
XIPS120::setRate(double hpm) {
	for(int i = 0; i < 2; i++) {
		if(fabs(getSweepRate() - hpm) < fieldResolution()) break;
		interface()->sendf("T%f", hpm);
		msecsleep(100);
	}
}

void
XPS120::setRate(double hpm) {
	int ihpm = lrint(hpm / fieldResolution());
	for(int i = 0; i < 2; i++) {
		if(fabs(getSweepRate() - hpm) < fieldResolution()) break;
		interface()->sendf("T%d", ihpm);
		msecsleep(100);
	}
}

void
XIPS120::open() throw (XInterface::XInterfaceError &) {
	interface()->send("$Q6");
	start();
}

XCryogenicSMS::XCryogenicSMS(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XMagnetPS>(name, runtime, ref(tr_meas), meas) {
/*
 * Notes not mentioned in the manufacturer's manual.
 * GET PER command does not return a value or delimiter when it is not in persistent mode or at zero field.
 * RAMP ... command does not reply.
 * PAUSE ... command does not reply the second line.
 * Some commands respond with a form of HH:MM:SS ....... (command).
 * Local button operations will emit status lines.
 *
 * This driver assumes...
 * (i) TPA (tesla per ampere) has been set properly.
 * (ii) PCSH is fitted.
 */
    interface()->setEOS("\r\n");
}
void
XCryogenicSMS::changePauseState(bool pause) {
// Lock before calling me.
//	XScopedLock<XInterface> lock( *interface());
	interface()->query("PAUSE");
	char buf[10];
	if(interface()->scanf("%*s PAUSE STATUS: %4s", buf) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
	if( !strncmp("ON", buf, 2)) {
		if(pause)
			return;
		interface()->query("PAUSE OFF");
		char buf[10];
		if(interface()->scanf("%*2d:%*2d:%*2d PAUSE STATUS: %4s", buf) != 1)
	        throw XInterface::XConvError(__FILE__, __LINE__);
//		interface()->receive();
//		double x;
//		if(interface()->scanf("%*2d:%*2d:%*2d RAMP STATUS: RAMPING FROM %lf", &x) != 1)
//	        throw XInterface::XInterfaceError(
//				i18n("Cannot start ramping."), __FILE__, __LINE__);
	}
	else {
		if( !pause)
			return;
		interface()->query("PAUSE ON");
		if(interface()->scanf("%*2d:%*2d:%*2d PAUSE STATUS: %4s", buf) != 1)
	        throw XInterface::XConvError(__FILE__, __LINE__);
//		interface()->receive();
//		double x;
//		if(interface()->scanf("%*2d:%*2d:%*2d RAMP STATUS: HOLDING ON PAUSE AT %lf", &x) != 1)
//	        throw XInterface::XInterfaceError(
//				i18n("Cannot pause."), __FILE__, __LINE__);
	}
}
void
XCryogenicSMS::toPersistent() {
	XScopedLock<XInterface> lock( *interface());
	changePauseState(true);
	interface()->query("HEATER OFF");
	char buf[12];
	if(interface()->scanf("%*s HEATER STATUS: %10s", buf) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
}
void
XCryogenicSMS::toNonPersistent() {
	XScopedLock<XInterface> lock( *interface());
	changePauseState(true);
	interface()->query("HEATER ON");
	char buf[12];
	if(interface()->scanf("%*s HEATER STATUS: %10s", buf) != 1)
        throw XInterface::XInterfaceError(
			i18n("Cannot activate heater."), __FILE__, __LINE__);
}
void
XCryogenicSMS::toZero() {
	XScopedLock<XInterface> lock( *interface());
	interface()->send("RAMP ZERO");
//	interface()->query("RAMP ZERO");
//	char buf[4];
//	if(interface()->scanf("%*2d:%*2d:%*2d RAMP TARGET: %4s", buf) != 1)
//		throw XInterface::XConvError(__FILE__, __LINE__);
	changePauseState(false);
}
void
XCryogenicSMS::toSetPoint() {
	XScopedLock<XInterface> lock( *interface());
	interface()->send("RAMP MID");
//	interface()->query("RAMP MID");
//	char buf[4];
//	if(interface()->scanf("%*2d:%*2d:%*2d RAMP TARGET: %4s", buf) != 1)
//		throw XInterface::XConvError(__FILE__, __LINE__);
	changePauseState(false);
}

void
XCryogenicSMS::setPoint(double field) {
	XScopedLock<XInterface> lock( *interface());
	interface()->query("TESLA ON");
	char buf[10];
	if(interface()->scanf("%*s UNITS: %5s", buf) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	interface()->query("GET OUTPUT");
	double x;
	if(interface()->scanf("%*2d:%*2d:%*2d OUTPUT: %lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	if(x * field < 0.0) {
		if(fabs(x) > fieldResolution()) {
			throw XInterface::XInterfaceError(
				i18n("First you should set to zero."), __FILE__, __LINE__);
		}
		if(field < 0.0)
			interface()->queryf("SET DIRECTION -");
		else
			interface()->queryf("SET DIRECTION +");

		if(interface()->scanf("%*2d:%*2d:%*2d CURRENT DIRECTION: %10s", buf) != 1)
			throw XInterface::XConvError(__FILE__, __LINE__);
	}
	interface()->queryf("SET MID %.5f", fabs(field));
	if(interface()->scanf("%*2d:%*2d:%*2d MID SETTING: %lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
}
void
XCryogenicSMS::setRate(double hpm) {
	interface()->query("GET TPA");
	double tesla_per_amp;
	if(interface()->scanf("%*2d:%*2d:%*2d %*s FIELD CONSTANT: %lf", &tesla_per_amp) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	double amp_per_sec = hpm / 60.0 / tesla_per_amp;
	interface()->queryf("SET RAMP %.5g", amp_per_sec);
	double x;
	if(interface()->scanf("%*2d:%*2d:%*2d RAMP RATE: %lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
}
double
XCryogenicSMS::getTargetField() {
	XScopedLock<XInterface> lock( *interface());
	interface()->query("TESLA ON");
	char buf[10];
	if(interface()->scanf("%*s UNITS: %5s", buf) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	interface()->query("GET MID");
	double x;
	if(interface()->scanf("%*2d:%*2d:%*2d %*s MID SETTING: %lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}
double
XCryogenicSMS::getSweepRate() {
	XScopedLock<XInterface> lock( *interface());
	interface()->query("TESLA ON");
	char buf[10];
	if(interface()->scanf("%*s UNITS: %5s", buf) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	interface()->query("GET TPA");
	double tesla_per_amp;
	if(interface()->scanf("%*2d:%*2d:%*2d %*s FIELD CONSTANT: %lf", &tesla_per_amp) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	interface()->query("GET RATE");
	double x;
	if(interface()->scanf("%*2d:%*2d:%*2d %*s RAMP RATE: %lf", &x) != 1)  //[A/s]
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x * tesla_per_amp * 60.0;
}
double
XCryogenicSMS::getOutputField() {
	XScopedLock<XInterface> lock( *interface());
	interface()->query("TESLA ON");
	char buf[10];
	if(interface()->scanf("%*s UNITS: %5s", buf) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	interface()->query("GET OUTPUT");
	double x;
	if(interface()->scanf("%*2d:%*2d:%*2d OUTPUT: %lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}
double
XCryogenicSMS::getMagnetField() {
	if(isPCSHeaterOn())
		return getOutputField();
	else
		return getPersistentField();
}
double
XCryogenicSMS::getPersistentField() {
	XScopedLock<XInterface> lock( *interface());
	interface()->query("TESLA ON");
	char buf[10];
	if(interface()->scanf("%*s UNITS: %5s", buf) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	interface()->query(
		"GET PER\r\n" //"GET PER" does not return value+delimiter if PER hasn't been recorded.
		"GET OUTPUT"); //Dummy, workaround against the damn firmware.
	double x;
	if(interface()->scanf("%*2d:%*2d:%*2d %lf %s", &x, buf) != 2) {
		return 0.0;
	}
	else {
		if( !strncmp(buf, "TESLA", 5)) {
			interface()->receive(); //For output.
			double y;
			if(interface()->scanf("%*2d:%*2d:%*2d OUTPUT: %lf", &y) != 1)
				throw XInterface::XConvError(__FILE__, __LINE__);
			return x;
		}
	}
	return 0.0;
}
double
XCryogenicSMS::getOutputVolt() {
	interface()->query("GET OUTPUT");
	double x;
	if(interface()->scanf("%*2d:%*2d:%*2d OUTPUT: %*s %*s AT %lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}
double
XCryogenicSMS::getOutputCurrent() {
	XScopedLock<XInterface> lock( *interface());
	interface()->query("TESLA OFF");
	char buf[10];
	if(interface()->scanf("%*2d:%*2d:%*2d UNITS: %5s", buf) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	interface()->query("GET OUTPUT");
	double x;
	if(interface()->scanf("%*2d:%*2d:%*2d OUTPUT: %lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}

//! Persistent Current Switch Heater
//! please return *TRUE* if no PCS fitted
bool
XCryogenicSMS::isPCSHeaterOn() {
	interface()->query("HEATER");
	char buf[10];
	if(interface()->scanf("%*s HEATER STATUS: %5s", buf) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	if( !strncmp("ON", buf, 2))
		return true;
	return false;
}
//! please return false if no PCS fitted
bool
XCryogenicSMS::isPCSFitted() {
	return true;
}
