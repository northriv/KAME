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
 * Notes not mentioned in the manufacturer's manual for ver 6.
 * GET PER command does not return a value or delimiter when it is not in persistent mode or at zero field.
 * RAMP/DIRECTION ... command does not reply.
 * PAUSE ... command does not reply the second line.
 * Some commands respond with a form of HH:MM:SS ....... (command).
 * Local button operations will emit status lines.
 *
 * This driver assumes...
 * (i) TPA (tesla per ampere) has been set properly.
 * (ii) PCSH is fitted, or "HEATER OUTPUT" is set to zero.
 */
    interface()->setEOS("\r\n");
}

std::string
XCryogenicSMS::receiveMessage(const char *title, bool is_stamp_required) {
	for(;;) {
		interface()->receive();
		bool has_stamp = false;
		if(strncmp( &interface()->buffer()[0], "........", 8)) {
			if( !strncmp( &interface()->buffer()[0], "------->", 8)) {
				//Error message is passed.
		        throw XInterface::XInterfaceError( &interface()->buffer()[8], __FILE__, __LINE__);
			}
			//Message w/ time stamp.
			int ss;
			if(sscanf( &interface()->buffer()[0], "%*2d:%*2d:%2d", &ss) != 1)
				throw XInterface::XConvError(__FILE__, __LINE__);
			has_stamp = true;
		}
		auto cl_pos = std::find(interface()->buffer().begin() + 8, interface()->buffer().end(), ':');
		if(cl_pos == interface()->buffer().end())
			throw XInterface::XConvError(__FILE__, __LINE__);
		int cnt = cl_pos - interface()->buffer().begin();
		if(cnt < 10)
			throw XInterface::XConvError(__FILE__, __LINE__);
		if( !strncmp( &interface()->buffer()[9], title, strlen(title))) {
			if(is_stamp_required && !has_stamp)
				throw XInterface::XConvError(__FILE__, __LINE__);
			cl_pos++; //skipping colon.
			while(cl_pos != interface()->buffer().end()) {
				if( *cl_pos != ' ')
					return &*cl_pos;
				cl_pos++; //skipping white space.
			}
			throw XInterface::XConvError(__FILE__, __LINE__);
		}
	}
}

void
XCryogenicSMS::open() throw (XInterface::XInterfaceError &) {
	interface()->send("SET TPA");
	if(sscanf(receiveMessage("FIELD CONSTANT").c_str(), "%lf", &m_tpa) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	//Reads again to flush buffer.
	interface()->send("SET TPA");
	if(sscanf(receiveMessage("FIELD CONSTANT").c_str(), "%lf", &m_tpa) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	start();
}
void
XCryogenicSMS::changePauseState(bool pause) {
// Lock before calling me.
//	XScopedLock<XInterface> lock( *interface());
	interface()->send("PAUSE");
	char buf[10];
	if(sscanf(receiveMessage("PAUSE STATUS").c_str(), "%4s", buf) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
	if( !strncmp("ON", buf, 2)) {
		if(pause)
			return;
		interface()->send("PAUSE OFF");
		receiveMessage("PAUSE STATUS", true);
	}
	else {
		if( !pause)
			return;
		interface()->send("PAUSE ON");
		receiveMessage("PAUSE STATUS", true);
	}
}
void
XCryogenicSMS::toPersistent() {
	XScopedLock<XInterface> lock( *interface());
	changePauseState(true);
	interface()->send("HEATER OFF");
	receiveMessage("HEATER STATUS");

	setRate(10.0); //Setting very high rate.
}
void
XCryogenicSMS::toNonPersistent() {
	XScopedLock<XInterface> lock( *interface());
	setRate(Snapshot( *this)[ *sweepRate()]);
	changePauseState(true);
	interface()->send("HEATER ON");
	receiveMessage("HEATER STATUS");
}
void
XCryogenicSMS::ramp(const char *str) {
	interface()->sendf("RAMP %s", str); //"RAMP..." does not respond for firmware > 6.
}
void
XCryogenicSMS::toZero() {
	XScopedLock<XInterface> lock( *interface());
	ramp("ZERO");
	changePauseState(false);
}
void
XCryogenicSMS::toSetPoint() {
	XScopedLock<XInterface> lock( *interface());
	ramp("MID");
	changePauseState(false);
}
void
XCryogenicSMS::changePolarity(int p) {
	for(int tcnt = 0; tcnt < 3; ++tcnt) {
		interface()->sendf(
			"DIRECTION %c\r\n" //"DIR..." does not respond for firmware > 6.
			"GET OUTPUT" //Dummy, workaround against the damn firmware.
			, (p > 0) ? '+' : '-');
		char c;
		if(sscanf(receiveMessage("OUTPUT", true).c_str(), "%c", &c) != 1)
			throw XInterface::XConvError(__FILE__, __LINE__);
		int x = (c != '-') ? 1 : -1;
		if(x * p > 0)
			return;
	}
	throw XInterface::XInterfaceError(i18n("Failed to reverse current direction."), __FILE__, __LINE__);
}
void
XCryogenicSMS::setPoint(double field) {
	XScopedLock<XInterface> lock( *interface());
	double x = getOutputField();

	if(fabs(x) < fieldResolution() * 10) {
		if(field < 0.0) {
			changePolarity(-1);
		}
		if(field > 0.0) {
			if( !isOutputPositive())
				changePolarity(+1);
		}
	}
	else if(x * field < 0) {
		throw XInterface::XInterfaceError(i18n("Failed to reverse current direction."), __FILE__, __LINE__);
	}

	interface()->sendf("SET MID %.5f", fabs(field));
	if(sscanf(receiveMessage("MID SETTING", true).c_str(), "%lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
}
void
XCryogenicSMS::setRate(double hpm) {
	XScopedLock<XInterface> lock( *interface());
	double amp_per_sec = hpm / 60.0 / teslaPerAmp();
	interface()->sendf("SET RAMP %.5g", amp_per_sec);
	double x;
	if(sscanf(receiveMessage("RAMP RATE", true).c_str(), "%lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
}
bool
XCryogenicSMS::isOutputPositive() {
	XScopedLock<XInterface> lock( *interface());
	interface()->send("GET OUTPUT");
	char c;
	if(sscanf(receiveMessage("OUTPUT", true).c_str(), "%c", &c) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return (c != '-');
}
double
XCryogenicSMS::getTargetField() {
	XScopedLock<XInterface> lock( *interface());
	interface()->send("SET MID");
	double x;
	if(sscanf(receiveMessage("MID SETTING").c_str(), "%lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x * (isOutputPositive() ? 1 : -1);
}
double
XCryogenicSMS::getSweepRate() {
	XScopedLock<XInterface> lock( *interface());
	interface()->send("TESLA ON");
	receiveMessage("UNITS");

	double x;
	interface()->send("SET RATE");
	if(sscanf(receiveMessage("RAMP RATE").c_str(), "%lf", &x) != 1)  //[A/s]
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x * teslaPerAmp() * 60.0;
}
double
XCryogenicSMS::getOutputField() {
	XScopedLock<XInterface> lock( *interface());
	interface()->send("TESLA ON");
	receiveMessage("UNITS");

	interface()->send("GET OUTPUT");
	double x;
	if(sscanf(receiveMessage("OUTPUT", true).c_str(), "%lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}
double
XCryogenicSMS::getPersistentField() {
	XScopedLock<XInterface> lock( *interface());
	interface()->send("TESLA ON");
	receiveMessage("UNITS");

	interface()->send("HEATER");
	std::string buf = receiveMessage("HEATER STATUS");
	if( !strncmp("ON", buf.c_str(), 2))
		throw XInterface::XInterfaceError(i18n("Trying to read persistent current while PCSH is on."), __FILE__, __LINE__);
	if( !strncmp("OFF", buf.c_str(), 3))
		return 0.0;
	double x;
	if(sscanf(buf.c_str(), "SWITCHED OFF AT %lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}
double
XCryogenicSMS::getOutputVolt() {
	XScopedLock<XInterface> lock( *interface());
	interface()->send("GET OUTPUT");
	double x;
	if(sscanf(receiveMessage("OUTPUT", true).c_str(), "%*s %*s AT %lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return x;
}
double
XCryogenicSMS::getOutputCurrent() {
	XScopedLock<XInterface> lock( *interface());
	double x = getOutputField();

	return lrint(x / fieldResolution() * 100) * fieldResolution() / 100.0 / teslaPerAmp();
}
double
XCryogenicSMS::fieldResolution() {
	return std::min(0.05, 0.04 * teslaPerAmp());
}
//! Persistent Current Switch Heater
bool
XCryogenicSMS::isPCSHeaterOn() {
	XScopedLock<XInterface> lock( *interface());
	interface()->send("HEATER");
	char buf[10];
	if(sscanf(receiveMessage("HEATER STATUS").c_str(), "%5s", buf) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	if( !strncmp("ON", buf, 2))
		return true;
	return false;
}
//! please return false if no PCS fitted
bool
XCryogenicSMS::isPCSFitted() {
	XScopedLock<XInterface> lock( *interface());
	interface()->send("SET HEATER"); //queries heater power setting by Volts.
	double x;
	if(sscanf(receiveMessage("HEATER OUTPUT").c_str(), "%lf", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	return (x > 0.01);
}
