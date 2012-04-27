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
#include "userlockinamp.h"
#include "charinterface.h"

REGISTER_TYPE(XDriverList, SR830, "Stanford Research SR830 lock-in amp.");
REGISTER_TYPE(XDriverList, LI5640, "NF LI5640 lock-in amp.");
REGISTER_TYPE(XDriverList, AH2500A, "Andeen-Hagerling 2500A capacitance bridge");

XSR830::XSR830(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XLIA>(name, runtime, ref(tr_meas), meas)
    , m_cCount(10) {

	const char *tc[] = {"1e-5sec", "3e-5s", "1e-4s", "3e-4s", "1e-3s", "3e-3s", "1e-2s",
						"3e-2", "0.1s", "0.3s", "1s", "3s", "10s", "30s", "100s", "300s", "1000s",
						"3000s", "10000s", "30000s", ""};
	const char *sens[] = {"2nV/fA", "5nV/fA", "10nV/fA", "20nV/fA", "50nV/fA", "100nV/fA",
						  "200nV/fA", "500nV/fA", "1uV/pA", "2uV/pA", "5uV/pA", "10uV/pA", "20uV/pA",
						  "50uV/pA", "100uV/pA", "200uV/pA", "500uV/pA", "1mV/nA", "2mV/nA", "5mV/nA",
						  "10mV/nA", "20mV/nA", "50mV/nA", "100mV/nA", "200mV/nA", "500mV/nA", "1V/uA",
						  ""};
	for(Transaction tr( *this);; ++tr) {
		for(int i = 0; strlen(tc[i]) > 0; i++) {
			tr[ *timeConst()].add(tc[i]);
		}
		for(int i = 0; strlen(sens[i]) > 0; i++) {
			tr[ *sensitivity()].add(sens[i]);
		}
		if(tr.commit())
			break;
	}
	//    UseSerialPollOnWrite = false;
	interface()->setGPIBWaitBeforeWrite(20);
	interface()->setGPIBWaitBeforeRead(20);
	interface()->setGPIBWaitBeforeSPoll(10);
}
void
XSR830::get(double *cos, double *sin) {
	double sens = 0;
	int idx;
	Snapshot shot( *this);
	bool autoscale_x = shot[ *autoScaleX()];
	bool autoscale_y = shot[ *autoScaleY()];
	if(autoscale_x || autoscale_y) {
		interface()->query("SENS?");
		idx = interface()->toInt();
		sens =  1e-9 * pow(10.0, rint(idx / 3));
		switch(idx % 3) {
		case 0: sens *= 2; break;
		case 1: sens *= 5; break;
		case 2: sens *= 10; break;
		}
	}
	interface()->query("SNAP?1,2");
	if(interface()->scanf("%lf,%lf", cos, sin) != 2)
		throw XInterface::XConvError(__FILE__, __LINE__);
	if((autoscale_x && (sqrt( *cos * *cos) > sens * 0.8)) ||
	   (autoscale_y && (sqrt( *sin * *sin) > sens * 0.8)))
		trans( *sensitivity()) = idx + 1;
	if((autoscale_x && (sqrt( *cos * *cos) < sens * 0.2)) ||
	   (autoscale_y && (sqrt( *sin * *sin) < sens * 0.2))) {
		m_cCount--;
		if(m_cCount == 0) {
			trans( *sensitivity()) = idx - 1;
			m_cCount = 10;
		}
	}
	else
		m_cCount = 10;
}
void
XSR830::open() throw (XInterface::XInterfaceError &) {
	interface()->query("OFLT?");
	trans( *timeConst()) = interface()->toInt();
	interface()->query("SENS?");
	trans( *sensitivity()) = interface()->toInt();
	interface()->query("SLVL?");
	trans( *output()) = interface()->toDouble();
	interface()->query("FREQ?");
	trans( *frequency()) = interface()->toDouble();
      
	start();
}
void
XSR830::afterStop() {
	try {
		interface()->send("LOCL 0");
	}
	catch (XInterface::XInterfaceError &e) {
		e.print(getLabel());
	}
    close();
}
void
XSR830::changeOutput(double x) {
	interface()->sendf("SLVL %f", x);
}
void
XSR830::changeSensitivity(int x) {
	interface()->sendf("SENS %d", x);
}
void
XSR830::changeTimeConst(int x) {
	interface()->sendf("OFLT %d", x);
}
void
XSR830::changeFreq(double x) {
	interface()->sendf("FREQ %g", x);
}

XLI5640::XLI5640(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XLIA>(name, runtime, ref(tr_meas), meas),
      m_cCount(10) {

	interface()->setEOS("\r\n");

	const char *tc[] = {"1e-5sec", "3e-5s", "1e-4s", "3e-4s", "1e-3s", "3e-3s", "1e-2s",
						"3e-2", "0.1s", "0.3s", "1s", "3s", "10s", "30s", "100s", "300s", "1000s",
						"3000s", "10000s", "30000s", ""};
	const char *sens[] = {"2nV", "5nV/fA", "10nV/fA", "20nV/fA", "50nV/fA", "100nV/fA",
						  "200nV/fA", "500nV/fA", "1uV/pA", "2uV/pA", "5uV/pA", "10uV/pA", "20uV/pA",
						  "50uV/pA", "100uV/pA", "200uV/pA", "500uV/pA", "1mV/nA", "2mV/nA", "5mV/nA",
						  "10mV/nA", "20mV/nA", "50mV/nA", "100mV/nA", "200mV/nA", "500mV/nA", "1V/uA",
						  ""};
	for(Transaction tr( *this);; ++tr) {
		for(int i = 0; strlen(tc[i]) > 0; i++) {
			tr[ *timeConst()].add(tc[i]);
		}
		for(int i = 0; strlen(sens[i]) > 0; i++) {
			tr[ *sensitivity()].add(sens[i]);
		}
		if(tr.commit())
			break;
	}
//	interface()->setGPIBUseSerialPollOnWrite(false);
//	interface()->setGPIBWaitBeforeWrite(20);
//	interface()->setGPIBWaitBeforeRead(20);
	interface()->setGPIBWaitBeforeSPoll(5);
}
void
XLI5640::get(double *cos, double *sin) {
	double sens = 0;
	int sidx;
	Snapshot shot( *this);
	bool autoscale_x = shot[ *autoScaleX()];
	bool autoscale_y = shot[ *autoScaleY()];
	interface()->query("DOUT?");
	if(interface()->scanf("%lf,%lf,%d", cos, sin, &sidx) != 3)
		throw XInterface::XConvError(__FILE__, __LINE__);

	if(autoscale_x || autoscale_y) {
		sens =  1e-9 * pow(10.0, rint(sidx / 3));
		switch(sidx % 3) {
		case 0: sens *= 2; break;
		case 1: sens *= 5; break;
		case 2: sens *= 10; break;
		}
	}
	if((autoscale_x && (sqrt( *cos * *cos) > sens * 0.8)) ||
	   (autoscale_y && (sqrt( *sin * *sin) > sens * 0.8)))
		trans( *sensitivity()) = sidx + 1;
	if((autoscale_x && (sqrt( *cos * *cos) < sens * 0.2)) ||
	   (autoscale_y && (sqrt( *sin * *sin) < sens * 0.2))) {
		m_cCount--;
		if(m_cCount == 0) {
			trans( *sensitivity()) = sidx - 1;
			m_cCount = 10;
		}
	}
	else
		m_cCount = 10;
}
void
XLI5640::open() throw (XInterface::XInterfaceError &) {
	interface()->query("TCON?");
	trans( *timeConst()) = interface()->toInt();
	interface()->query("AMPL?");
	double x;
	interface()->scanf("%lf", &x);
	trans( *output()) = x;
	interface()->query("FREQ?");
	trans( *frequency()) = interface()->toDouble();

	interface()->query("ISRC?");
	int src = interface()->toInt();
	m_currMode = (src >= 2);
	if(m_currMode)
		interface()->query("VSEN?");
	else
		interface()->query("ISEN?");
	trans( *sensitivity()) = interface()->toInt();

	interface()->send("DDEF 1,0"); //DATA1=x
	interface()->send("DDEF 2,0"); //DATA2=y
	interface()->send("OTYP 1,2,4"); //DATA1,DATA2,SENSITIVITY

	start();
}
void
XLI5640::afterStop() {

    close();
}
void
XLI5640::changeOutput(double x) {
	int range = 2;
	if(x < 0.5)
		range = 1;
	if(x < 0.05)
		range = 0;
	interface()->sendf("AMPL %f,%d", x, range);
}
void
XLI5640::changeSensitivity(int x) {
	interface()->sendf("VSEN %d", x);
}
void
XLI5640::changeTimeConst(int x) {
	interface()->sendf("TCON %d", x);
}
void
XLI5640::changeFreq(double x) {
	interface()->sendf("FREQ %g", x);
}

XAH2500A::XAH2500A(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XLIA>(name, runtime, ref(tr_meas), meas) {
	const char *tc[] = {"0.04s", "0.08s", "0.14s", "0.25s", "0.5s",
						"1s", "2s", "4s", "8s", "15s", "30s", "60s",
						"120s", "250s", "500s", "1000s", ""};
	const char *sens[] = {""};
	for(Transaction tr( *this);; ++tr) {
		for(int i = 0; strlen(tc[i]) > 0; i++) {
			tr[ *timeConst()].add(tc[i]);
		}
		for(int i = 0; strlen(sens[i]) > 0; i++) {
			tr[ *sensitivity()].add(sens[i]);
		}
		tr[ *fetchFreq()] = 10;

		tr[ *autoScaleX()].disable();
		tr[ *autoScaleY()].disable();
		tr[ *sensitivity()].disable();
		tr[ *frequency()].disable();
		if(tr.commit())
			break;
	}
	interface()->setGPIBUseSerialPollOnWrite(false);
	interface()->setGPIBWaitBeforeWrite(20);
	interface()->setGPIBWaitBeforeRead(20);
	interface()->setGPIBWaitBeforeSPoll(20);
	interface()->setGPIBMAVbit(0x80);
}
void
XAH2500A::get(double *cap, double *loss) {
    interface()->query("SI");
    if(interface()->scanf("C=%lf %*s L=%lf", cap, loss) != 2)
		throw XInterface::XConvError(__FILE__, __LINE__);
}
void
XAH2500A::open() throw (XInterface::XInterfaceError &) {
	interface()->query("SH AV");
	int d;
	if(interface()->scanf("%*s AVEREXP=%d", &d) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	trans( *timeConst()) = d;
	interface()->query("SH V");
	double f;
	if(interface()->scanf("%*s HIGHEST=%lf", &f) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	trans( *output()) = f;

	interface()->send("NREM");

	start();
}
void
XAH2500A::afterStop() {
	try {
		interface()->send("LOC");
	}
	catch (XInterface::XInterfaceError &e) {
		e.print(getLabel());
	}
	close();
}
void
XAH2500A::changeOutput(double x) {
	interface()->sendf("V %f", x);
}
void
XAH2500A::changeTimeConst(int x) {
	interface()->sendf("AV %d", x);
}
void
XAH2500A::changeSensitivity(int ) {
    throw XInterface::XInterfaceError(i18n("Operation not supported."), __FILE__, __LINE__);
}
void
XAH2500A::changeFreq(double ) {
    throw XInterface::XInterfaceError(i18n("Operation not supported."), __FILE__, __LINE__);
}
