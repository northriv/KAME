/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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
REGISTER_TYPE(XDriverList, LakeshoreM81LIA, "Lakeshore M81-SSM lock-in amp. module");
REGISTER_TYPE(XDriverList, SignalRecovery7265, "Signal Recovery/EG&G Model7265 lock-in amp.");
REGISTER_TYPE(XDriverList, LI5640, "NF LI5640 lock-in amp.");
REGISTER_TYPE(XDriverList, HP4284A, "Agilent/HP4284A Precision LCR Meter");
REGISTER_TYPE(XDriverList, AH2500A, "Andeen-Hagerling 2500A capacitance bridge");

XMutex XLakeshoreM81LIA::s_mutex;

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
	iterate_commit([=](Transaction &tr){
		for(int i = 0; strlen(tc[i]) > 0; i++) {
			tr[ *timeConst()].add(tc[i]);
		}
		for(int i = 0; strlen(sens[i]) > 0; i++) {
			tr[ *sensitivity()].add(sens[i]);
		}
    });
	//    UseSerialPollOnWrite = false;
	interface()->setGPIBWaitBeforeWrite(20);
	interface()->setGPIBWaitBeforeRead(20);
	interface()->setGPIBWaitBeforeSPoll(10);
}
void
XSR830::get(double *cos, double *sin) {
    XScopedLock<XInterface> lock( *interface());
    double sens = 0;
	int idx;
	bool ovld = false;
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
		interface()->query("LIAS?");
		ovld = (interface()->toInt() & 1);
	}
	interface()->query("SNAP?1,2");
	if(interface()->scanf("%lf,%lf", cos, sin) != 2)
		throw XInterface::XConvError(__FILE__, __LINE__);
    if(ovld || ((autoscale_x ? fabs( *cos) : 0) + (autoscale_y ? fabs( *sin) : 0) > sens * 0.9))
		trans( *sensitivity()) = idx + 1;
    if(autoscale_x || autoscale_y) {
        if((autoscale_x ? fabs( *cos) : 0) + (autoscale_y ? fabs( *sin) : 0) < sens * 0.15) {
            m_cCount--;
            if(m_cCount == 0) {
                trans( *sensitivity()) = idx - 1;
                m_cCount = 10;
            }
        }
	}
	else
		m_cCount = 10;
}
void
XSR830::open() {
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
XSR830::closeInterface() {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened())
		return;
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

XLakeshoreM81LIA::XLakeshoreM81LIA(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XLIA>(name, runtime, ref(tr_meas), meas) {

    const char *tc[] = {"1e-5sec", "3e-5s", "1e-4s", "3e-4s", "1e-3s", "3e-3s", "1e-2s",
                        "3e-2", "0.1s", "0.3s", "1s", "3s", "10s", "30s", "100s", "300s", "1000s",
                        "3000s", "10000s", "30000s", ""};
    const char *sens[] = {"2nV/fA", "5nV/fA", "10nV/fA", "20nV/fA", "50nV/fA", "100nV/fA",
                          "200nV/fA", "500nV/fA", "1uV/pA", "2uV/pA", "5uV/pA", "10uV/pA", "20uV/pA",
                          "50uV/pA", "100uV/pA", "200uV/pA", "500uV/pA", "1mV/nA", "2mV/nA", "5mV/nA",
                          "10mV/nA", "20mV/nA", "50mV/nA", "100mV/nA", "200mV/nA", "500mV/nA", "1V/uA",
                          ""};
    iterate_commit([=](Transaction &tr){
        for(int i = 0; strlen(tc[i]) > 0; i++) {
            tr[ *timeConst()].add(tc[i]);
        }
        for(int i = 0; strlen(sens[i]) > 0; i++) {
            tr[ *sensitivity()].add(sens[i]);
        }
    });
    autoScaleY()->disable();
    output()->disable();
    interface()->setGPIBUseSerialPollOnWrite(false);
    interface()->setGPIBUseSerialPollOnRead(false);
    interface()->setGPIBWaitBeforeWrite(40);
    //    ExclusiveWaitAfterWrite = 10;
    interface()->setSerialEOS("\r\n");
    interface()->setGPIBWaitBeforeRead(40);
    interface()->setSerialBaudRate(921600);
    interface()->setSerial7Bits(false);
    interface()->setSerialParity(XCharInterface::PARITY_NONE);

}
int
XLakeshoreM81LIA::channel() {
    //todo fix this bad hack.
    int ch;
    if(sscanf(getName().substr(getName().length() - 1).c_str(), "%i", &ch) != 1)
        throw XInterface::XInterfaceError("Cannot figure out module number from driver name.", __FILE__, __LINE__);
    return ch;
}
void
XLakeshoreM81LIA::get(double *cos, double *sin) {
    XScopedLock<XMutex> lock(s_mutex);
    int ch = channel();
//    interface()->queryf("FETCH:SENS%i:LIA:X?;Y?", ch);
    interface()->queryf("FETCH? MX,%i,MY,%i", ch, ch);
    if(interface()->scanf("%lf,%lf", cos, sin) != 2)
        throw XInterface::XConvError(__FILE__, __LINE__);
}
void
XLakeshoreM81LIA::open() {
    XScopedLock<XMutex> lock(s_mutex);
    int ch = channel();
//    interface()->queryf("SENS%i:VOLT:RANG:AUTO?", ch);
//    trans( *autoScaleX()) = interface()->toInt();
    interface()->queryf("SENS%i:LIA:TIME?", ch);
    double tc = interface()->toDouble();
    trans( *timeConst()) = (log10(tc) + 5) * 2;

    interface()->queryf("SENS%i:LIA:RSOURCE?", ch);
    int sch;
    if(interface()->scanf("S%i", &sch) == 1) {
        interface()->queryf("SOUR%i:FREQ?", sch);
        trans( *frequency()) = interface()->toDouble();
    }
    start();
}
void
XLakeshoreM81LIA::closeInterface() {
    XScopedLock<XInterface> lock( *interface());
    if( !interface()->isOpened())
        return;
    try {
        interface()->send("LOCL 0");
    }
    catch (XInterface::XInterfaceError &e) {
        e.print(getLabel());
    }
    close();
}
void
XLakeshoreM81LIA::changeOutput(double x) {
}
void
XLakeshoreM81LIA::changeSensitivity(int x) {
}
void
XLakeshoreM81LIA::changeTimeConst(int x) {
    XScopedLock<XMutex> lock(s_mutex);
    double tc = pow(10, x * 0.5 - 5);
    int ch = channel();
    interface()->sendf("SENS%i:LIA:TIME %f", ch, tc);
}
void
XLakeshoreM81LIA::changeFreq(double x) {
    XScopedLock<XMutex> lock(s_mutex);
    int ch = channel();
    interface()->queryf("SENS%i:LIA:RSOURCE?", ch);
    int sch;
    if(interface()->scanf("S%i", &sch) == 1) {
        interface()->sendf("SOUR%i:FREQ %f", sch, x);
    }
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
	iterate_commit([=](Transaction &tr){
		for(int i = 0; strlen(tc[i]) > 0; i++) {
			tr[ *timeConst()].add(tc[i]);
		}
		for(int i = 0; strlen(sens[i]) > 0; i++) {
			tr[ *sensitivity()].add(sens[i]);
		}
    });
//	interface()->setGPIBUseSerialPollOnWrite(false);
//	interface()->setGPIBWaitBeforeWrite(20);
//	interface()->setGPIBWaitBeforeRead(20);
	interface()->setGPIBWaitBeforeSPoll(5);
}
void
XLI5640::get(double *cos, double *sin) {
    XScopedLock<XInterface> lock( *interface());
    double sens = 0;
	int overlevel;
	int sidx;
	Snapshot shot( *this);
	bool autoscale_x = shot[ *autoScaleX()];
	bool autoscale_y = shot[ *autoScaleY()];
	interface()->query("DOUT?");
	if(interface()->scanf("%lf,%lf,%d, %d", cos, sin, &sidx, &overlevel) != 4)
		throw XInterface::XConvError(__FILE__, __LINE__);

	if(autoscale_x || autoscale_y) {
		sens =  1e-9 * pow(10.0, rint(sidx / 3));
		switch(sidx % 3) {
		case 0: sens *= 2; break;
		case 1: sens *= 5; break;
		case 2: sens *= 10; break;
		}
	}
    if(((autoscale_x ? fabs( *cos) : 0) + (autoscale_y ? fabs( *sin) : 0) > sens * 0.9) ||
	   overlevel)
		trans( *sensitivity()) = sidx + 1;
    if(autoscale_x || autoscale_y) {
        if((autoscale_x ? fabs( *cos) : 0) + (autoscale_y ? fabs( *sin) : 0) < sens * 0.15) {
            m_cCount--;
            if(m_cCount == 0) {
                trans( *sensitivity()) = sidx - 1;
                m_cCount = 10;
            }
        }
    }
	else
		m_cCount = 10;
}
void
XLI5640::open() {
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
	interface()->send("OTYP 1,2,4,5"); //DATA1,DATA2,SENSITIVITY,OVERLEVEL

	start();
}
void
XLI5640::closeInterface() {

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

XSignalRecovery7265::XSignalRecovery7265(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XLIA>(name, runtime, ref(tr_meas), meas)  {

    const char *tc[] = {"10us", "20us", "40us", "80us", "160us", "320us", "640us",
                        "5ms", "10ms", "20ms", "50ms", "100ms", "200ms", "500ms",
                        "1s", "2s", "5s", "10s", "20s", "50s", "100s", "200s", "500s",
                        "1ks", "2ks", "5ks", "10ks", "20ks", "50ks", "100ks", ""};
    const char *sens[] = {"2nV/fA", "5nV/fA", "10nV/fA", "20nV/fA", "50nV/fA", "100nV/fA",
                          "200nV/fA", "500nV/fA", "1uV/pA", "2uV/pA", "5uV/pA", "10uV/pA", "20uV/pA",
                          "50uV/pA", "100uV/pA", "200uV/pA", "500uV/pA", "1mV/nA", "2mV/nA", "5mV/nA",
                          "10mV/nA", "20mV/nA", "50mV/nA", "100mV/nA", "200mV/nA", "500mV/nA", "1V/uA",
                          ""};
    iterate_commit([=](Transaction &tr){
        for(int i = 0; strlen(tc[i]) > 0; i++) {
            tr[ *timeConst()].add(tc[i]);
        }
        for(int i = 0; strlen(sens[i]) > 0; i++) {
            tr[ *sensitivity()].add(sens[i]);
        }
    });
    autoScaleX()->disable();
    autoScaleY()->disable();
    interface()->setEOS("\r\n");
    interface()->setSerialBaudRate(9600);
    interface()->setSerialStopBits(1);
    interface()->setSerialParity(XCharInterface::PARITY_EVEN);
    interface()->setSerial7Bits(true);
    interface()->setGPIBMAVbit(0x80);
}
void
XSignalRecovery7265::get(double *cos, double *sin) {
    XScopedLock<XInterface> lock( *interface());
    interface()->query("X.");
    *cos = interface()->toDouble();
    interface()->query("Y.");
    *sin = interface()->toDouble();
}
void
XSignalRecovery7265::open() {
    interface()->query("TC");
    trans( *timeConst()) = interface()->toInt();
    interface()->query("SEN");
    trans( *sensitivity()) = interface()->toInt() - 1;
    interface()->query("OA.");
    trans( *output()) = interface()->toDouble();
    interface()->query("OF.");
    trans( *frequency()) = interface()->toDouble();

    start();
}
void
XSignalRecovery7265::closeInterface() {
    XScopedLock<XInterface> lock( *interface());
    close();
}
void
XSignalRecovery7265::changeOutput(double x) {
    interface()->sendf("OA. %.6g", x);
}
void
XSignalRecovery7265::changeSensitivity(int x) {
    interface()->sendf("SEN %d", x + 1);
}
void
XSignalRecovery7265::changeTimeConst(int x) {
    interface()->sendf("TC %d", x);
}
void
XSignalRecovery7265::changeFreq(double x) {
    interface()->sendf("OF. %.6g", x);
}


XHP4284A::XHP4284A(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XLIA>(name, runtime, ref(tr_meas), meas) {
    iterate_commit([=](Transaction &tr){
        for(auto &&s: {"SHORT", "MED", "LONG"}) {
            tr[ *timeConst()].add(s);
        }
        for(auto &&s: {"1", "10", "30", "100", "1000", "3000", "10000", "30000", "100000"}) {
            tr[ *sensitivity()].add(s);
        }
        tr[ *output()].disable();
        tr[ *fetchFreq()] = 0;
        tr[ *fetchFreq()].disable();
        tr[ *autoScaleX()].disable();
        tr[ *autoScaleY()].disable();
    });
}
void
XHP4284A::get(double *x, double *y) {
    interface()->query("TRIG:IMM;:FETCH?");
    int status, binno;
    if(interface()->scanf("%lf,%lf,%d,%d", x, y, &status, &binno) != 4)
        throw XInterface::XConvError(__FILE__, __LINE__);
}
void
XHP4284A::open() {
    interface()->send("*RST;*CLS");
    interface()->send("FORM ASCII");
    interface()->send("TRIG:SOUR BUS");
    interface()->send("COMP ON");
    interface()->send("INIT:CONT ON");

    interface()->query("APER?");
    auto s = interface()->toStrSimplified();
    s = s.substr(0, s.find(','));
    trans( *timeConst()) = s;

    interface()->query("FUNC:IMP:RANGE?");
    trans( *sensitivity()) = interface()->toStrSimplified();

    interface()->query("FREQ?");
    trans( *frequency()) = interface()->toDouble();

    start();
}
void
XHP4284A::closeInterface() {
    XScopedLock<XInterface> lock( *interface());
    if( !interface()->isOpened())
        return;
    try {
        interface()->send("*RST;*CLS");
    }
    catch (XInterface::XInterfaceError &e) {
        e.print(getLabel());
    }
    close();
}
void
XHP4284A::changeOutput(double x) {
    interface()->sendf("VOLT %g", x);
}
void
XHP4284A::changeTimeConst(int x) {
    interface()->send(("APER " + Snapshot(*this)[ *timeConst()].to_str()).c_str());
}
void
XHP4284A::changeSensitivity(int ) {
    double x = atof(Snapshot(*this)[ *sensitivity()].to_str().c_str());
    interface()->sendf("FUNC:IMP:RANGE %g", x);
}
void
XHP4284A::changeFreq(double x) {
    interface()->sendf("FREQ %g", x);
}

XAH2500A::XAH2500A(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XLIA>(name, runtime, ref(tr_meas), meas) {
	const char *tc[] = {"0.04s", "0.08s", "0.14s", "0.25s", "0.5s",
						"1s", "2s", "4s", "8s", "15s", "30s", "60s",
						"120s", "250s", "500s", "1000s", ""};
	const char *sens[] = {""};
	iterate_commit([=](Transaction &tr){
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
    });
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
XAH2500A::open() {
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
XAH2500A::closeInterface() {
    XScopedLock<XInterface> lock( *interface());
    if( !interface()->isOpened())
        return;
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
