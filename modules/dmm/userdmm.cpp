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
//---------------------------------------------------------------------------
//#include "pythondriver.h"

#include "userdmm.h"
#include "charinterface.h"

//#ifdef USE_PYBIND11
//static auto [pynode, pypoad] =
//    XPython::export_xdriver<XDMM, XPrimaryDriver>();
//#endif

REGISTER_TYPE(XDriverList, KE2000, "Keithley 2000/2001 DMM");
REGISTER_TYPE(XDriverList, KE2182, "Keithley 2182 nanovolt meter");
REGISTER_TYPE(XDriverList, KE2700_7700, "Keithley 2700 w/ 7700");
REGISTER_TYPE(XDriverList, KE6482, "Keithley 6482 picoam meter");
REGISTER_TYPE(XDriverList, HP34420A, "Agilent 34420A nanovolt meter");
REGISTER_TYPE(XDriverList, HP3458A, "Agilent 3458A DMM");
REGISTER_TYPE(XDriverList, HP3478A, "Agilent 3478A DMM");
REGISTER_TYPE(XDriverList, SanwaPC500, "SANWA PC500/510/520M DMM");
REGISTER_TYPE(XDriverList, SanwaPC5000, "SANWA PC5000 DMM");

void
XDMMSCPI::changeFunction() {
    XString func = ( **function())->to_str();
    if( !func.empty())
        interface()->sendf(":CONF:%s", func.c_str());
}
double
XDMMSCPI::fetch() {
    interface()->query(":FETC?");
    return interface()->toDouble();
}
double
XDMMSCPI::oneShotRead() {
    interface()->query(":READ?");
    return interface()->toDouble();
}
std::deque<double>
XDMMSCPI::oneShotMultiRead() {
    std::deque<double> var;
    for(unsigned int i = 0; i < maxNumOfChannels(); ++i) {
        interface()->sendf(":SENS:CHAN %u", i + 1);
        interface()->query(":READ?");
        var.push_back(interface()->toDouble());
    }
    return var;
}
/*
double
XDMMSCPI::measure(const XString &func)
{
    interface()->queryf(":MEAS:%s?", func.c_str());
    return interface()->toDouble();
}
*/


XKE2700_7700::XKE2700_7700(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XDMM>(name, runtime, ref(tr_meas), meas, 10) {
    function()->disable();
}
void XKE2700_7700::open() {
    XCharDeviceDriver<XDMM>::open();
    interface()->send("TRAC:CLE"); //Clears buffer.
    interface()->send("INIT:CONT OFF");
    interface()->send("TRIG:SOUR IMM"); //Immediate trigger.
    interface()->send("TRIG:COUN 1"); //1 scan.
}
std::deque<double>
XKE2700_7700::oneShotMultiRead() {
    std::deque<double> var;
    for(unsigned int i = 0; i < maxNumOfChannels(); ++i) {
        interface()->sendf("ROUT:CLOS (@1%1d%1d)", (i + 1) / 10, (i + 1) % 10);
        interface()->query("READ?");
        var.push_back(interface()->toDouble());
    }
    return var;
}

void
XKE6482::changeFunction() {
    XString func = ( **function())->to_str();
    if( !func.empty()) {
        interface()->send(":CONF:CURR");
        interface()->sendf(":FORM:ELEM %s", func.c_str());
    }
}

XHP3458A::XHP3458A(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XDMM>(name, runtime, ref(tr_meas), meas) {
	interface()->setGPIBMAVbit(0x80);
	interface()->setGPIBUseSerialPollOnWrite(false);
	iterate_commit([=](Transaction &tr){
        const char *funcs[] = {
            "DCV", "ACV", "ACDCV", "OHM", "OHMF", "DCI", "ACI", "ACDCI", "FREQ", "PER", "DSAC", "DSDC", "SSAC", "SSDC", ""
        };
        for(const char **func = funcs; strlen( *func); func++) {
			tr[ *function()].add( *func);
		}
    });
}
void
XHP3458A::changeFunction() {
    XString func = ( **function())->to_str();
    if( !func.empty())
        interface()->sendf("FUNC %s;ARANGE ON", func.c_str());
}
double
XHP3458A::fetch() {
    interface()->receive();
    return interface()->toDouble();
}
double
XHP3458A::oneShotRead() {
    interface()->query("END ALWAYS;OFORMAT ASCII;QFORMAT NUM;NRDGS 1;TRIG AUTO;TARM SGL");
    return interface()->toDouble();
}


XHP3478A::XHP3478A(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XDMM>(name, runtime, ref(tr_meas), meas) {
	interface()->setGPIBUseSerialPollOnWrite(false);
	interface()->setGPIBMAVbit(0x01);
//	setEOS("\r\n");
	iterate_commit([=](Transaction &tr){
        const char *funcs[] = {
            "DCV", "ACV", "OHM", "OHMF", "DCI", "ACI", ""
        };
        for(const char **func = funcs; strlen( *func); func++) {
			tr[ *function()].add( *func);
		}
    });
}
void
XHP3478A::changeFunction() {
    int func = ***function();
    if(func < 0)
		return;
//    		throw XInterface::XInterfaceError(i18n("Select function!"), __FILE__, __LINE__);
    interface()->sendf("F%dRAZ1", func + 1);
}
double
XHP3478A::fetch() {
    interface()->receive();
    return interface()->toDouble();
}
double
XHP3478A::oneShotRead() {
    interface()->query("T3");
    return interface()->toDouble();
}

XSanwaPC500::XSanwaPC500(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XDMM>(name, runtime, ref(tr_meas), meas) {
	interface()->setSerialBaudRate(9600);
	interface()->setSerialStopBits(2);
	
	iterate_commit([=](Transaction &tr){
        const char *funcs[] = {
            "AcV", "DcV", "Ac+DcV", "Cx", "Dx", "Dx", "TC", "TC", "TF", "Ohm",
            "Conti", "AcA", "DcA", "Ac+DcA", "Hz", "Duty%", "%mA", "dB", "?", ""
        };
        for(const char **func = funcs; strlen( *func); func++) {
			tr[ *function()].add( *func);
		}
		tr[ *function()].str(XString("?"));
    });
}
void
XSanwaPC500::changeFunction() {
}
double
XSanwaPC500::fetch() {
	msecsleep(200);
	requestData();
	interface()->receive(8);
	if((interface()->buffer()[0] != 0x10) ||
		(interface()->buffer()[1] != 0x02))
		throw XInterface::XInterfaceError(i18n("Format Error!"), __FILE__, __LINE__);
	if((interface()->buffer()[6] != 0x00) ||
		(interface()->buffer()[7] != 0x00))
		throw XInterface::XInterfaceError(i18n("Format Error!"), __FILE__, __LINE__);
	const int funcs[] = {0x05, 0x06, 0x07, 0x08, 0x04, 0x14, 0x00, 0x20, 0x40, 0x80,
		0x180, 0x201, 0x202, 0x203, 0x400, 0x800, 0x802, 0x2000
	};
	int f = (int)interface()->buffer()[4] + (int)interface()->buffer()[5] * 256u;
	for(int i = 0; i < (int)sizeof(funcs) / (int)sizeof(int); i++) {
		if(funcs[i] == f) {
			trans( *function()) = i;
		}
	}

	int dlen = interface()->buffer()[3] - 1;
	interface()->receive(dlen);
	std::vector<char> buf(dlen);
	memcpy(&buf[0], &interface()->buffer()[0], dlen);
	dbgPrint(XString(&buf[0]));
	if(buf.size() < 6)
		throw XInterface::XInterfaceError(i18n("Format Error!"), __FILE__, __LINE__);
	buf[dlen - 3] = '\0';
	if((XString( &buf[0]) == "+OL") || (XString( &buf[0]) == " OL")) {
		return 1e99;
	}
	if(XString( &buf[0]) == "-OL") {
		return -1e99;
	}
	if(buf.size() < 14)
		throw XInterface::XInterfaceError(i18n("Format Error!"), __FILE__, __LINE__);
	double x;
	if(sscanf( &buf[0], "%8lf", &x) != 1) {
		throw XInterface::XInterfaceError(i18n("Format Error!"), __FILE__, __LINE__);
	}
	double e;
	if(sscanf( &buf[8], "E%2lf", &e) != 1) {
		throw XInterface::XInterfaceError(i18n("Format Error!"), __FILE__, __LINE__);
	}
	return x * pow(10.0, e);
}
double
XSanwaPC500::oneShotRead() {
	return fetch();
}
void
XSanwaPC500::requestData() {
	char bytes[8] = {0x10, 0x02, 0x42, 0x00, 0x00, 0x00, 0x10, 0x03};
	interface()->write(bytes, sizeof(bytes));
}
XSanwaPC5000::XSanwaPC5000(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
		 XSanwaPC500(name, runtime, ref(tr_meas), meas) {
	
}

void
XSanwaPC5000::requestData() {
	char bytes[8] = {0x10, 0x02, 0x00, 0x00, 0x00, 0x00, 0x10, 0x03};
	interface()->write(bytes, sizeof(bytes));
}
