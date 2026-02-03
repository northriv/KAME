/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "analyzer.h"
#include "charinterface.h"
#include "usersignalgenerator.h"

REGISTER_TYPE(XDriverList, SG7130, "KENWOOD SG7130 signal generator");
REGISTER_TYPE(XDriverList, SG7200, "KENWOOD SG7200 signal generator");
REGISTER_TYPE(XDriverList, HP8643, "HP/Agilent 8643/8644 signal generator");
REGISTER_TYPE(XDriverList, HP8648, "HP/Agilent 8648 signal generator");
REGISTER_TYPE(XDriverList, HP8664, "HP/Agilent 8664/8665 signal generator");
REGISTER_TYPE(XDriverList, AgilentSGSCPI, "Keysight/Agilent E44xB signal generator SCPI");
REGISTER_TYPE(XDriverList, LibreVNASGSCPI, "LiberVNA signal generator SCPI");
REGISTER_TYPE(XDriverList, DPL32XGF, "DSTech. DPL-3.2XGF signal generator");
REGISTER_TYPE(XDriverList, RhodeSchwartzSMLSMV, "Rhode-Schwartz SML01/02/03/SMV03 signal generator");

XSG7200::XSG7200(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XSG>(name, runtime, ref(tr_meas), meas) {
	interface()->setGPIBUseSerialPollOnWrite(false);
	interface()->setGPIBUseSerialPollOnRead(false);
    amDepth()->disable();
    fmDev()->disable();
    amIntSrcFreq()->disable();
    fmIntSrcFreq()->disable();
    sweepFreqStart()->disable();
    sweepFreqStop()->disable();
    sweepAmplStart()->disable();
    sweepAmplStop()->disable();
    sweepDwellTime()->disable();
    sweepPoints()->disable();
    sweepMode()->disable();
}
XSG7130::XSG7130(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XSG7200(name, runtime, ref(tr_meas), meas) {
}
void
XSG7200::changeFreq(double mhz) {
	XScopedLock<XInterface> lock( *interface());
	interface()->sendf("FR%fMHZ", mhz);
	msecsleep(50); //wait stabilization of PLL
}
void
XSG7200::onRFONChanged(const Snapshot &shot, XValueNodeBase *) {
}
void
XSG7200::onOLevelChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("LE%fDBM", (double)shot[ *oLevel()]);
}
void
XSG7200::onFMONChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->send(shot[ *fmON()] ? "FMON" : "FMOFF");
}
void
XSG7200::onAMONChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->send(shot[ *amON()] ? "AMON" : "AMOFF");
}

XHP8643::XHP8643(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XSG>(name, runtime, ref(tr_meas), meas) {
	interface()->setGPIBUseSerialPollOnWrite(false);
//	interface()->setGPIBWaitBeforeWrite(10);
//	interface()->setGPIBWaitBeforeRead(10);
    amDepth()->disable();
    fmDev()->disable();
    amIntSrcFreq()->disable();
    fmIntSrcFreq()->disable();
    sweepFreqStart()->disable();
    sweepFreqStop()->disable();
    sweepAmplStart()->disable();
    sweepAmplStop()->disable();
    sweepDwellTime()->disable();
    sweepPoints()->disable();
    sweepMode()->disable();
}
double
XHP8643::getFreq() {
    interface()->query("FREQ:CW?");
    return interface()->toDouble() * 1e-6;
}
void
XHP8643::changeFreq(double mhz) {
	XScopedLock<XInterface> lock( *interface());
	interface()->sendf("FREQ:CW %f MHZ", mhz);
	msecsleep(75); //wait stabilization of PLL < 1GHz
}
void
XHP8643::onRFONChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("AMPL:STAT %s", shot[ *rfON()] ? "ON" : "OFF");
}
void
XHP8643::onOLevelChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("AMPL:LEV %f DBM", (double)shot[ *oLevel()]);
}
void
XHP8643::onFMONChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("FMSTAT %s", shot[ *fmON()] ? "ON" : "OFF");
}
void
XHP8643::onAMONChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("AMSTAT %s", shot[ *amON()] ? "ON" : "OFF");
}

XHP8648::XHP8648(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XHP8643(name, runtime, ref(tr_meas), meas) {
//	interface()->setGPIBUseSerialPollOnWrite(false);
//	interface()->setGPIBWaitBeforeWrite(10);
//	interface()->setGPIBWaitBeforeRead(10);
}
void
XHP8648::onRFONChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("OUTP:STAT %s", shot[ *rfON()] ? "ON" : "OFF");
}
void
XHP8648::onOLevelChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("POW:AMPL %f DBM", (double)shot[ *oLevel()]);
}

XAgilentSGSCPI::XAgilentSGSCPI(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XSG>(name, runtime, ref(tr_meas), meas) {
//	interface()->setGPIBUseSerialPollOnWrite(false);
//	interface()->setGPIBWaitBeforeWrite(10);
//	interface()->setGPIBWaitBeforeRead(10);
    interface()->setGPIBWaitBeforeSPoll(20);
    trans( *sweepMode()).add({"Off", "Freq.", "Ampl.", "Ampl Alt.", "Ampl 112 Alt.", "Ampl 1112 Alt."});
}
XHP8664::XHP8664(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XAgilentSGSCPI(name, runtime, ref(tr_meas), meas) {
    interface()->setGPIBUseSerialPollOnWrite(false);
//	interface()->setGPIBWaitBeforeWrite(10);
//	interface()->setGPIBWaitBeforeRead(10);
}
void
XHP8664::onRFONChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("AMPL:STAT %s", shot[ *rfON()] ? "ON" : "OFF");
}
void
XHP8664::onOLevelChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("AMPL %f DBM", (double)shot[ *oLevel()]);
}
double
XAgilentSGSCPI::getFreq() {
    interface()->query("FREQ:CW?");
    return interface()->toDouble() * 1e-6;
}
void
XAgilentSGSCPI::changeFreq(double mhz) {
	XScopedLock<XInterface> lock( *interface());
    interface()->sendf("FREQ:CW %f MHZ", mhz);
	msecsleep(50); //wait stabilization of PLL
}
void
XAgilentSGSCPI::onRFONChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("OUTP:STAT %s", shot[ *rfON()] ? "ON" : "OFF");
}
void
XAgilentSGSCPI::onOLevelChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("POW:AMPL %f DBM", (double)shot[ *oLevel()]);
}
void
XAgilentSGSCPI::onFMONChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("FM:STAT %s", shot[ *fmON()] ? "ON" : "OFF");
}
void
XAgilentSGSCPI::onAMONChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("AM:STAT %s", shot[ *amON()] ? "ON" : "OFF");
}
void
XAgilentSGSCPI::onAMDepthChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("AM:DEPTH %f", (double)shot[ *amDepth()]);
}
void
XAgilentSGSCPI::onFMDevChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("FM:DEV %f MHZ", (double)shot[ *fmDev()]);
}
void
XAgilentSGSCPI::onAMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("AM:INT:FREQ %f KHZ", (double)shot[ *amIntSrcFreq()]);
}
void
XAgilentSGSCPI::onFMIntSrcFreqChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("FM:INT:FREQ %f KHZ", (double)shot[ *fmIntSrcFreq()]);
}
void
XAgilentSGSCPI::onSweepCondChanged(const Snapshot &, XValueNodeBase *) {
    XScopedLock<XInterface> lock( *interface());
    interface()->send(":ABOR");
    Snapshot shot(*this);
    switch(shot[ *sweepMode()]) {
    case 0: //OFF
    default:
        interface()->send("FREQ:MODE CW");
        interface()->send("POW:MODE FIXED");
        break;
    case 1: //Freq
        interface()->send("FREQ:MODE LIST");
        interface()->send("POW:MODE FIXED");
        interface()->send("LIST:TYPE STEP");
        interface()->sendf("FREQ:START %f MHZ", (double)shot[ *sweepFreqStart()]);
        interface()->sendf("FREQ:STOP %f MHZ", (double)shot[ *sweepFreqStop()]);
        interface()->sendf("SWE:DWEL %f S", (double)shot[ *sweepDwellTime()]);
        interface()->sendf("SWE:POIN %u", (unsigned int)shot[ *sweepPoints()]);
        interface()->send("INIT:CONT ON");
        break;
    case 2: //Ampl
        interface()->send("POW:MODE LIST");
        interface()->send("FREQ:MODE CW");
        interface()->send("LIST:TYPE STEP");
        interface()->sendf("POW:START %f DB", (double)shot[ *sweepAmplStart()]);
        interface()->sendf("POW:STOP %f DB", (double)shot[ *sweepAmplStop()]);
        interface()->sendf("SWE:DWEL %f S", (double)shot[ *sweepDwellTime()]);
        interface()->sendf("SWE:POIN %u", (unsigned int)shot[ *sweepPoints()]);
        interface()->send("INIT:CONT ON");
        break;
    case 3: //Ampl Alt.
    {
        interface()->send("LIST:TYPE LIST");
        interface()->send("POW:MODE LIST");
        XString buf = "LIST:POW ";
        for(unsigned int i = 0; i < shot[ *sweepPoints()]; ++i) {
            double db = (i % 2 == 0) ? (double)shot[ *sweepAmplStart()] : (double)shot[ *sweepAmplStop()];
            if(i != 0)
                buf.append(",");
            buf.append(formatString("%f", db));
        }
        interface()->send(buf);
        interface()->send("FREQ:MODE CW");
        interface()->send("LIST:DWEL:TYPE STEP");
        interface()->sendf("SWE:DWEL %f S", (double)shot[ *sweepDwellTime()]);
        interface()->send("INIT:CONT OFF");
        interface()->send("INIT:IMM");
    }
        break;
    case 4: //Ampl 112 Alt.
    {
        interface()->send("LIST:TYPE LIST");
        interface()->send("POW:MODE LIST");
        XString buf = "LIST:POW ";
        for(unsigned int i = 0; i < shot[ *sweepPoints()]; ++i) {
            double db = (i % 3 != 2) ? (double)shot[ *sweepAmplStart()] : (double)shot[ *sweepAmplStop()];
            if(i != 0)
                buf.append(",");
            buf.append(formatString("%f", db));
        }
        interface()->send(buf);
        interface()->send("FREQ:MODE CW");
        interface()->send("LIST:DWEL:TYPE STEP");
        interface()->sendf("SWE:DWEL %f S", (double)shot[ *sweepDwellTime()]);
        interface()->send("INIT:CONT OFF");
        interface()->send("INIT:IMM");
    }
        break;
    case 5: //Ampl 1112 Alt.
    {
        interface()->send("LIST:TYPE LIST");
        interface()->send("POW:MODE LIST");
        XString buf = "LIST:POW ";
        for(unsigned int i = 0; i < shot[ *sweepPoints()]; ++i) {
            double db = (i % 4 != 3) ? (double)shot[ *sweepAmplStart()] : (double)shot[ *sweepAmplStop()];
            if(i != 0)
                buf.append(",");
            buf.append(formatString("%f", db));
        }
        interface()->send(buf);
        interface()->send("FREQ:MODE CW");
        interface()->send("LIST:DWEL:TYPE STEP");
        interface()->sendf("SWE:DWEL %f S", (double)shot[ *sweepDwellTime()]);
        interface()->send("INIT:CONT OFF");
        interface()->send("INIT:IMM");
    }
        break;
//        interface()->send("FREQ:MODE LIST");
//        buf = "LIST:FREQ ";
//        for(unsigned int i = 0; i < shot[ *sweepPoints()]; ++i) {
//            double x = (double)(i / 2 * 2) / ((shot[ *sweepPoints()] - 1) / 2 * 2);
//            double freq = (1 - x) * shot[ *sweepFreqStart()] + x * shot[ *sweepFreqStop()];
//            if(i != 0)
//                buf.append(",");
//            buf.append(formatString("%f", freq * 1e6));
//        }
//        interface()->send(buf);
    }
}

XLibreVNASGSCPI::XLibreVNASGSCPI(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XSG>(name, runtime, ref(tr_meas), meas) {
    interface()->setEOS("\n");
    interface()->device()->setUIEnabled(false);
    trans( *interface()->device()) = "TCP/IP";
    trans( *interface()->port()) = "127.0.0.1:19542";
    trans( *sweepMode()).add({"Off", "Freq.", "Ampl."});
    amON()->disable();
    fmON()->disable();
    amDepth()->disable();
    fmDev()->disable();
    amIntSrcFreq()->disable();
    fmIntSrcFreq()->disable();
    sweepFreqStart()->disable();
    sweepFreqStop()->disable();
    sweepAmplStart()->disable();
    sweepAmplStop()->disable();
    sweepDwellTime()->disable();
    sweepPoints()->disable();
    sweepMode()->disable();
}

double
XLibreVNASGSCPI::getFreq() {
    interface()->query(":GEN:FREQ?");
    return interface()->toDouble() * 1e-6;
}
void
XLibreVNASGSCPI::changeFreq(double mhz) {
    XScopedLock<XInterface> lock( *interface());
    interface()->queryf(":GEN:FREQ %.0f", mhz * 1e6);
    if(interface()->toStr() == "ERROR\n")
        throw XInterface::XConvError(__FILE__, __LINE__);
    msecsleep(50); //wait stabilization of PLL
}
void
XLibreVNASGSCPI::onRFONChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->queryf(":DEV:MODE %s", shot[ *rfON()] ? "SA" : "VNA");
    if(interface()->toStr() == "ERROR\n")
        throw XInterface::XConvError(__FILE__, __LINE__);
    interface()->queryf(":GEN:PORT %s", shot[ *rfON()] ? "1" : "0");
    if(interface()->toStr() == "ERROR\n")
        throw XInterface::XConvError(__FILE__, __LINE__);
}
void
XLibreVNASGSCPI::onOLevelChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->queryf(":GEN:LVL %.0f", (double)shot[ *oLevel()]);
    if(interface()->toStr() == "ERROR\n")
        throw XInterface::XConvError(__FILE__, __LINE__);
}

XDPL32XGF::XDPL32XGF(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XSG>(name, runtime, ref(tr_meas), meas) {
	interface()->setEOS("\r\n");
	interface()->setSerialBaudRate(9600);
	interface()->setSerialStopBits(1);
	interface()->setSerialFlushBeforeWrite(true);
    amON()->disable();
    fmON()->disable();
    amDepth()->disable();
    fmDev()->disable();
    amIntSrcFreq()->disable();
    fmIntSrcFreq()->disable();
    sweepFreqStart()->disable();
    sweepFreqStop()->disable();
    sweepAmplStart()->disable();
    sweepAmplStop()->disable();
    sweepDwellTime()->disable();
    sweepPoints()->disable();
    sweepMode()->disable();
}
void
XDPL32XGF::changeFreq(double mhz) {
	XScopedLock<XInterface> lock( *interface());
	interface()->queryf("F %fM", mhz);
	msecsleep(50); //wait stabilization of PLL
}
void
XDPL32XGF::onRFONChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->queryf("%s", shot[ *rfON()] ? "ON" : "OFF");
}
void
XDPL32XGF::onOLevelChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->queryf("A %.1f", (double)shot[ *oLevel()]);
}
void
XDPL32XGF::onFMONChanged(const Snapshot &shot, XValueNodeBase *) {
}
void
XDPL32XGF::onAMONChanged(const Snapshot &shot, XValueNodeBase *) {
}

XRhodeSchwartzSMLSMV::XRhodeSchwartzSMLSMV(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XSG>(name, runtime, ref(tr_meas), meas) {
	interface()->setEOS("\r\n");
	interface()->setSerialBaudRate(9600);
	interface()->setSerialStopBits(1);
	interface()->setSerialFlushBeforeWrite(true);
    amDepth()->disable();
    fmDev()->disable();
    amIntSrcFreq()->disable();
    fmIntSrcFreq()->disable();
    sweepFreqStart()->disable();
    sweepFreqStop()->disable();
    sweepAmplStart()->disable();
    sweepAmplStop()->disable();
    sweepDwellTime()->disable();
    sweepPoints()->disable();
    sweepMode()->disable();
}
void
XRhodeSchwartzSMLSMV::changeFreq(double mhz) {
	XScopedLock<XInterface> lock( *interface());
    interface()->sendf(":SOUR:FREQ %f", mhz * 1e6);
	msecsleep(50); //wait stabilization of PLL
}
void
XRhodeSchwartzSMLSMV::onRFONChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf(":OUTP:STAT %s", shot[ *rfON()] ? "ON" : "OFF");
}
void
XRhodeSchwartzSMLSMV::onOLevelChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf(":POW %.1f", (double)shot[ *oLevel()]);
}
void
XRhodeSchwartzSMLSMV::onFMONChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf(":FM:STAT %s", shot[ *fmON()] ? "ON" : "OFF");
}
void
XRhodeSchwartzSMLSMV::onAMONChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf(":AM:STAT %s", shot[ *amON()] ? "ON" : "OFF");
}
