/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
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
#include "thamwayprot.h"

REGISTER_TYPE(XDriverList, ThamwayT300ImpedanceAnalyzer, "Thamway T300-1049A Impedance Analyzer");
REGISTER_TYPE(XDriverList, ThamwayPROTSG, "Thamway PROT built-in signal generator");

XThamwayPROTSG::XThamwayPROTSG(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XCharDeviceDriver<XSG>(name, runtime, ref(tr_meas), meas) {
    interface()->setEOS("\n");
    amON()->disable();
    fmON()->disable();
    trans( *interface()->port()) = "localhost:5025";
    trans( *interface()->device()) = "TCP/IP";
}
void
XThamwayPROTSG::changeFreq(double mhz) {
    XScopedLock<XInterface> lock( *interface());
    interface()->sendf("FREQW%010.6f", mhz);
    msecsleep(50); //wait stabilization of PLL
}
void
XThamwayPROTSG::onRFONChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("RFSWW%s", shot[ *rfON()] ? "1" : "0");
}
void
XThamwayPROTSG::onOLevelChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("ATT1W%04.0f", (double)pow(10, shot[ *oLevel()] / 10.0) * 1024.0);
}
void
XThamwayPROTSG::onFMONChanged(const Snapshot &shot, XValueNodeBase *) {
}
void
XThamwayPROTSG::onAMONChanged(const Snapshot &shot, XValueNodeBase *) {
}


XThamwayT300ImpedanceAnalyzer::XThamwayT300ImpedanceAnalyzer(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XNetworkAnalyzer>(name, runtime, ref(tr_meas), meas) {
    interface()->setEOS("\r\n");
    interface()->setSerialBaudRate(115200);
    interface()->setSerialStopBits(1);
    interface()->setSerialFlushBeforeWrite(true);
    trans( *interface()->device()) = "SERIAL";

    average()->disable();

    calThru()->disable();
}

void
XThamwayT300ImpedanceAnalyzer::open() throw (XKameError &) {
    interface()->query("GET START?");
    double freq;
    if(interface()->scanf("START %lf", &freq) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    trans( *startFreq()) = freq;
    interface()->query("GET STOP?");
    if(interface()->scanf("STOP %lf", &freq) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    trans( *stopFreq()) = freq;
    unsigned int samples;
    interface()->query("GET SAMPLE?");
    if(interface()->scanf("SAMPLE %u", &samples) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    trans( *points()) = samples;

//    interface()->send("RJX FORMAT"); //R+jX format
    interface()->send("LOG FORMAT"); //log format

    start();
}
void
XThamwayT300ImpedanceAnalyzer::onStartFreqChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("SET START %f", (double)shot[ *startFreq()]);
}
void
XThamwayT300ImpedanceAnalyzer::onStopFreqChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("SET STOP %f", (double)shot[ *stopFreq()]);
}
void
XThamwayT300ImpedanceAnalyzer::onPointsChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("SET SAMPLE %u", (unsigned int)shot[ *points()]);
}
void
XThamwayT300ImpedanceAnalyzer::onCalOpenTouched(const Snapshot &shot, XTouchableNode *) {
    interface()->query("CAL OPEN");
    if(interface()->toStrSimplified() != "CAL OPEN DONE")
        throw XInterface::XInterfaceError("Calibration has failed.", __FILE__, __LINE__);
}
void
XThamwayT300ImpedanceAnalyzer::onCalShortTouched(const Snapshot &shot, XTouchableNode *) {
    interface()->query("CAL SHORT");
    if(interface()->toStrSimplified() != "CAL SHORT DONE")
        throw XInterface::XInterfaceError("Calibration has failed.", __FILE__, __LINE__);
}
void
XThamwayT300ImpedanceAnalyzer::onCalTermTouched(const Snapshot &shot, XTouchableNode *) {
    interface()->query("CAL LOAD");
    if(interface()->toStrSimplified() != "CAL LOAD DONE")
        throw XInterface::XInterfaceError("Calibration has failed.", __FILE__, __LINE__);
}

void
XThamwayT300ImpedanceAnalyzer::getMarkerPos(unsigned int num, double &x, double &y) {
    if(num > 2)
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    Snapshot shot( *this);
    const auto trace = shot[ *this].trace();
    double v = (num == 0) ? 1e10 : -1.0;
    unsigned int idx;
    for(unsigned int i = 0; i < shot[ *this].length(); i++) {
        double z = std::norm( trace[i]);
        if(num == 0) {
            if(v > z) {
                v = z;
                idx = i;
            }
        }
        else {
            if(v < z) {
                v = z;
                idx = i;
            }
        }
    }
    x *= idx * shot[ *this].freqInterval() + shot[ *this].startFreq();
    y = log10(v) * 20.0;
}
void
XThamwayT300ImpedanceAnalyzer::oneSweep() {
}
void
XThamwayT300ImpedanceAnalyzer::startContSweep() {
}
void
XThamwayT300ImpedanceAnalyzer::acquireTrace(shared_ptr<RawData> &writer, unsigned int ch) {
    XScopedLock<XInterface> lock( *interface());
    uint32_t len;
    interface()->query("MEAS ON");
    if(interface()->scanf("LOG MAG,F=,M=,%u", &len) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    writer->push((uint32_t)0); //LOG MAG
    writer->push(len);
    for(unsigned int i = 0; i < len; ++i) {
        interface()->receive();
        float freq, amp;
        if(interface()->scanf("%f,%f", &freq, &amp) != 2)
            throw XInterface::XConvError(__FILE__, __LINE__);
        writer->push(freq);
        writer->push(amp);
    }
}
void
XThamwayT300ImpedanceAnalyzer::convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    const Snapshot &shot(tr);
    uint32_t stype = reader.pop<uint32_t>();
    uint32_t samples = reader.pop<uint32_t>();
    switch(stype) {
    case 0: //Log format.
        break;
    default:
        throw XRecordError(i18n("Given format is not supported."), __FILE__, __LINE__);
    }

    tr[ *this].trace_().resize(samples);
    float startfreq = 0.0, stopfreq = 0.0;
    for(unsigned int i = 0; i < samples; i++) {
        float freq = reader.pop<float>();
        if(i == 0)
            startfreq = freq;
        if(i == samples - 1)
            stopfreq = freq;
        tr[ *this].trace_()[i] = pow(10.0, reader.pop<float>() / 20.0);
    }
    tr[ *this].m_startFreq = startfreq;
    tr[ *this].m_freqInterval = (stopfreq - startfreq) / (samples - 1);
}
