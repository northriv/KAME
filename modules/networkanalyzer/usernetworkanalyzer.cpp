/***************************************************************************
        Copyright (C) 2002-2018 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "usernetworkanalyzer.h"
#include "charinterface.h"
#include "analyzer.h"
#include <istream>
#include <sstream>

REGISTER_TYPE(XDriverList, HP8711, "HP/Agilent 8711/8712/8713/8714 Network Analyzer");
REGISTER_TYPE(XDriverList, AgilentE5061, "Agilent E5061/E5062 Network Analyzer");
REGISTER_TYPE(XDriverList, CopperMtTRVNA, "Copper Mountain TR1300/1,5048,4530 Network Analyzer");
REGISTER_TYPE(XDriverList, VNWA3ENetworkAnalyzer, "DG8SAQ VNWA3E/Custom Network Analyzer");
REGISTER_TYPE(XDriverList, VNWA3ENetworkAnalyzerTCPIP, "DG8SAQ VNWA3E Network Analyzer TCP/IP");
REGISTER_TYPE(XDriverList, LibreVNASCPI, "LiberVNA Network Analyzer SCPI");

//---------------------------------------------------------------------------
XAgilentNetworkAnalyzer::XAgilentNetworkAnalyzer(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XNetworkAnalyzer>(name, runtime, ref(tr_meas), meas) {
    iterate_commit([=](Transaction &tr){
        tr[ *points()].add({"3", "5", "11", "21", "51", "101", "201", "401", "801", "1601"});
    });

	calOpen()->disable();
	calShort()->disable();
	calTerm()->disable();
	calThru()->disable();

    interface()->setGPIBWaitBeforeRead(10);
    interface()->setGPIBWaitBeforeWrite(10);
}

void
XAgilentNetworkAnalyzer::open() {
	interface()->query("SENS:FREQ:START?");
	trans( *startFreq()) = interface()->toDouble() / 1e6;
	interface()->query("SENS:FREQ:STOP?");
	trans( *stopFreq()) = interface()->toDouble() / 1e6;
	interface()->query("SENS:AVER:STAT?");
	if(interface()->toUInt() == 0) {
		trans( *average()) = 1;
	}
	else {
		interface()->query("SENS:AVER:COUNT?");
		trans( *average()) = interface()->toUInt();
	}
	interface()->query("SENS:SWE:POIN?");
	trans( *points()).str(formatString("%u", interface()->toUInt()));
//	interface()->send("SENS:SWE:TIME:AUTO OFF");
//	interface()->query("SENS:SWE:TIME?");
//	double swet = interface()->toDouble();
//	interface()->sendf(":SENS:SWE:TIME %f S", std::min(1.0, std::max(0.3, swet)));
    interface()->query("SOUR1:POW?");
    trans( *power()) = interface()->toDouble();
    interface()->send("ABOR;INIT:CONT OFF");
	
	start();
}
void 
XAgilentNetworkAnalyzer::onStartFreqChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("SENS:FREQ:START %f MHZ", (double)shot[ *startFreq()]);
}
void 
XAgilentNetworkAnalyzer::onStopFreqChanged(const Snapshot &shot, XValueNodeBase *) {
	interface()->sendf("SENS:FREQ:STOP %f MHZ", (double)shot[ *stopFreq()]);
}
void
XAgilentNetworkAnalyzer::onAverageChanged(const Snapshot &shot, XValueNodeBase *) {
	unsigned int avg = shot[ *average()];
	if(avg >= 2)
		interface()->sendf("SENS:AVER:CLEAR;STAT ON;COUNT %u", avg);
	else
		interface()->send("SENS:AVER:STAT OFF");
}
void
XAgilentNetworkAnalyzer::onPointsChanged(const Snapshot &shot, XValueNodeBase *) {	
	interface()->sendf("SENS:SWE:POIN %s", shot[ *points()].to_str().c_str());
}
void
XAgilentNetworkAnalyzer::onPowerChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("SOUR1:POW %f", (double)shot[ *power()]);
}
void
XAgilentNetworkAnalyzer::getMarkerPos(unsigned int num, double &x, double &y) {
	XScopedLock<XInterface> lock( *interface());
	if(num >= 8)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	interface()->queryf("CALC:MARK%u:STAT?", num + 1u);
	if(interface()->toInt() != 1)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);		
	interface()->queryf("CALC:MARK%u:X?", num + 1u);
	x = interface()->toDouble() / 1e6;
	interface()->queryf("CALC:MARK%u:Y?", num + 1u);
	y = interface()->toDouble();
}
void
XAgilentNetworkAnalyzer::oneSweep() {
	interface()->query("INIT:IMM;*OPC?");
}
void
XAgilentNetworkAnalyzer::startContSweep() {
	interface()->send("INIT:CONT ON");
}
void
XAgilentNetworkAnalyzer::acquireTrace(shared_ptr<RawData> &writer, unsigned int ch) {
	XScopedLock<XInterface> lock( *interface());
	if(ch >= 2)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	interface()->queryf("SENS%u:FREQ:START?", ch + 1u);
	double start = interface()->toDouble() / 1e6;
	writer->push(start);
	interface()->queryf("SENS%u:FREQ:STOP?", ch + 1u);
	double stop = interface()->toDouble() / 1e6;
	writer->push(stop);
	interface()->queryf("SENS%u:SWE:POIN?", ch + 1u);
	uint32_t len = interface()->toUInt();
	writer->push(len);
    uint32_t ptfield_len = acquireTraceData(ch, len);
    writer->push(ptfield_len);
    writer->insert(writer->end(),
					 interface()->buffer().begin(), interface()->buffer().end());
}
void
XAgilentNetworkAnalyzer::convertRaw(RawDataReader &reader, Transaction &tr) {
	double start = reader.pop<double>();
	double stop = reader.pop<double>();
	unsigned int samples = reader.pop<uint32_t>();
	tr[ *this].m_startFreq = start;
	tr[ *this].m_freqInterval = (stop - start) / (samples - 1);
	tr[ *this].trace_().resize(samples);
    unsigned int ptfield_len = reader.pop<uint32_t>();
    convertRawBlock(reader, tr, ptfield_len);
}

unsigned int
XHP8711::acquireTraceData(unsigned int ch, unsigned int len) {
    interface()->send("FORM:DATA REAL,32;BORD SWAP");
	interface()->sendf("TRAC? CH%uFDATA", ch + 1u);
    interface()->receive(2);
    unsigned int ptfield_len;
    if(interface()->scanf("#%1u", &ptfield_len) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    interface()->receive(ptfield_len); //usually 6
    //! \todo complex data.
    unsigned int pts_len = interface()->toUInt();
    interface()->receive(pts_len + 1); //+ LF
    return pts_len;
}
void
XHP8711::convertRawBlock(RawDataReader &reader, Transaction &tr,
	unsigned int len) {
	unsigned int samples = tr[ *this].trace_().size();
	if(len / sizeof(float) < samples)
		throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	if(len / sizeof(float) > samples)
		throw XRecordError(i18n("Select scalar plot."), __FILE__, __LINE__);
	for(unsigned int i = 0; i < samples; i++) {
		tr[ *this].trace_()[i] = pow(10.0, reader.pop<float>() / 20.0);
	}
}

unsigned int
XAgilentE5061::acquireTraceData(unsigned int ch, unsigned int len) {
    interface()->send("FORM:DATA REAL32;BORD SWAP"); //binary float, little endian
    interface()->sendf("CALC%u:FORM  SCOMPLEX", ch + 1u); //smith r+jx
    interface()->sendf("CALC%u:DATA:FDAT?", ch + 1u);
    interface()->receive(2);
    unsigned int ptfield_len;
    if(interface()->scanf("#%1u", &ptfield_len) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    interface()->receive(ptfield_len);
    unsigned int pts_len = interface()->toUInt();
    interface()->receive(pts_len + 1); //binary + LF + (END)
    return pts_len;
}
void
XAgilentE5061::convertRawBlock(RawDataReader &reader, Transaction &tr,
	unsigned int len) {
	unsigned int samples = tr[ *this].trace_().size();
    if(len / sizeof(float) < samples * 2)
		throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	for(unsigned int i = 0; i < samples; i++) {
		tr[ *this].trace_()[i] = std::complex<double>(
			reader.pop<float>(), reader.pop<float>());
	}
}

XCopperMtTRVNA::XCopperMtTRVNA(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XAgilentE5061(name, runtime, ref(tr_meas), meas) {
    interface()->setEOS("\n");
    trans( *interface()->device()) = "TCP/IP";
    trans( *interface()->port()) = "127.0.0.1:5025";
}

XVNWA3ENetworkAnalyzer::XVNWA3ENetworkAnalyzer(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XNetworkAnalyzer>(name, runtime, ref(tr_meas), meas) {
	interface()->setEOS("\n");
    trans( *interface()->device()) = "TCP/IP";
    trans( *interface()->port()) = "127.0.0.1:12333";

	average()->disable();
	points()->disable();
    power()->disable();

	calOpen()->disable();
	calShort()->disable();
	calTerm()->disable();
	calThru()->disable();
}

void
XVNWA3ENetworkAnalyzer::open() {
	start();
}
void
XVNWA3ENetworkAnalyzer::onStartFreqChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("FSTART %f", (double)shot[ *startFreq()] * 1e6);
}
void
XVNWA3ENetworkAnalyzer::onStopFreqChanged(const Snapshot &shot, XValueNodeBase *) {
    interface()->sendf("FSTOP %f", (double)shot[ *stopFreq()] * 1e6);
}
void
XVNWA3ENetworkAnalyzer::getMarkerPos(unsigned int num, double &x, double &y) {
	if(num > 1)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	interface()->queryf("MARK%u?", num);
	if(interface()->scanf("MARK %*u %lf %lf", &x, &y) != 2)
		throw XInterface::XConvError(__FILE__, __LINE__);
	x *= 1e-6;
	y = log10(y) * 10.0;
}
void
XVNWA3ENetworkAnalyzer::oneSweep() {
    unsigned int num;
	interface()->query("ACQNUM?");
	if(interface()->scanf("ACQNUM %u", &num) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	if(num == 0)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
}
void
XVNWA3ENetworkAnalyzer::startContSweep() {
}
void
XVNWA3ENetworkAnalyzer::acquireTrace(shared_ptr<RawData> &writer, unsigned int ch) {
	XScopedLock<XInterface> lock( *interface());
	unsigned int len;
	interface()->query("DATA?");
	if(interface()->scanf("DATA %u", &len) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	interface()->receive(len);
	writer->insert(writer->end(),
					 interface()->buffer().begin(), interface()->buffer().end());
}
void
XVNWA3ENetworkAnalyzer::convertRaw(RawDataReader &reader, Transaction &tr) {
	const Snapshot &shot(tr);
    uint32_t hsize = reader.pop<uint32_t>();
	int stype = reader.pop<int32_t>();
	double start = reader.pop<double>() * 1e-6; //[MHz]
	double stop = reader.pop<double>() * 1e-6; //[MHz]
	int samples = reader.pop<int32_t>();
	int rec = reader.pop<int32_t>();
	double tm = reader.pop<double>();
	double temp = reader.pop<double>(); //4*4+8*4 = 48bytes
	for(int cnt = 0; cnt < hsize - 48; ++cnt)
		reader.pop<char>(); //skips remaining header.

	double df = (stop - start) / (samples - 1);
	tr[ *this].m_startFreq = start;
	tr[ *this].m_freqInterval = df;
	tr[ *this].trace_().resize(samples);

	switch(stype) {
	case 1: //Linear sweep.
		break;
	case 2:	//Log sweep.
	case 3: //Listed sweep.
	default:
		throw XRecordError(i18n("Log/Listed sweep is not supported."), __FILE__, __LINE__);
	}
	switch(rec) {
	case 1: //S21
	case 2: //S11
	case 3: //S12
	case 4: //S22
		break;
	case 5: //all
	default:
		throw XRecordError(i18n("Select one of record."), __FILE__, __LINE__);
	}

	for(unsigned int i = 0; i < samples; i++) {
		tr[ *this].trace_()[i] = std::complex<double>(reader.pop<double>(), reader.pop<double>());
	}
}


XVNWA3ENetworkAnalyzerTCPIP::XVNWA3ENetworkAnalyzerTCPIP(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XNetworkAnalyzer>(name, runtime, ref(tr_meas), meas),
    m_interface2(XNode::create<XCharInterface>("Interface2", false,
        dynamic_pointer_cast<XDriver>(this->shared_from_this()))) {
    meas->interfaces()->insert(tr_meas, m_interface2);
    interface()->setEOS("");
    interface()->device()->setUIEnabled(false);
    trans( *interface()->device()) = "TCP/IP";
    trans( *interface()->port()) = "127.0.0.1:55555";
    interface2()->setEOS("");
    interface2()->control()->setUIEnabled(false);
    interface2()->device()->setUIEnabled(false);
    trans( *interface2()->device()) = "TCP/IP";
    trans( *interface2()->port()) = "127.0.0.1:55556";

    iterate_commit([=](Transaction &tr){
        tr[ *points()].add({"3", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048"});
        tr[ *points()].str("512");
    });

    average()->disable();
    power()->disable();

    calOpen()->disable();
    calShort()->disable();
    calTerm()->disable();
    calThru()->disable();
}

void
XVNWA3ENetworkAnalyzerTCPIP::open() {
    interface2()->start();
//    interface()->write("stop", 5); //with null.
//    interface()->receive(); //"sweep stopped "
    start();
}
void
XVNWA3ENetworkAnalyzerTCPIP::close() {
    interface2()->stop();
    XCharDeviceDriver<XNetworkAnalyzer>::close();
}
void
XVNWA3ENetworkAnalyzerTCPIP::onStartFreqChanged(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
    auto buf = formatString("range %g %g",
        (double)shot_this[ *startFreq()] * 1e6, (double)shot_this[ *stopFreq()] * 1e6);
    interface()->write(buf.c_str(), buf.length() + 1); //with null.
    interface()->receive();
    int err;
    if(interface()->scanf("Error Code: %d", &err) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    if(err != 0)
        throw XInterface::XInterfaceError( &interface()->buffer()[0], __FILE__, __LINE__);
}
void
XVNWA3ENetworkAnalyzerTCPIP::onStopFreqChanged(const Snapshot &shot, XValueNodeBase *node) {
    onStartFreqChanged(shot, node);
}
void
XVNWA3ENetworkAnalyzerTCPIP::onPointsChanged(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
    auto buf = formatString("setgrid lin %g %g %u",
        (double)shot_this[ *startFreq()] * 1e6, (double)shot_this[ *stopFreq()] * 1e6,
        atoi(shot_this[ *points()].to_str().c_str()));
    interface()->write(buf.c_str(), buf.length() + 1); //with null.
    interface()->receive();
    int err;
    if(interface()->scanf("Error Code: %d", &err) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    if(err != 0)
        throw XInterface::XInterfaceError( &interface()->buffer()[0], __FILE__, __LINE__);
}

void
XVNWA3ENetworkAnalyzerTCPIP::getMarkerPos(unsigned int num, double &x, double &y) {
    throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
}
void
XVNWA3ENetworkAnalyzerTCPIP::oneSweep() {
    XString buf = "sweep S21 S11";
    interface()->write(buf.c_str(), buf.length() + 1); //with null.
    interface()->receive();
    int err;
    if(interface()->scanf("Error Code: %d", &err) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    if(err != 0)
        throw XInterface::XInterfaceError( &interface()->buffer()[0], __FILE__, __LINE__);
}
void
XVNWA3ENetworkAnalyzerTCPIP::startContSweep() {
}
void
XVNWA3ENetworkAnalyzerTCPIP::acquireTrace(shared_ptr<RawData> &writer, unsigned int ch) {
    XScopedLock<XInterface> lock( *interface());
    interface2()->receive();
    unsigned int num_pts, sweep_type;
    double start_freq, stop_freq;
    if(interface2()->scanf("sweep_start %u %lf %lf %u",
        &num_pts, &start_freq, &stop_freq, &sweep_type) != 4)
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    writer->push((uint32_t)num_pts);
    writer->push((int32_t)sweep_type);
    writer->push(start_freq);
    writer->push(stop_freq);
    for(unsigned int cnt = 0; cnt < num_pts; cnt++) {
        interface2()->receive();
        double freq, res11, ims11, res21, ims21;
        if(interface2()->scanf("data %*u %lf %lf %lf %lf %lf",
            &freq, &res11, &ims11, &res21, &ims21) != 5)
            throw XInterface::XConvError(__FILE__, __LINE__);
        writer->push(freq);
        writer->push(res11);
        writer->push(ims11);
        writer->push(res21);
        writer->push(ims21);
    }
    interface2()->receive(); //sweep_complete
    unsigned int subswp;
    if(interface2()->scanf("sweep_complete %u", &subswp) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    interface()->receive(); //sweep_complete
}
void
XVNWA3ENetworkAnalyzerTCPIP::convertRaw(RawDataReader &reader, Transaction &tr) {
    const Snapshot &shot(tr);

    uint32_t samples = reader.pop<uint32_t>();
    int stype = reader.pop<int32_t>();
    double start = reader.pop<double>() * 1e-6; //[MHz]
    double stop = reader.pop<double>() * 1e-6; //[MHz]

    double df = (stop - start) / (samples - 1);
    tr[ *this].m_startFreq = start;
    tr[ *this].m_freqInterval = df;
    tr[ *this].trace_().resize(samples);

    switch(stype & 0xfu) {
    case 1: //Linear sweep.
        break;
    case 2:	//Log sweep.
    case 3: //Listed sweep.
    default:
        throw XRecordError(i18n("Log/Listed sweep is not supported."), __FILE__, __LINE__);
    }
    double min_f = 1e10, max_f = -1e10, min_v = 1e10, max_v = -1e10;
    for(unsigned int i = 0; i < samples; i++) {
        double f = reader.pop<double>() * 1e-6; //freq [MHz]
        auto z = std::complex<double>(reader.pop<double>(), reader.pop<double>());
        tr[ *this].trace_()[i] = z;
        reader.pop<double>(); //s21re
        reader.pop<double>(); //s21im
        if(std::abs(z) < min_v) {
            min_v = std::abs(z);
            min_f = f;
        }
        if(std::abs(z) > max_v) {
            max_v = std::abs(z);
            max_f = f;
        }
    }
    tr[ *this].markers().resize(2);
    tr[ *this].markers()[0].first = min_f;
    tr[ *this].markers()[0].second = 20.0 * log10(min_v);
    tr[ *this].markers()[1].first = max_f;
    tr[ *this].markers()[1].second = 20.0 * log10(max_v);
}



XLibreVNASCPI::XLibreVNASCPI(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XNetworkAnalyzer>(name, runtime, ref(tr_meas), meas) {
    interface()->setEOS("\n");
    interface()->device()->setUIEnabled(false);
    trans( *interface()->device()) = "TCP/IP";
    trans( *interface()->port()) = "127.0.0.1:19542";

    iterate_commit([=](Transaction &tr){
        tr[ *points()].add({"51", "101", "201", "501", "1001", "2001", "5001"});
        tr[ *points()].str("501");
    });

    calOpen()->disable();
    calShort()->disable();
    calTerm()->disable();
    calThru()->disable();
}

void
XLibreVNASCPI::open() {
    //Before start(), which settles the sweep settings and would otherwise
    //send events without knowing how this GUI answers them.
    m_scpi.probeAPI(interface());
    //Whatever mode the GUI was left in; oneSweep() finds out afresh.  Set
    //before start() spawns the thread that alone touches these afterwards.
    m_continuous = false;
    m_lastSweepFreq = -1.0;
    m_warnedSweepFrozen = false;
    m_acquisitionStarted = {};
    this->start();
}

void
XLibreVNASCPI::rearrangeIFBW() {
    interface()->query(":VNA:ACQ:POINTS?");
    unsigned int pts = interface()->toUInt();
    interface()->query(":VNA:FREQ:START?");
    double start = interface()->toDouble();
    interface()->query(":VNA:FREQ:STOP?");
    double stop = interface()->toDouble();
    double ifbw = (stop - start) / (pts - 1);
    interface()->query(":DEV:INF:LIM:MAXIFBW?");
    double maxifbw = interface()->toDouble();
    maxifbw = std::min(maxifbw, 100e3); //guess >100kHz is unstable
    interface()->query(":DEV:INF:LIM:MINIFBW?");
    double minifbw = interface()->toDouble();
    ifbw = std::max(minifbw, std::min(ifbw, maxifbw));
    m_scpi.sendEvent(interface(), formatString(":VNA:ACQ:IFBW %.0f", ifbw));
}

void
XLibreVNASCPI::onStartFreqChanged(const Snapshot &shot, XValueNodeBase *) {
    XScopedLock<XInterface> lock( *interface());
    m_scpi.sendEvent(interface(),
        formatString(":VNA:FREQ:START %.0f", (double)shot[ *startFreq()] * 1e6));
    rearrangeIFBW();
}
void
XLibreVNASCPI::onStopFreqChanged(const Snapshot &shot, XValueNodeBase *node) {
    XScopedLock<XInterface> lock( *interface());
    m_scpi.sendEvent(interface(),
        formatString(":VNA:FREQ:STOP %.0f", (double)shot[ *stopFreq()] * 1e6));
    rearrangeIFBW();
}
void
XLibreVNASCPI::onPointsChanged(const Snapshot &shot, XValueNodeBase *) {
    XScopedLock<XInterface> lock( *interface());
    m_scpi.sendEvent(interface(),
        formatString(":VNA:ACQ:POINTS %s", shot[ *points()].to_str().c_str()));
    rearrangeIFBW();
}
void
XLibreVNASCPI::onAverageChanged(const Snapshot &shot, XValueNodeBase *) {
    m_scpi.sendEvent(interface(),
        formatString(":VNA:ACQ:AVG %u", (unsigned int)shot[ *average()]));
}
void
XLibreVNASCPI::onPowerChanged(const Snapshot &shot, XValueNodeBase *) {
    m_scpi.sendEvent(interface(),
        formatString(":VNA:STIM:LVL %.0f", (double)shot[ *power()]));
}
void
XLibreVNASCPI::getMarkerPos(unsigned int num, double &x, double &y) {
    double re, im;
    switch(num) {
    case 0:
        interface()->query(":VNA:TRAC:MINA? S11");
        if(interface()->scanf("%lf,%lf,%lf", &x, &re, &im) != 3)
            throw XInterface::XConvError(__FILE__, __LINE__);
        break;
    case 1:
        interface()->query(":VNA:TRAC:MAXA? S11");
        if(interface()->scanf("%lf,%lf,%lf", &x, &re, &im) != 3)
            throw XInterface::XConvError(__FILE__, __LINE__);
        break;
    default:
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    }
    x *= 1e-6; //[MHz]
    y = 10 * std::log10(re*re + im*im);
}
//! How often a running sweep's position is read.  It bounds both how many of
//! the next sweep's points a record carries, and how late an edge is seen.
static constexpr unsigned int SWEEP_POLL_MS = 10;

void
XLibreVNASCPI::singleSweep() {
    //SINGLE TRUE starts a sweep, and the GUI stops the device once its
    //average is full; FIN? says when.
    m_scpi.sendEvent(interface(), ":VNA:ACQ:SINGLE TRUE");
    XTime started{XTime::now()};
    while (XTime::now() - started < 1.0) {
        msecsleep(100);
        interface()->query(":VNA:ACQ:FIN?");
        if(interface()->toStr() == "ERROR\n")
            throw XInterface::XConvError(__FILE__, __LINE__);
        if(interface()->toStr() == "TRUE\n")
            break;
    }
}
void
XLibreVNASCPI::oneSweep() {
    //Unset unless a sweep read as it ran is returned below; a sweep asked for
    //starts after the loop's own time, which is then right as it is.
    m_acquisitionStarted = {};
    //Nothing before GUI 1.6.5 says where a running sweep is, so there the
    //only way to know one has completed is to ask for one.
    if( !m_scpi.atLeast(1, 6, 5)) {
        singleSweep();
        return;
    }
    //From 1.6.5 the sweep runs on and is read as it goes.  Asking for a single
    //sweep per record restarts the device every time -- the GUI stops it once
    //the average is full, and SINGLE TRUE starts it again with the average
    //reset -- which was both the slow part and the hard use of the
    //instrument (user).
    if( !m_continuous) {
        m_scpi.sendEvent(interface(), ":VNA:ACQ:SINGLE FALSE");
        m_continuous = true;
        m_lastSweepFreq = -1.0;
        //Edges from before are of another sweep, or of none at all.
        m_lastSweepEdge = {};
        m_sweepPeriod = 0.0;
    }
    //One record per completed sweep, never more: a consumer that discards
    //"the next record" to be rid of data taken while something moved -- the
    //Auto LC Tuner does exactly that after each motor step -- must be
    //discarding a sweep, not whatever arrived within one poll.  FREQ? is the
    //frequency of the point the GUI received last, so it drops back the moment
    //the next sweep begins: that is the edge between two sweeps.  The record
    //then fetched carries the new sweep's first few points; the shorter the
    //poll, the fewer.
    bool moved = false;
    XTime started{XTime::now()};
    while (XTime::now() - started < 1.0) {
        interface()->query(":VNA:ACQ:FREQ?");
        double freq = interface()->toDouble();
        bool wrapped = false;
        if(m_lastSweepFreq >= 0.0) {
            wrapped = (freq < m_lastSweepFreq);
            moved = moved || (freq != m_lastSweepFreq);
        }
        m_lastSweepFreq = freq;
        if(wrapped) {
            //The sweep's own period, measured edge to edge rather than worked
            //out from the points and the IF bandwidth: what each point costs
            //beyond 1/IFBW (PLL settling, dwell, USB) is not documented, and
            //an estimate that came out short would claim the data newer than
            //it is -- the one direction a timestamp must not err in.  A missed
            //edge only lengthens the period, which errs the safe way.
            XTime edge = XTime::now();
            double period = m_lastSweepEdge.isSet() ? (edge - m_lastSweepEdge) : 0.0;
            m_lastSweepEdge = edge;
            if(period > 0.0)
                m_sweepPeriod = period;
            //A sweep ended, but right after a setting has changed the average
            //is still filling up -- FIN? is what said "done" in singleSweep()
            //too.  A changed setting also restarts the sweep, which looks like
            //an end here and is caught by the same test.  Without a period yet
            //(the first edge after starting) the start cannot be told, so that
            //sweep goes unrecorded.
            interface()->query(":VNA:ACQ:FIN?");
            if((interface()->toStr() == "TRUE\n") && (m_sweepPeriod > 0.0)) {
                //The average is a moving one over AVG sweeps, so the oldest data
                //in it began that many sweeps back.  Both edges are seen up to
                //a poll late, hence two polls' padding a sweep: early is safe.
                interface()->query(":VNA:ACQ:AVG?");
                unsigned int avg = std::max(1u, interface()->toUInt());
                m_acquisitionStarted = edge;
                m_acquisitionStarted -= avg * (m_sweepPeriod + 2e-3 * SWEEP_POLL_MS);
                return;
            }
        }
        msecsleep(SWEEP_POLL_MS);
    }
    if( !moved) {
        //Not one new point in a second: the sweep is stopped, or is not a
        //frequency sweep, or is so narrow that the six significant digits
        //FREQ? prints cannot tell its ends apart.  Waiting on would never
        //record again and never say why, so ask for a sweep instead; the next
        //call sets it running and looks again.
        if( !m_warnedSweepFrozen) {
            gWarnPrint(getLabel() + i18n(": sweep position unreadable, using single sweeps."));
            m_warnedSweepFrozen = true;
        }
        m_continuous = false;
        singleSweep();
        return;
    }
    //Let the loop look at terminated; the edge is still found on the next
    //call, since the last frequency seen is kept.
    throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
}
XTime
XLibreVNASCPI::acquisitionStarted(const XTime &polled) {
    return m_acquisitionStarted.isSet() ? m_acquisitionStarted : polled;
}
void
XLibreVNASCPI::startContSweep() {
    m_scpi.sendEvent(interface(), ":VNA:ACQ:SINGLE FALSE");
}
void
XLibreVNASCPI::acquireTrace(shared_ptr<RawData> &writer, unsigned int ch) {
    XScopedLock<XInterface> lock( *interface());
    interface()->query(":VNA:TRAC:LIST?");
    std::vector<std::string> traces;
    std::stringstream ss{&interface()->buffer()[0]};
    std::string buf;
    while (std::getline(ss, buf, ',')) {
        traces.push_back(buf); //"S11" and so on
    }
    if(ch >= traces.size())
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);

    interface()->query(":VNA:ACQ:POINTS?");
    writer->push((uint32_t)interface()->toUInt());
    interface()->query(":VNA:FREQ:START?");
    writer->push((double)interface()->toDouble());
    interface()->query(":VNA:FREQ:STOP?");
    writer->push((double)interface()->toDouble());

    auto tr = traces[ch];
    writer->push((uint32_t)tr.size()); //3
    writer->insert(writer->end(), tr.begin(), tr.end()); //"S11" or trace name
    interface()->queryf(":VNA:TRAC:DATA? %s", tr.c_str());
    writer->push((uint32_t)interface()->buffer().size());
    writer->insert(writer->end(),
                     interface()->buffer().begin(), interface()->buffer().end());
}
void
XLibreVNASCPI::convertRaw(RawDataReader &reader, Transaction &tr) {
    const Snapshot &shot(tr);
    uint32_t samples = reader.pop<uint32_t>();
    double start = reader.pop<double>() * 1e-6; //[MHz]
    double stop = reader.pop<double>() * 1e-6; //[MHz]
    double df = (stop - start) / (samples - 1);
    tr[ *this].m_startFreq = start;
    tr[ *this].m_freqInterval = df;
    tr[ *this].trace_().resize(samples);

    ssize_t cnt = reader.pop<uint32_t>();
    std::string trace{reader.popIterator(), reader.popIterator() + cnt};
    reader.popIterator() += cnt;

    ssize_t size = reader.pop<uint32_t>();
    std::stringstream ss{std::string(reader.popIterator(), reader.popIterator() + size)};
    reader.popIterator() += size;
    std::string buf;
    unsigned int i = 0;
    while (std::getline(ss, buf, ']')) { //sequence of [*,*,*],
        if(buf[0] == '\n')
            break;
        if(buf[0] == ',')
            buf = buf.substr(1);
        double x, re, im;
        if(sscanf(buf.c_str(), "[%lf,%lf,%lf", &x, &re, &im) != 3)
            throw XInterface::XConvError(__FILE__, __LINE__);
        //Checked before the write, not after it: the trace can hold more points
        //than POINTS? said a moment earlier, if the count is raised between the
        //two queries, and writing first went one element past the end.
        if(i >= samples)
            throw XInterface::XConvError(__FILE__, __LINE__);
        tr[ *this].trace_()[i++] = std::complex<double>(re, im);
    }
}
