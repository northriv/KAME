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
#include "usernetworkanalyzer.h"
#include "charinterface.h"

REGISTER_TYPE(XDriverList, HP8711, "HP/Agilent 8711/8712/8713/8714 Network Analyzer");
REGISTER_TYPE(XDriverList, AgilentE5061, "Agilent E5061/E5062 Network Analyzer");
//---------------------------------------------------------------------------
XAgilentNetworkAnalyzer::XAgilentNetworkAnalyzer(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XCharDeviceDriver<XNetworkAnalyzer>(name, runtime, ref(tr_meas), meas) {
	const char *cand[] = {"3", "5", "11", "21", "51", "101", "201", "401", "801", "1601", ""};
	for(Transaction tr( *this);; ++tr) {
		for(const char **it = cand; strlen( *it); it++) {
			tr[ *points()].add( *it);
		}
		if(tr.commit())
			break;
	}
}

void
XAgilentNetworkAnalyzer::open() throw (XInterface::XInterfaceError &) {
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
	interface()->send("SENS:SWE:TIME:AUTO OFF");
	interface()->query("SENS:SWE:TIME?");
	double swet = interface()->toDouble();
	interface()->sendf(":SENS:SWE:TIME %f S", std::min(1.0, std::max(0.3, swet)));
	interface()->send("ABOR;INIT:CONT OFF");
	
	start();

	calOpen()->setUIEnabled(false);
	calShort()->setUIEnabled(false);
	calTerm()->setUIEnabled(false);
	calThru()->setUIEnabled(false);
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
	interface()->queryf("SENS%u:STAT?", ch + 1u);
	if(interface()->toInt() != 1)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);		
	interface()->queryf("SENS%u:FREQ:START?", ch + 1u);
	double start = interface()->toDouble() / 1e6;
	writer->push(start);
	interface()->queryf("SENS%u:FREQ:STOP?", ch + 1u);
	double stop = interface()->toDouble() / 1e6;
	writer->push(stop);
	interface()->queryf("SENS%u:SWE:POIN?", ch + 1u);
	unsigned int len = interface()->toUInt();
	writer->push(len);
	acquireTraceData(ch, len);
	writer->insert(writer->end(),
					 interface()->buffer().begin(), interface()->buffer().end());
}
void
XAgilentNetworkAnalyzer::convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
	double start = reader.pop<double>();
	double stop = reader.pop<double>();
	unsigned int samples = reader.pop<unsigned int>();
	tr[ *this].m_startFreq = start;
	char c = reader.pop<char>();
	if (c != '#') throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	char buf[11];
	buf[0] = reader.pop<char>();
	unsigned int len;
	sscanf(buf, "%1u", &len);
	for(unsigned int i = 0; i < len; i++) {
		buf[i] = reader.pop<char>();
	}
	buf[len] = '\0';
	sscanf(buf, "%u", &len);
	tr[ *this].m_freqInterval = (stop - start) / (samples - 1);
	tr[ *this].trace_().resize(samples);

	convertRawBlock(reader, tr, len);
}

void
XHP8711::acquireTraceData(unsigned int ch, unsigned int len) {
	interface()->send("FORM:DATA REAL,32;BORD SWAP");
	interface()->sendf("TRAC? CH%uFDATA", ch + 1u);
	interface()->receive(len * sizeof(float) + 12);
}
void
XHP8711::convertRawBlock(RawDataReader &reader, Transaction &tr,
	unsigned int len) throw (XRecordError&) {
	unsigned int samples = tr[ *this].trace_().size();
	if(len / sizeof(float) < samples)
		throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	if(len / sizeof(float) > samples)
		throw XRecordError(i18n("Select scalar plot."), __FILE__, __LINE__);
	for(unsigned int i = 0; i < samples; i++) {
		tr[ *this].trace_()[i] = reader.pop<float>();
	}
}

void
XAgilentE5061::acquireTraceData(unsigned int ch, unsigned int len) {
	interface()->send("FORM:DATA REAL32;BORD SWAP");
	interface()->sendf("CALC%u:DATA:FDAT?", ch + 1u);
	interface()->receive(len * sizeof(float) * 2 + 12);
}
void
XAgilentE5061::convertRawBlock(RawDataReader &reader, Transaction &tr,
	unsigned int len) throw (XRecordError&) {
	unsigned int samples = tr[ *this].trace_().size();
	if(len / sizeof(float) < samples * 2)
		throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	for(unsigned int i = 0; i < samples; i++) {
		tr[ *this].trace_()[i] = reader.pop<float>();
		reader.pop<float>();
	}
}
