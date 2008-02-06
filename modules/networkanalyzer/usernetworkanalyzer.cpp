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
#include "usernetworkanalyzer.h"
#include "charinterface.h"
#include "xwavengraph.h"

REGISTER_TYPE(XDriverList, HP8711, "HP/Agilent 8711/8712/8713/8714 Network Analyzer");
REGISTER_TYPE(XDriverList, AgilentE5061, "Agilent E5061/E5062 Network Analyzer");
//---------------------------------------------------------------------------
XAgilentNetworkAnalyzer::XAgilentNetworkAnalyzer(const char *name, bool runtime,
		   const shared_ptr<XScalarEntryList> &scalarentries,
		   const shared_ptr<XInterfaceList> &interfaces,
		   const shared_ptr<XThermometerList> &thermometers,
		   const shared_ptr<XDriverList> &drivers) :
	XCharDeviceDriver<XNetworkAnalyzer>(name, runtime, scalarentries, interfaces, thermometers, drivers) {
	const char *cand[] = {"3", "5", "11", "21", "51", "101", "201", "401", "801", "1601", ""};
	for(const char **it = cand; strlen(*it); it++) {
		points()->add(*it);
	}
}

void
XAgilentNetworkAnalyzer::open() throw (XInterface::XInterfaceError &)
{
	interface()->query("SENS:FREQ:START?");
	startFreq()->value(interface()->toDouble() / 1e6);
	interface()->query("SENS:FREQ:STOP?");
	stopFreq()->value(interface()->toDouble() / 1e6);
	interface()->query("SENS:AVER:STAT?");
	if(interface()->toUInt() == 0) {
		average()->value(1);
	}
	else {
		interface()->query("SENS:AVER:COUNT?");
		average()->value(interface()->toUInt());
	}
	interface()->query("SENS:SWE:POIN?");
	points()->str(formatString("%u", interface()->toUInt()));
	interface()->send("SENS:SWE:TIME:AUTO OFF");
	interface()->query("SENS:SWE:TIME?");
	double swet = interface()->toDouble();
	interface()->sendf(":SENS:SWE:TIME %f S", std::min(1.0, std::max(0.3, swet)));
	interface()->send("ABOR;INIT:CONT OFF");
	
	start();
}
void 
XAgilentNetworkAnalyzer::onStartFreqChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->sendf("SENS:FREQ:START %f MHZ", (double)*startFreq());
}
void 
XAgilentNetworkAnalyzer::onStopFreqChanged(const shared_ptr<XValueNodeBase> &) {
	interface()->sendf("SENS:FREQ:STOP %f MHZ", (double)*stopFreq());
}
void
XAgilentNetworkAnalyzer::onAverageChanged(const shared_ptr<XValueNodeBase> &) {
	unsigned int avg = *average();
	if(avg >= 2)
		interface()->sendf("SENS:AVER:CLEAR;STAT ON;COUNT %u", avg);
	else
		interface()->send("SENS:AVER:STAT OFF");
}
void
XAgilentNetworkAnalyzer::onPointsChanged(const shared_ptr<XValueNodeBase> &) {	
	interface()->sendf("SENS:SWE:POIN %s", points()->to_str().c_str());
}
void
XAgilentNetworkAnalyzer::getMarkerPos(unsigned int num, double &x, double &y) {
	XScopedLock<XInterface> lock(*interface());
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
XAgilentNetworkAnalyzer::acquireTrace(unsigned int ch) {
	XScopedLock<XInterface> lock(*interface());
	if(ch >= 2)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	interface()->queryf("SENS%u:STAT?", ch + 1u);
	if(interface()->toInt() != 1)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);		
	interface()->queryf("SENS%u:FREQ:START?", ch + 1u);
	double start = interface()->toDouble() / 1e6;
	push(start);
	interface()->queryf("SENS%u:FREQ:STOP?", ch + 1u);
	double stop = interface()->toDouble() / 1e6;
	push(stop);
	interface()->queryf("SENS%u:SWE:POIN?", ch + 1u);
	unsigned int len = interface()->toUInt();
	push(len);
	acquireTraceData(ch, len);
	rawData().insert(rawData().end(), 
					 interface()->buffer().begin(), interface()->buffer().end());
}
void
XAgilentNetworkAnalyzer::convertRaw() throw (XRecordError&) {
	double start = pop<double>();
	double stop = pop<double>();
	unsigned int samples = pop<unsigned int>();
	m_startFreqRecorded = start;
	char c = pop<char>();
	if (c != '#') throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	char buf[11];
	buf[0] = pop<char>();
	unsigned int len;
	sscanf(buf, "%1u", &len);
	for(unsigned int i = 0; i < len; i++) {
		buf[i] = pop<char>();
	}
	buf[len] = '\0';
	sscanf(buf, "%u", &len);
	m_freqIntervalRecorded = (stop - start) / (samples - 1);
	m_traceRecorded.resize(samples);

	convertRawBlock(len);
}

void
XHP8711::acquireTraceData(unsigned int ch, unsigned int len) {
	interface()->send("FORM:DATA REAL,32;BORD SWAP");
	interface()->sendf("TRAC? CH%uFDATA", ch + 1u);
	interface()->receive(len * sizeof(float) + 12);
}
void
XHP8711::convertRawBlock(unsigned int len) throw (XRecordError&) {
	unsigned int samples = m_traceRecorded.size();
	if(len / sizeof(float) < samples)
		throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	if(len / sizeof(float) > samples)
		throw XRecordError(KAME::i18n("Select scalar plot."), __FILE__, __LINE__);
	for(unsigned int i = 0; i < samples; i++) {
		m_traceRecorded[i] = pop<float>();
	}
}

void
XAgilentE5061::acquireTraceData(unsigned int ch, unsigned int len) {
	interface()->send("FORM:DATA REAL32;BORD SWAP");
	interface()->sendf("CALC%u:DATA:FDAT?", ch + 1u);
	interface()->receive(len * sizeof(float) * 2 + 12);
}
void
XAgilentE5061::convertRawBlock(unsigned int len) throw (XRecordError&) {
	unsigned int samples = m_traceRecorded.size();
	if(len / sizeof(float) < samples * 2)
		throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	for(unsigned int i = 0; i < samples; i++) {
		m_traceRecorded[i] = pop<float>();
		pop<float>();
	}
}
