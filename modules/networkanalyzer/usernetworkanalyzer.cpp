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
#include <klocale.h>

REGISTER_TYPE(XDriverList, HP8711, "HP/Agilent 8711/8712/8713/8714 Network Analyzer");

//---------------------------------------------------------------------------
XHP8711::XHP8711(const char *name, bool runtime,
		   const shared_ptr<XScalarEntryList> &scalarentries,
		   const shared_ptr<XInterfaceList> &interfaces,
		   const shared_ptr<XThermometerList> &thermometers,
		   const shared_ptr<XDriverList> &drivers) :
	XCharDeviceDriver<XNetworkAnalyzer>(name, runtime, scalarentries, interfaces, thermometers, drivers) {
	const char *points[] = {"3", "5", "11", "21", "51", "101", "201", "401", "801", "1601", ""};
	for(const char **it = points; strlen(*it); it++) {
		points()->add(*it);
	}
}

void
XHP8711::open() throw (XInterface::XInterfaceError &)
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
	interface()->send("ABOR;INIT:CONT OFF");
	
	start();
}
void 
XHP8711::onStartFreqChanged(const shared_ptr<XValueNodeBase> &) {
	XScopedLock<XInterface> lock(*interface());
	interface()->sendf("ABOR;SENS:FREQ:START %f MHZ;*WAI", (double)*startFreq());
	interface()->query("*OPC?");
}
void 
XHP8711::onStopFreqChanged(const shared_ptr<XValueNodeBase> &) {
	XScopedLock<XInterface> lock(*interface());
	interface()->sendf("ABOR;SENS:FREQ:STOP %f MHZ;*WAI", (double)*stopFreq());
	interface()->query("*OPC?");
}
void
XHP8711::onAverageChanged(const shared_ptr<XValueNodeBase> &) {
	XScopedLock<XInterface> lock(*interface());
	unsigned int avg = *average();
	if(avg >= 2)
		interface()->sendf("ABOR;SENS:AVER:CLEAR;STAT ON;COUNT %u;*WAI", avg);
	else
		interface()->send("ABOR;SENS:AVER:STAT OFF;*WAI");
	interface()->query("*OPC?");
}
void
XHP8711::onPointsChanged(const shared_ptr<XValueNodeBase> &) {	
	XScopedLock<XInterface> lock(*interface());
	interface()->sendf("ABOR;SENS:SWE:POIN %s;*WAI", points()->tostr());
	interface()->query("*OPC?");
}
void
XHP8711::getMarkerPos(unsigned int num, double &x, double &y) {
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
XHP8711::oneSweep() {
	interface()->query("INIT:IMM;*OPC?");
}
void
XHP8711::startContSweep() {
	interface()->send("INIT:CONT ON");
}
void
XHP8711::acquireTrace(unsigned int ch) {
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
	interface()->send("FORM:DATA REAL,32;BORD SWAP");
	interface()->sendf("TRAC? CH%uFDATA", ch + 1u);
	interface()->receive(len * sizeof(float) + 12);
	rawData().insert(rawData().end(), 
					 interface()->buffer().begin(), interface()->buffer().end());
}
void
XHP8711::convertRaw() throw (XRecordError&) {
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
	if(len / sizeof(float) < samples)
		throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	if(len / sizeof(float) > samples)
		throw XRecordError(KAME::i18n("Select scalar plot."), __FILE__, __LINE__);
	m_traceRecorded.resize(samples);
	m_freqIntervalRecorded = (stop - start) / (samples - 1);
	for(unsigned int i = 0; i < samples; i++) {
		m_traceRecorded[i] = pop<float>();
	}
}
