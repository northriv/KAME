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
}

void
XHP8711::open() throw (XInterface::XInterfaceError &)
{
	interface()->query("SENS:FREQ:START?");
	startFreq()->value(interface()->toDouble() / 1e6);
	interface()->query("SENS:FREQ:STOP?");
	stopFreq()->value(interface()->toDouble() / 1e6);

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
XHP8711::getMarkerPos(unsigned int num, double &x, double &y) {
	XScopedLock<XInterface> lock(*interface());
	if(num >= 8)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	try {
		interface()->queryf("CALC:MARK%u:X?", num + 1u);
		x = interface()->toDouble();
		interface()->queryf("CALC:MARK%u:Y?", num + 1u);
		y = interface()->toDouble();
	}
	catch (XInterface::XConvError&) {
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	}
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
	interface()->queryf("SENS%u:FREQ:START?", ch);
	double start = interface()->toDouble() / 1e6;
	push(start);
	interface()->queryf("SENS%u:FREQ:STOP?", ch);
	double stop = interface()->toDouble() / 1e6;
	push(stop);
	interface()->queryf("SENS%u:SWE:POIN?", ch);
	unsigned int len = interface()->toUInt();
	interface()->send("FORM:DATA INT,16;BORD NORM");
	interface()->sendf("TRAC? CH%uFDATA", ch);
	interface()->receive(len * 2 + 12);
	rawData().insert(rawData().end(), 
					 interface()->buffer().begin(), interface()->buffer().end());
}
void
XHP8711::convertRaw() throw (XRecordError&) {
	double start = pop<double>();
	double stop = pop<double>();
	m_startFreqRecorded = start;
	char c = pop<char>();
	if (c != '#') throw XBufferUnderflowRecordError(__FILE__, __LINE__);
	char buf[11];
	buf[0] = pop<char>();
	unsigned int len;
	scanf("%1u", &len);
	for(unsigned int i = 0; i < len; i++) {
		buf[i] = pop<char>();
	}
	buf[len] = '\0';
	sscanf(buf, "%u", &len);
	m_traceRecorded.resize(len);
	m_freqIntervalRecorded = (stop - start) / (len - 1);
	for(unsigned int i = 0; i < len; i++) {
		m_traceRecorded[i] = pop<unsigned short>();
	}
}
