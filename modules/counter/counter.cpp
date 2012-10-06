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
#include "counter.h"

#include "interface.h"
#include "analyzer.h"

XCounter::XCounter(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas) {
}

void
XCounter::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
	for(unsigned int ch = 0; ch < m_entries.size(); ch++) {
		m_entries[ch]->value(tr, reader.pop<double>());
	}
}
void
XCounter::visualize(const Snapshot &shot) {
}
void
XCounter::createChannels(Transaction &tr_meas, const shared_ptr<XMeasure> &meas,
    const char **channel_names) {
  shared_ptr<XScalarEntryList> entries(meas->scalarEntries());

  for(int i = 0; channel_names[i]; i++) {
	    shared_ptr<XScalarEntry> entry(create<XScalarEntry>(
	    	channel_names[i], false,
	       dynamic_pointer_cast<XDriver>(shared_from_this()), "%.8g"));
	     m_entries.push_back(entry);
	     entries->insert(tr_meas, entry);
    }
}
void *
XCounter::execute(const atomic<bool> &terminated) {
    while( !terminated) {
		msecsleep(50);

		shared_ptr<RawData> writer(new RawData);
		// try/catch exception of communication errors
		try {
			unsigned int num = m_entries.size();
			for(unsigned int ch = 0; ch < num; ch++)
				writer->push((double)getLevel(ch));
		}
		catch (XKameError &e) {
			e.print(getLabel());
			continue;
		}

		finishWritingRaw(writer, XTime::now(), XTime::now());
	}
	return NULL;
}

#include "charinterface.h"
#include "analyzer.h"

REGISTER_TYPE(XDriverList, MutohCounterNPS, "Mutoh Digital Counter NPS");

#define STX "\x02"

XMutohCounterNPS::XMutohCounterNPS(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XCounter>(name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = {"Ch0", 0L};
	createChannels(ref(tr_meas), meas, channels_create);

	interface()->setSerialParity(XCharInterface::PARITY_EVEN);
	interface()->setSerialBaudRate(19200);
	interface()->setSerial7Bits(true);
	interface()->setEOS("\x03"); //ETX
}

double
XMutohCounterNPS::getLevel(unsigned int ch) {
	XScopedLock<XInterface> lock( *interface());
	interface()->query(STX "00F102");
	int fun2;
	if(interface()->scanf(STX "00F202%d", &fun2) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	interface()->query(STX "00F105");
	int fun5;
	if(interface()->scanf(STX "00F205%d", &fun5) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);

	interface()->query(STX "00P1");
	int x;
	if(interface()->scanf(STX "00P2%d", &x) != 1)
		throw XInterface::XConvError(__FILE__, __LINE__);
	double z;
	switch(fun5) {
	case 00:
	case 01:
		z = x / pow(10.0, fun2 % 10);
		break;
	case 10:
	case 11:
	case 12:
		z = (x / 100) + ((x > 0) ? 1 : -1) * (abs(x) % 100) / 60.0;
		break;
	case 16:
	case 17:
	case 18:
	case 19:
		z = (x / 10000) + ((x > 0) ? 1 : -1) * ((abs(x) % 10000) / 100 + (abs(x) % 100) / 60.0) / 60.0;
		break;
	case 13:
	case 14:
	case 15:
		z = x / 10000.0;
		break;
	case 50:
		z = x / 10;
		break;
	default:
		throw XInterface::XConvError(__FILE__, __LINE__);
	}
	return z;
}
