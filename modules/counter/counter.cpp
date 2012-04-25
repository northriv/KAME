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
    XPrimaryDriver(name, runtime, ref(tr_meas), meas) {
}

void
XCounter::start() {
	m_thread.reset(new XThread<XLevelMeter>(shared_from_this(), &XLevelMeter::execute));
	m_thread->resume();
}
void
XCounter::stop() {
    if(m_thread) m_thread->terminate();
}

void
XCounter::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
	for(unsigned int ch = 0; ch < tr[ *this].m_levels.size(); ch++) {
		tr[ *this].m_levels[ch] = reader.pop<double>();
		m_entries[ch]->value(tr, tr[ *this].m_levels[ch]);
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
	       dynamic_pointer_cast<XDriver>(shared_from_this()), "%.4g"));
	     m_entries.push_back(entry);
	     entries->insert(tr_meas, entry);
    }
  for(Transaction tr( *this);; ++tr) {
	tr[ *this].m_levels.resize(m_entries.size());
  	if(tr.commit()) {
  		break;
  	}
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
	afterStop();
	return NULL;
}

REGISTER_TYPE(XDriverList, MutohCounter, "Mutoh Digital Counter NPS");

XMutohCounterNPS::XMutohCounterNPS(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XCounter>(name, runtime, ref(tr_meas), meas) {
	const char *channels_create[] = {"Ch0", 0L};
	createChannels(ref(tr_meas), meas, channels_create);

	interface()->setSerialParity(XCharInterface::PARITY_EVEN);
	interface()->setSerial7Bits(true);
	interface()->setEOS('\x03'); //ETX
}

double
XMutohCounterNPS::getLevel(unsigned int ch) {
	XScopedLock<XInterface> lock( *interface());
	char req[] = "\x0200F102"; //1st char is STX
	interface()->send(req);
	int fun2;
	if(interface()->scanf("00F202%d", &fun2) != 1)
		throw XInterface::XInterfaceError(__FILE__, __LINE__);

	char req[] = "\x0200F105"; //1st char is STX
	interface()->send(req);
	int fun5;
	if(interface()->scanf("00F205%d", &fun5) != 1)
		throw XInterface::XInterfaceError(__FILE__, __LINE__);

	char req[] = "\x0200P1"; //1st char is STX
	interface()->send(req);
	int x;
	if(interface()->scanf("00P2%d", &x) != 1)
		throw XInterface::XInterfaceError(__FILE__, __LINE__);
	double z;
	switch(fun5) {
	case 00:
	case 01:
		z = x / pow(10.0, fun2 % 10);
		break;
	case 10:
	case 11:
	case 12:
	case 16:
	case 17:
	case 18:
	case 19:
		z = (x / 10000) + ((x % 10000) / 100 + (x % 100) / 60.0) / 60.0;
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
		throw XInterface::XInterfaceError(__FILE__, __LINE__);
	}
	return z;
}
