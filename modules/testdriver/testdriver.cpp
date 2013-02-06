/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#include "testdriver.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <qstatusbar.h>
#include "rand.h"

REGISTER_TYPE(XDriverList, TestDriver, "Test driver: random number generation");

XTestDriver::XTestDriver(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XDummyDriver<XPrimaryDriverWithThread>(name, runtime, ref(tr_meas), meas),
    m_entryX(create<XScalarEntry>("X", false, 
    	static_pointer_cast<XDriver>(shared_from_this()), "%.3g")),
    m_entryY(create<XScalarEntry>("Y", false,
    	static_pointer_cast<XDriver>(shared_from_this()), "%+.4f[K]")) {

	meas->scalarEntries()->insert(tr_meas, m_entryX);
	meas->scalarEntries()->insert(tr_meas, m_entryY);
}

void
XTestDriver::showForms() {
// impliment form->show() here
}
void
XTestDriver::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    // Since raw buffer is FIFO, use the same sequence of push()es for pop()s
	tr[ *this].m_x = reader.pop<double>();
	tr[ *this].m_y = reader.pop<double>();
	m_entryX->value(tr, tr[ *this].m_x);
	m_entryY->value(tr, tr[ *this].m_y);
}
void
XTestDriver::visualize(const Snapshot &shot) {
}

void *
XTestDriver::execute(const atomic<bool> &terminated) {
	while( !terminated) {
		msecsleep(10);
		double x = randMT19937() - 0.2;
		double y = randMT19937()- 0.2;

		shared_ptr<RawData> writer(new RawData);
		writer->push(x);
		writer->push(y);
		finishWritingRaw(writer, XTime::now(), XTime::now());
	}
	return NULL;
}

