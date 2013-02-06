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
#include "driver.h"
#include "interface.h"

DECLARE_TYPE_HOLDER(XDriverList)

XDriverList::XDriverList(const char *name, bool runtime,
						 const shared_ptr<XMeasure> &measure) :
	XCustomTypeListNode<XDriver>(name, runtime),
	m_measure(measure) {
}

shared_ptr<XNode>
XDriverList::createByTypename(const XString &type, const XString& name) {
	shared_ptr<XMeasure> measure(m_measure.lock());
	shared_ptr<XNode> ptr;
	for(Transaction tr( *measure);; ++tr) {
		ptr = creator(type)
			(name.c_str(), false, ref(tr), measure);
		if(ptr)
			if( !insert(tr, ptr))
				continue;
		if(tr.commit())
			break;
	}
    return ptr;
}

XDriver::XBufferUnderflowRecordError::XBufferUnderflowRecordError(const char *file, int line) : 
    XRecordError(i18n("Buffer Underflow."), file, line) {}

XDriver::XDriver(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XNode(name, runtime) {
}

void
XDriver::record(Transaction &tr,
	const XTime &time_awared, const XTime &time_recorded) {
    tr[ *this].m_awaredTime = time_awared;
    tr[ *this].m_recordTime = time_recorded;
    tr.mark(tr[ *this].onRecord(), this);
}
