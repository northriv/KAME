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
#include "primarydriver.h"
#include <klocale.h>

XThreadLocal<std::vector<char> > XPrimaryDriver::s_tlRawData;
XThreadLocal<XPrimaryDriver::RawData_it> XPrimaryDriver::s_tl_pop_it;

XPrimaryDriver::XPrimaryDriver(const char *name, bool runtime, 
							   const shared_ptr<XScalarEntryList> &scalarentries,
							   const shared_ptr<XInterfaceList> &interfaces,
							   const shared_ptr<XThermometerList> &thermometers,
							   const shared_ptr<XDriverList> &drivers) :
    XDriver(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
}

void
XPrimaryDriver::finishWritingRaw(
    const XTime &time_awared, const XTime &time_recorded_org)
{
	if(tryRecord(time_awared, time_recorded_org))
		return;
	dbgPrint(formatString("%s: raw writing is delegated.", getLabel().c_str()));
	shared_ptr<XThread<XPrimaryDriver> > thread(new XThread<XPrimaryDriver>(shared_from_this(), &XPrimaryDriver::delegatedRawWriting));
	atomic_shared_ptr<DelegatedWriteRequest> reqnull, 
		req(new DelegatedWriteRequest(thread, rawData(), time_awared, time_recorded_org));
	for(;;) {
		if(req.compareAndSwap(reqnull, m_delegatedWriteRequest))
				break;
		msecsleep(5);
	}
	thread->resume();
}
bool XPrimaryDriver::tryRecord(
		const XTime &time_awared, const XTime &time_recorded_org) 
{
	if(!tryStartRecording())
		return false;
    XTime time_recorded = time_recorded_org;
	bool skipped = false;
    if(time_recorded) {
	    *s_tl_pop_it = rawData().begin();
	    try {
	        analyzeRaw();
	    }
	    catch (XSkippedRecordError&) {
	    	skipped = true;
	    }
	    catch (XRecordError& e) {
			time_recorded = XTime(); //record is invalid
			e.print(getLabel() + ": " + KAME::i18n("Record Error, because "));
	    }
    }
    if(skipped)
    	abortRecordingNReadLock();
	else {
	    finishRecordingNReadLock(time_awared, time_recorded);
	}
    visualize();
    readUnlockRecord();
    return true;
}
void *XPrimaryDriver::delegatedRawWriting(const atomic<bool> &terminated)
{
//	while(!terminated)
//	{
//		if(terminated) break;
		atomic_shared_ptr<DelegatedWriteRequest> req;
		req.swap(m_delegatedWriteRequest);
//		if(!req)
//			break;
		clearRaw();
		std::copy(req->raw_data.begin(), req->raw_data.end(), rawData().begin());
		for(;;) {
			if(tryRecord(req->time_awared, req->time_recorded_org))
				break;
			msecsleep(1);
		}
//	}
    return NULL;
}
