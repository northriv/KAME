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
	while(m_delegatedWriteThread)
		msecsleep(5);
	if(tryStartRecording()) {
		raw2record(time_awared, time_recorded_org);
	    readUnlockRecord();
	    return;
	}

	dbgPrint(formatString("%s: raw writing is delegated.", getLabel().c_str()));
	m_delegatedWriteThread.reset(new XThread<XPrimaryDriver>(
			shared_from_this(), &XPrimaryDriver::delegatedRawWriting));
	
	m_pDelegatedWriteData = &rawData();
	m_delegatedWriteTimeAwared = time_awared;
	m_delegatedWriteTimeRecordedOrg = time_recorded_org;
	m_delegatedWriteThread->resume();
}
void XPrimaryDriver::raw2record(
		const XTime &time_awared, const XTime &time_recorded_org) 
{
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
}
void *XPrimaryDriver::delegatedRawWriting(const atomic<bool> &/*terminated*/)
{
	clearRaw();
	for(;;) {
		if(tryStartRecording())
			break;
		msecsleep(5);
	}
	std::copy(m_pDelegatedWriteData->begin(), m_pDelegatedWriteData->end(), rawData().begin());
	raw2record(m_delegatedWriteTimeAwared, m_delegatedWriteTimeRecordedOrg);
	m_delegatedWriteThread.reset();
    readUnlockRecord();
	dbgPrint(formatString("%s: raw writing delegated done.", getLabel().c_str()));
    return NULL;
}
