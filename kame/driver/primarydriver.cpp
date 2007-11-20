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
    XTime time_recorded = time_recorded_org;
	bool skipped = false;
    for(unsigned int i = 0;; i++) {
    	if(tryStartRecording())
    		break;
    	msecsleep(i*10);
    	if(i > 30) {
    		gErrPrint(formatString(KAME::i18n(
    				"Dead lock deteceted on %s. Operation canceled.\nReport this bug to author(s)."),
    				getName().c_str()));
    		return;
    	}
    }
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
}

