/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifdef USE_PYBIND11
    #include <pybind11/pybind11.h>
#endif

#include "primarydriver.h"

XPrimaryDriver::XPrimaryDriver(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XDriver(name, runtime, tr_meas, meas) {
}

void
XPrimaryDriver::finishWritingRaw(const shared_ptr<const RawData> &rawdata,
    const XTime &time_awared, const XTime &time_recorded_org) {

    XTime time_recorded = time_recorded_org;
    XKameError err;
    Snapshot shot = iterate_commit([=, &time_recorded, &err](Transaction &tr){
		bool skipped = false;
        if(time_recorded.isSet()) {
			try {
				RawDataReader reader( *rawdata);
				tr[ *this].m_rawData = rawdata;
                analyzeRaw(reader, tr);
			}
#ifdef USE_PYBIND11
            catch (pybind11::error_already_set& e) {
                pybind11::gil_scoped_acquire guard;
                if(e.matches(PyExc_InterruptedError)) {
                    skipped = true;
                    err = XSkippedRecordError("", __FILE__, __LINE__);
                }
                else if(e.matches(PyExc_ValueError)) {
                    time_recorded = XTime(); //record is invalid
                    err = XRecordError(e.what(), __FILE__, __LINE__);
                }
                else {
                    gErrPrint(i18n("Python error: ") + e.what());
                    return;
                }
            }
#endif
//            catch (std::runtime_error &e) {
//                gErrPrint(std::string("Python KAME binding error: ") + e.what());
//                return;
//            }
            catch (XSkippedRecordError& e) {
				skipped = true;
				err = e;
			}
			catch (XRecordError& e) {
				time_recorded = XTime(); //record is invalid
				err = e;
			}
		}
		if( !skipped)
			record(tr, time_awared, time_recorded);
    });
    if(err.msg().length())
        err.print(getLabel() + ": ");
    try {
        visualize(shot);
    }
#ifdef USE_PYBIND11
    catch (pybind11::error_already_set& e) {
        pybind11::gil_scoped_acquire guard;
        gErrPrint(i18n("Python error: ") + e.what());
    }
#endif
    catch (std::runtime_error &e) {
        gErrPrint(std::string("Python KAME binding error: ") + e.what());
    }
}
