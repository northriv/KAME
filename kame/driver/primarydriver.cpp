/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifdef USE_PYBIND11
    #include <pybind11/pybind11.h>
#endif

#include "primarydriver.h"
#include <chrono>
#include <memory>

XPrimaryDriver::XPrimaryDriver(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XDriver(name, runtime, tr_meas, meas) {
}

void
XPrimaryDriver::finishWritingRaw(const shared_ptr<const RawData> &rawdata,
    const XTime &time_awared, const XTime &time_recorded_org) {

    XTime time_recorded = time_recorded_org;
    XKameError err;
    bool skipped = false;
    // Bounds the WAITING this call may do, at EVERY priority: past ~20 ms a
    // stalled record starts to distort the measurement whether or not the
    // acquisition thread is realtime.  See downstreamWaitBudgetUS() for the
    // default and for the throughput this trades away.
    //
    // One guard covers the whole call because the budget is an absolute
    // thread-local limit, not a per-scope duration.  On a HIGHEST thread it is
    // inert over the commit -- HIGHEST leaves the negotiator's round loop before
    // sleeping -- and binds the moment ScopedDemoteRealtime drops the priority to
    // NORMAL for the marked-message dispatch below and for
    // visualize()/onVisualization after, which is exactly where an acquisition
    // loop's period would otherwise be exposed.  On a NORMAL thread it binds
    // throughout, including the record commit itself.
    std::unique_ptr<Transactional::ScopedWaitBudget> _downstream_budget;
    if(unsigned int _b = downstreamWaitBudgetUS())
        _downstream_budget.reset(new Transactional::ScopedWaitBudget((int64_t)_b));
    // Telemetry only — see the counters' doc block in primarydriver.h.  Two
    // steady_clock reads and one comparison per record, against a commit that
    // already does a tree walk; nothing is printed here.
    const auto _t0 = std::chrono::steady_clock::now();
    Snapshot shot = iterate_commit([=, &time_recorded, &err, &skipped](Transaction &tr){
        //Reset reference-captured state on every CAS retry: iterate_commit
        //re-invokes this closure on each retry and these variables outlive the
        //lambda (they are read after commit for onVisualization/err.print), so
        //a first-attempt skip/error must not latch and suppress record() on a
        //subsequent successful retry.
        skipped = false;
        time_recorded = time_recorded_org;
        err = XKameError();
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
    {
        const auto _dt = (std::uint64_t)
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - _t0).count();
        m_recordCommits.fetch_add(1, std::memory_order_relaxed);
        if(_dt > SLOW_RECORD_COMMIT_NS)
            m_slowRecordCommits.fetch_add(1, std::memory_order_relaxed);
        auto prev = m_maxRecordCommitNS.load(std::memory_order_relaxed);
        while(_dt > prev &&
              !m_maxRecordCommitNS.compare_exchange_weak(prev, _dt,
                  std::memory_order_relaxed, std::memory_order_relaxed)) {}
    }
    if(err.msg().length())
        err.print(getLabel() + ": ");
    // Realtime ends with the record.  Everything below is downstream work --
    // visualize() touches graphs, and the onVisualization listeners are other
    // people's code -- and at HIGHEST it would inherit an exemption from
    // politeness it has no claim to, on paths that widen scope (a graph object
    // snapshots its plot; the secondary-driver chain snapshots the whole driver
    // list).  Two acquisition threads doing that concurrently put two realtime
    // threads on one Linkage, which is the invariant HIGHEST rests on.
    //
    // The onRecord listeners are NOT covered here: XDriver::record() marks the
    // talker, so they are dispatched inside the commit above.  kamestm demotes
    // there, at Transaction::finalizeCommitment's messaging loop.
    //
    // Demotes HIGHEST only.  A NORMAL driver is unaffected, and a lowprio
    // committer must not be raised.
    Transactional::ScopedDemoteRealtime _no_realtime_downstream;
    try {
        visualize(shot);
        trans( *this).onVisualization().talk(shot, !skipped, this);
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
