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
//! Class XMeasure
//! The root node of KAME
//---------------------------------------------------------------------------
#ifndef measureH
#define measureH

#include "xnode.h"
#include "xnodeconnector.h"

class XCalibrationCurveList;
class XDriverList;
class XInterfaceList;
class XStatusPrinter;
class XDriverList;
class XScalarEntryList;
class XGraphList;
class XChartList;
class XCalibratedEntryList;
class XTextWriter;
class XRawStreamRecorder;
class XJournalRecorder;
class XRawStreamRecordReader;
class XRuby;
class XPython;
class XNodeBrowser;

/*! The root object of KAME.
 */
class DECLSPEC_KAME XMeasure : public XNode {
public:
	XMeasure(const char *name, bool runtime);
	virtual ~XMeasure();

	//! call me before loading a measurement file.
	void initialize();
	//! clean all drivers, thermometers.
	void terminate();
    //! terminate() and clean up script supports.
    //! call this before quiting, since script supports hold shared_ptr<XMeasure>.
    void terminate_all();
    //! stop all drivers.
	void stop();

	const shared_ptr<XCalibrationCurveList> &thermometers() const {return m_thermometers;}
	const shared_ptr<XDriverList> &drivers() const {return m_drivers;}
	const shared_ptr<XInterfaceList> &interfaces() const {return m_interfaces;}
	const shared_ptr<XScalarEntryList> &scalarEntries() const {return m_scalarEntries;}
	const shared_ptr<XGraphList> &graphs() const {return m_graphList;}
	const shared_ptr<XChartList> &charts() const {return m_chartList;}
    const shared_ptr<XCalibratedEntryList> &calibratedEntries() const {return m_calibratedEntryList;}
	const shared_ptr<XTextWriter> &textWriter() const {return m_textWriter;}
	const shared_ptr<XRawStreamRecorder> &rawStreamRecorder() const {return m_rawStreamRecorder;}
	//! What the run is called, how much of it is kept, and what it costs.
	//! \sa doc/design/PROVENANCE.md
	const shared_ptr<XJournalRecorder> &journal() const {return m_journal;}
	const shared_ptr<XRawStreamRecordReader> &rawStreamRecordReader() const {return m_rawStreamRecordReader;}

	//! Null unless the build has the Ruby interpreter (USE_RUBY).
	const shared_ptr<XRuby> &ruby() const {return m_ruby;}
#ifdef USE_PYBIND11
    const shared_ptr<XPython> &python() const {return m_python;}
#endif
    //for description made by python monitor.
    const shared_ptr<XStringNode> &pyInfoForNodeBrowser() const {return m_pyInfoForNodeBrowser;}
    shared_ptr<XNode> &lastPointedByNodeBrowser() {return m_lastPointedByNodeBrowser;}
private:
	//! Declared unconditionally ON PURPOSE.  This header is included by
	//! libkame and by every module, and only kame.pro runs the ruby-header
	//! detection -- so USE_RUBY is NOT uniform across the targets that see
	//! this class.  Putting a member behind it split sizeof(XMeasure) by 16
	//! bytes between the app and the modules, which aliased the modules'
	//! m_interfaces onto the app's m_drivers.  A forward-declared
	//! shared_ptr costs 16 bytes and pulls in no libruby; only the
	//! construction in measure.cpp is gated.
	shared_ptr<XRuby> m_ruby;
    shared_ptr<XPython> m_python;

	const shared_ptr<XCalibrationCurveList> m_thermometers;
	const shared_ptr<XScalarEntryList> m_scalarEntries;
	const shared_ptr<XGraphList> m_graphList;
	const shared_ptr<XChartList> m_chartList;
	const shared_ptr<XInterfaceList> m_interfaces;
	const shared_ptr<XDriverList> m_drivers;
    const shared_ptr<XCalibratedEntryList> m_calibratedEntryList;
    const shared_ptr<XTextWriter> m_textWriter;
	const shared_ptr<XRawStreamRecorder> m_rawStreamRecorder;
	const shared_ptr<XJournalRecorder> m_journal;
	const shared_ptr<XRawStreamRecordReader> m_rawStreamRecordReader;

    shared_ptr<XNode> m_lastPointedByNodeBrowser;
    shared_ptr<XStringNode> m_pyInfoForNodeBrowser;

    const xqcon_ptr m_conRecordReader,
        m_conDrivers, m_conInterfaces, m_conEntries, m_conGraphs, m_conCalibEntries,
        m_conTextWrite, m_conTextURL, m_conTextLastLine,
        m_conLogURL, m_conLogWrite, m_conLogEvery,
        m_conJournalURL, m_conJournalMode, m_conJournalWrite, m_conJournalStats,
        m_conUrlRubyThread,
        m_conCalTable, m_conNodeBrowser;
	shared_ptr<Listener> m_lsnOnReleaseDriver;
	void onReleaseDriver(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e);
};

//! use this to show a floating information at the front of the main window.
//! \sa XStatusPrinter
extern DECLSPEC_KAME shared_ptr<XStatusPrinter> g_statusPrinter;

//---------------------------------------------------------------------------
#endif
