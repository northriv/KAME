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
#include "xpythonsupport.h"
#ifdef USE_RUBY
    #include "xrubysupport.h"
#endif
#include "measure.h"
#include "kame.h"

#include "primarydriver.h"
#include "interface.h"
#include "analyzer.h"
#include "rawstream.h"
#include "textwriter.h"
#include "xjournal.h"
#include "journalreader.h"

#include "thermometer.h"
#include "caltable.h"

#include "analyzer.h"
#include "driverlistconnector.h"
#include "interfacelistconnector.h"
#include "entrylistconnector.h"
#include "graphlistconnector.h"
#include "calibentryconnector.h"
#include "journalreaderconnector.h"
#include "nodebrowser.h"

#include "ui_caltableform.h"
#include "ui_drivercreate.h"
#include "ui_nodebrowserform.h"
#include "ui_journalreaderform.h"
#include "ui_scriptingthreadtool.h"
#include "ui_graphtool.h"
#include "ui_interfacetool.h"
#include "ui_drivertool.h"
#include "ui_scalarentrytool.h"

#include <QTextBrowser>

shared_ptr<XStatusPrinter> g_statusPrinter;

XMeasure::XMeasure(const char *name, bool runtime) :
XNode(name, runtime),
m_thermometers(create<XThermometerList>("Thermometers", false)),
m_scalarEntries(create<XScalarEntryList>("ScalarEntries", true)),
m_graphList(create<XGraphList>("GraphList", false, scalarEntries())),
m_chartList(create<XChartList>("ChartList", true, scalarEntries())),
m_interfaces(create<XInterfaceList>("Interfaces", true)),
m_drivers(create<XDriverList>("Drivers", false, static_pointer_cast<XMeasure>(shared_from_this()))),
m_calibratedEntryList(create<XCalibratedEntryList>("CalibratedEntries", false, scalarEntries(), thermometers(),
                                                       static_pointer_cast<XMeasure>(shared_from_this()))),
m_textWriter(create<XTextWriter>("TextWriter", false, drivers(), scalarEntries())),
m_journal(create<XJournal>("Journal", false, drivers())),
m_journalReader(create<XJournalReader>("JournalReader", false,
		drivers())),
m_conJournalReader(xqcon_create<XJournalReaderConnector>(
		journalReader(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmJournalReader)),
m_conDrivers(xqcon_create<XDriverListConnector>(
		m_drivers, dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmDriver)),
m_conInterfaces(xqcon_create<XInterfaceListConnector>(
		m_interfaces,
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmInterface->m_tblInterface)),
m_conEntries(xqcon_create<XEntryListConnector>(
		scalarEntries(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmScalarEntry->m_tblEntries,
		charts())),
m_conGraphs(xqcon_create<XGraphListConnector>(graphs(),
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmGraphList->m_tblGraphs,
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmGraphList->btnNewGraph,
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmGraphList->btnDeleteGraph)),
m_conCalibEntries(xqcon_create<XCalibratedEntryListConnector>(calibratedEntries(),
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmGraphList->m_tblCalibEntries,
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmGraphList->btnNewCalibEntry,
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmGraphList->btnDeleteCalibEntry)),
m_conTextWrite(xqcon_create<XQToggleButtonConnector>(
		textWriter()->recording(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmScalarEntry->m_ckbTextWrite)),
m_conTextURL(xqcon_create<XFilePathConnector>(
        textWriter()->filename(),
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmScalarEntry->m_edTextWriter,
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmScalarEntry->m_btnTextWriter,
        "Data files (*.dat);;All files (*.*)", true)),
m_conTextLastLine(xqcon_create<XQLineEditConnector>(
		textWriter()->lastLine(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmScalarEntry->m_edLastLine)),
m_conLogWrite(xqcon_create<XQToggleButtonConnector>(
		textWriter()->logRecording(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmScalarEntry->m_ckbLoggerWrite)),
m_conLogURL(xqcon_create<XFilePathConnector>(
		textWriter()->logFilename(),
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmScalarEntry->m_edLogFile,
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmScalarEntry->m_btnLogFile,
        "Data files (*.dat);;All files (*.*)", true)),
m_conLogEvery(xqcon_create<XQLineEditConnector>(
		textWriter()->logEvery(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmScalarEntry->m_edLoggerEvery)),
m_conJournalURL(xqcon_create<XFilePathConnector>(
        journal()->filename(),
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmDriver->m_edJournal,
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmDriver->m_btnJournal,
        "KAME journal (*.kamj);;All files (*.*)", true)),
//Read-only line edit rather than a label: a path has to be selectable and
//copyable, and a label clips a long one at whichever end the alignment
//chooses -- which hid the file name, the one part anybody wants.
m_conJournalSessionFile(xqcon_create<XQLineEditConnector>(
        journal()->sessionFile(),
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmDriver->m_edSessionFile)),
m_conJournalSession(xqcon_create<XQToggleButtonConnector>(
        journal()->sessionJournal(),
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmDriver->m_ckbSessionJournal)),
m_conJournalMode(xqcon_create<XQComboBoxConnector>(
        journal()->mode(),
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmDriver->m_cmbJournalMode,
        Snapshot( *journal()->mode()))),
m_conJournalWrite(xqcon_create<XQToggleButtonConnector>(
        journal()->recording(),
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmDriver->m_ckbJournalWrite)),
m_conJournalStats(xqcon_create<XQLabelConnector>(
        journal()->statistics(),
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmDriver->m_lblJournalStats)),
m_conUrlRubyThread(),
m_conCalTable(xqcon_create<XConCalTable>(
                m_thermometers, dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmCalTable)),
m_conNodeBrowser(xqcon_create<XNodeBrowser>(
        dynamic_pointer_cast<XMeasure>(shared_from_this()), dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmNodeBrowser)) {

	g_statusPrinter = XStatusPrinter::create();

    m_textWriter->addCalibratedEntrySource(m_calibratedEntryList);

	iterate_commit([=](Transaction &tr){
		m_lsnOnReleaseDriver = tr[ *drivers()].onRelease().connect(
			*this, &XMeasure::onReleaseDriver);
        tr[ *calibratedEntries()].onRelease().connect(m_lsnOnReleaseDriver);
    });

#ifdef USE_PYBIND11
    m_python = createOrphan<XPython>("PythonSupport", true,
        dynamic_pointer_cast<XMeasure>(shared_from_this()));
    //Start the worker AFTER the object is fully constructed (vtable installed
    //and owned by m_python) — execute() is pure virtual in the base, so
    //starting it from the base ctor raced into __cxa_pure_virtual.
    m_python->startExecutionThread();
#endif
    m_pyInfoForNodeBrowser = XNode::createOrphan<XStringNode>("PyInfoForNodeBrowser", true);

#ifdef USE_RUBY
    m_ruby = createOrphan<XRuby>("RubySupport", true,
        dynamic_pointer_cast<XMeasure>(shared_from_this()));
    m_ruby->startExecutionThread();
#endif

    initialize();
}

XMeasure::~XMeasure() {

}
void XMeasure::initialize() {
}
void XMeasure::terminate() {
	interfaces()->releaseAll();
    stop(); //notifies running threads of termination.
    graphs()->releaseAll();
    drivers()->releaseAll(); //still threads may hold their shared pointers.
    calibratedEntries()->releaseAll(); //releases m_entry from XScalarEntryList.
	thermometers()->releaseAll();
    Snapshot shot( *this);
	initialize();
}
void XMeasure::terminate_all() {
    //Every stage is isolated because the joins below are the safety-critical
    //part: terminate() runs releaseAll() on five lists plus the driver stops,
    //all of which can throw, and an unwound terminate_all() would leave the
    //scripting threads running on into static destruction -- which is fatal
    //(see FrmKameMain::closeEvent).  This is a guard, not the fix for the
    //2026-08-20 quit crash: the run that reproduced that crash reported no
    //failing stage once these markers existed, so nothing was throwing.  The
    //ordering in closeEvent was what mattered.
    //Reported to stderr, not through XKameError::print(), which would post to a
    //GUI that is already being torn down.
    auto stage = [](const char *what, auto &&fn) noexcept {
        try { fn(); }
        catch (XKameError &e) {
            fprintf(stderr, "kame: %s failed during shutdown: %s\n",
                what, (const char *)e.msg().c_str());
        }
        catch (std::exception &e) {
            fprintf(stderr, "kame: %s failed during shutdown: %s\n", what, e.what());
        }
        catch (...) {
            fprintf(stderr, "kame: %s failed during shutdown.\n", what);
        }
    };
    stage("releasing nodes", [&]{ terminate();});
    fprintf(stderr, "terminat");
#ifdef USE_RUBY
    stage("stopping the Ruby thread", [&]{ m_ruby->terminate(); m_ruby->join();});
    m_ruby.reset();
#endif
#ifdef USE_PYBIND11
    //pybind11 should free shared_ptr to XMeasure.
    //With IPython, sys.exit(0) is called, and stdout/err seem to be closed.
    stage("stopping the Python thread", [&]{ m_python->terminate(); m_python->join();});
    m_python.reset();
#endif
    stage("stopping the record reader", [&]{
        m_journalReader->terminate(); m_journalReader->join();});
    g_statusPrinter.reset();
    fprintf(stderr, "ed.\n");
}

void XMeasure::stop() {
	Snapshot shot( *drivers());
	if(shot.size()) {
		const XNode::NodeList &list( *shot.list());
		for(auto it = list.begin(); it != list.end(); it++) {
			auto driver = dynamic_pointer_cast<XPrimaryDriver> ( *it);
			if(driver)
				driver->stop();
		}
	}
}
void XMeasure::onReleaseDriver(const Snapshot &, const XListNodeBase::Payload::ReleaseEvent &e) {
	auto driver = static_pointer_cast<XDriver>(e.released);
	auto pridriver = dynamic_pointer_cast<XPrimaryDriver>(driver);
	if(pridriver)
		pridriver->stop();
	for(;;) {
		shared_ptr<XScalarEntry> entry;
		Snapshot shot( *scalarEntries());
		if(shot.size()) {
			const XNode::NodeList &list( *shot.list());
			for(auto it = list.begin(); it != list.end(); it++) {
				auto entr = dynamic_pointer_cast<XScalarEntry> ( *it);
				if(entr->driver() == driver) {
					entry = entr;
				}
			}
		}
		if( !entry)
			break;
		scalarEntries()->release(entry);
	}
	for(;;) {
        shared_ptr<XInterface> intf_release;
		Snapshot shot( *interfaces());
		if(shot.size()) {
			const XNode::NodeList &list( *shot.list());
			for(auto it = list.begin(); it != list.end(); it++) {
				auto intf = dynamic_pointer_cast<XInterface> ( *it);
				if(intf->driver() == driver) {
                    intf_release = intf;
				}
			}
		}
        if( !intf_release)
			break;
        interfaces()->release(intf_release);
	}
}
