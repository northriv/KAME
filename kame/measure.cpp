/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "xpythonsupport.h"
#include "xrubysupport.h"
#include "measure.h"
#include "kame.h"

#include "primarydriver.h"
#include "interface.h"
#include "analyzer.h"
#include "recorder.h"
#include "recordreader.h"

#include "thermometer.h"
#include "caltable.h"

#include "analyzer.h"
#include "driverlistconnector.h"
#include "interfacelistconnector.h"
#include "entrylistconnector.h"
#include "graphlistconnector.h"
#include "recordreaderconnector.h"
#include "nodebrowser.h"

#include "ui_caltableform.h"
#include "ui_drivercreate.h"
#include "ui_nodebrowserform.h"
#include "ui_recordreaderform.h"
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
m_graphList(create<XGraphList>("GraphList", true, scalarEntries())),
m_chartList(create<XChartList>("ChartList", true, scalarEntries())),
m_interfaces(create<XInterfaceList>("Interfaces", true)),
m_drivers(create<XDriverList>("Drivers", false, static_pointer_cast<XMeasure>(shared_from_this()))),
m_textWriter(create<XTextWriter>("TextWriter", false, drivers(), scalarEntries())),
m_rawStreamRecorder(create<XRawStreamRecorder>("RawStreamRecorder", false, drivers())),
m_rawStreamRecordReader(create<XRawStreamRecordReader>("RawStreamRecordReader", false,
		drivers())),
m_conRecordReader(xqcon_create<XRawStreamRecordReaderConnector>(
		rawStreamRecordReader(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmRecordReader)),
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
m_conBinURL(xqcon_create<XFilePathConnector>(
		rawStreamRecorder()->filename(),
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmDriver->m_edRec,
        dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmDriver->m_btnRec,
        "Binary files (*.bin);;All files (*.*)", true)),
m_conBinWrite(xqcon_create<XQToggleButtonConnector>(
		rawStreamRecorder()->recording(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmDriver->m_ckbBinRecWrite)),
m_conUrlRubyThread(),
m_conCalTable(xqcon_create<XConCalTable>(
                m_thermometers, dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmCalTable)),
m_conNodeBrowser(xqcon_create<XNodeBrowser>(
        dynamic_pointer_cast<XMeasure>(shared_from_this()), dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmNodeBrowser)) {

	g_statusPrinter = XStatusPrinter::create();

	iterate_commit([=](Transaction &tr){
		m_lsnOnReleaseDriver = tr[ *drivers()].onRelease().connect(
			*this, &XMeasure::onReleaseDriver);
    });

#ifdef USE_PYBIND11
    m_python = createOrphan<XPython>("PythonSupport", true,
        dynamic_pointer_cast<XMeasure>(shared_from_this()));
#endif
    m_pyInfoForNodeBrowser = XNode::createOrphan<XStringNode>("PyInfoForNodeBrowser", true);

    m_ruby = createOrphan<XRuby>("RubySupport", true,
        dynamic_pointer_cast<XMeasure>(shared_from_this()));

    initialize();
}

XMeasure::~XMeasure() {
    printf("terminate\n");
    m_rawStreamRecordReader->terminate();
    m_ruby->terminate();
#ifdef USE_PYBIND11
    m_python->terminate(); //pybind11 will free shared_ptr to XMeasure
#endif
#ifdef USE_PYBIND11
    m_python->join();
    m_python.reset();
#endif
    m_rawStreamRecordReader->join();
    m_ruby->join();
    m_ruby.reset();
    g_statusPrinter.reset();
}
void XMeasure::initialize() {
}
void XMeasure::terminate() {
	interfaces()->releaseAll();
    stop(); //notifies running threads of termination.
    drivers()->releaseAll(); //still threads may hold their shared pointers.
	thermometers()->releaseAll();
    Snapshot shot( *this);
	initialize();
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
void XMeasure::onReleaseDriver(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
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
