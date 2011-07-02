/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "measure.h"#include "kame.h"#include "xrubysupport.h"#include "primarydriver.h"#include "interface.h"#include "analyzer.h"#include "recorder.h"#include "recordreader.h"#include "thermometer.h"#include "caltable.h"#include "analyzer.h"
#include "driverlistconnector.h"#include "interfacelistconnector.h"#include "entrylistconnector.h"#include "graphlistconnector.h"#include "recordreaderconnector.h"#include "nodebrowser.h"
#include "ui_caltableform.h"
#include "ui_drivercreate.h"
#include "ui_nodebrowserform.h"
#include "ui_recordreaderform.h"
#include "ui_rubythreadtool.h"
#include "ui_graphtool.h"#include "ui_interfacetool.h"#include "ui_drivertool.h"#include "ui_scalarentrytool.h"#include <q3textbrowser.h>#include <kfiledialog.h>#include <kstandarddirs.h>#include <kmessagebox.h>
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
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmInterface->tblInterfaces)),
m_conEntries(xqcon_create<XEntryListConnector>(
		scalarEntries(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmScalarEntry->m_tblEntries,
		charts())),
m_conGraphs(xqcon_create<XGraphListConnector>(graphs(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmGraphList->tblGraphs,
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmGraphList->btnNewGraph,
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmGraphList->btnDeleteGraph)),
m_conTextWrite(xqcon_create<XQToggleButtonConnector>(
		textWriter()->recording(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmScalarEntry->m_ckbTextWrite)),
m_conTextURL(xqcon_create<XKURLReqConnector>(
		textWriter()->filename(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmScalarEntry->m_urlTextWriter,
		"*.dat|Data files (*.dat)\n*.*|All files (*.*)", true)),
m_conTextLastLine(xqcon_create<XQLineEditConnector>(
		textWriter()->lastLine(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmScalarEntry->m_edLastLine)),
m_conBinURL(xqcon_create<XKURLReqConnector>(
		rawStreamRecorder()->filename(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmDriver->m_urlBinRec,
		"*.bin|Binary files (*.bin)\n*.*|All files (*.*)", true)),
m_conBinWrite(xqcon_create<XQToggleButtonConnector>(
		rawStreamRecorder()->recording(),
		dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmDriver->m_ckbBinRecWrite)),
m_conUrlRubyThread(),
m_conCalTable(xqcon_create<XConCalTable>(
				m_thermometers, dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmCalTable)),
m_conNodeBrowser(xqcon_create<XNodeBrowser>(
		shared_from_this(), dynamic_cast<FrmKameMain*>(g_pFrmMain)->m_pFrmNodeBrowser)) {

	g_statusPrinter = XStatusPrinter::create();

	for(Transaction tr( *this);; ++tr) {
		m_lsnOnReleaseDriver = tr[ *drivers()].onRelease().connect(
			*this, &XMeasure::onReleaseDriver);
		if(tr.commit())
			break;
	}

	m_ruby = createOrphan<XRuby>("RubySupport", true,
		dynamic_pointer_cast<XMeasure>(shared_from_this()));

	m_ruby->resume();

	initialize();
}

XMeasure::~XMeasure() {
	printf("terminate\n");
	m_rawStreamRecordReader->terminate();
	m_ruby->terminate();
	m_ruby.reset();
	g_statusPrinter.reset();
}
void XMeasure::initialize() {
}
void XMeasure::terminate() {
	interfaces()->releaseAll();
	drivers()->releaseAll();
	thermometers()->releaseAll();
	initialize();
}
void XMeasure::stop() {
	Snapshot shot( *drivers());
	if(shot.size()) {
		const XNode::NodeList &list( *shot.list());
		for(XNode::const_iterator it = list.begin(); it != list.end(); it++) {
			shared_ptr<XPrimaryDriver> driver = dynamic_pointer_cast<
				XPrimaryDriver> ( *it);
			if(driver)
				driver->stop();
		}
	}
}
void XMeasure::onReleaseDriver(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
	shared_ptr<XDriver> driver = static_pointer_cast<XDriver>(e.released);
	shared_ptr<XPrimaryDriver> pridriver =
		dynamic_pointer_cast<XPrimaryDriver>(driver);
	if(pridriver)
		pridriver->stop();
	for(;;) {
		shared_ptr<XScalarEntry> entry;
		Snapshot shot( *scalarEntries());
		if(shot.size()) {
			const XNode::NodeList &list( *shot.list());
			for(XNode::const_iterator it = list.begin(); it != list.end(); it++) {
				shared_ptr<XScalarEntry> entr = dynamic_pointer_cast<
					XScalarEntry> ( *it);
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
		shared_ptr<XInterface> interface;
		Snapshot shot( *interfaces());
		if(shot.size()) {
			const XNode::NodeList &list( *shot.list());
			for(XNode::const_iterator it = list.begin(); it != list.end(); it++) {
				shared_ptr<XInterface> intf = dynamic_pointer_cast<
					XInterface> ( *it);
				if(intf->driver() == driver) {
					interface = intf;
				}
			}
		}
		if( !interface)
			break;
		interfaces()->release(interface);
	}
}
