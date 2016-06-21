/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#include "ui_dmmform.h"
#include <QStatusBar>
#include "dmm.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"

XDMM::XDMM(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_entry(create<XScalarEntry>("Value", false, 
								 dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_function(create<XComboNode>("Function", false)),
    m_waitInms(create<XUIntNode>("WaitInms", false)),
    m_form(new FrmDMM(g_pFrmMain)) {
	meas->scalarEntries()->insert(tr_meas, m_entry);
	iterate_commit([=](Transaction &tr){
		tr[ *m_waitInms] = 100;
    });
	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("DMM - ") + getLabel() );
	m_function->setUIEnabled(false);
	m_waitInms->setUIEnabled(false);
	m_conFunction = xqcon_create<XQComboBoxConnector>(m_function, m_form->m_cmbFunction, Snapshot( *m_function));
	m_conWaitInms = xqcon_create<XQSpinBoxUnsignedConnector>(m_waitInms, m_form->m_numWait);
}

void
XDMM::showForms() {
//! impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}

void
XDMM::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
	double x = reader.pop<double>();
	tr[ *this].write_(x);
	m_entry->value(tr, x);
}
void
XDMM::visualize(const Snapshot &shot) {
}

void
XDMM::onFunctionChanged(const Snapshot &shot, XValueNodeBase *node) {
    try {
        changeFunction();
    }
    catch (XKameError &e) {
		e.print(getLabel() + " " + i18n("DMM Error"));
    }
}

void *
XDMM::execute(const atomic<bool> &terminated) {
	changeFunction();

    m_function->setUIEnabled(true);
    m_waitInms->setUIEnabled(true);
        
	iterate_commit([=](Transaction &tr){
	    m_lsnOnFunctionChanged =
	        tr[ *function()].onValueChanged().connectWeakly(
				shared_from_this(), &XDMM::onFunctionChanged);
    });
	while( !terminated) {
		msecsleep( ***waitInms());
		if(( **function())->to_str().empty()) continue;
      
		auto writer = std::make_shared<RawData>();
		double x;
		XTime time_awared = XTime::now();
		// try/catch exception of communication errors
		try {
			x = oneShotRead();
		}
		catch (XKameError &e) {
			e.print(getLabel() + " " + i18n("DMM Read Error"));
			continue;
		}
		writer->push(x);
		finishWritingRaw(writer, time_awared, XTime::now());
	}
    
    m_function->setUIEnabled(false);
    m_waitInms->setUIEnabled(false);

    m_lsnOnFunctionChanged.reset();

    return NULL;
}

