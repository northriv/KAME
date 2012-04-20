/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
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
    XPrimaryDriver(name, runtime, ref(tr_meas), meas),
    m_entry(create<XScalarEntry>("Value", false, 
								 dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_function(create<XComboNode>("Function", false)),
    m_waitInms(create<XUIntNode>("WaitInms", false)),
    m_form(new FrmDMM(g_pFrmMain)) {
	meas->scalarEntries()->insert(tr_meas, m_entry);
	for(Transaction tr( *this);; ++tr) {
		tr[ *m_waitInms] = 100;
		if(tr.commit())
			break;
	}
	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("DMM - ") + getLabel() );
	m_function->setUIEnabled(false);
	m_waitInms->setUIEnabled(false);
	m_conFunction = xqcon_create<XQComboBoxConnector>(m_function, m_form->m_cmbFunction, Snapshot( *m_function));
	m_conWaitInms = xqcon_create<XQSpinBoxConnector>(m_waitInms, m_form->m_numWait);
}

void
XDMM::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}

void
XDMM::start() {
    m_thread.reset(new XThread<XDMM>(shared_from_this(), &XDMM::execute));
    m_thread->resume();

    m_function->setUIEnabled(true);
    m_waitInms->setUIEnabled(true);
}
void
XDMM::stop() {
    m_function->setUIEnabled(false);
    m_waitInms->setUIEnabled(false);

    if(m_thread) m_thread->terminate();
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
    try {
        changeFunction();
    }
    catch (XKameError &e) {
		e.print(getLabel() + " " + i18n("DMM Error"));
		afterStop();
		return NULL;
    }
        
	for(Transaction tr( *this);; ++tr) {
	    m_lsnOnFunctionChanged =
	        tr[ *function()].onValueChanged().connectWeakly(
				shared_from_this(), &XDMM::onFunctionChanged);
		if(tr.commit())
			break;
	}
	while( !terminated) {
		msecsleep( ***waitInms());
		if(( **function())->to_str().empty()) continue;
      
		shared_ptr<RawData> writer(new RawData);
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
    
    m_lsnOnFunctionChanged.reset();
        
    afterStop();
	return NULL;
}

