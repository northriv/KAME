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
#include "analyzer.h"
#include "signalgenerator.h"
#include "ui_signalgeneratorform.h"
#include <QStatusBar>

XSG::XSG(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XPrimaryDriver(name, runtime, ref(tr_meas), meas),
	  m_freq(create<XDoubleNode>("Freq", true, "%.13g")),
	  m_oLevel(create<XDoubleNode>("OutputLevel", true)),
	  m_fmON(create<XBoolNode>("FMON", true)),
	  m_amON(create<XBoolNode>("AMON", true)),
	  m_form(new FrmSG(g_pFrmMain)) {
	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("Signal Gen. Control - ") + getLabel() );

	m_conOLevel = xqcon_create<XQLineEditConnector>(m_oLevel, m_form->m_edOLevel);
	m_conFreq = xqcon_create<XQLineEditConnector>(m_freq, m_form->m_edFreq);
	m_conAMON = xqcon_create<XQToggleButtonConnector>(m_amON, m_form->m_ckbAMON);
	m_conFMON = xqcon_create<XQToggleButtonConnector>(m_fmON, m_form->m_ckbFMON);
      
	oLevel()->setUIEnabled(false);
	freq()->setUIEnabled(false);
	amON()->setUIEnabled(false);
	fmON()->setUIEnabled(false);
}
void
XSG::showForms() {
	m_form->show();
	m_form->raise();
}

void
XSG::start() {
	m_oLevel->setUIEnabled(true);
	m_freq->setUIEnabled(true);
	m_amON->setUIEnabled(true);
	m_fmON->setUIEnabled(true);
	
	m_lsnOLevel = oLevel()->onValueChanged().connectWeak(
		shared_from_this(), &XSG::onOLevelChanged);
	m_lsnAMON = amON()->onValueChanged().connectWeak(
		shared_from_this(), &XSG::onAMONChanged);
	m_lsnFMON = fmON()->onValueChanged().connectWeak(
		shared_from_this(), &XSG::onFMONChanged);
	m_lsnFreq = freq()->onValueChanged().connectWeak(
		shared_from_this(), &XSG::onFreqChanged);
}
void
XSG::stop() {
	m_oLevel->setUIEnabled(false);
	m_freq->setUIEnabled(false);
	m_amON->setUIEnabled(false);
	m_fmON->setUIEnabled(false);
	
	m_lsnOLevel.reset();
	m_lsnAMON.reset();
	m_lsnFMON.reset();
	m_lsnFreq.reset();

	afterStop();
}

void
XSG::analyzeRaw() throw (XRecordError&) {
    m_freqRecorded = pop<double>();
}
void
XSG::visualize() {
	//! impliment extra codes which do not need write-lock of record
	//! record is read-locked
}
void
XSG::onFreqChanged(const shared_ptr<XValueNodeBase> &) {
    double _freq = *freq();
    if(_freq <= 0) {
        gErrPrint(getLabel() + " " + i18n("Positive Value Needed."));
        return;
    }
    XTime time_awared(XTime::now());
    changeFreq(_freq);
    clearRaw();
    push(_freq);
    finishWritingRaw(time_awared, XTime::now());
}
