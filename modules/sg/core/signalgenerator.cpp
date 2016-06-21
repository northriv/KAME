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
#include "analyzer.h"
#include "signalgenerator.h"
#include "ui_signalgeneratorform.h"
#include <QStatusBar>

XSG::XSG(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XPrimaryDriver(name, runtime, ref(tr_meas), meas),
	  m_rfON(create<XBoolNode>("RFON", true)),
	  m_freq(create<XDoubleNode>("Freq", true, "%.13g")),
	  m_oLevel(create<XDoubleNode>("OutputLevel", true)),
	  m_fmON(create<XBoolNode>("FMON", true)),
	  m_amON(create<XBoolNode>("AMON", true)),
	  m_form(new FrmSG(g_pFrmMain)) {
	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("Signal Gen. Control - ") + getLabel() );

	m_conRFON = xqcon_create<XQToggleButtonConnector>(m_rfON, m_form->m_ckbRFON);
	m_conOLevel = xqcon_create<XQLineEditConnector>(m_oLevel, m_form->m_edOLevel);
	m_conFreq = xqcon_create<XQLineEditConnector>(m_freq, m_form->m_edFreq);
	m_conAMON = xqcon_create<XQToggleButtonConnector>(m_amON, m_form->m_ckbAMON);
	m_conFMON = xqcon_create<XQToggleButtonConnector>(m_fmON, m_form->m_ckbFMON);
      
	rfON()->setUIEnabled(false);
	oLevel()->setUIEnabled(false);
	freq()->setUIEnabled(false);
	amON()->setUIEnabled(false);
	fmON()->setUIEnabled(false);
}
void
XSG::showForms() {
    m_form->showNormal();
	m_form->raise();
}

void
XSG::start() {
	m_rfON->setUIEnabled(true);
	m_oLevel->setUIEnabled(true);
	m_freq->setUIEnabled(true);
	m_amON->setUIEnabled(true);
	m_fmON->setUIEnabled(true);
	
	iterate_commit([=](Transaction &tr){
		m_lsnRFON = tr[ *rfON()].onValueChanged().connectWeakly(
			shared_from_this(), &XSG::onRFONChanged);
		m_lsnOLevel = tr[ *oLevel()].onValueChanged().connectWeakly(
			shared_from_this(), &XSG::onOLevelChanged);
		m_lsnAMON = tr[ *amON()].onValueChanged().connectWeakly(
			shared_from_this(), &XSG::onAMONChanged);
		m_lsnFMON = tr[ *fmON()].onValueChanged().connectWeakly(
			shared_from_this(), &XSG::onFMONChanged);
		m_lsnFreq = tr[ *freq()].onValueChanged().connectWeakly(
			shared_from_this(), &XSG::onFreqChanged);
    });
}
void
XSG::stop() {
	m_rfON->setUIEnabled(false);
	m_oLevel->setUIEnabled(false);
	m_freq->setUIEnabled(false);
	m_amON->setUIEnabled(false);
	m_fmON->setUIEnabled(false);
	
	m_lsnRFON.reset();
	m_lsnOLevel.reset();
	m_lsnAMON.reset();
	m_lsnFMON.reset();
	m_lsnFreq.reset();

	closeInterface();
}

void
XSG::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
	tr[ *this].m_freq = reader.pop<double>();
}
void
XSG::visualize(const Snapshot &shot) {
}
void
XSG::onFreqChanged(const Snapshot &shot, XValueNodeBase *) {
    double freq__ = shot[ *freq()];
    if(freq__ <= 0) {
        gErrPrint(getLabel() + " " + i18n("Positive Value Needed."));
        return;
    }
    XTime time_awared(XTime::now());
    changeFreq(freq__);

    auto writer = std::make_shared<RawData>();
    writer->push(freq__);
    finishWritingRaw(writer, time_awared, XTime::now());
}
