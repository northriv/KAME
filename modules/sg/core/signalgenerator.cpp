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
      m_amDepth(create<XDoubleNode>("AMDepth", true)),
      m_fmDepth(create<XDoubleNode>("FMDepth", true)),
      m_amIntSrcFreq(create<XDoubleNode>("AMIntSrcFreq", true)),
      m_fmIntSrcFreq(create<XDoubleNode>("FMIntSrcFreq", true)),
      m_form(new FrmSG) {
	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("Signal Gen. Control - ") + getLabel() );

    m_conUIs = {
        xqcon_create<XQToggleButtonConnector>(m_rfON, m_form->m_ckbRFON),
        xqcon_create<XQLineEditConnector>(m_oLevel, m_form->m_edOLevel),
        xqcon_create<XQLineEditConnector>(m_freq, m_form->m_edFreq),
        xqcon_create<XQToggleButtonConnector>(m_amON, m_form->m_ckbAMON),
        xqcon_create<XQToggleButtonConnector>(m_fmON, m_form->m_ckbFMON)
    };
      
    iterate_commit([=](Transaction &tr){
        std::vector<shared_ptr<XNode>> runtime_ui = {
            rfON(), oLevel(), freq(), amON(), fmON(),
            amDepth(), fmDepth(), amIntSrcFreq(), fmIntSrcFreq()
        };
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });
}
void
XSG::showForms() {
    m_form->showNormal();
	m_form->raise();
}

void
XSG::start() {
	iterate_commit([=](Transaction &tr){
        std::vector<shared_ptr<XNode>> runtime_ui = {
            rfON(), oLevel(), freq(), amON(), fmON(),
            amDepth(), fmDepth(), amIntSrcFreq(), fmIntSrcFreq()
        };
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(true);
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
        m_lsnAMDepth = tr[ *amDepth()].onValueChanged().connectWeakly(
            shared_from_this(), &XSG::onAMDepthChanged);
        m_lsnFMDepth = tr[ *fmDepth()].onValueChanged().connectWeakly(
            shared_from_this(), &XSG::onFMDepthChanged);
        m_lsnAMIntSrcFreq = tr[ *amIntSrcFreq()].onValueChanged().connectWeakly(
            shared_from_this(), &XSG::onAMIntSrcFreqChanged);
        m_lsnFMIntSrcFreq = tr[ *fmIntSrcFreq()].onValueChanged().connectWeakly(
            shared_from_this(), &XSG::onFMIntSrcFreqChanged);
    });
}
void
XSG::stop() {
    iterate_commit([=](Transaction &tr){
        std::vector<shared_ptr<XNode>> runtime_ui = {
            rfON(), oLevel(), freq(), amON(), fmON(),
            amDepth(), fmDepth(), amIntSrcFreq(), fmIntSrcFreq()
        };
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });
	
	m_lsnRFON.reset();
	m_lsnOLevel.reset();
	m_lsnAMON.reset();
	m_lsnFMON.reset();
	m_lsnFreq.reset();
    m_lsnAMDepth.reset();
    m_lsnFMDepth.reset();
    m_lsnAMIntSrcFreq.reset();
    m_lsnFMIntSrcFreq.reset();
    closeInterface();
}

void
XSG::analyzeRaw(RawDataReader &reader, Transaction &tr) {
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
