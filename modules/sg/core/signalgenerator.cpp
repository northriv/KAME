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
    : XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
      m_entryFreq(create<XScalarEntry>("EntryFreq", false,
                                    dynamic_pointer_cast<XDriver>(shared_from_this()))),
      m_rfON(create<XBoolNode>("RFON", true)),
	  m_freq(create<XDoubleNode>("Freq", true, "%.13g")),
	  m_oLevel(create<XDoubleNode>("OutputLevel", true)),
	  m_fmON(create<XBoolNode>("FMON", true)),
	  m_amON(create<XBoolNode>("AMON", true)),
      m_amDepth(create<XDoubleNode>("AMDepth", true)),
      m_fmDev(create<XDoubleNode>("FMDev", true)),
      m_amIntSrcFreq(create<XDoubleNode>("AMIntSrcFreq", true)),
      m_fmIntSrcFreq(create<XDoubleNode>("FMIntSrcFreq", true)),
      m_sweepMode(create<XComboNode>("SweepMode", true, true)),
      m_sweepFreqStart(create<XDoubleNode>("SweepFreqStart", true)),
      m_sweepFreqStop(create<XDoubleNode>("SweepFreqStop", true)),
      m_sweepAmplStart(create<XDoubleNode>("SweepAmplStart", true)),
      m_sweepAmplStop(create<XDoubleNode>("SweepAmplStop", true)),
      m_sweepDwellTime(create<XDoubleNode>("SweepDwellTime", true)),
      m_sweepPoints(create<XUIntNode>("SweepPoints", true)),
      m_form(new FrmSG) {

    meas->scalarEntries()->insert(tr_meas, m_entryFreq);

	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("Signal Gen. Control - ") + getLabel() );

    m_conUIs = {
        xqcon_create<XQToggleButtonConnector>(m_rfON, m_form->m_ckbRFON),
        xqcon_create<XQLineEditConnector>(m_oLevel, m_form->m_edOLevel),
        xqcon_create<XQLineEditConnector>(m_freq, m_form->m_edFreq),
        xqcon_create<XQToggleButtonConnector>(m_amON, m_form->m_ckbAMON),
        xqcon_create<XQToggleButtonConnector>(m_fmON, m_form->m_ckbFMON),
        xqcon_create<XQLineEditConnector>(m_amDepth, m_form->m_edAMDepth),
        xqcon_create<XQLineEditConnector>(m_fmDev, m_form->m_edFMDev),
        xqcon_create<XQLineEditConnector>(m_amIntSrcFreq, m_form->m_edAMIntSrcFreq),
        xqcon_create<XQLineEditConnector>(m_fmIntSrcFreq, m_form->m_edFMIntSrcFreq),
        xqcon_create<XQComboBoxConnector>(m_sweepMode, m_form->m_cmbSweepMode, Snapshot( *m_sweepMode)),
        xqcon_create<XQLineEditConnector>(m_sweepFreqStart, m_form->m_edSweepFreqStart),
        xqcon_create<XQLineEditConnector>(m_sweepFreqStop, m_form->m_edSweepFreqStop),
        xqcon_create<XQLineEditConnector>(m_sweepAmplStart, m_form->m_edSweepAmplStart),
        xqcon_create<XQLineEditConnector>(m_sweepAmplStop, m_form->m_edSweepAmplStop),
        xqcon_create<XQLineEditConnector>(m_sweepDwellTime, m_form->m_edSweepDwellTime),
        xqcon_create<XQLineEditConnector>(m_sweepPoints, m_form->m_edSweepPoints),
    };
      
    iterate_commit([=](Transaction &tr){
        std::vector<shared_ptr<XNode>> runtime_ui = {
            rfON(), oLevel(), freq(), amON(), fmON(),
            amDepth(), fmDev(), amIntSrcFreq(), fmIntSrcFreq(),
            sweepMode(), sweepFreqStart(), sweepFreqStop(),
            sweepAmplStart(), sweepAmplStop(),
            sweepDwellTime(), sweepPoints()
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
XSG::analyzeRaw(RawDataReader &reader, Transaction &tr) {
    double freq = reader.pop<double>();
    tr[ *this].m_freq = freq;
    m_entryFreq->value(tr, freq);
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

void *
XSG::execute(const atomic<bool> &terminated) {
    std::vector<shared_ptr<XNode>> runtime_ui = {
        rfON(), oLevel(), freq(), amON(), fmON(),
        amDepth(), fmDev(), amIntSrcFreq(), fmIntSrcFreq(),
        sweepMode(), sweepFreqStart(), sweepFreqStop(),
        sweepAmplStart(), sweepAmplStop(),
        sweepDwellTime(), sweepPoints()
    };
    iterate_commit([=](Transaction &tr){
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
        m_lsnFMDev = tr[ *fmDev()].onValueChanged().connectWeakly(
            shared_from_this(), &XSG::onFMDevChanged);
        m_lsnAMIntSrcFreq = tr[ *amIntSrcFreq()].onValueChanged().connectWeakly(
            shared_from_this(), &XSG::onAMIntSrcFreqChanged);
        m_lsnFMIntSrcFreq = tr[ *fmIntSrcFreq()].onValueChanged().connectWeakly(
            shared_from_this(), &XSG::onFMIntSrcFreqChanged);
        m_lsnSweepCond = tr[ *sweepMode()].onValueChanged().connectWeakly(
            shared_from_this(), &XSG::onSweepCondChanged);
        tr[ *sweepFreqStart()].onValueChanged().connect(m_lsnSweepCond);
        tr[ *sweepFreqStop()].onValueChanged().connect(m_lsnSweepCond);
        tr[ *sweepAmplStart()].onValueChanged().connect(m_lsnSweepCond);
        tr[ *sweepAmplStop()].onValueChanged().connect(m_lsnSweepCond);
        tr[ *sweepDwellTime()].onValueChanged().connect(m_lsnSweepCond);
        tr[ *sweepPoints()].onValueChanged().connect(m_lsnSweepCond);
    });

    while( !terminated) {
        msecsleep(50);

        XTime time_awared = XTime::now();

        double freq;
        // try/catch exception of communication errors
        try {
            freq = getFreq();
        }
        catch (XInterface::XUnsupportedFeatureError &e) {
            msecsleep(200);
            continue;
        }
        catch (XKameError &e) {
            e.print(getLabel() + " " + i18n("Read Error, "));
            continue;
        }
        auto writer = std::make_shared<RawData>();
        writer->push(freq);
        finishWritingRaw(writer, time_awared, XTime::now());
    }

    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });

    m_lsnRFON.reset();
    m_lsnOLevel.reset();
    m_lsnAMON.reset();
    m_lsnFMON.reset();
    m_lsnFreq.reset();
    m_lsnAMDepth.reset();
    m_lsnFMDev.reset();
    m_lsnAMIntSrcFreq.reset();
    m_lsnFMIntSrcFreq.reset();
    m_lsnSweepCond.reset();
    return NULL;
}
