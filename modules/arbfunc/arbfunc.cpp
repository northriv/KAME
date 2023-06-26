/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "arbfunc.h"
#include "ui_arbfuncform.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"

XArbFuncGen::XArbFuncGen(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriver(name, runtime, ref(tr_meas), meas),
    m_output(create<XBoolNode>("Output", true)),
    m_burst(create<XBoolNode>("Burst", true)),
    m_burstPhase(create<XDoubleNode>("BurstPhase", true)),
    m_trigSrc(create<XComboNode>("TrigSrc", true)),
    m_waveform(create<XComboNode>("Waveform", true)),
    m_freq(create<XDoubleNode>("Freq", true)),
    m_ampl(create<XDoubleNode>("Ampl", true)),
    m_offset(create<XDoubleNode>("Offset", true)),
    m_duty(create<XDoubleNode>("Duty", true)),
    m_pulseWidth(create<XDoubleNode>("PulseWidth", true)),
    m_pulsePeriod(create<XDoubleNode>("PulsePeriod", true)),
    m_form(new FrmArbFuncGen) {

    m_conUIs = {
        xqcon_create<XQToggleButtonConnector>(m_output, m_form->m_ckbOutput),
        xqcon_create<XQToggleButtonConnector>(m_burst, m_form->m_ckbBurst),
        xqcon_create<XQLineEditConnector>(m_burstPhase, m_form->m_edBurstPhase),
        xqcon_create<XQComboBoxConnector>(m_trigSrc, m_form->m_cmbTrigSrc, Snapshot( *m_trigSrc)),
        xqcon_create<XQComboBoxConnector>(m_waveform, m_form->m_cmbWaveform, Snapshot( *m_waveform)),
        xqcon_create<XQLineEditConnector>(m_freq, m_form->m_edFreq),
        xqcon_create<XQLineEditConnector>(m_ampl, m_form->m_edAmpl),
        xqcon_create<XQLineEditConnector>(m_offset, m_form->m_edOffset),
        xqcon_create<XQLineEditConnector>(m_duty, m_form->m_edDuty),
        xqcon_create<XQLineEditConnector>(m_pulseWidth, m_form->m_edPulseWidth),
        xqcon_create<XQLineEditConnector>(m_pulsePeriod, m_form->m_edPulsePeriod),
    };

    iterate_commit([=](Transaction &tr){
        std::vector<shared_ptr<XNode>> runtime_ui{
            m_output, m_burst, m_burstPhase, m_trigSrc, m_waveform, m_freq, m_ampl, m_offset, m_duty,
            m_pulseWidth, m_pulsePeriod
        };
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });

    m_form->setWindowTitle(i18n("Arbitrary Func. Gen. - ") + getLabel());
}

void XArbFuncGen::showForms() {
    m_form->showNormal();
    m_form->raise();
}

void XArbFuncGen::onOutputChanged(const Snapshot &, XValueNodeBase *) {
    Snapshot shot( *this);
    try {
        changeOutput(shot[ *output()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error, "));
        return;
    }
}
void XArbFuncGen::onCondChanged(const Snapshot &, XValueNodeBase *) {
    Snapshot shot( *this);
    try {
        changePulseCond();
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error, "));
        return;
    }
}

void XArbFuncGen::analyzeRaw(RawDataReader &reader, Transaction &tr) {
}
void XArbFuncGen::visualize(const Snapshot &shot) {
}

void
XArbFuncGen::start() {
    std::vector<shared_ptr<XNode>> runtime_ui{
        m_output, m_burst, m_burstPhase, m_trigSrc, m_waveform, m_freq, m_ampl, m_offset, m_duty,
        m_pulseWidth, m_pulsePeriod
    };
    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(true);
    });
    iterate_commit([=](Transaction &tr){
        m_lsnOnOutputChanged = tr[ *m_output].onValueChanged().connectWeakly(
                    shared_from_this(), &XArbFuncGen::onOutputChanged);
        m_lsnOnCondChanged = tr[ *m_freq].onValueChanged().connectWeakly(
            shared_from_this(), &XArbFuncGen::onCondChanged);
        tr[ *m_burst].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *m_burstPhase].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *m_trigSrc].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *m_ampl].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *m_offset].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *m_duty].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *m_pulseWidth].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *m_pulsePeriod].onValueChanged().connect(m_lsnOnCondChanged);
    });
}
void
XArbFuncGen::stop() {
    m_lsnOnOutputChanged.reset();
    m_lsnOnCondChanged.reset();
    std::vector<shared_ptr<XNode>> runtime_ui{
        m_output, m_waveform, m_freq, m_ampl, m_offset, m_duty,
        m_pulseWidth, m_pulsePeriod
    };

    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });

    closeInterface();
}

