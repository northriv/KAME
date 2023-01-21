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
#include "odmrfm.h"
#include "ui_odmrfmform.h"

#include "analyzer.h"
#include "xnodeconnector.h"

REGISTER_TYPE(XDriverList, ODMRFMControl, "ODMR peak tracker by FM");

//---------------------------------------------------------------------------
XODMRFMControl::XODMRFMControl(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XSecondaryDriver(name, runtime, ref(tr_meas), meas),
        m_entryFreq(create<XScalarEntry>("Freq", false,
            dynamic_pointer_cast<XDriver>(shared_from_this()))),
        m_entryTesla(create<XScalarEntry>("Tesla", false,
            dynamic_pointer_cast<XDriver>(shared_from_this()))),
        m_entryTeslaErr(create<XScalarEntry>("TeslaErr", false,
            dynamic_pointer_cast<XDriver>(shared_from_this()))),
        m_entryFMIntens(create<XScalarEntry>("FMIntens", false,
            dynamic_pointer_cast<XDriver>(shared_from_this()))),
        m_sg(create<XItemNode<XDriverList, XSG> >("SG", false, ref(tr_meas), meas->drivers(), true)),
        m_lia(create<XItemNode<XDriverList, XLIA> >("LIA", false, ref(tr_meas), meas->drivers(), true)),
        m_gamma2pi(create<XDoubleNode>("Gamma2pi", false)),
        m_PhaseErrWithin(create<XDoubleNode>("PhaseErrWithin", false)),
        m_fmPhaseOrigin(create<XDoubleNode>("FMPhaseOrigin", false)),
        m_numReadings(create<XUIntNode>("NumReadings", false)),
        m_ctrlSG(create<XBoolNode>("ControlSG", true)),
        m_form(new FrmODMRFM) {

    connect(sg());
    connect(lia());

    meas->scalarEntries()->insert(tr_meas, entryFreq());
    meas->scalarEntries()->insert(tr_meas, entryTesla());
    meas->scalarEntries()->insert(tr_meas, entryTeslaErr());
    meas->scalarEntries()->insert(tr_meas, entryFMIntens());

    iterate_commit([=](Transaction &tr){
        tr[ *gamma2pi()] = 28024.95142; //g=2.002319
        tr[ *PhaseErrWithin()] = 5.0;
        tr[ *fmPhaseOrigin()] = -90.0;
        tr[ *numReadings()] = 20;
    });

    m_form->setWindowTitle(i18n("ODMR peak tracker by FM - ") + getLabel() );

    m_conUIs = {
        xqcon_create<XQLineEditConnector>(m_gamma2pi, m_form->m_gamma2pi),
        xqcon_create<XQLineEditConnector>(m_PhaseErrWithin, m_form->m_edPhaseErrWithin),
        xqcon_create<XQLineEditConnector>(m_fmPhaseOrigin, m_form->m_edFMPhaseOrigin),
        xqcon_create<XQSpinBoxUnsignedConnector>(numReadings(), m_form->m_spbAverage),
        xqcon_create<XQToggleButtonConnector>(m_ctrlSG, m_form->m_ckbControl),
        xqcon_create<XQComboBoxConnector>(sg(), m_form->m_cmbSG, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(lia(), m_form->m_cmbLIA, ref(tr_meas)),
    };


}
XODMRFMControl::~XODMRFMControl() {
}
void XODMRFMControl::showForms() {
    m_form->resize(100,100); //avoids bug on Windows.
    m_form->showNormal();
    m_form->raise();
}

bool XODMRFMControl::checkDependency(const Snapshot &shot_this,
    const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) const {
    const shared_ptr<XSG> sg__ = shot_this[ *sg()];
    if( !sg__)
        return false;
    const shared_ptr<XLIA> lia__ = shot_this[ *lia()];
    if (emitter != lia__.get())
        return false;
    return true;
}
void XODMRFMControl::analyze(Transaction &tr, const Snapshot &shot_emitter,
    const Snapshot &shot_others,
    XDriver *emitter) {
    const Snapshot &shot_this(tr);
    const shared_ptr<XLIA> lia__ = shot_this[ *lia()];
    assert(lia__);
    const shared_ptr<XSG> sg__ = shot_this[ *sg()];

//    if(shot_others[sg__].time() + 0.05 > shot_emitter[lia__].timeAwared())
//        tr[ *this].m_accumCounts = 0;
//    else
        tr[ *this].m_accumCounts++;
    unsigned int numread = shot_this[ *this].m_accumCounts;
    unsigned int countsToBeIgnored = shot_this[ *numReadings()] / 2 + 1; //transient data after SG change
    if(numread > countsToBeIgnored) {
        std::complex<double> z{shot_emitter[ *lia__].x(), shot_emitter[ *lia__].y()};
        tr[ *this].m_accum += z;
        double phase = shot_emitter[ *lia__].phase();
        tr[ *this].m_accum_arg += phase;
        tr[ *this].m_accum_arg_sq += phase * phase;
    }
    else {
        tr[ *this].m_accum = 0.0;
        tr[ *this].m_accum_arg = 0.0;
        tr[ *this].m_accum_arg_sq = 0.0;
    }
    if(numread < countsToBeIgnored + shot_this[ *numReadings()]) {
        throw XSkippedRecordError(__FILE__, __LINE__);
    }
    tr[ *this].m_accumCounts = 0;
    double phase = shot_this[ *this].m_accum_arg / numread; //std::arg, -pi < x < pi
    double phase_err = std::sqrt(shot_this[ *this].m_accum_arg_sq / numread - phase * phase);

    phase -= shot_this[ *fmPhaseOrigin()] / 180.0 * M_PI;
    phase -= floor((phase + M_PI) / (2 * M_PI)) * (2 * M_PI); //-pi < x < pi, assuming ramp FM
    double freq = shot_others[ *sg__->freq()] + shot_others[ *sg__->fmDev()] * phase / 2 / M_PI;
    double freq_err = shot_others[ *sg__->fmDev()] * phase_err / 2 / M_PI;
    if((shot_others[ *sg__->fmDev()] <= 1e-3) || ( !shot_others[ *sg__->fmON()]) )
        throw XSkippedRecordError(i18n("FM setting in SG driver may be not up-to-date"), __FILE__, __LINE__);

    tr[ *this].m_phase_err = phase_err;
    tr[ *this].m_freq = freq;
    tr[ *this].m_freq_err = freq_err;
    double tesla = freq / shot_this[ *gamma2pi()];
    double tesla_err = freq_err / shot_this[ *gamma2pi()];
    tr[ *this].m_tesla = tesla;
    tr[ *this].m_fmIntens = std::abs(shot_this[ *this].m_accum) / numread;

    entryFreq()->value(tr, freq);
    entryTesla()->value(tr, tesla);
    entryTeslaErr()->value(tr, tesla_err);
    entryFMIntens()->value(tr, shot_this[ *this].m_fmIntens);
}
void XODMRFMControl::visualize(const Snapshot &shot) {
    if(shot[ *m_ctrlSG]) {
        if(shot[ *this].phase_err() < shot[ *PhaseErrWithin()] / 2 / M_PI) {
            const shared_ptr<XSG> sg__ = shot[ *sg()];
            if(sg__) {
                trans( *sg__->freq()) = shot[ *this].freq();
            }
        }
    }
}

