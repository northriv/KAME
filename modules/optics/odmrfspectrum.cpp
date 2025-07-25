/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#include "ui_odmrfspectrumform.h"
#include "odmrimaging.h"
#include "digitalcamera.h"
#include "odmrfspectrum.h"
#include "signalgenerator.h"

REGISTER_TYPE(XDriverList, ODMRFSpectrum, "ODMR frequency-swept spectrum measurement");


//---------------------------------------------------------------------------
XODMRFSpectrum::XODMRFSpectrum(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
    : XSecondaryDriver(name, runtime, ref(tr_meas), meas),
    m_clear(create<XTouchableNode>("Clear", true)),
    m_form(new FrmODMRFSpectrum),
    m_spectrum(create<XWaveNGraph>("Spectrum", true, m_form->m_graph, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump,
            m_form->m_tlbMath, meas, static_pointer_cast<XDriver>(shared_from_this()))),
    m_sg1(create<XItemNode<XDriverList, XSG> >(
          "SG1", false, ref(tr_meas), meas->drivers(), true)),
    m_odmr(create<XItemNode<XDriverList, XODMRImaging> >(
          "DigitalCamera", false, ref(tr_meas), meas->drivers(), true)),
    m_centerFreq(create<XDoubleNode>("CenterFreq", false)),
    m_freqSpan(create<XDoubleNode>("FreqSpan", false)),
    m_freqStep(create<XDoubleNode>("FreqStep", false)),
    m_active(create<XBoolNode>("Active", true)),
    m_repeatedly(create<XBoolNode>("Repeatedly", false)),
    m_altUpdateSubRegion(create<XBoolNode>("AltUpdateSubRegion", false)),
    m_subRegionMinFreq(create<XDoubleNode>("SubRegionMinFreq", false)),
    m_subRegionMaxFreq(create<XDoubleNode>("SubRegionMaxFreq", false)) {

    connect(sg1());
    connect(odmr());

    m_form->setWindowTitle(i18n("ODMR Spectrum (Freq. Sweep) - ") + getLabel() );

    iterate_commit([=](Transaction &tr){
        setupGraph(tr);

        tr[ *centerFreq()] = 20;
        tr[ *freqSpan()] = 200;
        tr[ *freqStep()] = 1;

    });
//    m_form->m_btnClear->setIcon(QApplication::style()->standardIcon(QStyle::SP_DialogResetButton));

    m_conUIs = {
        xqcon_create<XQLineEditConnector>(m_centerFreq, m_form->m_edCenterFreq),
        xqcon_create<XQLineEditConnector>(m_freqSpan, m_form->m_edFreqSpan),
        xqcon_create<XQLineEditConnector>(m_freqStep, m_form->m_edFreqStep),
        xqcon_create<XQComboBoxConnector>(m_sg1, m_form->m_cmbSG1, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(m_odmr, m_form->m_cmbCamera, ref(tr_meas)),
        xqcon_create<XQToggleButtonConnector>(m_active, m_form->m_ckbActive),
        xqcon_create<XQToggleButtonConnector>(m_repeatedly, m_form->m_ckbRepeatedly),
        xqcon_create<XQToggleButtonConnector>(m_altUpdateSubRegion, m_form->m_ckbAltUpdateSubRegion),
        xqcon_create<XQLineEditConnector>(m_subRegionMinFreq, m_form->m_edSubRegionMinFreq),
        xqcon_create<XQLineEditConnector>(m_subRegionMaxFreq, m_form->m_edSubRegionMaxFreq),
    };

    iterate_commit([=](Transaction &tr){
        m_lsnOnActiveChanged = tr[ *active()].onValueChanged().connectWeakly(
            shared_from_this(), &XODMRFSpectrum::onActiveChanged);
        tr[ *centerFreq()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *freqSpan()].onValueChanged().connect(m_lsnOnCondChanged);
        tr[ *freqStep()].onValueChanged().connect(m_lsnOnCondChanged);
    });

    m_conBaseUIs = {
//        xqcon_create<XQButtonConnector>(m_clear, m_form->m_btnClear),
    };

    iterate_commit([=](Transaction &tr){
        m_lsnOnClear = tr[ *m_clear].onTouch().connectWeakly(
            shared_from_this(), &XODMRFSpectrum::onClear);
    });
}
//---------------------------------------------------------------------------

void
XODMRFSpectrum::showForms() {
    m_form->showNormal();
    m_form->raise();
}

void
XODMRFSpectrum::onCondChanged(const Snapshot &shot, XValueNodeBase *node) {
    trans( *this).m_lastFreqAcquired = -1000.0;
    requestAnalysis();
}
void
XODMRFSpectrum::onClear(const Snapshot &shot, XTouchableNode *) {
    trans( *this).m_timeClearRequested = XTime::now();
    requestAnalysis();
}
bool
XODMRFSpectrum::checkDependency(const Snapshot &shot_this,
    const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) const {
    shared_ptr<XODMRImaging> odmr__ = shot_this[ *odmr()];
    if( !odmr__) return false;
    if(emitter == this) return true;
    if(emitter != odmr__.get())
        return false;
    shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
    if( !sg1__) return false;
    double freq = shot_others[ *sg1__].freq();
    if(shot_this[ *this].m_lastFreqAcquired == freq) {
        return false; //skips for the same freq.
    }
    return true;
}
void
XODMRFSpectrum::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) {
    const Snapshot &shot_this(tr);

    shared_ptr<XODMRImaging> odmr__ = shot_this[ *odmr()];
    const Snapshot &shot_odmr((emitter == odmr__.get()) ? shot_emitter : shot_others);

    if(shot_odmr[ *odmr__->incrementalAverage()]) {
        gWarnPrint(i18n("Do NOT use incremental avg. Skipping."));
        throw XSkippedRecordError(__FILE__, __LINE__);
    }

    bool clear = (shot_this[ *this].m_timeClearRequested.isSet());
    tr[ *this].m_timeClearRequested = {};

    double res = shot_this[ *freqStep()] * 1e-3 * 1e6; //Hz
    double cfreq = shot_this[ *centerFreq()]; //MHz
    double freq_span = shot_this[ *freqSpan()] * 1e-3; //MHz
    double max__ = (cfreq + freq_span/2) * 1e6; //Hz
    double min__ = (cfreq - freq_span/2) * 1e6; //Hz

    if(max__ <= min__) {
        throw XSkippedRecordError(i18n("Invalid min. and max."), __FILE__, __LINE__);
    }
    if((shot_this[ *this].res() != res) || (shot_this[ *this].numChannels() != shot_odmr[ *odmr__].numSamples()) ||  clear) {
        tr[ *this].m_res = res;
        tr[ *this].data.clear();
        tr[ *this].data.resize(shot_odmr[ *odmr__].numSamples());
    }
    else {
        //expands/shrinks the begining of buffers.
        int diff = lrint(shot_this[ *this].min() / res) - lrint(min__ / res);
        for(unsigned int ch = 0; ch < shot_this[ *this].numChannels(); ch++) {
            auto &accum = tr[ *this].data[ch].m_accum;
            auto &accum_weights = tr[ *this].data[ch].m_accum_weights;
            for(int i = 0; i < diff; i++) {
                accum.push_front(0.0);
                accum_weights.push_front(0);
            }
            for(int i = 0; i < -diff; i++) {
                if( !accum.empty()) {
                    accum.pop_front();
                    accum_weights.pop_front();
                }
            }
        }
    }
    tr[ *this].m_min = min__;
    //expands/shrinks the end of buffers.
    int length = lrint((max__ - min__) / res);
    for(unsigned int ch = 0; ch < shot_this[ *this].numChannels(); ch++) {
        auto &accum = tr[ *this].data[ch].m_accum;
        auto &accum_weights = tr[ *this].data[ch].m_accum_weights;
        accum.resize(length, 0.0);
        accum_weights.resize(length, 0);
        auto &wave = tr[ *this].data[ch].m_wave;
        auto &weights = tr[ *this].data[ch].m_weights;
        wave.resize(length);
        std::fill(wave.begin(), wave.end(), 0.0);
        weights.resize(length);
        std::fill(weights.begin(), weights.end(), 0.0);
    }

    if(clear) {
        tr[ *m_spectrum].clearPoints();
        throw XSkippedRecordError(__FILE__, __LINE__);
    }

    if(emitter == odmr__.get()) {
        shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
        double freq = shot_others[ *sg1__].freq(); //MHz
        tr[ *this].m_lastFreqAcquired = freq; //suppresses double accumulation.
        freq *= 1e6; //Hz
        unsigned int idx = lrint((freq - min__) / res);
        if(idx < length) {
            for(unsigned int ch = 0; ch < shot_this[ *this].numChannels(); ch++) {
                auto &accum = tr[ *this].data[ch].m_accum;
                auto &accum_weights = tr[ *this].data[ch].m_accum_weights;
//                accum[idx] += shot_odmr[ *odmr__].dPL(ch);
                if(shot_this[ *repeatedly()]) {
                    accum[idx] = 0;
                    accum_weights[idx] = 0;
                }
                accum[idx] += shot_odmr[ *odmr__].dPLoPL(ch);
                accum_weights[idx]++;
            }
        }
        m_isInstrumControlRequested = true;
    }
    else
        m_isInstrumControlRequested = false;


    for(unsigned int ch = 0; ch < shot_this[ *this].numChannels(); ch++) {
        auto &accum = tr[ *this].data[ch].m_accum;
        auto &accum_weights = tr[ *this].data[ch].m_accum_weights;
        auto &wave = tr[ *this].data[ch].m_wave;
        auto &weights = tr[ *this].data[ch].m_weights;
        for(unsigned int i = 0; i < length; i++) {
            wave[i] = accum[i] / accum_weights[i];
            weights[i] = accum_weights[i];
        }
    }
}

void
XODMRFSpectrum::visualize(const Snapshot &shot) {
    if( !shot[ *this].time()) {
        iterate_commit([=](Transaction &tr){
            tr[ *m_spectrum].clearPoints();
        });
        return;
    }

    if(m_isInstrumControlRequested.compare_set_strong((int)true, (int)false))
        rearrangeInstrum(shot);

    if( !shot[ *this].numChannels())
        return;
    int length = shot[ *this].wave(0).size();
    std::vector<double> values;
    double min__ = shot[ *this].min();
    double res = shot[ *this].res();
    m_spectrum->iterate_commit([=](Transaction &tr){
        if( !setupGraph(tr))
            return;
        unsigned int totlen = length;
        tr[ *m_spectrum].setRowCount(totlen);
        std::vector<double> colf(totlen);
        for(int i = 0; i < length; i++) {
            colf[i] = (min__ + i * res) * 1e-6;
        }
        tr[ *m_spectrum].setColumn(0, std::move(colf), 9);
        for(unsigned int ch = 0; ch < shot[ *this].numChannels(); ch++) {
            std::vector<double> coli(totlen), colw(totlen);
            const double *wave( &shot[ *this].wave(ch)[0]);
            const double *weights( &shot[ *this].weights(ch)[0]);
            for(int i = 0; i < length; i++) {
                coli[i] = wave[i];
                colw[i] = weights[i];
            }
            tr[ *m_spectrum].setColumn(2*ch + 1, std::move(coli), 5);
            tr[ *m_spectrum].setColumn(2*ch + 2, std::move(colw), 4);
        }
        m_spectrum->drawGraph(tr);
    });
}



void
XODMRFSpectrum::onActiveChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
    if(shot_this[ *active()]) {
        onClear(shot_this, clear().get());
        trans( *this).m_lastFreqAcquired = -1000.0;
        double cfreq = shot_this[ *centerFreq()]; //MHz
        double freq_span = shot_this[ *freqSpan()] * 1e-3; //MHz
        double newf = cfreq - freq_span / 2;
		shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
        shared_ptr<XODMRImaging> odmr__ = shot_this[ *odmr()];
        Snapshot shot_odmr( *odmr__);
        unsigned int seq_len = shot_odmr[ *odmr__].sequenceLength();
        shared_ptr<XDigitalCamera> camera__ = shot_odmr[ *odmr__->camera()];
        if(sg1__ && odmr__ && camera__) {
            sg1__->iterate_commit([=](Transaction &tr){
                tr[ *sg1__->freq()] = newf;
            });
            trans( *odmr__->clearAverage()).touch();
            msecsleep(200);
            double exposure = Snapshot( *camera__)[ *camera__].exposureTime();
            sg1__->iterate_commit([=](Transaction &tr){
                unsigned int avg = shot_odmr[ *odmr__->average()];
                avg = std::max(1u, avg);
                tr[ *sg1__->sweepPoints()] = seq_len * (avg + shot_odmr[ *odmr__->precedingSkips()]);
                tr[ *sg1__->sweepDwellTime()] = exposure + 0.05; //+50ms
            });
        }
    }
}

bool
XODMRFSpectrum::setupGraph(Transaction &tr) {
    int numch = std::max((int)Snapshot( *this)[ *this].numChannels(), 1);
    if(numch * 2 + 1 == tr[ *m_spectrum].colCount())
        return true;
    std::deque<XString> strs;
    std::vector<const char*> labels = {"Freq. [MHz]"};
    for(int i = 0; i < numch; ++i) {
//        strs.push_back(formatString("dPL%i", i));
        strs.push_back(formatString("dPL/PL%i", i));
        labels.push_back(strs.back().c_str());
        strs.push_back(formatString("Weights%i", i));
        labels.push_back(strs.back().c_str());
    }
    assert(labels.size() == numch * 2 + 1);
    tr[ *m_spectrum].setColCount(labels.size(), &labels[0]);
    if( !m_spectrum->clearPlots(tr))
        return false;
    for(int i = 0; i < numch; ++i) {
        if( !tr[ *m_spectrum].insertPlot(tr, labels[1 + 2*i], 0, 1 + 2*i, -1, 2 + 2*i))
            return false;
    }
    tr[ *m_spectrum].setLabel(0, "Freq [MHz]");
    tr[ *tr[ *m_spectrum].axisx()->label()] = i18n("Freq [MHz]");
//    tr[ *tr[ *m_spectrum].axisy()->label()] = i18n("dPL");
    tr[ *tr[ *m_spectrum].axisy()->label()] = i18n("dPL/PL");
    tr[ *tr[ *m_spectrum].axisw()->label()] = i18n("Weight");
    tr[ *m_spectrum].clearPoints();
    return true;
}

void
XODMRFSpectrum::rearrangeInstrum(const Snapshot &shot_this) {
    shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
    if( ! sg1__)
        return;
    Snapshot shot_sg( *sg1__);
    if( !shot_sg[ *sg1__].time())
        return;
    double freq = shot_this[ *this].m_lastFreqAcquired; //MHz
    //sets new freq
	if(shot_this[ *active()]) {
	    double cfreq = shot_this[ *centerFreq()]; //MHz
		double freq_span = shot_this[ *freqSpan()] * 1e-3; //MHz
		if(cfreq <= freq_span / 2) {
			throw XRecordError(i18n("Invalid center freq."), __FILE__, __LINE__);
		}

        double freq_step = shot_this[ *freqStep()] * 1e-3; //MHz
        if(freq_span < freq_step * 1.5) {
			throw XRecordError(i18n("Too large freq. step."), __FILE__, __LINE__);
		}
	  
        double newf = freq + freq_step; //MHz
        if(shot_this[ *altUpdateSubRegion()]) {
            bool was_inside_subregion = (freq >= shot_this[ *subRegionMinFreq()] - 1e-9) && (freq <= shot_this[ *subRegionMaxFreq()] + 1e-9);
            if( !was_inside_subregion) {
                m_lastFreqOutsideSubRegion = freq;
                newf = shot_this[ *subRegionMinFreq()];
                double df = newf - (cfreq - freq_span / 2) - 1e-9;
                newf = ceil(df / freq_step) * freq_step + (cfreq - freq_span / 2);
            }
            if(was_inside_subregion && (newf - 1e-9 > shot_this[ *subRegionMaxFreq()])) {
                //coming back to main region.
                if((m_lastFreqOutsideSubRegion > shot_this[ *subRegionMaxFreq()]) || (m_lastFreqOutsideSubRegion + freq_step + 1e-9 < shot_this[ *subRegionMinFreq()]))
                    newf = m_lastFreqOutsideSubRegion + freq_step;
            }

        }
        newf = round(newf * 1e8) / 1e8; //rounds
        if(newf >= cfreq + freq_span / 2) {
            if(shot_this[ *repeatedly()]) {
                newf = cfreq - freq_span / 2; //restarts
            }
            else {
                trans( *active()) = false; //finish
                return;
            }
        }
        shared_ptr<XODMRImaging> odmr__ = shot_this[ *odmr()];
        Snapshot shot_odmr( *odmr__);
        if(sg1__ && odmr__) {
            sg1__->iterate_commit([=](Transaction &tr){
                tr[ *sg1__->freq()] = newf;
            });
            trans( *odmr__->clearAverage()).touch();
            msecsleep(50);
            sg1__->iterate_commit([=](Transaction &tr){
                tr[ *sg1__->sweepPoints()] = (unsigned int)shot_sg[ *sg1__->sweepPoints()]; //initiates one sweep.
            });
        }
    }
}


