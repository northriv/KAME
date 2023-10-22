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
//---------------------------------------------------------------------------
#include "ui_odmrfspectrumform.h"
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
            4, m_form->m_tlbMath, meas, static_pointer_cast<XDriver>(shared_from_this()))),
    m_sg1(create<XItemNode<XDriverList, XSG> >(
          "SG1", false, ref(tr_meas), meas->drivers(), true)),
    m_camera(create<XItemNode<XDriverList, XDigitalCamera> >(
          "DigitalCamera", false, ref(tr_meas), meas->drivers(), true)),
      m_centerFreq(create<XDoubleNode>("CenterFreq", false)),
      m_freqSpan(create<XDoubleNode>("FreqSpan", false)),
      m_freqStep(create<XDoubleNode>("FreqStep", false)),
      m_active(create<XBoolNode>("Active", true)) {

    connect(sg1());
    connect(camera());

    m_form->setWindowTitle(i18n("ODMR Spectrum (Freq. Sweep) - ") + getLabel() );

    iterate_commit([=](Transaction &tr){
        const char *labels[] = {"Ch", "Freq. [MHz]", "Intens", "Weights"};
        tr[ *m_spectrum].setColCount(4, labels);
        tr[ *m_spectrum].insertPlot(labels[2], 1, 2, -1, 3, 0);
        tr[ *tr[ *m_spectrum].axisy()->label()] = i18n("Intens.");
        tr[ *m_spectrum].clearPoints();

        tr[ *m_spectrum].setLabel(0, "Freq [MHz]");
        tr[ *tr[ *m_spectrum].axisx()->label()] = i18n("Freq [MHz]");

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
        xqcon_create<XQComboBoxConnector>(m_camera, m_form->m_cmbCamera, ref(tr_meas)),
        xqcon_create<XQToggleButtonConnector>(m_active, m_form->m_ckbActive),
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
    m_lastFreqAcquired = -1000.0;
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
    shared_ptr<XDigitalCamera> camera__ = shot_this[ *camera()];
    if( !camera__) return false;
//    if(emitter == this) return true;
    if(emitter != camera__.get())
        return false;
    shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
    if( !sg1__) return false;
    double freq = shot_others[ *sg1__].freq() * 1e6;
    if(m_lastFreqAcquired == freq) {
        return false; //skips for the same freq.
    }
    return true;
}
void
XODMRFSpectrum::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) {
    const Snapshot &shot_this(tr);

    shared_ptr<XDigitalCamera> camera__ = shot_this[ *camera()];
    const Snapshot &shot_camera((emitter == camera__.get()) ? shot_emitter : shot_others);

    if(shot_camera[ *camera__->incrementalAverage()]) {
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
    if((shot_this[ *this].res() != res) || (shot_this[ *this].numChannels() != 1) ||  clear) {
        //! todo multich
        tr[ *this].m_res = res;
        tr[ *this].data.clear();
        tr[ *this].data.resize(1);
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

    if(emitter == camera__.get()) {
        shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
        double freq = shot_others[ *sg1__].freq() * 1e6;
        unsigned int idx = lrint((freq - min__) / res);
        if(idx < length) {
            for(unsigned int ch = 0; ch < shot_this[ *this].numChannels(); ch++) {
                auto &accum = tr[ *this].data[ch].m_accum;
                auto &accum_weights = tr[ *this].data[ch].m_accum_weights;
                accum[idx] += 1;
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
        unsigned int totlen = length * shot[ *this].numChannels();
        tr[ *m_spectrum].setRowCount(totlen);
        std::vector<double> colf(totlen), colch(totlen), coli(totlen), colw(length);
        for(unsigned int ch = 0; ch < shot[ *this].numChannels(); ch++) {
            const double *wave( &shot[ *this].wave(ch)[0]);
            const double *weights( &shot[ *this].weights(ch)[0]);
            for(int i = 0; i < length; i++) {
                colch[i + length * ch] = ch;
                colf[i + length * ch] = (min__ + i * res) * 1e-6;
                coli[i + length * ch] = wave[i];
                colw[i + length * ch] = weights[i];
            }
        }
        tr[ *m_spectrum].setColumn(0, std::move(colch), 0);
        tr[ *m_spectrum].setColumn(1, std::move(colf), 9);
        tr[ *m_spectrum].setColumn(2, std::move(coli), 5);
        tr[ *m_spectrum].setColumn(3, std::move(colw), 4);
        m_spectrum->drawGraph(tr);
    });
}



void
XODMRFSpectrum::onActiveChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
    if(shot_this[ *active()]) {

		onClear(shot_this, clear().get());
        m_lastFreqAcquired = -1000.0;
        double cfreq = shot_this[ *centerFreq()]; //MHz
        double freq_span = shot_this[ *freqSpan()] * 1e-3; //MHz
        double newf = cfreq - freq_span / 2;
		shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
		if(sg1__)
			trans( *sg1__->freq()) = newf;
	}
}

void
XODMRFSpectrum::rearrangeInstrum(const Snapshot &shot_this) {
    shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
    if( ! sg1__)
        return;
    Snapshot shot_sg( *sg1__);
    if( !shot_sg[ *sg1__].time())
        return;
    double freq = shot_sg[ *sg1__].freq() * 1e6;
    m_lastFreqAcquired = freq; //suppresses double accumulation.
    freq *= 1e-6; //MHz
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
	  
		double newf = freq; //MHz
		newf += freq_step;
        newf = round(newf * 1e8) / 1e8; //rounds
        if((newf - (cfreq - freq_span / 2)) / freq_step > 0.99) {
            trans( *active()) = false; //finish
            return;
        }
        shared_ptr<XDigitalCamera> camera__ = shot_this[ *camera()];
        Snapshot shot_camera( *camera__);
        if(sg1__ && camera__) {
            sg1__->iterate_commit([=](Transaction &tr){
                tr[ *sg1__->freq()] = newf;
                unsigned int avg = shot_camera[ *camera__->average()];
                avg = std::max(1u, avg);
                tr[ *sg1__->sweepPoints()] = 2 * avg;
            });
        }
    }
}


