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
#include "ui_nmrfspectrumform.h"
#include "nmrfspectrum.h"
#include "signalgenerator.h"
#include "nmrspectrumbase_impl.h"
#include "autolctuner.h"
#include "pulserdriver.h"

REGISTER_TYPE(XDriverList, NMRFSpectrum, "NMR frequency-swept spectrum measurement");

//---------------------------------------------------------------------------
XNMRFSpectrum::XNMRFSpectrum(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
	: XNMRSpectrumBase<FrmNMRFSpectrum>(name, runtime, ref(tr_meas), meas),
	  m_sg1(create<XItemNode<XDriverList, XSG> >(
		  "SG1", false, ref(tr_meas), meas->drivers(), true)),
	  m_autoTuner(create<XItemNode<XDriverList, XAutoLCTuner> >(
          "AutoTuner", false, ref(tr_meas), meas->drivers(), true)),
	  m_pulser(create<XItemNode<XDriverList, XPulser> >(
		  "Pulser", false, ref(tr_meas), meas->drivers(), true)),
	  m_sg1FreqOffset(create<XDoubleNode>("SG1FreqOffset", false)),
	  m_centerFreq(create<XDoubleNode>("CenterFreq", false)),
	  m_freqSpan(create<XDoubleNode>("FreqSpan", false)),
	  m_freqStep(create<XDoubleNode>("FreqStep", false)),
	  m_active(create<XBoolNode>("Active", true)),
      m_tuneCycleStep(create<XDoubleNode>("TuneCycleStep", false)),
      m_tuneCycleStrategy(create<XComboNode>("TuneCycleStrategy", false, true)) {

	connect(sg1());
//	connect(autoTuner());
//	connect(pulser());

	m_form->setWindowTitle(i18n("NMR Spectrum (Freq. Sweep) - ") + getLabel() );

	iterate_commit([=](Transaction &tr){
		tr[ *m_spectrum].setLabel(0, "Freq [MHz]");
		tr[ *tr[ *m_spectrum].axisx()->label()] = i18n("Freq [MHz]");

		tr[ *centerFreq()] = 20;
        tr[ *sg1FreqOffset()] = 0;
		tr[ *freqSpan()] = 200;
		tr[ *freqStep()] = 1;

        for(auto &&x: {"As is", "Await Manual Tune", "Auto Tune", "Cyclic Avg. BPSK", "Cyclic Avg. QPSK", "Cyclic Avg. QPSKxP.I"})
            tr[ *tuneCycleStrategy()].add(x);
        tr[ *tuneCycleStrategy()] = (int)TuneCycleStrategy::ASIS;
    });
  
    m_conUIs = {
        xqcon_create<XQLineEditConnector>(m_sg1FreqOffset, m_form->m_edSG1FreqOffset),
        xqcon_create<XQLineEditConnector>(m_centerFreq, m_form->m_edCenterFreq),
        xqcon_create<XQLineEditConnector>(m_freqSpan, m_form->m_edFreqSpan),
        xqcon_create<XQLineEditConnector>(m_freqStep, m_form->m_edFreqStep),
        xqcon_create<XQComboBoxConnector>(m_sg1, m_form->m_cmbSG1, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(m_autoTuner, m_form->m_cmbAutoTuner, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector>(m_pulser, m_form->m_cmbPulser, ref(tr_meas)),
        xqcon_create<XQToggleButtonConnector>(m_active, m_form->m_ckbActive),
        xqcon_create<XQLineEditConnector>(m_tuneCycleStep, m_form->m_edTuneCycleStep),
        xqcon_create<XQComboBoxConnector>(m_tuneCycleStrategy, m_form->m_cmbTuneCycleStrategy, Snapshot( *m_tuneCycleStrategy))
    };

	iterate_commit([=](Transaction &tr){
		m_lsnOnActiveChanged = tr[ *active()].onValueChanged().connectWeakly(
			shared_from_this(), &XNMRFSpectrum::onActiveChanged);
		tr[ *centerFreq()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *freqSpan()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *freqStep()].onValueChanged().connect(m_lsnOnCondChanged);
    });
}

void
XNMRFSpectrum::onActiveChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
    if(shot_this[ *active()]) {
        switch((TuneCycleStrategy)(int)shot_this[ *tuneCycleStrategy()]) {
        case TuneCycleStrategy::ASIS:
            break;
        case TuneCycleStrategy::CYCLE_DBL:
        case TuneCycleStrategy::CYCLE_QUAD:
        case TuneCycleStrategy::CYCLE_OCT:
            {
                double x = shot_this[ *tuneCycleStep()] / shot_this[ *freqStep()];
                if((x < 0.9) || (fabs(x - lrint(x)) > 0.003 * x))
                    gErrPrint(i18n("Invalid cyclic step."));
            }
        case TuneCycleStrategy::TUNE_AWAIT:
        case TuneCycleStrategy::AUTOTUNE:
            {
                shared_ptr<XPulser> pulser__ = shot_this[ *pulser()];
                if( !pulser__)
                    gErrPrint(i18n("Pulser should be selected."));
                pulser__->iterate_commit([=](Transaction &tr){
                    tr[ *pulser__->firstPhase()] = 0;
                    tr[ *pulser__->invertPhase()] = false;
                });
            }
            if(shot_this[ *tuneCycleStep()]  <= 0.0)
                gErrPrint(i18n("Invalid tuning/cyclic step."));
            break;
        }

		onClear(shot_this, clear().get());
        m_lastFreqAcquired = -1000.0;
        m_tunedFreq = -1000.0;
        m_lastCycle = 0;
        double newf = getMinFreq(shot_this) * 1e-6; //MHz
        performTuning(shot_this, newf);
        newf += shot_this[ *sg1FreqOffset()];
		shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
		if(sg1__)
			trans( *sg1__->freq()) = newf;
	}
    else
    	m_lsnOnTuningChanged.reset();
}
bool
XNMRFSpectrum::onCondChangedImpl(const Snapshot &shot, XValueNodeBase *) {
    m_lastFreqAcquired = -1000.0;
    return false;
}
bool
XNMRFSpectrum::checkDependencyImpl(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
    shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
    if( !sg1__) return false;
    shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];
    if(emitter != pulse__.get()) return false;
    if(shot_emitter[ *pulse__].timeAwared() < shot_others[ *sg1__].time()) return false;
    double freq = getCurrentCenterFreq(shot_this, shot_others);
    if(m_lastFreqAcquired == freq) {
        return false; //skips for the same freq.
    }
    return true;
}
double
XNMRFSpectrum::getMinFreq(const Snapshot &shot_this) const{
	double cfreq = shot_this[ *centerFreq()]; //MHz
	double freq_span = shot_this[ *freqSpan()] * 1e-3; //MHz
	return (cfreq - freq_span/2) * 1e6;
}
double
XNMRFSpectrum::getMaxFreq(const Snapshot &shot_this) const{
	double cfreq = shot_this[ *centerFreq()]; //MHz
	double freq_span = shot_this[ *freqSpan()] * 1e-3; //MHz
	return (cfreq + freq_span/2) * 1e6;
}
double
XNMRFSpectrum::getFreqResHint(const Snapshot &shot_this) const {
	return 1e-6;
}
double
XNMRFSpectrum::getCurrentCenterFreq(const Snapshot &shot_this, const Snapshot &shot_others) const {
    shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
	assert( sg1__ );
	assert(shot_others[ *sg1__].time() );
    double freq = shot_others[ *sg1__].freq() - shot_this[ *sg1FreqOffset()]; //MHz
	return freq * 1e6;
}
void
XNMRFSpectrum::performTuning(const Snapshot &shot_this, double newf) {
    if((shot_this[ *tuneCycleStrategy()] != (int)TuneCycleStrategy::AUTOTUNE) &&
         (shot_this[ *tuneCycleStrategy()] != (int)TuneCycleStrategy::TUNE_AWAIT))
        return; //tuning is declined by user.

    if(fabs(m_tunedFreq - newf) <= shot_this[ *tuneCycleStep()] / 2 * 1e-3)
        return; //not needed yet

    newf += shot_this[ *tuneCycleStep()] / 2 * 1e-3; //to be tuned to

    shared_ptr<XPulser> pulser__ = shot_this[ *pulser()];
    if( !pulser__) {
        gWarnPrint(i18n("Pulser should be selected."));
        return;
    }
    //Tunes Capacitors.
    trans( *pulser__->output()) = false; // Pulse off.
    shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];
    if(pulse__)
        trans( *pulse__->avgClear()).touch();

    if((shot_this[ *tuneCycleStrategy()] == (int)TuneCycleStrategy::TUNE_AWAIT)) {
        g_statusPrinter->printMessage(getLabel() + " " + i18n("Tune it by yourself to") +
            formatString(" %.3f~MHz", newf) + i18n(", and turn pulse on again."), true, __FILE__, __LINE__, true);
    }
    if((shot_this[ *tuneCycleStrategy()] == (int)TuneCycleStrategy::AUTOTUNE)) {
        shared_ptr<XAutoLCTuner> autotuner = shot_this[ *autoTuner()];
        if( !autotuner) {
            gWarnPrint(i18n("AutoTuner should be selected."));
            return;
        }
        autotuner->iterate_commit([=](Transaction &tr){
            m_lsnOnTuningChanged = tr[ *autotuner->tuning()].onValueChanged().connectWeakly(
                shared_from_this(), &XNMRFSpectrum::onTuningChanged);
            tr[ *autotuner->target()] = newf;
        });
	}
    m_tunedFreq = newf;
}
void
XNMRFSpectrum::rearrangeInstrum(const Snapshot &shot_this) {
    m_lsnOnTuningChanged.reset();
    shared_ptr<XSG> sg1__ = shot_this[ *sg1()];
    if( ! sg1__)
        return;
    Snapshot shot_sg( *sg1__);
    if( !shot_sg[ *sg1__].time())
        return;
    double freq = getCurrentCenterFreq(shot_this, shot_sg);
    m_lastFreqAcquired = freq; //suppresses double accumulation.
    freq *= 1e-6; //MHz
    //sets new freq
	if(shot_this[ *active()]) {
	    double cfreq = shot_this[ *centerFreq()]; //MHz
		double freq_span = shot_this[ *freqSpan()] * 1e-3; //MHz
		if(cfreq <= freq_span / 2) {
			throw XRecordError(i18n("Invalid center freq."), __FILE__, __LINE__);
		}

        double freq_step = shot_this[ *tuneCycleStep()] * 1e-3; //MHz
        int num_psk_cycles = 0;
        switch((TuneCycleStrategy)(int)shot_this[ *tuneCycleStrategy()]) {
        default:
            freq_step = shot_this[ *freqStep()] * 1e-3; //MHz
            break;
        case TuneCycleStrategy::CYCLE_DBL:
            num_psk_cycles = 2; break;
        case TuneCycleStrategy::CYCLE_QUAD:
            num_psk_cycles = 4; break;
        case TuneCycleStrategy::CYCLE_OCT:
            num_psk_cycles = 8; break;
        }
        if(freq_span < freq_step * 1.5) {
			throw XRecordError(i18n("Too large freq. step."), __FILE__, __LINE__);
		}
	  
		double newf = freq; //MHz
		newf += freq_step;
		
        if(newf >= getMaxFreq(shot_this) * 1e-6) {
            double x = (newf - getMinFreq(shot_this) * 1e-6) / freq_step;
            newf -= floor(x + 0.01) * freq_step; //the first freq of this cycle.
            ++m_lastCycle;
            if(m_lastCycle >= num_psk_cycles) {
                m_lastCycle = 0;
                newf += shot_this[ *freqStep()] * 1e-3; //shifted.
                if((newf - getMinFreq(shot_this) * 1e-6) / freq_step > 0.99) {
                    trans( *active()) = false; //finish
                    return;
                }
            }
            shared_ptr<XPulser> pulser__ = shot_this[ *pulser()];
            if( !pulser__)
                throw XRecordError(i18n("Pulser should be selected."), __FILE__, __LINE__);
            if( ***pulser__->rtime() < 1000)
                throw XRecordError(i18n("Too short repetition period."), __FILE__, __LINE__);
            pulser__->iterate_commit([=](Transaction &tr){
                tr[ *pulser__->firstPhase()] = m_lastCycle % 4;
                tr[ *pulser__->invertPhase()] = (m_lastCycle / 4) ? true : false;
            });
        }

        newf = round(newf * 1e8) / 1e8; //rounds

        performTuning(shot_this, newf); //tunes a circuit if needed.

        newf += shot_this[ *sg1FreqOffset()]; //modifies SG freq.
        if(sg1__)
            trans( *sg1__->freq()) = newf;
    }
}
void
XNMRFSpectrum::onTuningChanged(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
    shared_ptr<XPulser> pulser__ = shot_this[ *pulser()];
    if( !pulser__) return;
//    if(shot_this[ *tuneStrategy()] != TUNEAUTOTUNER) return;
    {
        shared_ptr<XAutoLCTuner> autotuner = shot_this[ *autoTuner()];
        if(autotuner) {
            Snapshot shot_tuner( *autotuner);
            if(shot_tuner[ *autotuner->tuning()])
                return; //still tuner is running.
            if( !shot_tuner[ *autotuner->succeeded()])
                return; //awaiting manual tuning.
        }
    }
    m_lsnOnTuningChanged.reset();
    if(shot_this[ *active()]) {
        //Tuning has succeeded, go on.
        trans( *pulser__->output()) = true; // Pulse on.
        shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];
        if(pulse__)
            trans( *pulse__->avgClear()).touch();
    }
}
void
XNMRFSpectrum::getValues(const Snapshot &shot_this, std::vector<double> &values) const {
	int wave_size = shot_this[ *this].wave().size();
	double min__ = shot_this[ *this].min();
	double res = shot_this[ *this].res();
	values.resize(wave_size);
	for(unsigned int i = 0; i < wave_size; i++) {
		double freq = min__ + i * res;
		values[i] = freq * 1e-6;
	}
}
