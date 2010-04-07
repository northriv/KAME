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
//---------------------------------------------------------------------------
#include "ui_nmrfspectrumform.h"
#include "nmrfspectrum.h"
#include "signalgenerator.h"
#include "nmrspectrumbase_impl.h"

REGISTER_TYPE(XDriverList, NMRFSpectrum, "NMR frequency-swept spectrum measurement");

//---------------------------------------------------------------------------
XNMRFSpectrum::XNMRFSpectrum(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
	: XNMRSpectrumBase<FrmNMRFSpectrum>(name, runtime, ref(tr_meas), meas),
	  m_sg1(create<XItemNode<XDriverList, XSG> >(
		  "SG1", false, ref(tr_meas), meas->drivers(), true)),
	  m_sg1FreqOffset(create<XDoubleNode>("SG1FreqOffset", false)),
	  m_centerFreq(create<XDoubleNode>("CenterFreq", false)),
	  m_freqSpan(create<XDoubleNode>("FreqSpan", false)),
	  m_freqStep(create<XDoubleNode>("FreqStep", false)),
	  m_burstCount(create<XUIntNode>("BurstCount", false)),
	  m_active(create<XBoolNode>("Active", true)) {
	connect(sg1());

	m_form->setWindowTitle(i18n("NMR Spectrum (Freq. Sweep) - ") + getLabel() );

	for(Transaction tr( *m_spectrum);; ++tr) {
		tr[ *m_spectrum].setLabel(0, "Freq [MHz]");
		tr[ *tr[ *m_spectrum].axisx()->label()] = i18n("Freq [MHz]");
		if(tr.commit())
			break;
	}
  
	centerFreq()->value(20);
	sg1FreqOffset()->value(700);
	freqSpan()->value(200);
	freqStep()->value(1);

	m_conSG1FreqOffset = xqcon_create<XQLineEditConnector>(m_sg1FreqOffset, m_form->m_edSG1FreqOffset);
	m_conCenterFreq = xqcon_create<XQLineEditConnector>(m_centerFreq, m_form->m_edCenterFreq);
	m_conFreqSpan = xqcon_create<XQLineEditConnector>(m_freqSpan, m_form->m_edFreqSpan);
	m_conFreqStep = xqcon_create<XQLineEditConnector>(m_freqStep, m_form->m_edFreqStep);
	m_conSG1 = xqcon_create<XQComboBoxConnector>(m_sg1, m_form->m_cmbSG1, ref(tr_meas));
	m_form->m_numBurstCount->setRange(0, 15);
	m_conBurstCount = xqcon_create<XQSpinBoxConnector>(m_burstCount, m_form->m_numBurstCount);
	m_conActive = xqcon_create<XQToggleButtonConnector>(m_active, m_form->m_ckbActive);

	for(Transaction tr( *this);; ++tr) {
		m_lsnOnActiveChanged = tr[ *active()].onValueChanged().connectWeakly(
			shared_from_this(), &XNMRFSpectrum::onActiveChanged);
		tr[ *centerFreq()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *freqSpan()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *freqStep()].onValueChanged().connect(m_lsnOnCondChanged);
		if(tr.commit())
			break;
	}
}

void
XNMRFSpectrum::onActiveChanged(const Snapshot &shot, XValueNodeBase *) {
    if( *active()) {
		m_burstFreqCycleCount = 0;
		m_burstPhaseCycleCount = 0;
		shared_ptr<XSG> _sg1 = *sg1();
		if(_sg1)
			_sg1->freq()->value( *centerFreq() - *freqSpan()/2e3 + *sg1FreqOffset());
		Snapshot shot_this( *this);
		onClear(shot_this, clear().get());
	}
}
bool
XNMRFSpectrum::onCondChangedImpl(const Snapshot &shot, XValueNodeBase *) const {
    return false;
}
bool
XNMRFSpectrum::checkDependencyImpl(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
    shared_ptr<XSG> _sg1 = shot_this[ *sg1()];
    if( !_sg1) return false;
	shared_ptr<XNMRPulseAnalyzer> _pulse = shot_this[ *pulse()];
    if(shot_emitter[ *_pulse].timeAwared() < shot_others[ *_sg1].time()) return false;
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
    shared_ptr<XSG> _sg1 = shot_this[ *sg1()];
	ASSERT( _sg1 );
	ASSERT(shot_others[ *_sg1].time() );
    double freq = shot_others[ *_sg1].freq() - shot_this[ *sg1FreqOffset()]; //MHz
	return freq * 1e6;
}
void
XNMRFSpectrum::rearrangeInstrum(const Snapshot &shot_this) {
	//set new freq
	if(shot_this[ *active()]) {
	    shared_ptr<XSG> _sg1 = shot_this[ *sg1()];
		if( ! _sg1)
			return;
		Snapshot shot_sg( *_sg1);
		if( !shot_sg[ *_sg1].time())
			return;

		double freq = getCurrentCenterFreq(shot_this, shot_sg) * 1e-6;

	    double cfreq = shot_this[ *centerFreq()]; //MHz
		double freq_span = shot_this[ *freqSpan()] * 1e-3; //MHz
		double freq_step = shot_this[ *freqStep()] * 1e-3; //MHz
		if(cfreq <= freq_span / 2) {
			throw XRecordError(i18n("Invalid center freq."), __FILE__, __LINE__);
		}
		if(freq_span <= freq_step * 2) {
			throw XRecordError(i18n("Too large freq. step."), __FILE__, __LINE__);
		}
	  
		int burstcnt = shot_this[ *burstCount()];
		double newf = freq; //MHz
		m_burstFreqCycleCount++;
		if(burstcnt) {
			newf += freq_span / burstcnt;
			if(m_burstFreqCycleCount >= burstcnt) {
				m_burstFreqCycleCount = 0;
				newf -= freq_span;
				m_burstPhaseCycleCount++;
			}
		}
		if( !burstcnt || (m_burstPhaseCycleCount >= 4)) {
			m_burstPhaseCycleCount = 0;
			newf += freq_step;
		}
		
		if(_sg1)
			_sg1->freq()->value(newf + shot_this[ *sg1FreqOffset()]);
		if(newf >= getMaxFreq(shot_this) * 1e-6 - freq_step)
			trans( *active()) = false;
	}	
}

void
XNMRFSpectrum::getValues(const Snapshot &shot_this, std::vector<double> &values) const {
	values.resize(shot_this[ *this].wave().size());
	for(unsigned int i = 0; i < shot_this[ *this].wave().size(); i++) {
		double freq = shot_this[ *this].min() + i * shot_this[ *this].res();
		values[i] = freq * 1e-6;
	}
}
