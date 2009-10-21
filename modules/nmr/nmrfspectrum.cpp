/***************************************************************************
		Copyright (C) 2002-2009 Kentaro Kitagawa
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
							 const shared_ptr<XScalarEntryList> &scalarentries,
							 const shared_ptr<XInterfaceList> &interfaces,
							 const shared_ptr<XThermometerList> &thermometers,
							 const shared_ptr<XDriverList> &drivers)
	: XNMRSpectrumBase<FrmNMRFSpectrum>(name, runtime, scalarentries, interfaces, thermometers, drivers),
	  m_sg1(create<XItemNode<XDriverList, XSG> >("SG1", false, drivers, true)),
	  m_sg1FreqOffset(create<XDoubleNode>("SG1FreqOffset", false)),
	  m_centerFreq(create<XDoubleNode>("CenterFreq", false)),
	  m_freqSpan(create<XDoubleNode>("FreqSpan", false)),
	  m_freqStep(create<XDoubleNode>("FreqStep", false)),
	  m_burstCount(create<XUIntNode>("BurstCount", false)),
	  m_active(create<XBoolNode>("Active", true))
{
	connect(sg1(), true);

	m_form->setWindowTitle(i18n("NMR Spectrum (Freq. Sweep) - ") + getLabel() );

	m_spectrum->setLabel(0, "Freq [MHz]");
	m_spectrum->axisx()->label()->value(i18n("Freq [MHz]"));
  
	centerFreq()->value(20);
	sg1FreqOffset()->value(700);
	freqSpan()->value(200);
	freqStep()->value(1);

	m_conSG1FreqOffset = xqcon_create<XQLineEditConnector>(m_sg1FreqOffset, m_form->m_edSG1FreqOffset);
	m_conCenterFreq = xqcon_create<XQLineEditConnector>(m_centerFreq, m_form->m_edCenterFreq);
	m_conFreqSpan = xqcon_create<XQLineEditConnector>(m_freqSpan, m_form->m_edFreqSpan);
	m_conFreqStep = xqcon_create<XQLineEditConnector>(m_freqStep, m_form->m_edFreqStep);
	m_conSG1 = xqcon_create<XQComboBoxConnector>(m_sg1, m_form->m_cmbSG1);
	m_form->m_numBurstCount->setRange(0, 15);
	m_conBurstCount = xqcon_create<XQSpinBoxConnector>(m_burstCount, m_form->m_numBurstCount);
	m_conActive = xqcon_create<XQToggleButtonConnector>(m_active, m_form->m_ckbActive);

	m_lsnOnActiveChanged = active()->onValueChanged().connectWeak(
		shared_from_this(), &XNMRFSpectrum::onActiveChanged);
	centerFreq()->onValueChanged().connect(m_lsnOnCondChanged);
	freqSpan()->onValueChanged().connect(m_lsnOnCondChanged);
	freqStep()->onValueChanged().connect(m_lsnOnCondChanged);
}

void
XNMRFSpectrum::onActiveChanged(const shared_ptr<XValueNodeBase> &)
{
    if(*active()) {
		m_burstFreqCycleCount = 0;
		m_burstPhaseCycleCount = 0;
		shared_ptr<XSG> _sg1 = *sg1();
		if(_sg1) _sg1->freq()->value(*centerFreq() - *freqSpan()/2e3 + *sg1FreqOffset());
		onClear(shared_from_this());
	}
}
bool
XNMRFSpectrum::onCondChangedImpl(const shared_ptr<XValueNodeBase> &) const
{
    return false;
}
bool
XNMRFSpectrum::checkDependencyImpl(const shared_ptr<XDriver> &) const {
    shared_ptr<XSG> _sg1 = *sg1();
    if(!_sg1) return false;
	shared_ptr<XNMRPulseAnalyzer> _pulse = *pulse();
    if(_pulse->timeAwared() < _sg1->time()) return false;
    return true;
}
double
XNMRFSpectrum::getMinFreq() const{
	double cfreq = *centerFreq(); //MHz
	double freq_span = *freqSpan() * 1e-3; //MHz
	return (cfreq - freq_span/2) * 1e6;
}
double
XNMRFSpectrum::getMaxFreq() const{
	double cfreq = *centerFreq(); //MHz
	double freq_span = *freqSpan() * 1e-3; //MHz
	return (cfreq + freq_span/2) * 1e6;
}
double
XNMRFSpectrum::getFreqResHint() const {
	return 1e-6;
}
double
XNMRFSpectrum::getCurrentCenterFreq() const {
    shared_ptr<XSG> _sg1 = *sg1();
	ASSERT( _sg1 );
	ASSERT( _sg1->time() );
    double freq = _sg1->freqRecorded() - *sg1FreqOffset(); //MHz
	return freq * 1e6;
}
void
XNMRFSpectrum::afterFSSum() {
	double freq = getCurrentCenterFreq() * 1e-6;
	//set new freq
	if(*active()) {
	    shared_ptr<XSG> _sg1 = *sg1();
		ASSERT( _sg1 );
		ASSERT( _sg1->time() );

	    double cfreq = *centerFreq(); //MHz
		double freq_span = *freqSpan() * 1e-3; //MHz
		double freq_step = *freqStep() * 1e-3; //MHz
		if(cfreq <= freq_span/2) {
			throw XRecordError(i18n("Invalid center freq."), __FILE__, __LINE__);
		}
		if(freq_span <= freq_step*2) {
			throw XRecordError(i18n("Too large freq. step."), __FILE__, __LINE__);
		}
	  
		if(_sg1) unlockConnection(_sg1);
		
		int burstcnt = *burstCount();
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
		if(!burstcnt || (m_burstPhaseCycleCount >= 4)) {
			m_burstPhaseCycleCount = 0;
			newf += freq_step;
		}
		
		if(_sg1) _sg1->freq()->value(newf + *sg1FreqOffset());
		if(newf >= getMaxFreq() * 1e-6 - freq_step)
			active()->value(false);
	}	
}

void
XNMRFSpectrum::getValues(std::vector<double> &values) const {
	values.resize(wave().size());
	for(unsigned int i = 0; i < wave().size(); i++) {
		double freq = minRecorded() + i*resRecorded();
		values[i] = freq * 1e-6;
	}
}
