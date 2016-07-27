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
#include "ui_nmrspectrumform.h"
#include "nmrspectrum.h"
#include "dmm.h"
#include "magnetps.h"
#include "qdppms.h"

#include "nmrspectrumbase_impl.h"

REGISTER_TYPE(XDriverList, NMRSpectrum, "NMR field-swept spectrum measurement");

//---------------------------------------------------------------------------
XNMRSpectrum::XNMRSpectrum(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	  XNMRSpectrumBase<FrmNMRSpectrum>(name, runtime, ref(tr_meas), meas),
      m_magnet(create<XItemNode<XDriverList, XMagnetPS, XDMM, XQDPPMS> >(
		  "MagnetPS", false, ref(tr_meas), meas->drivers(), true)),
	  m_centerFreq(create<XDoubleNode>("CenterFreq", false)),
	  m_resolution(create<XDoubleNode>("Resolution", false)),
	  m_minValue(create<XDoubleNode>("FieldMin", false)),
	  m_maxValue(create<XDoubleNode>("FieldMax", false)),
	  m_fieldFactor(create<XDoubleNode>("FieldFactor", false)),
	  m_residualField(create<XDoubleNode>("ResidualField", false)) {

	connect(magnet());

	m_form->setWindowTitle(i18n("NMR Spectrum - ") + getLabel() );
    iterate_commit([=](Transaction &tr){
		tr[ *m_spectrum].setLabel(0, "Field [T]");
		tr[ *tr[ *m_spectrum].axisx()->label()] = i18n("Field [T]");

		tr[ *centerFreq()] = 20;
		tr[ *resolution()] = 0.001;
		tr[ *fieldFactor()] = 1;
		tr[ *maxValue()] = 5.0;
		tr[ *minValue()] = 3.0;
    });
  
	m_conCenterFreq = xqcon_create<XQLineEditConnector>(m_centerFreq, m_form->m_edFreq);
	m_conResolution = xqcon_create<XQLineEditConnector>(m_resolution, m_form->m_edResolution);
	m_conMin = xqcon_create<XQLineEditConnector>(m_minValue, m_form->m_edMin);
	m_conMax = xqcon_create<XQLineEditConnector>(m_maxValue, m_form->m_edMax);
	m_conFieldFactor = xqcon_create<XQLineEditConnector>(m_fieldFactor, m_form->m_edFieldFactor);
	m_conResidualField = xqcon_create<XQLineEditConnector>(m_residualField, m_form->m_edResidual);
	m_conMagnet = xqcon_create<XQComboBoxConnector>(m_magnet, m_form->m_cmbFieldEntry, ref(tr_meas));

    iterate_commit([=](Transaction &tr){
		tr[ *centerFreq()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *resolution()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *minValue()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *maxValue()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *fieldFactor()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *residualField()].onValueChanged().connect(m_lsnOnCondChanged);
    });
}
bool
XNMRSpectrum::onCondChangedImpl(const Snapshot &shot, XValueNodeBase *node) const {
    return (node == m_residualField.get()) || (node == m_fieldFactor.get()) || (node == m_resolution.get());
}
bool
XNMRSpectrum::checkDependencyImpl(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
    shared_ptr<XMagnetPS> magnet__ = shot_this[ *magnet()];
    shared_ptr<XDMM> dmm__ = shot_this[ *magnet()];
    shared_ptr<XQDPPMS> ppms__ = shot_this[ *magnet()];
    if( !(magnet__ || dmm__ || ppms__)) return false;
    if(emitter == magnet__.get()) return false;
    if(emitter == dmm__.get()) return false;
    if(emitter == ppms__.get()) return false;
    return true;
}
double
XNMRSpectrum::getFreqResHint(const Snapshot &shot_this) const {
	double res = fabs(shot_this[ *resolution()] / shot_this[ *maxValue()] * shot_this[ *centerFreq()]);
	res = std::min(res, fabs(shot_this[ *resolution()] / shot_this[ *minValue()] * shot_this[ *centerFreq()]));
	return res * 1e6;
}
double
XNMRSpectrum::getMinFreq(const Snapshot &shot_this) const {
	double freq = -log(shot_this[ *maxValue()]) * shot_this[ *centerFreq()];
	freq = std::min(freq, -log(shot_this[ *minValue()]) * shot_this[ *centerFreq()]);
	return freq * 1e6;
}
double
XNMRSpectrum::getMaxFreq(const Snapshot &shot_this) const {
	double freq = -log(shot_this[ *maxValue()]) * shot_this[ *centerFreq()];
    freq = std::max(freq, -log(shot_this[ *minValue()]) * shot_this[ *centerFreq()]);
    return freq * 1e6;
}
double
XNMRSpectrum::getCurrentCenterFreq(const Snapshot &shot_this, const Snapshot &shot_others) const {
	shared_ptr<XMagnetPS> magnet__ = shot_this[ *magnet()];
    shared_ptr<XDMM> dmm__ = shot_this[ *magnet()];
    shared_ptr<XQDPPMS> ppms__ = shot_this[ *magnet()];
    shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];

    assert( magnet__ || dmm__ || ppms__);
	double field;
	if(magnet__) {
		field = shot_others[ *magnet__].magnetField();
	}
    else if(dmm__ ) {
		field = shot_others[ *dmm__].value();
	}
    else {
        field = shot_others[ *ppms__].magnetField();
    }

	field *= shot_this[ *fieldFactor()];
	field += shot_this[ *residualField()];
 	
	return -log(field) * shot_this[ *centerFreq()] * 1e6;
}
void
XNMRSpectrum::getValues(const Snapshot &shot_this, std::vector<double> &values) const {
	int wave_size = shot_this[ *this].wave().size();
	double min__ = shot_this[ *this].min();
	double res = shot_this[ *this].res();
	double cfreq = shot_this[ *centerFreq()];
	values.resize(wave_size);
	for(unsigned int i = 0; i < wave_size; i++) {
		double freq = min__ + i * res;
		values[i] = exp( -freq * 1e-6 / cfreq);
	}
}
