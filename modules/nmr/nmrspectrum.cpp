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
#include "ui_nmrspectrumform.h"
#include "nmrspectrum.h"
#include "dmm.h"
#include "magnetps.h"

#include "nmrspectrumbase_impl.h"

REGISTER_TYPE(XDriverList, NMRSpectrum, "NMR field-swept spectrum measurement");

//---------------------------------------------------------------------------
XNMRSpectrum::XNMRSpectrum(const char *name, bool runtime,
						   const shared_ptr<XScalarEntryList> &scalarentries,
						   const shared_ptr<XInterfaceList> &interfaces,
						   const shared_ptr<XThermometerList> &thermometers,
						   const shared_ptr<XDriverList> &drivers)
	:
	  XNMRSpectrumBase<FrmNMRSpectrum>(name, runtime, scalarentries, interfaces, thermometers, drivers),
	  m_magnet(create<XItemNode<XDriverList, XMagnetPS, XDMM> >("MagnetPS", false, drivers, true)),
	  m_centerFreq(create<XDoubleNode>("CenterFreq", false)),
	  m_resolution(create<XDoubleNode>("Resolution", false)),
	  m_minValue(create<XDoubleNode>("FieldMin", false)),
	  m_maxValue(create<XDoubleNode>("FieldMax", false)),
	  m_fieldFactor(create<XDoubleNode>("FieldFactor", false)),
	  m_residualField(create<XDoubleNode>("ResidualField", false))
{
	connect(magnet());

	m_form->setWindowTitle(i18n("NMR Spectrum - ") + getLabel() );
	m_spectrum->setLabel(0, "Field [T]");
	m_spectrum->axisx()->label()->value(i18n("Field [T]"));
  
	centerFreq()->value(20);
	resolution()->value(0.001);
	fieldFactor()->value(1);
	maxValue()->value(5.0);
	minValue()->value(3.0);

	m_conCenterFreq = xqcon_create<XQLineEditConnector>(m_centerFreq, m_form->m_edFreq);
	m_conResolution = xqcon_create<XQLineEditConnector>(m_resolution, m_form->m_edResolution);
	m_conMin = xqcon_create<XQLineEditConnector>(m_minValue, m_form->m_edMin);
	m_conMax = xqcon_create<XQLineEditConnector>(m_maxValue, m_form->m_edMax);
	m_conFieldFactor = xqcon_create<XQLineEditConnector>(m_fieldFactor, m_form->m_edFieldFactor);
	m_conResidualField = xqcon_create<XQLineEditConnector>(m_residualField, m_form->m_edResidual);
	m_conMagnet = xqcon_create<XQComboBoxConnector>(m_magnet, m_form->m_cmbFieldEntry);

	centerFreq()->onValueChanged().connect(m_lsnOnCondChanged);
	resolution()->onValueChanged().connect(m_lsnOnCondChanged);
	minValue()->onValueChanged().connect(m_lsnOnCondChanged);
	maxValue()->onValueChanged().connect(m_lsnOnCondChanged);
	fieldFactor()->onValueChanged().connect(m_lsnOnCondChanged);
	residualField()->onValueChanged().connect(m_lsnOnCondChanged);
}
bool
XNMRSpectrum::onCondChangedImpl(const shared_ptr<XValueNodeBase> &node) const
{
    return (node == m_residualField) || (node == m_fieldFactor) || (node == m_resolution);
}
bool
XNMRSpectrum::checkDependencyImpl(const shared_ptr<XDriver> &emitter) const {
    shared_ptr<XMagnetPS> _magnet = *magnet();
    shared_ptr<XDMM> _dmm = *magnet();
    if(!(_magnet || _dmm)) return false;
    if(emitter == _magnet) return false;
    if(emitter == _dmm) return false;
    return true;
}
double
XNMRSpectrum::getFreqResHint() const {
	double res = fabs(*resolution() / *maxValue() * *centerFreq());	
	res = std::min(res, fabs(*resolution() / *minValue() * *centerFreq()));
	return res * 1e6;
}
double
XNMRSpectrum::getMinFreq() const {
	double freq = -log(*maxValue()) * *centerFreq();	
	freq = std::min(freq, -log(*minValue()) * *centerFreq());
	return freq * 1e6;
}
double
XNMRSpectrum::getMaxFreq() const {
	double freq = -log(*maxValue()) * *centerFreq();	
	freq = std::max(freq, -log(*minValue()) * *centerFreq());
	return freq * 1e6;
}
double
XNMRSpectrum::getCurrentCenterFreq() const {
	shared_ptr<XMagnetPS> _magnet = *magnet();
    shared_ptr<XDMM> _dmm = *magnet();
	shared_ptr<XNMRPulseAnalyzer> _pulse = *pulse();

	ASSERT( _magnet || _dmm );
	double field;
	if(_magnet)
		field = _magnet->magnetFieldRecorded();
	else
		field = _dmm->valueRecorded();

	field *= *fieldFactor();
	field += *residualField();
 	
	return -log(field) * *centerFreq() * 1e6;
}
void
XNMRSpectrum::getValues(std::vector<double> &values) const {
	values.resize(wave().size());
	for(unsigned int i = 0; i < wave().size(); i++) {
		double freq = minRecorded() + i*resRecorded();
		values[i] = exp(-freq * 1e-6 / *centerFreq());
	}
}
