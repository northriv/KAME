/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
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
#include "nmrspectrumform.h"
#include "nmrspectrum.h"
#include "dmm.h"
#include "magnetps.h"

#include "nmrspectrumbase.cpp"
REGISTER_TYPE(XDriverList, NMRSpectrum, "NMR field-swept spectrum measurement");

//---------------------------------------------------------------------------
XNMRSpectrum::XNMRSpectrum(const char *name, bool runtime,
						   const shared_ptr<XScalarEntryList> &scalarentries,
						   const shared_ptr<XInterfaceList> &interfaces,
						   const shared_ptr<XThermometerList> &thermometers,
						   const shared_ptr<XDriverList> &drivers)
	: XNMRSpectrumBase<FrmNMRSpectrum>(name, runtime, scalarentries, interfaces, thermometers, drivers),
	  m_magnet(create<XItemNode<XDriverList, XMagnetPS, XDMM> >("MagnetPS", false, drivers, true)),
	  m_centerFreq(create<XDoubleNode>("CenterFreq", false)),
	  m_resolution(create<XDoubleNode>("Resolution", false)),
	  m_minValue(create<XDoubleNode>("FieldMin", false)),
	  m_maxValue(create<XDoubleNode>("FieldMax", false)),
	  m_fieldFactor(create<XDoubleNode>("FieldFactor", false)),
	  m_residualField(create<XDoubleNode>("ResidualField", false))
{
	connect(magnet());

	m_form->setCaption(KAME::i18n("NMR Spectrum - ") + getLabel() );
	m_spectrum->setLabel(0, "Field [T]");
  
	centerFreq()->value(20);
	resolution()->value(0.001);
	fieldFactor()->value(1);
	maxValue()->value(0.1);

	centerFreq()->onValueChanged().connect(m_lsnOnCondChanged);
	resolution()->onValueChanged().connect(m_lsnOnCondChanged);
	minValue()->onValueChanged().connect(m_lsnOnCondChanged);
	maxValue()->onValueChanged().connect(m_lsnOnCondChanged);
	fieldFactor()->onValueChanged().connect(m_lsnOnCondChanged);
	residualField()->onValueChanged().connect(m_lsnOnCondChanged);

	m_conCenterFreq = xqcon_create<XQLineEditConnector>(m_centerFreq, m_form->m_edFreq);
	m_conResolution = xqcon_create<XQLineEditConnector>(m_resolution, m_form->m_edResolution);
	m_conMin = xqcon_create<XQLineEditConnector>(m_minValue, m_form->m_edMin);
	m_conMax = xqcon_create<XQLineEditConnector>(m_maxValue, m_form->m_edMax);
	m_conFieldFactor = xqcon_create<XQLineEditConnector>(m_fieldFactor, m_form->m_edFieldFactor);
	m_conResidualField = xqcon_create<XQLineEditConnector>(m_residualField, m_form->m_edResidual);
	m_conMagnet = xqcon_create<XQComboBoxConnector>(m_magnet, m_form->m_cmbFieldEntry);
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
XNMRSpectrum::getResolution() const{
	return *resolution();
}
double
XNMRSpectrum::getMinValue() const{
	return rint(*minValue() / *resolution()) * *resolution();
}
double
XNMRSpectrum::getMaxValue() const{
	return rint(*maxValue() / *resolution()) * *resolution();
}
void
XNMRSpectrum::fssum()
{
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
 
	int len = _pulse->ftWave().size();
	double df = _pulse->dFreq();
	int bw = lrint(*bandWidth() * 1000.0 / df);
	double cfreq = *centerFreq() * 1e6;
	if(cfreq == 0) {
		throw XRecordError(KAME::i18n("Invalid center freq."), __FILE__, __LINE__);  
	}
	for(int i = std::max(0, (len - bw) / 2); i < std::min(len, (len + bw) / 2); i++) {
		double freq = (i - len/2) * df;
		if(freq / cfreq  != -1) {
			int idx = lrint((field / (1 + freq / cfreq) - minRecorded()) / resRecorded());
			if((idx >= (int)wave().size()) || (idx < 0))
				continue;
			m_wave[idx] += _pulse->ftWave()[i];
			m_counts[idx]++;
		}
	}
}
