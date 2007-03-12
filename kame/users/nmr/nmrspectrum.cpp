/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#include "forms/nmrspectrumform.h"
#include "nmrspectrum.h"
#include "nmrpulse.h"

#include "graph.h"
#include "graphwidget.h"
#include "xwavengraph.h"
#include "users/magnetps/magnetps.h"

#include <klocale.h>
#include <qpushbutton.h>
#include <qcombobox.h>
#include <kapplication.h>
#include <kiconloader.h>

//---------------------------------------------------------------------------
XNMRSpectrum::XNMRSpectrum(const char *name, bool runtime,
						   const shared_ptr<XScalarEntryList> &scalarentries,
						   const shared_ptr<XInterfaceList> &interfaces,
						   const shared_ptr<XThermometerList> &thermometers,
						   const shared_ptr<XDriverList> &drivers)
	: XSecondaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
	  m_pulse(create<XItemNode<XDriverList, XNMRPulseAnalyzer> >("PulseAnalyzer", false, drivers, true)),
	  m_magnet(create<XItemNode<XDriverList, XMagnetPS> >("MagnetPS", false, drivers, true)),
	  m_centerFreq(create<XDoubleNode>("CenterFreq", false)),
	  m_bandWidth(create<XDoubleNode>("BandWidth", false)),
	  m_resolution(create<XDoubleNode>("Resolution", false)),
	  m_fieldFactor(create<XDoubleNode>("FieldFactor", false)),
	  m_residualField(create<XDoubleNode>("ResidualField", false)),
	  m_fieldMin(create<XDoubleNode>("FieldMin", false)),
	  m_fieldMax(create<XDoubleNode>("FieldMax", false)),
	  m_clear(create<XNode>("Clear", true)),
	  m_form(new FrmNMRSpectrum(g_pFrmMain)),
	  m_spectrum(create<XWaveNGraph>("Spectrum", true,
									 m_form->m_graph, m_form->m_urlDump, m_form->m_btnDump))
{
    m_form->m_btnClear->setIconSet(
		KApplication::kApplication()->iconLoader()->loadIconSet("editdelete", 
																KIcon::Toolbar, KIcon::SizeSmall, true ) );      
    
	connect(magnet());
	connect(pulse());

	m_form->setCaption(KAME::i18n("NMR Spectrum - ") + getLabel() );

	{
		const char *labels[] = {"Field [T]", "Re [V]", "Im [V]", "Counts"};
		m_spectrum->setColCount(4, labels);
		m_spectrum->selectAxes(0, 1, 2, 3);
		m_spectrum->plot1()->label()->value(KAME::i18n("real part"));
		m_spectrum->plot1()->drawPoints()->value(false);
		m_spectrum->plot2()->label()->value(KAME::i18n("imag. part"));
		m_spectrum->plot2()->drawPoints()->value(false);
		m_spectrum->clear();
	}
  
	centerFreq()->value(20);
	bandWidth()->value(50);
	resolution()->value(0.001);
	fieldFactor()->value(1);
	fieldMax()->value(0.1);

	m_lsnOnClear = m_clear->onTouch().connectWeak(
		shared_from_this(), &XNMRSpectrum::onClear);
	m_lsnOnCondChanged = centerFreq()->onValueChanged().connectWeak(
		shared_from_this(), &XNMRSpectrum::onCondChanged);
	m_bandWidth->onValueChanged().connect(m_lsnOnCondChanged);
	m_resolution->onValueChanged().connect(m_lsnOnCondChanged);
	m_fieldFactor->onValueChanged().connect(m_lsnOnCondChanged);
	m_residualField->onValueChanged().connect(m_lsnOnCondChanged);
	m_fieldMin->onValueChanged().connect(m_lsnOnCondChanged);
	m_fieldMax->onValueChanged().connect(m_lsnOnCondChanged);

	m_conCenterFreq = xqcon_create<XQLineEditConnector>(m_centerFreq, m_form->m_edFreq);
	m_conBandWidth = xqcon_create<XQLineEditConnector>(m_bandWidth, m_form->m_edBW);
	m_conResolution = xqcon_create<XQLineEditConnector>(m_resolution, m_form->m_edResolution);
	m_conFieldFactor = xqcon_create<XQLineEditConnector>(m_fieldFactor, m_form->m_edFieldFactor);
	m_conResidualField = xqcon_create<XQLineEditConnector>(m_residualField, m_form->m_edResidual);
	m_conFieldMin = xqcon_create<XQLineEditConnector>(m_fieldMin, m_form->m_edHMin);
	m_conFieldMax = xqcon_create<XQLineEditConnector>(m_fieldMax, m_form->m_edHMax);
	m_conMagnet = xqcon_create<XQComboBoxConnector>(m_magnet, m_form->m_cmbFieldEntry);
	m_conPulse = xqcon_create<XQComboBoxConnector>(m_pulse, m_form->m_cmbPulse);
	m_conClear = xqcon_create<XQButtonConnector>(m_clear, m_form->m_btnClear);
}
void
XNMRSpectrum::showForms()
{
	m_form->show();
	m_form->raise();
}
void 
XNMRSpectrum::onCondChanged(const shared_ptr<XValueNodeBase> &node)
{
    if((node == m_residualField) || (node == m_resolution) || (node == m_fieldFactor))
        m_timeClearRequested = XTime::now();
    requestAnalysis();
}
void
XNMRSpectrum::onClear(const shared_ptr<XNode> &)
{
    m_timeClearRequested = XTime::now();
    requestAnalysis();
}
bool
XNMRSpectrum::checkDependency(const shared_ptr<XDriver> &emitter) const {
    shared_ptr<XMagnetPS> _magnet = *magnet();
    shared_ptr<XNMRPulseAnalyzer> _pulse = *pulse();
    if(!_magnet || !_pulse) return false;
    if(emitter == _magnet) return false;
    if(emitter == shared_from_this()) return true;
    return (emitter == _pulse);
}
void
XNMRSpectrum::analyze(const shared_ptr<XDriver> &emitter) throw (XRecordError&)
{
	shared_ptr<XMagnetPS> _magnet = *magnet();
	shared_ptr<XNMRPulseAnalyzer> _pulse = *pulse();
	ASSERT( _magnet );
	ASSERT( _magnet->time() );
	ASSERT( _pulse );
	ASSERT( _pulse->time() );
	ASSERT( emitter != _magnet );
  
//  if(fabs(_pulse->time() - _magnet->time()) > 10)
//        m_statusPrinter->printWarning(KAME::i18n("Recorded time is older by 10 sec"));
	double field = _magnet->magnetFieldRecorded();

	field *= *fieldFactor();
	field += *residualField();
 
	double field_max = *fieldMax();
	double field_min = *fieldMin();
	if(field_max <= field_min) {
		throw XRecordError(KAME::i18n("Invalid min. and max."), __FILE__, __LINE__);
	}
	double res = *resolution();
	if(res < 1e-6) {
		throw XRecordError(KAME::i18n("Too small resolution."), __FILE__, __LINE__);
	}
  
	bool clear = (m_timeClearRequested > _pulse->timeAwared());
  
	if((dH() != res) || clear)
    {
		m_dH = res;
		m_wave.clear();
		m_counts.clear();
    }
	else {
		for(int i = 0; i < rint(m_hMin / dH()) - rint(field_min / dH()); i++) {
			m_wave.push_front(0.0);
			m_counts.push_front(0);
		}
		for(int i = 0; i < rint(field_min / dH()) - rint(m_hMin / dH()); i++) {
			if(!m_wave.empty()) {
				m_wave.pop_front();
				m_counts.pop_front();
			}
		}
	}
	m_hMin = field_min;
	int length = lrint((field_max - field_min) / dH());
	m_wave.resize(length, 0.0);
	m_counts.resize(length, 0);

	if(clear) {
		m_spectrum->clear();
		throw XSkippedRecordError(__FILE__, __LINE__);
	}
	if(emitter != _pulse) throw XSkippedRecordError(__FILE__, __LINE__);

	int len = _pulse->ftWave().size();
	double df = _pulse->dFreq();
	if((len == 0) || (df == 0)) {
		throw XRecordError(KAME::i18n("Invalid waveform."), __FILE__, __LINE__);  
	}
	int bw = lrint(*bandWidth() * 1000.0 / df);
	double cfreq = *centerFreq() * 1e6;
	if(cfreq == 0) {
		throw XRecordError(KAME::i18n("Invalid center freq."), __FILE__, __LINE__);  
	}
	for(int i = std::max(0, (len - bw) / 2); i < std::min(len, (len + bw) / 2); i++)
    {
		double freq = (i - len/2) * df;
		if(freq / cfreq  != -1)
			add(field / (1 + freq / cfreq), _pulse->ftWave()[i]);
    }
}
void
XNMRSpectrum::visualize()
{
	if(!time()) {
		m_spectrum->clear();
		return;
	}

	int length = wave().size();
	{   XScopedWriteLock<XWaveNGraph> lock(*m_spectrum);
	m_spectrum->setRowCount(length);
	for(int i = 0; i < length; i++)
	{
		m_spectrum->cols(0)[i] = hMin() + i * dH();
		m_spectrum->cols(1)[i] = (counts()[i] > 0) ? std::real(wave()[i]) / counts()[i] : 0;
		m_spectrum->cols(2)[i] = (counts()[i] > 0) ? std::imag(wave()[i]) / counts()[i] : 0;
		m_spectrum->cols(3)[i] = counts()[i];
	}
	}
}

void
XNMRSpectrum::add(double field, std::complex<double> c)
{
	int idx = lrint((field - hMin()) / dH());
	if((idx >= (int)wave().size()) || (idx < 0)) return;
	m_wave[idx] += c;
	m_counts[idx]++;
	return;
}
