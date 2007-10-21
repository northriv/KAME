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
#include "nmrspectrumbase.h"
#include "nmrpulse.h"

#include <graph.h>
#include <graphwidget.h>
#include <xwavengraph.h>

#include <klocale.h>
#include <qpushbutton.h>
#include <qcombobox.h>
#include <qcheckbox.h>
#include <kapplication.h>
#include <knuminput.h>
#include <kiconloader.h>
//---------------------------------------------------------------------------
template <class FRM>
XNMRSpectrumBase<FRM>::XNMRSpectrumBase(const char *name, bool runtime,
						   const shared_ptr<XScalarEntryList> &scalarentries,
						   const shared_ptr<XInterfaceList> &interfaces,
						   const shared_ptr<XThermometerList> &thermometers,
						   const shared_ptr<XDriverList> &drivers)
	: XSecondaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
	  m_pulse(create<XItemNode<XDriverList, XNMRPulseAnalyzer> >("PulseAnalyzer", false, drivers, true)),
	  m_bandWidth(create<XDoubleNode>("BandWidth", false)),
	  m_autoPhase(create<XBoolNode>("AutoPhase", false)),
	  m_phase(create<XDoubleNode>("Phase", false, "%.2f")),
	  m_clear(create<XNode>("Clear", true)),
	  m_form(new FRM(g_pFrmMain)),
	  m_spectrum(create<XWaveNGraph>("Spectrum", true,
									 m_form->m_graph, m_form->m_urlDump, m_form->m_btnDump))
{
    m_form->m_btnClear->setIconSet(
		KApplication::kApplication()->iconLoader()->loadIconSet("editdelete", 
																KIcon::Toolbar, KIcon::SizeSmall, true ) );      
    
	connect(pulse());

	{
		const char *labels[] = {"X", "Re [V]", "Im [V]", "Counts"};
		m_spectrum->setColCount(4, labels);
		m_spectrum->insertPlot(labels[1], 0, 1, -1, 3);
		m_spectrum->insertPlot(labels[2], 0, 2, -1, 3);
		m_spectrum->axisy()->label()->value(KAME::i18n("Intens. [V]"));
		m_spectrum->plot(0)->label()->value(KAME::i18n("real part"));
		m_spectrum->plot(0)->drawPoints()->value(false);
		m_spectrum->plot(1)->label()->value(KAME::i18n("imag. part"));
		m_spectrum->plot(1)->drawPoints()->value(false);
		m_spectrum->clear();
	}
  
	bandWidth()->value(50);

	m_lsnOnClear = m_clear->onTouch().connectWeak(
		shared_from_this(), &XNMRSpectrumBase<FRM>::onClear);
	m_lsnOnCondChanged = bandWidth()->onValueChanged().connectWeak(
		shared_from_this(), &XNMRSpectrumBase<FRM>::onCondChanged);
	autoPhase()->onValueChanged().connect(m_lsnOnCondChanged);
	phase()->onValueChanged().connect(m_lsnOnCondChanged);

	m_conBandWidth = xqcon_create<XQLineEditConnector>(m_bandWidth, m_form->m_edBW);
	m_conPhase = xqcon_create<XKDoubleNumInputConnector>(m_phase, m_form->m_numPhase);
	m_conAutoPhase = xqcon_create<XQToggleButtonConnector>(m_autoPhase, m_form->m_ckbAutoPhase);
	m_conPulse = xqcon_create<XQComboBoxConnector>(m_pulse, m_form->m_cmbPulse);
	m_conClear = xqcon_create<XQButtonConnector>(m_clear, m_form->m_btnClear);
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::showForms()
{
	m_form->show();
	m_form->raise();
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::onCondChanged(const shared_ptr<XValueNodeBase> &node)
{
    if((node == phase()) && *autoPhase()) return;
	if(onCondChangedImpl(node))
        m_timeClearRequested = XTime::now();
    requestAnalysis();
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::onClear(const shared_ptr<XNode> &)
{
    m_timeClearRequested = XTime::now();
    requestAnalysis();
}
template <class FRM>
bool
XNMRSpectrumBase<FRM>::checkDependency(const shared_ptr<XDriver> &emitter) const {
    shared_ptr<XNMRPulseAnalyzer> _pulse = *pulse();
    if(!_pulse) return false;
    if(emitter == shared_from_this()) return true;
    return (emitter == _pulse) && checkDependencyImpl(emitter);
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::analyze(const shared_ptr<XDriver> &emitter) throw (XRecordError&)
{
	shared_ptr<XNMRPulseAnalyzer> _pulse = *pulse();
	ASSERT( _pulse );
  
	bool clear = (m_timeClearRequested > _pulse->timeAwared());
  
	double res = getResolution();
	double _max = getMaxValue();
	double _min = getMinValue();

	if(_max <= _min) {
		throw XRecordError(KAME::i18n("Invalid min. and max."), __FILE__, __LINE__);
	}
	if(res < 2e-6 * (_max - _min)) {
		throw XRecordError(KAME::i18n("Too small resolution."), __FILE__, __LINE__);
	}

	if((resRecorded() != res) || clear) {
		m_resRecorded = res;
		m_wave.clear();
		m_counts.clear();
	}
	else {
		int diff = lrint(minRecorded() / res) - lrint(_min / res);
		for(int i = 0; i < diff; i++) {
			m_wave.push_front(0.0);
			m_counts.push_front(0);
		}
		for(int i = 0; i < -diff; i++) {
			if(!m_wave.empty()) {
				m_wave.pop_front();
				m_counts.pop_front();
			}
		}
	}
	m_minRecorded = _min;
	int length = lrint((_max - _min) / res);
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

	fssum();
	
	if(*autoPhase()) {
		std::complex<double> csum(0.0, 0.0);
		for(unsigned int i = 0; i < wave().size(); i++) {
			if(counts()[i] > 0)
				csum += wave()[i] / (double)counts()[i];
		}
		double ph = 180.0 / PI * atan2(std::imag(csum), std::real(csum));
		if(fabs(ph) < 180.0)
			phase()->value(ph);
	}
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::visualize()
{
	if(!time()) {
		m_spectrum->clear();
		return;
	}

	double ph = *phase() / 180.0 * PI;
	std::complex<double> cph(cos(ph), -sin(ph));
	int length = wave().size();
	{   XScopedWriteLock<XWaveNGraph> lock(*m_spectrum);
	m_spectrum->setRowCount(length);
	for(int i = 0; i < length; i++) {
		m_spectrum->cols(0)[i] = minRecorded() + i * resRecorded();
		std::complex<double> c = wave()[i] * cph;
		m_spectrum->cols(1)[i] = (counts()[i] > 0) ? std::real(c) / counts()[i] : 0;
		m_spectrum->cols(2)[i] = (counts()[i] > 0) ? std::imag(c) / counts()[i] : 0;
		m_spectrum->cols(3)[i] = counts()[i];
	}
	}
}
