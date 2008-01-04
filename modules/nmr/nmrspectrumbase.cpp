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
	  m_solverList(create<XComboNode>("SpectrumSolver", false, true)),
	  m_windowFunc(create<XComboNode>("WindowFunc", false, true)),
	  m_windowWidth(create<XDoubleNode>("WindowWidth", false)),
	  m_form(new FRM(g_pFrmMain)),
	  m_statusPrinter(XStatusPrinter::create(m_form.get())),
	  m_spectrum(create<XWaveNGraph>("Spectrum", true,
									 m_form->m_graph, m_form->m_urlDump, m_form->m_btnDump)),
	  m_ftLen(0),
	  m_solver(create<SpectrumSolverWrapper>("SpectrumSolver", true, m_solverList, m_windowFunc, m_windowWidth)) {
    m_form->m_btnClear->setIconSet(
		KApplication::kApplication()->iconLoader()->loadIconSet("editdelete", 
																KIcon::Toolbar, KIcon::SizeSmall, true ) );      
    
	connect(pulse());

	{
		const char *labels[] = {"X", "Re [V]", "Im [V]", "Weights"};
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
	autoPhase()->value(true);
	
	windowFunc()->str(std::string(SpectrumSolverWrapper::WINDOW_FUNC_DEFAULT));
	windowWidth()->value(100.0);

	m_lsnOnClear = m_clear->onTouch().connectWeak(
		shared_from_this(), &XNMRSpectrumBase<FRM>::onClear);
	m_lsnOnCondChanged = bandWidth()->onValueChanged().connectWeak(
		shared_from_this(), &XNMRSpectrumBase<FRM>::onCondChanged);
	autoPhase()->onValueChanged().connect(m_lsnOnCondChanged);
	phase()->onValueChanged().connect(m_lsnOnCondChanged);
	solverList()->onValueChanged().connect(m_lsnOnCondChanged);
	windowWidth()->onValueChanged().connect(m_lsnOnCondChanged);
	windowFunc()->onValueChanged().connect(m_lsnOnCondChanged);

	m_conBandWidth = xqcon_create<XQLineEditConnector>(m_bandWidth, m_form->m_edBW);
	m_conPhase = xqcon_create<XKDoubleNumInputConnector>(m_phase, m_form->m_numPhase);
	m_form->m_numPhase->setRange(-360.0, 360.0, 1.0, true);
	m_conAutoPhase = xqcon_create<XQToggleButtonConnector>(m_autoPhase, m_form->m_ckbAutoPhase);
	m_conPulse = xqcon_create<XQComboBoxConnector>(m_pulse, m_form->m_cmbPulse);
	m_conClear = xqcon_create<XQButtonConnector>(m_clear, m_form->m_btnClear);
	m_conSolverList = xqcon_create<XQComboBoxConnector>(m_solverList, m_form->m_cmbSolver);
	m_conWindowWidth = xqcon_create<XKDoubleNumInputConnector>(m_windowWidth,
		m_form->m_numWindowWidth);
	m_form->m_numWindowWidth->setRange(0.1, 200.0, 1.0, true);
	m_conWindowFunc = xqcon_create<XQComboBoxConnector>(m_windowFunc,
		m_form->m_cmbWindowFunc);
}
template <class FRM>
XNMRSpectrumBase<FRM>::~XNMRSpectrumBase() {
	if(m_ftLen) {
		fftw_destroy_plan(m_planIFT);
	}
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
  
	double interval = _pulse->interval();
	double df = _pulse->dFreq();
	
	double res = getFreqResHint();
	res = df * std::max(1L, lrint(res / df - 0.5));
	
	double _max = getMaxFreq();
	double _min = getMinFreq();
	
	if(_max <= _min) {
		throw XRecordError(KAME::i18n("Invalid min. and max."), __FILE__, __LINE__);
	}
	if(res * 65536 * 2 < _max - _min) {
		throw XRecordError(KAME::i18n("Too small resolution."), __FILE__, __LINE__);
	}

	if((resRecorded() != res) || clear) {
		m_resRecorded = res;
		m_accum.clear();
		m_weights.clear();
	}
	else {
		int diff = lrint(minRecorded() / res) - lrint(_min / res);
		for(int i = 0; i < diff; i++) {
			m_accum.push_front(0.0);
			m_weights.push_front(0);
		}
		for(int i = 0; i < -diff; i++) {
			if(!m_accum.empty()) {
				m_accum.pop_front();
				m_weights.pop_front();
			}
		}
	}
	m_minRecorded = _min;
	int length = lrint((_max - _min) / res);
	m_accum.resize(length, 0.0);
	m_weights.resize(length, 0);

	if(clear) {
		m_spectrum->clear();
		throw XSkippedRecordError(__FILE__, __LINE__);
	}

	if(emitter == _pulse) {
		fssum();
		afterFSSum();
	}
	
	analyzeIFT();
	if(*autoPhase()) {
		std::complex<double> csum(0.0, 0.0);
		for(unsigned int i = 0; i < wave().size(); i++) {
			if(weights()[i] > 0)
				csum += wave()[i] / weights()[i];
		}
		double ph = 180.0 / PI * atan2(std::imag(csum), std::real(csum));
		if(fabs(ph) < 180.0)
			phase()->value(ph);
	}
	double ph = *phase() / 180.0 * PI;
	std::complex<double> cph(cos(ph), -sin(ph));
	for(unsigned int i = 0; i < wave().size(); i++) {
		m_wave[i] *= cph;
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

	int length = wave().size();
	std::vector<double> values;
	getValues(values);
	ASSERT(values.size() == length);
	{   XScopedWriteLock<XWaveNGraph> lock(*m_spectrum);
	double th = SpectrumSolver::windowFuncHamming(0.2);
	m_spectrum->setRowCount(length);
	for(int i = 0; i < length; i++) {
		m_spectrum->cols(0)[i] = values[i];
		m_spectrum->cols(1)[i] = (weights()[i] > th) ? std::real(wave()[i]) / weights()[i] : 0;
		m_spectrum->cols(2)[i] = (weights()[i] > th) ? std::imag(wave()[i]) / weights()[i] : 0;
		m_spectrum->cols(3)[i] = weights()[i];
	}
	}
}

template <class FRM>
void
XNMRSpectrumBase<FRM>::fssum()
{
	shared_ptr<XNMRPulseAnalyzer> _pulse = *pulse();

	int len = _pulse->ftWave().size();
	double df = _pulse->dFreq();
	if((len == 0) || (df == 0)) {
		throw XRecordError(KAME::i18n("Invalid waveform."), __FILE__, __LINE__);  
	}
	int bw = abs(lrint(*bandWidth() * 1000.0 / df));
	bw *= 1.5; // for Hamming.
//	bw *= 3.6; // for FlatTop.
	double cfreq = getCurrentCenterFreq();
	if(cfreq == 0) {
		throw XRecordError(KAME::i18n("Invalid center freq."), __FILE__, __LINE__);  
	}
	if(_pulse->windowFunc()->to_str() != SpectrumSolverWrapper::WINDOW_FUNC_DEFAULT) {
		m_statusPrinter->printWarning(KAME::i18n("Do not use window function in the pulse analyzer. Skipping."), false);
		throw XSkippedRecordError(__FILE__, __LINE__);
	}
	if(_pulse->solverList()->to_str() != SpectrumSolverWrapper::SPECTRUM_SOLVER_ZF_FFT) {
		m_statusPrinter->printWarning(KAME::i18n("Use ZF-FFT in the pulse analyzer. Skipping."), false);
		throw XSkippedRecordError(__FILE__, __LINE__);
	}
	for(int i = std::max(0, (len - bw) / 2); i < std::min(len, (len + bw) / 2); i++) {
		double freq = (i - len/2) * df;
		int idx = lrint((cfreq + freq - minRecorded()) / resRecorded());
		if((idx >= (int)m_accum.size()) || (idx < 0))
			continue;
//		double w = SpectrumSolver::windowFuncFlatTop((double)(i - len/2) / bw);
		double w = SpectrumSolver::windowFuncHamming((double)(i - len/2) / bw);
//		double w = XNMRPulseAnalyzer::windowFuncRect((double)(i - len/2) / bw);
		m_accum[idx] += _pulse->ftWave()[i] * w;
		m_weights[idx] += w;
	}
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::analyzeIFT() {
	shared_ptr<XNMRPulseAnalyzer> _pulse = *pulse();

	double res = resRecorded();
	int iftlen = lrint(pow(2.0, (ceil(log(m_accum.size()) / log(2.0)))));
	int iftorigin = lrint(_pulse->waveFTPos() * _pulse->interval() * res * iftlen);
	int tdsize = lrint(_pulse->waveWidth() * _pulse->interval() * res * iftlen);
//	fprintf(stderr, "IFT: len=%d, org=%d, size=%d\n", iftlen, iftorigin, tdsize);
	
	if(m_ftLen != iftlen) {
		if(m_ftLen) {
			fftw_destroy_plan(m_planIFT);
		}
		m_ftLen = iftlen;
		m_planIFT = fftw_create_plan(m_ftLen, FFTW_BACKWARD, FFTW_ESTIMATE);
	}
	
	std::vector<fftw_complex> fftwave(iftlen), iftwave(iftlen);
	ASSERT(m_accum.size() <= fftwave.size());
	for(int i = 0; i < fftwave.size(); i++) {
		fftwave[i].re = 0.0;
		fftwave[i].im = 0.0;
	}
	for(int i = 0; i < m_accum.size(); i++) {
		int k = (i < m_accum.size()/2) ? i : (i - m_accum.size()); 
//		if(weights()[i] > 0.0) {
			k = (k + fftwave.size()) % fftwave.size();
			fftwave[k].re = m_accum[i].real();
			fftwave[k].im = m_accum[i].imag();
//		}
	}
	fftw_one(m_planIFT, &fftwave[0], &iftwave[0]);
	
	std::vector<fftw_complex> solverin(tdsize);
	for(unsigned int i = 0; i < solverin.size(); i++) {
		int k = (-iftorigin + i + iftlen) % iftlen;
		solverin[i].re = iftwave[k].re;
		solverin[i].im = iftwave[k].im;
	}
	shared_ptr<SpectrumSolver> solver = m_solver->solver();
	solver->exec(solverin, fftwave, -iftorigin, 0.1e-2, m_solver->windowFunc(), *windowWidth() / 100.0);

	m_wave.resize(m_accum.size());
	for(int i = 0; i < wave().size(); i++) {
		int k = (i < wave().size()/2) ? i : (i - wave().size()); 
		k = (k + fftwave.size()) % fftwave.size();
		m_wave[i] = std::complex<double>(fftwave[k].re, fftwave[k].im) / (double)iftlen;
	}
}
