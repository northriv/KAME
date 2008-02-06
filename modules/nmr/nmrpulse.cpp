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
#include "nmrpulse.h"
#include "nmrpulseform.h"
#include "ar.h"
#include "freqest.h"

#include <graph.h>
#include <graphwidget.h>
#include <graphnurlform.h>
#include <xwavengraph.h>
#include <analyzer.h>
#include <xnodeconnector.h>

#include <knuminput.h>
#include <qpushbutton.h>
#include <qcheckbox.h>
#include <kapplication.h>
#include <kiconloader.h>

REGISTER_TYPE(XDriverList, NMRPulseAnalyzer, "NMR FID/echo analyzer");

//---------------------------------------------------------------------------
XNMRPulseAnalyzer::XNMRPulseAnalyzer(const char *name, bool runtime,
	const shared_ptr<XScalarEntryList> &scalarentries,
	const shared_ptr<XInterfaceList> &interfaces,
	const shared_ptr<XThermometerList> &thermometers,
	const shared_ptr<XDriverList> &drivers) :
	XSecondaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
		m_entryPeakAbs(create<XScalarEntry>("PeakAbs", false,
			dynamic_pointer_cast<XDriver>(shared_from_this()))),
		m_entryPeakFreq(create<XScalarEntry>("PeakFreq", false,
			dynamic_pointer_cast<XDriver>(shared_from_this()))), 
		m_dso(create<XItemNode<XDriverList, XDSO> >("DSO", false, drivers, true)),
		m_fromTrig(create<XDoubleNode>("FromTrig", false)),
		m_width(create<XDoubleNode>("Width", false)),
		m_phaseAdv(create<XDoubleNode>("PhaseAdv", false)),
		m_useDNR(create<XBoolNode>("UseDNR", false)),
		m_solverList(create<XComboNode>("SpectrumSolver", false, true)),
		m_bgPos(create<XDoubleNode>("BGPos", false)),
		m_bgWidth(create<XDoubleNode>("BGWidth", false)),
		m_fftPos(create<XDoubleNode>("FFTPos", false)),
		m_fftLen(create<XUIntNode>("FFTLen", false)),
		m_windowFunc(create<XComboNode>("WindowFunc", false, true)),
		m_windowWidth(create<XDoubleNode>("WindowLength", false)),
		m_difFreq(create<XDoubleNode>("DIFFreq", false)),
		m_exAvgIncr(create<XBoolNode>("ExAvgIncr", false)),
		m_extraAvg(create<XUIntNode>("ExtraAvg", false)),
		m_numEcho(create<XUIntNode>("NumEcho", false)),
		m_echoPeriod(create<XDoubleNode>("EchoPeriod", false)),
		m_spectrumShow(create<XNode>("SpectrumShow", true)),
		m_avgClear(create<XNode>("AvgClear", true)),
		m_picEnabled(create<XBoolNode>("PICEnabled", false)),
		m_pulser(create<XItemNode<XDriverList, XPulser> >("Pulser", false, drivers, true)),
		m_form(new FrmNMRPulse(g_pFrmMain)),
		m_statusPrinter(XStatusPrinter::create(m_form.get())),
		m_spectrumForm(new FrmGraphNURL(g_pFrmMain)), m_waveGraph(create<XWaveNGraph>("Wave", true,
			m_form->m_graph, m_form->m_urlDump, m_form->m_btnDump)),
		m_ftWaveGraph(create<XWaveNGraph>("Spectrum", true, m_spectrumForm.get())),
		m_solver(create<SpectrumSolverWrapper>("SpectrumSolver", true, m_solverList, m_windowFunc, m_windowWidth)),
//		m_solverDNR(new CompositeSpectrumSolver<MEMStrict, EigenVectorMethod, true>) {
		m_solverDNR(new MEMStrict) {
//		m_solverDNR(new MEMBurg(&SpectrumSolver::icAICc)) {
	m_form->m_btnAvgClear->setIconSet(KApplication::kApplication()->iconLoader()->loadIconSet("editdelete", KIcon::Toolbar, KIcon::SizeSmall, true) );
	m_form->m_btnSpectrum->setIconSet(KApplication::kApplication()->iconLoader()->loadIconSet("graph", KIcon::Toolbar, KIcon::SizeSmall, true) );

	connect(dso());
	connect(pulser(), false);

	scalarentries->insert(entryPeakAbs());
	scalarentries->insert(entryPeakFreq());

	fromTrig()->value(-0.005);
	width()->value(0.02);
	bgPos()->value(0.03);
	bgWidth()->value(0.03);
	fftPos()->value(0.004);
	fftLen()->value(16384);
	numEcho()->value(1);
	windowFunc()->str(std::string(SpectrumSolverWrapper::WINDOW_FUNC_DEFAULT));
	windowWidth()->value(100.0);

	m_form->setCaption(KAME::i18n("NMR Pulse - ") + getLabel() );

	m_spectrumForm->setCaption(KAME::i18n("NMR-Spectrum - ") + getLabel() );

	m_conAvgClear = xqcon_create<XQButtonConnector>(m_avgClear,
		m_form->m_btnAvgClear);
	m_conSpectrumShow = xqcon_create<XQButtonConnector>(m_spectrumShow, m_form->m_btnSpectrum);

	m_conFromTrig = xqcon_create<XQLineEditConnector>(fromTrig(),
		m_form->m_edPos);
	m_conWidth = xqcon_create<XQLineEditConnector>(width(), m_form->m_edWidth);
	m_form->m_numPhaseAdv->setRange(-180.0, 180.0, 1.0, true);
	m_conPhaseAdv = xqcon_create<XKDoubleNumInputConnector>(phaseAdv(),
		m_form->m_numPhaseAdv);
	m_conUseDNR = xqcon_create<XQToggleButtonConnector>(useDNR(),
		m_form->m_ckbDNR);
	m_conSolverList = xqcon_create<XQComboBoxConnector>(solverList(),
		m_form->m_cmbSolver);
	m_conBGPos = xqcon_create<XQLineEditConnector>(bgPos(), m_form->m_edBGPos);
	m_conBGWidth = xqcon_create<XQLineEditConnector>(bgWidth(),
		m_form->m_edBGWidth);
	m_conFFTPos = xqcon_create<XQLineEditConnector>(fftPos(),
		m_form->m_edFFTPos);
	m_conFFTLen = xqcon_create<XQLineEditConnector>(fftLen(),
		m_form->m_edFFTLen);
	m_form->m_numExtraAvg->setRange(0, 100000);
	m_conExtraAv = xqcon_create<XQSpinBoxConnector>(extraAvg(),
		m_form->m_numExtraAvg);
	m_conExAvgIncr = xqcon_create<XQToggleButtonConnector>(exAvgIncr(),
		m_form->m_ckbIncrAvg);
	m_conNumEcho = xqcon_create<XQSpinBoxConnector>(numEcho(),
		m_form->m_numEcho);
	m_conEchoPeriod = xqcon_create<XQLineEditConnector>(echoPeriod(),
		m_form->m_edEchoPeriod);
	m_conWindowFunc = xqcon_create<XQComboBoxConnector>(windowFunc(),
		m_form->m_cmbWindowFunc);
	m_form->m_numWindowWidth->setRange(3.0, 200.0, 1.0, true);
	m_conWindowWidth = xqcon_create<XKDoubleNumInputConnector>(windowWidth(),
		m_form->m_numWindowWidth);
	m_conDIFFreq = xqcon_create<XQLineEditConnector>(difFreq(),
		m_form->m_edDIFFreq);

	m_conPICEnabled = xqcon_create<XQToggleButtonConnector>(m_picEnabled,
		m_form->m_ckbPICEnabled);

	m_conPulser = xqcon_create<XQComboBoxConnector>(m_pulser,
		m_form->m_cmbPulser);
	m_conDSO = xqcon_create<XQComboBoxConnector>(dso(), m_form->m_cmbDSO);

	{
		const char *labels[] = { "Time [ms]", "IFFT Re [V]", "IFFT Im [V]", "DSO CH1[V]", "DSO CH2[V]"};
		waveGraph()->setColCount(5, labels);
		waveGraph()->insertPlot(labels[1], 0, 1);
		waveGraph()->insertPlot(labels[2], 0, 2);
		waveGraph()->insertPlot(labels[3], 0, 3);
		waveGraph()->insertPlot(labels[4], 0, 4);
		waveGraph()->axisy()->label()->value(KAME::i18n("Intens. [V]"));
		waveGraph()->plot(0)->label()->value(KAME::i18n("IFFT Re."));
		waveGraph()->plot(0)->drawPoints()->value(false);
		waveGraph()->plot(0)->intensity()->value(1.5);
		waveGraph()->plot(1)->label()->value(KAME::i18n("IFFT Im."));
		waveGraph()->plot(1)->drawPoints()->value(false);
		waveGraph()->plot(1)->intensity()->value(1.5);
		waveGraph()->plot(2)->label()->value(KAME::i18n("DSO CH1"));
		waveGraph()->plot(2)->drawPoints()->value(false);
		waveGraph()->plot(2)->lineColor()->value(QColor(0xff, 0xa0, 0x00).rgb());
		waveGraph()->plot(2)->intensity()->value(0.6);
		waveGraph()->plot(3)->label()->value(KAME::i18n("DSO CH2"));
		waveGraph()->plot(3)->drawPoints()->value(false);
		waveGraph()->plot(3)->lineColor()->value(QColor(0x00, 0xa0, 0xff).rgb());
		waveGraph()->plot(3)->intensity()->value(0.6);
		waveGraph()->clear();
	}
	{
		const char *labels[] = { "Freq. [kHz]", "Re. [V]", "Im. [V]",
			"Abs. [V]", "Phase [deg]" };
		ftWaveGraph()->setColCount(5, labels);
		ftWaveGraph()->insertPlot(labels[3], 0, 3);
		ftWaveGraph()->insertPlot(labels[4], 0, -1, 4);
		ftWaveGraph()->plot(0)->label()->value(KAME::i18n("abs."));
		ftWaveGraph()->plot(0)->drawBars()->value(true);
		ftWaveGraph()->plot(0)->drawLines()->value(true);
		ftWaveGraph()->plot(0)->drawPoints()->value(false);
		ftWaveGraph()->plot(0)->intensity()->value(0.5);
		ftWaveGraph()->plot(1)->label()->value(KAME::i18n("phase"));
		ftWaveGraph()->plot(1)->drawPoints()->value(false);
		ftWaveGraph()->clear();
		{
			shared_ptr<XXYPlot> plot = ftWaveGraph()->graph()->plots()->create<XXYPlot>(
				"Peaks", true, ftWaveGraph()->graph());
			m_peakPlot = plot;
			plot->label()->value(KAME::i18n("Peaks"));
			plot->axisX()->value(ftWaveGraph()->axisx());
			plot->axisY()->value(ftWaveGraph()->axisy());
			plot->drawPoints()->value(false);
			plot->drawLines()->value(false);
			plot->drawBars()->value(true);
			plot->intensity()->value(2.0);
			plot->displayMajorGrid()->value(false);
			plot->pointColor()->value(QColor(0x40, 0x40, 0xa0).rgb());
			plot->barColor()->value(QColor(0x40, 0x40, 0xa0).rgb());
			plot->clearPoints()->setUIEnabled(false);
			plot->maxCount()->setUIEnabled(false);
		}
	}

	m_lsnOnAvgClear = m_avgClear->onTouch().connectWeak(shared_from_this(), &XNMRPulseAnalyzer::onAvgClear);
	m_lsnOnSpectrumShow = m_spectrumShow->onTouch().connectWeak(shared_from_this(), &XNMRPulseAnalyzer::onSpectrumShow,
		XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP);

	m_lsnOnCondChanged = fromTrig()->onValueChanged().connectWeak(shared_from_this(), &XNMRPulseAnalyzer::onCondChanged);
	width()->onValueChanged().connect(m_lsnOnCondChanged);
	phaseAdv()->onValueChanged().connect(m_lsnOnCondChanged);
	useDNR()->onValueChanged().connect(m_lsnOnCondChanged);
	solverList()->onValueChanged().connect(m_lsnOnCondChanged);
	bgPos()->onValueChanged().connect(m_lsnOnCondChanged);
	bgWidth()->onValueChanged().connect(m_lsnOnCondChanged);
	fftPos()->onValueChanged().connect(m_lsnOnCondChanged);
	fftLen()->onValueChanged().connect(m_lsnOnCondChanged);
	//	extraAvg()->onValueChanged().connect(m_lsnOnCondChanged);
	exAvgIncr()->onValueChanged().connect(m_lsnOnCondChanged);
	numEcho()->onValueChanged().connect(m_lsnOnCondChanged);
	echoPeriod()->onValueChanged().connect(m_lsnOnCondChanged);
	windowFunc()->onValueChanged().connect(m_lsnOnCondChanged);
	windowWidth()->onValueChanged().connect(m_lsnOnCondChanged);
	difFreq()->onValueChanged().connect(m_lsnOnCondChanged);
}
XNMRPulseAnalyzer::~XNMRPulseAnalyzer() {
}
void XNMRPulseAnalyzer::onSpectrumShow(const shared_ptr<XNode> &) {
	m_spectrumForm->show();
	m_spectrumForm->raise();
}
void XNMRPulseAnalyzer::showForms() {
	m_form->show();
	m_form->raise();
}

void XNMRPulseAnalyzer::backgroundSub(std::vector<std::complex<double> > &wave, 
	int pos, int length, int bgpos, int bglength) {

	if(bglength < length*2)
		m_statusPrinter->printWarning(KAME::i18n("Maybe, length for BG. sub. is too short."));
	
	std::complex<double> bg = 0;
	if (bglength) {
		double normalize = 0.0;
		for (int i = 0; i < bglength; i++) {
			double z = 1.0;
			if(!*useDNR())
				z = FFT::windowFuncHamming( (double)i / bglength - 0.5);
			bg += z * wave[pos + i + bgpos];
			normalize += z;
		}
		bg /= normalize;
	}

	for (int i = 0; i < wave.size(); i++) {
		wave[i] -= bg;
	}

	if (bglength && *useDNR()) {
		int dnrlength = FFT::fitLength((bglength + bgpos) * 4);
		std::vector<std::complex<double> > memin(bglength), memout(dnrlength);
		for(unsigned int i = 0; i < bglength; i++) {
			memin[i] = wave[pos + i + bgpos];
		}
		m_solverDNR->exec(memin, memout, bgpos, 2e-2, &FFT::windowFuncRect, 2.0);
		for(unsigned int i = 0; i < std::min((int)wave.size() - pos, (int)memout.size()); i++) {
			wave[i + pos] -= m_solverDNR->ifft()[i];
		}
	}
}
void XNMRPulseAnalyzer::rotNFFT(int ftpos, double ph,
	std::vector<std::complex<double> > &wave,
	std::vector<std::complex<double> > &ftwave,
	int diffreq) {
	int length = wave.size();
	//phase advance
	std::complex<double> cph(cos(ph), sin(ph));
	for (int i = 0; i < length; i++) {
		wave[i] *= cph;
	}

	int fftlen = ftwave.size();
	//fft
	std::vector<std::complex<double> > fftout(fftlen);
	m_solverRecorded = m_solver->solver();
	m_solverRecorded->exec(wave, fftout, -ftpos, 0.5e-2, m_solver->windowFunc(), *windowWidth() / 100.0);

	std::copy(fftout.begin(), fftout.end(), ftwave.begin());
}
void XNMRPulseAnalyzer::onAvgClear(const shared_ptr<XNode> &) {
	m_timeClearRequested = XTime::now();
	requestAnalysis();
}
void XNMRPulseAnalyzer::onCondChanged(const shared_ptr<XValueNodeBase> &node) {
	if ((node == numEcho()) || (node == difFreq()))
		m_timeClearRequested = XTime::now();
	if (node == exAvgIncr())
		m_timeClearRequested = XTime::now();
	if (node == exAvgIncr())
		extraAvg()->value(0);
	requestAnalysis();
}
bool XNMRPulseAnalyzer::checkDependency(const shared_ptr<XDriver> &emitter) const {
	const shared_ptr<XPulser> _pulser = *pulser();
	if (emitter == _pulser)
		return false;
	const shared_ptr<XDSO> _dso = *dso();
	if (!_dso)
		return false;
	//    //Request for clear.
	//    if(m_timeClearRequested > _dso->timeAwared()) return true;
	//    if(_pulser && (_dso->timeAwared() < _pulser->time())) return false;
	return true;
}
void XNMRPulseAnalyzer::analyze(const shared_ptr<XDriver> &emitter)
	throw (XRecordError&) {
	const shared_ptr<XDSO> _dso = *dso();
	ASSERT(_dso);
	ASSERT(_dso->time() );

	if (_dso->numChannelsRecorded() < 1) {
		throw XSkippedRecordError(KAME::i18n("No record in DSO"), __FILE__, __LINE__);
	}
	if (_dso->numChannelsRecorded() < 2) {
		throw XSkippedRecordError(KAME::i18n("Two channels needed in DSO"), __FILE__, __LINE__);
	}

	double interval = _dso->timeIntervalRecorded();
	if (interval <= 0) {
		throw XSkippedRecordError(KAME::i18n("Invalid time interval in waveforms."), __FILE__, __LINE__);
	}
	int pos = lrint(*fromTrig() *1e-3 / interval + _dso->trigPosRecorded());
	double starttime = (pos - _dso->trigPosRecorded()) * interval;
	if (pos >= (int)_dso->lengthRecorded()) {
		throw XSkippedRecordError(KAME::i18n("Position beyond waveforms."), __FILE__, __LINE__);
	}
	if (pos < 0) {
		throw XSkippedRecordError(KAME::i18n("Position beyond waveforms."), __FILE__, __LINE__);
	}
	int length = lrint(*width() / 1000 / interval);
	if (pos + length >= (int)_dso->lengthRecorded()) {
		throw XSkippedRecordError(KAME::i18n("Invalid length."), __FILE__, __LINE__);
	}
	if (length <= 0) {
		throw XSkippedRecordError(KAME::i18n("Invalid length."), __FILE__, __LINE__);
	}
	
	int bgpos = lrint((*bgPos() - *fromTrig()) / 1000 / interval);
	if(pos + bgpos >= (int)_dso->lengthRecorded()) {
		throw XSkippedRecordError(KAME::i18n("Position for BG. sub. beyond waveforms."), __FILE__, __LINE__);
	}
	if(bgpos < 0) {
		throw XSkippedRecordError(KAME::i18n("Position for BG. sub. beyond waveforms."), __FILE__, __LINE__);
	}
	int bglength = lrint(*bgWidth() / 1000 / interval);
	if(pos + bgpos + bglength >= (int)_dso->lengthRecorded()) {
		throw XSkippedRecordError(KAME::i18n("Invalid Length for BG. sub."), __FILE__, __LINE__);
	}
	if(bglength < 0) {
		throw XSkippedRecordError(KAME::i18n("Invalid Length for BG. sub."), __FILE__, __LINE__);
	}
	if((bgpos < length) && (bgpos + bglength > 0))
		m_statusPrinter->printWarning(KAME::i18n("Maybe, position for BG. sub. is overrapped against echoes"), true);

	int echoperiod = lrint(*echoPeriod() / 1000 /interval);
	int numechoes = *numEcho();
	bool bg_after_last_echo = (echoperiod < bgpos + bglength);
	if(numechoes > 1) {
		if(pos + echoperiod * (numechoes - 1) + length >= (int)_dso->lengthRecorded()) {
			throw XSkippedRecordError(KAME::i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
		}
		if(echoperiod < length) {
			throw XSkippedRecordError(KAME::i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
		}
		if(!bg_after_last_echo) {
			if(bgpos + bglength > echoperiod) {
				throw XSkippedRecordError(KAME::i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
			}
			if(pos + echoperiod * (numechoes - 1) + bgpos + bglength >= (int)_dso->lengthRecorded()) {
				throw XSkippedRecordError(KAME::i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
			}
		}
	}

	if((m_startTime != starttime) || (length != m_waveWidth)) {
		double t = length * interval * 1e3;
		m_waveGraph->axisx()->autoScale()->value(false);
		m_waveGraph->axisx()->minValue()->value(starttime * 1e3 - t * 0.3);
		m_waveGraph->axisx()->maxValue()->value(starttime * 1e3 + t * 1.3);
	}
	m_waveWidth = length;
	bool skip = (m_timeClearRequested > _dso->timeAwared());
	bool avgclear = skip;

	if(interval != m_interval) {
		//[sec]
		m_interval = interval;
		avgclear = true;
	}
	if(m_startTime != starttime) {
		//[sec]
		m_startTime = starttime;
		avgclear = true;
	}
	
	if (length > (int)m_waveSum.size()) {
		avgclear = true;
	}
	m_wave.resize(length);
	m_waveSum.resize(length);
	std::fill(m_wave.begin(), m_wave.end(), 0.0);

	// Phase Inversion Cycling
	bool picenabled = *m_picEnabled;
	shared_ptr<XPulser> _pulser(*pulser());
	bool inverted = false;
	if (picenabled && (!_pulser || !_pulser->time())) {
		picenabled = false;
		gErrPrint(getLabel() + ": " + KAME::i18n("No active pulser!"));
	}
	if (picenabled) {
		inverted = _pulser->invertPhaseRecorded();
	}

	int avgnum = std::max((unsigned int)*extraAvg(), 1u) * (picenabled ? 2 : 1);
	
	if (!*exAvgIncr() && (avgnum <= m_avcount)) {
		avgclear = true;
	}
	if (avgclear) {
		std::fill(m_waveSum.begin(), m_waveSum.end(), 0.0);
		m_avcount = 0;
		if(*exAvgIncr()) {
			extraAvg()->value(0);
		}
	}

	if(skip) {
		throw XSkippedRecordError(__FILE__, __LINE__);
	}

	m_dsoWave.resize(_dso->lengthRecorded());
	{
		const double *rawwavecos, *rawwavesin = NULL;
		ASSERT( _dso->numChannelsRecorded() );
		rawwavecos = _dso->waveRecorded(0);
		rawwavesin = _dso->waveRecorded(1);
		for(unsigned int i = 0; i < _dso->lengthRecorded(); i++) {
			m_dsoWave[i] = std::complex<double>(rawwavecos[i], rawwavesin[i]) * (inverted ? -1.0 : 1.0);
		}
	}
	m_dsoWaveStartPos = pos;

	for(int i = 1; i < numechoes; i++) {
		int rpos = pos + i * echoperiod;
		for(int j = 0;
		j < (!bg_after_last_echo ? std::max(bgpos + bglength, length) : length); j++) {
			int k = rpos + j;
			ASSERT(k < (int)_dso->lengthRecorded());
			if(i == 1)
				m_dsoWave[pos + j] /= (double)numechoes;
			m_dsoWave[pos + j] += m_dsoWave[k] / (double)numechoes;
		}
	}
	//background subtraction or dynamic noise reduction
	backgroundSub(m_dsoWave, pos, length, bgpos, bglength);
	if((emitter == _dso) || (!m_avcount)) {	
		for(int i = 0; i < length; i++) {
			m_waveSum[i] += m_dsoWave[pos + i];
		}
		m_avcount++;
		if(*exAvgIncr()) {
			extraAvg()->value(m_avcount);
		}
	}
	double normalize = 1.0 / m_avcount;
	for(int i = 0; i < length; i++) {
		m_wave[i] = m_waveSum[i] * normalize;
	}

	int ftpos = lrint(*fftPos() * 1e-3 / interval + _dso->trigPosRecorded() - pos);
	//	if((windowfunc != &windowFuncRect) && (abs(ftpos - length/2) > length*0.1))
	//		m_statusPrinter->printWarning(KAME::i18n("FFTPos is off-centered for window func."));  
	double ph = *phaseAdv() * M_PI / 180;
	m_waveFTPos = ftpos;
	int fftlen = FFT::fitLength(*fftLen());
	//[Hz]
	m_dFreq = 1.0 / fftlen / interval;
	m_ftWave.resize(fftlen);
	rotNFFT(ftpos, ph, m_wave, m_ftWave, lrint(*difFreq() * 1000.0 / dFreq()) );
	
	if(m_solverRecorded->peaks().size()) {
		entryPeakAbs()->value(m_solverRecorded->peaks()[0].first / (double)m_wave.size());
		double x = m_solverRecorded->peaks()[0].second;
		x = (x > fftlen / 2) ? (x - fftlen) : x;
		entryPeakFreq()->value(0.001 * x * m_dFreq);
	}
	
	if(picenabled && (m_avcount % 2 == 1) && (emitter == _dso)) {
		ASSERT( _pulser->time() );
		unlockConnection(_pulser);
		_pulser->invertPhase()->value(!inverted);
	}
	if(!*exAvgIncr() && (avgnum != m_avcount))
		throw XSkippedRecordError(__FILE__, __LINE__);
}
void XNMRPulseAnalyzer::visualize() {
	if (!time() || !m_avcount) {
		ftWaveGraph()->clear();
		waveGraph()->clear();
		m_peakPlot->maxCount()->value(0);
		return;
	}

	int ftsize = m_ftWave.size();
	{
		XScopedWriteLock<XWaveNGraph> lock(*ftWaveGraph());
		ftWaveGraph()->setRowCount(ftsize);
		double normalize = 1.0 / m_wave.size();
		for (int i = 0; i < ftsize; i++) {
			int j = (i - ftsize/2 + ftsize) % ftsize;
			ftWaveGraph()->cols(0)[i] = 0.001 * (i - ftsize/2) * m_dFreq;
			std::complex<double> z = m_ftWave[j] * normalize;
			ftWaveGraph()->cols(1)[i] = std::real(z);
			ftWaveGraph()->cols(2)[i] = std::imag(z);
			ftWaveGraph()->cols(3)[i] = std::abs(z);
			ftWaveGraph()->cols(4)[i] = std::arg(z) / M_PI * 180;
		}
		m_peakPlot->maxCount()->value(m_solverRecorded->peaks().size());
		std::deque<XGraph::ValPoint> &points(m_peakPlot->points());
		points.resize(m_solverRecorded->peaks().size());
		for(int i = 0; i < m_solverRecorded->peaks().size(); i++) {
			double x = m_solverRecorded->peaks()[i].second;
			x = (x > ftsize / 2) ? (x - ftsize) : x;
			points[i] = XGraph::ValPoint(0.001 * x * m_dFreq,
				m_solverRecorded->peaks()[i].first * normalize);
		}
	}
	{
		int length = m_dsoWave.size();
		XScopedWriteLock<XWaveNGraph> lock(*waveGraph());
		waveGraph()->setRowCount(length);
		for (int i = 0; i < length; i++) {
			int j = i - m_dsoWaveStartPos;
			waveGraph()->cols(0)[i] = (startTime() + j * m_interval) * 1e3;
			if(abs(j) < ftsize / 2) {
				j = (j - m_waveFTPos + ftsize) % ftsize;
				waveGraph()->cols(1)[i] = std::real(m_solverRecorded->ifft()[j]);
				waveGraph()->cols(2)[i] = std::imag(m_solverRecorded->ifft()[j]);
			}
			else {
				waveGraph()->cols(1)[i] = 0.0;
				waveGraph()->cols(2)[i] = 0.0;				
			}
			waveGraph()->cols(3)[i] = m_dsoWave[i].real();
			waveGraph()->cols(4)[i] = m_dsoWave[i].imag();				
		}
	}
}

