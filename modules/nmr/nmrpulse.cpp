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
#include "nmrpulse.h"
#include "ui_nmrpulseform.h"
#include "freqestleastsquare.h"

#include <graph.h>
#include <graphwidget.h>
#include <ui_graphnurlform.h>
#include <analyzer.h>
#include <xnodeconnector.h>
#include <kiconloader.h>

REGISTER_TYPE(XDriverList, NMRPulseAnalyzer, "NMR FID/echo analyzer");

//---------------------------------------------------------------------------
XNMRPulseAnalyzer::XNMRPulseAnalyzer(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XSecondaryDriver(name, runtime, ref(tr_meas), meas),
		m_entryPeakAbs(create<XScalarEntry>("PeakAbs", false,
			dynamic_pointer_cast<XDriver>(shared_from_this()))),
		m_entryPeakFreq(create<XScalarEntry>("PeakFreq", false,
			dynamic_pointer_cast<XDriver>(shared_from_this()))), 
		m_dso(create<XItemNode<XDriverList, XDSO> >("DSO", false, ref(tr_meas), meas->drivers(), true)),
		m_fromTrig(create<XDoubleNode>("FromTrig", false)),
		m_width(create<XDoubleNode>("Width", false)),
		m_phaseAdv(create<XDoubleNode>("PhaseAdv", false)),
		m_usePNR(create<XBoolNode>("UsePNR", false)),
		m_pnrSolverList(create<XComboNode>("PNRSpectrumSolver", false, true)),
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
		m_pulser(create<XItemNode<XDriverList, XPulser> >("Pulser", false, ref(tr_meas), meas->drivers(), true)),
		m_form(new FrmNMRPulse(g_pFrmMain)),
		m_statusPrinter(XStatusPrinter::create(m_form.get())),
		m_spectrumForm(new FrmGraphNURL(g_pFrmMain)), m_waveGraph(create<XWaveNGraph>("Wave", true,
			m_form->m_graph, m_form->m_urlDump, m_form->m_btnDump)),
		m_ftWaveGraph(create<XWaveNGraph>("Spectrum", true, m_spectrumForm.get())),
		m_solver(create<SpectrumSolverWrapper>("SpectrumSolverWrapper", true, m_solverList, m_windowFunc, m_windowWidth)),
		m_solverPNR(create<SpectrumSolverWrapper>("PNRSpectrumSolverWrapper", true, m_pnrSolverList, shared_ptr<XComboNode>(), shared_ptr<XDoubleNode>(), true)) {
	m_form->m_btnAvgClear->setIcon(KIconLoader::global()->loadIcon("editdelete", KIconLoader::Toolbar, KIconLoader::SizeSmall, true) );
	m_form->m_btnSpectrum->setIcon(KIconLoader::global()->loadIcon("graph", KIconLoader::Toolbar, KIconLoader::SizeSmall, true) );
	
	m_pnrSolverList->str(XString(SpectrumSolverWrapper::SPECTRUM_SOLVER_LS_MDL));

	connect(dso());
	connect(pulser(), false);

	meas->scalarEntries()->insert(tr_meas, entryPeakAbs());
	meas->scalarEntries()->insert(tr_meas, entryPeakFreq());

	fromTrig()->value(-0.005);
	width()->value(0.02);
	bgPos()->value(0.03);
	bgWidth()->value(0.03);
	fftPos()->value(0.004);
	fftLen()->value(16384);
	numEcho()->value(1);
	windowFunc()->str(XString(SpectrumSolverWrapper::WINDOW_FUNC_DEFAULT));
	windowWidth()->value(100.0);

	m_form->setWindowTitle(i18n("NMR Pulse - ") + getLabel() );

	m_spectrumForm->setWindowTitle(i18n("NMR-Spectrum - ") + getLabel() );

	m_conAvgClear = xqcon_create<XQButtonConnector>(m_avgClear,
		m_form->m_btnAvgClear);
	m_conSpectrumShow = xqcon_create<XQButtonConnector>(m_spectrumShow, m_form->m_btnSpectrum);

	m_conFromTrig = xqcon_create<XQLineEditConnector>(fromTrig(),
		m_form->m_edPos);
	m_conWidth = xqcon_create<XQLineEditConnector>(width(), m_form->m_edWidth);
	m_form->m_numPhaseAdv->setRange(-180.0, 180.0, 10.0, true);
	m_conPhaseAdv = xqcon_create<XKDoubleNumInputConnector>(phaseAdv(),
		m_form->m_numPhaseAdv);
	m_conUsePNR = xqcon_create<XQToggleButtonConnector>(usePNR(),
		m_form->m_ckbPNR);
	m_conPNRSolverList = xqcon_create<XQComboBoxConnector>(pnrSolverList(),
		m_form->m_cmbPNRSolver, Snapshot( *pnrSolverList()));
	m_conSolverList = xqcon_create<XQComboBoxConnector>(solverList(),
		m_form->m_cmbSolver, Snapshot( *solverList()));
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
		m_form->m_cmbWindowFunc, Snapshot( *windowFunc()));
	m_form->m_numWindowWidth->setRange(3.0, 200.0, 1.0, true);
	m_conWindowWidth = xqcon_create<XKDoubleNumInputConnector>(windowWidth(),
		m_form->m_numWindowWidth);
	m_conDIFFreq = xqcon_create<XQLineEditConnector>(difFreq(),
		m_form->m_edDIFFreq);

	m_conPICEnabled = xqcon_create<XQToggleButtonConnector>(m_picEnabled,
		m_form->m_ckbPICEnabled);

	m_conPulser = xqcon_create<XQComboBoxConnector>(m_pulser,
		m_form->m_cmbPulser, ref(tr_meas));
	m_conDSO = xqcon_create<XQComboBoxConnector>(dso(), m_form->m_cmbDSO, ref(tr_meas));

	for(Transaction tr( *waveGraph());; ++tr) {
		const char *labels[] = { "Time [ms]", "IFFT Re [V]", "IFFT Im [V]", "DSO CH1[V]", "DSO CH2[V]"};
		tr[ *waveGraph()].setColCount(5, labels);
		tr[ *waveGraph()].insertPlot(labels[1], 0, 1);
		tr[ *waveGraph()].insertPlot(labels[2], 0, 2);
		tr[ *waveGraph()].insertPlot(labels[3], 0, 3);
		tr[ *waveGraph()].insertPlot(labels[4], 0, 4);
		tr[ *tr[ *waveGraph()].axisy()->label()] = i18n("Intens. [V]");
		tr[ *tr[ *waveGraph()].plot(0)->label()] = i18n("IFFT Re.");
		tr[ *tr[ *waveGraph()].plot(0)->drawPoints()] = false;
		tr[ *tr[ *waveGraph()].plot(0)->intensity()] = 2.0;
		tr[ *tr[ *waveGraph()].plot(1)->label()] = i18n("IFFT Im.");
		tr[ *tr[ *waveGraph()].plot(1)->drawPoints()] = false;
		tr[ *tr[ *waveGraph()].plot(1)->intensity()] = 2.0;
		tr[ *tr[ *waveGraph()].plot(2)->label()] = i18n("DSO CH1");
		tr[ *tr[ *waveGraph()].plot(2)->drawPoints()] = false;
		tr[ *tr[ *waveGraph()].plot(2)->lineColor()] = QColor(0xff, 0xa0, 0x00).rgb();
		tr[ *tr[ *waveGraph()].plot(2)->intensity()] = 0.3;
		tr[ *tr[ *waveGraph()].plot(3)->label()] = i18n("DSO CH2");
		tr[ *tr[ *waveGraph()].plot(3)->drawPoints()] = false;
		tr[ *tr[ *waveGraph()].plot(3)->lineColor()] = QColor(0x00, 0xa0, 0xff).rgb();
		tr[ *tr[ *waveGraph()].plot(3)->intensity()] = 0.3;
		if(tr.commit())
			break;
	}
	waveGraph()->clear();
	for(Transaction tr( *ftWaveGraph());; ++tr) {
		const char *labels[] = { "Freq. [kHz]", "Re. [V]", "Im. [V]",
			"Abs. [V]", "Phase [deg]", "Dark. [V]" };
		tr[ *ftWaveGraph()].setColCount(6, labels);
		tr[ *ftWaveGraph()].insertPlot(labels[3], 0, 3);
		tr[ *ftWaveGraph()].insertPlot(labels[4], 0, -1, 4);
		tr[ *ftWaveGraph()].insertPlot(labels[5], 0, 5);
		tr[ *tr[ *ftWaveGraph()].axisy()->label()] = i18n("Intens. [V]");
		tr[ *tr[ *ftWaveGraph()].plot(0)->label()] = i18n("abs.");
		tr[ *tr[ *ftWaveGraph()].plot(0)->drawBars()] = true;
		tr[ *tr[ *ftWaveGraph()].plot(0)->drawLines()] = true;
		tr[ *tr[ *ftWaveGraph()].plot(0)->drawPoints()] = false;
		tr[ *tr[ *ftWaveGraph()].plot(0)->intensity()] = 0.5;
		tr[ *tr[ *ftWaveGraph()].plot(1)->label()] = i18n("phase");
		tr[ *tr[ *ftWaveGraph()].plot(1)->drawPoints()] = false;
		tr[ *tr[ *ftWaveGraph()].plot(1)->intensity()] = 0.3;
		tr[ *tr[ *ftWaveGraph()].plot(2)->label()] = i18n("dark");
		tr[ *tr[ *ftWaveGraph()].plot(2)->drawBars()] = false;
		tr[ *tr[ *ftWaveGraph()].plot(2)->drawLines()] = true;
		tr[ *tr[ *ftWaveGraph()].plot(2)->lineColor()] = QColor(0xa0, 0xa0, 0x00).rgb();
		tr[ *tr[ *ftWaveGraph()].plot(2)->drawPoints()] = false;
		tr[ *tr[ *ftWaveGraph()].plot(2)->intensity()] = 0.5;
		{
			shared_ptr<XXYPlot> plot = ftWaveGraph()->graph()->plots()->create<XXYPlot>(
				tr, "Peaks", true, ref(tr), ftWaveGraph()->graph());
			m_peakPlot = plot;
			tr[ *plot->label()] = i18n("Peaks");
			tr[ *plot->axisX()] = tr[ *ftWaveGraph()].axisx();
			tr[ *plot->axisY()] = tr[ *ftWaveGraph()].axisy();
			tr[ *plot->drawPoints()] = false;
			tr[ *plot->drawLines()] = false;
			tr[ *plot->drawBars()] = true;
			tr[ *plot->intensity()] = 0.5;
			tr[ *plot->displayMajorGrid()] = false;
			tr[ *plot->pointColor()] = QColor(0x40, 0x40, 0xa0).rgb();
			tr[ *plot->barColor()] = QColor(0x40, 0x40, 0xa0).rgb();
			tr[ *plot->clearPoints()].setUIEnabled(false);
			tr[ *plot->maxCount()].setUIEnabled(false);
		}
		if(tr.commit())
			break;
	}
	ftWaveGraph()->clear();

	m_lsnOnAvgClear = m_avgClear->onTouch().connectWeak(shared_from_this(), &XNMRPulseAnalyzer::onAvgClear);
	m_lsnOnSpectrumShow = m_spectrumShow->onTouch().connectWeak(shared_from_this(), &XNMRPulseAnalyzer::onSpectrumShow,
		XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP);

	m_lsnOnCondChanged = fromTrig()->onValueChanged().connectWeak(shared_from_this(), &XNMRPulseAnalyzer::onCondChanged);
	width()->onValueChanged().connect(m_lsnOnCondChanged);
	phaseAdv()->onValueChanged().connect(m_lsnOnCondChanged);
	usePNR()->onValueChanged().connect(m_lsnOnCondChanged);
	pnrSolverList()->onValueChanged().connect(m_lsnOnCondChanged);
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
	std::complex<double> bg = 0;
	if (bglength) {
		double normalize = 0.0;
		for (int i = 0; i < bglength; i++) {
			double z = 1.0;
			if(!*usePNR())
				z = FFT::windowFuncHamming( (double)i / bglength - 0.5);
			bg += z * wave[pos + i + bgpos];
			normalize += z;
		}
		bg /= normalize;
	}

	for (int i = 0; i < wave.size(); i++) {
		wave[i] -= bg;
	}

	shared_ptr<SpectrumSolver> solverPNR = m_solverPNR->solver();
	if (bglength) {
		if(*usePNR() && solverPNR) {
			int dnrlength = FFT::fitLength((bglength + bgpos) * 4);
			std::vector<std::complex<double> > memin(bglength), memout(dnrlength);
			for(unsigned int i = 0; i < bglength; i++) {
				memin[i] = wave[pos + i + bgpos];
			}
			try {
				solverPNR->exec(memin, memout, bgpos, 0.5e-2, &FFT::windowFuncRect, 1.0);
			}
			catch (XKameError &e) {
				throw XSkippedRecordError(e.msg(), __FILE__, __LINE__);
			}
			int imax = std::min((int)wave.size() - pos, (int)memout.size());
			for(unsigned int i = 0; i < imax; i++) {
				wave[i + pos] -= solverPNR->ifft()[i];
			}
		}
	}
}
void XNMRPulseAnalyzer::rotNFFT(int ftpos, double ph,
	std::vector<std::complex<double> > &wave,
	std::vector<std::complex<double> > &ftwave) {
	int length = wave.size();
	//phase advance
	std::complex<double> cph(std::polar(1.0, ph));
	for (int i = 0; i < length; i++) {
		wave[i] *= cph;
	}

	int fftlen = ftwave.size();
	//fft
	std::vector<std::complex<double> > fftout(fftlen);
	FFT::twindowfunc wndfunc = m_solver->windowFunc();
	double wndwidth = *windowWidth() / 100.0;
	try {
		m_solverRecorded->exec(wave, fftout, -ftpos, 0.3e-2, wndfunc, wndwidth);
	}
	catch (XKameError &e) {
		throw XSkippedRecordError(e.msg(), __FILE__, __LINE__);
	}

	std::copy(fftout.begin(), fftout.end(), ftwave.begin());
	
	if(m_solverRecorded->isFT()) {
		std::vector<double> weight;
		SpectrumSolver::window(length, -ftpos, wndfunc, wndwidth, weight);
		double w = 0;
		for(int i = 0; i < length; i++)
			w += weight[i] * weight[i];
		m_ftWavePSDCoeff = w/(double)length;
	}
	else {
		m_ftWavePSDCoeff = 1.0;
	}
}
void XNMRPulseAnalyzer::onAvgClear(const shared_ptr<XNode> &) {
	m_timeClearRequested = XTime::now();
	requestAnalysis();

	const shared_ptr<XDSO> _dso = *dso();
	if(_dso)
		_dso->restart()->touch(); //Restart averaging in DSO.
}
void XNMRPulseAnalyzer::onCondChanged(const shared_ptr<XValueNodeBase> &node) {
	if (node == exAvgIncr())
		extraAvg()->value(0);
	if ((node == numEcho()) || (node == difFreq()) || (node == exAvgIncr()))
		onAvgClear(node);
	else
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
		throw XSkippedRecordError(i18n("No record in DSO"), __FILE__, __LINE__);
	}
	if (_dso->numChannelsRecorded() < 2) {
		throw XSkippedRecordError(i18n("Two channels needed in DSO"), __FILE__, __LINE__);
	}
	if (!*_dso->singleSequence()) {
		m_statusPrinter->printWarning(i18n("Use sequential average in DSO."));
	}

	double interval = _dso->timeIntervalRecorded();
	if (interval <= 0) {
		throw XSkippedRecordError(i18n("Invalid time interval in waveforms."), __FILE__, __LINE__);
	}
	int pos = lrint(*fromTrig() *1e-3 / interval + _dso->trigPosRecorded());
	double starttime = (pos - _dso->trigPosRecorded()) * interval;
	if (pos >= (int)_dso->lengthRecorded()) {
		throw XSkippedRecordError(i18n("Position beyond waveforms."), __FILE__, __LINE__);
	}
	if (pos < 0) {
		throw XSkippedRecordError(i18n("Position beyond waveforms."), __FILE__, __LINE__);
	}
	int length = lrint(*width() / 1000 / interval);
	if (pos + length >= (int)_dso->lengthRecorded()) {
		throw XSkippedRecordError(i18n("Invalid length."), __FILE__, __LINE__);
	}
	if (length <= 0) {
		throw XSkippedRecordError(i18n("Invalid length."), __FILE__, __LINE__);
	}
	
	int bgpos = lrint((*bgPos() - *fromTrig()) / 1000 / interval);
	if(pos + bgpos >= (int)_dso->lengthRecorded()) {
		throw XSkippedRecordError(i18n("Position for BG. sub. beyond waveforms."), __FILE__, __LINE__);
	}
	if(bgpos < 0) {
		throw XSkippedRecordError(i18n("Position for BG. sub. beyond waveforms."), __FILE__, __LINE__);
	}
	int bglength = lrint(*bgWidth() / 1000 / interval);
	if(pos + bgpos + bglength >= (int)_dso->lengthRecorded()) {
		throw XSkippedRecordError(i18n("Invalid Length for BG. sub."), __FILE__, __LINE__);
	}
	if(bglength < 0) {
		throw XSkippedRecordError(i18n("Invalid Length for BG. sub."), __FILE__, __LINE__);
	}

	shared_ptr<XPulser> _pulser(*pulser());
	
	int echoperiod = lrint(*echoPeriod() / 1000 /interval);
	int numechoes = *numEcho();
	numechoes = std::max(1, numechoes);
	bool bg_after_last_echo = (echoperiod < bgpos + bglength);

	if(bglength && (bglength < length * (bg_after_last_echo ? numechoes : 1) * 3))
		m_statusPrinter->printWarning(i18n("Maybe, length for BG. sub. is too short."));
	
	if((bgpos < length + (bg_after_last_echo ? (echoperiod * (numechoes - 1)) : 0)) 
		&& (bgpos + bglength > 0))
		m_statusPrinter->printWarning(i18n("Maybe, position for BG. sub. is overrapped against echoes"), true);

	if(numechoes > 1) {
		if(pos + echoperiod * (numechoes - 1) + length >= (int)_dso->lengthRecorded()) {
			throw XSkippedRecordError(i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
		}
		if(echoperiod < length) {
			throw XSkippedRecordError(i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
		}
		if(!bg_after_last_echo) {
			if(bgpos + bglength > echoperiod) {
				throw XSkippedRecordError(i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
			}
			if(pos + echoperiod * (numechoes - 1) + bgpos + bglength >= (int)_dso->lengthRecorded()) {
				throw XSkippedRecordError(i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
			}
		}
		if(_pulser) {
			if((numechoes > _pulser->echoNumRecorded()) ||
				(fabs(*echoPeriod()*1e3 / (_pulser->tauRecorded()*2.0) - 1.0) > 1e-4)) {
				m_statusPrinter->printWarning(i18n("Invalid Multiecho settings."), true);
			}
		}
	}

	if((m_startTime != starttime) || (length != m_waveWidth)) {
		double t = length * interval * 1e3;
		for(Transaction tr( *waveGraph());; ++tr) {
			tr[ *tr[ *waveGraph()].axisx()->autoScale()] = false;
			tr[ *tr[ *waveGraph()].axisx()->minValue()] = starttime * 1e3 - t * 0.3;
			tr[ *tr[ *waveGraph()].axisx()->maxValue()] = starttime * 1e3 + t * 1.3;
			if(tr.commit())
				break;
		}
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
	int fftlen = FFT::fitLength(*fftLen());
	if(fftlen != m_darkPSD.size()) {
		avgclear = true;		
	}
	m_darkPSD.resize(fftlen);
	m_darkPSDSum.resize(fftlen);
	std::fill(m_wave.begin(), m_wave.end(), 0.0);

	// Phase Inversion Cycling
	bool picenabled = *m_picEnabled;
	bool inverted = false;
	if (picenabled && (!_pulser || !_pulser->time())) {
		picenabled = false;
		gErrPrint(getLabel() + ": " + i18n("No active pulser!"));
	}
	if (_pulser) {
		inverted = _pulser->invertPhaseRecorded();
	}

	int avgnum = std::max((unsigned int)*extraAvg(), 1u) * (picenabled ? 2 : 1);
	
	if (!*exAvgIncr() && (avgnum <= m_avcount)) {
		avgclear = true;
	}
	if (avgclear) {
		std::fill(m_waveSum.begin(), m_waveSum.end(), 0.0);
		std::fill(m_darkPSDSum.begin(), m_darkPSDSum.end(), 0.0);
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

	//background subtraction or dynamic noise reduction
	if(bg_after_last_echo)
		backgroundSub(m_dsoWave, pos, length, bgpos, bglength);
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
	if(!bg_after_last_echo)
		backgroundSub(m_dsoWave, pos, length, bgpos, bglength);

	//Incremental/Sequential average.
	if((emitter == _dso) || (!m_avcount)) {	
		for(int i = 0; i < length; i++) {
			m_waveSum[i] += m_dsoWave[pos + i];
		}
		{
			//Estimate power spectral density of dark side.
			if(!m_ftDark || (m_ftDark->length() != m_darkPSD.size())) {
				m_ftDark.reset(new FFT(-1, *fftLen()));
			}
			std::vector<std::complex<double> > darkin(fftlen, 0.0), darkout(fftlen);
			int bginplen = std::min(bglength, fftlen);
			double normalize = 0.0;
			//Twist background not to be affected by the dc subtraction.
			for(int i = 0; i < bginplen; i++) {
				double tw = sin(2.0*M_PI*i/(double)bginplen);
				darkin[i] = m_dsoWave[pos + i + bgpos] * tw;
				normalize += tw * tw;
			}
			normalize = 1.0 / normalize * interval;
			m_ftDark->exec(darkin, darkout);
			//Convolution for the rectangular window.
			for(int i = 0; i < fftlen; i++) {
				darkin[i] = std::norm(darkout[i]) * normalize;
			}
			m_ftDark->exec(darkin, darkout); //FT of PSD.
			std::vector<std::complex<double> > sigma2(darkout);
			std::fill(darkin.begin(), darkin.end(), 0.0);
			double x = sqrt(1.0 / length / fftlen);
			for(int i = 0; i < length; i++) {
				darkin[i] = x;
			}
			m_ftDark->exec(darkin, darkout); //FT of rect. window.
			for(int i = 0; i < fftlen; i++) {
				darkin[i] = std::norm(darkout[i]);
			}
			m_ftDark->exec(darkin, darkout); //FT of norm of (FT of rect. window). 
			for(int i = 0; i < fftlen; i++) {
				darkin[i] = std::conj(darkout[i] * sigma2[i]);
			}
			m_ftDark->exec(darkin, darkout); //Convolution.
			normalize = 1.0 / fftlen;
			for(int i = 0; i < fftlen; i++) {
				m_darkPSDSum[i] += std::real(darkout[i]) * normalize; //[V^2/Hz]
			}
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
	double darknormalize = normalize * normalize;
	if(bg_after_last_echo)
		darknormalize /= (double)numechoes;
	for(int i = 0; i < fftlen; i++) {
		m_darkPSD[i] = m_darkPSDSum[i] * darknormalize;
	}
	int ftpos = lrint(*fftPos() * 1e-3 / interval + _dso->trigPosRecorded() - pos);

	if(*difFreq() != 0.0) {
		//Digital IF.
		double omega = -2.0 * M_PI * *difFreq() * 1e3 * interval;
		for(int i = 0; i < length; i++) {
			m_wave[i] *= std::polar(1.0, omega * (i - ftpos));
		}
	}

	//	if((windowfunc != &windowFuncRect) && (abs(ftpos - length/2) > length*0.1))
	//		m_statusPrinter->printWarning(i18n("FFTPos is off-centered for window func."));
	double ph = *phaseAdv() * M_PI / 180;
	m_waveFTPos = ftpos;
	//[Hz]
	m_dFreq = 1.0 / fftlen / interval;
	m_ftWave.resize(fftlen);
	m_solverRecorded = m_solver->solver();

	rotNFFT(ftpos, ph, m_wave, m_ftWave);	
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
	for(Transaction tr( *ftWaveGraph());; ++tr) {
		tr[ *ftWaveGraph()].setRowCount(ftsize);
		double normalize = 1.0 / m_wave.size();
		double darknormalize = m_ftWavePSDCoeff / (m_wave.size() * interval());
		for (int i = 0; i < ftsize; i++) {
			int j = (i - ftsize/2 + ftsize) % ftsize;
			tr[ *ftWaveGraph()].cols(0)[i] = 0.001 * (i - ftsize/2) * m_dFreq;
			std::complex<double> z = m_ftWave[j] * normalize;
			tr[ *ftWaveGraph()].cols(1)[i] = std::real(z);
			tr[ *ftWaveGraph()].cols(2)[i] = std::imag(z);
			tr[ *ftWaveGraph()].cols(3)[i] = std::abs(z);
			tr[ *ftWaveGraph()].cols(4)[i] = std::arg(z) / M_PI * 180;
			tr[ *ftWaveGraph()].cols(5)[i] = sqrt(m_darkPSD[j] * darknormalize);
		}
		tr[ *m_peakPlot->maxCount()] = m_solverRecorded->peaks().size();
		std::deque<XGraph::ValPoint> &points(tr[ *m_peakPlot].points());
		points.resize(m_solverRecorded->peaks().size());
		for(int i = 0; i < m_solverRecorded->peaks().size(); i++) {
			double x = m_solverRecorded->peaks()[i].second;
			x = (x > ftsize / 2) ? (x - ftsize) : x;
			points[i] = XGraph::ValPoint(0.001 * x * m_dFreq,
				m_solverRecorded->peaks()[i].first * normalize);
		}
		ftWaveGraph()->drawGraph(tr);
		if(tr.commit()) {
			break;
		}
	}
	for(Transaction tr( *waveGraph());; ++tr) {
		int length = m_dsoWave.size();
		tr[ *waveGraph()].setRowCount(length);
		for (int i = 0; i < length; i++) {
			int j = i - m_dsoWaveStartPos;
			tr[ *waveGraph()].cols(0)[i] = (startTime() + j * m_interval) * 1e3;
			if(abs(j) < ftsize / 2) {
				j = (j - m_waveFTPos + ftsize) % ftsize;
				tr[ *waveGraph()].cols(1)[i] = std::real(m_solverRecorded->ifft()[j]);
				tr[ *waveGraph()].cols(2)[i] = std::imag(m_solverRecorded->ifft()[j]);
			}
			else {
				tr[ *waveGraph()].cols(1)[i] = 0.0;
				tr[ *waveGraph()].cols(2)[i] = 0.0;
			}
			tr[ *waveGraph()].cols(3)[i] = m_dsoWave[i].real();
			tr[ *waveGraph()].cols(4)[i] = m_dsoWave[i].imag();
		}
		waveGraph()->drawGraph(tr);
		if(tr.commit()) {
			break;
		}
	}
}

