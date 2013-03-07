/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp

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
		m_spectrumShow(create<XTouchableNode>("SpectrumShow", true)),
		m_avgClear(create<XTouchableNode>("AvgClear", true)),
		m_picEnabled(create<XBoolNode>("PICEnabled", false)),
		m_pulser(create<XItemNode<XDriverList, XPulser> >("Pulser", false, ref(tr_meas), meas->drivers(), true)),
		m_form(new FrmNMRPulse(g_pFrmMain)),
		m_statusPrinter(XStatusPrinter::create(m_form.get())),
		m_spectrumForm(new FrmGraphNURL(g_pFrmMain)), m_waveGraph(create<XWaveNGraph>("Wave", true,
			m_form->m_graph, m_form->m_urlDump, m_form->m_btnDump)),
		m_ftWaveGraph(create<XWaveNGraph>("Spectrum", true, m_spectrumForm.get())),
		m_solver(create<SpectrumSolverWrapper>("SpectrumSolverWrapper", true, m_solverList, m_windowFunc, m_windowWidth)),
		m_solverPNR(create<SpectrumSolverWrapper>("PNRSpectrumSolverWrapper", true, m_pnrSolverList, shared_ptr<XComboNode>(), shared_ptr<XDoubleNode>(), true)) {
	m_form->m_btnAvgClear->setIcon(KIconLoader::global()->loadIcon("edit-clear", KIconLoader::Toolbar, KIconLoader::SizeSmall, true) );
	m_form->m_btnSpectrum->setIcon(KIconLoader::global()->loadIcon("graph", KIconLoader::Toolbar, KIconLoader::SizeSmall, true) );
	
	connect(dso());
	connect(pulser());

	meas->scalarEntries()->insert(tr_meas, entryPeakAbs());
	meas->scalarEntries()->insert(tr_meas, entryPeakFreq());

	for(Transaction tr( *this);; ++tr) {
		tr[ *m_pnrSolverList].str(XString(SpectrumSolverWrapper::SPECTRUM_SOLVER_LS_MDL));
		tr[ *fromTrig()] = -0.005;
		tr[ *width()] = 0.02;
		tr[ *bgPos()] = 0.03;
		tr[ *bgWidth()] = 0.03;
		tr[ *fftPos()] = 0.004;
		tr[ *fftLen()] = 16384;
		tr[ *numEcho()] = 1;
		tr[ *windowFunc()].str(XString(SpectrumSolverWrapper::WINDOW_FUNC_DEFAULT));
		tr[ *windowWidth()] = 100.0;

		if(tr.commit())
			break;
	}

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
		tr[ *waveGraph()].clearPoints();
		if(tr.commit())
			break;
	}
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
		tr[ *ftWaveGraph()].clearPoints();
		if(tr.commit())
			break;
	}

	for(Transaction tr( *this);; ++tr) {
		m_lsnOnAvgClear = tr[ *m_avgClear].onTouch().connectWeakly(
			shared_from_this(), &XNMRPulseAnalyzer::onAvgClear);
		m_lsnOnSpectrumShow = tr[ *m_spectrumShow].onTouch().connectWeakly(
			shared_from_this(), &XNMRPulseAnalyzer::onSpectrumShow,
			XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP);

		m_lsnOnCondChanged = tr[ *fromTrig()].onValueChanged().connectWeakly(
			shared_from_this(), &XNMRPulseAnalyzer::onCondChanged);
		tr[ *width()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *phaseAdv()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *usePNR()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *pnrSolverList()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *solverList()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *bgPos()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *bgWidth()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *fftPos()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *fftLen()].onValueChanged().connect(m_lsnOnCondChanged);
		//	extraAvg()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *exAvgIncr()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *numEcho()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *echoPeriod()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *windowFunc()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *windowWidth()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *difFreq()].onValueChanged().connect(m_lsnOnCondChanged);
		if(tr.commit())
			break;
	}
}
XNMRPulseAnalyzer::~XNMRPulseAnalyzer() {
}
void XNMRPulseAnalyzer::onSpectrumShow(const Snapshot &shot, XTouchableNode *) {
	m_spectrumForm->show();
	m_spectrumForm->raise();
}
void XNMRPulseAnalyzer::showForms() {
	m_form->show();
	m_form->raise();
}

void XNMRPulseAnalyzer::backgroundSub(Transaction &tr,
	std::vector<std::complex<double> > &wave,
	int pos, int length, int bgpos, int bglength) {
	Snapshot &shot(tr);

	std::complex<double> bg = 0;
	if(bglength) {
		double normalize = 0.0;
		for(int i = 0; i < bglength; i++) {
			double z = 1.0;
			if( !shot[ *usePNR()])
				z = FFT::windowFuncHamming( (double)i / bglength - 0.5);
			bg += z * wave[pos + i + bgpos];
			normalize += z;
		}
		bg /= normalize;
	}

	for(int i = 0; i < wave.size(); i++) {
		wave[i] -= bg;
	}

	SpectrumSolver &solverPNR(tr[ *m_solverPNR].solver());
	if(bglength) {
		if(shot[ *usePNR()]) {
			int dnrlength = FFT::fitLength((bglength + bgpos) * 4);
			std::vector<std::complex<double> > memin(bglength), memout(dnrlength);
			for(unsigned int i = 0; i < bglength; i++) {
				memin[i] = wave[pos + i + bgpos];
			}
			try {
				solverPNR.exec(memin, memout, bgpos, 0.5e-2, &FFT::windowFuncRect, 1.0);
				int imax = std::min((int)wave.size() - pos, (int)memout.size());
				for(unsigned int i = 0; i < imax; i++) {
					wave[i + pos] -= solverPNR.ifft()[i];
				}
			}
			catch (XKameError &e) {
				e.print();
//				throw XSkippedRecordError(e.msg(), __FILE__, __LINE__);
			}
		}
	}
}
void XNMRPulseAnalyzer::rotNFFT(Transaction &tr, int ftpos, double ph,
	std::vector<std::complex<double> > &wave,
	std::vector<std::complex<double> > &ftwave) {
	Snapshot &shot(tr);

	int length = wave.size();
	//phase advance
	std::complex<double> cph(std::polar(1.0, ph));
	for(int i = 0; i < length; i++) {
		wave[i] *= cph;
	}

	int fftlen = ftwave.size();
	//fft
	std::vector<std::complex<double> > fftout(fftlen);
	SpectrumSolver &solver(tr[ *m_solver].solver());
	FFT::twindowfunc wndfunc = m_solver->windowFunc(shot);
	double wndwidth = shot[ *windowWidth()] / 100.0;
	try {
		solver.exec(wave, fftout, -ftpos, 0.3e-2, wndfunc, wndwidth);
	}
	catch (XKameError &e) {
		throw XSkippedRecordError(e.msg(), __FILE__, __LINE__);
	}

	std::copy(fftout.begin(), fftout.end(), ftwave.begin());
	
	if(solver.isFT()) {
		std::vector<double> weight;
		SpectrumSolver::window(length, -ftpos, wndfunc, wndwidth, weight);
		double w = 0;
		for(int i = 0; i < length; i++)
			w += weight[i] * weight[i];
		tr[ *this].m_ftWavePSDCoeff = w/(double)length;
	}
	else {
		tr[ *this].m_ftWavePSDCoeff = 1.0;
	}
}
void XNMRPulseAnalyzer::onAvgClear(const Snapshot &shot, XTouchableNode *) {
	trans( *this).m_timeClearRequested = XTime::now();

	Snapshot shot_this( *this);
	requestAnalysis();

	const shared_ptr<XDSO> dso__ = shot_this[ *dso()];
	if(dso__)
		trans( *dso__->restart()).touch(); //Restart averaging in DSO.
}
void XNMRPulseAnalyzer::onCondChanged(const Snapshot &shot, XValueNodeBase *node) {
	if(node == exAvgIncr().get())
		trans( *extraAvg()) = 0;
	if((node == numEcho().get()) || (node == difFreq().get()) || (node == exAvgIncr().get()))
		onAvgClear(shot, avgClear().get());
	else
		requestAnalysis();
}
bool XNMRPulseAnalyzer::checkDependency(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
	const shared_ptr<XPulser> pulse__ = shot_this[ *pulser()];
	if (emitter == pulse__.get())
		return false;
	const shared_ptr<XDSO> dso__ = shot_this[ *dso()];
	if( !dso__)
		return false;
	//    //Request for clear.
	//    if(m_timeClearRequested > dso__->timeAwared()) return true;
	//    if(pulse__ && (dso__->timeAwared() < pulse__->time())) return false;
	return true;
}
void XNMRPulseAnalyzer::analyze(Transaction &tr, const Snapshot &shot_emitter,
	const Snapshot &shot_others,
	XDriver *emitter) throw (XRecordError&) {
	const Snapshot &shot_this(tr);
	const shared_ptr<XDSO> dso__ = shot_this[ *dso()];
	assert(dso__);

	const Snapshot &shot_dso((emitter == dso__.get()) ? shot_emitter : shot_others);
	assert(shot_dso[ *dso__].time() );

	if(shot_dso[ *dso__].numChannels() < 1) {
		throw XSkippedRecordError(i18n("No record in DSO"), __FILE__, __LINE__);
	}
	if(shot_dso[ *dso__].numChannels() < 2) {
		throw XSkippedRecordError(i18n("Two channels needed in DSO"), __FILE__, __LINE__);
	}
	if( !shot_dso[ *dso__->singleSequence()]) {
		m_statusPrinter->printWarning(i18n("Use sequential average in DSO."));
	}
	int dso_len = shot_dso[ *dso__].length();

	double interval = shot_dso[ *dso__].timeInterval();
	if (interval <= 0) {
		throw XSkippedRecordError(i18n("Invalid time interval in waveforms."), __FILE__, __LINE__);
	}
	int pos = lrint(shot_this[ *fromTrig()] *1e-3 / interval + shot_dso[ *dso__].trigPos());
	double starttime = (pos - shot_dso[ *dso__].trigPos()) * interval;
	if(pos >= dso_len) {
		throw XSkippedRecordError(i18n("Position beyond waveforms."), __FILE__, __LINE__);
	}
	if(pos < 0) {
		throw XSkippedRecordError(i18n("Position beyond waveforms."), __FILE__, __LINE__);
	}
	int length = lrint( shot_this[ *width()] / 1000 / interval);
	if(pos + length >= dso_len) {
		throw XSkippedRecordError(i18n("Invalid length."), __FILE__, __LINE__);
	}
	if(length <= 0) {
		throw XSkippedRecordError(i18n("Invalid length."), __FILE__, __LINE__);
	}
	
	int bgpos = lrint((shot_this[ *bgPos()] - shot_this[ *fromTrig()]) / 1000 / interval);
	if(pos + bgpos >= dso_len) {
		throw XSkippedRecordError(i18n("Position for BG. sub. beyond waveforms."), __FILE__, __LINE__);
	}
	if(bgpos < 0) {
		throw XSkippedRecordError(i18n("Position for BG. sub. beyond waveforms."), __FILE__, __LINE__);
	}
	int bglength = lrint(shot_this[ *bgWidth()] / 1000 / interval);
	if(pos + bgpos + bglength >= dso_len) {
		throw XSkippedRecordError(i18n("Invalid Length for BG. sub."), __FILE__, __LINE__);
	}
	if(bglength < 0) {
		throw XSkippedRecordError(i18n("Invalid Length for BG. sub."), __FILE__, __LINE__);
	}

	shared_ptr<XPulser> pulse__(shot_this[ *pulser()]);
	
	int echoperiod = lrint(shot_this[ *echoPeriod()] / 1000 /interval);
	int numechoes = shot_this[ *numEcho()];
	numechoes = std::max(1, numechoes);
	bool bg_after_last_echo = (echoperiod < bgpos + bglength);

	if(bglength && (bglength < length * (bg_after_last_echo ? numechoes : 1) * 3))
		m_statusPrinter->printWarning(i18n("Maybe, length for BG. sub. is too short."));
	
	if((bgpos < length + (bg_after_last_echo ? (echoperiod * (numechoes - 1)) : 0)) 
		&& (bgpos + bglength > 0))
		m_statusPrinter->printWarning(i18n("Maybe, position for BG. sub. is overrapped against echoes"), true);

	if(numechoes > 1) {
		if(pos + echoperiod * (numechoes - 1) + length >= dso_len) {
			throw XSkippedRecordError(i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
		}
		if(echoperiod < length) {
			throw XSkippedRecordError(i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
		}
		if( !bg_after_last_echo) {
			if(bgpos + bglength > echoperiod) {
				throw XSkippedRecordError(i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
			}
			if(pos + echoperiod * (numechoes - 1) + bgpos + bglength >= dso_len) {
				throw XSkippedRecordError(i18n("Invalid Multiecho settings."), __FILE__, __LINE__);
			}
		}
		if(pulse__) {
			if((numechoes > shot_others[ *pulse__].echoNum()) ||
				(fabs(shot_this[ *echoPeriod()] * 1e3 / (shot_others[ *pulse__].tau() * 2.0) - 1.0) > 1e-4)) {
				m_statusPrinter->printWarning(i18n("Invalid Multiecho settings."), true);
			}
		}
	}
	if(pulse__) {
		if(shot_others[ *pulse__].isPulseAnalyzerMode()) {
			m_statusPrinter->printWarning(i18n("Built-In Network Analyzer Mode."), false);
			throw XSkippedRecordError(__FILE__, __LINE__);
		}
	}

	if((shot_this[ *this].m_startTime != starttime) || (length != shot_this[ *this].m_waveWidth)) {
		double t = length * interval * 1e3;
		tr[ *tr[ *waveGraph()].axisx()->autoScale()] = false;
		tr[ *tr[ *waveGraph()].axisx()->minValue()] = starttime * 1e3 - t * 0.3;
		tr[ *tr[ *waveGraph()].axisx()->maxValue()] = starttime * 1e3 + t * 1.3;
	}
	tr[ *this].m_waveWidth = length;
	bool skip = (shot_this[ *this].m_timeClearRequested > shot_dso[ *dso__].timeAwared());
	bool avgclear = skip;

	if(interval != shot_this[ *this].m_interval) {
		//[sec]
		tr[ *this].m_interval = interval;
		avgclear = true;
	}
	if(shot_this[ *this].m_startTime != starttime) {
		//[sec]
		tr[ *this].m_startTime = starttime;
		avgclear = true;
	}
	
	if(length > (int)shot_this[ *this].m_waveSum.size()) {
		avgclear = true;
	}
	tr[ *this].m_wave.resize(length);
	tr[ *this].m_waveSum.resize(length);
	int fftlen = FFT::fitLength(shot_this[ *fftLen()]);
	if(fftlen != shot_this[ *this].m_darkPSD.size()) {
		avgclear = true;		
	}
	if(length > fftlen) {
		throw XSkippedRecordError(i18n("FFT length is too short."), __FILE__, __LINE__);
	}
	tr[ *this].m_darkPSD.resize(fftlen);
	tr[ *this].m_darkPSDSum.resize(fftlen);
	std::fill(tr[ *this].m_wave.begin(), tr[ *this].m_wave.end(), std::complex<double>(0.0));

	// Phase Inversion Cycling
	bool picenabled = shot_this[ *m_picEnabled];
	bool inverted = false;
	if(picenabled && ( !pulse__ || !shot_others[ *pulse__].time())) {
		picenabled = false;
		gErrPrint(getLabel() + ": " + i18n("No active pulser!"));
	}
	if(pulse__) {
		inverted = shot_others[ *pulse__].invertPhase();
	}

	int avgnum = std::max((unsigned int)shot_this[ *extraAvg()], 1u) * (picenabled ? 2 : 1);
	
	if( !shot_this[ *exAvgIncr()] && (avgnum <= shot_this[ *this].m_avcount)) {
		avgclear = true;
	}
	if(avgclear) {
		std::fill(tr[ *this].m_waveSum.begin(), tr[ *this].m_waveSum.end(), std::complex<double>(0.0));
		std::fill(tr[ *this].m_darkPSDSum.begin(), tr[ *this].m_darkPSDSum.end(), 0.0);
		tr[ *this].m_avcount = 0;
		if(shot_this[ *exAvgIncr()]) {
			tr[ *extraAvg()] = 0;
		}
	}

	if(skip) {
		throw XSkippedRecordError(__FILE__, __LINE__);
	}

	tr[ *this].m_dsoWave.resize(dso_len);
	std::complex<double> *dsowave( &tr[ *this].m_dsoWave[0]);
	{
		const double *rawwavecos, *rawwavesin = NULL;
		assert(shot_dso[ *dso__].numChannels() );
		rawwavecos = shot_dso[ *dso__].wave(0);
		rawwavesin = shot_dso[ *dso__].wave(1);
		for(unsigned int i = 0; i < dso_len; i++) {
			dsowave[i] = std::complex<double>(rawwavecos[i], rawwavesin[i]) * (inverted ? -1.0 : 1.0);
		}
	}
	tr[ *this].m_dsoWaveStartPos = pos;

	//Background subtraction or dynamic noise reduction
	if(bg_after_last_echo)
		backgroundSub(tr, tr[ *this].m_dsoWave, pos, length, bgpos, bglength);
	for(int i = 1; i < numechoes; i++) {
		int rpos = pos + i * echoperiod;
		for(int j = 0;
		j < ( !bg_after_last_echo ? std::max(bgpos + bglength, length) : length); j++) {
			int k = rpos + j;
			assert(k < dso_len);
			if(i == 1)
				dsowave[pos + j] /= (double)numechoes;
			dsowave[pos + j] += dsowave[k] / (double)numechoes;
		}
	}
	//Background subtraction or dynamic noise reduction
	if( !bg_after_last_echo)
		backgroundSub(tr, tr[ *this].m_dsoWave, pos, length, bgpos, bglength);

	std::complex<double> *wavesum( &tr[ *this].m_waveSum[0]);
	double *darkpsdsum( &tr[ *this].m_darkPSDSum[0]);
	//Incremental/Sequential average.
	if((emitter == dso__.get()) || ( !shot_this[ *this].m_avcount)) {
		for(int i = 0; i < length; i++) {
			wavesum[i] += dsowave[pos + i];
		}
		if(bglength) {
			//Estimates power spectral density in the background.
			if( !shot_this[ *this].m_ftDark ||
				(shot_this[ *this].m_ftDark->length() != shot_this[ *this].m_darkPSD.size())) {
				tr[ *this].m_ftDark.reset(new FFT(-1, shot_this[ *fftLen()]));
			}
			std::vector<std::complex<double> > darkin(fftlen, 0.0), darkout(fftlen);
			int bginplen = std::min(bglength, fftlen);
			double normalize = 0.0;
			//Twists background not to be affected by the dc subtraction.
			for(int i = 0; i < bginplen; i++) {
				double tw = sin(2.0*M_PI*i/(double)bginplen);
				darkin[i] = dsowave[pos + i + bgpos] * tw;
				normalize += tw * tw;
			}
			normalize = 1.0 / normalize * interval;
			tr[ *this].m_ftDark->exec(darkin, darkout);
			//Convolution for the rectangular window.
			for(int i = 0; i < fftlen; i++) {
				darkin[i] = std::norm(darkout[i]) * normalize;
			}
			tr[ *this].m_ftDark->exec(darkin, darkout); //FT of PSD.
			std::vector<std::complex<double> > sigma2(darkout);
			std::fill(darkin.begin(), darkin.end(), std::complex<double>(0.0));
			double x = sqrt(1.0 / length / fftlen);
			for(int i = 0; i < length; i++) {
				darkin[i] = x;
			}
			tr[ *this].m_ftDark->exec(darkin, darkout); //FT of rect. window.
			for(int i = 0; i < fftlen; i++) {
				darkin[i] = std::norm(darkout[i]);
			}
			tr[ *this].m_ftDark->exec(darkin, darkout); //FT of norm of (FT of rect. window).
			for(int i = 0; i < fftlen; i++) {
				darkin[i] = std::conj(darkout[i] * sigma2[i]);
			}
			tr[ *this].m_ftDark->exec(darkin, darkout); //Convolution.
			normalize = 1.0 / fftlen;
			for(int i = 0; i < fftlen; i++) {
				darkpsdsum[i] += std::real(darkout[i]) * normalize; //[V^2/Hz]
			}
		}
		tr[ *this].m_avcount++;
		if( shot_this[ *exAvgIncr()]) {
			tr[ *extraAvg()] = shot_this[ *this].m_avcount;
		}
	}
	std::complex<double> *wave( &tr[ *this].m_wave[0]);
	double normalize = 1.0 / shot_this[ *this].m_avcount;
	for(int i = 0; i < length; i++) {
		wave[i] = wavesum[i] * normalize;
	}
	double darknormalize = normalize * normalize;
	if(bg_after_last_echo)
		darknormalize /= (double)numechoes;
	double *darkpsd( &tr[ *this].m_darkPSD[0]);
	for(int i = 0; i < fftlen; i++) {
		darkpsd[i] = darkpsdsum[i] * darknormalize;
	}
	int ftpos = lrint( shot_this[ *fftPos()] * 1e-3 / interval + shot_dso[ *dso__].trigPos() - pos);

	if(shot_this[ *difFreq()] != 0.0) {
		//Digital IF.
		double omega = -2.0 * M_PI * shot_this[ *difFreq()] * 1e3 * interval;
		for(int i = 0; i < length; i++) {
			wave[i] *= std::polar(1.0, omega * (i - ftpos));
		}
	}

	//	if((windowfunc != &windowFuncRect) && (abs(ftpos - length/2) > length*0.1))
	//		m_statusPrinter->printWarning(i18n("FFTPos is off-centered for window func."));
	double ph = shot_this[ *phaseAdv()] * M_PI / 180;
	tr[ *this].m_waveFTPos = ftpos;
	//[Hz]
	tr[ *this].m_dFreq = 1.0 / fftlen / interval;
	tr[ *this].m_ftWave.resize(fftlen);

	rotNFFT(tr, ftpos, ph, tr[ *this].m_wave, tr[ *this].m_ftWave); //Generates a new SpectrumSolver.
	const SpectrumSolver &solver(shot_this[ *m_solver].solver());
	if(solver.peaks().size()) {
		entryPeakAbs()->value(tr,
			solver.peaks()[0].first / (double)shot_this[ *this].m_wave.size());
		double x = solver.peaks()[0].second;
		x = (x > fftlen / 2) ? (x - fftlen) : x;
		entryPeakFreq()->value(tr,
			0.001 * x * shot_this[ *this].m_dFreq);
	}
	
	m_isPulseInversionRequested = picenabled && (shot_this[ *this].m_avcount % 2 == 1) && (emitter == dso__.get());

	if( !shot_this[ *exAvgIncr()] && (avgnum != shot_this[ *this].m_avcount))
		throw XSkippedRecordError(__FILE__, __LINE__);
}
void XNMRPulseAnalyzer::visualize(const Snapshot &shot) {
	for(Transaction tr( *this);; ++tr) {
		Snapshot &shot(tr);
		if(shot[ *this].time() && shot[ *this].m_avcount)
			break;

		tr[ *ftWaveGraph()].clearPoints();
		tr[ *waveGraph()].clearPoints();
		tr[ *m_peakPlot->maxCount()] = 0;
		if(tr.commit())
			return;
	}

	if(m_isPulseInversionRequested.compareAndSet((int)true, (int)false)) {
		shared_ptr<XPulser> pulse__ = shot[ *pulser()];
		if(pulse__) {
			for(Transaction tr( *pulse__);; ++tr) {
				if(tr[ *pulse__].time()) {
					tr[ *pulse__->invertPhase()] = !tr[ *pulse__->invertPhase()];
				}
				if(tr.commit())
					break;
			}
		}
	}

	for(Transaction tr( *this);; ++tr) {
		Snapshot &shot(tr);
		const SpectrumSolver &solver(shot[ *m_solver].solver());

		int ftsize = shot[ *this].m_ftWave.size();

		tr[ *ftWaveGraph()].setRowCount(ftsize);
		double normalize = 1.0 / shot[ *this].m_wave.size();
		double darknormalize =
			shot[ *this].m_ftWavePSDCoeff / (shot[ *this].m_wave.size() * shot[ *this].interval());
		double dfreq = shot[ *this].m_dFreq;
		const double *darkpsd( &shot[ *this].m_darkPSD[0]);
		const std::complex<double> *ftwave( &shot[ *this].m_ftWave[0]);
		double *colf(tr[ *ftWaveGraph()].cols(0));
		double *colr(tr[ *ftWaveGraph()].cols(1));
		double *coli(tr[ *ftWaveGraph()].cols(2));
		double *colabs(tr[ *ftWaveGraph()].cols(3));
		double *colarg(tr[ *ftWaveGraph()].cols(4));
		double *coldark(tr[ *ftWaveGraph()].cols(5));
		for (int i = 0; i < ftsize; i++) {
			int j = (i - ftsize/2 + ftsize) % ftsize;
			colf[i] = 0.001 * (i - ftsize/2) * dfreq;
			std::complex<double> z = ftwave[j] * normalize;
			colr[i] = std::real(z);
			coli[i] = std::imag(z);
			colabs[i] = std::abs(z);
			colarg[i] = std::arg(z) / M_PI * 180;
			coldark[i] = sqrt(darkpsd[j] * darknormalize);
		}
		const std::vector<std::pair<double, double> > &peaks(solver.peaks());
		int peaks_size = peaks.size();
		tr[ *m_peakPlot->maxCount()] = peaks_size;
		std::deque<XGraph::ValPoint> &points(tr[ *m_peakPlot].points());
		points.resize(peaks_size);
		for(int i = 0; i < peaks_size; i++) {
			double x = peaks[i].second;
			x = (x > ftsize / 2) ? (x - ftsize) : x;
			points[i] = XGraph::ValPoint(0.001 * x * dfreq, peaks[i].first * normalize);
		}
		ftWaveGraph()->drawGraph(tr);

		int length = shot[ *this].m_dsoWave.size();
		const std::complex<double> *dsowave( &shot[ *this].m_dsoWave[0]);
		const std::complex<double> *ifft( &solver.ifft()[0]);
		int dsowavestartpos = shot[ *this].m_dsoWaveStartPos;
		double interval = shot[ *this].m_interval;
		double starttime = shot[ *this].startTime();
		int waveftpos = shot[ *this].m_waveFTPos;
		tr[ *waveGraph()].setRowCount(length);
		double *colt(tr[ *waveGraph()].cols(0));
		double *colfr(tr[ *waveGraph()].cols(1));
		double *colfi(tr[ *waveGraph()].cols(2));
		double *colrr(tr[ *waveGraph()].cols(3));
		double *colri(tr[ *waveGraph()].cols(4));
		for (int i = 0; i < length; i++) {
			int j = i - dsowavestartpos;
			colt[i] = (starttime + j * interval) * 1e3;
			if(abs(j) < ftsize / 2) {
				j = (j - waveftpos + ftsize) % ftsize;
				colfr[i] = std::real(ifft[j]);
				colfi[i] = std::imag(ifft[j]);
			}
			else {
				colfr[i] = 0.0;
				colfi[i] = 0.0;
			}
			colrr[i] = dsowave[i].real();
			colri[i] = dsowave[i].imag();
		}
		waveGraph()->drawGraph(tr);
		if(tr.commit()) {
			break;
		}
	}
}

