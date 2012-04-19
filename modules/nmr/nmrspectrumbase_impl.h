/***************************************************************************
		Copyright (C) 2002-2011 Kentaro Kitagawa
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

#include <QPushButton>
#include <QComboBox>
#include <QCheckBox>
#include <knuminput.h>
#include <kiconloader.h>
//---------------------------------------------------------------------------
template <class FRM>
XNMRSpectrumBase<FRM>::XNMRSpectrumBase(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
	: XSecondaryDriver(name, runtime, ref(tr_meas), meas),
	m_pulse(create<XItemNode<XDriverList, XNMRPulseAnalyzer> >(
	  "PulseAnalyzer", false, ref(tr_meas), meas->drivers(), true)),
	m_bandWidth(create<XDoubleNode>("BandWidth", false)),
	m_bwList(create<XComboNode>("BandWidthList", false, true)),
	m_autoPhase(create<XBoolNode>("AutoPhase", false)),
	m_phase(create<XDoubleNode>("Phase", false, "%.2f")),
	m_clear(create<XTouchableNode>("Clear", true)),
	m_solverList(create<XComboNode>("SpectrumSolver", false, true)),
	m_windowFunc(create<XComboNode>("WindowFunc", false, true)),
	m_windowWidth(create<XDoubleNode>("WindowWidth", false)),
	m_solver(create<SpectrumSolverWrapper>("SpectrumSolver", true, m_solverList, m_windowFunc, m_windowWidth)),
	m_form(new FRM(g_pFrmMain)),
	m_statusPrinter(XStatusPrinter::create(m_form.get())),
	m_spectrum(create<XWaveNGraph>("Spectrum", true, m_form->m_graph, m_form->m_urlDump, m_form->m_btnDump)) {
	m_form->m_btnClear->setIcon(
    	KIconLoader::global()->loadIcon("edit-clear",
		KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
    
	connect(pulse());

	for(Transaction tr( *this);; ++tr) {
		const char *labels[] = {"X", "Re [V]", "Im [V]", "Weights", "Abs [V]", "Dark [V]"};
		tr[ *m_spectrum].setColCount(6, labels);
		tr[ *m_spectrum].insertPlot(labels[4], 0, 4, -1, 3);
		tr[ *m_spectrum].insertPlot(labels[1], 0, 1, -1, 3);
		tr[ *m_spectrum].insertPlot(labels[2], 0, 2, -1, 3);
		tr[ *m_spectrum].insertPlot(labels[5], 0, 5, -1, 3);
		tr[ *tr[ *m_spectrum].axisy()->label()] = i18n("Intens. [V]");
		tr[ *tr[ *m_spectrum].plot(1)->label()] = i18n("real part");
		tr[ *tr[ *m_spectrum].plot(1)->drawPoints()] = false;
		tr[ *tr[ *m_spectrum].plot(1)->lineColor()] = clRed;
		tr[ *tr[ *m_spectrum].plot(2)->label()] = i18n("imag. part");
		tr[ *tr[ *m_spectrum].plot(2)->drawPoints()] = false;
		tr[ *tr[ *m_spectrum].plot(2)->lineColor()] = clGreen;
		tr[ *tr[ *m_spectrum].plot(0)->label()] = i18n("abs.");
		tr[ *tr[ *m_spectrum].plot(0)->drawPoints()] = false;
		tr[ *tr[ *m_spectrum].plot(0)->drawLines()] = true;
		tr[ *tr[ *m_spectrum].plot(0)->drawBars()] = true;
		tr[ *tr[ *m_spectrum].plot(0)->barColor()] = QColor(0x60, 0x60, 0xc0).rgb();
		tr[ *tr[ *m_spectrum].plot(0)->lineColor()] = QColor(0x60, 0x60, 0xc0).rgb();
		tr[ *tr[ *m_spectrum].plot(0)->intensity()] = 0.5;
		tr[ *tr[ *m_spectrum].plot(3)->label()] = i18n("dark");
		tr[ *tr[ *m_spectrum].plot(3)->drawBars()] = false;
		tr[ *tr[ *m_spectrum].plot(3)->drawLines()] = true;
		tr[ *tr[ *m_spectrum].plot(3)->lineColor()] = QColor(0xa0, 0xa0, 0x00).rgb();
		tr[ *tr[ *m_spectrum].plot(3)->drawPoints()] = false;
		tr[ *tr[ *m_spectrum].plot(3)->intensity()] = 0.5;
		{
			shared_ptr<XXYPlot> plot = m_spectrum->graph()->plots()->create<XXYPlot>(
				tr, "Peaks", true, ref(tr), m_spectrum->graph());
			m_peakPlot = plot;
			tr[ *plot->label()] = i18n("Peaks");
			tr[ *plot->axisX()] = tr[ *m_spectrum].axisx();
			tr[ *plot->axisY()] = tr[ *m_spectrum].axisy();
			tr[ *plot->drawPoints()] = false;
			tr[ *plot->drawLines()] = false;
			tr[ *plot->drawBars()] = true;
			tr[ *plot->intensity()] = 2.0;
			tr[ *plot->displayMajorGrid()] = false;
			tr[ *plot->pointColor()] = QColor(0xa0, 0x00, 0xa0).rgb();
			tr[ *plot->barColor()] = QColor(0xa0, 0x00, 0xa0).rgb();
			tr[ *plot->clearPoints()].setUIEnabled(false);
			tr[ *plot->maxCount()].setUIEnabled(false);
		}
		tr[ *m_spectrum].clearPoints();

		tr[ *bandWidth()] = 50;
		tr[ *bwList()].add("50%");
		tr[ *bwList()].add("100%");
		tr[ *bwList()].add("200%");
		tr[ *bwList()] = 1;
		tr[ *autoPhase()] = true;

		tr[ *windowFunc()].str(XString(SpectrumSolverWrapper::WINDOW_FUNC_DEFAULT));
		tr[ *windowWidth()] = 100.0;

		if(tr.commit())
			break;
	}
  
	m_conBandWidth = xqcon_create<XQLineEditConnector>(m_bandWidth, m_form->m_edBW);
	m_conBWList = xqcon_create<XQComboBoxConnector>(m_bwList, m_form->m_cmbBWList, Snapshot( *m_bwList));
	m_conPhase = xqcon_create<XKDoubleNumInputConnector>(m_phase, m_form->m_numPhase);
	m_form->m_numPhase->setRange(-360.0, 360.0, 10.0, true);
	m_conAutoPhase = xqcon_create<XQToggleButtonConnector>(m_autoPhase, m_form->m_ckbAutoPhase);
	m_conPulse = xqcon_create<XQComboBoxConnector>(m_pulse, m_form->m_cmbPulse, ref(tr_meas));
	m_conClear = xqcon_create<XQButtonConnector>(m_clear, m_form->m_btnClear);
	m_conSolverList = xqcon_create<XQComboBoxConnector>(m_solverList, m_form->m_cmbSolver, Snapshot( *m_solverList));
	m_conWindowWidth = xqcon_create<XKDoubleNumInputConnector>(m_windowWidth,
		m_form->m_numWindowWidth);
	m_form->m_numWindowWidth->setRange(0.1, 200.0, 1.0, true);
	m_conWindowFunc = xqcon_create<XQComboBoxConnector>(m_windowFunc,
		m_form->m_cmbWindowFunc, Snapshot( *m_windowFunc));

	for(Transaction tr( *this);; ++tr) {
		m_lsnOnClear = tr[ *m_clear].onTouch().connectWeakly(
			shared_from_this(), &XNMRSpectrumBase<FRM>::onClear);
		m_lsnOnCondChanged = tr[ *bandWidth()].onValueChanged().connectWeakly(
			shared_from_this(), &XNMRSpectrumBase<FRM>::onCondChanged);
		tr[ *autoPhase()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *phase()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *solverList()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *windowWidth()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *windowFunc()].onValueChanged().connect(m_lsnOnCondChanged);
		tr[ *bwList()].onValueChanged().connect(m_lsnOnCondChanged);
		if(tr.commit())
			break;
	}
}
template <class FRM>
XNMRSpectrumBase<FRM>::~XNMRSpectrumBase() {
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::showForms() {
	m_form->show();
	m_form->raise();
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::onCondChanged(const Snapshot &shot, XValueNodeBase *node) {
//    if((node == phase()) && *autoPhase()) return;
	if((node == bandWidth().get()) || onCondChangedImpl(shot, node))
        trans( *this).m_timeClearRequested = XTime::now();
    requestAnalysis();
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::onClear(const Snapshot &shot, XTouchableNode *) {
    trans( *this).m_timeClearRequested = XTime::now();
    requestAnalysis();
}
template <class FRM>
bool
XNMRSpectrumBase<FRM>::checkDependency(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
    shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];
    if( !pulse__) return false;
    if(emitter == this) return true;
    return (emitter == pulse__.get()) && checkDependencyImpl(shot_this, shot_emitter, shot_others, emitter);
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) throw (XRecordError&) {
	const Snapshot &shot_this(tr);

	shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];
	assert( pulse__ );
	const Snapshot &shot_pulse((emitter == pulse__.get()) ? shot_emitter : shot_others);
 
	if(shot_pulse[ *pulse__->exAvgIncr()]) {
		m_statusPrinter->printWarning(i18n("Do NOT use incremental avg. Skipping."));
		throw XSkippedRecordError(__FILE__, __LINE__);
	}

	bool clear = (shot_this[ *this].m_timeClearRequested > shot_pulse[ *pulse__].timeAwared());
  
//	double interval = shot_pulse[ *pulse__].interval();
	double df = shot_pulse[ *pulse__].dFreq();
	
	double res = getFreqResHint(shot_this);
	res = df * std::max(1L, lrint(res / df - 0.5));
	
	double max__ = getMaxFreq(shot_this);
	double min__ = getMinFreq(shot_this);
	
	if(max__ <= min__) {
		throw XSkippedRecordError(i18n("Invalid min. and max."), __FILE__, __LINE__);
	}
	if(res * 65536 * 2 < max__ - min__) {
		throw XSkippedRecordError(i18n("Too small resolution."), __FILE__, __LINE__);
	}

	if(fabs(log(shot_this[ *this].res() / res)) < log(2.0))
		res = shot_this[ *this].res();
	
	if((shot_this[ *this].res() != res) || clear) {
		tr[ *this].m_res = res;
		for(int bank = 0; bank < Payload::ACCUM_BANKS; bank++) {
			tr[ *this].m_accum[bank].clear();
			tr[ *this].m_accum_weights[bank].clear();
			tr[ *this].m_accum_dark[bank].clear();
		}
	}
	else {
		int diff = lrint(shot_this[ *this].min() / res) - lrint(min__ / res);
		for(int bank = 0; bank < Payload::ACCUM_BANKS; bank++) {
			for(int i = 0; i < diff; i++) {
				tr[ *this].m_accum[bank].push_front(0.0);
				tr[ *this].m_accum_weights[bank].push_front(0);
				tr[ *this].m_accum_dark[bank].push_front(0.0);
			}
			for(int i = 0; i < -diff; i++) {
				if( !shot_this[ *this].m_accum[bank].empty()) {
					tr[ *this].m_accum[bank].pop_front();
					tr[ *this].m_accum_weights[bank].pop_front();
					tr[ *this].m_accum_dark[bank].pop_front();
				}
			}
		}
	}
	tr[ *this].m_min = min__;
	int length = lrint((max__ - min__) / res);
	for(int bank = 0; bank < Payload::ACCUM_BANKS; bank++) {
		tr[ *this].m_accum[bank].resize(length, 0.0);
		tr[ *this].m_accum_weights[bank].resize(length, 0);
		tr[ *this].m_accum_dark[bank].resize(length, 0.0);
	}
	tr[ *this].m_wave.resize(length);
	std::fill(tr[ *this].m_wave.begin(), tr[ *this].m_wave.end(), std::complex<double>(0.0));
	tr[ *this].m_weights.resize(length);
	std::fill(tr[ *this].m_weights.begin(), tr[ *this].m_weights.end(), 0.0);
	tr[ *this].m_darkPSD.resize(length);
	std::fill(tr[ *this].m_darkPSD.begin(), tr[ *this].m_darkPSD.end(), 0.0);

	if(clear) {
		tr[ *m_spectrum].clearPoints();
		tr[ *this].m_peaks.clear();
		trans( *pulse__->avgClear()).touch();
		throw XSkippedRecordError(__FILE__, __LINE__);
	}

	if(emitter == pulse__.get()) {
		fssum(tr, shot_pulse, shot_others);
		m_isInstrumControlRequested = true;
	}
	else
		m_isInstrumControlRequested = false;
	
	analyzeIFT(tr, shot_pulse);
	std::vector<std::complex<double> > &wave(tr[ *this].m_wave);
	const std::vector<double> &weights(shot_this[ *this].weights());
	int wave_size = shot_this[ *this].wave().size();
	if(shot_this[ *autoPhase()]) {
		std::complex<double> csum(0.0, 0.0);
		for(unsigned int i = 0; i < wave_size; i++)
			csum += wave[i] * weights[i];
		double ph = 180.0 / M_PI * atan2(std::imag(csum), std::real(csum));
		if(fabs(ph) < 180.0)
			tr[ *phase()] = ph;
		tr.unmark(m_lsnOnCondChanged); //avoiding recursive signaling.
	}
	double ph = shot_this[ *phase()] / 180.0 * M_PI;
	std::complex<double> cph = std::polar(1.0, -ph);
	for(unsigned int i = 0; i < wave_size; i++)
		wave[i] *= cph;
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::visualize(const Snapshot &shot) {
	if( !shot[ *this].time()) {
		for(Transaction tr( *this);; ++tr) {
			tr[ *m_spectrum].clearPoints();
			tr[ *m_peakPlot->maxCount()] = 0;
			if(tr.commit())
				break;
		}
		return;
	}

	if(m_isInstrumControlRequested.compareAndSet((int)true, (int)false))
		rearrangeInstrum(shot);

	int length = shot[ *this].wave().size();
	std::vector<double> values;
	getValues(shot, values);
	assert(values.size() == length);
	for(Transaction tr( *m_spectrum);; ++tr) {
		double th = FFT::windowFuncHamming(0.1);
		const std::complex<double> *wave( &shot[ *this].wave()[0]);
		const double *weights( &shot[ *this].weights()[0]);
		const double *darkpsd( &shot[ *this].darkPSD()[0]);
		tr[ *m_spectrum].setRowCount(length);
		double *colx(tr[ *m_spectrum].cols(0));
		double *colr(tr[ *m_spectrum].cols(1));
		double *coli(tr[ *m_spectrum].cols(2));
		double *colw(tr[ *m_spectrum].cols(3));
		double *colabs(tr[ *m_spectrum].cols(4));
		double *coldark(tr[ *m_spectrum].cols(5));
		for(int i = 0; i < length; i++) {
			colx[i] = values[i];
			colr[i] = std::real(wave[i]);
			coli[i] = std::imag(wave[i]);
			colw[i] = (weights[i] > th) ? weights[i] : 0.0;
			colabs[i] = std::abs(wave[i]);
			coldark[i] = sqrt(darkpsd[i]);
		}
		const std::deque<std::pair<double, double> > &peaks(shot[ *this].m_peaks);
		int peaks_size = peaks.size();
		tr[ *m_peakPlot->maxCount()] = peaks_size;
		std::deque<XGraph::ValPoint> &points(tr[ *m_peakPlot].points());
		points.resize(peaks_size);
		for(int i = 0; i < peaks_size; i++) {
			double x = peaks[i].second;
			int j = lrint(x - 0.5);
			j = std::min(std::max(0, j), length - 2);
			double a = values[j] + (values[j + 1] - values[j]) * (x - j);
			points[i] = XGraph::ValPoint(a, peaks[i].first);
		}
		m_spectrum->drawGraph(tr);
		if(tr.commit()) {
			break;
		}
	}
}

template <class FRM>
void
XNMRSpectrumBase<FRM>::fssum(Transaction &tr, const Snapshot &shot_pulse, const Snapshot &shot_others) {
	const Snapshot &shot_this(tr);
	shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];

	int len = shot_pulse[ *pulse__].ftWidth();
	double df = shot_pulse[ *pulse__].dFreq();
	if((len == 0) || (df == 0)) {
		throw XRecordError(i18n("Invalid waveform."), __FILE__, __LINE__);
	}
	//bw *= 1.8; // for Hamming.
	//	bw *= 3.6; // for FlatTop.
	//bw *= 2.2; // for Kaiser3.
	int bw = abs(lrint(shot_this[ *bandWidth()] * 1000.0 / df * 2.2));
	if(bw >= len) {
		throw XRecordError(i18n("BW beyond Nyquist freq."), __FILE__, __LINE__);
	}
	double cfreq = getCurrentCenterFreq(shot_this, shot_others);
	if(cfreq == 0) {
		throw XRecordError(i18n("Invalid center freq."), __FILE__, __LINE__);
	}
	std::vector<std::complex<double> > ftwavein(len, 0.0), ftwaveout(len);
	if( !shot_this[ *this].m_preFFT || (shot_this[ *this].m_preFFT->length() != len)) {
		tr[ *this].m_preFFT.reset(new FFT(-1, len));
	}
	int wlen = std::min(len, (int)shot_pulse[ *pulse__].wave().size());
	int woff = -shot_pulse[ *pulse__].waveFTPos()
		+ len * ((shot_pulse[ *pulse__].waveFTPos() > 0) ? ((int)shot_pulse[ *pulse__].waveFTPos() / len + 1) : 0);
	const std::complex<double> *pulse_wave( &shot_pulse[ *pulse__].wave()[0]);
	for(int i = 0; i < wlen; i++) {
		int j = (i + woff) % len;
		ftwavein[j] = pulse_wave[i];
	}
	tr[ *this].m_preFFT->exec(ftwavein, ftwaveout);
	bw /= 2.0;
	double normalize = 1.0 / (double)shot_pulse[ *pulse__].wave().size();
	double darknormalize = shot_this[ *this].res() / df;
	for(int bank = 0; bank < Payload::ACCUM_BANKS; bank++) {
		double min = shot_this[ *this].min();
		double res = shot_this[ *this].res();
		int size = (int)shot_this[ *this].m_accum[bank].size();
		std::deque<std::complex<double> > &accum_wave(tr[ *this].m_accum[bank]);
		std::deque<double> &accum_weights(tr[ *this].m_accum_weights[bank]);
		std::deque<double> &accum_dark(tr[ *this].m_accum_dark[bank]);
		const double *pulse_dark( &shot_pulse[ *pulse__].darkPSD()[0]);
		for(int i = -bw / 2; i <= bw / 2; i++) {
			double freq = i * df;
			int idx = lrint((cfreq + freq - min) / res);
			if((idx >= size) || (idx < 0))
				continue;
			double w = FFT::windowFuncKaiser1((double)i / bw);
			int j = (i + len) % len;
			accum_wave[idx] += ftwaveout[j] * w * normalize;
			accum_weights[idx] += w;
			accum_dark[idx] += pulse_dark[j] * w * w * darknormalize;
		}
		bw *= 2.0;
	}
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::analyzeIFT(Transaction &tr, const Snapshot &shot_pulse) {
	const Snapshot &shot_this(tr);
	int bank = shot_this[ *bwList()];
	if((bank < 0) || (bank >= Payload::ACCUM_BANKS))
		throw XSkippedRecordError(__FILE__, __LINE__);
	double bw_coeff = 0.5 * pow(2.0, (double)bank);
	
	double th = FFT::windowFuncHamming(0.49);
	int max_idx = 0;
	const std::deque<std::complex<double> > &accum_wave(shot_this[ *this].m_accum[bank]);
	const std::deque<double> &accum_weights(shot_this[ *this].m_accum_weights[bank]);
	const std::deque<double> &accum_dark(shot_this[ *this].m_accum_dark[bank]);
	int accum_size = accum_wave.size();
	int min_idx = accum_size - 1;
	int taps_max = 0; 
	for(int i = 0; i < accum_size; i++) {
		if(accum_weights[i] > th) {
			min_idx = std::min(min_idx, i);
			max_idx = std::max(max_idx, i);
			taps_max++;
		}
	}
	if(max_idx <= min_idx)
		throw XSkippedRecordError(__FILE__, __LINE__);
	shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];
	double res = shot_this[ *this].res();
	int iftlen = max_idx - min_idx + 1;
	double wave_period = shot_pulse[ *pulse__].waveWidth() * shot_pulse[ *pulse__].interval();
	int npad = lrint(
		6.0 / (res * wave_period) + 0.5); //# of pads in frequency domain.
	//Truncation factor for IFFT.
	int trunc2 = lrint(pow(2.0, ceil(log(iftlen * 0.03) / log(2.0))));
	if(trunc2 < 1)
		throw XSkippedRecordError(__FILE__, __LINE__);
	iftlen = ((iftlen * 3 / 2 + npad) / trunc2 + 1) * trunc2;
	int tdsize = lrint(wave_period * res * iftlen);
	int iftorigin = lrint(shot_pulse[ *pulse__].waveFTPos() * shot_pulse[ *pulse__].interval() * res * iftlen);
	int bwinv = abs(lrint(1.0 / (shot_this[ *bandWidth()] * bw_coeff * 1000.0 * shot_pulse[ *pulse__].interval() * res * iftlen)));
	
	if( !shot_this[ *this].m_ift || (shot_this[ *this].m_ift->length() != iftlen)) {
		tr[ *this].m_ift.reset(new FFT(1, iftlen));
	}
	
	std::vector<std::complex<double> > fftwave(iftlen), iftwave(iftlen);
	std::fill(fftwave.begin(), fftwave.end(), std::complex<double>(0.0));
	for(int i = min_idx; i <= max_idx; i++) {
		int k = (i - (max_idx + min_idx) / 2 + iftlen) % iftlen;
		if(accum_weights[i] > th)
			fftwave[k] = accum_wave[i] / accum_weights[i];
	}
	tr[ *this].m_ift->exec(fftwave, iftwave);
	
	SpectrumSolver &solver(tr[ *m_solver].solver());
	std::vector<std::complex<double> > solverin;
	FFT::twindowfunc wndfunc = m_solver->windowFunc(shot_this);
	double wndwidth = shot_this[ *windowWidth()] / 100.0;
	double psdcoeff = 1.0;
	if(solver.isFT()) {
		std::vector<double> weight;
		SpectrumSolver::window(tdsize, -iftorigin, wndfunc, wndwidth, weight);
		double w = 0;
		for(int i = 0; i < tdsize; i++)
			w += weight[i] * weight[i];
		psdcoeff = w / (double)tdsize;
		//Compensate broadening due to convolution.
		solverin.resize(iftlen);
		double wlen = SpectrumSolver::windowLength(tdsize, -iftorigin, wndwidth);
		wlen += bwinv * 2; //effect of convolution.
		wndwidth = wlen / solverin.size();
		iftorigin = solverin.size() / 2;
	}
	else {
		solverin.resize(tdsize);
	}
	for(int i = 0; i < (int)solverin.size(); i++) {
		int k = (-iftorigin + i + iftlen) % iftlen;
		assert(k >= 0);
		solverin[i] = iftwave[k];
	}
	try {
		solver.exec(solverin, fftwave, -iftorigin, 0.1e-2, wndfunc, wndwidth);
	}
	catch (XKameError &e) {
		throw XSkippedRecordError(e.msg(), __FILE__, __LINE__);
	}

	std::vector<std::complex<double> > &wave(tr[ *this].m_wave);
	std::vector<double> &weights(tr[ *this].m_weights);
	std::vector<double> &darkpsd(tr[ *this].m_darkPSD);
	psdcoeff /= wave_period;
	for(int i = min_idx; i <= max_idx; i++) {
		int k = (i - (max_idx + min_idx) / 2 + iftlen) % iftlen;
		wave[i] = fftwave[k] / (double)iftlen;
		double w = accum_weights[i];
		weights[i] = w;
		darkpsd[i] = accum_dark[i] / (w * w) * psdcoeff;
	}
	th = FFT::windowFuncHamming(0.1);
	tr[ *this].m_peaks.clear();
	int weights_size = shot_this[ *this].weights().size();
	std::deque<std::pair<double, double> > &peaks(tr[ *this].m_peaks);
	for(int i = 0; i < solver.peaks().size(); i++) {
		double k = solver.peaks()[i].second;
		double j = (k > iftlen / 2) ? (k - iftlen) : k;
		j += (max_idx + min_idx) / 2;
		int l = lrint(j);
		if((l >= 0) && (l < weights_size) && (weights[l] > th))
			peaks.push_back(std::pair<double, double>(
				solver.peaks()[i].first / (double)iftlen, j));
	}
}
