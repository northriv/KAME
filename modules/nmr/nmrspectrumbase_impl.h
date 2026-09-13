/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
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
	m_solver(create<SpectrumSolverWrapper>("SpectrumSolverWrapper", true, m_solverList, m_windowFunc, m_windowWidth)), //was "SpectrumSolver", colliding with the m_solverList combo; matches the XNMRPulseAnalyzer convention.
    m_form(new FRM),
	m_statusPrinter(XStatusPrinter::create(m_form.get())),
    m_spectrum(create<XWaveNGraph>("Spectrum", false, m_form->m_graph, m_form->m_edDump, m_form->m_tbDump, m_form->m_btnDump,
            m_form->m_tlbMath, meas, static_pointer_cast<XDriver>(shared_from_this()))) {
    m_form->m_btnClear->setIcon(QApplication::style()->standardIcon(QStyle::SP_DialogResetButton));
    
	connect(pulse());

	iterate_commit([=](Transaction &tr){
		const char *labels[] = {"X", "Re [V]", "Im [V]", "Weights", "Abs [V]", "Dark [V]"};
		tr[ *m_spectrum].setColCount(6, labels);
        if( !tr[ *m_spectrum].insertPlot(tr, labels[4], 0, 4, -1, 3)) return;
        if( !tr[ *m_spectrum].insertPlot(tr, labels[1], 0, 1, -1, 3)) return;
        if( !tr[ *m_spectrum].insertPlot(tr, labels[2], 0, 2, -1, 3)) return;
        if( !tr[ *m_spectrum].insertPlot(tr, labels[5], 0, 5, -1, 3)) return;
		tr[ *tr[ *m_spectrum].axisy()->label()] = i18n("Intens. [V]");
		tr[ *tr[ *m_spectrum].plot(1)->label()] = i18n("real part");
        tr[ *tr[ *m_spectrum].plot(3)->lineColor()] = clLime; //QColor(0xa0, 0xa0, 0x00).rgb();
        tr[ *tr[ *m_spectrum].plot(2)->lineColor()] = (unsigned int)tr[ *tr[ *m_spectrum].plot(1)->lineColor()];
        tr[ *tr[ *m_spectrum].plot(1)->lineColor()] = (unsigned int)tr[ *tr[ *m_spectrum].plot(0)->lineColor()];
        tr[ *tr[ *m_spectrum].plot(0)->lineColor()] = clWhite;
        tr[ *tr[ *m_spectrum].plot(0)->barColor()] = QColor(0xf0, 0xf0, 0xc0).rgb();
//		tr[ *tr[ *m_spectrum].plot(0)->barColor()] = QColor(0x60, 0x60, 0xc0).rgb();
//		tr[ *tr[ *m_spectrum].plot(0)->lineColor()] = QColor(0x60, 0x60, 0xc0).rgb();
        tr[ *tr[ *m_spectrum].plot(1)->drawPoints()] = false;
		tr[ *tr[ *m_spectrum].plot(2)->label()] = i18n("imag. part");
		tr[ *tr[ *m_spectrum].plot(2)->drawPoints()] = false;
		tr[ *tr[ *m_spectrum].plot(0)->label()] = i18n("abs.");
		tr[ *tr[ *m_spectrum].plot(0)->drawPoints()] = false;
		tr[ *tr[ *m_spectrum].plot(0)->drawLines()] = true;
		tr[ *tr[ *m_spectrum].plot(0)->drawBars()] = true;
		tr[ *tr[ *m_spectrum].plot(0)->intensity()] = 0.5;
		tr[ *tr[ *m_spectrum].plot(3)->label()] = i18n("dark");
		tr[ *tr[ *m_spectrum].plot(3)->drawBars()] = false;
		tr[ *tr[ *m_spectrum].plot(3)->drawLines()] = true;
		tr[ *tr[ *m_spectrum].plot(3)->drawPoints()] = false;
		tr[ *tr[ *m_spectrum].plot(3)->intensity()] = 0.5;
		{
			shared_ptr<XXYPlot> plot = m_spectrum->graph()->plots()->template create<XXYPlot>(
				tr, "Peaks", true, tr, m_spectrum->graph());
            if( !plot) return;
			m_peakPlot = plot;
			tr[ *plot->label()] = i18n("Peaks");
			tr[ *plot->axisX()] = tr[ *m_spectrum].axisx();
			tr[ *plot->axisY()] = tr[ *m_spectrum].axisy();
			tr[ *plot->drawPoints()] = false;
			tr[ *plot->drawLines()] = false;
			tr[ *plot->drawBars()] = true;
            tr[ *plot->intensity()] = 0.3;
			tr[ *plot->displayMajorGrid()] = false;
            tr[ *plot->pointColor()] = clWhite; //QColor(0xa0, 0x00, 0xa0).rgb();
            tr[ *plot->barColor()] = clWhite; //QColor(0xa0, 0x00, 0xa0).rgb();
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
    });
  
    //Ranges should be preset in prior to connectors.
    m_form->m_dblPhase->setRange(-360.0, 360.0);
    m_form->m_dblPhase->setSingleStep(10.0);
    m_form->m_dblWindowWidth->setRange(0.1, 200.0);
    m_form->m_dblWindowWidth->setSingleStep(1.0);

    m_conBaseUIs = {
        xqcon_create<XQLineEditConnector>(m_bandWidth, m_form->m_edBW),
        xqcon_create<XQComboBoxConnector>(m_bwList, m_form->m_cmbBWList, Snapshot( *m_bwList)),
        xqcon_create<XQDoubleSpinBoxConnector>(m_phase, m_form->m_dblPhase, m_form->m_slPhase),
        xqcon_create<XQToggleButtonConnector>(m_autoPhase, m_form->m_ckbAutoPhase),
        xqcon_create<XQComboBoxConnector>(m_pulse, m_form->m_cmbPulse, ref(tr_meas)),
        xqcon_create<XQButtonConnector>(m_clear, m_form->m_btnClear),
        xqcon_create<XQComboBoxConnector>(m_solverList, m_form->m_cmbSolver, Snapshot( *m_solverList)),
        xqcon_create<XQDoubleSpinBoxConnector>(m_windowWidth, m_form->m_dblWindowWidth, m_form->m_slWindowWidth),
        xqcon_create<XQComboBoxConnector>(m_windowFunc, m_form->m_cmbWindowFunc, Snapshot( *m_windowFunc)),
    };

	iterate_commit([=](Transaction &tr){
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
    });
}
template <class FRM>
XNMRSpectrumBase<FRM>::~XNMRSpectrumBase() {
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
	XDriver *emitter) {
	const Snapshot &shot_this(tr);

	shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];
	assert( pulse__ );
	const Snapshot &shot_pulse((emitter == pulse__.get()) ? shot_emitter : shot_others);
 
	if(shot_pulse[ *pulse__->exAvgIncr()]) {
		m_statusPrinter->printWarning(i18n("Do NOT use incremental avg. Skipping."));
		throw XSkippedRecordError(__FILE__, __LINE__);
	}

    bool clear = (shot_this[ *this].m_timeClearRequested.isSet());
    tr[ *this].m_timeClearRequested = {};
  
//	double interval = shot_pulse[ *pulse__].interval();
	double df = shot_pulse[ *pulse__].dFreq();
	
	double res = getFreqResHint(shot_this);
	res = df * std::max(1L, lrint(res / df - 0.5));

	double max__ = getMaxFreq(shot_this);
	double min__ = getMinFreq(shot_this);
	
	if(max__ <= min__) {
		throw XSkippedRecordError(i18n("Invalid min. and max."), __FILE__, __LINE__);
	}
    constexpr ssize_t MAX_ACCUM_LEN = 65536 * 2;
    if(res * MAX_ACCUM_LEN < max__ - min__) {
        //restricts a size of accumulation buffers, due to memory consumption and friendliness.
        res = df * lrint((max__ - min__) / MAX_ACCUM_LEN / df - 0.5);
//		throw XSkippedRecordError(i18n("Too small resolution."), __FILE__, __LINE__);
    }
    //small change is discarded.
	if(fabs(log(shot_this[ *this].res() / res)) < log(2.0))
		res = shot_this[ *this].res();
	//The time bins of a relaxation map ride on this same axis, and are told
	//below whether it was rebuilt or merely shifted.
	bool axis_rebuilt = (shot_this[ *this].res() != res) || clear;
	int diff = 0;
	if(axis_rebuilt) {
		tr[ *this].m_res = res;
		for(int bank = 0; bank < Payload::ACCUM_BANKS; bank++) {
			tr[ *this].m_accum[bank].clear();
			tr[ *this].m_accum_weights[bank].clear();
			tr[ *this].m_accum_dark[bank].clear();
		}
	}
	else {
        //expands/shrinks the begining of buffers.
        diff = lrint(shot_this[ *this].min() / res) - lrint(min__ / res);
		for(int bank = 0; bank < Payload::ACCUM_BANKS; bank++) {
            auto &accum = tr[ *this].m_accum[bank];
            auto &accum_weights = tr[ *this].m_accum_weights[bank];
            auto &accum_dark = tr[ *this].m_accum_dark[bank];
            for(int i = 0; i < diff; i++) {
                accum.push_front(0.0);
                accum_weights.push_front(0);
                accum_dark.push_front(0.0);
			}
			for(int i = 0; i < -diff; i++) {
                if( !accum.empty()) {
                    accum.pop_front();
                    accum_weights.pop_front();
                    accum_dark.pop_front();
				}
			}
		}
	}
	tr[ *this].m_min = min__;
    //expands/shrinks the end of buffers.
    int length = lrint((max__ - min__) / res);
	for(int bank = 0; bank < Payload::ACCUM_BANKS; bank++) {
		tr[ *this].m_accum[bank].resize(length, 0.0);
		tr[ *this].m_accum_weights[bank].resize(length, 0);
		tr[ *this].m_accum_dark[bank].resize(length, 0.0);
	}
	updateMapBins(tr, shot_pulse, axis_rebuilt, diff, length);
	tr[ *this].m_wave.resize(length);
	std::fill(tr[ *this].m_wave.begin(), tr[ *this].m_wave.end(), std::complex<double>(0.0));
	tr[ *this].m_weights.resize(length);
	std::fill(tr[ *this].m_weights.begin(), tr[ *this].m_weights.end(), 0.0);
	tr[ *this].m_darkPSD.resize(length);
	std::fill(tr[ *this].m_darkPSD.begin(), tr[ *this].m_darkPSD.end(), 0.0);

	if(clear) {
		tr[ *m_spectrum].clearPoints();
		tr[ *this].m_peaks.clear();
		//NOT trans( *pulse__->avgClear()).touch() here: that commits immediately
		//and restarts the DSO sequence through onAvgClear/onRestartTouched, a
		//side effect this transaction cannot roll back.  analyze() re-runs on
		//every commit retry -- and the m_timeClearRequested reset above is rolled
		//back with it, so `clear` is true again -- which would restart the
		//hardware once per failed commit.  visualize() does it exactly once.
		m_isAvgClearRequested = true;
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
	//Deferred from analyze(); runs outside any transaction, so the DSO restart
	//this triggers may block on the interface mutex.  Must precede
	//rearrangeInstrum() below -- the average is cleared before the pulser/SG is
	//moved -- and precedes the !time() early return so a clear requested on the
	//first record is not held over to the next cycle.
	if(m_isAvgClearRequested.compare_set_strong((int)true, (int)false)) {
		shared_ptr<XNMRPulseAnalyzer> pulse__ = shot[ *pulse()];
		if(pulse__)
			trans( *pulse__->avgClear()).touch();
	}

	if( !shot[ *this].time()) {
		iterate_commit([=](Transaction &tr){
			tr[ *m_spectrum].clearPoints();
			tr[ *m_peakPlot->maxCount()] = 0;
        });
		return;
	}

    if(m_isInstrumControlRequested.compare_set_strong((int)true, (int)false))
		rearrangeInstrum(shot);

	int length = shot[ *this].wave().size();
	std::vector<double> values;
	getValues(shot, values);
	assert(values.size() == length);
    m_spectrum->iterate_commit([=](Transaction &tr){
		double th = FFT::windowFuncHamming(0.1);
		//data(), not &...[0]: visualize() is called even when analyze() threw
		//an XSkippedRecordError, and these Payload vectors are then still empty.
		//The size-guarded loop below never dereferences them, but forming the
		//pointer with operator[](0) on an empty vector is UB and aborts under a
		//hardened libstdc++ (the debug build enables _GLIBCXX_ASSERTIONS via Qt).
		const std::complex<double> *wave(shot[ *this].wave().data());
		const double *weights(shot[ *this].weights().data());
		const double *darkpsd(shot[ *this].darkPSD().data());
		tr[ *m_spectrum].setRowCount(length);
        std::vector<double> colx(length);
        std::vector<float> colr(length), coli(length), colw(length),
            colabs(length), coldark(length);
		for(int i = 0; i < length; i++) {
			colx[i] = values[i];
			colr[i] = std::real(wave[i]);
			coli[i] = std::imag(wave[i]);
			colw[i] = (weights[i] > th) ? weights[i] : 0.0;
			colabs[i] = std::abs(wave[i]);
			coldark[i] = sqrt(darkpsd[i]);
		}
        tr[ *m_spectrum].setColumn(0, std::move(colx), 9);
        tr[ *m_spectrum].setColumn(1, std::move(colr), 5);
        tr[ *m_spectrum].setColumn(2, std::move(coli), 5);
        tr[ *m_spectrum].setColumn(3, std::move(colw), 4);
        tr[ *m_spectrum].setColumn(4, std::move(colabs), 5);
        tr[ *m_spectrum].setColumn(5, std::move(coldark), 4);
        const auto &peaks(shot[ *this].m_peaks);
		int peaks_size = peaks.size();
		tr[ *m_peakPlot->maxCount()] = peaks_size;
        auto &points(tr[ *m_peakPlot].points());
		points.resize(peaks_size);
		for(int i = 0; i < peaks_size; i++) {
			double x = peaks[i].second;
			int j = lrint(x - 0.5);
			j = std::min(std::max(0, j), length - 2);
			double a = values[j] + (values[j + 1] - values[j]) * (x - j);
			points[i] = XGraph::ValPoint(a, peaks[i].first);
		}
		m_spectrum->drawGraph(tr);
    });
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
	int bw_org = bw; //before the banks halve and redouble it.
	double cfreq = getCurrentCenterFreq(shot_this, shot_others);
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
	//Time-resolved accumulation for a relaxation map, if the driver wants one.
	fssumTimeResolved(tr, shot_pulse, len, df, cfreq, bw_org);
}
template <class FRM>
const std::vector<std::complex<double> > &
XNMRSpectrumBase<FRM>::waveOfRecord(const Snapshot &shot_pulse,
	const XNMRPulseAnalyzer &pulse, int) const {
	return shot_pulse[pulse].wave();
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::updateMapBins(Transaction &tr, const Snapshot &shot_pulse,
	bool axis_rebuilt, int head_shift, int length) {
	const Snapshot &shot_this(tr);
	MapBinning binning;
	std::vector<std::vector<double> > times;
	if(mapBinning(shot_this, shot_pulse, binning) && (binning.binCount > 0)) {
		times.resize(binning.binCount);
		int nrec = (int)std::min(binning.binOfRecord.size(), binning.timeOfRecord.size());
		for(int r = 0; r < nrec; ++r) {
			int b = binning.binOfRecord[r];
			if((b >= 0) && (b < binning.binCount))
				times[b].push_back(binning.timeOfRecord[r]);
		}
	}
	//A copy of the pointers: the payload below is rewritten under our feet.
	auto bins = shot_this[ *this].m_mapBins;
	if(times.empty() && bins.empty())
		return; //no map at all, which is what this costs in the common case.
	//bwList() is display-time for the spectrum -- all three banks are summed in
	//parallel and one is read -- but the map sums only the bank in use, so
	//switching it changes the excitation weighting of everything that follows.
	//What is already in the bins cannot be re-weighted, so the bins go, and
	//only the bins: the spectrum's own banks are untouched and keep the sweep.
	int bank = shot_this[ *bwList()];
	bool rebuild = axis_rebuilt || (bins.size() != times.size())
		|| (shot_this[ *this].m_mapBank != bank);
	for(size_t b = 0; !rebuild && (b < bins.size()); ++b)
		rebuild = (bins[b]->times != times[b]);
	if(rebuild) {
		//The axis or the binning moved and what was accumulated cannot be
		//reinterpreted, so it goes.  Only the bins: turning a map on, or
		//regrouping the echoes, must not throw a whole sweep of the spectrum away.
		std::vector<shared_ptr<const typename Payload::MapBin> > fresh(times.size());
		for(size_t b = 0; b < times.size(); ++b) {
			auto bin = std::make_shared<typename Payload::MapBin>();
			bin->times = times[b];
			bin->accum.resize(length, 0.0);
			bin->accum_weights.resize(length, 0.0);
			bin->accum_dark.resize(length, 0.0);
			fresh[b] = bin; //filled before publishing (pointer-to-const).
		}
		tr[ *this].m_mapBins = std::move(fresh);
		tr[ *this].m_mapBank = bank;
		return;
	}
	if( !head_shift && !bins.empty() && ((int)bins[0]->accum.size() == length))
		return; //the axis did not move.
	for(size_t b = 0; b < bins.size(); ++b) {
		//Clone-on-write: a committed bin is shared with every live Snapshot.
		auto bin = std::make_shared<typename Payload::MapBin>( *bins[b]);
		for(int i = 0; i < head_shift; i++) {
			bin->accum.push_front(0.0);
			bin->accum_weights.push_front(0.0);
			bin->accum_dark.push_front(0.0);
		}
		for(int i = 0; i < -head_shift; i++) {
			if( !bin->accum.empty()) {
				bin->accum.pop_front();
				bin->accum_weights.pop_front();
				bin->accum_dark.pop_front();
			}
		}
		bin->accum.resize(length, 0.0);
		bin->accum_weights.resize(length, 0.0);
		bin->accum_dark.resize(length, 0.0);
		tr[ *this].m_mapBins[b] = bin;
	}
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::filterMapBins(Transaction &tr, int min_idx, int max_idx,
	int iftlen, int iftorigin, int tdsize) {
	const Snapshot &shot_this(tr);
	auto bins = shot_this[ *this].m_mapBins; //a copy of the pointers.
	if(bins.empty())
		return;
	FFT::twindowfunc wndfunc = mapWindowFunc(shot_this);
	if( !wndfunc || (tdsize < 2) || (iftlen < 4) ||
		!shot_this[ *this].m_ift || (shot_this[ *this].m_ift->length() != iftlen)) {
		//Nothing to lay down, so the accumulators are read as they stand.
		tr[ *this].m_mapWindowPSDCoeff = 1.0;
		if( !bins[0]->filtered.empty()) {
			for(size_t b = 0; b < bins.size(); ++b) {
				auto bin = std::make_shared<typename Payload::MapBin>( *bins[b]);
				bin->filtered.clear();
				tr[ *this].m_mapBins[b] = bin;
			}
		}
		return;
	}
	if( !shot_this[ *this].m_mapFFT || (shot_this[ *this].m_mapFFT->length() != iftlen))
		tr[ *this].m_mapFFT.reset(new FFT( -1, iftlen));

	//\a iftorigin is where the image has its origin, not where the spectrum's
	//solver moved it to: SpectrumSolver::window() measures the width from it,
	//and off by iftlen/2 the whole window falls outside its own support --
	//every sample zero as soon as the width leaves 100%, which is what a map
	//that stopped changing looked like (user).  The solver's convolution
	//compensation is deliberately not copied: the width asked for here is the
	//width laid down.
	std::vector<double> wnd;
	SpectrumSolver::window(tdsize, -iftorigin, wndfunc, mapWindowWidth(shot_this), wnd);
	double wsq = 0.0;
	for(int i = 0; i < tdsize; i++)
		wsq += wnd[i] * wnd[i];
	tr[ *this].m_mapWindowPSDCoeff = wsq / tdsize;

	double th = FFT::windowFuncHamming(0.49);
	int centre = (max_idx + min_idx) / 2;
	std::vector<std::complex<double> > in(iftlen), out(iftlen), td(iftlen);
	for(size_t b = 0; b < bins.size(); ++b) {
		const typename Payload::MapBin &bin( *bins[b]);
		if((int)bin.accum.size() <= max_idx)
			continue;
		//Back to the time domain on the spectrum's own grid, windowed there,
		//and forward again: one linear filter along the frequency axis, the
		//same one for every bin (\sa mapWindowFunc()).
		std::fill(in.begin(), in.end(), std::complex<double>(0.0));
		for(int i = min_idx; i <= max_idx; i++) {
			double w = bin.accum_weights[i];
			if(w > th)
				in[(i - centre + iftlen) % iftlen] = bin.accum[i] / w;
		}
		shot_this[ *this].m_ift->exec(in, td);
		std::fill(in.begin(), in.end(), std::complex<double>(0.0));
		for(int i = 0; i < tdsize; i++) {
			int k = ( -iftorigin + i + iftlen) % iftlen;
			in[k] = td[k] * wnd[i];
		}
		shot_this[ *this].m_mapFFT->exec(in, out);
		//Clone-on-write, as everywhere a committed bin is touched.
		auto fresh = std::make_shared<typename Payload::MapBin>(bin);
		fresh->filtered.assign(bin.accum.size(), std::complex<double>(0.0));
		for(int i = min_idx; i <= max_idx; i++)
			fresh->filtered[i] = out[(i - centre + iftlen) % iftlen] / (double)iftlen;
		tr[ *this].m_mapBins[b] = fresh;
	}
}
template <class FRM>
void
XNMRSpectrumBase<FRM>::fssumTimeResolved(Transaction &tr, const Snapshot &shot_pulse,
	int len, double df, double cfreq, int bw_org) {
	const Snapshot &shot_this(tr);
	auto bins = shot_this[ *this].m_mapBins; //a copy of the pointers, see above.
	if(bins.empty())
		return;
	MapBinning binning;
	if( !mapBinning(shot_this, shot_pulse, binning) || (binning.binCount != (int)bins.size()))
		return;
	shared_ptr<XNMRPulseAnalyzer> pulse__ = shot_this[ *pulse()];
	if( !pulse__ || !shot_this[ *this].m_preFFT)
		return;
	int bank = shot_this[ *bwList()];
	if((bank < 0) || (bank >= Payload::ACCUM_BANKS))
		return;
	//The map follows the bank the spectrum is displaying, so that the two weight
	//the excitation profile identically.
	int bw = bw_org / 2;
	for(int i = 0; i < bank; ++i)
		bw *= 2;
	if((bw < 2) || (bw >= len))
		return;
	double min = shot_this[ *this].min();
	double res = shot_this[ *this].res();
	double darknormalize = res / df;
	const std::vector<double> &pulse_dark(shot_pulse[ *pulse__].darkPSD());
	//One fresh copy per bin, published only once every record of this
	//acquisition has been summed: several records may share a bin, and a
	//committed bin is shared with live Snapshots.  A commit retry re-runs this
	//from the committed state, so the sum is applied exactly once.
	std::vector<shared_ptr<typename Payload::MapBin> > fresh(bins.size());
	for(size_t b = 0; b < bins.size(); ++b)
		fresh[b] = std::make_shared<typename Payload::MapBin>( *bins[b]);

	std::vector<std::complex<double> > ftwavein(len, 0.0), ftwaveout(len);
	int nrec = (int)binning.binOfRecord.size();
	for(int r = 0; r < nrec; ++r) {
		int b = binning.binOfRecord[r];
		if((b < 0) || (b >= (int)fresh.size()))
			continue;
		const std::vector<std::complex<double> > &wave(waveOfRecord(shot_pulse, *pulse__, r));
		if(wave.empty())
			continue;
		int wlen = std::min(len, (int)wave.size());
		int ftpos = shot_pulse[ *pulse__].waveFTPos();
		int woff = -ftpos + len * ((ftpos > 0) ? (ftpos / len + 1) : 0);
		std::fill(ftwavein.begin(), ftwavein.end(), std::complex<double>(0.0));
		for(int i = 0; i < wlen; i++)
			ftwavein[(i + woff) % len] = wave[i];
		shot_this[ *this].m_preFFT->exec(ftwavein, ftwaveout);
		double normalize = 1.0 / (double)wave.size();
		auto &bin( *fresh[b]);
		int size = (int)bin.accum.size();
		for(int i = -bw / 2; i <= bw / 2; i++) {
			double freq = i * df;
			int idx = lrint((cfreq + freq - min) / res);
			if((idx >= size) || (idx < 0))
				continue;
			double w = FFT::windowFuncKaiser1((double)i / bw);
			int j = (i + len) % len;
			bin.accum[idx] += ftwaveout[j] * w * normalize;
			bin.accum_weights[idx] += w;
			bin.accum_dark[idx] += pulse_dark[j] * w * w * darknormalize;
		}
		bin.avgCount++;
	}
	auto &dst(tr[ *this].m_mapBins);
	for(size_t b = 0; b < fresh.size(); ++b)
		dst[b] = fresh[b]; //publish (pointer-to-const).
	double wave_period = shot_pulse[ *pulse__].waveWidth() * shot_pulse[ *pulse__].interval();
	if(wave_period > 0.0)
		tr[ *this].m_mapPSDCoeff = mapNoiseFactor(shot_pulse, *pulse__) / wave_period;
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
	//Where the time-domain image actually has its origin.  The solver's branch
	//below moves iftorigin to the middle of the input it builds for itself,
	//and everything after that means the solver's frame, not this one.
	const int iftorigin_td = iftorigin;
	int bwinv = abs(lrint(1.0 / (shot_this[ *bandWidth()] * bw_coeff * 1000.0 * shot_pulse[ *pulse__].interval() * res * iftlen)));
	
	if( !shot_this[ *this].m_ift || (shot_this[ *this].m_ift->length() != iftlen)) {
		tr[ *this].m_ift.reset(new FFT(1, iftlen));
	}
	
	std::vector<std::complex<double> > fftwave(iftlen), iftwave(iftlen);
	std::fill(fftwave.begin(), fftwave.end(), std::complex<double>(0.0));
	for(int i = min_idx; i <= max_idx; i++) {
        int k = (i - (max_idx + min_idx) / 2 + iftlen) % iftlen;
        assert(k >= 0);
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
        assert(k >= 0);
		wave[i] = fftwave[k] / (double)iftlen;
		double w = accum_weights[i];
		weights[i] = w;
		darkpsd[i] = accum_dark[i] / (w * w) * psdcoeff;
	}
	//The map's bins ride on the same axis and the same geometry; they are
	//filtered here rather than at draw time so that a window can be changed
	//without a sweep being thrown away.
	filterMapBins(tr, min_idx, max_idx, iftlen, iftorigin_td, tdsize);

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
            peaks.emplace_back(solver.peaks()[i].first / (double)iftlen, j);
	}
}
