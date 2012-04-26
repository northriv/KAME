/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
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
#include "pulseanalyzer.h"
#include "analyzer.h"

REGISTER_TYPE(XDriverList, NMRBuiltInNetworkAnalyzer, "NMR Built-In Network Analyzer");

XNMRBuiltInNetworkAnalyzer::XNMRBuiltInNetworkAnalyzer(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
		XSecondaryDriverInterface<XNetworkAnalyzer>(name, runtime, ref(tr_meas), meas),
		m_pulser(create<XItemNode<XDriverList, XPulser> >("Pulser", false, ref(tr_meas), meas->drivers(), true)),
		m_dso(create<XItemNode<XDriverList, XDSO> >("DSO", false, ref(tr_meas), meas->drivers(), true)),
		m_sg(create<XItemNode<XDriverList, XSG> >("SG", false, ref(tr_meas), meas->drivers(), true)) {
	connect(m_dso);
	connect(m_pulser);
	connect(m_sg);

	const char *cand[] = {"OFF", "11", "21", "51", "101", "201", "401", "801", "1601", "3201", ""};
	for(Transaction tr( *this);; ++tr) {
		for(const char **it = cand; strlen( *it); it++) {
			tr[ *points()].add( *it);
		}
		tr[ *this].m_sweeping = false;
		if(tr.commit())
			break;
	}
	XNetworkAnalyzer::start();
}
void
XNMRBuiltInNetworkAnalyzer::clear() {
	restart(CAL_NONE, true);
}
void
XNMRBuiltInNetworkAnalyzer::onCalOpenTouched(const Snapshot &shot, XTouchableNode *) {
	restart(CAL_OPEN);
}
void
XNMRBuiltInNetworkAnalyzer::onCalShortTouched(const Snapshot &shot, XTouchableNode *) {
	restart(CAL_SHORT);
}
void
XNMRBuiltInNetworkAnalyzer::onCalTermTouched(const Snapshot &shot, XTouchableNode *) {
	restart(CAL_TERM);
}
void
XNMRBuiltInNetworkAnalyzer::onCalThruTouched(const Snapshot &shot, XTouchableNode *) {
	restart(CAL_THRU);
}
void
XNMRBuiltInNetworkAnalyzer::onStartFreqChanged(const Snapshot &shot, XValueNodeBase *) {
	clear();
}
void
XNMRBuiltInNetworkAnalyzer::onStopFreqChanged(const Snapshot &shot, XValueNodeBase *) {
	clear();
}
void
XNMRBuiltInNetworkAnalyzer::onAverageChanged(const Snapshot &shot, XValueNodeBase *) {
	clear();
}
void
XNMRBuiltInNetworkAnalyzer::onPointsChanged(const Snapshot &shot, XValueNodeBase *) {
	clear();
	int pts = atoi(Snapshot( *this)[ *points()].to_str().c_str());
	if( !pts){
		try {
			startContSweep();
		}
		catch (XInterface::XInterfaceError &e) {
			gErrPrint(e.msg());
		}
//		stop();
	}
}
void
XNMRBuiltInNetworkAnalyzer::getMarkerPos(unsigned int num, double &x, double &y) {
	Snapshot shot( *this);
	switch(num) {
	case 0:
		x = shot[ *this].m_marker_min.first;
		y = shot[ *this].m_marker_min.second;
		break;
	case 1:
		x = shot[ *this].m_marker_max.first;
		y = shot[ *this].m_marker_max.second;
		break;
	default:
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	}
}
void
XNMRBuiltInNetworkAnalyzer::oneSweep() {
	bool ret = restart(CAL_NONE);
	if( !ret)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	while(Snapshot( *this)[ *this].m_sweeping) {
		msecsleep(30);
	}
}
bool
XNMRBuiltInNetworkAnalyzer::restart(int calmode, bool clear) {
	bool ret = false;
	for(Transaction tr( *this);; ++tr) {
		try {
			ret = false;
			restart(tr, calmode, clear);
			ret = true;
		}
		catch (XDriver::XSkippedRecordError &) {
		}
		catch (XInterface::XInterfaceError &e) {
			gErrPrint(e.msg());
		}
		if(tr.commit()) {
			break;
		}
	}
	return ret;
}
void
XNMRBuiltInNetworkAnalyzer::restart(Transaction &tr, int calmode, bool clear) {
	Snapshot &shot_this(tr);

	tr[ *this].m_sweeping = true;

	int pts = atoi(shot_this[ *points()].to_str().c_str());
	tr[ *this].m_sweepPoints = pts;
	if( !pts)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);

	tr[ *this].m_ftsum.clear();
	tr[ *this].m_ftsum_weight.clear();
	tr[ *this].m_calMode = calmode;
	if(clear || tr[ *this].m_raw_open.empty()) {
		tr[ *this].m_raw_open.clear();
		tr[ *this].m_raw_short.clear();
		tr[ *this].m_raw_term.clear();
		tr[ *this].m_raw_thru.clear();
		tr[ *this].m_raw_open.resize(pts, 1.0);
		tr[ *this].m_raw_short.resize(pts, -1.0);
		tr[ *this].m_raw_term.resize(pts, 0.0);
		tr[ *this].m_raw_thru.resize(pts, 1.0);
	}

	shared_ptr<XPulser> pulse = shot_this[ *m_pulser];
	if( !pulse)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	shared_ptr<XSG> sg = shot_this[ *m_sg];
	if( !sg)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	shared_ptr<XDSO> dso = shot_this[ *m_dso];
	if( !dso)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);

	Snapshot shot_dso( *dso);
	double interval;
	int dso_len;
	if( !shot_dso[ *dso].time() || !shot_dso[ *dso].numChannels()) {
		interval = 1e-6; //temporary
		dso_len = 10000; //temporary
	}
	else {
		interval = shot_dso[ *dso].timeInterval();
		dso_len = shot_dso[ *dso].length();
	}

	double fmax = shot_this[ *stopFreq()];
	tr[ *this].m_sweepStop = fmax;
	double fmin = shot_this[ *startFreq()];
	tr[ *this].m_sweepStart = fmin;
	if((fmax <= fmin) || (fmin <= 0.1))
		throw XDriver::XSkippedRecordError(i18n("Invalid frequency settings"), __FILE__, __LINE__);

	for(Transaction trp( *pulse);; ++trp) {
		double plsbw = trp[ *pulse].paPulseBW() * 1e-3; //[MHz]
		double fstep = plsbw *1.0;
		fstep = std::max(fstep, (fmax - fmin) / (pts - 1));
		if(fstep < 0.001)
			throw XDriver::XSkippedRecordError(i18n("Invalid frequency settings"), __FILE__, __LINE__);
		tr[ *this].m_sweepStep = fstep;
		trp[ *pulse->pulseAnalyzerMode()] = true;
		double rept_ms = std::max(2.0 / ((fmax - fmin) / (pts - 1)) * 1e-3 * 2, 0.2);
		rept_ms = interval * 1e3 * lrint(rept_ms / (interval * 1e3)); //round to DSO interval.
		trp[ *pulse->paPulseRept()] = rept_ms;
		trp[ *pulse->output()] = true;
		if(trp.commit()) {
			break;
		}
	}

	int avg = std::max(1L, lrint(0.03 / (interval * dso_len)));
	avg *= std::min(1u, (unsigned int)shot_this[ *average()]);
	trans( *dso->average()) = (avg + 3) / 4 * 4; //round to phase cycling for NMR.

	trans( *sg->freq()) = fmin;

	for(Transaction trd( *dso);; ++trd) {
		trd[ *dso->firEnabled()] = false;
		trd[ *dso->restart()].touch(); //Restart averaging in DSO.
		if(trd.commit()) {
			break;
		}
	}
}
void
XNMRBuiltInNetworkAnalyzer::startContSweep() {
	Snapshot shot_this( *this);
	shared_ptr<XPulser> pulse = shot_this[ *m_pulser];
	if(pulse) {
		for(Transaction tr( *pulse);; ++tr) {
			tr[ *pulse->pulseAnalyzerMode()] = false;
			tr[ *pulse->output()] = false;
			if(tr.commit()) {
				break;
			}
		}
	}
    shared_ptr<XSG> sg = shot_this[ *m_sg];
    if(sg) {
    	trans( *sg->freq()) = (double)shot_this[ *this].m_marker_min.first;
    }
}
void
XNMRBuiltInNetworkAnalyzer::acquireTrace(shared_ptr<RawData> &, unsigned int ch) {

}
void
XNMRBuiltInNetworkAnalyzer::convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
}
void
XNMRBuiltInNetworkAnalyzer::writeTraceAndMarkers(Transaction &tr) {
	Snapshot &shot_this(tr);
	double fmin = shot_this[ *this].m_sweepStart;
	double fmax = shot_this[ *this].m_sweepStop;
	int pts = shot_this[ *this].m_sweepPoints;
//	if( !pts)
//		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);

	tr[ *this].m_startFreq = fmin;
	double df = (fmax - fmin) / (pts - 1);
	tr[ *this].m_freqInterval = df;
	tr[ *this].trace_().resize(pts);

	auto ftsum = &tr[ *this].m_ftsum[0];
	auto ftsum_weight = &tr[ *this].m_ftsum_weight[0];
	for(unsigned int i = 0; i < pts; i++) {
		ftsum[i] /= ftsum_weight[i];
	}

	//Stores calibration curves.
	switch(tr[ *this].m_calMode) {
	case CAL_NONE:
		break;
	case CAL_OPEN:
		tr[ *this].m_raw_open.resize(pts);
		std::copy(tr[ *this].m_ftsum.begin(), tr[ *this].m_ftsum.end(), tr[ *this].m_raw_open.begin());
		break;
	case CAL_SHORT:
		tr[ *this].m_raw_short.resize(pts);
		std::copy(tr[ *this].m_ftsum.begin(), tr[ *this].m_ftsum.end(), tr[ *this].m_raw_short.begin());
		break;
	case CAL_TERM:
		tr[ *this].m_raw_term.resize(pts);
		std::copy(tr[ *this].m_ftsum.begin(), tr[ *this].m_ftsum.end(), tr[ *this].m_raw_term.begin());
		break;
	case CAL_THRU:
		tr[ *this].m_raw_thru.resize(pts);
		std::copy(tr[ *this].m_ftsum.begin(), tr[ *this].m_ftsum.end(), tr[ *this].m_raw_thru.begin());
		break;
	}

	auto rawopen = &tr[ *this].m_raw_open[0];
	auto rawshort = &tr[ *this].m_raw_short[0];
	auto rawterm = &tr[ *this].m_raw_term[0];
	auto trace = &tr[ *this].trace_()[0];
	for(unsigned int i = 0; i < pts; i++) {
		auto zport_in = - rawopen[i] / rawshort[i]; //Impedance of the port connected to LNA.
		auto s11_dut = 1.0 - 2.0 / ((1.0 + zport_in) / (1.0 - (ftsum[i] - rawterm[i]) / rawopen[i]) + 1.0 - zport_in);
		trace[i] = 20.0 * log10(std::abs(s11_dut));
	}

	//Tracking markers.
	auto &mkmin = tr[ *this].m_marker_min;
	mkmin.second = 1000;
	auto &mkmax = tr[ *this].m_marker_max;
	mkmax.second = -1000;
	for(unsigned int i = 0; i < pts; i++) {
		if(trace[i] < mkmin.second)
			mkmin = std::pair<double, double>(i * df + fmin, trace[i]);
		if(trace[i] > mkmax.second)
			mkmax = std::pair<double, double>(i * df + fmin, trace[i]);
	}
}
void
XNMRBuiltInNetworkAnalyzer::open() throw (XInterface::XInterfaceError &) {

}
bool
XNMRBuiltInNetworkAnalyzer::checkDependency(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
	if( !shot_this[ *this].m_sweeping)
		return false;
	shared_ptr<XPulser> pulse = shot_this[ *m_pulser];
	if( !pulse) return false;
	shared_ptr<XDSO> dso = shot_this[ *m_dso];
	if( !dso) return false;
    shared_ptr<XSG> sg = shot_this[ *m_sg];
    if( !sg) return false;
	if (emitter != dso.get())
		return false;
//    if(shot_emitter[ *pulse].timeAwared() < shot_others[ *sg].time()) return false;
	return true;
}
void
XNMRBuiltInNetworkAnalyzer::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) throw (XRecordError&) {
	const Snapshot &shot_this(tr);
	int pts = shot_this[ *this].m_sweepPoints;
	if( !pts)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);

	const Snapshot &shot_dso(shot_emitter);
	const Snapshot &shot_sg(shot_others);
	const Snapshot &shot_pulse(shot_others);
	shared_ptr<XPulser> pulse = shot_this[ *m_pulser];
	shared_ptr<XDSO> dso = shot_this[ *m_dso];
    shared_ptr<XSG> sg = shot_this[ *m_sg];

	assert(shot_dso[ *dso].time() );

	if(shot_dso[ *dso].numChannels() < 1) {
		throw XSkippedRecordError(i18n("No record in DSO"), __FILE__, __LINE__);
	}
	if(shot_dso[ *dso].numChannels() < 2) {
		throw XSkippedRecordError(i18n("Two channels needed in DSO"), __FILE__, __LINE__);
	}
	if( !shot_dso[ *dso->singleSequence()]) {
		g_statusPrinter->printWarning(i18n("Use sequential average in DSO."));
	}
	int dso_len = shot_dso[ *dso].length();

	double interval = shot_dso[ *dso].timeInterval();
	if (interval <= 0) {
		throw XSkippedRecordError(i18n("Invalid time interval in waveforms."), __FILE__, __LINE__);
	}
	int pos = lrint(shot_dso[ *dso].trigPos());
	if(pos >= dso_len) {
		throw XSkippedRecordError(i18n("Position beyond waveforms."), __FILE__, __LINE__);
	}
	if(pos < 0) {
		throw XSkippedRecordError(i18n("Position beyond waveforms."), __FILE__, __LINE__);
	}

	if(pulse) {
		if( !shot_pulse[ *pulse].isPulseAnalyzerMode())
			throw XSkippedRecordError(i18n("Pulser configured not in Built-In Network Analyzer Mode."), __FILE__, __LINE__);
	}
	double rept = shot_pulse[ *pulse].rtime() * 1e-3; //[s]
	double plsorg = shot_pulse[ *pulse].paPulseOrigin() * 1e-6; //[s]

	double fmin = shot_this[ *this].m_sweepStart;
	double fmax = shot_this[ *this].m_sweepStop;

	unsigned int fftlen = (unsigned int)floor(std::max(plsorg / interval, 1e-6 / ((fmax - fmin) / (pts - 1)) / interval * 2));
	fftlen = FFT::fitLength(std::min(fftlen, (unsigned int)floor(rept / interval)));
	if( !m_fft || m_fft->length() != fftlen) {
		m_fft.reset(new FFT(-1, fftlen));
		m_fftin.resize(fftlen);
		m_fftout.resize(fftlen);
	}
	std::fill(m_fftin.begin(), m_fftin.end(), 0.0);
	unsigned int avg_in_wave = floor((dso_len - pos) / (rept / interval));
	if(avg_in_wave < 2)
		throw XSkippedRecordError(i18n("Too short waveforms."), __FILE__, __LINE__);
	avg_in_wave = avg_in_wave / 2 * 2;

	for(unsigned int av = 0; av < avg_in_wave; ++av) {
		int lpos = lrint(pos + (rept * av) / interval);
		int org = lrint(pos + (plsorg + rept * av) / interval);
		bool inverted = (av % 2 == 1);
		const double *wavecos = &shot_dso[ *dso].wave(0)[lpos];
		const double *wavesin = &shot_dso[ *dso].wave(1)[lpos];
		for(unsigned int i = fftlen - (org - lpos); i < fftlen; ++i) {
			m_fftin[i] += std::complex<double>( *wavecos++,  *wavesin++) * (inverted ? -1.0 : 1.0);
		}
		for(unsigned int i = 0; i < fftlen - (org - lpos); ++i) {
			m_fftin[i] += std::complex<double>( *wavecos++,  *wavesin++) * (inverted ? -1.0 : 1.0);
		}
	}
	double fft_df = 1.0 / (interval * fftlen) * 1e-6; //[MHz]
	m_fft->exec(m_fftin, m_fftout);

	double plsbw = shot_pulse[ *pulse].paPulseBW() * 1e-3; //[MHz]

	if(tr[ *this].m_ftsum.size() != pts) {
		tr[ *this].m_ftsum.resize(pts);
		tr[ *this].m_ftsum_weight.resize(pts);
		std::fill(tr[ *this].m_ftsum.begin(), tr[ *this].m_ftsum.end(), 0.0);
		std::fill(tr[ *this].m_ftsum_weight.begin(), tr[ *this].m_ftsum_weight.end(), 0);
	}
	double freq = shot_sg[ *sg].freq();
	tr[ *this].m_lastCenterFreq = freq;
	if(freq < fmin - plsbw/2) {
		restart(tr, shot_this[ *this].m_calMode);
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	}
	double normalize = 1.0 / avg_in_wave / fftlen;
	auto ftsum = &tr[ *this].m_ftsum[0];
	auto ftsum_weight = &tr[ *this].m_ftsum_weight[0];
	for(int i = 0; i < fftlen; ++i) {
		double f = fft_df * ((i >= fftlen / 2) ? i - (int)fftlen : i);
		if(abs(f) > plsbw / 2)
			continue;
		f += freq;
		int j = lrint((f - fmin) / (fmax - fmin) * (pts - 1));
		if((j < 0) || (j >= pts))
			continue;
		ftsum[j] += m_fftout[i] * normalize;
		++ftsum_weight[j];
	}
	if( !shot_this[ *this].m_ftsum.size()) {
		restart(tr, shot_this[ *this].m_calMode);
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	}
	double fstep = shot_this[ *this].m_sweepStep;
	if(freq + fstep / 2 > fmax) {
		writeTraceAndMarkers(tr);
		tr[ *this].m_sweeping = false;
	}
	throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
}
void
XNMRBuiltInNetworkAnalyzer::visualize(const Snapshot &shot) {
	if(shot[ *this].m_sweeping) {
		double freq = shot[ *this].m_lastCenterFreq;
		double fstep = shot[ *this].m_sweepStep;

		freq += fstep;
	    shared_ptr<XSG> sg = shot[ *m_sg];
	    assert(sg);
		trans( *sg->freq()) = freq; //setting new freq.
	}
	else
		XNetworkAnalyzer::visualize(shot);
}

