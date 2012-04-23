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
	int pts = atoi(shot[ *points()].to_str().c_str());
	if( !pts) {
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
	restart(CAL_NONE);

	m_sweeping = true;
	while(m_sweeping) {
		msecsleep(10);
	}
}
void
XNMRBuiltInNetworkAnalyzer::restart(int calmode, bool clear) {
	for(Transaction tr( *this);; ++tr) {
		try {
			restart(tr, calmode, clear);
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
}
void
XNMRBuiltInNetworkAnalyzer::restart(Transaction &tr, int calmode, bool clear) {
	Snapshot &shot_this(tr);
	shared_ptr<XPulser> pulse = shot_this[ *m_pulser];
	if( !pulse)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	shared_ptr<XSG> sg = shot_this[ *m_sg];
	if( !sg)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	shared_ptr<XDSO> dso = shot_this[ *m_dso];
	if( !dso)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);

	int pts = atoi(shot_this[ *points()].to_str().c_str());
	if( !pts)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	tr[ *this].m_sweepPoints = pts;

	double fmax = shot_this[ *stopFreq()];
	tr[ *this].m_sweepStop = fmax;
	double fmin = shot_this[ *startFreq()];
	tr[ *this].m_sweepStart = fmin;
	if((fmax <= fmin) || (fmin <= 0.1))
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);

	for(Transaction tr( *pulse);; ++tr) {
		double plsbw = tr[ *pulse].paPulseBW() * 1e-3; //[MHz]
		double fstep = plsbw *1.0;
		fstep = std::max(fstep, (fmax - fmin) / (pts - 1));
		tr[ *this].m_sweepStep = fstep;
		tr[ *pulse->pulseAnalyzerMode()] = true;
		tr[ *pulse->paPulseRept()] = std::max(1.0 / ((fmax - fmin) / (pts - 1)) * 1e-3 * 2, 0.5);
		tr[ *pulse->output()] = true;
		if(tr.commit()) {
			break;
		}
	}

	trans( *sg->freq()) = fmin;

	trans( *dso->restart()).touch(); //Restart averaging in DSO.

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
    	trans( *sg->freq()) = (double)shot_this[ *marker1X()->value()];
    }
}
void
XNMRBuiltInNetworkAnalyzer::acquireTrace(shared_ptr<RawData> &, unsigned int ch) {

}
void
XNMRBuiltInNetworkAnalyzer::convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
	Snapshot &shot_this(tr);
	double fmin = shot_this[ *this].m_sweepStart;
	double fmax = shot_this[ *this].m_sweepStop;
	int pts = shot_this[ *this].m_sweepPoints;
	if( !pts)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);

	tr[ *this].m_startFreq = fmin;
	double df = (fmax - fmin) / (pts - 1);
	tr[ *this].m_freqInterval = df;
	tr[ *this].trace_().resize(pts);

	auto ftsum = &tr[ *this].m_ftsum[0];
	auto ftsum_weight = &tr[ *this].m_ftsum_weight[0];
	auto rawopen = &tr[ *this].m_raw_open[0];
	auto rawterm = &tr[ *this].m_raw_term[0];
	auto trace = &tr[ *this].trace_()[0];
	for(unsigned int i = 0; i < pts; i++) {
		ftsum[i] /= ftsum_weight[i];
		trace[i] = 20.0 * log10(std::abs((ftsum[i] - rawterm[i])/ (rawopen[i] -  rawterm[i])));
	}

	//Tracking markers.
	auto &mkmin = tr[ *this].m_marker_min;
	mkmin.second = 1000;
	auto &mkmax = tr[ *this].m_marker_min;
	mkmax.second = -1000;
	for(unsigned int i = 0; i < pts; i++) {
		if(trace[i] < mkmin.second)
			mkmin = std::pair<double, double>(i * df + fmin, trace[i]);
		if(trace[i] > mkmax.second)
			mkmax = std::pair<double, double>(i * df + fmin, trace[i]);
	}

	//Preserves calibration curves.
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
}
void
XNMRBuiltInNetworkAnalyzer::open() throw (XInterface::XInterfaceError &) {

}
bool
XNMRBuiltInNetworkAnalyzer::checkDependency(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
	if( !m_sweeping)
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
	double fstep = shot_this[ *this].m_sweepStep;
	if((fmax <= fmin) || (fmin <= 0.1) || (fstep < 0.001))
		throw XDriver::XSkippedRecordError(i18n("Invalid frequency settings"), __FILE__, __LINE__);
	int pts = shot_this[ *this].m_sweepPoints;
	if( !pts)
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);

	unsigned int fftlen = (unsigned int)floor(std::max(plsorg / interval * 2, 1e-6 / ((fmax - fmin) / (pts - 1)) / interval * 2));
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

//	unsigned int avg = shot_this[ *average()];

	if(freq < fmin - plsbw/2) {
		restart(tr, shot_this[ *this].m_calMode);
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	}
	if(freq + fstep / 2 > fmax) {
		m_sweeping = false;
		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
	}
	freq += fstep;
	trans( *sg->freq()) = freq;
	if( !shot_this[ *this].m_ftsum.size())
		restart(tr, shot_this[ *this].m_calMode);
	throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
}
void
XNMRBuiltInNetworkAnalyzer::visualize(const Snapshot &shot) {
	XNetworkAnalyzer::visualize(shot);
}

