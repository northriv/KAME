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

	const char *cand[] = {"3", "5", "11", "21", "51", "101", "201", "401", "801", "1601", "3201", ""};
	for(Transaction tr( *this);; ++tr) {
		for(const char **it = cand; strlen( *it); it++) {
			tr[ *points()].add( *it);
		}
		if(tr.commit())
			break;
	}
}
void
XNMRBuiltInNetworkAnalyzer::onStartFreqChanged(const Snapshot &shot, XValueNodeBase *) {

}
void
XNMRBuiltInNetworkAnalyzer::onStopFreqChanged(const Snapshot &shot, XValueNodeBase *) {
}
void
XNMRBuiltInNetworkAnalyzer::onAverageChanged(const Snapshot &shot, XValueNodeBase *) {

}
void
XNMRBuiltInNetworkAnalyzer::onPointsChanged(const Snapshot &shot, XValueNodeBase *) {

}
void
XNMRBuiltInNetworkAnalyzer::getMarkerPos(unsigned int num, double &x, double &y) {

}
void
XNMRBuiltInNetworkAnalyzer::oneSweep() {
	Snapshot shot_this( *this);
	shared_ptr<XPulser> pulse = shot_this[ *m_pulser];
	if( !pulse) return;
	for(Transaction tr( *pulse);; ++tr) {
		tr[ *pulse->pulseAnalyzerMode()] = true;
		tr[ *pulse->output()] = true;
		if(tr.commit()) {
			break;
		}
	}
	double fmin = shot_this[ *startFreq()];
	double fmax = shot_this[ *stopFreq()];

    shared_ptr<XSG> sg = shot_this[ *m_sg];
	if( !sg) return;
	trans( *sg->freq()) = fmin;

	shared_ptr<XDSO> dso = shot_this[ *m_dso];
	if( !dso) return;
		trans( *dso->restart()).touch(); //Restart averaging in DSO.

	m_ftsum.clear();
	m_ftsum_weight.clear();

	m_sweeping = true;
	while(m_sweeping) {
		msecsleep(10);
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
	Snapshot shot_this( *this);
	double fmin = shot_this[ *startFreq()];
	double fmax = shot_this[ *stopFreq()];
	int pts = atoi(shot_this[ *points()].to_str().c_str());

	tr[ *this].m_startFreq = fmin;
	tr[ *this].m_freqInterval = (fmax - fmin) / (pts - 1);
	tr[ *this].trace_().resize(pts);

	for(unsigned int i = 0; i < pts; i++) {
		tr[ *this].trace_()[i] = 20.0 * log10(std::abs(m_ftsum[i] / (double)m_ftsum_weight[i]));
	}
}
void
XNMRBuiltInNetworkAnalyzer::open() throw (XInterface::XInterfaceError &) {

}
bool
XNMRBuiltInNetworkAnalyzer::checkDependency(const Snapshot &shot_this,
	const Snapshot &shot_emitter, const Snapshot &shot_others,
	XDriver *emitter) const {
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
	shared_ptr<XPulser> pulse = shot_pulse[ *m_pulser];
	shared_ptr<XDSO> dso = shot_dso[ *m_dso];
    shared_ptr<XSG> sg = shot_sg[ *m_sg];

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
	double plsbw = shot_pulse[ *pulse].paPulseBW() * 1e-3; //[MHz]

	double fmin = shot_this[ *startFreq()];
	double fmax = shot_this[ *stopFreq()];
	double fstep = plsbw *0.7;
	int pts = atoi(shot_this[ *points()].to_str().c_str());
	fstep = std::min(fstep, (fmax - fmin) / (pts - 1));

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

	if(m_ftsum.size() != pts) {
		m_ftsum.resize(pts);
		m_ftsum_weight.resize(pts);
		std::fill(m_ftsum.begin(), m_ftsum.end(), 0.0);
		std::fill(m_ftsum_weight.begin(), m_ftsum_weight.end(), 0);
	}
	double freq = shot_sg[ *sg].freq();
	double normalize = 1.0 / avg_in_wave / fftlen;
	for(int i = 0; i < fftlen; ++i) {
		double f = fft_df * ((i >= fftlen / 2) ? i - (int)fftlen : i);
		if(abs(f) > plsbw / 2)
			continue;
		f += freq;
		int j = lrint((f - fmin) / (fmax - fmin) * (pts - 1));
		if((j < 0) || (j >= pts))
			continue;
		m_ftsum[j] += m_fftout[i] * normalize;
		++m_ftsum_weight[j];
	}

	unsigned int avg = shot_this[ *average()];

	if(freq < fmin - plsbw/2) {
		trans( *sg->freq()) = fmin;
		throw XSkippedRecordError;
	}
	if(freq + fstep / 2 > fmax) {
		m_sweeping = false;
		throw XSkippedRecordError;
	}
	freq += fstep;
	trans( *sg->freq()) = freq;
	throw XSkippedRecordError;
}
void
XNMRBuiltInNetworkAnalyzer::visualize(const Snapshot &shot) {
//	XNetworkAnalyzer::visualize(shot);
}

