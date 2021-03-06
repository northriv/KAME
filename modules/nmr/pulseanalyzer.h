/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef PULSEANALYZER_H_
#define PULSEANALYZER_H_

#include "networkanalyzer.h"
#include "pulserdriver.h"
#include "dso.h"
#include "signalgenerator.h"
#include "secondarydriverinterface.h"

//! Built-in network analyzer using a directional coupler in coherent pulse NMR system.
class XNMRBuiltInNetworkAnalyzer : public XSecondaryDriverInterface<XNetworkAnalyzer> {
public:
	XNMRBuiltInNetworkAnalyzer(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XNMRBuiltInNetworkAnalyzer() {}

	struct Payload : public XSecondaryDriverInterface<XNetworkAnalyzer>::Payload {
	private:
		friend class XNMRBuiltInNetworkAnalyzer;
		std::vector<std::complex<double> > m_ftsum, m_raw_open, m_raw_short, m_raw_term, m_raw_thru;
		std::vector<int> m_ftsum_weight;
		int m_calMode;
		std::pair<double, double> m_marker_min, m_marker_max;
		double m_sweepStep;
		double m_sweepStart;
		double m_sweepStop;
		int m_sweepPoints;
		bool m_sweeping;
		double m_lastCenterFreq;
	};
	virtual void showForms() {XNetworkAnalyzer::showForms();}

	virtual void onStartFreqChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onStopFreqChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onPointsChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onCalOpenTouched(const Snapshot &shot, XTouchableNode *);
	virtual void onCalShortTouched(const Snapshot &shot, XTouchableNode *);
	virtual void onCalTermTouched(const Snapshot &shot, XTouchableNode *);
	virtual void onCalThruTouched(const Snapshot &shot, XTouchableNode *);
	virtual void getMarkerPos(unsigned int num, double &x, double &y);
	virtual void oneSweep();
	virtual void startContSweep();
	virtual void acquireTrace(shared_ptr<RawData> &, unsigned int ch);
	//! Converts raw to dispaly-able
	virtual void convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);

	//! Be called just after opening interface. Call start() inside this routine appropriately.
	virtual void open() throw (XKameError &);
	//! Be called for closing interfaces.
	//! This function should not cause an exception.
	virtual void closeInterface() {}

protected:
	//! This function is called when a connected driver emit a signal
	virtual void analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
		XDriver *emitter) throw (XRecordError&);
	//! This function is called inside analyze() or analyzeRaw()
	//! this must be reentrant unlike analyze()
	virtual void visualize(const Snapshot &shot);
	//! Checks if the connected drivers have valid time stamps.
	//! \return true if dependency is resolved.
	//! This function must be reentrant unlike analyze().
	virtual bool checkDependency(const Snapshot &shot_this,
		const Snapshot &shot_emitter, const Snapshot &shot_others,
		XDriver *emitter) const;

private:
	const shared_ptr<XItemNode<XDriverList, XPulser> > m_pulser;
	const shared_ptr<XItemNode<XDriverList, XDSO> > m_dso;
	const shared_ptr<XItemNode<XDriverList, XSG> > m_sg;

	//for FFT.
	shared_ptr<FFT> m_fft;
	std::vector<std::complex<double> > m_fftin, m_fftout;

	void clear();
	void restart(Transaction &tr, int calmode, bool clear = false);
	bool restart(int calmode, bool clear = false);
	enum {CAL_NONE = 0, CAL_OPEN = 1, CAL_SHORT = 2, CAL_TERM = 3, CAL_THRU = 4};

	void writeTraceAndMarkers(Transaction &tr);
};

#endif /* PULSEANALYZER_H_ */
