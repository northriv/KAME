/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef networkanalyzerH
#define networkanalyzerH
//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"
#include <complex>

class XScalarEntry;
class QMainWindow;
class Ui_FrmNetworkAnalyzer;
typedef QForm<QMainWindow, Ui_FrmNetworkAnalyzer> FrmNetworkAnalyzer;

class XGraph;
class XWaveNGraph;
class XXYPlot;

//! Base class for digital storage oscilloscope.
class DECLSPEC_SHARED XNetworkAnalyzer : public XPrimaryDriverWithThread {
public:
	XNetworkAnalyzer(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do.
	virtual ~XNetworkAnalyzer() {}
	//! Shows all forms belonging to driver.
	virtual void showForms();

	struct Payload : public XPrimaryDriver::Payload {
		double startFreq() const {return m_startFreq;} //! [MHz]
		double freqInterval() const {return m_freqInterval;} //! [MHz]
		unsigned int length() const {return m_trace.size();}
		const std::complex<double> *trace() const {return &m_trace.at(0);}

		std::vector<std::complex<double> > &trace_() {return m_trace;}
		double m_startFreq;
		double m_freqInterval;
	private:
		friend class XNetworkAnalyzer;
		std::vector<std::complex<double> > m_trace;
		std::deque<std::pair<double, double> > m_markers;
	};
protected:
	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
	virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);
  
	//! driver specific part below
	const shared_ptr<XScalarEntry> &marker1X() const {return m_marker1X;}
	const shared_ptr<XScalarEntry> &marker1Y() const {return m_marker1Y;}
	const shared_ptr<XScalarEntry> &marker2X() const {return m_marker2X;}
	const shared_ptr<XScalarEntry> &marker2Y() const {return m_marker2Y;}
	const shared_ptr<XDoubleNode> &startFreq() const {return m_startFreq;}
	const shared_ptr<XDoubleNode> &stopFreq() const {return m_stopFreq;}
	const shared_ptr<XComboNode> &points() const {return m_points;}
	const shared_ptr<XUIntNode> &average() const {return m_average;}
	const shared_ptr<XTouchableNode> &calOpen() const {return m_calOpen;}
	const shared_ptr<XTouchableNode> &calShort() const {return m_calShort;}
	const shared_ptr<XTouchableNode> &calTerm() const {return m_calTerm;}
	const shared_ptr<XTouchableNode> &calThru() const {return m_calThru;}
protected:
	virtual void onStartFreqChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onStopFreqChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onPointsChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onCalOpenTouched(const Snapshot &shot, XTouchableNode *) = 0;
	virtual void onCalShortTouched(const Snapshot &shot, XTouchableNode *) = 0;
	virtual void onCalTermTouched(const Snapshot &shot, XTouchableNode *) = 0;
	virtual void onCalThruTouched(const Snapshot &shot, XTouchableNode *) = 0;
	virtual void getMarkerPos(unsigned int num, double &x, double &y) = 0;
	virtual void oneSweep() = 0;
	virtual void startContSweep() = 0;
	virtual void acquireTrace(shared_ptr<RawData> &, unsigned int ch) = 0;
	//! Converts raw to dispaly-able
	virtual void convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) = 0;
private:
	const shared_ptr<XWaveNGraph> &waveForm() const {return m_waveForm;}
	const shared_ptr<XScalarEntry> m_marker1X;
	const shared_ptr<XScalarEntry> m_marker1Y;
	const shared_ptr<XScalarEntry> m_marker2X;
	const shared_ptr<XScalarEntry> m_marker2Y;
	const shared_ptr<XDoubleNode> m_startFreq;
	const shared_ptr<XDoubleNode> m_stopFreq;
	const shared_ptr<XComboNode> m_points;
	const shared_ptr<XUIntNode> m_average;
	const shared_ptr<XTouchableNode> m_calOpen, m_calShort, m_calTerm, m_calThru;

	const qshared_ptr<FrmNetworkAnalyzer> m_form;
	const shared_ptr<XWaveNGraph> m_waveForm;

	shared_ptr<XListener> m_lsnOnStartFreqChanged;
	shared_ptr<XListener> m_lsnOnStopFreqChanged;
	shared_ptr<XListener> m_lsnOnPointsChanged;
	shared_ptr<XListener> m_lsnOnAverageChanged;
	shared_ptr<XListener> m_lsnCalOpen, m_lsnCalShort, m_lsnCalTerm, m_lsnCalThru;
  
	xqcon_ptr m_conStartFreq, m_conStopFreq, m_conPoints, m_conAverage,
		m_conCalOpen, m_conCalShort, m_conCalTerm, m_conCalThru;
 
	shared_ptr<XGraph> m_graph;
	shared_ptr<XXYPlot> m_markerPlot;
	
	void *execute(const atomic<bool> &);
};

//---------------------------------------------------------------------------

#endif
