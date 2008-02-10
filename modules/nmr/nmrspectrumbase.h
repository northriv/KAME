/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef nmrspectrumbaseH
#define nmrspectrumbaseH
//---------------------------------------------------------------------------
#include <secondarydriver.h>
#include <xnodeconnector.h>
#include <complex>
#include "nmrspectrumsolver.h"

class XNMRPulseAnalyzer;
class XWaveNGraph;
class XXYPlot;

template <class FRM>
class XNMRSpectrumBase : public XSecondaryDriver
{
	XNODE_OBJECT
protected:
	XNMRSpectrumBase(const char *name, bool runtime,
				 const shared_ptr<XScalarEntryList> &scalarentries,
				 const shared_ptr<XInterfaceList> &interfaces,
				 const shared_ptr<XThermometerList> &thermometers,
				 const shared_ptr<XDriverList> &drivers);
public:
	//! ususally nothing to do
	virtual ~XNMRSpectrumBase();
  
	//! show all forms belonging to driver
	virtual void showForms();
protected:

	//! this is called when connected driver emit a signal
	//! unless dependency is broken
	//! all connected drivers are readLocked
	virtual void analyze(const shared_ptr<XDriver> &emitter) throw (XRecordError&);
	//! this is called after analyze() or analyzeRaw()
	//! record is readLocked
	virtual void visualize();
	//! check connected drivers have valid time
	//! \return true if dependency is resolved
	virtual bool checkDependency(const shared_ptr<XDriver> &emitter) const;
public:
	//! driver specific part below 
	const shared_ptr<XItemNode<XDriverList, XNMRPulseAnalyzer> > &pulse() const {return m_pulse;}

	const shared_ptr<XDoubleNode> &bandWidth() const {return m_bandWidth;}
	//! Deduce phase from data
	const shared_ptr<XBoolNode> &autoPhase() const {return m_autoPhase;}
	//! (Deduced) phase of echoes [deg.]
	const shared_ptr<XDoubleNode> &phase() const {return m_phase;}
	//! Spectrum solvers.
	const shared_ptr<XComboNode> &solverList() const {return m_solverList;}
	/// FFT Window Function
	const shared_ptr<XComboNode> &windowFunc() const {return m_windowFunc;}
	//! Changing width of time-domain image [%]
	const shared_ptr<XDoubleNode> &windowWidth() const {return m_windowWidth;}
	

	//! records below.
	const std::deque<std::complex<double> > &wave() const {return m_wave;}
	//! averaged weights
	const std::deque<double> &weights() const {return m_weights;}
	//! resolution [Hz]
	double resRecorded() const {return m_resRecorded;}
	//! value of the first point [Hz]
	double minRecorded() const {return m_minRecorded;}
protected:
	//! Records
	std::deque<double> m_weights;
	std::deque<std::complex<double> > m_accum;

	shared_ptr<XListener> m_lsnOnClear, m_lsnOnCondChanged;
    
	//! \return true to be cleared.
	virtual bool onCondChangedImpl(const shared_ptr<XValueNodeBase> &) const = 0;
	//! [Hz]
	virtual double getFreqResHint() const = 0;
	//! [Hz]
	virtual double getMinFreq() const = 0;
	//! [Hz]
	virtual double getMaxFreq() const = 0;
	//! [Hz]
	virtual double getCurrentCenterFreq() const = 0;
	virtual void afterFSSum() {}
	virtual void getValues(std::vector<double> &values) const = 0;
	virtual bool checkDependencyImpl(const shared_ptr<XDriver> &emitter) const = 0;
private:
	//! Fourier Step Summation.
	void fssum();

	//! Records
	std::deque<std::complex<double> > m_wave;

	const shared_ptr<XItemNode<XDriverList, XNMRPulseAnalyzer> > m_pulse;
 
	const shared_ptr<XDoubleNode> m_bandWidth;
	const shared_ptr<XBoolNode> m_autoPhase;
	const shared_ptr<XDoubleNode> m_phase;
	const shared_ptr<XNode> m_clear;
	const shared_ptr<XComboNode> m_solverList;
	const shared_ptr<XComboNode> m_windowFunc;
	const shared_ptr<XDoubleNode> m_windowWidth;
	
	double m_resRecorded, m_minRecorded;
  
	xqcon_ptr m_conBandWidth;
	xqcon_ptr m_conPulse;
	xqcon_ptr m_conPhase, m_conAutoPhase;
	xqcon_ptr m_conClear, m_conSolverList, m_conWindowWidth, m_conWindowFunc;

	shared_ptr<FFT> m_ift;
	shared_ptr<SpectrumSolverWrapper> m_solver;
	shared_ptr<XXYPlot> m_peakPlot;
	std::vector<std::pair<double, double> > m_peaks;
	FFTSolver m_preFFT;

	void analyzeIFT();
	
	void onCondChanged(const shared_ptr<XValueNodeBase> &);

	XTime m_timeClearRequested;
protected:
	const qshared_ptr<FRM> m_form;
	const shared_ptr<XStatusPrinter> m_statusPrinter;
	const shared_ptr<XWaveNGraph> m_spectrum;
	void onClear(const shared_ptr<XNode> &);
};

#endif
