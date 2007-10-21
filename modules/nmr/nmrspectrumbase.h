/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
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

class XNMRPulseAnalyzer;
class XWaveNGraph;

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
	~XNMRSpectrumBase() {}
  
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

	//! records below.
	const std::deque<std::complex<double> > &wave() const {return m_wave;}
	//! averaged count
	const std::deque<int> &counts() const {return m_counts;}
	//! resolution
	double resRecorded() const {return m_resRecorded;}
	//! value of the first point
	double minRecorded() const {return m_minRecorded;}
protected:
	//! Records
	std::deque<int> m_counts;
	std::deque<std::complex<double> > m_wave;

	shared_ptr<XListener> m_lsnOnClear, m_lsnOnCondChanged;
    
	//! \return true to be cleared.
	virtual bool onCondChangedImpl(const shared_ptr<XValueNodeBase> &) const = 0;
	//! Fourier Step Summation.
	virtual void fssum() = 0;
	virtual double getResolution() const = 0;
	virtual double getMinValue() const = 0;
	virtual double getMaxValue() const = 0;
	virtual bool checkDependencyImpl(const shared_ptr<XDriver> &emitter) const = 0;

private:
	const shared_ptr<XItemNode<XDriverList, XNMRPulseAnalyzer> > m_pulse;
 
	const shared_ptr<XDoubleNode> m_bandWidth;
	const shared_ptr<XBoolNode> m_autoPhase;
	const shared_ptr<XDoubleNode> m_phase;
	const shared_ptr<XNode> m_clear;
	
	double m_resRecorded, m_minRecorded;
  
	xqcon_ptr m_conBandWidth;
	xqcon_ptr m_conPulse;
	xqcon_ptr m_conPhase, m_conAutoPhase;
	xqcon_ptr m_conClear;

	void onCondChanged(const shared_ptr<XValueNodeBase> &);

	XTime m_timeClearRequested;
protected:
	const qshared_ptr<FRM> m_form;
	const shared_ptr<XWaveNGraph> m_spectrum;
	void onClear(const shared_ptr<XNode> &);
};

#endif
