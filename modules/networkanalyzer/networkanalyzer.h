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
//---------------------------------------------------------------------------

#ifndef networkanalyzerH
#define networkanalyzerH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "xnodeconnector.h"

class XScalarEntry;
class FrmNetworkAnalyzer;
class XWaveNGraph;
class FrmGraphNURL;

//! Base class for digital storage oscilloscope.
class XNetworkAnalyzer : public XPrimaryDriver
{
	XNODE_OBJECT
protected:
	XNetworkAnalyzer(const char *name, bool runtime,
		 const shared_ptr<XScalarEntryList> &scalarentries,
		 const shared_ptr<XInterfaceList> &interfaces,
		 const shared_ptr<XThermometerList> &thermometers,
		 const shared_ptr<XDriverList> &drivers);
public:
	//! usually nothing to do.
	virtual ~XNetworkAnalyzer() {}
	//! show all forms belonging to driver.
	virtual void showForms();
protected:
	//! Start up your threads, connect GUI, and activate signals
	virtual void start();
	//! Shut down your threads, unconnect GUI, and deactivate signals
	//! this may be called even if driver has already stopped.
	virtual void stop();
  
	//! this is called when raw is written 
	//! unless dependency is broken
	//! convert raw to record
	virtual void analyzeRaw() throw (XRecordError&);
	//! this is called after analyze() or analyzeRaw()
	//! record is readLocked
	virtual void visualize();
  
	//! driver specific part below
	const shared_ptr<XScalarEntry> &marker1X() const {return m_marker1X;}
	const shared_ptr<XScalarEntry> &marker1Y() const {return m_marker1Y;}
	const shared_ptr<XScalarEntry> &marker2X() const {return m_marker2X;}
	const shared_ptr<XScalarEntry> &marker2Y() const {return m_marker2Y;}
	const shared_ptr<XDoubleNode> &startFreq() const {return m_startFreq;}
	const shared_ptr<XDoubleNode> &stopFreq() const {return m_stopFreq;}

	//! Records below
	double startFreqRecorded() const {return m_startFreqRecorded;} //! [MHz]
	double freqIntervalRecorded() const {return m_freqIntervalRecorded;} //! [MHz]
	unsigned int lengthRecorded() const {return m_traceRecorded.size();}
	const double *traceRecorded() const;
protected:
	virtual void onStartFreqChanged(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onStopFreqChanged(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void getMarkerPos(unsigned int num, double &x, double &y) = 0;
	virtual void oneSweep() = 0;
	virtual void acquireTrace(unsigned int ch) = 0;
	//! convert raw to dispaly-able
	virtual void convertRaw() throw (XRecordError&) = 0;
	
	std::vector<double> m_traceRecorded;	
	double m_startFreqRecorded;
	double m_freqIntervalRecorded;
private:
	const shared_ptr<XWaveNGraph> &waveForm() const {return m_waveForm;}
	const shared_ptr<XScalarEntry> m_marker1X;
	const shared_ptr<XScalarEntry> m_marker1Y;
	const shared_ptr<XScalarEntry> m_marker2X;
	const shared_ptr<XScalarEntry> m_marker2Y;
	const shared_ptr<XDoubleNode> m_startFreq;
	const shared_ptr<XDoubleNode> m_stopFreq;

	const qshared_ptr<FrmNetworkAnalyzer> m_form;
	const shared_ptr<XWaveNGraph> m_waveForm;

	shared_ptr<XListener> m_lsnOnStartFreqChanged;
	shared_ptr<XListener> m_lsnOnStopFreqChanged;
  
	const xqcon_ptr m_conStartFreq, m_conStopFreq;
 
	shared_ptr<XThread<XNetworkAnalyzer> > m_thread;
  
	void *execute(const atomic<bool> &);
};

//---------------------------------------------------------------------------

#endif
