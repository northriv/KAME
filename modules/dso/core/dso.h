/***************************************************************************
		Copyright (C) 2002-2009 Kentaro Kitagawa
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

#ifndef dsoH
#define dsoH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "xnodeconnector.h"

class XScalarEntry;
class FIR;

class QMainWindow;
class Ui_FrmDSO;
typedef QForm<QMainWindow, Ui_FrmDSO> FrmDSO;
#include "xwavengraph.h"

//! Base class for digital storage oscilloscope.
class XDSO : public XPrimaryDriver
{
	XNODE_OBJECT
protected:
	XDSO(const char *name, bool runtime,
		 const shared_ptr<XScalarEntryList> &scalarentries,
		 const shared_ptr<XInterfaceList> &interfaces,
		 const shared_ptr<XThermometerList> &thermometers,
		 const shared_ptr<XDriverList> &drivers);
public:
	//! usually nothing to do.
	virtual ~XDSO() {}
	//! Shows all forms belonging to driver.
	virtual void showForms();
protected:
	//! Start up your threads, connect GUI, and activate signals
	virtual void start();
	//! Shut down your threads, unconnect GUI, and deactivate signals
	//! this may be called even if driver has already stopped.
	virtual void stop();
  
	//! This is called after analyze() or analyzeRaw()
	//! The record will be read-locked.
	//! This might be called even if the record is broken (time() == false).
	virtual void visualize();
  
	//! driver specific part below
public:
	const shared_ptr<XUIntNode> &average() const {return m_average;}
	//! If true, an acquisition will be paused with a averaging count.
	const shared_ptr<XBoolNode> &singleSequence() const {return m_singleSequence;}
	const shared_ptr<XComboNode> &trigSource() const {return m_trigSource;}
	const shared_ptr<XDoubleNode> &trigPos() const {return m_trigPos;}
	const shared_ptr<XDoubleNode> &trigLevel() const {return m_trigLevel;}
	const shared_ptr<XBoolNode> &trigFalling() const {return m_trigFalling;}
	const shared_ptr<XDoubleNode> &timeWidth() const {return m_timeWidth;}
	const shared_ptr<XComboNode> &vFullScale1() const {return m_vFullScale1;}
	const shared_ptr<XComboNode> &vFullScale2() const {return m_vFullScale2;}
	const shared_ptr<XComboNode> &vFullScale3() const {return m_vFullScale3;}
	const shared_ptr<XComboNode> &vFullScale4() const {return m_vFullScale4;}
	const shared_ptr<XDoubleNode> &vOffset1() const {return m_vOffset1;}
	const shared_ptr<XDoubleNode> &vOffset2() const {return m_vOffset2;}
	const shared_ptr<XDoubleNode> &vOffset3() const {return m_vOffset3;}
	const shared_ptr<XDoubleNode> &vOffset4() const {return m_vOffset4;}
	const shared_ptr<XUIntNode> &recordLength() const {return m_recordLength;}
	const shared_ptr<XNode> &forceTrigger() const {return m_forceTrigger;}  
	const shared_ptr<XNode> &restart() const {return m_restart;}

	const shared_ptr<XComboNode> &trace1() const {return m_trace1;}
	const shared_ptr<XComboNode> &trace2() const {return m_trace2;}
	const shared_ptr<XComboNode> &trace3() const {return m_trace3;}
	const shared_ptr<XComboNode> &trace4() const {return m_trace4;}
  
	const shared_ptr<XComboNode> &fetchMode() const {return m_fetchMode;}
  
	const shared_ptr<XBoolNode> &firEnabled() const {return m_firEnabled;}
	const shared_ptr<XDoubleNode> &firBandWidth() const {return m_firBandWidth;} ///< [kHz]
	const shared_ptr<XDoubleNode> &firCenterFreq() const {return m_firCenterFreq;} ///< [kHz]
	const shared_ptr<XDoubleNode> &firSharpness() const {return m_firSharpness;}

	//! Records below
	double trigPosRecorded() const {return m_trigPosRecorded;} ///< unit is interval
	unsigned int numChannelsRecorded() const {return m_numChannelsRecorded;}
	double timeIntervalRecorded() const {return m_timeIntervalRecorded;} //! [sec]
	unsigned int lengthRecorded() const;
	const double *waveRecorded(unsigned int ch) const;
  
protected:
	virtual void onTrace1Changed(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onTrace2Changed(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onTrace3Changed(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onTrace4Changed(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onAverageChanged(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onSingleChanged(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onTrigSourceChanged(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onTrigPosChanged(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onTrigLevelChanged(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onTrigFallingChanged(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onTimeWidthChanged(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onVFullScale1Changed(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onVFullScale2Changed(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onVFullScale3Changed(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onVFullScale4Changed(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onVOffset1Changed(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onVOffset2Changed(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onVOffset3Changed(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onVOffset4Changed(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onRecordLengthChanged(const shared_ptr<XValueNodeBase> &) = 0;
	virtual void onForceTriggerTouched(const shared_ptr<XNode> &) = 0;
	virtual void onRestartTouched(const shared_ptr<XNode> &) {
		startSequence();
	}

	virtual double getTimeInterval() = 0;

	//! Clears the count or starts a sequence measurement
	virtual void startSequence() = 0;

	//! \arg seq_busy true if the sequence is not finished.
	virtual int acqCount(bool *seq_busy) = 0;

	//! Loads the waveform and settings from the instrument.
	virtual void getWave(std::deque<XString> &channels) = 0;
	//! Converts the raw to a display-able style.
	virtual void convertRaw() throw (XRecordError&) = 0;
	void setParameters(unsigned int channels, double startpos, double interval, unsigned int length);
	//! For displaying.
	unsigned int lengthDisp() const;
	double *waveDisp(unsigned int ch);
	double trigPosDisp() const {return m_trigPosDisp;} ///< unit is interval
	unsigned int numChannelsDisp() const {return m_numChannelsDisp;}
	double timeIntervalDisp() const {return m_timeIntervalDisp;} //! [sec]
  
	//! This is called when the raw data is written.
	//! unless dependency is broken.
	//! convert raw to record
	virtual void analyzeRaw() throw (XRecordError&);
  
	const shared_ptr<XStatusPrinter> &statusPrinter() const {return m_statusPrinter;}
private:
	enum {FETCHMODE_NEVER = 0, FETCHMODE_AVG = 1, FETCHMODE_SEQ = 2};
 
	const shared_ptr<XWaveNGraph> &waveForm() const {return m_waveForm;}
  
	const shared_ptr<XUIntNode> m_average;
	//! If true, pause acquision after averaging count
	const shared_ptr<XBoolNode> m_singleSequence;
	const shared_ptr<XComboNode> m_trigSource;
	const shared_ptr<XBoolNode> m_trigFalling;
	const shared_ptr<XDoubleNode> m_trigPos;
	const shared_ptr<XDoubleNode> m_trigLevel;
	const shared_ptr<XDoubleNode> m_timeWidth;
	const shared_ptr<XComboNode> m_vFullScale1;
	const shared_ptr<XComboNode> m_vFullScale2;
	const shared_ptr<XComboNode> m_vFullScale3;
	const shared_ptr<XComboNode> m_vFullScale4;
	const shared_ptr<XDoubleNode> m_vOffset1;
	const shared_ptr<XDoubleNode> m_vOffset2;
	const shared_ptr<XDoubleNode> m_vOffset3;
	const shared_ptr<XDoubleNode> m_vOffset4;
	const shared_ptr<XUIntNode> m_recordLength;
	const shared_ptr<XNode> m_forceTrigger;  
	const shared_ptr<XNode> m_restart;
	const shared_ptr<XComboNode> m_trace1;
	const shared_ptr<XComboNode> m_trace2;
	const shared_ptr<XComboNode> m_trace3;
	const shared_ptr<XComboNode> m_trace4;
	const shared_ptr<XComboNode> m_fetchMode;
	const shared_ptr<XBoolNode> m_firEnabled;
	const shared_ptr<XDoubleNode> m_firBandWidth; ///< [kHz]
	const shared_ptr<XDoubleNode> m_firCenterFreq; ///< [kHz]
	const shared_ptr<XDoubleNode> m_firSharpness;

	const qshared_ptr<FrmDSO> m_form;
	const shared_ptr<XWaveNGraph> m_waveForm;
  
	//! These are parts of a record.
	double m_trigPosRecorded; ///< unit is interval
	unsigned int m_numChannelsRecorded;
	double m_timeIntervalRecorded; //! [sec]
	std::vector<double> m_wavesRecorded;
	//! for displaying.
	bool m_rawDisplayOnly;
	double m_trigPosDisp; ///< unit is interval
	unsigned int m_numChannelsDisp;
	double m_timeIntervalDisp; //! [sec]
	std::vector<double> m_wavesDisp;
	XRecursiveMutex m_dispMutex;
	//! Convert the raw to a display-able style and performs extra digital processing.
	void convertRawToDisp() throw (XRecordError&);
  
	shared_ptr<XListener> m_lsnOnSingleChanged;
	shared_ptr<XListener> m_lsnOnAverageChanged;
	shared_ptr<XListener> m_lsnOnTrigSourceChanged;
	shared_ptr<XListener> m_lsnOnTrigPosChanged;
	shared_ptr<XListener> m_lsnOnTrigLevelChanged;
	shared_ptr<XListener> m_lsnOnTrigFallingChanged;
	shared_ptr<XListener> m_lsnOnTimeWidthChanged;
	shared_ptr<XListener> m_lsnOnTrace1Changed;
	shared_ptr<XListener> m_lsnOnTrace2Changed;
	shared_ptr<XListener> m_lsnOnTrace3Changed;
	shared_ptr<XListener> m_lsnOnTrace4Changed;
	shared_ptr<XListener> m_lsnOnVFullScale1Changed;
	shared_ptr<XListener> m_lsnOnVFullScale2Changed;
	shared_ptr<XListener> m_lsnOnVFullScale3Changed;
	shared_ptr<XListener> m_lsnOnVFullScale4Changed;
	shared_ptr<XListener> m_lsnOnVOffset1Changed;
	shared_ptr<XListener> m_lsnOnVOffset2Changed;
	shared_ptr<XListener> m_lsnOnVOffset3Changed;
	shared_ptr<XListener> m_lsnOnVOffset4Changed;
	shared_ptr<XListener> m_lsnOnRecordLengthChanged;
	shared_ptr<XListener> m_lsnOnForceTriggerTouched;
	shared_ptr<XListener> m_lsnOnRestartTouched;
	shared_ptr<XListener> m_lsnOnCondChanged;
  
	void onCondChanged(const shared_ptr<XValueNodeBase> &);
  
	const xqcon_ptr m_conAverage, m_conSingle,
		m_conTrace1, m_conTrace2, m_conTrace3, m_conTrace4;
	const xqcon_ptr m_conFetchMode, m_conTimeWidth,
		m_conVFullScale1, m_conVFullScale2, m_conVFullScale3, m_conVFullScale4;
	const xqcon_ptr m_conTrigSource, m_conTrigPos, m_conTrigLevel, m_conTrigFalling;
	const xqcon_ptr m_conVOffset1, m_conVOffset2, m_conVOffset3, m_conVOffset4,
		m_conForceTrigger, m_conRecordLength;
	const xqcon_ptr m_conFIREnabled, m_conFIRBandWidth, m_conFIRSharpness, m_conFIRCenterFreq;
 
	shared_ptr<XThread<XDSO> > m_thread;
	const shared_ptr<XStatusPrinter> m_statusPrinter;

	shared_ptr<FIR> m_fir;
  
	void *execute(const atomic<bool> &);
  
	static const char *s_trace_names[];
	static const unsigned int s_trace_colors[];
};

//---------------------------------------------------------------------------

#endif
