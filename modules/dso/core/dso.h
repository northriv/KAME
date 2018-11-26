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
//---------------------------------------------------------------------------

#ifndef dsoH
#define dsoH
//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"
#include <complex>

class XScalarEntry;
class FIR;
class XSG;

class QMainWindow;
class Ui_FrmDSO;
typedef QForm<QMainWindow, Ui_FrmDSO> FrmDSO;
#include "xwavengraph.h"

//! Base class for digital storage oscilloscope.
class DECLSPEC_SHARED XDSO : public XPrimaryDriverWithThread {
public:
	XDSO(const char *name, bool runtime,
		 Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do.
	virtual ~XDSO() {}
	//! Shows all forms belonging to driver.
	virtual void showForms();
protected:
	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
	virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);
  
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
	const shared_ptr<XTouchableNode> &forceTrigger() const {return m_forceTrigger;}
	const shared_ptr<XTouchableNode> &restart() const {return m_restart;}

	const shared_ptr<XComboNode> &trace1() const {return m_trace1;}
	const shared_ptr<XComboNode> &trace2() const {return m_trace2;}
	const shared_ptr<XComboNode> &trace3() const {return m_trace3;}
	const shared_ptr<XComboNode> &trace4() const {return m_trace4;}
  
	const shared_ptr<XComboNode> &fetchMode() const {return m_fetchMode;}
  
	const shared_ptr<XBoolNode> &firEnabled() const {return m_firEnabled;}
	const shared_ptr<XDoubleNode> &firBandWidth() const {return m_firBandWidth;} ///< [kHz]
	const shared_ptr<XDoubleNode> &firCenterFreq() const {return m_firCenterFreq;} ///< [kHz]
	const shared_ptr<XDoubleNode> &firSharpness() const {return m_firSharpness;}

	enum DRFMODE {DRFMODE_OFF = 0, DRFMODE_GIVEN_FREQ = 1, DRFMODE_FREQ_BY_SG = 2, DRFMODE_COHERENT_SG = 3};
	const shared_ptr<XComboNode> &dRFMode() const {return m_dRFMode;}
	const shared_ptr<XItemNode<XDriverList, XSG> > &dRFSG() const {return m_dRFSG;}
	const shared_ptr<XDoubleNode> &dRFFreq() const {return m_dRFFreq;}

    struct DECLSPEC_SHARED Payload : public XPrimaryDriver::Payload {
		Payload() : m_rawDisplayOnly(false), m_numChannelsDisp(0) {}
		double trigPos() const {return m_trigPos;} ///< unit is interval
		unsigned int numChannels() const {return m_numChannels;}
		double timeInterval() const {return m_timeInterval;} //! [sec]
		unsigned int length() const;
        const double *wave(unsigned int ch) const;

		void setParameters(unsigned int channels, double startpos, double interval, unsigned int length);
		//! For displaying.
		unsigned int lengthDisp() const;
		double *waveDisp(unsigned int ch);
		const double *waveDisp(unsigned int ch) const;
		double trigPosDisp() const {return m_trigPosDisp;} ///< unit is interval
		unsigned int numChannelsDisp() const {return m_numChannelsDisp;}
		double timeIntervalDisp() const {return m_timeIntervalDisp;} //! [sec]
	private:
		friend class XDSO;
		double m_trigPos; ///< unit is interval
		unsigned int m_numChannels;
		double m_timeInterval; //! [sec]
        std::vector<double> m_waves;

		//! for displaying.
		bool m_rawDisplayOnly; ///< flag for skipping to record.
		double m_trigPosDisp; ///< unit is interval
		unsigned int m_numChannelsDisp;
		double m_timeIntervalDisp; //! [sec]
		std::vector<double> m_wavesDisp;

		shared_ptr<FIR> m_fir;
		shared_ptr<std::vector<std::complex<double> > > m_dRFRefWave; ///< exp(i omega t)
	};
protected:
	virtual void onTrace1Changed(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onTrace2Changed(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onTrace3Changed(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onTrace4Changed(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onSingleChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onTrigSourceChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onTrigPosChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onTrigLevelChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onTrigFallingChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onTimeWidthChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onVFullScale1Changed(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onVFullScale2Changed(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onVFullScale3Changed(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onVFullScale4Changed(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onVOffset1Changed(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onVOffset2Changed(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onVOffset3Changed(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onVOffset4Changed(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onRecordLengthChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onForceTriggerTouched(const Snapshot &shot, XTouchableNode *) = 0;
	virtual void onRestartTouched(const Snapshot &shot, XTouchableNode *);

	virtual double getTimeInterval() = 0;

	//! Clears the count or starts a sequence measurement
	virtual void startSequence() = 0;

	//! \param seq_busy true if the sequence is not finished.
	virtual int acqCount(bool *seq_busy) = 0;

	//! Loads waveforms and settings from the instrument.
	virtual void getWave(shared_ptr<RawData> &writer, std::deque<XString> &channels) = 0;
	//! Converts the raw to a display-able style.
	//! In the coherent SG mode,  real and imaginary parts should be stored in \a Payload::waveDisp().
	virtual void convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) = 0;

	virtual bool isDRFCoherentSGSupported() const {return false;}

	//! Calculates RF phase for coherent detection, at given count.
	double phaseOfRF(const Snapshot &shot_of_this, uint64_t count, double interval);

	const shared_ptr<XStatusPrinter> &statusPrinter() const {return m_statusPrinter;}

    shared_ptr<Listener> m_lsnOnTrigSourceChanged;
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
	const shared_ptr<XTouchableNode> m_forceTrigger;
	const shared_ptr<XTouchableNode> m_restart;
	const shared_ptr<XComboNode> m_trace1;
	const shared_ptr<XComboNode> m_trace2;
	const shared_ptr<XComboNode> m_trace3;
	const shared_ptr<XComboNode> m_trace4;
	const shared_ptr<XComboNode> m_fetchMode;
	const shared_ptr<XBoolNode> m_firEnabled;
	const shared_ptr<XDoubleNode> m_firBandWidth; ///< [kHz]
	const shared_ptr<XDoubleNode> m_firCenterFreq; ///< [kHz]
	const shared_ptr<XDoubleNode> m_firSharpness;

	const shared_ptr<XComboNode> m_dRFMode;
	const shared_ptr<XItemNode<XDriverList, XSG> > m_dRFSG;
	const shared_ptr<XDoubleNode> m_dRFFreq;

	const qshared_ptr<FrmDSO> m_form;
	const shared_ptr<XWaveNGraph> m_waveForm;
  
	//! Converts the raw to a display-able style and performs extra digital processing.
	void convertRawToDisp(RawDataReader &reader, Transaction &tr) throw (XRecordError&);
	//! Digital direct conversion.
    void demodulateDisp(Transaction &tr) throw (XRecordError&);
  
    shared_ptr<Listener> m_lsnOnSingleChanged;
    shared_ptr<Listener> m_lsnOnAverageChanged;
    shared_ptr<Listener> m_lsnOnTrigPosChanged;
    shared_ptr<Listener> m_lsnOnTrigLevelChanged;
    shared_ptr<Listener> m_lsnOnTrigFallingChanged;
    shared_ptr<Listener> m_lsnOnTimeWidthChanged;
    shared_ptr<Listener> m_lsnOnTrace1Changed;
    shared_ptr<Listener> m_lsnOnTrace2Changed;
    shared_ptr<Listener> m_lsnOnTrace3Changed;
    shared_ptr<Listener> m_lsnOnTrace4Changed;
    shared_ptr<Listener> m_lsnOnVFullScale1Changed;
    shared_ptr<Listener> m_lsnOnVFullScale2Changed;
    shared_ptr<Listener> m_lsnOnVFullScale3Changed;
    shared_ptr<Listener> m_lsnOnVFullScale4Changed;
    shared_ptr<Listener> m_lsnOnVOffset1Changed;
    shared_ptr<Listener> m_lsnOnVOffset2Changed;
    shared_ptr<Listener> m_lsnOnVOffset3Changed;
    shared_ptr<Listener> m_lsnOnVOffset4Changed;
    shared_ptr<Listener> m_lsnOnRecordLengthChanged;
    shared_ptr<Listener> m_lsnOnForceTriggerTouched;
    shared_ptr<Listener> m_lsnOnRestartTouched;
    shared_ptr<Listener> m_lsnOnCondChanged;
    shared_ptr<Listener> m_lsnOnDRFCondChanged;
  
	void onCondChanged(const Snapshot &shot, XValueNodeBase *);
	void onDRFCondChanged(const Snapshot &shot, XValueNodeBase *);
  
    std::deque<xqcon_ptr> m_conUIs;

	const shared_ptr<XStatusPrinter> m_statusPrinter;

	void *execute(const atomic<bool> &);
  
	static const char *s_trace_names[];
	static const unsigned int s_trace_colors[];

	atomic<XTime> m_timeSequenceStarted;
};

//---------------------------------------------------------------------------

#endif
