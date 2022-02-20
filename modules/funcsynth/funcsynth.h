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
#ifndef funcsynthH
#define funcsynthH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "xnodeconnector.h"

class QMainWindow;
class Ui_FrmFuncSynth;
typedef QForm<QMainWindow, Ui_FrmFuncSynth> FrmFuncSynth;

class XFuncSynth : public XPrimaryDriver {
public:
	XFuncSynth(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
	virtual ~XFuncSynth() {}
	//! Shows all forms belonging to driver
	virtual void showForms();
protected:
	//! Starts up your threads, connects GUI, and activates signals.
	virtual void start();
	//! Shuts down your threads, unconnects GUI, and deactivates signals
	//! This function may be called even if driver has already stopped.
	virtual void stop();
  
	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
	virtual void analyzeRaw(RawDataReader &reader, Transaction &tr);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);
  
	//! driver specific part below
	const shared_ptr<XBoolNode> &output() const {return m_output;}
	const shared_ptr<XTouchableNode> &trig() const {return m_trig;} //!< trigger to burst
	const shared_ptr<XComboNode> &mode() const {return m_mode;}
	const shared_ptr<XComboNode> &function() const {return m_function;}
	const shared_ptr<XDoubleNode> &freq() const {return m_freq;} //!< [Hz]
	const shared_ptr<XDoubleNode> &amp() const {return m_amp;} //!< [Vp-p]
	const shared_ptr<XDoubleNode> &phase() const {return m_phase;} //!< [deg.]
	const shared_ptr<XDoubleNode> &offset() const {return m_offset;} //!< [V]
protected:
	virtual void onOutputChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onTrigTouched(const Snapshot &shot, XTouchableNode *) = 0;
	virtual void onModeChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onFunctionChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onFreqChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onAmpChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onPhaseChanged(const Snapshot &shot, XValueNodeBase *) = 0;
	virtual void onOffsetChanged(const Snapshot &shot, XValueNodeBase *) = 0;
private:
	const shared_ptr<XBoolNode>  m_output;
	const shared_ptr<XTouchableNode>  m_trig;
	const shared_ptr<XComboNode>  m_mode;
	const shared_ptr<XComboNode>  m_function;
	const shared_ptr<XDoubleNode>  m_freq;
	const shared_ptr<XDoubleNode>  m_amp;
	const shared_ptr<XDoubleNode>  m_phase;
	const shared_ptr<XDoubleNode>  m_offset;
	shared_ptr<Listener> m_lsnOutput, m_lsnMode, m_lsnFunction,
		m_lsnFreq, m_lsnAmp, m_lsnPhase, m_lsnOffset;
	shared_ptr<Listener> m_lsnTrig;
	xqcon_ptr m_conOutput, m_conTrig, m_conMode, m_conFunction,
		m_conFreq, m_conAmp, m_conPhase, m_conOffset;

	const qshared_ptr<FrmFuncSynth> m_form;
};

#endif
