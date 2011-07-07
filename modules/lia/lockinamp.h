/***************************************************************************
		Copyright (C) 2002-2011 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef lockinampH
#define lockinampH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "xnodeconnector.h"

class XScalarEntry;
class QMainWindow;
class Ui_FrmLIA;
typedef QForm<QMainWindow, Ui_FrmLIA> FrmLIA;

class XLIA : public XPrimaryDriver {
public:
	XLIA(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
	virtual ~XLIA() {}
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
	virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);
  
	//! driver specific part below
	const shared_ptr<XScalarEntry> &valueX() const {return m_valueX;}
	const shared_ptr<XScalarEntry> &valueY() const {return m_valueY;}
   
	const shared_ptr<XDoubleNode> & output() const {return m_output;}
	const shared_ptr<XDoubleNode> & frequency() const {return m_frequency;}
	const shared_ptr<XComboNode> & sensitivity() const {return m_sensitivity;}
	const shared_ptr<XComboNode> & timeConst() const {return m_timeConst;}
	const shared_ptr<XBoolNode> & autoScaleX() const {return m_autoScaleX;}
	const shared_ptr<XBoolNode> & autoScaleY() const {return m_autoScaleY;}
	const shared_ptr<XDoubleNode> & fetchFreq() const {return m_fetchFreq;}
protected:
	virtual void get(double *cos, double *sin) = 0;
	virtual void changeOutput(double volt) = 0;
	virtual void changeFreq(double freq) = 0;
	virtual void changeSensitivity(int) = 0;
	virtual void changeTimeConst(int) = 0;
private:
	const shared_ptr<XScalarEntry> m_valueX, m_valueY;
 
	const shared_ptr<XDoubleNode> m_output;
	const shared_ptr<XDoubleNode> m_frequency;
	const shared_ptr<XComboNode> m_sensitivity;
	const shared_ptr<XComboNode> m_timeConst;
	const shared_ptr<XBoolNode> m_autoScaleX;
	const shared_ptr<XBoolNode> m_autoScaleY;
	const shared_ptr<XDoubleNode> m_fetchFreq; //Data Acquision Frequency to Time Constant
	shared_ptr<XListener> m_lsnOutput, m_lsnSens, m_lsnTimeConst, m_lsnFreq;
	xqcon_ptr m_conSens, m_conTimeConst, m_conOutput, m_conFreq;
	xqcon_ptr m_conAutoScaleX, m_conAutoScaleY, m_conFetchFreq;
 
	shared_ptr<XThread<XLIA> > m_thread;
	const qshared_ptr<FrmLIA> m_form;
  
	void onOutputChanged(const Snapshot &shot, XValueNodeBase *);
	void onFreqChanged(const Snapshot &shot, XValueNodeBase *);
	void onSensitivityChanged(const Snapshot &shot, XValueNodeBase *);
	void onTimeConstChanged(const Snapshot &shot, XValueNodeBase *);

	void *execute(const atomic<bool> &);
  
};


#endif
