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
#ifndef lockinampH
#define lockinampH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "xnodeconnector.h"

class XScalarEntry;
class FrmLIA;

class XLIA : public XPrimaryDriver
{
	XNODE_OBJECT
protected:
	XLIA(const char *name, bool runtime,
		 const shared_ptr<XScalarEntryList> &scalarentries,
		 const shared_ptr<XInterfaceList> &interfaces,
		 const shared_ptr<XThermometerList> &thermometers,
		 const shared_ptr<XDriverList> &drivers);
public:
	//! usually nothing to do
	virtual ~XLIA() {}
	//! show all forms belonging to driver
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
  
	void onOutputChanged(const shared_ptr<XValueNodeBase> &);
	void onFreqChanged(const shared_ptr<XValueNodeBase> &);
	void onSensitivityChanged(const shared_ptr<XValueNodeBase> &);
	void onTimeConstChanged(const shared_ptr<XValueNodeBase> &);

	void *execute(const atomic<bool> &);
  
};


#endif
