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
#ifndef magnetpsH
#define magnetpsH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "xnodeconnector.h"

class XScalarEntry;
class FrmMagnetPS;

class XMagnetPS : public XPrimaryDriver
{
	XNODE_OBJECT
protected:
	XMagnetPS(const char *name, bool runtime,
			  const shared_ptr<XScalarEntryList> &scalarentries,
			  const shared_ptr<XInterfaceList> &interfaces,
			  const shared_ptr<XThermometerList> &thermometers,
			  const shared_ptr<XDriverList> &drivers);
public:
	//! usually nothing to do
	virtual ~XMagnetPS() {}
	//! show all forms belonging to driver
	virtual void showForms();
 
	//! Records
	double magnetFieldRecorded() const {return m_magnetFieldRecorded;}
	double outputCurrentRecorded() const {return m_outputCurrentRecorded;}

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
	const shared_ptr<XScalarEntry> &field() const {return m_field;}
	const shared_ptr<XScalarEntry> &current() const {return m_current;}

	const shared_ptr<XDoubleNode> &targetField() const {return m_targetField;}
	const shared_ptr<XDoubleNode> &sweepRate() const {return m_sweepRate;}
	const shared_ptr<XBoolNode> &allowPersistent() const {return m_allowPersistent;}
	//! averaged err between magnet field and target one
	const shared_ptr<XDoubleNode> &stabilized() const {return m_stabilized;}
protected:
	const shared_ptr<XDoubleNode> &magnetField() const {return m_magnetField;}
	const shared_ptr<XDoubleNode> &outputField() const {return m_outputField;}
	const shared_ptr<XDoubleNode> &outputCurrent() const {return m_outputCurrent;}
	const shared_ptr<XDoubleNode> &outputVolt() const {return m_outputVolt;}
	const shared_ptr<XBoolNode> &pcsHeater() const {return m_pcsHeater;}
	const shared_ptr<XBoolNode> &persistent() const {return m_persistent;}
  
	virtual double fieldResolution() = 0;
	virtual void toNonPersistent() = 0;
	virtual void toPersistent() = 0;
	virtual void toZero() = 0;
	virtual void toSetPoint() = 0;
	virtual void setPoint(double field) = 0;
	virtual void setRate(double hpm) = 0;
	virtual double getPersistentField() = 0;
	virtual double getOutputField() = 0;
	virtual double getMagnetField() = 0;
	virtual double getTargetField() = 0;
	virtual double getSweepRate() = 0;
	virtual double getOutputVolt() = 0;
	virtual double getOutputCurrent() = 0;
	//! Persistent Current Switch Heater
	//! please return *TRUE* if no PCS fitted
	virtual bool isPCSHeaterOn() = 0;
	//! please return false if no PCS fitted
	virtual bool isPCSFitted() = 0;
private:
	virtual void onRateChanged(const shared_ptr<XValueNodeBase> &);
  
	const shared_ptr<XScalarEntry> m_field, m_current;

	const shared_ptr<XDoubleNode> m_targetField;
	const shared_ptr<XDoubleNode> m_sweepRate;
	const shared_ptr<XBoolNode> m_allowPersistent;
	//! averaged err between magnet field and target one
	const shared_ptr<XDoubleNode> m_stabilized;
	const shared_ptr<XDoubleNode> m_magnetField, m_outputField, m_outputCurrent, m_outputVolt;
	const shared_ptr<XBoolNode> m_pcsHeater, m_persistent;

	shared_ptr<XListener> m_lsnRate;
  
	xqcon_ptr m_conAllowPersistent;
	xqcon_ptr m_conTargetField, m_conSweepRate;
	xqcon_ptr m_conMagnetField, m_conOutputField, m_conOutputCurrent, m_conOutputVolt;
	xqcon_ptr m_conPCSH, m_conPersist;
 
	shared_ptr<XThread<XMagnetPS> > m_thread;
	const qshared_ptr<FrmMagnetPS> m_form;
	const shared_ptr<XStatusPrinter> m_statusPrinter;
  
	//! Records
	double m_magnetFieldRecorded;
	double m_outputCurrentRecorded;
    
	void *execute(const atomic<bool> &);
  
};

#endif
