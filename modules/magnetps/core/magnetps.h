/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
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
class QMainWindow;
class Ui_FrmMagnetPS;
typedef QForm<QMainWindow, Ui_FrmMagnetPS> FrmMagnetPS;

class XMagnetPS : public XPrimaryDriver {
public:
	XMagnetPS(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
	virtual ~XMagnetPS() {}
	//! Shows all forms belonging to driver
	virtual void showForms();
 
	struct Payload : public XPrimaryDriver::Payload {
		double magnetField() const {return m_magnetField;}
		double outputCurrent() const {return m_outputCurrent;}
	private:
		friend class XMagnetPS;
		double m_magnetField;
		double m_outputCurrent;
	};
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
  
	void *execute(const atomic<bool> &);
  
};

#endif
