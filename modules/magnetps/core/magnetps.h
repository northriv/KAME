/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
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
class Ui_FrmMagnetPSConfig;
typedef QForm<QMainWindow, Ui_FrmMagnetPSConfig> FrmMagnetPSConfig;

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
	virtual void onRateChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onConfigShow(const Snapshot &shot, XTouchableNode *);

	const shared_ptr<XScalarEntry> m_field, m_current;
	const shared_ptr<XScalarEntryList> m_entries;

	const shared_ptr<XDoubleNode> m_targetField;
	const shared_ptr<XDoubleNode> m_sweepRate;
	const shared_ptr<XBoolNode> m_allowPersistent;
	//! averaged err between magnet field and target one
	const shared_ptr<XDoubleNode> m_stabilized;
	const shared_ptr<XDoubleNode> m_magnetField, m_outputField, m_outputCurrent, m_outputVolt;
	const shared_ptr<XBoolNode> m_pcsHeater, m_persistent, m_aborting;

	const shared_ptr<XTouchableNode> m_configShow;
	 //! Rate limiting. [T/min] and [T].
	const shared_ptr<XDoubleNode> m_rateLimit1, m_rateLimit1UBound;
	const shared_ptr<XDoubleNode> m_rateLimit2, m_rateLimit2UBound;
	const shared_ptr<XDoubleNode> m_rateLimit3, m_rateLimit3UBound;
	const shared_ptr<XDoubleNode> m_rateLimit4, m_rateLimit4UBound;
	const shared_ptr<XDoubleNode> m_rateLimit5, m_rateLimit5UBound;

	const shared_ptr<XDoubleNode> m_secondaryPSMultiplier; //!< For Shim coil. [T/T]
	const shared_ptr<XItemNode<XDriverList, XMagnetPS> > m_secondaryPS; //!< For Shim coil

	//! Configuration for safe conditions.
	const shared_ptr<XItemNode<XScalarEntryList, XScalarEntry> > m_safeCond1Entry;
	const shared_ptr<XDoubleNode> m_safeCond1Min, m_safeCond1Max;
	const shared_ptr<XItemNode<XScalarEntryList, XScalarEntry> > m_safeCond2Entry;
	const shared_ptr<XDoubleNode> m_safeCond2Min, m_safeCond2Max;
	const shared_ptr<XItemNode<XScalarEntryList, XScalarEntry> > m_safeCond3Entry;
	const shared_ptr<XDoubleNode> m_safeCond3Min, m_safeCond3Max;

	const shared_ptr<XItemNode<XScalarEntryList, XScalarEntry> > m_persistentCondEntry;
	const shared_ptr<XDoubleNode> m_persistentCondMax;
	const shared_ptr<XItemNode<XScalarEntryList, XScalarEntry> > m_nonPersistentCondEntry;
	const shared_ptr<XDoubleNode> m_nonPersistentCondMin;

	const shared_ptr<XDoubleNode> m_pcshWait; //!< [sec]

	shared_ptr<XListener> m_lsnRate, m_lsnConfigShow;
  
	xqcon_ptr m_conAllowPersistent;
	xqcon_ptr m_conTargetField, m_conSweepRate;
	xqcon_ptr m_conMagnetField, m_conOutputField, m_conOutputCurrent, m_conOutputVolt;
	xqcon_ptr m_conPCSH, m_conPersist, m_conAborting;
	xqcon_ptr m_conConfigShow;
	xqcon_ptr m_conRateLimit1, m_conRateLimit1UBound;
	xqcon_ptr m_conRateLimit2, m_conRateLimit2UBound;
	xqcon_ptr m_conRateLimit3, m_conRateLimit3UBound;
	xqcon_ptr m_conRateLimit4, m_conRateLimit4UBound;
	xqcon_ptr m_conRateLimit5, m_conRateLimit5UBound;
	xqcon_ptr m_conSecondaryPS, m_conSecondaryPSMultiplier;
	xqcon_ptr m_conSafeCond1Entry, m_conSafeCond1Min, m_conSafeCond1Max;
	xqcon_ptr m_conSafeCond2Entry, m_conSafeCond2Min, m_conSafeCond2Max;
	xqcon_ptr m_conSafeCond3Entry, m_conSafeCond3Min, m_conSafeCond3Max;
	xqcon_ptr m_conPersistentCondEntry, m_conPersistentCondMax;
	xqcon_ptr m_conNonPersistentCondEntry, m_conNonPersistentCondMin;
	xqcon_ptr m_conPCSHWait;
 
	shared_ptr<XThread<XMagnetPS> > m_thread;
	const qshared_ptr<FrmMagnetPS> m_form;
	const qshared_ptr<FrmMagnetPSConfig> m_formConfig;
	const shared_ptr<XStatusPrinter> m_statusPrinter;
  
	void *execute(const atomic<bool> &);
  
	bool isSafeConditionSatisfied(const Snapshot &shot, const Snapshot &shot_entries);
	bool isPersistentStabilized(const Snapshot &shot, const Snapshot &shot_entries, const XTime &pcsh_off_time);
	bool isNonPersistentStabilized(const Snapshot &shot, const Snapshot &shot_entries, const XTime &pcsh_on_time);
	double limitSweepRate(double field, double rate, const Snapshot &shot);
	double limitTargetField(double field, const Snapshot &shot);
};

#endif
