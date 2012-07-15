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
#ifndef motorH
#define motorH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "xnodeconnector.h"

class XScalarEntry;
class QMainWindow;
class Ui_FrmMotorDriver;
typedef QForm<QMainWindow, Ui_FrmMotorDriver> FrmMotorDriver;

class XMotorDriver : public XPrimaryDriver {
public:
	XMotorDriver(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
	virtual ~XMotorDriver() {}
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
	const shared_ptr<XScalarEntry> &position() const {return m_position;}

	const shared_ptr<XDoubleNode> &target() const {return m_target;} //!< [step]
	const shared_ptr<XUIntNode> &step() const {return m_step;} //!< [step]
	const shared_ptr<XDoubleNode> &currentStopping() const {return m_currentStopping;} //!< [%]
	const shared_ptr<XDoubleNode> &currentRunning() const {return m_currentRunning;} //!< [%]
	const shared_ptr<XDoubleNode> &speed() const {return m_speed;} //!< [Hz]
	const shared_ptr<XDoubleNode> &timeAcc() const {return m_timeAcc;} //!< [s]
	const shared_ptr<XDoubleNode> &timeDec() const {return m_timeDec;} //!< [s]
	const shared_ptr<XBoolNode> &active() const {return m_active;}
	const shared_ptr<XBoolNode> &ready() const {return m_ready;}
	const shared_ptr<XBoolNode> &slipping() const {return m_slipping;}
	const shared_ptr<XBoolNode> &microStep() const {return m_microStep;}

protected:
	virtual void getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) = 0;
	virtual void changeConditions(const Snapshot &shot) = 0;
	virtual void getConditions(Transaction &tr) = 0;
	virtual void setTarget(const Snapshot &shot, double target) = 0;
	virtual void setActive(bool active) = 0;
private:
	const shared_ptr<XScalarEntry> m_position;

	const shared_ptr<XDoubleNode> m_target;
	const shared_ptr<XUIntNode> m_step;
	const shared_ptr<XDoubleNode> m_currentStopping;
	const shared_ptr<XDoubleNode> m_currentRunning;
	const shared_ptr<XDoubleNode> m_speed;
	const shared_ptr<XDoubleNode> m_timeAcc;
	const shared_ptr<XDoubleNode> m_timeDec;
	const shared_ptr<XBoolNode> m_active;
	const shared_ptr<XBoolNode> m_ready;
	const shared_ptr<XBoolNode> m_slipping;
	const shared_ptr<XBoolNode> m_microStep;

	shared_ptr<XListener> m_lsnTarget, m_lsnConditions;
	xqcon_ptr m_conPosition, m_conTarget, m_conStep,
		m_conCurrentStopping, m_conCurrentRunning, m_conSpeed,
		m_conTimeAcc, m_conTimeDec, m_conActive, m_conReady, m_conSlipping, m_conMicroStep;

	shared_ptr<XThread<XMotorDriver> > m_thread;
	const qshared_ptr<FrmMotorDriver> m_form;

	void onTargetChanged(const Snapshot &shot, XValueNodeBase *);
	void onConditionsChanged(const Snapshot &shot, XValueNodeBase *);

	void *execute(const atomic<bool> &);

};


#endif
