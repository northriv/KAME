/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef flowcontrollerH
#define flowcontrollerH
//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"

class XScalarEntry;
class QMainWindow;
class Ui_FrmFlowController;
typedef QForm<QMainWindow, Ui_FrmFlowController> FrmFlowController;

//! Base class for mass flow monitors/controllers.
class DECLSPEC_SHARED XFlowControllerDriver : public XPrimaryDriverWithThread {
public:
	XFlowControllerDriver(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
	virtual ~XFlowControllerDriver() {}
	//! Shows all forms belonging to driver
	virtual void showForms();
protected:
	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
	virtual void analyzeRaw(RawDataReader &reader, Transaction &tr);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);

public:
	//! driver specific part below
	const shared_ptr<XScalarEntry> &flow() const {return m_flow;} //!< [Given Unit]

	const shared_ptr<XDoubleNode> &target() const {return m_target;} //!< [Given Unit]
	const shared_ptr<XDoubleNode> &valve() const {return m_valve;} //!< [V]
	const shared_ptr<XDoubleNode> &rampTime() const {return m_rampTime;} //!< [ms]
	const shared_ptr<XTouchableNode> &openValve() const {return m_openValve;}
	const shared_ptr<XTouchableNode> &closeValve() const {return m_closeValve;}
	const shared_ptr<XBoolNode> &warning() const {return m_warning;}
	const shared_ptr<XBoolNode> &alarm() const {return m_alarm;}
	const shared_ptr<XBoolNode> &control() const {return m_control;}

    struct DECLSPEC_SHARED Payload : public XPrimaryDriver::Payload {
		double fullScale() const {return m_fullScale;}
        const XString &unit() const {return m_unit;}
		XString m_unit;
		double m_fullScale;
	};
protected:
	virtual bool isController() = 0; //! distinguishes monitors and controllers.
	virtual bool isUnitInSLM() = 0; //! false for SCCM.
	virtual double getFullScale() = 0;

	virtual void getStatus(double &flow_in_slm, double &valve_v, bool &alarm, bool &warning) = 0;
	virtual void setValveState(bool open) = 0;
	virtual void changeControl(bool ctrl) = 0;
	virtual void changeSetPoint(double target) = 0;
	virtual void setRampTime(double time) = 0;
private:
	const shared_ptr<XScalarEntry> m_flow;

	const shared_ptr<XDoubleNode> m_target;
	const shared_ptr<XDoubleNode> m_valve;
	const shared_ptr<XDoubleNode> m_rampTime;
	const shared_ptr<XTouchableNode> m_openValve;
	const shared_ptr<XTouchableNode> m_closeValve;
	const shared_ptr<XBoolNode> m_warning;
	const shared_ptr<XBoolNode> m_alarm;
	const shared_ptr<XBoolNode> m_control;

	shared_ptr<Listener> m_lsnTarget, m_lsnOpenValve, m_lsnCloseValve, m_lsnControl, m_lsnRampTime;
	xqcon_ptr m_conFlow, m_conAlarm, m_conWarning, m_conTarget,
		m_conRampTime, m_conValve, m_conControl, m_conOpenValve, m_conCloseValve;

	const qshared_ptr<FrmFlowController> m_form;

	void onTargetChanged(const Snapshot &shot, XValueNodeBase *);
	void onRampTimeChanged(const Snapshot &shot, XValueNodeBase *);
	void onControlChanged(const Snapshot &shot, XValueNodeBase *);
	void onOpenValveTouched(const Snapshot &shot, XTouchableNode *);
	void onCloseValveTouched(const Snapshot &shot, XTouchableNode *);

	void *execute(const atomic<bool> &);

};


#endif
