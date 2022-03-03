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
#ifndef motorH
#define motorH
//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"

class XScalarEntry;
class QMainWindow;
class Ui_FrmMotorDriver;
typedef QForm<QMainWindow, Ui_FrmMotorDriver> FrmMotorDriver;

class DECLSPEC_SHARED XMotorDriver : public XPrimaryDriverWithThread {
public:
	XMotorDriver(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
	virtual ~XMotorDriver() {}
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
	const shared_ptr<XScalarEntry> &position() const {return m_position;} //!< [deg.]

	const shared_ptr<XDoubleNode> &target() const {return m_target;} //!< [deg.]
	const shared_ptr<XUIntNode> &stepMotor() const {return m_stepMotor;} //!< [steps per rot.]
	const shared_ptr<XUIntNode> &stepEncoder() const {return m_stepEncoder;} //!< [steps per rot.]
	const shared_ptr<XDoubleNode> &currentStopping() const {return m_currentStopping;} //!< [%]
	const shared_ptr<XDoubleNode> &currentRunning() const {return m_currentRunning;} //!< [%]
	const shared_ptr<XDoubleNode> &speed() const {return m_speed;} //!< [Hz]
	const shared_ptr<XDoubleNode> &timeAcc() const {return m_timeAcc;} //!< [ms/kHz]
	const shared_ptr<XDoubleNode> &timeDec() const {return m_timeDec;} //!< [ms/kHz]
	const shared_ptr<XBoolNode> &active() const {return m_active;}
	const shared_ptr<XBoolNode> &ready() const {return m_ready;}
	const shared_ptr<XBoolNode> &slipping() const {return m_slipping;}
	const shared_ptr<XBoolNode> &microStep() const {return m_microStep;}
	const shared_ptr<XBoolNode> &hasEncoder() const {return m_hasEncoder;}
	const shared_ptr<XTouchableNode> &store() const {return m_store;}
	const shared_ptr<XTouchableNode> &clear() const {return m_clear;}
	const shared_ptr<XUIntNode> &auxBits() const {return m_auxBits;}
	const shared_ptr<XBoolNode> &round() const {return m_round;}
	const shared_ptr<XUIntNode> &roundBy() const {return m_roundBy;}
	const shared_ptr<XTouchableNode> &forwardMotor() const {return m_forwardMotor;}
	const shared_ptr<XTouchableNode> &reverseMotor() const {return m_reverseMotor;}
	const shared_ptr<XTouchableNode> &stopMotor() const {return m_stopMotor;}
protected:
	virtual void getStatus(const Snapshot &shot, double *position, bool *slipping, bool *ready) = 0;
	virtual void changeConditions(const Snapshot &shot) = 0;
    virtual void getConditions() = 0;
	virtual void setTarget(const Snapshot &shot, double target) = 0;
	virtual void setForward() = 0;
	virtual void setReverse() = 0;
	virtual void stopRotation() = 0;
	virtual void setActive(bool active) = 0;
	virtual void setAUXBits(unsigned int bits) = 0;
	//! stores current settings to the NV memory of the instrument.
	virtual void storeToROM() = 0;
	virtual void clearPosition() = 0;
private:
	const shared_ptr<XScalarEntry> m_position;

	const shared_ptr<XDoubleNode> m_target;
	const shared_ptr<XUIntNode> m_stepMotor;
	const shared_ptr<XUIntNode> m_stepEncoder;
	const shared_ptr<XDoubleNode> m_currentStopping;
	const shared_ptr<XDoubleNode> m_currentRunning;
	const shared_ptr<XDoubleNode> m_speed;
	const shared_ptr<XDoubleNode> m_timeAcc;
	const shared_ptr<XDoubleNode> m_timeDec;
	const shared_ptr<XBoolNode> m_active;
	const shared_ptr<XBoolNode> m_ready;
	const shared_ptr<XBoolNode> m_slipping;
	const shared_ptr<XBoolNode> m_microStep;
	const shared_ptr<XBoolNode> m_hasEncoder;
	const shared_ptr<XUIntNode> m_auxBits;
	const shared_ptr<XTouchableNode> m_clear;
	const shared_ptr<XTouchableNode> m_store;
	const shared_ptr<XBoolNode> m_round;
	const shared_ptr<XUIntNode> m_roundBy;
	const shared_ptr<XTouchableNode> m_forwardMotor;
	const shared_ptr<XTouchableNode>  m_reverseMotor;
	const shared_ptr<XTouchableNode> m_stopMotor;

	shared_ptr<Listener> m_lsnTarget, m_lsnConditions,
		m_lsnClear, m_lsnStore, m_lsnForwardMotor, m_lsnReverseMotor, m_lsnStopMotor, m_lsnAUX;
	xqcon_ptr m_conPosition, m_conTarget, m_conStepMotor, m_conStepEncoder,
		m_conCurrentStopping, m_conCurrentRunning, m_conSpeed,
		m_conTimeAcc, m_conTimeDec, m_conActive, m_conReady, m_conSlipping,
		m_conMicroStep, m_conHasEncoder, m_conClear, m_conStore,
		m_conForwardMotor, m_conReverseMotor, m_conStopMotor,
		m_conAUXBits, m_conRound, m_conRoundBy;

	const qshared_ptr<FrmMotorDriver> m_form;

	void onTargetChanged(const Snapshot &shot, XValueNodeBase *);
	void onAUXChanged(const Snapshot &shot, XValueNodeBase *);
	void onConditionsChanged(const Snapshot &shot, XValueNodeBase *);
	void onClearTouched(const Snapshot &shot, XTouchableNode *);
	void onStoreTouched(const Snapshot &shot, XTouchableNode *);
	void onForwardMotorTouched(const Snapshot &shot, XTouchableNode *);
	void onReverseMotorTouched(const Snapshot &shot, XTouchableNode *);
	void onStopMotorTouched(const Snapshot &shot, XTouchableNode *);

	void *execute(const atomic<bool> &);

    XTime m_timeMovementStarted;
};


#endif
