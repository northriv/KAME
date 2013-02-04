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
//---------------------------------------------------------------------------
#include "ui_flowcontrollerform.h"
#include "flowcontroller.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"

XFlowController::XFlowController(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_flow(create<XScalarEntry>("Flow", false,
								  dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_target(create<XDoubleNode>("Target", true)),
    m_valve(create<XDoubleNode>("Valve", true)),
    m_rampTime(create<XDoubleNode>("RampTime", true)),
    m_openValve(create<XTouchableNode>("OpenValve", true)),
    m_closeValve(create<XTouchableNode>("CloseValve", true)),
    m_warning(create<XBoolNode>("Warning", true)),
    m_alarm(create<XBoolNode>("Alarm", true)),
    m_control(create<XBoolNode>("Control", true)),
    m_form(new FrmFlowController(g_pFrmMain)) {

	for(Transaction tr( *this);; ++tr) {
		tr[ *control()] = false;
		if(tr.commit())
			break;
	}

	meas->scalarEntries()->insert(tr_meas, m_flow);

	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("Flow Controller - ") + getLabel() );

	m_target->setUIEnabled(false);
	m_valve->setUIEnabled(false);
	m_rampTime->setUIEnabled(false);
	m_openValve->setUIEnabled(false);
	m_closeValve->setUIEnabled(false);
	m_alarm->setUIEnabled(false);
	m_warning->setUIEnabled(false);
	m_control->setUIEnabled(false);

	m_conFlow = xqcon_create<XQLCDNumberConnector>(m_flow->value(), m_form->m_lcdFlow);
	m_conValve = xqcon_create<XQLCDNumberConnector>(m_valve, m_form->m_lcdValve);
	m_conTarget = xqcon_create<XKDoubleNumInputConnector>(m_target, m_form->m_dblTarget);
	m_form->m_dblTarget->setRange(0.0, 1000.0, 1, true);
	m_conRampTime = xqcon_create<XQLineEditConnector>(m_rampTime, m_form->m_edRampTime);
	m_conControl = xqcon_create<XQToggleButtonConnector>(m_control, m_form->m_ckbControl);
	m_conAlarm = xqcon_create<XKLedConnector>(m_alarm, m_form->m_ledAlarm);
	m_conWarning = xqcon_create<XKLedConnector>(m_warning, m_form->m_ledWarning);
	m_conOpenValve = xqcon_create<XQButtonConnector>(m_openValve, m_form->m_btnOpenValve);
	m_conCloseValve = xqcon_create<XQButtonConnector>(m_closeValve, m_form->m_btnCloseValve);
}

void
XFlowController::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}

void
XFlowController::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    double flow, valve;
    flow = reader.pop<double>();
    valve = reader.pop<double>();
    bool alarm = reader.pop<uin16_t>();
    bool warning = reader.pop<uin16_t>();
    tr[ *m_flow] = flow;
    tr[ *m_valve] = valve;
    tr[ *alarm] = alarm;
    tr[ *warning] = warning;
}
void
XFlowController::visualize(const Snapshot &shot) {
}

void
XFlowController::onTargetChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        changeSetPoint(shot[ *target()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while changing target, "));
        return;
    }
}
void
XFlowController::onRampTimeChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        changeRampTime(shot[ *rampTime()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while changing conditions, "));
        return;
    }
}
void
XFlowController::onOpenValveTouched(const Snapshot &shot, XTouchableNode *) {
    try {
        openValve();
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while opening valve, "));
        return;
    }
}
void
XFlowController::onCloseValveTouched(const Snapshot &shot, XTouchableNode *) {
    try {
        closeValve();
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while closeing valve, "));
        return;
    }
}
void
XFlowController::onControlChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        changeControl(shot[ *control()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error, "));
        return;
    }
}
void *
XFlowController::execute(const atomic<bool> &terminated) {
	m_target->setUIEnabled(true);
	m_valve->setUIEnabled(true);
	m_rampTime->setUIEnabled(true);
	m_openValve->setUIEnabled(true);
	m_closeValve->setUIEnabled(true);
	m_alarm->setUIEnabled(true);
	m_warning->setUIEnabled(true);
	m_control->setUIEnabled(true);





	for(Transaction tr( *this);; ++tr) {
		m_lsnTarget = tr[ *target()].onValueChanged().connectWeakly(shared_from_this(), &XFlowController::onTargetChanged);
		m_lsnConditions = tr[ *stepMotor()].onValueChanged().connectWeakly(shared_from_this(), &XFlowController::onConditionsChanged);
		tr[ *stepEncoder()].onValueChanged().connect(m_lsnConditions);
		tr[ *currentStopping()].onValueChanged().connect(m_lsnConditions);
		tr[ *currentRunning()].onValueChanged().connect(m_lsnConditions);
		tr[ *speed()].onValueChanged().connect(m_lsnConditions);
		tr[ *timeAcc()].onValueChanged().connect(m_lsnConditions);
		tr[ *timeDec()].onValueChanged().connect(m_lsnConditions);
		tr[ *active()].onValueChanged().connect(m_lsnConditions);
		tr[ *microStep()].onValueChanged().connect(m_lsnConditions);
		tr[ *round()].onValueChanged().connect(m_lsnConditions);
		tr[ *roundBy()].onValueChanged().connect(m_lsnConditions);
		m_lsnClear = tr[ *clear()].onTouch().connectWeakly(shared_from_this(), &XFlowController::onClearTouched);
		m_lsnStore = tr[ *store()].onTouch().connectWeakly(shared_from_this(), &XFlowController::onStoreTouched);
		m_lsnForwardMotor = tr[ *forwardMotor()].onTouch().connectWeakly(shared_from_this(), &XFlowController::onForwardMotorTouched);
		m_lsnReverseMotor = tr[ *reverseMotor()].onTouch().connectWeakly(shared_from_this(), &XFlowController::onReverseMotorTouched);
		m_lsnStopMotor = tr[ *stopMotor()].onTouch().connectWeakly(shared_from_this(), &XFlowController::onStopMotorTouched);
		m_lsnAUX = tr[ *auxBits()].onValueChanged().connectWeakly(shared_from_this(), &XFlowController::onAUXChanged);
		if(tr.commit())
			break;
	}

	while( !terminated) {
		msecsleep(100);
		XTime time_awared = XTime::now();
		double pos = 0;
		bool slip = false, isready = false;
		// try/catch exception of communication errors
		try {
			Snapshot shot( *this);
			getStatus(shot, &pos, &slip, &isready);
		}
		catch (XKameError &e) {
			e.print(getLabel() + " " + i18n("Read Error, "));
			continue;
		}
		shared_ptr<RawData> writer(new RawData);
		writer->push(pos);
		writer->push((uint16_t)slip);
		writer->push((uint16_t)isready);
		finishWritingRaw(writer, time_awared, XTime::now());
	}

	m_target->setUIEnabled(false);
	m_valve->setUIEnabled(false);
	m_rampTime->setUIEnabled(false);
	m_openValve->setUIEnabled(false);
	m_closeValve->setUIEnabled(false);
	m_alarm->setUIEnabled(false);
	m_warning->setUIEnabled(false);
	m_control->setUIEnabled(false);

	m_lsnTarget.reset();
	m_lsnConditions.reset();
	m_lsnClear.reset();
	m_lsnStore.reset();
	m_lsnAUX.reset();
	m_lsnForwardMotor.reset();
	m_lsnReverseMotor.reset();
	m_lsnStopMotor.reset();
	return NULL;
}
