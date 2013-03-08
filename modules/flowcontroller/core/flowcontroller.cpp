/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
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
#include <QStatusBar>

XFlowControllerDriver::XFlowControllerDriver(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_flow(create<XScalarEntry>("Flow", false,
								  dynamic_pointer_cast<XDriver>(shared_from_this()), "%.4f")),
    m_target(create<XDoubleNode>("Target", true)),
    m_valve(create<XDoubleNode>("Valve", true, "%.4f")),
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
	m_rampTime->setUIEnabled(false);
	m_openValve->setUIEnabled(false);
	m_closeValve->setUIEnabled(false);
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
XFlowControllerDriver::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}

void
XFlowControllerDriver::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    double flow, valve;
    flow = reader.pop<double>();
    valve = reader.pop<double>();
    bool alarm = reader.pop<uint16_t>();
    bool warning = reader.pop<uint16_t>();
    m_flow->value(tr, flow);
    tr[ *m_valve] = valve;
    tr[ *m_alarm] = alarm;
    tr[ *m_warning] = warning;
}
void
XFlowControllerDriver::visualize(const Snapshot &shot) {
	if(m_form->m_lblUnit->text().isEmpty()) {
		m_form->m_dblTarget->setRange(0.0, shot[ *this].m_fullScale, 1, true);
		m_form->m_lblUnit->setText(shot[ *this].m_unit);
	}
}

void
XFlowControllerDriver::onTargetChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        changeSetPoint(shot[ *target()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while changing target, "));
        return;
    }
}
void
XFlowControllerDriver::onRampTimeChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        setRampTime(shot[ *rampTime()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while changing conditions, "));
        return;
    }
}
void
XFlowControllerDriver::onOpenValveTouched(const Snapshot &shot, XTouchableNode *) {
    try {
        setValveState(true);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while opening valve, "));
        return;
    }
}
void
XFlowControllerDriver::onCloseValveTouched(const Snapshot &shot, XTouchableNode *) {
    try {
        setValveState(false);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while closeing valve, "));
        return;
    }
}
void
XFlowControllerDriver::onControlChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        changeControl(shot[ *control()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error, "));
        return;
    }
}
void *
XFlowControllerDriver::execute(const atomic<bool> &terminated) {
	double fs;
	bool unit_in_slm;
	// try/catch exception of communication errors
	try {
		fs = getFullScale();
		unit_in_slm = isUnitInSLM();
		if(isController()) {
			m_target->setUIEnabled(true);
			m_rampTime->setUIEnabled(true);
			m_openValve->setUIEnabled(true);
			m_closeValve->setUIEnabled(true);
			m_control->setUIEnabled(true);
		}
		else
			m_valve->setUIEnabled(false);
	}
	catch (XKameError &e) {
		e.print(getLabel() + " " + i18n("Read Error, "));
	}

	for(Transaction tr( *this);; ++tr) {
		tr[ *this].m_fullScale = fs;
		tr[ *this].m_unit = unit_in_slm ? "SLM" : "SCCM";
		m_lsnTarget = tr[ *target()].onValueChanged().connectWeakly(shared_from_this(), &XFlowControllerDriver::onTargetChanged);
		m_lsnRampTime = tr[ *rampTime()].onValueChanged().connectWeakly(shared_from_this(), &XFlowControllerDriver::onRampTimeChanged);
		m_lsnControl = tr[ *control()].onValueChanged().connectWeakly(shared_from_this(), &XFlowControllerDriver::onControlChanged);
		m_lsnCloseValve = tr[ *closeValve()].onTouch().connectWeakly(shared_from_this(), &XFlowControllerDriver::onCloseValveTouched);
		m_lsnOpenValve = tr[ *openValve()].onTouch().connectWeakly(shared_from_this(), &XFlowControllerDriver::onOpenValveTouched);
		if(tr.commit())
			break;
	}

	while( !terminated) {
		msecsleep(100);
		XTime time_awared = XTime::now();
		double flow = 0, valve = 0;
		bool warning = false, alarm = false;
		// try/catch exception of communication errors
		try {
			getStatus(flow, valve, alarm, warning);
		}
		catch (XKameError &e) {
			e.print(getLabel() + " " + i18n("Read Error, "));
			continue;
		}
		shared_ptr<RawData> writer(new RawData);
		writer->push(flow);
		writer->push(valve);
		writer->push((uint16_t)alarm);
		writer->push((uint16_t)warning);
		finishWritingRaw(writer, time_awared, XTime::now());
	}

	m_valve->setUIEnabled(false);
	m_rampTime->setUIEnabled(false);
	m_openValve->setUIEnabled(false);
	m_closeValve->setUIEnabled(false);
	m_control->setUIEnabled(false);

	m_lsnTarget.reset();
	m_lsnRampTime.reset();
	m_lsnControl.reset();
	m_lsnCloseValve.reset();
	m_lsnOpenValve.reset();
	return NULL;
}
