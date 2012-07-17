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
#include "ui_motorform.h"
#include "motor.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <QStatusBar>

XMotorDriver::XMotorDriver(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriver(name, runtime, ref(tr_meas), meas),
    m_position(create<XScalarEntry>("Position", false,
								  dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_target(create<XDoubleNode>("Target", true)),
    m_step(create<XUIntNode>("Step", true)),
    m_currentStopping(create<XDoubleNode>("CurrentStopping", true)),
    m_currentRunning(create<XDoubleNode>("CurrentRunning", true)),
    m_speed(create<XDoubleNode>("Speed", true)),
    m_timeAcc(create<XDoubleNode>("TimeAcc", true)),
    m_timeDec(create<XDoubleNode>("TimeDec", true)),
    m_active(create<XBoolNode>("Active", true)),
    m_ready(create<XBoolNode>("Ready", true)),
    m_slipping(create<XBoolNode>("Slipping", true)),
    m_microStep(create<XBoolNode>("MicroStep", true)),
    m_form(new FrmMotorDriver(g_pFrmMain)) {

	for(Transaction tr( *this);; ++tr) {
		tr[ *active()] = false;
		if(tr.commit())
			break;
	}

	meas->scalarEntries()->insert(tr_meas, m_position);

	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("Motor Driver - ") + getLabel() );

	m_position->setUIEnabled(false);
	m_target->setUIEnabled(false);
	m_step->setUIEnabled(false);
	m_currentStopping->setUIEnabled(false);
	m_currentRunning->setUIEnabled(false);
	m_speed->setUIEnabled(false);
	m_timeAcc->setUIEnabled(false);
	m_timeDec->setUIEnabled(false);
	m_active->setUIEnabled(false);
	m_ready->setUIEnabled(false);
	m_slipping->setUIEnabled(false);
	m_microStep->setUIEnabled(false);

	m_conPosition = xqcon_create<XQLCDNumberConnector>(m_position->value(), m_form->m_lcdPosition);
	m_conTarget = xqcon_create<XKDoubleNumInputConnector>(m_target, m_form->m_dblTarget);
	m_form->m_dblTarget->setRange(-3600.0, 3600.0, 1.0, true);
	m_conStep = xqcon_create<XQLineEditConnector>(m_step, m_form->m_edStep);
	m_conCurrentStopping = xqcon_create<XQLineEditConnector>(m_currentStopping, m_form->m_edCurrStopping);
	m_conCurrentRunning = xqcon_create<XQLineEditConnector>(m_currentRunning, m_form->m_edCurrRunning);
	m_conSpeed = xqcon_create<XQLineEditConnector>(m_speed, m_form->m_edSpeed);
	m_conTimeAcc = xqcon_create<XQLineEditConnector>(m_timeAcc, m_form->m_edTimeAcc);
	m_conTimeDec = xqcon_create<XQLineEditConnector>(m_timeDec, m_form->m_edTimeDec);
	m_conActive = xqcon_create<XQToggleButtonConnector>(m_active, m_form->m_ckbActive);
	m_conMicroStep = xqcon_create<XQToggleButtonConnector>(m_microStep, m_form->m_ckbMicroStepping);
	m_conSlipping = xqcon_create<XKLedConnector>(m_slipping, m_form->m_ledSlipping);
	m_conReady = xqcon_create<XKLedConnector>(m_ready, m_form->m_ledReady);
}

void
XMotorDriver::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}

void

XMotorDriver::start() {
    m_thread.reset(new XThread<XMotorDriver>(shared_from_this(), &XMotorDriver::execute));
    m_thread->resume();
	m_position->setUIEnabled(true);
	m_target->setUIEnabled(true);
	m_step->setUIEnabled(true);
	m_currentStopping->setUIEnabled(true);
	m_currentRunning->setUIEnabled(true);
	m_speed->setUIEnabled(true);
	m_timeAcc->setUIEnabled(true);
	m_timeDec->setUIEnabled(true);
	m_active->setUIEnabled(true);
	m_ready->setUIEnabled(true);
	m_slipping->setUIEnabled(true);
	m_microStep->setUIEnabled(true);
	for(Transaction tr( *this);; ++tr) {
		getConditions(tr);
		if(tr.commit())
			break;
	}
}
void

XMotorDriver::stop() {
	m_position->setUIEnabled(false);
	m_target->setUIEnabled(false);
	m_step->setUIEnabled(false);
	m_currentStopping->setUIEnabled(false);
	m_currentRunning->setUIEnabled(false);
	m_speed->setUIEnabled(false);
	m_timeAcc->setUIEnabled(false);
	m_timeDec->setUIEnabled(false);
	m_active->setUIEnabled(false);
	m_ready->setUIEnabled(false);
	m_slipping->setUIEnabled(false);
	m_microStep->setUIEnabled(false);
    if(m_thread) m_thread->terminate();
}

void

XMotorDriver::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    double pos;
    bool slip, isready;
    pos = reader.pop<double>();
    slip = reader.pop<bool>();
    isready = reader.pop<bool>();
    m_position->value(tr, pos);
    tr[ *m_slipping] = slip;
    tr[ *m_ready] = isready;
}
void

XMotorDriver::visualize(const Snapshot &shot) {
}

void
XMotorDriver::onTargetChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
    try {
        setTarget(shot_this, shot[ *target()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while changing target, "));
        return;
    }
}
void
XMotorDriver::onConditionsChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
    try {
        changeConditions(shot_this);
        setActive(shot_this[ *active()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while changing conditions, "));
        return;
    }
}

void *
XMotorDriver::execute(const atomic<bool> &terminated) {

	for(Transaction tr( *this);; ++tr) {
		m_lsnTarget = tr[ *target()].onValueChanged().connectWeakly(shared_from_this(), &XMotorDriver::onTargetChanged);
		m_lsnConditions = tr[ *step()].onValueChanged().connectWeakly(shared_from_this(), &XMotorDriver::onConditionsChanged);
		tr[ *currentStopping()].onValueChanged().connect(m_lsnConditions);
		tr[ *currentRunning()].onValueChanged().connect(m_lsnConditions);
		tr[ *speed()].onValueChanged().connect(m_lsnConditions);
		tr[ *timeAcc()].onValueChanged().connect(m_lsnConditions);
		tr[ *timeDec()].onValueChanged().connect(m_lsnConditions);
		tr[ *active()].onValueChanged().connect(m_lsnConditions);
		tr[ *microStep()].onValueChanged().connect(m_lsnConditions);
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
		writer->push(slip);
		writer->push(isready);
		finishWritingRaw(writer, time_awared, XTime::now());
	}

	m_lsnTarget.reset();
	m_lsnConditions.reset();
	afterStop();
	return NULL;
}
