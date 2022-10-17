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
//---------------------------------------------------------------------------
#include "ui_motorform.h"
#include "motor.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <QStatusBar>

XMotorDriver::XMotorDriver(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_position(create<XScalarEntry>("Position", false,
								  dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_target(create<XDoubleNode>("Target", true)),
    m_stepMotor(create<XUIntNode>("StepMotor", true)),
    m_stepEncoder(create<XUIntNode>("StepEncoder", true)),
    m_currentStopping(create<XDoubleNode>("CurrentStopping", true)),
    m_currentRunning(create<XDoubleNode>("CurrentRunning", true)),
    m_speed(create<XDoubleNode>("Speed", true)),
    m_timeAcc(create<XDoubleNode>("TimeAcc", true)),
    m_timeDec(create<XDoubleNode>("TimeDec", true)),
    m_active(create<XBoolNode>("Active", true)),
    m_ready(create<XBoolNode>("Ready", true)),
    m_slipping(create<XBoolNode>("Slipping", true)),
    m_microStep(create<XBoolNode>("MicroStep", true)),
    m_hasEncoder(create<XBoolNode>("HasEncoder", false)),
    m_pushing(create<XBoolNode>("PushingMode", true)),
    m_auxBits(create<XHexNode>("AUXBits", false)),
    m_clear(create<XTouchableNode>("Clear", true)),
    m_store(create<XTouchableNode>("Store", true)),
    m_round(create<XBoolNode>("Round", true)),
    m_roundBy(create<XUIntNode>("RoundBy", true)),
    m_forwardMotor(create<XTouchableNode>("ForwardMotor", true)),
    m_reverseMotor(create<XTouchableNode>("ReverseMotor", true)),
    m_stopMotor(create<XTouchableNode>("StopMotor", true)),
    m_form(new FrmMotorDriver) {

	iterate_commit([=](Transaction &tr){
		tr[ *active()] = false;
    });

	meas->scalarEntries()->insert(tr_meas, m_position);

	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("Motor Driver - ") + getLabel() );

	m_position->setUIEnabled(false);
	m_target->setUIEnabled(false);
	m_stepMotor->setUIEnabled(false);
	m_stepEncoder->setUIEnabled(false);
	m_currentStopping->setUIEnabled(false);
	m_currentRunning->setUIEnabled(false);
	m_speed->setUIEnabled(false);
	m_timeAcc->setUIEnabled(false);
	m_timeDec->setUIEnabled(false);
	m_active->setUIEnabled(false);
	m_ready->setUIEnabled(false);
	m_slipping->setUIEnabled(false);
	m_microStep->setUIEnabled(false);
	m_auxBits->setUIEnabled(false);
	m_clear->setUIEnabled(false);
	m_store->setUIEnabled(false);
	m_round->setUIEnabled(false);
	m_roundBy->setUIEnabled(false);
	m_forwardMotor->setUIEnabled(false);
	m_reverseMotor->setUIEnabled(false);
	m_stopMotor->setUIEnabled(false);
    m_pushing->setUIEnabled(false);
//	m_hasEncoder->setUIEnabled(true);

    m_conUIs = {
        xqcon_create<XQLCDNumberConnector>(m_position->value(), m_form->m_lcdPosition),
        xqcon_create<XQLineEditConnector>(m_target, m_form->m_edTarget),
        xqcon_create<XQLineEditConnector>(m_stepMotor, m_form->m_edStepMotor),
        xqcon_create<XQLineEditConnector>(m_stepEncoder, m_form->m_edStepEncoder),
        xqcon_create<XQLineEditConnector>(m_currentStopping, m_form->m_edCurrStopping),
        xqcon_create<XQLineEditConnector>(m_currentRunning, m_form->m_edCurrRunning),
        xqcon_create<XQLineEditConnector>(m_speed, m_form->m_edSpeed),
        xqcon_create<XQLineEditConnector>(m_timeAcc, m_form->m_edTimeAcc),
        xqcon_create<XQLineEditConnector>(m_timeDec, m_form->m_edTimeDec),
        xqcon_create<XQToggleButtonConnector>(m_active, m_form->m_ckbActive),
        xqcon_create<XQToggleButtonConnector>(m_microStep, m_form->m_ckbMicroStepping),
        xqcon_create<XQLedConnector>(m_slipping, m_form->m_ledSlipping),
        xqcon_create<XQLedConnector>(m_ready, m_form->m_ledReady),
        xqcon_create<XQToggleButtonConnector>(m_hasEncoder, m_form->m_ckbHasEncoder),
        xqcon_create<XQToggleButtonConnector>(m_pushing, m_form->m_ckbPushing),
        xqcon_create<XQLineEditConnector>(m_auxBits, m_form->m_edAUXBits),
        xqcon_create<XQButtonConnector>(m_clear, m_form->m_btnClear),
        xqcon_create<XQButtonConnector>(m_store, m_form->m_btnStore),
        xqcon_create<XQToggleButtonConnector>(m_round, m_form->m_ckbRound),
        xqcon_create<XQLineEditConnector>(m_roundBy, m_form->m_edRoundBy),
        xqcon_create<XQButtonConnector>(m_forwardMotor, m_form->m_btnFWD),
        xqcon_create<XQButtonConnector>(m_reverseMotor, m_form->m_btnRVS),
        xqcon_create<XQButtonConnector>(m_stopMotor, m_form->m_btnSTOP),
    };
}

void
XMotorDriver::showForms() {
//! impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}

void
XMotorDriver::analyzeRaw(RawDataReader &reader, Transaction &tr) {
    double pos;
    bool slip, isready;
    pos = reader.pop<double>();
    slip = reader.pop<uint16_t>();
    isready = reader.pop<uint16_t>();
    m_position->value(tr, pos);
    tr[ *m_slipping] = slip;
    if(m_timeMovementStarted > tr[* this].timeAwared())
        isready = false; //Motor started moving after the previous timeAwared().
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
        m_timeMovementStarted = XTime::now();
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
void
XMotorDriver::onClearTouched(const Snapshot &shot, XTouchableNode *) {
	Snapshot shot_this( *this);
    try {
        clearPosition();
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while clearing position, "));
        return;
    }
    iterate_commit([=](Transaction &tr){
    	tr[ *target()] = 0.0;
    	tr.unmark(m_lsnTarget);
    });
}
void
XMotorDriver::onStoreTouched(const Snapshot &shot, XTouchableNode *) {
	Snapshot shot_this( *this);
    try {
        storeToROM();
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while storing to NV, "));
        return;
    }
}
void
XMotorDriver::onAUXChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        setAUXBits(shot[ *auxBits()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error, "));
        return;
    }
}
void
XMotorDriver::onForwardMotorTouched(const Snapshot &shot, XTouchableNode *) {
	Snapshot shot_this( *this);
    try {
        setForward();
        m_timeMovementStarted = XTime::now();
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error, "));
        return;
    }
}
void
XMotorDriver::onReverseMotorTouched(const Snapshot &shot, XTouchableNode *) {
	Snapshot shot_this( *this);
    try {
        setReverse();
        m_timeMovementStarted = XTime::now();
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error, "));
        return;
    }
}
void
XMotorDriver::onStopMotorTouched(const Snapshot &shot, XTouchableNode *) {
	Snapshot shot_this( *this);
    try {
        stopRotation();
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error, "));
        return;
    }
    iterate_commit([=](Transaction &tr){
    	tr[ *target()] = (double)tr[ *(m_position->value())];
    	tr.unmark(m_lsnTarget);
    });
}
void *
XMotorDriver::execute(const atomic<bool> &terminated) {
    m_timeMovementStarted = {};
    getConditions();

	m_position->setUIEnabled(true);
	m_target->setUIEnabled(true);
	m_stepMotor->setUIEnabled(true);
	m_stepEncoder->setUIEnabled(true);
	m_currentStopping->setUIEnabled(true);
	m_currentRunning->setUIEnabled(true);
	m_speed->setUIEnabled(true);
	m_timeAcc->setUIEnabled(true);
	m_timeDec->setUIEnabled(true);
	m_active->setUIEnabled(true);
	m_ready->setUIEnabled(true);
	m_slipping->setUIEnabled(true);
	m_microStep->setUIEnabled(true);
	m_clear->setUIEnabled(true);
	m_store->setUIEnabled(true);
	m_auxBits->setUIEnabled(true);
	m_round->setUIEnabled(true);
	m_roundBy->setUIEnabled(true);
	m_forwardMotor->setUIEnabled(true);
	m_reverseMotor->setUIEnabled(true);
	m_stopMotor->setUIEnabled(true);
    m_pushing->setUIEnabled(true);
//	m_hasEncoder->setUIEnabled(true);

	iterate_commit([=](Transaction &tr){
		m_lsnTarget = tr[ *target()].onValueChanged().connectWeakly(shared_from_this(), &XMotorDriver::onTargetChanged);
		m_lsnConditions = tr[ *stepMotor()].onValueChanged().connectWeakly(shared_from_this(), &XMotorDriver::onConditionsChanged);
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
        tr[ *pushing()].onValueChanged().connect(m_lsnConditions);
        m_lsnClear = tr[ *clear()].onTouch().connectWeakly(shared_from_this(), &XMotorDriver::onClearTouched);
		m_lsnStore = tr[ *store()].onTouch().connectWeakly(shared_from_this(), &XMotorDriver::onStoreTouched);
		m_lsnForwardMotor = tr[ *forwardMotor()].onTouch().connectWeakly(shared_from_this(), &XMotorDriver::onForwardMotorTouched);
		m_lsnReverseMotor = tr[ *reverseMotor()].onTouch().connectWeakly(shared_from_this(), &XMotorDriver::onReverseMotorTouched);
		m_lsnStopMotor = tr[ *stopMotor()].onTouch().connectWeakly(shared_from_this(), &XMotorDriver::onStopMotorTouched);
		m_lsnAUX = tr[ *auxBits()].onValueChanged().connectWeakly(shared_from_this(), &XMotorDriver::onAUXChanged);
    });

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
		auto writer = std::make_shared<RawData>();
		writer->push(pos);
		writer->push((uint16_t)slip);
		writer->push((uint16_t)isready);
		finishWritingRaw(writer, time_awared, XTime::now());
	}

	m_position->setUIEnabled(false);
	m_target->setUIEnabled(false);
	m_stepMotor->setUIEnabled(false);
	m_stepEncoder->setUIEnabled(false);
	m_currentStopping->setUIEnabled(false);
	m_currentRunning->setUIEnabled(false);
	m_speed->setUIEnabled(false);
	m_timeAcc->setUIEnabled(false);
	m_timeDec->setUIEnabled(false);
	m_active->setUIEnabled(false);
	m_ready->setUIEnabled(false);
	m_slipping->setUIEnabled(false);
	m_microStep->setUIEnabled(false);
	m_clear->setUIEnabled(false);
	m_store->setUIEnabled(false);
	m_auxBits->setUIEnabled(false);
	m_round->setUIEnabled(false);
	m_roundBy->setUIEnabled(false);
	m_forwardMotor->setUIEnabled(false);
	m_reverseMotor->setUIEnabled(false);
	m_stopMotor->setUIEnabled(false);
    m_pushing->setUIEnabled(false);
//	m_hasEncoder->setUIEnabled(true);

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
