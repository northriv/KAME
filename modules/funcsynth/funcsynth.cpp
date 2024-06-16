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
#include "funcsynth.h"
#include "ui_funcsynthform.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <QPushButton>
#include <qstatusbar.h>
#include <QCheckBox>

XFuncSynth::XFuncSynth(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriver(name, runtime, ref(tr_meas), meas),
    m_output(create<XBoolNode>("Output", true)),
    m_trig(create<XTouchableNode>("Trigger", true)),
    m_mode(create<XComboNode>("Mode", false)),
    m_function(create<XComboNode>("Function", false)),
    m_freq(create<XDoubleNode>("Freq", false)),
    m_amp(create<XDoubleNode>("Amplitude", false)),
    m_phase(create<XDoubleNode>("Phase", false)),
    m_offset(create<XDoubleNode>("Offset", false)),
    m_form(new FrmFuncSynth) {

	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("Func. Synth. - ") + getLabel() );

	m_conOutput = xqcon_create<XQToggleButtonConnector>(m_output, m_form->m_ckbOutput);
	m_conTrig = xqcon_create<XQButtonConnector>(m_trig, m_form->m_btnTrig);
	m_conMode = xqcon_create<XQComboBoxConnector>(m_mode, m_form->m_cmbMode, Snapshot( *m_mode));
	m_conFreq = xqcon_create<XQLineEditConnector>(m_freq, m_form->m_edFreq);
	m_conFunction = xqcon_create<XQComboBoxConnector>(m_function, m_form->m_cmbFunc, Snapshot( *m_function));
	m_conAmp = xqcon_create<XQLineEditConnector>(m_amp, m_form->m_edAmp);
	m_conPhase = xqcon_create<XQLineEditConnector>(m_phase, m_form->m_edPhase);
	m_conOffset = xqcon_create<XQLineEditConnector>(m_offset, m_form->m_edOffset);

    m_output->setUIEnabled(false);
    m_trig->setUIEnabled(false);
    m_mode->setUIEnabled(false);
    m_freq->setUIEnabled(false);
    m_function->setUIEnabled(false);
    m_amp->setUIEnabled(false);
    m_phase->setUIEnabled(false);
    m_offset->setUIEnabled(false);
}

void
XFuncSynth::showForms() {
//! impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}

void
XFuncSynth::start() {
    m_output->setUIEnabled(true);
    m_trig->setUIEnabled(true);
    m_mode->setUIEnabled(true);
    m_freq->setUIEnabled(true);
    m_function->setUIEnabled(true);
    m_amp->setUIEnabled(true);
    m_phase->setUIEnabled(true);
    m_offset->setUIEnabled(true);
        
	iterate_commit([=](Transaction &tr){
		m_lsnOutput = tr[ *output()].onValueChanged().connectWeakly(
			shared_from_this(), &XFuncSynth::onOutputChanged);
		m_lsnMode = tr[ *mode()].onValueChanged().connectWeakly(
			shared_from_this(), &XFuncSynth::onModeChanged);
		m_lsnFreq = tr[ *freq()].onValueChanged().connectWeakly(
			shared_from_this(), &XFuncSynth::onFreqChanged);
		m_lsnFunction = tr[ *function()].onValueChanged().connectWeakly(
			shared_from_this(), &XFuncSynth::onFunctionChanged);
		m_lsnAmp = tr[ *amp()].onValueChanged().connectWeakly(
			shared_from_this(), &XFuncSynth::onAmpChanged);
		m_lsnPhase = tr[ *phase()].onValueChanged().connectWeakly(
			shared_from_this(), &XFuncSynth::onPhaseChanged);
		m_lsnOffset = tr[ *offset()].onValueChanged().connectWeakly(
			shared_from_this(), &XFuncSynth::onOffsetChanged);
    });

	iterate_commit([=](Transaction &tr){
		m_lsnTrig = tr[ *trig()].onTouch().connectWeakly(
			shared_from_this(), &XFuncSynth::onTrigTouched);
    });
}
void
XFuncSynth::stop() {
	m_lsnOutput.reset();
	m_lsnTrig.reset();
	m_lsnMode.reset();
	m_lsnFreq.reset();
	m_lsnFunction.reset();
	m_lsnAmp.reset();
	m_lsnPhase.reset();
	m_lsnOffset.reset();
  
    m_output->setUIEnabled(false);
    m_trig->setUIEnabled(false);
    m_mode->setUIEnabled(false);
    m_freq->setUIEnabled(false);
    m_function->setUIEnabled(false);
    m_amp->setUIEnabled(false);
    m_phase->setUIEnabled(false);
    m_offset->setUIEnabled(false);

	closeInterface();
}

void
XFuncSynth::analyzeRaw(RawDataReader &reader, Transaction &tr) {
}
void
XFuncSynth::visualize(const Snapshot &shot) {
}

