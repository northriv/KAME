/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
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
#include "forms/funcsynthform.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <qpushbutton.h>
#include <qstatusbar.h>
#include <qcheckbox.h>

XFuncSynth::XFuncSynth(const char *name, bool runtime, 
					   const shared_ptr<XScalarEntryList> &scalarentries,
					   const shared_ptr<XInterfaceList> &interfaces,
					   const shared_ptr<XThermometerList> &thermometers,
					   const shared_ptr<XDriverList> &drivers) : 
    XPrimaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
    m_output(create<XBoolNode>("Output", true)),
    m_trig(create<XNode>("Trigger", true)),
    m_mode(create<XComboNode>("Mode", false)),
    m_freq(create<XDoubleNode>("Freq", false)),
    m_function(create<XComboNode>("Function", false)),
    m_amp(create<XDoubleNode>("Amplitude", false)),
    m_phase(create<XDoubleNode>("Phase", false)),
    m_offset(create<XDoubleNode>("Offset", false)),
    m_form(new FrmFuncSynth(g_pFrmMain))
{
	m_form->statusBar()->hide();
	m_form->setCaption(KAME::i18n("Func. Synth. - ") + getLabel() );

	m_conOutput = xqcon_create<XQToggleButtonConnector>(m_output, m_form->m_ckbOutput);
	m_conTrig = xqcon_create<XQButtonConnector>(m_trig, m_form->m_btnTrig);
	m_conMode = xqcon_create<XQComboBoxConnector>(m_mode, m_form->m_cmbMode);
	m_conFreq = xqcon_create<XQLineEditConnector>(m_freq, m_form->m_edFreq);
	m_conFunction = xqcon_create<XQComboBoxConnector>(m_function, m_form->m_cmbFunc);
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
    m_form->show();
    m_form->raise();
}

void
XFuncSynth::start()
{
    m_output->setUIEnabled(true);
    m_trig->setUIEnabled(true);
    m_mode->setUIEnabled(true);
    m_freq->setUIEnabled(true);
    m_function->setUIEnabled(true);
    m_amp->setUIEnabled(true);
    m_phase->setUIEnabled(true);
    m_offset->setUIEnabled(true);
        
	m_lsnOutput = output()->onValueChanged().connectWeak(
		shared_from_this(), &XFuncSynth::onOutputChanged);
	m_lsnTrig = trig()->onTouch().connectWeak(
		shared_from_this(), &XFuncSynth::onTrigTouched);
	m_lsnMode = mode()->onValueChanged().connectWeak(
		shared_from_this(), &XFuncSynth::onModeChanged);
	m_lsnFreq = freq()->onValueChanged().connectWeak(
		shared_from_this(), &XFuncSynth::onFreqChanged);
	m_lsnFunction = function()->onValueChanged().connectWeak(
		shared_from_this(), &XFuncSynth::onFunctionChanged);
	m_lsnAmp = amp()->onValueChanged().connectWeak(
		shared_from_this(), &XFuncSynth::onAmpChanged);
	m_lsnPhase = phase()->onValueChanged().connectWeak(
		shared_from_this(), &XFuncSynth::onPhaseChanged);
	m_lsnOffset = offset()->onValueChanged().connectWeak(
		shared_from_this(), &XFuncSynth::onOffsetChanged);
}
void
XFuncSynth::stop()
{  
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

	afterStop();
}

void
XFuncSynth::analyzeRaw() throw (XRecordError&)
{
}
void
XFuncSynth::visualize()
{
	//! impliment extra codes which do not need write-lock of record
	//! record is read-locked
}

