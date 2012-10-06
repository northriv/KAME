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
#include "ui_lockinampform.h"
#include "lockinamp.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <QStatusBar>

XLIA::XLIA(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_valueX(create<XScalarEntry>("ValueX", false, 
								  dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_valueY(create<XScalarEntry>("ValueY", false, 
								  dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_output(create<XDoubleNode>("Output", false)),
    m_frequency(create<XDoubleNode>("Frequency", false)),
    m_sensitivity(create<XComboNode>("Sensitivity", false, true)),
    m_timeConst(create<XComboNode>("TimeConst", false, true)),
    m_autoScaleX(create<XBoolNode>("AutoScaleX", false)),
    m_autoScaleY(create<XBoolNode>("AutoScaleY", false)),
    m_fetchFreq(create<XDoubleNode>("FetchFreq", false)),
    m_form(new FrmLIA(g_pFrmMain)) {
	for(Transaction tr( *this);; ++tr) {
		tr[ *fetchFreq()] = 1;
		if(tr.commit())
			break;
	}
  
	meas->scalarEntries()->insert(tr_meas, m_valueX);
	meas->scalarEntries()->insert(tr_meas, m_valueY);

	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("Lock-in-Amp - ") + getLabel() );

	m_output->setUIEnabled(false);
	m_frequency->setUIEnabled(false);
	m_sensitivity->setUIEnabled(false);
	m_timeConst->setUIEnabled(false);
	m_autoScaleX->setUIEnabled(false);
	m_autoScaleY->setUIEnabled(false);
	m_fetchFreq->setUIEnabled(false);

	m_conSens = xqcon_create<XQComboBoxConnector>(m_sensitivity, m_form->m_cmbSens, Snapshot( *m_sensitivity));
	m_conTimeConst = xqcon_create<XQComboBoxConnector>(m_timeConst, m_form->m_cmbTimeConst, Snapshot( *m_timeConst));
	m_conFreq = xqcon_create<XQLineEditConnector>(m_frequency, m_form->m_edFreq);
	m_conOutput = xqcon_create<XQLineEditConnector>(m_output, m_form->m_edOutput);
	m_conAutoScaleX = xqcon_create<XQToggleButtonConnector>(m_autoScaleX, m_form->m_ckbAutoScaleX);
	m_conAutoScaleY = xqcon_create<XQToggleButtonConnector>(m_autoScaleY, m_form->m_ckbAutoScaleY);
	m_conFetchFreq = xqcon_create<XQLineEditConnector>(m_fetchFreq, m_form->m_edFetchFreq);
}

void
XLIA::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}

void
XLIA::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    double x, y;
    x = reader.pop<double>();
    y = reader.pop<double>();
    m_valueX->value(tr, x);
    m_valueY->value(tr, y);
}
void
XLIA::visualize(const Snapshot &shot) {
}

void 
XLIA::onOutputChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        changeOutput(shot[ *output()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while changing output, "));
        return;
    }
}
void 
XLIA::onFreqChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        changeFreq(shot[ *frequency()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while changing frequency, "));
        return;
    }
}
void 
XLIA::onSensitivityChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        changeSensitivity(shot[ *sensitivity()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while changing sensitivity, "));
        return;
    }
}
void 
XLIA::onTimeConstChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
        changeTimeConst(shot[ *timeConst()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error while changing time const., "));
        return;
    }
}

void *
XLIA::execute(const atomic<bool> &terminated) {
	m_output->setUIEnabled(true);
	m_frequency->setUIEnabled(true);
	m_sensitivity->setUIEnabled(true);
	m_timeConst->setUIEnabled(true);
	m_autoScaleX->setUIEnabled(true);
	m_autoScaleY->setUIEnabled(true);
	m_fetchFreq->setUIEnabled(true);

	for(Transaction tr( *this);; ++tr) {
		m_lsnOutput = tr[ *output()].onValueChanged().connectWeakly(
			shared_from_this(), &XLIA::onOutputChanged);
		m_lsnFreq = tr[ *frequency()].onValueChanged().connectWeakly(
			shared_from_this(), &XLIA::onFreqChanged);
		m_lsnSens = tr[ *sensitivity()].onValueChanged().connectWeakly(
			shared_from_this(), &XLIA::onSensitivityChanged);
		m_lsnTimeConst = tr[ *timeConst()].onValueChanged().connectWeakly(
			shared_from_this(), &XLIA::onTimeConstChanged);
		if(tr.commit())
			break;
	}

	while( !terminated) {
		double fetch_freq = ***fetchFreq();
		double wait = 0;
		if(fetch_freq > 0) {
			sscanf(( **timeConst())->to_str().c_str(), "%lf", &wait);
			wait *= 1000.0 / fetch_freq;
		}
		if(wait > 0) msecsleep(lrint(wait));
      
		double x, y;
		XTime time_awared = XTime::now();
		// try/catch exception of communication errors
		try {
			get(&x, &y);
		}
		catch (XKameError &e) {
			e.print(getLabel() + " " + i18n("Read Error, "));
			continue;
		}
		shared_ptr<RawData> writer(new RawData);
		writer->push(x);
		writer->push(y);
		finishWritingRaw(writer, time_awared, XTime::now());
	}
	m_output->setUIEnabled(false);
	m_frequency->setUIEnabled(false);
	m_sensitivity->setUIEnabled(false);
	m_timeConst->setUIEnabled(false);
	m_autoScaleX->setUIEnabled(false);
	m_autoScaleY->setUIEnabled(false);
	m_fetchFreq->setUIEnabled(false);

	m_lsnOutput.reset();
	m_lsnFreq.reset();
	m_lsnSens.reset();
	m_lsnTimeConst.reset();
	return NULL;
}
