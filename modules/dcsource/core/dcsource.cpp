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
#include "ui_dcsourceform.h"
#include "dcsource.h"
#include "xnodeconnector.h"
#include <QStatusBar>
#include <QPushButton>
#include <QCheckBox>

XDCSource::XDCSource(const char *name, bool runtime, 
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriver(name, runtime, ref(tr_meas), meas),
    m_entryValue(create<XScalarEntry>("EntryValue", false,
                                 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.8f")),
    m_function(create<XComboNode>("Function", false)),
    m_output(create<XBoolNode>("Output", true)),
    m_value(create<XDoubleNode>("Value", false)),
    m_channel(create<XComboNode>("Channel", false, true)),
    m_range(create<XComboNode>("Range", false, true)),
    m_form(new FrmDCSource(g_pFrmMain)) {
	m_form->statusBar()->hide();
	m_form->setWindowTitle(i18n("DC Source - ") + getLabel() );

    meas->scalarEntries()->insert(tr_meas, m_entryValue);

	m_output->setUIEnabled(false);
	m_function->setUIEnabled(false);
	m_value->setUIEnabled(false);
	m_channel->setUIEnabled(false);
	m_range->setUIEnabled(false);

	m_conFunction = xqcon_create<XQComboBoxConnector>(m_function, m_form->m_cmbFunction, Snapshot( *m_function));
	m_conOutput = xqcon_create<XQToggleButtonConnector>(m_output, m_form->m_ckbOutput);
	m_conValue = xqcon_create<XQLineEditConnector>(m_value, m_form->m_edValue);
	m_conChannel = xqcon_create<XQComboBoxConnector>(m_channel, m_form->m_cmbChannel, Snapshot( *m_channel));
	m_conRange = xqcon_create<XQComboBoxConnector>(m_range, m_form->m_cmbRange, Snapshot( *m_range));
}

void
XDCSource::showForms() {
//! impliment form->show() here
    m_form->show();
    m_form->raise();
}

void
XDCSource::start() {
	m_output->setUIEnabled(true);
	m_function->setUIEnabled(true);
	m_value->setUIEnabled(true);
	m_channel->setUIEnabled(true);
	m_range->setUIEnabled(true);

	iterate_commit([=](Transaction &tr){
		m_lsnOutput = tr[ *output()].onValueChanged().connectWeakly(
							  shared_from_this(), &XDCSource::onOutputChanged);
		m_lsnFunction = tr[ *function()].onValueChanged().connectWeakly(
							  shared_from_this(), &XDCSource::onFunctionChanged);
		m_lsnValue = tr[ *value()].onValueChanged().connectWeakly(
							shared_from_this(), &XDCSource::onValueChanged);
		m_lsnChannel = tr[ *channel()].onValueChanged().connectWeakly(
							  shared_from_this(), &XDCSource::onChannelChanged);
		m_lsnRange = tr[ *range()].onValueChanged().connectWeakly(
							  shared_from_this(), &XDCSource::onRangeChanged);
    });
	updateStatus();
}
void
XDCSource::stop() {
	m_lsnChannel.reset();
	m_lsnOutput.reset();
	m_lsnFunction.reset();
	m_lsnValue.reset();

	m_output->setUIEnabled(false);
	m_function->setUIEnabled(false);
	m_value->setUIEnabled(false);
	m_channel->setUIEnabled(false);
	m_range->setUIEnabled(false);

	closeInterface();
}

void
XDCSource::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&)  {
    m_entryValue->value(tr, tr[ *value()]);
}
void
XDCSource::visualize(const Snapshot &shot) {
}

void
XDCSource::finish(const XTime &time_awared) {
	auto writer = std::make_shared<RawData>();
	if( !Snapshot( *this)[ *output()])
		finishWritingRaw(writer, XTime(), XTime());
	else
		finishWritingRaw(writer, time_awared, XTime::now());
}
void 
XDCSource::onOutputChanged(const Snapshot &shot, XValueNodeBase *) {
	XTime time_awared(XTime::now());
	int ch = ***channel();
    try {
        changeOutput(ch, shot[ *output()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + i18n(": Error while changing output, "));
        return;
    }
    finish(time_awared);
}
void 
XDCSource::onFunctionChanged(const Snapshot &shot, XValueNodeBase *) {
	XTime time_awared(XTime::now());
	int ch = ***channel();
    try {
        changeFunction(ch, shot[ *function()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + i18n(": Error while changing function, "));
        return;
    }
    finish(time_awared);
}
void 
XDCSource::onValueChanged(const Snapshot &shot, XValueNodeBase *) {
	XTime time_awared(XTime::now());
	int ch = ***channel();
    try {
        changeValue(ch, shot[ *value()], true);
    }
    catch (XKameError& e) {
        e.print(getLabel() + i18n(": Error while changing value, "));
        return;
    }
    finish(time_awared);
}
void 
XDCSource::onRangeChanged(const Snapshot &shot, XValueNodeBase *) {
	XTime time_awared(XTime::now());
	int ch = ***channel();
    try {
        changeRange(ch, shot[ *range()]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + i18n(": Error while changing value, "));
        return;
    }
    finish(time_awared);
}
void 
XDCSource::onChannelChanged(const Snapshot &shot, XValueNodeBase *) {
    try {
    	iterate_commit([=](Transaction &tr){
    		int ch = tr[ *channel()];
            queryStatus(tr, ch);
            tr.unmark(m_lsnOutput);
            tr.unmark(m_lsnFunction);
            tr.unmark(m_lsnRange);
            tr.unmark(m_lsnValue);
        });
    }
    catch (XKameError& e) {
        e.print(getLabel());
        return;
    }
}
