/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#include "ui_relayform.h"
#include "relaydriver.h"
#include "xnodeconnector.h"
#include <QStatusBar>
#include <QCheckBox>
#include <QComboBox>
#include <QLabel>

XRelayDriver::XRelayDriver(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas,
    unsigned int num_channels) :
    XPrimaryDriver(name, runtime, ref(tr_meas), meas),
    m_form(new FrmRelay) {
    m_form->statusBar()->hide();
    m_form->setWindowTitle(i18n("Relay Controller - ") + getLabel() );

    //Master-device selector is used only by adapter drivers, e.g. XRelayViaSTM.
    m_form->m_lblMaster->setEnabled(false);
    m_form->m_cmbMaster->setEnabled(false);

    QCheckBox *ckbs[maxNumChannels] = {
        m_form->m_ckbCh1, m_form->m_ckbCh2, m_form->m_ckbCh3, m_form->m_ckbCh4,
        m_form->m_ckbCh5, m_form->m_ckbCh6, m_form->m_ckbCh7, m_form->m_ckbCh8};
    num_channels = std::min(num_channels, maxNumChannels);
    for(unsigned int i = 0; i < maxNumChannels; ++i) {
        if(i < num_channels) {
            auto ch = create<XBoolNode>(formatString("Channel%u", i + 1).c_str(), true);
            ch->setUIEnabled(false);
            m_conChannels.push_back(xqcon_create<XQToggleButtonConnector>(ch, ckbs[i]));
            m_channelOutputs.push_back(ch);
        }
        else
            ckbs[i]->setEnabled(false);
    }
}

void
XRelayDriver::showForms() {
    m_form->showNormal();
    m_form->raise();
}

void
XRelayDriver::start() {
    for(auto &ch: m_channelOutputs)
        ch->setUIEnabled(true);

    iterate_commit([=](Transaction &tr){
        m_lsnOutputs.clear();
        for(auto &ch: m_channelOutputs)
            m_lsnOutputs.push_back(tr[ *ch].onValueChanged().connectWeakly(
                shared_from_this(), &XRelayDriver::onOutputChanged));
    });
    //Syncs UI with the hardware states if readback is supported.
    iterate_commit([=](Transaction &tr){
        queryStatus(tr);
        for(auto &lsn: m_lsnOutputs)
            tr.unmark(lsn);
    });
}
void
XRelayDriver::stop() {
    m_lsnOutputs.clear();

    for(auto &ch: m_channelOutputs)
        ch->setUIEnabled(false);

    closeInterface();
}

void
XRelayDriver::analyzeRaw(RawDataReader &reader, Transaction &tr) {
    tr[ *this].m_bits = (unsigned int)(uint32_t)reader.pop<int32_t>();
}
void
XRelayDriver::visualize(const Snapshot &shot) {
}

void
XRelayDriver::finish(const XTime &time_awared) {
    int32_t bits = 0;
    Snapshot shot( *this);
    for(unsigned int i = 0; i < numChannels(); ++i)
        if(shot[ *channelOutput(i)])
            bits |= 1 << i;
    auto writer = std::make_shared<RawData>();
    writer->push(bits);
    finishWritingRaw(writer, time_awared, XTime::now());
}
void
XRelayDriver::onOutputChanged(const Snapshot &shot, XValueNodeBase *node) {
    XTime time_awared(XTime::now());
    unsigned int ch;
    for(ch = 0; ch < numChannels(); ++ch)
        if(m_channelOutputs[ch].get() == node)
            break;
    if(ch >= numChannels())
        return;
    try {
        changeOutput(ch, shot[ *channelOutput(ch)]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + i18n(": Error while changing output, "));
        return;
    }
    finish(time_awared);
}
