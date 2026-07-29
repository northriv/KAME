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
#include "lasermodule.h"
#include "ui_lasermoduleform.h"

#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <cmath>
#include <cstdint>

// ===========================================================================================
// XLaserModule::LaserChannel
// ===========================================================================================
XLaserModule::LaserChannel::LaserChannel(const char *name, bool runtime, unsigned int slot,
    const shared_ptr<XLaserModule> &driver) :
    XNode(name, runtime),
    m_slot(slot),
    m_driver(driver),
    // Scalar-entry names must be unique across the whole measurement's entry list; encode slot.
    m_current(create<XScalarEntry>(formatString("Laser%u.Current", slot).c_str(), false,
        dynamic_pointer_cast<XDriver>(driver), "%.4g")), //[mA]
    m_power(create<XScalarEntry>(formatString("Laser%u.Power", slot).c_str(), false,
        dynamic_pointer_cast<XDriver>(driver), "%.4g")), //[mW]
    m_voltage(create<XScalarEntry>(formatString("Laser%u.Voltage", slot).c_str(), false,
        dynamic_pointer_cast<XDriver>(driver), "%.4g")), //[V]
    m_setCurrent(create<XDoubleNode>("SetCurrent", true, "%.4g")),
    m_setPower(create<XDoubleNode>("SetPower", true, "%.4g")),
    m_enabled(create<XBoolNode>("Enabled", true)) {
    // Disabled until the driver opens (reversible setUIEnabled(), never irreversible disable());
    // re-enabled in start(). Nested iterate_commit on this node's own subtree.
    iterate_commit([=](Transaction &tr){
        for(auto &&x: std::vector<shared_ptr<XNode>>{m_setCurrent, m_setPower, m_enabled})
            tr[ *x].setUIEnabled(false);
    });
}
void
XLaserModule::LaserChannel::start() {
    iterate_commit([=](Transaction &tr){
        m_lsnSetCurrent = tr[ *m_setCurrent].onValueChanged().connectWeakly(
            shared_from_this(), &XLaserModule::LaserChannel::onSetCurrentChanged);
        m_lsnSetPower = tr[ *m_setPower].onValueChanged().connectWeakly(
            shared_from_this(), &XLaserModule::LaserChannel::onSetPowerChanged);
        m_lsnEnabled = tr[ *m_enabled].onValueChanged().connectWeakly(
            shared_from_this(), &XLaserModule::LaserChannel::onEnabledChanged);
        for(auto &&x: std::vector<shared_ptr<XNode>>{m_setCurrent, m_setPower, m_enabled})
            tr[ *x].setUIEnabled(true);
    });
}
void
XLaserModule::LaserChannel::stop() {
    iterate_commit([=](Transaction &tr){
        for(auto &&x: std::vector<shared_ptr<XNode>>{m_setCurrent, m_setPower, m_enabled})
            tr[ *x].setUIEnabled(false);
    });
    m_lsnSetCurrent.reset();
    m_lsnSetPower.reset();
    m_lsnEnabled.reset();
}
void
XLaserModule::LaserChannel::onSetCurrentChanged(const Snapshot &shot, XValueNodeBase *) {
    if(auto d = m_driver.lock())
        d->setLaserCurrent(m_slot, (double)shot[ *m_setCurrent]);
}
void
XLaserModule::LaserChannel::onSetPowerChanged(const Snapshot &shot, XValueNodeBase *) {
    if(auto d = m_driver.lock())
        d->setLaserPower(m_slot, (double)shot[ *m_setPower]);
}
void
XLaserModule::LaserChannel::onEnabledChanged(const Snapshot &shot, XValueNodeBase *) {
    if(auto d = m_driver.lock())
        d->setLaserOutput(m_slot, (bool)shot[ *m_enabled]);
}

// ===========================================================================================
// XLaserModule::TecChannel
// ===========================================================================================
XLaserModule::TecChannel::TecChannel(const char *name, bool runtime, unsigned int slot,
    const shared_ptr<XLaserModule> &driver) :
    XNode(name, runtime),
    m_slot(slot),
    m_driver(driver),
    m_temp(create<XScalarEntry>(formatString("TEC%u.Temp", slot).c_str(), false,
        dynamic_pointer_cast<XDriver>(driver), "%.5g")), //[degC]
    m_setTemp(create<XDoubleNode>("SetTemp", true, "%.5g")),
    m_enabled(create<XBoolNode>("Enabled", true)) {
    iterate_commit([=](Transaction &tr){
        for(auto &&x: std::vector<shared_ptr<XNode>>{m_setTemp, m_enabled})
            tr[ *x].setUIEnabled(false);
    });
}
void
XLaserModule::TecChannel::start() {
    iterate_commit([=](Transaction &tr){
        m_lsnSetTemp = tr[ *m_setTemp].onValueChanged().connectWeakly(
            shared_from_this(), &XLaserModule::TecChannel::onSetTempChanged);
        m_lsnEnabled = tr[ *m_enabled].onValueChanged().connectWeakly(
            shared_from_this(), &XLaserModule::TecChannel::onEnabledChanged);
        for(auto &&x: std::vector<shared_ptr<XNode>>{m_setTemp, m_enabled})
            tr[ *x].setUIEnabled(true);
    });
}
void
XLaserModule::TecChannel::stop() {
    iterate_commit([=](Transaction &tr){
        for(auto &&x: std::vector<shared_ptr<XNode>>{m_setTemp, m_enabled})
            tr[ *x].setUIEnabled(false);
    });
    m_lsnSetTemp.reset();
    m_lsnEnabled.reset();
}
void
XLaserModule::TecChannel::onSetTempChanged(const Snapshot &shot, XValueNodeBase *) {
    if(auto d = m_driver.lock())
        d->setTecTemp(m_slot, (double)shot[ *m_setTemp]);
}
void
XLaserModule::TecChannel::onEnabledChanged(const Snapshot &shot, XValueNodeBase *) {
    if(auto d = m_driver.lock())
        d->setTecOutput(m_slot, (bool)shot[ *m_enabled]);
}

// ===========================================================================================
// XLaserModule
// ===========================================================================================
XLaserModule::XLaserModule(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_status(create<XStringNode>("Status", true)),
    m_form(new FrmLaserModule) {
    // The form ships with both panels (Laser/TEC) and 4 pages each; hide everything up front.
    // createLaserChannels()/createTecChannels() reveal only what the concrete driver declares
    // (mirrors XTempControl hiding its unused loop pages).
    m_form->m_grpLaser->hide();
    m_form->m_grpTec->hide();
    m_conUIs.push_back(xqcon_create<XQLabelConnector>(m_status, m_form->m_lblStatus));
    m_form->setWindowTitle(i18n("Laser Module - ") + getLabel());
}
void
XLaserModule::showForms() {
    m_form->showNormal();
    m_form->raise();
}
void
XLaserModule::createLaserChannels(Transaction &tr_meas, const shared_ptr<XMeasure> &meas, unsigned int n) {
    if(n == 0)
        return; //panel stays hidden.
    if(n > 4) n = 4; //the form provides 4 pages.
    m_form->m_grpLaser->show();
    auto self = dynamic_pointer_cast<XLaserModule>(shared_from_this());
    for(unsigned int slot = 1; slot <= n; ++slot) {
        shared_ptr<LaserChannel> ch;
        iterate_commit([&, slot](Transaction &tr){
            ch = create<LaserChannel>(tr, formatString("Laser%u", slot).c_str(), false, slot, self);
        });
        m_laserChannels.push_back(ch);
        meas->scalarEntries()->insert(tr_meas, ch->current());
        meas->scalarEntries()->insert(tr_meas, ch->power());
        meas->scalarEntries()->insert(tr_meas, ch->voltage());
    }
    // Drop the unused channels' tabs. Two steps are needed (matching XTempControl): removeItem()
    // detaches the tab, but the page widget it evicts is left parented-and-visible, so it must
    // also be hidden. Remove from the highest index down so the remaining items don't reindex.
    QWidget *pages[4] = {m_form->m_pageLaser1, m_form->m_pageLaser2, m_form->m_pageLaser3, m_form->m_pageLaser4};
    for(int i = 3; i >= (int)n; --i) {
        m_form->m_toolBoxLaser->removeItem(i);
        pages[i]->hide();
    }
    if(n >= 1) m_conUIs.insert(m_conUIs.end(), {
        xqcon_create<XQLCDNumberConnector>(laserChannel(1)->current()->value(), m_form->m_lcdCurrentL1),
        xqcon_create<XQLCDNumberConnector>(laserChannel(1)->power()->value(), m_form->m_lcdPowerL1),
        xqcon_create<XQLCDNumberConnector>(laserChannel(1)->voltage()->value(), m_form->m_lcdVoltageL1),
        xqcon_create<XQLineEditConnector>(laserChannel(1)->setCurrent(), m_form->m_edSetCurrentL1),
        xqcon_create<XQLineEditConnector>(laserChannel(1)->setPower(), m_form->m_edSetPowerL1),
        xqcon_create<XQToggleButtonConnector>(laserChannel(1)->enabled(), m_form->m_ckbLaserL1),
    });
    if(n >= 2) m_conUIs.insert(m_conUIs.end(), {
        xqcon_create<XQLCDNumberConnector>(laserChannel(2)->current()->value(), m_form->m_lcdCurrentL2),
        xqcon_create<XQLCDNumberConnector>(laserChannel(2)->power()->value(), m_form->m_lcdPowerL2),
        xqcon_create<XQLCDNumberConnector>(laserChannel(2)->voltage()->value(), m_form->m_lcdVoltageL2),
        xqcon_create<XQLineEditConnector>(laserChannel(2)->setCurrent(), m_form->m_edSetCurrentL2),
        xqcon_create<XQLineEditConnector>(laserChannel(2)->setPower(), m_form->m_edSetPowerL2),
        xqcon_create<XQToggleButtonConnector>(laserChannel(2)->enabled(), m_form->m_ckbLaserL2),
    });
    if(n >= 3) m_conUIs.insert(m_conUIs.end(), {
        xqcon_create<XQLCDNumberConnector>(laserChannel(3)->current()->value(), m_form->m_lcdCurrentL3),
        xqcon_create<XQLCDNumberConnector>(laserChannel(3)->power()->value(), m_form->m_lcdPowerL3),
        xqcon_create<XQLCDNumberConnector>(laserChannel(3)->voltage()->value(), m_form->m_lcdVoltageL3),
        xqcon_create<XQLineEditConnector>(laserChannel(3)->setCurrent(), m_form->m_edSetCurrentL3),
        xqcon_create<XQLineEditConnector>(laserChannel(3)->setPower(), m_form->m_edSetPowerL3),
        xqcon_create<XQToggleButtonConnector>(laserChannel(3)->enabled(), m_form->m_ckbLaserL3),
    });
    if(n >= 4) m_conUIs.insert(m_conUIs.end(), {
        xqcon_create<XQLCDNumberConnector>(laserChannel(4)->current()->value(), m_form->m_lcdCurrentL4),
        xqcon_create<XQLCDNumberConnector>(laserChannel(4)->power()->value(), m_form->m_lcdPowerL4),
        xqcon_create<XQLCDNumberConnector>(laserChannel(4)->voltage()->value(), m_form->m_lcdVoltageL4),
        xqcon_create<XQLineEditConnector>(laserChannel(4)->setCurrent(), m_form->m_edSetCurrentL4),
        xqcon_create<XQLineEditConnector>(laserChannel(4)->setPower(), m_form->m_edSetPowerL4),
        xqcon_create<XQToggleButtonConnector>(laserChannel(4)->enabled(), m_form->m_ckbLaserL4),
    });
}
void
XLaserModule::createTecChannels(Transaction &tr_meas, const shared_ptr<XMeasure> &meas, unsigned int n) {
    if(n == 0)
        return;
    if(n > 4) n = 4;
    m_form->m_grpTec->show();
    auto self = dynamic_pointer_cast<XLaserModule>(shared_from_this());
    for(unsigned int slot = 1; slot <= n; ++slot) {
        shared_ptr<TecChannel> ch;
        iterate_commit([&, slot](Transaction &tr){
            ch = create<TecChannel>(tr, formatString("Tec%u", slot).c_str(), false, slot, self);
        });
        m_tecChannels.push_back(ch);
        meas->scalarEntries()->insert(tr_meas, ch->temp());
    }
    // Drop unused tabs AND hide the evicted page widgets (see createLaserChannels()).
    QWidget *pages[4] = {m_form->m_pageTec1, m_form->m_pageTec2, m_form->m_pageTec3, m_form->m_pageTec4};
    for(int i = 3; i >= (int)n; --i) {
        m_form->m_toolBoxTec->removeItem(i);
        pages[i]->hide();
    }
    if(n >= 1) m_conUIs.insert(m_conUIs.end(), {
        xqcon_create<XQLCDNumberConnector>(tecChannel(1)->temp()->value(), m_form->m_lcdTempT1),
        xqcon_create<XQLineEditConnector>(tecChannel(1)->setTemp(), m_form->m_edSetTempT1),
        xqcon_create<XQToggleButtonConnector>(tecChannel(1)->enabled(), m_form->m_ckbTecT1),
    });
    if(n >= 2) m_conUIs.insert(m_conUIs.end(), {
        xqcon_create<XQLCDNumberConnector>(tecChannel(2)->temp()->value(), m_form->m_lcdTempT2),
        xqcon_create<XQLineEditConnector>(tecChannel(2)->setTemp(), m_form->m_edSetTempT2),
        xqcon_create<XQToggleButtonConnector>(tecChannel(2)->enabled(), m_form->m_ckbTecT2),
    });
    if(n >= 3) m_conUIs.insert(m_conUIs.end(), {
        xqcon_create<XQLCDNumberConnector>(tecChannel(3)->temp()->value(), m_form->m_lcdTempT3),
        xqcon_create<XQLineEditConnector>(tecChannel(3)->setTemp(), m_form->m_edSetTempT3),
        xqcon_create<XQToggleButtonConnector>(tecChannel(3)->enabled(), m_form->m_ckbTecT3),
    });
    if(n >= 4) m_conUIs.insert(m_conUIs.end(), {
        xqcon_create<XQLCDNumberConnector>(tecChannel(4)->temp()->value(), m_form->m_lcdTempT4),
        xqcon_create<XQLineEditConnector>(tecChannel(4)->setTemp(), m_form->m_edSetTempT4),
        xqcon_create<XQToggleButtonConnector>(tecChannel(4)->enabled(), m_form->m_ckbTecT4),
    });
}
void
XLaserModule::analyzeRaw(RawDataReader &reader, Transaction &tr) {
    // Symmetric with execute()'s push order (m_laserChannels then m_tecChannels then status).
    // Record each scalar only when the slot is present and the field is finite.
    for(auto &&ch: m_laserChannels) {
        char present = reader.pop<char>();
        double current = reader.pop<double>();
        double power = reader.pop<double>();
        double voltage = reader.pop<double>();
        if(present) {
            if(std::isfinite(current)) ch->current()->value(tr, current);
            if(std::isfinite(power)) ch->power()->value(tr, power);
            if(std::isfinite(voltage)) ch->voltage()->value(tr, voltage);
        }
    }
    for(auto &&ch: m_tecChannels) {
        char present = reader.pop<char>();
        double temp = reader.pop<double>();
        if(present && std::isfinite(temp))
            ch->temp()->value(tr, temp);
    }
    uint32_t len = reader.pop<uint32_t>();
    std::string s;
    s.resize(len);
    for(uint32_t i = 0; i < len; ++i)
        s[i] = reader.pop<char>();
    tr[ *m_status] = XString(s);
}
void
XLaserModule::visualize(const Snapshot &shot) {
}
void *
XLaserModule::execute(const atomic<bool> &terminated) {
    for(auto &&ch: m_laserChannels)
        ch->start();
    for(auto &&ch: m_tecChannels)
        ch->start();

    while( !terminated) {
        XTime time_awared = XTime::now();
        auto writer = std::make_shared<RawData>();
        // All instrument I/O (read* hooks over the single shared interface) happens here,
        // OUTSIDE any transaction; the results are serialised into the raw stream and committed
        // by analyzeRaw(). A finishWritingRaw()/iterate_commit() retry must never re-issue
        // hardware commands, so nothing here touches the record transaction.
        struct LR {bool present; double cur, pow, volt; bool on;};
        struct TR {bool present; double temp; bool on;};
        std::deque<LR> lr;
        std::deque<TR> tr_;
        XString errs;
        try {
            for(auto &&ch: m_laserChannels) {
                LR r{false, 0, 0, 0, false};
                r.present = readLaser(ch->slot(), r.cur, r.pow, r.volt, r.on);
                lr.push_back(r);
            }
            for(auto &&ch: m_tecChannels) {
                TR r{false, 0, false};
                r.present = readTec(ch->slot(), r.temp, r.on);
                tr_.push_back(r);
            }
            errs = readErrors();
        }
        catch (XDriver::XSkippedRecordError&) {
            msecsleep(100);
            continue;
        }
        catch (XKameError &e) {
            e.print(getLabel());
            msecsleep(100);
            continue;
        }

        XString status;
        for(size_t i = 0; i < lr.size(); ++i)
            if(lr[i].present)
                status += formatString("L%u:%s ", m_laserChannels[i]->slot(), lr[i].on ? "ON" : "OFF");
        for(size_t i = 0; i < tr_.size(); ++i)
            if(tr_[i].present)
                status += formatString("T%u:%s ", m_tecChannels[i]->slot(), tr_[i].on ? "ON" : "OFF");
        if( !errs.empty())
            status += XString("Err-") + errs;

        for(auto &&r: lr) {
            writer->push((char)(r.present ? 1 : 0));
            writer->push(r.cur);
            writer->push(r.pow);
            writer->push(r.volt);
        }
        for(auto &&r: tr_) {
            writer->push((char)(r.present ? 1 : 0));
            writer->push(r.temp);
        }
        writer->push((uint32_t)status.size());
        for(char c: status)
            writer->push(c);
        finishWritingRaw(writer, time_awared, XTime::now());
    }

    for(auto &&ch: m_laserChannels)
        ch->stop();
    for(auto &&ch: m_tecChannels)
        ch->stop();
    return NULL;
}
