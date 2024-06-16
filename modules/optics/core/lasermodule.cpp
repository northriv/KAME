/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#include "lasermodule.h"
#include "ui_lasermoduleform.h"

#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"

XLaserModule::XLaserModule(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_temperature(create<XScalarEntry>("Temperature", false,
                                  dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_current(create<XScalarEntry>("Current", false,
                                  dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_status(create<XStringNode>("Status", true)),
    m_enabled(create<XBoolNode>("Enabled", true)),
    m_form(new FrmLaserModule) {

    meas->scalarEntries()->insert(tr_meas, m_temperature);
    meas->scalarEntries()->insert(tr_meas, m_current);

    m_conUIs = {
        xqcon_create<XQLCDNumberConnector>(temperature()->value(), m_form->m_lcdTemperature),
        xqcon_create<XQLCDNumberConnector>(current()->value(), m_form->m_lcdCurrent),
        xqcon_create<XQLabelConnector>(status(), m_form->m_lblStatus),
        xqcon_create<XQToggleButtonConnector>(enabled(), m_form->m_ckbEnabled)
    };

    std::vector<shared_ptr<XNode>> runtime_ui{
        status(), enabled()
    };
    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });
}
void
XLaserModule::showForms() {
// impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}

void
XLaserModule::analyzeRaw(RawDataReader &reader, Transaction &tr)  {
    double temp = reader.pop<double>();
    double curr = reader.pop<double>();
    m_temperature->value(tr, temp);
    m_current->value(tr, curr);
    tr[ *this].m_current = curr;
    tr[ *this].m_temperature = temp;
}
void
XLaserModule::visualize(const Snapshot &shot) {
}

void *
XLaserModule::execute(const atomic<bool> &terminated) {

    std::vector<shared_ptr<XNode>> runtime_ui{
        status(), enabled()
        };

    iterate_commit([=](Transaction &tr){
        m_lsnOnEnabledChanged = tr[ *enabled()].onValueChanged().connectWeakly(
                    shared_from_this(), &XLaserModule::onEnabledChanged);
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(true);
    });

    while( !terminated) {
        XTime time_awared = XTime::now();
        auto writer = std::make_shared<RawData>();
        // try/catch exception of communication errors
        try {
            auto stat = readStatus();
            writer->push(stat.temperature);
            writer->push(stat.current);
        }
        catch (XDriver::XSkippedRecordError&) {
            msecsleep(100);
            continue;
        }
        catch (XKameError &e) {
            e.print(getLabel());
            continue;
        }
        finishWritingRaw(writer, time_awared, XTime::now());
    }

    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });

    m_lsnOnEnabledChanged.reset();
    return NULL;
}
