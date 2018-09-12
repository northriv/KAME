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
#include "pumpcontroller.h"
#include "ui_pumpcontrollerform.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <QStatusBar>
#include <QLabel>
#include <QToolBox>


XPumpControl::XPumpControl(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_entryPressure(create<XScalarEntry>("Pressure", false,
        dynamic_pointer_cast<XDriver>(shared_from_this()))),
    m_activate(create<XBoolNode>("Activate", true)),
    m_heating(create<XBoolNode>("Heating", true)),
    m_standby(create<XBoolNode>("Standby", true)),
    m_running(create<XBoolNode>("Running", true)),
    m_warning(create<XBoolNode>("Warning", true)),
    m_error(create<XBoolNode>("Error", true)),
    m_rotationSpeed(create<XDoubleNode>("RotationSpeed", true)),
    m_runtime(create<XDoubleNode>("RotationSpeed", true)),
    m_standbyRotationSpeed(create<XDoubleNode>("StandbyRotationSpeed", true)),
    m_maxDrivePower(create<XDoubleNode>("MaxDriverPower", true)),
    m_form(new FrmPumpControl(g_pFrmMain)) {

    meas->scalarEntries()->insert(tr_meas, m_entryPressure);

    m_conUIs = {
        xqcon_create<XQToggleButtonConnector>(m_activate, m_form->m_ckbActivate),
        xqcon_create<XQToggleButtonConnector>(m_heating, m_form->m_ckbHeat),
        xqcon_create<XQToggleButtonConnector>(m_standby, m_form->m_ckbStanby),
        xqcon_create<XQLedConnector>(m_running, m_form->m_ledPumping),
        xqcon_create<XQLedConnector>(m_warning, m_form->m_ledWarning),
        xqcon_create<XQLedConnector>(m_error, m_form->m_ledError),
        xqcon_create<XQLCDNumberConnector>(m_rotationSpeed, m_form->m_lcdRotationSpeed),
        xqcon_create<XQLCDNumberConnector>(m_entryPressure->value(), m_form->m_lcdPressure),
        xqcon_create<XQLCDNumberConnector>(m_runtime, m_form->m_lcdRuntime),
        xqcon_create<XQLineEditConnector>(m_standbyRotationSpeed, m_form->m_edStanbyRotationSpeed),
        xqcon_create<XQLineEditConnector>(m_maxDrivePower, m_form->m_edMaxDrivePower),
    };

    iterate_commit([=](Transaction &tr){
        std::vector<shared_ptr<XNode>> runtime_ui{
            m_activate, m_heating, m_standby,
            m_standbyRotationSpeed, m_maxDrivePower
        };
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });

    m_form->statusBar()->hide();
    m_form->setWindowTitle(i18n("Pump Controller - ") + getLabel());
}

void XPumpControl::showForms() {
    //! impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}

void XPumpControl::onModeChanged(const Snapshot &, XValueNodeBase *) {
    Snapshot shot( *this);
    try {
        changeMode(shot[ *m_activate], shot[ *m_heating], shot[ *m_standby]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error, "));
        return;
    }
}
void XPumpControl::onStandbyRotationSpeedChanged(const Snapshot &, XValueNodeBase *) {
    Snapshot shot( *this);
    try {
        changeStandbyRotationSpeed(shot[ *m_standbyRotationSpeed]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error, "));
        return;
    }
}
void XPumpControl::onMaxDriverPowerChanged(const Snapshot &, XValueNodeBase *) {
    Snapshot shot( *this);
    try {
        changeMaxDrivePower(shot[ *m_maxDrivePower]);
    }
    catch (XKameError& e) {
        e.print(getLabel() + " " + i18n("Error, "));
        return;
    }
}

void XPumpControl::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    try {
        for(;;) {
            //! Since raw buffer is Fast-in Fast-out, use the same sequence of push()es for pop()s
            auto hz = reader.pop<float>();
            auto pressure = reader.pop<float>();
            auto runtime = reader.pop<float>();
            tr[ *m_rotationSpeed] = hz;
            tr[ *m_runtime] = runtime;
            m_entryPressure->value(tr, pressure);
            auto warn = reader.pop<uint16_t>();
            tr[ *m_warning] = (bool)warn;
            auto err = reader.pop<uint16_t>();
            tr[ *m_error] = (bool)err;
            auto num_temps = reader.pop<uint16_t>();
            if(num_temps > m_temps.size())
                throw XRecordError{"", __FILE__, __LINE__};
            for(auto &&node: m_temps) {
                tr[ *node] = reader.pop<float>();
            }
        }
    }
    catch(XRecordError&) {
    }
}
void XPumpControl::visualize(const Snapshot &shot) {
}


void *
XPumpControl::execute(const atomic<bool> &terminated) {
    std::vector<shared_ptr<XNode>> runtime_ui{
        m_activate, m_heating, m_standby,
        m_standbyRotationSpeed, m_maxDrivePower
    };
    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(true);
    });
    if(m_temps.empty()) {
        auto labels = getTempLabels();
        QLCDNumber *runtime_uis[] = {
            m_form->m_lcdTemp1, m_form->m_lcdTemp2, m_form->m_lcdTemp3,
            m_form->m_lcdTemp4, m_form->m_lcdTemp5, m_form->m_lcdTemp6,
            nullptr};
        QLabel *lbl_uis[] = {
            m_form->m_lblTemp1, m_form->m_lblTemp2, m_form->m_lblTemp3,
            m_form->m_lblTemp4, m_form->m_lblTemp5, m_form->m_lblTemp6,
            nullptr};
        auto runtime_ui = runtime_uis;
        auto lbl_ui = lbl_uis;
        for(auto &&label: labels) {
            if( !runtime_ui)
                break;
            auto node = create<XDoubleNode>(label.c_str(), true);
            m_temps.push_back(node);
            (*lbl_ui)->setText(label);
            m_conTempUIs.push_back(
                xqcon_create<XQLCDNumberConnector>(node, *runtime_ui));
            runtime_ui++;
            lbl_ui++;
        }
    }
    iterate_commit([=](Transaction &tr){
        m_lsnOnStanbyRotationSpeedChanged = tr[ *m_standbyRotationSpeed].onValueChanged().connectWeakly(
                    shared_from_this(), &XPumpControl::onStandbyRotationSpeedChanged);
        m_lsnOnModeChanged = tr[ *m_activate].onValueChanged().connectWeakly(
            shared_from_this(), &XPumpControl::onModeChanged);
        tr[ *m_heating].onValueChanged().connect(m_lsnOnModeChanged);
        tr[ *m_standby].onValueChanged().connect(m_lsnOnModeChanged);
        m_lsnMaxDriverPowerChanged = tr[ *m_maxDrivePower].onValueChanged().connectWeakly(
            shared_from_this(), &XPumpControl::onMaxDriverPowerChanged);
    });


    while( !terminated) {
        msecsleep(100);

        auto writer = std::make_shared<RawData>();
        XTime time_awared = XTime::now();
        // try/catch exception of communication errors
        try {
            writer->push((float)getRotationSpeed());
            writer->push((float)getPressure());
            writer->push((float)getRuntime());
            auto errors = getWarning();
            writer->push((uint16_t)errors.first);
            if(errors.first)
                gWarnPrint(getLabel() + ": " + errors.second);
            errors = getError();
            writer->push((uint16_t)errors.first);
            if(errors.first)
                gErrPrint(getLabel() + ": " + errors.second);
            auto temps = getTemps();
            writer->push((int16_t)temps.size());
            for(auto t: temps)
                writer->push((float)t);
        }
        catch(XKameError &e) {
            e.print(getLabel() + "; ");
            continue;
        }
        finishWritingRaw(writer, time_awared, XTime::now());
    }
    m_lsnOnModeChanged.reset();
    m_lsnMaxDriverPowerChanged.reset();
    m_lsnOnStanbyRotationSpeedChanged.reset();
    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });
    return NULL;
}

