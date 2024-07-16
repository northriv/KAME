/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#include "filterwheelstmdriven.h"
#include "ui_filterwheel.h"
#include "motor.h"
#include "xnodeconnector.h"

REGISTER_TYPE(XDriverList, FilterWheelSTMDriven, "Filter wheel driver using STM");

XFilterWheelSTMDriven::XFilterWheelSTMDriven(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XFilterWheel(name, runtime, ref(tr_meas), meas) {

    connect(stm());
    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(stm(), m_form->m_cmbSTM, ref(tr_meas)),
    };
}
void
XFilterWheelSTMDriven::goAround() {

}

void XFilterWheelSTMDriven::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
                        XDriver *emitter) {
    Snapshot &shot_this(tr);

}

void XFilterWheelSTMDriven::visualize(const Snapshot &shot) {

}

bool XFilterWheelSTMDriven::checkDependency(const Snapshot &shot_this,
    const Snapshot &shot_emitter, const Snapshot &shot_others,
                                XDriver *emitter) const {
    const shared_ptr<XMotorDriver> stm__ = shot_this[ *stm()];
    if(!stm__)
        return false;
    if(emitter != stm__.get())
        return false;
    return true;
}
