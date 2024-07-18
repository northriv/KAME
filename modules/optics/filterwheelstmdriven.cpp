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
#include "ui_filterwheelform.h"
#include "motor.h"
#include "xnodeconnector.h"
#include "analyzer.h"

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
    Snapshot shot_this = iterate_commit([&](Transaction &tr){
      unsigned int dwellidx = tr[ *this].m_dwellIndex;
      unsigned int idx = tr[ *this].m_nextWheelIndex;
      dwellidx++;
      while(dwellidx >= tr[ *dwellCount(idx)]) {
          dwellidx = 0;
          idx++;
          if(idx > filterCount())
              idx = 0;
          if(idx == tr[ *this].m_wheelIndex)
              throw XDriver::XRecordError(i18n("No valid wheel setting."), __FILE__, __LINE__);
      }
      tr[ *this].m_dwellIndex = dwellidx;
      tr[ *this].m_nextWheelIndex = idx;
      tr[ *this].m_timeFilterMoved = XTime::now();
      tr[ *this].m_wheelIndex = -1;
    });
    shared_ptr<XMotorDriver> stm__ = shot_this[ *stm()];
    if( !stm__)
        throw XDriver::XRecordError(i18n("No valid STM setting."), __FILE__, __LINE__);
    trans( *stm__->target()) = (double)shot_this[ *stmAngle(shot_this[ *this].m_nextWheelIndex)];
}

void XFilterWheelSTMDriven::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
                        XDriver *emitter) {
    Snapshot &shot_this(tr);
    shared_ptr<XMotorDriver> stm__ = shot_this[ *stm()];
    const Snapshot &shot_stm((emitter == stm__.get()) ? shot_emitter : shot_others);
    if( !shot_stm[ *stm__->ready()]) {
        tr[ *this].m_timeFilterMoved = XTime::now();
        tr[ *this].m_wheelIndex = -1;
    }
    else {
        for(unsigned int i = 0; i < filterCount(); ++i) {
            if(fabs(shot_stm[ *stm__->position()->value()] - shot_this[ *stmAngle(i)]) < shot_this[ *angleErrorWithin()]) {
                //finds filter from the current angle.
                tr[ *this].m_wheelIndex = i;
            }
        }
        if(tr[ *this].m_wheelIndex >= 0) {
            if(XTime::now() - shot_this[ *this].m_timeFilterMoved < shot_this[ *waitAfterMove()])
                tr[ *this].m_wheelIndex = -1; //unstable yet.
        }
        else
            tr[ *this].m_timeFilterMoved = XTime::now(); //no filter found yet.
    }
    currentWheelIndex()->value(ref(tr), shot_this[ *this].m_wheelIndex);
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
