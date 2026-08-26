/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#include "digitalcamera.h"
#include "filterwheel.h"
#include "ui_filterwheelform.h"
#include "xnodeconnector.h"
#include "analyzer.h"

//REGISTER_TYPE(XDriverList, FilterWheel, "Filter wheel manager");

XFilterWheel::XFilterWheel(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XSecondaryDriver(name, runtime, ref(tr_meas), meas),
    m_camera(create<XItemNode<XDriverList, XDigitalCamera> >(
          "DigitalCamera", false, ref(tr_meas), meas->drivers(), true)),
    m_target(create<XUIntNode>("Target", true)),
    m_angleErrorWithin(create<XDoubleNode>("AngleErrorWithin", false)),
    m_waitAfterMove(create<XDoubleNode>("WaitAfterMove", false)),
    m_goAroundAfterShot(create<XBoolNode>("GoAroundAfterShot", true)),
    m_currentWheelIndex(create<XScalarEntry>("CurrWheelIndex", true, dynamic_pointer_cast<XDriver>(shared_from_this()), "%.0f")),
    m_form(new FrmFilterWheel) {

    connect(camera());

    meas->scalarEntries()->insert(tr_meas, currentWheelIndex());

    m_conUIs = {
        xqcon_create<XQComboBoxConnector>(m_camera, m_form->m_cmbCamera, ref(tr_meas)),
        xqcon_create<XQLCDNumberConnector>(m_currentWheelIndex->value(), m_form->m_lcdCurrentPos),
        xqcon_create<XQLineEditConnector>(m_waitAfterMove, m_form->m_edWaitAfterMove),
        xqcon_create<XQLineEditConnector>(m_angleErrorWithin, m_form->m_edPhaseErrWithin),
        xqcon_create<XQSpinBoxUnsignedConnector>(m_target, m_form->m_spbTarget),
        xqcon_create<XQToggleButtonConnector>(m_goAroundAfterShot, m_form->m_ckbGoAround),
    };

    QLineEdit *uiangles[] = {m_form->m_edAngle0, m_form->m_edAngle1, m_form->m_edAngle2, m_form->m_edAngle3, m_form->m_edAngle4, m_form->m_edAngle5};
    QSpinBox *uidwells[] = {m_form->m_spbCounts0, m_form->m_spbCounts1, m_form->m_spbCounts2, m_form->m_spbCounts3, m_form->m_spbCounts4, m_form->m_spbCounts5};
    QLineEdit *uilabels[] = {m_form->m_edLabel0, m_form->m_edLabel1, m_form->m_edLabel2, m_form->m_edLabel3, m_form->m_edLabel4, m_form->m_edLabel5};

    for(unsigned int i = 0; i < MaxFilterCount; ++i) {
        m_filterLabels.push_back(create<XStringNode>(formatString("FilterLabel%u", i).c_str(), false));
        m_dwellCounts.push_back(create<XUIntNode>(formatString("DwellCount%u", i).c_str(), false));
        m_stmAngles.push_back(create<XDoubleNode>(formatString("STMAngle%u", i).c_str(), false));
        m_conUIs.push_back(xqcon_create<XQSpinBoxUnsignedConnector>(dwellCount(i), uidwells[i]));
        m_conUIs.push_back(xqcon_create<XQLineEditConnector>(filterLabel(i), uilabels[i]));
        m_conUIs.push_back(xqcon_create<XQLineEditConnector>(stmAngle(i), uiangles[i]));
    }

    m_form->setWindowTitle(i18n("Filter Wheel - ") + getLabel() );
    iterate_commit([=](Transaction &tr){
        m_lsnOnTargetChanged = tr[ *target()].onValueChanged().connectWeakly(
            shared_from_this(), &XFilterWheel::onTargetChangedInternal);
        tr[ *this].m_timeFilterMoved = XTime::now();
    });
}
XFilterWheel::~XFilterWheel() {
}
void
XFilterWheel::showForms() {
// impliment form->show() here
    m_form->showNormal();
    m_form->raise();
}
void XFilterWheel::analyze(Transaction &tr, const Snapshot &shot_emitter, const Snapshot &shot_others,
                        XDriver *emitter) {
    Snapshot &shot_this(tr);
    shared_ptr<XDigitalCamera> camera__ = shot_this[ *camera()];
    if(emitter == camera__.get()) {
        int wheelidx = tr[ *this].wheelIndexOfFrame(shot_emitter[ *camera__].time(),
                shot_emitter[ *camera__].timeAwared());
        if(wheelidx < 0)
            throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
        //Records just before possible go-around, for the succeeding secondary drivers.
        tr[ *this].m_timeLastFrame = shot_emitter[ *camera__].time();
        tr[ *this].m_wheelIndexOfLastFrame = wheelidx;
        if(shot_this[ *goAroundAfterShot()]) {
            //finds next wheel
            unsigned int dwellidx = tr[ *this].m_dwellIndex;
            unsigned int idx = tr[ *this].m_nextWheelIndex;
            dwellidx++;
            while(dwellidx >= tr[ *dwellCount(idx)]) {
                dwellidx = 0;
                idx++;
                if(idx >= filterCount())
                    idx = 0;
                if(idx == tr[ *this].m_nextWheelIndex)
                    throw XDriver::XRecordError(i18n("No valid wheel setting."), __FILE__, __LINE__);
            }
            tr[ *this].m_dwellIndex = dwellidx;
            tr[ *this].m_nextWheelIndex = idx;
            if(idx != tr[ *target()]) {
                tr[ *this].m_timeFilterMoved = XTime::now();
                tr[ *this].m_wheelIndex = -1;
                tr[ *this].m_timeFilterStabled = {};
                //Together with the target, and inside the same transaction: a STM
                //record analyzed before onTargetChanged() has even reached the
                //hardware must not be able to call the wheel stable again.
                tr[ *this].m_wheelIndexCommanded = idx;
                tr[ *target()] = idx;
            }
        }
    }
    else {
        int pos = currentWheelPosition(shot_this, shot_emitter);
        int commanded = shot_this[ *this].m_wheelIndexCommanded;
        if((pos >= 0) && (commanded >= 0) && (pos != commanded))
            pos = -1; //Resting at a filter, but not at the commanded one: still moving, or not started.
        tr[ *this].m_wheelIndex = pos;
        if(tr[ *this].m_wheelIndex >= 0) {
            if(XTime::now().diff_sec(shot_this[ *this].m_timeFilterMoved) < shot_this[ *waitAfterMove()])
                tr[ *this].m_wheelIndex = -1; //unstable (still vibrating) yet.
            else if( !tr[ *this].m_timeFilterStabled) {
                tr[ *this].m_timeFilterStabled = XTime::now(); //timestamp when wheel becomes stable.
            }
        }
        else {
            tr[ *this].m_timeFilterStabled = {};
            tr[ *this].m_timeFilterMoved = XTime::now(); //filter is not found yet., for timestamp when filter reached.
        }
    }
    currentWheelIndex()->value(ref(tr), shot_this[ *this].m_wheelIndex);
}

void
XFilterWheel::onTargetChangedInternal(const Snapshot &shot, XValueNodeBase *node) {
    //The wheel is, from now on, no longer where it was. Invalidating here as well
    //as in analyze() covers a target set by hand, by a script, or by .kam loading.
    //The auto-advance has already written the same index inside its own
    //transaction, so this commit is skipped for it.
    iterate_commit([&](Transaction &tr){
        unsigned int idx = tr[ *target()];
        if(tr[ *this].m_wheelIndexCommanded == (int)idx)
            return; //auto-advance, already accounted for.
        tr[ *this].m_wheelIndexCommanded = idx;
        tr[ *this].m_wheelIndex = -1;
        tr[ *this].m_timeFilterStabled = {};
        tr[ *this].m_timeFilterMoved = XTime::now();
        //Resume the go-around from where the wheel was just sent, instead of
        //stepping from a stale index and jumping several filters at once.
        tr[ *this].m_nextWheelIndex = idx;
        tr[ *this].m_dwellIndex = 0;
    });
    onTargetChanged(shot, node);
}

void XFilterWheel::visualize(const Snapshot &shot) {

}

