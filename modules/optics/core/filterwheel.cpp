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
#include "filterwheel.h"
#include "ui_filterwheelform.h"
#include "xnodeconnector.h"
#include "analyzer.h"

//REGISTER_TYPE(XDriverList, FilterWheel, "Filter wheel manager");

XFilterWheel::XFilterWheel(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XSecondaryDriver(name, runtime, ref(tr_meas), meas),
    m_target(create<XUIntNode>("Target", true)),
    m_angleErrorWithin(create<XDoubleNode>("AngleErrorWithin", false)),
    m_waitAfterMove(create<XDoubleNode>("WaitAfterMove", false)),
    m_form(new FrmFilterWheel) {

    meas->scalarEntries()->insert(tr_meas, currentWheelIndex());

    m_conUIs = {
        xqcon_create<XQLCDNumberConnector>(m_currentWheelIndex->value(), m_form->m_lcdCurrentPos),
        xqcon_create<XQLineEditConnector>(m_waitAfterMove, m_form->m_edWaitAfterMove),
        xqcon_create<XQLineEditConnector>(m_angleErrorWithin, m_form->m_edPhaseErrWithin),
        xqcon_create<XQSpinBoxUnsignedConnector>(m_target, m_form->m_spbTarget),
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
            shared_from_this(), &XFilterWheel::onTargetChanged);
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
