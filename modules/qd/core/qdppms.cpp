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
#include "qdppms.h"
#include "analyzer.h"
#include "ui_qdppmsform.h"

XQDPPMS::XQDPPMS(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_temp(create<XScalarEntry>("Temp", false,
                                 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.3f")),
    m_temp_rotator(create<XScalarEntry>("TempRotator", false,
                                 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.3f")),
    m_field(create<XScalarEntry>("Field", false,
                                 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.3f")),
    m_position(create<XScalarEntry>("Position", false,
                                 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.3f")),
    m_heliumLevel(create<XDoubleNode>("HeliumLevel", false)),
    m_targetField(create<XDoubleNode>("TargetField",true)),
    m_fieldSweepRate(create<XDoubleNode>("FieldSweepRate",true)),
    m_fieldApproachMode(create<XComboNode>("FieldApproachMode",true,true)),
    m_fieldMagnetMode(create<XComboNode>("FieldMagnetMode",true,true)),
    m_fieldStatus(create<XStringNode>("FieldStatus",false)),
    m_targetPosition(create<XDoubleNode>("TargetPosition",true)),
    m_positionApproachMode(create<XComboNode>("PositionApproachMode",true,true)),
    m_positionSlowDownCode(create<XIntNode>("PositionSlowDownCode",true)),
    m_positionStatus(create<XStringNode>("PositionStatus",false)),
    m_targetTemp(create<XDoubleNode>("TargetTemp",true)),
    m_tempSweepRate(create<XDoubleNode>("TempdSweepRate",true)),
    m_tempApproachMode(create<XComboNode>("TempApproachMode",true,true)),
    m_tempStatus(create<XStringNode>("TempStatus",false)),
    m_form(new FrmQDPPMS(g_pFrmMain)) {
    meas->scalarEntries()->insert(tr_meas, m_temp);
    meas->scalarEntries()->insert(tr_meas, m_temp_rotator);
    meas->scalarEntries()->insert(tr_meas, m_field);
    meas->scalarEntries()->insert(tr_meas, m_position);

    m_form->setWindowTitle(XString("QDPPMS - " + getLabel() ));

    m_conTemp = xqcon_create<XQLCDNumberConnector>(temp()->value(), m_form->m_lcdTemp);
    m_conTempRotator = xqcon_create<XQLCDNumberConnector>(temp_rotator()->value(), m_form->m_lcdTempRotator);
    m_conField = xqcon_create<XQLCDNumberConnector>(field()->value(), m_form->m_lcdField);
    m_conPosition = xqcon_create<XQLCDNumberConnector>(position()->value(), m_form->m_lcdPosition);
    m_conHeliumLevel = xqcon_create<XQLCDNumberConnector>(heliumLevel(), m_form->m_lcdHeliumLevel);
    m_conTargetField = xqcon_create<XQLineEditConnector>(targetField(), m_form->m_edTargetField);
    m_conFieldSweepRate = xqcon_create<XQLineEditConnector>(fieldSweepRate(), m_form->m_edFieldSweepRate);
    m_conFieldApproachMode = xqcon_create<XQComboBoxConnector>(fieldApproachMode(), m_form->m_cmbFieldApproachMode, Snapshot( *m_fieldApproachMode));
    m_conFieldMagnetMode = xqcon_create<XQComboBoxConnector>(fieldMagnetMode(), m_form->m_cmbMagnetMode, Snapshot( *m_fieldMagnetMode));
    m_conFieldStatus = xqcon_create<XQLabelConnector>(fieldStatus(), m_form->m_labelFieldStatus);
    m_conTargetTemp = xqcon_create<XQLineEditConnector>(targetTemp(), m_form->m_edTargetTemp);
    m_conTempSweepRate = xqcon_create<XQLineEditConnector>(tempSweepRate(), m_form->m_edTempSweepRate);
    m_conTempApproachMode = xqcon_create<XQComboBoxConnector>(tempApproachMode(), m_form->m_cmbTempApproachMode, Snapshot( *m_tempApproachMode));
    m_conTempStatus = xqcon_create<XQLabelConnector>(tempStatus(), m_form->m_labelTempStatus);
    m_conTargetPosition = xqcon_create<XQLineEditConnector>(targetPosition(), m_form->m_edTargetPosition);
    m_conPositionApproachMode = xqcon_create<XQComboBoxConnector>(positionApproachMode(), m_form->m_cmbPositionApproachMode, Snapshot( *m_positionApproachMode));
    m_conPositionSlowDownCode = xqcon_create<XQLineEditConnector>(positionSlowDownCode(), m_form->m_edPositionSlowDownCode);
    m_conPostionStatus = xqcon_create<XQLabelConnector>(positionStatus(), m_form->m_labelPositionStatus);
    iterate_commit([=](Transaction &tr){
        tr[ *targetField()].setUIEnabled(false);
        tr[ *fieldSweepRate()].setUIEnabled(false);
        tr[ *fieldApproachMode()].add("Linear");
        tr[ *fieldApproachMode()].add("No Overshoot");
        tr[ *fieldApproachMode()].add("Oscillate");
        tr[ *fieldMagnetMode()].add("Persistent");
        tr[ *fieldMagnetMode()].add("Driven");
        tr[ *targetPosition()].setUIEnabled(false);
        tr[ *positionApproachMode()].add("default");
        tr[ *positionSlowDownCode()] = 0;
        tr[ *positionSlowDownCode()].setUIEnabled(false);
        tr[ *targetTemp()].setUIEnabled(false);
        tr[ *tempSweepRate()].setUIEnabled(false);
        tr[ *tempApproachMode()].add("FastSettle");
        tr[ *tempApproachMode()].add("No Overshoot");
    });
}
void
XQDPPMS::showForms() {
    m_form->showNormal();
    m_form->raise();
}
void
XQDPPMS::analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    tr[ *this].m_sampleTemp = reader.pop<float>();
    tr[ *this].m_sampleTempRotator = reader.pop<float>();
    tr[ *this].m_magnetField = reader.pop<float>();
    tr[ *this].m_samplePosition = reader.pop<float>();
    m_temp->value(tr, tr[ *this].m_sampleTemp);
    m_temp_rotator->value(tr, tr[ *this].m_sampleTempRotator);
    m_field->value(tr, tr[ *this].m_magnetField);
    m_position->value(tr, tr[*this].m_samplePosition);
}
void
XQDPPMS::visualize(const Snapshot &shot) {
}


void
XQDPPMS::onFieldChanged(const Snapshot &shot,  XValueNodeBase *){
    double sweepRate = ***fieldSweepRate();
    int approachMode = ***fieldApproachMode();
    int magnetMode = ***fieldMagnetMode();
    try {
        setField(shot[ *targetField()], sweepRate,
                approachMode, magnetMode);
    }
    catch (XKameError &e) {
        e.print(getLabel() + "; ");
    }
}

void
XQDPPMS::onPositionChanged(const Snapshot &shot,  XValueNodeBase *){
    int approachMode = ***positionApproachMode();
    int slowDownCode = ***positionSlowDownCode();
    try {
        setPosition(shot[ *targetPosition()], approachMode, slowDownCode);
    }
    catch (XKameError &e) {
        e.print(getLabel() + "; ");
    }
}

void
XQDPPMS::onTempChanged(const Snapshot &shot,  XValueNodeBase *){
    double sweepRate = ***tempSweepRate();
    int approachMode = ***tempApproachMode();
    try {
        setTemp(shot[ *targetTemp()], sweepRate, approachMode);
    }
    catch (XKameError &e) {
        e.print(getLabel() + "; ");
    }
    catch (NodeNotFoundError &e){
    }
}

void *
XQDPPMS::execute(const atomic<bool> &terminated) {

    targetField()->setUIEnabled(true);
    fieldSweepRate()->setUIEnabled(true);
    fieldApproachMode()->setUIEnabled(true);
    fieldMagnetMode()->setUIEnabled(true);
    targetTemp()->setUIEnabled(true);
    tempSweepRate()->setUIEnabled(true);
    tempApproachMode()->setUIEnabled(true);
    targetPosition()->setUIEnabled(true);
    positionApproachMode()->setUIEnabled(true);
    positionSlowDownCode()->setUIEnabled(true);

    iterate_commit([=](Transaction &tr){
        m_lsnFieldSet = tr[ *targetField()].onValueChanged().connectWeakly(
                    shared_from_this(), &XQDPPMS::onFieldChanged);
        m_lsnTempSet = tr[ *targetTemp()].onValueChanged().connectWeakly(
                    shared_from_this(), &XQDPPMS::onTempChanged);
        m_lsnPositionSet = tr[ *targetPosition()].onValueChanged().connectWeakly(
                    shared_from_this(), &XQDPPMS::onPositionChanged);
    });

    while( !terminated) {
        msecsleep(100);
        double magnet_field;
        double sample_temp;
        double sample_temp_rotator;
        double sample_position;
        double helium_level;
        int status;

        try {
            // Reading....
            magnet_field = getField();
            sample_temp = getTemp();
            sample_temp_rotator = getTempRotator();
            sample_position = getPosition();
            helium_level = getHeliumLevel();
            status = getStatus();
        }
        catch (XKameError &e) {
            e.print(getLabel() + "; ");
            continue;
        }
        auto writer = std::make_shared<RawData>();
        writer->push((float)sample_temp);
        writer->push((float)sample_temp_rotator);
        writer->push((float)magnet_field);
        writer->push((float)sample_position);

        finishWritingRaw(writer, XTime::now(), XTime::now());

        iterate_commit([=](Transaction &tr){
            tr[ *heliumLevel()] = helium_level;
            tr[ *tempStatus()] = mp_temp_status.at(status & 0xf);
            tr[ *fieldStatus()] = mp_field_status.at((status >> 4) & 0xf);
            tr[ *positionStatus()] = mp_position_status.at((status >> 12) & 0xf);
        });
    }

    targetField()->setUIEnabled(false);
    fieldSweepRate()->setUIEnabled(false);
    fieldApproachMode()->setUIEnabled(false);
    fieldMagnetMode()->setUIEnabled(false);
    targetTemp()->setUIEnabled(false);
    tempSweepRate()->setUIEnabled(false);
    tempApproachMode()->setUIEnabled(false);
    targetPosition()->setUIEnabled(false);
    positionApproachMode()->setUIEnabled(false);
    positionSlowDownCode()->setUIEnabled(false);

    m_lsnFieldSet.reset();
    m_lsnTempSet.reset();
    m_lsnPositionSet.reset();

    return NULL;
}
