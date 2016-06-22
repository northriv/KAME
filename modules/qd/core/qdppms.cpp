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
    m_user_temp(create<XScalarEntry>("UserTemp", false,
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
    meas->scalarEntries()->insert(tr_meas, m_user_temp);
    meas->scalarEntries()->insert(tr_meas, m_field);
    meas->scalarEntries()->insert(tr_meas, m_position);

    m_form->setWindowTitle(XString("QDPPMS - " + getLabel() ));

    m_conTemp = xqcon_create<XQLCDNumberConnector>(temp()->value(), m_form->m_lcdTemp);
    m_conUserTemp = xqcon_create<XQLCDNumberConnector>(user_temp()->value(), m_form->m_lcdUserTemp);
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
        tr[ *fieldApproachMode()].add({"Linear", "No Overshoot", "Oscillate"});
        tr[ *fieldMagnetMode()].add({"Persistent", "Driven"});
        tr[ *positionApproachMode()].add("default");
        tr[ *positionSlowDownCode()] = 0;
        tr[ *tempApproachMode()].add({"FastSettle", "No Overshoot"});
        std::vector<shared_ptr<XNode>> runtime_ui{
            targetField(), fieldSweepRate(), fieldApproachMode(),
            fieldMagnetMode(), targetTemp(), tempSweepRate(), tempApproachMode(),
            targetPosition(), positionApproachMode(), positionSlowDownCode()
            };
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
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
    tr[ *this].m_sampleUserTemp = reader.pop<float>();
    tr[ *this].m_magnetField = reader.pop<float>();
    tr[ *this].m_samplePosition = reader.pop<float>();
    m_temp->value(tr, tr[ *this].m_sampleTemp);
    m_user_temp->value(tr, tr[ *this].m_sampleUserTemp);
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
}

void *
XQDPPMS::execute(const atomic<bool> &terminated) {
    std::vector<shared_ptr<XNode>> runtime_ui{
        targetField(), fieldSweepRate(), fieldApproachMode(),
        fieldMagnetMode(), targetTemp(), tempSweepRate(), tempApproachMode(),
        targetPosition(), positionApproachMode(), positionSlowDownCode()
        };
    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(true);

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
        double sample_user_temp;
        double sample_position;
        double helium_level;
        int status;

        try {
            // Reading....
            magnet_field = getField();
            sample_temp = getTemp();
            sample_user_temp = getUserTemp();
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
        writer->push((float)sample_user_temp);
        writer->push((float)magnet_field);
        writer->push((float)sample_position);

        finishWritingRaw(writer, XTime::now(), XTime::now());

        try{
            iterate_commit([=](Transaction &tr){
                tr[ *heliumLevel()] = helium_level;
                tr[ *tempStatus()] = std::map<int,std::string>{
                    {0x0,"Unknown"},
                    {0x1,"Persistent Stable"},
                    {0x2,"Persist Warming"},
                    {0x3,"Persist Cooling"},
                    {0x4,"Driven Stable"},
                    {0x5,"Driven Approach"},
                    {0x6,"Charging"},
                    {0x7,"Unchaging"},
                    {0x8,"Current Error"},
                    {0xf,"Failure"}
                }.at(status & 0xf);
                tr[ *fieldStatus()] = std::map<int,std::string>{
                    {0x0,"Unknown"},
                    {0x1,"Stable"},
                    {0x2,"Tracking"},
                    {0x5,"Wait"},
                    {0x6,"not Valid"},
                    {0x7,"Fill/Empty Reservoir"},
                    {0xa,"Standby"},
                    {0xd,"Control Disabled"},
                    {0xe,"Cannot Complete"},
                    {0xf,"Failure"}
                }.at((status >> 4) & 0xf);
                tr[ *positionStatus()] = std::map<int,std::string>{
                    {0x0,"Unknown"},
                    {0x1,"Stopped"},
                    {0x5,"Moving"},
                    {0x8,"Limit"},
                    {0x9,"Index"},
                    {0xf,"Failure"}
                }.at((status >> 12) & 0xf);
            });
        }
        catch (std::out_of_range &) {
            gErrPrint(i18n("PPMS: unknown status has been returned."));
        }
    }

    iterate_commit([=](Transaction &tr){
        for(auto &&x: runtime_ui)
            tr[ *x].setUIEnabled(false);
    });

    m_lsnFieldSet.reset();
    m_lsnTempSet.reset();
    m_lsnPositionSet.reset();

    return NULL;
}
