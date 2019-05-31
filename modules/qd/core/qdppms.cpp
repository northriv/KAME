/***************************************************************************
        Copyright (C) 2002-2016 Shota Suetsugu and Kentaro Kitagawa
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

constexpr double MIN_MODEL6700_SWEEPRATE = (10.0 / 10000); //T/s, 9.3 Oe/s is the actual limit.

XQDPPMS::XQDPPMS(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPrimaryDriverWithThread(name, runtime, ref(tr_meas), meas),
    m_temp(create<XScalarEntry>("Temp", false,
                                 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.3f")),
    m_user_temp(create<XScalarEntry>("UserTemp", false,
                                 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.3f")),
    m_field(create<XScalarEntry>("Field", false,
                                 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.4f")),
    m_position(create<XScalarEntry>("Position", false,
                                 dynamic_pointer_cast<XDriver>(shared_from_this()), "%.3f")),
    m_heliumLevel(create<XDoubleNode>("HeliumLevel", true)),
    m_targetField(create<XDoubleNode>("TargetField", true)),
    m_fieldSweepRate(create<XDoubleNode>("FieldSweepRate",true)),
    m_fieldApproachMode(create<XComboNode>("FieldApproachMode",true,true)),
    m_fieldMagnetMode(create<XComboNode>("FieldMagnetMode",true,true)),
    m_fieldStatus(create<XStringNode>("FieldStatus", true)),
    m_targetPosition(create<XDoubleNode>("TargetPosition",true)),
    m_positionApproachMode(create<XComboNode>("PositionApproachMode",true,true)),
    m_positionSlowDownCode(create<XIntNode>("PositionSlowDownCode",true)),
    m_positionStatus(create<XStringNode>("PositionStatus",true)),
    m_targetTemp(create<XDoubleNode>("TargetTemp",true)),
    m_tempSweepRate(create<XDoubleNode>("TempSweepRate",true)),
    m_tempApproachMode(create<XComboNode>("TempApproachMode",true,true)),
    m_tempStatus(create<XStringNode>("TempStatus",true)),
    m_form(new FrmQDPPMS) {
    meas->scalarEntries()->insert(tr_meas, m_temp);
    meas->scalarEntries()->insert(tr_meas, m_user_temp);
    meas->scalarEntries()->insert(tr_meas, m_field);
    meas->scalarEntries()->insert(tr_meas, m_position);

    m_form->setWindowTitle(XString("QDPPMS - " + getLabel() ));

    m_conUIs = {
        xqcon_create<XQLCDNumberConnector>(temp()->value(), m_form->m_lcdTemp),
        xqcon_create<XQLCDNumberConnector>(user_temp()->value(), m_form->m_lcdUserTemp),
        xqcon_create<XQLCDNumberConnector>(field()->value(), m_form->m_lcdField),
        xqcon_create<XQLCDNumberConnector>(position()->value(), m_form->m_lcdPosition),
        xqcon_create<XQLCDNumberConnector>(heliumLevel(), m_form->m_lcdHeliumLevel),
        xqcon_create<XQLineEditConnector>(targetField(), m_form->m_edTargetField),
        xqcon_create<XQLineEditConnector>(fieldSweepRate(), m_form->m_edFieldSweepRate),
        xqcon_create<XQComboBoxConnector>(fieldApproachMode(), m_form->m_cmbFieldApproachMode, Snapshot( *m_fieldApproachMode)),
        xqcon_create<XQComboBoxConnector>(fieldMagnetMode(), m_form->m_cmbMagnetMode, Snapshot( *m_fieldMagnetMode)),
        xqcon_create<XQLabelConnector>(fieldStatus(), m_form->m_labelFieldStatus),
        xqcon_create<XQLineEditConnector>(targetTemp(), m_form->m_edTargetTemp),
        xqcon_create<XQLineEditConnector>(tempSweepRate(), m_form->m_edTempSweepRate),
        xqcon_create<XQComboBoxConnector>(tempApproachMode(), m_form->m_cmbTempApproachMode, Snapshot( *m_tempApproachMode)),
        xqcon_create<XQLabelConnector>(tempStatus(), m_form->m_labelTempStatus),
        xqcon_create<XQLineEditConnector>(targetPosition(), m_form->m_edTargetPosition),
        xqcon_create<XQComboBoxConnector>(positionApproachMode(), m_form->m_cmbPositionApproachMode, Snapshot( *m_positionApproachMode)),
        xqcon_create<XQLineEditConnector>(positionSlowDownCode(), m_form->m_edPositionSlowDownCode),
        xqcon_create<XQLabelConnector>(positionStatus(), m_form->m_labelPositionStatus)
    };

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
XQDPPMS::onFieldChanged(const Snapshot &,  XValueNodeBase *){
    Snapshot shot( *this);
    double sweepRate = shot[ *fieldSweepRate()]; //T/s
    if(sweepRate < 1e-7)
        throw XInterface::XInterfaceError(i18n("Too small sweep rate."), __FILE__, __LINE__);
    int approachMode = shot[ *fieldApproachMode()];
    int magnetMode = shot[ *fieldMagnetMode()];
    if(shot[ *fieldSweepRate()] >= MIN_MODEL6700_SWEEPRATE) {
        try {
            setField(shot[ *targetField()], sweepRate,
                    approachMode, magnetMode);
        }
        catch (XKameError &e) {
            e.print(getLabel() + "; ");
        }
    }
}

void
XQDPPMS::onPositionChanged(const Snapshot &,  XValueNodeBase *){
    Snapshot shot( *this);
    int approachMode = shot[ *positionApproachMode()];
    int slowDownCode = shot[ *positionSlowDownCode()];
    try {
        setPosition(shot[ *targetPosition()], approachMode, slowDownCode);
    }
    catch (XKameError &e) {
        e.print(getLabel() + "; ");
    }
}

void
XQDPPMS::onTempChanged(const Snapshot &,  XValueNodeBase *){
    Snapshot shot( *this);
    double sweepRate = shot[ *tempSweepRate()]; //K/min
    if(sweepRate < 0.01)
        throw XInterface::XInterfaceError(i18n("Too small sweep rate."), __FILE__, __LINE__);
    int approachMode = shot[ *tempApproachMode()];
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
//        tr[ *fieldSweepRate()].onValueChanged().connect(m_lsnFieldSet);
        m_lsnTempSet = tr[ *targetTemp()].onValueChanged().connectWeakly(
                    shared_from_this(), &XQDPPMS::onTempChanged);
//        tr[ *tempSweepRate()].onValueChanged().connect(m_lsnTempSet);
        m_lsnPositionSet = tr[ *targetPosition()].onValueChanged().connectWeakly(
                    shared_from_this(), &XQDPPMS::onPositionChanged);
    });

    auto setfield_prevtime = XTime::now();
    double field_by_hardware = 0.0;
    auto field_by_hardware_time = XTime::now();
    while( !terminated) {
        msecsleep(100);
        double magnet_field; //Oe
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
            //displays status strings.
            iterate_commit([=](Transaction &tr){
                tr[ *heliumLevel()] = helium_level;
                tr[ *tempStatus()] = std::map<int,std::string>{
                    {0x0,"Unknown"},
                    {0x1,"Stable"},
                    {0x2,"Tracking"},
                    {0x5,"Near"},
                    {0x6,"Chasing"},
                    {0x7,"Pot Ops"},
                    {0xa,"Standby"},
                    {0xd,"Control Disabled"},
                    {0xe,"Cannot Complete"},
                    {0xf,"Failure"}
                }.at(status & 0xf);
                tr[ *positionStatus()] = std::map<int,std::string>{
                    {0x0,"Unknown"},
                    {0x1,"Stopped"},
                    {0x5,"Moving"},
                    {0x8,"Limit"},
                    {0x9,"Index"},
                    {0xf,"Failure"}
                }.at((status >> 12) & 0xf);
                tr[ *fieldStatus()] = std::map<int,std::string>{
                    {0x0,"Unknown"},
                    {0x1,"Persistent"},
                    {0x2,"SW-Warm"},
                    {0x3,"SW-Cool"},
                    {0x4,"Holding"},
                    {0x5,"Itelate"},
                    {0x6,"Charging"},
                    {0x7,"Discharging"},
                    {0x8,"CurrentError"},
                    {0xf,"Failure"}
                }.at((status >> 4) & 0xf);
            });
        }
        catch (std::out_of_range &) {
            gErrPrint(i18n("PPMS: unknown status has been returned."));
            continue;
        }
        try {
            Snapshot shot( *this);
            if(shot[ *fieldSweepRate()] < MIN_MODEL6700_SWEEPRATE) {
                switch ((status >> 4) & 0xf) {
                case 3: //"SW-Cool"
                case 4: //"Holding"
                {
                    //Field sweep control for slow ramp by software
                    //Model6700 hardware cannot handle a rate below 9.3Oe/sec..
                    setfield_prevtime = XTime::now();
                    double sweeprate = fabs(shot[ *fieldSweepRate()]); //T/s
                    if( shot[ *targetField()] < field_by_hardware)
                        sweeprate *= -1;
                    double newfield = field_by_hardware + (XTime::now() - field_by_hardware_time + 4) * sweeprate; //expected field 4 sec after.
                    int approach_mode = shot[ *fieldApproachMode()];
                    if(fabs(shot[ *targetField()] - magnet_field * 1e-4) > std::max(10 * sweeprate, 2e-4)) {
                        if(((newfield > shot[ *targetField()]) && (sweeprate > 0)) ||
                            ((newfield < shot[ *targetField()]) && (sweeprate < 0))) {
                            dbgPrint( "Magnet field now approaching to the set point.");
                            setField(shot[ *targetField()], MIN_MODEL6700_SWEEPRATE, approach_mode, shot[ *fieldMagnetMode()]);
                        }
                        else {
                            setField(newfield, MIN_MODEL6700_SWEEPRATE, 1 /*no overshoot*/, 1 /*driven*/);
                        }
                        break;
                    }
                    field_by_hardware = magnet_field * 1e-4; //T
                    field_by_hardware_time = XTime::now();
                    break;
                }
                case 5: //"Iterate"
                case 6: //"Charging"
                case 7: //"Disharging"
                default:
                    break;
                }
            }
            else {
                field_by_hardware = magnet_field * 1e-4; //T
                field_by_hardware_time = XTime::now();
            }
        }
        catch (XKameError &e) {
            e.print(getLabel() + "; ");
            continue;
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
