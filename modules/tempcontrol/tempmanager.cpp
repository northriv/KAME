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
#include "tempmanager.h"
#include "ui_tempmanagerform.h"
#include "interface.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <QStatusBar>
#include <QToolBox>
#include <QLineEdit>
#include <QPushButton>
#include <QTableWidget>
#include <QLabel>
#include <QComboBox>

REGISTER_TYPE(XDriverList, TempManager, "Temperature Management System");

XTempManager::XZone::XZone(const char *name, bool runtime,
    Transaction &tr_list, const shared_ptr<XThermometerList> &list) :
	XNode(name, runtime),
    m_upperTemp(create<XDoubleNode>("UpperTemp", false, "%.3g")),
    m_maxRampRate(create<XDoubleNode>("MaxRampRate", false, "%.3g")),
    m_channel(create<XComboNode>("Channel", false)),
    m_excitation(create<XComboNode>("Excitation", false)),
    m_thermometer(create<XItemNode<XThermometerList,
		XThermometer> > ("Thermometer", false, ref(tr_list), list)),
    m_loop(create<XComboNode>("Loop", false)),
    m_powerRange(create<XComboNode>("PowerRange", false)),
    m_prop(create<XDoubleNode>("P", false, "%.4g")),
    m_interv(create<XDoubleNode>("I", false, "%.4g")),
    m_deriv(create<XDoubleNode>("D", false, "%.4g")),
    m_thermometers(list) {
    for(unsigned int i = 0; i < XTempManager::maxNumOfAUXDevices; ++i)
        m_auxDeviceValues.push_back(create<XDoubleNode>(
            formatString("AUXDevice%dValue", i + 1).c_str(), false, "%.4g"));
}

XTempManager::XTempManager(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XSecondaryDriver(name, runtime, ref(tr_meas), meas),
    m_isActivated(create<XBoolNode> ("IsActivated", true)),
    m_targetTemp(create<XDoubleNode> ("TargetTemp" , true, "%.5g")),
    m_rampRate(create<XDoubleNode> ("RampRate", true, "%.3g")),
    m_dupZone(create<XTouchableNode> ("DuplicateZone", true)),
    m_delZone(create<XTouchableNode> ("DeleteZone", true)),
    m_extDevice(create<XItemExtDevice>("ExtDevice", false, ref(tr_meas), meas->drivers())),
    m_extIsPositive(create<XBoolNode> ("ExtIsPositive", false)),
    m_hysteresisOnZoneTr(create<XDoubleNode> ("HysteresisOnZoneTransition" , false, "%.3g")),
    m_doesMixTemp(create<XBoolNode> ("MixTempOnSensorChange" , false)),
    m_mainDevice(create<XItemMainDevice>("MainDevice", false, ref(tr_meas), meas->drivers(), true)),
    m_zones(create<XZoneList> ("ZoneList" , false, meas->thermometers())),
    m_stabilized(create<XDoubleNode> ("Stabilized", true, "%.3g")),
    m_statusStr(create<XStringNode> ("Status", true)),
    m_tempStatusStr(create<XStringNode> ("TempStatus", true)),
    m_heaterStatusStr(create<XStringNode> ("HeaterStatus", true)),
    m_entryTemp(create<XScalarEntry>("Temp", false,
        dynamic_pointer_cast<XDriver>(shared_from_this()), "%.5g")),
    m_entryPow(create<XScalarEntry>("HeaterPower", false,
        dynamic_pointer_cast<XDriver>(shared_from_this()), "%.5g")),
    m_form(new FrmTempManager(g_pFrmMain)) {
    for(unsigned int i = 0; i < maxNumOfAUXDevices; ++i) {
        m_auxDevices.push_back(create<XItemAUXDevice>(
            formatString("AUXDevice%d", i + 1).c_str(), false, ref(tr_meas), meas->drivers()));
        m_auxDevChs.push_back(create<XComboNode>(
            formatString("AUXDevice%dChannel", i + 1).c_str(), false, true));
        m_auxDevModes.push_back(create<XComboNode>(
            formatString("AUXDevice%dMode", i + 1).c_str(), false, true));
    }

    connect(mainDevice());

    meas->scalarEntries()->insert(tr_meas, m_entryTemp);
    meas->scalarEntries()->insert(tr_meas, m_entryPow);

    m_conUIs = {
        xqcon_create<XQToggleButtonConnector>(m_isActivated, m_form->m_ckbActivateEngine),
        xqcon_create<XQLCDNumberConnector> (m_entryTemp->value(), m_form->m_lcdTemp),
        xqcon_create<XQLCDNumberConnector> (m_entryPow->value(), m_form->m_lcdHeater),
        xqcon_create<XQLineEditConnector> (m_targetTemp, m_form->m_edTargetTemp),
        xqcon_create<XQLineEditConnector> (m_rampRate, m_form->m_edRampRate),
        xqcon_create<XQLineEditConnector> (m_hysteresisOnZoneTr, m_form->m_edZoneHysteresis),
        xqcon_create<XQToggleButtonConnector>(m_doesMixTemp, m_form->m_ckbMixTempOnChange),
        xqcon_create<XQComboBoxConnector> (m_mainDevice, m_form->m_cmbMainDevice, ref(tr_meas)),
        xqcon_create<XQComboBoxConnector> (m_extDevice, m_form->m_cmbExtDevice, ref(tr_meas)),
        xqcon_create<XQToggleButtonConnector>(m_extIsPositive, m_form->m_ckbExtIsPositive),
        xqcon_create<XQButtonConnector>(m_dupZone, m_form->m_btnZoneDup),
        xqcon_create<XQButtonConnector>(m_delZone, m_form->m_btnZoneDelete),
        xqcon_create<XQLabelConnector>(m_statusStr, m_form->m_lblStatus),
        xqcon_create<XQLabelConnector>(m_tempStatusStr, m_form->m_lblTempStatus),
        xqcon_create<XQLabelConnector>(m_heaterStatusStr, m_form->m_lblHeaterStatus)
    };
    int i = 0;
    for(auto &&ui :{m_form->m_cmbSubDevice1, m_form->m_cmbSubDevice2, m_form->m_cmbSubDevice3,
               m_form->m_cmbSubDevice4, m_form->m_cmbSubDevice5, m_form->m_cmbSubDevice6}) {
        m_conUIs.push_back(xqcon_create<XQComboBoxConnector>(auxDevice(i++), ui, ref(tr_meas)));
    }
    i = 0;
    for(auto &&ui :{m_form->m_cmbSubDev1Ch, m_form->m_cmbSubDev2Ch, m_form->m_cmbSubDev3Ch,
            m_form->m_cmbSubDev4Ch, m_form->m_cmbSubDev5Ch, m_form->m_cmbSubDev6Ch}) {
        m_conUIs.push_back(xqcon_create<XQComboBoxConnector>(auxDevCh(i), ui, Snapshot( *auxDevCh(i))));
        i++;
    }
    i = 0;
    for(auto &&ui :{m_form->m_cmbSubDev1Mode, m_form->m_cmbSubDev2Mode, m_form->m_cmbSubDev3Mode,
            m_form->m_cmbSubDev4Mode, m_form->m_cmbSubDev5Mode, m_form->m_cmbSubDev6Mode}) {
        m_conUIs.push_back(xqcon_create<XQComboBoxConnector>(auxDevMode(i), ui, Snapshot( *auxDevMode(i))));
        i++;
    }

    iterate_commit([=](Transaction &tr){
        tr[ *hysteresisOnZoneTr()] = 5;
        tr[ *doesMixTemp()] = true;
        m_lsnOnActivateChanged = tr[ *m_isActivated].onValueChanged().connectWeakly(
            shared_from_this(), &XTempManager::onActivateChanged);
        m_lsnOnAUXDeviceChanged = tr[ *auxDevice(0)].onValueChanged().connectWeakly(
            shared_from_this(), &XTempManager::onAUXDeviceChanged,
            Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP);
        for(unsigned int i = 1; i < maxNumOfAUXDevices; ++i)
            tr[ *auxDevice(i)].onValueChanged().connect(m_lsnOnAUXDeviceChanged);
        m_lsnOnMainDeviceChanged = tr[ *mainDevice()].onValueChanged().connectWeakly(
            shared_from_this(), &XTempManager::onMainDeviceChanged,
            Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP);
        m_lsnOnExtDeviceChanged = tr[ *extDevice()].onValueChanged().connectWeakly(
            shared_from_this(), &XTempManager::onExtDeviceChanged,
            Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP);
        m_lsnOnDupTouched = tr[ *m_dupZone].onTouch().connectWeakly(
            shared_from_this(), &XTempManager::onDupTouched,
            Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP);
        m_lsnOnDelTouched = tr[ *m_delZone].onTouch().connectWeakly(
            shared_from_this(), &XTempManager::onDeleteTouched,
            Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP);
    });

    onActivateChanged(Snapshot( *this), isActivated().get());
    refreshZoneUIs();
    m_currZoneNo = 0;
    m_currLoopNo = 0;
    m_currCh = 0;
    m_tempStarted = 0.0;
    m_timeStarted = XTime::now();
}

XTempManager::~XTempManager() {
    m_conZoneUIs.clear();
    trans( *isActivated()) = false;
    for(unsigned int j = 0; j < maxNumOfAUXDevices; ++j)
        m_lsnAUXDevOnListChanged[j].reset();
    m_lsnMainDevOnListChanged.reset();
    m_lsnExtDevOnListChanged.reset();
    m_lsnOnAUXDeviceChanged.reset();
    m_lsnOnMainDeviceChanged.reset();
    m_lsnOnExtDeviceChanged.reset();
}

void XTempManager::showForms() {
    m_form->showNormal();
    m_form->raise();
}

void XTempManager::visualize(const Snapshot &shot) {
    if( !shot[ *isActivated()]) {
        return;
    }
    const shared_ptr<XTempControl> maindev = shot[ *mainDevice()];
    if( !maindev) return;
    Snapshot shot_emitter( *maindev);

    auto zone = currentZone(shot);
    shared_ptr<XFlowControllerDriver> extflowctrl = shot[ *extDevice()];
    shared_ptr<XDCSource> extdcsrc = shot[ *extDevice()];
    int loop = m_currLoopNo;
    if( !extflowctrl && !extdcsrc && (m_currCh >= 0)) {
        if((loop >= 0) && (loop < maindev->numOfLoops())) {
            XString chname = maindev->currentChannel(loop)->itemStrings(shot_emitter).at(m_currCh).name;
            if(chname != shot_emitter[ *maindev->currentChannel(loop)].to_str()) {
                trans( *maindev->currentChannel(loop)).str(chname);
                gMessagePrint(
                    getLabel() + ": " + formatString_tr(I18N_NOOP("Source sensor channel is changed to %s."), chname.c_str()));
                return;
            }
        }
    }
    if((m_currCh >= 0) && (m_currCh < shot_emitter.size(maindev->channels()))) {
        auto ch = dynamic_pointer_cast<XTempControl::XChannel>(
            shot_emitter.list(maindev->channels())->at(m_currCh));
        if((m_currExcitaion >= 0) &&
            (m_currExcitaion != shot_emitter[ *ch->excitation()])) {
            trans( *ch->excitation()) = m_currExcitaion;
            fprintf(stderr, "Excitation is changed to %d.\n", m_currExcitaion);
            return;
        }
    }

    XString msg = shot[ *statusStr()];
    double stemp = m_setpointTemp;

    double p = shot[ *zone->prop()];
    double i = shot[ *zone->interv()];
    double d = shot[ *zone->deriv()];
    msg += formatString(", P=%.3g, I=%.3g, D=%.3g", p, i, d);
    double power = pid(shot, XTime::now(), stemp, shot[ *m_entryTemp->value()]);
    if(extdcsrc) {
        if((loop >= 0) && (shot[ *extdcsrc->channel()] != loop))
            trans( *extdcsrc->channel()) = loop;
        double limit_max = extdcsrc->max(loop, false);
        power = limit_max * sqrt(power) / 10.0;
        extdcsrc->changeValue(loop, power, false);
//        msg += formatString(", using ext.dc.src. %s, P=%.4g of %.3g",
//            shot[ *extdcsrc->channel()].to_str().c_str(), power, limit_max);
    }
    else if(extflowctrl) {
        double limit_max = Snapshot( *extflowctrl)[ *extflowctrl].fullScale();
        power = limit_max * power / 100.0;
        trans( *extflowctrl->target()) = power;
//        msg += formatString(", using ext.flow cntl. %.4g of %.3g", power, limit_max);
    }
    else {
        if((loop >= 0) && (loop < maindev->numOfLoops())) {
//            msg += formatString(", using %s", maindev->loopLabel(loop).c_str());
            maindev->iterate_commit([=](Transaction &tr){
                if(tr[ *maindev->targetTemp(loop)] != stemp)
                    tr[ *maindev->targetTemp(loop)] = stemp;
                if((shot[ *zone->powerRange()] >= 0) &&
                    (tr[ *maindev->powerRange(loop)] != (int)shot[ *zone->powerRange()])) {
                    tr[ *maindev->powerRange(loop)] = (int)shot[ *zone->powerRange()];
                }

                if(tr[ *maindev->prop(loop)] != p) {
                    tr[ *maindev->prop(loop)] = p;
                }
                if(tr[ *maindev->interval(loop)] != i) {
                    tr[ *maindev->interval(loop)] = i;
                }
                if(tr[ *maindev->deriv(loop)] != d) {
                    tr[ *maindev->deriv(loop)] = d;
                }
            });
        }
    }

    for(unsigned int i = 0; i < maxNumOfAUXDevices; ++i) {
        double v = shot[ *zone->auxDeviceValues(i)];
        XString auxname = shot[ *auxDevice(i)].to_str();
        shared_ptr<XDCSource> dcsrc = shot[ *auxDevice(i)];
        shared_ptr<XFlowControllerDriver> flowctrl = shot[ *auxDevice(i)];
        shared_ptr<XTempControl> tempctrl = shot[ *auxDevice(i)];
        int ch = shot[ *auxDevCh(i)];
        int mode = shot[ *auxDevMode(i)];
        if(tempctrl) {
            if((ch < 0) || (ch >= tempctrl->numOfLoops()))
                break;
            if(mode == 0) {
                tempctrl->iterate_commit([=](Transaction &tr){
                    if(tr[ *tempctrl->targetTemp(ch)] != v)
                        tr[ *tempctrl->targetTemp(ch)] = v;
                });
                msg += formatString(", %s %s SV=%.3g K",
                    auxname.c_str(), tempctrl->loopLabel(ch).c_str(), v);
            }
            else if(mode == 1) {
                tempctrl->iterate_commit([=](Transaction &tr){
                    if(tr[ *tempctrl->manualPower(ch)] != v)
                        tr[ *tempctrl->manualPower(ch)] = v;
                });
                msg += formatString(", %s %s P=%.3g",
                    auxname.c_str(), tempctrl->loopLabel(ch).c_str(), v);
            }
        }
        if(dcsrc) {
            dcsrc->iterate_commit([=](Transaction &tr){
                if((ch >= 0) && (tr[ *dcsrc->channel()] != ch))
                    tr[ *dcsrc->channel()] = ch;
               if(tr[ *dcsrc->value()] != v)
                   tr[ *dcsrc->value()] = v;
            });
            msg += formatString(", %s %s %.3g",
                auxname.c_str(), Snapshot( *dcsrc)[ *dcsrc->channel()].to_str().c_str(), v);
        }
        if(flowctrl) {
            flowctrl->iterate_commit([=](Transaction &tr){
                if(tr[ *flowctrl->target()] != v)
                    tr[ *flowctrl->target()] = v;
             });
            msg += formatString(", %s target=%.3g", auxname.c_str(), v);
        }
    }
    trans( *statusStr()) = msg;
}

bool XTempManager::checkDependency(const Snapshot &shot_this,
    const Snapshot &shot_emitter, const Snapshot &shot_others,
    XDriver *emitter) const {
    const shared_ptr<XTempControl> maindev = shot_this[ *mainDevice()];
    if(emitter != maindev.get())
        return false;
//    if( !shot_this[ *isActivated()])
//        return false;
    return true;
}
void XTempManager::analyze(Transaction &tr, const Snapshot &shot_emitter,
    const Snapshot &shot_others,
    XDriver *emitter) throw (XRecordError&) {
    const Snapshot &shot_this(tr);
    const shared_ptr<XTempControl> maindev = shot_this[ *mainDevice()];
    assert(maindev);
    tr[ *statusStr()] = i18n("No available zone setting.");
    auto currZone = currentZone(shot_this);
    if( !currZone) {
        if( !shot_this.size(zones())) {
            msecsleep(100);
            throw XRecordError(i18n("No available zone setting."), __FILE__, __LINE__);
        }
        m_currZoneNo = 0;
        currZone = currentZone(shot_this);
    }
    tr[ *statusStr()] = formatString("Zone#%d", m_currZoneNo + 1);
    //lambda fn.
    XString chstr;
    auto get_temp = [&](int zoneno)->double {
        auto zone = dynamic_pointer_cast<XZone>(shot_this.list(zones())->at(zoneno));
        int currCh = shot_this[ *zone->channel()];
        double temp = 0.0;
        if((currCh < 0) || (currCh >= shot_emitter.size(maindev->channels())))
            currCh = m_currCh;
        if((currCh < 0) || (currCh >= shot_emitter.size(maindev->channels())))
            currCh = 0;
        chstr = shot_emitter.list(maindev->channels())->at(currCh)->getLabel();
        temp = shot_emitter[ *maindev->entryTemp(currCh)->value()];
//            temp = shot_emitter[ *maindev->sourceTemp(currloop)];
        //converts a raw value to temp using calibration table.
        shared_ptr<XThermometer> thermo = shot_this[ *zone->thermometer()];
        if( !thermo) thermo = m_currThermometer;
        if(thermo) {
            temp = thermo->getTemp(shot_emitter[ *maindev->entryRaw(currCh)->value()]);
            chstr += "(" + thermo->getLabel() + ")";
        }
        return temp;
    };
    double temp = get_temp(currentZoneNo());
    chstr = maindev->getLabel() + " " + chstr;
    tr[ *m_tempStatusStr] = i18n("Temperature") + " [" + chstr + "]";
    m_entryTemp->value(tr, temp);

    //calculates std. deviations in some periods
    double tau = 10.0;
    XTime newtime = XTime::now();
    double dt = newtime - m_lasttime;
    m_lasttime = newtime;
    double terr = temp - shot_this[ *targetTemp()];
    terr = std::min(fabs(terr), (double)shot_this[ *targetTemp()]);
    m_tempAvg = (m_tempAvg - temp) * exp( -dt / tau) + temp;
    m_tempErrAvg = (m_tempErrAvg - terr * terr) * exp( -dt / tau) + terr * terr;
    m_tempErrAvg = std::min(m_tempErrAvg, temp * temp * 0.04);
    tr[ *m_stabilized] = sqrt(m_tempErrAvg); //std.dev.
    tr[ *statusStr()] = XString(shot_this[ *statusStr()]) +
            formatString(", %.2gsec.deviation=%.2g K", tau, sqrt(m_tempErrAvg));

    int currloop = shot_this[ *currZone->loop()];
    if((currloop < 0) || (currloop >= maindev->numOfLoops()))
        currloop = m_currLoopNo;
    shared_ptr<XFlowControllerDriver> extflowctrl = shot_this[ *extDevice()];
    shared_ptr<XDCSource> extdcsrc = shot_this[ *extDevice()];

    double power = 0.0;
    if(extflowctrl) {
        power = shot_others[ *extflowctrl->flow()->value()];
        tr[ *m_heaterStatusStr] = i18n("Mass Flow") +
            formatString(" (%s) [%s max=%.3g]",
            shot_others[ *extflowctrl].unit().c_str(),
            extflowctrl->getLabel().c_str(),
            shot_others[ *extflowctrl].fullScale());
    }
    else if(extdcsrc) {
        power = shot_others[ *extdcsrc->value()];
        double limit_max = extdcsrc->max(currloop, false);
        tr[ *m_heaterStatusStr] = i18n("Voltage/Current") +
            formatString(" [%s %s max=%.3g]", extdcsrc->getLabel().c_str(),
            shot_others[ *extdcsrc->channel()].to_str().c_str(), limit_max);
    }
    else {
        if((currloop >= 0) && (currloop < maindev->numOfLoops())) {
            power = shot_emitter[ *maindev->heaterPower(currloop)];
            tr[ *m_heaterStatusStr] = i18n("Heater Power") +
                formatString(" (%%) [%s %s]", maindev->loopLabel(currloop).c_str(),
                shot_emitter[ *maindev->powerRange(currloop)].to_str().c_str());
        }
    }
    m_entryPow->value(tr, power);

    double signed_ramprate = -0.001;
    if(shot_this[ *isActivated()]) {
        signed_ramprate = ((shot_this[ *targetTemp()] > m_tempStarted) ? 1.0 : -1.0) *
            fabs(shot_this[ *rampRate()]);
        double dt = (XTime::now() - m_timeStarted) * signed_ramprate / 60.0;
        double stemp = m_tempStarted + dt;
        XString msg = formatString(", SetPoint=%.4g K", stemp);
        if(fabs(shot_this[ *targetTemp()] - m_tempStarted) > fabs(dt)) {
            stemp = shot_this[ *targetTemp()]; //reached to the target temp.
            msg = formatString(", SetPoint(=Target)=%.4g K", stemp);
            if(shot_this[ *stabilized()] < fabs(signed_ramprate) * 1.0) {
                signed_ramprate *= shot_this[ *stabilized()] / fabs(signed_ramprate); //stabilizing.
                msg += ", Stabilizing";
            }
        }
        if(fabs(m_setpointTemp - stemp) > fabs(signed_ramprate) / 60 * 3) {
            m_setpointTemp = stemp; //every 3 sec.
        }
        msg += formatString(", Rate=%.3g K/min.", signed_ramprate);
        tr[ *statusStr()] = shot_this[ *statusStr()].to_str() + msg;
    }

    int nextzone = firstMatchingZone(shot_this, temp, signed_ramprate, true);
    if(nextzone < 0) {
        if(shot_this[ *isActivated()]) {
            tr[ *isActivated()] = false;
            throw XRecordError(i18n("Temperature exceeds the limitation."), __FILE__, __LINE__);
        }
        throw XSkippedRecordError(__FILE__, __LINE__);
    }
    double temp_plus_hys = temp * (1 + 1e-2 * shot_this[ *hysteresisOnZoneTr()]);
    double temp_minus_hys = temp * (1 - 1e-2 * shot_this[ *hysteresisOnZoneTr()]);
    int upperzone = firstMatchingZone(shot_this, temp_plus_hys, signed_ramprate);
    int lowerzone = firstMatchingZone(shot_this, temp_minus_hys, signed_ramprate);
    if((currentZoneNo() != upperzone) && (currentZoneNo() != lowerzone)) {
        m_currZoneNo = nextzone;
        currZone = currentZone(shot_this);
//        temp = get_temp(currentZoneNo());
    }

    if(shot_this[ *doesMixTemp()] && (upperzone >= 0) && (lowerzone >= 0)) {
        double temp_u = get_temp(upperzone);
        auto chstr_u = chstr;
        double temp_l = get_temp(lowerzone);
        auto zone = dynamic_pointer_cast<XZone>(shot_this.list(zones())->at(lowerzone));
        double x = shot_this[ *zone->upperTemp()];
        x = (temp - temp_minus_hys) / (temp_plus_hys - temp_minus_hys);
        if((x > 0.0) && (x < 1.0))
            temp = temp_u * x + temp_l * (1 - x);
        tr[ *m_tempStatusStr] = i18n("Temperature") + " [" + chstr_u +
            formatString(" %.2g%% + %s %.2g%%]", 1e2 * (1 - x), chstr.c_str(), x * 1e2);
    }
    m_entryTemp->value(tr, temp);
}

void
XTempManager::onTargetChanged(const Snapshot &, XValueNodeBase *) {
    Snapshot shot( *this);
    if(fabs(shot[ *m_rampRate]) * 60 * 24 * 10 <
        fabs(shot[ *m_entryTemp->value()] - shot[ *m_targetTemp])) {
        gWarnPrint(getLabel() + i18n("Too small ramp rate."));
    }
    m_tempStarted = shot[ *m_entryTemp->value()];
    m_timeStarted = XTime::now();
}

int
XTempManager::firstMatchingZone(const Snapshot &shot, double temp, double signed_ramprate,
    bool update_missinginfo) {
   assert(shot.size(zones()));
   int zno = -1;
   for(auto &&x: *shot.list(zones())) {
       auto zone = dynamic_pointer_cast<XZone>(x);
       if((temp > shot[ *zone->upperTemp()]) || (signed_ramprate > shot[ *zone->maxRampRate()]))
           return zno;
       zno++;
       if(update_missinginfo) {
           int currloop = shot[ *zone->loop()];
           if(currloop >= 0)
                m_currLoopNo = currloop;
           shared_ptr<XThermometer> thermo = shot[ *zone->thermometer()];
           if(thermo) m_currThermometer = thermo;
           int ch = shot[ *zone->channel()];
           if(ch >= 0)
               m_currCh = ch;
           int exc = shot[ *zone->excitation()];
           if(exc >= 0)
               m_currExcitaion = exc;
       }
   }
   return zno;
}

void
XTempManager::onActivateChanged(const Snapshot &shot, XValueNodeBase *) {
    Snapshot shot_this( *this);
    bool actv = shot_this[ *isActivated()];
    shared_ptr<XTempControl> maindev = shot_this[ *mainDevice()];
    m_lsnOnTargetChanged.reset();
    if(actv && !maindev) {
        trans( *isActivated()) = false;
        gErrPrint(i18n("None of device is under control."));
        return;
    }
    if(actv) {
        iterate_commit([=](Transaction &tr){
            m_lsnOnTargetChanged = tr[ *m_targetTemp].onValueChanged().connectWeakly(
                shared_from_this(), &XTempManager::onTargetChanged);
            tr[ *m_rampRate].onValueChanged().connect(m_lsnOnTargetChanged);
        });
    }
    iterate_commit([=](Transaction &tr){
        {
            std::deque<shared_ptr<XNode>> uis = {
                targetTemp(), rampRate()
            };
            for(auto &&ui : uis)
                tr[ *ui].setUIEnabled(actv);
        }
        {
            std::deque<shared_ptr<XNode>> uis = {
                dupZone(), delZone(),
                extDevice(), extIsPositive(),
                mainDevice()
            };
            for(auto &&ui : uis)
                tr[ *ui].setUIEnabled( !actv);
            for(unsigned int i = 0; i < maxNumOfAUXDevices; ++i) {
                tr[ *auxDevice(i)].setUIEnabled( !actv);
                tr[ *auxDevCh(i)].setUIEnabled( !actv);
                tr[ *auxDevMode(i)].setUIEnabled( !actv);
            }
        }
    });
//    dynamic_pointer_cast<XTempControl>(maindev)->targetTemp()->setUIEnabled( !avtv);
    if( !actv) {
        //resets status for stabilized, PID control.
        m_tempAvg = 0.0;
        m_tempErrAvg = 0.0;
        m_lasttime = XTime::now();

        m_pidAccum = 0;
        m_pidLastTime = XTime::now();
        m_pidLastTemp = 0.0;
        //turns off.
        shared_ptr<XFlowControllerDriver> extflowctrl = shot_this[ *extDevice()];
        shared_ptr<XDCSource> extdcsrc = shot_this[ *extDevice()];
        if(extflowctrl)
            trans( *extflowctrl->target()) = 0.0;
        else if(extdcsrc)
            trans( *extdcsrc->value()) = 0.0;
        else if(maindev) {
            if((m_currLoopNo >= 0) && (m_currLoopNo < maindev->numOfLoops()))
                trans( *maindev->targetTemp(m_currLoopNo)) = 0.0;
        }
    }
}

void
XTempManager::onMainDeviceChanged(const Snapshot &shot, XValueNodeBase *) {
    refreshZoneUIs();

    m_lsnMainDevOnListChanged.reset();
    shared_ptr<XTempControl> maindev = shot[ *mainDevice()];
    if(maindev) {
        maindev->iterate_commit([=](Transaction &tr){
            for(unsigned int i = 0; i < maindev->numOfLoops(); ++i) {
                if( !m_lsnMainDevOnListChanged) {
                    m_lsnMainDevOnListChanged =
                        tr[ *maindev->powerRange(i)].onListChanged().connectWeakly(
                            shared_from_this(), &XTempManager::onChListChanged,
                            Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP);
                }
                else
                    tr[ *maindev->powerRange(i)].onListChanged().connect(m_lsnMainDevOnListChanged);
                tr[ *maindev->currentChannel(i)].onListChanged().connect(m_lsnMainDevOnListChanged);
            }
        });
    }
}

void
XTempManager::onAUXDeviceChanged(const Snapshot &, XValueNodeBase *) {
    Snapshot shot( *this);
//    refreshZoneUIs();
    for(unsigned int j = 0; j < maxNumOfAUXDevices; ++j) {
        m_lsnAUXDevOnListChanged[j].reset();
//        shared_ptr<XTempControl> tempctl = shot[ *subDevice(j)];
        shared_ptr<XDCSource> dcsrc = shot[ *auxDevice(j)];
        if(dcsrc) {
            dcsrc->iterate_commit([=](Transaction &tr){
                if( !m_lsnAUXDevOnListChanged[j]) {
                    m_lsnAUXDevOnListChanged[j] =
                        tr[ *dcsrc->channel()].onListChanged().connectWeakly(
                            shared_from_this(), &XTempManager::onChListChanged,
                            Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP);
                }
                else {
                    tr[ *dcsrc->channel()].onListChanged().connect(m_lsnAUXDevOnListChanged[j]);
                }
            });
        }
    }
    onChListChanged(shot, {Snapshot{ *this}, nullptr});
}
void
XTempManager::onExtDeviceChanged(const Snapshot &shot, XValueNodeBase *) {
    refreshZoneUIs();

    m_lsnExtDevOnListChanged.reset();
    shared_ptr<XDCSource> dcsrc = shot[ *extDevice()];
    if(dcsrc) {
        dcsrc->iterate_commit([=](Transaction &tr){
            m_lsnExtDevOnListChanged =
                tr[ *dcsrc->channel()].onListChanged().connectWeakly(
                    shared_from_this(), &XTempManager::onChListChanged,
                    Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP);
        });
    }
}

void
XTempManager::onChListChanged(const Snapshot &, XItemNodeBase::Payload::ListChangeEvent e) {
    refreshZoneUIs();
    Snapshot shot(*this);
    iterate_commit([=](Transaction &tr){
        for(unsigned int j = 0; j < maxNumOfAUXDevices; ++j) {
            shared_ptr<XTempControl> tempctl = shot[ *auxDevice(j)];
            shared_ptr<XDCSource> dcsrc = shot[ *auxDevice(j)];
//            shared_ptr<XFlowControllerDriver> flowctl = shot[ *subDevice(j)];
            tr[ *auxDevCh(j)].clear();
            tr[ *auxDevCh(j)].setUIEnabled(false);
            tr[ *auxDevMode(j)].clear();
            tr[ *auxDevMode(j)].setUIEnabled(false);
            if(tempctl) {
                tr[ *auxDevMode(j)].add({"Target Temp.", "Manual Heater"});
                tr[ *auxDevMode(j)].setUIEnabled(true);
                for(int i = 0; i < tempctl->numOfLoops(); ++i)
                    tr[ *auxDevCh(j)].add(tempctl->loopLabel(i));
                tr[ *auxDevCh(j)].setUIEnabled(true);
            }
            if(dcsrc) {
                tr[ *auxDevCh(j)].add(dcsrc->channel()->itemStrings(Snapshot( *dcsrc)));
                tr[ *auxDevCh(j)].setUIEnabled(true);
            }
        }
    });
}

void
XTempManager::refreshZoneUIs() {
    Snapshot shot( *this);
    auto tbl = m_form->m_tblZone;
    QStringList labels;
    labels += i18n("UpperTemp.");
    labels += i18n("MaxRampRate");
    labels += i18n("Ch.");
    labels += i18n("Exc.");
    labels += i18n("Cal.Tbl.");
    labels += i18n("Loop#");
    labels += i18n("MaxPow");
    labels += i18n("P");
    labels += i18n("I");
    labels += i18n("D");
    labels += i18n("AUX1Val");
    labels += i18n("AUX2Val");
    labels += i18n("AUX3Val");
    labels += i18n("AUX4Val");
    labels += i18n("AUX5Val");
    labels += i18n("AUX6Val");
    tbl->setColumnCount(labels.size());
    double sizes[] = {2.0, 2.0, 2.0, 2.0, 2.5, 1.5, 2.0,
        1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
    for(int i = 0; i < labels.size(); ++i) {
        tbl->setColumnWidth(i, (int)(sizes[i] * 30));
    }
    tbl->setHorizontalHeaderLabels(labels);

    m_conZoneUIs.clear();
    m_conZoneUIs.resize(shot.size(zones()));
    tbl->setRowCount(shot.size(zones()));
    delZone()->setUIEnabled(false);
    if(shot.size(zones())) {
        delZone()->setUIEnabled(true);
        for(int i = 0; i < shot.size(zones()); ++i) {
            auto zone = dynamic_pointer_cast<XZone>(shot.list(zones())->at(i));
            auto &uis = m_conZoneUIs[i].conUIs;
            auto le = new QLineEdit(m_form->m_tblZone);
            int col = 0;
            tbl->setCellWidget(i, col++, le);
            uis.push_back(xqcon_create<XQLineEditConnector>(zone->upperTemp(), le));
            le = new QLineEdit(m_form->m_tblZone);
            tbl->setCellWidget(i, col++, le);
            uis.push_back(xqcon_create<XQLineEditConnector>(zone->maxRampRate(), le));
            auto cmb = new QComboBox(m_form->m_tblZone);
            tbl->setCellWidget(i, col++, cmb);
            uis.push_back(xqcon_create<XQComboBoxConnector>(zone->channel(), cmb, Snapshot( *zone->channel())));
            cmb = new QComboBox(m_form->m_tblZone);
            tbl->setCellWidget(i, col++, cmb);
            uis.push_back(xqcon_create<XQComboBoxConnector>(zone->excitation(), cmb, Snapshot( *zone->excitation())));
            cmb = new QComboBox(m_form->m_tblZone);
            tbl->setCellWidget(i, col++, cmb);
            uis.push_back(xqcon_create<XQComboBoxConnector>(zone->thermometer(), cmb, Snapshot( *zones()->thermometers())));
            cmb = new QComboBox(m_form->m_tblZone);
            tbl->setCellWidget(i, col++, cmb);
            uis.push_back(xqcon_create<XQComboBoxConnector>(zone->loop(), cmb, Snapshot( *zone->loop())));
            cmb = new QComboBox(m_form->m_tblZone);
            tbl->setCellWidget(i, col++, cmb);
            uis.push_back(xqcon_create<XQComboBoxConnector>(zone->powerRange(), cmb, Snapshot( *zone->powerRange())));
            le = new QLineEdit(m_form->m_tblZone);
            tbl->setCellWidget(i, col++, le);
            uis.push_back(xqcon_create<XQLineEditConnector>(zone->prop(), le));
            le = new QLineEdit(m_form->m_tblZone);
            tbl->setCellWidget(i, col++, le);
            uis.push_back(xqcon_create<XQLineEditConnector>(zone->interv(), le));
            le = new QLineEdit(m_form->m_tblZone);
            tbl->setCellWidget(i, col++, le);
            uis.push_back(xqcon_create<XQLineEditConnector>(zone->deriv(), le));
            for(unsigned int j = 0; j < maxNumOfAUXDevices; ++j) {
                le = new QLineEdit(m_form->m_tblZone);
                tbl->setCellWidget(i, col++, le);
                uis.push_back(xqcon_create<XQLineEditConnector>(zone->auxDeviceValues(j), le));
            }
        }
    }
    iterate_commit([=](Transaction &tr){
        const Snapshot &shot(tr);
        int loop = 0;
        if(shot.size(zones())) {
            for(int i = 0; i < shot.size(zones()); ++i) {
                auto zone = dynamic_pointer_cast<XZone>(shot.list(zones())->at(i));
                tr[ *zone->loop()].clear();
                tr[ *zone->channel()].clear();
                tr[ *zone->excitation()].clear();
                tr[ *zone->powerRange()].clear();
                shared_ptr<XDCSource> dcsrc = shot[ *extDevice()];
                shared_ptr<XTempControl> maindev = shot[ *mainDevice()];
                if(maindev) {
                    if(dcsrc) {
                        tr[ *zone->loop()].add(dcsrc->channel()->itemStrings(Snapshot( *dcsrc)));
                        loop = 0;
                    }
                    else {
                        for(unsigned int j = 0; j < maindev->numOfLoops(); ++j)
                            tr[ *zone->loop()].add(maindev->loopLabel(j));
                        if(shot[ *zone->loop()] >= 0)
                            loop = shot[ *zone->loop()];
                    }
                    if(loop < maindev->numOfLoops()) {
                        tr[ *zone->channel()].add(maindev->currentChannel(loop)->itemStrings(Snapshot( *maindev)));
                        auto cch = dynamic_pointer_cast<XTempControl::XChannel>(maindev->currentChannel(loop));
                        if(cch)
                            tr[ *zone->excitation()].add(cch->excitation()->itemStrings(Snapshot( *maindev)));
                        if(dcsrc)
                            tr[ *zone->powerRange()].add(dcsrc->range()->itemStrings(Snapshot( *dcsrc)));
                        else
                            tr[ *zone->powerRange()].add(maindev->powerRange(loop)->itemStrings(Snapshot( *maindev)));
                    }
                }
            }
        }
    });
}

void
XTempManager::onDupTouched(const Snapshot &shot, XTouchableNode *) {
    zones()->thermometers()->iterate_commit_if([=](Transaction &tr_th){
        Transaction tr( *this);
        //nameless
        auto zone = zones()->create<XZone>(
            tr, "", false, tr_th, zones()->thermometers());
        int i = m_form->m_tblZone->currentRow();
        if((i >= 0) && (i + 1 < tr.size(zones()))) {
            //duplicates
            auto zone_old = dynamic_pointer_cast<XZone>(tr.list(zones())->at(i));
            tr[ *zone->upperTemp()] = (double)tr[ *zone_old->upperTemp()];
            tr[ *zone->maxRampRate()] = (double)tr[ *zone_old->maxRampRate()];
            tr[ *zone->prop()] = (double)tr[ *zone_old->prop()];
            tr[ *zone->interv()] = (double)tr[ *zone_old->interv()];
            tr[ *zone->deriv()] = (double)tr[ *zone_old->deriv()];
            for(int i = 0; i < maxNumOfAUXDevices; ++i)
                tr[ *zone->auxDeviceValues(i)] = (double)tr[ *zone_old->auxDeviceValues(i)];
            zones()->swap(tr, zone, tr.list(zones())->at(i + 1));
        }
        return tr.commit();
    });
    refreshZoneUIs();
}

void
XTempManager::onDeleteTouched(const Snapshot &shot, XTouchableNode *) {
    iterate_commit_while([=](Transaction &tr){
        int i = m_form->m_tblZone->currentRow();
        if((i < 0) || (i >= tr.size(zones())))
            return false;
        zones()->release(tr, tr.list(zones())->at(i));
        return true;
    });
    refreshZoneUIs();
}

double
XTempManager::pid(const Snapshot &shot, XTime time, double sv, double pv) {
    auto zone = currentZone(shot);
    if( !zone) return 0.0;
    double p = shot[ *zone->prop()];
    double i = shot[ *zone->interv()];
    double d = shot[ *zone->deriv()];

    double dt = pv - sv;
    if(shot[ *m_extIsPositive])
        dt *= -1.0;
    double dxdt = 0.0;
    double acc = 0.0;
    if((i > 0) && (time - m_pidLastTime < i)) {
        m_pidAccum += (time - m_pidLastTime) * dt;
        dxdt = (pv - m_pidLastTemp) / (time - m_pidLastTime);
        acc = m_pidAccum / i;
        acc = -std::min(std::max( -acc * p, -2.0), 100.0) / p;
        m_pidAccum = acc * i;
    }
    else
        m_pidAccum = 0;

    m_pidLastTime = time;
    m_pidLastTemp = pv;

    return -(dt + acc + dxdt * d) * p;
}
