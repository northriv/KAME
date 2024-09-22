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

#include "userlasermodule.h"
#include "charinterface.h"

REGISTER_TYPE(XDriverList, CoherentStingray, "Coherent Stingray Laser module");
REGISTER_TYPE(XDriverList, LDX3200, "Newport/ILX LDX-3200 series precision current source");
REGISTER_TYPE(XDriverList, LDC3700, "Newport/ILX LDC-3700(C) series laser controller");

XCoherentStingray::XCoherentStingray(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) : XCharDeviceDriver<XLaserModule>(name, runtime, ref(tr_meas), meas) {
    trans(*interface()->device()) = "SERIAL";
    interface()->setSerialBaudRate(115200);

    std::vector<shared_ptr<XNode>> disable_ui{
        setCurrent(), setPower(), setTemp(),
    };
    iterate_commit([=](Transaction &tr){
        for(auto &&x: disable_ui)
            tr[ *x].disable();
    });
}
XCoherentStingray::ModuleStatus
XCoherentStingray::readStatus() {
    ModuleStatus stat;
    interface()->query("SYST:INF:WAV?");
    stat.status += interface()->toStr() + " nm";
    interface()->query("SYST:INF:POW?");
    stat.status += " " + interface()->toStr() + "W";
    interface()->query("SYST:DIOD:HOUR?");
    stat.status += ", " + interface()->toStr();
    interface()->query("SYST:STAT?");
    stat.status += ", StatWord: " + interface()->toStr();
    interface()->query("SOUR:AM:SOUR?");
    stat.status += ", Source: " + interface()->toStr();
    interface()->query("SOUR:CURR:LEV?");
    stat.current = interface()->toDouble();
    interface()->query("SOUR:POW:LEV?"); //[W]
    stat.power = interface()->toDouble() * 1e3;
    interface()->query("SOUR:TEMP:DIOD?");
    stat.temperature = interface()->toDouble();
    interface()->query("SYST:FAUL?");
    stat.status += ", FaultWord: " + interface()->toStr();
    return stat;

}
void
XCoherentStingray::onEnabledChanged(const Snapshot &shot, XValueNodeBase *node) {
    interface()->query(XString("SOUR:AM:STAT ") + (shot[ *enabled()] ? "ON" : "OFF"));
}

XLDX3200::XLDX3200(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) : XCharDeviceDriver<XLaserModule>(name, runtime, ref(tr_meas), meas) {
    trans(*interface()->device()) = "GPIB";
}
XLDX3200::ModuleStatus
XLDX3200::readStatus() {
    ModuleStatus stat;
    interface()->query("LAS:LDI?"); //[uA]
    stat.current = interface()->toDouble() * 1e-3;
    interface()->query("LAS:MDP?"); //[mW]
    stat.power = interface()->toDouble();
    interface()->query("LAS:LDV?"); //[V]
    stat.voltage = interface()->toDouble();
    interface()->query("LAS:OUT?");
    int i = interface()->toInt();
    stat.status += XString("Laser ") + ((i > 0) ? "ON" : "OFF");
    interface()->query("ERR?");
    i = interface()->toInt();
    if(i)
        stat.status += formatString("Error-%i", i);
    return stat;

}
void
XLDX3200::onEnabledChanged(const Snapshot &shot, XValueNodeBase *node) {
    interface()->query(XString("LAS:OUT ") + (shot[ *enabled()] ? "ON" : "OFF"));
}
void
XLDX3200::onCurrentChanged(const Snapshot &shot, XValueNodeBase *node) {
    interface()->sendf("LAS:LDI %f", shot[ *setCurrent()] * 1e3);
}
void
XLDX3200::onPowerChanged(const Snapshot &shot, XValueNodeBase *node) {
    interface()->sendf("LAS:MDP %f", (double)shot[ *setPower()]);
}


XLDC3700::XLDC3700(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) : XLDX3200(name, runtime, ref(tr_meas), meas) {
}
XLDC3700::ModuleStatus
XLDC3700::readStatus() {
    ModuleStatus stat = XLDX3200::readStatus();
    interface()->query("TEC:T?");
    stat.temperature = interface()->toDouble();
    return stat;
}
void
XLDC3700::onTempChanged(const Snapshot &shot, XValueNodeBase *node) {
    interface()->sendf("TEC:T %f", (double)shot[ *setTemp()]);
}
