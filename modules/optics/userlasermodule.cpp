/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

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

XCoherentStingray::XCoherentStingray(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) : XCharDeviceDriver<XLaserModule>(name, runtime, ref(tr_meas), meas) {
    trans(*interface()->device()) = "SERIAL";
    interface()->setSerialBaudRate(115200);

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
    interface()->query("SOUR:POW:LEV?");
    stat.status += ", Power: " + interface()->toStr() + " [W]";
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
