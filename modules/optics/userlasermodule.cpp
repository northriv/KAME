/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/

#include "userlasermodule.h"
#include "charinterface.h"
#include "analyzer.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>

REGISTER_TYPE(XDriverList, CoherentStingray, "Coherent Stingray Laser module");
REGISTER_TYPE(XDriverList, LDX3200, "Newport/ILX LDX-3200 series precision current source");
REGISTER_TYPE(XDriverList, LDC3700, "Newport/ILX LDC-3700(C) series laser controller");
REGISTER_TYPE(XDriverList, LDC3900, "Newport/ILX LDC-3900 series modular laser diode controller (up to 4 channels)");

//! Parses a SCPI ERR?/ERRors? response, which may hold UP TO 10 comma-separated integer error
//! codes (e.g. "201,407"; "0" means no errors), into a summary of ALL queued codes -- a naive
//! toInt() on the raw string only captures the first and drops the rest. Confirmed against the
//! official LDX-3200 / LDC-3700(C) manuals (both document ERR? as a comma-separated list) and
//! the LDC-3900 manual. Returns an empty string when there are no (nonzero) errors queued.
static XString
formatAllErrorCodes(const XString &raw) {
    XString errs;
    std::string s(raw);
    size_t pos = 0;
    while(pos <= s.size()) {
        size_t comma = s.find(',', pos);
        std::string tok = s.substr(pos, (comma == std::string::npos) ? std::string::npos : comma - pos);
        int code = atoi(tok.c_str());
        if(code)
            errs += (errs.empty() ? "" : ",") + std::to_string(code);
        if(comma == std::string::npos)
            break;
        pos = comma + 1;
    }
    return errs;
}

// ===========================================================================================
// XCoherentStingray (serial, fixed single laser + read-only diode temperature)
// ===========================================================================================
XCoherentStingray::XCoherentStingray(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XLaserModule>(name, runtime, ref(tr_meas), meas) {
    trans(*interface()->device()) = "SERIAL";
    interface()->setSerialBaudRate(115200);
    createLaserChannels(tr_meas, meas, 1);
    createTecChannels(tr_meas, meas, 1);
    // Stingray's output is fixed: current/power are read-only, and its "TEC" line is a diode-
    // temperature readback only. Permanently disable the setpoints/TEC-enable (irreversible
    // disable() is intended here); only the laser output on/off stays controllable.
    iterate_commit([=](Transaction &tr){
        tr[ *laserChannel(1)->setCurrent()].disable();
        tr[ *laserChannel(1)->setPower()].disable();
        tr[ *tecChannel(1)->setTemp()].disable();
        tr[ *tecChannel(1)->enabled()].disable();
    });
}
bool
XCoherentStingray::readLaser(unsigned int, double &current_mA, double &power_mW,
    double &voltage_V, bool &output_on) {
    interface()->query("SOUR:CURR:LEV?");
    current_mA = interface()->toDouble();
    interface()->query("SOUR:POW:LEV?"); //[W]
    power_mW = interface()->toDouble() * 1e3;
    voltage_V = 0.0; //no voltage readback on this instrument.
    output_on = false; //this instrument is not queried for its output state.
    return true;
}
bool
XCoherentStingray::readTec(unsigned int, double &temp_C, bool &output_on) {
    interface()->query("SOUR:TEMP:DIOD?");
    temp_C = interface()->toDouble();
    output_on = false;
    return true;
}
void
XCoherentStingray::setLaserOutput(unsigned int, bool on) {
    interface()->query(XString("SOUR:AM:STAT ") + (on ? "ON" : "OFF"));
}

// ===========================================================================================
// XLDX3200 (GPIB, single laser current source, no TEC)
// ===========================================================================================
XLDX3200::XLDX3200(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XLaserModule>(name, runtime, ref(tr_meas), meas) {
    trans(*interface()->device()) = "GPIB";
    createLaserChannels(tr_meas, meas, 1);
}
bool
XLDX3200::readLaser(unsigned int, double &current_mA, double &power_mW,
    double &voltage_V, bool &output_on) {
    // LAS:LDI/MDP/LDV units are mA/mW/V -- confirmed against the official Newport LDX-3200
    // (doc 70028204) and LDC-3700C (doc 70041001) manuals; no scale factor (a previous uA
    // scaling was a confirmed, now-removed bug -- see git history).
    interface()->query("LAS:LDI?"); //[mA]
    current_mA = interface()->toDouble();
    interface()->query("LAS:MDP?"); //[mW]
    power_mW = interface()->toDouble();
    interface()->query("LAS:LDV?"); //[V]
    voltage_V = interface()->toDouble();
    interface()->query("LAS:OUT?");
    output_on = interface()->toInt() > 0;
    return true;
}
void
XLDX3200::setLaserCurrent(unsigned int, double mA) {
    interface()->sendf("LAS:LDI %f", mA);
}
void
XLDX3200::setLaserPower(unsigned int, double mW) {
    interface()->sendf("LAS:MDP %f", mW);
}
void
XLDX3200::setLaserOutput(unsigned int, bool on) {
    interface()->query(XString("LAS:OUT ") + (on ? "ON" : "OFF"));
}
XString
XLDX3200::readErrors() {
    interface()->query("ERR?");
    return formatAllErrorCodes(interface()->toStr());
}

// ===========================================================================================
// XLDC3700 (GPIB, LDX-3200 laser + TEC)
// ===========================================================================================
XLDC3700::XLDC3700(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XLDX3200(name, runtime, ref(tr_meas), meas) {
    // XLDX3200's constructor already created the single laser channel; add the TEC channel.
    createTecChannels(tr_meas, meas, 1);
}
bool
XLDC3700::readTec(unsigned int, double &temp_C, bool &output_on) {
    interface()->query("TEC:T?"); //[degC]
    temp_C = interface()->toDouble();
    interface()->query("TEC:OUT?");
    output_on = interface()->toInt() > 0;
    return true;
}
void
XLDC3700::setTecTemp(unsigned int, double degC) {
    interface()->sendf("TEC:T %f", degC);
}
void
XLDC3700::setTecOutput(unsigned int, bool on) {
    // XLDC3700's previous implementation never sent TEC:OUT at all; this now toggles it.
    interface()->query(XString("TEC:OUT ") + (on ? "ON" : "OFF"));
}

// ===========================================================================================
// XLDC3900 (GPIB modular mainframe, 4 laser + 4 TEC channels, one shared interface)
// ===========================================================================================
XLDC3900::XLDC3900(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XLaserModule>(name, runtime, ref(tr_meas), meas) {
    trans(*interface()->device()) = "GPIB";
    createLaserChannels(tr_meas, meas, 4);
    createTecChannels(tr_meas, meas, 4);
}
bool
XLDC3900::readLaser(unsigned int slot, double &current_mA, double &power_mW,
    double &voltage_V, bool &output_on) {
    // Every command carries its own "LAS:CHAN n;" prefix (SCPI tree-walking: after "LAS:CHAN n"
    // the remembered level is "LAS:", so ";LDI?" resolves there in one round trip). Presence is
    // probed via LDI?: an absent laser/combination module returns the "-INF" sentinel (parsed by
    // toDouble() as -inf, not an exception); skip the rest to avoid piling per-cycle E533 "no
    // module" codes into the mainframe error queue.
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "LAS:CHAN %u;LDI?", slot);
    interface()->query(cmd); //[mA]
    double c = interface()->toDouble();
    if( !std::isfinite(c))
        return false;
    current_mA = c;
    snprintf(cmd, sizeof(cmd), "LAS:CHAN %u;MDP?", slot);
    interface()->query(cmd); //[mW]
    power_mW = interface()->toDouble();
    snprintf(cmd, sizeof(cmd), "LAS:CHAN %u;LDV?", slot);
    interface()->query(cmd); //[V]
    voltage_V = interface()->toDouble();
    snprintf(cmd, sizeof(cmd), "LAS:CHAN %u;OUT?", slot);
    interface()->query(cmd);
    output_on = interface()->toInt() > 0;
    return true;
}
bool
XLDC3900::readTec(unsigned int slot, double &temp_C, bool &output_on) {
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "TEC:CHAN %u;T?", slot);
    interface()->query(cmd); //[degC]
    double t = interface()->toDouble();
    if( !std::isfinite(t))
        return false;
    temp_C = t;
    snprintf(cmd, sizeof(cmd), "TEC:CHAN %u;OUT?", slot);
    interface()->query(cmd);
    output_on = interface()->toInt() > 0;
    return true;
}
void
XLDC3900::setLaserCurrent(unsigned int slot, double mA) {
    interface()->sendf("LAS:CHAN %u;LDI %f", slot, mA); //[mA], no scale factor.
}
void
XLDC3900::setLaserPower(unsigned int slot, double mW) {
    interface()->sendf("LAS:CHAN %u;MDP %f", slot, mW); //[mW], no scale factor.
}
void
XLDC3900::setLaserOutput(unsigned int slot, bool on) {
    char cmd[32];
    snprintf(cmd, sizeof(cmd), "LAS:CHAN %u;OUT %s", slot, on ? "ON" : "OFF");
    interface()->query(cmd);
}
void
XLDC3900::setTecTemp(unsigned int slot, double degC) {
    interface()->sendf("TEC:CHAN %u;T %f", slot, degC); //[degC], no scale factor.
}
void
XLDC3900::setTecOutput(unsigned int slot, bool on) {
    char cmd[32];
    snprintf(cmd, sizeof(cmd), "TEC:CHAN %u;OUT %s", slot, on ? "ON" : "OFF");
    interface()->query(cmd);
}
XString
XLDC3900::readErrors() {
    // ERR? is instrument-wide (COMMON query, no CHAN prefix); report all queued codes.
    interface()->query("ERR?");
    return formatAllErrorCodes(interface()->toStr());
}
