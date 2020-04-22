/***************************************************************************
        Copyright (C) 2002-2018 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#include "userpumpcontroller.h"
#include "charinterface.h"

REGISTER_TYPE(XDriverList, PfeifferTC110, "Pfeiffer TC110 turbopump controller");

XPfeifferTC110::XPfeifferTC110(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPfeifferProtocolDriver<XPumpControl>(name, runtime, ref(tr_meas), meas) {
    trans( *interface()->device()) = "SERIAL";
    trans( *interface()->address()) = 1;
}

double
XPfeifferTC110::getRotationSpeed() {
    return interface()->requestUInt(Snapshot( *this)[ *interface()->address()], DATATYPE::U_INTEGER, 309);
}
double
XPfeifferTC110::getRuntime() {
    return interface()->requestUInt(Snapshot( *this)[ *interface()->address()], DATATYPE::U_INTEGER, 311);
}
double
XPfeifferTC110::getPressure() {
    return 0.0;
//    return requestUInt(Snapshot( *this), DATATYPE::U_EXPO_NEW, 340);
}
std::deque<XString>
XPfeifferTC110::getTempLabels() {
    return {"Temp Elec", "Temp Pump Btm", "Temp Bearing", "Temp Motor"};
}
std::deque<double>
XPfeifferTC110::getTemps() {
    unsigned int address = Snapshot( *this)[ *interface()->address()];
    std::deque<double> temps;
    temps.push_back(interface()->requestUInt(address, DATATYPE::U_INTEGER, 326));
    temps.push_back(interface()->requestUInt(address, DATATYPE::U_INTEGER, 330));
    temps.push_back(interface()->requestUInt(address, DATATYPE::U_INTEGER, 342));
    temps.push_back(interface()->requestUInt(address, DATATYPE::U_INTEGER, 346));
    return temps;
}
std::pair<unsigned int, XString>
XPfeifferTC110::getWarning() {
    unsigned int address = Snapshot( *this)[ *interface()->address()];
    auto code = interface()->requestString(address, DATATYPE::STRING, 303);
    if(code.substr(0, 3) == "Wrn")
        return {atoi(code.substr(3, 3).c_str()), code};
    return {0, {}};
}
std::pair<unsigned int, XString>
XPfeifferTC110::getError() {
    unsigned int address = Snapshot( *this)[ *interface()->address()];
    auto code = interface()->requestString(address, DATATYPE::STRING, 303);
    if(code.substr(0, 3) == "Err")
        return {atoi(code.substr(3, 3).c_str()), code};
    return {0, {}};
}

void
XPfeifferTC110::changeMode(bool active, bool stby, bool heating) {
    unsigned int address = Snapshot( *this)[ *interface()->address()];
    interface()->control(address, DATATYPE::BOOLEAN_OLD, 2, stby);
    interface()->control(address, DATATYPE::BOOLEAN_OLD, 1, heating);
    interface()->control(address, DATATYPE::BOOLEAN_OLD, 10, active);
}

void
XPfeifferTC110::changeMaxDrivePower(double p){
    unsigned int address = Snapshot( *this)[ *interface()->address()];
    interface()->control(address, DATATYPE::U_SHORT_INT, 708,
        (unsigned int)std::max(0.0, std::min(100.0, p)));
}

void
XPfeifferTC110::changeStandbyRotationSpeed(double p) {
    unsigned int address = Snapshot( *this)[ *interface()->address()];
    interface()->control(address, DATATYPE::U_REAL, 717, std::max(0.0, std::min(100.0, p)));
}
void
XPfeifferTC110::open() throw (XKameError &) {
    unsigned int address = Snapshot( *this)[ *interface()->address()];
    double dp = interface()->requestUInt(address, DATATYPE::U_SHORT_INT, 708);
    double rs = interface()->requestReal(address, DATATYPE::U_REAL, 717);
    bool ac = interface()->requestBool(address, DATATYPE::BOOLEAN_OLD, 10);
    bool st = interface()->requestBool(address, DATATYPE::BOOLEAN_OLD, 2);
    bool ht = interface()->requestBool(address, DATATYPE::BOOLEAN_OLD, 1);
    iterate_commit([=](Transaction &tr){
        tr[ *maxDrivePower()] = dp;
        tr[ *standbyRotationSpeed()] = rs;
        tr[ *activate()] = ac;
        tr[ *standby()] = st;
        tr[ *heating()] = ht;
    });
    start();
}

