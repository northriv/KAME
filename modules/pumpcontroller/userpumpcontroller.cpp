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
    XCharDeviceDriver<XPumpControl>(name, runtime, ref(tr_meas), meas) {
    trans( *interface()->device()) = "SERIAL";
    trans( *interface()->address()) = 1;
    interface()->setSerialBaudRate(9600);
    interface()->setSerialStopBits(1);
    interface()->setSerialParity(XCharInterface::PARITY_NONE);
    interface()->setEOS("\r");
}

XString
XPfeifferTC110::action(const Snapshot &shot_this, bool iscontrol,
    unsigned int param_no, const XString &str) {
    XString buf;
    XScopedLock<XInterface> lock( *interface());
    unsigned int addr = shot_this[ *interface()->address()];
    buf = formatString("%03u%02u%03u%02u", addr,
        iscontrol ? 10u : 0u, param_no, (unsigned int)str.length());
    buf += str;
    unsigned int csum = 0;
    for(auto c: buf)
        csum += c;
    csum = csum % 0x100u;
    buf += formatString("%03u", csum);
    interface()->query(buf.c_str());
    unsigned int res_addr, res_action, res_param_no, res_len;
    if(interface()->scanf("%3u%2u%3u%2u", &res_addr, &res_action, &res_param_no, &res_len) != 4)
        throw XInterface::XConvError(__FILE__, __LINE__);
    if((addr != res_addr) || (res_action != 10) || (param_no != res_param_no))
        throw XInterface::XConvError(__FILE__, __LINE__);
    try {
        buf = &interface()->buffer()[0];
        csum = 0;
        for(size_t i = 0; i < 10 + res_len; ++i)
            csum += buf[i];
        csum = csum % 0x100u;
        unsigned int res_csum = atoi(buf.substr(10 + res_len, 3).c_str());
        if(csum != res_csum)
            throw XInterface::XConvError(__FILE__, __LINE__);
        buf = buf.substr(10, res_len);
    }
    catch (std::out_of_range &) {
        throw XInterface::XConvError(__FILE__, __LINE__);
    }
    if(buf == "NO_DEF")
        throw XInterface::XInterfaceError(
            getLabel() + i18n(": no definition error."), __FILE__, __LINE__);
    if(buf == "_RANGE")
        throw XInterface::XInterfaceError(
            getLabel() + i18n(": out-of-range error."), __FILE__, __LINE__);
    if(buf == "_LOGIC")
        throw XInterface::XInterfaceError(
            getLabel() + i18n(": logic error."), __FILE__, __LINE__);
    return buf;
}

unsigned int
XPfeifferTC110::requestUInt(const Snapshot &shot_this, DATATYPE data_type, unsigned int param_no) {
    auto buf = action(shot_this, false, param_no, "=?");
    int sizes[] = {6, 6, 6, 6, 1, 3, 6, 16};
    if(((int)data_type > 7) || (buf.size() != sizes[(int)data_type]))
        throw XInterface::XConvError(__FILE__, __LINE__);
    return atoi(buf.c_str());
}

bool
XPfeifferTC110::requestBool(const Snapshot &shot_this, DATATYPE data_type, unsigned int param_no) {
    auto buf = action(shot_this, false, param_no, "=?");
    int sizes[] = {6, 6, 6, 6, 1, 3, 6, 16};
    if(((int)data_type > 7) || (buf.size() != sizes[(int)data_type]))
        throw XInterface::XConvError(__FILE__, __LINE__);
    if(buf[0] == '0')
        return false;
    if(buf[1] == '1')
        return true;
    throw XInterface::XConvError(__FILE__, __LINE__);
}

double
XPfeifferTC110::requestReal(const Snapshot &shot_this, DATATYPE data_type, unsigned int param_no) {
    auto buf = action(shot_this, false, param_no, "=?");
    int sizes[] = {6, 6, 6, 6, 1, 3, 6, 16};
    if(((int)data_type > 7) || (buf.size() != sizes[(int)data_type]))
        throw XInterface::XConvError(__FILE__, __LINE__);
    switch (data_type) {
    case DATATYPE::U_REAL:
        return atoi(buf.c_str()) / 100.0;
    case DATATYPE::U_EXPO_NEW:
        unsigned int v, e;
        if(sscanf(buf.c_str(), "%4u%2u", &v, &e) != 2)
            throw XInterface::XConvError(__FILE__, __LINE__);
        return v * pow(10.0, e);
    default:
        throw XInterface::XConvError(__FILE__, __LINE__);
    }
}

XString
XPfeifferTC110::requestString(const Snapshot &shot_this, DATATYPE data_type, unsigned int param_no) {
    auto buf = action(shot_this, false, param_no, "=?");
    int sizes[] = {6, 6, 6, 6, 1, 3, 6, 16};
    if(((int)data_type > 7) || (buf.size() != sizes[(int)data_type]))
        throw XInterface::XConvError(__FILE__, __LINE__);
    switch (data_type) {
    case DATATYPE::STRING:
    case DATATYPE::STRING_LONG:
        return buf;
    default:
        throw XInterface::XConvError(__FILE__, __LINE__);
    }
}


void
XPfeifferTC110::control(const Snapshot &shot_this, DATATYPE data_type, unsigned int param_no, bool data) {
    XString buf;
    switch (data_type) {
    case DATATYPE::BOOLEAN_OLD:
        buf = data ? "111111" : "000000";
        break;
    case DATATYPE::BOOLEAN_NEW:
        buf = data ? "1" : "0";
        break;
    default:
        throw;
    }
    action(shot_this, true, param_no, buf);
}
void
XPfeifferTC110::control(const Snapshot &shot_this, DATATYPE data_type, unsigned int param_no, unsigned int data) {
    XString buf;
    switch (data_type) {
    case DATATYPE::U_INTEGER:
        buf = formatString("%06u", data);
        break;
    case DATATYPE::U_SHORT_INT:
        buf = formatString("%03u", data);
        break;
    default:
        throw;
    }
    action(shot_this, true, param_no, buf);
}
void
XPfeifferTC110::control(const Snapshot &shot_this, DATATYPE data_type, unsigned int param_no, double data) {
    XString buf;
    switch (data_type) {
    case DATATYPE::U_REAL:
        buf = formatString("%06u", (unsigned int)lrint(100 * data));
        break;
    case DATATYPE::U_EXPO_NEW:
    {
        unsigned int e = floor(log10(data));
        unsigned int v = lrint(1000 * data / pow(10.0, e));
        buf = formatString("%04u%02u", v, e);
        break;
    }
    default:
        throw;
    }
    action(shot_this, true, param_no, buf);
}
void
XPfeifferTC110::control(const Snapshot &shot_this, DATATYPE data_type, unsigned int param_no, const XString &data) {
    XString buf;
    switch (data_type) {
    case DATATYPE::STRING:
        buf = formatString("%6s", data.c_str());
        break;
    case DATATYPE::STRING_LONG:
        buf = formatString("%16s", data.c_str());
        break;
    default:
        throw;
    }
    action(shot_this, true, param_no, buf);
}
double
XPfeifferTC110::getRotationSpeed() {
    return requestUInt(Snapshot( *this), DATATYPE::U_INTEGER, 309);
}
double
XPfeifferTC110::getRuntime() {
    return requestUInt(Snapshot( *this), DATATYPE::U_INTEGER, 311);
}
double
XPfeifferTC110::getPressure() {
    return requestUInt(Snapshot( *this), DATATYPE::U_EXPO_NEW, 340);
}
std::deque<XString>
XPfeifferTC110::getTempLabels() {
    return {"Temp Elec", "Temp Pump Btm", "Temp Bearing", "Temp Motor"};
}
std::deque<double>
XPfeifferTC110::getTemps() {
    std::deque<double> temps;
    temps.push_back(requestUInt(Snapshot( *this), DATATYPE::U_INTEGER, 326));
    temps.push_back(requestUInt(Snapshot( *this), DATATYPE::U_INTEGER, 330));
    temps.push_back(requestUInt(Snapshot( *this), DATATYPE::U_INTEGER, 342));
    temps.push_back(requestUInt(Snapshot( *this), DATATYPE::U_INTEGER, 346));
    return temps;
}
std::pair<unsigned int, XString>
XPfeifferTC110::getWarning() {
    auto code = requestString(Snapshot( *this), DATATYPE::STRING, 303);
    if(code.substr(0, 3) == "Wrn")
        return {atoi(code.substr(3, 3).c_str()), code};
    return {0, {}};
}
std::pair<unsigned int, XString>
XPfeifferTC110::getError() {
    auto code = requestString(Snapshot( *this), DATATYPE::STRING, 303);
    if(code.substr(0, 3) == "Err")
        return {atoi(code.substr(3, 3).c_str()), code};
    return {0, {}};
}

void
XPfeifferTC110::changeMode(bool active, bool stby, bool heating) {
    control(Snapshot( *this), DATATYPE::BOOLEAN_OLD, 2, stby);
    control(Snapshot( *this), DATATYPE::BOOLEAN_OLD, 1, heating);
    control(Snapshot( *this), DATATYPE::BOOLEAN_OLD, 10, active);
}

void
XPfeifferTC110::changeMaxDrivePower(double p){
    control(Snapshot( *this), DATATYPE::U_SHORT_INT, 708,
        (unsigned int)std::max(0.0, std::min(100.0, p)));
}

void
XPfeifferTC110::changeStandbyRotationSpeed(double p) {
    control(Snapshot( *this), DATATYPE::U_REAL, 717, std::max(0.0, std::min(100.0, p)));
}
void
XPfeifferTC110::open() throw (XKameError &) {
    double dp = requestUInt(Snapshot( *this), DATATYPE::U_SHORT_INT, 708);
    double rs = requestReal(Snapshot( *this), DATATYPE::U_REAL, 717);
    bool ac = requestBool(Snapshot( *this), DATATYPE::BOOLEAN_OLD, 10);
    bool st = requestBool(Snapshot( *this), DATATYPE::BOOLEAN_OLD, 2);
    bool ht = requestBool(Snapshot( *this), DATATYPE::BOOLEAN_OLD, 1);
    iterate_commit([=](Transaction &tr){
        tr[ *maxDrivePower()] = dp;
        tr[ *standbyRotationSpeed()] = rs;
        tr[ *activate()] = ac;
        tr[ *standby()] = st;
        tr[ *heating()] = ht;
    });

}

void
XPfeifferTC110::closeInterface() {

}
