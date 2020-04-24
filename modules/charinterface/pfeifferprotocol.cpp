/***************************************************************************
        Copyright (C) 2002-2020 Kentaro Kitagawa
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
#include "pfeifferprotocol.h"

std::deque<weak_ptr<XPort> > XPfeifferProtocolInterface::s_openedPorts;
XMutex XPfeifferProtocolInterface::s_lock;

XPfeifferProtocolInterface::XPfeifferProtocolInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
    XCharInterface(name, runtime, driver) {
       setEOS("\r");
       setSerialBaudRate(9600);
       setSerialStopBits(1);
       setSerialParity(XCharInterface::PARITY_NONE);
}

void
XPfeifferProtocolInterface::open() throw (XInterfaceError &) {
    XScopedLock<XPfeifferProtocolInterface> lock( *this);
    {
        Snapshot shot( *this);
        XScopedLock<XMutex> glock(s_lock);
        for(auto it = s_openedPorts.begin(); it != s_openedPorts.end();) {
            if(auto pt = it->lock()) {
                if(pt->portString() == (XString)shot[ *port()]) {
                    m_openedPort = pt;
                    //The COMM port has been already opened by m_master.
                    return;
                }
                ++it;
            }
            else
                it = s_openedPorts.erase(it); //cleans garbage.
        }
    }
    //Opens new COMM device.
    XCharInterface::open();
    m_openedPort = openedPort();
    s_openedPorts.push_back(m_openedPort);
}
void
XPfeifferProtocolInterface::close() throw (XInterfaceError &) {
    XScopedLock<XPfeifferProtocolInterface> lock( *this);
    XScopedLock<XMutex> glock(s_lock);
    m_openedPort.reset(); //release shared_ptr to the port if any.
    XCharInterface::close(); //release shared_ptr to the port if any.
}

XString
XPfeifferProtocolInterface::action(unsigned int addr, bool iscontrol,
    unsigned int param_no, const XString &str) {
    XScopedLock<XPfeifferProtocolInterface> lock( *this);
    XString buf;
    buf = formatString("%03u%02u%03u%02u", addr,
        iscontrol ? 10u : 0u, param_no, (unsigned int)str.length());
    buf += str;
    unsigned int csum = 0;
    for(auto c: buf)
        csum += c;
    csum = csum % 0x100u;
    buf += formatString("%03u", csum);
    query(buf.c_str());
    unsigned int res_addr, res_action, res_param_no, res_len;
    if(scanf("%3u%2u%3u%2u", &res_addr, &res_action, &res_param_no, &res_len) != 4)
        throw XInterface::XConvError(__FILE__, __LINE__);
    if((addr != res_addr) || (res_action != 10) || (param_no != res_param_no))
        throw XInterface::XConvError(__FILE__, __LINE__);
    try {
        buf = &buffer()[0];
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
XPfeifferProtocolInterface::requestUInt(unsigned int address, DATATYPE data_type, unsigned int param_no) {
    auto buf = action(address, false, param_no, "=?");
    int sizes[] = {6, 6, 6, 6, 1, 3, 6, 16};
    if(((int)data_type > 7) || (buf.size() != sizes[(int)data_type]))
        throw XInterface::XConvError(__FILE__, __LINE__);
    return atoi(buf.c_str());
}

bool
XPfeifferProtocolInterface::requestBool(unsigned int address, DATATYPE data_type, unsigned int param_no) {
    auto buf = action(address, false, param_no, "=?");
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
XPfeifferProtocolInterface::requestReal(unsigned int address, DATATYPE data_type, unsigned int param_no) {
    auto buf = action(address, false, param_no, "=?");
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
        return v * pow(10.0, e - 23);
    default:
        throw XInterface::XConvError(__FILE__, __LINE__);
    }
}

XString
XPfeifferProtocolInterface::requestString(unsigned int address, DATATYPE data_type, unsigned int param_no) {
    auto buf = action(address, false, param_no, "=?");
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
XPfeifferProtocolInterface::control(unsigned int address, DATATYPE data_type, unsigned int param_no, bool data) {
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
    action(address, true, param_no, buf);
}
void
XPfeifferProtocolInterface::control(unsigned int address, DATATYPE data_type, unsigned int param_no, unsigned int data) {
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
    action(address, true, param_no, buf);
}
void
XPfeifferProtocolInterface::control(unsigned int address, DATATYPE data_type, unsigned int param_no, double data) {
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
    action(address, true, param_no, buf);
}
void
XPfeifferProtocolInterface::control(unsigned int address, DATATYPE data_type, unsigned int param_no, const XString &data) {
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
    action(address, true, param_no, buf);
}

