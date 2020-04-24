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

#ifndef pfeifferprotocolH
#define pfeifferprotocolH

#include "charinterface.h"
#include "chardevicedriver.h"

//! Pfeiffer protocol for RS485
class XPfeifferProtocolInterface : public XCharInterface {
public:
    XPfeifferProtocolInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
    virtual ~XPfeifferProtocolInterface() {}

    enum class DATATYPE {
        BOOLEAN_OLD, U_INTEGER, U_REAL, STRING, BOOLEAN_NEW, U_SHORT_INT, U_EXPO_NEW, STRING_LONG
    };
    bool requestBool(unsigned int address, DATATYPE data_type, unsigned int param_no);
    unsigned int requestUInt(unsigned int address, DATATYPE data_type, unsigned int param_no);
    double requestReal(unsigned int address, DATATYPE data_type, unsigned int param_no);
    XString requestString(unsigned int address, DATATYPE data_type, unsigned int param_no);
    template <typename X>
    void control(unsigned int address,
        DATATYPE data_type, unsigned int param_no, X data);
    void control(unsigned int address,
        DATATYPE data_type, unsigned int param_no, bool data);
    void control(unsigned int address,
        DATATYPE data_type, unsigned int param_no, unsigned int data);
    void control(unsigned int address,
        DATATYPE data_type, unsigned int param_no, const XString &data);
    void control(unsigned int address,
        DATATYPE data_type, unsigned int param_no, double data);
protected:
    virtual void open() throw (XInterfaceError &);
    //! This can be called even if has already closed.
    virtual void close() throw (XInterfaceError &);

    virtual bool isOpened() const {return !!m_openedPort;}
private:
    shared_ptr<XPort> m_openedPort;
    static XMutex s_lock;
    static std::deque<weak_ptr<XPort> > s_openedPorts; //guarded by s_lock.

    XString action(unsigned int address,
        bool iscontrol, unsigned int param_no, const XString &str);
};

template <class T>
class XPfeifferProtocolDriver : public XCharDeviceDriver<T, XPfeifferProtocolInterface> {
public:
    XPfeifferProtocolDriver(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
        XCharDeviceDriver<T, XPfeifferProtocolInterface>(name, runtime, ref(tr_meas), meas) {}
    virtual ~XPfeifferProtocolDriver() {}
};

#endif
