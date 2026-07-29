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
#ifndef userrelayH
#define userrelayH

#include "chardevicedriver.h"
#include "dummydriver.h"
#include "relaydriver.h"
#include "xitemnode.h"

class XMotorDriver;

//! LCTech LCUS-x series USB(CH340 serial) relay modules.
//! Write-only protocol; the states cannot be read back.
class XLCUSRelay : public XCharDeviceDriver<XRelayDriver> {
public:
    XLCUSRelay(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas,
        unsigned int num_channels);
    virtual void changeOutput(unsigned int ch, bool on) override;
};

//! LCTech LCUS-1 1ch USB relay module.
class XLCUS1 : public XLCUSRelay {
public:
    XLCUS1(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
        : XLCUSRelay(name, runtime, tr_meas, meas, 1) {}
};
//! LCTech LCUS-2 2ch USB relay module.
class XLCUS2 : public XLCUSRelay {
public:
    XLCUS2(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
        : XLCUSRelay(name, runtime, tr_meas, meas, 2) {}
};
//! LCTech LCUS-4 4ch USB relay module.
class XLCUS4 : public XLCUSRelay {
public:
    XLCUS4(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
        : XLCUSRelay(name, runtime, tr_meas, meas, 4) {}
};
//! LCTech LCUS-8 8ch USB relay module.
class XLCUS8 : public XLCUSRelay {
public:
    XLCUS8(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
        : XLCUSRelay(name, runtime, tr_meas, meas, 8) {}
};

//! Adapter exposing the AUX output bits of a stepping-motor controller
//! (XMotorDriver) as relay channels.
class XRelayViaSTM : public XDummyDriver<XRelayDriver> {
public:
    XRelayViaSTM(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual void changeOutput(unsigned int ch, bool on) override;
protected:
    const shared_ptr<XItemNode<XDriverList, XMotorDriver>> &stm() const {return m_stm;}
private:
    const shared_ptr<XItemNode<XDriverList, XMotorDriver>> m_stm;
    std::deque<xqcon_ptr> m_conUIs;
};

#endif
