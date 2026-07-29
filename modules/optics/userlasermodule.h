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
#ifndef USERLASERMODULE_H
#define USERLASERMODULE_H

#include "lasermodule.h"
#include "chardevicedriver.h"
//---------------------------------------------------------------------------

//! COHERENT Stingray laser module (serial). Fixed-output single laser: current/power are
//! read-only, only the laser output on/off is controllable. Exposed as 1 laser channel + 1 TEC
//! channel (diode-temperature readback only; TEC setpoint/enable disabled).
class XCoherentStingray : public XCharDeviceDriver<XLaserModule> {
public:
    XCoherentStingray(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XCoherentStingray() {}
protected:
    virtual bool readLaser(unsigned int slot, double &current_mA, double &power_mW,
        double &voltage_V, bool &output_on) override;
    virtual bool readTec(unsigned int slot, double &temp_C, bool &output_on) override;
    virtual void setLaserOutput(unsigned int slot, bool on) override;
};

//! Newport/ILX LDX-3200 series precision current source (GPIB, single laser channel, no TEC).
//! LAS:LDI/MDP units are mA/mW (confirmed against the official LDX-3200 & LDC-3700C manuals;
//! see git history for the removal of a previous erroneous uA scaling).
class XLDX3200 : public XCharDeviceDriver<XLaserModule> {
public:
    XLDX3200(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XLDX3200() {}
protected:
    virtual bool readLaser(unsigned int slot, double &current_mA, double &power_mW,
        double &voltage_V, bool &output_on) override;
    virtual void setLaserCurrent(unsigned int slot, double mA) override;
    virtual void setLaserPower(unsigned int slot, double mW) override;
    virtual void setLaserOutput(unsigned int slot, bool on) override;
    virtual XString readErrors() override;
};

//! Newport/ILX LDC-3700(C) series laser controller (GPIB): an LDX-3200 current source plus a
//! TEC (1 laser + 1 TEC channel). Inherits the laser side from XLDX3200 and adds TEC:T/TEC:OUT.
class XLDC3700 : public XLDX3200 {
public:
    XLDC3700(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XLDC3700() {}
protected:
    virtual bool readTec(unsigned int slot, double &temp_C, bool &output_on) override;
    virtual void setTecTemp(unsigned int slot, double degC) override;
    virtual void setTecOutput(unsigned int slot, bool on) override;
};

//! Newport/ILX LDC-3900 modular mainframe (GPIB): up to 4 slots, each a CSM (current-only),
//! TCM (TEC-only), or LCM (combination) module. Exposed as 4 laser + 4 TEC channels sharing the
//! ONE mainframe interface; an absent half of a slot reports "-INF" and is silently skipped.
//! Every command carries its own "LAS:CHAN n;"/"TEC:CHAN n;" prefix (SCPI tree-walking) so it
//! never relies on a globally-persisted channel selection. All the multi-channel machinery is
//! in XLaserModule; this class only supplies the LDC-3900 SCPI.
class XLDC3900 : public XCharDeviceDriver<XLaserModule> {
public:
    XLDC3900(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XLDC3900() {}
protected:
    virtual bool readLaser(unsigned int slot, double &current_mA, double &power_mW,
        double &voltage_V, bool &output_on) override;
    virtual bool readTec(unsigned int slot, double &temp_C, bool &output_on) override;
    virtual void setLaserCurrent(unsigned int slot, double mA) override;
    virtual void setLaserPower(unsigned int slot, double mW) override;
    virtual void setLaserOutput(unsigned int slot, bool on) override;
    virtual void setTecTemp(unsigned int slot, double degC) override;
    virtual void setTecOutput(unsigned int slot, bool on) override;
    virtual XString readErrors() override;
};

#endif // USERLASERMODULE_H
