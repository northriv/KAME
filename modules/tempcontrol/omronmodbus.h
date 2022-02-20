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
//---------------------------------------------------------------------------

#ifndef usertempcontrolH
#define usertempcontrolH

#include "tempcontrol.h"
#include "modbusrtuinterface.h"
#include "chardevicedriver.h"

//OMRON E5*C controller via Modbus link.
class XOmronE5_CModbus : public XModbusRTUDriver<XTempControl> {
public:
    XOmronE5_CModbus(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XOmronE5_CModbus() {}

protected:
    //! reads sensor value from the instrument
    virtual double getRaw(shared_ptr<XChannel> &channel);
    //! reads a value in Kelvin from the instrument
    virtual double getTemp(shared_ptr<XChannel> &channel);
    //! obtains current heater power
    //! \sa m_heaterPowerUnit()
    virtual double getHeater(unsigned int loop);
    //! ex. "W", "dB", or so
    virtual const char *m_heaterPowerUnit(unsigned int loop) {return "%";}

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open();

    virtual void onPChanged(unsigned int loop, double p);
    virtual void onIChanged(unsigned int loop, double i);
    virtual void onDChanged(unsigned int loop, double d);
    virtual void onTargetTempChanged(unsigned int loop, double temp);
    virtual void onManualPowerChanged(unsigned int loop, double pow);
    virtual void onHeaterModeChanged(unsigned int loop, int mode);
    virtual void onPowerRangeChanged(unsigned int loop, int range);
    virtual void onPowerMaxChanged(unsigned int, double v) {}
    virtual void onPowerMinChanged(unsigned int, double v) {}
    virtual void onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch);

    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc);
private:
};

#endif
