/***************************************************************************
        Copyright (C) 2002-2018 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef userpumpcontrollerH
#define userpumpcontrollerH

#include "pumpcontroller.h"
#include "pfeifferprotocol.h"

//! Pfeiffer TC110
class XPfeifferTC110:public XPfeifferProtocolDriver<XPumpControl> {
public:
    XPfeifferTC110(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    ~XPfeifferTC110() {}

protected:
    //! reads a rotation speed from the instrument
    virtual double getRotationSpeed() override; //[Hz]
    //! reads runtime value from the instrument
    virtual double getRuntime() override; //[hrs]
    //! reads pressure sensor value from the instrument
    virtual double getPressure() override; //[Pa]
    //! reads temperatures from the instrument
    virtual std::deque<XString> getTempLabels() override;
    virtual std::deque<double> getTemps() override; //[degC]
    //! reads warning status from the instrument
    virtual std::pair<unsigned int, XString> getWarning() override;
    //! reads error status from the instrument
    virtual std::pair<unsigned int, XString> getError() override;

    virtual void changeMode(bool active, bool stby, bool heating) override;
    virtual void changeMaxDrivePower(double p) override; //[%]
    virtual void changeStandbyRotationSpeed(double p) override; //[%]

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;
private:
    using DATATYPE = XPfeifferProtocolInterface::DATATYPE;

};

#endif
