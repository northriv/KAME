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

#ifndef userpumpcontrollerH
#define userpumpcontrollerH

#include "pumpcontroller.h"
#include "chardevicedriver.h"

//! Pfeiffer TC110
class XPfeifferTC110:public XCharDeviceDriver<XPumpControl> {
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
    virtual void open() throw (XKameError &) override;
    //! Be called for closing interfaces.
    virtual void closeInterface() override;
private:
    XString action(const Snapshot &shot_this,
        bool iscontrol, unsigned int param_no, const XString &str);
    enum class DATATYPE {
        BOOLEAN_OLD, U_INTEGER, U_REAL, STRING, BOOLEAN_NEW, U_SHORT_INT, U_EXPO_NEW, STRING_LONG
    };
    bool requestBool(const Snapshot &shot_this, DATATYPE data_type, unsigned int param_no);
    unsigned int requestUInt(const Snapshot &shot_this, DATATYPE data_type, unsigned int param_no);
    double requestReal(const Snapshot &shot_this, DATATYPE data_type, unsigned int param_no);
    XString requestString(const Snapshot &shot_this, DATATYPE data_type, unsigned int param_no);
    template <typename X>
    void control(const Snapshot &shot_this,
        DATATYPE data_type, unsigned int param_no, X data);
    void control(const Snapshot &shot_this,
        DATATYPE data_type, unsigned int param_no, bool data);
    void control(const Snapshot &shot_this,
        DATATYPE data_type, unsigned int param_no, unsigned int data);
    void control(const Snapshot &shot_this,
        DATATYPE data_type, unsigned int param_no, const XString &data);
    void control(const Snapshot &shot_this,
        DATATYPE data_type, unsigned int param_no, double data);
};



#endif
