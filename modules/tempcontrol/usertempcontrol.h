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
#include "oxforddriver.h"
#include "chardevicedriver.h"
//---------------------------------------------------------------------------
//! ITC503 Oxford
class XITC503 : public XOxfordDriver<XTempControl> {
public:
    XITC503(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XITC503() {}

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

//! Picowatt/Oxford AVS47-IB
//! AVS47 and TS530A
class XAVS47IB:public XCharDeviceDriver<XTempControl> {
public:
    XAVS47IB(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    ~XAVS47IB() {}

protected:
    //! reads sensor value from the instrument
    virtual double getRaw(shared_ptr<XChannel> &channel);
    //! reads a value in Kelvin from the instrument
    virtual double getTemp(shared_ptr<XChannel> &channel);
    //! obtains current heater power
    //! \sa m_heaterPowerUnit()
    virtual double getHeater(unsigned int loop);
    //! ex. "W", "dB", or so
    virtual const char *m_heaterPowerUnit(unsigned int loop) {return "W";}

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open();
    //! Be called for closing interfaces.
    virtual void closeInterface();

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
    double read(const char *str);

    void setTemp(double temp);
    void setHeaterMode(int /*mode*/) {}
    int setPoint();
    //AVS-47 COMMANDS
    int setRange(unsigned int range);
    double getRes();
    int getRange();
    //TS-530 COMMANDS
    int setBias(unsigned int bias);
    void setPowerRange(int range);

    int m_autorange_wait;
};

class XCryoconCharInterface : public XCharInterface {
public:
    XCryoconCharInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
        : XCharInterface(name, runtime, driver) {}
    virtual ~XCryoconCharInterface() {}
    virtual void send(const XString &str) { ::msecsleep(20); XCharInterface::send(str);}
    virtual void send(const char *str) { ::msecsleep(20); XCharInterface::send(str);}
    virtual void write(const char *sendbuf, int size)  { ::msecsleep(20); XCharInterface::write(sendbuf, size);}
};

//! Cryo-con base class
class XCryocon : public XCharDeviceDriver<XTempControl, XCryoconCharInterface> {
public:
    XCryocon(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XCryocon() {}

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
    virtual void onPowerMinChanged(unsigned int loop, double v) {}
    virtual void onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch);

    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc);
private:
    void setTemp(unsigned int loop, double temp);
    //        void SetChannel(XChannel *channel);
    void setHeaterMode(unsigned int loop);
    void getChannel(unsigned int loop);
    int control();
    int stopControl();
    double getInput(shared_ptr<XChannel> &channel);
    int setHeaterSetPoint(unsigned int loop, double value);
protected:
    virtual const char *loopString(unsigned int loop) = 0;
};

//! Cryo-con Model 32 Cryogenic Inst.
class XCryoconM32:public XCryocon {
public:
    XCryoconM32(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XCryoconM32() {}

protected:
    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open();

    virtual void onPowerMaxChanged(unsigned int loop, double v);
    virtual const char *loopString(unsigned int loop) {
        return (loop == 0) ? "LOOP 1" : "LOOP 2";
    }
};

//! Cryo-con Model 62 Cryogenic Inst.
class XCryoconM62 : public XCryocon {
public:
    XCryoconM62(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XCryoconM62() {}

protected:
    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open();

    virtual void onPowerMaxChanged(unsigned int loop, double v) {}
    virtual const char *loopString(unsigned int loop) {
        return (loop == 0) ? "HEATER" : "AOUT";
    }
};

//! Linear-Research 700 AC resistance bridge
class XLinearResearch700 : public XCharDeviceDriver<XTempControl> {
public:
    XLinearResearch700(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XLinearResearch700() {}

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
    virtual void onPowerMaxChanged(unsigned int loop, double v) {}
    virtual void onPowerMinChanged(unsigned int loop, double v) {}
    virtual void onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch);

    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc);
private:
    double parseResponseMessage();
};

//! Neocera LTC-21.
class XNeoceraLTC21 : public XCharDeviceDriver<XTempControl> {
public:
    XNeoceraLTC21(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XNeoceraLTC21() {}

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
    virtual void onPowerMaxChanged(unsigned int loop, double v);
    virtual void onPowerMinChanged(unsigned int loop, double v) {}
    virtual void onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch);

    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc);
private:
    //! set the system into the control mode.
    void control();
    //! leave the control mode.
    void monitor();
    //! set PID, manual power.
    void setHeater(unsigned int loop);
};

//! Base class for LakeShore 340/370
class XLakeShoreBridge : public XCharDeviceDriver<XTempControl> {
public:
    XLakeShoreBridge(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XLakeShoreBridge() = default;

protected:
    //! reads sensor value from the instrument
    virtual double getRaw(shared_ptr<XChannel> &channel) override;
    //! reads a value in Kelvin from the instrument
    virtual double getTemp(shared_ptr<XChannel> &channel) override;
    //! ex. "W", "dB", or so
    virtual const char *m_heaterPowerUnit(unsigned int loop) override {return "%";}
    virtual double currentIntervalSettingInSec(const Snapshot &shot, unsigned int lp) override {return 1000.0 / shot[ *interval(lp)];}
    virtual void onPChanged(unsigned int loop, double p) override;
    virtual void onIChanged(unsigned int loop, double i) override;
    virtual void onDChanged(unsigned int loop, double d) override;
    virtual void onManualPowerChanged(unsigned int loop, double pow) override;
    virtual void onPowerMinChanged(unsigned int loop, double v) override {}
    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc) override;
};

//! LakeShore 340
class XLakeShore340 : public XLakeShoreBridge {
public:
    XLakeShore340(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XLakeShore340() = default;
protected:
    //! obtains current heater power
    //! \sa m_heaterPowerUnit()
    virtual double getHeater(unsigned int loop) override;

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;

    virtual void onTargetTempChanged(unsigned int loop, double temp) override;
    virtual void onHeaterModeChanged(unsigned int loop, int mode) override;
    virtual void onPowerRangeChanged(unsigned int loop, int range) override;
    virtual void onPowerMaxChanged(unsigned int loop, double v) override;
    virtual void onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch) override;
private:
};
//! LakeShore 350
class XLakeShore350 : public XLakeShoreBridge {
public:
    XLakeShore350(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XLakeShore350() = default;
protected:
    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;

    //! obtains current heater power
    //! \sa m_heaterPowerUnit()
    virtual double getHeater(unsigned int loop) override;

    virtual void onTargetTempChanged(unsigned int loop, double temp) override;
    virtual void onHeaterModeChanged(unsigned int loop, int mode) override;
    virtual void onPowerRangeChanged(unsigned int loop, int range) override;
    virtual void onPowerMaxChanged(unsigned int loop, double v) override;
    virtual void onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch) override;
private:
};
//! LakeShore 370
class XLakeShore370 : public XLakeShoreBridge {
public:
    XLakeShore370(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XLakeShore370() = default;

protected:
    //! reads sensor value from the instrument
    virtual double getRaw(shared_ptr<XChannel> &channel) override;
    //! reads a value in Kelvin from the instrument
    virtual double getTemp(shared_ptr<XChannel> &channel) override;
    //! obtains current heater power
    //! \sa m_heaterPowerUnit()
    virtual double getHeater(unsigned int loop) override;
    //! ex. "W", "dB", or so
    virtual const char *m_heaterPowerUnit(unsigned int loop) override {return "%";}

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;

    virtual void onPChanged(unsigned int loop, double p) override;
    virtual void onIChanged(unsigned int loop, double i) override;
    virtual void onDChanged(unsigned int loop, double d) override;
    virtual void onTargetTempChanged(unsigned int loop, double temp) override;
    virtual void onManualPowerChanged(unsigned int loop, double pow) override;
    virtual void onHeaterModeChanged(unsigned int loop, int mode) override;
    virtual void onPowerRangeChanged(unsigned int loop, int range) override;
    virtual void onPowerMaxChanged(unsigned int loop, double v) override {}
    virtual void onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch) override;

    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc) override;
private:
};

//! Keithley Integra 2700 w/ 7700 switching module.
class XKE2700w7700 : public XCharDeviceDriver<XTempControl> {
public:
    XKE2700w7700(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XKE2700w7700() {}

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

    virtual void onPChanged(unsigned int loop, double p) {}
    virtual void onIChanged(unsigned int loop, double i) {}
    virtual void onDChanged(unsigned int loop, double d) {}
    virtual void onTargetTempChanged(unsigned int loop, double temp) {}
    virtual void onManualPowerChanged(unsigned int loop, double pow) {}
    virtual void onHeaterModeChanged(unsigned int loop, int mode) {}
    virtual void onPowerRangeChanged(unsigned int loop, int range) {}
    virtual void onPowerMaxChanged(unsigned int, double v) {}
    virtual void onPowerMinChanged(unsigned int, double v) {}
    virtual void onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch) {}

    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc) {}
private:
};
#endif
