/***************************************************************************
        Copyright (C) 2002-2025 Kentaro Kitagawa
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
    virtual const char *m_heaterPowerUnit(unsigned int) {return "%";}

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

    virtual void onSetupChannelChanged(const shared_ptr<XChannel> &) {} //for updating UIs.
    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc);
    virtual void onChannelEnableChanged(const shared_ptr<XChannel> &, bool) {}
    virtual void onScanDwellSecChanged(const shared_ptr<XChannel> &, double) {}
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

    virtual void onSetupChannelChanged(const shared_ptr<XChannel> &ch) {
        trans( *ch->excitation()).clear();
        trans( *ch->excitation()).add(
            {"0", "3uV", "10uV", "30uV", "100uV", "300uV", "1mV", "3mV"});
    }
    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc);
    virtual void onChannelEnableChanged(const shared_ptr<XChannel> &, bool) {}
    virtual void onScanDwellSecChanged(const shared_ptr<XChannel> &, double) {}
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
    virtual void onPowerMinChanged(unsigned int, double) {}
    virtual void onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch);

    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc);
    virtual void onChannelEnableChanged(const shared_ptr<XChannel> &, bool) {}
    virtual void onScanDwellSecChanged(const shared_ptr<XChannel> &, double) {}
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

    virtual void onSetupChannelChanged(const shared_ptr<XChannel> &ch) {
        trans( *ch->excitation()).clear();
        trans( *ch->excitation()).add({"CI", "10MV", "3MV", "1MV"});
    }
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

    virtual void onSetupChannelChanged(const shared_ptr<XChannel> &ch) {
        trans( *ch->excitation()).clear();
        trans( *ch->excitation()).add({"10UV", "30UV", "100UV", "333UV", "1.0MV", "3.3MV"});
    }
    virtual void onPowerMaxChanged(unsigned int, double v) {}
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
    virtual void onPowerMaxChanged(unsigned int, double) {}
    virtual void onPowerMinChanged(unsigned int, double) {}
    virtual void onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch);

    virtual void onSetupChannelChanged(const shared_ptr<XChannel> &ch) {
        trans( *ch->excitation()).clear();
        trans( *ch->excitation()).add(
            {"20uV", "60uV", "200uV", "600uV", "2mV", "6mV", "20mV"});
    }
    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc);
    virtual void onChannelEnableChanged(const shared_ptr<XChannel> &, bool) {}
    virtual void onScanDwellSecChanged(const shared_ptr<XChannel> &, double) {}
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
    virtual void onPowerMinChanged(unsigned int, double) {}
    virtual void onCurrentChannelChanged(unsigned int loop, const shared_ptr<XChannel> &ch);

    virtual void onSetupChannelChanged(const shared_ptr<XChannel> &ch) {
        trans( *ch->excitation()).clear();
        trans( *ch->excitation()).add(
        {});//"1mV", "320uV", "100uV", "32uV", "10uV"
    }
    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc);
    virtual void onChannelEnableChanged(const shared_ptr<XChannel> &, bool) {}
    virtual void onScanDwellSecChanged(const shared_ptr<XChannel> &, double) {}
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
    virtual const char *m_heaterPowerUnit(unsigned int) override {return "%";}
    virtual double currentIntervalSettingInSec(const Snapshot &shot, unsigned int lp) override {return 1000.0 / shot[ *interval(lp)];}
    virtual void onPChanged(unsigned int loop, double p) override;
    virtual void onIChanged(unsigned int loop, double i) override;
    virtual void onDChanged(unsigned int loop, double d) override;
    virtual void onManualPowerChanged(unsigned int loop, double pow) override;
    virtual void onPowerMinChanged(unsigned int , double ) override {}
};


//! LakeShore 218
class XLakeShore218 : public XLakeShoreBridge {
public:
    XLakeShore218(const char *name, bool runtime,
                  Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XLakeShore218() = default;
protected:
    //! obtains current heater power
    //! \sa m_heaterPowerUnit()
    virtual double getHeater(unsigned int loop) override {}

    virtual void onTargetTempChanged(unsigned int, double) override {}
    virtual void onHeaterModeChanged(unsigned int, int) override {}
    virtual void onPowerRangeChanged(unsigned int, int) override {}
    virtual void onPowerMaxChanged(unsigned int, double) override {}
    virtual void onCurrentChannelChanged(unsigned int, const shared_ptr<XChannel> &) override {}
    virtual void onSetupChannelChanged(const shared_ptr<XChannel> &) override {}
    virtual void onExcitationChanged(const shared_ptr<XChannel> &, int) override {}
    virtual void onChannelEnableChanged(const shared_ptr<XChannel> &, bool) override {}
    virtual void onScanDwellSecChanged(const shared_ptr<XChannel> &, double) override {}
private:
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
    virtual void onSetupChannelChanged(const shared_ptr<XChannel> &) override {}
    virtual void onExcitationChanged(const shared_ptr<XChannel> &, int) override {}
    virtual void onChannelEnableChanged(const shared_ptr<XChannel> &, bool) override {}
    virtual void onScanDwellSecChanged(const shared_ptr<XChannel> &, double) override {}
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
    virtual void onSetupChannelChanged(const shared_ptr<XChannel> &ch) override;
    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc) override;
    virtual void onChannelEnableChanged(const shared_ptr<XChannel> &, bool) override {}
    virtual void onScanDwellSecChanged(const shared_ptr<XChannel> &, double) override {}
private:
};
//! Base class for LakeShore 370
class XLakeShore370 : public XLakeShoreBridge {
public:
    XLakeShore370(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas, bool create_8ch = true);
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
    virtual void onSetupChannelChanged(const shared_ptr<XChannel> &ch) override;
    virtual void onExcitationChanged(const shared_ptr<XChannel> &ch, int exc) override;
    virtual void onChannelEnableChanged(const shared_ptr<XChannel> &ch, bool enable) override;
    virtual void onScanDwellSecChanged(const shared_ptr<XChannel> &ch, double sec) override;

    virtual bool is372() const {return m_is372;}
private:
    bool m_is372 = false;
};

class XLakeShore370_1CH : public XLakeShore370 {
public:
    XLakeShore370_1CH(const char *name, bool runtime,
                                 Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
        XLakeShore370(name, runtime, ref(tr_meas), meas, false) {
        createChannels(ref(tr_meas), meas, true,
                       {"1"},
                       {"Loop"},
                       false, false);
    }
};
using XLakeShore370_8CH = XLakeShore370;
class XLakeShore370_16CH : public XLakeShore370 {
public:
    XLakeShore370_16CH(const char *name, bool runtime,
                                         Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
        XLakeShore370(name, runtime, ref(tr_meas), meas, false) {
        createChannels(ref(tr_meas), meas, true,
                       {"1", "2", "3", "4", "5", "6", "7", "8",
                        "9", "10", "11", "12", "13", "14", "15", "16"},
                       {"Loop"},
                       true, true); //scanner is used.
    }
};
#endif
