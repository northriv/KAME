/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef OCEANOPTICSUSB_H
#define OCEANOPTICSUSB_H

#include "cyfxusb.h"
#include <vector>

struct OceanOpticsUSBDevice : public CyFXUSBDevice {};

//! interfaces OceanOptics/SeaBreeze spectrometers
class XOceanOpticsUSBInterface : public XCyFXUSBInterface<OceanOpticsUSBDevice> {
    using USBDevice = OceanOpticsUSBDevice;
public:
    XOceanOpticsUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
        : XCyFXUSBInterface<OceanOpticsUSBDevice>(name, runtime, driver) {
        initialize();
    }

    virtual ~XOceanOpticsUSBInterface() {
        finalize();
    }

    virtual void send(const char *str) override {}
    virtual void receive() override {}

    void initDevice();
    void setIntegrationTime(unsigned int us);
    void enableStrobe(bool);
    void setupStrobeCond(double singlestrobe_to_high_sec, double singlestrobe_to_low_sec);
    enum class TrigMode {NORMAL=0,SOFTWARE=1,EXT_HARDWARE=2, EXT_SYNC=3, EXT_HARDWARE_EDGE=4};
    void setupTrigCond(TrigMode mode, double delay_sec);
    void setAnalogOutput(double);

    int readSpectrum(std::vector<uint8_t> &buf, uint16_t pixels, bool usb_highspeed);

    std::vector<uint8_t> readInstrumStatus();
    struct InstrumConfig {
        std::string serialNo, wavelenCalib[4], strayLightConst, nonlinCorr[8], nlpoly, opticalBenchConfig, spectrometerConfig;
    };
    InstrumConfig readConfigurations();


    enum class Register {
        MasterClockCounterDivisor = 0x00, FPGAFirmwareVersion = 0x04,
        ContinuousStrobeTimerIntervalDivisor = 0x08, ContinuousStrobeBaseClock = 0x0c,
        IntegrationPeriodBaseClock = 0x10, BaseClock = 0x14, IntegrationClockTimeDivisor = 0x18,
        HardwareTriggerDelay = 0x28, TriggerMode = 0x2c,
        SingleStrobeHighClockTransition = 0x38, SingleStrobeLowClockTransition = 0x3c,
        LampEnable = 0x40, GPIOMuxRegister = 0x48, GPIOOutputEnable = 0x50, GPIODataRegister = 0x54,
        EnableExternalMasterClock = 0x5c,
    };

    void writeRegInfo(Register reg, uint16_t word);
    uint16_t readRegInfo(Register reg);
protected:
    virtual DEVICE_STATUS examineDeviceBeforeFWLoad(const shared_ptr<CyFXUSBDevice> &dev) override;
    virtual std::string examineDeviceAfterFWLoad(const shared_ptr<CyFXUSBDevice> &dev) override;
    virtual XString gpifWave(const shared_ptr<CyFXUSBDevice> &dev) override {return {};}
    virtual XString firmware(const shared_ptr<CyFXUSBDevice> &dev) override {return {};}
    virtual void setWave(const shared_ptr<CyFXUSBDevice> &dev, const uint8_t *wave) override {}
private:
    uint8_t m_ep_in_others = 1, m_ep_in_config = 1, m_ep_in_spec = 2, m_ep_in_spec_first1Kpixels = 6, m_ep_cmd = 1;
    enum class CMD {
        INIT=0x01, SET_INTEGRATION_TIME=0x02, SET_STROBE_ENABLE_STAT=0x03, SET_SHUTDOWN_MODE=0x04,QUERY_INFO=0x05,
        WRITE_INFO=0x06,WRITE_SERIALNO=0x07,GET_SERIALNO=0x08,REQUEST_SPECTRA=0x09,
        SET_TRIG_MODE=0x0a, QUERY_NPLUGINS=0x0b, QUERY_PLUGIN_IDS=0x0c, DETECT_PLUGINGS=0x0d,
        LED_STATUS=0x12,
        LS450_READ_TEMPERTURE = 0x20, LS450_SET_LED_MODE = 0x21, LS450_QUERY_CALIB_CONST = 0x23, LS450_SEND_CALIB_CONST = 0x24,
        LS450_SET_ANALOG_OUT = 0x25, LS450_LOAD_ALL_CALIB_VALUES = 0x26, LS450_WRITE_ALL_CALIB_COEFFS = 0x27,
        GENERAL_I2C_READ = 0x60, GENERAL_I2C_WRITE = 0x61, GENERAL_SPI_IO = 0x62,
        PSOC_READ=0x68, PSOC_WRITE=0x69,
        WRITE_REG=0x6a, READ_REG=0x6b, READ_PCB_TEMP=0x6c, READ_IRRAD_CALIB=0x6d, WRITE_IRRAD_CALIB=0x6e,
        QUERY_OP_INFO=0xfe};
    constexpr static unsigned int CMD_READ_SIZE = 18;
    unsigned int m_bytesInSpec = 4097 * 2;
};

#endif // OCEANOPTICSUSB_H
