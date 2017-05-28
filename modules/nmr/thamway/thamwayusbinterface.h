/***************************************************************************
        Copyright (C) 2002-2017 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/

#ifndef THAMWAYUSBINTERFACE_H
#define THAMWAYUSBINTERFACE_H

#include "chardevicedriver.h"
#include "charinterface.h"
#include "cyfxusb.h"
#include <vector>
#include "softtrigger.h"

struct ThamwayCyFX2USBDevice : public CyFXUSBDevice {};

//! Interfaces Thamway's PROT/AD/Pulser based on FX2LP device.
class XThamwayFX2USBInterface : public XCyFXUSBInterface<ThamwayCyFX2USBDevice> {
public:
    XThamwayFX2USBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver, uint8_t addr_offset, const char* id);
    virtual ~XThamwayFX2USBInterface();

    virtual void open() throw (XInterfaceError &) override;
    //! This can be called even if has already closed.
    virtual void close() throw (XInterfaceError &) override;

    struct ScopedBulkWriter{
        ScopedBulkWriter() = default;
        ScopedBulkWriter(const shared_ptr<XThamwayFX2USBInterface> intf) : m_intf(intf) {
            if(m_intf) m_intf->deferWritings();
        }
        ~ScopedBulkWriter() {if(m_intf) m_intf->resetBulkWrite();}
        ScopedBulkWriter(const ScopedBulkWriter&) = delete;
        ScopedBulkWriter(ScopedBulkWriter&&) = default;
        void flush() {m_intf->bulkWriteStored();}
    private:
        shared_ptr<XThamwayFX2USBInterface> m_intf;
    };
    void resetBulkWrite() noexcept;

    void writeToRegister8(unsigned int addr, uint8_t data);
    void writeToRegister16(unsigned int addr, uint16_t data);

    void burstRead(unsigned int addr, uint8_t *buf, unsigned int cnt);
    uint8_t singleRead(unsigned int addr);
    uint16_t readRegister8(unsigned int addr) {return singleRead(addr);}
    uint16_t readRegister16(unsigned int addr);

    XString getIDN(int maxlen = 255) {return getIDN(usb(), maxlen); }

    virtual void send(const char *str) override;
    virtual void receive() override;
protected:
    virtual DEVICE_STATUS examineDeviceBeforeFWLoad(const shared_ptr<CyFXUSBDevice> &dev) override;
    virtual std::string examineDeviceAfterFWLoad(const shared_ptr<CyFXUSBDevice> &dev) override;
    virtual XString gpifWave(const shared_ptr<CyFXUSBDevice> &dev) override;
    virtual XString firmware(const shared_ptr<CyFXUSBDevice> &dev) override;
    virtual void setWave(const shared_ptr<CyFXUSBDevice> &dev, const uint8_t *wave) override;
    void deferWritings();
    void bulkWriteStored();
private:
    XString getIDN(const shared_ptr<CyFXUSBDevice> &dev, int maxlen = 255) {
        XString str = getIDN(dev, maxlen, m_addrOffset);
        if(str.empty() || (str.find(m_idString,0) != 0) || m_idString.empty())
             return {};
        return str;
    }
    static XString getIDN(const shared_ptr<CyFXUSBDevice> &dev, int maxlen, int offsetaddr);
    static void setLED(const shared_ptr<CyFXUSBDevice> &dev, uint8_t data);
    static uint8_t readDIPSW(const shared_ptr<CyFXUSBDevice> &dev);
    static uint8_t singleRead(const shared_ptr<CyFXUSBDevice> &dev, unsigned int addr, unsigned int addroffset);

    uint8_t m_addrOffset;
    XString m_idString;
    bool m_bBurstWrite;
    std::vector<uint8_t> m_buffer; //writing buffer for a burst write.
};

struct ThamwayCyFX3USBDevice : public CyFXUSBDevice {};

//! Interfaces Thamway's PROT data acquision device based on FX3 device.
class XThamwayFX3USBInterface : public XCyFXUSBInterface<ThamwayCyFX3USBDevice> {
public:
    XThamwayFX3USBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
    virtual ~XThamwayFX3USBInterface();

    virtual void open() throw (XInterfaceError &) override;
    //! This can be called even if has already closed.
    virtual void close() throw (XInterfaceError &) override;

    virtual void send(const char *str) override;
    virtual void receive() override;

    unique_ptr<CyFXUSBDevice::AsyncIO> asyncReceive(char *buf, ssize_t size);

    static SoftwareTriggerManager &softwareTriggerManager() {return s_softwareTriggerManager;}
protected:
    virtual DEVICE_STATUS examineDeviceBeforeFWLoad(const shared_ptr<CyFXUSBDevice> &dev) override;
    virtual std::string examineDeviceAfterFWLoad(const shared_ptr<CyFXUSBDevice> &dev) override;
    virtual XString gpifWave(const shared_ptr<CyFXUSBDevice> &dev) override {return {};}
    virtual XString firmware(const shared_ptr<CyFXUSBDevice> &dev) override {return {};}
    virtual void setWave(const shared_ptr<CyFXUSBDevice> &dev, const uint8_t *wave) override {}
private:
    static SoftwareTriggerManager s_softwareTriggerManager;
};

#endif // THAMWAYUSBINTERFACE_H
