/***************************************************************************
        Copyright (C) 2002-2022 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

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

#include "chardevicedriver.h"
#include "charinterface.h"
#include <vector>

struct OceanOpticsUSBDevice {
    OceanOpticsUSBDevice(const OceanOpticsUSBDevice&) = default;
    virtual ~OceanOpticsUSBDevice() = default;

    using List = std::vector<shared_ptr<OceanOpticsUSBDevice>>;
    //! \return a list of connected OceanOptics USB devices.
    static List enumerateDevices();

    //! \arg buf be sure that the user buffer stays alive during an asynchronous IO.
    int64_t bulkWrite(uint8_t ep, const uint8_t *buf, int len);
    //! \arg ep 0x80 will be or-operated
    //! \arg buf be sure that the user buffer stays alive during an asynchronous IO.
    int64_t bulkRead(uint8_t ep, uint8_t* buf, int len);

    virtual void open() = 0;
    virtual void close() = 0;
    virtual XString getString(int descid) = 0;

    enum class CtrlReq : uint8_t  {
        GET_STATUS = 0x00, CLEAR_FEATURE = 0x01, SET_FEATURE = 0x03, SET_ADDRESS = 0x05,
        GET_DESCRIPTOR = 0x06, SET_DESCRIPTOR = 0x07, GET_CONFIGURATION = 0x08, SET_CONFIGURATION = 0x09,
        GET_INTERFACE = 0x0A, SET_INTERFACE = 0x0B, SYNCH_FRAME = 0x0C, SET_SEL = 0x30,
        USB_SET_ISOCH_DELAY = 0x31
    };
    enum class CtrlReqType : uint8_t {
        STANDARD = (0x00 << 5),
        CLASS = (0x01 << 5),
        VENDOR = (0x02 << 5),
        RESERVED = (0x03 << 5),
        USB_RECIPIENT_DEVICE = 0x00,
        USB_RECIPIENT_INTERFACE = 0x01,
        USB_RECIPIENT_ENDPOINT = 0x02,
        USB_RECIPIENT_OTHER = 0x03 };
    virtual int controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                             uint16_t index, const uint8_t *buf, int len) = 0;
    virtual int controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                            uint16_t index, uint8_t *buf, int len) = 0;

    unsigned int vendorID() const {return m_vendorID;}
    unsigned int productID() const {return m_productID;}

    class AsyncIO {
    public:
        AsyncIO() = default;
        AsyncIO(const AsyncIO&) = delete;
        AsyncIO(AsyncIO&&) noexcept = default;
        virtual ~AsyncIO() = default;
        void finalize(int64_t count_imm) {
            m_count_imm = count_imm;
        }
        virtual bool hasFinished() const noexcept {return false;} //gcc doesn't accept pure virtual.
        virtual int64_t waitFor() {return 0;} //gcc doesn't accept pure virtual.
        //! \return true if a cancelation is successfully requested.
        virtual bool abort() noexcept {return false;} //gcc doesn't accept pure virtual.

        static XThreadLocal<std::vector<uint8_t>> stl_bufferGarbage;
    protected:
        int64_t m_count_imm = -1; //byte count of received user data, not incl. header.
    };
    virtual unique_ptr<AsyncIO> asyncBulkWrite(uint8_t ep, const uint8_t *buf, int len, unsigned int timeout_ms = 0) = 0;
    virtual unique_ptr<AsyncIO> asyncBulkRead(uint8_t ep, uint8_t *buf, int len, unsigned int timeout_ms = 0) = 0;

    XRecursiveMutex mutex;

    enum {USB_DEVICE_DESCRIPTOR_TYPE = 1, USB_CONFIGURATION_DESCRIPTOR_TYPE = 2,
        USB_STRING_DESCRIPTOR_TYPE = 3, USB_INTERFACE_DESCRIPTOR_TYPE = 4,
        USB_ENDPOINT_DESCRIPTOR_TYPE = 5};
protected:
    OceanOpticsUSBDevice() = default;
    uint16_t m_vendorID, m_productID;
};

//! interfaces OceanOptics/SeaBreeze spectrometers
class XOceanOpticsUSBInterface : public XCustomCharInterface {
    using USBDevice = OceanOpticsUSBDevice;
public:
    XOceanOpticsUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
    virtual ~XOceanOpticsUSBInterface();

    virtual void open() override;
    //! This can be called even if has already closed.
    virtual void close() override;

    void lock() {m_usbDevice->mutex.lock();} //!<overrides XInterface::lock().
    void unlock() {m_usbDevice->mutex.unlock();}
    bool isLocked() const {return m_usbDevice->mutex.isLockedByCurrentThread();}

    virtual void send(const char *str) override {}
    virtual void receive() override {}

    void initDevice();
    void setIntegrationTime(unsigned int us);

    unique_ptr<typename USBDevice::AsyncIO> asyncReceive(char *buf, ssize_t size);

    virtual bool isOpened() const override {return !!usb();}

    int readSpectrum(std::vector<uint8_t> &buf, uint16_t pixels, bool usb_highspeed);

    std::vector<uint8_t> readInstrumStatus();
    struct InstrumConfig {
        std::string serialNo, wavelenCalib[4], strayLightConst, nonlinCorr[8], nlpoly, opticalBenchConfig, spectrometerConfig;
    };
    InstrumConfig readConfigurations();
protected:

    const shared_ptr<USBDevice> &usb() const {return m_usbDevice;}
private:
    shared_ptr<OceanOpticsUSBDevice> m_usbDevice;
    static XMutex s_mutex;
    static typename USBDevice::List s_devices;
    static int s_refcnt;
    unique_ptr<XThread> m_threadInit;
    std::map<XString, shared_ptr<USBDevice>> m_candidates;
    void openAllUSBdevices();
    void closeAllUSBdevices();

    uint8_t m_ep_in_others = 1, m_ep_in_spec = 2, m_ep_in_spec_first1Kpixels = 6, m_ep_cmd = 1;
    enum class CMD {
        INIT=0x01, SET_INTEGRATION_TIME=0x02, SET_STROBE_ENABLE_STAT=0x03, SET_SHUTDOWN_MODE=0x04,QUERY_INFO=0x05,
        WRITE_INFO=0x06,WRITE_SERIALNO=0x07,GET_SERIALNO=0x08,REQUEST_SPECTRA=0x09,
        SET_TRIG_MODE=0x0a, QUERY_NPLUGINS=0x0b, QUERY_PLUGIN_IDS=0x0c, DETECT_PLUGINGS=0x0d,
        LED_STATUS=0x12, GENERAL_I2C_READ = 0x60, GENERAL_I2C_WRITE = 0x61, GENERAL_SPI_IO = 0x62,
        PSOC_READ=0x68, PSOC_WRITE=0x69,
        WRITE_REG=0x6a, READ_REG=0x6b, READ_PCB_TEMP=0x6c, READ_IRRAD_CALIB=0x6d, WRITE_IRRAD_CALIB=0x6e,
        QUERY_OP_INFO=0xfe};
    enum class TRIG_MODE {NORMAL=0,SOFTWARE=1,EXT_HARDWARE=2, EXT_SYNC=3, EXT_HARDWARE_EDGE=4};
    constexpr static unsigned int CMD_READ_SIZE = 18;
    unsigned int m_bytesInSpec = 4097 * 2;
};

#endif // OCEANOPTICSUSB_H
