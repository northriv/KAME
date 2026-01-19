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
#ifndef CYFXUSB_H
#define CYFXUSB_H

#include "chardevicedriver.h"
#include "charinterface.h"
#include <vector>


struct CyFXUSBDevice {
    CyFXUSBDevice(const CyFXUSBDevice&) = delete;
    virtual ~CyFXUSBDevice() = default;

    bool operator==(const CyFXUSBDevice &d) const {
        return (productID() == d.productID()) && (vendorID() == d.vendorID()) && (serialNo() == d.serialNo());}

    using List = std::vector<shared_ptr<CyFXUSBDevice>>;
    //! \return a list of connected USB devices, perhaps including non-cypress devices.
    static List enumerateDevices();

    virtual void open() = 0;
    virtual void close() = 0;

    void openForSharing() {
        //allows to be shared.
        m_refcnt++;
        open();
    }
    void unref() {
        if(--m_refcnt)
            close();
    }
    void halt();
    void run();
    virtual XString getString(int descid) = 0;

    void downloadFX2(const uint8_t* image, int len);

    //! \arg buf be sure that the user buffer stays alive during an asynchronous IO.
    int64_t bulkWrite(uint8_t ep, const uint8_t *buf, int len);
    //! \arg ep 0x80 will be or-operated
    //! \arg buf be sure that the user buffer stays alive during an asynchronous IO.
    int64_t bulkRead(uint8_t ep, uint8_t* buf, int len);

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
    unsigned int serialNo() const {return m_serialNo;}

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
    CyFXUSBDevice() = default;
    uint16_t m_vendorID, m_productID, m_serialNo;
    int m_refcnt = 0;
};

//! interfaces Cypress FX2LP/FX3 devices
template <class USBDevice = CyFXUSBDevice>
class XCyFXUSBInterface : public XCustomCharInterface {
public:
    XCyFXUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
    virtual ~XCyFXUSBInterface();

    virtual void open() override;
    //! This can be called even if has already closed.
    virtual void close() override;

    //! must be called during the constructor of the inherited class.
    //! If instatiation == false, setups newly discovered devices only.
    void initialize(bool instatiation = true);
    void finalize();

    virtual void lock() override { m_usbDevice->mutex.lock();} //!<overrides XInterface::lock().
    virtual void unlock() override { m_usbDevice->mutex.unlock();}
    virtual bool isLocked() const override {return m_usbDevice->mutex.isLockedByCurrentThread();}

    virtual void send(const char *) override {}
    virtual void receive() override {}

    virtual bool isOpened() const override {return !!usb();}
protected:
    enum class DEVICE_STATUS {FW_NOT_LOADED, READY, UNSUPPORTED};
    virtual DEVICE_STATUS examineDeviceBeforeFWLoad(const shared_ptr<CyFXUSBDevice> &dev) = 0;
    //! \return device string to be shown in the list box, if it is supported.
    virtual std::string examineDeviceAfterFWLoad(const shared_ptr<CyFXUSBDevice> &dev) = 0;
    //! \return Relative path to the GPIB wave file.
    virtual XString gpifWave(const shared_ptr<CyFXUSBDevice> &dev) = 0;
    //! \return Relative path to the firmware file.
    virtual XString firmware(const shared_ptr<CyFXUSBDevice> &dev) = 0;
    virtual void setWave(const shared_ptr<CyFXUSBDevice> &dev, const uint8_t *wave) = 0;

    const shared_ptr<CyFXUSBDevice> &usb() const {return m_usbDevice;}
private:
    typename USBDevice::List pickupNewDev(const typename USBDevice::List &enumerated);

    shared_ptr<CyFXUSBDevice> m_usbDevice;
    static XMutex s_mutex;
    static typename USBDevice::List s_devices;
    static int s_refcnt;
    unique_ptr<XThread> m_threadInit;
    std::map<XString, shared_ptr<CyFXUSBDevice>> m_candidates;
    void openAllEZUSBdevices();
    void closeAllEZUSBdevices();

    void onItemRefreshRequested(const Snapshot &shot, XItemNodeBase *node) {
        initialize(false);
    }
    shared_ptr<Listener> m_lsnOnItemRefreshRequested;
};

#endif

