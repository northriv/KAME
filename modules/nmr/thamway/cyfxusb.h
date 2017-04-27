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
#include "chardevicedriver.h"
#include "charinterface.h"
#include <vector>


struct CyFXUSBDevice {
    CyFXUSBDevice() = delete;
    virtual ~CyFXUSBDevice() = default;

    using List = std::vector<shared_ptr<CyFXUSBDevice>>;
    static List enumerateDevices();

    virtual void open() = 0;
    virtual void close() = 0;

    virtual int initialize() = 0;
    virtual void finalize() = 0;

    void halt();
    void run();
    XString getString(int descid);

    void downloadFX2(const uint8_t* image, int len);

    int bulkWrite(int pipe, const uint8_t *buf, int len);
    int bulkRead(int pipe, uint8_t* buf, int len);

    virtual int controlWrite(uint8_t request, uint16_t value,
                             uint16_t index, const uint8_t *buf, int len) = 0;
    virtual int controlRead(uint8_t request, uint16_t value,
                            uint16_t index, uint8_t *buf, int len) = 0;

    virtual unsigned int vendorID() = 0;
    virtual unsigned int productID() = 0;

    struct AsyncIO {
        class Transfer;
        AsyncIO(unique_ptr<Transfer>&& t);
        AsyncIO(const AsyncIO&) = delete;
        AsyncIO(AsyncIO &&) = default;
        void finalize(int64_t count_imm) {
            m_count_imm = count_imm;
        }
        bool hasFinished() const;
        int64_t waitFor(uint8_t *rdbuf = nullptr);
        Transfer *ptr() {return m_transfer.get();}
    private:
        unique_ptr<Transfer> m_transfer;
        int64_t m_count_imm = -1;
    };
    virtual AsyncIO asyncBulkWrite(int pipe, const uint8_t *buf, int len) = 0;
    virtual AsyncIO asyncBulkRead(int pipe, uint8_t *buf, int len) = 0;

    XRecursiveMutex mutex;
    XString label;

//    //AE18AA60-7F6A-11d4-97DD-00010229B959
//    static const std::initializer_list<uint32_t> CYPRESS_GUID = {0xae18aa60, 0x7f6a, 0x11d4, 0x97, 0xdd, 0x0, 0x1, 0x2, 0x29, 0xb9, 0x59};
};

//! interfaces Cypress FX2LP/FX3 devices
template <class USBDevice = CyFXUSBDevice>
class XCyFXUSBInterface : public XCustomCharInterface {
public:
    XCyFXUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
    virtual ~XCyFXUSBInterface();

    virtual void open() throw (XInterfaceError &);
    //! This can be called even if has already closed.
    virtual void close() throw (XInterfaceError &);

    void lock() {m_usbDevice->mutex->lock();} //!<overrides XInterface::lock().
    void unlock() {m_usbDevice->mutex->unlock();}
    bool isLocked() const {return m_usbDevice->isLockedByCurrentThread();}

    virtual void send(const char *) throw (XCommError &) override {}
    virtual void receive() throw (XCommError &) override {}

    virtual bool isOpened() const override {return usb();}
protected:
    //\return true if device is supported by this interface.
    virtual bool examineDeviceBeforeFWLoad(const USBDevice &dev) = 0;
    //\return device string to be shown in the list box, if it is supported.
    virtual std::string examineDeviceAfterFWLoad(const USBDevice &dev) = 0;
    //\return Relative path to the GPIB wave file.
    virtual XString gpifWave() = 0;
    //\return Relative path to the firmware file.
    virtual XString firmware() = 0;

    const shared_ptr<USBDevice> &usb() const {return m_usbDevice;}
private:
    shared_ptr<USBDevice> m_usbDevice;
    static XMutex s_mutex;
    static typename USBDevice::List s_devices;
    static int s_refcnt;
    static void openAllEZUSBdevices();
    static void setWave(void *handle, const uint8_t *wave);
    static void closeAllEZUSBdevices();
};

