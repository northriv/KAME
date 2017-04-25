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
    virtual ~CyFXUSBDevice() {close();}

    using List = std::vector<shared_ptr<CyFXUSBDevice>>;
    static List enumerateDevices();

    virtual void open();
    virtual void close();

    virtual int initialize() = 0;
    virtual void finalize() = 0;

    virtual void halt(const USBDevice &dev) = 0;
    virtual void run(const USBDevice &dev) = 0;
    XString virtual getString(const USBDevice &dev, int descid) = 0;
    virtual void download(const USBDevice &dev, uint8_t* image, int len) = 0;
    virtual int bulkWrite(int pipe, uint8_t *buf, int len) = 0;
    virtual int bulkRead(int pipe, const uint8_t* buf, int len) = 0;

    virtual unsigned int vendorID() = 0;
    virtual unsigned int productID() = 0;

    struct AsyncIO {
        AsyncIO(void *h);
        ~AsyncIO();
        void waitFor();
        void abort();
    private:
        class Transfer;
        scoped_ptr<Transfer> m_status;
    };
    virtual AsyncIO asyncBulkWrite(int pipe, uint8_t *buf, int len) = 0;
    virtual AsyncIO asyncBulkRead(int pipe, const uint8_t *buf, int len) = 0;

    XRecursiveMutex mutex;
    XString label;
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
    bool isLocked() const {return m_mutex->isLockedByCurrentThread();}

    virtual void send(const char *) throw (XCommError &) override {}
    virtual void receive() throw (XCommError &) override {}

    virtual bool isOpened() const override {return usb();}
protected:
    //\return true if device is supported by this interface.
    virtual bool examineDeviceBeforeFWLoad(const USBDevice &dev) = 0;
    //\return device string to be shown in the list box, if it is supported.
    virtual std::string examineDeviceAfterFWLoad(const USBDevice &dev) = 0;
    //\return Relative path to the GPIB wave file.
    virtual XStinrg gpifWave() = 0;
    //\return Relative path to the firmware file.
    virtual XStinrg firmware() = 0;

    const shared_ptr<USBDevice> &usb() const {return m_usbDevice;}
private:
    shared_ptr<USBDevice> m_usbDevice;
    static XMutex s_mutex;
    static USBDevice::List s_devices;
    static int s_refcnt;
    static void openAllEZUSBdevices();
    static void setWave(void *handle, const uint8_t *wave);
    static void closeAllEZUSBdevices();
};

