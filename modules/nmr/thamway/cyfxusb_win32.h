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
#ifndef CYFXUSB_WIN32_H
#define CYFXUSB_WIN32_H

#include "cyfxusb.h"

struct CyFXWin32USBDevice : public CyFXUSBDvice {
    CyFXWin32USBDevice(HANDLE handle, const XString &n);
    virtual ~CyFXWin32USBDevice()  {close();}

    virtual void open() final;
    virtual void close() final;

    virtual int bulkWrite(int ep, const uint8_t *buf, int len) final;
    virtual int bulkRead(int ep, uint8_t* buf, int len) final;
protected:
    struct DeviceDescriptor {
        uint8_t bLength;
        uint8_t bDescriptorType;
        uint16_t bcdUSB;
        uint8_t bDeviceClass;
        uint8_t bDeviceSubClass;
        uint8_t bDeviceProtocol;
        uint8_t bMaxPacketSize0;
        uint16_t idVendor;
        uint16_t idProduct;
        uint16_t bcdDevice;
        uint8_t iManufacturer;
        uint8_t iProduct;
        uint8_t iSerialNumber;
        uint8_t bNumConfigurations;
    }
private:
    HANDLE handle;
    XString name;

    ASyncIO async_ioctl(uint64_t code, const void *in, ssize_t size_in, void *out = NULL, ssize_t size_out = 0);
    int ioctl(uint64_t code, const void *in, ssize_t size_in, void *out = NULL, ssize_t size_out = 0);

    //AE18AA60-7F6A-11d4-97DD-00010229B959
    constexpr tGUID GUID = {0xae18aa60, 0x7f6a, 0x11d4, 0x97, 0xdd, 0x0, 0x1, 0x2, 0x29, 0xb9, 0x59};
};

struct CyFXEzUSBDevice : public CyFXWin32USBDevice {
    CyFXEzUSBDevice(HANDLE handle, const XString &n) : CyFXWin32USBDevice(handle, n)  {}
    virtual ~CyFXEzUSBDevice();

    virtual void finalize() final {}

    XString virtual getString(int descid) final;

    virtual AsyncIO asyncBulkWrite(int ep, const uint8_t *buf, int len) final;
    virtual AsyncIO asyncBulkRead(int ep, uint8_t *buf, int len) final;

    virtual int controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                             uint16_t index, const uint8_t *buf, int len) final;
    virtual int controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                            uint16_t index, uint8_t *buf, int len) final;
private:
    struct VendorRequestCtrl {
        uint8_t bRequest;
        uint16_t wValue, wIndex, wLength;
        uint8_t direction;
    };
    struct StringDescCtrl {
        uint8_t Index;
        uint16_t LanguageId;
    };
    struct BulkTransferCtrl {
        uint32_t pipeNum;
    };
};

struct CyUSB3Device : public CyFXWin32USBDevice {
    CyUSBDevice(HANDLE handle, const XString &n) : CyFXWin32USBDevice(handle, n) {}

    virtual void finalize() final;

    XString virtual getString(int descid) final;

    virtual AsyncIO asyncBulkWrite(int ep, const uint8_t *buf, int len) final;
    virtual AsyncIO asyncBulkRead(int ep, uint8_t *buf, int len) final;

    virtual int controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                             uint16_t index, const uint8_t *buf, int len) final;
    virtual int controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                            uint16_t index, uint8_t *buf, int len) final;

private:
    struct SingleTransfer {
        uint8_t bmRequest;
        uint8_t bRequest;
        uint16_t wValue, wIndex, wLength;
        uint32_t timeOut;
        uint8_t bReserved2, bEndpointAddress, bNtStatus;
        uint32_t usbdStatus, isoPacketOffset, isoPacketLength;
        uint32_t bufferOffset, bufferLength;
    };
};

#endif // CYFXUSB_WIN32_H
