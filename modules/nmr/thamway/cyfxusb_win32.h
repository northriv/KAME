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
    XString wcstoxstr(const wchar_t *);
private:
    HANDLE handle;
    XString name;

    ASyncIO async_ioctl(uint64_t code, const void *in, ssize_t size_in, void *out = NULL, ssize_t size_out = 0);
    int ioctl(uint64_t code, const void *in, ssize_t size_in, void *out = NULL, ssize_t size_out = 0);
};

//! FX2(LP) devices under control of ezusb.sys.
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

//! FX3, FX2LP devices under control of CyUSB3.sys.
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

    XString friendlyName();
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
