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

#define _WIN32_WINNT 0x0600 //vista or later
#define NOMINMAX
#include <windows.h>

struct CyFXWin32USBDevice : public CyFXUSBDevice {
    CyFXWin32USBDevice(HANDLE handle, const XString &n);
    virtual ~CyFXWin32USBDevice()  {close();}

    virtual void open() override;
    virtual void close() override;

    //! retrieves vendor/product IDs.
    void setIDs();

    struct AsyncIO : public CyFXUSBDevice::AsyncIO {
        AsyncIO() = default;
        virtual ~AsyncIO() = default;

        virtual bool hasFinished() const override;
        virtual int64_t waitFor() override;
        virtual bool abort() override;

        OVERLAPPED overlap = {}; //zero clear
        HANDLE handle;
        std::vector<uint8_t> ioctlbuf; //buffer during the transfer.
        uint8_t *ioctlbuf_rdpos = nullptr; //location of the incoming data of concern, part of \a ioctrlbuf.
        uint8_t *rdbuf = nullptr; //user buffer, already passed by read function.
    };

protected:
    unique_ptr<AsyncIO> asyncIOCtrl(uint64_t code, const void *in, ssize_t size_in, void *out = NULL, ssize_t size_out = 0);
    int64_t ioCtrl(uint64_t code, const void *in, ssize_t size_in, void *out = NULL, ssize_t size_out = 0);
private:
    HANDLE handle;
    XString name;
};

//! FX2(LP) devices under control of ezusb.sys.
struct CyFXEzUSBDevice : public CyFXWin32USBDevice {
    CyFXEzUSBDevice(HANDLE handle, const XString &n) : CyFXWin32USBDevice(handle, n)  {}

    virtual void finalize() final {}

    XString virtual getString(int descid) override;

    virtual unique_ptr<CyFXUSBDevice::AsyncIO> asyncBulkWrite(uint8_t ep, const uint8_t *buf, int len) override;
    virtual unique_ptr<CyFXUSBDevice::AsyncIO> asyncBulkRead(uint8_t ep, uint8_t *buf, int len) override;

    virtual int controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                             uint16_t index, const uint8_t *buf, int len) override;
    virtual int controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                            uint16_t index, uint8_t *buf, int len) override;
private:
    struct VendorRequestCtrl {
//        uint8_t bRequest;
//        uint16_t wValue, wIndex, wLength;
//        uint8_t direction;

            BYTE    bRequest;
            WORD    wValue;
            WORD    wIndex;
            WORD    wLength;
            BYTE    direction;
            BYTE    bData;
    };
    struct StringDescCtrl {
//        uint8_t Index;
//        uint16_t LanguageId;
        UCHAR    Index;
        USHORT   LanguageId;
    };
    struct BulkTransferCtrl {
        ULONG pipeNum;
//        uint32_t pipeNum;
    };
};

//! FX3, FX2LP devices under control of CyUSB3.sys.
struct CyUSB3Device : public CyFXWin32USBDevice {
    CyUSB3Device(HANDLE handle, const XString &n) : CyFXWin32USBDevice(handle, n) {}

    virtual void finalize() override;

    XString virtual getString(int descid) override;

    virtual unique_ptr<CyFXUSBDevice::AsyncIO> asyncBulkWrite(uint8_t ep, const uint8_t *buf, int len) override;
    virtual unique_ptr<CyFXUSBDevice::AsyncIO> asyncBulkRead(uint8_t ep, uint8_t *buf, int len) override;

    virtual int controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                             uint16_t index, const uint8_t *buf, int len) override;
    virtual int controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                            uint16_t index, uint8_t *buf, int len) override;

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
