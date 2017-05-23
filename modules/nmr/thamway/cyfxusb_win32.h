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

#define _WIN32_WINNT 0x0600 //vista or later is required for CancelIoEx
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
        AsyncIO(AsyncIO&&) noexcept = default;
        virtual ~AsyncIO();

        virtual bool hasFinished() const noexcept override;
        virtual int64_t waitFor() override;
        virtual bool abort() noexcept override;

        void setBufferOffset(uint8_t *ioctlbufrdpos, ssize_t prepad) {
            ssize_t offset = ioctlbufrdpos - &ioctlbuf[prepad];
            if(m_count_imm >= offset) m_count_imm -= offset;
            prepadding = prepad;
            ioctlbuf_rdpos = ioctlbufrdpos;
        }

        OVERLAPPED overlap = {}; //zero clear
        HANDLE handle = nullptr;
        std::vector<uint8_t> ioctlbuf; //buffer during the transfer.
        ssize_t prepadding = 0;
        uint8_t *ioctlbuf_rdpos = nullptr; //location of the incoming data of concern, part of \a ioctrlbuf.
        uint8_t *rdbuf = nullptr; //user buffer, already passed by read function.
    };

protected:
    unique_ptr<AsyncIO> asyncIOCtrl(uint64_t code, const void *in, ssize_t size_in, void *out = NULL, ssize_t size_out = 0);
    int64_t ioCtrl(uint64_t code, const void *in, ssize_t size_in, void *out = NULL, ssize_t size_out = 0);
    XString name;
private:
    HANDLE handle;
};

//! FX2(LP) devices under control of ezusb.sys.
struct CyFXEzUSBDevice : public CyFXWin32USBDevice {
    CyFXEzUSBDevice(HANDLE handle, const XString &n) : CyFXWin32USBDevice(handle, n)  {}

    XString virtual getString(int descid) override;

    virtual unique_ptr<CyFXUSBDevice::AsyncIO> asyncBulkWrite(uint8_t ep, const uint8_t *buf, int len, unsigned int timeout_ms = 0) override;
    virtual unique_ptr<CyFXUSBDevice::AsyncIO> asyncBulkRead(uint8_t ep, uint8_t *buf, int len, unsigned int timeout_ms = 0) override;

    virtual int controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                             uint16_t index, const uint8_t *buf, int len) override;
    virtual int controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                            uint16_t index, uint8_t *buf, int len) override;
private:
    struct VendorRequestCtrl {
        uint8_t bRequest, bReserve;
        uint16_t wValue, wIndex, wLength;
        uint8_t direction, bData;
    };
    struct StringDescCtrl {
        uint8_t Index, bReserve;
        uint16_t LanguageId;
    };
    struct BulkTransferCtrl {
        uint32_t pipeNum;
    };
};

//! FX3, FX2LP devices under control of CyUSB3.sys.
struct CyUSB3Device : public CyFXWin32USBDevice {
    CyUSB3Device(HANDLE handle, const XString &n) : CyFXWin32USBDevice(handle, n) {}

    XString virtual getString(int descid) override;

    virtual unique_ptr<CyFXUSBDevice::AsyncIO> asyncBulkWrite(uint8_t ep, const uint8_t *buf, int len, unsigned int timeout_ms = 0) override;
    virtual unique_ptr<CyFXUSBDevice::AsyncIO> asyncBulkRead(uint8_t ep, uint8_t *buf, int len, unsigned int timeout_ms = 0) override;

    virtual int controlWrite(CtrlReq request, CtrlReqType type, uint16_t value,
                             uint16_t index, const uint8_t *buf, int len) override;
    virtual int controlRead(CtrlReq request, CtrlReqType type, uint16_t value,
                            uint16_t index, uint8_t *buf, int len) override;

    XString friendlyName();
private:
    std::vector<uint8_t> setupSingleTransfer(uint8_t ep, CtrlReq request,
        CtrlReqType type, uint16_t value,
        uint16_t index, int len, uint32_t timeout_ms = 0);
    enum {SIZEOF_SINGLE_TRANSFER = 38, PAD_BEFORE = 2};
};

#endif // CYFXUSB_WIN32_H
