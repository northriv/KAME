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

struct CyFXEasyUSBDevice {
    CyFXEasyUSBDevice(HANDLE handle, const XString &n) : handle(h), name(n) {}
    CyFXEasyUSBDevice() : handle(nullptr), device(nullptr) {}
    virtual ~CyFXEasyUSBDevice();

    virtual int initialize() final;
    virtual void finalize() final;

    virtual void halt(const USBDevice &dev) final;
    virtual void run(const USBDevice &dev) final;
    XString virtual getString(const USBDevice &dev, int descid) final;
    virtual void download(const USBDevice &dev, uint8_t* image, int len) final;
    virtual int bulkWrite(int pipe, uint8_t *buf, int len) final;
    virtual int bulkRead(int pipe, const uint8_t* buf, int len) final;

    virtual unsigned int vendorID() final;
    virtual unsigned int productID() final;

    virtual AsyncIO asyncBulkWrite(int pipe, uint8_t *buf, int len) final;
    virtual AsyncIO asyncBulkRead(int pipe, const uint8_t *buf, int len) final;

private:
    HANDLE handle;
    XString name;

    //AE18AA60-7F6A-11d4-97DD-00010229B959
    constexpr tGUID GUID = {0xae18aa60, 0x7f6a, 0x11d4, 0x97, 0xdd, 0x0, 0x1, 0x2, 0x29, 0xb9, 0x59};
};

#endif // CYFXUSB_WIN32_H
