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
#include "cyfxusb.h"

//Timeout in libusb_fill_control_transfer() causes device freeze!
//This value should be longer than timeout in waitFor().
static constexpr unsigned int TIMEOUT_MS_LONG_ENOUGH = 10000;

XThreadLocal<std::vector<uint8_t>>
CyFXUSBDevice::AsyncIO::stl_bufferGarbage;

int64_t
CyFXUSBDevice::bulkWrite(uint8_t ep, const uint8_t *buf, int len) {
    auto async = asyncBulkWrite(ep, buf, len, TIMEOUT_MS_LONG_ENOUGH);
    auto ret = async->waitFor();
    return ret;
}

int64_t
CyFXUSBDevice::bulkRead(uint8_t ep, uint8_t* buf, int len) {
    auto async = asyncBulkRead(ep, buf, len, TIMEOUT_MS_LONG_ENOUGH);
    return async->waitFor();
}

void
CyFXUSBDevice::halt() {
    //Writes the CPUCS register of i8051.
    uint8_t buf[1] = {1};
    controlWrite((CtrlReq)0xA0, CtrlReqType::VENDOR, 0xe600, 0x00, buf, 1);
}

void
CyFXUSBDevice::run() {
    //Writes the CPUCS register of i8051.
    uint8_t buf[1] = {0};
    controlWrite((CtrlReq)0xA0, CtrlReqType::VENDOR, 0xe600, 0x00, buf, 1);
}


void
CyFXUSBDevice::downloadFX2(const uint8_t* image, int len) {
    int addr = 0;
    //A0 anchor download.
#if defined __MACOSX__ || defined __APPLE__
    controlWrite((CtrlReq)0xA0, CtrlReqType::VENDOR, addr, 0x00, image, len);
#else
    //winUSB cannot send data > 4kB at once.
    constexpr int chunk = 4096;
    while(len) {
        int size = std::min(chunk, len);
        controlWrite((CtrlReq)0xA0, CtrlReqType::VENDOR, addr, 0x00, image + addr, size);
        addr += size;
        len -= size;
    }
#endif
}
