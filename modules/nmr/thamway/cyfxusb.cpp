/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "cyfxusb.h"

XThreadLocal<std::vector<uint8_t>>
CyFXUSBDevice::AsyncIO::s_tlBufferGarbage;

int64_t
CyFXUSBDevice::bulkWrite(uint8_t ep, const uint8_t *buf, int len) {
    auto async = asyncBulkWrite(ep, buf, len);
    auto ret = async->waitFor();
    return ret;
}

int64_t
CyFXUSBDevice::bulkRead(uint8_t ep, uint8_t* buf, int len) {
    auto async = asyncBulkRead(ep, buf, len);
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
    controlWrite((CtrlReq)0xA0, CtrlReqType::VENDOR, addr, 0x00, image, len);
}


