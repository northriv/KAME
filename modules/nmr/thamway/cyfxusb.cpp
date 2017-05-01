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

CyFXUSBDevice::halt() {
    //Writes the CPUCS register of i8051.
    uint8_t buf[1] = {1};
    if(controlWrite((CtrlReq)0xA0, CtrlReqType::VENDOR, 0xe600, 0x00, buf, 1) != 1)
        throw XInterfaceError(i18n("i8051 halt err."));
}

void
CyFXUSBDevice::run() {
    //Writes the CPUCS register of i8051.
    uint8_t buf[1] = {0};
    if(controlWrite((CtrlReq)0xA0, CtrlReqType::VENDOR, 0xe600, 0x00, buf, 1) != 1)
        throw XInterfaceError(i18n("i8051 run err."));
}


void
CyFXUSBDevice::downloadFX2(const uint8_t* image, int len) {
    int addr = 0;
    //A0 anchor download.
    if(controlWrite((CtrlReq)0xA0, CtrlReqType::VENDOR, addr, 0x00, image, len) != 1)
        throw XInterfaceError(i18n("Error: FX2 write to RAM failed."));
}


