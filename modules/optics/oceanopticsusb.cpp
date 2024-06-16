/***************************************************************************
        Copyright (C) 2002-2022 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/

#if defined USE_OCEANOPTICS_USB

#include "oceanopticsusb.h"
#include "cyfxusbinterface_impl.h"
#include "interface.h"
#include <cstring>

//incl. static members, used for enumeration of supported devices, and intefaces
template class XCyFXUSBInterface<OceanOpticsUSBDevice>;

static constexpr unsigned int OCEANOPTICS_VENDOR_ID = 0x2457;
static const std::map<unsigned int, std::string> cs_oceanOpticsModels = {
    {0x1011, "HR4000"},
    {0x1012, "HR2000+/4000"}, //tested
    {0x1016, "HR2000+"}, //tested.
    {0x101e, "USB2000+"},
    {0x1022, "USB4000"}, //tested.
};

XOceanOpticsUSBInterface::DEVICE_STATUS
XOceanOpticsUSBInterface::examineDeviceBeforeFWLoad(const shared_ptr<CyFXUSBDevice> &dev) {
    try {
        if(dev->vendorID() == OCEANOPTICS_VENDOR_ID) {
            cs_oceanOpticsModels.at(dev->productID());
            return DEVICE_STATUS::READY;
        }
    }
    catch (std::out_of_range &e) {
        return DEVICE_STATUS::UNSUPPORTED;
    }
    return DEVICE_STATUS::UNSUPPORTED;
}
std::string
XOceanOpticsUSBInterface::examineDeviceAfterFWLoad(const shared_ptr<CyFXUSBDevice> &dev) {
    return cs_oceanOpticsModels.at(dev->productID());
}

void
XOceanOpticsUSBInterface::initDevice() {
    uint8_t cmds[] = {(uint8_t)CMD::INIT};
    usb()->bulkWrite(m_ep_cmd, cmds, sizeof(cmds));
}

void
XOceanOpticsUSBInterface::setIntegrationTime(unsigned int us) {
    uint8_t hh = us / 0x1000000uL;
    uint8_t hl = (us / 0x10000uL) % 0x100uL;
    uint8_t lh = (us / 0x100uL) % 0x100uL;
    uint8_t ll = us % 0x100uL;
    uint8_t cmds[] = {(uint8_t)CMD::SET_INTEGRATION_TIME, ll, lh, hl, hh}; //littleendian
    usb()->bulkWrite(m_ep_cmd, cmds, sizeof(cmds));
}

std::vector<uint8_t>
XOceanOpticsUSBInterface::readInstrumStatus() {
    XScopedLock<XOceanOpticsUSBInterface> lock( *this);
    uint8_t cmds[] = {(uint8_t)CMD::QUERY_OP_INFO};
    usb()->bulkWrite(m_ep_cmd, cmds, sizeof(cmds));
    std::vector<uint8_t> stat(16);
    usb()->bulkRead(m_ep_in_others, (uint8_t*)&stat[0], stat.size());
    return stat;
}

XOceanOpticsUSBInterface::InstrumConfig
XOceanOpticsUSBInterface::readConfigurations() {
    InstrumConfig config;
    auto fn_query_conf = [this](uint8_t no){
        XScopedLock<XOceanOpticsUSBInterface> lock( *this);
        uint8_t cmds[] = {(uint8_t)CMD::QUERY_INFO, no};
        usb()->bulkWrite(m_ep_cmd, cmds, sizeof(cmds));
        uint8_t buf[CMD_READ_SIZE + 1];
        buf[CMD_READ_SIZE] = '\0';
        int size = usb()->bulkRead(m_ep_in_others, buf, CMD_READ_SIZE);
        if((buf[0] != cmds[0]) || (buf[1] != cmds[1]))
            throw XInterface::XConvError(__FILE__, __LINE__);
        return std::string((char*)&buf[2]);
    };
    config.serialNo = fn_query_conf(0);
    for(unsigned int i = 0; i < 4; ++i)
        config.wavelenCalib[i] = fn_query_conf(i + 1);
    config.strayLightConst = fn_query_conf(5);
    for(unsigned int i = 0; i < 8; ++i)
        config.nonlinCorr[i] = fn_query_conf(i + 6);
    config.nlpoly = fn_query_conf(14);
    config.opticalBenchConfig = fn_query_conf(15);
    config.spectrometerConfig = fn_query_conf(16);

    return config;
}

int
XOceanOpticsUSBInterface::readSpectrum(std::vector<uint8_t> &buf, uint16_t pixels, bool usb_highspeed) {
    XScopedLock<XOceanOpticsUSBInterface> lock( *this);
    uint8_t cmds[] = {(uint8_t)CMD::REQUEST_SPECTRA};
    usb()->bulkWrite(m_ep_cmd, cmds, sizeof(cmds));

    buf.resize(2 * pixels + 1);
    int len = 0;
    if(usb_highspeed && (pixels > 2048)) {
        //HR4000, first 2K pixels use the other end point.
        len += usb()->bulkRead(m_ep_in_spec_first1Kpixels, &buf[0], 1024 * 2);
    }
    len += usb()->bulkRead(m_ep_in_spec, &buf[len], buf.size() - len);

    return len;
}
#endif // OCEANOPTICSUSB_H
