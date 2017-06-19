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
#include "cyfxusbinterface_impl.h"
#include "thamwayusbinterface.h"
#include "charinterface.h"
#include "fx2fw.h"
#include <cstring>

constexpr size_t FX2FW_MAX_BURST_SIZE_USB2 = 512; //Thamway's value was 40000(WR)/512(RD).
constexpr size_t FX2FW_MAX_BURST_SIZE_USB1_1 = 512; //64 //Thamway's value was 40000(WR)/512(RD).

#define CMD_DIPSW 0x11u
#define CMD_LED 0x12u

#define DEV_ADDR_PROT 0x6

#define ADDR_IDN 0x1fu
#define ADDR_CHARINTF 0xa0u

#define THAMWAY_USB_FIRMWARE_FILE "fx2fw.bix"
#define THAMWAY_USB_GPIFWAVE1_FILE "slow_dat.bin" //for USB1.1
#define THAMWAY_USB_GPIFWAVE2_FILE "fullspec_dat.bin" //for USB2.0 burst-transfer enabled

#define FX2_DEF_VID 0x4b4
#define FX2_DEF_PID 0x8613 //cypress default FX2.
#define THAMWAY_VID 0x547
#define THAMWAY_PID 0x1002

#define EPOUT2 2 //TFIFO
#define EPOUT8 8 //CPIPE
#define EPIN6 6 //RFIFO

#define FX3_DEF_VID 0x4b4
#define FX3_DEF_PID 0x00f1 //cypress default FX3 for FIFOSYNC.

#define EPIN1 1
#define EPOUT1 1

template class XCyFXUSBInterface<ThamwayCyFX2USBDevice>;
template class XCyFXUSBInterface<ThamwayCyFX3USBDevice>;

SoftwareTriggerManager XThamwayFX3USBInterface::s_softwareTriggerManager;

XThamwayFX2USBInterface::XThamwayFX2USBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver,
        uint8_t addr_offset, const char* id) :
     XCyFXUSBInterface<ThamwayCyFX2USBDevice>(name, runtime, driver),
    m_addrOffset(addr_offset), m_idString(id) {
    initialize(); //open all supported USB devices and loads firmware, writes GPIF waves.
}

XThamwayFX2USBInterface::~XThamwayFX2USBInterface() {
    finalize();
}

XThamwayFX2USBInterface::DEVICE_STATUS
XThamwayFX2USBInterface::examineDeviceBeforeFWLoad(const shared_ptr<CyFXUSBDevice> &dev) {
    if((dev->productID() != FX2_DEF_PID) || (dev->vendorID() != FX2_DEF_VID)) {
        if((dev->productID() == THAMWAY_PID) && (dev->vendorID() == THAMWAY_VID)) {
            dev->open();
            constexpr char Manufacturer_sym[] = "F2FW";
            try {
                XString s1 = dev->getString(1);
                dbgPrint(formatString("USB: Device: %s", s1.c_str()));
                if(s1 == Manufacturer_sym) {
                    return DEVICE_STATUS::READY;
                }
            }
            catch (XInterface::XInterfaceError &) {
                dbgPrint("USB: ???");
                return DEVICE_STATUS::FW_NOT_LOADED;
            }
        }
        else
            return DEVICE_STATUS::UNSUPPORTED;
    }
    else
        dev->open();
    constexpr char Serial_sym[] = "20070627";
    try {
        XString s2 = dev->getString(2);
        dbgPrint(formatString("USB: Ver: %s\n", s2.c_str()));
        if(s2[0] != Serial_sym[0]) {
            dbgPrint("USB: Not Thamway's device");
            dev->close();
            return DEVICE_STATUS::UNSUPPORTED;
        }
        unsigned int version = atoi(s2.c_str());
        if(version < atoi(Serial_sym))
            return DEVICE_STATUS::FW_NOT_LOADED;
        return DEVICE_STATUS::UNSUPPORTED;
    }
    catch (XInterfaceError& e) {
        dbgPrint("USB: ???");
        return DEVICE_STATUS::FW_NOT_LOADED;
    }
}

std::string
XThamwayFX2USBInterface::examineDeviceAfterFWLoad(const shared_ptr<CyFXUSBDevice> &dev) {
    uint8_t dipsw = readDIPSW(dev);
    XString idn;
    if(m_idString.empty()) {
        idn = "PROT";
    }
    else {
        //for PG and DV series.
        idn = getIDN(dev, 8);
        if( !idn.length()) return {};
    }
    idn = formatString("%d:%s", (int)dipsw, idn.c_str());
    return idn;
}

XString
XThamwayFX2USBInterface::gpifWave(const shared_ptr<CyFXUSBDevice> &dev) {
    try {
        uint8_t dipsw = readDIPSW(dev);
        if(dipsw != DEV_ADDR_PROT)
//           return {THAMWAY_USB_GPIFWAVE2_FILE};
            return {THAMWAY_USB_GPIFWAVE1_FILE}; //Thamway recommends slow_dat.bin always.
    }
    catch (XInterfaceError &) {
        gWarnPrint("Reading DIPSW value resulted in failure, continuing...");
    }
    return {THAMWAY_USB_GPIFWAVE1_FILE};
}

XString
XThamwayFX2USBInterface::firmware(const shared_ptr<CyFXUSBDevice> &dev) {
    return THAMWAY_USB_FIRMWARE_FILE;
}

void
XThamwayFX2USBInterface::setWave(const shared_ptr<CyFXUSBDevice> &dev, const uint8_t *wave) {
    std::vector<uint8_t> buf;
    buf.insert(buf.end(), {CMD_MODE, MODE_GPIF | MODE_8BIT | MODE_ADDR | MODE_NOFLOW | MODE_DEBG, CMD_GPIF});
    buf.insert(buf.end(), wave, wave + 8);
    buf.insert(buf.end(), {MODE_FLOW});
    buf.insert(buf.end(), wave + 8 + 32*4, wave + 8 + 32*4 + 36);
    dev->bulkWrite(EPOUT8, &buf[0], buf.size());
    const uint8_t cmdwaves[] = {CMD_WAVE0 /*SingleRead*/, CMD_WAVE1/*SingleWrite*/, CMD_WAVE2/*BurstRead*/, CMD_WAVE3/*BurstWrite*/};
    for(int i = 0; i < sizeof(cmdwaves); ++i) {
        buf.clear();
        buf.insert(buf.end(), cmdwaves + i, cmdwaves + i + 1);
        buf.insert(buf.end(), wave + 8 + 32*i, wave + 8 + 32*(i + 1));
        dev->bulkWrite(EPOUT8, &buf[0], buf.size());
    }
    msecsleep(200);
}

void
XThamwayFX2USBInterface::open() throw (XInterfaceError &) {
    XCyFXUSBInterface<ThamwayCyFX2USBDevice>::open();
//    for(int i = 0; i < 1; ++i) {
//        //blinks LED
//        setLED(usb(), 0x00u);
//        msecsleep(30);
//        setLED(usb(), 0xf0u);
//        msecsleep(30);
//    }
    resetBulkWrite();

    uint8_t cmds[] = {CMD_USBCS};
    usb()->bulkWrite(EPOUT8, cmds, sizeof(cmds));
    uint8_t buf[10];
    usb()->bulkRead(EPIN6, buf, 1);
    bool is_usb2 = buf[0] & 0x80u;
    m_maxBurstRWSize = is_usb2 ? FX2FW_MAX_BURST_SIZE_USB2 : FX2FW_MAX_BURST_SIZE_USB1_1;
    dbgPrint(formatString("FX2FW connected to %s, max_bsize=%u", is_usb2 ? "USB2" : "USB1.1", (unsigned int)m_maxBurstRWSize));
}

void
XThamwayFX2USBInterface::close() throw (XInterfaceError &) {
//    if(isOpened()) setLED(usb(), 0);
    XCyFXUSBInterface<ThamwayCyFX2USBDevice>::close();
}

void
XThamwayFX2USBInterface::resetBulkWrite() noexcept {
    m_isDeferredWritingOn = false;
    m_buffer.clear();
    m_buffer.reserve(FX2FW_MAX_BURST_SIZE_USB2);
}
void
XThamwayFX2USBInterface::deferWritings() {
    assert(m_buffer.size() == 0);
    m_isDeferredWritingOn = true;
}
void
XThamwayFX2USBInterface::writeToRegister8(unsigned int addr, uint8_t data) {
    addr += m_addrOffset;
    assert(addr < 0x100u);

    if(m_isDeferredWritingOn) {
        if(m_buffer.size() >= m_maxBurstRWSize) {
            XScopedLock<XThamwayFX2USBInterface> lock( *this);
            bulkWriteStored();
            deferWritings();
        }
        m_buffer.push_back(addr);
        m_buffer.push_back(data);
    }
    else {
        XScopedLock<XThamwayFX2USBInterface> lock( *this);
        dbgPrint(driver()->getLabel() + formatString(" SingleWriting @ %x; %x", addr, (unsigned int)data));
        uint8_t cmds[] = {CMD_BWRITE, 2, 0}; //2bytes to be written.
        usb()->bulkWrite(EPOUT8, cmds, sizeof(cmds));
        uint8_t cmds2[] = {(uint8_t)(addr), data};
        usb()->bulkWrite(EPOUT2, cmds2, sizeof(cmds2));
    }
}
void
XThamwayFX2USBInterface::writeToRegister16(unsigned int addr, uint16_t data) {
    if(m_isDeferredWritingOn) {
        writeToRegister8(addr, data % 0x100u);
        writeToRegister8(addr + 1, data / 0x100u);
    }
    else {
        XScopedLock<XThamwayFX2USBInterface> lock( *this);
        writeToRegister8(addr, data % 0x100u);
        writeToRegister8(addr + 1, data / 0x100u);
    }
}
void
XThamwayFX2USBInterface::bulkWriteStored() {
    XScopedLock<XThamwayFX2USBInterface> lock( *this);

    uint16_t len = m_buffer.size();
    dbgPrint(driver()->getLabel() + formatString(" BurstWriting for %x bytes", (unsigned int)len));
    uint8_t cmds[] = {CMD_BWRITE, (uint8_t)(len % 0x100u), (uint8_t)(len / 0x100u)};
    usb()->bulkWrite(EPOUT8, cmds, sizeof(cmds));
    usb()->bulkWrite(EPOUT2, (uint8_t*) &m_buffer[0], len);

    resetBulkWrite();
}

void
XThamwayFX2USBInterface::setLED(const shared_ptr<CyFXUSBDevice> &dev, uint8_t data) {
    XScopedLock<XRecursiveMutex> lock(dev->mutex);
    uint8_t cmds[] = {CMD_LED, data};
    dev->bulkWrite(EPOUT8, cmds, sizeof(cmds));
}

uint8_t
XThamwayFX2USBInterface::readDIPSW(const shared_ptr<CyFXUSBDevice> &dev) {
    XScopedLock<XRecursiveMutex> lock(dev->mutex);
    uint8_t cmds[] = {CMD_DIPSW};
    dev->bulkWrite(EPOUT8, cmds, sizeof(cmds));
    uint8_t buf[10];
    dev->bulkRead(EPIN6, buf, 1);
    return buf[0];
}

XString
XThamwayFX2USBInterface::getIDN(const shared_ptr<CyFXUSBDevice> &dev, int maxlen, int addroffset) {
    XScopedLock<XRecursiveMutex> lock(dev->mutex);
    //ignores till \0
    for(int i = 0; ; ++i) {
        char c = singleRead(dev, ADDR_IDN, addroffset);
        if( !c)
            break;
        if(i > 255) {
            return {}; //failed
        }
    }
    XString idn;
    for(int i = 0; ; ++i) {
        char c = singleRead(dev, ADDR_IDN, addroffset);
        if( !c)
            break;
        idn += c;
        if(i >= maxlen) {
            break;
        }
    }
    fprintf(stderr, "getIDN:%s\n", idn.c_str());
    return idn;
}
uint8_t
XThamwayFX2USBInterface::singleRead(unsigned int addr) {
    return singleRead(usb(), addr, m_addrOffset);
}

uint8_t
XThamwayFX2USBInterface::singleRead(const shared_ptr<CyFXUSBDevice> &dev, unsigned int addr, unsigned int addroffset) {
    XScopedLock<XRecursiveMutex> lock(dev->mutex);
    addr += addroffset;
    assert(addr < 0x100u);
    dbgPrint(formatString("FX2USB: SingleReading @ %x", addr));
    {
        uint8_t cmds[] = {CMD_SWRITE, (uint8_t)(addr)};
        dev->bulkWrite(EPOUT8, cmds, sizeof(cmds));
    }
    {
        uint8_t cmds[] = {CMD_SREAD};
        dev->bulkWrite(EPOUT8, cmds, sizeof(cmds));
        uint8_t buf[10];
        dev->bulkRead(EPIN6, buf, 1);
        dbgPrint(formatString(" Received; %x", (unsigned int)buf[0]));
        return buf[0];
    }
}
uint16_t
XThamwayFX2USBInterface::readRegister16(unsigned int addr) {
    XScopedLock<XThamwayFX2USBInterface> lock( *this);
    return singleRead(addr) + singleRead(addr + 1) * (uint16_t)0x100u;
}

void
XThamwayFX2USBInterface::burstRead(unsigned int addr, uint8_t *buf, unsigned int cnt) {
    XScopedLock<XThamwayFX2USBInterface> lock( *this);
    addr += m_addrOffset;
    assert(addr < 0x100u);
    {
        uint8_t cmds[] = {CMD_SWRITE, (uint8_t)(addr)};
        usb()->bulkWrite(EPOUT8, cmds, sizeof(cmds));
    }
    std::vector<uint8_t> bbuf(m_maxBurstRWSize);
    std::uint8_t cmds[] = {CMD_BREAD, 0, 0};
    cmds[1] = bbuf.size() % 0x100u;
    cmds[2] = bbuf.size() / 0x100u;
    dbgPrint(driver()->getLabel() + formatString(" BurstReading @%x for %u bytes", addr, cnt));
    for(; cnt;) {
        usb()->bulkWrite(EPOUT8, cmds, sizeof(cmds));
        //BREAD is only allowed in a unit of packet size???.
        int i = usb()->bulkRead(EPIN6, &bbuf[0], bbuf.size());
        unsigned int n = std::min(cnt, (unsigned int)i);
        std::memcpy(buf, &bbuf[0], n);
        buf += n;
        cnt -= n;
    }
}

void
XThamwayFX2USBInterface::send(const char *str) {
    XScopedLock<XInterface> lock(*this);
    XScopedLock<XThamwayFX2USBInterface> lock2( *this);
    try {
        dbgPrint(driver()->getLabel() + " Sending:\"" + dumpCString(str) + "\"");
        XString buf = str + eos();
        for(int i = 0; i < buf.length(); ++i) {
            writeToRegister8(ADDR_CHARINTF, (uint8_t)buf[i]);
        }
    }
    catch (XInterfaceError &e) {
        e.print(driver()->getLabel() + i18n(" SendError, because "));
        throw e;
    }
}
void
XThamwayFX2USBInterface::receive() {
    XScopedLock<XInterface> lock(*this);
    XScopedLock<XThamwayFX2USBInterface> lock2( *this);
    msecsleep(20);
    buffer_receive().clear();
    try {
        dbgPrint(driver()->getLabel() + " Receiving...");
        for(int i = 0; ; ++i) {
            uint8_t c = singleRead(ADDR_CHARINTF);
            if( !c || (c == 0xffu))
                break;
            if( i > 256 )
                throw XInterface::XCommError(i18n("USB string length exceeded the limit."), __FILE__, __LINE__);
            buffer_receive().push_back(c);
        }
        buffer_receive().push_back('\0');
        dbgPrint(driver()->getLabel() + " Received;\"" +
                 dumpCString((const char*)&buffer()[0]) + "\"");
    }
    catch (XInterfaceError &e) {
        e.print(driver()->getLabel() + i18n(" ReceiveError, because "));
        throw e;
    }
}


XThamwayFX3USBInterface::XThamwayFX3USBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
    XCyFXUSBInterface<ThamwayCyFX3USBDevice>(name, runtime, driver) {
    initialize();
}

XThamwayFX3USBInterface::~XThamwayFX3USBInterface() {
    finalize();
}

void
XThamwayFX3USBInterface::open() throw (XInterfaceError &) {
    XCyFXUSBInterface<ThamwayCyFX3USBDevice>::open();
}

void
XThamwayFX3USBInterface::close() throw (XInterfaceError &) {
    XCyFXUSBInterface<ThamwayCyFX3USBDevice>::close();
}

XThamwayFX3USBInterface::DEVICE_STATUS
XThamwayFX3USBInterface::examineDeviceBeforeFWLoad(const shared_ptr<CyFXUSBDevice> &dev) {
    if((dev->productID() != FX3_DEF_PID) || (dev->vendorID() != FX3_DEF_VID)) {
        return DEVICE_STATUS::UNSUPPORTED;
    }
    dev->open();
    constexpr char Manufactor_sym[] = "THAMWAY";
    constexpr char Product_sym[] = "SF,WATER=8,1227A"; //SyncFIFO
    try {
        XString s1 = dev->getString(1);
        XString s2 = dev->getString(2);
        dbgPrint(formatString("USB: Manu: %s, Prod: %s.", s1.c_str(), s2.c_str()));
        if(s1 != Manufactor_sym) {
            dbgPrint("USB: Not Thamway's device.");
            return DEVICE_STATUS::UNSUPPORTED;
        }
        if(s2 != Product_sym) {
            dbgPrint("USB: Unsupported device.");
            return DEVICE_STATUS::UNSUPPORTED;
        }
    }
    catch (XInterfaceError& e) {
        dbgPrint("USB: ???");
        return DEVICE_STATUS::UNSUPPORTED;
    }
    return DEVICE_STATUS::READY;
}
std::string
XThamwayFX3USBInterface::examineDeviceAfterFWLoad(const shared_ptr<CyFXUSBDevice> &dev) {
    return {"FX3"};
}
void
XThamwayFX3USBInterface::send(const char *str) {
    usb()->bulkWrite(EPOUT1, (const uint8_t*)str, strlen(str));
}
void
XThamwayFX3USBInterface::receive() {
    buffer_receive().resize(16*2048);
    int ret = usb()->bulkRead(EPIN1, (uint8_t *)&buffer_receive()[0], buffer_receive().size());
    buffer_receive().resize(ret);
}

unique_ptr<CyFXUSBDevice::AsyncIO>
XThamwayFX3USBInterface::asyncReceive(char *buf, ssize_t size) {
    return usb()->asyncBulkRead(EPIN1, (uint8_t *)buf, size);
}


