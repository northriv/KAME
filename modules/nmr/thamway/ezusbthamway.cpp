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
#include "ezusbthamway.h"
#include "charinterface.h"
#include <QFile>
#include <QDir>
#include <QApplication>
#include <QStandardPaths>

#define CUSB_BULK_WRITE_SIZE 40000

#ifdef USE_EZUSB_FX2FW
    #define NOMINMAX
    #include <windows.h>
    extern "C" {
        #include "cusb.h"
    }
    #define KAME_EZUSB_DIR ""
    inline int cusblib_initialize() {return 8;}
    inline void cusblib_finalize() {}
    using usb_handle = HANDLE;
#endif
#ifdef USE_EZUSB_CYUSB
    #include "cusb2cyusb.h"
#endif
#ifdef USE_EZUSB
    extern "C" {
        #include "fx2fw.h"
    }
#endif

#define CMD_DIPSW 0x11u
#define CMD_LED 0x12u

#define ADDR_IDN 0x1fu
#define ADDR_CHARINTF 0xa0u

#define THAMWAY_USB_FIRMWARE_FILE "fx2fw.bix"
#define THAMWAY_USB_GPIFWAVE1_FILE "slow_dat.bin" //for USB1.1
#define THAMWAY_USB_GPIFWAVE2_FILE "fullspec_dat.bin" //for USB2.0 burst-transfer enabled
#define THAMWAY_USB_GPIFWAVE_SIZE 172

XMutex XFX2FWUSBInterface::s_mutex;
int XFX2FWUSBInterface::s_refcnt = 0;
std::deque<XFX2FWUSBInterface::USBDevice> XFX2FWUSBInterface::s_devices;

void
XFX2FWUSBInterface::openAllEZUSBdevices() {
    QDir dir(QApplication::applicationDirPath());
    auto load_firm = [&dir](char *data, int expected_size, const char *filename){
        QString path =
    #ifdef WITH_KDE
            KStandardDirs::locate("appdata", filename);
    #else
            QStandardPaths::locate(QStandardPaths::DataLocation, filename);
        if(path.isEmpty()) {
            //for macosx/win
            QDir dir(QApplication::applicationDirPath());
    #if defined __MACOSX__ || defined __APPLE__
            //For macosx application bundle.
            dir.cdUp();
    #endif
            QString path2 = KAME_EZUSB_DIR;
            path2 += filename;
            dir.filePath(path2);
            if(dir.exists())
                path = dir.absoluteFilePath(path2);
        }
    #endif
        dir.filePath(path);
        if( !dir.exists())
            throw XInterface::XInterfaceError(i18n_noncontext("USB GPIF/firmware file ") +
                filename + i18n_noncontext(" not found."), __FILE__, __LINE__);
        QFile file(dir.absoluteFilePath(path));
        if( !file.open(QIODevice::ReadOnly))
            throw XInterface::XInterfaceError(i18n_noncontext("USB GPIF/firmware file ") +
                filename + i18n_noncontext(" not found."), __FILE__, __LINE__);
        int size = file.read(data, expected_size);
        if(size != expected_size)
            throw XInterface::XInterfaceError(i18n_noncontext("USB GPIF/firmware file ") +
                filename + i18n_noncontext(" is not proper."), __FILE__, __LINE__);
    };

    char firmware[CUSB_DWLSIZE];
    load_firm(firmware, CUSB_DWLSIZE, THAMWAY_USB_FIRMWARE_FILE);
    char gpifwave1[THAMWAY_USB_GPIFWAVE_SIZE];
    load_firm(gpifwave1, THAMWAY_USB_GPIFWAVE_SIZE, THAMWAY_USB_GPIFWAVE1_FILE);
    char gpifwave2[THAMWAY_USB_GPIFWAVE_SIZE];
    bool always_slow_usb = false;
    try {
        load_firm(gpifwave2, THAMWAY_USB_GPIFWAVE_SIZE, THAMWAY_USB_GPIFWAVE2_FILE);
    }
    catch (XInterface::XInterfaceError& e) {
        e.print();
        gMessagePrint(i18n_noncontext("Continues with slower USB speed."));
        always_slow_usb = true;
    }

    int num_devices = cusblib_initialize();
    if(num_devices < 0) {
        throw XInterface::XInterfaceError(i18n_noncontext("Error during initialization of libusb.")
                                          , __FILE__, __LINE__);
    }

    for(int i = 0; i < num_devices; ++i) {
        usb_handle handle = 0;
        fprintf(stderr, "cusb_init #%d\n", i);
        if(cusb_init(i, &handle, (uint8_t *)firmware,
            (signed char*)"F2FW", (signed char*)"20070627")) {
            //no device, or incompatible firmware.
            continue;
        }
        uint8_t sw = readDIPSW(handle);
        USBDevice dev;
        dev.handle = handle;
        dev.addr = sw;
        dev.mutex.reset(new XRecursiveMutex);
        s_devices.push_back(dev);
        fprintf(stderr, "Setting GPIF waves for handle 0x%x, DIPSW=%x\n", (unsigned int)(uintptr_t)handle, (unsigned int)sw);
        char *gpifwave = gpifwave2;
        if(always_slow_usb || (dev.addr == DEV_ADDR_PROT))
            gpifwave = gpifwave1;
        setWave(handle, (const uint8_t*)gpifwave);

        for(int i = 0; i < 3; ++i) {
            //blinks LED
            setLED(handle, 0x00u);
            msecsleep(70);
            setLED(handle, 0xf0u);
            msecsleep(60);
        }
    }
    if(s_devices.empty())
        throw XInterface::XInterfaceError(i18n_noncontext("USB-device open has failed."), __FILE__, __LINE__);
}

void
XFX2FWUSBInterface::setWave(void *handle, const uint8_t *wave) {
    std::vector<uint8_t> buf;
    buf.insert(buf.end(), {CMD_MODE, MODE_GPIF | MODE_8BIT | MODE_ADDR | MODE_NOFLOW | MODE_DEBG, CMD_GPIF});
    buf.insert(buf.end(), wave, wave + 8);
    buf.insert(buf.end(), {MODE_FLOW});
    buf.insert(buf.end(), wave + 8 + 32*4, wave + 8 + 32*4 + 36);
    if(usb_bulk_write( (usb_handle*)&handle, CPIPE, &buf[0], buf.size()) < 0)
        throw XInterface::XInterfaceError(i18n_noncontext("USB bulk writing has failed."), __FILE__, __LINE__);
    const uint8_t cmdwaves[] = {CMD_WAVE0 /*SingleRead*/, CMD_WAVE1/*SingleWrite*/, CMD_WAVE2/*BurstRead*/, CMD_WAVE3/*BurstWrite*/};
    for(int i = 0; i < sizeof(cmdwaves); ++i) {
        buf.clear();
        buf.insert(buf.end(), cmdwaves + i, cmdwaves + i + 1);
        buf.insert(buf.end(), wave + 8 + 32*i, wave + 8 + 32*(i + 1));
        if(usb_bulk_write( (usb_handle*)&handle, CPIPE, &buf[0], buf.size()) < 0)
            throw XInterface::XInterfaceError(i18n_noncontext("USB bulk writing has failed."), __FILE__, __LINE__);
    }
}
void
XFX2FWUSBInterface::closeAllEZUSBdevices() {
    for(auto it = s_devices.begin(); it != s_devices.end(); ++it) {
        try {
            setLED(it->handle, 0);
        }
        catch (XInterface::XInterfaceError &e) {
            e.print();
        }

        fprintf(stderr, "cusb_close\n");
        usb_close( (usb_handle*)&it->handle);
    }
    cusblib_finalize();
    s_devices.clear();
}

XFX2FWUSBInterface::XFX2FWUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver, uint8_t addr_offset, const char* id)
    : XCustomCharInterface(name, runtime, driver), m_handle(0), m_idString(id), m_addrOffset(addr_offset) {
    XScopedLock<XMutex> slock(s_mutex);
    try {
        if( !s_refcnt)
            openAllEZUSBdevices();
        s_refcnt++;

        iterate_commit([=](Transaction &tr){
            for(auto it = s_devices.begin(); it != s_devices.end(); ++it) {
                XString idn;
                if(strlen(id)) {
                    //for PG and DV series.
                    idn = getIDN(it->handle, 7);
                    if( !idn.length()) continue;
                }
                else {
                    //for PROT
                    if(it->addr != DEV_ADDR_PROT) continue;
                    idn = "PROT";
                }
                idn = formatString("%d:%s", it->addr, idn.c_str());
                tr[ *device()].add(idn);
            }
        });
    }
    catch (XInterface::XInterfaceError &e) {
        e.print();
    }
}

XFX2FWUSBInterface::~XFX2FWUSBInterface() {
    if(isOpened()) close();

    XScopedLock<XMutex> slock(s_mutex);
    s_refcnt--;
    if( !s_refcnt)
        closeAllEZUSBdevices();
}

void
XFX2FWUSBInterface::open() throw (XInterfaceError &) {
    Snapshot shot( *this);
    try {
        for(auto it = s_devices.begin(); it != s_devices.end(); ++it) {
            int addr;
            if(sscanf(shot[ *device()].to_str().c_str(), "%d:", &addr) != 1)
                throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
            if(addr == it->addr) {
                m_handle = it->handle;
                m_mutex = it->mutex;
            }
        }
        if( !m_handle)
            throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);
    }
    catch (XInterface::XInterfaceError &e) {
        m_handle = 0;
        m_mutex.reset();
        throw;
    }
    resetBulkWrite();
}

void
XFX2FWUSBInterface::close() throw (XInterfaceError &) {
    m_handle = 0;
    m_mutex.reset();
}

void
XFX2FWUSBInterface::resetBulkWrite() {
    m_bBurstWrite = false;
    m_buffer.clear();
}
void
XFX2FWUSBInterface::deferWritings() {
    assert(m_buffer.size() == 0);
    m_bBurstWrite = true;
}
void
XFX2FWUSBInterface::writeToRegister16(unsigned int addr, uint16_t data) {
    if(m_bBurstWrite) {
        writeToRegister8(addr, data % 0x100u);
        writeToRegister8(addr + 1, data / 0x100u);
    }
    else {
        XScopedLock<XFX2FWUSBInterface> lock( *this);
        writeToRegister8(addr, data % 0x100u);
        writeToRegister8(addr + 1, data / 0x100u);
    }
}
void
XFX2FWUSBInterface::writeToRegister8(unsigned int addr, uint8_t data) {
    addr += m_addrOffset;
    assert(addr < 0x100u);

    if(m_bBurstWrite) {
        if(m_buffer.size() > CUSB_BULK_WRITE_SIZE) {
            XScopedLock<XFX2FWUSBInterface> lock( *this);
            bulkWriteStored();
            deferWritings();
        }
        m_buffer.push_back(addr);
        m_buffer.push_back(data);
    }
    else {
        XScopedLock<XFX2FWUSBInterface> lock( *this);
        uint8_t cmds[] = {CMD_BWRITE, 2, 0}; //2bytes to be written.
        if(usb_bulk_write( (usb_handle*)&m_handle, CPIPE, cmds, sizeof(cmds)) < 0)
            throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
        uint8_t cmds2[] = {(uint8_t)(addr), data};
        if(usb_bulk_write( (usb_handle*)&m_handle, TFIFO, cmds2, sizeof(cmds2)) < 0)
            throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
    }
}
void
XFX2FWUSBInterface::bulkWriteStored() {
    XScopedLock<XFX2FWUSBInterface> lock( *this);

    uint16_t len = m_buffer.size();
    uint8_t cmds[] = {CMD_BWRITE, (uint8_t)(len % 0x100u), (uint8_t)(len / 0x100u)};
    if(usb_bulk_write( (usb_handle*)&m_handle, CPIPE, cmds, sizeof(cmds)) < 0)
        throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
    if(usb_bulk_write( (usb_handle*)&m_handle, TFIFO, (uint8_t*) &m_buffer[0], len) < 0)
        throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);

    resetBulkWrite();
}

void
XFX2FWUSBInterface::setLED(void *handle, uint8_t data) {
    uint8_t cmds[] = {CMD_LED, data};
    if(usb_bulk_write( (usb_handle*)&handle, CPIPE, cmds, sizeof(cmds)) < 0)
        throw XInterface::XInterfaceError(i18n_noncontext("USB bulk writing has failed."), __FILE__, __LINE__);
}

uint8_t
XFX2FWUSBInterface::readDIPSW(void *handle) {
    uint8_t cmds[] = {CMD_DIPSW};
    if(usb_bulk_write( (usb_handle*)&handle, CPIPE, cmds, sizeof(cmds)) < 0)
        throw XInterface::XInterfaceError(i18n_noncontext("USB bulk writing has failed."), __FILE__, __LINE__);
    uint8_t buf[10];
    if(usb_bulk_read( (usb_handle*)&handle, RFIFO, buf, 1) != 1)
        throw XInterface::XInterfaceError(i18n_noncontext("USB bulk reading has failed."), __FILE__, __LINE__);
    return buf[0];
}

XString
XFX2FWUSBInterface::getIDN(void *handle, int maxlen, int addroffset) {
    //ignores till \0
    for(int i = 0; ; ++i) {
        char c = singleRead(handle, ADDR_IDN, addroffset);
        if( !c)
            break;
        if(i > 255) {
            return {}; //failed
        }
    }
    XString idn;
    for(int i = 0; ; ++i) {
        char c = singleRead(handle, ADDR_IDN, addroffset);
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
XFX2FWUSBInterface::singleRead(unsigned int addr) {
    XScopedLock<XFX2FWUSBInterface> lock( *this);
    return singleRead(m_handle, addr, m_addrOffset);
}

uint8_t
XFX2FWUSBInterface::singleRead(void *handle, unsigned int addr, unsigned int addroffset) {
    addr += addroffset;
    assert(addr < 0x100u);
    {
        uint8_t cmds[] = {CMD_SWRITE, (uint8_t)(addr)};
        if(usb_bulk_write( (usb_handle*)&handle, CPIPE, cmds, sizeof(cmds)) < 0)
            throw XInterface::XInterfaceError(i18n_noncontext("USB bulk writing has failed."), __FILE__, __LINE__);
    }
    {
        uint8_t cmds[] = {CMD_SREAD};
        if(usb_bulk_write( (usb_handle*)&handle, CPIPE, cmds, sizeof(cmds)) < 0)
            throw XInterface::XInterfaceError(i18n_noncontext("USB bulk writing has failed."), __FILE__, __LINE__);
        uint8_t buf[10];
        if(usb_bulk_read( (usb_handle*)&handle, RFIFO, buf, 1) != 1)
            throw XInterface::XInterfaceError(i18n_noncontext("USB bulk reading has failed."), __FILE__, __LINE__);
        return buf[0];
    }
}
uint16_t
XFX2FWUSBInterface::readRegister16(unsigned int addr) {
    XScopedLock<XFX2FWUSBInterface> lock( *this);
    return singleRead(addr) + singleRead(addr + 1) * (uint16_t)0x100u;
}

void
XFX2FWUSBInterface::burstRead(unsigned int addr, uint8_t *buf, unsigned int cnt) {
    XScopedLock<XFX2FWUSBInterface> lock( *this);
    addr += m_addrOffset;
    assert(addr < 0x100u);
    {
        uint8_t cmds[] = {CMD_SWRITE, (uint8_t)(addr)};
        if(usb_bulk_write( (usb_handle*)&m_handle, CPIPE, cmds, sizeof(cmds)) < 0)
            throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
    }
    const unsigned int blocksize = 512;
    uint8_t cmds[] = {CMD_BREAD, blocksize % 0x100u, blocksize / 0x100u};
    uint8_t bbuf[blocksize];
    for(; cnt;) {
        if(usb_bulk_write( (usb_handle*)&m_handle, CPIPE, cmds, sizeof(cmds)) < 0)
            throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
        int i = usb_bulk_read( (usb_handle*)&m_handle, RFIFO, bbuf, blocksize);
        if(i <= 0)
            throw XInterface::XInterfaceError(i18n("USB bulk reading has failed."), __FILE__, __LINE__);
        unsigned int n = std::min(cnt, (unsigned int)i);
        std::copy(bbuf, bbuf + n, buf);
        buf += n;
        cnt -= n;
    }
}

void
XFX2FWUSBInterface::send(const char *str) throw (XCommError &) {
    XScopedLock<XInterface> lock(*this);
    XScopedLock<XFX2FWUSBInterface> lock2( *this);
    try {
        dbgPrint(driver()->getLabel() + " Sending:\"" + dumpCString(str) + "\"");
        XString buf = str + eos();
        for(int i = 0; i < buf.length(); ++i) {
            writeToRegister8(ADDR_CHARINTF, (uint8_t)buf[i]);
        }
    }
    catch (XCommError &e) {
        e.print(driver()->getLabel() + i18n(" SendError, because "));
        throw;
    }
}
void
XFX2FWUSBInterface::receive() throw (XCommError &) {
    XScopedLock<XInterface> lock(*this);
    XScopedLock<XFX2FWUSBInterface> lock2( *this);
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
    catch (XCommError &e) {
        e.print(driver()->getLabel() + i18n(" ReceiveError, because "));
        throw;
    }
}
