/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
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

#define CUSB_BULK_WRITE_SIZE 40000

#include <windows.h>
extern "C" {
    #include "cusb.h"
    #include "fx2fw.h"
}

#define CMD_DIPSW 0x11
#define CMD_LED 0x12

#define THAMWAY_USB_FIRMWARE_FILE "fx2fw.bix"
#define THAMWAY_USB_GPIFWAVE_FILE "fullspec_dat.bin"
#define THAMWAY_USB_GPIFWAVE_SIZE 172

XMutex XWinCUSBInterface::s_mutex;
int XWinCUSBInterface::s_refcnt = 0;
std::deque<void *> XWinCUSBInterface::s_handles;

void
XWinCUSBInterface::openAllEZUSBdevices() {
    QDir dir(QApplication::applicationDirPath());
    char firmware[CUSB_DWLSIZE];
    {
        QString path = THAMWAY_USB_FIRMWARE_FILE;
        dir.filePath(path);
        if( !dir.exists())
            throw XInterface::XInterfaceError(i18n_noncontext("USB firmware file not found"), __FILE__, __LINE__);
        QFile file(dir.absoluteFilePath(path));
        if( !file.open(QIODevice::ReadOnly))
            throw XInterface::XInterfaceError(i18n_noncontext("USB firmware file is not proper"), __FILE__, __LINE__);
        int size = file.read(firmware, CUSB_DWLSIZE);
        if(size != CUSB_DWLSIZE)
            throw XInterface::XInterfaceError(i18n_noncontext("USB firmware file is not proper"), __FILE__, __LINE__);
    }
    char gpifwave[THAMWAY_USB_GPIFWAVE_SIZE];
    {
        QString path = THAMWAY_USB_GPIFWAVE_FILE;
        dir.filePath(path);
        if( !dir.exists())
            throw XInterface::XInterfaceError(i18n_noncontext("USB GPIF wave file not found"), __FILE__, __LINE__);
        QFile file(dir.absoluteFilePath(path));
        if( !file.open(QIODevice::ReadOnly))
            throw XInterface::XInterfaceError(i18n_noncontext("USB GPIF wave file is not proper"), __FILE__, __LINE__);
        int size = file.read(gpifwave, THAMWAY_USB_GPIFWAVE_SIZE);
        if(size != THAMWAY_USB_GPIFWAVE_SIZE)
            throw XInterface::XInterfaceError(i18n_noncontext("USB GPIF wave file is not proper"), __FILE__, __LINE__);
    }
    for(int i = 0; i < 8; ++i) {
        void *handle = 0;
        fprintf(stderr, "cusb_init #%d\n", i);
        if(cusb_init(i, &handle, (uint8_t *)firmware,
            (signed char*)"F2FW", (signed char*)"20070627")) {
            //no device, or incompatible firmware.
            continue;
        }
        s_handles.push_back(handle);
        uint8_t sw = readDIPSW(handle);
        fprintf(stderr, "Setting GPIF waves for handle 0x%x, DIPSW=%x\n", (unsigned int)handle, (unsigned int)sw);
        setWave(handle, (const uint8_t*)gpifwave);

        for(int i = 0; i < 3; ++i) {
            //blinks LED
            setLED(handle, 0x00u);
            msecsleep(70);
            setLED(handle, 0xf0u);
            msecsleep(60);
        }
    }
    if(s_handles.empty())
        throw XInterface::XInterfaceError(i18n_noncontext("USB-device open has failed."), __FILE__, __LINE__);
}

void
XWinCUSBInterface::setWave(void *handle, const uint8_t *wave) {
    std::vector<uint8_t> buf;
    buf.insert(buf.end(), {CMD_MODE, MODE_GPIF | MODE_8BIT | MODE_ADDR | MODE_NOFLOW | MODE_DEBG, CMD_GPIF});
    buf.insert(buf.end(), wave, wave + 8);
    buf.insert(buf.end(), {MODE_FLOW});
    buf.insert(buf.end(), wave + 8 + 32*4, wave + 8 + 32*4 + 36);
    if(usb_bulk_write( &handle, CPIPE, &buf[0], buf.size()) < 0)
        throw XInterface::XInterfaceError(i18n_noncontext("USB bulk writing has failed."), __FILE__, __LINE__);
    const uint8_t cmdwaves[] = {CMD_WAVE0 /*SingleRead*/, CMD_WAVE1/*SingleWrite*/, CMD_WAVE2/*BurstRead*/, CMD_WAVE3/*BurstWrite*/};
    for(int i = 0; i < sizeof(cmdwaves); ++i) {
        buf.clear();
        buf.insert(buf.end(), cmdwaves + i, cmdwaves + i + 1);
        buf.insert(buf.end(), wave + 8 + 32*i, wave + 8 + 32*(i + 1));
        if(usb_bulk_write( &handle, CPIPE, &buf[0], buf.size()) < 0)
            throw XInterface::XInterfaceError(i18n_noncontext("USB bulk writing has failed."), __FILE__, __LINE__);
    }
}
void
XWinCUSBInterface::closeAllEZUSBdevices() {
    for(auto it = s_handles.begin(); it != s_handles.end();) {
        setLED( *it, 0);

        fprintf(stderr, "cusb_close\n");
        usb_close( &*it);
        ++it;
    }
    s_handles.clear();
}

XWinCUSBInterface::XWinCUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver, uint8_t addr_idn, const char* id)
    : XInterface(name, runtime, driver), m_handle(0), m_idString(id), m_addrIDN(addr_idn) {
    XScopedLock<XMutex> slock(s_mutex);
    try {
        if( !s_refcnt)
            openAllEZUSBdevices();
        s_refcnt++;

        for(Transaction tr( *this);; ++tr) {
            int i = 0;
            for(auto it = s_handles.begin(); it != s_handles.end(); ++it) {
                XString idn = getIDN( *it, 7);
                if(idn.empty()) continue;
                tr[ *device()].add(idn);
                ++i;
            }
            if(tr.commit())
                break;
        }
    }
    catch (XInterface::XInterfaceError &e) {
        e.print();
    }
}

XWinCUSBInterface::~XWinCUSBInterface() {
    if(isOpened()) close();

    XScopedLock<XMutex> slock(s_mutex);
    s_refcnt--;
    if( !s_refcnt)
        closeAllEZUSBdevices();
}

void
XWinCUSBInterface::open() throw (XInterfaceError &) {
    Snapshot shot( *this);
    try {
        int dev = shot[ *device()];
        if((dev < 0) || (dev >= s_handles.size()))
            throw XInterface::XOpenInterfaceError(__FILE__, __LINE__);

        m_handle = s_handles.at(dev);

//        uint8_t sw = readDIPSW() % 8;
//        if(sw != shot[ *address()]) {
//          close();
//          continue; //go to the next device.
//        }
    }
    catch (XInterface::XInterfaceError &e) {
        m_handle = 0;
        throw e;
    }
}

void
XWinCUSBInterface::close() throw (XInterfaceError &) {
    m_handle = 0;
}

void
XWinCUSBInterface::resetBulkWrite() {
    m_bBulkWrite = false;
    m_buffer.clear();
}
void
XWinCUSBInterface::deferWritings() {
    assert(m_buffer.size() == 0);
    m_bBulkWrite = true;
}
void
XWinCUSBInterface::writeToRegister8(unsigned int addr, uint8_t data) {
    if(m_bBulkWrite) {
        if(m_buffer.size() > CUSB_BULK_WRITE_SIZE) {
            bulkWriteStored();
            deferWritings();
        }
        m_buffer.push_back(addr % 0x100u);
        m_buffer.push_back(data);
    }
    else {
        uint8_t cmds[] = {CMD_BWRITE, 2, 0}; //2bytes to be written.
        if(usb_bulk_write( &m_handle, CPIPE, cmds, sizeof(cmds)) < 0)
            throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
        uint8_t cmds2[] = {(uint8_t)(addr % 0x100u), data};
        if(usb_bulk_write( &m_handle, TFIFO, cmds2, sizeof(cmds2)) < 0)
            throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
    }
}
void
XWinCUSBInterface::bulkWriteStored() {
    uint16_t len = m_buffer.size();
    uint8_t cmds[] = {CMD_BWRITE, (uint8_t)(len % 0x100u), (uint8_t)(len / 0x100u)};
    if(usb_bulk_write( &m_handle, CPIPE, cmds, sizeof(cmds)) < 0)
        throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
    if(usb_bulk_write( &m_handle, TFIFO, (uint8_t*) &m_buffer[0], len) < 0)
        throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);

    resetBulkWrite();
}

void
XWinCUSBInterface::setLED(void *handle, uint8_t data) {
    uint8_t cmds[] = {CMD_LED, data};
    if(usb_bulk_write( &handle, CPIPE, cmds, sizeof(cmds)) < 0)
        throw XInterface::XInterfaceError(i18n_noncontext("USB bulk writing has failed."), __FILE__, __LINE__);
}

uint8_t
XWinCUSBInterface::readDIPSW(void *handle) {
    uint8_t cmds[] = {CMD_DIPSW};
    if(usb_bulk_write( &handle, CPIPE, cmds, sizeof(cmds)) < 0)
        throw XInterface::XInterfaceError(i18n_noncontext("USB bulk writing has failed."), __FILE__, __LINE__);
    uint8_t buf[10];
    if(usb_bulk_read( &handle, RFIFO, buf, 1) != 1)
        throw XInterface::XInterfaceError(i18n_noncontext("USB bulk reading has failed."), __FILE__, __LINE__);
    return buf[0];
}

XString
XWinCUSBInterface::getIDN(void *handle, int maxlen, int addr) {
    //ignores till \0
    for(int i = 0; ; ++i) {
        char c = singleRead(handle, addr);
        if( !c)
            break;
        if(i > 255) {
            return XString(); //failed
        }
    }
    XString idn;
    for(int i = 0; ; ++i) {
        char c = singleRead(handle, addr);
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
XWinCUSBInterface::singleRead(void *handle, unsigned int addr) {
    {
        uint8_t cmds[] = {CMD_SWRITE, (uint8_t)(addr % 0x100u)};
        if(usb_bulk_write( &handle, CPIPE, cmds, sizeof(cmds)) < 0)
            throw XInterface::XInterfaceError(i18n_noncontext("USB bulk writing has failed."), __FILE__, __LINE__);
    }
    {
        uint8_t cmds[] = {CMD_SREAD};
        if(usb_bulk_write( &handle, CPIPE, cmds, sizeof(cmds)) < 0)
            throw XInterface::XInterfaceError(i18n_noncontext("USB bulk writing has failed."), __FILE__, __LINE__);
        uint8_t buf[10];
        if(usb_bulk_read( &handle, RFIFO, buf, 1) != 1)
            throw XInterface::XInterfaceError(i18n_noncontext("USB bulk reading has failed."), __FILE__, __LINE__);
        return buf[0];
    }
}
uint16_t
XWinCUSBInterface::readRegister16(unsigned int addr) {
    return singleRead(addr) + singleRead(addr + 1) * (uint16_t)0x100u;
}

void
XWinCUSBInterface::burstRead(unsigned int addr, uint8_t *buf, unsigned int cnt) {
    assert(isLocked());
    {
        uint8_t cmds[] = {CMD_SWRITE, (uint8_t)(addr % 0x100u)};
        if(usb_bulk_write( &m_handle, CPIPE, cmds, sizeof(cmds)) < 0)
            throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
    }
    const unsigned int blocksize = 512;
    uint8_t cmds[] = {CMD_BREAD, blocksize % 0x100u, blocksize / 0x100u};
    uint8_t bbuf[blocksize];
    for(; cnt;) {
        if(usb_bulk_write( &m_handle, CPIPE, cmds, sizeof(cmds)) < 0)
            throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
        if(usb_bulk_read( &m_handle, RFIFO, bbuf, blocksize) != 1)
            throw XInterface::XInterfaceError(i18n("USB bulk reading has failed."), __FILE__, __LINE__);
        unsigned int n = std::min(cnt, blocksize);
        std::copy(bbuf, bbuf + n, buf);
        buf += n;
        cnt -= n;
    }
}

