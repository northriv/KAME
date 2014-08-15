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
        int size = file.read(firmware, THAMWAY_USB_GPIFWAVE_SIZE);
        if(size != THAMWAY_USB_GPIFWAVE_SIZE)
            throw XInterface::XInterfaceError(i18n_noncontext("USB GPIF wave file is not proper"), __FILE__, __LINE__);
    }
    for(int i = 0; i < 8; ++i) {
        void *handle = 0;
        fprintf(stderr, "cusb_init #%d\n", i);
        //finds out the first available device.
        if(cusb_init(i, &handle, (uint8_t *)firmware,
            (signed char*)"F2FW", (signed char*)"20070627")) {
            //no device, or incompatible firmware.
            continue;
        }
        fprintf(stderr, "Setting GPIF waves for handle 0x%x\n", (unsigned int)handle);
        setWave(handle, (const uint8_t*)gpifwave);
        s_handles.push_back(handle);
    }
    if(s_handles.empty())
        throw XInterface::XInterfaceError(i18n_noncontext("USB-device open has failed."), __FILE__, __LINE__);
}

void
XWinCUSBInterface::setWave(void *handle, const uint8_t *wave) {
    const uint8_t cmd[] = {CMD_MODE, MODE_GPIF | MODE_8BIT | MODE_ADDR | MODE_NOFLOW | MODE_DEBG, CMD_GPIF};
    std::vector<uint8_t> buf;
    buf.insert(buf.end(), cmd, cmd + sizeof(cmd));
    buf.insert(buf.end(), wave, wave + 8);
    const uint8_t cmd2[] = {MODE_FLOW};
    buf.insert(buf.end(), cmd2, cmd2 + sizeof(cmd2));
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
    for(auto it = s_handles.begin(); it != s_handles.end(); ++it) {
        fprintf(stderr, "cusb_close\n");
        usb_close( &*it);
    }
    s_handles.clear();
}

XWinCUSBInterface::XWinCUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
    : XInterface(name, runtime, driver), m_handle(0) {
    XScopedLock<XMutex> slock(s_mutex);
    try {
        if( !s_refcnt)
            openAllEZUSBdevices();
        s_refcnt++;

        for(Transaction tr( *this);; ++tr) {
            for(auto it = s_handles.begin(); it != s_handles.end(); ++it) {
                tr[ *device()].add(getIDN( *it).substr(0,7));
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

        for(int i = 0; i < 3; ++i) {
            //blinks LED
            setLED(0x00u);
            msecsleep(70);
            setLED(0xf0u);
            msecsleep(60);
        }
    }
    catch (XInterface::XInterfaceError &e) {
        m_handle = 0;
        throw e;
    }
}

void
XWinCUSBInterface::close() throw (XInterfaceError &) {
    if(m_handle) {
        try {
            setLED(0);
        }
        catch (XInterface::XInterfaceError &e) {
            e.print();
        }
    }
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
        uint16_t len = m_buffer.size();
        uint8_t cmds[] = {CMD_BWRITE, 2, 0}; //2bytes to be written.
        if(usb_bulk_write( &m_handle, CPIPE, cmds, sizeof(cmds)) < 0)
            throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
        uint8_t cmds2[] = {addr % 0x100u, data};
        if(usb_bulk_write( &m_handle, TFIFO, cmds2, sizeof(cmds2)) < 0)
            throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
    }
}
void
XWinCUSBInterface::bulkWriteStored() {
    uint16_t len = m_buffer.size();
    uint8_t cmds[] = {CMD_BWRITE, len % 0x100u, len / 0x100u};
    if(usb_bulk_write( &m_handle, CPIPE, cmds, sizeof(cmds)) < 0)
        throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
    if(usb_bulk_write( &m_handle, TFIFO, (uint8_t*) &m_buffer[0], len) < 0)
        throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);

    resetBulkWrite();
}

void
XWinCUSBInterface::setLED(uint8_t data) {
    uint8_t cmds[] = {CMD_LED, data};
    if(usb_bulk_write( &m_handle, CPIPE, cmds, sizeof(cmds)) < 0)
        throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
}

uint8_t
XWinCUSBInterface::readDIPSW() {
    uint8_t cmds[] = {CMD_DIPSW};
    if(usb_bulk_write( &m_handle, CPIPE, cmds, sizeof(cmds)) < 0)
        throw XInterface::XInterfaceError(i18n("USB bulk writing has failed."), __FILE__, __LINE__);
    uint8_t buf[10];
    if(usb_bulk_read( &m_handle, RFIFO, buf, 1) != 1)
        throw XInterface::XInterfaceError(i18n("USB bulk reading has failed."), __FILE__, __LINE__);
    return buf[0];
}

XString
XWinCUSBInterface::getIDN(void *handle) {
    //ignores till \0
    for(int i = 0; ; ++i) {
        if( !singleRead(handle, 0x1f))
            break;
        if(i > 256) {
            throw XInterface::XInterfaceError(i18n_noncontext("USB getting IDN has failed."), __FILE__, __LINE__);
        }
    }
    XString idn;
    for(int i = 0; ; ++i) {
        char c = singleRead(handle, 0x1f);
        idn += c;
        if( !c)
            break;
        if(i > 256) {
            throw XInterface::XInterfaceError(i18n_noncontext("USB getting IDN has failed."), __FILE__, __LINE__);
        }
    }
    return idn;
}
uint8_t
XWinCUSBInterface::singleRead(void *handle, unsigned int addr) {
    {
        uint8_t cmds[] = {CMD_SWRITE, addr % 0x100};
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
void
XWinCUSBInterface::burstRead(unsigned int addr, uint8_t *buf, unsigned int cnt) {
    assert(isLocked());
    {
        uint8_t cmds[] = {CMD_SWRITE, addr % 0x100u};
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

