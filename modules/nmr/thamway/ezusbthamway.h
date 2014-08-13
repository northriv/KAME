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
#include "chardevicedriver.h"
#include <vector>

//! interfaces chameleon USB, found at http://optimize.ath.cx/cusb
class XWinCUSBInterface : public XInterface {
public:
    XWinCUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
    virtual ~XWinCUSBInterface();

    virtual void open() throw (XInterfaceError &);
    //! This can be called even if has already closed.
    virtual void close() throw (XInterfaceError &);

    virtual bool isOpened() const {return m_handle != 0;}

    void deferWritings();
    void writeToRegister8(unsigned int addr, uint8_t data);
    void writeToRegister16(unsigned int addr, uint16_t data) {
        writeToRegister8(addr, data % 0x100u);
        writeToRegister8(addr + 1, data / 0x100u);
    }
    void bulkWriteStored();
    void resetBulkWrite();

    void burstRead(unsigned int addr, uint8_t *buf, unsigned int cnt);

    XString getIDN() {return getIDN(m_handle);}
protected:
    void setLED(uint8_t data);
    uint8_t readDIPSW();
private:
    static uint8_t singleRead(void *handle, unsigned int addr);
    static XMutex s_mutex;
    static std::deque<void *> s_handles;
    static int s_refcnt;
    static void openAllEZUSBdevices();
    static void setWave(void *handle, const uint8_t *wave);
    static void closeAllEZUSBdevices();
    static XString getIDN(void *handle);
    void* m_handle;
    bool m_bBulkWrite;
    std::deque<uint8_t> m_buffer;
};
