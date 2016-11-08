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
#include "chardevicedriver.h"
#include "charinterface.h"
#include <vector>

#define DEV_ADDR_PROT 0x6

//! interfaces chameleon USB, found at http://optimize.ath.cx/cusb
class XFX2FWUSBInterface : public XCustomCharInterface {
public:
    XFX2FWUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver, uint8_t addr_offset, const char* id);
    virtual ~XFX2FWUSBInterface();

    virtual void open() throw (XInterfaceError &);
    //! This can be called even if has already closed.
    virtual void close() throw (XInterfaceError &);

    void deferWritings();
    void writeToRegister8(unsigned int addr, uint8_t data);
    void writeToRegister16(unsigned int addr, uint16_t data);
    void bulkWriteStored();
    void resetBulkWrite();

    void burstRead(unsigned int addr, uint8_t *buf, unsigned int cnt);
    uint8_t singleRead(unsigned int addr);
    uint16_t readRegister8(unsigned int addr) {return singleRead(addr);}
    uint16_t readRegister16(unsigned int addr);

    XString getIDN(int maxlen = 255) {return getIDN(m_handle, maxlen); }

    void lock() {m_mutex->lock();} //!<overrides XInterface::lock().
    void unlock() {m_mutex->unlock();}
    bool isLocked() const {return m_mutex->isLockedByCurrentThread();}

    virtual void send(const char *str) throw (XCommError &);
    virtual void receive() throw (XCommError &);

    virtual bool isOpened() const {return m_handle != 0;}
protected:
private:
    XString getIDN(void *handle, int maxlen = 255) {
        XString str = getIDN(handle, maxlen, m_addrOffset);
        if(str.empty() || (str.find(m_idString,0) != 0))
             return {};
        return str;
    }
    static void setLED(void *handle, uint8_t data);
    static uint8_t readDIPSW(void *handle);
    static uint8_t singleRead(void *handle, unsigned int addr, unsigned int addroffset);
    static XMutex s_mutex;
    struct USBDevice {
        shared_ptr<XRecursiveMutex> mutex;
        void *handle;
        int addr;
    };
    static std::deque<USBDevice> s_devices;
    static int s_refcnt;
    static void openAllEZUSBdevices();
    static void setWave(void *handle, const uint8_t *wave);
    static void closeAllEZUSBdevices();
    static XString getIDN(void *handle, int maxlen, int offsetaddr);
    void* m_handle;
    shared_ptr<XRecursiveMutex> m_mutex;
    XString m_idString;
    uint8_t m_addrOffset;
    bool m_bBurstWrite;
    std::vector<uint8_t> m_buffer; //writing buffer for a burst write.
};
