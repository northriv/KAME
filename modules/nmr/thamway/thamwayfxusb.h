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
#include "cyfxusb.h"

#define DEV_ADDR_PROT 0x6

class XThamwayFX2LPUSBInterface : public XCyFXUSBInterface<THAMWAY_DEVID, CYPRESS_GUID, 0> {
public:
    XThamwayFX2LPUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver, uint8_t addr_offset, const char* id);
    virtual ~XThamwayFX2LPUSBInterface();

    void deferWritings();
    void writeToRegister8(unsigned int addr, uint8_t data);
    void writeToRegister16(unsigned int addr, uint16_t data);
    void bulkWriteStored();
    void resetBulkWrite();

    void burstRead(unsigned int addr, uint8_t *buf, unsigned int cnt);
    uint8_t singleRead(unsigned int addr);
    uint16_t readRegister8(unsigned int addr) {return singleRead(addr);}
    uint16_t readRegister16(unsigned int addr);

    virtual void send(const char *str) throw (XCommError &) override;
    virtual void receive() throw (XCommError &) override;

    XString getIDN(int maxlen = 255) {return getIDN(m_handle, maxlen); }

protected:
    virtual bool examineDeviceBeforeFWLoad(void *handle, std::vector &fw);
    virtual std::string examineDeviceAfterFWLoad(void *handle);
    virtual std::vector gpifWave();
    std::vector gpibWave();
    static uint8_t singleRead(void *handle, unsigned int addr, unsigned int addroffset);
    uint8_t m_addrOffset;
private:
    XString getIDN(void *handle, int maxlen = 255) {
        XString str = getIDN(handle, maxlen, m_addrOffset);
        if(str.empty() || (str.find(m_idString,0) != 0))
             return {};
        return str;
    }
    static XString getIDN(void *handle, int maxlen, int offsetaddr);
    static void setLED(void *handle, uint8_t data);
    static uint8_t readDIPSW(void *handle);
    XString m_idString;
    bool m_bBurstWrite;
    std::vector<uint8_t> m_buffer; //writing buffer for a burst write.
};
