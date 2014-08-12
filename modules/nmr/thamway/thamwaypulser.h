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
#include "pulserdriver.h"
#include "chardevicedriver.h"
#include <vector>

//! pulser driver
template<class tInterface>
class XThamwayPulser : public XCharDeviceDriver<XPulser, tInterface> {
public:
	XThamwayPulser(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XThamwayPulser() {}

	struct Payload : public XPulser::Payload {
	private:
		friend class XThamwayPulser;
		struct Pulse {
			uint32_t term_n_cmd;
			uint32_t data;
		};
		std::deque<Pulse> m_patterns;
	};

	//! time resolution [ms]
    virtual double resolution() const;
protected:
    virtual void open() throw (XKameError &);
    //! Sends patterns to pulser or turns off.
    virtual void changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern);
    //! Converts RelPatList to native patterns
    virtual void createNativePatterns(Transaction &tr);
    virtual double resolutionQAM() const {return 0.0;}
    //! minimum period of pulses [ms]
    virtual double minPulseWidth() const;
    //! existense of AO ports.
    virtual bool hasQAMPorts() const {return false;}
private:
	//! Add 1 pulse pattern
	//! \param term a period of the pattern to appear
	//! \param pattern a pattern for digital, to appear
	int pulseAdd(Transaction &tr, uint64_t term, uint16_t pattern);
};

//! interfaces chameleon USB, found at http://optimize.ath.cx/cusb
class XWinCUSBInterface : public XInterface {
public:
    XWinCUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
        : XInterface(name, runtime, driver), m_handle(0)
    {}
    virtual ~XWinCUSBInterface() {
        if(isOpened()) close();
    }

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

    XString getIDN();
protected:
    void setLED(uint8_t data);
    uint8_t readDIPSW();
    void setWave(const uint8_t *wave);
    uint8_t singleRead(unsigned int addr);
private:
    void* m_handle;
    bool m_bBulkWrite;
    std::deque<uint8_t> m_buffer;
};

typedef XThamwayPulser<XWinCUSBInterface> XThamwayUSBPulser;
