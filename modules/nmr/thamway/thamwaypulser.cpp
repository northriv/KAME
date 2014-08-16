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
#include "thamwaypulser.h"
#include "charinterface.h"

#if defined USE_EZUSB
    #include "ezusbthamway.h"
    typedef XThamwayPulser<XThamwayPGCUSBInterface> XThamwayUSBPulser;
    REGISTER_TYPE(XDriverList, ThamwayUSBPulser, "NMR pulser Thamway N210-1026 PG32U40(USB)");
#endif
typedef XThamwayPulser<XCharInterface> XThamwayCharPulser;
REGISTER_TYPE(XDriverList, ThamwayCharPulser, "NMR pulser Thamway N210-1026S/T (GPIB/TCP)");

#define MAX_PATTERN_SIZE 256*1024u
//[ms]
#define TIMER_PERIOD (1.0/(40.0e3))
//[ms]
#define MIN_PULSE_WIDTH 0.0002


#define ADDR_REG_ADDR_L 0x00
#define ADDR_REG_ADDR_H 0x02
#define ADDR_REG_STS 0x03
#define ADDR_REG_TIME_LSW 0x04
#define ADDR_REG_TIME_MSW 0x06
#define ADDR_REG_DATA_LSW 0x08
#define ADDR_REG_REP_N 0x0a
#define ADDR_REG_MODE 0x0c
#define ADDR_REG_CTRL 0x0d
#define ADDR_REG_DATA_MSW 0x0e
#define CMD_TRG 0xff80uL
#define CMD_STOP 0xff40uL
#define CMD_JP 0xff20uL
#define CMD_INT 0xff10uL

template<class tInterface>
double XThamwayPulser<tInterface>::resolution() const {
	return TIMER_PERIOD;
}
template<class tInterface>
double XThamwayPulser<tInterface>::minPulseWidth() const {
	return MIN_PULSE_WIDTH;
}

template<class tInterface>
XThamwayPulser<tInterface>::XThamwayPulser(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XPulser, tInterface>(name, runtime, ref(tr_meas), meas) {

    const int ports[] = {
        XPulser::PORTSEL_GATE, XPulser::PORTSEL_PREGATE, XPulser::PORTSEL_TRIG1, XPulser::PORTSEL_TRIG2,
        XPulser::PORTSEL_QPSK_A, XPulser::PORTSEL_QPSK_B, XPulser::PORTSEL_ASW, XPulser::PORTSEL_UNSEL,
        XPulser::PORTSEL_PULSE1, XPulser::PORTSEL_PULSE2, XPulser::PORTSEL_COMB, XPulser::PORTSEL_UNSEL,
        XPulser::PORTSEL_QPSK_OLD_PSGATE, XPulser::PORTSEL_QPSK_OLD_NONINV, XPulser::PORTSEL_QPSK_OLD_INV, XPulser::PORTSEL_COMB_FM
    };
	for(Transaction tr( *this);; ++tr) {
	    for(unsigned int i = 0; i < sizeof(ports)/sizeof(int); i++) {
            tr[ *XPulser::portSel(i)] = ports[i];
		}
		if(tr.commit())
			break;
	}
}

template<class tInterface>
void
XThamwayPulser<tInterface>::createNativePatterns(Transaction &tr) {
	const Snapshot &shot(tr);
	tr[ *this].m_patterns.clear();
    uint16_t pat = (uint16_t)(shot[ *this].relPatList().back().pattern & XPulser::PAT_DO_MASK);
    pulseAdd(tr, 10, pat); //leading blanks
    pulseAdd(tr, 10, pat);
    unsigned int startaddr = 2;
    for(typename Payload::RelPatList::const_iterator it = shot[ *this].relPatList().begin();
		it != shot[ *this].relPatList().end(); it++) {
		pulseAdd(tr, it->toappear, pat);
        pat = (uint16_t)(it->pattern & XPulser::PAT_DO_MASK);
	}
    typename Payload::Pulse p;
    p.term_n_cmd = CMD_JP * 0x10000uL + startaddr;
	p.data = 0;
	tr[ *this].m_patterns.push_back(p);
    p.term_n_cmd = CMD_STOP * 0x10000uL;
    p.data = 0;
    tr[ *this].m_patterns.push_back(p);
}
template<class tInterface>
int
XThamwayPulser<tInterface>::pulseAdd(Transaction &tr, uint64_t term, uint16_t pattern) {
	term = std::max(term, (uint64_t)lrint(MIN_PULSE_WIDTH / TIMER_PERIOD));
	for(; term;) {
        uint32_t t = std::min((unsigned long)term, 0xfe000000uL);
		term -= t;
        typename Payload::Pulse p;
		p.term_n_cmd = t;
		p.data = pattern;
		tr[ *this].m_patterns.push_back(p);
	}
	return 0;
}
template<class tInterface>
void
XThamwayPulser<tInterface>::changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern) {
    XScopedLock<XInterface> lock( *this->interface());
    if( !this->interface()->isOpened())
        return;

    this->interface()->resetBulkWrite();
    this->interface()->writeToRegister8(ADDR_REG_CTRL, 0); //stops it
    this->interface()->writeToRegister8(ADDR_REG_MODE, 2); //direct output on.
    this->interface()->writeToRegister16(ADDR_REG_DATA_LSW, blankpattern % 0x10000uL);
    this->interface()->writeToRegister16(ADDR_REG_DATA_MSW, blankpattern / 0x10000uL);
    this->interface()->writeToRegister8(ADDR_REG_MODE, 0); //direct output off.
    this->interface()->writeToRegister16(ADDR_REG_ADDR_L, 0);
    this->interface()->writeToRegister8(ADDR_REG_ADDR_H, 0);
	if(output) {
        if(shot[ *this].m_patterns.size() >= MAX_PATTERN_SIZE) {
            throw XInterface::XInterfaceError(i18n("Number of patterns exceeded the size limit."), __FILE__, __LINE__);
        }
        this->interface()->deferWritings();
        for(auto it = shot[ *this].m_patterns.begin(); it != shot[ *this].m_patterns.end(); ++it) {
            this->interface()->writeToRegister16(ADDR_REG_TIME_LSW, it->term_n_cmd % 0x10000uL);
            this->interface()->writeToRegister16(ADDR_REG_TIME_MSW, it->term_n_cmd / 0x10000uL);
            this->interface()->writeToRegister16(ADDR_REG_DATA_LSW, it->data % 0x10000uL);
            this->interface()->writeToRegister16(ADDR_REG_DATA_MSW, it->data / 0x10000uL);
            this->interface()->writeToRegister8(ADDR_REG_CTRL, 2); //addr++
		}
        this->interface()->bulkWriteStored();

        this->interface()->writeToRegister8(ADDR_REG_STS, 0); //clears STS.
        this->interface()->writeToRegister8(ADDR_REG_REP_N, 0); //infinite loops
        this->interface()->writeToRegister16(ADDR_REG_ADDR_L, 0);
        this->interface()->writeToRegister8(ADDR_REG_ADDR_H, 0);
        bool ext_clock;
        getStatus(0, &ext_clock);
        this->interface()->writeToRegister8(ADDR_REG_MODE, 8 | (ext_clock ? 4 : 0)); //external Trig
        this->interface()->writeToRegister8(ADDR_REG_CTRL, 1); //starts it
    }
}

template<>
void
XThamwayPulser<XCharInterface>::changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern) {
    XScopedLock<XInterface> lock( *this->interface());
    if( !this->interface()->isOpened())
        return;

    this->interface()->send("STOP");
    this->interface()->send("SAVER 0"); //not needed for TCP/IP version.
    this->interface()->send("SETMODE 1"); //extended mode
    this->interface()->send("MEMCLR 0");
    if(output) {
        if(shot[ *this].m_patterns.size() >= MAX_PATTERN_SIZE) {
            throw XInterface::XInterfaceError(i18n("Number of patterns exceeded the size limit."), __FILE__, __LINE__);
        }

        unsigned int addr = 0;
        for(auto it = shot[ *this].m_patterns.begin(); it != shot[ *this].m_patterns.end(); ++it) {
            this->interface()->sendf("POKE 0x%x,0x%x,0x%x", addr, it->term_n_cmd, it->data);
            addr++;
        }
        this->interface()->send("START 0"); //infinite loops
    }
}

template<class tInterface>
void
XThamwayPulser<tInterface>::open() throw (XKameError &) {
    XString idn = this->interface()->getIDN();
    gWarnPrint("Pulser IDN=" + idn);
    this->start();
}

template<>
void
XThamwayPulser<XCharInterface>::open() throw (XKameError &) {
    this->interface()->setEOS("\r\n");
    this->start();
}

template<class tInterface>
void
XThamwayPulser<tInterface>::getStatus(bool *running, bool *extclk_det) {
    uint8_t sta = this->interface()->singleRead(ADDR_REG_STS);
    if(running)
        *running = sta & 0x2;
    if(extclk_det)
        *extclk_det = sta & 0x20;
    this->interface()->writeToRegister8(ADDR_REG_STS, 0);
}
template<>
void
XThamwayPulser<XCharInterface>::getStatus(bool *running, bool *extclk_det) {
    if(running) {
        this->interface()->query("ISRUN?");
        *running = (this->interface()->toStrSimplified() == "RUN");
    }
    if(extclk_det)
        *extclk_det = true; //uncertain
}


#if defined USE_EZUSB
    template class XThamwayPulser<class XWinCUSBInterface>;
#endif
template class XThamwayPulser<class XCharInterface>;
