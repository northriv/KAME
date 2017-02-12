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
#include "thamwaypulser.h"
#include "charinterface.h"

#if defined USE_THAMWAY_USB
    REGISTER_TYPE(XDriverList, ThamwayUSBPulser, "NMR pulser Thamway N210-1026 PG32U40(USB)");
    REGISTER_TYPE(XDriverList, ThamwayUSBPulserWithQAM, "NMR pulser Thamway N210-1026 PG027QAM(USB)");
#endif
REGISTER_TYPE(XDriverList, ThamwayCharPulser, "NMR pulser Thamway N210-1026S/T (GPIB/TCP)");

#define MAX_PATTERN_SIZE 256*1024u
//[ms]
#define TIMER_PERIOD (1.0/(40.0e3))
//[ms]
#define MIN_PULSE_WIDTH 0.0001 //100ns, perhaps 50ns is enough?


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

#define QAM_ADDR_REG_ADDR_L 0x00
#define QAM_ADDR_REG_ADDR_H 0x02
#define QAM_ADDR_REG_CTRL 0x0d
#define QAM_ADDR_REG_DATA_LSW 0x08
#define QAM_ADDR_REG_REP_N 0x0a

double XThamwayPulser::resolution() const {
	return TIMER_PERIOD;
}

double XThamwayPulser::minPulseWidth() const {
	return MIN_PULSE_WIDTH;
}


XThamwayPulser::XThamwayPulser(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XPulser(name, runtime, ref(tr_meas), meas) {

    const int ports[] = {
        XPulser::PORTSEL_GATE, XPulser::PORTSEL_PREGATE, XPulser::PORTSEL_TRIG1, XPulser::PORTSEL_QPSK_OLD_PSGATE,
        XPulser::PORTSEL_QPSK_OLD_INV, XPulser::PORTSEL_GATE, XPulser::PORTSEL_TRIG2, XPulser::PORTSEL_TRIG1,
        XPulser::PORTSEL_PULSE1, XPulser::PORTSEL_ASW, XPulser::PORTSEL_UNSEL, XPulser::PORTSEL_UNSEL,
        XPulser::PORTSEL_UNSEL, XPulser::PORTSEL_UNSEL, XPulser::PORTSEL_UNSEL, XPulser::PORTSEL_UNSEL
    };
	iterate_commit([=](Transaction &tr){
	    for(unsigned int i = 0; i < sizeof(ports)/sizeof(int); i++) {
            tr[ *XPulser::portSel(i)] = ports[i];
		}
    });
}


void
XThamwayPulser::createNativePatterns(Transaction &tr) {
	const Snapshot &shot(tr);
	tr[ *this].m_patterns.clear();
    uint16_t pat = (uint16_t)(shot[ *this].relPatList().back().pattern & XPulser::PAT_DO_MASK);
    pulseAdd(tr, 10, pat); //leading blanks
    pulseAdd(tr, 10, pat);
    uint32_t startaddr = 2;
    for(auto it = shot[ *this].relPatList().begin();
		it != shot[ *this].relPatList().end(); it++) {
		pulseAdd(tr, it->toappear, pat);
        pat = (uint16_t)(it->pattern & XPulser::PAT_DO_MASK);
	}
//    if(tr[ *this].m_patterns.back().term_n_cmd < 2) {
//        pulseAdd(tr, lrint(2 * minPulseWidth() / resolution()), pat);
//    }
    Payload::Pulse p;
    p.term_n_cmd = CMD_JP * 0x10000uL + startaddr;
	p.data = 0;
	tr[ *this].m_patterns.push_back(p);
    p.term_n_cmd = CMD_STOP * 0x10000uL;
    p.data = 0;
    tr[ *this].m_patterns.push_back(p);
}

int
XThamwayPulser::pulseAdd(Transaction &tr, uint64_t term, uint16_t pattern) {
	term = std::max(term, (uint64_t)lrint(MIN_PULSE_WIDTH / TIMER_PERIOD));
	for(; term;) {
        uint64_t t = std::min((uint64_t)term, (uint64_t)0xfe000000uL);
		term -= t;
        Payload::Pulse p;
		p.term_n_cmd = t;
		p.data = pattern;
		tr[ *this].m_patterns.push_back(p);
	}
	return 0;
}

void
XThamwayCharPulser::changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern) {
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

void
XThamwayCharPulser::getStatus(bool *running, bool *extclk_det) {
    if(running) {
        this->interface()->query("ISRUN?");
        *running = (this->interface()->toStrSimplified() == "RUN");
    }
    if(extclk_det)
        *extclk_det = true; //uncertain
}

void
XThamwayCharPulser::open() throw (XKameError &) {
    this->interface()->setEOS("\r\n");
    this->start();
}

#if defined USE_THAMWAY_USB

XThamwayUSBPulserWithQAM::XThamwayUSBPulserWithQAM(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) : XThamwayUSBPulser(name, runtime, ref(tr_meas), meas),
    m_interfaceQAM(XNode::create<XThamwayPGQAMCUSBInterface>("SubInterface", false,
                                                dynamic_pointer_cast<XDriver>(this->shared_from_this()))) {
    meas->interfaces()->insert(tr_meas, m_interfaceQAM);
    m_interfaceQAM->control()->setUIEnabled(false);
}


void
XThamwayUSBPulser::open() throw (XKameError &) {
    XString idn = this->interface()->getIDN();
    fprintf(stderr, "PG IDN=%s\n", idn.c_str());
    this->start();
}

void
XThamwayUSBPulserWithQAM::open() throw (XKameError &) {
    interfaceQAM()->start();
    if(!interfaceQAM()->isOpened())
        throw XInterface::XInterfaceError(i18n("QAM device cannot be configured."), __FILE__, __LINE__);

    XThamwayUSBPulser::open();
}
void
XThamwayUSBPulserWithQAM::close() throw (XKameError &) {
    interfaceQAM()->stop();
    XThamwayUSBPulser::close();
}

void
XThamwayUSBPulser::changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern) {
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
    int addr = 0;
    if(hasQAMPorts()) {
        this->interfaceQAM()->writeToRegister16(QAM_ADDR_REG_ADDR_L, 0);
        this->interfaceQAM()->writeToRegister8(QAM_ADDR_REG_ADDR_H, 0);
    }
    if(output) {
        this->interface()->deferWritings();
        if(hasQAMPorts()) this->interfaceQAM()->deferWritings();

        //lambda for one pulse
        auto addPulse = [&](uint32_t term, uint32_t data, uint16_t iq) {
            this->interface()->writeToRegister16(ADDR_REG_TIME_LSW, term % 0x10000uL);
            this->interface()->writeToRegister16(ADDR_REG_TIME_MSW, term / 0x10000uL);
            this->interface()->writeToRegister16(ADDR_REG_DATA_LSW, data % 0x10000uL);
            this->interface()->writeToRegister16(ADDR_REG_DATA_MSW, data / 0x10000uL);
            this->interface()->writeToRegister8(ADDR_REG_CTRL, 2); //addr++
            if(hasQAMPorts()) {
                this->interfaceQAM()->writeToRegister16(QAM_ADDR_REG_DATA_LSW, iq);
                this->interfaceQAM()->writeToRegister8(QAM_ADDR_REG_CTRL, 2); //addr++
            }
            addr += 2;
            if(addr >= MAX_PATTERN_SIZE) {
                throw XInterface::XInterfaceError(i18n("Number of patterns exceeded the size limit."), __FILE__, __LINE__);
            }
        };
        //lambda to determine IQ levels
        auto qamIQ = [&](const std::complex<double> &c) -> uint16_t {
            auto offset = std::complex<double>{ shot[ *qamOffset1()], shot[ *qamOffset2()]};
            auto z = 127.0 * (c  + offset);
            if(std::abs(z) > 127) z /= std::abs(z) / 127;
            return 256 * lrint(std::real(z * (double)shot[ *qamLevel1()])) + lrint(std::imag(z * (double)shot[ *qamLevel2()])); //I * 256 + Q, abs(z) <= 127.
        };
        auto qam0 = qamIQ(0.0); //zero levels

        for(auto it = shot[ *this].m_patterns.begin(); it != shot[ *this].m_patterns.end(); ++it) {
            unsigned int pnum = (it->data & PAT_QAM_PULSE_IDX_MASK)/PAT_QAM_PULSE_IDX;
            if( !hasQAMPorts() || (pnum == 0)) {
                addPulse(it->term_n_cmd, it->data, qam0);
            }
            else {
                unsigned int phase = (it->data & PAT_QAM_PHASE_MASK)/PAT_QAM_PHASE;
                auto z = std::polar(pow(10.0, shot[ *this].masterLevel() / 20.0), M_PI / 2.0 * phase);
                auto &wave = shot[ *this].qamWaveForm(pnum);
                int idx = 0;
                for(int64_t t = it->term_n_cmd; t > 0; t -= QAM_PERIOD) {
                    uint16_t iq = qamIQ(z * wave[idx]);
                    addPulse(std::min(t, (int64_t)QAM_PERIOD), it->data, iq);
                    idx++;
                }
            }
        }

        this->interface()->bulkWriteStored();
        this->interface()->writeToRegister8(ADDR_REG_STS, 0); //clears STS.
        this->interface()->writeToRegister16(ADDR_REG_REP_N, 0); //infinite loops
        this->interface()->writeToRegister16(ADDR_REG_ADDR_L, 0);
        this->interface()->writeToRegister8(ADDR_REG_ADDR_H, 0);
        bool ext_clock = false;
//        getStatus(0, &ext_clock); //PROT does not use ext. clock.
        this->interface()->writeToRegister8(ADDR_REG_MODE, 8 | (ext_clock ? 4 : 0)); //external Trig
        this->interface()->writeToRegister8(ADDR_REG_CTRL, 1); //starts it

        if(hasQAMPorts()) {
            this->interfaceQAM()->bulkWriteStored();
            this->interfaceQAM()->writeToRegister16(QAM_ADDR_REG_REP_N, 0); //infinite loops
        }
    }
}


void
XThamwayUSBPulser::getStatus(bool *running, bool *extclk_det) {
    uint8_t sta = this->interface()->singleRead(ADDR_REG_STS);
    if(running)
        *running = sta & 0x2;
    if(extclk_det)
        *extclk_det = !(sta & 0x20);
    this->interface()->writeToRegister8(ADDR_REG_STS, 0);
}


#endif //USE_THAMWAY_USB
