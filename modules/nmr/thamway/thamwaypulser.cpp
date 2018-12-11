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

constexpr size_t MAX_PATTERN_SIZE = 256*1024u;
constexpr size_t MAX_QAM_PATTERN_SIZE = 32*8192u;
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
#define CMD_MASK 0xfff00000uL

#define QAM_ADDR_REG_ADDR_L 0x00
#define QAM_ADDR_REG_ADDR_H 0x02
#define QAM_ADDR_REG_CTRL 0x0d
#define QAM_ADDR_REG_DATA_LSW 0x08
#define QAM_ADDR_REG_REP_N 0x0a

#define PAT_TX1 1

#define CHAR_TIMER_PERIOD (1.0/(40.0e3))

constexpr unsigned int LEADING_BLANKS = 40;

double XThamwayPulser::resolution() const {
    return CHAR_TIMER_PERIOD; //25MSPS
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
        tr[ *masterLevel()] = -3.0;
        tr[ *aswSetup()] = 0.2;
    });
}

XThamwayUSBPulser::XThamwayUSBPulser(const char *name, bool runtime, Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
 : XCharDeviceDriver<XThamwayPulser, XThamwayPGCUSBInterface>(name, runtime, ref(tr_meas), meas) {

    //!\todo this may cause nested transaction via onSoftTrigChanged().
    m_softwareTrigger = XThamwayFX3USBInterface::softwareTriggerManager().create(name, NUM_DO_PORTS);

    setPrefillingSampsBeforeArm(LEADING_BLANKS);
}
XThamwayUSBPulser::~XThamwayUSBPulser() {
    XThamwayFX3USBInterface::softwareTriggerManager().unregister(m_softwareTrigger);
}

void
XThamwayPulser::createNativePatterns(Transaction &tr) {
	const Snapshot &shot(tr);
	tr[ *this].m_patterns.clear();
    auto pat = shot[ *this].relPatList().back().pattern;
    pulseAdd(tr, LEADING_BLANKS / 2, pat); //leading blanks
    pulseAdd(tr, LEADING_BLANKS / 2, pat);
    uint32_t startaddr = 2;
    for(auto it = shot[ *this].relPatList().begin();
		it != shot[ *this].relPatList().end(); it++) {
		pulseAdd(tr, it->toappear, pat);
        pat = it->pattern;
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
XThamwayPulser::pulseAdd(Transaction &tr, uint64_t term, uint32_t pattern) {
    term = std::max(term, (uint64_t)lrint(MIN_PULSE_WIDTH / CHAR_TIMER_PERIOD));
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
XThamwayCharPulser::changeOutput(const Snapshot &shot, bool output, unsigned int) {
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
            this->interface()->sendf("POKE 0x%x,0x%x,0x%x", addr,
                   it->term_n_cmd, it->data & XPulser::PAT_DO_MASK);
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
    m_interfaceQAM->control()->disable();
}


void
XThamwayUSBPulser::open() throw (XKameError &) {
    XString idn = this->interface()->getIDN();
    fprintf(stderr, "PG IDN=%s\n", idn.c_str());
    auto pos = idn.find("CLK=");
    if(pos == std::string::npos)
        throw XInterface::XConvError(__FILE__, __LINE__);
    int clk;
    if(sscanf(idn.c_str() + pos, "CLK=%dMHZ", &clk) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    m_resolution = 1e-3 / clk;
    this->start();
}

void
XThamwayUSBPulserWithQAM::open() throw (XKameError &) {
    interfaceQAM()->start();
    XString idn = this->interfaceQAM()->getIDN();
    gMessagePrint(formatString_tr(I18N_NOOP("%s successfully opened\n"), idn.c_str()));
    auto pos = idn.find("SPS=");
    if(pos == std::string::npos)
        throw XInterface::XConvError(__FILE__, __LINE__);
    int clk;
    if(sscanf(idn.c_str() + pos, "SPS=%dMHZ", &clk) != 1)
        throw XInterface::XConvError(__FILE__, __LINE__);
    if(!interfaceQAM()->isOpened())
        throw XInterface::XInterfaceError(i18n("QAM device cannot be configured."), __FILE__, __LINE__);

    XThamwayUSBPulser::open();

    m_qamPeriod = 1e-3 / clk / resolution();
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

    fprintf(stderr, "Pulser stopping.\n");
    //mimics PULBOAD.BAS:StopBrd(0)
    bool ext_clock = false;
    //        getStatus(0, &ext_clock); //PROT does not use ext. clock.
    this->interface()->writeToRegister8(ADDR_REG_CTRL, 0); //stops it
    {
        XThamwayFX2USBInterface::ScopedBulkWriter writer(this->interface());
        XThamwayFX2USBInterface::ScopedBulkWriter writerQAM(this->interfaceQAM());
        this->interface()->writeToRegister8(ADDR_REG_MODE, 2 | (ext_clock ? 4 : 0)); //direct output on.
        this->interface()->writeToRegister16(ADDR_REG_DATA_LSW, blankpattern % 0x10000uL);
        this->interface()->writeToRegister16(ADDR_REG_DATA_MSW, blankpattern / 0x10000uL);
        this->interface()->writeToRegister8(ADDR_REG_MODE, 0 | (ext_clock ? 4 : 0)); //direct output off.

        //mimics PULBOAD.BAS:TransBrd2Mem(0)
        this->interface()->writeToRegister16(ADDR_REG_ADDR_L, 0);
        this->interface()->writeToRegister8(ADDR_REG_ADDR_H, 0);
        if(hasQAMPorts()) {
            this->interfaceQAM()->writeToRegister16(QAM_ADDR_REG_ADDR_L, 0);
            this->interfaceQAM()->writeToRegister8(QAM_ADDR_REG_ADDR_H, 0);
            writerQAM.flush();
        }
        writer.flush();
    }
    size_t addr = 0, addr_qam = 0;
    bool is_saturated = false;
    if(output) {
        {
            fprintf(stderr, "Pulser start");
            XThamwayFX2USBInterface::ScopedBulkWriter writer(this->interface());
            XThamwayFX2USBInterface::ScopedBulkWriter writerQAM(this->interfaceQAM());

            //lambda for one pulse
            auto addPulse = [&](uint32_t term, uint32_t data) {
                this->interface()->writeToRegister16(ADDR_REG_TIME_LSW, term % 0x10000uL);
                this->interface()->writeToRegister16(ADDR_REG_TIME_MSW, term / 0x10000uL);
                this->interface()->writeToRegister16(ADDR_REG_DATA_LSW, data % 0x10000uL);
                this->interface()->writeToRegister16(ADDR_REG_DATA_MSW, data / 0x10000uL);
                this->interface()->writeToRegister8(ADDR_REG_CTRL, 2); //addr++
                addr += 2;
                if(addr >= MAX_PATTERN_SIZE) {
                    throw XInterface::XInterfaceError(i18n("Number of patterns exceeded the size limit."), __FILE__, __LINE__);
                }
            };
            auto qam_offset = std::complex<double>{ shot[ *qamOffset1()], shot[ *qamOffset2()]};
            double qam_lvl1 = shot[ *qamLevel1()], qam_lvl2 = shot[ *qamLevel2()];
            //lambda to determine IQ levels
            auto qamIQ = [&](const std::complex<double> &c) -> uint16_t {
                auto z = 127.0 * (c  + qam_offset);
                auto x = std::real(z) * qam_lvl1;
                auto y = std::imag(z) * qam_lvl2;
                if(std::norm(z) > 127.0 * 127.0) {
                    is_saturated = true;
                    x *= 127.0 / std::abs(z);
                    y *= 127.0 / std::abs(z);
                }
                uint8_t i = lrint(x), q = lrint(y);
                return 0x100u * i + q; //I * 256 + Q, abs(z) <= 127 / sqrt(2)?.
            };
            unsigned int pnum_prev = 0xffffu;
            unsigned int qamidx = 0; //idx in one pulse waveform.
            auto qamz = std::polar(0.0, 0.0); //intens. during a decimation.
            for(auto &&pat: shot[ *this].m_patterns) {
                addPulse(pat.term_n_cmd, pat.data & PAT_DO_MASK);
                if(hasQAMPorts()) {
                    unsigned int pnum = (pat.data & PAT_QAM_PULSE_IDX_MASK)/PAT_QAM_PULSE_IDX;
                    if(pnum && !(pat.term_n_cmd & CMD_MASK)) {
                        if(pnum != pnum_prev) {
                            qamidx = 0; //waveform has been changed.
                            qamz = 0.0;
                        }
                        unsigned int phase = (pat.data & PAT_QAM_PHASE_MASK)/PAT_QAM_PHASE;
                        auto z0 = std::polar(pow(10.0, shot[ *this].masterLevel() / 20.0), M_PI / 2.0 * phase);
                        z0 /= m_qamPeriod;
                        auto &wave = shot[ *this].qamWaveForm(pnum - 1);
                        assert(wave.size() >= qamidx + pat.term_n_cmd);
                        for(int i = 0; i < pat.term_n_cmd; ++i) {
                            qamz += wave[qamidx++];
                            addr_qam++;
                            if(addr_qam % m_qamPeriod == 0) { //decimation.
                                uint16_t iq = qamIQ(z0 * qamz);
//                                fprintf(stderr, " %04x", iq);
                                this->interfaceQAM()->writeToRegister16(QAM_ADDR_REG_DATA_LSW, iq);
                                this->interfaceQAM()->writeToRegister8(QAM_ADDR_REG_CTRL, 2); //addr++
                                this->interfaceQAM()->writeToRegister8(QAM_ADDR_REG_CTRL, 0); //?
                                qamz = 0.0;
                            }
                        }
//                        fprintf(stderr, "<- QAM waveform %u %u %u\n", qamidx, phase, pnum);
                        if(addr_qam / m_qamPeriod >= MAX_QAM_PATTERN_SIZE) {
                            throw XInterface::XInterfaceError(i18n("Number of QAM patterns exceeded the size limit."), __FILE__, __LINE__);
                        }
                    }
                    pnum_prev = pnum;
                }
            }
            this->interface()->writeToRegister8(ADDR_REG_STS, 0); //clears STS.
            this->interface()->writeToRegister16(ADDR_REG_REP_N, 0); //infinite loops
            //mimics PULBOAD.BAS:StartBrd(0)
            this->interface()->writeToRegister16(ADDR_REG_ADDR_L, 0);
            this->interface()->writeToRegister8(ADDR_REG_ADDR_H, 0);
            this->interface()->writeToRegister8(ADDR_REG_MODE, 8 | (ext_clock ? 4 : 0)); //external Trig

            writer.flush(); //sends above commands here.
            if(hasQAMPorts()) {
                this->interfaceQAM()->writeToRegister16(QAM_ADDR_REG_REP_N, 0); //infinite loops
//                //this readout procedure is necessary for unknown reasons!
//                //mimics modQAM.bas:dump_qam
                std::vector<uint8_t> buf(addr_qam / m_qamPeriod * 2);
                this->interfaceQAM()->burstRead(QAM_ADDR_REG_DATA_LSW, &buf[0], buf.size());
                writerQAM.flush(); //sends above commands here.
            }
        }
        if(is_saturated)
            gErrPrint(i18n("QAM levels may exceeds a limit voltage."));

        //For PROT3, this readout procedure is necessary for unknown reasons!
        std::vector<uint8_t> buf(addr);
        this->interface()->burstRead(ADDR_REG_TIME_LSW, &buf[0], buf.size());
        this->interface()->burstRead(ADDR_REG_TIME_MSW, &buf[0], buf.size());
//        this->interface()->burstRead(ADDR_REG_DATA_LSW, &buf[0], buf.size());
//        this->interface()->burstRead(ADDR_REG_DATA_MSW, &buf[0], buf.size());

        this->interface()->writeToRegister16(ADDR_REG_ADDR_L, 0);
        this->interface()->writeToRegister8(ADDR_REG_ADDR_H, 0);

        this->interface()->writeToRegister8(ADDR_REG_CTRL, 1); //starts it

        fprintf(stderr, "ed.\n");
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
