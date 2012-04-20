/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "pulserdriverh8.h"
#include "charinterface.h"

REGISTER_TYPE(XDriverList, H8Pulser, "NMR pulser handmade-H8");

#define MAX_PATTERN_SIZE 2048u
//[ms]
#define TIMER_PERIOD (1.0/(25.0e3))
//[ms]
#define MIN_PULSE_WIDTH 0.001

double XH8Pulser::resolution() const {
	return TIMER_PERIOD;
}
double XH8Pulser::minPulseWidth() const {
	return MIN_PULSE_WIDTH;
}

XH8Pulser::XH8Pulser(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XCharDeviceDriver<XPulser>(name, runtime, ref(tr_meas), meas) {

    interface()->setEOS("\r\n");
	interface()->setSerialBaudRate(115200);
	interface()->setSerialStopBits(2);
    
    const int ports[] = {
    	PORTSEL_GATE, PORTSEL_PREGATE, PORTSEL_TRIG1, PORTSEL_TRIG2,
    	PORTSEL_QPSK_A, PORTSEL_QPSK_B, PORTSEL_ASW, PORTSEL_UNSEL,
    	PORTSEL_PULSE1, PORTSEL_PULSE2, PORTSEL_COMB, PORTSEL_UNSEL,
    	PORTSEL_QPSK_OLD_PSGATE, PORTSEL_QPSK_OLD_NONINV, PORTSEL_QPSK_OLD_INV, PORTSEL_COMB_FM
    };
	for(Transaction tr( *this);; ++tr) {
	    for(unsigned int i = 0; i < sizeof(ports)/sizeof(int); i++) {
	    	tr[ *portSel(i)] = ports[i];
		}
		if(tr.commit())
			break;
	}
}
void
XH8Pulser::open() throw (XInterface::XInterfaceError &) {
	start();
}

void
XH8Pulser::createNativePatterns(Transaction &tr) {
	const Snapshot &shot(tr);
	tr[ *this].m_zippedPatterns.clear();
	for(Payload::RelPatList::const_iterator it = shot[ *this].relPatList().begin();
		it != shot[ *this].relPatList().end(); it++) {
		pulseAdd(tr, it->toappear, (uint16_t)(it->pattern & PAT_DO_MASK));
	}
}
int
XH8Pulser::pulseAdd(Transaction &tr, uint64_t term, uint16_t pattern) {
	static_assert(sizeof(long long) == 8, "");

	term = std::max(term, (uint64_t)lrint(MIN_PULSE_WIDTH / TIMER_PERIOD));

	uint32_t ulen = (uint32_t)((term - 1) / 0x8000uLL);
	uint32_t llen = (uint32_t)((term - 1) % 0x8000uLL);

	switch(ulen) {
		Payload::h8ushort x;
	case 0:
		x.msb = llen / 0x100;
		x.lsb = llen % 0x100;
		tr[ *this].m_zippedPatterns.push_back(x);
		x.msb = pattern / 0x100;
		x.lsb = pattern % 0x100;
		tr[ *this].m_zippedPatterns.push_back(x);
		break;
	default:
		x.msb = (ulen % 0x8000u + 0x8000u) / 0x100;
		x.lsb = (ulen % 0x8000u + 0x8000u) % 0x100;
		tr[ *this].m_zippedPatterns.push_back(x);
		x.msb = (ulen / 0x8000u) / 0x100;
		x.lsb = (ulen / 0x8000u) % 0x100;
		tr[ *this].m_zippedPatterns.push_back(x);
		x.msb = (llen + 0x8000u) / 0x100;
		x.lsb = (llen + 0x8000u) % 0x100;
		tr[ *this].m_zippedPatterns.push_back(x);
		x.msb = pattern / 0x100;
		x.lsb = pattern % 0x100;
		tr[ *this].m_zippedPatterns.push_back(x);
		break;
	}
	return 0;
}
static uint16_t makesum(unsigned char *start, uint32_t bytes) {
	uint16_t sum = 0;

	for(; bytes > 0; bytes--)
		sum += *start++;
	return sum;
}
void
XH8Pulser::changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened())
		return;

	if(output) {
		if(shot[ *this].m_zippedPatterns.empty() |
		   (shot[ *this].m_zippedPatterns.size() >= MAX_PATTERN_SIZE ))
			throw XInterface::XInterfaceError(i18n("Pulser Invalid pattern"), __FILE__, __LINE__);
		for(unsigned int retry = 0; ; retry++) {
			try {
				interface()->sendf("$poff %x", blankpattern);
				interface()->send("$pclear");
				unsigned int size = shot[ *this].m_zippedPatterns.size();
				unsigned int pincr = size;
				interface()->sendf("$pload %x %x", size, pincr);
				interface()->receive();
				interface()->write(">", 1);
				msecsleep(1);
				for(unsigned int j=0; j < size; j += pincr) {
					interface()->write(
						(char *) &shot[ *this].m_zippedPatterns[j], pincr * 2);
					uint16_t sum = 
						makesum((unsigned char *) &shot[ *this].m_zippedPatterns[j], pincr * 2);
					Payload::h8ushort nsum;
					nsum.lsb = sum % 0x100; nsum.msb = sum / 0x100;
					interface()->write((char *)&nsum, 2);
				}
				interface()->write("    \n", 5);
				interface()->receive();
				unsigned int ret;
				if(interface()->scanf("%x", &ret) != 1)
					throw XInterface::XConvError(__FILE__, __LINE__);
				if(ret != size)
					throw XInterface::XInterfaceError(i18n("Pulser Check Sum Error"), __FILE__, __LINE__);
			}
			catch (XKameError &e) {
				if(retry > 0) throw e;
				e.print(getLabel() + ": " + i18n("try to continue") + ", ");
				continue;
			}
			break;
		}
	}
	else {
		interface()->sendf("$poff %x", blankpattern); //Pulser turned off.
	}
}
