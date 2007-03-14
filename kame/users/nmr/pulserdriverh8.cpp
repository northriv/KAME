/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
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
#include <klocale.h>

REGISTER_TYPE(XDriverList, H8Pulser, "NMR pulser handmade-H8");

static const unsigned int MAX_PATTERN_SIZE = 2048;

//[ms]
static const double TIMER_PERIOD = (1.0/(25.0e3));
//[ms]
static const double MIN_PULSE_WIDTH = 0.001;

double XH8Pulser::resolution() const {
	return TIMER_PERIOD;
}
double XH8Pulser::minPulseWidth() const {
	return MIN_PULSE_WIDTH;
}

XH8Pulser::XH8Pulser(const char *name, bool runtime,
					 const shared_ptr<XScalarEntryList> &scalarentries,
					 const shared_ptr<XInterfaceList> &interfaces,
					 const shared_ptr<XThermometerList> &thermometers,
					 const shared_ptr<XDriverList> &drivers) :
    XCharDeviceDriver<XPulser>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
    interface()->setEOS("\r\n");
    interface()->baudrate()->value(115200);
    
    const int ports[] = {
    	PORTSEL_GATE, PORTSEL_PREGATE, PORTSEL_TRIG1, PORTSEL_TRIG2,
    	PORTSEL_QPSK_A, PORTSEL_QPSK_B, PORTSEL_ASW, PORTSEL_UNSEL,
    	PORTSEL_PULSE1, PORTSEL_PULSE2, PORTSEL_COMB, PORTSEL_UNSEL,
    	PORTSEL_QPSK_OLD_PSGATE, PORTSEL_QPSK_OLD_NONINV, PORTSEL_QPSK_OLD_INV, PORTSEL_COMB_FM
    };
    for(unsigned int i = 0; i < sizeof(ports)/sizeof(int); i++) {
    	portSel(i)->value(ports[i]);
	}

}
void
XH8Pulser::open() throw (XInterface::XInterfaceError &)
{  
	start();
}

void
XH8Pulser::createNativePatterns()
{
	m_zippedPatterns.clear();
	for(RelPatListIterator it = m_relPatList.begin(); it != m_relPatList.end(); it++)
	{
		pulseAdd(it->toappear, (unsigned short)(it->pattern & PAT_DO_MASK));
	}
}
int
XH8Pulser::pulseAdd(uint64_t term, unsigned short pattern)
{
	term = std::max(term, (uint64_t)lrint(MIN_PULSE_WIDTH / TIMER_PERIOD));

	uint32_t ulen = (uint32_t)((term - 1) / 0x8000uLL);
	uint32_t llen = (uint32_t)((term - 1) % 0x8000uLL);

	switch(ulen)
	{
		h8ushort x;
	case 0:
		x.msb = llen / 0x100;
		x.lsb = llen % 0x100;
		m_zippedPatterns.push_back(x);
		x.msb = pattern / 0x100;
		x.lsb = pattern % 0x100;
		m_zippedPatterns.push_back(x);
		break;
	default:
		x.msb = (ulen % 0x8000u + 0x8000u) / 0x100;
		x.lsb = (ulen % 0x8000u + 0x8000u) % 0x100;
		m_zippedPatterns.push_back(x);
		x.msb = (ulen / 0x8000u) / 0x100;
		x.lsb = (ulen / 0x8000u) % 0x100;
		m_zippedPatterns.push_back(x);
		x.msb = (llen + 0x8000u) / 0x100;
		x.lsb = (llen + 0x8000u) % 0x100;
		m_zippedPatterns.push_back(x);
		x.msb = pattern / 0x100;
		x.lsb = pattern % 0x100;
		m_zippedPatterns.push_back(x);
		break;
	}
	return 0;
}
static unsigned short makesum(unsigned char *start, uint32_t bytes)
{
	unsigned short sum = 0;

	for(; bytes > 0; bytes--)
		sum += *start++;
	return sum;
}
void
XH8Pulser::changeOutput(bool output, unsigned int blankpattern)
{
	if(output)
	{
		if(m_zippedPatterns.empty() |
		   (m_zippedPatterns.size() >= MAX_PATTERN_SIZE ))
			throw XInterface::XInterfaceError(KAME::i18n("Pulser Invalid pattern"), __FILE__, __LINE__);
		for(unsigned int retry = 0; ; retry++) {
			try {
				interface()->sendf("$poff %x", blankpattern);
				interface()->send("$pclear");
				unsigned int size = m_zippedPatterns.size();
				unsigned int pincr = size;
				interface()->sendf("$pload %x %x", size, pincr);
				interface()->receive();
				interface()->write(">", 1);
				msecsleep(1);
				for(unsigned int j=0; j < size; j += pincr)
				{
					interface()->write(
						(char *)&m_zippedPatterns[j], pincr * 2);
					unsigned short sum = 
						makesum((unsigned char *)&m_zippedPatterns[j], pincr * 2);
					h8ushort nsum;
					nsum.lsb = sum % 0x100; nsum.msb = sum / 0x100;
					interface()->write((char *)&nsum, 2);
				}
				interface()->write("    \n", 5);
				interface()->receive();
				unsigned int ret;
				if(interface()->scanf("%x", &ret) != 1)
					throw XInterface::XConvError(__FILE__, __LINE__);
				if(ret != size)
					throw XInterface::XInterfaceError(KAME::i18n("Pulser Check Sum Error"), __FILE__, __LINE__);
			}
			catch (XKameError &e) {
				if(retry > 0) throw e;
				e.print(getLabel() + ": " + KAME::i18n("try to continue") + ", ");
				continue;
			}
			break;
		}
	}
	else
	{
		interface()->sendf("$poff %x", blankpattern);
	}
}
