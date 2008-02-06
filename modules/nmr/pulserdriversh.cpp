/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "pulserdriversh.h"
#include <charinterface.h>

REGISTER_TYPE(XDriverList, SHPulser, "NMR pulser handmade-SH2");

using std::max;
using std::min;

//[ms]
static const double DMA_PERIOD = (1.0/(28.64e3/2));

double XSHPulser::resolution() const {
	return DMA_PERIOD;
}

//[ms]
static const double MIN_MTU_LEN = 50e-3;
//[ms]
static const double MTU_PERIOD = (1.0/(28.64e3/1));


static const unsigned int NUM_BANK = 2;
static const unsigned int PATTERNS_ZIPPED_MAX = 40000;

//dma time commands
static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_END = 0;
//+1: a phase by 90deg.
//+2,3: from DMA start 
//+4,5: src neg. offset from here
static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_COPY_HBURST = 1;
//+1,2: time to appear
//+2,3: pattern to appear
static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_LSET_LONG = 2;
//+0: time to appear + START
//+1,2: pattern to appear
static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_LSET_START = 0x10;
static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_LSET_END = 0xffu;

//off-dma time commands
static const unsigned char PATTERN_ZIPPED_COMMAND_END = 0;
//+1,2 : TimerL
static const unsigned char PATTERN_ZIPPED_COMMAND_WAIT = 1;
//+1,2 : TimerL
//+3,4: LSW of TimerU
static const unsigned char PATTERN_ZIPPED_COMMAND_WAIT_LONG = 2;
//+1,2 : TimerL
//+3,4: MSW of TimerU
//+5,6: LSW of TimerU
static const unsigned char PATTERN_ZIPPED_COMMAND_WAIT_LONG_LONG = 3;
//+1: byte
static const unsigned char PATTERN_ZIPPED_COMMAND_AUX1 = 4;
//+1: byte
static const unsigned char PATTERN_ZIPPED_COMMAND_AUX3 = 5;
//+1: address
//+2,3: value
static const unsigned char PATTERN_ZIPPED_COMMAND_AUX2_DA = 6;
//+1,2: loops
static const unsigned char PATTERN_ZIPPED_COMMAND_DO = 7;
static const unsigned char PATTERN_ZIPPED_COMMAND_LOOP = 8;
static const unsigned char PATTERN_ZIPPED_COMMAND_LOOP_INF = 9;
static const unsigned char PATTERN_ZIPPED_COMMAND_BREAKPOINT = 0xa; 
static const unsigned char PATTERN_ZIPPED_COMMAND_PULSEON = 0xb;
//+1,2: last pattern
static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_SET = 0xc;
//+1,2: size
//+2n: patterns
static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_HBURST = 0xd;
//+1 (signed char): QAM1 offset
//+2 (signed char): QAM2 offset
static const unsigned char PATTERN_ZIPPED_COMMAND_SET_DA_TUNE_OFFSET = 0xe;
//+1 (signed char): QAM1 level
//+2 (signed char): QAM2 level
static const unsigned char PATTERN_ZIPPED_COMMAND_SET_DA_TUNE_LEVEL = 0xf;
//+1 (signed char): QAM1 delay
//+2 (signed char): QAM2 delay
static const unsigned char PATTERN_ZIPPED_COMMAND_SET_DA_TUNE_DELAY = 0x10;

XSHPulser::XSHPulser(const char *name, bool runtime,
					 const shared_ptr<XScalarEntryList> &scalarentries,
					 const shared_ptr<XInterfaceList> &interfaces,
					 const shared_ptr<XThermometerList> &thermometers,
					 const shared_ptr<XDriverList> &drivers) :
    XCharDeviceDriver<XPulser>(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
    interface()->setEOS("\n");
    interface()->baudrate()->value(115200);

    const int ports[] = {
    	PORTSEL_GATE, PORTSEL_PREGATE, PORTSEL_TRIG1, PORTSEL_TRIG2,
    	PORTSEL_GATE3, PORTSEL_COMB, PORTSEL_QSW, PORTSEL_ASW,
    	PORTSEL_PULSE1, PORTSEL_PULSE2, PORTSEL_COMB_FM, PORTSEL_COMB
    };
    for(unsigned int i = 0; i < sizeof(ports)/sizeof(int); i++) {
    	portSel(i)->value(ports[i]);
	}
}

void
XSHPulser::createNativePatterns()
{
	//dry-run to determin LastPattern, DMATime
	m_dmaTerm = 0;
	m_lastPattern = 0;
	uint32_t pat = 0;
	insertPreamble((unsigned short)pat);
	for(RelPatListIterator it = m_relPatList.begin(); it != m_relPatList.end(); it++)
	{
		pulseAdd(it->toappear, it->pattern, (it == m_relPatList.begin() ), true );
		pat = it->pattern;
	}
  
	insertPreamble((unsigned short)pat);

	for(unsigned int i = 0; i < PAT_QAM_PULSE_IDX_MASK/PAT_QAM_PULSE_IDX; i++) {
	  	const unsigned short word = qamWaveForm(i).size();
		if(!word) continue;
		m_waveformPos[i] = m_zippedPatterns.size();
		m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_HBURST);
		m_zippedPatterns.push_back((unsigned char)(word / 0x100) );
		m_zippedPatterns.push_back((unsigned char)(word % 0x100) );
		for(std::vector<std::complex<double> >::const_iterator it = 
				qamWaveForm(i).begin(); it != qamWaveForm(i).end(); it++) {
			double x = max(min(it->real() * 125.0, 124.0), -124.0);
			double y = max(min(it->imag() * 125.0, 124.0), -124.0);
			m_zippedPatterns.push_back( (unsigned char)(char)x );	
			m_zippedPatterns.push_back( (unsigned char)(char)y );	
		}
	}
	
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DO);
	m_zippedPatterns.push_back(0);
	m_zippedPatterns.push_back(0);
	for(RelPatListIterator it = m_relPatList.begin(); it != m_relPatList.end(); it++)
	{
		pulseAdd(it->toappear, it->pattern, (it == m_relPatList.begin() ), false );
		pat = it->pattern;
	}
	finishPulse();

	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_END);
}

int
XSHPulser::setAUX2DA(double volt, int addr)
{
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_AUX2_DA);
	m_zippedPatterns.push_back((unsigned char) addr);
	volt = max(volt, 0.0);
	unsigned short word = (unsigned short)rint(4096u * volt / 2 / 2.5);
	word = min(word, (unsigned short)4095u);
	m_zippedPatterns.push_back((unsigned char)(word / 0x100) );
	m_zippedPatterns.push_back((unsigned char)(word % 0x100) );
	return 0;
}
int
XSHPulser::insertPreamble(unsigned short startpattern)
{
	const double masterlevel = pow(10.0, *masterLevel() / 20.0);
	const double qamlevel1 = *qamLevel1();
	const double qamlevel2 = *qamLevel2();
	m_zippedPatterns.clear();
	
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_SET_DA_TUNE_OFFSET);
	m_zippedPatterns.push_back((unsigned char)(signed char)rint(127.5 * *qamOffset1() *1e-2 / masterlevel ) );
	m_zippedPatterns.push_back((unsigned char)(signed char)rint(127.5 * *qamOffset2() *1e-2 / masterlevel ) );
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_SET_DA_TUNE_LEVEL);
	m_zippedPatterns.push_back((unsigned char)(signed char)rint(qamlevel1 * 0x100) );
	m_zippedPatterns.push_back((unsigned char)(signed char)rint(qamlevel2 * 0x100) );
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_SET_DA_TUNE_DELAY);
//obsolete
//	m_zippedPatterns.push_back((unsigned char)(signed char)rint(*qamDelay1() / DMA_PERIOD * 1e-3) );
//	m_zippedPatterns.push_back((unsigned char)(signed char)rint(*qamDelay2() / DMA_PERIOD * 1e-3) );
	m_zippedPatterns.push_back((unsigned char)(signed char)rint(0) );
	m_zippedPatterns.push_back((unsigned char)(signed char)rint(0) );
	
	uint32_t len;	
	//wait for 1 ms
	len = lrint(1.0 / MTU_PERIOD);
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_WAIT_LONG);
	m_zippedPatterns.push_back((unsigned char)((len % 0x10000) / 0x100) );
	m_zippedPatterns.push_back((unsigned char)((len % 0x10000) % 0x100) );
	m_zippedPatterns.push_back((unsigned char)((len / 0x10000) / 0x100) );
	m_zippedPatterns.push_back((unsigned char)((len / 0x10000) % 0x100) );
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_SET);
	m_zippedPatterns.push_back((unsigned char)(startpattern / 0x100) );
	m_zippedPatterns.push_back((unsigned char)(startpattern % 0x100) );
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_LSET_START + 2);
	m_zippedPatterns.push_back((unsigned char)(startpattern / 0x100) );
	m_zippedPatterns.push_back((unsigned char)(startpattern % 0x100) );
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_END);
	
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_PULSEON);
	
	//wait for 10 ms
	len = lrint(10.0 / MTU_PERIOD);
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_WAIT_LONG);
	m_zippedPatterns.push_back((unsigned char)((len % 0x10000) / 0x100) );
	m_zippedPatterns.push_back((unsigned char)((len % 0x10000) % 0x100) );
	m_zippedPatterns.push_back((unsigned char)((len / 0x10000) / 0x100) );
	m_zippedPatterns.push_back((unsigned char)((len / 0x10000) % 0x100) );
	
/*	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_AUX1);
	int aswfilter = 3;
	if(aswFilter()->to_str() == ASW_FILTER_1) aswfilter = 1;
	if(aswFilter()->to_str() == ASW_FILTER_2) aswfilter = 2;
	m_zippedPatterns.push_back((unsigned char)aswfilter);
*/
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_AUX1);
	m_zippedPatterns.push_back((unsigned char)3);
/*	setAUX2DA(*portLevel8(), 1);
	setAUX2DA(*portLevel9(), 2);
	setAUX2DA(*portLevel10(), 3);
	setAUX2DA(*portLevel11(), 4);
	setAUX2DA(*portLevel12(), 5);
	setAUX2DA(*portLevel13(), 6);
	setAUX2DA(*portLevel14(), 7);*/
	setAUX2DA(0.0, 1);
	setAUX2DA(0.0, 2);
	setAUX2DA(0.0, 3);
	setAUX2DA(0.0, 4);
	setAUX2DA(0.0, 5);
	setAUX2DA(0.0, 6);
	setAUX2DA(0.0, 7);
	setAUX2DA(1.6 * masterlevel, 0); //tobe 5V
	
	return 0;	
}
int
XSHPulser::finishPulse(void)
{
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_END);
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_LOOP_INF);
	return 0;
}
int
XSHPulser::pulseAdd(uint64_t term, uint32_t pattern, bool firsttime, bool dryrun)
{
	const double msec = term * resolution();
	int64_t mtu_term = term * llrint(resolution() / MTU_PERIOD);
	if( (msec > MIN_MTU_LEN) && ((m_lastPattern & PAT_QAM_PULSE_IDX_MASK)/PAT_QAM_PULSE_IDX == 0) ) {
		//insert long wait
		if(!firsttime) {
			m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_END);
		}
		mtu_term += m_dmaTerm;
		uint32_t ulen = (uint32_t)(mtu_term / 0x10000uLL);
		unsigned short ulenh = (unsigned short)(ulen / 0x10000uL);
		unsigned short ulenl = (unsigned short)(ulen % 0x10000uL);	
		unsigned short dlen = (uint32_t)(mtu_term % 0x10000uLL);
		if(ulenh) {
			m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_WAIT_LONG_LONG);
			m_zippedPatterns.push_back((unsigned char)(dlen / 0x100) );
			m_zippedPatterns.push_back((unsigned char)(dlen % 0x100) );
			m_zippedPatterns.push_back((unsigned char)(ulenh / 0x100) );
			m_zippedPatterns.push_back((unsigned char)(ulenh % 0x100) );
			m_zippedPatterns.push_back((unsigned char)(ulenl / 0x100) );
			m_zippedPatterns.push_back((unsigned char)(ulenl % 0x100) );
		}
		else { if(ulenl) {
			m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_WAIT_LONG);
			m_zippedPatterns.push_back((unsigned char)(dlen / 0x100) );
			m_zippedPatterns.push_back((unsigned char)(dlen % 0x100) );
			m_zippedPatterns.push_back((unsigned char)(ulenl / 0x100) );
			m_zippedPatterns.push_back((unsigned char)(ulenl % 0x100) );
		}
		else {
			m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_WAIT);
			m_zippedPatterns.push_back((unsigned char)(dlen / 0x100) );
			m_zippedPatterns.push_back((unsigned char)(dlen % 0x100) );
		}
		}
		mtu_term -= ulen*0x10000uL + dlen;
		mtu_term = max(0LL, mtu_term);
		m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_SET);
		m_zippedPatterns.push_back((unsigned char)(m_lastPattern / 0x100) );
		m_zippedPatterns.push_back((unsigned char)(m_lastPattern % 0x100) );
		m_dmaTerm = 0;
	}
	m_dmaTerm += mtu_term;
	unsigned long pos_l = m_dmaTerm / llrint(resolution() / MTU_PERIOD);
	if(pos_l >= 0x7000u)
		throw XInterface::XInterfaceError(KAME::i18n("Too long DMA."), __FILE__, __LINE__);
	unsigned short pos = (unsigned short)pos_l;
	unsigned short len = mtu_term / llrint(resolution() / MTU_PERIOD);
	if( ((m_lastPattern & PAT_QAM_PULSE_IDX_MASK)/PAT_QAM_PULSE_IDX == 0) && ((pattern & PAT_QAM_PULSE_IDX_MASK)/PAT_QAM_PULSE_IDX > 0) ) {
		unsigned short qam_pos = m_waveformPos[(pattern & PAT_QAM_PULSE_IDX_MASK)/PAT_QAM_PULSE_IDX - 1];
		if(!dryrun) {
			if(!qam_pos || (m_zippedPatterns[qam_pos] != PATTERN_ZIPPED_COMMAND_DMA_HBURST))
				throw XInterface::XInterfaceError(KAME::i18n("No waveform."), __FILE__, __LINE__);
			unsigned short word = m_zippedPatterns.size() - qam_pos;
			m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_COPY_HBURST);
			m_zippedPatterns.push_back((unsigned char)((pattern & PAT_QAM_PHASE_MASK)/PAT_QAM_PHASE));
			m_zippedPatterns.push_back((unsigned char)(pos / 0x100) );
			m_zippedPatterns.push_back((unsigned char)(pos % 0x100) );
			m_zippedPatterns.push_back((unsigned char)(word / 0x100) );
			m_zippedPatterns.push_back((unsigned char)(word % 0x100) );
		}
	}
	if(len > PATTERN_ZIPPED_COMMAND_DMA_LSET_END - PATTERN_ZIPPED_COMMAND_DMA_LSET_START) {
		m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_LSET_LONG);
		m_zippedPatterns.push_back((unsigned char)(len / 0x100) );
		m_zippedPatterns.push_back((unsigned char)(len % 0x100) );
		m_zippedPatterns.push_back((unsigned char)(pattern / 0x100) );
		m_zippedPatterns.push_back((unsigned char)(pattern % 0x100) );
	}
	else {
		m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_LSET_START + (unsigned char)len);
		m_zippedPatterns.push_back((unsigned char)(pattern / 0x100) );
		m_zippedPatterns.push_back((unsigned char)(pattern % 0x100) );
	}
	m_lastPattern = pattern;
	return 0;
}

void
XSHPulser::changeOutput(bool output, unsigned int /*blankpattern*/)
{
	if(output)
	{
		if(m_zippedPatterns.empty() )
			throw XInterface::XInterfaceError(KAME::i18n("Pulser Invalid pattern"), __FILE__, __LINE__);
		for(unsigned int retry = 0; ; retry++) {
			try {
				interface()->write("!", 1); //poff
				interface()->receive();
				char buf[3];
				if((interface()->scanf("Pulse %3s", buf) != 1) || strncmp(buf, "Off", 3))
					throw XInterface::XConvError(__FILE__, __LINE__);
				unsigned int size = m_zippedPatterns.size();
				interface()->sendf("$pload %x", size );
				interface()->receive();
				interface()->write(">", 1);
				unsigned short sum = 0;
				for(unsigned int i = 0; i < m_zippedPatterns.size(); i++) {
					sum += m_zippedPatterns[i];
				} 
				msecsleep(1);
				interface()->write((char*)&m_zippedPatterns[0], size);
          
				interface()->receive();
				unsigned int ret;
				if(interface()->scanf("%x", &ret) != 1)
					throw XInterface::XConvError(__FILE__, __LINE__);
				if(ret != sum)
					throw XInterface::XInterfaceError(KAME::i18n("Pulser Check Sum Error"), __FILE__, __LINE__);
				interface()->send("$pon");
				interface()->receive();
				if((interface()->scanf("Pulse %2s", buf) != 1) || strncmp(buf, "On", 2))
					throw XInterface::XConvError(__FILE__, __LINE__);
			}
			catch (XKameError &e) {
				if(retry > 1) throw e;
				e.print(getLabel() + ": " + KAME::i18n("try to continue") + ", ");
				continue;
			}
			break;
		}
	}
	else
	{
		interface()->write("!", 1); //poff
		interface()->receive();
	}
	return;
}

