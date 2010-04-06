/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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

//! My pulser driver
class XSHPulser : public XCharDeviceDriver<XPulser> {
public:
	XSHPulser(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XSHPulser() {}

	struct Payload : public XPulser::Payload {
	private:
		friend class XSHPulser;
		uint32_t m_lastPattern;
		uint64_t m_dmaTerm;
		std::vector<unsigned char> m_zippedPatterns;
		int m_waveformPos[PAT_QAM_PULSE_IDX_MASK / PAT_QAM_PULSE_IDX];
	};

	//! time resolution [ms]
    virtual double resolution() const;
protected:
    //! Sends patterns to pulser or turn-off
    virtual void changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern);
    //! Converts RelPatList to native patterns
    virtual void createNativePatterns(Transaction &tr);
    virtual double resolutionQAM() const {return resolution();}
    //! minimum period of pulses [ms]
    virtual double minPulseWidth() const {return resolution();}
    //! existense of AO ports.
    virtual bool haveQAMPorts() const {return true;}

private:
	int setAUX2DA(Transaction &tr, double volt, int addr);
	int insertPreamble(Transaction &tr, uint16_t startpattern);
	int finishPulse(Transaction &tr);
	//! Add 1 pulse pattern
	//! \param term a period to next pattern
	//! \param pattern a pattern for digital, to appear
	int pulseAdd(Transaction &tr, uint64_t term, uint32_t pattern, bool firsttime, bool dryrun);

	//dma time commands
	static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_END;
	//+1: a phase by 90deg.
	//+2,3: from DMA start
	//+4,5: src neg. offset from here
	static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_COPY_HBURST;
	//+1,2: time to appear
	//+2,3: pattern to appear
	static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_LSET_LONG;
	//+0: time to appear + START
	//+1,2: pattern to appear
	static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_LSET_START;
	static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_LSET_END;

	//off-dma time commands
	static const unsigned char PATTERN_ZIPPED_COMMAND_END;
	//+1,2 : TimerL
	static const unsigned char PATTERN_ZIPPED_COMMAND_WAIT;
	//+1,2 : TimerL
	//+3,4: LSW of TimerU
	static const unsigned char PATTERN_ZIPPED_COMMAND_WAIT_LONG;
	//+1,2 : TimerL
	//+3,4: MSW of TimerU
	//+5,6: LSW of TimerU
	static const unsigned char PATTERN_ZIPPED_COMMAND_WAIT_LONG_LONG;
	//+1: byte
	static const unsigned char PATTERN_ZIPPED_COMMAND_AUX1;
	//+1: byte
	static const unsigned char PATTERN_ZIPPED_COMMAND_AUX3;
	//+1: address
	//+2,3: value
	static const unsigned char PATTERN_ZIPPED_COMMAND_AUX2_DA;
	//+1,2: loops
	static const unsigned char PATTERN_ZIPPED_COMMAND_DO;
	static const unsigned char PATTERN_ZIPPED_COMMAND_LOOP;
	static const unsigned char PATTERN_ZIPPED_COMMAND_LOOP_INF;
	static const unsigned char PATTERN_ZIPPED_COMMAND_BREAKPOINT;
	static const unsigned char PATTERN_ZIPPED_COMMAND_PULSEON;
	//+1,2: last pattern
	static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_SET;
	//+1,2: size
	//+2n: patterns
	static const unsigned char PATTERN_ZIPPED_COMMAND_DMA_HBURST;
	//+1 (signed char): QAM1 offset
	//+2 (signed char): QAM2 offset
	static const unsigned char PATTERN_ZIPPED_COMMAND_SET_DA_TUNE_OFFSET;
	//+1 (signed char): QAM1 level
	//+2 (signed char): QAM2 level
	static const unsigned char PATTERN_ZIPPED_COMMAND_SET_DA_TUNE_LEVEL;
	//+1 (signed char): QAM1 delay
	//+2 (signed char): QAM2 delay
	static const unsigned char PATTERN_ZIPPED_COMMAND_SET_DA_TUNE_DELAY;
};
