/***************************************************************************
		Copyright (C) 2002-2011 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef PULSERDRIVERNIDAQMX_H_
#define PULSERDRIVERNIDAQMX_H_

#include "pulserdriver.h"

#include "nidaqmxdriver.h"

#include <vector>

class XNIDAQmxPulser : public XNIDAQmxDriver<XPulser> {
public:
	XNIDAQmxPulser(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XNIDAQmxPulser();

    //! time resolution [ms]
    virtual double resolution() const {return m_resolutionDO;}

    struct Payload : public XNIDAQmxDriver<XPulser>::Payload {
    private:
    	friend class XNIDAQmxPulser;
    };
protected:
	virtual void open() throw (XInterface::XInterfaceError &) = 0;
	virtual void close() throw (XInterface::XInterfaceError &);
	
    double resolutionQAM() const {return m_resolutionAO;}
	//! \return Existense of AO ports.
    virtual bool haveQAMPorts() const = 0;

 	virtual const shared_ptr<XNIDAQmxInterface> &intfDO() const {return interface();}
	virtual const shared_ptr<XNIDAQmxInterface> &intfAO() const {return interface();} 
	virtual const shared_ptr<XNIDAQmxInterface> &intfCtr() const {return interface();} 
       	 
    //! Sends patterns to pulser or turns off.
    virtual void changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern);
    //! Converts RelPatList to native patterns.
    virtual void createNativePatterns(Transaction &tr);

    //! \return Minimum period of pulses [ms]
    virtual double minPulseWidth() const {return resolution();}
    
	void openDO(bool use_ao_clock = false) throw (XInterface::XInterfaceError &);
	void openAODO() throw (XInterface::XInterfaceError &);
private:
	void startPulseGen(const Snapshot &shot) throw (XInterface::XInterfaceError &);
	void stopPulseGen();
	void abortPulseGen();
	
	void clearTasks();
	void setupTasksDO(bool use_ao_clock);
	void setupTasksAODO();

 	typedef int16 tRawAO;
	typedef uInt16 tRawDO;
	struct GenPattern {
		GenPattern(uint32_t pat, uint64_t next) :
	        pattern(pat), tonext(next) {}
		uint32_t pattern;
		uint64_t tonext; //!< in samps for buffer.
	};

	static int32 onTaskDone_(TaskHandle task, int32 status, void*);
	void onTaskDone(TaskHandle task, int32 status);

	scoped_ptr<std::vector<GenPattern> > m_genPatternList;
	scoped_ptr<std::vector<GenPattern> > m_genPatternListNext;
	typedef std::vector<GenPattern>::iterator GenPatternIterator;

	GenPatternIterator m_genLastPatIt;
	uint64_t m_genRestSamps;
	enum { NUM_AO_CH = 2};
	unsigned int m_genAOIndex;
	shared_ptr<XNIDAQmxInterface::SoftwareTrigger> m_softwareTrigger;
	unsigned int m_pausingBit;
	unsigned int m_aswBit;
	unsigned int m_pausingCount;
	XString m_pausingCh;
	XString m_pausingSrcTerm;
	XString m_pausingGateTerm;
	unsigned int m_preFillSizeDO;
	unsigned int m_preFillSizeAO;
	unsigned int m_transferSizeHintDO;
	unsigned int m_transferSizeHintAO;
	uint64_t m_genTotalCount;
	bool m_running;
protected:	
	double m_resolutionDO, m_resolutionAO;
	TaskHandle m_taskAO, m_taskDO,
		m_taskDOCtr, m_taskGateCtr;
private:
	enum {PORTSEL_PAUSING = 16};

	template <typename T>
	struct RingBuffer {
		enum {CHUNK_DIVISOR = 16};
		void reserve(ssize_t s) {m_data.resize(s); m_curReadPos = 0; m_endOfWritten = 0; m_end = s;}
		const T*curReadPos() const { return &m_data[m_curReadPos];}
		ssize_t writtenSize() const {
			ssize_t end_of_written = m_endOfWritten;
			if(m_curReadPos <= end_of_written) {
				return end_of_written - m_curReadPos;
			}
			return m_end - m_curReadPos;
		}
		void finReading(ssize_t size_read) {
			ssize_t p = m_curReadPos + size_read;
			ASSERT(p <= m_end);
			if((m_endOfWritten < m_curReadPos) && (p == m_end)) p = 0;
			m_curReadPos = p;
		}
		ssize_t chunkSize() {
			return m_data.size() / CHUNK_DIVISOR;
		}
		T *curWritePos() {
			ssize_t readpos = m_curReadPos;
			if(m_endOfWritten + chunkSize() > m_data.size()) {
				if(readpos == 0)
					return NULL;
				m_end = m_endOfWritten;
				m_endOfWritten = 0;
			}
			if((readpos > m_endOfWritten) && (readpos <= m_endOfWritten + chunkSize()))
				return NULL;
			return &m_data[m_endOfWritten];
		}
		void finWriting(T *pend) {
			ASSERT(pend - m_endOfWritten <= chunkSize());
			ssize_t pos = pend - &m_data[0];
			m_endOfWritten = pos;
		}
	private:
		atomic<ssize_t> m_curReadPos, m_endOfWritten;
		atomic<ssize_t> m_end;
		std::vector<T> m_data;
	};
	typedef struct {tRawAO ch[NUM_AO_CH];} tRawAOSet;
	RingBuffer<tRawDO> m_patBufDO; //!< Buffer containing generated patterns for DO.
	RingBuffer<tRawAOSet>  m_patBufAO; //!< Buffer containing generated patterns for AO.

	tRawAOSet m_genAOZeroLevel;

	scoped_ptr<std::vector<tRawAOSet> > m_genPulseWaveAO[PAT_QAM_MASK / PAT_QAM_PHASE];
	scoped_ptr<std::vector<tRawAOSet> > m_genPulseWaveNextAO[PAT_QAM_MASK / PAT_QAM_PHASE];
	enum { CAL_POLY_ORDER = 4};
	double m_coeffAO[NUM_AO_CH][CAL_POLY_ORDER];
	double m_coeffAODev[NUM_AO_CH][CAL_POLY_ORDER];
	double m_upperLimAO[NUM_AO_CH];
	double m_lowerLimAO[NUM_AO_CH];
	inline tRawAOSet aoVoltToRaw(const std::complex<double> &volt);

	shared_ptr<XThread<XNIDAQmxPulser> > m_threadWriter;
	bool m_isThreadWriterReady;
	//! \return Succeeded or not.
	template <bool UseAO>
	inline bool fillBuffer();
	//! \return Counts being sent.
	ssize_t writeToDAQmxDO(const tRawDO *pDO, ssize_t samps);
	ssize_t writeToDAQmxAO(const tRawAOSet *pAO, ssize_t samps);
	void *executeWriter(const atomic<bool> &);
	void *executeFillBuffer(const atomic<bool> &);

	int makeWaveForm(int num, double pw, tpulsefunc func, double dB, double freq = 0.0, double phase = 0.0);
	XRecursiveMutex m_totalLock;
};

#endif /*PULSERDRIVERNIDAQMX_H_*/
