/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
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

private:
	enum { NUM_AO_CH = 2};
	enum { CAL_POLY_ORDER = 4};
 	typedef int16 tRawAO;
	typedef struct {tRawAO ch[NUM_AO_CH];} tRawAOSet;
	typedef uInt16 tRawDO;
	struct GenPattern {
		GenPattern(uint32_t pat, uint64_t next) :
	        pattern(pat), tonext(next) {}
		uint32_t pattern;
		uint64_t tonext; //!< in samps for buffer.
	};
public:
	struct Payload : public XNIDAQmxDriver<XPulser>::Payload {
    private:
    	friend class XNIDAQmxPulser;
    	//! The pattern passed from \a XPulser to be output.
		//! \sa createNativePatterns()
    	shared_ptr<std::vector<GenPattern> > m_genPatternListNext;
    	shared_ptr<std::vector<tRawAOSet> > m_genPulseWaveNextAO[PAT_QAM_MASK / PAT_QAM_PHASE];
    	tRawAOSet m_genAOZeroLevelNext;
    };
protected:
	virtual void open() throw (XKameError &) = 0;
	virtual void close() throw (XKameError &);
	
    double resolutionQAM() const {return m_resolutionAO;}
	//! \return Existence of AO ports.
    virtual bool hasQAMPorts() const = 0;

 	virtual const shared_ptr<XNIDAQmxInterface> &intfDO() const {return interface();}
	virtual const shared_ptr<XNIDAQmxInterface> &intfAO() const {return interface();} 
	virtual const shared_ptr<XNIDAQmxInterface> &intfCtr() const {return interface();} 
       	 
    //! Sends patterns to pulser or turns off.
    virtual void changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern);
    //! Converts RelPatList to native patterns.
    virtual void createNativePatterns(Transaction &tr);

    //! \return Minimum period of pulses [ms]
    virtual double minPulseWidth() const {return resolution();}
    
	void openDO(bool use_ao_clock = false) throw (XKameError &);
	void openAODO() throw (XKameError &);
private:
	void startPulseGen(const Snapshot &shot) throw (XKameError &);
	void stopPulseGen();
	void abortPulseGen();
	
	void stopPulseGenFreeRunning(unsigned int blankpattern);
	void startPulseGenFromFreeRun(const Snapshot &shot);

	void clearTasks();
	void setupTasksDO(bool use_ao_clock);
	void setupTasksAODO();

	static int32 onTaskDone_(TaskHandle task, int32 status, void*);
	void onTaskDone(TaskHandle task, int32 status);

	shared_ptr<std::vector<GenPattern> > m_genPatternList;

	typedef std::vector<GenPattern>::iterator GenPatternIterator;
	GenPatternIterator m_genLastPatIt;

	unsigned int m_genAOIndex; //!< Preserved index for \a m_genPulseWaveAO[], \sa fillBuffer().
	shared_ptr<XNIDAQmxInterface::SoftwareTrigger> m_softwareTrigger;
	unsigned int m_pausingBit; //!< Pausing bit triggers counter that stops DO/AO for a certain period.
	unsigned int m_aswBit;
	unsigned int m_pausingCount;
	XString m_pausingCh;
	XString m_pausingSrcTerm;
	XString m_pausingGateTerm;
	unsigned int m_preFillSizeDO; //!< Determines stating condition,\sa m_isThreadWriterReady.
	unsigned int m_preFillSizeAO;
	unsigned int m_transferSizeHintDO; //!< Max size of data transfer to DAQmx lib.
	unsigned int m_transferSizeHintAO;
	//! During the pattern generation, indicates total # of clocks at the next pattern change that has elapsed from the origin.
	uint64_t m_genTotalCount;
	uint64_t m_genRestCount; //!< indicates the remaining # of clocks to the next pattern change.
	 //! During the pattern generation, total # of samplings that has been generated for DO buffered device.
	//! \a m_genTotalSamps is different from \a m_genTotalCount when pausing clocks are active.
	uint64_t m_genTotalSamps;
	uint64_t m_totalWrittenSampsDO, m_totalWrittenSampsAO; //!< # of samps written to DAQmx lib.
	bool m_running;
protected:	
	double m_resolutionDO, m_resolutionAO;
	TaskHandle m_taskAO, m_taskDO,
		m_taskDOCtr, m_taskGateCtr;
private:
	enum {PORTSEL_PAUSING = 17};
	//! Ring buffer storing AO/DO patterns being transfered to DAQmx lib.
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
		//! Tags as read.
		void finReading(ssize_t size_read) {
			ssize_t p = m_curReadPos + size_read;
			assert(p <= m_end);
			if((m_endOfWritten < m_curReadPos) && (p == m_end)) p = 0;
			m_curReadPos = p;
		}
		//! Size of a writing space beginning with \a curWritePos().
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
		//! Tags as written.
		void finWriting(T *pend) {
			assert(pend - &m_data[m_endOfWritten] <= chunkSize());
			ssize_t pos = pend - &m_data[0];
			m_endOfWritten = pos;
		}
	private:
		atomic<ssize_t> m_curReadPos, m_endOfWritten;
		atomic<ssize_t> m_end;
		std::vector<T> m_data;
	};

	RingBuffer<tRawDO> m_patBufDO; //!< Buffer containing generated patterns for DO.
	RingBuffer<tRawAOSet>  m_patBufAO; //!< Buffer containing generated patterns for AO.

	shared_ptr<std::vector<tRawAOSet> > m_genPulseWaveAO[PAT_QAM_MASK / PAT_QAM_PHASE];
	tRawAOSet m_genAOZeroLevel;

	double m_coeffAODev[NUM_AO_CH][CAL_POLY_ORDER];
	double m_upperLimAO[NUM_AO_CH];
	double m_lowerLimAO[NUM_AO_CH];
	//! Converts voltage to an internal value for DAC and compensates it.
	inline tRawAOSet aoVoltToRaw(const double poly_coeff[NUM_AO_CH][CAL_POLY_ORDER], const std::complex<double> &volt);

	shared_ptr<XThread<XNIDAQmxPulser> > m_threadWriter;
	bool m_isThreadWriterReady; //!< indicates DAQmx lib. has been (partially) filled with generated patterns.
	//! \return Succeeded or not.
	template <bool UseAO>
	inline bool fillBuffer();
	void rewindBufPos(double ms_from_gen_pos);
	//! \return Counts being sent.
	ssize_t writeToDAQmxDO(const tRawDO *pDO, ssize_t samps);
	ssize_t writeToDAQmxAO(const tRawAOSet *pAO, ssize_t samps);
	void startBufWriter();
	void stopBufWriter();
	void *executeWriter(const atomic<bool> &);
	void *executeFillBuffer(const atomic<bool> &);
	void preparePatternGen(const Snapshot &shot,
			bool use_dummypattern, unsigned int blankpattern);

	int makeWaveForm(int num, double pw, tpulsefunc func, double dB, double freq = 0.0, double phase = 0.0);
	XRecursiveMutex m_stateLock;

	typedef std::deque<std::pair<uint64_t, uint64_t> > QueueTimeGenCnt;
	QueueTimeGenCnt m_queueTimeGenCnt; //!< pairs of real time and # of DO samps, which are identical if pausing feature is off.
};

#endif /*PULSERDRIVERNIDAQMX_H_*/
