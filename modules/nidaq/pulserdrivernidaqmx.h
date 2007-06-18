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
#ifndef PULSERDRIVERNIDAQMX_H_
#define PULSERDRIVERNIDAQMX_H_

#include "pulserdriver.h"

#include "nidaqmxdriver.h"

#ifdef HAVE_NI_DAQMX

#include <vector>

class XNIDAQmxPulser : public XNIDAQmxDriver<XPulser>
{
	XNODE_OBJECT
protected:
	XNIDAQmxPulser(const char *name, bool runtime,
				   const shared_ptr<XScalarEntryList> &scalarentries,
				   const shared_ptr<XInterfaceList> &interfaces,
				   const shared_ptr<XThermometerList> &thermometers,
				   const shared_ptr<XDriverList> &drivers);
public:
	virtual ~XNIDAQmxPulser();

protected:
	virtual void open() throw (XInterface::XInterfaceError &) = 0;
	virtual void close() throw (XInterface::XInterfaceError &);
	
    //! time resolution [ms]
    virtual double resolution() const {return m_resolutionDO;}
    double resolutionQAM() const {return m_resolutionAO;}
	//! existense of AO ports.
    virtual bool haveQAMPorts() const = 0;

 	virtual const shared_ptr<XNIDAQmxInterface> &intfDO() const {return interface();}
	virtual const shared_ptr<XNIDAQmxInterface> &intfAO() const {return interface();} 
	virtual const shared_ptr<XNIDAQmxInterface> &intfCtr() const {return interface();} 
       	 
    //! send patterns to pulser or turn-off
    virtual void changeOutput(bool output, unsigned int blankpattern);
    //! convert RelPatList to native patterns
    virtual void createNativePatterns();

    //! minimum period of pulses [ms]
    virtual double minPulseWidth() const {return resolution();}
    
	void openDO(bool use_ao_clock = false) throw (XInterface::XInterfaceError &);
	void openAODO() throw (XInterface::XInterfaceError &);
private:
	void startPulseGen() throw (XInterface::XInterfaceError &);
	void stopPulseGen();
	
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

	static int32 _onTaskDone(TaskHandle task, int32 status, void*);
	void onTaskDone(TaskHandle task, int32 status);

	scoped_ptr<std::vector<GenPattern> > m_genPatternList;
	scoped_ptr<std::vector<GenPattern> > m_genPatternListNext;
	typedef std::vector<GenPattern>::iterator GenPatternIterator;

	GenPatternIterator m_genLastPatItAO, m_genLastPatItDO;
	uint64_t m_genRestSampsAO, m_genRestSampsDO;
	enum { NUM_AO_CH = 2};
	unsigned int m_genAOIndex;
	shared_ptr<XNIDAQmxInterface::SoftwareTrigger> m_softwareTrigger;
	unsigned int m_pausingBit;
	unsigned int m_aswBit;
	unsigned int m_pausingCount;
	std::string m_pausingCh;
	std::string m_pausingSrcTerm;
	std::string m_pausingGateTerm;
	unsigned int m_bufSizeHintDO;
	unsigned int m_bufSizeHintAO;
	unsigned int m_transferSizeHintDO;
	unsigned int m_transferSizeHintAO;
	uint64_t m_genTotalCountDO;
	atomic<bool> m_running;
	atomic<bool> m_suspendDO, m_suspendAO;
protected:	
	double m_resolutionDO, m_resolutionAO;
	TaskHandle m_taskAO, m_taskDO,
		m_taskDOCtr, m_taskGateCtr;
private:
	enum {PORTSEL_PAUSING = 16};
	std::vector<tRawDO> m_genBufDO;
	typedef struct {tRawAO ch[NUM_AO_CH];} tRawAOSet;
	std::vector<tRawAOSet> m_genBufAO;
	tRawAOSet m_genAOZeroLevel;
	scoped_ptr<std::vector<tRawAOSet> > m_genPulseWaveAO[PAT_QAM_MASK / PAT_QAM_PHASE];
	scoped_ptr<std::vector<tRawAOSet> > m_genPulseWaveNextAO[PAT_QAM_MASK / PAT_QAM_PHASE];
	enum { CAL_POLY_ORDER = 4};
	double m_coeffAO[NUM_AO_CH][CAL_POLY_ORDER];
	double m_coeffAODev[NUM_AO_CH][CAL_POLY_ORDER];
	double m_upperLimAO[NUM_AO_CH];
	double m_lowerLimAO[NUM_AO_CH];
	
	inline tRawAOSet aoVoltToRaw(const std::complex<double> &volt);
	void genBankDO();
	void genBankAO();
	shared_ptr<XThread<XNIDAQmxPulser> > m_threadWriteAO;
	shared_ptr<XThread<XNIDAQmxPulser> > m_threadWriteDO;
	void writeBufAO(const atomic<bool> &terminated, const atomic<bool> &suspended);
	void writeBufDO(const atomic<bool> &terminated, const atomic<bool> &suspended);
	void *executeWriteAO(const atomic<bool> &);
	void *executeWriteDO(const atomic<bool> &);
	
	int makeWaveForm(int num, double pw, tpulsefunc func, double dB, double freq = 0.0, double phase = 0.0);
	XRecursiveMutex m_totalLock;
	XRecursiveMutex m_mutexAO, m_mutexDO;
  
	inline bool tryOutputSuspend(const atomic<bool> &flag,
								 XRecursiveMutex &mutex, const atomic<bool> &terminated);
};

#endif //HAVE_NI_DAQMX

#endif /*PULSERDRIVERNIDAQMX_H_*/
