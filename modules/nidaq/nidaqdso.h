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
#ifndef nidaqdsoH
#define nidaqdsoH

#include "dso.h"

#include "nidaqmxdriver.h"

//! Software DSO w/ NI DAQmx
class XNIDAQmxDSO : public XNIDAQmxDriver<XDSO> {
public:
	XNIDAQmxDSO(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XNIDAQmxDSO();
	//! Converts raw to record
	virtual void convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);
protected:
	//! Be called just after opening interface. Call start() inside this routine appropriately.
	virtual void open() throw (XKameError &);
	//! Be called during stopping driver. Call interface()->stop() inside this routine.
	virtual void close() throw (XKameError &);

	virtual void onTrace1Changed(const Snapshot &shot, XValueNodeBase *);
	virtual void onTrace2Changed(const Snapshot &shot, XValueNodeBase *);
	virtual void onTrace3Changed(const Snapshot &shot, XValueNodeBase *);
	virtual void onTrace4Changed(const Snapshot &shot, XValueNodeBase *);
	virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onSingleChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onTrigSourceChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onTrigPosChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onTrigLevelChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onTrigFallingChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onTimeWidthChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onVFullScale1Changed(const Snapshot &shot, XValueNodeBase *);
	virtual void onVFullScale2Changed(const Snapshot &shot, XValueNodeBase *);
	virtual void onVFullScale3Changed(const Snapshot &shot, XValueNodeBase *);
	virtual void onVFullScale4Changed(const Snapshot &shot, XValueNodeBase *);
	virtual void onVOffset1Changed(const Snapshot &shot, XValueNodeBase *);
	virtual void onVOffset2Changed(const Snapshot &shot, XValueNodeBase *);
	virtual void onVOffset3Changed(const Snapshot &shot, XValueNodeBase *);
	virtual void onVOffset4Changed(const Snapshot &shot, XValueNodeBase *);
	virtual void onRecordLengthChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onForceTriggerTouched(const Snapshot &shot, XTouchableNode *);

	virtual double getTimeInterval();
	//! Clears count or start sequence measurement
	virtual void startSequence();
	virtual int acqCount(bool *seq_busy);

	//! Loads waveform and settings from instrument
	virtual void getWave(shared_ptr<RawData> &writer, std::deque<XString> &channels);

	virtual bool isDRFCoherentSGSupported() const {return true;}
private:
	typedef int16 tRawAI;
	shared_ptr<XNIDAQmxInterface::SoftwareTrigger> m_softwareTrigger;
	shared_ptr<XListener> m_lsnOnSoftTrigStarted, m_lsnOnSoftTrigChanged;
	void onSoftTrigStarted(const shared_ptr<XNIDAQmxInterface::SoftwareTrigger> &);
	void onSoftTrigChanged(const shared_ptr<XNIDAQmxInterface::SoftwareTrigger> &);
	shared_ptr<XThread<XNIDAQmxDSO> > m_threadReadAI;
	void *executeReadAI(const atomic<bool> &);
	atomic<bool> m_suspendRead;
	atomic<bool> m_running;
	std::vector<tRawAI> m_recordBuf;
	enum {CAL_POLY_ORDER = 4};
    float64 m_coeffAI[4][CAL_POLY_ORDER];
	inline float64 aiRawToVolt(const float64 *pcoeff, float64 raw);
	struct DSORawRecord {
		unsigned int numCh;
		unsigned int accumCount;
		unsigned int recordLength;
		int acqCount;
		bool isComplex; //true in the coherent SG mode.
		std::vector<int32_t> record;
		atomic<int> locked;
		bool tryLock() {
            bool ret = locked.compare_set_strong(false, true);
			readBarrier();
			return ret;
		}
		void unlock() {
			assert(locked);
			writeBarrier();
			locked = false;
		}
	};
	DSORawRecord m_dsoRawRecordBanks[2];
	int m_dsoRawRecordBankLatest;
	//! for moving av.
	std::deque<std::vector<tRawAI> > m_record_av; 
	TaskHandle m_task;
	double m_interval;
	unsigned int m_preTriggerPos;
	void clearAcquision();
	void setupAcquision();
	void disableTrigger();
	void setupTrigger();
	void clearStoredSoftwareTrigger();
	void setupSoftwareTrigger();
	void setupTiming();
	void createChannels();
	void acquire(const atomic<bool> &terminated);
	static int32 onTaskDone_(TaskHandle task, int32 status, void*);
	void onTaskDone(TaskHandle task, int32 status);

	XRecursiveMutex m_readMutex;

	inline bool tryReadAISuspend(const atomic<bool> &terminated);
};

#endif
