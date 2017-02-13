/***************************************************************************
        Copyright (C) 2002-2017 Kentaro Kitagawa
                           kitagawa@phys.s.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU Library General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU Library General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------
#ifndef DSOREALTIMEACQ_H
#define DSOREALTIMEACQ_H

#include "dso.h"
#include "softtrigger.h"

//! Software DSO using continuous AD read.
//! Trigger position is typically calculated by a synchronized pulse generator.
//! \sa SoftwareTrigger
template <class tDriver>
class XRealTimeAcqDSO : public tDriver {
public:
    XRealTimeAcqDSO(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XRealTimeAcqDSO();
    //! Converts raw to record
    virtual void convertRaw(typename tDriver::RawDataReader &reader, Transaction &tr) throw (typename tDriver::XRecordError&) override;
protected:
    virtual void startAcquision() = 0;
    virtual void commitAcquision() = 0;
    virtual void suspendAcquision() = 0;
    virtual void stopAcquision() = 0;
    virtual void clearAcquision() = 0;
    virtual void getNumOfChannels() = 0;
    virtual std::initializer_list<XString> hardwareTriggerNames() = 0;
    virtual void setupTimeBase() = 0;
    virtual void setupChannels() = 0;
    virtual void setupHardwareTrigger() = 0;
    virtual void disableHardwareTriggers() = 0;
    virtual uint64_t getTotalSampsAcquired() = 0;
    virtual uint32_t readAcqBuffer(uint64_t pos, uint32_t size, double *buf) = 0;

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() throw (XKameError &) override;
    //! Be called during stopping driver. Call interface()->stop() inside this routine.
    virtual void close() throw (XKameError &) override;

    virtual void onTrace1Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTrace2Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTrace3Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTrace4Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onSingleChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTrigSourceChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTrigPosChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTrigLevelChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTrigFallingChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onTimeWidthChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVFullScale1Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVFullScale2Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVFullScale3Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVFullScale4Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVOffset1Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVOffset2Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVOffset3Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onVOffset4Changed(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onRecordLengthChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onForceTriggerTouched(const Snapshot &shot, XTouchableNode *) override;

    virtual double getTimeInterval() override;
    //! Clears count or start sequence measurement
    virtual void startSequence() override;
    virtual int acqCount(bool *seq_busy) override;

    //! Loads waveform and settings from instrument
    virtual void getWave(shared_ptr<typename tDriver::RawData> &writer, std::deque<XString> &channels) override;
private:
    using tRawAI = int16_t;
    shared_ptr<SoftwareTrigger> m_softwareTrigger;
    shared_ptr<Listener> m_lsnOnSoftTrigStarted, m_lsnOnSoftTrigChanged;
    void onSoftTrigStarted(const shared_ptr<SoftwareTrigger> &);
    void onSoftTrigChanged(const shared_ptr<SoftwareTrigger> &);
    shared_ptr<XThread<XRealTimeAcqDSO<tDriver>>> m_threadReadAI;
    void *executeReadAI(const atomic<bool> &);
    atomic<bool> m_suspendRead;
    atomic<bool> m_running;
    std::vector<tRawAI> m_recordBuf;
    enum {CAL_POLY_ORDER = 4};
    double m_coeffAI[4][CAL_POLY_ORDER];
    inline double aiRawToVolt(const double *pcoeff, double raw);
    struct DSORawRecord {
        DSORawRecord() { locked = false;}
        unsigned int numCh;
        unsigned int accumCount;
        unsigned int recordLength;
        int acqCount;
        bool isComplex; //true in the coherent SG mode.
        std::vector<int32_t> record;
        atomic<int> locked;
        bool tryLock() {
            bool ret = locked.compare_set_strong(false, true);
            return ret;
        }
        void unlock() {
            assert(locked);
            locked = false;
        }
    };
    DSORawRecord m_dsoRawRecordBanks[2];
    int m_dsoRawRecordBankLatest;
    //! for moving av.
    std::deque<std::vector<tRawAI> > m_record_av;
    double m_interval;
    unsigned int m_preTriggerPos;
    void setupAcquision();
    void disableTrigger();
    void setupTrigger();
    void clearStoredSoftwareTrigger();
    void setupSoftwareTrigger();
    void setupTiming();
    void createChannels();
    void acquire(const atomic<bool> &terminated);

    XRecursiveMutex m_readMutex;

    inline bool tryReadAISuspend(const atomic<bool> &terminated);
};

#endif // DSOREALTIMEACQ_H
