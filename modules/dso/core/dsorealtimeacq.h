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
    using tRawAI = int16_t;
    //! Changes the instrument state so that it can wait for a trigger (arm).
    virtual void startAcquision() = 0;
    //! Prepares the instrument state just before startAcquision().
    virtual void commitAcquision() = 0;
    //! From a triggerable state to a commited state.
    virtual void stopAcquision() = 0;
    //! From any state to unconfigured state.
    virtual void clearAcquision() = 0;
    //! \return # of configured channels.
    virtual unsigned int getNumOfChannels() = 0;
    //! \return Additional informations of channels to be stored.
    virtual XString getChannelInfoStrings() = 0;
    //! \return Trigger candidates
    virtual std::deque<XString> hardwareTriggerNames() = 0;
    //! Prepares instrumental setups for timing.
    virtual double setupTimeBase() = 0;
    //! Prepares instrumental setups for channels.
    virtual void setupChannels() = 0;
    //! Prepares instrumental setups for trigger.
    virtual void setupHardwareTrigger() = 0;
    //! Clears trigger settings.
    virtual void disableHardwareTriggers() = 0;
    //! \return # of samples per channel acquired from the arm.
    virtual uint64_t getTotalSampsAcquired() = 0;
    //! \return # of new samples per channel stored in the driver's ring buffer from the current read position.
    virtual uint32_t getNumSampsToBeRead() = 0;
    //! Sets the position for the next reading operated by a readAcqBuffer() function.
    //! \arg pos position from the hardware arm, or from the first point for pre-trigger acquision.
    //! \return true if the operation is sucessful
    virtual bool setReadPosition(uint64_t pos) = 0;
    //! Copies data from driver's ring buffer from the current read position.
    //! The position for the next reading will be advanced by the return value.
    //! \arg buf to which 16bitxChannels stream is stored, packed by channels first.
    //! \return # of samples per channel read.
    virtual uint32_t readAcqBuffer(uint32_t size, tRawAI *buf) = 0;

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

    bool hasSoftwareTrigger() const {return !!m_softwareTrigger;}
    shared_ptr<SoftwareTrigger> softwareTrigger() const {return m_softwareTrigger;}

    enum {CAL_POLY_ORDER = 4};
    double m_coeffAI[4][CAL_POLY_ORDER];
    unsigned int m_preTriggerPos;

    unsigned int sizeofRecordBuf() const {return m_recordBuf.size();}

    void suspendAcquision();
private:
    shared_ptr<SoftwareTrigger> m_softwareTrigger;
    shared_ptr<Listener> m_lsnOnSoftTrigStarted, m_lsnOnSoftTrigChanged;
    void onSoftTrigStarted(const shared_ptr<SoftwareTrigger> &);
    void onSoftTrigChanged(const shared_ptr<SoftwareTrigger> &);
    shared_ptr<XThread<XRealTimeAcqDSO<tDriver>>> m_threadReadAI;
    void *executeReadAI(const atomic<bool> &);
    atomic<bool> m_suspendRead;
    atomic<bool> m_running;
    std::vector<tRawAI> m_recordBuf;
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
    void setupAcquision();
    void clearAll();
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
