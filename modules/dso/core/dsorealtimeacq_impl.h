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

#include "dsorealtimeacq.h"
#include <qmessagebox.h>
#include "xwavengraph.h"

template <class tDriver> XRealTimeAcqDSO<tDriver>::XRealTimeAcqDSO(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    tDriver(name, runtime, ref(tr_meas), meas),
    m_dsoRawRecordBankLatest(0) {

    this->iterate_commit([=](Transaction &tr){
        tr[ *this->recordLength()] = 2000;
        tr[ *this->timeWidth()] = 1e-2;
        tr[ *this->average()] = 1;
    });
    if(isMemLockAvailable()) {
        //Suppress swapping.
        mlock(this, sizeof(this));
    }
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::onSoftTrigChanged(const shared_ptr<SoftwareTrigger> &) {
    this->iterate_commit([=](Transaction &tr){
        tr[ *this->trigSource()].clear();
        auto names = hardwareTriggerNames();
        for(auto &&x: names) {
            tr[ *this->trigSource()].add(x);
        }
        auto list(this->interface()->softwareTriggerManager().list());
        for(auto it = list->begin(); it != list->end(); ++it) {
            for(unsigned int i = 0; i < ( *it)->bits(); i++) {
                tr[ *this->trigSource()].add(
                    formatString("%s/line%d", ( *it)->label(), i));
            }
        }
        tr.unmark(this->m_lsnOnTrigSourceChanged); //avoids nested transactions.
    });
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::open() throw (XKameError &) {
    XScopedLock<XInterface> lock( *this->interface());
    m_running = false;

    onSoftTrigChanged(shared_ptr<SoftwareTrigger>());

    m_suspendRead = true;
    m_threadReadAI.reset(new XThread<XRealTimeAcqDSO<tDriver>>(
        this->shared_from_this(),
        &XRealTimeAcqDSO<tDriver>::executeReadAI));
    m_threadReadAI->resume();

    this->start();

    m_lsnOnSoftTrigChanged =
        this->interface()->softwareTriggerManager().onListChanged().connectWeakly(
            this->shared_from_this(), &XRealTimeAcqDSO<tDriver>::onSoftTrigChanged,
            Listener::FLAG_MAIN_THREAD_CALL);
    createChannels();
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::close() throw (XKameError &) {
    XScopedLock<XInterface> lock( *this->interface());

    m_lsnOnSoftTrigChanged.reset();

    clearAll();

    if(m_threadReadAI) {
        m_threadReadAI->terminate();
    }

    m_recordBuf.clear();
    m_record_av.clear();

    this->interface()->stop();
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::clearAll() {
    XScopedLock<XInterface> lock( *this->interface());
    m_suspendRead = true;
    XScopedLock<XRecursiveMutex> lock2(m_readMutex);

    try {
        disableTrigger();
        clearAcquision();
    }
    catch (XInterface::XInterfaceError &e) {
        e.print();
    }
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::disableTrigger() {
    XScopedLock<XInterface> lock( *this->interface());
    m_suspendRead = true;
    XScopedLock<XRecursiveMutex> lock2(m_readMutex);

    if(m_running) {
        m_running = false;
        stopAcquision();
    }
    disableHardwareTriggers();

    m_preTriggerPos = 0;

    //reset virtual trigger setup.
    if(m_softwareTrigger)
        m_softwareTrigger->disconnect();
    m_lsnOnSoftTrigStarted.reset();
    m_softwareTrigger.reset();
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::setupTrigger() {
    XScopedLock<XInterface> lock( *this->interface());
    Snapshot shot( *this);
    m_suspendRead = true;
    XScopedLock<XRecursiveMutex> lock2(m_readMutex);

    unsigned int pretrig = lrint(shot[ *this->trigPos()] / 100.0 * shot[ *this->recordLength()]);
    m_preTriggerPos = pretrig;

    setupHardwareTrigger();
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::setupSoftwareTrigger() {
    Snapshot shot( *this);
    XString src = shot[ *this->trigSource()].to_str();
    //setup virtual trigger.
    auto list(this->interface()->softwareTriggerManager().list());
    for(auto it = list->begin(); it != list->end(); ++it) {
        for(unsigned int i = 0; i < ( *it)->bits(); i++) {
            if(src == formatString("%s/line%d", ( *it)->label(), i)) {
                m_softwareTrigger = *it;
                m_softwareTrigger->connect(
                    !shot[ *this->trigFalling()] ? (1uL << i) : 0,
                    shot[ *this->trigFalling()] ? (1uL << i) : 0);
            }
        }
    }
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::setupTiming() {
    XScopedLock<XInterface> lock( *this->interface());
    Snapshot shot( *this);
    m_suspendRead = true;
    XScopedLock<XRecursiveMutex> lock2(m_readMutex);

    if(m_running) {
        m_running = false;
        stopAcquision();
    }

    uint32_t num_ch = getNumOfChannels();
    if(num_ch == 0) return;

    disableTrigger();
    setupSoftwareTrigger();

    const unsigned int len = shot[ *this->recordLength()];
    for(unsigned int i = 0; i < 2; i++) {
        DSORawRecord &rec = m_dsoRawRecordBanks[i];
        rec.record.resize(len * num_ch * (rec.isComplex ? 2 : 1));
        assert(rec.numCh == num_ch);
        if(isMemLockAvailable()) {
            mlock(&rec.record[0], rec.record.size() * sizeof(int32_t));
        }
    }
    m_recordBuf.resize(len * num_ch);
    if(isMemLockAvailable()) {
        mlock( &m_recordBuf[0], m_recordBuf.size() * sizeof(tRawAI));
    }

    m_interval = setupTimeBase();

    setupTrigger();

    startSequence();
}

template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::createChannels() {
    XScopedLock<XInterface> lock( *this->interface());
    Snapshot shot( *this);
    m_suspendRead = true;
    XScopedLock<XRecursiveMutex> lock2(m_readMutex);

    clearAll();

    setupChannels();

    uint32_t num_ch = getNumOfChannels();

    //accumlation buffer.
    for(unsigned int i = 0; i < 2; i++) {
        DSORawRecord &rec(m_dsoRawRecordBanks[i]);
        rec.acqCount = 0;
        rec.accumCount = 0;
        rec.numCh = num_ch;
        rec.isComplex = (shot[ *this->dRFMode()] == this->DRFMODE_COHERENT_SG);
    }

    if(num_ch == 0)  {
        return;
    }

    setupTiming();
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::clearStoredSoftwareTrigger() {
    uint64_t total_samps = 0;
    if(m_running)
        total_samps = getTotalSampsAcquired();
    m_softwareTrigger->clear(total_samps, 1.0 / m_interval);
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::onSoftTrigStarted(const shared_ptr<SoftwareTrigger> &) {
    XScopedLock<XInterface> lock( *this->interface());
    m_suspendRead = true;
    XScopedLock<XRecursiveMutex> lock2(m_readMutex);

    if(m_running) {
        m_running = false;
        stopAcquision();
    }

    const DSORawRecord &rec(m_dsoRawRecordBanks[m_dsoRawRecordBankLatest]);
    m_softwareTrigger->setBlankTerm(m_interval * rec.recordLength);
//	fprintf(stderr, "Virtual trig start.\n");

    uint32_t num_ch = getNumOfChannels();
    if(num_ch > 0) {
        setupTrigger();
        startAcquision();
        m_suspendRead = false;
        m_running = true;
    }
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::suspendAcquision() {
    m_suspendRead = true;
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::onForceTriggerTouched(const Snapshot &shot, XTouchableNode *) {
    XScopedLock<XInterface> lock( *this->interface());
    m_suspendRead = true;
    XScopedLock<XRecursiveMutex> lock2(m_readMutex);

    if(m_softwareTrigger) {
        if(m_running) {
            uint64_t total_samps = getTotalSampsAcquired();
            m_softwareTrigger->forceStamp(total_samps, 1.0 / m_interval);
            m_suspendRead = false;
        }
    }
    else {
        disableTrigger();
        startAcquision();
        m_suspendRead = false;
        m_running = true;
    }
}
template <class tDriver>
bool
XRealTimeAcqDSO<tDriver>::tryReadAISuspend(const atomic<bool> &terminated) {
    if(m_suspendRead) {
        m_readMutex.unlock();
        while(m_suspendRead && !terminated) msecsleep(30);
        m_readMutex.lock();
        return true;
    }
    return false;
}
template <class tDriver>
void *
XRealTimeAcqDSO<tDriver>::executeReadAI(const atomic<bool> &terminated) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::HIGHEST);
    while( !terminated) {
        try {
            acquire(terminated);
        }
        catch (XInterface::XInterfaceError &e) {
            e.print(this->getLabel());
            m_suspendRead = true;
        }
    }
    return NULL;
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::acquire(const atomic<bool> &terminated) {
    XScopedLock<XRecursiveMutex> lock(m_readMutex);
    while( !terminated) {

        if( !m_running) {
            tryReadAISuspend(terminated);
            msecsleep(30);
            return;
        }

        uint32_t num_ch = getNumOfChannels();
        if(num_ch == 0) {
            tryReadAISuspend(terminated);
            msecsleep(30);
            return;
        }

        const DSORawRecord &old_rec(m_dsoRawRecordBanks[m_dsoRawRecordBankLatest]);
        if(num_ch != old_rec.numCh)
            throw XInterface::XInterfaceError(i18n("Inconsistent channel number."), __FILE__, __LINE__);

        const unsigned int size = m_recordBuf.size() / num_ch;
        const double freq = 1.0 / m_interval;
        unsigned int cnt = 0;

        uint64_t samplecnt_at_trigger = 0;
        if(m_softwareTrigger) {
            shared_ptr<SoftwareTrigger> &vt(m_softwareTrigger);

            while( !terminated) {
                if(tryReadAISuspend(terminated))
                    return;
                uint64_t total_samps = getTotalSampsAcquired();
                samplecnt_at_trigger = vt->tryPopFront(total_samps, freq);
                if(samplecnt_at_trigger) {
                    if( !setReadPositionAbsolute(samplecnt_at_trigger - m_preTriggerPos)) {
                        gWarnPrint(i18n("Buffer Overflow."));
                        continue;
                    }
                    break;
                }
                msecsleep(lrint(1e3 * size * m_interval / 6));
            }
        }
        else {
            setReadPositionFirstPoint();
        }
        if(terminated)
            return;

        const unsigned int num_samps = std::min(size, 8192u);
        for(; cnt < size;) {
            int samps;
            samps = std::min(size - cnt, num_samps);
            while( !terminated) {
                if(tryReadAISuspend(terminated))
                    return;
                uint32_t space = getNumSampsToBeRead();
                if(space >= samps)
                    break;
                msecsleep(lrint(1e3 * (samps - space) * m_interval));
            }
            if(terminated)
                return;
            samps = readAcqBuffer(samps, &m_recordBuf[cnt * num_ch]);
            cnt += samps;
        }

        Snapshot shot( *this);
        const unsigned int av = shot[ *this->average()];
        const bool sseq = shot[ *this->singleSequence()];
        //obtain unlocked bank.
        int bank;
        for(;;) {
            bank = 1 - m_dsoRawRecordBankLatest;
            if(m_dsoRawRecordBanks[bank].tryLock())
                break;
            bank = m_dsoRawRecordBankLatest;
            if(m_dsoRawRecordBanks[bank].tryLock())
                break;
        }
        assert((bank >= 0) && (bank < 2));
        DSORawRecord &new_rec(m_dsoRawRecordBanks[bank]);
        unsigned int accumcnt = old_rec.accumCount;

        if( !sseq || (accumcnt < av)) {
            if( !m_softwareTrigger) {
                if(m_running) {
                    m_running = false;
                    stopAcquision();
                }
                startAcquision();
                m_running = true;
            }
        }

        cnt = std::min(cnt, old_rec.recordLength);
        new_rec.recordLength = cnt;
        //	num_ch = std::min(num_ch, old_rec->numCh);
        new_rec.numCh = num_ch;
        const unsigned int bufsize = new_rec.recordLength * num_ch;
        tRawAI *pbuf = &m_recordBuf[0];
        const int32_t *pold = &old_rec.record[0];
        int32_t *paccum = &new_rec.record[0];
        //Optimized accumlation.
        unsigned int div = bufsize / 4;
        unsigned int rest = bufsize % 4;
        if(new_rec.isComplex) {
            double ph = this->phaseOfRF(shot, samplecnt_at_trigger, m_interval);
            double cosph = cos(ph);
            double sinph = sin(ph);
            //real part.
            for(unsigned int i = 0; i < div; i++) {
                *paccum++ = *pold++ + *pbuf++ * cosph;
                *paccum++ = *pold++ + *pbuf++ * cosph;
                *paccum++ = *pold++ + *pbuf++ * cosph;
                *paccum++ = *pold++ + *pbuf++ * cosph;
            }
            for(unsigned int i = 0; i < rest; i++)
                *paccum++ = *pold++ + *pbuf++ * cosph;
            //imag part.
            for(unsigned int i = 0; i < div; i++) {
                *paccum++ = *pold++ + *pbuf++ * sinph;
                *paccum++ = *pold++ + *pbuf++ * sinph;
                *paccum++ = *pold++ + *pbuf++ * sinph;
                *paccum++ = *pold++ + *pbuf++ * sinph;
            }
            for(unsigned int i = 0; i < rest; i++)
                *paccum++ = *pold++ + *pbuf++ * cosph;
        }
        else {
            for(unsigned int i = 0; i < div; i++) {
                *paccum++ = *pold++ + *pbuf++;
                *paccum++ = *pold++ + *pbuf++;
                *paccum++ = *pold++ + *pbuf++;
                *paccum++ = *pold++ + *pbuf++;
            }
            for(unsigned int i = 0; i < rest; i++)
                *paccum++ = *pold++ + *pbuf++;
        }
        new_rec.acqCount = old_rec.acqCount + 1;
        accumcnt++;

        while( !sseq && (av <= m_record_av.size()) && !m_record_av.empty())  {
            if(new_rec.isComplex)
                throw XInterface::XInterfaceError(i18n("Moving average with coherent SG is not supported."), __FILE__, __LINE__);
            int32_t *paccum = &(new_rec.record[0]);
            tRawAI *psub = &(m_record_av.front()[0]);
            unsigned int div = bufsize / 4;
            unsigned int rest = bufsize % 4;
            for(unsigned int i = 0; i < div; i++) {
                *paccum++ -= *psub++;
                *paccum++ -= *psub++;
                *paccum++ -= *psub++;
                *paccum++ -= *psub++;
            }
            for(unsigned int i = 0; i < rest; i++)
                *paccum++ -= *psub++;
            m_record_av.pop_front();
            accumcnt--;
        }
        new_rec.accumCount = accumcnt;
        // substitute the record with the working set.
        m_dsoRawRecordBankLatest = bank;
        new_rec.unlock();
        if( !sseq) {
            m_record_av.push_back(m_recordBuf);
        }
        if(sseq && (accumcnt >= av))  {
            if(m_softwareTrigger) {
                if(m_running) {
                    m_suspendRead = true;
                }
            }
        }
    }
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::startSequence() {
    XScopedLock<XInterface> lock( *this->interface());
    m_suspendRead = true;
    XScopedLock<XRecursiveMutex> lock2(m_readMutex);

    {
        m_dsoRawRecordBankLatest = 0;
        for(unsigned int i = 0; i < 2; i++) {
            DSORawRecord &rec(m_dsoRawRecordBanks[i]);
            rec.acqCount = 0;
            rec.accumCount = 0;
        }
        DSORawRecord &rec(m_dsoRawRecordBanks[0]);
        if(!rec.numCh)
            return;
        rec.recordLength = rec.record.size() / rec.numCh / (rec.isComplex ? 2 : 1);
        memset(&rec.record[0], 0, rec.record.size() * sizeof(int32_t));
    }
    m_record_av.clear();

    if(m_softwareTrigger) {
        if( !m_lsnOnSoftTrigStarted)
            m_lsnOnSoftTrigStarted = m_softwareTrigger->onStart().connectWeakly(
                this->shared_from_this(), &XRealTimeAcqDSO<tDriver>::onSoftTrigStarted);
        if(m_running) {
            clearStoredSoftwareTrigger();
            m_suspendRead = false;
        }
        else {
            commitAcquision();
            this->statusPrinter()->printMessage(i18n("Restart the software-trigger source."));
        }
    }
    else {
        if(m_running) {
            m_running = false;
            stopAcquision();
        }
        uint32_t num_ch = getNumOfChannels();
        if(num_ch > 0) {
            startAcquision();
            m_suspendRead = false;
            m_running = true;
        }
    }
}

template <class tDriver>
int
XRealTimeAcqDSO<tDriver>::acqCount(bool *seq_busy) {
    const DSORawRecord &rec(m_dsoRawRecordBanks[m_dsoRawRecordBankLatest]);
    Snapshot shot( *this);
    *seq_busy = ((unsigned int)rec.acqCount < shot[ *this->average()]);
    return rec.acqCount;
}

template <class tDriver>
double
XRealTimeAcqDSO<tDriver>::getTimeInterval() {
    return m_interval;
}

template <class tDriver>
double
XRealTimeAcqDSO<tDriver>::aiRawToVolt(const double *pcoeff, double raw) {
    double x = 1.0;
    double y = 0.0;
    for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
        y += *(pcoeff++) * x;
        x *= raw;
    }
    return y;
}

template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::getWave(shared_ptr<typename tDriver::RawData> &writer, std::deque<XString> &) {
    XScopedLock<XInterface> lock( *this->interface());

    int bank;
    for(;;) {
        bank = m_dsoRawRecordBankLatest;
        if(m_dsoRawRecordBanks[bank].tryLock())
            break;
        bank = 1 - bank;
        if(m_dsoRawRecordBanks[bank].tryLock())
            break;
    }
    readBarrier();
    assert((bank >= 0) && (bank < 2));
    DSORawRecord &rec(m_dsoRawRecordBanks[bank]);

    if(rec.accumCount == 0) {
        rec.unlock();
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    }
    uint32_t num_ch = rec.numCh;
    uint32_t len = rec.recordLength;

    if(rec.isComplex)
        num_ch *= 2;
    writer->push((uint32_t)num_ch);
    writer->push((uint32_t)m_preTriggerPos);
    writer->push((uint32_t)len);
    writer->push((uint32_t)rec.accumCount);
    writer->push((double)m_interval);
    for(unsigned int ch = 0; ch < num_ch; ch++) {
        for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
            int ch_real = ch;
            if(rec.isComplex) ch_real = ch / 2;
            writer->push((double)m_coeffAI[ch_real][i]);
        }
    }
    const int32_t *p = &(rec.record[0]);
    const unsigned int size = len * num_ch;
    for(unsigned int i = 0; i < size; i++)
        writer->push((int32_t)*p++);
    XString str = getChannelInfoStrings();
    writer->insert(writer->end(), str.begin(), str.end());
    str = ""; //reserved/
    writer->insert(writer->end(), str.begin(), str.end());

    rec.unlock();
}
template <class tDriver>
void
XRealTimeAcqDSO<tDriver>::convertRaw(typename tDriver::RawDataReader &reader, Transaction &tr) throw (typename tDriver::XRecordError&) {
    const unsigned int num_ch = reader.template pop<uint32_t>();
    const unsigned int pretrig = reader.template pop<uint32_t>();
    const unsigned int len = reader.template pop<uint32_t>();
    const unsigned int accumCount = reader.template pop<uint32_t>();
    const double interval = reader.template pop<double>();

    tr[ *this].setParameters(num_ch, - (double)pretrig * interval, interval, len);

    double *wave[num_ch * 2];
    double coeff[num_ch * 2][CAL_POLY_ORDER];
    for(unsigned int j = 0; j < num_ch; j++) {
        for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
            coeff[j][i] = reader.template pop<double>();
        }

        wave[j] = tr[ *this].waveDisp(j);
    }

    const double prop = 1.0 / accumCount;
    for(unsigned int i = 0; i < len; i++) {
        for(unsigned int j = 0; j < num_ch; j++)
            *(wave[j])++ = aiRawToVolt(coeff[j], reader.template pop<int32_t>() * prop);
    }
}

template <class tDriver> void XRealTimeAcqDSO<tDriver>::onAverageChanged(const Snapshot &shot, XValueNodeBase *) {
    startSequence();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onSingleChanged(const Snapshot &shot, XValueNodeBase *) {
    startSequence();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onTrigPosChanged(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onTrigSourceChanged(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onTrigLevelChanged(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onTrigFallingChanged(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onTimeWidthChanged(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onTrace1Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onTrace2Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onTrace3Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onTrace4Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onVFullScale1Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onVFullScale2Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onVFullScale3Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onVFullScale4Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onVOffset1Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onVOffset2Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onVOffset3Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onVOffset4Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
template <class tDriver> void XRealTimeAcqDSO<tDriver>::onRecordLengthChanged(const Snapshot &shot, XValueNodeBase *) {
    createChannels();
}
