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
#include "dwfdso.h"
#include "interface.h"
#include "xwavengraph.h"
#include <dwf.h>

REGISTER_TYPE(XDriverList, DigilentWFDSO, "Digilent WaveForms AIN DSO");

//---------------------------------------------------------------------------
XDigilentWFInterface::XDigilentWFInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) :
    XInterface(name, runtime, driver), m_hdwf(hdwfNone) {
    int nDevice;
    if( !FDwfEnum(enumfilterAll, &nDevice))
        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
    for(int i = 0; i < nDevice; ++nDevice) {
        char name[32] = {};
        if( !FDwfEnumDeviceName(i, name))
            throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
        iterate_commit([=](Transaction &tr){
            tr[ *device()].add(name);
        });
    }
    address()->disable();
    port()->disable();
}
void
XDigilentWFInterface::open() throw (XInterfaceError &) {
    if( !FDwfDeviceOpen(***device(), &m_hdwf))
        throwWFError(i18n("WaveForms device open failed: "), __FILE__, __LINE__);
}
void
XDigilentWFInterface::close() throw (XInterfaceError &) {
    FDwfDeviceClose(hdwf());
    m_hdwf = hdwfNone;
}

bool
XDigilentWFInterface::isOpened() const {
    return hdwf() != hdwfNone;
}

void
XDigilentWFInterface::throwWFError(XString &&msg, const char *file, int line) {
    char buf[512] = {};
    FDwfGetLastErrorMsg(buf);
    throw XInterface::XInterfaceError(msg + buf, file, line);
}

XDigilentWFDSO::XDigilentWFDSO(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XDigilentWFDriver<XDSO>(name, runtime, tr_meas, meas) {

    trans( *recordLength()) = 8192;
}

void
XDigilentWFDSO::open() throw (XKameError &) {
    int num_ch;
    if( !FDwfAnalogInChannelCount(hdwf(), &num_ch))
        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
    iterate_commit([=](Transaction &tr){
        for(int i = 0; i < num_ch; ++i) {
            for(auto &&x: {trace1(), trace2(), trace3(), trace4()})
                tr[ *x].add(formatString("CH%d", i + 1));
            //        FDwfAnalogInChannelEnableSet(hdwf(), i, true);
        }
    });
    {
        double rgVoltsStep[32];
        int nSteps;
        if( !FDwfAnalogInChannelRangeSteps(hdwf(), rgVoltsStep, &nSteps))
            throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
        assert(nSteps <= 32);
        iterate_commit([=](Transaction &tr){
            for(int i = 0; i < nSteps; ++i) {
                for(auto &&x: {vFullScale1(), vFullScale2(), vFullScale3(), vFullScale4()})
                    tr[ *x].add(formatString("%f", rgVoltsStep[i]));
            }
        });
    }
    {
        int fstrigsrc;
        if( !FDwfAnalogInTriggerSourceInfo(hdwf(), &fstrigsrc))
            throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
        iterate_commit([=](Transaction &tr){
            tr[ *trigSource()].add("None");
            if(IsBitSet(fstrigsrc, trigsrcPC))
                tr[ *trigSource()].add("PC");
            if(IsBitSet(fstrigsrc, trigsrcDetectorAnalogIn))
                tr[ *trigSource()].add("DetectorAnalogIn");
            if(IsBitSet(fstrigsrc, trigsrcDetectorDigitalIn))
                tr[ *trigSource()].add("DetectorDigitalIn");
            if(IsBitSet(fstrigsrc, trigsrcAnalogIn))
                tr[ *trigSource()].add("AnalogIn");
            if(IsBitSet(fstrigsrc, trigsrcDigitalIn))
                tr[ *trigSource()].add("DigitalIn");
            if(IsBitSet(fstrigsrc, trigsrcAnalogOut1))
                tr[ *trigSource()].add("AnalogOut1");
            if(IsBitSet(fstrigsrc, trigsrcAnalogOut2))
                tr[ *trigSource()].add("AnalogOut2");
            if(IsBitSet(fstrigsrc, trigsrcAnalogOut3))
                tr[ *trigSource()].add("AnalogOut3");
            if(IsBitSet(fstrigsrc, trigsrcAnalogOut4))
                tr[ *trigSource()].add("AnalogOut4");
            if(IsBitSet(fstrigsrc, trigsrcExternal1))
                tr[ *trigSource()].add("External1");
            if(IsBitSet(fstrigsrc, trigsrcExternal2))
                tr[ *trigSource()].add("External2");
            if(IsBitSet(fstrigsrc, trigsrcExternal3))
                tr[ *trigSource()].add("External3");
            if(IsBitSet(fstrigsrc, trigsrcExternal4))
                tr[ *trigSource()].add("External4");
        });
    }

    m_threadReadAI.reset(new XThread<XDigilentWFDSO>(shared_from_this(),
        &XDigilentWFDSO::executeReadAI));
    m_threadReadAI->resume();

	start();

    createChannels(Snapshot( *this));
}
void
XDigilentWFDSO::close() throw (XKameError &) {
    XScopedLock<XInterface> lock( *interface());

    if(m_threadReadAI) {
        m_threadReadAI->terminate();
    }
    clearAcquision();

    iterate_commit([=](Transaction &tr){
        for(auto &&x: {trace1(), trace2(), trace3(), trace4()})
            tr[ *x].clear();
        for(auto &&x: {vFullScale1(), vFullScale2(), vFullScale3(), vFullScale4()})
            tr[ *x].clear();
        tr[ *trigSource()].clear();
    });

    m_record_av.clear();

    interface()->stop();
}
void
XDigilentWFDSO::clearAcquision() {
    XScopedLock<XInterface> lock( *interface());
    try {
        disableChannels();
    }
    catch (XInterface::XInterfaceError &e) {
        e.print();
    }
}
void
XDigilentWFDSO::disableChannels() {
    if(hdwf() == hdwfNone) return;
    int num_ch;
    if( !FDwfAnalogInChannelCount(hdwf(), &num_ch))
        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
    for(int ch = 0; ch < num_ch; ++ch) {
        if( !FDwfAnalogInChannelEnableSet( hdwf(), ch, false))
            throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
    }
    m_preTriggerPos = 0;
    m_accum.reset();
}

void
XDigilentWFDSO::createChannels(const Snapshot &shot) {
    XScopedLock<XInterface> lock( *interface());

    disableChannels();

    //accumlation buffer.
    DSORawRecord rec;
    rec.acqCount = 0;
    rec.accumCount = 0;

    auto traces =
        {std::make_tuple(trace1(), vFullScale1(), vOffset1()), std::make_tuple(trace2(), vFullScale2(), vOffset2()),
        std::make_tuple(trace3(), vFullScale3(), vOffset3()), std::make_tuple(trace4(), vFullScale4(), vOffset4())};
    for(auto &&trace: traces) {
        int ch = shot[ *std::get<0>(trace)];
        if(ch >= 0) {
            rec.channels.push_back(ch);
            if( !FDwfAnalogInChannelEnableSet( hdwf(), ch, true))
                throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
            if( !FDwfAnalogInChannelOffsetSet( hdwf(), ch, shot[ *std::get<2>(trace)]))
                throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
            if( !FDwfAnalogInChannelRangeSet(hdwf(), ch, shot[ *std::get<1>(trace)]))
                throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
        }
    }
    m_accum = make_local_shared<DSORawRecord>(std::move(rec));

    setupTiming(shot);
}
void
XDigilentWFDSO::setupTiming(const Snapshot &shot) {
    XScopedLock<XInterface> lock( *interface());
    int max_len;
    if( !FDwfAnalogInBufferSizeInfo(hdwf(), NULL, &max_len))
        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
    int len = shot[ *recordLength()];
    len = std::min(len, max_len);
    if( !FDwfAnalogInBufferSizeSet(hdwf(), len))
        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
    if( !FDwfAnalogInBufferSizeGet(hdwf(), &len))
        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
    local_shared_ptr<DSORawRecord> rec = m_accum;
    if( !rec) return;
    rec->record.resize(len * rec->numCh());

//    if( !FDwfAnalogInRecordLengthSet(hdwf(), len))
//        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
//    if( !FDwfAnalogInRecordLengthGet(hdwf(), &len))
//        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);

    if( !FDwfAnalogInFrequencySet(hdwf(), len / shot[ *timeWidth()])) //Hz
        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
    double freq;
    if( !FDwfAnalogInFrequencyGet(hdwf(), &freq)) //Hz
        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
    rec->interval = 1.0/freq;

    setupTrigger(shot);

    startSequence();
}
void
XDigilentWFDSO::setupTrigger(const Snapshot &shot) {
    XScopedLock<XInterface> lock( *interface());
    if( !FDwfAnalogInTriggerSourceSet(hdwf(), shot[ *trigSource()]))
        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
//    if( !FDwfAnalogInTriggerAutoTimeoutSet(hdwf(), 10.0))
//        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
//    if( !FDwfAnalogInTriggerChannelSet(hdwf(), 0))
//        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
//    if( !FDwfAnalogInTriggerTypeSet(hdwf(), shot[ *trig()]))
//        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
    if( !FDwfAnalogInTriggerLevelSet(hdwf(), shot[ *trigLevel()]))
        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
    if( !FDwfAnalogInTriggerConditionSet(hdwf(), shot[ *trigFalling()] ? trigcondFallingNegative : trigcondRisingPositive))
        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);

    local_shared_ptr<DSORawRecord> rec = m_accum;
    if( !rec) return;
    unsigned int pretrig = lrint(shot[ *trigPos()] / 100.0 * rec->recordLength());
    m_preTriggerPos = pretrig;
    double pos = (m_preTriggerPos - rec->recordLength() / 2) * rec->interval;
    if( !FDwfAnalogInTriggerPositionSet(hdwf(), pos))
        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
}
void
XDigilentWFDSO::startSequence() {
    XScopedLock<XInterface> lock( *interface());
    m_record_av.clear();
    local_shared_ptr<DSORawRecord> rec = m_accum;
    if( !rec) return;
    rec->acqCount = 0;
    rec->accumCount = 0;
    if(rec->numCh() == 0) return;
    if( !FDwfAnalogInConfigure(hdwf(), false, true))
        throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
}

void *
XDigilentWFDSO::executeReadAI(const atomic<bool> &terminated) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::HIGHEST);
    while( !terminated) {
        try {
            acquire(terminated);
        }
        catch (XInterface::XInterfaceError &e) {
            e.print(getLabel());
        }
    }
    return NULL;
}
void
XDigilentWFDSO::acquire(const atomic<bool> &terminated) {
    XScopedLock<XInterface> lock( *interface());

    //RCU pattern.
    local_shared_ptr<DSORawRecord> newrec(m_accum);
    if( !newrec) {
        msecsleep(10);
        return;
    }
    //allocations prior to spining
    newrec.reset( new DSORawRecord( *newrec));
    int num_ch = newrec->numCh();
    const unsigned int size = newrec->recordLength();
    std::vector<double> record_buf(size * num_ch);

    while( !terminated) {
        STS sts;
        if( !FDwfAnalogInStatus(hdwf(), true, &sts))
            throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
        if(sts != DwfStateDone) {
            double term = size * newrec->interval;
            if((sts != DwfStateTriggered) || (term > 20e-3)) {
                msecsleep(std::min(50.0, term / 2 * 1000.0));
                return;
            }
            //spining is preferred.
            msecsleep(term / 10 * 1000.0);
            continue;
        }
        break;
    }
    double *pbuf = &record_buf[0];
    for(auto &&ch : newrec->channels) {
        if( !FDwfAnalogInStatusData( hdwf(), ch, pbuf, size))
            throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
        pbuf += size;
    }

    Snapshot shot( *this);
    const unsigned int av = shot[ *average()];
    const bool sseq = shot[ *singleSequence()];

    if( !sseq || (newrec->accumCount < av)) {
        if( !FDwfAnalogInConfigure(hdwf(), false, true))
            throwWFError(i18n("WaveForms error: "), __FILE__, __LINE__);
    }

    const unsigned int bufsize = newrec->recordLength() * num_ch;
    pbuf = &record_buf[0];
    double *paccum = &newrec->record[0];
    //Optimized accumlation.
    unsigned int div = bufsize / 4;
    unsigned int rest = bufsize % 4;
    for(unsigned int i = 0; i < div; i++) {
        *paccum++ += *pbuf++;
        *paccum++ += *pbuf++;
        *paccum++ += *pbuf++;
        *paccum++ += *pbuf++;
    }
    for(unsigned int i = 0; i < rest; i++)
        *paccum++ += *pbuf++;

    newrec->acqCount++;
    newrec->accumCount++;

    while( !sseq && (av <= m_record_av.size()) && !m_record_av.empty())  {
        double *paccum = &(newrec->record[0]);
        double *psub = &(m_record_av.front()[0]);
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
        newrec->accumCount--;
    }
    m_accum = std::move(newrec);
    if( !sseq) {
        m_record_av.push_back(std::move(record_buf));
    }
}
void 
XDigilentWFDSO::onAverageChanged(const Snapshot &shot, XValueNodeBase *) {
    setupTiming(Snapshot( *this));
}

void
XDigilentWFDSO::onSingleChanged(const Snapshot &shot, XValueNodeBase *) {
    setupTiming(Snapshot( *this));
}
void
XDigilentWFDSO::onTrigSourceChanged(const Snapshot &shot, XValueNodeBase *) {
    setupTrigger(Snapshot( *this));
}
void
XDigilentWFDSO::onTrigPosChanged(const Snapshot &shot, XValueNodeBase *) {
    setupTrigger(Snapshot( *this));
}
void
XDigilentWFDSO::onTrigLevelChanged(const Snapshot &shot, XValueNodeBase *) {
    setupTrigger(Snapshot( *this));
}
void
XDigilentWFDSO::onTrigFallingChanged(const Snapshot &shot, XValueNodeBase *) {
    setupTrigger(Snapshot( *this));
}
void
XDigilentWFDSO::onTimeWidthChanged(const Snapshot &shot, XValueNodeBase *) {
    setupTiming(Snapshot( *this));
}
void
XDigilentWFDSO::onVFullScale1Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels(Snapshot( *this));
}
void
XDigilentWFDSO::onVFullScale2Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels(Snapshot( *this));
}
void
XDigilentWFDSO::onVFullScale3Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels(Snapshot( *this));
}
void
XDigilentWFDSO::onVFullScale4Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels(Snapshot( *this));
}
void
XDigilentWFDSO::onVOffset1Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels(Snapshot( *this));
}
void
XDigilentWFDSO::onVOffset2Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels(Snapshot( *this));
}
void
XDigilentWFDSO::onVOffset3Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels(Snapshot( *this));
}
void
XDigilentWFDSO::onVOffset4Changed(const Snapshot &shot, XValueNodeBase *) {
    createChannels(Snapshot( *this));
}
void
XDigilentWFDSO::onRecordLengthChanged(const Snapshot &shot, XValueNodeBase *) {
    setupTiming(Snapshot( *this));
}
void
XDigilentWFDSO::onForceTriggerTouched(const Snapshot &shot, XTouchableNode *) {
    createChannels(Snapshot( *this));
}

int
XDigilentWFDSO::acqCount(bool *seq_busy) {
    Snapshot shot( *this);
    local_shared_ptr<DSORawRecord> rec = m_accum;
    if( !rec) return 0;
    *seq_busy = ((unsigned int)rec->acqCount < shot[ *average()]);
    return rec->acqCount;
}

double
XDigilentWFDSO::getTimeInterval() {
    local_shared_ptr<DSORawRecord> rec = m_accum;
    if( !rec) return 0.0;
    return rec->interval;
}

void
XDigilentWFDSO::getWave(shared_ptr<RawData> &writer, std::deque<XString> &channels) {
    local_shared_ptr<DSORawRecord> rec = m_accum;

    if( !rec || (rec->accumCount == 0)) {
        throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
    }
    unsigned int num_ch = rec->numCh();
    unsigned int len = rec->recordLength();

    writer->push((uint32_t)num_ch);
    writer->push((uint32_t)m_preTriggerPos);
    writer->push((uint32_t)len);
    writer->push((uint32_t)rec->accumCount);
    writer->push((double)rec->interval);
    const double *p = &(rec->record[0]);
    const unsigned int size = len * num_ch;
    for(unsigned int i = 0; i < size; i++)
        writer->push<double>( *p++);
    XString str = ""; //reserved/
    writer->insert(writer->end(), str.begin(), str.end());
}
void
XDigilentWFDSO::convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
    const unsigned int num_ch = reader.pop<uint32_t>();
    const unsigned int pretrig = reader.pop<uint32_t>();
    const unsigned int len = reader.pop<uint32_t>();
    const unsigned int accumCount = reader.pop<uint32_t>();
    const double interval = reader.pop<double>();

    tr[ *this].setParameters(num_ch, - (double)pretrig * interval, interval, len);

    const double prop = 1.0 / accumCount;
    for(unsigned int j = 0; j < num_ch; j++) {
        double *pwave = tr[ *this].waveDisp(j);
        for(unsigned int i = 0; i < len; i++)
            *pwave++ = reader.pop<double>() * prop;
    }
}
