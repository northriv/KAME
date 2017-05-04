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

#include "thamwayrealtimedso.h"
#include "dsorealtimeacq_impl.h"

constexpr double SMPL_PER_SEC = 5000000.0; //5MSmps/s

REGISTER_TYPE(XDriverList, ThamwayPROT3DSO, "Thamway PROT3 digital streaming DSO");

#define NUM_MAX_CH 2

XThamwayPROT3DSO::XThamwayPROT3DSO(const char *name, bool runtime,
    Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XRealTimeAcqDSO<XCharDeviceDriver<XDSO, XThamwayFX3USBInterface>>(name, runtime, tr_meas, meas) {

    iterate_commit([=](Transaction &tr){
        for(auto &&x: {vFullScale1(), vFullScale2(), vFullScale3(), vFullScale4()}) {
            x->disable();
        }
    });
    vOffset1()->disable();
    vOffset2()->disable();
    vOffset3()->disable();
    vOffset4()->disable();
}

void
XThamwayPROT3DSO::startAcquision() {
//    CHECK_DAQMX_RET(DAQmxStartTask(m_task));
}
void
XThamwayPROT3DSO::commitAcquision() {
//    CHECK_DAQMX_RET(DAQmxTaskControl(m_task, DAQmx_Val_Task_Commit));
}
void
XThamwayPROT3DSO::stopAcquision() {
//    CHECK_DAQMX_RET(DAQmxStopTask(m_task));
}

void
XThamwayPROT3DSO::clearAcquision() {
//        CHECK_DAQMX_RET(DAQmxClearTask(m_task));
//    m_task = TASK_UNDEF;
}

unsigned int
XThamwayPROT3DSO::getNumOfChannels() {
//    CHECK_DAQMX_RET(DAQmxGetReadNumChans(m_task, &num_ch));
}

XString
XThamwayPROT3DSO::getChannelInfoStrings() {
//    CHECK_DAQMX_RET(DAQmxGetReadChannelsToRead(m_task, buf, sizeof(buf)));
}

std::deque<XString>
XThamwayPROT3DSO::hardwareTriggerNames() {
    return {};
}

double
XThamwayPROT3DSO::setupTimeBase() {
    return 1.0 / SMPL_PER_SEC;
}

void
XThamwayPROT3DSO::setupChannels() {
}

void
XThamwayPROT3DSO::setupHardwareTrigger() {
}

void
XThamwayPROT3DSO::disableHardwareTriggers() {
}

uint64_t
XThamwayPROT3DSO::getTotalSampsAcquired() {
//    uInt64 total_samps;
//    CHECK_DAQMX_RET(DAQmxGetReadTotalSampPerChanAcquired(m_task, &total_samps));
//    return total_samps;
}

uint32_t
XThamwayPROT3DSO::getNumSampsToBeRead() {
//    uInt32 space;
//    int ret = DAQmxGetReadAvailSampPerChan(m_task, &space);
//    return space;
}

bool
XThamwayPROT3DSO::setReadPositionAbsolute(uint64_t pos) {
//    uInt32 bufsize;
//    CHECK_DAQMX_RET(DAQmxGetBufInputBufSize(m_task, &bufsize));
//    uint64_t total_samps = getTotalSampsAcquired();
//    if(total_samps - pos > bufsize * 4 / 5) {
//        return false;
//    }
//    //set read pos.
//    int16 tmpbuf[NUM_MAX_CH];
//    int32 samps;
//    CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_MostRecentSamp));
//    CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, -1));
//    CHECK_DAQMX_RET(DAQmxReadBinaryI16(m_task, 1,
//                                       0, DAQmx_Val_GroupByScanNumber,
//                                       tmpbuf, NUM_MAX_CH, &samps, NULL
//                                       ));
//    CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_CurrReadPos));
//    CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, 0));
//    uInt64 curr_rdpos;
//    CHECK_DAQMX_RET(DAQmxGetReadCurrReadPos(m_task, &curr_rdpos));
//    int32 offset = pos - curr_rdpos;
//    CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, offset));
//    //					fprintf(stderr, "hit! %d %d %d\n", (int)offset, (int)lastcnt, (int)m_preTriggerPos);
//    return true;
}
void
XThamwayPROT3DSO::setReadPositionFirstPoint() {
//    if(m_preTriggerPos) {
//        CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_FirstPretrigSamp));
//    }
//    else {
//        CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_FirstSample));
//    }
//    CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, 0));
}

uint32_t
XThamwayPROT3DSO::readAcqBuffer(uint32_t size, tRawAI *buf) {
//    int32 samps;
//    int32_t num_ch = getNumOfChannels();
//    CHECK_DAQMX_RET(DAQmxReadBinaryI16(m_task, size,
//       0.0, DAQmx_Val_GroupByScanNumber,
//       buf, size * num_ch, &samps, NULL
//       ));
//    CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_CurrReadPos));
//    CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, 0));
//    return samps;
}


void
XThamwayPROT3DSO::open() throw (XKameError &) {
    XRealTimeAcqDSO<XCharDeviceDriver<XDSO, XThamwayFX3USBInterface>>::open();
}
void
XThamwayPROT3DSO::close() throw (XKameError &) {
    XScopedLock<XInterface> lock( *interface());

    iterate_commit([=](Transaction &tr){
        for(auto &&x: {trace1(), trace2(), trace3(), trace4()})
            tr[ *x].clear();
    });
    XRealTimeAcqDSO<XCharDeviceDriver<XDSO, XThamwayFX3USBInterface>>::close();
}

void*
XThamwayPROT3DSO::execute(const atomic<bool> &) {
    return nullptr;
}

