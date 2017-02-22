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
//! \todo sub-sampling rate synchronization.

#include "nidaqdso.h"
#include "dsorealtimeacq_impl.h"

#ifndef HAVE_NI_DAQMX
	#define DAQmx_Val_FallingSlope 0
	#define DAQmx_Val_RisingSlope 0
	#define DAQmx_Val_DigEdge 0
	#define DAQmxGetReadAvailSampPerChan(x,y) 0
#endif //HAVE_NI_DAQMX

#include <qmessagebox.h>
#include "xwavengraph.h"

REGISTER_TYPE(XDriverList, NIDAQmxDSO, "National Instruments DAQ as DSO");

#define TASK_UNDEF ((TaskHandle)-1)
#define NUM_MAX_CH 4

XNIDAQmxDSO::XNIDAQmxDSO(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
    XRealTimeAcqDSO<XNIDAQmxDriver<XDSO>>(name, runtime, ref(tr_meas), meas),
	m_task(TASK_UNDEF) {

	iterate_commit([=](Transaction &tr){
        for(auto &&x: {vFullScale1(), vFullScale2(), vFullScale3(), vFullScale4()}) {
            tr[ *x].add({"0.4", "1", "2", "4", "10", "20", "40", "84"});
            tr[ *x] = "20";
        }
    });
	vOffset1()->disable();
	vOffset2()->disable();
	vOffset3()->disable();
	vOffset4()->disable();
}

void
XNIDAQmxDSO::startAcquision() {
    CHECK_DAQMX_RET(DAQmxStartTask(m_task));
}
void
XNIDAQmxDSO::commitAcquision() {
    CHECK_DAQMX_RET(DAQmxTaskControl(m_task, DAQmx_Val_Task_Commit));
}
void
XNIDAQmxDSO::stopAcquision() {
    if(m_task != TASK_UNDEF)
        CHECK_DAQMX_RET(DAQmxStopTask(m_task));
}

void
XNIDAQmxDSO::clearAcquision() {
    if(m_task != TASK_UNDEF) {
        CHECK_DAQMX_RET(DAQmxClearTask(m_task));
    }
    m_task = TASK_UNDEF;
}

unsigned int
XNIDAQmxDSO::getNumOfChannels() {
    uInt32 num_ch;
    CHECK_DAQMX_RET(DAQmxGetReadNumChans(m_task, &num_ch));
    return num_ch;
}

XString
XNIDAQmxDSO::getChannelInfoStrings() {
    char buf[2048];
    CHECK_DAQMX_RET(DAQmxGetReadChannelsToRead(m_task, buf, sizeof(buf)));
    return buf;
}

std::deque<XString>
XNIDAQmxDSO::hardwareTriggerNames() {
    std::deque<XString> list;
    XString series = interface()->productSeries();
    char buf[2048];
    {
        CHECK_DAQMX_RET(DAQmxGetDevAIPhysicalChans(interface()->devName(), buf, sizeof(buf)));
        XNIDAQmxInterface::parseList(buf, list);
    }
    //M series
    const char* sc_m[] = {
        "PFI0", "PFI1", "PFI2", "PFI3", "PFI4", "PFI5", "PFI6", "PFI7",
        "PFI8", "PFI9", "PFI10", "PFI11", "PFI12", "PFI13", "PFI14", "PFI15",
        "Ctr0InternalOutput", "Ctr1InternalOutput",
        "Ctr0Source",
        "Ctr0Gate",
        "Ctr1Source",
        "Ctr1Gate",
        "FrequencyOutput",
        0L};
    //S series
    const char* sc_s[] = {
        "PFI0", "PFI1", "PFI2", "PFI3", "PFI4", "PFI5", "PFI6", "PFI7",
        "PFI8", "PFI9",
        "Ctr0InternalOutput",
        "OnboardClock",
        "Ctr0Source",
        "Ctr0Gate",
        0L};
    const char **sc = sc_m;
    if(series == "S")
        sc = sc_s;
    for(const char **it = sc; *it; it++) {
        XString str(formatString("/%s/%s", interface()->devName(), *it));
        list.emplace_back(str);
    }
    return list;
}

double
XNIDAQmxDSO::setupTimeBase() {
    uInt32 onbrd_size;
    CHECK_DAQMX_RET(DAQmxGetBufInputOnbrdBufSize(m_task, &onbrd_size));
    fprintf(stderr, "Using on-brd bufsize=%d\n", (int)onbrd_size);
    Snapshot shot( *this);
    uint32_t num_ch = getNumOfChannels();
    const unsigned int len = shot[ *this->recordLength()];
    unsigned int bufsize = len;
    if(hasSoftwareTrigger()) {
        bufsize = std::max(bufsize * 8, (unsigned int)lrint((len / shot[ *timeWidth()]) * 1.0));
        bufsize = std::max(bufsize, (unsigned int)(onbrd_size / num_ch));
    }

    //! debug!
    //		formatString("/%s/Ctr0InternalOutput", interface()->devName()),
    CHECK_DAQMX_RET(
        DAQmxCfgSampClkTiming(m_task,
          NULL, // internal source
          len / shot[ *timeWidth()],
          DAQmx_Val_Rising,
          !hasSoftwareTrigger() ? DAQmx_Val_FiniteSamps : DAQmx_Val_ContSamps,
          bufsize
          ));

    interface()->synchronizeClock(m_task);

    {
        uInt32 size;
        CHECK_DAQMX_RET(DAQmxGetBufInputBufSize(m_task, &size));
        fprintf(stderr, "Using buffer size of %d\n", (int)size);
        if(size != bufsize) {
            fprintf(stderr, "Try to modify buffer size from %d to %d\n", (int)size, (int)bufsize);
            CHECK_DAQMX_RET(DAQmxCfgInputBuffer(m_task, bufsize));
        }
    }

    CHECK_DAQMX_RET(DAQmxSetExportedSampClkOutputTerm(m_task, formatString("/%s/PFI7", interface()->devName()).c_str()));
//    m_sampleClockRoute.reset(new XNIDAQmxInterface::XNIDAQmxRoute(
//    										formatString("/%s/ai/SampleClock", interface()->devName()).c_str(),
//    										formatString("/%s/PFI7", interface()->devName()).c_str()));

    float64 rate;
    //	CHECK_DAQMX_RET(DAQmxGetRefClkRate(m_task, &rate));
    //	dbgPrint(QString("Reference Clk rate = %1.").arg(rate));
    CHECK_DAQMX_RET(DAQmxGetSampClkRate(m_task, &rate));
    return 1.0 / rate;
}

void
XNIDAQmxDSO::setupChannels() {
    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_task));
    assert(m_task != TASK_UNDEF);

    Snapshot shot( *this);
    auto traces =
        {std::make_tuple(trace1(), vFullScale1(), vOffset1()), std::make_tuple(trace2(), vFullScale2(), vOffset2()),
        std::make_tuple(trace3(), vFullScale3(), vOffset3()), std::make_tuple(trace4(), vFullScale4(), vOffset4())};
    int ch_num = 0;
    for(auto &&trace: traces) {
        int ch = shot[ *std::get<0>(trace)];
        if(ch >= 0) {
            auto str = shot[ *std::get<0>(trace)].to_str();
            CHECK_DAQMX_RET(
                DAQmxCreateAIVoltageChan(m_task,
                     str.c_str(),
                     "",
                     DAQmx_Val_Cfg_Default,
                     -atof(shot[ *std::get<1>(trace)].to_str().c_str()) / 2.0,
                     atof(shot[ *std::get<1>(trace)].to_str().c_str()) / 2.0,
                     DAQmx_Val_Volts,
                     NULL
                     ));

            //obtain range info.
            for(unsigned int i = 0; i < CAL_POLY_ORDER; i++)
                m_coeffAI[ch_num][i] = 0.0;
            CHECK_DAQMX_RET(
                DAQmxGetAIDevScalingCoeff(m_task,
                      str.c_str(),
                      m_coeffAI[ch_num], CAL_POLY_ORDER));
            ch_num++;
        }
    }
    CHECK_DAQMX_RET(DAQmxRegisterDoneEvent(m_task, 0, &XNIDAQmxDSO::onTaskDone_, this));
}

void
XNIDAQmxDSO::setupHardwareTrigger() {
    Snapshot shot( *this);
    XString atrig;
    XString dtrig;
    XString src = shot[ *trigSource()].to_str();
    unsigned int pretrig = m_preTriggerPos;

    char buf[2048];
    {
        CHECK_DAQMX_RET(DAQmxGetDevAIPhysicalChans(interface()->devName(), buf, sizeof(buf)));
        std::deque<XString> chans;
        XNIDAQmxInterface::parseList(buf, chans);
        for(auto&& x: chans) {
            if(src == x)
                atrig = x;
        }
    }
    if( !atrig.length())
        dtrig = src;

    int32 trig_spec = shot[ *trigFalling()] ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope;

    if(hasSoftwareTrigger()) {
        dtrig = softwareTrigger()->armTerm();
        trig_spec = DAQmx_Val_RisingSlope;
        pretrig = 0;
        CHECK_DAQMX_RET(DAQmxSetReadOverWrite(m_task, DAQmx_Val_OverwriteUnreadSamps));
        softwareTrigger()->setPersistentCoherentMode(shot[ *dRFMode()] >= 1);
    }

    //Small # of pretriggers is not allowed for ReferenceTrigger.
    if( !hasSoftwareTrigger() && (pretrig < 2)) {
        pretrig = 0;
        m_preTriggerPos = pretrig;
    }

    if( !pretrig) {
        if(atrig.length()) {
            CHECK_DAQMX_RET(
                DAQmxCfgAnlgEdgeStartTrig(m_task,
                                          atrig.c_str(), trig_spec, shot[ *trigLevel()]));
        }
        if(dtrig.length()) {
            CHECK_DAQMX_RET(
                DAQmxCfgDigEdgeStartTrig(m_task,
                                         dtrig.c_str(), trig_spec));
        }
    }
    else {
        if(atrig.length()) {
            CHECK_DAQMX_RET(
                DAQmxCfgAnlgEdgeRefTrig(m_task,
                                        atrig.c_str(), trig_spec, shot[ *trigLevel()], pretrig));
        }
        if(dtrig.length()) {
            CHECK_DAQMX_RET(
                DAQmxCfgDigEdgeRefTrig(m_task,
                                       dtrig.c_str(), trig_spec, pretrig));
        }
    }

    char ch[256];
    CHECK_DAQMX_RET(DAQmxGetTaskChannels(m_task, ch, sizeof(ch)));
    if(interface()->productFlags() & XNIDAQmxInterface::FLAG_BUGGY_DMA_AI) {
        CHECK_DAQMX_RET(DAQmxSetAIDataXferMech(m_task, ch,
                                               DAQmx_Val_Interrupts));
    }
    if(interface()->productFlags() & XNIDAQmxInterface::FLAG_BUGGY_XFER_COND_AI) {
        uInt32 bufsize;
        CHECK_DAQMX_RET(DAQmxGetBufInputOnbrdBufSize(m_task, &bufsize));
        CHECK_DAQMX_RET(
            DAQmxSetAIDataXferReqCond(m_task, ch,
                                      (hasSoftwareTrigger() || (bufsize/2 < sizeofRecordBuf())) ? DAQmx_Val_OnBrdMemNotEmpty :
                                      DAQmx_Val_OnBrdMemMoreThanHalfFull));
    }

}

void
XNIDAQmxDSO::disableHardwareTriggers() {
    if(m_task != TASK_UNDEF) {
        uInt32 num_ch;
        CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));
        if(num_ch) {
            CHECK_DAQMX_RET(DAQmxDisableStartTrig(m_task));
            CHECK_DAQMX_RET(DAQmxDisableRefTrig(m_task));
        }
    }
}

uint64_t
XNIDAQmxDSO::getTotalSampsAcquired() {
    uInt64 total_samps;
    CHECK_DAQMX_RET(DAQmxGetReadTotalSampPerChanAcquired(m_task, &total_samps));
    return total_samps;
}

uint32_t
XNIDAQmxDSO::getNumSampsToBeRead() {
    uInt32 space;
    int ret = DAQmxGetReadAvailSampPerChan(m_task, &space);
    return space;
}

bool
XNIDAQmxDSO::setReadPositionAbsolute(uint64_t pos) {
    uInt32 bufsize;
    CHECK_DAQMX_RET(DAQmxGetBufInputBufSize(m_task, &bufsize));
    uint64_t total_samps = getTotalSampsAcquired();
    if(total_samps - pos > bufsize * 4 / 5) {
        return false;
    }
    //set read pos.
    int16 tmpbuf[NUM_MAX_CH];
    int32 samps;
    CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_MostRecentSamp));
    CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, -1));
    CHECK_DAQMX_RET(DAQmxReadBinaryI16(m_task, 1,
                                       0, DAQmx_Val_GroupByScanNumber,
                                       tmpbuf, NUM_MAX_CH, &samps, NULL
                                       ));
    CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_CurrReadPos));
    CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, 0));
    uInt64 curr_rdpos;
    CHECK_DAQMX_RET(DAQmxGetReadCurrReadPos(m_task, &curr_rdpos));
    int32 offset = pos - curr_rdpos;
    CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, offset));
    //					fprintf(stderr, "hit! %d %d %d\n", (int)offset, (int)lastcnt, (int)m_preTriggerPos);
    return true;
}
void
XNIDAQmxDSO::setReadPositionFirstPoint() {
    if(m_preTriggerPos) {
        CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_FirstPretrigSamp));
    }
    else {
        CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_FirstSample));
    }
    CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, 0));
}

uint32_t
XNIDAQmxDSO::readAcqBuffer(uint32_t size, tRawAI *buf) {
    int32 samps;
    int32_t num_ch = getNumOfChannels();
    CHECK_DAQMX_RET(DAQmxReadBinaryI16(m_task, size,
       0.0, DAQmx_Val_GroupByScanNumber,
       buf, size * num_ch, &samps, NULL
       ));
    CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_CurrReadPos));
    CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, 0));
    return samps;
}


void
XNIDAQmxDSO::open() throw (XKameError &) {
	XScopedLock<XInterface> lock( *interface());
	char buf[2048];
	{
		CHECK_DAQMX_RET(DAQmxGetDevAIPhysicalChans(interface()->devName(), buf, sizeof(buf)));
		std::deque<XString> chans;
		XNIDAQmxInterface::parseList(buf, chans);
		iterate_commit([=](Transaction &tr){
            for(auto it = chans.cbegin(); it != chans.cend(); ++it) {
                for(auto &&x: {trace1(), trace2(), trace3(), trace4()})
                    tr[ *x].add(it->c_str());
            }
        });
	}
    XRealTimeAcqDSO<XNIDAQmxDriver<XDSO>>::open();
}
void
XNIDAQmxDSO::close() throw (XKameError &) {
	XScopedLock<XInterface> lock( *interface());

    iterate_commit([=](Transaction &tr){
        for(auto &&x: {trace1(), trace2(), trace3(), trace4()})
            tr[ *x].clear();
    });
    XRealTimeAcqDSO<XNIDAQmxDriver<XDSO>>::close();
}

int32
XNIDAQmxDSO::onTaskDone_(TaskHandle task, int32 status, void *data) {
	XNIDAQmxDSO *obj = static_cast<XNIDAQmxDSO*>(data);
	obj->onTaskDone(task, status);
	return status;
}
void
XNIDAQmxDSO::onTaskDone(TaskHandle /*task*/, int32 status) {
	if(status) {
		gErrPrint(getLabel() + XNIDAQmxInterface::getNIDAQmxErrMessage(status));
        suspendAcquision();
	}
}

