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
//! \todo sub-sampling rate synchronization.

#include "nidaqdso.h"

#ifndef HAVE_NI_DAQMX
	#define DAQmx_Val_FallingSlope 0
	#define DAQmx_Val_RisingSlope 0
	#define DAQmx_Val_DigEdge 0
	#define DAQmxGetReadAvailSampPerChan(x,y) 0
#endif //HAVE_NI_DAQMX

#include <qmessagebox.h>
#include <kmessagebox.h>
#include "xwavengraph.h"

REGISTER_TYPE(XDriverList, NIDAQmxDSO, "National Instruments DAQ as DSO");

#define TASK_UNDEF ((TaskHandle)-1)
#define NUM_MAX_CH 4

XNIDAQmxDSO::XNIDAQmxDSO(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
	XNIDAQmxDriver<XDSO>(name, runtime, ref(tr_meas), meas),
	m_dsoRawRecordBankLatest(0),
	m_task(TASK_UNDEF), m_taskCounterOrigin(TASK_UNDEF) {

	const char* sc[] = {"0.4", "1", "2", "4", "10", "20", "40", "84", 0L};
	for(Transaction tr( *this);; ++tr) {
		tr[ *recordLength()] = 2000;
		tr[ *timeWidth()] = 1e-2;
		tr[ *average()] = 1;
		for(int i = 0; sc[i]; i++) {
			tr[ *vFullScale1()].add(sc[i]);
			tr[ *vFullScale2()].add(sc[i]);
			tr[ *vFullScale3()].add(sc[i]);
			tr[ *vFullScale4()].add(sc[i]);
		}
		tr[ *vFullScale1()] = "20";
		tr[ *vFullScale2()] = "20";
		tr[ *vFullScale3()] = "20";
		tr[ *vFullScale4()] = "20";
		if(tr.commit())
			break;
	}
	if(g_bUseMLock) {
		const void *FIRST_OF_MLOCK_MEMBER = &m_recordBuf;
		const void *LAST_OF_MLOCK_MEMBER = &m_task;
		//Suppress swapping.
		mlock(FIRST_OF_MLOCK_MEMBER, (size_t)LAST_OF_MLOCK_MEMBER - (size_t)FIRST_OF_MLOCK_MEMBER);
	}

	vOffset1()->disable();
	vOffset2()->disable();
	vOffset3()->disable();
	vOffset4()->disable();
}
XNIDAQmxDSO::~XNIDAQmxDSO() {
	clearAcquision();
}
void
XNIDAQmxDSO::onSoftTrigChanged(const shared_ptr<XNIDAQmxInterface::SoftwareTrigger> &) {
	for(Transaction tr( *this);; ++tr) {
		tr[ *trigSource()].clear();
		XString series = interface()->productSeries();
		{
			char buf[2048];
			{
				CHECK_DAQMX_RET(DAQmxGetDevAIPhysicalChans(interface()->devName(), buf, sizeof(buf)));
				std::deque<XString> chans;
				XNIDAQmxInterface::parseList(buf, chans);
				for(std::deque<XString>::iterator it = chans.begin(); it != chans.end(); it++) {
					tr[ *trigSource()].add(it->c_str());
				}
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
				tr[ *trigSource()].add(str);
			}
			local_shared_ptr<XNIDAQmxInterface::SoftwareTrigger::SoftwareTriggerList>
				list(XNIDAQmxInterface::SoftwareTrigger::virtualTrigList());
			for(XNIDAQmxInterface::SoftwareTrigger::SoftwareTriggerList_it
					it = list->begin(); it != list->end(); it++) {
				for(unsigned int i = 0; i < ( *it)->bits(); i++) {
					tr[ *trigSource()].add(
						formatString("%s/line%d", ( *it)->label(), i));
				}
			}
		}
		if(tr.commit())
			break;
	}
}
void
XNIDAQmxDSO::open() throw (XKameError &) {
	XScopedLock<XInterface> lock( *interface());
	m_running = false;
	char buf[2048];
	{
		CHECK_DAQMX_RET(DAQmxGetDevAIPhysicalChans(interface()->devName(), buf, sizeof(buf)));
		std::deque<XString> chans;
		XNIDAQmxInterface::parseList(buf, chans);
		for(Transaction tr( *this);; ++tr) {
			for(std::deque<XString>::iterator it = chans.begin(); it != chans.end(); it++) {
				tr[ *trace1()].add(it->c_str());
				tr[ *trace2()].add(it->c_str());
				tr[ *trace3()].add(it->c_str());
				tr[ *trace4()].add(it->c_str());
			}
			if(tr.commit())
				break;
		}
	}
	onSoftTrigChanged(shared_ptr<XNIDAQmxInterface::SoftwareTrigger>());

	m_suspendRead = true;
	m_threadReadAI.reset(new XThread<XNIDAQmxDSO>(shared_from_this(),
												  &XNIDAQmxDSO::executeReadAI));
	m_threadReadAI->resume();

	this->start();

	m_lsnOnSoftTrigChanged = XNIDAQmxInterface::SoftwareTrigger::onChange().connectWeak(
		shared_from_this(), &XNIDAQmxDSO::onSoftTrigChanged,
		XListener::FLAG_MAIN_THREAD_CALL);
	createChannels();
}
void
XNIDAQmxDSO::close() throw (XKameError &) {
	XScopedLock<XInterface> lock( *interface());

	m_lsnOnSoftTrigChanged.reset();

	clearAcquision();

	if(m_threadReadAI) {
		m_threadReadAI->terminate();
	}

	for(Transaction tr( *this);; ++tr) {
		tr[ *trace1()].clear();
		tr[ *trace2()].clear();
		tr[ *trace3()].clear();
		tr[ *trace4()].clear();
		if(tr.commit())
			break;
	}

	m_recordBuf.clear();
	m_record_av.clear();

	interface()->stop();
}
void
XNIDAQmxDSO::clearAcquision() {
	XScopedLock<XInterface> lock( *interface());
	m_suspendRead = true;
	XScopedLock<XRecursiveMutex> lock2(m_readMutex);

	try {
		disableTrigger();
	}
	catch (XInterface::XInterfaceError &e) {
		e.print();
	}

	if(m_task != TASK_UNDEF) {
		CHECK_DAQMX_RET(DAQmxClearTask(m_task));
	}
	m_task = TASK_UNDEF;
}
void
XNIDAQmxDSO::disableTrigger() {
	XScopedLock<XInterface> lock( *interface());
	m_suspendRead = true;
	XScopedLock<XRecursiveMutex> lock2(m_readMutex);

	if(m_running) {
		m_running = false;
		CHECK_DAQMX_RET(DAQmxStopTask(m_task));
	}
	if(m_task != TASK_UNDEF) {
		uInt32 num_ch;
		CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));
		if(num_ch) {
			CHECK_DAQMX_RET(DAQmxDisableStartTrig(m_task));
			CHECK_DAQMX_RET(DAQmxDisableRefTrig(m_task));
		}
	}

	m_preTriggerPos = 0;

	//reset virtual trigger setup.
	if(m_softwareTrigger)
		m_softwareTrigger->disconnect();
	m_lsnOnSoftTrigStarted.reset();
	m_softwareTrigger.reset();

	//reset HW trigger counter.
	if(m_taskCounterOrigin != TASK_UNDEF) {
		CHECK_DAQMX_RET(DAQmxStopTask(m_taskCounterOrigin));
		CHECK_DAQMX_RET(DAQmxClearTask(m_taskCounterOrigin));
	}
	m_taskCounterOrigin = TASK_UNDEF;
	m_countOrigin = 0;
}
void
XNIDAQmxDSO::setupTrigger() {
	XScopedLock<XInterface> lock( *interface());
	Snapshot shot( *this);
	m_suspendRead = true;
	XScopedLock<XRecursiveMutex> lock2(m_readMutex);

	unsigned int pretrig = lrint(shot[ *trigPos()] / 100.0 * shot[ *recordLength()]);
	m_preTriggerPos = pretrig;

	XString atrig;
	XString dtrig;
	XString src = shot[ *trigSource()].to_str();

	char buf[2048];
	{
		CHECK_DAQMX_RET(DAQmxGetDevAIPhysicalChans(interface()->devName(), buf, sizeof(buf)));
		std::deque<XString> chans;
		XNIDAQmxInterface::parseList(buf, chans);
		for(std::deque<XString>::iterator it = chans.begin(); it != chans.end(); it++) {
			if(src == *it)
				atrig = *it;
		}
	}
	if( !atrig.length())
		dtrig = src;

	int32 trig_spec = shot[ *trigFalling()] ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope;

	if(m_softwareTrigger) {
		dtrig = m_softwareTrigger->armTerm();
		trig_spec = DAQmx_Val_RisingSlope;
		pretrig = 0;
		CHECK_DAQMX_RET(DAQmxSetReadOverWrite(m_task, DAQmx_Val_OverwriteUnreadSamps));
	}

	//Small # of pretriggers is not allowed for ReferenceTrigger.
	if( !m_softwareTrigger && (pretrig < 2)) {
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


	//Setups counter for HW trigger/origin of SW trigger.
	m_countOrigin = 0;
	CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskCounterOrigin));
	XString ctrdev = formatString("%s/ctr0", interface()->devName());
	CHECK_DAQMX_RET(DAQmxCreateCIPeriodChan(
		m_taskCounterOrigin, ctrdev.c_str(), "", 1.0, 10000000, DAQmx_Val_Ticks,
		DAQmx_Val_Rising, DAQmx_Val_LowFreq1Ctr, 1, 4, NULL));
	char ch_ctr[256];
	CHECK_DAQMX_RET(DAQmxGetTaskChannels(m_taskCounterOrigin, ch_ctr, sizeof(ch_ctr)));
	CHECK_DAQMX_RET(DAQmxCfgImplicitTiming(m_taskCounterOrigin, DAQmx_Val_ContSamps, 1000));
	CHECK_DAQMX_RET(DAQmxSetCICtrTimebaseRate(m_taskCounterOrigin, ch_ctr, 1.0 / m_interval));
	interface()->synchronizeClock(m_taskCounterOrigin);
	XString hwcounter_input_term;
	if( !pretrig) {
		hwcounter_input_term = formatString("%s/aiStartTrigger", interface()->devName());
	}
	else {
		hwcounter_input_term = formatString("%s/aiReferenceTrigger", interface()->devName());
	}
	CHECK_DAQMX_RET(DAQmxSetCIPeriodTerm(m_taskCounterOrigin, ch_ctr, hwcounter_input_term.c_str()));

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
									  (m_softwareTrigger || (bufsize/2 < m_recordBuf.size())) ? DAQmx_Val_OnBrdMemNotEmpty :
									  DAQmx_Val_OnBrdMemMoreThanHalfFull));
	}
}
void
XNIDAQmxDSO::setupSoftwareTrigger() {
	Snapshot shot( *this);
	XString src = shot[ *trigSource()].to_str();
	//setup virtual trigger.
	local_shared_ptr<XNIDAQmxInterface::SoftwareTrigger::SoftwareTriggerList>
		list(XNIDAQmxInterface::SoftwareTrigger::virtualTrigList());
	for(XNIDAQmxInterface::SoftwareTrigger::SoftwareTriggerList_it
			it = list->begin(); it != list->end(); it++) {
		for(unsigned int i = 0; i < ( *it)->bits(); i++) {
			if(src == formatString("%s/line%d", ( *it)->label(), i)) {
				m_softwareTrigger = *it;
				m_softwareTrigger->connect(
					!shot[ *trigFalling()] ? (1uL << i) : 0,
					shot[ *trigFalling()] ? (1uL << i) : 0);
			}
		}
	}
}
void
XNIDAQmxDSO::setupTiming() {
	XScopedLock<XInterface> lock( *interface());
	Snapshot shot( *this);
	m_suspendRead = true;
	XScopedLock<XRecursiveMutex> lock2(m_readMutex);

	if(m_running) {
		m_running = false;
		CHECK_DAQMX_RET(DAQmxStopTask(m_task));
	}

	uInt32 num_ch;
	CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));
	if(num_ch == 0) return;

	disableTrigger();
	setupSoftwareTrigger();

	const unsigned int len = shot[ *recordLength()];
	for(unsigned int i = 0; i < 2; i++) {
		DSORawRecord &rec = m_dsoRawRecordBanks[i];
		rec.record.resize(len * num_ch * (rec.isComplex ? 2 : 1));
		assert(rec.numCh == num_ch);
		if(g_bUseMLock) {
			mlock(&rec.record[0], rec.record.size() * sizeof(int32_t));
		}
	}
	m_recordBuf.resize(len * num_ch);
	if(g_bUseMLock) {
		mlock( &m_recordBuf[0], m_recordBuf.size() * sizeof(tRawAI));
	}

	uInt32 onbrd_size;
	CHECK_DAQMX_RET(DAQmxGetBufInputOnbrdBufSize(m_task, &onbrd_size));
	fprintf(stderr, "Using on-brd bufsize=%d\n", (int)onbrd_size);
	unsigned int bufsize = len;
	if(m_softwareTrigger) {
		bufsize = std::max(bufsize * 8, (unsigned int)lrint((len / shot[ *timeWidth()]) * 1.0));
		bufsize = std::max(bufsize, (unsigned int)(onbrd_size / num_ch));
	}

	CHECK_DAQMX_RET(
		DAQmxCfgSampClkTiming(m_task,
							  //! debug!
							  //		formatString("/%s/Ctr0InternalOutput", interface()->devName()),
							  NULL, // internal source
							  len / shot[ *timeWidth()],
							  DAQmx_Val_Rising,
							  !m_softwareTrigger ? DAQmx_Val_FiniteSamps : DAQmx_Val_ContSamps,
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
	m_interval = 1.0 / rate;

	setupTrigger();

	startSequence();
}

void
XNIDAQmxDSO::createChannels() {
	XScopedLock<XInterface> lock( *interface());
	Snapshot shot( *this);
	m_suspendRead = true;
	XScopedLock<XRecursiveMutex> lock2(m_readMutex);

	clearAcquision();

	CHECK_DAQMX_RET(DAQmxCreateTask("", &m_task));
	assert(m_task != TASK_UNDEF);

	if(shot[ *trace1()] >= 0) {
		CHECK_DAQMX_RET(
			DAQmxCreateAIVoltageChan(m_task,
									 shot[ *trace1()].to_str().c_str(),
									 "",
									 DAQmx_Val_Cfg_Default,
									 -atof(shot[ *vFullScale1()].to_str().c_str()) / 2.0,
									 atof(shot[ *vFullScale1()].to_str().c_str()) / 2.0,
									 DAQmx_Val_Volts,
									 NULL
									 ));

		//obtain range info.
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++)
			m_coeffAI[0][i] = 0.0;
		CHECK_DAQMX_RET(
			DAQmxGetAIDevScalingCoeff(m_task,
									  shot[ *trace1()].to_str().c_str(),
									  m_coeffAI[0], CAL_POLY_ORDER));
	}
	if(shot[ *trace2()] >= 0) {
		CHECK_DAQMX_RET(
			DAQmxCreateAIVoltageChan(m_task,
									 shot[ *trace2()].to_str().c_str(),
									 "",
									 DAQmx_Val_Cfg_Default,
									 -atof(shot[ *vFullScale2()].to_str().c_str()) / 2.0,
									 atof(shot[ *vFullScale2()].to_str().c_str()) / 2.0,
									 DAQmx_Val_Volts,
									 NULL
									 ));
		//obtain range info.
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++)
			m_coeffAI[1][i] = 0.0;
		CHECK_DAQMX_RET(DAQmxGetAIDevScalingCoeff(m_task,
												  shot[ *trace2()].to_str().c_str(),
												  m_coeffAI[1], CAL_POLY_ORDER));
	}
	if(shot[ *trace3()] >= 0) {
		CHECK_DAQMX_RET(
			DAQmxCreateAIVoltageChan(m_task,
									 shot[ *trace3()].to_str().c_str(),
									 "",
									 DAQmx_Val_Cfg_Default,
									 -atof(shot[ *vFullScale3()].to_str().c_str()) / 2.0,
									 atof(shot[ *vFullScale3()].to_str().c_str()) / 2.0,
									 DAQmx_Val_Volts,
									 NULL
									 ));
		//obtain range info.
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++)
			m_coeffAI[2][i] = 0.0;
		CHECK_DAQMX_RET(DAQmxGetAIDevScalingCoeff(m_task,
												  shot[ *trace3()].to_str().c_str(),
												  m_coeffAI[2], CAL_POLY_ORDER));
	}
	if(shot[ *trace4()] >= 0) {
		CHECK_DAQMX_RET(
			DAQmxCreateAIVoltageChan(m_task,
									 shot[ *trace4()].to_str().c_str(),
									 "",
									 DAQmx_Val_Cfg_Default,
									 -atof(shot[ *vFullScale4()].to_str().c_str()) / 2.0,
									 atof(shot[ *vFullScale4()].to_str().c_str()) / 2.0,
									 DAQmx_Val_Volts,
									 NULL
									 ));
		//obtain range info.
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++)
			m_coeffAI[3][i] = 0.0;
		CHECK_DAQMX_RET(DAQmxGetAIDevScalingCoeff(m_task,
												  shot[ *trace4()].to_str().c_str(),
												  m_coeffAI[3], CAL_POLY_ORDER));
	}

	uInt32 num_ch;
	CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));

	//accumlation buffer.
	for(unsigned int i = 0; i < 2; i++) {
		DSORawRecord &rec(m_dsoRawRecordBanks[i]);
		rec.acqCount = 0;
		rec.accumCount = 0;
		rec.numCh = num_ch;
		rec.isComplex = (shot[ *dRFMode()] == DRFMODE_COHERENT_SG);
	}

	if(num_ch == 0)  {
		return;
	}

	CHECK_DAQMX_RET(DAQmxRegisterDoneEvent(m_task, 0, &XNIDAQmxDSO::onTaskDone_, this));

	setupTiming();
}
void
XNIDAQmxDSO::clearStoredSoftwareTrigger() {
	uInt64 total_samps = 0;
	if(m_running)
		CHECK_DAQMX_RET(DAQmxGetReadTotalSampPerChanAcquired(m_task, &total_samps));
	m_softwareTrigger->clear(total_samps, 1.0 / m_interval);
}
void
XNIDAQmxDSO::onSoftTrigStarted(const shared_ptr<XNIDAQmxInterface::SoftwareTrigger> &) {
	XScopedLock<XInterface> lock( *interface());
	m_suspendRead = true;
	XScopedLock<XRecursiveMutex> lock2(m_readMutex);

	if(m_running) {
		m_running = false;
		CHECK_DAQMX_RET(DAQmxStopTask(m_task));
	}

	const DSORawRecord &rec(m_dsoRawRecordBanks[m_dsoRawRecordBankLatest]);
	m_softwareTrigger->setBlankTerm(m_interval * rec.recordLength);
//	fprintf(stderr, "Virtual trig start.\n");

	uInt32 num_ch;
	CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));
	if(num_ch > 0) {
		int32 type;
		CHECK_DAQMX_RET(DAQmxGetStartTrigType(m_task, &type));
		if(type != DAQmx_Val_DigEdge) {
			setupTrigger();
		}
		CHECK_DAQMX_RET(DAQmxStartTask(m_task));
		m_suspendRead = false;
		m_running = true;
	}
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
		m_suspendRead = true;
	}
}
void
XNIDAQmxDSO::onForceTriggerTouched(const Snapshot &shot, XTouchableNode *) {
	XScopedLock<XInterface> lock( *interface());
	m_suspendRead = true;
	XScopedLock<XRecursiveMutex> lock2(m_readMutex);

	if(m_softwareTrigger) {
		if(m_running) {
			uInt64 total_samps;
			CHECK_DAQMX_RET(DAQmxGetReadTotalSampPerChanAcquired(m_task, &total_samps));
			m_softwareTrigger->forceStamp(total_samps, 1.0 / m_interval);
			m_suspendRead = false;
		}
	}
	else {
		disableTrigger();
		CHECK_DAQMX_RET(DAQmxStartTask(m_task));
		m_suspendRead = false;
		m_running = true;
	}
}
inline bool
XNIDAQmxDSO::tryReadAISuspend(const atomic<bool> &terminated) {
	if(m_suspendRead) {
		m_readMutex.unlock();
		while(m_suspendRead && !terminated) usleep(30000);
		m_readMutex.lock();
		return true;
	}
	return false;
}
void *
XNIDAQmxDSO::executeReadAI(const atomic<bool> &terminated) {
	while( !terminated) {
		try {
			acquire(terminated);
		}
		catch (XInterface::XInterfaceError &e) {
			e.print(getLabel());
			m_suspendRead = true;
		}
	}
	return NULL;
}
uint64_t
XNIDAQmxDSO::storeCountOrigin() {
	for(;;) {
	//Checks available data
		uInt32 st_count;
		CHECK_DAQMX_RET(DAQmxGetReadAvailSampPerChan(m_taskCounterOrigin, &st_count));
		if( !st_count)
			break;
		uInt32 count_lsw;
		CHECK_DAQMX_RET(DAQmxReadCounterScalarU32(m_taskCounterOrigin, 0, &count_lsw, NULL));
		m_countOrigin += count_lsw;
		checkOverflowForCounterOrigin();
		fprintf(stderr, "sC %f\n", (double)count_lsw);
	}
	return m_countOrigin;
}
bool
XNIDAQmxDSO::checkOverflowForCounterOrigin() {
	char ch_ctr[256];
	CHECK_DAQMX_RET(DAQmxGetTaskChannels(m_taskCounterOrigin, ch_ctr, sizeof(ch_ctr)));
	bool32 reached;
	CHECK_DAQMX_RET(DAQmxGetCITCReached(m_taskCounterOrigin, ch_ctr, &reached));
	if(reached) {
		float64 count_max;
		CHECK_DAQMX_RET(DAQmxGetCIMax(m_taskCounterOrigin, ch_ctr, &count_max));
		fprintf(stderr, "cm %f\n", (double)count_max);
		m_countOrigin += llrint(count_max + 1);
	}
	return reached;
}
void
XNIDAQmxDSO::acquire(const atomic<bool> &terminated) {
	XScopedLock<XRecursiveMutex> lock(m_readMutex);
	while( !terminated) {
		checkOverflowForCounterOrigin();

		if( !m_running) {
			tryReadAISuspend(terminated);
			msecsleep(30);
			return;
		}

		uInt32 num_ch;
		CHECK_DAQMX_RET(DAQmxGetReadNumChans(m_task, &num_ch));
		if(num_ch == 0) {
			tryReadAISuspend(terminated);
			msecsleep(30);
			return;
		}

		const DSORawRecord &old_rec(m_dsoRawRecordBanks[m_dsoRawRecordBankLatest]);
		if(num_ch != old_rec.numCh)
			throw XInterface::XInterfaceError(i18n("Inconsistent channel number."), __FILE__, __LINE__);

		const unsigned int size = m_recordBuf.size() / num_ch;
		const float64 freq = 1.0 / m_interval;
		unsigned int cnt = 0;

		uint64_t samplecnt_at_trigger = 0;
		if(m_softwareTrigger) {
			shared_ptr<XNIDAQmxInterface::SoftwareTrigger> &vt(m_softwareTrigger);

			while( !terminated) {
				if(tryReadAISuspend(terminated))
					return;
				uInt64 total_samps;
				CHECK_DAQMX_RET(DAQmxGetReadTotalSampPerChanAcquired(m_task, &total_samps));
				samplecnt_at_trigger = vt->tryPopFront(total_samps, freq);
				if(samplecnt_at_trigger) {
					uInt32 bufsize;
					CHECK_DAQMX_RET(DAQmxGetBufInputBufSize(m_task, &bufsize));
					if(total_samps - samplecnt_at_trigger + m_preTriggerPos > bufsize * 4 / 5) {
						gWarnPrint(i18n("Buffer Overflow."));
						continue;
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
					int32 offset = samplecnt_at_trigger - m_preTriggerPos - curr_rdpos;
					CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, offset));
					//					fprintf(stderr, "hit! %d %d %d\n", (int)offset, (int)lastcnt, (int)m_preTriggerPos);
					break;
				}
				usleep(lrint(1e6 * size * m_interval / 6));
			}
		}
		else {
			if(m_preTriggerPos) {
				CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_FirstPretrigSamp));
			}
			else {
				CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_FirstSample));
			}

			CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, 0));
		}
		 //Reads count for the origin/trigger.
		storeCountOrigin();
		samplecnt_at_trigger += m_countOrigin;

		if(terminated)
			return;

		const unsigned int num_samps = std::min(size, 8192u);
		for(; cnt < size;) {
			int32 samps;
			samps = std::min(size - cnt, num_samps);
			while( !terminated) {
				if(tryReadAISuspend(terminated))
					return;
				uInt32 space;
				int ret = DAQmxGetReadAvailSampPerChan(m_task, &space);
				if( !ret && (space >= (uInt32)samps))
					break;
				usleep(lrint(1e6 * (samps - space) * m_interval));
			}
			if(terminated)
				return;
			CHECK_DAQMX_RET(DAQmxReadBinaryI16(m_task, samps,
											   0.0, DAQmx_Val_GroupByScanNumber,
											   &m_recordBuf[cnt * num_ch], samps * num_ch, &samps, NULL
											   ));
			cnt += samps;
			if( !m_softwareTrigger) {
				CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, cnt));
			}
			else {
				CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, 0));
			}
		}

		Snapshot shot( *this);
		const unsigned int av = shot[ *average()];
		const bool sseq = shot[ *singleSequence()];
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
					CHECK_DAQMX_RET(DAQmxStopTask(m_task));
				}
				CHECK_DAQMX_RET(DAQmxStartTask(m_task));
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
			double ph = phaseOfRF(shot, samplecnt_at_trigger, m_interval);
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
void
XNIDAQmxDSO::startSequence() {
	XScopedLock<XInterface> lock( *interface());
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
			m_lsnOnSoftTrigStarted = m_softwareTrigger->onStart().connectWeak(
				shared_from_this(), &XNIDAQmxDSO::onSoftTrigStarted);
		if(m_running) {
			clearStoredSoftwareTrigger();
			m_suspendRead = false;
		}
		else {
			CHECK_DAQMX_RET(DAQmxTaskControl(m_task, DAQmx_Val_Task_Commit));
			statusPrinter()->printMessage(i18n("Restart the software-trigger source."));
		}
	}
	else {
		if(m_running) {
			m_running = false;
			if(m_task != TASK_UNDEF)
				CHECK_DAQMX_RET(DAQmxStopTask(m_task));
		}
		uInt32 num_ch;
		CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));
		if(num_ch > 0) {
			CHECK_DAQMX_RET(DAQmxStartTask(m_task));
			m_suspendRead = false;
			m_running = true;
		}
	}
	//	CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, 0));
}

int
XNIDAQmxDSO::acqCount(bool *seq_busy) {
	const DSORawRecord &rec(m_dsoRawRecordBanks[m_dsoRawRecordBankLatest]);
	Snapshot shot( *this);
	*seq_busy = ((unsigned int)rec.acqCount < shot[ *average()]);
	return rec.acqCount;
}

double
XNIDAQmxDSO::getTimeInterval() {
	return m_interval;
}

inline float64
XNIDAQmxDSO::aiRawToVolt(const float64 *pcoeff, float64 raw) {
	float64 x = 1.0;
	float64 y = 0.0;
	for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
		y += *(pcoeff++) * x;
		x *= raw;
	}
	return y;
}

void
XNIDAQmxDSO::getWave(shared_ptr<RawData> &writer, std::deque<XString> &) {
	XScopedLock<XInterface> lock( *interface());

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
	uInt32 num_ch = rec.numCh;
	uInt32 len = rec.recordLength;

	char buf[2048];
	CHECK_DAQMX_RET(DAQmxGetReadChannelsToRead(m_task, buf, sizeof(buf)));

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
		writer->push<int32_t>( *p++);
	XString str(buf);
	writer->insert(writer->end(), str.begin(), str.end());
	str = ""; //reserved/
	writer->insert(writer->end(), str.begin(), str.end());

	rec.unlock();
}
void
XNIDAQmxDSO::convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&) {
	const unsigned int num_ch = reader.pop<uint32_t>();
	const unsigned int pretrig = reader.pop<uint32_t>();
	const unsigned int len = reader.pop<uint32_t>();
	const unsigned int accumCount = reader.pop<uint32_t>();
	const double interval = reader.pop<double>();

	tr[ *this].setParameters(num_ch, - (double)pretrig * interval, interval, len);

	double *wave[NUM_MAX_CH * 2];
	float64 coeff[NUM_MAX_CH * 2][CAL_POLY_ORDER];
	for(unsigned int j = 0; j < num_ch; j++) {
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
			coeff[j][i] = reader.pop<double>();
		}

		wave[j] = tr[ *this].waveDisp(j);
	}

	const float64 prop = 1.0 / accumCount;
	for(unsigned int i = 0; i < len; i++) {
		for(unsigned int j = 0; j < num_ch; j++)
			*(wave[j])++ = aiRawToVolt(coeff[j], reader.pop<int32_t>() * prop);
	}
}

void
XNIDAQmxDSO::onAverageChanged(const Snapshot &shot, XValueNodeBase *) {
	startSequence();
}

void
XNIDAQmxDSO::onSingleChanged(const Snapshot &shot, XValueNodeBase *) {
	startSequence();
}
void
XNIDAQmxDSO::onTrigPosChanged(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onTrigSourceChanged(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onTrigLevelChanged(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onTrigFallingChanged(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onTimeWidthChanged(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onTrace1Changed(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onTrace2Changed(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onTrace3Changed(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onTrace4Changed(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onVFullScale1Changed(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onVFullScale2Changed(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onVFullScale3Changed(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onVFullScale4Changed(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onVOffset1Changed(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onVOffset2Changed(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onVOffset3Changed(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onVOffset4Changed(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
void
XNIDAQmxDSO::onRecordLengthChanged(const Snapshot &shot, XValueNodeBase *) {
	createChannels();
}
