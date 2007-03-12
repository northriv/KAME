/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
						   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
//! \todo initial items, physical items.

#include "nidaqdso.h"

#ifdef HAVE_NI_DAQMX

#include <qmessagebox.h>
#include <kmessagebox.h>
#include "xwavengraph.h"
#include <klocale.h>

#define TASK_UNDEF ((TaskHandle)-1)
#define NUM_MAX_CH 2

XNIDAQmxDSO::XNIDAQmxDSO(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
 XNIDAQmxDriver<XDSO>(name, runtime, scalarentries, interfaces, thermometers, drivers),
 m_dsoRawRecordBankLatest(0),
 m_task(TASK_UNDEF)
{
	recordLength()->value(2000);
	timeWidth()->value(1e-2);
	average()->value(1);
	
	const char* sc[] = {"0.4", "1", "2", "4", "10", "20", "40", "84", 0L};
	for(int i = 0; sc[i]; i++)
	{
		vFullScale1()->add(sc[i]);
		vFullScale2()->add(sc[i]);
	}
	vFullScale1()->value("20");
	vFullScale2()->value("20");
	
 	if(g_bUseMLock) {
		const void *FIRST_OF_MLOCK_MEMBER = &m_recordBuf;
		const void *LAST_OF_MLOCK_MEMBER = &m_task;
		//Suppress swapping.
		mlock(FIRST_OF_MLOCK_MEMBER, (size_t)LAST_OF_MLOCK_MEMBER - (size_t)FIRST_OF_MLOCK_MEMBER);	
 	}
}
XNIDAQmxDSO::~XNIDAQmxDSO()
{
	clearAcquision();
}
void
XNIDAQmxDSO::onSoftTrigChanged(const shared_ptr<XNIDAQmxInterface::SoftwareTrigger> &) {
	trigSource()->clear();
	std::string series = interface()->productSeries();
	{
		char buf[2048];
		{
			DAQmxGetDevAIPhysicalChans(interface()->devName(), buf, sizeof(buf));
			std::deque<std::string> chans;
			XNIDAQmxInterface::parseList(buf, chans);
			for(std::deque<std::string>::iterator it = chans.begin(); it != chans.end(); it++) {
				trigSource()->add(it->c_str());
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
		for(const char**it = sc; *it; it++) {
			std::string str(formatString("/%s/%s", interface()->devName(), *it));
			trigSource()->add(str);
		}
		atomic_shared_ptr<XNIDAQmxInterface::SoftwareTrigger::SoftwareTriggerList> 
			list(XNIDAQmxInterface::SoftwareTrigger::virtualTrigList());
		for(XNIDAQmxInterface::SoftwareTrigger::SoftwareTriggerList_it
			it = list->begin(); it != list->end(); it++) {
			for(unsigned int i = 0; i < (*it)->bits(); i++) {
				trigSource()->add(
					formatString("%s/line%d", (*it)->label(), i));
			}
		}
	}
}
void
XNIDAQmxDSO::open() throw (XInterface::XInterfaceError &)
{
	XScopedLock<XInterface> lock(*interface());
	m_running = false;
	char buf[2048];
	{
		DAQmxGetDevAIPhysicalChans(interface()->devName(), buf, sizeof(buf));
		std::deque<std::string> chans;
		XNIDAQmxInterface::parseList(buf, chans);
		for(std::deque<std::string>::iterator it = chans.begin(); it != chans.end(); it++) {
			trace1()->add(it->c_str());
			trace2()->add(it->c_str());
		}
	}
	onSoftTrigChanged(shared_ptr<XNIDAQmxInterface::SoftwareTrigger>());

	m_suspendRead = true;
	m_threadReadAI.reset(new XThread<XNIDAQmxDSO>(shared_from_this(),
		 &XNIDAQmxDSO::executeReadAI));
	m_threadReadAI->resume();
	
	this->start();
	
	vOffset1()->setUIEnabled(false);
	vOffset2()->setUIEnabled(false);

 	
	m_lsnOnSoftTrigChanged = XNIDAQmxInterface::SoftwareTrigger::onChange().connectWeak(
		shared_from_this(), &XNIDAQmxDSO::onSoftTrigChanged,
		XListener::FLAG_MAIN_THREAD_CALL);
	createChannels();
}
void
XNIDAQmxDSO::close() throw (XInterface::XInterfaceError &)
{
	XScopedLock<XInterface> lock(*interface());
 	
	m_lsnOnSoftTrigChanged.reset();

	clearAcquision();
 	
	if(m_threadReadAI) {
		m_threadReadAI->terminate();
	}
 	
	trace1()->clear();
	trace2()->clear();
	
	m_recordBuf.clear();
	m_record_av.clear();

	interface()->stop();
}
void
XNIDAQmxDSO::clearAcquision() {
	XScopedLock<XInterface> lock(*interface());
	m_suspendRead = true;
 	XScopedLock<XRecursiveMutex> lock2(m_readMutex);
	
 	disableTrigger();

	if(m_task != TASK_UNDEF) {
		DAQmxClearTask(m_task);
	}
	m_task = TASK_UNDEF;
}
void
XNIDAQmxDSO::disableTrigger()
{
	XScopedLock<XInterface> lock(*interface());
	m_suspendRead = true;
 	XScopedLock<XRecursiveMutex> lock2(m_readMutex);
	
	if(m_running) {
		m_running = false;
		DAQmxStopTask(m_task);
	}
	if(m_task != TASK_UNDEF) {
		DAQmxDisableStartTrig(m_task);
		DAQmxDisableRefTrig(m_task);
	}
	
	m_preTriggerPos = 0;
	m_trigRoute.reset();

	//reset virtual trigger setup.
	if(m_softwareTrigger)
		m_softwareTrigger->disconnect();
	m_lsnOnSoftTrigStarted.reset();
	m_softwareTrigger.reset();
}
void
XNIDAQmxDSO::setupTrigger()
{
	XScopedLock<XInterface> lock(*interface());
	m_suspendRead = true;
 	XScopedLock<XRecursiveMutex> lock2(m_readMutex);
	
	unsigned int pretrig = lrint(*trigPos() / 100.0 * *recordLength());
	m_preTriggerPos = pretrig;
	
	std::string atrig;
	std::string dtrig;
	std::string src = trigSource()->to_str();

	char buf[2048];
	{
		DAQmxGetDevAIPhysicalChans(interface()->devName(), buf, sizeof(buf));
		std::deque<std::string> chans;
		XNIDAQmxInterface::parseList(buf, chans);
		for(std::deque<std::string>::iterator it = chans.begin(); it != chans.end(); it++) {
			if(src == *it)
				atrig = *it;
		}
	}
	if(!atrig.length())
		 dtrig = src;
	
	int32 trig_spec = *trigFalling() ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope;
	
	if(m_softwareTrigger) {
		dtrig = m_softwareTrigger->armTerm();
		trig_spec = DAQmx_Val_RisingSlope;
		pretrig = 0;					
		CHECK_DAQMX_RET(DAQmxSetReadOverWrite(m_task, DAQmx_Val_OverwriteUnreadSamps));
	}
	
	//Small # of pretriggers is not allowed for ReferenceTrigger.
	if(!m_softwareTrigger && (pretrig < 2)) {
		pretrig = 0;
		m_preTriggerPos = pretrig;
	}
	
	if(!pretrig) {
		if(atrig.length()) {
			CHECK_DAQMX_RET(DAQmxCfgAnlgEdgeStartTrig(m_task,
				atrig.c_str(), trig_spec, *trigLevel()));
		}
		if(dtrig.length()) {
			CHECK_DAQMX_RET(DAQmxCfgDigEdgeStartTrig(m_task,
				dtrig.c_str(), trig_spec));
		}
	}
	else {
		if(atrig.length()) {
			CHECK_DAQMX_RET(DAQmxCfgAnlgEdgeRefTrig(m_task,
				atrig.c_str(), trig_spec, *trigLevel(), pretrig));
		}
		if(dtrig.length()) {
			CHECK_DAQMX_RET(DAQmxCfgDigEdgeRefTrig(m_task,
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
		CHECK_DAQMX_RET(DAQmxSetAIDataXferReqCond(m_task, ch, 
			(m_softwareTrigger || (bufsize/2 < m_recordBuf.size())) ? DAQmx_Val_OnBrdMemNotEmpty :
				DAQmx_Val_OnBrdMemMoreThanHalfFull));
	}	
	startSequence();
}
void
XNIDAQmxDSO::setupSoftwareTrigger()
{
	std::string src = trigSource()->to_str();
	//setup virtual trigger.
	atomic_shared_ptr<XNIDAQmxInterface::SoftwareTrigger::SoftwareTriggerList> 
		list(XNIDAQmxInterface::SoftwareTrigger::virtualTrigList());
	for(XNIDAQmxInterface::SoftwareTrigger::SoftwareTriggerList_it
		it = list->begin(); it != list->end(); it++) {
		for(unsigned int i = 0; i < (*it)->bits(); i++) {
			if(src == formatString("%s/line%d", (*it)->label(), i)) {
				m_softwareTrigger = *it;
				m_softwareTrigger->connect(
				!*trigFalling() ? (1uL << i) : 0,
				*trigFalling() ? (1uL << i) : 0);
			}
		}
	}
}
void
XNIDAQmxDSO::setupTiming()
{
	XScopedLock<XInterface> lock(*interface());
	m_suspendRead = true;
 	XScopedLock<XRecursiveMutex> lock2(m_readMutex);

	if(m_running) {
		m_running = false;
		DAQmxStopTask(m_task);
	}

	uInt32 num_ch;
	CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));	
	if(num_ch == 0) return;

	disableTrigger();
	setupSoftwareTrigger();

	const unsigned int len = *recordLength();
	for(unsigned int i = 0; i < 2; i++) {
		DSORawRecord &rec = m_dsoRawRecordBanks[i];
		rec.record.resize(len * num_ch);
		rec.numCh = num_ch;
		if(g_bUseMLock) {
			mlock(&rec.record[0], rec.record.size() * sizeof(int32_t));
		}
	}
	m_recordBuf.resize(len * num_ch);
	if(g_bUseMLock) {
		mlock(&m_recordBuf[0], m_recordBuf.size() * sizeof(tRawAI));	
	}

	uInt32 onbrd_size;
	CHECK_DAQMX_RET(DAQmxGetBufInputOnbrdBufSize(m_task, &onbrd_size));
	fprintf(stderr, "Using on-brd bufsize=%d\n", (int)onbrd_size);
	unsigned int bufsize = len;
	if(m_softwareTrigger) {
		bufsize = std::max(bufsize * 8, (unsigned int)lrint((len / *timeWidth()) * 1.0));
		bufsize = std::max(bufsize, (unsigned int)onbrd_size);
	}
	
	CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_task,
//! debug!
//		formatString("/%s/Ctr0InternalOutput", interface()->devName()),
		NULL, // internal source
		len / *timeWidth(),
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
	
	float64 rate;
//	CHECK_DAQMX_RET(DAQmxGetRefClkRate(m_task, &rate));
//	dbgPrint(QString("Reference Clk rate = %1.").arg(rate));
	CHECK_DAQMX_RET(DAQmxGetSampClkRate(m_task, &rate));
	m_interval = 1.0 / rate;

	setupTrigger();
}
void
XNIDAQmxDSO::createChannels()
{
	XScopedLock<XInterface> lock(*interface());
	m_suspendRead = true;
 	XScopedLock<XRecursiveMutex> lock2(m_readMutex);
 	
	clearAcquision();
	
	CHECK_DAQMX_RET(DAQmxCreateTask("", &m_task));
	ASSERT(m_task != TASK_UNDEF);   
	
	if(*trace1() >= 0) {
		CHECK_DAQMX_RET(DAQmxCreateAIVoltageChan(m_task,
			trace1()->to_str().c_str(),
			  "",
			DAQmx_Val_Cfg_Default,
			-atof(vFullScale1()->to_str().c_str()) / 2.0,
			atof(vFullScale1()->to_str().c_str()) / 2.0,
			DAQmx_Val_Volts,
			NULL
			));

		//obtain range info.
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++)
			m_coeffAI[0][i] = 0.0;
		CHECK_DAQMX_RET(DAQmxGetAIDevScalingCoeff(m_task, 
			trace1()->to_str().c_str(),
			m_coeffAI[0], CAL_POLY_ORDER));
	}
	if(*trace2() >= 0) {
		CHECK_DAQMX_RET(DAQmxCreateAIVoltageChan(m_task,
			trace2()->to_str().c_str(),
			  "",
			DAQmx_Val_Cfg_Default,
			-atof(vFullScale2()->to_str().c_str()) / 2.0,
			atof(vFullScale2()->to_str().c_str()) / 2.0,
			DAQmx_Val_Volts,
			NULL
			));
		//obtain range info.
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++)
			m_coeffAI[1][i] = 0.0;
		CHECK_DAQMX_RET(DAQmxGetAIDevScalingCoeff(m_task, 
			trace2()->to_str().c_str(),
			m_coeffAI[1], CAL_POLY_ORDER));
	}

	uInt32 num_ch;
	CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));	
	if(num_ch == 0) 
		return;
	{
/*		char chans[256];
		CHECK_DAQMX_RET(DAQmxGetTaskChannels(m_task, chans, sizeof(chans)));
		bool32 ret;
		CHECK_DAQMX_RET(DAQmxGetAIChanCalHasValidCalInfo(m_task, chans, &ret));
		if(!ret) {
			statusPrinter()->printMessage(KAME::i18n("Performing self calibration."));
			QMessageBox::warning(g_pFrmMain, "KAME", KAME::i18n("Performing self calibration. Wait for minutes.") );
			CHECK_DAQMX_RET(DAQmxSelfCal(interface()->devName()));
			statusPrinter()->printMessage(KAME::i18n("Self calibration done."));
		}
*/	}

	CHECK_DAQMX_RET(DAQmxRegisterDoneEvent(m_task, 0, &XNIDAQmxDSO::_onTaskDone, this));
   	
	CHECK_DAQMX_RET(DAQmxSetRealTimeReportMissedSamp(m_task, true));

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
	XScopedLock<XInterface> lock(*interface());
	m_suspendRead = true;
 	XScopedLock<XRecursiveMutex> lock2(m_readMutex);

	if(m_running) {
		m_running = false;
		DAQmxStopTask(m_task);
	}
	
	const DSORawRecord &rec(m_dsoRawRecordBanks[m_dsoRawRecordBankLatest]);
	m_softwareTrigger->setBlankTerm(m_interval * rec.recordLength);
	fprintf(stderr, "Virtual trig start.\n");

	uInt32 num_ch;
	CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));	
	if(num_ch > 0) {
		CHECK_DAQMX_RET(DAQmxStartTask(m_task));
		m_suspendRead = false;
		m_running = true;
	}
}
int32
XNIDAQmxDSO::_onTaskDone(TaskHandle task, int32 status, void *data) {
	XNIDAQmxDSO *obj = reinterpret_cast<XNIDAQmxDSO*>(data);
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
XNIDAQmxDSO::onForceTriggerTouched(const shared_ptr<XNode> &)
{
	XScopedLock<XInterface> lock(*interface());
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
		while(m_suspendRead && !terminated) msecsleep(30);
		m_readMutex.lock();
		return true;
	}
	return false;
}
void *
XNIDAQmxDSO::executeReadAI(const atomic<bool> &terminated)
{
	while(!terminated) {
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
void
XNIDAQmxDSO::acquire(const atomic<bool> &terminated)
{
	XScopedLock<XRecursiveMutex> lock(m_readMutex);
  while(!terminated) {

	 if(!m_running) {
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
		throw XInterface::XInterfaceError(KAME::i18n("Inconsistent channel number."), __FILE__, __LINE__);

	const unsigned int size = m_recordBuf.size() / num_ch;
	const float64 freq = 1.0 / m_interval;
	unsigned int cnt = 0;

	if(m_softwareTrigger) {
		shared_ptr<XNIDAQmxInterface::SoftwareTrigger> &vt(m_softwareTrigger);
		
		while(!terminated) {
			if(tryReadAISuspend(terminated))
				return;
			uInt64 total_samps;
			CHECK_DAQMX_RET(DAQmxGetReadTotalSampPerChanAcquired(m_task, &total_samps));
			if(uint64_t lastcnt = vt->tryPopFront(total_samps, freq)) {
				uInt32 bufsize;
				CHECK_DAQMX_RET(DAQmxGetBufInputBufSize(m_task, &bufsize));
				if(total_samps - lastcnt + m_preTriggerPos > bufsize * 4 / 5) {
					gWarnPrint(KAME::i18n("Buffer Overflow."));
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
				int32 offset = lastcnt - m_preTriggerPos - curr_rdpos;
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
	if(terminated)
		return;

	const unsigned int num_samps = std::min(size, 8192u);
	for(; cnt < size;) {
		int32 samps;
		samps = std::min(size - cnt, num_samps);
		while(!terminated) {
			if(tryReadAISuspend(terminated))
				return;
		uInt32 space;
			int ret = DAQmxGetReadAvailSampPerChan(m_task, &space);
			if(!ret && (space >= (uInt32)samps))
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
		if(!m_softwareTrigger) {
			CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, cnt));
		}
		else {
			CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, 0));
		}
	}

	const unsigned int av = *average();
	const bool sseq = *singleSequence();
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
	writeBarrier();
	ASSERT((bank >= 0) && (bank < 2));
	DSORawRecord &new_rec(m_dsoRawRecordBanks[bank]);
	unsigned int accumcnt = old_rec.accumCount;
	
	if(!sseq || (accumcnt < av)) {
		if(!m_softwareTrigger) {
			if(m_running) {
				m_running = false;
				DAQmxStopTask(m_task);
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
	//for optimization.
	unsigned int div = bufsize / 4;
	unsigned int rest = bufsize % 4;
	for(unsigned int i = 0; i < div; i++) {
		*paccum++ = *pold++ + *pbuf++;
		*paccum++ = *pold++ + *pbuf++;
		*paccum++ = *pold++ + *pbuf++;
		*paccum++ = *pold++ + *pbuf++;
	}
	for(unsigned int i = 0; i < rest; i++)
		*paccum++ = *pold++ + *pbuf++;
	new_rec.acqCount = old_rec.acqCount + 1;
	accumcnt++;

	while(!sseq && (av <= m_record_av.size()) && !m_record_av.empty())  {
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
	writeBarrier();
	new_rec.unlock();
	if(!sseq) {
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
XNIDAQmxDSO::startSequence()
{
	XScopedLock<XInterface> lock(*interface());
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
		ASSERT(rec.numCh);
		rec.recordLength = rec.record.size() / rec.numCh;
		memset(&rec.record[0], 0, rec.record.size() * sizeof(int32_t));
	}
	m_record_av.clear();   	
	
	if(m_softwareTrigger) {
		if(!m_lsnOnSoftTrigStarted)
			m_lsnOnSoftTrigStarted = m_softwareTrigger->onStart().connectWeak(
				shared_from_this(), &XNIDAQmxDSO::onSoftTrigStarted);
		if(m_running) {
			clearStoredSoftwareTrigger();
			m_suspendRead = false;
		}
		else {
			statusPrinter()->printMessage(KAME::i18n("Restart the software-trigger source."));
		}
	}
	else {
		if(m_running) {
			m_running = false;
			if(m_task != TASK_UNDEF)
				DAQmxStopTask(m_task);
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
XNIDAQmxDSO::acqCount(bool *seq_busy)
{
	const DSORawRecord &rec(m_dsoRawRecordBanks[m_dsoRawRecordBankLatest]);
	*seq_busy = ((unsigned int)rec.acqCount < *average());
	return rec.acqCount;
}

double
XNIDAQmxDSO::getTimeInterval()
{
	return m_interval;
}

inline float64
XNIDAQmxDSO::aiRawToVolt(const float64 *pcoeff, float64 raw)
{
	float64 x = 1.0;
	float64 y = 0.0;
	for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
		y += *(pcoeff++) * x;
		x *= raw;
	}
	return y;
}

void
XNIDAQmxDSO::getWave(std::deque<std::string> &)
{
	XScopedLock<XInterface> lock(*interface());
	
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
	ASSERT((bank >= 0) && (bank < 2));
	DSORawRecord &rec(m_dsoRawRecordBanks[bank]);

 	if(rec.accumCount == 0) {
		rec.unlock();
 		throw XDriver::XSkippedRecordError(__FILE__, __LINE__);
 	}
	const uInt32 num_ch = rec.numCh;
	const uInt32 len = rec.recordLength;
	
	char buf[2048];
	CHECK_DAQMX_RET(DAQmxGetReadChannelsToRead(m_task, buf, sizeof(buf)));
	
	push((uint32_t)num_ch);
	push((uint32_t)m_preTriggerPos);
	push((uint32_t)len);
	push((uint32_t)rec.accumCount);
	push((double)m_interval);
	for(unsigned int ch = 0; ch < num_ch; ch++) {
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
			push((double)m_coeffAI[ch][i]);
		}
	}
	const int32_t *p = &(rec.record[0]);
	std::vector<char> &raw_data(rawData());
	const unsigned int size = len * num_ch;
	for(unsigned int i = 0; i < size; i++)
		push(*p++, raw_data);
	std::string str(buf);
	rawData().insert(rawData().end(), str.begin(), str.end());
	str = ""; //reserved/
	rawData().insert(rawData().end(), str.begin(), str.end());

	rec.unlock();
}
void
XNIDAQmxDSO::convertRaw() throw (XRecordError&)
{
	const unsigned int num_ch = pop<uint32_t>();
	const unsigned int pretrig = pop<uint32_t>();
	const unsigned int len = pop<uint32_t>();
	const unsigned int accumCount = pop<uint32_t>();
	const double interval = pop<double>();

	setParameters(num_ch, - (double)pretrig * interval, interval, len);
	
	double *wave[NUM_MAX_CH];
	float64 coeff[NUM_MAX_CH][CAL_POLY_ORDER];
	for(unsigned int j = 0; j < num_ch; j++) {
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
			coeff[j][i] = pop<double>();
		}
		
		wave[j] = waveDisp(j);
	}

	std::vector<char>::iterator &raw_data_it(rawDataPopIterator());
	const float64 prop = 1.0 / accumCount;
	for(unsigned int i = 0; i < len; i++) {
		for(unsigned int j = 0; j < num_ch; j++)
			  *(wave[j])++ = aiRawToVolt(coeff[j], pop<int32_t>(raw_data_it) * prop);
	}
}

void 
XNIDAQmxDSO::onAverageChanged(const shared_ptr<XValueNodeBase> &) {
	startSequence();
}

void
XNIDAQmxDSO::onSingleChanged(const shared_ptr<XValueNodeBase> &) {
	startSequence();
}
void
XNIDAQmxDSO::onTrigPosChanged(const shared_ptr<XValueNodeBase> &) {
	createChannels();
}
void
XNIDAQmxDSO::onTrigSourceChanged(const shared_ptr<XValueNodeBase> &) {
	createChannels();
}
void
XNIDAQmxDSO::onTrigLevelChanged(const shared_ptr<XValueNodeBase> &) {
	createChannels();
}
void
XNIDAQmxDSO::onTrigFallingChanged(const shared_ptr<XValueNodeBase> &) {
	createChannels();
}
void
XNIDAQmxDSO::onTimeWidthChanged(const shared_ptr<XValueNodeBase> &) {
	createChannels();
}
void
XNIDAQmxDSO::onTrace1Changed(const shared_ptr<XValueNodeBase> &) {
	createChannels();
}
void
XNIDAQmxDSO::onTrace2Changed(const shared_ptr<XValueNodeBase> &) {
	createChannels();
}
void
XNIDAQmxDSO::onVFullScale1Changed(const shared_ptr<XValueNodeBase> &) {
	createChannels();
}
void
XNIDAQmxDSO::onVFullScale2Changed(const shared_ptr<XValueNodeBase> &) {
	createChannels();
}
void
XNIDAQmxDSO::onVOffset1Changed(const shared_ptr<XValueNodeBase> &) {
	createChannels();
}
void
XNIDAQmxDSO::onVOffset2Changed(const shared_ptr<XValueNodeBase> &) {
	createChannels();
}
void
XNIDAQmxDSO::onRecordLengthChanged(const shared_ptr<XValueNodeBase> &) {
	createChannels();
}

#endif //HAVE_NI_DAQMX
