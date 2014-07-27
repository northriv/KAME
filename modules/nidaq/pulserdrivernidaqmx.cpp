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
#include "pulserdrivernidaqmx.h"

//! \warning Current version of DAQmx (8.0.2) software for Linux may lack thread-safety in many-core systems.
//! When an app attempts concurrent data writing to two devices,
//! the driver sometimes causes freezing, ends with error, or leaves a syslog message,
//! "BUG: sleeping function called from invalid context at mm/slub.c:..."
//! Perhaps a global lock on DAQmx functions is necessary....

#define PAUSING_BLANK_BEFORE 1u
#define PAUSING_BLANK_AFTER 1u

#include "interface.h"

#define TASK_UNDEF ((TaskHandle)-1)
#define RESOLUTION_UNDEF 1e-5

template <typename T>
inline T *fastFill(T* p, T x, unsigned int cnt);

XNIDAQmxPulser::XNIDAQmxPulser(const char *name, bool runtime,
	Transaction &tr_meas, const shared_ptr<XMeasure> &meas):
    XNIDAQmxDriver<XPulser>(name, runtime, ref(tr_meas), meas),
    m_pausingBit(0), m_pausingCount(0),
	m_running(false),
    m_resolutionDO(RESOLUTION_UNDEF),
    m_resolutionAO(RESOLUTION_UNDEF),
	m_taskAO(TASK_UNDEF),
	m_taskDO(TASK_UNDEF),
	m_taskDOCtr(TASK_UNDEF),
	m_taskGateCtr(TASK_UNDEF) {

	for(Transaction tr( *this);; ++tr) {
		for(unsigned int i = 0; i < NUM_DO_PORTS; i++)
			tr[ *portSel(i)].add("Pausing(PFI4)");
	    const int ports[] = {
	    	PORTSEL_GATE, PORTSEL_PREGATE, PORTSEL_TRIG1, PORTSEL_TRIG2,
	    	PORTSEL_GATE3, PORTSEL_COMB, PORTSEL_QSW, PORTSEL_ASW
	    };
	    for(unsigned int i = 0; i < sizeof(ports)/sizeof(int); i++) {
	    	tr[ *portSel(i)] = ports[i];
		}
		if(tr.commit())
			break;
	}

	m_softwareTrigger = XNIDAQmxInterface::SoftwareTrigger::create(name, NUM_DO_PORTS);

	m_pausingCount = (PAUSING_BLANK_BEFORE + PAUSING_BLANK_AFTER) * 47;

#if !defined __WIN32__ && !defined WINDOWS
	//memory locks.
 	if(g_bUseMLock) {
		const void *FIRST_OF_MLOCK_MEMBER = &m_genPatternList;
		const void *LAST_OF_MLOCK_MEMBER = &m_lowerLimAO[NUM_AO_CH];
		mlock(FIRST_OF_MLOCK_MEMBER, (size_t)LAST_OF_MLOCK_MEMBER - (size_t)FIRST_OF_MLOCK_MEMBER);
 	}
#endif
}

XNIDAQmxPulser::~XNIDAQmxPulser() {
	clearTasks();
	XNIDAQmxInterface::SoftwareTrigger::unregister(m_softwareTrigger);
}

void
XNIDAQmxPulser::openDO(bool use_ao_clock) throw (XKameError &) {
	XScopedLock<XRecursiveMutex> tlock(m_stateLock);

	if(intfDO()->maxDORate(1) == 0)
		throw XInterface::XInterfaceError(i18n("HW-timed transfer needed."), __FILE__, __LINE__);

	if(m_resolutionDO == RESOLUTION_UNDEF)
		m_resolutionDO = 1.0 / intfDO()->maxDORate(1);
	fprintf(stderr, "Using DO rate = %f[kHz]\n", 1.0/m_resolutionDO);
	setupTasksDO(use_ao_clock);
}

void
XNIDAQmxPulser::openAODO() throw (XKameError &) {
	XScopedLock<XRecursiveMutex> tlock(m_stateLock);

	if(intfDO()->maxDORate(1) == 0)
		throw XInterface::XInterfaceError(i18n("HW-timed transfer needed."), __FILE__, __LINE__);
	if(intfAO()->maxAORate(2) == 0)
		throw XInterface::XInterfaceError(i18n("HW-timed transfer needed."), __FILE__, __LINE__);

	if((m_resolutionDO == RESOLUTION_UNDEF) || (m_resolutionAO == RESOLUTION_UNDEF))
	{
		double do_rate = intfDO()->maxDORate(1);
		double ao_rate = intfAO()->maxAORate(2);
		if(ao_rate <= do_rate)
			do_rate = ao_rate;
		else {
			//Oversampling is unstable.
			int oversamp = 1; //lrint(ao_rate / do_rate);
			ao_rate = do_rate * oversamp;
		}
		m_resolutionDO = 1.0 / do_rate;
		m_resolutionAO = 1.0 / ao_rate;
	}
	fprintf(stderr, "Using AO rate = %f[kHz]\n", 1.0/m_resolutionAO);

	setupTasksAODO();
}

void
XNIDAQmxPulser::close() throw (XKameError &) {
	XScopedLock<XRecursiveMutex> tlock(m_stateLock);

	try {
		stopPulseGen();
	}
	catch (XInterface::XInterfaceError &e) {
		e.print();
	}

	{
		clearTasks();

		m_resolutionDO = RESOLUTION_UNDEF;
		m_resolutionAO = RESOLUTION_UNDEF;

		intfDO()->stop();
		intfAO()->stop();
		intfCtr()->stop();
	}
}
void
XNIDAQmxPulser::clearTasks() {
	if(m_taskAO != TASK_UNDEF)
	    CHECK_DAQMX_RET(DAQmxClearTask(m_taskAO));
	if(m_taskDO != TASK_UNDEF)
	    CHECK_DAQMX_RET(DAQmxClearTask(m_taskDO));
	if(m_taskDOCtr != TASK_UNDEF)
	    CHECK_DAQMX_RET(DAQmxClearTask(m_taskDOCtr));
	if(m_taskGateCtr != TASK_UNDEF)
	    CHECK_DAQMX_RET(DAQmxClearTask(m_taskGateCtr));
	m_taskAO = TASK_UNDEF;
	m_taskDO = TASK_UNDEF;
	m_taskDOCtr = TASK_UNDEF;
	m_taskGateCtr = TASK_UNDEF;
}

void
XNIDAQmxPulser::setupTasksDO(bool use_ao_clock) {
	if(m_taskDO != TASK_UNDEF)
	    CHECK_DAQMX_RET(DAQmxClearTask(m_taskDO));
	if(m_taskDOCtr != TASK_UNDEF)
	    CHECK_DAQMX_RET(DAQmxClearTask(m_taskDOCtr));
	if(m_taskGateCtr != TASK_UNDEF)
	    CHECK_DAQMX_RET(DAQmxClearTask(m_taskGateCtr));

	float64 freq = 1e3 / resolution();

	CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskDO));
    CHECK_DAQMX_RET(DAQmxCreateDOChan(m_taskDO,
									  formatString("%s/port0", intfDO()->devName()).c_str(),
									  "", DAQmx_Val_ChanForAllLines));
	CHECK_DAQMX_RET(DAQmxRegisterDoneEvent(m_taskDO, 0, &XNIDAQmxPulser::onTaskDone_, this));

	XString do_clk_src;

	if(use_ao_clock) {
		do_clk_src = formatString("/%s/ao/SampleClock", intfAO()->devName());
	    fprintf(stderr, "Using ao/SampleClock for DO.\n");
	}
	else {
		do_clk_src = formatString("/%s/Ctr0InternalOutput", intfCtr()->devName());
		XString ctrdev = formatString("%s/ctr0", intfCtr()->devName());
		//Continuous pulse train generation. Duty = 50%.
	    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskDOCtr));
		CHECK_DAQMX_RET(DAQmxCreateCOPulseChanFreq(m_taskDOCtr,
												   ctrdev.c_str(), "", DAQmx_Val_Hz, DAQmx_Val_Low, 0.0,
												   freq, 0.5));
	   	CHECK_DAQMX_RET(DAQmxRegisterDoneEvent(m_taskDOCtr, 0, &XNIDAQmxPulser::onTaskDone_, this));
		CHECK_DAQMX_RET(DAQmxCfgImplicitTiming(m_taskDOCtr, DAQmx_Val_ContSamps, 1000));
	    intfCtr()->synchronizeClock(m_taskDOCtr);
		m_softwareTrigger->setArmTerm(do_clk_src.c_str());
	}

	unsigned int buf_size_hint = (unsigned int)lrint(1.0 * freq); //1sec.
	//M series needs an external sample clock and trigger for DO channels.
	CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskDO,
										  do_clk_src.c_str(),
										  freq, DAQmx_Val_Rising, DAQmx_Val_ContSamps, buf_size_hint));
//    intfDO()->synchronizeClock(m_taskDO); //not applicable for M series.

	uInt32 onbrdsize, bufsize;
	CHECK_DAQMX_RET(DAQmxGetBufOutputOnbrdBufSize(m_taskDO, &onbrdsize));
	fprintf(stderr, "DO On-board bufsize = %d\n", (int)onbrdsize);
	if(m_pausingBit)
		buf_size_hint /= 4;
	CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskDO, &bufsize));
	if(bufsize < buf_size_hint)
		CHECK_DAQMX_RET(DAQmxCfgOutputBuffer(m_taskDO, std::max((uInt32)buf_size_hint, onbrdsize / 4)));
	CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskDO, &bufsize));
	fprintf(stderr, "DO Using bufsize = %d, freq = %f\n", (int)bufsize, freq);
	m_preFillSizeDO = bufsize / 2;
	m_transferSizeHintDO = std::min((unsigned int)onbrdsize / 2, m_preFillSizeDO / 16);
	CHECK_DAQMX_RET(DAQmxSetWriteRegenMode(m_taskDO, DAQmx_Val_DoNotAllowRegen));

	{
//		CHECK_DAQMX_RET(DAQmxSetWriteWaitMode(m_taskDO, DAQmx_Val_Poll));
//		CHECK_DAQMX_RET(DAQmxSetWriteSleepTime(m_taskDO, 0.001));
		char ch[256];
		CHECK_DAQMX_RET(DAQmxGetTaskChannels(m_taskDO, ch, sizeof(ch)));
		if(intfDO()->productFlags() & XNIDAQmxInterface::FLAG_BUGGY_DMA_DO) {
			CHECK_DAQMX_RET(DAQmxSetDODataXferMech(m_taskDO, ch,
												   DAQmx_Val_Interrupts));
		}
		if(intfDO()->productFlags() & XNIDAQmxInterface::FLAG_BUGGY_XFER_COND_DO) {
			CHECK_DAQMX_RET(DAQmxSetDODataXferReqCond(m_taskDO, ch,
													  DAQmx_Val_OnBrdMemNotFull));
		}
	}

	if(m_pausingBit) {
		m_pausingGateTerm = formatString("/%s/PFI4", intfDO()->devName());
		unsigned int ctr_no = use_ao_clock ? 0 : 1;
		m_pausingCh = formatString("%s/ctr%u", intfCtr()->devName(), ctr_no);
		m_pausingSrcTerm = formatString("/%s/Ctr%uInternalOutput", intfCtr()->devName(), ctr_no);
		//set idle state to low level for synchronization.
		CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskGateCtr));
		CHECK_DAQMX_RET(DAQmxCreateCOPulseChanTime(m_taskGateCtr,
												   m_pausingCh.c_str(), "", DAQmx_Val_Seconds, DAQmx_Val_Low,
												   PAUSING_BLANK_BEFORE * resolution() * 1e-3,
												   PAUSING_BLANK_AFTER * resolution() * 1e-3,
												   m_pausingCount * resolution() * 1e-3));
		CHECK_DAQMX_RET(DAQmxCfgImplicitTiming(m_taskGateCtr,
											   DAQmx_Val_FiniteSamps, 1));
		CHECK_DAQMX_RET(DAQmxSetCOCtrTimebaseActiveEdge(m_taskGateCtr,
			 m_pausingCh.c_str(), DAQmx_Val_Rising));
		intfCtr()->synchronizeClock(m_taskGateCtr);

		CHECK_DAQMX_RET(DAQmxCfgDigEdgeStartTrig(m_taskGateCtr,
												 m_pausingGateTerm.c_str(),
												 DAQmx_Val_Rising));
		CHECK_DAQMX_RET(DAQmxSetStartTrigRetriggerable(m_taskGateCtr, true));
//		CHECK_DAQMX_RET(DAQmxSetDigEdgeStartTrigDigSyncEnable(m_taskGateCtr, true));
		if( !use_ao_clock) {
			char doch[256];
			CHECK_DAQMX_RET(DAQmxGetTaskChannels(m_taskDOCtr, doch, 256));
			CHECK_DAQMX_RET(DAQmxSetCOCtrTimebaseActiveEdge(m_taskDOCtr,
				 doch, DAQmx_Val_Falling));
			CHECK_DAQMX_RET(DAQmxSetPauseTrigType(m_taskDOCtr, DAQmx_Val_DigLvl));
			CHECK_DAQMX_RET(DAQmxSetDigLvlPauseTrigSrc(m_taskDOCtr, m_pausingSrcTerm.c_str()));
			CHECK_DAQMX_RET(DAQmxSetDigLvlPauseTrigWhen(m_taskDOCtr, DAQmx_Val_High));
//			CHECK_DAQMX_RET(DAQmxSetDigLvlPauseTrigDigSyncEnable(m_taskDOCtr, true));
		}
	}
}
void
XNIDAQmxPulser::setupTasksAODO() {
	if(m_taskAO != TASK_UNDEF)
	    CHECK_DAQMX_RET(DAQmxClearTask(m_taskAO));

    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskAO));
	CHECK_DAQMX_RET(DAQmxCreateAOVoltageChan(m_taskAO,
											 formatString("%s/ao0:1", intfAO()->devName()).c_str(), "",
											 -1.0, 1.0, DAQmx_Val_Volts, NULL));
	CHECK_DAQMX_RET(DAQmxRegisterDoneEvent(m_taskAO, 0, &XNIDAQmxPulser::onTaskDone_, this));

	float64 freq = 1e3 / resolutionQAM();
	unsigned int buf_size_hint = (unsigned int)lrint(1.0 * freq); //1sec.

	CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskAO, "",
										  freq, DAQmx_Val_Rising, DAQmx_Val_ContSamps, buf_size_hint));
    intfAO()->synchronizeClock(m_taskAO);

    int oversamp = lrint(resolution() / resolutionQAM());
	setupTasksDO(oversamp == 1);
    if(oversamp != 1) {
		//Synchronizes ARM.
		if(m_pausingBit) {
			CHECK_DAQMX_RET(DAQmxSetArmStartTrigType(m_taskDOCtr, DAQmx_Val_DigEdge));
			CHECK_DAQMX_RET(DAQmxSetDigEdgeArmStartTrigSrc(m_taskDOCtr,
														   formatString("/%s/ao/StartTrigger", intfAO()->devName()).c_str()));
			CHECK_DAQMX_RET(DAQmxSetDigEdgeArmStartTrigEdge(m_taskDOCtr,
															DAQmx_Val_Rising));
		}
		else {
			CHECK_DAQMX_RET(DAQmxCfgDigEdgeStartTrig(m_taskDOCtr,
													 formatString("/%s/ao/StartTrigger", intfAO()->devName()).c_str(),
													 DAQmx_Val_Rising));
		}
    }

	if(m_pausingBit) {
		CHECK_DAQMX_RET(DAQmxSetSampClkTimebaseActiveEdge(m_taskAO, DAQmx_Val_Rising));
		CHECK_DAQMX_RET(DAQmxSetPauseTrigType(m_taskAO, DAQmx_Val_DigLvl));
		CHECK_DAQMX_RET(DAQmxSetDigLvlPauseTrigSrc(m_taskAO, m_pausingSrcTerm.c_str()));
		CHECK_DAQMX_RET(DAQmxSetDigLvlPauseTrigWhen(m_taskAO, DAQmx_Val_High));
	}

	m_softwareTrigger->setArmTerm(
		formatString("/%s/ao/SampleClock", intfAO()->devName()).c_str());

	//Buffer setup.
	uInt32 onbrdsize, bufsize;
//	CHECK_DAQMX_RET(DAQmxSetBufOutputOnbrdBufSize(m_taskAO, 4096u));
	CHECK_DAQMX_RET(DAQmxGetBufOutputOnbrdBufSize(m_taskAO, &onbrdsize));
	fprintf(stderr, "AO On-board bufsize = %d\n", (int)onbrdsize);
	if(m_pausingBit)
		buf_size_hint /= 4;
	CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskAO, &bufsize));
	if(bufsize < buf_size_hint)
		CHECK_DAQMX_RET(DAQmxCfgOutputBuffer(m_taskAO, std::max((uInt32)buf_size_hint, onbrdsize / 4)));
	CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskAO, &bufsize));
	fprintf(stderr, "AO Using bufsize = %d\n", (int)bufsize);
	m_preFillSizeAO = m_preFillSizeDO * oversamp;
	m_transferSizeHintAO = std::min((unsigned int)onbrdsize / 2, m_transferSizeHintDO * oversamp);
	CHECK_DAQMX_RET(DAQmxSetWriteRegenMode(m_taskAO, DAQmx_Val_DoNotAllowRegen));

	{
//		CHECK_DAQMX_RET(DAQmxSetWriteWaitMode(m_taskAO, DAQmx_Val_Poll));
//		CHECK_DAQMX_RET(DAQmxSetWriteSleepTime(m_taskAO, 0.001));
		char ch[256];
		CHECK_DAQMX_RET(DAQmxGetTaskChannels(m_taskAO, ch, sizeof(ch)));
		if(intfAO()->productFlags() & XNIDAQmxInterface::FLAG_BUGGY_DMA_AO) {
			CHECK_DAQMX_RET(DAQmxSetAODataXferMech(m_taskAO, ch,
												   DAQmx_Val_Interrupts));
		}
		if(intfAO()->productFlags() & XNIDAQmxInterface::FLAG_BUGGY_XFER_COND_AO) {
			CHECK_DAQMX_RET(DAQmxSetAODataXferReqCond(m_taskAO, ch,
													  DAQmx_Val_OnBrdMemNotFull));
		}
//		CHECK_DAQMX_RET(DAQmxSetAOReglitchEnable(m_taskAO, ch, false));
	}
	//Voltage calibration/ranges.
	for(unsigned int ch = 0; ch < NUM_AO_CH; ch++) {
		//obtain range info.
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++)
			m_coeffAODev[ch][i] = 0.0;
		CHECK_DAQMX_RET(DAQmxGetAODevScalingCoeff(m_taskAO,
												  formatString("%s/ao%d", intfAO()->devName(), ch).c_str(),
												  m_coeffAODev[ch], CAL_POLY_ORDER));
		CHECK_DAQMX_RET(DAQmxGetAODACRngHigh(m_taskAO,
											 formatString("%s/ao%d", intfAO()->devName(), ch).c_str(),
											 &m_upperLimAO[ch]));
		CHECK_DAQMX_RET(DAQmxGetAODACRngLow(m_taskAO,
											formatString("%s/ao%d", intfAO()->devName(), ch).c_str(),
											&m_lowerLimAO[ch]));
	}

/*	CHECK_DAQMX_RET(DAQmxSetAOIdleOutputBehavior(m_taskAO,
	formatString("%s/ao0:1", intfAO()->devName()).c_str(),
	DAQmx_Val_ZeroVolts));
*/
}
int32
XNIDAQmxPulser::onTaskDone_(TaskHandle task, int32 status, void *data) {
	XNIDAQmxPulser *obj = static_cast<XNIDAQmxPulser*>(data);
	obj->onTaskDone(task, status);
	return status;
}
void
XNIDAQmxPulser::onTaskDone(TaskHandle task, int32 status) {
	if(status) {
		XString str;
		if(task == m_taskDO) { str = "DO"; }
		if(task == m_taskDOCtr) { str = "DOCtr"; }
		if(task == m_taskAO) { str = "AO"; }
		if(task == m_taskGateCtr) { str = "GateCtr"; }
		gErrPrint(getLabel() + "\n" + str + "\n" + XNIDAQmxInterface::getNIDAQmxErrMessage(status));
		try {
			stopPulseGen();
		}
		catch (XInterface::XInterfaceError &e) {
//			e.print();
		}
	}
}
template <typename T>
inline T *
fastFill(T* p, T x, unsigned int cnt) {
	if(cnt > 100) {
		for(;(intptr_t)p % (sizeof(uint64_t) / sizeof(T)); cnt--)
			*p++ = x;
		uint64_t *pp = (uint64_t *)p;
		union {
			uint64_t dw; T w[sizeof(uint64_t) / sizeof(T)];
		} pack;
		for(unsigned int i = 0; i < (sizeof(uint64_t) / sizeof(T)); i++)
			pack.w[i] = x;
		unsigned int pcnt = cnt / (sizeof(uint64_t) / sizeof(T));
		cnt = cnt % (sizeof(uint64_t) / sizeof(T));
		for(unsigned int i = 0; i < pcnt; i++)
			*pp++ = pack.dw;
		p = (T*)pp;
	}
	for(unsigned int i = 0; i < cnt; i++)
		*p++ = x;
	return p;
}

void
XNIDAQmxPulser::preparePatternGen(const Snapshot &shot,
		bool use_dummypattern, unsigned int blankpattern) {
	if(use_dummypattern) {
		//Creates dummy pattern.
    	shared_ptr<std::vector<GenPattern> > patlist_dummy(new std::vector<GenPattern>());
    	patlist_dummy->push_back(GenPattern(blankpattern, 100000));
    	patlist_dummy->push_back(GenPattern(blankpattern, 100000));
		m_genPatternList = patlist_dummy;
		for(unsigned int j = 0; j < PAT_QAM_MASK / PAT_QAM_PHASE; j++) {
			m_genPulseWaveAO[j].reset();
		}
	}
	else {
		//Copies generated pattern from Payload.
		m_genPatternList = shot[ *this].m_genPatternListNext;
		for(unsigned int j = 0; j < PAT_QAM_MASK / PAT_QAM_PHASE; j++) {
			m_genPulseWaveAO[j] = shot[ *this].m_genPulseWaveNextAO[j];
		}
		m_genAOZeroLevel = shot[ *this].m_genAOZeroLevelNext;
	}

	//prepares ring buffers for pattern generation.
	m_genLastPatIt = m_genPatternList->begin();
	m_genRestCount = m_genPatternList->front().tonext;
	m_genTotalCount += m_genPatternList->front().tonext;
	m_patBufDO.reserve(m_preFillSizeDO);
	if(m_taskAO != TASK_UNDEF) {
		m_genAOIndex = 0;
		m_patBufAO.reserve(m_preFillSizeAO);
	}

	startBufWriter();
}

void
XNIDAQmxPulser::startPulseGen(const Snapshot &shot) throw (XKameError &) {
	XScopedLock<XRecursiveMutex> tlock(m_stateLock);
	if(m_running && m_softwareTrigger->isPersistentCoherentMode()) {
		startPulseGenFromFreeRun(shot);
		return;
	}

	stopPulseGen();

	unsigned int pausingbit = selectedPorts(shot, PORTSEL_PAUSING);
	m_aswBit = selectedPorts(shot, PORTSEL_ASW);

	if((m_taskDO == TASK_UNDEF) ||
	   (m_pausingBit != pausingbit)) {
		m_pausingBit = pausingbit;
		clearTasks();
		if(hasQAMPorts())
			setupTasksAODO();
		else
			setupTasksDO(false);
	}
	{
		uInt32 bufsize;
		CHECK_DAQMX_RET(DAQmxGetBufOutputOnbrdBufSize(m_taskDO, &bufsize));
		if( !m_pausingBit & (bufsize < 2047uL))
			throw XInterface::XInterfaceError(
				i18n("Use the pausing feature for a cheap DAQmx board.") + "\n"
						   + i18n("Look at the port-selection table."), __FILE__, __LINE__);
	}
	if(m_taskAO != TASK_UNDEF) {
		uInt32 bufsize;
		CHECK_DAQMX_RET(DAQmxGetBufOutputOnbrdBufSize(m_taskAO, &bufsize));
		if( !m_pausingBit & (bufsize < 8192uL))
			throw XInterface::XInterfaceError(
				i18n("Use the pausing feature for a cheap DAQmx board.") + "\n"
						   + i18n("Look at the port-selection table."), __FILE__, __LINE__);
	}

	m_genTotalCount = 0;
	m_genTotalSamps = 0;

	const unsigned int cnt_prezeros = 1000;
	m_genTotalCount += cnt_prezeros;
	m_genTotalSamps += cnt_prezeros;
	//prefilling of the buffers.
	const unsigned int oversamp_ao = lrint(resolution() / resolutionQAM());
	if(m_taskAO != TASK_UNDEF) {
		//Pads preceding zeros.
		CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskAO, DAQmx_Val_FirstSample));
		CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskAO, 0));
		const unsigned int cnt_prezeros_ao = cnt_prezeros * oversamp_ao - 0;
		m_genAOZeroLevel = shot[ *this].m_genAOZeroLevelNext;
		std::vector<tRawAOSet> zeros(cnt_prezeros_ao, m_genAOZeroLevel);
		int32 samps;
		CHECK_DAQMX_RET(DAQmxWriteBinaryI16(m_taskAO, cnt_prezeros_ao,
											false, 0.5,
											DAQmx_Val_GroupByScanNumber,
											zeros[0].ch,
											&samps, NULL));
		CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskAO, DAQmx_Val_CurrWritePos));
		CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskAO, 0));
	}
	//Pads preceding zeros.
	std::vector<tRawDO> zeros(cnt_prezeros, 0);

	CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskDO, DAQmx_Val_FirstSample));
	CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskDO, 0));
	int32 samps;
	CHECK_DAQMX_RET(DAQmxWriteDigitalU16(m_taskDO, cnt_prezeros,
										 false, 0.0,
										 DAQmx_Val_GroupByScanNumber,
										 &zeros[0],
										 &samps, NULL));
	CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskDO, DAQmx_Val_CurrWritePos));
	CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskDO, 0));

	//synchronizes with the software trigger.
	m_softwareTrigger->start(1e3 / resolution());

	m_totalWrittenSampsDO = cnt_prezeros;
	m_totalWrittenSampsAO = cnt_prezeros * oversamp_ao;

	m_isThreadWriterReady = false; //to be set to true in the buffer-writing thread.

	preparePatternGen(shot, false, 0);

	CHECK_DAQMX_RET(DAQmxTaskControl(m_taskDO, DAQmx_Val_Task_Commit));
	if(m_taskDOCtr != TASK_UNDEF)
		CHECK_DAQMX_RET(DAQmxTaskControl(m_taskDOCtr, DAQmx_Val_Task_Commit));
	if(m_taskGateCtr != TASK_UNDEF)
		CHECK_DAQMX_RET(DAQmxTaskControl(m_taskGateCtr, DAQmx_Val_Task_Commit));
	if(m_taskAO != TASK_UNDEF)
		CHECK_DAQMX_RET(DAQmxTaskControl(m_taskAO, DAQmx_Val_Task_Commit));

	//Waits for buffer filling.
	while( !m_isThreadWriterReady) {
		if(m_threadWriter->isTerminated())
			return;
		usleep(1000);
	}
	{
		//Recovers from open state.
		char ch[256];
		CHECK_DAQMX_RET(DAQmxGetTaskChannels(m_taskDO, ch, sizeof(ch)));
		CHECK_DAQMX_RET(DAQmxSetDOTristate(m_taskDO, ch, false));
	}
	//	fprintf(stderr, "Prefilling done.\n");
	//Slaves must start before the master.
	CHECK_DAQMX_RET(DAQmxStartTask(m_taskDO));
	if(m_taskGateCtr != TASK_UNDEF)
		CHECK_DAQMX_RET(DAQmxStartTask(m_taskGateCtr));
	if(m_taskDOCtr != TASK_UNDEF)
		CHECK_DAQMX_RET(DAQmxStartTask(m_taskDOCtr));
//	fprintf(stderr, "Starting AO....\n");
	if(m_taskAO != TASK_UNDEF)
		CHECK_DAQMX_RET(DAQmxStartTask(m_taskAO));
	m_running = true;
}
void
XNIDAQmxPulser::startBufWriter() {
	//Starting threads that writing buffers concurrently.
	m_threadWriter.reset(new XThread<XNIDAQmxPulser>(shared_from_this(),
													  &XNIDAQmxPulser::executeWriter));
	m_threadWriter->resume();
}
void
XNIDAQmxPulser::stopBufWriter() {
	//Stops threads.
	if(m_threadWriter) {
		m_threadWriter->terminate();
		m_threadWriter->waitFor();
		m_threadWriter.reset();
	}
}
void
XNIDAQmxPulser::stopPulseGen() {
	XScopedLock<XRecursiveMutex> tlock(m_stateLock);
	stopBufWriter();
	abortPulseGen();
}
void
XNIDAQmxPulser::abortPulseGen() {
	XScopedLock<XRecursiveMutex> tlock(m_stateLock);
	{
		m_softwareTrigger->stop();
		if(m_running) {
			{
				char ch[256];
				CHECK_DAQMX_RET(DAQmxGetTaskChannels(m_taskDO, ch, sizeof(ch)));
				uInt32 num_lines;
				CHECK_DAQMX_RET(DAQmxGetDONumLines(m_taskDO, ch, &num_lines));
				//Sets outputs to open state except for the pausing bit.
				XString chtri;
				for(unsigned int i = 0; i < num_lines; i++) {
					if(m_pausingBit &&
						(lrint(log(m_pausingBit)/log(2.0)) == (int)i))
						continue;
					if(chtri.length())
						chtri = chtri + ",";
					chtri = chtri + formatString("%s/line%u", ch, i);
				}
				CHECK_DAQMX_RET(DAQmxSetDOTristate(m_taskDO, chtri.c_str(), true));
			}
			if(m_taskAO != TASK_UNDEF)
				CHECK_DAQMX_RET(DAQmxStopTask(m_taskAO));
			if(m_taskDOCtr != TASK_UNDEF)
				CHECK_DAQMX_RET(DAQmxStopTask(m_taskDOCtr));
			CHECK_DAQMX_RET(DAQmxStopTask(m_taskDO));
			if(m_taskGateCtr != TASK_UNDEF) {
				CHECK_DAQMX_RET(DAQmxWaitUntilTaskDone (m_taskGateCtr, 0.1));
				CHECK_DAQMX_RET(DAQmxStopTask(m_taskGateCtr));
			}
			if(m_taskAO != TASK_UNDEF)
				CHECK_DAQMX_RET(DAQmxTaskControl(m_taskAO, DAQmx_Val_Task_Unreserve));
			if(m_taskDOCtr != TASK_UNDEF)
				CHECK_DAQMX_RET(DAQmxTaskControl(m_taskDOCtr, DAQmx_Val_Task_Unreserve));
			CHECK_DAQMX_RET(DAQmxTaskControl(m_taskDO, DAQmx_Val_Task_Unreserve));
			if(m_taskGateCtr != TASK_UNDEF)
				CHECK_DAQMX_RET(DAQmxTaskControl(m_taskGateCtr, DAQmx_Val_Task_Unreserve));
		}

		m_running = false;
	}
}

void
XNIDAQmxPulser::rewindBufPos(double ms_from_gen_pos) {
	int32 cnt_from_gen_pos = lrint(ms_from_gen_pos / resolution());
	const unsigned int oversamp_ao = lrint(resolution() / resolutionQAM());
	uInt64 samp_gen;
	CHECK_DAQMX_RET(DAQmxGetWriteTotalSampPerChanGenerated(m_taskDO, &samp_gen));
	if(m_taskAO != TASK_UNDEF) {
		uInt64 samp_gen_ao, currpos_ao;
		CHECK_DAQMX_RET(DAQmxGetWriteTotalSampPerChanGenerated(m_taskAO, &samp_gen_ao));
		samp_gen = std::max(samp_gen_ao / oversamp_ao, samp_gen);
	}
	uint64_t count_gen;
	uint64_t count_old = m_genTotalCount;
	assert(m_genRestCount == 0);
	for(auto it = m_queueTimeGenCnt.begin(); it != m_queueTimeGenCnt.end(); ++it) {
		if(it->second > samp_gen) {
			count_gen = it->first;
			break;
		}
	}
	for(auto it = m_queueTimeGenCnt.begin(); it != m_queueTimeGenCnt.end(); ++it) {
		if(it->first > count_gen + cnt_from_gen_pos) {
			m_genTotalCount = it->first;
			m_genTotalSamps = it->second;
			m_genRestCount = 0;
			break;
		}
	}
	uint64_t currsamps = m_totalWrittenSampsDO;
	if(m_taskAO != TASK_UNDEF)
		currsamps  = std::min(currsamps, m_totalWrittenSampsAO / oversamp_ao);
	if(m_genTotalSamps > currsamps) {
		//Requested time is beyond the current buffer writing position.
		for(auto rit = m_queueTimeGenCnt.rbegin(); rit != m_queueTimeGenCnt.rend(); ++rit) {
			if(rit->second <= currsamps) {
				m_genTotalCount = rit->first;
				m_genTotalSamps = rit->second;
				m_genRestCount = 0;
				break;
			}
		}
	}
	if(m_genTotalSamps > currsamps) {
		m_genTotalSamps = currsamps;
		m_genTotalCount = count_old;
		m_genRestCount = 0;
	}

	CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskDO,  -(int32_t)(m_totalWrittenSampsDO - m_genTotalSamps)));
	if(m_taskAO != TASK_UNDEF) {
		CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskAO,   -(int32_t)(m_totalWrittenSampsAO - m_genTotalSamps * oversamp_ao)));
	}
	fprintf(stderr, "Rewind: %g,%g,%g,%g,%g\n", (double)samp_gen, (double)m_totalWrittenSampsDO,
		(double)count_gen,(double)m_genTotalCount, (double)m_genTotalSamps);;

	m_totalWrittenSampsDO = m_genTotalSamps;
	m_totalWrittenSampsAO = m_genTotalSamps * oversamp_ao;

	//Writes dummy data, because DAQmxGetWriteSpaceAvail() cannot be performed after setting offset.
	const unsigned int cnt_prezeros = 1000;
	msecsleep(cnt_prezeros * resolution());

	m_genTotalCount += cnt_prezeros;
	m_genTotalSamps += cnt_prezeros;
	//prefilling of the buffers.
	if(m_taskAO != TASK_UNDEF) {
		//Pads preceding zeros.
		const unsigned int oversamp_ao = lrint(resolution() / resolutionQAM());
		const unsigned int cnt_prezeros_ao = cnt_prezeros * oversamp_ao - 0;
		std::vector<tRawAOSet> zeros(cnt_prezeros_ao, m_genAOZeroLevel);
		int32 samps;
		CHECK_DAQMX_RET(DAQmxWriteBinaryI16(m_taskAO, cnt_prezeros_ao,
											false, 0.5,
											DAQmx_Val_GroupByScanNumber,
											zeros[0].ch,
											&samps, NULL));
		CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskAO, DAQmx_Val_CurrWritePos));
		CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskAO, 0));
	}
	//Pads preceding zeros.
	std::vector<tRawDO> zeros(cnt_prezeros, 0);

	int32 samps;
	CHECK_DAQMX_RET(DAQmxWriteDigitalU16(m_taskDO, cnt_prezeros,
										 false, 0.0,
										 DAQmx_Val_GroupByScanNumber,
										 &zeros[0],
										 &samps, NULL));
	CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskDO, DAQmx_Val_CurrWritePos));
	CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskDO, 0));

	m_totalWrittenSampsDO += cnt_prezeros;
	m_totalWrittenSampsAO += cnt_prezeros * oversamp_ao;
}
void
XNIDAQmxPulser::stopPulseGenFreeRunning(unsigned int blankpattern) {
	XScopedLock<XRecursiveMutex> tlock(m_stateLock);
	{
		//clears sent software triggers.
		m_softwareTrigger->clear();
		stopBufWriter();

		//sets position padding=150ms. after the current generating position.
		rewindBufPos(150.0);
		preparePatternGen(Snapshot( *this), true, blankpattern);
	}
}
void
XNIDAQmxPulser::startPulseGenFromFreeRun(const Snapshot &shot) {
	//clears sent software triggers.
	m_softwareTrigger->clear();
	stopBufWriter();

	//sets position padding=150ms. after the current generating position.
	rewindBufPos(150.0);
	preparePatternGen(shot, false, 0);
}

inline XNIDAQmxPulser::tRawAOSet
XNIDAQmxPulser::aoVoltToRaw(const double poly_coeff[NUM_AO_CH][CAL_POLY_ORDER], const std::complex<double> &volt) {
	const double volts[] = {volt.real(), volt.imag()};
	tRawAOSet z;
	for(unsigned int ch = 0; ch < NUM_AO_CH; ch++) {
		double x = 1.0;
		double y = 0.0;
		const double *pco = poly_coeff[ch];
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
			y += ( *pco++) * x;
			x *= volts[ch];
		}
		z.ch[ch] = lrint(y);
	}
	return z;
}

void *
XNIDAQmxPulser::executeWriter(const atomic<bool> &terminating) {
 	double dma_do_period = resolution();
 	double dma_ao_period = resolutionQAM();
 	uint64_t written_total_ao = 0, written_total_do = 0;

 	//Starting a child thread generating patterns concurrently.
	XThread<XNIDAQmxPulser> th_genbuf(shared_from_this(),
													  &XNIDAQmxPulser::executeFillBuffer);
	th_genbuf.resume();

	while( !terminating) {
		const tRawDO *pDO = m_patBufDO.curReadPos();
		ssize_t samps_do = m_patBufDO.writtenSize();
		const tRawAOSet *pAO = NULL;
		ssize_t samps_ao = 0;
		if(m_taskAO != TASK_UNDEF) {
			pAO = m_patBufAO.curReadPos();
			samps_ao = m_patBufAO.writtenSize();
		}
		if( !samps_do && !samps_ao) {
			usleep(lrint(std::min(1e3 * m_transferSizeHintDO * dma_do_period,
				1e3 * m_transferSizeHintAO * dma_ao_period) / 2));
			continue;
		}
		try {
			ssize_t written;
			if(samps_ao > samps_do) {
				written = writeToDAQmxAO(pAO, std::min(samps_ao, (ssize_t)m_transferSizeHintAO));
				if(written) m_patBufAO.finReading(written);
				written_total_ao += written;
			}
			else {
				written = writeToDAQmxDO(pDO, std::min(samps_do, (ssize_t)m_transferSizeHintDO));
				if(written) m_patBufDO.finReading(written);
				written_total_do += written;
			}
			if((written_total_do > m_preFillSizeDO) && ( !pAO || (written_total_ao > m_preFillSizeAO)))
				m_isThreadWriterReady = true; //Count written into the devices has exceeded a certain value.
		}
		catch (XInterface::XInterfaceError &e) {
			e.print(getLabel());

			m_threadWriter->terminate();
			XScopedLock<XRecursiveMutex> tlock(m_stateLock);
			if(m_running) {
				try {
					abortPulseGen();
				}
				catch (XInterface::XInterfaceError &e) {
		//			e.print(getLabel());
				}
				break;
			}
		}
	}
	m_totalWrittenSampsDO += written_total_do;
	m_totalWrittenSampsAO += written_total_ao;

	th_genbuf.terminate();
	th_genbuf.waitFor();
	return NULL;
}

ssize_t
XNIDAQmxPulser::writeToDAQmxAO(const tRawAOSet *pAO, ssize_t samps) {
	uInt32 space;
	CHECK_DAQMX_RET(DAQmxGetWriteSpaceAvail(m_taskAO, &space));
	if(space < (uInt32)samps)
		return 0;
	int32 written;
	CHECK_DAQMX_RET(DAQmxWriteBinaryI16(m_taskAO, samps, false, 0.0,
										DAQmx_Val_GroupByScanNumber,
										const_cast<tRawAOSet *>(pAO)->ch,
										&written, NULL));
	return written;
}
ssize_t
XNIDAQmxPulser::writeToDAQmxDO(const tRawDO *pDO, ssize_t samps) {
	uInt32 space;
	CHECK_DAQMX_RET(DAQmxGetWriteSpaceAvail(m_taskDO, &space));
	if(space < (uInt32)samps)
		return 0;
	int32 written;
	CHECK_DAQMX_RET(DAQmxWriteDigitalU16(m_taskDO, samps, false, 0.0,
										 DAQmx_Val_GroupByScanNumber,
										 const_cast<tRawDO *>(pDO),
										 &written, NULL));
	return written;
}

template <bool UseAO>
inline bool
XNIDAQmxPulser::fillBuffer() {
	unsigned int oversamp_ao = lrint(resolution() / resolutionQAM());

	GenPatternIterator it = m_genLastPatIt;
	uint32_t pat = it->pattern;
	uint64_t tonext = m_genRestCount;
	uint64_t total = m_genTotalCount;
	unsigned int aoidx = m_genAOIndex;
	tRawDO pausingbit = m_pausingBit;
	tRawDO aswbit = m_aswBit;
	uint64_t pausing_cnt = m_pausingCount;
	uint64_t pausing_cnt_blank_before = PAUSING_BLANK_BEFORE + PAUSING_BLANK_AFTER;
	uint64_t pausing_cnt_blank_after = 1;
	uint64_t pausing_period = pausing_cnt + pausing_cnt_blank_before + pausing_cnt_blank_after;
	uint64_t pausing_cost = std::max(16uLL, pausing_cnt_blank_before + pausing_cnt_blank_after);

	shared_ptr<XNIDAQmxInterface::SoftwareTrigger> &vt = m_softwareTrigger;

	tRawDO *pDO = m_patBufDO.curWritePos();
	tRawDO *pDOorg = pDO;
	if( !pDO)
		return false;
	tRawAOSet *pAO = (UseAO) ? m_patBufAO.curWritePos() : NULL;
	if( !pAO && UseAO)
		return false;
	tRawAOSet raw_zero = m_genAOZeroLevel;
	ssize_t capacity = m_patBufDO.chunkSize();
	if(UseAO)
		capacity = std::min(capacity, m_patBufAO.chunkSize() / (ssize_t)oversamp_ao);
	for(unsigned int samps_rest = capacity; samps_rest;) {
		unsigned int pidx = (pat & PAT_QAM_PULSE_IDX_MASK) / PAT_QAM_PULSE_IDX;
		//Bits for digital lines.
		tRawDO patDO = PAT_DO_MASK & pat;
		//Case:  QAM and ASW is off.
		if(pausingbit && (pidx == 0) && !(pat & aswbit)) {
			//generates a pausing trigger.
			assert(tonext > 0);
			unsigned int lps = (unsigned int)std::min(
				(uint64_t)(samps_rest / pausing_cost), (tonext - 1) / pausing_period);
			patDO &= ~pausingbit;
			if(lps) {
				tonext -= lps * pausing_period;
				samps_rest -= lps * pausing_cost;
				tRawDO patDO_or_p = patDO | pausingbit;
				for(unsigned int lp = 0; lp < lps; lp++) {
					for(int i = 0; i < pausing_cnt_blank_before; i++)
						*pDO++ = patDO_or_p;
					for(int i = 0; i < pausing_cnt_blank_after; i++)
						*pDO++ = patDO;
				}
				if(UseAO)
					pAO = fastFill(pAO, raw_zero, lps * oversamp_ao * (pausing_cnt_blank_before + pausing_cnt_blank_after));
			}
			assert(tonext > 0);
			if(samps_rest < pausing_cost)
				break; //necessary for frequent tagging of buffer.
		}
		//number of samples to be written into buffer.
		unsigned int gen_cnt = std::min((uint64_t)samps_rest, tonext);
		//writes digital pattern.
		pDO = fastFill(pDO, patDO, gen_cnt);

		if(UseAO) {
			if(pidx == 0) {
				//writes a blank in analog lines.
				aoidx = 0;
				pAO = fastFill(pAO, raw_zero, gen_cnt * oversamp_ao);
			}
			else {
				unsigned int pnum = (pat & PAT_QAM_MASK) / PAT_QAM_PHASE - (PAT_QAM_PULSE_IDX/PAT_QAM_PHASE);
				assert(pnum < PAT_QAM_PULSE_IDX_MASK / PAT_QAM_PULSE_IDX);
				if( !m_genPulseWaveAO[pnum].get() ||
				   (m_genPulseWaveAO[pnum]->size() < gen_cnt * oversamp_ao + aoidx))
					throw XInterface::XInterfaceError(i18n("Invalid pulse patterns."), __FILE__, __LINE__);
				tRawAOSet *pGenAO = &m_genPulseWaveAO[pnum]->at(aoidx);
				memcpy(pAO, pGenAO, gen_cnt * oversamp_ao * sizeof(tRawAOSet));
				pAO += gen_cnt * oversamp_ao;
				aoidx += gen_cnt * oversamp_ao;
			}
		}
		tonext -= gen_cnt;
		samps_rest -= gen_cnt;
		if(tonext == 0) {
			it++;
			if(it == m_genPatternList->end()) {
				it = m_genPatternList->begin();
//				printf("p.\n");
			}
			tonext = it->tonext;
			vt->changeValue(pat, it->pattern, total);
			pat = it->pattern;
			total += tonext;
		}
	}
	m_genRestCount = tonext;
	m_genLastPatIt = it;
	m_genAOIndex = aoidx;
	m_genTotalCount = total;
	m_genTotalSamps += pDO - pDOorg;
	//Here ensures the pattern data is ready.
	m_patBufDO.finWriting(pDO);
	if(UseAO)
		m_patBufAO.finWriting(pAO);
	return true;
}
void *
XNIDAQmxPulser::executeFillBuffer(const atomic<bool> &terminating) {
	m_queueTimeGenCnt.clear();
	while( !terminating) {
		bool buffer_not_full;
		if(m_taskAO != TASK_UNDEF) {
			buffer_not_full = fillBuffer<true>();
		}
		else {
			buffer_not_full = fillBuffer<false>();
		}
		if( !m_queueTimeGenCnt.size() ||
			(m_genTotalCount - m_genRestCount - m_queueTimeGenCnt.back().first > lrint(20.0 / resolution()))) {
			m_queueTimeGenCnt.push_back(std::pair<uint64_t, uint64_t>(
				m_genTotalCount - m_genRestCount, m_genTotalSamps)); //preserves every 20ms.
		}
		while(m_genTotalCount -  m_genRestCount - m_queueTimeGenCnt.front().first > lrint(20000.0 / resolution())) {
			m_queueTimeGenCnt.pop_front(); //limits only within last 20s.
		}
		if( !buffer_not_full) {
			//Waiting until previous data have been sent.
		 	double dma_do_period = resolution();
			usleep(lrint(1e3 * m_transferSizeHintDO * dma_do_period / 2));
		}
	}
	m_genTotalCount -= m_genRestCount;
	m_genRestCount = 0;
	return NULL;
}
void
XNIDAQmxPulser::createNativePatterns(Transaction &tr) {
	const Snapshot &shot(tr);
	tr[ *this].m_genPatternListNext.reset(new std::vector<GenPattern>);
	uint32_t pat = shot[ *this].relPatList().back().pattern;
	for(Payload::RelPatList::const_iterator it = shot[ *this].relPatList().begin();
		it != shot[ *this].relPatList().end(); it++) {
		uint64_t tonext = it->toappear;

		GenPattern genpat(pat, tonext);
		tr[ *this].m_genPatternListNext->push_back(genpat);
		pat = it->pattern;
	}

	if(hasQAMPorts()) {
		double offset[] = { shot[ *qamOffset1()], shot[ *qamOffset2()]};
		double level[] = { shot[ *qamLevel1()], shot[ *qamLevel2()]};
    	double coeffAO[NUM_AO_CH][CAL_POLY_ORDER];

		for(unsigned int ch = 0; ch < NUM_AO_CH; ch++) {
			//arranges range info.
			double x = 1.0;
			for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
				coeffAO[ch][i] = m_coeffAODev[ch][i] * x
					+ ((i == 0) ? offset[ch] : 0);
				x *= level[ch];
			}
		}
		tr[ *this].m_genAOZeroLevelNext = aoVoltToRaw(coeffAO, std::complex<double>(0.0));
		std::complex<double> c(pow(10.0, shot[ *this].masterLevel() / 20.0), 0);
		for(unsigned int i = 0; i < PAT_QAM_PULSE_IDX_MASK/PAT_QAM_PULSE_IDX; i++) {
			for(unsigned int qpsk = 0; qpsk < 4; qpsk++) {
				const unsigned int pnum = i * (PAT_QAM_PULSE_IDX/PAT_QAM_PHASE) + qpsk;
				tr[ *this].m_genPulseWaveNextAO[pnum].reset(new std::vector<tRawAOSet>);
				for(std::vector<std::complex<double> >::const_iterator it =
						shot[ *this].qamWaveForm(i).begin(); it != shot[ *this].qamWaveForm(i).end(); it++) {
					std::complex<double> z( *it * c);
					tr[ *this].m_genPulseWaveNextAO[pnum]->push_back(aoVoltToRaw(coeffAO, z));
				}
				c *= std::complex<double>(0,1);
			}
		}
  	}
}


void
XNIDAQmxPulser::changeOutput(const Snapshot &shot, bool output, unsigned int blankpattern) {
	XScopedLock<XInterface> lock( *interface());
	if( !interface()->isOpened())
		return;
	XScopedLock<XRecursiveMutex> tlock(m_stateLock);
	if(output) {
		if( !shot[ *this].m_genPatternListNext || shot[ *this].m_genPatternListNext->empty() )
			throw XInterface::XInterfaceError(i18n("Pulser Invalid pattern"), __FILE__, __LINE__);
		startPulseGen(shot);
	}
	else {
		if(m_running && m_softwareTrigger->isPersistentCoherentMode())
			stopPulseGenFreeRunning(blankpattern);
		else
			stopPulseGen();
	}
}
