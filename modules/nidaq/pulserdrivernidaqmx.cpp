/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "pulserdrivernidaqmx.h"

#define CLEAR_TASKS_EVERYTIME 0u

#define PAUSING_BLANK_BEFORE 1u
#define PAUSING_BLANK_AFTER 1u

#include "interface.h"

#define TASK_UNDEF ((TaskHandle)-1)
#define RESOLUTION_UNDEF 1e-5

template <typename T>
inline T *fastFill(T* p, T x, unsigned int cnt);

XNIDAQmxPulser::XNIDAQmxPulser(const char *name, bool runtime,
							   const shared_ptr<XScalarEntryList> &scalarentries,
							   const shared_ptr<XInterfaceList> &interfaces,
							   const shared_ptr<XThermometerList> &thermometers,
							   const shared_ptr<XDriverList> &drivers) :
    XNIDAQmxDriver<XPulser>(name, runtime, scalarentries, interfaces, thermometers, drivers),
    m_pausingBit(0), m_pausingCount(0), 
	m_running(false),
    m_resolutionDO(RESOLUTION_UNDEF),
    m_resolutionAO(RESOLUTION_UNDEF),
	m_taskAO(TASK_UNDEF),
	m_taskDO(TASK_UNDEF),
	m_taskDOCtr(TASK_UNDEF),
	m_taskGateCtr(TASK_UNDEF)
{
	for(unsigned int i = 0; i < NUM_DO_PORTS; i++)
		portSel(i)->add("Pausing(PFI4)");
    const int ports[] = {
    	PORTSEL_GATE, PORTSEL_PREGATE, PORTSEL_TRIG1, PORTSEL_TRIG2,
    	PORTSEL_GATE3, PORTSEL_COMB, PORTSEL_QSW, PORTSEL_ASW
    };
    for(unsigned int i = 0; i < sizeof(ports)/sizeof(int); i++) {
    	portSel(i)->value(ports[i]);
	}
	m_softwareTrigger = XNIDAQmxInterface::SoftwareTrigger::create(name, NUM_DO_PORTS);
	
	m_pausingCount = (PAUSING_BLANK_BEFORE + PAUSING_BLANK_AFTER) * 47;
	
	//memory locks.
 	if(g_bUseMLock) {
		const void *FIRST_OF_MLOCK_MEMBER = &m_genPatternList;
		const void *LAST_OF_MLOCK_MEMBER = &m_lowerLimAO[NUM_AO_CH];
		mlock(FIRST_OF_MLOCK_MEMBER, (size_t)LAST_OF_MLOCK_MEMBER - (size_t)FIRST_OF_MLOCK_MEMBER);
 	}
}

XNIDAQmxPulser::~XNIDAQmxPulser()
{
	clearTasks();
	XNIDAQmxInterface::SoftwareTrigger::unregister(m_softwareTrigger);
}

void
XNIDAQmxPulser::openDO(bool use_ao_clock) throw (XInterface::XInterfaceError &)
{
	XScopedLock<XRecursiveMutex> tlock(m_totalLock);

	if(intfDO()->maxDORate(1) == 0)
		throw XInterface::XInterfaceError(i18n("HW-timed transfer needed."), __FILE__, __LINE__);
	
	if(m_resolutionDO == RESOLUTION_UNDEF)
		m_resolutionDO = 1.0 / intfDO()->maxDORate(1);
	fprintf(stderr, "Using DO rate = %f[kHz]\n", 1.0/m_resolutionDO);
	setupTasksDO(use_ao_clock);
		
	m_suspendDO = true; 	
	m_threadWriteDO.reset(new XThread<XNIDAQmxPulser>(shared_from_this(),
													  &XNIDAQmxPulser::executeWriteDO));
	m_threadWriteDO->resume();
}

void
XNIDAQmxPulser::openAODO() throw (XInterface::XInterfaceError &)
{
	XScopedLock<XRecursiveMutex> tlock(m_totalLock);
	
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

	m_suspendDO = true; 	
	m_threadWriteDO.reset(new XThread<XNIDAQmxPulser>(shared_from_this(),
													  &XNIDAQmxPulser::executeWriteDO));
	m_threadWriteDO->resume();

	m_suspendAO = true;
	m_threadWriteAO.reset(new XThread<XNIDAQmxPulser>(shared_from_this(),
													  &XNIDAQmxPulser::executeWriteAO));
	m_threadWriteAO->resume();
}

void
XNIDAQmxPulser::close() throw (XInterface::XInterfaceError &)
{
	XScopedLock<XRecursiveMutex> tlock(m_totalLock);

	try {
		stopPulseGen();
	}
	catch (XInterface::XInterfaceError &e) {
		e.print();
	}
	
	if(m_threadWriteAO) {
		m_threadWriteAO->terminate();
	}
	if(m_threadWriteDO) {
		m_threadWriteDO->terminate();
	}

	XScopedLock<XRecursiveMutex> lockAO(m_mutexAO);
	XScopedLock<XRecursiveMutex> lockDO(m_mutexDO);
	
	clearTasks();

	m_resolutionDO = RESOLUTION_UNDEF;    
	m_resolutionAO = RESOLUTION_UNDEF;    

	intfDO()->stop();
	intfAO()->stop();
	intfCtr()->stop();
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
	CHECK_DAQMX_RET(DAQmxRegisterDoneEvent(m_taskDO, 0, &XNIDAQmxPulser::_onTaskDone, this));

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
	   	CHECK_DAQMX_RET(DAQmxRegisterDoneEvent(m_taskDOCtr, 0, &XNIDAQmxPulser::_onTaskDone, this));
		CHECK_DAQMX_RET(DAQmxCfgImplicitTiming(m_taskDOCtr, DAQmx_Val_ContSamps, 1000));
	    intfCtr()->synchronizeClock(m_taskDOCtr);
		m_softwareTrigger->setArmTerm(do_clk_src.c_str());
	}
   
	unsigned int buf_size_hint = (unsigned int)lrint(2.0 * freq);
	//M series needs an external sample clock and trigger for DO channels.
	CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskDO,
										  do_clk_src.c_str(),
										  freq, DAQmx_Val_Rising, DAQmx_Val_ContSamps, buf_size_hint));
//    intfDO()->synchronizeClock(m_taskDO);
	
	uInt32 onbrdsize, bufsize;
	CHECK_DAQMX_RET(DAQmxGetBufOutputOnbrdBufSize(m_taskDO, &onbrdsize));
	fprintf(stderr, "On-board bufsize = %d\n", (int)onbrdsize);
	if(onbrdsize > buf_size_hint * 2) {
/*		if(m_pausingBit) {
			CHECK_DAQMX_RET(DAQmxSetBufOutputOnbrdBufSize(m_taskDO, buf_size_hint * 2));
			CHECK_DAQMX_RET(DAQmxGetBufOutputOnbrdBufSize(m_taskDO, &onbrdsize));
			fprintf(stderr, "On-board bufsize is modified to %d\n", (int)onbrdsize);
		}
*/		buf_size_hint = onbrdsize / 2;
	}
	if(m_pausingBit)
		buf_size_hint /= 4;
	CHECK_DAQMX_RET(DAQmxCfgOutputBuffer(m_taskDO, buf_size_hint));
	CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskDO, &bufsize));
	fprintf(stderr, "Using bufsize = %d, freq = %f\n", (int)bufsize, freq);
	m_bufSizeHintDO = bufsize / 8;
	if(m_pausingBit)
		m_bufSizeHintDO = std::min(m_bufSizeHintDO, 16384u);
	m_transferSizeHintDO = std::min((unsigned int)onbrdsize / 4, m_bufSizeHintDO / 4);
	CHECK_DAQMX_RET(DAQmxSetWriteRegenMode(m_taskDO, DAQmx_Val_DoNotAllowRegen));
	
	{
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
		m_pausingGateTerm = formatString("/%s/PFI4", intfCtr()->devName());
		m_pausingCh = formatString("%s/ctr1", intfCtr()->devName());
		m_pausingSrcTerm = formatString("/%s/Ctr1InternalOutput", intfCtr()->devName());
		//set idle state to high level for synchronization.
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
		CHECK_DAQMX_RET(DAQmxSetDigEdgeStartTrigDigSyncEnable(m_taskGateCtr, true));
		if(!use_ao_clock) {
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
	CHECK_DAQMX_RET(DAQmxRegisterDoneEvent(m_taskAO, 0, &XNIDAQmxPulser::_onTaskDone, this));
		
	float64 freq = 1e3 / resolutionQAM();
	unsigned int buf_size_hint = (unsigned int)lrint(2.0 * freq);
	
	CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskAO, "",
										  freq, DAQmx_Val_Rising, DAQmx_Val_ContSamps, buf_size_hint));
    intfAO()->synchronizeClock(m_taskAO);
    
    int oversamp = lrint(resolution() / resolutionQAM());
	setupTasksDO(oversamp == 1);
    if(oversamp != 1) {
		//Synchronize ARM.
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
		CHECK_DAQMX_RET(DAQmxSetSampClkTimebaseActiveEdge(m_taskAO, DAQmx_Val_Falling));
		CHECK_DAQMX_RET(DAQmxSetPauseTrigType(m_taskAO, DAQmx_Val_DigLvl));
		CHECK_DAQMX_RET(DAQmxSetDigLvlPauseTrigSrc(m_taskAO, m_pausingSrcTerm.c_str()));
		CHECK_DAQMX_RET(DAQmxSetDigLvlPauseTrigWhen(m_taskAO, DAQmx_Val_High));
//		CHECK_DAQMX_RET(DAQmxSetDigLvlPauseTrigDigSyncEnable(m_taskAO, true));
	}

	m_softwareTrigger->setArmTerm(
		formatString("/%s/ao/SampleClock", intfAO()->devName()).c_str());

	//Buffer setup.
	uInt32 onbrdsize, bufsize;
	CHECK_DAQMX_RET(DAQmxGetBufOutputOnbrdBufSize(m_taskAO, &onbrdsize));
	fprintf(stderr, "On-board bufsize = %d\n", (int)onbrdsize);
	if(onbrdsize > buf_size_hint * 2) {
/*		if(m_pausingBit) {
			CHECK_DAQMX_RET(DAQmxSetBufOutputOnbrdBufSize(m_taskAO, buf_size_hint * 2));
			CHECK_DAQMX_RET(DAQmxGetBufOutputOnbrdBufSize(m_taskAO, &onbrdsize));
			fprintf(stderr, "On-board bufsize is modified to %d\n", (int)onbrdsize);
		}
*/		buf_size_hint = onbrdsize / 2;
	}
	if(m_pausingBit)
		buf_size_hint /= 4;
	CHECK_DAQMX_RET(DAQmxCfgOutputBuffer(m_taskAO, buf_size_hint));
	CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskAO, &bufsize));
	fprintf(stderr, "Using bufsize = %d\n", (int)bufsize);
	m_bufSizeHintAO = bufsize / 8;
	if(m_pausingBit)
		m_bufSizeHintAO = std::min(m_bufSizeHintAO, 16384u);
	
	m_transferSizeHintAO = std::min((unsigned int)onbrdsize / 4, m_bufSizeHintAO / 4);
	CHECK_DAQMX_RET(DAQmxSetWriteRegenMode(m_taskAO, DAQmx_Val_DoNotAllowRegen));

	{
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
		CHECK_DAQMX_RET(DAQmxSetAOReglitchEnable(m_taskAO, ch, false));
	}

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
XNIDAQmxPulser::_onTaskDone(TaskHandle task, int32 status, void *data) {
	XNIDAQmxPulser *obj = reinterpret_cast<XNIDAQmxPulser*>(data);
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
		m_suspendDO = true;
		m_suspendAO = true;
	}
}
template <typename T>
inline T *
fastFill(T* p, T x, unsigned int cnt) {
	if(cnt > 100) {
		for(;(int)p % (sizeof(uint64_t) / sizeof(T)); cnt--)
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
XNIDAQmxPulser::startPulseGen() throw (XInterface::XInterfaceError &)
{
	XScopedLock<XRecursiveMutex> tlock(m_totalLock);
	{
		m_suspendDO = true;
		m_suspendAO = true;
		XScopedLock<XRecursiveMutex> lockAO(m_mutexAO);
		XScopedLock<XRecursiveMutex> lockDO(m_mutexDO);
		
		stopPulseGen();
		
		unsigned int pausingbitnext = selectedPorts(PORTSEL_PAUSING);
		m_aswBit = selectedPorts(PORTSEL_ASW);
	
		if(CLEAR_TASKS_EVERYTIME ||
		   (m_taskDO == TASK_UNDEF) ||
		   (m_pausingBit != pausingbitnext)) {
			m_pausingBit = pausingbitnext;
			clearTasks();
			if(haveQAMPorts())
				setupTasksAODO();
			else
				setupTasksDO(false);
		}
		{
			uInt32 bufsize;
			CHECK_DAQMX_RET(DAQmxGetBufOutputOnbrdBufSize(m_taskDO, &bufsize));
			if(!m_pausingBit & (bufsize < 2047uL))
				throw XInterface::XInterfaceError(
					i18n("Use the pausing feature for a cheap DAQmx board.") + "\n"
							   + i18n("Look at the port-selection table."), __FILE__, __LINE__);
		}
		if(m_taskAO != TASK_UNDEF) {
			uInt32 bufsize;
			CHECK_DAQMX_RET(DAQmxGetBufOutputOnbrdBufSize(m_taskAO, &bufsize));
			if(!m_pausingBit & (bufsize < 8192uL))
				throw XInterface::XInterfaceError(
					i18n("Use the pausing feature for a cheap DAQmx board.") + "\n"
							   + i18n("Look at the port-selection table."), __FILE__, __LINE__);
		}

		//swap generated pattern lists to new ones.
		m_genPatternList.reset();
		m_genPatternListNext.swap(m_genPatternList);
		for(unsigned int j = 0; j < PAT_QAM_MASK / PAT_QAM_PHASE; j++) {
			m_genPulseWaveAO[j].reset();
			m_genPulseWaveNextAO[j].swap(m_genPulseWaveAO[j]);
		}

		//prepare pattern generation.
		m_genLastPatItDO = m_genPatternList->begin();
		m_genRestSampsDO = m_genPatternList->front().tonext;
		m_genTotalCountDO = m_genPatternList->front().tonext;
		m_genBufDO.resize(m_bufSizeHintDO);
		if(m_taskAO != TASK_UNDEF) {
			m_genLastPatItAO = m_genPatternList->begin();
			m_genRestSampsAO = m_genPatternList->front().tonext;
			m_genAOIndex = 0;
			m_genBufAO.resize(m_bufSizeHintAO);
		}
		
		const unsigned int cnt_preample = 1000;
		m_genTotalCountDO += cnt_preample;
		//synchronize the software trigger.
		m_softwareTrigger->start(1e3 / resolution());

		//prefilling of the buffers.
		if(m_taskAO != TASK_UNDEF) {
			//write pre-ample.
			const unsigned int oversamp_ao = lrint(resolution() / resolutionQAM());
			CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskAO, DAQmx_Val_FirstSample));
			CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskAO, 0));
			const unsigned int cnt_preample_ao = cnt_preample * oversamp_ao - 0;
			fastFill(&m_genBufAO[0], m_genAOZeroLevel, cnt_preample_ao);
			int32 samps;
			CHECK_DAQMX_RET(DAQmxWriteBinaryI16(m_taskAO, cnt_preample_ao,
												false, 0.5, 
												DAQmx_Val_GroupByScanNumber,
												m_genBufAO[0].ch,
												&samps, NULL));
			CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskAO, DAQmx_Val_CurrWritePos));
			CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskAO, 0));
			uInt32 bufsize;
			CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskAO, &bufsize));
			for(unsigned int rest = bufsize; rest >= bufsize / 2;) {
				genBankAO();
				unsigned int size = m_genBufAO.size();
				for(unsigned int cnt = 0; cnt < size;) {
					samps = std::min(size - cnt, m_transferSizeHintAO);
					CHECK_DAQMX_RET(DAQmxWriteBinaryI16(m_taskAO, samps,
														false, 0.0,
														DAQmx_Val_GroupByScanNumber,
														m_genBufAO[cnt].ch,
														&samps, NULL));
					cnt += samps;
					rest -= samps;
				}
			}
			genBankAO();
		}
		//write pre-ample.
		fastFill(&m_genBufDO[0], (tRawDO)0, cnt_preample);
		
		CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskDO, DAQmx_Val_FirstSample));
		CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskDO, 0));
		int32 samps;
		CHECK_DAQMX_RET(DAQmxWriteDigitalU16(m_taskDO, cnt_preample,
											 false, 0.0, 
											 DAQmx_Val_GroupByScanNumber,
											 &m_genBufDO[0],
											 &samps, NULL));
		CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskDO, DAQmx_Val_CurrWritePos));
		CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskDO, 0));
		uInt32 bufsize;
		CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskDO, &bufsize));
		for(unsigned int rest = bufsize; rest >= bufsize / 2;) {
			genBankDO();
			unsigned int size = m_genBufDO.size();
			for(unsigned int cnt = 0; cnt < size;) {
				samps = std::min(size - cnt, m_transferSizeHintDO);
				CHECK_DAQMX_RET(DAQmxWriteDigitalU16(m_taskDO, samps,
													 false, 0.0, 
													 DAQmx_Val_GroupByScanNumber,
													 &m_genBufDO[cnt],
													 &samps, NULL));
				cnt += samps;
				rest -= samps;
			}
		}
		genBankDO();
		
		{
			char ch[256];
			CHECK_DAQMX_RET(DAQmxGetTaskChannels(m_taskDO, ch, sizeof(ch)));
			CHECK_DAQMX_RET(DAQmxSetDOTristate(m_taskDO, ch, false));
		}
		CHECK_DAQMX_RET(DAQmxTaskControl(m_taskDO, DAQmx_Val_Task_Commit));
		if(m_taskDOCtr != TASK_UNDEF)
			CHECK_DAQMX_RET(DAQmxTaskControl(m_taskDOCtr, DAQmx_Val_Task_Commit));
		if(m_taskGateCtr != TASK_UNDEF)
			CHECK_DAQMX_RET(DAQmxTaskControl(m_taskGateCtr, DAQmx_Val_Task_Commit));
		if(m_taskAO != TASK_UNDEF)
			CHECK_DAQMX_RET(DAQmxTaskControl(m_taskAO, DAQmx_Val_Task_Commit));
	}
	fprintf(stderr, "Prefilling done.\n");
	//slave must start before the master.
	CHECK_DAQMX_RET(DAQmxStartTask(m_taskDO));
	if(m_taskGateCtr != TASK_UNDEF)
		CHECK_DAQMX_RET(DAQmxStartTask(m_taskGateCtr));
	if(m_taskDOCtr != TASK_UNDEF)
		CHECK_DAQMX_RET(DAQmxStartTask(m_taskDOCtr));
	m_suspendDO = false;
	fprintf(stderr, "Starting AO....\n");
	if(m_taskAO != TASK_UNDEF)
		CHECK_DAQMX_RET(DAQmxStartTask(m_taskAO));	
	m_suspendAO = false;
	m_running = true;	
}
void
XNIDAQmxPulser::stopPulseGen()
{
	XScopedLock<XRecursiveMutex> tlock(m_totalLock);

	m_suspendAO = true;
	m_suspendDO = true;
	XScopedLock<XRecursiveMutex> lockAO(m_mutexAO);
	XScopedLock<XRecursiveMutex> lockDO(m_mutexDO);

	m_softwareTrigger->stop();

	if(m_running) {
		{
			char ch[256];
			CHECK_DAQMX_RET(DAQmxGetTaskChannels(m_taskDO, ch, sizeof(ch)));
			uInt32 num_lines;
			CHECK_DAQMX_RET(DAQmxGetDONumLines(m_taskDO, ch, &num_lines));
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
		if(m_taskGateCtr != TASK_UNDEF)
		    CHECK_DAQMX_RET(DAQmxStopTask(m_taskGateCtr));
		if(!CLEAR_TASKS_EVERYTIME) {
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
inline XNIDAQmxPulser::tRawAOSet
XNIDAQmxPulser::aoVoltToRaw(const std::complex<double> &volt)
{
	const double volts[] = {volt.real(), volt.imag()};
	tRawAOSet z;
	for(unsigned int ch = 0; ch < 2; ch++) {
		double x = 1.0;
		double y = 0.0;
		double *pco = m_coeffAO[ch];
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
			y += (*pco++) * x;
			x *= volts[ch];
		}
		z.ch[ch] = lrint(y);
	}
	return z;
}
inline bool
XNIDAQmxPulser::tryOutputSuspend(const atomic<bool> &flag, 
								 XRecursiveMutex &mutex, const atomic<bool> &terminated) {
	if(flag) {
		mutex.unlock();
		while(flag && !terminated) usleep(30000);
		mutex.lock();
		return true;
	}
	return false;
}
void *
XNIDAQmxPulser::executeWriteAO(const atomic<bool> &terminated)
{
	while(!terminated) {
		writeBufAO(terminated, m_suspendAO);
	}
	return NULL;
}
void *
XNIDAQmxPulser::executeWriteDO(const atomic<bool> &terminated)
{
	while(!terminated) {
		writeBufDO(terminated, m_suspendDO);
	}
	return NULL;
}

void
XNIDAQmxPulser::writeBufAO(const atomic<bool> &terminated, const atomic<bool> &suspended)
{
	XScopedLock<XRecursiveMutex> lock(m_mutexAO);

	if(tryOutputSuspend(suspended, m_mutexAO, terminated))
		return;

 	const double dma_ao_period = resolutionQAM();
	const unsigned int size = m_genBufAO.size();
	try {
		for(unsigned int cnt = 0; cnt < size;) {
			int32 samps;
			samps = std::min(size - cnt, m_transferSizeHintAO);
			while(!terminated) {
				if(tryOutputSuspend(suspended, m_mutexAO, terminated))
					return;
				uInt32 space;
				CHECK_DAQMX_RET(DAQmxGetWriteSpaceAvail(m_taskAO, &space));
				if(space >= (uInt32)samps)
					break;
				usleep(lrint(1e3 * samps * dma_ao_period));
			}
			if(terminated)
				break;
			int32 written;
			CHECK_DAQMX_RET(DAQmxWriteBinaryI16(m_taskAO, samps, false, 0.0, 
												DAQmx_Val_GroupByScanNumber,
												m_genBufAO[cnt].ch,
												&written, NULL));
			if(written != samps)
				fprintf(stderr, "%d != %d\n", (int)written, (int)samps);
			cnt += written;
		}
		if(terminated)
			return;
	 	genBankAO();
	}
	catch (XInterface::XInterfaceError &e) {
		e.print(getLabel());
		m_suspendDO = true;
		m_suspendAO = true;
		return;
	}
	return;
}
void
XNIDAQmxPulser::writeBufDO(const atomic<bool> &terminated, const atomic<bool> &suspended)
{
	XScopedLock<XRecursiveMutex> lock(m_mutexDO);

	if(tryOutputSuspend(suspended, m_mutexDO, terminated))
		return;

 	const double dma_do_period = resolution();
	const unsigned int size = m_genBufDO.size();
	try {
		for(unsigned int cnt = 0; cnt < size;) {
			int32 samps;
			samps = std::min(size - cnt, m_transferSizeHintDO);
			while(!terminated) {
				if(tryOutputSuspend(suspended, m_mutexDO, terminated))
					return;
				uInt32 space;
				CHECK_DAQMX_RET(DAQmxGetWriteSpaceAvail(m_taskDO, &space));
				if(space >= (uInt32)samps)
					break;
				usleep(lrint(1e3 * samps * dma_do_period));
			}
			if(terminated)
				break;
			int32 written;
			CHECK_DAQMX_RET(DAQmxWriteDigitalU16(m_taskDO, samps, false, 0.0, 
												 DAQmx_Val_GroupByScanNumber,
												 &m_genBufDO[cnt],
												 &written, NULL));
			if(written != samps)
				fprintf(stderr, "%d != %d\n", (int)written, (int)samps);
			cnt += written;
		}
		if(terminated)
			return;
	 	genBankDO();
	}
	catch (XInterface::XInterfaceError &e) {
		e.print(getLabel());
		m_suspendDO = true;
		m_suspendAO = true;
 		return; 	
	}
	return;
}
void
XNIDAQmxPulser::genBankDO()
{
	GenPatternIterator it = m_genLastPatItDO;
	uint32_t pat = it->pattern;
	uint64_t tonext = m_genRestSampsDO;
	uint64_t total = m_genTotalCountDO;
	const tRawDO pausingbit = m_pausingBit;
	const tRawDO aswbit = m_aswBit;
	const uint64_t pausing_cnt = m_pausingCount;
	const uint64_t pausing_cnt_blank_before = PAUSING_BLANK_BEFORE + PAUSING_BLANK_AFTER;
	const uint64_t pausing_cnt_blank_after = 1;
	const uint64_t pausing_period = pausing_cnt + pausing_cnt_blank_before + pausing_cnt_blank_after;
	
	shared_ptr<XNIDAQmxInterface::SoftwareTrigger> &vt = m_softwareTrigger;
	
	tRawDO *pDO = &m_genBufDO[0];
	const unsigned int size = m_bufSizeHintDO;
	for(unsigned int samps_rest = size; samps_rest;) {
		//pattern of digital lines.
		tRawDO patDO = PAT_DO_MASK & pat;
		//No QAM nor ASW.
		if(pausingbit && ((pat & PAT_QAM_PULSE_IDX_MASK) == 0) && !(pat & aswbit)) {
			//generate a pausing trigger.
			ASSERT(tonext > 0);
			unsigned int lps = (unsigned int)std::min(
				(uint64_t)(samps_rest / (pausing_cnt_blank_before + pausing_cnt_blank_after)),
				(tonext - 1) / pausing_period);
			patDO &= ~pausingbit;
			if(lps) {
				samps_rest -= lps * (pausing_cnt_blank_before + pausing_cnt_blank_after);
				tonext -= lps * pausing_period;
				tRawDO patDO_or_p = patDO | pausingbit;
				for(unsigned int lp = 0; lp < lps; lp++) {
					for(int i = 0; i < pausing_cnt_blank_before; i++)
						*pDO++ = patDO_or_p;
					for(int i = 0; i < pausing_cnt_blank_after; i++)
						*pDO++ = patDO;
				}
			}
			if(samps_rest < pausing_cnt_blank_before + pausing_cnt_blank_after)
				break; 
		}
		//number of samples to be written into buffer.
		unsigned int gen_cnt = std::min((uint64_t)samps_rest, tonext);
		//write digital pattern.
		pDO = fastFill(pDO, patDO, gen_cnt);
		
		tonext -= gen_cnt;
		samps_rest -= gen_cnt;
		ASSERT(samps_rest < size);
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
	m_genBufDO.resize((unsigned int)(pDO - &m_genBufDO[0]));
	ASSERT(pDO == &m_genBufDO[m_genBufDO.size()]);
	m_genRestSampsDO = tonext;
	m_genLastPatItDO = it;
	m_genTotalCountDO = total;
}
void
XNIDAQmxPulser::genBankAO()
{
	const unsigned int oversamp_ao = lrint(resolution() / resolutionQAM());

	GenPatternIterator it = m_genLastPatItAO;
	uint32_t pat = it->pattern;
	uint64_t tonext = m_genRestSampsAO;
	unsigned int aoidx = m_genAOIndex;
	const tRawDO pausingbit = m_pausingBit;
	const tRawDO aswbit = m_aswBit;
	uint64_t pausing_cnt = m_pausingCount;
	uint64_t pausing_cnt_blank_before = PAUSING_BLANK_BEFORE + PAUSING_BLANK_AFTER;
	uint64_t pausing_cnt_blank_after = 1;
	const uint64_t pausing_period = pausing_cnt + pausing_cnt_blank_before + pausing_cnt_blank_after;

	tRawAOSet *pAO = &m_genBufAO[0];
	const tRawAOSet raw_zero = m_genAOZeroLevel;
	const unsigned int size = m_bufSizeHintAO;
	for(unsigned int samps_rest = size; samps_rest >= oversamp_ao;) {
		unsigned int pidx = (pat & PAT_QAM_PULSE_IDX_MASK) / PAT_QAM_PULSE_IDX;

		//No QAM nor ASW.
		if(pausingbit && (pidx == 0) && !(pat & aswbit)) {
			//generate a pausing trigger.
			ASSERT(tonext > 0);
			unsigned int lps = (unsigned int)std::min(
				(uint64_t)(samps_rest / oversamp_ao / (pausing_cnt_blank_before + pausing_cnt_blank_after)),
				(tonext - 1) / pausing_period);
			if(lps) {
				tonext -= lps * pausing_period;
				lps *= oversamp_ao * (pausing_cnt_blank_before + pausing_cnt_blank_after);
				samps_rest -= lps;
				pAO = fastFill(pAO, raw_zero, lps);
			}
			if(samps_rest / oversamp_ao < pausing_cnt_blank_before + pausing_cnt_blank_after)
				break; 
		}
		//number of samples to be written into buffer.
		unsigned int gen_cnt = std::min((uint64_t)samps_rest, tonext * oversamp_ao);
		gen_cnt = (gen_cnt / oversamp_ao) * oversamp_ao;
		if(pidx == 0) {
			//write blank in analog lines.
			aoidx = 0;
			pAO = fastFill(pAO, raw_zero, gen_cnt);
		}
		else {
			unsigned int pnum = (pat & PAT_QAM_MASK) / PAT_QAM_PHASE - (PAT_QAM_PULSE_IDX/PAT_QAM_PHASE);
			ASSERT(pnum < PAT_QAM_PULSE_IDX_MASK / PAT_QAM_PULSE_IDX);
			if(!m_genPulseWaveAO[pnum].get() ||
			   (m_genPulseWaveAO[pnum]->size() < gen_cnt + aoidx))
				throw XInterface::XInterfaceError(i18n("Invalid pulse patterns."), __FILE__, __LINE__);
			tRawAOSet *pGenAO = &m_genPulseWaveAO[pnum]->at(aoidx);
/*			for(unsigned int cnt = 0; cnt < gen_cnt; cnt++)
 *pAO++ = *pGenAO++;*/
			memcpy(pAO, pGenAO, gen_cnt * sizeof(tRawAOSet));
			pAO += gen_cnt;
			aoidx += gen_cnt;
		}
		tonext -= gen_cnt / oversamp_ao;
		samps_rest -= gen_cnt;
		ASSERT(samps_rest < size);
		if(tonext == 0) {
			it++;
			if(it == m_genPatternList->end()) {
				it = m_genPatternList->begin();
//				printf("p.\n");
			}
			pat = it->pattern;
			tonext = it->tonext;
		}
	}
	m_genBufAO.resize((unsigned int)(pAO - &m_genBufAO[0]));
	ASSERT(pAO == &m_genBufAO[m_genBufAO.size()]);
	ASSERT(m_genBufAO.size() % oversamp_ao == 0);
	m_genRestSampsAO = tonext;
	m_genLastPatItAO = it;
	m_genAOIndex = aoidx;
}

void
XNIDAQmxPulser::createNativePatterns()
{
	m_genPatternListNext.reset(new std::vector<GenPattern>);
	uint32_t pat = m_relPatList.back().pattern;
	for(RelPatListIterator it = m_relPatList.begin(); it != m_relPatList.end(); it++)
	{
		uint64_t tonext = it->toappear;

		GenPattern genpat(pat, tonext);
		m_genPatternListNext->push_back(genpat);
		pat = it->pattern;
	}

	if(haveQAMPorts()) {
		const double offset[] = {*qamOffset1(), *qamOffset2()};
		const double level[] = {*qamLevel1(), *qamLevel2()};
		  			  	
		for(unsigned int ch = 0; ch < NUM_AO_CH; ch++) {
			//arrange range info.
			double x = 1.0;
			for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
				m_coeffAO[ch][i] = m_coeffAODev[ch][i] * x
					+ ((i == 0) ? offset[ch] : 0);
				x *= level[ch];
			}
		}
	    m_genAOZeroLevel = aoVoltToRaw(std::complex<double>(0.0));
		std::complex<double> c(pow(10.0, *masterLevel()/20.0), 0);
		for(unsigned int i = 0; i < PAT_QAM_PULSE_IDX_MASK/PAT_QAM_PULSE_IDX; i++) {
			for(unsigned int qpsk = 0; qpsk < 4; qpsk++) {
				const unsigned int pnum = i * (PAT_QAM_PULSE_IDX/PAT_QAM_PHASE) + qpsk;
				m_genPulseWaveNextAO[pnum].reset(new std::vector<tRawAOSet>);
				for(std::vector<std::complex<double> >::const_iterator it = 
						qamWaveForm(i).begin(); it != qamWaveForm(i).end(); it++) {
					std::complex<double> z(*it * c);
					m_genPulseWaveNextAO[pnum]->push_back(aoVoltToRaw(z));
				}
				c *= std::complex<double>(0,1);
			}
		}
  	}
}


void
XNIDAQmxPulser::changeOutput(bool output, unsigned int /*blankpattern*/)
{
	if(output)
	{
		if(!m_genPatternListNext || m_genPatternListNext->empty() )
			throw XInterface::XInterfaceError(i18n("Pulser Invalid pattern"), __FILE__, __LINE__);
		startPulseGen();
	}
	else
	{
		stopPulseGen();
	}
	return;
}

