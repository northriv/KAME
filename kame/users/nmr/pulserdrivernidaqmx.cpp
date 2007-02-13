#include "pulserdrivernidaqmx.h"

#ifdef HAVE_NI_DAQMX

static const bool USE_FINITE_AO = false;

#define TASK_UNDEF ((TaskHandle)-1)

#include "interface.h"
#include <klocale.h>

using std::max;
using std::min;

//[ms]
static const double DMA_DO_PERIOD = (5.0/(1e3));

static const unsigned int OVERSAMP_AO = 1;
static const unsigned int OVERSAMP_DO = 1;
//[ms]
static const double DMA_AO_PERIOD = ((DMA_DO_PERIOD * OVERSAMP_DO) / OVERSAMP_AO);

static const unsigned int BUF_SIZE_HINT = 65536*4;

double XNIDAQmxPulser::resolution() {
     return DMA_DO_PERIOD * OVERSAMP_DO;
}

static const unsigned int g3mask = 0x0010;
static const unsigned int g2mask = 0x0002;
static const unsigned int g1mask = (0x0001 | g3mask);
static const unsigned int trig1mask = 0x0004;
static const unsigned int trig2mask = 0x0008;
static const unsigned int aswmask =	0x0080;
static const unsigned int qswmask = 0x0040;
static const unsigned int allmask = 0xffff;
static const unsigned int pulse1mask = 0x0100;
static const unsigned int pulse2mask = 0x0200;
static const unsigned int combmask = 0x0800;
static const unsigned int combfmmask = 0x0400;
static const unsigned int qpskbit = 0x10000;
static const unsigned int qpskmask = (qpskbit*3);
static const unsigned int pulsebit = 0x40000;
static const unsigned int pulsemask = (pulsebit*7);
static const unsigned int PULSE_P1 = (1*pulsebit);
static const unsigned int PULSE_P2 = (2*pulsebit);
static const unsigned int PULSE_COMB = (3*pulsebit);
static const unsigned int PULSE_INDUCE_EMISSION = (4*pulsebit);

static const unsigned int ctrtrigbit = g2mask;
static const unsigned int ctrtrigline = lrint(log((double)g2mask)/log(2.0));

XNIDAQmxPulser::XNIDAQmxPulser(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    XNIDAQmxDriver<XPulser>(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_ao_interface(XNode::create<XNIDAQmxInterface>("Interface2", false,
            dynamic_pointer_cast<XDriver>(this->shared_from_this()))),
	 m_taskAO(TASK_UNDEF),
	 m_taskDO(TASK_UNDEF),
 	 m_taskDOCtr(TASK_UNDEF),
 	 m_taskAOCtr(TASK_UNDEF)	 
{
    interfaces->insert(m_ao_interface);
    m_lsnOnOpenAO = m_ao_interface->onOpen().connectWeak(false,
    	 this->shared_from_this(), &XNIDAQmxPulser::onOpenAO);
    m_lsnOnCloseAO = m_ao_interface->onClose().connectWeak(false, 
    	this->shared_from_this(), &XNIDAQmxPulser::onCloseAO);

}
XNIDAQmxPulser::~XNIDAQmxPulser()
{
	if(m_taskAO != TASK_UNDEF)
	    DAQmxClearTask(m_taskAO);
	if(m_taskDO != TASK_UNDEF)
	    DAQmxClearTask(m_taskDO);
	if(m_taskDOCtr != TASK_UNDEF)
	    DAQmxClearTask(m_taskDOCtr);
	if(m_taskAOCtr != TASK_UNDEF)
	    DAQmxClearTask(m_taskAOCtr);
}
void
XNIDAQmxPulser::open() throw (XInterface::XInterfaceError &)
{
 	openDO();

	this->start();	
}
void
XNIDAQmxPulser::openDO() throw (XInterface::XInterfaceError &)
{
 	XScopedLock<XInterface> lock(*intfDO());
	if(m_taskDO != TASK_UNDEF)
	    DAQmxClearTask(m_taskDO);
	if(m_taskDOCtr != TASK_UNDEF)
	    DAQmxClearTask(m_taskDOCtr);

//	std::string ctrdev = formatString("%s/freqout", intfDO()->devName()).c_str();
//	std::string ctrout = formatString("/%s/FrequencyOutput", intfDO()->devName()).c_str();
	std::string ctrdev = formatString("%s/ctr0", intfDO()->devName()).c_str();
	std::string ctrout = formatString("/%s/Ctr0InternalOutput", intfDO()->devName()).c_str();
	if(m_taskAO != TASK_UNDEF) {
		ctrdev = formatString("%s/ctr1", intfAO()->devName()).c_str();
		ctrout = formatString("/%s/Ctr1InternalOutput", intfAO()->devName()).c_str();
	}

	float64 freq = 1e3 / DMA_DO_PERIOD;

	//Continuous pulse train generation. Duty = 50%.
    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskDOCtr));
	CHECK_DAQMX_RET(DAQmxCreateCOPulseChanFreq(m_taskDOCtr, 
    	ctrdev.c_str(), "", DAQmx_Val_Hz, DAQmx_Val_Low, 0.0,
    	freq, 0.5));
    //config. of timing is needed for some reasons.
	CHECK_DAQMX_RET(DAQmxCfgImplicitTiming(m_taskDOCtr, DAQmx_Val_ContSamps, 1000));
   
	CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskDO));

    CHECK_DAQMX_RET(DAQmxCreateDOChan(m_taskDO, 
    	formatString("%s/port0/line0:7", intfDO()->devName()).c_str(),
    	 "", DAQmx_Val_ChanForAllLines));

	//M series needs an external sample clock and trigger for DO channels.
	CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskDO,
		ctrout.c_str(),
		freq, DAQmx_Val_Rising, DAQmx_Val_ContSamps, BUF_SIZE_HINT * OVERSAMP_DO));
	
	//Buffer setup.
/*	CHECK_DAQMX_RET(DAQmxSetDODataXferReqCond(m_taskDO, 
    	formatString("%s/port0", intfDO()->devName()).c_str(),
		DAQmx_Val_OnBrdMemHalfFullOrLess));
*/
	CHECK_DAQMX_RET(DAQmxCfgOutputBuffer(m_taskDO, BUF_SIZE_HINT * OVERSAMP_DO));
	uInt32 bufsize;
	CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskDO, &bufsize));
	printf("Using bufsize = %d, freq = %f\n", (int)bufsize, freq);
	if(bufsize < BUF_SIZE_HINT * OVERSAMP_DO)
		throw XInterface::XInterfaceError(KAME::i18n("Insufficient size of NIDAQmx buffer."), __FILE__, __LINE__);
	CHECK_DAQMX_RET(DAQmxSetWriteRegenMode(m_taskDO, DAQmx_Val_DoNotAllowRegen));
}
void
XNIDAQmxPulser::onOpenAO(const shared_ptr<XInterface> &)
{
 	XScopedLock<XInterface> lockAO(*intfAO());
 	XScopedLock<XInterface> lockDO(*intfDO());
	try {		
		stopPulseGen();
		
		if(m_taskAO != TASK_UNDEF)
		    DAQmxClearTask(m_taskAO);
		if(m_taskAOCtr != TASK_UNDEF)
		    DAQmxClearTask(m_taskAOCtr);
		
	    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskAO));
	
		CHECK_DAQMX_RET(DAQmxCreateAOVoltageChan(m_taskAO,
	    	formatString("%s/ao0:1", intfAO()->devName()).c_str(), "",
	    	-1.0, 1.0, DAQmx_Val_Volts, NULL));
			
		//DMA is slower than interrupts!
		CHECK_DAQMX_RET(DAQmxSetAODataXferMech(m_taskAO, 
	    	formatString("%s/ao0:1", intfAO()->devName()).c_str(),
			DAQmx_Val_Interrupts));

		openDO();
		
		float64 freq = 1e3 / DMA_AO_PERIOD;
		
		if(USE_FINITE_AO) {
			std::string ctrdev = formatString("%s/ctr0", intfDO()->devName()).c_str();
			std::string ctrout = formatString("/%s/Ctr0InternalOutput", intfDO()->devName()).c_str();

		    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskAOCtr));
			CHECK_DAQMX_RET(DAQmxCreateCOPulseChanFreq(m_taskAOCtr, 
		    	ctrdev.c_str(), "", DAQmx_Val_Hz, DAQmx_Val_Low, 0.0,
		    	freq, 0.5));
			CHECK_DAQMX_RET(DAQmxCfgImplicitTiming(m_taskAOCtr, DAQmx_Val_FiniteSamps, 1));

		    CHECK_DAQMX_RET(DAQmxCfgDigEdgeStartTrig(m_taskAOCtr,
				formatString("/%s/PFI0", intfAO()->devName()).c_str(),
		    	DAQmx_Val_Rising));

			CHECK_DAQMX_RET(DAQmxSetStartTrigRetriggerable(m_taskAOCtr, true));

			CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskAO,
				ctrout.c_str(),
				1e3 / DMA_AO_PERIOD, DAQmx_Val_Rising, DAQmx_Val_ContSamps,
				BUF_SIZE_HINT * OVERSAMP_AO));
		}
		else {
			CHECK_DAQMX_RET(DAQmxCfgDigEdgeStartTrig(m_taskDOCtr,
				formatString("/%s/ao/StartTrigger", intfAO()->devName()).c_str(),
				DAQmx_Val_Rising));

			CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskAO, "",
				1e3 / DMA_AO_PERIOD, DAQmx_Val_Rising, DAQmx_Val_ContSamps,
				BUF_SIZE_HINT * OVERSAMP_AO));
		}
			

		//Buffer setup.
		CHECK_DAQMX_RET(DAQmxCfgOutputBuffer(m_taskAO, BUF_SIZE_HINT * OVERSAMP_AO));
		uInt32 bufsize;
		CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskAO, &bufsize));
		printf("Using bufsize = %d\n", (int)bufsize);
		if(bufsize < BUF_SIZE_HINT * OVERSAMP_AO)
			throw XInterface::XInterfaceError(
				KAME::i18n("Insufficient size of NIDAQmx buffer."), __FILE__, __LINE__);
		CHECK_DAQMX_RET(DAQmxGetBufOutputOnbrdBufSize(m_taskAO, &bufsize));
		printf("On-board bufsize = %d\n", bufsize);
		CHECK_DAQMX_RET(DAQmxSetWriteRegenMode(m_taskAO, DAQmx_Val_DoNotAllowRegen));
	
		for(unsigned int ch = 0; ch < NUM_AO_CH; ch++) {
		//obtain range info.
			for(unsigned int i = 0; i < CAL_POLY_ORDER; i++)
				m_coeffAO[ch][i] = 0.0;
			CHECK_DAQMX_RET(DAQmxGetAODevScalingCoeff(m_taskAO, 
				formatString("%s/ao%d", intfAO()->devName(), ch).c_str(),
				m_coeffAO[ch], CAL_POLY_ORDER));
			CHECK_DAQMX_RET(DAQmxGetAODACRngHigh(m_taskAO,
				formatString("%s/ao%d", intfAO()->devName(), ch).c_str(),
				&m_upperLimAO[ch]));
			CHECK_DAQMX_RET(DAQmxGetAODACRngLow(m_taskAO,
				formatString("%s/ao%d", intfAO()->devName(), ch).c_str(),
				&m_lowerLimAO[ch]));
		}

	}
	catch (XInterface::XInterfaceError &e) {
		e.print(getLabel());
	    close();
	}
}
void
XNIDAQmxPulser::onCloseAO(const shared_ptr<XInterface> &)
{
	stop();
}

void
XNIDAQmxPulser::close() throw (XInterface::XInterfaceError &)
{
	stopPulseGen();
 	XScopedLock<XInterface> lockao(*intfAO());
 	XScopedLock<XInterface> lockdo(*intfDO());
	if(m_taskAO != TASK_UNDEF)
	    DAQmxClearTask(m_taskAO);
	if(m_taskAOCtr != TASK_UNDEF)
	    DAQmxClearTask(m_taskAOCtr);
	if(m_taskDO != TASK_UNDEF)
	    DAQmxClearTask(m_taskDO);
	if(m_taskDOCtr != TASK_UNDEF)
	    DAQmxClearTask(m_taskDOCtr);
	m_taskAO = TASK_UNDEF;
	m_taskDO = TASK_UNDEF;
	m_taskAOCtr = TASK_UNDEF;
	m_taskDOCtr = TASK_UNDEF;
    
	intfDO()->stop();
	intfAO()->stop();
}

unsigned int
XNIDAQmxPulser::finiteAOSamps(unsigned int finiteaosamps)
{
	unsigned int finiteaocnt = 0;
	unsigned int finiteaorest = 0;
	for(GenPatternIterator it = m_genPatternList.begin(); it != m_genPatternList.end(); it++) {
		unsigned int pat = it->pattern;
		unsigned int gen_cnt = it->tonext;
		tRawDO patDO = allmask & pat;
		unsigned int pidx = (pat & pulsemask) / pulsebit;
		if(patDO & ctrtrigbit) {
			finiteaocnt += finiteaorest;
			if(!finiteaocnt) {
				finiteaocnt = 0;
			}
			finiteaocnt += gen_cnt;
			finiteaorest = 0;
		}
		else {
			if(finiteaocnt) {
				finiteaorest += gen_cnt;
				if(finiteaorest > 10) {
					finiteaosamps = std::max(finiteaosamps, finiteaocnt);
					if(finiteaocnt + finiteaorest > finiteaosamps) {
						finiteaocnt = 0;
						finiteaorest = 0;
					}
				}
			}
		}
	}
	return finiteaosamps;
}
void
XNIDAQmxPulser::startPulseGen() throw (XInterface::XInterfaceError &)
{
	 stopPulseGen();
 	{
 	XScopedLock<XInterface> lockao(*intfAO());
 	XScopedLock<XInterface> lockdo(*intfDO());
		   
	//	std::deque<GenPattern> m_genPatternList;
		m_genLastPatItAODO = m_genPatternList.begin();
		m_genRestSampsAODO = m_genPatternList.back().tonext;
		m_genAOIndex = 0;
		m_genBankWrittenLast = NUM_BUF_BANK - 1;
		m_genBankDO = 0;
		m_genBankAO = 0;
	
		for(unsigned int bank = 0; bank < NUM_BUF_BANK; bank++) {
			m_genBufDO[bank].resize(BUF_SIZE_HINT * OVERSAMP_DO);
		}
		for(unsigned int bank = 0; bank < NUM_BUF_BANK; bank++) {
			m_genBufAO[bank].resize(BUF_SIZE_HINT * NUM_AO_CH * OVERSAMP_AO);
		}
	
		if(USE_FINITE_AO) {
			unsigned int oldv = 0;
			unsigned int newv = 2;
			while(oldv != newv) {
				oldv = newv;
				newv = finiteAOSamps(oldv);
			}
			m_genFiniteAOSamps = newv;
			printf("Using finite ao = %u.\n", newv);
			if(m_taskAOCtr != TASK_UNDEF)
				CHECK_DAQMX_RET(DAQmxSetSampQuantSampPerChan(m_taskAOCtr, newv));
		}
		
	const void *FIRST_OF_MLOCK_MEMBER = &m_genPatternList;
	const void *LAST_OF_MLOCK_MEMBER = &m_lowerLimAO[NUM_AO_CH];
		//Suppress swapping.
		mlock(FIRST_OF_MLOCK_MEMBER, (size_t)LAST_OF_MLOCK_MEMBER - (size_t)FIRST_OF_MLOCK_MEMBER);
		for(unsigned int bank = 0; bank < NUM_BUF_BANK; bank++) {
			mlock(&m_genBufDO[bank][0], m_genBufDO[bank].size() * sizeof(tRawDO));
			mlock(&m_genBufAO[bank][0], m_genBufAO[bank].size() * sizeof(tRawAO));
		}
		for(unsigned int ch = 0; ch < NUM_AO_CH; ch++) {
			for(unsigned int i = 0; i < 32; i++)
				mlock(&m_genPulseWaveAO[ch][i], m_genPulseWaveAO[ch][i].size() * sizeof(tRawAO));
		}
		
		//prefilling of banks.
		for(unsigned int bank = 0; bank < 2; bank++) {
			genBankAODO();
		}
		
		CHECK_DAQMX_RET(DAQmxTaskControl(m_taskDOCtr, DAQmx_Val_Task_Commit));
		CHECK_DAQMX_RET(DAQmxTaskControl(m_taskDO, DAQmx_Val_Task_Commit));
	    CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskDO, DAQmx_Val_FirstSample));
	    CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskDO, 0));
		if(m_taskAOCtr != TASK_UNDEF)
			CHECK_DAQMX_RET(DAQmxTaskControl(m_taskAOCtr, DAQmx_Val_Task_Commit));
		if(m_taskAO != TASK_UNDEF) {
			CHECK_DAQMX_RET(DAQmxTaskControl(m_taskAO, DAQmx_Val_Task_Commit));
		    CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskAO, DAQmx_Val_FirstSample));
		    CHECK_DAQMX_RET(DAQmxSetWriteOffset(m_taskAO, 0));
		}
 	}
 	
	m_threadWriteDO.reset(new XThread<XNIDAQmxPulser>(shared_from_this(),
		 &XNIDAQmxPulser::executeWriteDO));
	m_threadWriteDO->resume();
	while(m_genBankDO < 1) {
		msecsleep(1);
	}
	if(m_taskAO != TASK_UNDEF) {
		m_threadWriteAO.reset(new XThread<XNIDAQmxPulser>(shared_from_this(),
			 &XNIDAQmxPulser::executeWriteAO));
		m_threadWriteAO->resume();
		while(m_genBankAO < 1) {
			msecsleep(1);
		}
	}

 	{
		//slave must start before the master.
		if(USE_FINITE_AO && (m_taskAOCtr != TASK_UNDEF)) {
			if(m_taskAO != TASK_UNDEF) {
			    CHECK_DAQMX_RET(DAQmxStartTask(m_taskAO));
			}
			if(m_taskAOCtr != TASK_UNDEF) {
			    CHECK_DAQMX_RET(DAQmxStartTask(m_taskAOCtr));
			}
		    CHECK_DAQMX_RET(DAQmxStartTask(m_taskDOCtr));
		    CHECK_DAQMX_RET(DAQmxStartTask(m_taskDO));
		}
		else {
		    CHECK_DAQMX_RET(DAQmxStartTask(m_taskDOCtr));
		    CHECK_DAQMX_RET(DAQmxStartTask(m_taskDO));
			if(m_taskAOCtr != TASK_UNDEF) {
			    CHECK_DAQMX_RET(DAQmxStartTask(m_taskAOCtr));
			}
			if(m_taskAO != TASK_UNDEF) {
			    CHECK_DAQMX_RET(DAQmxStartTask(m_taskAO));
			}
		}
 	}
}
void
XNIDAQmxPulser::stopPulseGen()
{
	if(m_threadWriteAO) {
		m_threadWriteAO->terminate();
	}
	if(m_threadWriteDO) {
		m_threadWriteDO->terminate();
	}
	{
 	XScopedLock<XInterface> lockao(*intfAO());
 	XScopedLock<XInterface> lockdo(*intfDO());
		if(m_taskAOCtr != TASK_UNDEF)
		    DAQmxStopTask(m_taskAOCtr);
		if(m_taskDOCtr != TASK_UNDEF)
		    DAQmxStopTask(m_taskDOCtr);
	}
	{
 	XScopedLock<XInterface> lockao(*intfAO());
 	XScopedLock<XInterface> lockdo(*intfDO());
		if(m_taskDO != TASK_UNDEF)
		    DAQmxStopTask(m_taskDO);
		if(m_taskAO != TASK_UNDEF)
		    DAQmxStopTask(m_taskAO);
	}
}
inline XNIDAQmxPulser::tRawAO
XNIDAQmxPulser::aoVoltToRaw(int ch, float64 volt)
{
//	volt = std::max(volt, m_lowerLimAO[ch]);
//	volt = std::min(volt, m_upperLimAO[ch]);
	float64 x = 1.0;
	float64 y = 0.0;
	float64 *pco = m_coeffAO[ch];
	for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
		y += (*pco++) * x;
		x *= volt;
	}
	return lrint(y);
}
void *
XNIDAQmxPulser::executeWriteAO(const atomic<bool> &terminated)
{
	while(!terminated) {
		writeBankAO(terminated);
	}
	return NULL;
}
void *
XNIDAQmxPulser::executeWriteDO(const atomic<bool> &terminated)
{
	while(!terminated) {
		writeBankDO(terminated);
	}
	return NULL;
}

void
XNIDAQmxPulser::writeBankAO(const atomic<bool> &terminated)
{
 	XScopedLock<XInterface> lockao(*intfAO());
	unsigned int bank = m_genBankAO;
	unsigned int size = m_genBufAO[bank].size() / NUM_AO_CH;
	try {
		const unsigned int num_samps = 256;
		for(unsigned int cnt = 0; cnt < size;) {
			int32 samps;
			samps = std::min(size - cnt, num_samps);
			while(!terminated) {
				if(bank == m_genBankWrittenLast)
					throw XInterface::XInterfaceError(KAME::i18n("AO buffer underrun."), __FILE__, __LINE__);
			uInt32 space;
				int ret = DAQmxGetWriteSpaceAvail(m_taskAO, &space);
				if(!ret && (space >= samps))
					break;
				usleep(lrint(1e3 * samps * DMA_AO_PERIOD));
			}
			if(terminated)
				break;
			CHECK_DAQMX_RET(DAQmxWriteBinaryI16(m_taskAO, samps, false, DAQmx_Val_WaitInfinitely, 
				DAQmx_Val_GroupByScanNumber,
				&m_genBufAO[bank][cnt * NUM_AO_CH],
				&samps, NULL));
		   CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskAO, DAQmx_Val_CurrWritePos));
			cnt += samps;
		}
	}
	catch (XInterface::XInterfaceError &e) {
		e.print(getLabel());
		stopPulseGen();
		return;
	}
	bank++;
	if(bank == NUM_BUF_BANK)
		bank = 0;
	m_genBankAO = bank;
	return;
}
void
XNIDAQmxPulser::writeBankDO(const atomic<bool> &terminated)
{
 	XScopedLock<XInterface> lockdo(*intfDO());
	unsigned int bank = m_genBankDO;
	unsigned int size = m_genBufDO[bank].size();
	try {
		const unsigned int num_samps = 256;
		for(unsigned int cnt = 0; cnt < size;) {
			int32 samps;
			samps = std::min(size - cnt, num_samps);
			while(!terminated) {
				if(bank == m_genBankWrittenLast)
					throw XInterface::XInterfaceError(KAME::i18n("DO buffer underrun."), __FILE__, __LINE__);
			uInt32 space;
				int ret = DAQmxGetWriteSpaceAvail(m_taskDO, &space);
				if(!ret && (space >= samps))
					break;
				usleep(lrint(1e3 * samps * DMA_DO_PERIOD));
			}
			if(terminated)
				break;
			CHECK_DAQMX_RET(DAQmxWriteDigitalU16(m_taskDO, samps, false, DAQmx_Val_WaitInfinitely, 
				DAQmx_Val_GroupByScanNumber,
				&m_genBufDO[bank][cnt],
				&samps, NULL));
		   CHECK_DAQMX_RET(DAQmxSetWriteRelativeTo(m_taskDO, DAQmx_Val_CurrWritePos));
			cnt += samps;
		}
	}
	catch (XInterface::XInterfaceError &e) {
		e.print(getLabel());
		stopPulseGen();
 		return; 	
	}
	bank++;
	if(bank == NUM_BUF_BANK)
		bank = 0;
	m_genBankDO = bank;
 	genBankAODO();
	return;
}
void
XNIDAQmxPulser::genBankAODO()
{
	GenPatternIterator it = m_genLastPatItAODO;
	uint32_t pat = it->pattern;
	long long int tonext = m_genRestSampsAODO;
	unsigned int aoidx = m_genAOIndex;
	unsigned int bank = m_genBankWrittenLast;
	bank++;
	if(bank == NUM_BUF_BANK)
		bank = 0;
	unsigned int finiteaorest = m_genFiniteAORestSamps;
	unsigned int finiteaosamps = m_genFiniteAOSamps;
	
	C_ASSERT(sizeof(long long int) > sizeof(int32_t));
	
	tRawDO *pDO = &m_genBufDO[bank][0];
	tRawAO *pAO = &m_genBufAO[bank][0];
	tRawAO raw_ao0_zero = aoVoltToRaw(0, 0.0);
	tRawAO raw_ao1_zero = aoVoltToRaw(1, 0.0);
	for(unsigned int samps_rest = BUF_SIZE_HINT; samps_rest;) {
		tRawDO patDO = allmask & pat;
		unsigned int pidx = (pat & pulsemask) / pulsebit;
		unsigned int gen_cnt = std::min((long long int)samps_rest, tonext);
		
		for(unsigned int cnt = 0; cnt < gen_cnt * OVERSAMP_DO; cnt++) {
			*pDO++ = patDO;
		}
		if(USE_FINITE_AO)
		{
			if((patDO & ctrtrigbit) && (finiteaorest == 0)) {
				finiteaorest = finiteaosamps;
			}
		}
		if(pidx == 0) {
			aoidx = 0;
			unsigned int zerocnt = gen_cnt;
			if(USE_FINITE_AO) {
				zerocnt = std::min(finiteaorest, gen_cnt);
				finiteaorest -= zerocnt;
			}
			for(unsigned int cnt = 0; cnt < zerocnt * OVERSAMP_AO; cnt++) {
				*pAO++ = raw_ao0_zero;
//				*pAO++ = raw_ao1_zero;
				*pAO++ = (cnt %2 == 0) ? raw_ao1_zero : -1000uL;
			}
		}
		else {
			if(USE_FINITE_AO) {
				finiteaorest -= gen_cnt;
			}
			unsigned int qpskidx = (pat & qpskmask) / qpskbit;
			unsigned int pnum = (pidx - 1) * (pulsebit/qpskbit) + qpskidx;
			tRawAO *pGenAO0 = &m_genPulseWaveAO[0][pnum][aoidx];
			tRawAO *pGenAO1 = &m_genPulseWaveAO[1][pnum][aoidx];
			ASSERT(m_genPulseWaveAO[0][pnum].size());
			ASSERT(m_genPulseWaveAO[1][pnum].size());
			if(m_genPulseWaveAO[0][pnum].size() <= aoidx + gen_cnt * OVERSAMP_AO) {
				dbgPrint("Oops. This should not happen.");
				int lps = m_genPulseWaveAO[0][pnum].size() - aoidx;
				lps = std::max(0, lps);
				for(int cnt = 0; cnt < lps; cnt++) {
					*pAO++ = *pGenAO0++;
					*pAO++ = *pGenAO1++;
				}
				for(int cnt = 0; cnt < gen_cnt * OVERSAMP_AO - lps; cnt++) {
					*pAO++ = raw_ao0_zero;
					*pAO++ = raw_ao1_zero;
				}
			}
			else {
				for(unsigned int cnt = 0; cnt < gen_cnt * OVERSAMP_AO; cnt++) {
					*pAO++ = *pGenAO0++;
					*pAO++ = *pGenAO1++;
				}
				aoidx += gen_cnt * OVERSAMP_AO;
			}
		}
		tonext -= gen_cnt;
		samps_rest -= gen_cnt;
		if(tonext == 0) {
			it++;
			if(it == m_genPatternList.end()) {
				it = m_genPatternList.begin();
				printf("p.\n");
			}
			pat = it->pattern;
			tonext = it->tonext;
		}
	}
	ASSERT(pDO == &m_genBufDO[bank][m_genBufDO[bank].size()]);
	if(USE_FINITE_AO)
		m_genBufAO[bank].resize((int)(pAO - &m_genBufAO[bank][0]));
	else
		ASSERT(pAO == &m_genBufAO[bank][m_genBufAO[bank].size()]);
	m_genRestSampsAODO = tonext;
	m_genLastPatItAODO = it;
	m_genAOIndex = aoidx;
	m_genBankWrittenLast = bank;
	m_genFiniteAORestSamps = finiteaorest;
}

void
XNIDAQmxPulser::createNativePatterns()
{
  double _master = *masterLevel();
  double _tau = m_tauRecorded;
  double _pw1 = m_pw1Recorded;
  double _pw2 = m_pw2Recorded;
  double _comb_pw = m_combPWRecorded;
  double _dif_freq = m_difFreqRecorded;

  bool _induce_emission = *induceEmission();
  double _induce_emission_pw = _comb_pw;
  double _induce_emission_phase = *induceEmissionPhase() / 180.0 * PI;
      
  m_genPatternList.clear();
  uint32_t lastpat = m_relPatList.back().pattern;
  for(RelPatListIterator it = m_relPatList.begin(); it != m_relPatList.end(); it++)
  {
  long long int tonext = llrint(it->toappear / DMA_DO_PERIOD);
	 	GenPattern pat(lastpat, tonext);
	 	lastpat = it->pattern;
  		m_genPatternList.push_back(pat);
  }
    
  makeWaveForm(PULSE_P1/pulsebit - 1, _pw1/1000.0, pulseFunc(p1Func()->to_str() ),
  	 _master + *p1Level()
    , _dif_freq * 1000.0, -2 * PI * _dif_freq * 2 * _tau);
  makeWaveForm(PULSE_P2/pulsebit - 1, _pw2/1000.0, pulseFunc(p2Func()->to_str() ),
  	_master + *p2Level()
    , _dif_freq * 1000.0, -2 * PI * _dif_freq * 2 * _tau);
  makeWaveForm(PULSE_COMB/pulsebit - 1, _comb_pw/1000.0, pulseFunc(combFunc()->to_str() ),
         _master + *combLevel(), *combOffRes() + _dif_freq *1000.0);
  if(_induce_emission) {
      makeWaveForm(PULSE_INDUCE_EMISSION/pulsebit - 1, _induce_emission_pw/1000.0, pulseFunc(combFunc()->to_str() ),
         _master + *combLevel(), *combOffRes() + _dif_freq *1000.0, _induce_emission_phase);
  }
}

int
XNIDAQmxPulser::makeWaveForm(int num, double pw, tpulsefunc func, double dB, double freq, double phase)
{
	for(unsigned int qpsk = 0; qpsk < 4; qpsk++) {
		unsigned int pnum = num * (pulsebit/qpskbit) + qpsk;
		m_genPulseWaveAO[0][pnum].clear();
		m_genPulseWaveAO[1][pnum].clear();
		ASSERT(pnum < 32);
	  	unsigned short word = (unsigned short)lrint(pw / DMA_AO_PERIOD) + 2;
		double dx = DMA_AO_PERIOD / pw;
		double dp = 2*PI*freq*DMA_AO_PERIOD;
		double z = pow(10.0, dB/20.0);
		for(int i = 0; i < word; i++) {
			double w = z * func((i - word / 2.0) * dx) * 1.0;
			double x = w * cos((i - word / 2.0) * dp + PI/4 + phase);
			double y = w * sin((i - word / 2.0) * dp + PI/4 + phase);
			m_genPulseWaveAO[0][pnum].push_back(aoVoltToRaw(0, x));
			m_genPulseWaveAO[1][pnum].push_back(aoVoltToRaw(1, y));
		}
		phase += PI/2;
	}
	return 0;
}

void
XNIDAQmxPulser::changeOutput(bool output)
{
  if(output)
    {
      if(m_genPatternList.empty() )
              throw XInterface::XInterfaceError(KAME::i18n("Pulser Invalid pattern"), __FILE__, __LINE__);
	  startPulseGen();
    }
  else
    {
      stopPulseGen();
    }
  return;
}

#include <set>

void
XNIDAQmxPulser::rawToRelPat() throw (XRecordError&)
{
  double _rtime = m_rtimeRecorded;
  double _tau = m_tauRecorded;
  double _asw_setup = m_aswSetupRecorded;
  double _asw_hold = m_aswHoldRecorded;
  double _alt_sep = m_altSepRecorded;
  double _pw1 = m_pw1Recorded;
  double _pw2 = m_pw2Recorded;
  double _comb_pw = m_combPWRecorded;
  double _comb_pt = m_combPTRecorded;
  double _comb_p1 = m_combP1Recorded;
  double _comb_p1_alt = m_combP1AltRecorded;
  double _g2_setup = *g2Setup();
  int _echo_num = m_echoNumRecorded;
  int _comb_num = m_combNumRecorded;
  int _comb_mode = m_combModeRecorded;
  int _rt_mode = m_rtModeRecorded;
  int _num_phase_cycle = m_numPhaseCycleRecorded;
  if(_comb_mode == N_COMB_MODE_OFF) _num_phase_cycle = std::min(_num_phase_cycle, 4);
  
  bool comb_mode_alt = ((_comb_mode == N_COMB_MODE_P1_ALT) ||
            (_comb_mode == N_COMB_MODE_COMB_ALT));
  bool saturation_wo_comb = (_comb_num == 0);
  bool driven_equilibrium = *drivenEquilibrium();
  double _qsw_delay = *qswDelay();
  double _qsw_width = *qswWidth();
  bool _qsw_pi_only = *qswPiPulseOnly();
  int comb_rot_num = lrint(*combOffRes() * (_comb_pw / 1000.0) * 4);
  
  bool _induce_emission = *induceEmission();
  double _induce_emission_pw = _comb_pw;
  
  //unit of phase is pi/2
  #define qpsk(phase) ((phase % 4)*qpskbit)
  #define qpskinv(phase) (qpsk(((phase) + 2) % 4))

  //comb phases
  const uint32_t comb[MAX_NUM_PHASE_CYCLE] = {
    1, 3, 0, 2, 3, 1, 2, 0, 0, 2, 1, 3, 2, 0, 3, 1
  };

  //pi/2 pulse phases
  const uint32_t p1single[MAX_NUM_PHASE_CYCLE] = {
    0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2
  };
  //pi pulse phases
  const uint32_t p2single[MAX_NUM_PHASE_CYCLE] = {
    0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3
  };
  //pi/2 pulse phases for multiple echoes
  const uint32_t p1multi[MAX_NUM_PHASE_CYCLE] = {
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
  };
  //pi pulse phases for multiple echoes
  const uint32_t p2multi[MAX_NUM_PHASE_CYCLE] = {
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3
  };
  
  const uint32_t _qpsk_driven_equilibrium[4] = {2, 1, 0, 3};
  #define qpsk_driven_equilibrium(phase) qpsk(_qpsk_driven_equilibrium[(phase) % 4])
  #define qpsk_driven_equilibrium_inv(phase) (qpsk_driven_equilibrium(((phase) + 2) % 4))

  typedef std::multiset<tpat, std::less<tpat> > tpatset;
  tpatset patterns;  // patterns
  typedef std::multiset<tpat, std::less<tpat> >::iterator tpatset_it;

  m_relPatList.clear();
  
  double pos = 0;
            
  int echonum = _echo_num;
  const uint32_t *p1 = (echonum > 1) ? p1multi : p1single;
  const uint32_t *p2 = (echonum > 1) ? p2multi : p2single;
  
  //dice for alternative modes
  bool former_of_alt = ((double)KAME::rand() / (RAND_MAX - 1) > 0.5);
  
  for(int i = 0; i < _num_phase_cycle * (comb_mode_alt ? 2 : 1); i++)
    {
      int j = (i / (comb_mode_alt ? 2 : 1)) % _num_phase_cycle; //index for phase cycling
      former_of_alt = !former_of_alt;
      bool comb_off_res = ((_comb_mode != N_COMB_MODE_COMB_ALT) || former_of_alt) && (comb_rot_num != 0);
            
      double _p1 = 0;
      if((_comb_mode != N_COMB_MODE_OFF) &&
     !((_comb_mode == N_COMB_MODE_COMB_ALT) && former_of_alt && !(comb_rot_num != 0)))
    {
         _p1 = ((former_of_alt || (_comb_mode != N_COMB_MODE_P1_ALT)) ? _comb_p1 : _comb_p1_alt);
    }

      double rest;
      if(_rt_mode == N_RT_MODE_FIXREST)
            rest = _rtime;
      else
        rest = _rtime - _p1;
    
      if(saturation_wo_comb && (_p1 > 0)) rest = 0;
      
      if(rest > 0) pos += rest;
      
      //comb pulses
      if((_p1 > 0) && !saturation_wo_comb)
     {
     double combpt = max((double)_comb_pt, (double)_comb_pw)/1000.0;
     double cpos = pos - combpt*_comb_num;
     
      patterns.insert(tpat(cpos - _comb_pw/1000.0/2 - _g2_setup/1000.0,
                     g2mask, g2mask));
      patterns.insert(tpat(cpos - _comb_pw/1000.0/2 - _g2_setup/1000.0, comb_off_res ? ~(uint32_t)0 : 0, combfmmask));
      patterns.insert(tpat(cpos - _comb_pw/1000.0/2 - _g2_setup/1000.0, ~(uint32_t)0, combmask));
      for(int k = 0; k < _comb_num; k++) {
          patterns.insert(tpat(cpos + _comb_pw/2/1000.0 , qpsk(comb[j]), qpskmask));
          cpos += combpt;
          cpos -= _comb_pw/2/1000.0;
          patterns.insert(tpat(cpos, ~(uint32_t)0, g1mask));
          patterns.insert(tpat(cpos, PULSE_COMB, pulsemask));
          cpos += _comb_pw/1000.0;      
          patterns.insert(tpat(cpos, 0 , g1mask));
          patterns.insert(tpat(cpos, 0, pulsemask));

          cpos -= _comb_pw/2/1000.0;
      }
      patterns.insert(tpat(cpos + _comb_pw/2/1000.0, 0, g2mask));
      patterns.insert(tpat(cpos + _comb_pw/2/1000.0, 0, combmask));
      patterns.insert(tpat(cpos + _comb_pw/1000.0/2, ~(uint32_t)0, combfmmask));
      if(! _qsw_pi_only) {
          patterns.insert(tpat(cpos + _comb_pw/2/1000.0 + _qsw_delay/1000.0, ~(uint32_t)0 , qswmask));
          patterns.insert(tpat(cpos + _comb_pw/2/1000.0 + (_qsw_delay + _qsw_width)/1000.0, 0 , qswmask));
      }
    }   
       pos += _p1;
       
       //pi/2 pulse
      //on
      patterns.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, qpsk(p1[j]), qpskmask));
      patterns.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, ~(uint32_t)0, pulse1mask));
      patterns.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, ~(uint32_t)0, g2mask));
      patterns.insert(tpat(pos - _pw1/2.0/1000.0, PULSE_P1, pulsemask));
      patterns.insert(tpat(pos - _pw1/2.0/1000.0, ~(uint32_t)0, g1mask | trig2mask));
      //off
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, g1mask));
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, pulsemask));
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, pulse1mask));
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, qpsk(p2[j]), qpskmask));
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, ~(uint32_t)0, pulse2mask));
      if(! _qsw_pi_only) {
          patterns.insert(tpat(pos + _pw1/2.0/1000.0 + _qsw_delay/1000.0, ~(uint32_t)0 , qswmask));
          patterns.insert(tpat(pos + _pw1/2.0/1000.0 + (_qsw_delay + _qsw_width)/1000.0, 0 , qswmask));
      }
     
      //2tau
      pos += 2*_tau/1000.0;
      //    patterns.insert(tpat(pos - _asw_setup, -1, aswmask | rfswmask, 0));
      patterns.insert(tpat(pos - _asw_setup, ~(uint32_t)0, aswmask));
      patterns.insert(tpat(pos -
               ((!former_of_alt && comb_mode_alt) ?
                (double)_alt_sep : 0.0), ~(uint32_t)0, trig1mask));
                
      //induce emission
      if(_induce_emission) {
          patterns.insert(tpat(pos - _induce_emission_pw/2.0/1000.0, ~(uint32_t)0, g3mask));
          patterns.insert(tpat(pos - _induce_emission_pw/2.0/1000.0, PULSE_INDUCE_EMISSION, pulsemask));
          patterns.insert(tpat(pos - _induce_emission_pw/2.0/1000.0, 0, qpskmask));
          patterns.insert(tpat(pos + _induce_emission_pw/2.0/1000.0, 0, pulsemask));
          patterns.insert(tpat(pos + _induce_emission_pw/2.0/1000.0, 0, g3mask));
      }

      //pi pulses 
      pos -= 3*_tau/1000.0;
      for(int k = 0;k < echonum; k++)
      {
          pos += 2*_tau/1000.0;
          //on
      if(k >= 1) {
              patterns.insert(tpat(pos - _pw2/2.0/1000.0 - _g2_setup/1000.0, ~(uint32_t)0, g2mask));
      }
          patterns.insert(tpat(pos - _pw2/2.0/1000.0, 0, trig2mask));
          patterns.insert(tpat(pos - _pw2/2.0/1000.0, PULSE_P2, pulsemask));
          patterns.insert(tpat(pos - _pw2/2.0/1000.0, ~(uint32_t)0, g1mask));
          //off
          patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, pulse2mask));
          patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, pulsemask));
          patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, g1mask));
          patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, g2mask));
          patterns.insert(tpat(pos + _pw2/2.0/1000.0 + _qsw_delay/1000.0, ~(uint32_t)0 , qswmask));
          patterns.insert(tpat(pos + _pw2/2.0/1000.0 + (_qsw_delay + _qsw_width)/1000.0, 0 , qswmask));
      }

       patterns.insert(tpat(pos + _tau/1000.0 + _asw_hold, 0, aswmask | trig1mask));
      //induce emission
      if(_induce_emission) {
          patterns.insert(tpat(pos + _tau/1000.0 + _asw_hold - _induce_emission_pw/2.0/1000.0, ~(uint32_t)0, g3mask));
          patterns.insert(tpat(pos + _tau/1000.0 + _asw_hold - _induce_emission_pw/2.0/1000.0, PULSE_INDUCE_EMISSION, pulsemask));
          patterns.insert(tpat(pos + _tau/1000.0 + _asw_hold - _induce_emission_pw/2.0/1000.0, 0, qpskmask));
          patterns.insert(tpat(pos + _tau/1000.0 + _asw_hold + _induce_emission_pw/2.0/1000.0, 0, pulsemask));
          patterns.insert(tpat(pos + _tau/1000.0 + _asw_hold + _induce_emission_pw/2.0/1000.0, 0, g3mask));
      }

      if(driven_equilibrium)
      {
        pos += 2*_tau/1000.0;
        //pi pulse 
        //on
        patterns.insert(tpat(pos - _pw2/2.0/1000.0 - _g2_setup/1000.0, qpsk_driven_equilibrium(p2[j]), qpskmask));
        patterns.insert(tpat(pos - _pw2/2.0/1000.0 - _g2_setup/1000.0, ~(uint32_t)0, g2mask));
        patterns.insert(tpat(pos - _pw2/2.0/1000.0 - _g2_setup/1000.0, ~(uint32_t)0, pulse2mask));
        patterns.insert(tpat(pos - _pw2/2.0/1000.0, PULSE_P2, pulsemask));
        patterns.insert(tpat(pos - _pw2/2.0/1000.0, ~(uint32_t)0, g1mask));
        //off
        patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, pulse2mask));
        patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, pulsemask));
        patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, g1mask | g2mask));
        patterns.insert(tpat(pos + _pw2/2.0/1000.0 + _qsw_delay/1000.0, ~(uint32_t)0 , qswmask));
        patterns.insert(tpat(pos + _pw2/2.0/1000.0 + (_qsw_delay + _qsw_width)/1000.0, 0 , qswmask));
        pos += _tau/1000.0;
         //pi/2 pulse
        //on
        patterns.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, qpskinv(p1[j]), qpskmask));
        patterns.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, ~(uint32_t)0, pulse1mask));
        patterns.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, ~(uint32_t)0, g2mask));
        patterns.insert(tpat(pos - _pw1/2.0/1000.0, PULSE_P1, pulsemask));
        patterns.insert(tpat(pos - _pw1/2.0/1000.0, ~(uint32_t)0, g1mask));
        //off
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, pulsemask));
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, g1mask));
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, pulse1mask));
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, qpsk(p1[j]), qpskmask));
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, g2mask));
        if(! _qsw_pi_only) {
            patterns.insert(tpat(pos + _pw1/2.0/1000.0 + _qsw_delay/1000.0, ~(uint32_t)0 , qswmask));
            patterns.insert(tpat(pos + _pw1/2.0/1000.0 + (_qsw_delay + _qsw_width)/1000.0, 0 , qswmask));
        }
      }
    }

  double curpos = patterns.begin()->pos;
  double lastpos = 0;
  uint32_t pat = 0;
  for(tpatset_it it = patterns.begin(); it != patterns.end(); it++)
    {
      lastpos = it->pos - pos;
      pat &= ~it->mask;
      pat |= (it->pat & it->mask);
    }
    
  for(tpatset_it it = patterns.begin(); it != patterns.end();)
    {
      pat &= ~it->mask;
      pat |= (it->pat & it->mask);
      it++;
      if((it == patterns.end()) ||
             (fabs(it->pos - curpos) > resolution()))
        {
        RelPat relpat(pat, curpos, curpos - lastpos);
        
            m_relPatList.push_back(relpat);
                
              if(it == patterns.end()) break;
              lastpos = curpos;
              curpos = it->pos;
        }
    }
}
#endif //HAVE_NI_DAQMX
