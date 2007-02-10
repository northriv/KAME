#include "pulserdrivernidaqmx.h"

#ifdef HAVE_NI_DAQMX

#define TASK_UNDEF ((TaskHandle)-1)

#include "interface.h"
#include <klocale.h>

using std::max;
using std::min;


//[ms]
#define DMA_DO_PERIOD (1.0/(1e3))

#define SAMPS_AO_PER_DO 1
//[ms]
#define DMA_AO_PERIOD (DMA_DO_PERIOD / SAMPS_AO_PER_DO)

#define BUF_SIZE_HINT 65536
#define CB_TRANSFER_SIZE (BUF_SIZE_HINT/2)

double XNIDAQmxPulser::resolution() {
     return DMA_DO_PERIOD;
}

#define g3mask 0x0010
#define g2mask 0x0002
#define g1mask (0x0001 | g3mask)
#define trig1mask 0x0004
#define trig2mask 0x0008
#define aswmask	0x0080
#define qswmask 0x0040
#define allmask 0xffff
#define pulse1mask 0x0100
#define pulse2mask 0x0200
#define combmask 0x0820
#define combfmmask 0x0400
#define qpskbit 0x10000
#define qpskmask (qpskbit*3)
#define pulsebit 0x40000
#define pulsemask (pulsebit*7)
#define PULSE_P1 (1*pulsebit)
#define PULSE_P2 (2*pulsebit)
#define PULSE_COMB (3*pulsebit)
#define PULSE_INDUCE_EMISSION (4*pulsebit)

XNIDAQmxPulser::XNIDAQmxPulser(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    XNIDAQmxDriver<XPulser>(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_ao_interface(XNode::create<XNIDAQmxInterface>("Interface2", false,
            dynamic_pointer_cast<XDriver>(this->shared_from_this()))),
	 m_taskAO(TASK_UNDEF),
	 m_taskDO(TASK_UNDEF)
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
	if(m_taskDO != TASK_UNDEF) {
	    DAQmxClearTask(m_taskDO);
	    DAQmxStopTask(m_taskCtr);
	    DAQmxClearTask(m_taskCtr);
	}
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
//	std::string freqdev = formatString("%s/freqout", intfDO()->devName()).c_str();
//	std::string freqout = formatString("/%s/FrequencyOutput", intfDO()->devName()).c_str();
	std::string freqdev = formatString("%s/ctr0", intfDO()->devName()).c_str();
	std::string freqout = formatString("/%s/Ctr0InternalOutput", intfDO()->devName()).c_str();
	if(m_taskAO != TASK_UNDEF) {
		freqdev = formatString("%s/ctr0", intfAO()->devName()).c_str();
		freqout = formatString("/%s/Ctr0InternalOutput", intfAO()->devName()).c_str();
	}

 	XScopedLock<XInterface> lock(*intfDO());
	if(m_taskDO != TASK_UNDEF) {
	    DAQmxClearTask(m_taskDO);
	    DAQmxClearTask(m_taskCtr);
	}
	
	float64 freq = 1e3 / DMA_DO_PERIOD;

    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskCtr));
	CHECK_DAQMX_RET(DAQmxCreateCOPulseChanFreq(m_taskCtr, 
    	freqdev.c_str(), "", DAQmx_Val_Hz, DAQmx_Val_Low, 0.0,
    	freq, 0.5));
    //config. of timing is needed for some reasons.
	CHECK_DAQMX_RET(DAQmxCfgImplicitTiming(m_taskCtr, DAQmx_Val_ContSamps, 1000));
    	
	if(m_taskAO != TASK_UNDEF) {
		CHECK_DAQMX_RET(DAQmxCfgDigEdgeStartTrig(m_taskCtr,
			formatString("/%s/ao/StartTrigger", intfAO()->devName()).c_str(),
			DAQmx_Val_Rising));
	}
   CHECK_DAQMX_RET(DAQmxStartTask(m_taskCtr));
   
	CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskDO));

    CHECK_DAQMX_RET(DAQmxCreateDOChan(m_taskDO, 
    	formatString("%s/port0/line0:7", intfDO()->devName()).c_str(),
    	 "", DAQmx_Val_ChanForAllLines));

	CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskDO,
		freqout.c_str(),
		freq, DAQmx_Val_Rising, DAQmx_Val_ContSamps, BUF_SIZE_HINT));
	
	//Buffer setup.
	CHECK_DAQMX_RET(DAQmxCfgOutputBuffer(m_taskDO, BUF_SIZE_HINT));
	uInt32 bufsize;
	CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskDO, &bufsize));
	printf("Using bufsize = %d, freq = %f\n", (int)bufsize, freq);
	if(bufsize < CB_TRANSFER_SIZE * 2)
		throw XInterface::XInterfaceError(KAME::i18n("Insufficient size of NIDAQmx buffer."), __FILE__, __LINE__);
	CHECK_DAQMX_RET(DAQmxSetWriteRegenMode(m_taskDO, DAQmx_Val_DoNotAllowRegen));
	
	CHECK_DAQMX_RET(DAQmxRegisterEveryNSamplesEvent(m_taskDO,
		DAQmx_Val_Transferred_From_Buffer, CB_TRANSFER_SIZE, 0,
		&XNIDAQmxPulser::_genCallBackDO, this));
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
	    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskAO));
	
		openDO();
	
		CHECK_DAQMX_RET(DAQmxCreateAOVoltageChan(m_taskAO,
	    	formatString("%s/ao0:1", intfAO()->devName()).c_str(), "",
	    	-1.0, 1.0, DAQmx_Val_Volts, NULL));
	if(0)
		CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskAO, 
			formatString("/%s/FrequencyOutput", intfDO()->devName()).c_str(),
			1e3 / DMA_AO_PERIOD, DAQmx_Val_Rising, DAQmx_Val_ContSamps,
			BUF_SIZE_HINT * SAMPS_AO_PER_DO));

	if(1) {
		CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskAO, "",
			1e3 / DMA_AO_PERIOD, DAQmx_Val_Rising, DAQmx_Val_ContSamps,
			BUF_SIZE_HINT * SAMPS_AO_PER_DO));
	}

/*		shared_ptr<XNIDAQmxInterface::XNIDAQmxRoute> route;
		route.reset(new XNIDAQmxInterface::XNIDAQmxRoute(
			formatString("/%s/20MHzTimebase", intfDO()->devName()).c_str(),
			formatString("/%s/RTSI7", intfDO()->devName()).c_str()));
		m_routes.push_back(route);
		route.reset(new XNIDAQmxInterface::XNIDAQmxRoute(
			formatString("/%s/RTSI7", intfAO()->devName()).c_str(),
			formatString("/%s/MasterTimebase", intfAO()->devName()).c_str()));
		m_routes.push_back(route);
			
*/		//Buffer setup.
		CHECK_DAQMX_RET(DAQmxCfgOutputBuffer(m_taskAO, BUF_SIZE_HINT * SAMPS_AO_PER_DO));
		uInt32 bufsize;
		CHECK_DAQMX_RET(DAQmxGetBufOutputBufSize(m_taskAO, &bufsize));
		printf("Using bufsize = %d\n", (int)bufsize);
		if(bufsize < CB_TRANSFER_SIZE * 2 * SAMPS_AO_PER_DO)
			throw XInterface::XInterfaceError(
				KAME::i18n("Insufficient size of NIDAQmx buffer."), __FILE__, __LINE__);
		CHECK_DAQMX_RET(DAQmxSetWriteRegenMode(m_taskAO, DAQmx_Val_DoNotAllowRegen));
	
		//obtain range info.
		for(unsigned int ch = 0; ch < NUM_AO_CH; ch++) {
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
		CHECK_DAQMX_RET(DAQmxRegisterEveryNSamplesEvent(m_taskAO,
			DAQmx_Val_Transferred_From_Buffer, CB_TRANSFER_SIZE, 0,
			&XNIDAQmxPulser::_genCallBackAO, this));
		
	}
	catch (XInterface::XInterfaceError &e) {
		e.print(getLabel());
	    DAQmxClearTask(m_taskAO);
	    m_taskAO = TASK_UNDEF;
	    intfAO()->stop();
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
 	XScopedLock<XInterface> lockao(*intfAO());
 	XScopedLock<XInterface> locko(*intfDO());
	if(m_taskAO != TASK_UNDEF)
	    DAQmxClearTask(m_taskAO);
	if(m_taskDO != TASK_UNDEF) {
	    DAQmxClearTask(m_taskDO);
	    DAQmxClearTask(m_taskCtr);
	}
	m_routes.clear();
	m_taskAO = TASK_UNDEF;
	m_taskDO = TASK_UNDEF;
    
	intfDO()->stop();
	intfAO()->stop();
}

void
XNIDAQmxPulser::startPulseGen() throw (XInterface::XInterfaceError &)
{
 	XScopedLock<XInterface> lockao(*intfAO());
 	XScopedLock<XInterface> lockdo(*intfDO());
 	
    DAQmxStopTask(m_taskDO);
	if(m_taskAO != TASK_UNDEF)
	   DAQmxStopTask(m_taskAO);
//	std::deque<GenPattern> m_genPatternList;
	m_genLastPatIt = m_genPatternList.begin();
	m_genLastPattern = m_genPatternList.back().pattern;
	m_genRestSamps = m_genPatternList.back().toappear;
	m_genAOIndex = 0;
	m_genBufDO.resize(CB_TRANSFER_SIZE);
	m_genBufAO.resize(CB_TRANSFER_SIZE * NUM_AO_CH * SAMPS_AO_PER_DO);
	
	//prefilling of our-side buffer.
	genPulseBuffer(CB_TRANSFER_SIZE);
	//transfer twice
	genCallBackDO(m_taskDO, CB_TRANSFER_SIZE);
	genCallBackDO(m_taskDO, CB_TRANSFER_SIZE);
	
	//slave must start before the master.
	if(m_taskAO != TASK_UNDEF)
	    CHECK_DAQMX_RET(DAQmxStartTask(m_taskAO));
    CHECK_DAQMX_RET(DAQmxStartTask(m_taskDO));
}
void
XNIDAQmxPulser::stopPulseGen()
{
 	XScopedLock<XInterface> lockao(*intfAO());
 	XScopedLock<XInterface> lockdo(*intfDO());
	if(m_taskAO != TASK_UNDEF)
	    DAQmxStopTask(m_taskAO);
	if(m_taskDO != TASK_UNDEF) {
	    DAQmxStopTask(m_taskDO);
	    DAQmxStopTask(m_taskCtr);
	}
}
int32
XNIDAQmxPulser::_genCallBackDO(TaskHandle task, int32 /*type*/, uInt32 num_samps, void *data)
{
    XNIDAQmxPulser *obj = reinterpret_cast<XNIDAQmxPulser*>(data);
    return obj->genCallBackDO(task, num_samps);
}
int32
XNIDAQmxPulser::_genCallBackAO(TaskHandle task, int32 /*type*/, uInt32 num_samps, void *data)
{
	printf("a\n");
	
    XNIDAQmxPulser *obj = reinterpret_cast<XNIDAQmxPulser*>(data);
    return obj->genCallBackAO(task, num_samps);
}

XNIDAQmxPulser::tRawAO
XNIDAQmxPulser::aoVoltToRaw(int ch, float64 volt)
{
	volt = std::max(volt, m_lowerLimAO[ch]);
	volt = std::min(volt, m_upperLimAO[ch]);
	float64 x = 1.0;
	float64 y = 0.0;
	float64 *pco = m_coeffAO[ch];
	for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
		y += *(pco++) * x;
		x *= volt;
	}
	return lrint(y);
}
void
XNIDAQmxPulser::genPulseBuffer(uInt32 num_samps)
{
	uint32_t pat = m_genLastPattern;
	GenPatternIterator it = m_genLastPatIt;
	long long int toappear = m_genRestSamps;
	unsigned int aoidx = m_genAOIndex;
	
	C_ASSERT(sizeof(long long int) > sizeof(int32_t));
	
	tRawDO *pDO = &m_genBufDO[0];
	tRawAO *pAO = &m_genBufAO[0];
	tRawAO raw_ao0_zero = aoVoltToRaw(0, 0.0);
	tRawAO raw_ao1_zero = aoVoltToRaw(1, 0.0);
	for(unsigned int samps_rest = num_samps; samps_rest;) {
		unsigned int gen_cnt = std::min((long long int)samps_rest, toappear);
		tRawDO patDO = allmask & pat;
		unsigned int pidx = (pat & pulsemask) / pulsebit;
		C_ASSERT(pulsebit > qpskbit);
		if(pidx == 0) {
			aoidx = 0;
			for(unsigned int cnt = 0; cnt < gen_cnt; cnt++) {
				*pDO++ = patDO;
				for(unsigned int i = 0; i < SAMPS_AO_PER_DO; i++) {
					*pAO++ = raw_ao0_zero;
					*pAO++ = raw_ao1_zero;
				}
			}
		}
		else {
			unsigned int qpskidx = (pat & qpskmask) / qpskbit;
			unsigned int pnum = (pidx - 1) * (pulsebit/qpskbit) + qpskidx;
			tRawAO *pGenAO0 = &m_genPulseWaveAO[0][pnum][aoidx];
			tRawAO *pGenAO1 = &m_genPulseWaveAO[1][pnum][aoidx];
			ASSERT(m_genPulseWaveAO[0][pidx - pulsebit/qpskbit].size() >= aoidx + gen_cnt);
			for(unsigned int cnt = 0; cnt < gen_cnt; cnt++) {
				*pDO++ = patDO;
				for(unsigned int i = 0; i < SAMPS_AO_PER_DO; i++) {
					*pAO++ = *pGenAO0++;
					*pAO++ = *pGenAO1++;
					aoidx++;
				}
			}
		}
		toappear -= gen_cnt;
		samps_rest -= gen_cnt;
		if(toappear == 0) {
			it++;
			if(it == m_genPatternList.end()) {
				it = m_genPatternList.begin();
				printf("p.\n");
			}
			pat = it->pattern;
			toappear = it->toappear;
		}
	}
	ASSERT(pDO == &m_genBufDO[num_samps]);
	ASSERT(pAO == &m_genBufAO[num_samps * SAMPS_AO_PER_DO * NUM_AO_CH]);
	m_genLastPattern = pat;
	m_genRestSamps = toappear;
	m_genLastPatIt = it;
	m_genAOIndex = aoidx;
}
int32
XNIDAQmxPulser::genCallBackDO(TaskHandle /*task*/, uInt32 transfer_size)
{
	try {
	 	XScopedLock<XInterface> lockao(*intfAO());
	 	XScopedLock<XInterface> lockdo(*intfDO());
	 	#define NUM_CB_DIV 2
		for(int cnt = 0; cnt < NUM_CB_DIV; cnt++) {
			uInt32 num_samps = transfer_size / NUM_CB_DIV;
				
			int32 samps;
			if(m_taskDO != TASK_UNDEF) {
				ASSERT(NUM_CB_DIV * num_samps == m_genBufDO.size());
				CHECK_DAQMX_RET(DAQmxWriteDigitalU16(m_taskDO, num_samps, false, 0.3, 
					DAQmx_Val_GroupByChannel, &m_genBufDO[cnt * num_samps], &samps, NULL));
				if(samps != (int32)num_samps) {
					throw XInterface::XInterfaceError("DO: buffer underrun", __FILE__, __LINE__);
				}
			}
			if(m_taskAO != TASK_UNDEF) {
				ASSERT(NUM_CB_DIV * num_samps * SAMPS_AO_PER_DO * NUM_AO_CH== m_genBufAO.size());
				CHECK_DAQMX_RET(DAQmxWriteBinaryI16(m_taskAO, num_samps * SAMPS_AO_PER_DO, false, 0.3, 
					DAQmx_Val_GroupByScanNumber, &m_genBufAO[cnt * num_samps * SAMPS_AO_PER_DO * NUM_AO_CH],
					 &samps, NULL));
				if(samps != (int32)num_samps*SAMPS_AO_PER_DO) {
					throw XInterface::XInterfaceError("AO: buffer underrun", __FILE__, __LINE__);
				}
			}
		}
	}
	catch (XInterface::XInterfaceError &e) {
		e.print(getLabel());
		stopPulseGen();
		return -1;
	}
	//refill our-side buffer.
	genPulseBuffer(transfer_size);
	return 0;
}
int32
XNIDAQmxPulser::genCallBackAO(TaskHandle /*task*/, uInt32 transfer_size)
{
	return 0;
}
void
XNIDAQmxPulser::createNativePatterns()
{
  double _tau = m_tauRecorded;
  double _pw1 = m_pw1Recorded;
  double _pw2 = m_pw2Recorded;
  double _comb_pw = m_combPWRecorded;
  double _dif_freq = m_difFreqRecorded;

  bool _induce_emission = *induceEmission();
  double _induce_emission_pw = _comb_pw;
  double _induce_emission_phase = *induceEmissionPhase() / 180.0 * PI;
      
  m_genPatternList.clear();
  double rest = 0.0;
  for(RelPatListIterator it = m_relPatList.begin(); it != m_relPatList.end(); it++)
  {
	  rest += it->toappear;
  long long int toappear = llrint(rest / DMA_DO_PERIOD);
	  rest -= toappear * DMA_DO_PERIOD;
  GenPattern pat(it->pattern, toappear);
  		m_genPatternList.push_back(pat);
  }
    
  makeWaveForm(PULSE_P1/pulsebit - 1, _pw1/1000.0, pulseFunc(p1Func()->to_str() ), *p1Level()
    , _dif_freq * 1000.0, -2 * PI * _dif_freq * 2 * _tau);
  makeWaveForm(PULSE_P2/pulsebit - 1, _pw2/1000.0, pulseFunc(p2Func()->to_str() ), *p2Level()
    , _dif_freq * 1000.0, -2 * PI * _dif_freq * 2 * _tau);
  makeWaveForm(PULSE_COMB/pulsebit - 1, _comb_pw/1000.0, pulseFunc(combFunc()->to_str() ),
         *combLevel(), *combOffRes() + _dif_freq *1000.0);
  if(_induce_emission) {
      makeWaveForm(PULSE_INDUCE_EMISSION/pulsebit - 1, _induce_emission_pw/1000.0, pulseFunc(combFunc()->to_str() ),
         *combLevel(), *combOffRes() + _dif_freq *1000.0, _induce_emission_phase);
  }
}

int
XNIDAQmxPulser::makeWaveForm(int num, double pw, tpulsefunc func, double dB, double freq, double phase)
{
	for(unsigned int qpsk = 0; qpsk < 4; qpsk++) {
		unsigned int pnum = num * pulsebit/qpskbit + qpsk;
		ASSERT(pnum < 32);
	  	unsigned short word = (unsigned short)lrint(pw / DMA_AO_PERIOD) + SAMPS_AO_PER_DO*2;
		double dx = DMA_AO_PERIOD / pw;
		double dp = 2*PI*freq*DMA_AO_PERIOD + PI/2*qpsk;
		double z = pow(10.0, dB/20.0);
		for(int i = 0; i < word; i++) {
			double w = z * func((i - word / 2.0) * dx) * 1.0;
			double x = w * cos((i - word / 2.0) * dp + PI/4 + phase);
			double y = w * sin((i - word / 2.0) * dp + PI/4 + phase);
			m_genPulseWaveAO[0][pnum].push_back(aoVoltToRaw(0, x));
			m_genPulseWaveAO[1][pnum].push_back(aoVoltToRaw(1, y));
		}
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
          for(int k = 0; k < _comb_num; k++)
        {
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
