#include "pulserdrivernidaqmx.h"

#ifdef HAVE_NI_DAQMX

#define TASK_UNDEF ((TaskHandle)-1)
#define BUF_SIZE_HINT 10000

#include "interface.h"
#include <klocale.h>

using std::max;
using std::min;


//[ms]
#define DMA_PERIOD (1.0/(1e3))

double XNIDAQmxPulser::resolution() {
     return DMA_PERIOD;
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
    XNIDAQmxDriver<XDSO>(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_ao_interface(XNode::create<XNIDAQmxInterface>("Interface2", false,
            dynamic_pointer_cast<XDriver>(this->shared_from_this()))),
	 m_taskAO(TASK_UNDEF),
	 m_taskDO(TASK_UNDEF)
{
    interfaces->insert(m_ao_interface);

}
XNIDAQmxPulser::~XNIDAQmxPulser()
{
	if(m_taskAO != TASK_UNDEF)
	    DAQmxClearTask(m_taskAO);
	if(m_taskDO != TASK_UNDEF)
	    DAQmxClearTask(m_taskDO);
}

void
XNIDAQmxPulser::open() throw (XInterface::XInterfaceError &)
{
 	XScopedLock<XInterface> lock(*intfDO());
    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskDO));

    CHECK_DAQMX_RET(DAQmxCreateDOVoltageChan(m_taskDO, 
    	(QString("%1/port0/line0:7").arg(intfDO()->devName())), "", DAQmx_Val_ChanForAllLines));
	
	CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskDO, NULL,
		1e3 / DMA_PERIOD, DAQmx_Val_Rising, DAQmx_Val_ContSamps, BUF_SIZE_HINT));
	
	this->start();	
}
void
XNIDAQmxPulser::openAODevice() throw (XInterface::XInterfaceError &)
{
 	XScopedLock<XInterface> lock(*intfAO());
    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_taskAO));

	CHECK_DAQMX_RET(DAQmxCreateAOVoltageChan(m_taskAO,
    	(QString("%1/ao0:1").arg(intfAO()->devName())), "",
    	-1.0, 1.0, DAQmx_Val_Volts, NULL));

	CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_taskAO, NULL,
		1e3 / DMA_PERIOD, DAQmx_Val_Rising, DAQmx_Val_ContSamps, BUF_SIZE_HINT));
		
}
void
XNIDAQmxPulser::close() throw (XInterface::XInterfaceError &)
{
 	XScopedLock<XInterface> lock(*intfAO());
 	XScopedLock<XInterface> lock(*intfDO());
	if(m_taskAO != TASK_UNDEF)
	    DAQmxClearTask(m_taskAO);
	if(m_taskDO != TASK_UNDEF)
	    DAQmxClearTask(m_taskDO);
	m_taskAO = TASK_UNDEF;
	m_taskDO = TASK_UNDEF;
    
	intfDO()->stop();
	intfAO()->stop();
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
      
  //dry-run to determin LastPattern, DMATime
  m_dmaTerm = 0.0;
  m_lastPattern = 0;
  uint32_t pat = 0;
  insertPreamble((unsigned short)pat);
  for(RelPatListIterator it = m_relPatList.begin(); it != m_relPatList.end(); it++)
  {
    pulseAdd(it->toappear, it->pattern, (it == m_relPatList.begin() ) );
    pat = it->pattern;
  }
  
  insertPreamble((unsigned short)pat);
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
  m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DO);
  m_zippedPatterns.push_back(0);
  m_zippedPatterns.push_back(0);
  for(RelPatListIterator it = m_relPatList.begin(); it != m_relPatList.end(); it++)
  {
    pulseAdd(it->toappear, it->pattern, (it == m_relPatList.begin() ) );
    pat = it->pattern;
  }
  finishPulse();

  m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_END);
}

int
XNIDAQmxPulser::makeWaveForm(int num, double pw, tpulsefunc func, double dB, double freq, double phase)
{
	m_waveformPos[num] = m_zippedPatterns.size();
  	unsigned short word = (unsigned short)rint(pw / DMA_PERIOD);
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_HBURST);
	m_zippedPatterns.push_back((unsigned char)(word / 0x100) );
	m_zippedPatterns.push_back((unsigned char)(word % 0x100) );
	double dx = DMA_PERIOD / pw;
	double dp = 2*PI*freq*DMA_PERIOD;
	double z = pow(10.0, dB/20.0);
	for(int i = 0; i < word; i++) {
		double w = z * func((i - word / 2.0) * dx) * 125.0;
		double x = w * cos((i - word / 2.0) * dp + PI/4 + phase);
		double y = w * sin((i - word / 2.0) * dp + PI/4 + phase);
		x = max(min(x, 124.0), -124.0);
		y = max(min(y, 124.0), -124.0);
		m_zippedPatterns.push_back( (unsigned char)(char)rint(x) );	
		m_zippedPatterns.push_back( (unsigned char)(char)rint(y) );	
	}
	return 0;
}
int
XNIDAQmxPulser::setAUX2DA(double volt, int addr)
{
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_AUX2_DA);
	m_zippedPatterns.push_back((unsigned char) addr);
	volt = max(volt, 0.0);
	unsigned short word = (unsigned short)rint(4096u * volt / 2 / 2.5);
	word = min(word, (unsigned short)4095u);
	m_zippedPatterns.push_back((unsigned char)(word / 0x100) );
	m_zippedPatterns.push_back((unsigned char)(word % 0x100) );
	return 0;
}
int
XNIDAQmxPulser::insertPreamble(unsigned short startpattern)
{
double masterlevel = pow(10.0, *masterLevel() / 20.0);
double qamlevel1 = *qamLevel1();
double qamlevel2 = *qamLevel2();
	m_zippedPatterns.clear();
	
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_SET_DA_TUNE_OFFSET);
	m_zippedPatterns.push_back((unsigned char)(signed char)rint(127.5 * *qamOffset1() *1e-2 / masterlevel ) );
	m_zippedPatterns.push_back((unsigned char)(signed char)rint(127.5 * *qamOffset2() *1e-2 / masterlevel ) );
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_SET_DA_TUNE_LEVEL);
	m_zippedPatterns.push_back((unsigned char)(signed char)rint(qamlevel1 * 0x100) );
	m_zippedPatterns.push_back((unsigned char)(signed char)rint(qamlevel2 * 0x100) );
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_SET_DA_TUNE_DELAY);
	m_zippedPatterns.push_back((unsigned char)(signed char)rint(*qamDelay1() / DMA_PERIOD * 1e-3) );
	m_zippedPatterns.push_back((unsigned char)(signed char)rint(*qamDelay2() / DMA_PERIOD * 1e-3) );
	
uint32_t len;	
	//wait for 1 ms
	len = lrint(1.0 / MTU_PERIOD);
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_WAIT_LONG);
	m_zippedPatterns.push_back((unsigned char)((len % 0x10000) / 0x100) );
	m_zippedPatterns.push_back((unsigned char)((len % 0x10000) % 0x100) );
	m_zippedPatterns.push_back((unsigned char)((len / 0x10000) / 0x100) );
	m_zippedPatterns.push_back((unsigned char)((len / 0x10000) % 0x100) );
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_SET);
	m_zippedPatterns.push_back((unsigned char)(startpattern / 0x100) );
	m_zippedPatterns.push_back((unsigned char)(startpattern % 0x100) );
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_LSET_START + 2);
	m_zippedPatterns.push_back((unsigned char)(startpattern / 0x100) );
	m_zippedPatterns.push_back((unsigned char)(startpattern % 0x100) );
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_END);
	
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_PULSEON);
	
	//wait for 10 ms
	len = lrint(10.0 / MTU_PERIOD);
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_WAIT_LONG);
	m_zippedPatterns.push_back((unsigned char)((len % 0x10000) / 0x100) );
	m_zippedPatterns.push_back((unsigned char)((len % 0x10000) % 0x100) );
	m_zippedPatterns.push_back((unsigned char)((len / 0x10000) / 0x100) );
	m_zippedPatterns.push_back((unsigned char)((len / 0x10000) % 0x100) );
	
	
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_AUX1);
	int aswfilter = 3;
	if(aswFilter()->to_str() == ASW_FILTER_1) aswfilter = 1;
	if(aswFilter()->to_str() == ASW_FILTER_2) aswfilter = 2;
	m_zippedPatterns.push_back((unsigned char)aswfilter);

	setAUX2DA(*portLevel8(), 1);
	setAUX2DA(*portLevel9(), 2);
	setAUX2DA(*portLevel10(), 3);
	setAUX2DA(*portLevel11(), 4);
	setAUX2DA(*portLevel12(), 5);
	setAUX2DA(*portLevel13(), 6);
	setAUX2DA(*portLevel14(), 7);
	setAUX2DA(1.6 * masterlevel, 0); //tobe 5V
	
	return 0;	
}
int
XNIDAQmxPulser::finishPulse(void)
{
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_END);
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_LOOP_INF);
	return 0;
}
int
XNIDAQmxPulser::pulseAdd(double msec, uint32_t pattern, bool firsttime)
{
  if( (msec > MIN_MTU_LEN) && ((m_lastPattern & pulsemask)/pulsebit == 0) ) {
  //insert long wait
	if(!firsttime) {
		m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_END);
	}
	msec += m_dmaTerm;
	uint32_t ulen = (uint32_t)floor((msec / MTU_PERIOD) / (double)0x10000);
	unsigned short ulenh = (unsigned short)(ulen / 0x10000uL);
	unsigned short ulenl = (unsigned short)(ulen % 0x10000uL);	
	unsigned short dlen = (uint32_t)floor((msec / MTU_PERIOD) - ulen * (double)0x10000);
	if(ulenh) {
		m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_WAIT_LONG_LONG);
		m_zippedPatterns.push_back((unsigned char)(dlen / 0x100) );
		m_zippedPatterns.push_back((unsigned char)(dlen % 0x100) );
		m_zippedPatterns.push_back((unsigned char)(ulenh / 0x100) );
		m_zippedPatterns.push_back((unsigned char)(ulenh % 0x100) );
		m_zippedPatterns.push_back((unsigned char)(ulenl / 0x100) );
		m_zippedPatterns.push_back((unsigned char)(ulenl % 0x100) );
	}
	else { if(ulenl) {
		m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_WAIT_LONG);
		m_zippedPatterns.push_back((unsigned char)(dlen / 0x100) );
		m_zippedPatterns.push_back((unsigned char)(dlen % 0x100) );
		m_zippedPatterns.push_back((unsigned char)(ulenl / 0x100) );
		m_zippedPatterns.push_back((unsigned char)(ulenl % 0x100) );
	  }
	 else {
		m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_WAIT);
		m_zippedPatterns.push_back((unsigned char)(dlen / 0x100) );
		m_zippedPatterns.push_back((unsigned char)(dlen % 0x100) );
	  }
	}
	msec -= ((double)ulen*0x10000ul + dlen) * MTU_PERIOD;
	msec = max(0.0, msec);
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_SET);
	m_zippedPatterns.push_back((unsigned char)(m_lastPattern / 0x100) );
	m_zippedPatterns.push_back((unsigned char)(m_lastPattern % 0x100) );
	m_dmaTerm = 0.0;
  }
  m_dmaTerm += msec;
  unsigned long pos_l = rintl(m_dmaTerm / DMA_PERIOD);
  if(pos_l >= 0x7000u)
     throw XInterface::XInterfaceError(KAME::i18n("Too long DMA."), __FILE__, __LINE__);
  unsigned short pos = (unsigned short)pos_l;
  unsigned short len = (unsigned short)rintl(msec / DMA_PERIOD);
  if( ((m_lastPattern & pulsemask)/pulsebit == 0) && ((pattern & pulsemask)/pulsebit > 0) ) {
	unsigned short word = m_zippedPatterns.size() - m_waveformPos[(pattern & pulsemask)/pulsebit - 1];
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_COPY_HBURST);
	m_zippedPatterns.push_back((unsigned char)((pattern & qpskmask)/qpskbit));
	m_zippedPatterns.push_back((unsigned char)(pos / 0x100) );
	m_zippedPatterns.push_back((unsigned char)(pos % 0x100) );
	m_zippedPatterns.push_back((unsigned char)(word / 0x100) );
	m_zippedPatterns.push_back((unsigned char)(word % 0x100) );
  }
  if(len > PATTERN_ZIPPED_COMMAND_DMA_LSET_END - PATTERN_ZIPPED_COMMAND_DMA_LSET_START) {
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_LSET_LONG);
	m_zippedPatterns.push_back((unsigned char)(len / 0x100) );
	m_zippedPatterns.push_back((unsigned char)(len % 0x100) );
	m_zippedPatterns.push_back((unsigned char)(pattern / 0x100) );
	m_zippedPatterns.push_back((unsigned char)(pattern % 0x100) );
  }
  else {
	m_zippedPatterns.push_back(PATTERN_ZIPPED_COMMAND_DMA_LSET_START + (unsigned char)len);
	m_zippedPatterns.push_back((unsigned char)(pattern / 0x100) );
	m_zippedPatterns.push_back((unsigned char)(pattern % 0x100) );
  }
  m_lastPattern = pattern;
  return 0;
}

void
XNIDAQmxPulser::changeOutput(bool output)
{
  if(output)
    {
      if(m_zippedPatterns.empty() )
              throw XInterface::XInterfaceError(KAME::i18n("Pulser Invalid pattern"), __FILE__, __LINE__);
      XScopedLock<XInterface> lock(*interface());
      for(unsigned int retry = 0; ; retry++) {
          try {
              interface()->write("!", 1); //poff
              interface()->receive();
              char buf[3];
              if((interface()->scanf("Pulse %3s", buf) != 1) || strncmp(buf, "Off", 3))
                    throw XInterface::XConvError(__FILE__, __LINE__);
              unsigned int size = m_zippedPatterns.size();
              interface()->sendf("$pload %x", size );
              interface()->receive();
              interface()->write(">", 1);
              unsigned short sum = 0;
              for(unsigned int i = 0; i < m_zippedPatterns.size(); i++) {
        	       sum += m_zippedPatterns[i];
              } 
              msecsleep(1);
              interface()->write((char*)&m_zippedPatterns[0], size);
          
              interface()->receive();
              unsigned int ret;
              if(interface()->scanf("%x", &ret) != 1)
                    throw XInterface::XConvError(__FILE__, __LINE__);
              if(ret != sum)
                    throw XInterface::XInterfaceError(KAME::i18n("Pulser Check Sum Error"), __FILE__, __LINE__);
              interface()->send("$pon");
              interface()->receive();
              if((interface()->scanf("Pulse %2s", buf) != 1) || strncmp(buf, "On", 2))
                    throw XInterface::XConvError(__FILE__, __LINE__);
          }
          catch (XKameError &e) {
              if(retry > 1) throw e;
              e.print(getLabel() + ": " + KAME::i18n("try to continue") + ", ");
              continue;
          }
          break;
      }
    }
  else
    {
    XScopedLock<XInterface> lock(*interface());
      interface()->write("!", 1); //poff
      interface()->receive();
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
      int j = (i / (comb_mode_alt ? 2 : 1) ^ m_phase_xor) % _num_phase_cycle; //index for phase cycling
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
      patterns.insert(tpat(cpos - _comb_pw/1000.0/2 - _g2_setup/1000.0, comb_off_res ? ~0 : 0, combfmmask));
      patterns.insert(tpat(cpos - _comb_pw/1000.0/2 - _g2_setup/1000.0, ~0, combmask));
          for(int k = 0; k < _comb_num; k++)
        {
              patterns.insert(tpat(cpos + _comb_pw/2/1000.0 , qpsk(comb[j]), qpskmask));
          cpos += combpt;
          cpos -= _comb_pw/2/1000.0;
          patterns.insert(tpat(cpos, ~0, g1mask));
          patterns.insert(tpat(cpos, PULSE_COMB, pulsemask));
          cpos += _comb_pw/1000.0;      
          patterns.insert(tpat(cpos, 0 , g1mask));
          patterns.insert(tpat(cpos, 0, pulsemask));

          cpos -= _comb_pw/2/1000.0;
        }
      patterns.insert(tpat(cpos + _comb_pw/2/1000.0, 0, g2mask));
      patterns.insert(tpat(cpos + _comb_pw/2/1000.0, 0, combmask));
      patterns.insert(tpat(cpos + _comb_pw/1000.0/2, ~0, combfmmask));
      if(! _qsw_pi_only) {
          patterns.insert(tpat(cpos + _comb_pw/2/1000.0 + _qsw_delay/1000.0, ~0 , qswmask));
          patterns.insert(tpat(cpos + _comb_pw/2/1000.0 + (_qsw_delay + _qsw_width)/1000.0, 0 , qswmask));
      }
    }   
       pos += _p1;
       
       //pi/2 pulse
      //on
      patterns.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, qpsk(p1[j]), qpskmask));
      patterns.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, ~0, pulse1mask));
      patterns.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, ~0, g2mask));
      patterns.insert(tpat(pos - _pw1/2.0/1000.0, PULSE_P1, pulsemask));
      patterns.insert(tpat(pos - _pw1/2.0/1000.0, ~0, g1mask | trig2mask));
      //off
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, g1mask));
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, pulsemask));
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, pulse1mask));
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, qpsk(p2[j]), qpskmask));
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, ~0, pulse2mask));
      if(! _qsw_pi_only) {
          patterns.insert(tpat(pos + _pw1/2.0/1000.0 + _qsw_delay/1000.0, ~0 , qswmask));
          patterns.insert(tpat(pos + _pw1/2.0/1000.0 + (_qsw_delay + _qsw_width)/1000.0, 0 , qswmask));
      }
     
      //2tau
      pos += 2*_tau/1000.0;
      //    patterns.insert(tpat(pos - _asw_setup, -1, aswmask | rfswmask, 0));
      patterns.insert(tpat(pos - _asw_setup, ~0, aswmask));
      patterns.insert(tpat(pos -
               ((!former_of_alt && comb_mode_alt) ?
                (double)_alt_sep : 0.0), ~0, trig1mask));
                
      //induce emission
      if(_induce_emission) {
          patterns.insert(tpat(pos - _induce_emission_pw/2.0/1000.0, ~0, g3mask));
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
              patterns.insert(tpat(pos - _pw2/2.0/1000.0 - _g2_setup/1000.0, ~0, g2mask));
      }
          patterns.insert(tpat(pos - _pw2/2.0/1000.0, 0, trig2mask));
          patterns.insert(tpat(pos - _pw2/2.0/1000.0, PULSE_P2, pulsemask));
          patterns.insert(tpat(pos - _pw2/2.0/1000.0, ~0, g1mask));
          //off
          patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, pulse2mask));
          patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, pulsemask));
          patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, g1mask));
          patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, g2mask));
          patterns.insert(tpat(pos + _pw2/2.0/1000.0 + _qsw_delay/1000.0, ~0 , qswmask));
          patterns.insert(tpat(pos + _pw2/2.0/1000.0 + (_qsw_delay + _qsw_width)/1000.0, 0 , qswmask));
      }

       patterns.insert(tpat(pos + _tau/1000.0 + _asw_hold, 0, aswmask | trig1mask));
      //induce emission
      if(_induce_emission) {
          patterns.insert(tpat(pos + _tau/1000.0 + _asw_hold - _induce_emission_pw/2.0/1000.0, ~0, g3mask));
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
        patterns.insert(tpat(pos - _pw2/2.0/1000.0 - _g2_setup/1000.0, ~0, g2mask));
        patterns.insert(tpat(pos - _pw2/2.0/1000.0 - _g2_setup/1000.0, ~0, pulse2mask));
        patterns.insert(tpat(pos - _pw2/2.0/1000.0, PULSE_P2, pulsemask));
        patterns.insert(tpat(pos - _pw2/2.0/1000.0, ~0, g1mask));
        //off
        patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, pulse2mask));
        patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, pulsemask));
        patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, g1mask | g2mask));
        patterns.insert(tpat(pos + _pw2/2.0/1000.0 + _qsw_delay/1000.0, ~0 , qswmask));
        patterns.insert(tpat(pos + _pw2/2.0/1000.0 + (_qsw_delay + _qsw_width)/1000.0, 0 , qswmask));
        pos += _tau/1000.0;
         //pi/2 pulse
        //on
        patterns.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, qpskinv(p1[j]), qpskmask));
        patterns.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, ~0, pulse1mask));
        patterns.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, ~0, g2mask));
        patterns.insert(tpat(pos - _pw1/2.0/1000.0, PULSE_P1, pulsemask));
        patterns.insert(tpat(pos - _pw1/2.0/1000.0, ~0, g1mask));
        //off
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, pulsemask));
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, g1mask));
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, pulse1mask));
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, qpsk(p1[j]), qpskmask));
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, g2mask));
        if(! _qsw_pi_only) {
            patterns.insert(tpat(pos + _pw1/2.0/1000.0 + _qsw_delay/1000.0, ~0 , qswmask));
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
