#include "pulserdriverh8.h"
#include "interface.h"
#include <klocale.h>

using std::max;
using std::min;

#define MAX_PATTERN_SIZE 2048

//[ms]
#define TIMER_PERIOD (1.0/(25.0e3))
//[ms]
#define MIN_PULSE_WIDTH 0.001

double XH8Pulser::resolution() {
     return TIMER_PERIOD;
}

#define g1mask 0x0001u
#define g2mask 0x0002u
#define trig1mask 0x0004u
#define trig2mask 0x0008u
#define qpsk1bit 0x0010u
#define qpsk2bit 0x1000u
#define qpskmask ((qpsk1bit*3) | (qpsk2bit*7))
#define aswmask 0x0040u
#define allmask 0xffffu
#define pulse1mask 0x0100u
#define pulse2mask 0x0200u
#define combmask 0x0400u
#define combfmmask 0x8000u
#define pulsebit 0x0000 //nothing
#define pulsemask (pulsebit*7)
#define PULSE_P1 (1*pulsebit)
#define PULSE_P2 (2*pulsebit)
#define PULSE_COMB (3*pulsebit)

#define BLANK_PATTERN combfmmask

XH8Pulser::XH8Pulser(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    XPulser(name, runtime, scalarentries, interfaces, thermometers, drivers)
{
    interface()->setEOS("\r\n");
    interface()->baudrate()->value(115200);
}
void
XH8Pulser::afterStart()
{
  p1Func()->setUIEnabled(false);
  p2Func()->setUIEnabled(false);
  combFunc()->setUIEnabled(false);
  p1Level()->setUIEnabled(false);
  p2Level()->setUIEnabled(false);
  combLevel()->setUIEnabled(false);
  masterLevel()->setUIEnabled(false);
  aswFilter()->setUIEnabled(false);
  portLevel8()->setUIEnabled(false);
  portLevel9()->setUIEnabled(false);
  portLevel10()->setUIEnabled(false);
  portLevel11()->setUIEnabled(false);
  portLevel12()->setUIEnabled(false);
  portLevel13()->setUIEnabled(false);
  portLevel14()->setUIEnabled(false);
  qamOffset1()->setUIEnabled(false);
  qamOffset2()->setUIEnabled(false);
  qamLevel1()->setUIEnabled(false);
  qamLevel2()->setUIEnabled(false);
  qamDelay1()->setUIEnabled(false);
  qamDelay2()->setUIEnabled(false);
  difFreq()->setUIEnabled(false);    
}

void
XH8Pulser::createNativePatterns()
{
  m_zippedPatterns.clear();
  for(RelPatListIterator it = m_relPatList.begin(); it != m_relPatList.end(); it++)
  {
    pulseAdd(it->toappear, it->pattern);
  }
}
int
XH8Pulser::pulseAdd(double msec, unsigned short pattern)
{
  msec = max(msec, MIN_PULSE_WIDTH);

  uint32_t ulen = (uint32_t)floor(
    (msec / TIMER_PERIOD - 1) / 0x8000u);
  uint32_t llen = lrint(
    (msec / TIMER_PERIOD - 1) - ulen * 0x8000u);
  if(llen >= 0x8000u) llen = 0;

  switch(ulen)
    {
   h8ushort x;
    case 0:
      x.msb = llen / 0x100;
      x.lsb = llen % 0x100;
      m_zippedPatterns.push_back(x);
      x.msb = pattern / 0x100;
      x.lsb = pattern % 0x100;
      m_zippedPatterns.push_back(x);
      break;
    default:
      x.msb = (ulen % 0x8000u + 0x8000u) / 0x100;
      x.lsb = (ulen % 0x8000u + 0x8000u) % 0x100;
      m_zippedPatterns.push_back(x);
      x.msb = (ulen / 0x8000u) / 0x100;
      x.lsb = (ulen / 0x8000u) % 0x100;
      m_zippedPatterns.push_back(x);
      x.msb = (llen + 0x8000u) / 0x100;
      x.lsb = (llen + 0x8000u) % 0x100;
      m_zippedPatterns.push_back(x);
      x.msb = pattern / 0x100;
      x.lsb = pattern % 0x100;
      m_zippedPatterns.push_back(x);
      break;
    }
  return 0;
}
static unsigned short makesum(unsigned char *start, uint32_t bytes)
{
  unsigned short sum = 0;

  for(; bytes > 0; bytes--)
    sum += *start++;
  return sum;
}
void
XH8Pulser::changeOutput(bool output)
{
  if(output)
    {
      if(m_zippedPatterns.empty() |
        (m_zippedPatterns.size() >= MAX_PATTERN_SIZE ))
              throw XInterface::XInterfaceError(i18n("Pulser Invalid pattern"), __FILE__, __LINE__);
      XScopedLock<XInterface> lock(*interface());
      for(unsigned int retry = 0; ; retry++) {
          try {
              interface()->sendf("$poff %x", BLANK_PATTERN);
              interface()->send("$pclear");
              unsigned int size = m_zippedPatterns.size();
              unsigned int pincr = size;
              interface()->sendf("$pload %x %x", size, pincr);
              interface()->receive();
              interface()->write(">", 1);
              for(unsigned int j=0; j < size; j += pincr)
                {
                  interface()->write(
                    (char *)&m_zippedPatterns[j], pincr * 2);
                  unsigned short sum = 
                    makesum((unsigned char *)&m_zippedPatterns[j], pincr * 2);
                  h8ushort nsum;
                  nsum.lsb = sum % 0x100; nsum.msb = sum / 0x100;
                  interface()->write((char *)&nsum, 2);
                }
              interface()->write("    \n", 5);
              interface()->receive();
              unsigned int ret;
              if(interface()->scanf("%x", &ret) != 1)
                    throw XInterface::XConvError(__FILE__, __LINE__);
              if(ret != size)
                  throw XInterface::XInterfaceError(i18n("Pulser Check Sum Error"), __FILE__, __LINE__);
          }
          catch (XKameError &e) {
              if(retry > 0) throw e;
              e.print(getName() + ": " + i18n("try to continue") + ", ");
              continue;
          }
          break;
      }
    }
  else
    {
       XScopedLock<XInterface> lock(*interface());
      interface()->sendf("$poff %x", BLANK_PATTERN);
    }
}


#include <set>

void
XH8Pulser::rawToRelPat() throw (XRecordError&)
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
  int comb_rot_num = lrint(*combOffRes() * (_comb_pw / 1000.0) * 4);
    //patterns correspoinding to 0, pi/2, pi, -pi/2
  const unsigned short qpsk1[4] = {0, 1, 3, 2};
  const unsigned short qpsk2[4] = {2, 3, 4, 5};
  //unit of phase is pi/2
  #define qpsk(phase) (qpsk1[(phase) % 4]*qpsk1bit + qpsk2[(phase) % 4]*qpsk2bit)
  #define qpskinv(phase) (qpsk(((phase) + 2) % 4))

  //comb phases
  const unsigned short comb[MAX_NUM_PHASE_CYCLE] = {
    1, 3, 0, 2, 3, 1, 2, 0, 0, 2, 1, 3, 2, 0, 3, 1
  };

  //pi/2 pulse phases
  const unsigned short p1single[MAX_NUM_PHASE_CYCLE] = {
    0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2
  };
  //pi pulse phases
  const unsigned short p2single[MAX_NUM_PHASE_CYCLE] = {
    0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3
  };
  //pi/2 pulse phases for multiple echoes
  const unsigned short p1multi[MAX_NUM_PHASE_CYCLE] = {
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
  };
  //pi pulse phases for multiple echoes
  const unsigned short p2multi[MAX_NUM_PHASE_CYCLE] = {
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3
  };
  
  const unsigned short _qpsk_driven_equilibrium[4] = {2, 1, 0, 3};
  #define qpsk_driven_equilibrium(phase) qpsk(_qpsk_driven_equilibrium[(phase) % 4])
  #define qpsk_driven_equilibrium_inv(phase) (qpsk_driven_equilibrium(((phase) + 2) % 4))


  typedef std::multiset<tpat, std::less<tpat> > tpatset;
  typedef std::multiset<tpat, std::less<tpat> >::iterator tpatset_it;
  tpatset patterns;  //High priority (accuracy in time) patterns
  tpatset patterns_cheap; //Low priority patterns

  m_relPatList.clear();

  double pos = 0;

  int echonum = _echo_num;
  const unsigned short *p1 = (echonum > 1) ? p1multi : p1single;
  const unsigned short *p2 = (echonum > 1) ? p2multi : p2single;
  
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
     
      patterns_cheap.insert(tpat(cpos - _comb_pw/1000.0/2 - _g2_setup/1000.0,
                     g2mask, g2mask));
      patterns_cheap.insert(tpat(cpos - _comb_pw/1000.0/2 - _g2_setup/1000.0, comb_off_res ? ~0 : 0, combfmmask));
      patterns_cheap.insert(tpat(cpos - _comb_pw/1000.0/2 - _g2_setup/1000.0, ~0, combmask));
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
    }   
       pos += _p1;
       
       //pi/2 pulse
      //on
      patterns_cheap.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, qpsk(p1[j]), qpskmask));
      patterns_cheap.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, ~0, pulse1mask));
      patterns_cheap.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, ~0, g2mask));
      patterns.insert(tpat(pos - _pw1/2.0/1000.0, PULSE_P1, pulsemask));
      patterns.insert(tpat(pos - _pw1/2.0/1000.0, ~0, g1mask | trig2mask));
      //off
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, g1mask));
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, pulsemask));
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, pulse1mask));
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, qpsk(p2[j]), qpskmask));
      patterns.insert(tpat(pos + _pw1/2.0/1000.0, ~0, pulse2mask));
     
      //2tau
      pos += 2*_tau/1000.0;
      //    patterns.insert(tpat(pos - _asw_setup, -1, aswmask | rfswmask, 0));
      patterns_cheap.insert(tpat(pos - _asw_setup, ~0, aswmask));
      patterns.insert(tpat(pos -
               ((!former_of_alt && comb_mode_alt) ?
                (double)_alt_sep : 0.0), ~0, trig1mask));

      //pi pulses 
      pos -= 3*_tau/1000.0;
      for(int k = 0;k < echonum; k++)
      {
          pos += 2*_tau/1000.0;
          //on
      if(k >= 1) {
              patterns_cheap.insert(tpat(pos - _pw2/2.0/1000.0 - _g2_setup/1000.0, ~0, g2mask));
      }
          patterns.insert(tpat(pos - _pw2/2.0/1000.0, 0, trig2mask));
          patterns.insert(tpat(pos - _pw2/2.0/1000.0, PULSE_P2, pulsemask));
          patterns.insert(tpat(pos - _pw2/2.0/1000.0, ~0, g1mask));
          //off
          patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, pulse2mask));
          patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, pulsemask));
          patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, g1mask));
          patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, g2mask));
      }
      
       patterns_cheap.insert(tpat(pos + _tau/1000.0 + _asw_hold, 0, aswmask | trig1mask));
      
      if(driven_equilibrium)
      {
        pos += 2*_tau/1000.0;
        //pi pulse 
        //on
        patterns_cheap.insert(tpat(pos - _pw2/2.0/1000.0 - _g2_setup/1000.0, qpsk_driven_equilibrium(p2[j]), qpskmask));
        patterns_cheap.insert(tpat(pos - _pw2/2.0/1000.0 - _g2_setup/1000.0, ~0, g2mask));
        patterns_cheap.insert(tpat(pos - _pw2/2.0/1000.0 - _g2_setup/1000.0, ~0, pulse2mask));
        patterns.insert(tpat(pos - _pw2/2.0/1000.0, PULSE_P2, pulsemask));
        patterns.insert(tpat(pos - _pw2/2.0/1000.0, ~0, g1mask));
        //off
        patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, pulse2mask));
        patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, pulsemask));
        patterns.insert(tpat(pos + _pw2/2.0/1000.0, 0, g1mask | g2mask));
        pos += _tau/1000.0;
         //pi/2 pulse
        //on
        patterns_cheap.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, qpskinv(p1[j]), qpskmask));
        patterns_cheap.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, ~0, pulse1mask));
        patterns_cheap.insert(tpat(pos - _pw1/2.0/1000.0 - _g2_setup/1000.0, ~0, g2mask));
        patterns.insert(tpat(pos - _pw1/2.0/1000.0, PULSE_P1, pulsemask));
        patterns.insert(tpat(pos - _pw1/2.0/1000.0, ~0, g1mask));
        //off
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, pulsemask));
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, g1mask));
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, pulse1mask));
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, qpsk(p1[j]), qpskmask));
        patterns.insert(tpat(pos + _pw1/2.0/1000.0, 0, g2mask));
      }   
    }

  //insert low priority (cheap) pulses into pattern set
  for(tpatset_it it = patterns_cheap.begin(); it != patterns_cheap.end(); it++)
    {
      double npos = it->pos;
      for(tpatset_it kit = patterns.begin(); kit != patterns.end(); kit++)
    {
          //Avoid overrapping within 1 us
      double diff = fabs(kit->pos - npos);
      diff -= pos * floor(diff / pos);
      if(diff <= MIN_PULSE_WIDTH)
        {
          npos = kit->pos;
          break;
        }
    }
      patterns.insert(tpat(npos, it->pat, it->mask));
    }

  double curpos = patterns.begin()->pos;
  double lastpos = 0;
  unsigned short pat = BLANK_PATTERN;
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
