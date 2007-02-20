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
//---------------------------------------------------------------------------
#ifndef pulserdriverH
#define pulserdriverH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "xitemnode.h"
#include "xnodeconnector.h"
#include <complex>

//! Modified Bessel Function, 1st type
double bessel_i0(double x);

class FrmPulser;
class FrmPulserMore;
class XQPulserDriverConnector;

//! Base class of NMR Pulsers
class XPulser : public XPrimaryDriver
{	
 XNODE_OBJECT
 protected:
  XPulser(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  virtual ~XPulser() {}
  //! show all forms belonging to driver
  virtual void showForms();

  //! driver specific part below

    
    //! \sa combMode(), combModeRecorded().
	enum {N_COMB_MODE_OFF = 0, N_COMB_MODE_ON = 1, N_COMB_MODE_P1_ALT = 2, N_COMB_MODE_COMB_ALT = 3}; 
    //! \sa rtMode(), rtModeRecorded().
    enum {N_RT_MODE_FIXREP = 0, N_RT_MODE_FIXREST = 1};
	//! \sa numPhaseCycle(), numPhaseCycleRecorded().
	enum {MAX_NUM_PHASE_CYCLE = 16};
	//! # of digital-pulse ports.
	enum {NUM_DO_PORTS= 16};
	
	const shared_ptr<XBoolNode> &output() const {return m_output;}
 	const shared_ptr<XComboNode> &combMode() const {return m_combMode;} //!< see above definitions in header file
      //! Control period to next pulse sequence
      //! Fix Repetition Time or Fix Rest Time which means time between pulse sequences 
 	const shared_ptr<XComboNode> &rtMode() const {return m_rtMode;}
    const shared_ptr<XComboNode> &numPhaseCycle() const {return m_numPhaseCycle;} //!< How many cycles in phase cycling
    const shared_ptr<XDoubleNode> &rtime() const {return m_rt;} //!< Repetition/Rest Time [ms]
    const shared_ptr<XDoubleNode> &tau() const {return m_tau;}  //!< [us]
    const shared_ptr<XDoubleNode> &combPW() const {return m_combPW;} //!< PulseWidths [us]
    const shared_ptr<XDoubleNode> &pw1() const {return m_pw1;} //!< PulseWidths [us]
    const shared_ptr<XDoubleNode> &pw2() const {return m_pw2;} //!< PulseWidths [us]
    const shared_ptr<XDoubleNode> &combPT() const {return m_combPT;} //!< Comb pulse periodic term [us]
    const shared_ptr<XDoubleNode> &combP1() const {return m_combP1;} //!< P1 and P1 alternative
    const shared_ptr<XDoubleNode> &combP1Alt() const {return m_combP1Alt;} //!< P1 and P1 alternative
    const shared_ptr<XDoubleNode> &aswSetup() const {return m_aswSetup;} //!< Analog switch setting, setup(proceeding) time before the first spin echo
    const shared_ptr<XDoubleNode> &aswHold() const {return m_aswHold;}  //!< Analog switch setting, hold time after the last spin echo
    const shared_ptr<XDoubleNode> &altSep() const {return m_altSep;} //!< Separation time in DSO record, cause a shift of trigger of DSO in alternatively mode
    const shared_ptr<XDoubleNode> &g2Setup() const {return m_g2Setup;} //!< Setup time of pre-gating port and QPSK
    const shared_ptr<XUIntNode> &combNum() const {return m_combNum;} //!< # of comb pulses
    const shared_ptr<XUIntNode> &echoNum() const {return m_echoNum;} //!< # of Spin echoes (i.e. pi pulses)
    const shared_ptr<XBoolNode> &drivenEquilibrium() const {return m_drivenEquilibrium;} //!< polarize spins after pulse sequence or not
    const shared_ptr<XDoubleNode> &combOffRes() const {return m_combOffRes;} //!< off-resonance comb pulses
    const shared_ptr<XComboNode> &combFunc() const {return m_combFunc;} //!< Pulse Modulation
    const shared_ptr<XComboNode> &p1Func() const {return m_p1Func;} //!< Pulse Modulation
    const shared_ptr<XComboNode> &p2Func() const {return m_p2Func;} //!< Pulse Modulation
    const shared_ptr<XDoubleNode> &combLevel() const {return m_combLevel;} //!< [dB], Pulse Modulation
    const shared_ptr<XDoubleNode> &p1Level() const {return m_p1Level;} //!< [dB], Pulse Modulation
    const shared_ptr<XDoubleNode> &p2Level() const {return m_p2Level;} //!< [dB], Pulse Modulation
    const shared_ptr<XDoubleNode> &masterLevel() const {return m_masterLevel;} //!< [dB]
    const shared_ptr<XBoolNode> &induceEmission() const {return m_induceEmission;}
    const shared_ptr<XDoubleNode> &induceEmissionPhase() const {return m_induceEmissionPhase;}
    const shared_ptr<XDoubleNode> &qamOffset1() const {return m_qamOffset1;}
    const shared_ptr<XDoubleNode> &qamOffset2() const {return m_qamOffset2;} //!< [%F.S.]
    const shared_ptr<XDoubleNode> &qamLevel1() const {return m_qamLevel1;} //! < Quadrature Amplitude Modulation. Amplitude compensation factor.
    const shared_ptr<XDoubleNode> &qamLevel2() const {return m_qamLevel2;}
    const shared_ptr<XDoubleNode> &qamDelay1() const {return m_qamDelay1;} //! < Delaying compensation [us].
    const shared_ptr<XDoubleNode> &qamDelay2() const {return m_qamDelay2;} //!< [us]
    const shared_ptr<XDoubleNode> &difFreq() const {return m_difFreq;} //!< [MHz]
    const shared_ptr<XDoubleNode> &qswDelay() const {return m_qswDelay;} //!< Q-switch setting, period after the end-edge of pulses [us].
    const shared_ptr<XDoubleNode> &qswWidth() const {return m_qswWidth;} //!< Q-switch setting, width of suppression [us].
    const shared_ptr<XBoolNode> &qswPiPulseOnly() const {return m_qswPiPulseOnly;} //!< Q-switch setting, use QSW only for pi pulses.
    const shared_ptr<XBoolNode> &invertPhase() const {return m_invertPhase;}
    const shared_ptr<XComboNode> &portSel(unsigned int port) const {
    	ASSERT(port < NUM_DO_PORTS);
    	return m_portSel[port];
    }
    
    //! ver 1 records.
    short combModeRecorded() const {return m_combModeRecorded;}
    double rtimeRecorded() const {return m_rtimeRecorded;}
    double tauRecorded() const {return m_tauRecorded;}
    double pw1Recorded() const {return m_pw1Recorded;}
    double pw2Recorded() const {return m_pw2Recorded;}
    double combP1Recorded() const {return m_combP1Recorded;}
    double altSepRecorded() const {return m_altSepRecorded;}
    double combP1AltRecorded() const {return m_combP1AltRecorded;}
    double aswSetupRecorded() const {return m_aswSetupRecorded;}
    double aswHoldRecorded() const {return m_aswHoldRecorded;}
    //! ver 2 records.
    double difFreqRecorded() const {return m_difFreqRecorded;}
    double combPWRecorded() const {return m_combPWRecorded;}
    double combPTRecorded() const {return m_combPTRecorded;}
    unsigned short echoNumRecorded() const {return m_echoNumRecorded;}
    unsigned short combNumRecorded() const {return m_combNumRecorded;}
    short rtModeRecorded() const {return m_rtModeRecorded;}
    unsigned short numPhaseCycleRecorded() const {return m_numPhaseCycleRecorded;}
    //! ver 3 records [experimental].
    bool invertPhaseRecorded() const {return m_invertPhaseRecorded;}
    
    //! periodic term of one cycle [ms].
    double periodicTermRecorded() const;
 protected:
	//! indice for return values of portSel().
  enum {PORTSEL_UNSEL = -1,
  	PORTSEL_GATE = 0, PORTSEL_PREGATE = 1, PORTSEL_GATE3 = 2,
  	PORTSEL_TRIG1 = 3, PORTSEL_TRIG2 = 4, PORTSEL_ASW = 5, PORTSEL_QSW = 6,
  	PORTSEL_PULSE1 = 7, PORTSEL_PULSE2 = 8, 
  	PORTSEL_COMB = 9, PORTSEL_COMB_FM = 10,
  	PORTSEL_QPSK_A = 11, PORTSEL_QPSK_B = 12,
  	PORTSEL_QPSK_OLD_NONINV = 13, PORTSEL_QPSK_OLD_INV = 14,
  	PORTSEL_QPSK_OLD_PSGATE = 15,
  	PORTSEL_PAUSING = 16};
  //! \arg func e.g. PORTSEL_GATE.
  //! \return bit mask.
  unsigned int selectedPorts(int func) const;
  	//! for RelPatList patterns. \sa RelPatList.
  enum {PAT_DO_MASK = (1 << NUM_DO_PORTS) - 1,
  	PAT_QAM_PHASE = (1 << NUM_DO_PORTS),
  	PAT_QAM_PHASE_MASK = PAT_QAM_PHASE * 3,
  	PAT_QAM_PULSE_IDX = PAT_QAM_PHASE * 4,
  	PAT_QAM_PULSE_IDX_P1 = PAT_QAM_PULSE_IDX * 1,
  	PAT_QAM_PULSE_IDX_P2 = PAT_QAM_PULSE_IDX * 2,
  	PAT_QAM_PULSE_IDX_PCOMB = PAT_QAM_PULSE_IDX * 3,
  	PAT_QAM_PULSE_IDX_INDUCE_EMISSION = PAT_QAM_PULSE_IDX * 4,
  	PAT_QAM_PULSE_IDX_MASK = PAT_QAM_PULSE_IDX * 15,
  	PAT_QAM_MASK = PAT_QAM_PHASE_MASK | PAT_QAM_PULSE_IDX_MASK,
  	};
 
  //! Start up your threads, connect GUI, and activate signals
  virtual void start();
  //! Shut down your threads, unconnect GUI, and deactivate signals
  //! this may be called even if driver has already stopped.
  virtual void stop();
  
  //! this is called when raw is written 
  //! unless dependency is broken
  //! convert raw to record
  virtual void analyzeRaw() throw (XRecordError&);
  //! this is called after analyze() or analyzeRaw()
  //! record is readLocked
  virtual void visualize();

  friend class XQPulserDriverConnector;
  
    //! ver 1 records
    short m_combModeRecorded;
    double m_rtimeRecorded;
    double m_tauRecorded;
    double m_pw1Recorded;
    double m_pw2Recorded;
    double m_combP1Recorded;
    double m_altSepRecorded;
    double m_combP1AltRecorded;
    double m_aswSetupRecorded;
    double m_aswHoldRecorded;
    //! ver 2 records
    double m_difFreqRecorded;
    double m_combPWRecorded;
    double m_combPTRecorded;
    unsigned short m_echoNumRecorded;
    unsigned short m_combNumRecorded;
    short m_rtModeRecorded;
    unsigned short m_numPhaseCycleRecorded;        
    //! ver 3 records
    bool m_invertPhaseRecorded;

  struct RelPat {
      RelPat(uint32_t pat, uint64_t t, uint64_t toapp) :
        pattern(pat), time(t), toappear(toapp) {}
      uint32_t pattern;
      uint64_t time; //!< unit of resolution().
      uint64_t toappear; //!< term between this pattern and the previous. unit of resolution().
  };

  typedef std::deque<RelPat> RelPatList;
  RelPatList m_relPatList;
  typedef RelPatList::iterator RelPatListIterator;

  //! push parameters.
  //! use this after clearRaw()
  void writeRaw();
  
  typedef double (*tpulsefunc)(double x);
  tpulsefunc pulseFunc(const std::string &str) const;
  static double pulseFuncRect(double x);
  static double pulseFuncHanning(double x);
  static double pulseFuncHamming(double x);
  static double pulseFuncBlackman(double x);
  static double pulseFuncBlackmanHarris(double x);
  static double pulseFuncKaiser(double x, double alpha);
  static double pulseFuncKaiser1(double x);
  static double pulseFuncKaiser2(double x);
  static double pulseFuncKaiser3(double x);
  static double pulseFuncFlatTop(double x);
  static double pulseFuncFlatTopLong(double x);
  static double pulseFuncFlatTopLongLong(double x);
  static double pulseFuncHalfCos(double x);
  static double pulseFuncChoppedHalfCos(double x);
  
  
    //! send patterns to pulser or turn-off
    virtual void changeOutput(bool output, unsigned int blankpattern) = 0;
    //! convert RelPatList to native patterns
    virtual void createNativePatterns() = 0;
    //! time resolution [ms]
    virtual double resolution() const = 0;
    virtual double resolutionQAM() const = 0;
    //! minimum period of pulses [ms]
    virtual double minPulseWidth() const = 0;
    //! existense of AO ports.
    virtual bool haveQAMPorts() const = 0;

  	const std::vector<std::complex<double> > &qamWaveForm(unsigned int idx) const
  	 	 {return m_qamWaveForm[idx];}

  
 private:
    const shared_ptr<XBoolNode> m_output;
    const shared_ptr<XComboNode> m_combMode; //!< see above definitions in header file
      //! Control period to next pulse sequence
      //! Fix Repetition Time or Fix Rest Time which means time between pulse sequences 
    const shared_ptr<XComboNode> m_rtMode;
    const shared_ptr<XDoubleNode> m_rt; //!< Repetition/Rest Time [ms]
    const shared_ptr<XDoubleNode> m_tau;  //!< [us]
    const shared_ptr<XDoubleNode> m_combPW, m_pw1, m_pw2; //!< PulseWidths [us]
    const shared_ptr<XUIntNode> m_combNum; //!< # of comb pulses
    const shared_ptr<XDoubleNode> m_combPT; //!< Comb pulse periodic term [us]
    const shared_ptr<XDoubleNode> m_combP1, m_combP1Alt; //!< P1 and P1 alternative
    const shared_ptr<XDoubleNode> m_aswSetup; //!< Analog switch setting, setup(proceeding) time before the first spin echo
    const shared_ptr<XDoubleNode> m_aswHold;  //!< Analog switch setting, hold time after the last spin echo
    const shared_ptr<XDoubleNode> m_altSep; //!< Separation time in DSO record, cause a shift of trigger of DSO in alternatively mode
    const shared_ptr<XDoubleNode> m_g2Setup; //!< Setup time of pre-gating port and QPSK
    const shared_ptr<XUIntNode> m_echoNum; //!< # of Spin echoes (i.e. pi pulses)
    const shared_ptr<XDoubleNode> m_combOffRes; //!< off-resonance comb pulses
    const shared_ptr<XBoolNode> m_drivenEquilibrium; //!< polarize spins after pulse sequence or not
    const shared_ptr<XComboNode> m_numPhaseCycle; //!< How many cycles in phase cycling
    const shared_ptr<XComboNode> m_p1Func, m_p2Func, m_combFunc; //!< Pulse Modulation
    const shared_ptr<XDoubleNode> m_p1Level, m_p2Level, m_combLevel; //!< [dB], Pulse Modulation
    const shared_ptr<XDoubleNode> m_masterLevel; //!< [dB]
    const shared_ptr<XDoubleNode> m_qamOffset1;
    const shared_ptr<XDoubleNode> m_qamOffset2; //!< [%F.S.]
    const shared_ptr<XDoubleNode> m_qamLevel1;
    const shared_ptr<XDoubleNode> m_qamLevel2;
    const shared_ptr<XDoubleNode> m_qamDelay1;
    const shared_ptr<XDoubleNode> m_qamDelay2; //!< [us]
    const shared_ptr<XDoubleNode> m_difFreq; //!< [MHz]
    const shared_ptr<XBoolNode> m_induceEmission; 
    const shared_ptr<XDoubleNode> m_induceEmissionPhase; 
    const shared_ptr<XDoubleNode> m_qswDelay;
    const shared_ptr<XDoubleNode> m_qswWidth;
    const shared_ptr<XBoolNode> m_invertPhase;
    const shared_ptr<XBoolNode> m_qswPiPulseOnly;
    shared_ptr<XComboNode> m_portSel[NUM_DO_PORTS];
    
  const shared_ptr<XNode> m_moreConfigShow;
  xqcon_ptr m_conOutput;
  xqcon_ptr m_conCombMode, m_conRTMode;
  xqcon_ptr m_conRT, m_conTau, m_conCombPW, m_conPW1, m_conPW2,
    m_conCombNum, m_conCombPT, m_conCombP1, m_conCombP1Alt,
    m_conASWHold, m_conASWSetup, m_conALTSep, m_conG2Setup,
    m_conEchoNum, m_conDrivenEquilibrium, m_conNumPhaseCycle, m_conCombOffRes, m_conInvertPhase,
    m_conCombFunc, m_conP1Func, m_conP2Func,
    m_conCombLevel, m_conP1Level, m_conP2Level,
    m_conMasterLevel,
    m_conQAMOffset1, m_conQAMOffset2,
    m_conQAMLevel1, m_conQAMLevel2,
    m_conQAMDelay1, m_conQAMDelay2,
    m_conMoreConfigShow,
    m_conDIFFreq,
    m_conInduceEmission, m_conInduceEmissionPhase,
    m_conQSWDelay, m_conQSWWidth, m_conQSWPiPulseOnly;
   xqcon_ptr m_conPortSel[NUM_DO_PORTS];
  shared_ptr<XListener> m_lsnOnPulseChanged;
  shared_ptr<XListener> m_lsnOnMoreConfigShow;
  void onMoreConfigShow(const shared_ptr<XNode> &);

  const qshared_ptr<FrmPulser> m_form;
  const qshared_ptr<FrmPulserMore> m_formMore;
  
  xqcon_ptr m_conPulserDriver;
    
  void onPulseChanged(const shared_ptr<XValueNodeBase> &);

  //! create RelPatList
  void rawToRelPat() throw (XRecordError&);

  //! prepare waveforms for QAM.
  void makeWaveForm(unsigned int pnum_minus_1, 
	 double pw, unsigned int to_center,
  	 tpulsefunc func, double dB, double freq = 0.0, double phase = 0.0);
  std::vector<std::complex<double> > m_qamWaveForm[PAT_QAM_PULSE_IDX_MASK/PAT_QAM_PULSE_IDX];
  
  //! truncate time by resolution().
  inline double rintTermMilliSec(double msec) const;
  inline double rintTermMicroSec(double usec) const;
  inline uint64_t ceilSampsMicroSec(double us) const;
  inline uint64_t rintSampsMicroSec(double us) const;
  inline uint64_t rintSampsMilliSec(double ms) const;
 };

#endif
