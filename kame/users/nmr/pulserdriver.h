//---------------------------------------------------------------------------
#ifndef pulserdriverH
#define pulserdriverH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "xitemnode.h"
#include "xnodeconnector.h"

//! Modified Bessel Function, 1st type
double bessel_i0(double x);

class FrmPulser;
class FrmPulserMore;
class XQPulserDriverConnector;

#define N_COMB_MODE_OFF 0
#define N_COMB_MODE_ON 1
#define N_COMB_MODE_P1_ALT 2
#define N_COMB_MODE_COMB_ALT 3

#define N_RT_MODE_FIXREP 0
#define N_RT_MODE_FIXREST 1

#define NUM_PHASE_CYCLE_1 "1"
#define NUM_PHASE_CYCLE_2 "2"
#define NUM_PHASE_CYCLE_4 "4"
#define NUM_PHASE_CYCLE_8 "8"
#define NUM_PHASE_CYCLE_16 "16"
#define MAX_NUM_PHASE_CYCLE 16


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
  const   shared_ptr<XBoolNode> &output() const {return m_output;}
    const shared_ptr<XComboNode> &combMode() const {return m_combMode;} //!< see above definitions in header file
      //! Control period to next pulse sequence
      //! Fix Repetition Time or Fix Rest Time which means time between pulse sequences 
  const   shared_ptr<XComboNode> &rtMode() const {return m_rtMode;}
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
    const shared_ptr<XComboNode> &aswFilter() const {return m_aswFilter;}
    const shared_ptr<XBoolNode> &induceEmission() const {return m_induceEmission;}
    const shared_ptr<XDoubleNode> &induceEmissionPhase() const {return m_induceEmissionPhase;}
    const shared_ptr<XDoubleNode> &portLevel8() const {return m_portLevel8;}
    const shared_ptr<XDoubleNode> &portLevel9() const {return m_portLevel9;}
    const shared_ptr<XDoubleNode> &portLevel10() const {return m_portLevel10;}
    const shared_ptr<XDoubleNode> &portLevel11() const {return m_portLevel11;}
    const shared_ptr<XDoubleNode> &portLevel12() const {return m_portLevel12;}
    const shared_ptr<XDoubleNode> &portLevel13() const {return m_portLevel13;}
    const shared_ptr<XDoubleNode> &portLevel14() const {return m_portLevel14;} //!< [V]
    const shared_ptr<XDoubleNode> &qamOffset1() const {return m_qamOffset1;}
    const shared_ptr<XDoubleNode> &qamOffset2() const {return m_qamOffset2;} //!< [%F.S.]
    const shared_ptr<XDoubleNode> &qamLevel1() const {return m_qamLevel1;}
    const shared_ptr<XDoubleNode> &qamLevel2() const {return m_qamLevel2;}
    const shared_ptr<XDoubleNode> &qamDelay1() const {return m_qamDelay1;}
    const shared_ptr<XDoubleNode> &qamDelay2() const {return m_qamDelay2;} //!< [us]
    const shared_ptr<XDoubleNode> &difFreq() const {return m_difFreq;} //!< [MHz]

    //! ver 1 records
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
    //! ver 2 records
    double difFreqRecorded() const {return m_difFreqRecorded;}
    double combPWRecorded() const {return m_combPWRecorded;}
    double combPTRecorded() const {return m_combPTRecorded;}
    unsigned short echoNumRecorded() const {return m_echoNumRecorded;}
    unsigned short combNumRecorded() const {return m_combNumRecorded;}
    short rtModeRecorded() const {return m_rtModeRecorded;}
    unsigned short numPhaseCycleRecorded() const {return m_numPhaseCycleRecorded;}
    
    //! periodic term of one cycle [ms]
    double periodicTermRecorded() const;

    void setPhaseCycleOrder(unsigned int x) {m_phase_xor = x;}
 protected:
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
  
    //! send patterns to pulser or turn-off
    virtual void changeOutput(bool output) = 0;
    //! convert RelPatList to native patterns
    virtual void createNativePatterns() = 0;
    //! time resolution [ms]
    virtual double resolution() = 0;
    //! create RelPatList
    virtual void rawToRelPat() throw (XRecordError&) = 0;

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

  struct RelPat {
      RelPat(uint32_t pat, double t, double toapp) :
        pattern(pat), time(t), toappear(toapp) {}
      uint32_t pattern;
      double time; //!< [ms]
      double toappear; //!< term between this pattern and the previous [ms]
  };

  typedef std::deque<RelPat> RelPatList;
  RelPatList m_relPatList;
  typedef RelPatList::iterator RelPatListIterator;

  virtual void afterStart() = 0;
  //! push parameters.
  //! use this after clearRaw()
  void writeRaw();
  
  typedef double (*tpulsefunc)(double x);
  tpulsefunc pulseFunc(const std::string &str);
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
  
  unsigned int m_phase_xor;
 private:
    shared_ptr<XBoolNode> m_output;
    shared_ptr<XComboNode> m_combMode; //!< see above definitions in header file
      //! Control period to next pulse sequence
      //! Fix Repetition Time or Fix Rest Time which means time between pulse sequences 
    shared_ptr<XComboNode> m_rtMode;
    shared_ptr<XComboNode> m_numPhaseCycle; //!< How many cycles in phase cycling
    shared_ptr<XDoubleNode> m_rt; //!< Repetition/Rest Time [ms]
    shared_ptr<XDoubleNode> m_tau;  //!< [us]
    shared_ptr<XDoubleNode> m_combPW, m_pw1, m_pw2; //!< PulseWidths [us]
    shared_ptr<XDoubleNode> m_combPT; //!< Comb pulse periodic term [us]
    shared_ptr<XDoubleNode> m_combP1, m_combP1Alt; //!< P1 and P1 alternative
    shared_ptr<XDoubleNode> m_aswSetup; //!< Analog switch setting, setup(proceeding) time before the first spin echo
    shared_ptr<XDoubleNode> m_aswHold;  //!< Analog switch setting, hold time after the last spin echo
    shared_ptr<XDoubleNode> m_altSep; //!< Separation time in DSO record, cause a shift of trigger of DSO in alternatively mode
    shared_ptr<XDoubleNode> m_g2Setup; //!< Setup time of pre-gating port and QPSK
    shared_ptr<XUIntNode> m_combNum; //!< # of comb pulses
    shared_ptr<XUIntNode> m_echoNum; //!< # of Spin echoes (i.e. pi pulses)
    shared_ptr<XBoolNode> m_drivenEquilibrium; //!< polarize spins after pulse sequence or not
    shared_ptr<XDoubleNode> m_combOffRes; //!< off-resonance comb pulses
    shared_ptr<XComboNode> m_combFunc, m_p1Func, m_p2Func; //!< Pulse Modulation
    shared_ptr<XDoubleNode> m_combLevel, m_p1Level, m_p2Level; //!< [dB], Pulse Modulation
    shared_ptr<XDoubleNode> m_masterLevel; //!< [dB]
    shared_ptr<XComboNode> m_aswFilter;
    shared_ptr<XDoubleNode> m_portLevel8;
    shared_ptr<XDoubleNode> m_portLevel9;
    shared_ptr<XDoubleNode> m_portLevel10;
    shared_ptr<XDoubleNode> m_portLevel11;
    shared_ptr<XDoubleNode> m_portLevel12;
    shared_ptr<XDoubleNode> m_portLevel13;
    shared_ptr<XDoubleNode> m_portLevel14; //!< [V]
    shared_ptr<XDoubleNode> m_qamOffset1;
    shared_ptr<XDoubleNode> m_qamOffset2; //!< [%F.S.]
    shared_ptr<XDoubleNode> m_qamLevel1;
    shared_ptr<XDoubleNode> m_qamLevel2;
    shared_ptr<XDoubleNode> m_qamDelay1;
    shared_ptr<XDoubleNode> m_qamDelay2; //!< [us]
    shared_ptr<XDoubleNode> m_difFreq; //!< [MHz]
    shared_ptr<XBoolNode> m_induceEmission; 
    shared_ptr<XDoubleNode> m_induceEmissionPhase; 
    
  shared_ptr<XNode> m_moreConfigShow;
  xqcon_ptr m_conOutput;
  xqcon_ptr m_conCombMode, m_conRTMode;
  xqcon_ptr m_conRT, m_conTau, m_conCombPW, m_conPW1, m_conPW2,
    m_conCombNum, m_conCombPT, m_conCombP1, m_conCombP1Alt,
    m_conASWHold, m_conASWSetup, m_conALTSep, m_conG2Setup,
    m_conEchoNum, m_conDrivenEquilibrium, m_conNumPhaseCycle, m_conCombOffRes,
    m_conCombFunc, m_conP1Func, m_conP2Func,
    m_conCombLevel, m_conP1Level, m_conP2Level,
    m_conMasterLevel,
    m_conASWFilter,
    m_conPortLevel8, m_conPortLevel9, m_conPortLevel10, m_conPortLevel11, m_conPortLevel12, m_conPortLevel13, m_conPortLevel14,
    m_conQAMOffset1, m_conQAMOffset2,
    m_conQAMLevel1, m_conQAMLevel2,
    m_conQAMDelay1, m_conQAMDelay2,
    m_conMoreConfigShow,
    m_conDIFFreq,
    m_conInduceEmission, m_conInduceEmissionPhase;
  shared_ptr<XListener> m_lsnOnPulseChanged;
  shared_ptr<XListener> m_lsnOnMoreConfigShow;
  void onMoreConfigShow(const shared_ptr<XNode> &);

  qshared_ptr<FrmPulser> m_form;
  qshared_ptr<FrmPulserMore> m_formMore;
  
  xqcon_ptr m_conPulserDriver;
    
  void onPulseChanged(const shared_ptr<XValueNodeBase> &);
 };

#endif
