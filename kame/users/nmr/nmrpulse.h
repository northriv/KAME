#ifndef nmrpulseH
#define nmrpulseH
//---------------------------------------------------------------------------
#include <fftw.h>
#include <vector>
#include "secondarydriver.h"
#include "users/dso/dso.h"
#include "pulserdriver.h"
#include <complex>
//---------------------------------------------------------------------------

class FrmNMRPulse;
class XWaveNGraph;
class FrmGraphNURL;

class XNMRPulseAnalyzer : public XSecondaryDriver
{
 XNODE_OBJECT
 protected:
  XNMRPulseAnalyzer(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  virtual ~XNMRPulseAnalyzer();
  
  //! show all forms belonging to driver
  virtual void showForms();
 protected:

  //! this is called when connected driver emit a signal
  //! unless dependency is broken
  //! all connected drivers are readLocked
  virtual void analyze(const shared_ptr<XDriver> &emitter) throw (XRecordError&);
  //! this is called after analyze() or analyzeRaw()
  //! record is readLocked
  virtual void visualize();
  //! check connected drivers have valid time
  //! \return true if dependency is resolved
  virtual bool checkDependency(const shared_ptr<XDriver> &emitter) const;
 
 public:
  //! driver specific part below 
  
  const shared_ptr<XScalarEntry> &entryCosAv() const {return m_entryCosAv;}    ///< Entry storing dc
  const shared_ptr<XScalarEntry> &entrySinAv() const {return m_entrySinAv;}    ///< Entry storing dc

  const shared_ptr<XItemNode<XDriverList, XDSO> > &dso() const {return m_dso;}
  const shared_ptr<XItemNode<XDriverList, XPulser> > &pulser() const {return m_pulser;}

  void acquire();

  //! Origin from trigger [ms]
  const shared_ptr<XDoubleNode> &fromTrig() const {return m_fromTrig;}
  //! length of data points [ms]
  const shared_ptr<XDoubleNode> &width() const {return m_width;}

  const shared_ptr<XDoubleNode> &phaseAdv() const {return m_phaseAdv;}   ///< [deg]
  /// Dynamic Noise Reduction
  const shared_ptr<XBoolNode> &useDNR() const {return m_useDNR;}
  /// Position from trigger, for background subtraction or DNR [ms]
  const shared_ptr<XDoubleNode> &bgPos() const {return m_bgPos;}
  /// length for background subtraction or DNR [ms]  
  const shared_ptr<XDoubleNode> &bgWidth() const {return m_bgWidth;}
  /// Phase 0 deg. position of FT component from trigger [ms]
  const shared_ptr<XDoubleNode> &fftPos() const {return m_fftPos;}
  /// If exceeding Width, do zerofilling
  const shared_ptr<XUIntNode> &fftLen() const {return m_fftLen;}
  /// FFT Window Function
  const shared_ptr<XComboNode> &windowFunc() const {return m_windowFunc;}
  /// Set Digital IF frequency
  const shared_ptr<XDoubleNode> &difFreq() const {return m_difFreq;}

  /// Extra Average with infinite steps
  const shared_ptr<XBoolNode> &exAvgIncr() const {return m_exAvgIncr;}
  /// Extra Average Steps
  const shared_ptr<XUIntNode> &extraAvg() const {return m_extraAvg;}

  /// # of echoes
  const shared_ptr<XUIntNode> &numEcho() const {return m_numEcho;}
  /// If NumEcho > 1, need periodic term of echoes [ms]
  const shared_ptr<XDoubleNode> &echoPeriod() const {return m_echoPeriod;}

  //! records below.

  /// FFT Wave without extra avg
  const std::deque<std::complex<double> > &ftWave() const {return m_ftWave;}
  /// Wave without extra avg
  const std::deque<std::complex<double> > &wave() const {return m_wave;}
  //! freq. resolution [Hz]
  double dFreq() const {return m_dFreq;}
  //! [V^2] noise factor deduced from background
  //! \sa UseDNR, BGPos, BGWidth
  double noisePower() const {return m_noisePower;}
  //! time resolution [sec.]
  double interval() const {return m_interval;}
  //! time diff. of the first point from trigger [sec.]
  double startTime() const {return m_startTime;}
 private:
  /// Stored Wave for display with extra avg
  const shared_ptr<XWaveNGraph> &waveGraph() const {return m_waveGraph;}
  /// Stored FFT Wave for display with extra avg
  const shared_ptr<XWaveNGraph> &ftWaveGraph() const {return m_ftWaveGraph;}

  shared_ptr<XScalarEntry> m_entryCosAv;    ///< Entry storing dc
  shared_ptr<XScalarEntry> m_entrySinAv;    ///< Entry storing dc

  shared_ptr<XItemNode<XDriverList, XDSO> > m_dso;
 
  shared_ptr<XDoubleNode> m_fromTrig;
  shared_ptr<XDoubleNode> m_width;

  shared_ptr<XDoubleNode> m_phaseAdv;   ///< [deg]
  shared_ptr<XBoolNode> m_useDNR;
  shared_ptr<XDoubleNode> m_bgPos;
  shared_ptr<XDoubleNode> m_bgWidth;
  shared_ptr<XDoubleNode> m_fftPos;
  shared_ptr<XUIntNode> m_fftLen;
  shared_ptr<XComboNode> m_windowFunc;
  shared_ptr<XDoubleNode> m_difFreq;

  shared_ptr<XBoolNode> m_exAvgIncr;
  shared_ptr<XUIntNode> m_extraAvg;

  shared_ptr<XUIntNode> m_numEcho;
  shared_ptr<XDoubleNode> m_echoPeriod;

  shared_ptr<XNode> m_fftShow;
  shared_ptr<XNode> m_avgClear;

	//! Echo Phase Cycling
  shared_ptr<XBoolNode> m_epcEnabled;
  shared_ptr<XBoolNode> m_epc4x;
  shared_ptr<XItemNode<XDriverList, XPulser> > m_pulser;
  unsigned int m_epccnt;
  
  //! Records
  //! these are without avg.
  std::deque<std::complex<double> > m_ftWave;
  std::deque<std::complex<double> > m_wave;
  double m_dFreq;  ///< Hz per point
  double m_noisePower;
  int m_avcount;
  /// Stored FFT Wave with avg.
  std::deque<std::complex<double> > m_ftWaveSum;
  /// Stored Waves for avg.
  std::deque<std::complex<double> > m_waveSum;
  /// Stored Waves for moving avg.
  std::deque<std::deque<std::complex<double> > > m_waveAv;
  /// Stored Waves for avg.
  std::deque<std::complex<double> > m_rawWaveSum;
  //! time resolution
  double m_interval;
  //! time diff. of the first point from trigger
  double m_startTime;
  
  xqcon_ptr m_conFromTrig, m_conWidth, m_conPhaseAdv, m_conBGPos, m_conUseDNR,
    m_conBGWidth, m_conFFTPos, m_conFFTLen, m_conExtraAv;
  xqcon_ptr m_conExAvgIncr;
  xqcon_ptr m_conAvgClear, m_conFFTShow, m_conWindowFunc, m_conDIFFreq;
  xqcon_ptr m_conNumEcho, m_conEchoPeriod;
  xqcon_ptr m_conDSO;
  xqcon_ptr m_conEPC4x, m_conPulser, m_conEPCEnabled;

  shared_ptr<XListener> m_lsnOnFFTShow, m_lsnOnAvgClear;
  shared_ptr<XListener> m_lsnOnCondChanged;
    
  qshared_ptr<FrmNMRPulse> m_form;
  shared_ptr<XStatusPrinter> m_statusPrinter;
  qshared_ptr<FrmGraphNURL> m_fftForm;

  shared_ptr<XWaveNGraph> m_waveGraph;
  shared_ptr<XWaveNGraph> m_ftWaveGraph;
  
  void onCondChanged(const shared_ptr<XValueNodeBase> &);
  void onFFTShow(const shared_ptr<XNode> &);
  void onAvgClear(const shared_ptr<XNode> &);
  
  //for Window Func.
  typedef double (*twindowfunc)(double x);
  static double windowFuncRect(double x);
  static double windowFuncHanning(double x);
  static double windowFuncHamming(double x);
  static double windowFuncFlatTop(double x);
  static double windowFuncBlackman(double x);
  static double windowFuncBlackmanHarris(double x);
  static double windowFuncKaiser(double x, double alpha);
  static double windowFuncKaiser1(double x);
  static double windowFuncKaiser2(double x);
  static double windowFuncKaiser3(double x);
  
  //for DNR
  int m_dnrsubfftlen, m_dnrpulsefftlen;
  std::vector<fftw_complex> m_dnrsubfftin, m_dnrsubfftout;
  fftw_plan m_dnrsubfftplan;    
  std::vector<fftw_complex> m_dnrpulsefftin, m_dnrpulsefftout;
  fftw_plan m_dnrpulsefftplan;
  void backgroundSub(const std::deque<std::complex<double> > &wave,
     int length, int bgpos, int bglength, twindowfunc windowfunc, double phase_shift);
  
  //for FFT
  int m_fftlen;
  std::vector<fftw_complex> m_fftin, m_fftout;
  fftw_plan m_fftplan;
  void rotNFFT(int ftpos, double ph, 
    std::deque<std::complex<double> > &wave, std::deque<std::complex<double> > &ftwave,
    twindowfunc windowfunc, int diffreq);
    
  XTime m_timeClearRequested;
};



#endif
