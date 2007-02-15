#ifndef nmrrelaxH
#define nmrrelaxH

#include "secondarydriver.h"
#include "xnodeconnector.h"
//#include "pulserdriver.h"
//#include "nmrpulse.h"
//#include "nmrrelaxfit.h"
#include <complex>

class XNMRPulseAnalyzer;
class XPulser;
class FrmNMRT1;
class XWaveNGraph;
class XRelaxFunc;
class XRelaxFuncList;
class XRelaxFuncPlot;
class XScalarEntry;

//! Measure Relaxation Curve
class XNMRT1 : public XSecondaryDriver
{
 XNODE_OBJECT
 protected:
  XNMRT1(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  ~XNMRT1 () {}
  
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
  //! Holds 1/T1 or 1/T2 and its std. deviation
  const shared_ptr<XScalarEntry> &t1inv() const {return m_t1inv;}
  const shared_ptr<XScalarEntry> &t1invErr() const {return m_t1invErr;}

  const shared_ptr<XItemNode < XDriverList, XPulser > > &pulser() const {return m_pulser;}
  const shared_ptr<XItemNode < XDriverList, XNMRPulseAnalyzer > > &pulse1() const {return m_pulse1;}
  const shared_ptr<XItemNode < XDriverList, XNMRPulseAnalyzer > > &pulse2() const {return m_pulse2;}

  //! If active, a control to Pulser is allowed
  const shared_ptr<XBoolNode> &active() const {return m_active;}
  //! Deduce phase from data
  const shared_ptr<XBoolNode> &autoPhase() const {return m_autoPhase;}
  //! Use absolute value, ignoring phase
  const shared_ptr<XBoolNode> &absFit() const {return m_absFit;}
  //! Region of P1 or 2tau for fitting, display, control of pulser [ms]
  const shared_ptr<XDoubleNode> &p1Min() const {return m_p1Min;}
  const shared_ptr<XDoubleNode> &p1Max() const {return m_p1Max;}
  //! (Deduced) phase of echoes [deg.]
  const shared_ptr<XDoubleNode> &phase() const {return m_phase;}
  //! Center freq and band width of echoes [kHz].
  const shared_ptr<XDoubleNode> &freq() const {return m_freq;}
  const shared_ptr<XDoubleNode> &bandWidth() const {return m_bandWidth;}
  //! Do T2 measurement
  const shared_ptr<XBoolNode> &t2Mode() const {return m_t2Mode;}
  //! # of Samples for fitting and display
  const shared_ptr<XUIntNode> &smoothSamples() const {return m_smoothSamples;}
  //! Distribution of P1 or 2tau
  const shared_ptr<XComboNode> &p1Dist() const {return m_p1Dist;}
  //! Relaxation Function
  const shared_ptr<XItemNode < XRelaxFuncList, XRelaxFunc > >  &relaxFunc() const {return m_relaxFunc;}

 private:
  //! List of relaxation functions
  shared_ptr<XRelaxFuncList> m_relaxFuncs;
  
 friend class XRelaxFunc;
 friend class XRelaxFuncPlot;
 
  //! Holds 1/T1 or 1/T2 and its std. deviation
  shared_ptr<XScalarEntry> m_t1inv;
  shared_ptr<XScalarEntry> m_t1invErr;

  shared_ptr<XItemNode < XDriverList, XPulser > > m_pulser;
  shared_ptr<XItemNode < XDriverList, XNMRPulseAnalyzer > > m_pulse1;
  shared_ptr<XItemNode < XDriverList, XNMRPulseAnalyzer > > m_pulse2;

  shared_ptr<XBoolNode> m_active;
  shared_ptr<XBoolNode> m_autoPhase;
  shared_ptr<XBoolNode> m_absFit;
  shared_ptr<XDoubleNode> m_p1Min;
  shared_ptr<XDoubleNode> m_p1Max;
  shared_ptr<XDoubleNode> m_phase;
  shared_ptr<XDoubleNode> m_freq;
  shared_ptr<XDoubleNode> m_bandWidth;
  shared_ptr<XBoolNode> m_t2Mode;
  shared_ptr<XUIntNode> m_smoothSamples;
  shared_ptr<XComboNode> m_p1Dist;
  shared_ptr<XItemNode < XRelaxFuncList, XRelaxFunc > >  m_relaxFunc;
  shared_ptr<XNode> m_resetFit, m_clearAll;
  shared_ptr<XStringNode> m_fitStatus;

  //! For fitting and display
  struct Pt
  {
    double var; /// auto-phase- or absolute value
    std::complex<double> c;
    double p1;
    double isigma; /// weight
  };

  //! for Non-Lenear-Least-Square fitting
  struct NLLS
  {
    std::deque< Pt > *pts; //pointer to data
    shared_ptr<XRelaxFunc> func; //pointer to the current relaxation function
  };
 
  shared_ptr<XListener> m_lsnOnClearAll, m_lsnOnResetFit;
  shared_ptr<XListener> m_lsnOnActiveChanged;
  shared_ptr<XListener> m_lsnOnCondChanged;
  void onClearAll (const shared_ptr<XNode> &);
  void onResetFit (const shared_ptr<XNode> &);
  void onActiveChanged (const shared_ptr<XValueNodeBase> &);
  void onCondChanged (const shared_ptr<XValueNodeBase> &);
  xqcon_ptr m_conP1Min, m_conP1Max, m_conPhase, m_conFreq, m_conBW,
    m_conSmoothSamples, m_conASWClearance;
  xqcon_ptr m_conFitStatus;
  xqcon_ptr m_conP1Dist, m_conRelaxFunc;
  xqcon_ptr m_conClearAll, m_conResetFit;
  xqcon_ptr m_conActive, m_conAutoPhase, m_conAbsFit;
  xqcon_ptr m_conT2Mode;
  xqcon_ptr m_conPulser, m_conPulse1, m_conPulse2;

  //! Raw measured points
  struct RawPt
  {
    std::complex<double> c;
    double p1;
    double isigma2; //! weight^2, probably deduced from noise in background
  };
  //! Store all measured points
  std::deque< RawPt > m_pts;
  //! Store reduced points to manage fitting and display
  std::deque< Pt > m_sumpts;

  qshared_ptr<FrmNMRT1> m_form;
  shared_ptr<XStatusPrinter> m_statusPrinter;
  //! Store reduced points
  //! \sa m_pt, m_sumpts
  shared_ptr<XWaveNGraph> m_wave;

  double m_params[3]; //!< fitting parameters; 1/T1, c, a; ex. f(t) = c*exp(-t/T1) + a
  double m_errors[3]; //!< std. deviations
  
  //! Do fitting iterations \a itercnt times
  //! \param relax a pointer to a realaxation function
  //! \param itercnt counts 
  //! \param buf a message will be passed
  std::string iterate(shared_ptr<XRelaxFunc> &relax, int itercnt);

  std::complex<double> acuSpectrum (
    const std::deque< std::complex<double> >&wave, double df, double cf,
		      double bw);
 		      
  XTime m_timeClearRequested;
};

//---------------------------------------------------------------------------
#endif
