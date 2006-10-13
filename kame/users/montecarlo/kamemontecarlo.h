#ifndef KAMEMONTECARLO_H_
#define KAMEMONTECARLO_H_

#include "primarydriver.h"
#include <fftw.h>

class XScalarEntry;
class MonteCarlo;
class FrmMonteCarlo;
class XWaveNGraph;

class XMonteCarloDriver : public XPrimaryDriver
{
 XNODE_OBJECT
 protected:
  XMonteCarloDriver(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  //! usually nothing to do
  virtual ~XMonteCarloDriver() {}
  //! show all forms belonging to driver
  virtual void showForms();
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
 private:
  shared_ptr<XDoubleNode> m_targetTemp;
  shared_ptr<XDoubleNode> m_targetField;
  shared_ptr<XDoubleNode> m_hdirx;
  shared_ptr<XDoubleNode> m_hdiry;
  shared_ptr<XDoubleNode> m_hdirz;
  shared_ptr<XUIntNode> m_L;
  shared_ptr<XDoubleNode> m_cutoffReal;
  shared_ptr<XDoubleNode> m_cutoffRec;
  shared_ptr<XDoubleNode> m_alpha;
  shared_ptr<XDoubleNode> m_minTests;
  shared_ptr<XDoubleNode> m_minFlips;
  shared_ptr<XNode> m_step;
  shared_ptr<XComboNode> m_graph3D;
  shared_ptr<XScalarEntry> m_entryT, m_entryH,
    m_entryU, m_entryC, m_entryCoT,
    m_entryS, m_entryM, m_entry2in2, m_entry1in3;
  shared_ptr<MonteCarlo> m_loop, m_store;
  
  xqcon_ptr m_conLength, m_conCutoffReal, m_conCutoffRec, m_conAlpha,
    m_conTargetTemp, m_conTargetField,
    m_conHDirX, m_conHDirY, m_conHDirZ, m_conMinTests, m_conMinFlips, m_conStep,
    m_conGraph3D;
  qshared_ptr<FrmMonteCarlo> m_form;
  shared_ptr<XWaveNGraph> m_wave3D;
  long double m_sumDU, m_sumDS, m_sumDUav;
  long double m_testsTotal;
  double m_flippedTotal;
  double m_dU;
  double m_DUav, m_Mav;
  double m_lastTemp;
  //! along field direction.
  double m_lastField, m_lastMagnetization;
  void execute(int flips, long double tests);
  void onTargetChanged(const shared_ptr<XValueNodeBase> &);
  void onGraphChanged(const shared_ptr<XValueNodeBase> &);
  void onStepTouched(const shared_ptr<XNode> &);
  shared_ptr<XListener> m_lsnTargetChanged, m_lsnStepTouched, m_lsnGraphChanged;
  shared_ptr<XStatusPrinter> m_statusPrinter;
  int m_fftlen;
  std::vector<fftw_complex> m_fftin[3], m_fftout[3];
  fftwnd_plan m_fftplan;  
};

#endif /*KAMEMONTECARLO_H_*/
