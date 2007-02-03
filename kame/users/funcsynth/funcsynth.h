#ifndef funcsynthH
#define funcsynthH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "xnodeconnector.h"

class FrmFuncSynth;

class XFuncSynth : public XPrimaryDriver
{
 XNODE_OBJECT
 protected:
  XFuncSynth(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  //! usually nothing to do
  virtual ~XFuncSynth() {}
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
  
  //! driver specific part below
  const shared_ptr<XBoolNode> &output() const {return m_output;}
  const shared_ptr<XNode> &trig() const {return m_trig;} //!< trigger to burst
  const shared_ptr<XComboNode> &mode() const {return m_mode;}
  const shared_ptr<XComboNode> &function() const {return m_function;}
  const shared_ptr<XDoubleNode> &freq() const {return m_freq;} //!< [Hz]
  const shared_ptr<XDoubleNode> &amp() const {return m_amp;} //!< [Vp-p]
  const shared_ptr<XDoubleNode> &phase() const {return m_phase;} //!< [deg.]
  const shared_ptr<XDoubleNode> &offset() const {return m_offset;} //!< [V]
 protected:
  virtual void onOutputChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onTrigTouched(const shared_ptr<XNode> &) = 0;
  virtual void onModeChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onFunctionChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onFreqChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onAmpChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onPhaseChanged(const shared_ptr<XValueNodeBase> &) = 0;
  virtual void onOffsetChanged(const shared_ptr<XValueNodeBase> &) = 0;
 private:
  shared_ptr<XBoolNode>  m_output;
  shared_ptr<XNode>  m_trig;
  shared_ptr<XComboNode>  m_mode;
  shared_ptr<XComboNode>  m_function;
  shared_ptr<XDoubleNode>  m_freq;
  shared_ptr<XDoubleNode>  m_amp;
  shared_ptr<XDoubleNode>  m_phase;
  shared_ptr<XDoubleNode>  m_offset;
  shared_ptr<XListener> m_lsnOutput, m_lsnMode, m_lsnFunction,
         m_lsnFreq, m_lsnAmp, m_lsnPhase, m_lsnOffset;
  shared_ptr<XListener> m_lsnTrig;
  xqcon_ptr m_conOutput, m_conTrig, m_conMode, m_conFunction,
         m_conFreq, m_conAmp, m_conPhase, m_conOffset;

  qshared_ptr<FrmFuncSynth> m_form;
};

#endif
