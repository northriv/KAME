#ifndef dmmH
#define dmmH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "xnodeconnector.h"

class XScalarEntry;
class FrmDMM;

class XDMM : public XPrimaryDriver
{
 XNODE_OBJECT
 protected:
  XDMM(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  //! usually nothing to do
  virtual ~XDMM() {}
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
  const shared_ptr<XComboNode> &function() const {return m_function;}
  const shared_ptr<XUIntNode> &waitInms() const {return m_waitInms;}
 protected:
  //! one-shot reading
  virtual double oneShotRead() = 0; 
  //! called when m_function is changed
  virtual void changeFunction() = 0;
  
  //! This should not cause an exception.
  virtual void afterStop() = 0;
 private:
  //! called when m_function is changed
  void onFunctionChanged(const shared_ptr<XValueNodeBase> &node);
  
  shared_ptr<XScalarEntry> m_entry;
  shared_ptr<XComboNode> m_function;
  shared_ptr<XUIntNode> m_waitInms;
  shared_ptr<XListener> m_lsnOnFunctionChanged;
  xqcon_ptr m_conFunction, m_conWaitInms;
 
  shared_ptr<XThread<XDMM> > m_thread;
  qshared_ptr<FrmDMM> m_form;
  
  void *execute(const atomic<bool> &);
  
};

#endif
