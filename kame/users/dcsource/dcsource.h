//---------------------------------------------------------------------------

#ifndef dcsourceH
#define dcsourceH

#include "primarydriver.h"
#include "xnodeconnector.h"

class FrmDCSource;

class XDCSource : public XPrimaryDriver
{
 XNODE_OBJECT
 protected:
  XDCSource(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  //! usually nothing to do
  virtual ~XDCSource() {}
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
  const shared_ptr<XBoolNode> &output() const {return m_output;}
  const shared_ptr<XDoubleNode> &value() const {return m_value;}
    
 protected:
  virtual void changeFunction(int x) = 0;
  virtual void changeOutput(bool x) = 0;
  virtual void changeValue(double x) = 0;
 private:
 
  xqcon_ptr m_conFunction, m_conOutput, m_conValue;
  shared_ptr<XComboNode> m_function;
  shared_ptr<XBoolNode> m_output;
  shared_ptr<XDoubleNode> m_value;
  shared_ptr<XListener> m_lsnFunction, m_lsnOutput, m_lsnValue;
  
  virtual void onFunctionChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onOutputChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onValueChanged(const shared_ptr<XValueNodeBase> &);
  
  qshared_ptr<FrmDCSource> m_form;
};


//YOKOGAWA 7551 DC V/DC A source
class XYK7651:public XDCSource
{
 public:
  XYK7651(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 protected:
  virtual void changeFunction(int x);
  virtual void changeOutput(bool x);
  virtual void changeValue(double x);
};

#endif
 
