//---------------------------------------------------------------------------

#ifndef testdriverH
#define testdriverH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "dummydriver.h"

class XScalarEntry;

class XTestDriver : public XDummyDriver<XPrimaryDriver>
{
 XNODE_OBJECT
 protected:
  XTestDriver(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  //! usually nothing to do
  virtual ~XTestDriver() {}
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
  shared_ptr<XThread<XTestDriver> > m_thread;
  double m_x,m_y;
  const shared_ptr<XScalarEntry> m_entryX, m_entryY;
  void *execute(const atomic<bool> &);
  
};

//---------------------------------------------------------------------------
#endif
