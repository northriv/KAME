//---------------------------------------------------------------------------
#include "testdriver.h"
#include "analyzer.h"
#include "xnodeconnector.h"
#include <qstatusbar.h>

XTestDriver::XTestDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    XPrimaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
    m_entryX(create<XScalarEntry>("X", false, 
        dynamic_pointer_cast<XDriver>(shared_from_this()), "%.3g")),
    m_entryY(create<XScalarEntry>("Y", false,
        dynamic_pointer_cast<XDriver>(shared_from_this()), "%+.4f[K]"))
{
  scalarentries->insert(m_entryX);
  scalarentries->insert(m_entryY);
}

void
XTestDriver::showForms() {
//! impliment form->show() here
}
void
XTestDriver::start()
{
    m_thread.reset(new XThread<XTestDriver>(shared_from_this(), &XTestDriver::execute));
    m_thread->resume();
}
void
XTestDriver::stop()
{
    if(m_thread) m_thread->terminate();
//    m_thread->waitFor();
}
void
XTestDriver::analyzeRaw() throw (XRecordError&)
{
    //! Since raw buffer is Fast-in Fast-out, use the same sequence of push()es for pop()s
    m_x = pop<double>();
    m_y = pop<double>();
    m_entryX->value(m_x);
    m_entryY->value(m_y);
}
void
XTestDriver::visualize()
{
 //! impliment extra codes which do not need write-lock of record
 //! record is read-locked
}

void *
XTestDriver::execute(const atomic<bool> &terminated)
{
  while(!terminated)
    {
      msecsleep(10);
      double x = (double)KAME::rand() / RAND_MAX - 0.2;
      double y = (double)KAME::rand() / RAND_MAX - 0.2;
      clearRaw();
      push(x);
      push(y);
      finishWritingRaw(XTime::now(), XTime::now());
    }
  return NULL;
}

