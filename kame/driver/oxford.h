#ifndef OXFORD_H_
#define OXFORD_H_

#include "interface.h"
#include "primarydriver.h"

class XOxfordInterface : public XInterface
{
 XNODE_OBJECT
protected:
  XOxfordInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
 public:
  virtual void open() throw (XInterfaceError &);
  virtual void close();
  
  virtual void send(const char *str) throw (XCommError &);
  //! don't use me
  virtual void write(const char *, int) throw (XCommError &) {
    ASSERT(false);
  }
  virtual void receive() throw (XCommError &);
  virtual void receive(int length) throw (XCommError &);
  virtual void query(const char *str) throw (XCommError &);
};

template <class tPrimaryDriver>
class XOxfordDriver : public tPrimaryDriver
{
 XNODE_OBJECT
protected:
  XOxfordDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers)
   : tPrimaryDriver(name, runtime, scalarentries, interfaces, thermometers, drivers)
   {
        tPrimaryDriver::replaceInterface(
            XNode::create<XOxfordInterface>(
                tPrimaryDriver::interface()->getName().c_str(),
                tPrimaryDriver::interface()->isRunTime(),
                tPrimaryDriver::interface()->driver()), interfaces);
   }
  double read(int arg) throw (XInterface::XInterfaceError &);
};
template <class tPrimaryDriver>
double
XOxfordDriver<tPrimaryDriver>::read(int arg) throw (XInterface::XInterfaceError &)
{
  double x;
  tPrimaryDriver::interface()->queryf("R%d", arg);
  int ret = tPrimaryDriver::interface()->scanf("R%lf", &x);
  if(ret != 1) throw XInterface::XConvError(__FILE__, __LINE__);
  return x;
}

#endif /*OXFORD_H_*/
