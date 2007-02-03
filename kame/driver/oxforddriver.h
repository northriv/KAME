#ifndef OXFORDDRIVER_H_
#define OXFORDDRIVER_H_

#include "charinterface.h"
#include "chardevicedriver.h"
#include "primarydriver.h"

class XOxfordInterface : public XCharInterface
{
 XNODE_OBJECT
protected:
  XOxfordInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
 public:
  virtual void open() throw (XInterfaceError &);
  virtual void close() throw (XInterfaceError &);
  
  virtual void send(const char *str) throw (XCommError &);
  //! don't use me
  virtual void write(const char *, int) throw (XCommError &) {
    ASSERT(false);
  }
  virtual void receive() throw (XCommError &);
  virtual void receive(int length) throw (XCommError &);
  virtual void query(const char *str) throw (XCommError &);
};

template <class tDriver>
class XOxfordDriver : public XCharDeviceDriver<tDriver>
{
 XNODE_OBJECT
protected:
  XOxfordDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers)
   : XCharDeviceDriver<tDriver>(name, runtime, scalarentries, interfaces, thermometers, drivers) {}
  double read(int arg) throw (XInterface::XInterfaceError &);
 protected:
 private:
};

template<class tDriver>
double
XOxfordDriver<tDriver>::read(int arg) throw (XInterface::XInterfaceError &)
{
  double x;
  this->interface()->queryf("R%d", arg);
  int ret = this->interface()->scanf("R%lf", &x);
  if(ret != 1) throw XInterface::XConvError(__FILE__, __LINE__);
  return x;
}

#endif /*OXFORDDRIVER_H_*/
