#ifndef OXFORDDRIVER_H_
#define OXFORDDRIVER_H_

#include "charinterface.h"
#include "primarydriver.h"

class XOxfordInterface : public XCharInterface
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

template <class tDriver>
class XOxfordDriver : public tDriver
{
 XNODE_OBJECT
protected:
  XOxfordDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
  double read(int arg) throw (XInterface::XInterfaceError &);
 protected:
  //! open all interfaces.
  virtual void openInterfaces() throw (XInterface::XInterfaceError &);
  //! close all interfaces.
  virtual void closeInterfaces();
  const shared_ptr<XCharInterface> &interface() const {return m_interface;}  
 private:
  shared_ptr<XCharInterface> m_interface;
};



template<class tDriver>
XOxfordDriver<tDriver>::XOxfordDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers)
   : tDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_interface(XNode::create<XOxfordInterface>("Interface", false,
            dynamic_pointer_cast<XDriver>(tDriver::shared_from_this())))
{
    interfaces->insert(m_interface);
}

template<class tDriver>
void
XOxfordDriver<tDriver>::openInterfaces() throw (XInterface::XInterfaceError &)
{
	interface()->open();
}
template<class tDriver>
void
XOxfordDriver<tDriver>::closeInterfaces()
{
	interface()->close();
}

template<class tDriver>
double
XOxfordDriver<tDriver>::read(int arg) throw (XInterface::XInterfaceError &)
{
  double x;
  interface()->queryf("R%d", arg);
  int ret = interface()->scanf("R%lf", &x);
  if(ret != 1) throw XInterface::XConvError(__FILE__, __LINE__);
  return x;
}

#endif /*OXFORDDRIVER_H_*/
