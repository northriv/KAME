#ifndef DUMMYDRIVER_H_
#define DUMMYDRIVER_H_

#include "driver.h"
#include "interface.h"

class XDummyInterface : public XInterface
{
 XNODE_OBJECT
 XDummyInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
 : XInterface(name, runtime, driver), m_bOpened(false)
  {}
public:
 virtual ~XDummyInterface() {}

  virtual void open() throw (XInterfaceError &) {m_bOpened = true;}
  //! This can be called even if has already closed.
  virtual void close() {m_bOpened = false;}

  virtual bool isOpened() const {return m_bOpened;}
private:
	bool m_bOpened;
};
template<class tDriver>
class XDummyDriver : public tDriver
{
 XNODE_OBJECT
 protected:
  XDummyDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  virtual ~XDummyDriver() {}
 protected:
  const shared_ptr<XDummyInterface> &interface() const {return m_interface;}
  //! open all interfaces.
  virtual void openInterfaces() throw (XInterface::XInterfaceError &);
  //! close all interfaces.
  virtual void closeInterfaces();
 private:
  shared_ptr<XDummyInterface> m_interface;
};

template<class tDriver>
XDummyDriver<tDriver>::XDummyDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    tDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_interface(XNode::create<XDummyInterface>("Interface", false,
            dynamic_pointer_cast<XDriver>(tDriver::shared_from_this())))
{
    interfaces->insert(m_interface);
}
template<class tDriver>
void
XDummyDriver<tDriver>::openInterfaces() throw (XInterface::XInterfaceError &)
{
	interface()->open();
}
template<class tDriver>
void
XDummyDriver<tDriver>::closeInterfaces()
{
	interface()->close();
}


#endif /*DUMMYDRIVER_H_*/
