#ifndef CHARDEVICEDRIVER_H_
#define CHARDEVICEDRIVER_H_

#include "driver.h"
#include "interface.h"
class XCharInterface;

template<class tDriver>
class XCharDeviceDriver : public tDriver
{
 XNODE_OBJECT
 protected:
  XCharDeviceDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  virtual ~XCharDeviceDriver() {}
 protected:
  const shared_ptr<XCharInterface> &interface() const {return m_interface;}
  //! open all interfaces.
  virtual void openInterfaces() throw (XInterface::XInterfaceError &);
  //! close all interfaces.
  virtual void closeInterfaces();
 private:
  shared_ptr<XCharInterface> m_interface;
};

template<class tDriver>
XCharDeviceDriver<tDriver>::XCharDeviceDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    tDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_interface(XNode::create<XCharInterface>("Interface", false,
            dynamic_pointer_cast<XDriver>(tDriver::shared_from_this())))
{
    interfaces->insert(m_interface);
}
template<class tDriver>
void
XCharDeviceDriver<tDriver>::openInterfaces() throw (XInterface::XInterfaceError &)
{
	interface()->open();
}
template<class tDriver>
void
XCharDeviceDriver<tDriver>::closeInterfaces()
{
	interface()->close();
}

#endif /*CHARDEVICEDRIVER_H_*/
