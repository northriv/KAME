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
  //! Be called just after opening interface. Call start() inside this routine appropriately.
  virtual void open() throw (XInterface::XInterfaceError &) {this->start();}
  //! Be called during stopping driver. Call interface()->stop() inside this routine.
  virtual void close() throw (XInterface::XInterfaceError &) {interface()->stop();}
  void onOpen(const shared_ptr<XInterface> &);
  void onClose(const shared_ptr<XInterface> &);
  virtual void afterStop() {close();}
 private:
  shared_ptr<XListener> m_lsnOnOpen, m_lsnOnClose;
  
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
            dynamic_pointer_cast<XDriver>(this->shared_from_this())))
{
    interfaces->insert(m_interface);
    m_lsnOnOpen = interface()->onOpen().connectWeak(false,
    	 this->shared_from_this(), &XCharDeviceDriver<tDriver>::onOpen);
    m_lsnOnClose = interface()->onClose().connectWeak(false, 
    	this->shared_from_this(), &XCharDeviceDriver<tDriver>::onClose);
}
template<class tDriver>
void
XCharDeviceDriver<tDriver>::onOpen(const shared_ptr<XInterface> &)
{
	try {
		open();
	}
	catch (XInterface::XInterfaceError& e) {
		e.print(this->getLabel() + KAME::i18n(": Opening interface failed, because "));
		close();
	}
}
template<class tDriver>
void
XCharDeviceDriver<tDriver>::onClose(const shared_ptr<XInterface> &)
{
	try {
		this->stop();
	}
	catch (XInterface::XInterfaceError& e) {
		e.print(this->getLabel() + KAME::i18n(": Stopping driver failed, because "));
	}
}
#endif /*CHARDEVICEDRIVER_H_*/
