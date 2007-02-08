#ifndef NIDAQMXDRIVER_H_
#define NIDAQMXDRIVER_H_

#ifdef HAVE_CONFIG_H
 #include <config.h>
#endif /*HAVE_CONFIG_H*/

#ifdef HAVE_NI_DAQMX

#include "interface.h"
#include "driver.h"
#include <NIDAQmx.h>

class XNIDAQmxTask;

class XNIDAQmxInterface : public XInterface
{
 XNODE_OBJECT
protected:
 XNIDAQmxInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver);
public:
 virtual ~XNIDAQmxInterface() {}
 
  static QString getNIDAQmxErrMessage();
  static QString getNIDAQmxErrMessage(int status);
  static int checkDAQmxError(int ret, const char*file, int line);

  virtual bool isOpened() const {return m_devname.length();}
  
  const char*devName() const {return m_devname.c_str();}
  
  static void parseList(const char *list, std::deque<std::string> &buf);
  
  struct XNIDAQmxRoute {
  	XNIDAQmxRoute(const char*src, const char*dst);
  	~XNIDAQmxRoute();
  };
protected:
  virtual void open() throw (XInterfaceError &);
  //! This can be called even if has already closed.
  virtual void close() throw (XInterfaceError &);
private:
	std::string m_devname;
};

#define CHECK_DAQMX_ERROR(ret) XNIDAQmxInterface::checkDAQmxError(ret, __FILE__, __LINE__)

//#define CHECK_DAQMX_RET(ret, msg) {dbgPrint(# ret);\
//	if(CHECK_DAQMX_ERROR(ret, msg) > 0) {gWarnPrint(QString(msg) + " " + XNIDAQmxInterface::getNIDAQmxErrMessage()); } }
#define CHECK_DAQMX_RET(ret) {\
	if(CHECK_DAQMX_ERROR(ret) > 0) {gWarnPrint(XNIDAQmxInterface::getNIDAQmxErrMessage()); } }

template<class tDriver>
class XNIDAQmxDriver : public tDriver
{
 XNODE_OBJECT
 protected:
  XNIDAQmxDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
 public:
  virtual ~XNIDAQmxDriver() {}
 protected:
  const shared_ptr<XNIDAQmxInterface> &interface() const {return m_interface;}
  //! Be called just after opening interface. Call start() inside this routine appropriately.
  virtual void open() throw (XInterface::XInterfaceError &) {this->start();}
  //! Be called during stopping driver. Call interface()->stop() inside this routine.
  virtual void close() throw (XInterface::XInterfaceError &) {interface()->stop();}
  void onOpen(const shared_ptr<XInterface> &);
  void onClose(const shared_ptr<XInterface> &);
  //! This should not cause an exception.
  virtual void afterStop() {close();}
 private:
  shared_ptr<XListener> m_lsnOnOpen, m_lsnOnClose;
  
  shared_ptr<XNIDAQmxInterface> m_interface;
};
template<class tDriver>
XNIDAQmxDriver<tDriver>::XNIDAQmxDriver(const char *name, bool runtime, 
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
    tDriver(name, runtime, scalarentries, interfaces, thermometers, drivers),
	m_interface(XNode::create<XNIDAQmxInterface>("Interface", false,
            dynamic_pointer_cast<XDriver>(this->shared_from_this())))
{
    interfaces->insert(m_interface);
    m_lsnOnOpen = interface()->onOpen().connectWeak(false,
    	 this->shared_from_this(), &XNIDAQmxDriver<tDriver>::onOpen);
    m_lsnOnClose = interface()->onClose().connectWeak(false, 
    	this->shared_from_this(), &XNIDAQmxDriver<tDriver>::onClose);
}
template<class tDriver>
void
XNIDAQmxDriver<tDriver>::onOpen(const shared_ptr<XInterface> &)
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
XNIDAQmxDriver<tDriver>::onClose(const shared_ptr<XInterface> &)
{
	try {
		this->stop();
	}
	catch (XInterface::XInterfaceError& e) {
		e.print(this->getLabel() + KAME::i18n(": Stopping driver failed, because "));
	}
}
#endif //HAVE_NI_DAQMX

#endif /*NIDAQMXDRIVER_H_*/
