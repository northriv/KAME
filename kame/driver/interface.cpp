//---------------------------------------------------------------------------
#include "measure.h"
#include "interface.h"
#include "xnodeconnector.h"
#include <string>
#include "driver.h"
#include "klocale.h"

//---------------------------------------------------------------------------

XInterface::XInterfaceError::XInterfaceError(const QString &msg, const char *file, int line)
 : XKameError(msg, file, line) {}
XInterface::XConvError::XConvError(const char *file, int line)
 : XInterfaceError(KAME::i18n("Conversion Error"), file, line) {}
XInterface::XCommError::XCommError(const QString &msg, const char *file, int line)
     :  XInterfaceError(KAME::i18n("Communication Error") + ", " + msg, file, line) {}
XInterface::XOpenInterfaceError::XOpenInterfaceError(const char *file, int line)
     :  XInterfaceError(KAME::i18n("Open Interface Error"), file, line) {}


XInterface::XInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) : 
    XNode(name, runtime), 
    m_driver(driver),
    m_device(create<XComboNode>("Device", false)),
    m_port(create<XStringNode>("Port", false)),
    m_address(create<XUIntNode>("Address", false)),
    m_baudrate(create<XUIntNode>("BaudRate", false)),
    m_control(create<XBoolNode>("Control", true))
{
  lsnOnControlChanged = control()->onValueChanged().connectWeak(
        false, shared_from_this(), &XInterface::onControlChanged);
}

std::string
XInterface::getLabel() const
{
	if(m_label.empty())
		return driver()->getLabel();
	else
		return driver()->getLabel() + ":" + m_label;
}

void
XInterface::onControlChanged(const shared_ptr<XValueNodeBase> &)
{
	if(*control()) {
		start();
	}
	else {
		m_tlkOnClose.talk(dynamic_pointer_cast<XInterface>(shared_from_this()));
//		stop();
	}
}

void
XInterface::start()
{
  XScopedLock<XInterface> lock(*this);
  try {
      if(isOpened()) {
          throw XInterfaceError(KAME::i18n("Port has already opened"), __FILE__, __LINE__);
      }
      open();
  }
  catch (XInterfaceError &e) {
        e.print(getLabel() + KAME::i18n(": Opening port failed, because"));
		lsnOnControlChanged->mask();
		control()->value(false);
		lsnOnControlChanged->unmask();
		return;
  }

  device()->setUIEnabled(false);
  port()->setUIEnabled(false);
  address()->setUIEnabled(false);
  baudrate()->setUIEnabled(false);

  m_tlkOnOpen.talk(dynamic_pointer_cast<XInterface>(shared_from_this()));
}
void
XInterface::stop()
{
  XScopedLock<XInterface> lock(*this);
  
  try {
      close();
  }
  catch (XInterfaceError &e) {
        e.print(getLabel() + KAME::i18n(": Closing port failed, because"));
  }

  device()->setUIEnabled(true);
  port()->setUIEnabled(true);
  address()->setUIEnabled(true);
  baudrate()->setUIEnabled(true);
  //g_statusPrinter->clear();
}
