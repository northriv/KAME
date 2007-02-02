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
    m_opened(create<XBoolNode>("Opened", true))
{
}
