#include "comediinterface.h"

QString
XComediPort::errmsg(const QString &msg)
{
	return QString(msg) + ": " + comedi_strerror(comedi_errno());
}

XComediPort::XComediPort(XInterface *interface)
 : XPort(interface)
{
	comedi_loglevel(3);
}

XComediPort::~XComediPort()
{
    try {
        comedi_close();
    }
    catch(...) {
    }
} 
void
XComediPort::open() throw (XInterface::XCommError &)
{
	comedi_t *pdev = comedi_open(m_pInterface->port()->to_str());
	pDev = pdev;
	if(!pdev) {
		throw XInterface::XCommError(
            errmsg(KAME::i18n("opening comedi device faild")), __FILE__, __LINE__);
	}
}
void
XComediPort::comedi_close() throw (XInterface::XCommError &)
{
	if(pDev)
		comedi_close(pDev);
}

void
XComediPort::send(const char *str) throw (XInterface::XCommError &)
{
  ASSERT(m_pInterface->isOpened());
}
void
XComediPort::write(const char *sendbuf, int size) throw (XInterface::XCommError &)
{
  ASSERT(m_pInterface->isOpened());
}
void
XComediPort::receive() throw (XInterface::XCommError &) {
}
void
XComediPort::receive(unsigned int length) throw (XInterface::XCommError &)
{
}
