#include "comediinterface.h"
 #ifdef HAVE_COMEDI
 
QString
XComediInterface::errmsg(const QString &msg)
{
	return QString(msg) + ": " + comedi_strerror(comedi_errno());
}

XComediInterface::XComediInterface(XInterface *interface)
 : XPort(interface)
{
	comedi_loglevel(3);
}

XComediInterface::~XComediInterface()
{
    try {
        comedi_close();
    }
    catch(...) {
    }
} 
void
XComediInterface::open() throw (XInterfaceError &)
{
	comedi_t *pdev = comedi_open(m_pInterface->port()->to_str());
	pDev = pdev;
	if(!pdev) {
		throw XInterface::XCommError(
            errmsg(KAME::i18n("opening comedi device faild")), __FILE__, __LINE__);
	}
}
void
XComediInterface::close() throw (XInterface::XCommError &)
{
	if(pDev)
		comedi_close(pDev);
}


 #endif //HAVE_COMEDI
