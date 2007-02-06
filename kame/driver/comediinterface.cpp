#include "comediinterface.h"
#ifdef HAVE_COMEDI
 
QString
XComediInterface::errmsg(const QString &msg)
{
	return QString(msg) + ": " + comedi_strerror(comedi_errno());
}

XComediInterface::XComediInterface(const char *name, bool runtime,
 const shared_ptr<XDriver> &driver, int subdevice_type)
 : m_pDev(0), m_subdevice_type(subdevice_type)
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
	comedi_t *pdev = comedi_open(port()->to_str());
	if(!pdev) {
		throw XInterface::XCommError(
            errmsg(KAME::i18n("opening comedi device failed")), __FILE__, __LINE__);
	}
	m_subdevice = comedi_find_subdevice_by_type(pdev, m_subdevice_type, *address());
	if(comedi_subdev() < 0) {
		throw XInterface::XCommError(
            errmsg(KAME::i18n("opening comedi device failed")), __FILE__, __LINE__);
	}	
	int flag = comedi_get_subdevice_flags(pdev, comedi_subdev());
	if(flag < 0) {
		throw XInterface::XCommError(
            errmsg(KAME::i18n("opening comedi sub-device failed")), __FILE__, __LINE__);
	}
	if(flag & (SDF_RUNNING | SDF_BUSY_OWNER | SDF_LOCKED)) {
		throw XInterface::XCommError(
            errmsg(KAME::i18n("opening comedi sub-device failed")), __FILE__, __LINE__);
	}
	if(comedi_lock(pdev, comedi_subdev()) < 0) {
		throw XInterface::XCommError(
            errmsg(KAME::i18n("locking comedi sub-device failed")), __FILE__, __LINE__);
	}
	if(flag & SDF_CMD) {
		fcntl(comedi_fileno(pdev),F_SETFL,O_NONBLOCK);
	}
	
	address()->value(comedi_subdev());
	m_pDev = pdev;
}
void
XComediInterface::close() throw (XInterface::XCommError &)
{
	comedi_t *dev = comedi_dev();
	m_pDev = 0;
	if(dev) {
		comedi_close(dev);
		if(comedi_unlock(dev, m_subdevice) < 0) {
			throw XInterface::XCommError(
	            errmsg(KAME::i18n("unlocking comedi sub-device failed")), __FILE__, __LINE__);
		}
	}
}
int
XComediInterface::numChannels()
{
	int ret = comedi_get_n_channels(comedi_dev(), comedi_subdev());
	if(ret < 0)
		throw XInterface::XCommError(
            errmsg(KAME::i18n("Comedi comm. failed")), __FILE__, __LINE__);
    return ret;
}

void
XComediInterface::comedi_command_test(comedi_cmd *cmd)
{
	int ret = comedi_command_test(comedi_dev(), cmd);
	switch(ret) {
		case 0:
			break;
		case 1:
			break;
		case 2:
			break;
		case 3:
			break;
		case 4:
			break;
		case 5:
			break;
		default:
			break;
	}
}

#endif //HAVE_COMEDI
