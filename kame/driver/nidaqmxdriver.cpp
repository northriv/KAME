#include "nidaqmxdriver.h"

#ifdef HAVE_NI_DAQMX

QString
XNIDAQmxInterface::getNIDAQmxErrMessage()
{
char str[2048];
	DAQmxGetExtendedErrorInfo(str, sizeof(str));
	return QString(str);
}
int
XNIDAQmxInterface::checkDAQmxError(const QString &msg, const char*file, int line) {
	throw XInterface::XInterfaceError(msg + getNIDAQmxErrMessage(ret), file, line);
}

void
XNIDAQmxInterface::parseList(const char *str, std::deque<std::string> &list)
{
	QStringList qlist = QString(str).split(QRegExp("\\s+"));
	for(QListIterator<QString> it = qlist.constBegin(); it != qlist.constEnd(); it++)
		list.push_back(std::string(*it));
}


XNIDAQmxInterface::XNIDAQmxInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) : 
    XInterface(name, runtime, driver)
{
char buf[2048];
	CHECK_DAQMX_RET(DAQmxGetSysDevNames(buf, sizeof(buf)), "");
std::deque<std::string> list;
	parseList(buf, list);
	for(std::deque<std::string>::iterator it = list.begin(); it != list.end(); it++) {
		CHECK_DAQMX_RET(DAQmxGetDevProductType(*it, buf, sizeof(buf)), "");
//		int32 type;
//		CHECK_DAQMX_RET(DAQmxGetDevBusType(*it, &type), "");
//		std::string ts;
//		switch(type) {
//			case DAQmx_Val_PCI: ts = "PCI"; break;
//			case DAQmx_Val_PCIe: ts = "PCIe"; break;
//			case DAQmx_Val_PXI: ts = "PXI"; break;
//			case DAQmx_Val_SCXI: ts = "SCXI"; break;
//			case DAQmx_Val_USB: ts = "USB"; break;
//			default: ts = "Unknown"; break;
//		}
//		device()->add(*it + " [" + buf + "-" + ts + "]");
		device()->add(*it + " [" + buf + "]");
	}
}
void
XNIDAQmxInterface::open() throw (XInterfaceError &)
{
char buf[256];
	if(sscanf(device()->to_str().c_str(), "%256s", buf) != 1)
          	throw XOpenInterfaceError(__FILE__, __LINE__);
	m_devname = buf;
}
void
XNIDAQmxInterface::close() throw (XInterfaceError &)
{
	m_devname.clear();
}
#endif //HAVE_NI_DAQMX