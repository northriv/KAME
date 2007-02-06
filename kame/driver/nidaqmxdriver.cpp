#include "nidaqmxdriver.h"

#ifdef HAVE_NI_DAQMX

QString
XNIDAQmxInterface::getNIDAQmxErrMessage()
{
char str[2048];
	DAQmxGetExtendedErrorInfo(str, sizeof(str));
	return QString(str);
}
QString
XNIDAQmxInterface::getNIDAQmxErrMessage(int status)
{
char str[2048];
	DAQmxGetErrorString(status, str, sizeof(str));
	return QString(str);
}
int
XNIDAQmxInterface::checkDAQmxError(const QString &msg, const char*file, int line) {
	throw XInterface::XInterfaceError(msg + " " + getNIDAQmxErrMessage(), file, line);
}

void
XNIDAQmxInterface::parseList(const char *str, std::deque<std::string> &list)
{
	list.clear();
	std::string org(str);
	const char *del = " \t";
	for(unsigned int pos = 0; pos != std::string::npos; ) {
		unsigned int spos = org.find_first_not_of(del, pos);
		if(spos == std::string::npos) break;
		pos = org.find_first_of(del, spos);
		if(pos == std::string::npos)
			list.push_back(org.substr(spos));
		else
			list.push_back(org.substr(spos, pos - spos));
	}
}


XNIDAQmxInterface::XNIDAQmxInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) : 
    XInterface(name, runtime, driver)
{
char buf[2048];
	CHECK_DAQMX_RET(DAQmxGetSysDevNames(buf, sizeof(buf)), "");
std::deque<std::string> list;
	parseList(buf, list);
	for(std::deque<std::string>::iterator it = list.begin(); it != list.end(); it++) {
		CHECK_DAQMX_RET(DAQmxGetDevProductType(it->c_str(), buf, sizeof(buf)), "");
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
