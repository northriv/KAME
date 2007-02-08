#include "nidaqmxdriver.h"

#ifdef HAVE_NI_DAQMX

static int g_daqmx_open_cnt;
static XMutex g_daqmx_mutex;
static TaskHandle g_task_sync_timebase;
static void XNIDAQmxGlobalOpen();
static void XNIDAQmxGlobalClose();

static void
XNIDAQmxGlobalOpen()
{
	XScopedLock<XMutex> lock(g_daqmx_mutex);
	if(g_daqmx_open_cnt == 0) {
	    CHECK_DAQMX_RET(DAQmxCreateTask("", &g_task_sync_master));
		
	char buf[2048];
		CHECK_DAQMX_RET(DAQmxGetSysDevNames(buf, sizeof(buf)), "");
	std::deque<std::string> list;
		XNIDAQmxInterface::parseList(buf, list);
		for(std::deque<std::string>::iterator it = list.begin(); it != list.end(); it++) {
			if(it == list.begin()) {
				dbgPrint(QString("Export %1/20MHzTimebase as the global MasterTimebase.").arg(*it));
				CHECK_DAQMX_RET(DAQmxExportSignal(g_task_sync_master,
					DAQmx_Val_20MHzTimebaseClock, QString("%1/RTSI7").arg(*it)));
//				CHECK_DAQMX_RET(DAQmxConnectTerms(QString("%1/20MHzTimebase").arg(*it),
//					QString("%1/RTSI7").arg(*it), DAQmx_Val_DoNotInvertPolarity));
			}
			else {
				CHECK_DAQMX_RET(DAQmxSetMasterTimebaseSrc(g_task_sync_master,
					QString("%1/RTSI7").arg(*it)));
//				CHECK_DAQMX_RET(DAQmxConnectTerms(QString("%1/RTSI7").arg(*it),
//					QString("%1/MasterTimebase").arg(*it),
//					DAQmx_Val_DoNotInvertPolarity));
			}
		}
	}
	g_daqmx_open_cnt++;
}
static void
XNIDAQmxGlobalClose()
{
	XScopedLock<XMutex> lock(g_daqmx_mutex);
	g_daqmx_open_cnt--;
	if(g_daqmx_open_cnt == 0) {
		CHECK_DAQMX_RET(DAQmxClearTask(g_task_sync_master));
//	char buf[2048];
//		CHECK_DAQMX_RET(DAQmxGetSysDevNames(buf, sizeof(buf)), "");
//	std::deque<std::string> list;
//		XNIDAQmxInterface::parseList(buf, list);
//		for(std::deque<std::string>::iterator it = list.begin(); it != list.end(); it++) {
//			if(it == list.begin()) {
//				CHECK_DAQMX_RET(DAQmxDisconnectTerms(QString("%1/20MHzTimebase").arg(*it),
//					QString("%1/RTSI7").arg(*it)));
//			}
//			else {
//				CHECK_DAQMX_RET(DAQmxDisconnectTerms(QString("%1/RTSI7").arg(*it),
//					QString("%1/MasterTimebase").arg(*it)));
//			}
//		}
	}
}

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
XNIDAQmxInterface::checkDAQmxError(int ret, const char*file, int line) {
	if(ret >= 0) return ret;
	throw XInterface::XInterfaceError(getNIDAQmxErrMessage(), file, line);
	return 0;
}

void
XNIDAQmxInterface::parseList(const char *str, std::deque<std::string> &list)
{
	list.clear();
	std::string org(str);
	const char *del = ", \t";
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
XNIDAQmxRoute::XNIDAQmxRoute(const char*src, const char*dst)
{
	try {
	    CHECK_DAQMX_RET(DAQmxConnectTerms(src, dst, DAQmx_Val_DoNotInvertPolarity));
	}
	catch (XInterface::XInterfaceError &e) {
		e.print();
	}
}
XNIDAQmxRoute::~XNIDAQmxRoute()
{
	try {
	    CHECK_DAQMX_RET(DAQmxDisconnectTerms(src, dst));
	}
	catch (XInterface::XInterfaceError &e) {
		e.print();
	}
}
void
XNIDAQmxInterface::open() throw (XInterfaceError &)
{
char buf[256];
	XNIDAQmxGlobalOpen();
	if(sscanf(device()->to_str().c_str(), "%256s", buf) != 1)
          	throw XOpenInterfaceError(__FILE__, __LINE__);
	m_devname = buf;
}
void
XNIDAQmxInterface::close() throw (XInterfaceError &)
{
	m_devname.clear();
	XNIDAQmxGlobalClose();
}
#endif //HAVE_NI_DAQMX
