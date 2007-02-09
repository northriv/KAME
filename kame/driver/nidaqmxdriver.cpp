#include "nidaqmxdriver.h"

#ifdef HAVE_NI_DAQMX

static int g_daqmx_open_cnt;
static XMutex g_daqmx_mutex;
static std::deque<shared_ptr<XNIDAQmxInterface::XNIDAQmxRoute> > g_daqmx_sync_routes;

static void
XNIDAQmxGlobalOpen()
{
	XScopedLock<XMutex> lock(g_daqmx_mutex);
	if(g_daqmx_open_cnt == 0) {
//	    CHECK_DAQMX_RET(DAQmxCreateTask("", &g_task_sync_master));
	char buf[2048];		
		CHECK_DAQMX_RET(DAQmxGetSysDevNames(buf, sizeof(buf)));
	std::deque<std::string> list;
		XNIDAQmxInterface::parseList(buf, list);
		std::string master10MHz;
		std::string master20MHz;
		std::deque<std::string> pcidevs;
		for(std::deque<std::string>::iterator it = list.begin(); it != list.end(); it++) {
			DAQmxResetDevice(it->c_str());
			int32 bus;
			DAQmxGetDevBusType(it->c_str(), &bus);
			if((bus == DAQmx_Val_PCI) || (bus == DAQmx_Val_PCIe)) {
				pcidevs.push_back(*it);
				//RTSI synchronizations.
				shared_ptr<XNIDAQmxInterface::XNIDAQmxRoute> route;
				if(!master10MHz.length()) {
					int pret;
					route.reset(new XNIDAQmxInterface::XNIDAQmxRoute(
						QString("/%1/10MHzRefClock").arg(*it),
						QString("/%1/RTSI6").arg(*it), &pret));
					if(pret >= 0) {
						master10MHz = *it;
						g_daqmx_sync_routes.push_back(route);
						printf("10MHzRefClk found on %s\n", it->c_str());
						route.reset(new XNIDAQmxInterface::XNIDAQmxRoute(
							QString("/%1/20MHzTimebase").arg(*it),
							QString("/%1/RTSI7").arg(*it), &pret));
						if(pret >= 0) {
							master20MHz = *it;
							printf("20MHzRefClk found on %s\n", it->c_str());
							g_daqmx_sync_routes.push_back(route);
						}
					}
				}
			}
		}
		for(std::deque<std::string>::iterator it = pcidevs.begin(); it != pcidevs.end(); it++) {
			shared_ptr<XNIDAQmxInterface::XNIDAQmxRoute> route;
			if(!master20MHz.length()) {
				int pret;
				route.reset(new XNIDAQmxInterface::XNIDAQmxRoute(
					QString("/%1/20MHzTimebase").arg(*it),
					QString("/%1/RTSI7").arg(*it), &pret));
				if(pret >= 0) {
					master20MHz = *it;
					printf("20MHzRefClk found on %s\n", it->c_str());
					g_daqmx_sync_routes.push_back(route);
				}
			}
		}
/*		if(pcidevs.size() > 1) {
			for(std::deque<std::string>::iterator it = pcidevs.begin(); it != pcidevs.end(); it++) {
				shared_ptr<XNIDAQmxInterface::XNIDAQmxRoute> route;
				if(*it == master10MHz) continue;
				if(*it == master20MHz) continue;
				int pret = -1;
				if(master20MHz.length())
					route.reset(new XNIDAQmxInterface::XNIDAQmxRoute(
						QString("/%1/RTSI7").arg(*it),
						QString("/%1/MasterTimebase").arg(*it), &pret));
				if(pret >= 0) {
					g_daqmx_sync_routes.push_back(route);
				}
				else {
					if(master10MHz.length())
						route.reset(new XNIDAQmxInterface::XNIDAQmxRoute(
							QString("/%1/RTSI6").arg(*it),
							QString("/%1/ExternalRefereceClock").arg(*it), &pret));
					if(pret >= 0) {
						g_daqmx_sync_routes.push_back(route);
					}
					else {
						gErrPrint(KAME::i18n("No synchronization route found on ") + *it);
					}
				}
			}
		}*/
	}
	g_daqmx_open_cnt++;
}
static void
XNIDAQmxGlobalClose()
{
	XScopedLock<XMutex> lock(g_daqmx_mutex);
	g_daqmx_open_cnt--;
	if(g_daqmx_open_cnt == 0) {
		g_daqmx_sync_routes.clear();
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
	CHECK_DAQMX_RET(DAQmxGetSysDevNames(buf, sizeof(buf)));
std::deque<std::string> list;
	parseList(buf, list);
	for(std::deque<std::string>::iterator it = list.begin(); it != list.end(); it++) {
		CHECK_DAQMX_RET(DAQmxGetDevProductType(it->c_str(), buf, sizeof(buf)));
		device()->add(*it + " [" + buf + "]");
	}
}
XNIDAQmxInterface::XNIDAQmxRoute::XNIDAQmxRoute(const char*src, const char*dst, int *pret)
 : m_src(src), m_dst(dst)
{
	int ret;
	try {
	    ret = CHECK_DAQMX_ERROR(DAQmxConnectTerms(src, dst, DAQmx_Val_DoNotInvertPolarity));
	    dbgPrint(QString("Connect route from %1 to %2.").arg(src).arg(dst));
	}
	catch (XInterface::XInterfaceError &e) {
		if(!pret) {
			e.print();
		}
		else {
			dbgPrint(e.msg());
		}
		m_src.clear();
	}
	if(pret)
		*pret = ret;
}
XNIDAQmxInterface::XNIDAQmxRoute::~XNIDAQmxRoute()
{
	if(!m_src.length()) return;
	try {
	    CHECK_DAQMX_RET(DAQmxDisconnectTerms(m_src.c_str(), m_dst.c_str()));
	    dbgPrint(QString("Dsiconnect route from %1 to %2.").arg(m_src).arg(m_dst));
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
