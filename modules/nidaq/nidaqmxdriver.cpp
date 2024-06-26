/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "nidaqmxdriver.h"
#if !defined __WIN32__ && !defined WINDOWS && !defined _WIN32
    #include <sys/errno.h>
#endif

//struct ProductInfo {
//    const char *type;
//    const char *series;
//    int flags;
//    unsigned long ai_max_rate; //!< [kHz]
//    unsigned long ao_max_rate; //!< [kHz]
//    unsigned long di_max_rate; //!< [kHz]
//    unsigned long do_max_rate; //!< [kHz]
//};
const XNIDAQmxInterface::ProductInfo
XNIDAQmxInterface::sc_productInfoList[] = {
	{"PCI-6110", "S", 0, 5000uL, 1000uL, 0, 0},
	{"PXI-6110", "S", 0, 5000uL, 1000uL, 0, 0},
	{"PCI-6111", "S", 0, 5000uL, 1000uL, 0, 0},
	{"PXI-6111", "S", 0, 5000uL, 1000uL, 0, 0},
	{"PCI-6115", "S", 0, 10000uL, 2500uL, 10000uL, 10000uL},
    {"PXI-6115", "S", 0, 10000uL, 2500uL, 10000uL, 10000uL},
    {"PCI-6120", "S", 0, 800uL, 2500uL, 5000uL, 5000uL},
	{"PCI-6220", "M", 0, 250uL, 0, 1000uL, 1000uL},
	{"PXI-6220", "M", 0, 250uL, 0, 1000uL, 1000uL},
	{"PCI-6221", "M", 0, 250uL, 740uL, 1000uL, 1000uL},
	{"PXI-6221", "M", 0, 250uL, 740uL, 1000uL, 1000uL},
	{"PCI-6224", "M", 0, 250uL, 0, 1000uL, 1000uL},
	{"PXI-6224", "M", 0, 250uL, 0, 1000uL, 1000uL},
	{"PCI-6229", "M", 0, 250uL, 625uL, 1000uL, 1000uL},
	{"PXI-6229", "M", 0, 250uL, 625uL, 1000uL, 1000uL},
	{"PCI-6250", "M", 0, 1000uL, 0uL, 5000uL, 5000uL},
	{"PXI-6250", "M", 0, 1000uL, 0uL, 5000uL, 5000uL},
	{"PCI-6251", "M", 0, 1000uL, 1000uL, 5000uL, 5000uL},
	{"PCIe-6251", "M", 0, 1000uL, 1000uL, 5000uL, 5000uL},
	{"PXI-6251", "M", 0, 1000uL, 1000uL, 5000uL, 5000uL},
	{"PCI-6254", "M", 0, 1000uL, 0uL, 5000uL, 5000uL},
	{"PXI-6254", "M", 0, 1000uL, 0uL, 5000uL, 5000uL},
	{"PCI-6255", "M", 0, 750uL, 1000uL, 5000uL, 5000uL},
	{"PXI-6255", "M", 0, 750uL, 1000uL, 5000uL, 5000uL},
	{"PCI-6255", "M", 0, 1000uL, 1000uL, 5000uL, 5000uL},
	{"PCIe-6255", "M", 0, 1000uL, 1000uL, 5000uL, 5000uL},
	{"PXI-6255", "M", 0, 1000uL, 1000uL, 5000uL, 5000uL},
    {"PCIe-6321", "X", 0, 250uL, 900uL, 1000uL, 1000uL},
    {"PCIe-6323", "X", 0, 250uL, 900uL, 1000uL, 1000uL},
    {"PCIe-6341", "X", 0, 500uL, 900uL, 1000uL, 1000uL},
    {"PCIe-6343", "X", 0, 500uL, 900uL, 1000uL, 1000uL},
    {"PCIe-6351", "X", 0, 1000uL, 1000uL, 5000uL, 5000uL},
    {"PCIe-6353", "X", 0, 1000uL, 1000uL, 5000uL, 5000uL},
    {"PCIe-6361", "X", 0, 1000uL, 1000uL, 5000uL, 5000uL},
    {"PCIe-6363", "X", 0, 1000uL, 1000uL, 5000uL, 5000uL},
    {"PCIe-6374", "X", 0, 3570uL, 3300uL, 10000uL, 10000uL},
    {"PCIe-6376", "X", 0, 3570uL, 3300uL, 10000uL, 10000uL},
    {0, 0, 0, 0, 0, 0, 0},
};

//for synchronization.
static XString g_pciClockMaster; //Device exporting clock to RTSI7.
static float64 g_pciClockMasterRate = 0.0;
static XString g_pciClockExtRefTerm; //External Reference Clock
static float64 g_pciClockExtRefRate;
static TaskHandle g_pciClockMasterTask = (TaskHandle)-1;
static int g_daqmx_open_cnt;
static XMutex g_daqmx_mutex;
static std::deque<shared_ptr<XNIDAQmxInterface::XNIDAQmxRoute> > g_daqmx_sync_routes;

SoftwareTriggerManager XNIDAQmxInterface::s_softwareTriggerManager;

const char *
XNIDAQmxInterface::busArchType() const {
#ifdef HAVE_NI_DAQMX
	int32 bus;
	DAQmxGetDevBusType(devName(), &bus);
	switch(bus) {
	case DAQmx_Val_PCI:
	case DAQmx_Val_PCIe:
		return "PCI";
	case DAQmx_Val_PXI:
//	case DAQmx_Val_PXIe:
		return "PXI";
	case DAQmx_Val_USB:
		return "USB";
	default:
		return "Unknown";
	}
#else
	return "n/a";
#endif //HAVE_NI_DAQMX
}
void
XNIDAQmxInterface::synchronizeClock(TaskHandle task) {
	float64 rate = g_pciClockMasterRate;
	XString src = formatString("/%s/RTSI7", devName());
	if(devName() == g_pciClockMaster) {
		src = g_pciClockExtRefTerm;
		rate = g_pciClockExtRefRate;
		if( !src.length())
			return;
	}
	
    if((productSeries() == XString("M")) || (productSeries() == XString("X"))) {
		if(busArchType() == XString("PCI")) {
			CHECK_DAQMX_RET(DAQmxSetRefClkSrc(task, src.c_str()));
			CHECK_DAQMX_RET(DAQmxSetRefClkRate(task, rate));
		}
		if(busArchType() == XString("PXI")) {
			CHECK_DAQMX_RET(DAQmxSetRefClkSrc(task,"PXI_Clk10"));
			CHECK_DAQMX_RET(DAQmxSetRefClkRate(task, 10e6));
		}
	}
	if(productSeries() == XString("S")) {
		if(busArchType() == XString("PCI")) {
			CHECK_DAQMX_RET(DAQmxSetMasterTimebaseSrc(task, src.c_str()));
			CHECK_DAQMX_RET(DAQmxSetMasterTimebaseRate(task, rate));
		}
		if(busArchType() == XString("PXI")) {
			CHECK_DAQMX_RET(DAQmxSetMasterTimebaseSrc(task,"PXI_Clk10"));
			CHECK_DAQMX_RET(DAQmxSetMasterTimebaseRate(task, 10e6));
		}
	}
}

XString
XNIDAQmxInterface::getNIDAQmxErrMessage()
{
#ifdef HAVE_NI_DAQMX
	char str[2048];
	DAQmxGetExtendedErrorInfo(str, sizeof(str));
	errno = 0;
    return {str};
#else
    return {};
#endif //HAVE_NI_DAQMX
}
XString
XNIDAQmxInterface::getNIDAQmxErrMessage(int status) {
#ifdef HAVE_NI_DAQMX
	char str[2048];
	DAQmxGetErrorString(status, str, sizeof(str));
	errno = 0;
    return {str};
#else
    return {};
#endif //HAVE_NI_DAQMX
}
int
XNIDAQmxInterface::checkDAQmxError(int ret, const char*file, int line) {
	if(ret >= 0) return ret;
	errno = 0;
	throw XInterface::XInterfaceError(getNIDAQmxErrMessage(), file, line);
	return 0;
}

void
XNIDAQmxInterface::parseList(const char *str, std::deque<XString> &list)
{
	list.clear();
	XString org(str);
	const char *del = ", \t";
	for(int pos = 0; pos != std::string::npos; ) {
		int spos = org.find_first_not_of(del, pos);
		if(spos == std::string::npos) break;
		pos = org.find_first_of(del, spos);
		if(pos == std::string::npos)
			list.push_back(org.substr(spos));
		else
			list.push_back(org.substr(spos, pos - spos));
	}
}


XNIDAQmxInterface::XNIDAQmxInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver) : 
    XInterface(name, runtime, driver),
    m_productInfo(0L) {
#ifdef HAVE_NI_DAQMX
	char buf[2048];
	CHECK_DAQMX_RET(DAQmxGetSysDevNames(buf, sizeof(buf)));
	std::deque<XString> list;
	parseList(buf, list);
    iterate_commit([=, &buf](Transaction &tr){
        for(auto it = list.cbegin(); it != list.cend(); ++it) {
            CHECK_DAQMX_RET(DAQmxGetDevProductType(it->c_str(), buf, sizeof(buf)));
			tr[ *device()].add(*it + " [" + buf + "]");
		}
    });
#endif //HAVE_NI_DAQMX
    address()->disable();
    port()->disable();
}
XNIDAQmxInterface::XNIDAQmxRoute::XNIDAQmxRoute(const char*src, const char*dst, int *pret)
	: m_src(src), m_dst(dst) {
#ifdef HAVE_NI_DAQMX
	if(pret) {
		int ret = 0;
	    ret = DAQmxConnectTerms(src, dst, DAQmx_Val_DoNotInvertPolarity);
	    if(ret < 0)
			m_src.clear();
		*pret = ret;
	}
	else {
		try {
		    CHECK_DAQMX_ERROR(DAQmxConnectTerms(src, dst, DAQmx_Val_DoNotInvertPolarity));
		    dbgPrint(QString("Connect route from %1 to %2.").arg(src).arg(dst));
		}
		catch (XInterface::XInterfaceError &e) {
			e.print();
			m_src.clear();
		}
	}
#endif //HAVE_NI_DAQMX
}
XNIDAQmxInterface::XNIDAQmxRoute::~XNIDAQmxRoute() {
	if(!m_src.length()) return;
	try {
	    CHECK_DAQMX_RET(DAQmxDisconnectTerms(m_src.c_str(), m_dst.c_str()));
	    dbgPrint(QString("Disconnect route from %1 to %2.").arg(m_src).arg(m_dst));
	}
	catch (XInterface::XInterfaceError &e) {
		e.print();
	}
}
void
XNIDAQmxInterface::open() {
#ifdef HAVE_NI_DAQMX
	char buf[256];

	Snapshot shot( *this);
    if(sscanf(shot[ *device()].to_str().c_str(), "%255s", buf) != 1)
		throw XOpenInterfaceError(__FILE__, __LINE__);

	XScopedLock<XMutex> lock(g_daqmx_mutex);
	if(g_daqmx_open_cnt == 0) {
		//Routes the master clock for synchronizations.
		g_pciClockExtRefTerm.clear();
//	    CHECK_DAQMX_RET(DAQmxCreateTask("", &g_task_sync_master));
		char buf[2048];		
		CHECK_DAQMX_RET(DAQmxGetSysDevNames(buf, sizeof(buf)));
		std::deque<XString> list;
		XNIDAQmxInterface::parseList(buf, list);
		std::deque<XString> pcidevs;
        for(std::deque<XString>::iterator it = list.begin(); it != list.end(); ++it) {
			// Device reset.
			DAQmxResetDevice(it->c_str());
			// Search for master clock among PCI(e) devices.
			int32 bus;
			DAQmxGetDevBusType(it->c_str(), &bus);
			if((bus == DAQmx_Val_PCI) || (bus == DAQmx_Val_PCIe)) {
				pcidevs.push_back(*it);
			}
		}
		if(pcidevs.size() > 1) {
            for(std::deque<XString>::iterator it = pcidevs.begin(); it != pcidevs.end(); ++it) {
				//M series only.
				CHECK_DAQMX_RET(DAQmxGetDevProductType(it->c_str(), buf, sizeof(buf)));
				XString type = buf;
				for(const ProductInfo *pit = sc_productInfoList; pit->type; pit++) {
                    if((pit->type == type) && ((pit->series == XString("M")) || (pit->series == XString("X")) )) {
						XString inp_term = formatString("/%s/PFI0", it->c_str());
						//Detects external clock source.
						if(routeExternalClockSource(it->c_str(),  inp_term.c_str())) {
							fprintf(stderr, "Reference Clock for PLL Set to %s\n", inp_term.c_str());
							g_pciClockMaster = *it;
							pcidevs.clear();
							pcidevs.push_back(g_pciClockMaster);
						}
						break;
					}
				}
				if(g_pciClockMaster.length())
					break;
			}
            for(std::deque<XString>::iterator it = pcidevs.begin(); it != pcidevs.end(); ++it) {
				//AO of S series device cannot handle external master clock properly.
				//Thus S should be a master when high-speed AOs are used.
				CHECK_DAQMX_RET(DAQmxGetDevProductType(it->c_str(), buf, sizeof(buf)));
				XString type = buf;
                for(const ProductInfo *pit = sc_productInfoList; pit->type; ++pit) {
					if((pit->type == type) && (pit->series == XString("S"))) {
						//RTSI synchronizations.
						shared_ptr<XNIDAQmxInterface::XNIDAQmxRoute> route;
						float64 freq = 20.0e6;
						route.reset(new XNIDAQmxInterface::XNIDAQmxRoute(
										formatString("/%s/20MHzTimebase", it->c_str()).c_str(),
										formatString("/%s/RTSI7", it->c_str()).c_str()));
						g_daqmx_sync_routes.push_back(route);
						fprintf(stderr, "20MHz Reference Clock exported from %s\n", it->c_str());
						g_pciClockMaster = *it;
						g_pciClockMasterRate = freq;
						pcidevs.clear();
						break;
					}
				}
				if(g_pciClockMaster.length())
					break;
			}
            for(std::deque<XString>::iterator it = pcidevs.begin(); it != pcidevs.end(); ++it) {
				//M series only.
				CHECK_DAQMX_RET(DAQmxGetDevProductType(it->c_str(), buf, sizeof(buf)));
				XString type = buf;
				for(const ProductInfo *pit = sc_productInfoList; pit->type; pit++) {
                    if((pit->type == type) && ((pit->series == XString("M")) || (pit->series == XString("X")) )) {
                        //Detects external clock source.
						fprintf(stderr, "20MHz Reference Clock exported from %s\n", it->c_str());
						//M series device cannot export 20MHzTimebase freely.
						XString ctrdev = formatString("%s/ctr1", it->c_str());
						//Continuous pulse train generation. Duty = 50%.
						CHECK_DAQMX_RET(DAQmxCreateTask("", &g_pciClockMasterTask));
						float64 freq = 20e6; //20MHz
						CHECK_DAQMX_RET(DAQmxCreateCOPulseChanFreq(g_pciClockMasterTask,
																   ctrdev.c_str(), "", DAQmx_Val_Hz, DAQmx_Val_Low, 0.0,
																   freq, 0.5));
						CHECK_DAQMX_RET(DAQmxCfgImplicitTiming(g_pciClockMasterTask, DAQmx_Val_ContSamps, 1000));
						CHECK_DAQMX_RET(DAQmxSetCOPulseTerm(g_pciClockMasterTask, ctrdev.c_str(), formatString("/%s/RTSI7", it->c_str()).c_str()));
						if(g_pciClockExtRefTerm.length()) {
							CHECK_DAQMX_RET(DAQmxSetRefClkSrc(g_pciClockMasterTask, g_pciClockExtRefTerm.c_str()));
							CHECK_DAQMX_RET(DAQmxSetRefClkRate(g_pciClockMasterTask, g_pciClockExtRefRate));
						}
						CHECK_DAQMX_RET(DAQmxStartTask(g_pciClockMasterTask));
						g_pciClockMaster = *it;
						g_pciClockMasterRate = freq;
						pcidevs.clear();
						break;
					}
				}
				if(g_pciClockMaster.length())
					break;
			}
		}
	}
	g_daqmx_open_cnt++;

	XString devname = buf;
	CHECK_DAQMX_RET(DAQmxGetDevProductType(devname.c_str(), buf, sizeof(buf)));
	XString type = buf;
	
	m_productInfo = NULL;
	for(const ProductInfo *it = sc_productInfoList; it->type; it++) {
		if(it->type == type) {
			m_productInfo = it;
			m_devname = devname;
			return;
		}
	}
	throw XInterfaceError(i18n("No device info. for product [%1].").arg(type), __FILE__, __LINE__);
#endif //HAVE_NI_DAQMX
}
bool
XNIDAQmxInterface::routeExternalClockSource(const char *dev, const char *inp_term) {
#ifdef HAVE_NI_DAQMX
    XString ctrdev = formatString("/%s/ctr0", dev);
	TaskHandle taskCounting = 0;
	//Measures an external source frequency.
	CHECK_DAQMX_RET(DAQmxCreateTask("",&taskCounting));
	CHECK_DAQMX_RET(DAQmxCreateCIFreqChan(taskCounting,
		ctrdev.c_str(), "", 1000000, 25000000, DAQmx_Val_Hz,
		DAQmx_Val_Rising, DAQmx_Val_LargeRng2Ctr, 0.01, 100, NULL));
	CHECK_DAQMX_RET(DAQmxCfgImplicitTiming(taskCounting, DAQmx_Val_ContSamps, 1000));
	CHECK_DAQMX_RET(DAQmxSetCIFreqTerm(taskCounting, ctrdev.c_str(), inp_term));

	CHECK_DAQMX_RET(DAQmxStartTask(taskCounting));
	float64 data[1000];
	int32 cnt;
	int32 ret = DAQmxReadCounterF64(taskCounting, 1000, 0.05, data,1000, &cnt, 0);
	float64 freq = 0.0;
	for(int i = cnt / 2; i < cnt; ++i)
		freq += data[i];
	freq /= cnt - cnt / 2; //averaging.
	DAQmxStopTask(taskCounting);
	DAQmxClearTask(taskCounting);
	if(ret)
		return false;
	fprintf(stderr, "%.5g Hz detected at the counter input term %s.\n", (double)freq, inp_term);

	uint64_t freq_cand[] = {10000000, 20000000, 0};
	for(uint64_t *f = freq_cand; *f; ++f) {
		if(fabs(freq - *f) < *f * 0.001) {
			g_pciClockExtRefTerm = inp_term;
			g_pciClockExtRefRate = *f;
			return true;
		}
	}
#endif //HAVE_NI_DAQMX

	return false;
}
void
XNIDAQmxInterface::close() {
	m_productInfo = NULL;
	if(m_devname.length()) {
		m_devname.clear();

		XScopedLock<XMutex> lock(g_daqmx_mutex);
		g_daqmx_open_cnt--;
		if(g_daqmx_open_cnt == 0) {
            if(g_pciClockMasterTask != (TaskHandle)-1) {
				CHECK_DAQMX_RET(DAQmxClearTask(g_pciClockMasterTask));
			}
            g_pciClockMasterTask = (TaskHandle)-1;
			g_daqmx_sync_routes.clear();
		}
	}
}
