/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "nidaqmxdriver.h"
#include <sys/errno.h>
#include <boost/math/common_factor.hpp>
using boost::math::lcm;
using boost::math::gcd;

const XNIDAQmxInterface::ProductInfo
XNIDAQmxInterface::sc_productInfoList[] = {
	{"PCI-6110", "S", 0, 5000uL, 1000uL, 0, 0},
	{"PXI-6110", "S", 0, 5000uL, 1000uL, 0, 0},
	{"PCI-6111", "S", 0, 5000uL, 1000uL, 0, 0},
	{"PXI-6111", "S", 0, 5000uL, 1000uL, 0, 0},
	{"PCI-6115", "S", 0, 10000uL, 2500uL, 10000uL, 10000uL},
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
	{0, 0, 0, 0, 0, 0, 0},
};

//for synchronization.
static XString g_pciClockMaster; //Device exporting clock to RTSI7.
static float64 g_pciClockMasterRate = 0.0;
static XString g_pciClockExtRefTerm; //External Reference Clock
static float64 g_pciClockExtRefRate;
static TaskHandle g_pciClockMasterTask = -1;
static int g_daqmx_open_cnt;
static XMutex g_daqmx_mutex;
static std::deque<shared_ptr<XNIDAQmxInterface::XNIDAQmxRoute> > g_daqmx_sync_routes;

atomic_shared_ptr<XNIDAQmxInterface::SoftwareTrigger::SoftwareTriggerList>
XNIDAQmxInterface::SoftwareTrigger::s_virtualTrigList(new XNIDAQmxInterface::SoftwareTrigger::SoftwareTriggerList);
XTalker<shared_ptr<XNIDAQmxInterface::SoftwareTrigger> >
XNIDAQmxInterface::SoftwareTrigger::s_onChange;

shared_ptr<XNIDAQmxInterface::SoftwareTrigger>
XNIDAQmxInterface::SoftwareTrigger::create(const char *label, unsigned int bits) {
	shared_ptr<SoftwareTrigger> p(new SoftwareTrigger(label, bits));
	
	//inserting the new trigger source to the list atomically.
	for(local_shared_ptr<SoftwareTriggerList> old_list(s_virtualTrigList);;) {
        local_shared_ptr<SoftwareTriggerList> new_list(new SoftwareTriggerList( *old_list));
        new_list->push_back(p);
        if(s_virtualTrigList.compareAndSwap(old_list, new_list)) break;
    }
    onChange().talk(p);
    return p;
}

XNIDAQmxInterface::SoftwareTrigger::SoftwareTrigger(const char *label, unsigned int bits)
	: m_label(label), m_bits(bits),
	  m_risingEdgeMask(0u), m_fallingEdgeMask(0u) {
 	clear_();
}
void
XNIDAQmxInterface::SoftwareTrigger::unregister(const shared_ptr<SoftwareTrigger> &p) {
	//performing it atomically.
	for(local_shared_ptr<SoftwareTriggerList> old_list(s_virtualTrigList);;) {
		local_shared_ptr<SoftwareTriggerList> new_list(new SoftwareTriggerList( *old_list));
		new_list->erase(std::find(new_list->begin(), new_list->end(), p));
		if(s_virtualTrigList.compareAndSwap(old_list, new_list)) break;
	}
	onChange().talk(p);
}
void
XNIDAQmxInterface::SoftwareTrigger::clear_() {
	uint64_t x;
	while(FastQueue::key t = m_fastQueue.atomicFront(&x)) {
		m_fastQueue.atomicPop(t);
	}
	m_slowQueue.clear();
	m_slowQueueSize = 0;
}
void
XNIDAQmxInterface::SoftwareTrigger::stamp(uint64_t cnt) {
	readBarrier();
	if(cnt < m_endOfBlank) return;
	if(cnt == 0) return; //ignore.
	try {
		m_fastQueue.push(cnt);
	}
	catch (FastQueue::nospace_error&) {
		XScopedLock<XMutex> lock(m_mutex);
		fprintf(stderr, "Slow queue!\n");
		m_slowQueue.push_back(cnt);
		if(m_slowQueue.size() > 100000u)
			m_slowQueue.pop_front();
		else
			++m_slowQueueSize;
	}
	m_endOfBlank = cnt + m_blankTerm;
}
void
XNIDAQmxInterface::SoftwareTrigger::start(float64 freq) {
	{
		XScopedLock<XMutex> lock(m_mutex);
		m_endOfBlank = 0;
		if(!m_blankTerm) m_blankTerm = lrint(0.02 * freq);
		m_freq = freq;
		clear_();
	}
	onStart().talk(shared_from_this());
}

void
XNIDAQmxInterface::SoftwareTrigger::stop() {
	XScopedLock<XMutex> lock(m_mutex);
	clear_();
	m_endOfBlank = (uint64_t)-1LL;
}
void
XNIDAQmxInterface::SoftwareTrigger::connect(uint32_t rising_edge_mask, 
											uint32_t falling_edge_mask) throw (XInterface::XInterfaceError &) {
	XScopedLock<XMutex> lock(m_mutex);
	clear_();
	if(m_risingEdgeMask || m_fallingEdgeMask)
		throw XInterface::XInterfaceError(
			i18n("Duplicated connection to virtual trigger is not supported."), __FILE__, __LINE__);
	m_risingEdgeMask = rising_edge_mask;
	m_fallingEdgeMask = falling_edge_mask;
}
void
XNIDAQmxInterface::SoftwareTrigger::disconnect() {
	XScopedLock<XMutex> lock(m_mutex);
	clear_();
	m_risingEdgeMask = 0;
	m_fallingEdgeMask = 0;
}
uint64_t
XNIDAQmxInterface::SoftwareTrigger::tryPopFront(uint64_t threshold, float64 freq__) {
	unsigned int freq_em = lrint(freq());
	unsigned int freq_rc = lrint(freq__);
	unsigned int gcd__ = gcd(freq_em, freq_rc);
	
	uint64_t cnt;
	if(m_slowQueueSize) {
		XScopedLock<XMutex> lock(m_mutex);
		if(FastQueue::key t = m_fastQueue.atomicFront(&cnt)) {
			if((cnt < m_slowQueue.front()) || !m_slowQueueSize) {
				cnt = (cnt * (freq_rc / gcd__)) / (freq_em / gcd__);
				if(cnt >= threshold)
					return 0uLL;
				if(m_fastQueue.atomicPop(t))
					return cnt;
				return 0uLL;
			}
		}
		if( !m_slowQueueSize)
			return 0uLL;
		cnt = m_slowQueue.front();
		cnt = (cnt * (freq_rc / gcd__)) / (freq_em / gcd__);
		if(cnt >= threshold)
			return 0uLL;
		m_slowQueue.pop_front();			
		--m_slowQueueSize;
		return cnt;
	}
	if(FastQueue::key t = m_fastQueue.atomicFront(&cnt)) {
		cnt = (cnt * (freq_rc / gcd__)) / (freq_em / gcd__);
		if(cnt >= threshold)
			return 0uLL;
		if(m_fastQueue.atomicPop(t))
			return cnt;
	}
	return 0uLL;
}

void
XNIDAQmxInterface::SoftwareTrigger::clear(uint64_t now, float64 freq__) {
	unsigned int freq_em= lrint(freq());
	unsigned int freq_rc = lrint(freq__);
	unsigned int gcd__ = gcd(freq_em, freq_rc);
	now = (now * (freq_em / gcd__)) / (freq_rc / gcd__);

	XScopedLock<XMutex> lock(m_mutex);
	uint64_t x;
	while(FastQueue::key t = m_fastQueue.atomicFront(&x)) {
		if(x <= now)
			m_fastQueue.atomicPop(t);
		else
			break;
	}
	while(m_slowQueue.size() && (m_slowQueue.front() <= now)) {
		m_slowQueue.pop_front();
		--m_slowQueueSize;
	}
}
void
XNIDAQmxInterface::SoftwareTrigger::forceStamp(uint64_t now, float64 freq__) {
	unsigned int freq_em= lrint(freq());
	unsigned int freq_rc = lrint(freq__);
	unsigned int gcd__ = gcd(freq_em, freq_rc);
	now = (now * (freq_em / gcd__)) / (freq_rc / gcd__);
		
	XScopedLock<XMutex> lock(m_mutex);
	++m_slowQueueSize;
	m_slowQueue.push_front(now);
	std::sort(m_slowQueue.begin(), m_slowQueue.end());
}

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
	
	if(productSeries() == XString("M")) {
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
	return XString(str);
#else
	return XString();
#endif //HAVE_NI_DAQMX
}
XString
XNIDAQmxInterface::getNIDAQmxErrMessage(int status) {
#ifdef HAVE_NI_DAQMX
	char str[2048];
	DAQmxGetErrorString(status, str, sizeof(str));
	errno = 0;
	return XString(str);
#else
	return XString();
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
	for(Transaction tr( *this);; ++tr) {
		for(std::deque<XString>::iterator it = list.begin(); it != list.end(); it++) {
			CHECK_DAQMX_RET(DAQmxGetDevProductType(it->c_str(), buf, sizeof(buf)));
			tr[ *device()].add(*it + " [" + buf + "]");
		}
		if(tr.commit())
			break;
	}
#endif //HAVE_NI_DAQMX
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
XNIDAQmxInterface::open() throw (XInterfaceError &) {
#ifdef HAVE_NI_DAQMX
	char buf[256];

	Snapshot shot( *this);
	if(sscanf(shot[ *device()].to_str().c_str(), "%256s", buf) != 1)
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
		for(std::deque<XString>::iterator it = list.begin(); it != list.end(); it++) {
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
//			for(std::deque<XString>::iterator it = pcidevs.begin(); it != pcidevs.end(); it++) {
//				//M series only.
//				CHECK_DAQMX_RET(DAQmxGetDevProductType(it->c_str(), buf, sizeof(buf)));
//				XString type = buf;
//				for(const ProductInfo *pit = sc_productInfoList; pit->type; pit++) {
//					if((pit->type == type) && (pit->series == XString("M"))) {
//						XString inp_term = formatString("/%s/PFI0", it->c_str());
//						//Detects external clock source.
//						if(routeExternalClockSource(it->c_str(),  inp_term.c_str())) {
//							fprintf(stderr, "Reference Clock for PLL Set to %s\n", inp_term.c_str());
//							g_pciClockMaster = *it;
//							pcidevs.clear();
//							pcidevs.push_back(g_pciClockMaster);
//						}
//						break;
//					}
//					if(g_pciClockMaster.length())
//						break;
//				}
//			}
			for(std::deque<XString>::iterator it = pcidevs.begin(); it != pcidevs.end(); it++) {
				//M series only.
				//M series device has better time accuracy than S.
				CHECK_DAQMX_RET(DAQmxGetDevProductType(it->c_str(), buf, sizeof(buf)));
				XString type = buf;
				for(const ProductInfo *pit = sc_productInfoList; pit->type; pit++) {
					if((pit->type == type) && (pit->series == XString("M"))) {
						//Detects external clock source.
						fprintf(stderr, "20MHz Reference Clock exported from %s\n", it->c_str());
						XString ctrdev = formatString("%s/ctr1", it->c_str());
						//Continuous pulse train generation. Duty = 50%.
						CHECK_DAQMX_RET(DAQmxCreateTask("", &g_pciClockMasterTask));
						double freq = 20e6; //20MHz
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
			for(std::deque<XString>::iterator it = pcidevs.begin(); it != pcidevs.end(); it++) {
				CHECK_DAQMX_RET(DAQmxGetDevProductType(it->c_str(), buf, sizeof(buf)));
				XString type = buf;
				for(const ProductInfo *pit = sc_productInfoList; pit->type; pit++) {
					if((pit->type == type) && (pit->series == XString("S"))) {
						//S series device cannot export 20MHzTimebase freely.
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
	return false;
}
void
XNIDAQmxInterface::close() throw (XInterfaceError &) {
	m_productInfo = NULL;
	if(m_devname.length()) {
		m_devname.clear();

		XScopedLock<XMutex> lock(g_daqmx_mutex);
		g_daqmx_open_cnt--;
		if(g_daqmx_open_cnt == 0) {
			if(g_pciClockMasterTask != -1) {
				CHECK_DAQMX_RET(DAQmxClearTask(g_pciClockMasterTask));
			}
			g_pciClockMasterTask = -1;
			g_daqmx_sync_routes.clear();
		}
	}
}
