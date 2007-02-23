/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "nidaqdso.h"

#ifdef HAVE_NI_DAQMX

#include <qmessagebox.h>
#include <kmessagebox.h>
#include "xwavengraph.h"
#include <klocale.h>

#define TASK_UNDEF ((TaskHandle)-1)
#define NUM_MAX_CH 2
//---------------------------------------------------------------------------
XNIDAQmxDSO::XNIDAQmxDSO(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
 XNIDAQmxDriver<XDSO>(name, runtime, scalarentries, interfaces, thermometers, drivers),
 m_task(TASK_UNDEF)
{
	recordLength()->value(2000);
	timeWidth()->value(1e-2);
	average()->value(1);
	
    const char* sc[] = {"0.4", "1", "2", "4", "10", "20", "40", "84", 0L};
    for(int i = 0; sc[i]; i++)
    {
        vFullScale1()->add(sc[i]);
        vFullScale2()->add(sc[i]);
    }
    vFullScale1()->value("20");
    vFullScale2()->value("20");
    
	const void *FIRST_OF_MLOCK_MEMBER = &m_recordBuf;
	const void *LAST_OF_MLOCK_MEMBER = &m_acqCount;
	//Suppress swapping.
	mlock(FIRST_OF_MLOCK_MEMBER, (size_t)LAST_OF_MLOCK_MEMBER - (size_t)FIRST_OF_MLOCK_MEMBER);    
}
XNIDAQmxDSO::~XNIDAQmxDSO()
{
	clearAcquision();
}

void
XNIDAQmxDSO::open() throw (XInterface::XInterfaceError &)
{
	XScopedLock<XInterface> lock(*interface());
	m_running = false;
    char buf[2048];
    {
	    DAQmxGetDevAIPhysicalChans(interface()->devName(), buf, sizeof(buf));
	    std::deque<std::string> chans;
	    XNIDAQmxInterface::parseList(buf, chans);
	    for(std::deque<std::string>::iterator it = chans.begin(); it != chans.end(); it++) {
	        trace1()->add(it->c_str());
	        trace2()->add(it->c_str());
	        m_analogTrigSrc.push_back(*it);
	        trigSource()->add(it->c_str());
    	}
    }
    {
		//M series
	    const char* sc_m[] = {
	    "PFI0", "PFI1", "PFI2", "PFI3", "PFI4", "PFI5", "PFI6", "PFI7",
	    "PFI8", "PFI9", "PFI10", "PFI11", "PFI12", "PFI13", "PFI14", "PFI15",
//    	"RTSI0", "RTSI1", "RTSI2", "RTSI3", "RTSI4", "RTSI5", "RTSI6", "RTSI7", 
    	"Ctr0InternalOutput", "Ctr1InternalOutput",
    	//via PFI or RTSI
    	"ao/SampleClock",
    	"ao/StartTrigger",
    	"ao/PauseTrigger",
    	"Ctr0Source",
    	"Ctr0Gate",
    	"Ctr1Source",
    	"Ctr1Gate",
    	"FrequencyOutput",
    	"di/SampleClock",
    	"do/SampleClock",
    	 0L};
    	 //S series
	    const char* sc_s[] = {
	    //internally
	    "PFI0", "PFI1", "PFI2", "PFI3", "PFI4", "PFI5", "PFI6", "PFI7",
	    "PFI8", "PFI9",
//    	"RTSI0", "RTSI1", "RTSI2", "RTSI3", "RTSI4", "RTSI5", "RTSI6",
    	"Ctr0InternalOutput",
    	"OnboardClock",
    	//via PFI or RTSI
    	"ao/SampleClock",
    	"ao/StartTrigger",
    	"ao/PauseTrigger",
    	"Ctr0Source",
    	"Ctr0Gate",
    	"Ctr1Source",
    	"Ctr1Gate",
    	 0L};
    	 const char **sc = sc_s;
    	 if(interface()->productSeries() == std::string("M"))
    	 	sc = sc_m;
	    for(const char**it = sc; *it; it++) {
	    	std::string str(formatString("/%s/%s", interface()->devName(), *it));
	        trigSource()->add(str);
	    }
	    m_virtualTriggerList = XNIDAQmxInterface::VirtualTrigger::virtualTrigList();
	    if(m_virtualTriggerList) {
		    for(XNIDAQmxInterface::VirtualTrigger::VirtualTriggerList_it
		    	it = m_virtualTriggerList->begin(); it != m_virtualTriggerList->end(); it++) {
	    		if(shared_ptr<XNIDAQmxInterface::VirtualTrigger> vt = it->lock()) {
					for(unsigned int i = 0; i < vt->bits(); i++) {
				    	trigSource()->add(
				    		formatString("%s/line%d", vt->label(), i));
					}
	    		}
		    }
	    }
    }

	m_acqCount = 0;

	this->start();
	
	vOffset1()->setUIEnabled(false);
	vOffset2()->setUIEnabled(false);

//    createChannels();
}
void
XNIDAQmxDSO::close() throw (XInterface::XInterfaceError &)
{
	XScopedLock<XInterface> lock(*interface());
 	
	clearAcquision();
 	
    m_analogTrigSrc.clear();
    trace1()->clear();
    trace2()->clear();
    trigSource()->clear();
    
	interface()->stop();
}
void
XNIDAQmxDSO::clearAcquision() {
	XScopedLock<XInterface> lock(*interface());
	if(m_threadReadAI) {
		m_threadReadAI->terminate();
	}
 	XScopedLock<XRecursiveMutex> lock2(m_readMutex);
	
 	disableTrigger();

    if(m_task != TASK_UNDEF) {
	    DAQmxClearTask(m_task);
    }
	m_task = TASK_UNDEF;
}
void
XNIDAQmxDSO::disableTrigger()
{
	XScopedLock<XInterface> lock(*interface());
	ScopedReadAILock lockRead(*this);
	
	if(m_running) {
		m_running = false;
    	DAQmxStopTask(m_task);
	}
    if(m_task != TASK_UNDEF) {
	    DAQmxDisableStartTrig(m_task);
	    DAQmxDisableRefTrig(m_task);
    }
    
    m_preTriggerPos = 0;
    m_trigRoute.reset();
    //reset virtual trigger setup.
	if(m_virtualTrigger)
    	m_virtualTrigger->disconnect();
    m_lsnOnVirtualTrigStart.reset();
    m_virtualTrigger.reset();
}
void
XNIDAQmxDSO::setupTrigger()
{
	XScopedLock<XInterface> lock(*interface());
	ScopedReadAILock lockRead(*this);
	
	disableTrigger();
	
    unsigned int pretrig = lrint(*trigPos() / 100.0 * *recordLength());
	m_preTriggerPos = pretrig;
    
    std::string atrig;
    std::string dtrig;
    std::string src = trigSource()->to_str();

    if(std::find(m_analogTrigSrc.begin(), m_analogTrigSrc.end(), src)
         != m_analogTrigSrc.end()) {
         atrig = src;
    }
    else {
         dtrig = src;
    }
    
    int32 trig_spec = *trigFalling() ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope;
    
    //setup virtual trigger.
    if(m_virtualTriggerList) {
	    for(XNIDAQmxInterface::VirtualTrigger::VirtualTriggerList_it
	    	it = m_virtualTriggerList->begin(); it != m_virtualTriggerList->end(); it++) {
			if(shared_ptr<XNIDAQmxInterface::VirtualTrigger> vt = it->lock()) {
				for(unsigned int i = 0; i < vt->bits(); i++) {
		    		if(src == formatString("%s/line%d", vt->label(), i)) {
			    		m_virtualTrigger = vt;
			    		vt->connect(
			    			!*trigFalling() ? (1uL << i) : 0,
			    			*trigFalling() ? (1uL << i) : 0);
					    dtrig = vt->armTerm();
					    trig_spec = DAQmx_Val_RisingSlope;
					    pretrig = 0;				    
					    CHECK_DAQMX_RET(DAQmxSetReadOverWrite(m_task, DAQmx_Val_OverwriteUnreadSamps));
		    		}
	    		}
			}
	    }
    }
    
    //Small # of pretriggers is not allowed for ReferenceTrigger.
    if(!m_virtualTrigger && (pretrig < 2)) {
    	pretrig = 0;
		m_preTriggerPos = pretrig;
    }
    
    if(!pretrig) {
	    if(atrig.length()) {
	        CHECK_DAQMX_RET(DAQmxCfgAnlgEdgeStartTrig(m_task,
	            atrig.c_str(), trig_spec, *trigLevel()));
	    }
	    if(dtrig.length()) {
	        CHECK_DAQMX_RET(DAQmxCfgDigEdgeStartTrig(m_task,
	            dtrig.c_str(), trig_spec));
	    }
	    DAQmxSetReadOverWrite(m_task, DAQmx_Val_DoNotOverwriteUnreadSamps);
    }
    else {
	    if(atrig.length()) {
	        CHECK_DAQMX_RET(DAQmxCfgAnlgEdgeRefTrig(m_task,
	            atrig.c_str(), trig_spec, *trigLevel(), pretrig));
	    }
	    if(dtrig.length()) {
	        CHECK_DAQMX_RET(DAQmxCfgDigEdgeRefTrig(m_task,
	            dtrig.c_str(), trig_spec, pretrig));
	    }
	    CHECK_DAQMX_RET(DAQmxSetReadOverWrite(m_task, DAQmx_Val_OverwriteUnreadSamps));
    }
    
	startSequence();
}
void
XNIDAQmxDSO::setupTiming()
{
	XScopedLock<XInterface> lock(*interface());
	ScopedReadAILock lockRead(*this);

	if(m_running) {
		m_running = false;
    	DAQmxStopTask(m_task);
	}

	uInt32 num_ch;
    CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));	
    if(num_ch == 0) return;

	const unsigned int len = *recordLength();
	m_record.resize(len * NUM_MAX_CH);
	m_recordBuf.resize(len * NUM_MAX_CH);
	mlock(&m_record[0], m_record.size() * sizeof(tRawAI));
	mlock(&m_recordBuf[0], m_recordBuf.size() * sizeof(int32_t));    

    CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_task,
        NULL, // internal source
        len / *timeWidth(),
        DAQmx_Val_Rising,
//        m_virtualTrigger ? DAQmx_Val_ContSamps : DAQmx_Val_FiniteSamps,
        DAQmx_Val_FiniteSamps,
        len
        ));

    interface()->synchronizeClock(m_task);

    float64 rate;
//    CHECK_DAQMX_RET(DAQmxGetRefClkRate(m_task, &rate));
//	dbgPrint(QString("Reference Clk rate = %1.").arg(rate));
    CHECK_DAQMX_RET(DAQmxGetSampClkRate(m_task, &rate));
    m_interval = 1.0 / rate;
    
    setupTrigger();
}
void
XNIDAQmxDSO::createChannels()
{
	XScopedLock<XInterface> lock(*interface());
	ScopedReadAILock lockRead(*this);
 	
	clearAcquision();
	
    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_task));
	ASSERT(m_task != TASK_UNDEF);   
    
    if(*trace1() >= 0) {
        CHECK_DAQMX_RET(DAQmxCreateAIVoltageChan(m_task,
            trace1()->to_str().c_str(),
              "",
            DAQmx_Val_Cfg_Default,
            -atof(vFullScale1()->to_str().c_str()) / 2.0,
            atof(vFullScale1()->to_str().c_str()) / 2.0,
            DAQmx_Val_Volts,
            NULL
            ));

		//obtain range info.
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++)
			m_coeffAI[0][i] = 0.0;
		CHECK_DAQMX_RET(DAQmxGetAIDevScalingCoeff(m_task, 
            trace1()->to_str().c_str(),
			m_coeffAI[0], CAL_POLY_ORDER));
    }
    if(*trace2() >= 0) {
        CHECK_DAQMX_RET(DAQmxCreateAIVoltageChan(m_task,
            trace2()->to_str().c_str(),
              "",
            DAQmx_Val_Cfg_Default,
            -atof(vFullScale2()->to_str().c_str()) / 2.0,
            atof(vFullScale2()->to_str().c_str()) / 2.0,
            DAQmx_Val_Volts,
            NULL
            ));
		//obtain range info.
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++)
			m_coeffAI[1][i] = 0.0;
		CHECK_DAQMX_RET(DAQmxGetAIDevScalingCoeff(m_task, 
            trace2()->to_str().c_str(),
			m_coeffAI[1], CAL_POLY_ORDER));
    }

	{
/*		char chans[256];
		CHECK_DAQMX_RET(DAQmxGetTaskChannels(m_task, chans, sizeof(chans)));
		bool32 ret;
		CHECK_DAQMX_RET(DAQmxGetAIChanCalHasValidCalInfo(m_task, chans, &ret));
		if(!ret) {
			statusPrinter()->printMessage(KAME::i18n("Performing self calibration."));
	        QMessageBox::warning(g_pFrmMain, "KAME", KAME::i18n("Performing self calibration. Wait for minutes.") );
			CHECK_DAQMX_RET(DAQmxSelfCal(interface()->devName()));
			statusPrinter()->printMessage(KAME::i18n("Self calibration done."));
		}
*/	}

    setupTiming();
	
	m_threadReadAI.reset(new XThread<XNIDAQmxDSO>(shared_from_this(),
		 &XNIDAQmxDSO::executeReadAI));
	m_threadReadAI->resume();
	
	if(m_virtualTrigger) {
		uInt32 num_ch;
	    CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));	
	    if(num_ch > 0) {
		    CHECK_DAQMX_RET(DAQmxStartTask(m_task));
		    m_running = true;
	    }
	}
}
void
XNIDAQmxDSO::onVirtualTrigStart(const shared_ptr<XNIDAQmxInterface::VirtualTrigger> &) {
	XScopedLock<XInterface> lock(*interface());
	ScopedReadAILock lockRead(*this);

	if(m_running) {
		m_running = false;
    	DAQmxStopTask(m_task);
	}

    startSequence();

	uInt32 num_ch;
    CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));	
    if(num_ch > 0) {
	    CHECK_DAQMX_RET(DAQmxStartTask(m_task));
	    m_running = true;
    }
}
void
XNIDAQmxDSO::onForceTriggerTouched(const shared_ptr<XNode> &)
{
	XScopedLock<XInterface> lock(*interface());
	ScopedReadAILock lockRead(*this);

	disableTrigger();

    CHECK_DAQMX_RET(DAQmxStartTask(m_task));
    m_running = true;
}
inline bool
XNIDAQmxDSO::tryReadAISuspend() {
	if(m_suspendRead) {
		m_readMutex.unlock();
		while(m_suspendRead) msecsleep(10);
		m_readMutex.lock();
		return true;
	}
	return false;
}
void *
XNIDAQmxDSO::executeReadAI(const atomic<bool> &terminated)
{
	while(!terminated) {
	    try {
			acquire(terminated);
	    }
	    catch (XInterface::XInterfaceError &e) {
	    	e.print(getLabel());
	    }
	}
	return NULL;
}
void
XNIDAQmxDSO::acquire(const atomic<bool> &terminated)
{
    uInt32 num_ch;
	unsigned int cnt = 0;
	{
		XScopedLock<XRecursiveMutex> lock(m_readMutex);
	
		const unsigned int size = m_record.size() / NUM_MAX_CH;
	
		 if(!m_running) {
			tryReadAISuspend();
			msecsleep(10);
			return;
		 }
	
	    CHECK_DAQMX_RET(DAQmxGetReadNumChans(m_task, &num_ch));
	    if(num_ch == 0) {
			tryReadAISuspend();
			msecsleep(10);
			return;
		 }
	    
	    float64 freq = 1.0 / m_interval;
	
		if(m_virtualTrigger) {
			shared_ptr<XNIDAQmxInterface::VirtualTrigger> &vt(m_virtualTrigger);
			while(!terminated) {
				if(tryReadAISuspend())
					return;
				uInt64 total_samps;
				CHECK_DAQMX_RET(DAQmxGetReadTotalSampPerChanAcquired(m_task, &total_samps));
				uint64_t lastcnt = vt->front(freq);
				if(lastcnt && (lastcnt < total_samps)) {
					uInt32 bufsize;
					CHECK_DAQMX_RET(DAQmxGetBufInputBufSize(m_task, &bufsize));
					if(total_samps - lastcnt + m_preTriggerPos > bufsize * 4 / 5) {
						vt->pop();
						gWarnPrint(KAME::i18n("Buffer Overflow."));
						continue;
					}
	//				uInt64 currpos;
	//				CHECK_DAQMX_RET(DAQmxGetReadCurrReadPos(m_task, &currpos));
	//				int32 offset = ((lastcnt - currpos) % (uInt64)bufsize) - m_preTriggerPos;
					int32 offset = ((lastcnt - m_preTriggerPos + (uInt64)bufsize) % (uInt64)bufsize);
				    CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, offset));
					vt->pop();
					fprintf(stderr, "hit!\n");
					break;
				}
				usleep(lrint(1e6 * size * m_interval / 2));
			}
		}
		else {
			if(m_preTriggerPos) {
				CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_FirstPretrigSamp));
			}
			else {
				CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_CurrReadPos));
			}
				
		    CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, 0));
		}
		if(terminated)
			return;
	
		const unsigned int num_samps = size; //std::min(size, 1024u);
		for(; cnt < size;) {
			int32 samps;
			samps = std::min(size - cnt, num_samps);
			while(!terminated) {
				if(tryReadAISuspend())
					return;
			uInt32 space;
				int ret = DAQmxGetReadAvailSampPerChan(m_task, &space);
				if(!ret && (space >= (uInt32)samps))
					break;
				usleep(lrint(1e6 * (samps - space) * m_interval));
			}
			if(terminated)
				return;
		    CHECK_DAQMX_RET(DAQmxReadBinaryI16(m_task, samps,
		        0.01, DAQmx_Val_GroupByScanNumber,
		        &m_recordBuf[cnt * num_ch], samps * num_ch, &samps, NULL
		        ));
		    cnt += samps;
			if(m_preTriggerPos && !m_virtualTrigger) {
				CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, cnt));
			}
		}
	} //end of readMutex

	XScopedLock<XInterface> lock(*interface());
	
	const unsigned int av = *average();
	const bool sseq = *singleSequence();
	
    if(!sseq || ((unsigned int)m_accumCount < av)) {
		if(!m_virtualTrigger) {
			if(m_running) {
				m_running = false;
		    	DAQmxStopTask(m_task);
			}
		    CHECK_DAQMX_RET(DAQmxStartTask(m_task));
		    m_running = true;
		}
    }

    m_recordLength = std::min(cnt, m_recordLength);
      for(unsigned int i = 0; i < cnt * num_ch; i++) {
        m_record[i] += m_recordBuf[i];
      }
    m_acqCount++;
    m_accumCount++;

    while(!sseq && (av <= m_record_av.size()) && !m_record_av.empty())  {
      for(unsigned int i = 0; i < m_record.size(); i++) {
        m_record[i] -= m_record_av.front()[i];
      }
      m_record_av.pop_front();
      m_accumCount--;
    }
    
    if(!sseq) {
      m_record_av.push_back(m_recordBuf);
    }
}
void
XNIDAQmxDSO::startSequence()
{
	XScopedLock<XInterface> lock(*interface());
	ScopedReadAILock lockRead(*this);

    m_acqCount = 0;
    m_accumCount = 0;
	std::fill(m_record.begin(), m_record.end(), 0);
	m_record_av.clear();   	
	m_recordLength = m_record.size() / NUM_MAX_CH;
    
	if(m_virtualTrigger) {
		uInt32 bufsize = std::max(m_recordLength * 4, (unsigned int)lrint(0.1 / m_interval));
		CHECK_DAQMX_RET(DAQmxCfgInputBuffer(m_task, bufsize));
		m_virtualTrigger->setBlankTerm(m_interval * m_recordLength);
		uInt64 total_samps;
		CHECK_DAQMX_RET(DAQmxGetReadTotalSampPerChanAcquired(m_task, &total_samps));
		m_virtualTrigger->clear(total_samps, 1.0 / m_interval);
		if(!m_lsnOnVirtualTrigStart)
			m_lsnOnVirtualTrigStart = m_virtualTrigger->onStart().connectWeak(false,
				shared_from_this(), &XNIDAQmxDSO::onVirtualTrigStart);

	    CHECK_DAQMX_RET(DAQmxSetSampQuantSampMode(m_task, DAQmx_Val_ContSamps));
		CHECK_DAQMX_RET(DAQmxSetReadRelativeTo(m_task, DAQmx_Val_FirstSample));
	}
	else {
		if(m_running) {
			m_running = false;
		    if(m_task != TASK_UNDEF)
		    	DAQmxStopTask(m_task);
		}
	    CHECK_DAQMX_RET(DAQmxSetSampQuantSampMode(m_task, DAQmx_Val_FiniteSamps));
		uInt32 num_ch;
	    CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));	
	    if(num_ch > 0) {
		    CHECK_DAQMX_RET(DAQmxStartTask(m_task));
		    m_running = true;
	    }
	}
//    CHECK_DAQMX_RET(DAQmxSetReadOffset(m_task, 0));
}

int
XNIDAQmxDSO::acqCount(bool *seq_busy)
{
	XScopedLock<XInterface> lock(*interface());
    *seq_busy = ((unsigned int)m_acqCount < *average());
    return m_acqCount;
}

double
XNIDAQmxDSO::getTimeInterval()
{
	return m_interval;
}

inline float64
XNIDAQmxDSO::aiRawToVolt(const float64 *pcoeff, float64 raw)
{
	float64 x = 1.0;
	float64 y = 0.0;
	for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
		y += *(pcoeff++) * x;
		x *= raw;
	}
	return y;
}

void
XNIDAQmxDSO::getWave(std::deque<std::string> &)
{
	XScopedLock<XInterface> lock(*interface());

 	if(m_accumCount == 0) throw XInterface::XInterfaceError(KAME::i18n("No sample."), __FILE__, __LINE__);
    uInt32 num_ch;
    CHECK_DAQMX_RET(DAQmxGetReadNumChans(m_task, &num_ch));
//    bool32 overload;
//    CHECK_DAQMX_RET(DAQmxGetReadOverloadedChansExist(m_task, &overload), "Overload");
//    if(overload) {
//    	gWarnPrint(getLabel() + KAME::i18n(": Overload Detected!"));
//    }
    uInt32 len;
//    CHECK_DAQMX_RET(DAQmxGetReadAvailSampPerChan(m_task, &len), "SampPerChan");
	len = m_recordLength;
    
    char buf[2048];
    CHECK_DAQMX_RET(DAQmxGetReadChannelsToRead(m_task, buf, sizeof(buf)));
	
    push((uint32_t)num_ch);
    push((uint32_t)m_preTriggerPos);
    push((uint32_t)len);
    push((uint32_t)m_accumCount);
    push((double)m_interval);
    for(unsigned int ch = 0; ch < num_ch; ch++) {
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
			push((double)m_coeffAI[ch][i]);
		}
    }
    int32_t *p = &m_record[0];
    const unsigned int size = len * num_ch;
    for(unsigned int i = 0; i < size; i++)
    	push(*p++);
    std::string str(buf);
    rawData().insert(rawData().end(), str.begin(), str.end());
    push((char)0);
}
void
XNIDAQmxDSO::convertRaw() throw (XRecordError&)
{
	const unsigned int num_ch = pop<uint32_t>();
	const unsigned int pretrig = pop<uint32_t>();
	const unsigned int len = pop<uint32_t>();
	const unsigned int accumCount = pop<uint32_t>();
	const double interval = pop<double>();

	setParameters(num_ch, - (double)pretrig * interval, interval, len);
	
	double *wave[NUM_MAX_CH];
	float64 coeff[NUM_MAX_CH][CAL_POLY_ORDER];
	for(unsigned int j = 0; j < num_ch; j++) {
		for(unsigned int i = 0; i < CAL_POLY_ORDER; i++) {
			coeff[j][i] = pop<double>();
		}
		
		wave[j] = waveDisp(j);
    }

	const float64 prop = 1.0 / accumCount;
	//for optimaization: -funroll-loops.
	switch(num_ch) {
	case 1:
	    for(unsigned int i = 0; i < len; i++) {
    	  *(wave[0])++ = aiRawToVolt(coeff[0], pop<int32_t>() * prop);
    	}
    	break;
	case 2:
	    for(unsigned int i = 0; i < len; i++) {
    	  *(wave[0])++ = aiRawToVolt(coeff[0], pop<int32_t>() * prop);
    	  *(wave[1])++ = aiRawToVolt(coeff[1], pop<int32_t>() * prop);
    	}
    	break;
	default:
	    for(unsigned int i = 0; i < len; i++) {
 			for(unsigned int j = 0; j < num_ch; j++)
	        	  *(wave[j])++ = aiRawToVolt(coeff[j], pop<int32_t>() * prop);
    	}
    }
}

void 
XNIDAQmxDSO::onAverageChanged(const shared_ptr<XValueNodeBase> &) {
	startSequence();
}

void
XNIDAQmxDSO::onSingleChanged(const shared_ptr<XValueNodeBase> &) {
	startSequence();
}
void
XNIDAQmxDSO::onTrigPosChanged(const shared_ptr<XValueNodeBase> &) {
    setupTrigger();
}
void
XNIDAQmxDSO::onTrigSourceChanged(const shared_ptr<XValueNodeBase> &) {
    setupTrigger();
}
void
XNIDAQmxDSO::onTrigLevelChanged(const shared_ptr<XValueNodeBase> &) {
    setupTrigger();
}
void
XNIDAQmxDSO::onTrigFallingChanged(const shared_ptr<XValueNodeBase> &) {
    setupTrigger();
}
void
XNIDAQmxDSO::onTimeWidthChanged(const shared_ptr<XValueNodeBase> &) {
    setupTiming();
}
void
XNIDAQmxDSO::onTrace1Changed(const shared_ptr<XValueNodeBase> &) {
    createChannels();
}
void
XNIDAQmxDSO::onTrace2Changed(const shared_ptr<XValueNodeBase> &) {
    createChannels();
}
void
XNIDAQmxDSO::onVFullScale1Changed(const shared_ptr<XValueNodeBase> &) {
    createChannels();
}
void
XNIDAQmxDSO::onVFullScale2Changed(const shared_ptr<XValueNodeBase> &) {
    createChannels();
}
void
XNIDAQmxDSO::onVOffset1Changed(const shared_ptr<XValueNodeBase> &) {
}
void
XNIDAQmxDSO::onVOffset2Changed(const shared_ptr<XValueNodeBase> &) {
}
void
XNIDAQmxDSO::onRecordLengthChanged(const shared_ptr<XValueNodeBase> &) {
    setupTiming();
}

#endif //HAVE_NI_DAQMX
