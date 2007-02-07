#include "nidaqdso.h"

#ifdef HAVE_NI_DAQMX

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
    vFullScale1()->value("84");
    vFullScale2()->value("84");
}
XNIDAQmxDSO::~XNIDAQmxDSO()
{
	if(m_task != TASK_UNDEF)
    DAQmxClearTask(m_task);
}
void
XNIDAQmxDSO::open() throw (XInterface::XInterfaceError &)
{
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
	    DAQmxGetDevDILines(interface()->devName(), buf, sizeof(buf));
	    std::deque<std::string> chans;
	    XNIDAQmxInterface::parseList(buf, chans);
	    for(std::deque<std::string>::iterator it = chans.begin(); it != chans.end(); it++) {
	        m_digitalTrigSrc.push_back(*it);
	        trigSource()->add(it->c_str());
    	}
    }

//    createChannels();
//    setupTrigger();

	this->start();
}
void
XNIDAQmxDSO::close() throw (XInterface::XInterfaceError &)
{
 	XScopedLock<XInterface> lock(*interface());
	if(m_task != TASK_UNDEF)
	    DAQmxClearTask(m_task);
	m_task = TASK_UNDEF;
    m_analogTrigSrc.clear();
    m_digitalTrigSrc.clear();
    trace1()->clear();
    trace2()->clear();
    trigSource()->clear();
    
	interface()->stop();
}
void
XNIDAQmxDSO::setupTrigger()
{
 	XScopedLock<XInterface> lock(*interface());
 	{
 		char buf[2048];
	    CHECK_DAQMX_RET(DAQmxGetTaskChannels(m_task, buf, sizeof(buf)), "");
	    printf("%s\n", buf);
	    CHECK_DAQMX_RET(DAQmxGetTaskDevices(m_task, buf, sizeof(buf)), "");
	    printf("%s\n", buf);
 	}

    CHECK_DAQMX_RET(DAQmxStopTask(m_task), "Stop Task");
    
	printf("trigsource:%s\n", trigSource()->to_str().c_str());
    CHECK_DAQMX_RET(DAQmxDisableStartTrig(m_task), "");
    CHECK_DAQMX_RET(DAQmxDisableRefTrig(m_task), "");
	
    unsigned int pretrig = lrintl(*trigPos() / 100.0 * *recordLength());
    if(pretrig < 2) {
	    if(std::find(m_analogTrigSrc.begin(), m_analogTrigSrc.end(), trigSource()->to_str())
	         != m_analogTrigSrc.end()) {
	        CHECK_DAQMX_RET(DAQmxCfgAnlgEdgeStartTrig(m_task,
	            trigSource()->to_str().c_str(),
	            *trigFalling() ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope,
	            *trigLevel()),
	            "Trigger Setup");
	    }
	    if(std::find(m_digitalTrigSrc.begin(), m_digitalTrigSrc.end(), trigSource()->to_str())
	         != m_digitalTrigSrc.end()) {
	        CHECK_DAQMX_RET(DAQmxCfgDigEdgeStartTrig(m_task,
	            trigSource()->to_str().c_str(),
	            *trigFalling() ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope),
	            "Trigger Setup");
	    }
    }
    else {
	    if(std::find(m_analogTrigSrc.begin(), m_analogTrigSrc.end(), trigSource()->to_str())
	         != m_analogTrigSrc.end()) {
	        CHECK_DAQMX_RET(DAQmxCfgAnlgEdgeRefTrig(m_task,
	            trigSource()->to_str().c_str(),
	            *trigFalling() ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope,
	            *trigLevel(),
	            pretrig),
	            "Trigger Setup");
	    }
	    if(std::find(m_digitalTrigSrc.begin(), m_digitalTrigSrc.end(), trigSource()->to_str())
	         != m_digitalTrigSrc.end()) {
	        CHECK_DAQMX_RET(DAQmxCfgDigEdgeRefTrig(m_task,
	            trigSource()->to_str().c_str(),
	            *trigFalling() ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope,
	            pretrig),
	            "Trigger Setup");
	    }
    }
    
	startSequence();
}
void
XNIDAQmxDSO::setupTiming()
{
 	XScopedLock<XInterface> lock(*interface());

	uInt32 num_ch;
    CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch), "");	
    if(num_ch == 0) return;

    CHECK_DAQMX_RET(DAQmxStopTask(m_task), "Stop Task");

	unsigned int len = *recordLength();
	m_record.resize(len * NUM_MAX_CH);
	m_record_buf.resize(len * NUM_MAX_CH);
    CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_task,
        NULL, // internal source
        len / *timeWidth(),
        DAQmx_Val_Rising,
        DAQmx_Val_FiniteSamps,
        len
        ), "Set Timing");
    float64 rate;
    CHECK_DAQMX_RET(DAQmxGetSampClkRate(m_task, &rate
    	), "Get Rate");
    m_interval = 1.0 / rate;
    
	startSequence();
}
void
XNIDAQmxDSO::createChannels()
{
 	XScopedLock<XInterface> lock(*interface());
	if(m_task != TASK_UNDEF) {
	    CHECK_DAQMX_RET(DAQmxStopTask(m_task), "Stop Task");
	    CHECK_DAQMX_RET(DAQmxClearTask(m_task), "Clear Task");
	}
	m_task = TASK_UNDEF;
    CHECK_DAQMX_RET(DAQmxCreateTask("", &m_task), "Task Creation");
	ASSERT(m_task != TASK_UNDEF);   
    
    if(*trace1() >= 0) {
    	 printf("%s\n", trace1()->to_str().c_str());
        CHECK_DAQMX_RET(DAQmxCreateAIVoltageChan(m_task,
            trace1()->to_str().c_str(),
              "",
            DAQmx_Val_Cfg_Default,
            -atof(vFullScale1()->to_str().c_str()) / 2.0,
            atof(vFullScale1()->to_str().c_str()) / 2.0,
            DAQmx_Val_Volts,
            NULL
            ), "Channel Creation");
    }
    if(*trace2() >= 0) {
    	 printf("%s\n", trace2()->to_str().c_str());
        CHECK_DAQMX_RET(DAQmxCreateAIVoltageChan(m_task,
            trace2()->to_str().c_str(),
              "",
            DAQmx_Val_Cfg_Default,
            -atof(vFullScale2()->to_str().c_str()) / 2.0,
            atof(vFullScale2()->to_str().c_str()) / 2.0,
            DAQmx_Val_Volts,
            NULL
            ), "Channel Creation");
    }

    m_bPollMode = (DAQmxRegisterDoneEvent(m_task, 0, &XNIDAQmxDSO::_acqCallBack, this) < 0);
    if(m_bPollMode)
    	dbgPrint(getLabel() + ": Polling mode enabled.");

    setupTiming();
}
void 
XNIDAQmxDSO::onAverageChanged(const shared_ptr<XValueNodeBase> &) {
}

void
XNIDAQmxDSO::onSingleChanged(const shared_ptr<XValueNodeBase> &)
{
}
void
XNIDAQmxDSO::onTrigPosChanged(const shared_ptr<XValueNodeBase> &)
{
    setupTrigger();
}
void
XNIDAQmxDSO::onTrigSourceChanged(const shared_ptr<XValueNodeBase> &)
{
    setupTrigger();
}
void
XNIDAQmxDSO::onTrigLevelChanged(const shared_ptr<XValueNodeBase> &)
{
    setupTrigger();
}
void
XNIDAQmxDSO::onTrigFallingChanged(const shared_ptr<XValueNodeBase> &)
{
    setupTrigger();
}
void
XNIDAQmxDSO::onTimeWidthChanged(const shared_ptr<XValueNodeBase> &)
{
    setupTiming();
}
void
XNIDAQmxDSO::onTrace1Changed(const shared_ptr<XValueNodeBase> &)
{
    createChannels();
}
void
XNIDAQmxDSO::onTrace2Changed(const shared_ptr<XValueNodeBase> &)
{
    createChannels();
}
void
XNIDAQmxDSO::onVFullScale1Changed(const shared_ptr<XValueNodeBase> &)
{
    createChannels();
}
void
XNIDAQmxDSO::onVFullScale2Changed(const shared_ptr<XValueNodeBase> &)
{
    createChannels();
}
void
XNIDAQmxDSO::onVOffset1Changed(const shared_ptr<XValueNodeBase> &)
{
}
void
XNIDAQmxDSO::onVOffset2Changed(const shared_ptr<XValueNodeBase> &)
{
}
void
XNIDAQmxDSO::onRecordLengthChanged(const shared_ptr<XValueNodeBase> &)
{
    setupTiming();
}
void
XNIDAQmxDSO::onForceTriggerTouched(const shared_ptr<XNode> &)
{
 	XScopedLock<XInterface> lock(*interface());
//    CHECK_DAQMX_RET(DAQmxSendSoftwareTrigger(m_task, DAQmx_Val_AdvanceTrigger), "");
    CHECK_DAQMX_RET(DAQmxStopTask(m_task), "");
    CHECK_DAQMX_RET(DAQmxDisableStartTrig(m_task), "");
    CHECK_DAQMX_RET(DAQmxDisableRefTrig(m_task), "");
    CHECK_DAQMX_RET(DAQmxStartTask(m_task), "");
}
int32
XNIDAQmxDSO::_acqCallBack(TaskHandle task, int32 status, void *data)
{
    XNIDAQmxDSO *obj = reinterpret_cast<XNIDAQmxDSO*>(data);
    return obj->acqCallBack(task, status);
}
int32
XNIDAQmxDSO::acqCallBack(TaskHandle task, int32 status)
{
 	XScopedLock<XInterface> lock(*interface());
    if(status) {
    	gErrPrint(XNIDAQmxInterface::getNIDAQmxErrMessage(status));
    	if(status < 0) return status;
    }
    try {
		acquire(task);
    }
    catch (XInterface::XInterfaceError &e) {
    	e.print(getLabel());
    }
    return status;
}
void
XNIDAQmxDSO::acquire(TaskHandle task)
{
    uInt32 num_ch;
    CHECK_DAQMX_RET(DAQmxGetTaskNumChans(task, &num_ch), "");	
    if(num_ch == 0) return;
    CHECK_DAQMX_RET(DAQmxGetReadNumChans(task, &num_ch), "");
    if(num_ch == 0) return;
    uInt32 len;
    CHECK_DAQMX_RET(DAQmxGetReadAvailSampPerChan(m_task, &len), "SampPerChan");
    if(len < m_record.size() / NUM_MAX_CH) return;
    int32 cnt;
    CHECK_DAQMX_RET(DAQmxReadAnalogF64(task, DAQmx_Val_Auto,
        0, DAQmx_Val_GroupByChannel,
        &m_record_buf[0], m_record_buf.size(), &cnt, NULL
        ), "Read");
    ASSERT(cnt <= len);
    m_record_length = cnt;
      for(unsigned int i = 0; i < m_record.size(); i++) {
        m_record[i] += m_record_buf[i];
      }
    m_acqCount++;
    m_accumCount++;

	unsigned int av = *average();
	bool sseq = *singleSequence();
	
    while(!sseq && (av <= m_record_av.size()) && !m_record_av.empty())  {
      for(unsigned int i = 0; i < m_record.size(); i++) {
        m_record[i] -= m_record_av.front()[i];
      }
      m_record_av.pop_front();
      m_accumCount--;
    }
    
    if(!sseq) {
      m_record_av.push_back(m_record_buf);
    }

    if(!sseq || ((unsigned int)m_accumCount < av)) {
	    CHECK_DAQMX_RET(DAQmxStartTask(m_task), "Start Task");
    }
}
void
XNIDAQmxDSO::startSequence()
{
 	XScopedLock<XInterface> lock(*interface());
    CHECK_DAQMX_RET(DAQmxStopTask(m_task), "Stop Task");
    m_acqCount = 0;
    m_accumCount = 0;
	std::fill(m_record.begin(), m_record.end(), 0.0);
	m_record_av.clear();   	

	uInt32 num_ch;
    CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch), "");	
    if(num_ch > 0) {
	    CHECK_DAQMX_RET(DAQmxStartTask(m_task), "Start Task");
    }
}

int
XNIDAQmxDSO::acqCount(bool *seq_busy)
{
	if(m_bPollMode) {
	 	XScopedLock<XInterface> lock(*interface());
	    if(DAQmxWaitUntilTaskDone(m_task, 0.5) < 0) {
	    //time elapsed.
	    }
	    else {
	    bool32 done;
	    	CHECK_DAQMX_RET(DAQmxGetTaskComplete(m_task, &done), "");
	    	if(done) {
		    //task done.
		    	acquire(m_task);
	    	}
	    }
	}
    *seq_busy = ((unsigned int)m_acqCount < *average());
    return m_acqCount;
}

double
XNIDAQmxDSO::getTimeInterval()
{
	return m_interval;
}

void
XNIDAQmxDSO::getWave(std::deque<std::string> &)
{
 	XScopedLock<XInterface> lock(*interface());
 	if(m_accumCount == 0) return;
    uInt32 num_ch;
    CHECK_DAQMX_RET(DAQmxGetReadNumChans(m_task, &num_ch), "# of ch");
//    bool32 overload;
//    CHECK_DAQMX_RET(DAQmxGetReadOverloadedChansExist(m_task, &overload), "Overload");
//    if(overload) {
//    	gWarnPrint(getLabel() + KAME::i18n(": Overload Detected!"));
//    }
    uInt32 len;
//    CHECK_DAQMX_RET(DAQmxGetReadAvailSampPerChan(m_task, &len), "SampPerChan");
	len = m_record_length;
    
    char buf[2048];
    CHECK_DAQMX_RET(DAQmxGetReadChannelsToRead(m_task, buf, sizeof(buf)), "");
    
    uInt32 pretrig;
    CHECK_DAQMX_RET(DAQmxGetRefTrigPretrigSamples(m_task, &pretrig), "");
    
    push((uint32_t)num_ch);
    push((uint32_t)pretrig);
    push((uint32_t)len);
    push((uint32_t)m_accumCount);
    push((double)m_interval);
    double coeff =  1.0 / m_accumCount;
    float64 *p = &m_record[0];
    for(unsigned int ch = 0; ch < num_ch; ch++) {
	    for(unsigned int i = 0; i < len; i++) {
	    	push((double)*(p++) * coeff);
	    }
    }
    std::string str(buf);
    rawData().insert(rawData().end(), str.begin(), str.end());
    push((char)0);
}
void
XNIDAQmxDSO::convertRaw() throw (XRecordError&)
{
	unsigned int num_ch = pop<uint32_t>();
	unsigned int pretrig = pop<uint32_t>();
	unsigned int len = pop<uint32_t>();
	unsigned int accumCount = pop<uint32_t>();
	double interval = pop<double>();

	printf("%d %f %f %d\n", num_ch, - pretrig * interval, interval, len);
	setRecordDim(num_ch, - pretrig * interval, interval, len);
	
  for(unsigned int j = 0; j < num_ch; j++)
    {
      double *wave = waveRecorded(j);
      for(unsigned int i = 0; i < len; i++)
		{
        	  *(wave++) = pop<double>();
		}
    }
}

#endif //HAVE_NI_DAQMX
