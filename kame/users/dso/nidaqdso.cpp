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
    {
	    const char* sc[] = {
	    "PFI0", "PFI1", "PFI2", "PFI3", "PFI4", "PFI5", "PFI6", "PFI7",
    	"RTSI0", "RTSI1", "RTSI2", "RTSI3", "RTSI4", "RTSI5", "RTSI6",
    	"crt0", "crt1",
    	 0L};
	    for(int i = 0; sc[i]; i++)
	    {
	    	QString str(QString("/%1/%2").arg(interface()->devName()).arg(sc[i]));
	        trigSource()->add(str);
//	        m_digitalTrigSrc.push_back(str);
	    }
    }

	m_acqCount = 0;
//    createChannels();
//    setupTrigger();

	this->start();
	
	vOffset1()->setUIEnabled(false);
	vOffset2()->setUIEnabled(false);
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
    m_trigRoute.reset();
    
	interface()->stop();
}
void
XNIDAQmxDSO::setupTrigger()
{
 	XScopedLock<XInterface> lock(*interface());

    CHECK_DAQMX_RET(DAQmxStopTask(m_task));
    CHECK_DAQMX_RET(DAQmxDisableStartTrig(m_task));
    CHECK_DAQMX_RET(DAQmxDisableRefTrig(m_task));
    m_trigRoute.reset();
	
    unsigned int pretrig = lrint(*trigPos() / 100.0 * *recordLength());
    const char *atrig = 0;
    const char *dtrig = 0;
    std::string src = trigSource()->to_str();
//    if(!src.length()) return;
    if(std::find(m_analogTrigSrc.begin(), m_analogTrigSrc.end(), src)
         != m_analogTrigSrc.end()) {
         atrig = src.c_str();
    }
    if(std::find(m_digitalTrigSrc.begin(), m_digitalTrigSrc.end(), src)
         != m_digitalTrigSrc.end()) {
    	dtrig = QString("/%1/PFI0").arg(interface()->devName());
    	m_trigRoute.reset(new XNIDAQmxInterface::XNIDAQmxRoute(src.c_str(), dtrig));
    }
    if(!atrig && !dtrig && src.length()) {
         dtrig = src.c_str();
    }
    
    if(pretrig < 2) {
	    if(atrig) {
	        CHECK_DAQMX_RET(DAQmxCfgAnlgEdgeStartTrig(m_task,
	            atrig,
	            *trigFalling() ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope,
	            *trigLevel()));
	    }
	    if(dtrig) {
	        CHECK_DAQMX_RET(DAQmxCfgDigEdgeStartTrig(m_task,
	            dtrig,
	            *trigFalling() ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope));
	    }
//		CHECK_DAQMX_RET(DAQmxSetStartTrigRetriggerable(m_task, true), "");
    }
    else {
	    if(atrig) {
	        CHECK_DAQMX_RET(DAQmxCfgAnlgEdgeRefTrig(m_task,
	            atrig,
	            *trigFalling() ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope,
	            *trigLevel(),
	            pretrig));
	    }
	    if(dtrig) {
	        CHECK_DAQMX_RET(DAQmxCfgDigEdgeRefTrig(m_task,
	            dtrig,
	            *trigFalling() ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope,
	            pretrig));
	    }
//		CHECK_DAQMX_RET(DAQmxSetRefTrigRetriggerable(m_task, true), "");
    }
    
	startSequence();
}
void
XNIDAQmxDSO::setupTiming()
{
 	XScopedLock<XInterface> lock(*interface());

	uInt32 num_ch;
    CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));	
    if(num_ch == 0) return;

    CHECK_DAQMX_RET(DAQmxStopTask(m_task));

	unsigned int len = *recordLength();
	m_record.resize(len * NUM_MAX_CH);
	m_record_buf.resize(len * NUM_MAX_CH);
    CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_task,
        NULL, // internal source
        len / *timeWidth(),
        DAQmx_Val_Rising,
        DAQmx_Val_FiniteSamps,
        len
        ));
    float64 rate;
//    CHECK_DAQMX_RET(DAQmxGetRefClkRate(m_task, &rate));
//	dbgPrint(QString("Reference Clk rate = %1.").arg(rate));
    CHECK_DAQMX_RET(DAQmxGetSampClkRate(m_task, &rate));
    m_interval = 1.0 / rate;
}
void
XNIDAQmxDSO::createChannels()
{
 	XScopedLock<XInterface> lock(*interface());
	if(m_task != TASK_UNDEF) {
	    CHECK_DAQMX_RET(DAQmxStopTask(m_task));
	    CHECK_DAQMX_RET(DAQmxClearTask(m_task));
	}
	m_task = TASK_UNDEF;
	
	m_acqCount = 0;
	
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
    }

    m_bPollMode = (DAQmxRegisterDoneEvent(m_task, 0, &XNIDAQmxDSO::_acqCallBack, this) < 0);
    if(m_bPollMode)
    	dbgPrint(getLabel() + ": Polling mode enabled.");

    setupTiming();
	 setupTrigger();
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
	startSequence();
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
	startSequence();
}
void
XNIDAQmxDSO::onForceTriggerTouched(const shared_ptr<XNode> &)
{
 	XScopedLock<XInterface> lock(*interface());
//    CHECK_DAQMX_RET(DAQmxSendSoftwareTrigger(m_task, DAQmx_Val_AdvanceTrigger), "");
    CHECK_DAQMX_RET(DAQmxStopTask(m_task));

    int32 trigtype;
	CHECK_DAQMX_RET(DAQmxGetRefTrigType(m_task, &trigtype));
	if(trigtype != DAQmx_Val_None) {
	    CHECK_DAQMX_RET(DAQmxDisableRefTrig(m_task));
	}
	CHECK_DAQMX_RET(DAQmxGetStartTrigType(m_task, &trigtype));
	if(trigtype != DAQmx_Val_None) {
	    CHECK_DAQMX_RET(DAQmxDisableStartTrig(m_task));
	}
    CHECK_DAQMX_RET(DAQmxStartTask(m_task));
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
    CHECK_DAQMX_RET(DAQmxGetTaskNumChans(task, &num_ch));	
    if(num_ch == 0) return;
    CHECK_DAQMX_RET(DAQmxGetReadNumChans(task, &num_ch));
    if(num_ch == 0) return;
    uInt32 len;
    CHECK_DAQMX_RET(DAQmxGetReadAvailSampPerChan(m_task, &len));
    if(len < m_record.size() / NUM_MAX_CH) return;
    int32 cnt;
    CHECK_DAQMX_RET(DAQmxReadAnalogF64(task, DAQmx_Val_Auto,
        0, DAQmx_Val_GroupByChannel,
        &m_record_buf[0], m_record_buf.size(), &cnt, NULL
        ));
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
	    CHECK_DAQMX_RET(DAQmxStopTask(m_task));
	    CHECK_DAQMX_RET(DAQmxStartTask(m_task));
    }
}
void
XNIDAQmxDSO::startSequence()
{
 	XScopedLock<XInterface> lock(*interface());

    CHECK_DAQMX_RET(DAQmxStopTask(m_task));

    if(*singleSequence()) {
	    m_acqCount = 0;
	    m_accumCount = 0;
		std::fill(m_record.begin(), m_record.end(), 0.0);
		m_record_av.clear();   	
    }

	uInt32 num_ch;
    CHECK_DAQMX_RET(DAQmxGetTaskNumChans(m_task, &num_ch));	
    if(num_ch > 0) {
	    CHECK_DAQMX_RET(DAQmxStartTask(m_task));
    }
//	setupTrigger();
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
	    	CHECK_DAQMX_RET(DAQmxGetTaskComplete(m_task, &done));
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
	len = m_record_length;
    
    char buf[2048];
    CHECK_DAQMX_RET(DAQmxGetReadChannelsToRead(m_task, buf, sizeof(buf)));
    int32 reftrigtype;
	CHECK_DAQMX_RET(DAQmxGetRefTrigType(m_task, &reftrigtype));
    uInt32 pretrig = 0;
	if(reftrigtype != DAQmx_Val_None) {
	    pretrig = lrint(*trigPos() / 100.0 * *recordLength());
	}
	
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
	printf("strange pre-prig num:%d %f\n", (uint32_t)pretrig, - pretrig * interval);

//	printf("%d %f %f %d\n", num_ch, - pretrig * interval, interval, len);
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
