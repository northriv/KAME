#include "nidaqdso.h"

#ifdef HAVE_NI_DAQMX

#include "xwavengraph.h"
#include <klocale.h>

#define TASK_UNDEF ((TaskHandle)-1)

//---------------------------------------------------------------------------
XNIDAQmxDSO::XNIDAQmxDSO(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
 XNIDAQmxDriver<XDSO>(name, runtime, scalarentries, interfaces, thermometers, drivers),
 m_task(TASK_UNDEF)
{
}
void
XNIDAQmxDSO::open() throw (XInterface::XInterfaceError &)
{
	this->start();

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
    const char* sc[] = {"0.4", "1", "2", "4", "10", "20", "40", "84", 0L};
    for(int i = 0; sc[i]; i++)
    {
        vFullScale1()->add(sc[i]);
        vFullScale2()->add(sc[i]);
    }
    createChannels();
    setupTrigger();
    setupTiming();
}
void
XNIDAQmxDSO::close() throw (XInterface::XInterfaceError &)
{
	XScopedLock<XMutex> lock(m_tasklock);
	m_task = TASK_UNDEF;
    CHECK_DAQMX_RET(DAQmxClearTask(m_task), "Clear Task");
    m_analogTrigSrc.clear();
    m_digitalTrigSrc.clear();
    trace1()->clear();
    trace2()->clear();
    trigSource()->clear();
    vFullScale1()->clear();
    vFullScale2()->clear();
	interface()->stop();
}
void
XNIDAQmxDSO::setupTrigger()
{
	XScopedLock<XMutex> lock(m_tasklock);
    CHECK_DAQMX_RET(DAQmxStopTask(m_task), "Stop Task");
    if(std::find(m_analogTrigSrc.begin(), m_analogTrigSrc.end(), trigSource()->to_str())
         != m_analogTrigSrc.end()) {
        CHECK_DAQMX_RET(DAQmxCfgAnlgEdgeRefTrig(m_task,
            trigSource()->to_str().c_str(),
            *trigFalling ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope,
            *trigLevel(),
            - *trigPos() / getTimeInterval()),
            "Trigger Setup");
    }
    if(std::find(m_digitalTrigSrc.begin(), m_digitalTrigSrc.end(), trigSource()->to_str())
         != m_digitalTrigSrc.end()) {
        CHECK_DAQMX_RET(DAQmxCfgDigEdgeRefTrig(m_task,
            trigSource()->to_str().c_str(),
            *trigFalling ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope,
            *trigLevel(),
            - *trigPos() / getTimeInterval()),
            "Trigger Setup");
    }
    CHECK_DAQMX_RET(DAQmxStartTask(m_task), "Start Task");
}
void
XNIDAQmxDSO::setupTiming()
{
    m_acqCount = 0;
	XScopedLock<XMutex> lock(m_tasklock);
    CHECK_DAQMX_RET(DAQmxStopTask(m_task), "Stop Task");

    CHECK_DAQMX_RET(DAQmxCfgSampClkTiming(m_task,
        NULL, // internal source
        *recordLength() / *timeWidth(),
        DAQmx_Val_Rising,
        DAQmx_Val_FiniteSamps,
        *recordLength()
        ), "Set Timing");
    float64 rate;
    CHECK_DAQMX_RET(DAQmxGetSampClkRate(m_task, &rate
    	), "Get Rate");
    m_interval = 1.0 / rate;
    CHECK_DAQMX_RET(DAQmxStartTask(m_task), "Start Task");
}
void
XNIDAQmxDSO::createChannels()
{
	XScopedLock<XMutex> lock(m_tasklock);
	if(m_task == TASK_UNDEF)
	    CHECK_DAQMX_RET(DAQmxClearTask(m_task), "Clear Task");
    CHECK_DAQMX_RET(DAQmxCreateTask("XNIDAQmxDSO", &m_task), "Task Creation");
    
    if(*trace1() > 0) {
        CHECK_DAQMX_RET(DAQmxCreateAIVoltageChan(m_task,
            trace1()->to_str().c_str(),
            trace1()->to_str().c_str(),
            DAQmx_Val_Cfg_Default,
            -atof(vFullScale1()->to_str().c_str()) / 2.0,
            atof(vFullScale1()->to_str().c_str()) / 2.0,
            DAQmx_Val_Volts,
            NULL
            ), "Channel Creation");
    }
    if(*trace2() > 0) {
        CHECK_DAQMX_RET(DAQmxCreateAIVoltageChan(m_task,
            trace2()->to_str().c_str(),
            trace2()->to_str().c_str(),
            DAQmx_Val_Cfg_Default,
            -atof(vFullScale2()->to_str().c_str()) / 2.0,
            atof(vFullScale2()->to_str().c_str()) / 2.0,
            DAQmx_Val_Volts,
            NULL
            ), "Channel Creation");
    }
    CHECK_DAQMX_RET(DAQmxRegisterDoneEvent(m_task, 0, &XNIDAQmxDSO::_acqCallBack, this),
        "Register Event");
    CHECK_DAQMX_RET(DAQmxStartTask(m_task), "Start Task");
    m_acqCount = 0;
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
	XScopedLock<XMutex> lock(m_tasklock);
    CHECK_DAQMX_RET(DAQmxSendSoftwareTrigger(m_task, DAQmx_Val_AdvanceTrigger),
        "Force Trigger");
}
int32
XNIDAQmxDSO::_acqCallBack(TaskHandle task, int32 status, void *data)
{
    XNIDAQmxDSO *obj = reinterpret_cast<XNIDAQmxDSO>(data);
    return obj->acqCallBack(task, status);
}
int32
XNIDAQmxDSO::acqCallBack(TaskHandle task, int32 status)
{
 	XScopedLock<XMutex> lock(m_tasklock);
    CHECK_DAQMX_RET(status, "Event");
    int len = *recordLength();
    int32 cnt;
    std::vector<float64> buf(len * 2);
    CHECK_DAQMX_RET(DAQmxReadAnalogF64(m_task, DAQmx_Val_Auto,
        0, DAQmx_Val_GroupByChannel,
        &buf[0], len * 2, &cnt, NULL
        ), "Read");
    ASSERT(cnt <= len);
    for(int ch = 0; ch < 1; ch++) {
        m_records[ch].resize(cnt);
        double *prec = &m_records[ch][0];
        float64 *pbuf = &buf[cnt * ch];
        for(int i = 0; i < cnt; i++) {
            (*prec++) += (*pbuf++)
        }
    }
    m_acqCount++;
    if(*singleSequence() && (m_acqCount >= *average())) {
        CHECK_DAQMX_RET(DAQmxDisableStartTrigger(m_task), "Disable Trigger");       
    }
}
void
XNIDAQmxDSO::startSequence()
{
    m_acqCount = 0;
    setupTrigger();
}

int
XNIDAQmxDSO::acqCount(bool *seq_busy)
{
//    int32 late;
//    int ret = DAQmxWaitForNextSampleClock(*m_task, 1.0, &late);
//    if(late)
//        gWarnPrint(i18n("Real Time Operation Failed."));
    *seq_busy = (m_acqCount < *average());
    return m_acqCount;
}

double
XNIDAQmxDSO::getTimeInterval()
{
	return m_interval;
}

void
XNIDAQmxDSO::getWave(std::deque<std::string> &channels)
{
    for(std::deque<std::string>::iterator it = channels.begin();
        it != channels.end(); it++)
    {
    }
}
void
XNIDAQmxDSO::convertRaw() throw (XRecordError&) {

  setRecordDim(ch_cnt, xoff, xin, width);
  
  cp = buf;
  for(int j = 0; j < ch_cnt; j++)
    {
      double *wave = waveRecorded(j);
      for(; i < std::min(width, yyy/2); i++)
        	{
        	  double val = *((unsigned char *)cp) * 0x100;
        	  val += *((unsigned char *)cp + 1);
        	  *(wave++) += yin[j] * (val - yoff[j] - 0.5);
        	  cp += 2;
        	}
      for(; i < width; i++) {
      	  *(wave++) = 0.0;
      }
    }  
}

#endif //HAVE_NI_DAQMX
