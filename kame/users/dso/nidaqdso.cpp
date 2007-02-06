#include "nidaqdso.h"

#ifdef HAVE_NI_DAQMX

#include "xwavengraph.h"
#include <klocale.h>
#include "NIDAQmx.h"
#include <QRegExp>

#define TASK_UNDEF ((TaskHandle)-1)

#define CheckError(code, msg) _checkError(code, msg, __FILE__, __LINE__)
void
XNIDAQmxDSO::_checkError(int code, const char *msg, const char *file, int line)
{
    if(code < 0)
        throw XInterface::XInterfaceError(msg, file, line);
    if(code > 0)
        _gWarnPrint(msg, file, line);
}

//---------------------------------------------------------------------------
XNIDAQmxDSO::XNIDAQmxDSO(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers) :
 XDSO(name, runtime, scalarentries, interfaces, thermometers, drivers),
 m_task(TASK_UNDEF)
{
    char buf[2048];
    DAQmxGetSysDevNames(buf, sizeof(buf));
    std::deque<std::string> dev;
    splitList(buf, dev);
    for(std::deque<std::string>::iterator it = dev.begin(); it != dev.end(); it++) {
        DAQmxGetDevAIPhysicalChans(it->c_str(), buf, sizeof(buf));
        std::deque<std::string> chans;
        splitList(buf, chans);
        for(std::deque<std::string>::iterator it = chans.begin(); it != chans.end(); it++) {
            trace1()->add(it->c_str());
            trace2()->add(it->c_str());
        }
    }
    const char* sc[] = {"0.4", "1", "2", "4", "10", "20", "40", "84", 0L};
    for(int i = 0; sc[i]; i++)
    {
        vFullScale1()->add(sc[i]);
        vFullScale2()->add(sc[i]);
    }
    
    TaskHandle task;
    DAQmxCreateTask("", &task);
    DAQmxGetDigEdgeStartTrigSrc(task, buf, sizeof(size));
    splitList(buf, m_digitalTrigSrc);
    for(std::deque<std::string>::iterator it = m_digitalTrigSrc.begin(); it != m_digitalTrigSrc.end(); it++) {
        trigSource()->add(it->c_str());
    }
    DAQmxGetAnlgEdgeStartTrigSrc(task, buf, sizeof(size));
    splitList(buf, m_analogTrigSrc);
    for(std::deque<std::string>::iterator it = m_analogTrigSrc.begin(); it != m_analogTrigSrc.end(); it++) {
        trigSource()->add(it->c_str());
    }
    DAQmxClearTask(task);
}
void
XNIDAQmxDSO::afterStart()
{
    createChannels();
    setupTrigger();
    setupTiming();
}
void
XNIDAQmxDSO::beforeStop()
{
	XScopedLock<XMutex> lock(m_tasklock);
	m_task = TASK_UNDEF;
    CheckError(DAQmxStopTask(m_task), "Stop Task");
    int ret = DAQmxClearTask(m_task);
    ASSERT(ret >= 0);
}
void
XNIDAQmxDSO::setupTrigger()
{
	XScopedLock<XMutex> lock(m_tasklock);
    CheckError(DAQmxStopTask(m_task), "Stop Task");
    if(std::find(m_analogTrigSrc.begin(), m_analogTrigSrc.end(), trigSource()->to_str())
         != m_analogTrigSrc.end()) {
        CheckError(DAQmxCfgAnlgEdgeStartTrig(m_task,
            trigSource()->to_str().c_str(),
            *trigFalling ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope,
            *trigLevel()),
            "Trigger Setup");
    }
    if(std::find(m_digitalTrigSrc.begin(), m_digitalTrigSrc.end(), trigSource()->to_str())
         != m_digitalTrigSrc.end()) {
        CheckError(DAQmxCfgDigEdgeStartTrig(m_task,
            trigSource()->to_str().c_str(),
            *trigFalling ? DAQmx_Val_FallingSlope : DAQmx_Val_RisingSlope,
            *trigLevel()),
            "Trigger Setup");
    }
    CheckError(DAQmxStartTask(m_task), "Start Task");
}
void
XNIDAQmxDSO::setupTiming()
{
    m_acqCount = 0;
	XScopedLock<XMutex> lock(m_tasklock);
    CheckError(DAQmxStopTask(m_task), "Stop Task");
    m_timeInterval = 
    CheckError(DAQmxCfgSampClkTiming(m_task,
        NULL, // internal source
        *recordLength() / *timeWidth(),
        DAQmx_Val_Rising,
        DAQmx_Val_FiniteSamps,
        *recordLength()
        ), "Set Timing");
    CheckError(DAQmxStartTask(m_task), "Start Task");
}
void
XNIDAQmxDSO::createChannels()
{
	XScopedLock<XMutex> lock(m_tasklock);
	if(m_task == TASK_UNDEF) {
	    CheckError(DAQmxStopTask(m_task), "Stop Task");
	    int ret = DAQmxClearTask(m_task);
	    ASSERT(ret >= 0);
	}

    CheckError(DAQmxCreateTask("XNIDAQmxDSO", &m_task), "Task Creation");
    if(*trace1() > 0) {
        CheckError(DAQmxCreateAIVoltageChan(m_task,
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
        CheckError(DAQmxCreateAIVoltageChan(m_task,
            trace2()->to_str().c_str(),
            trace2()->to_str().c_str(),
            DAQmx_Val_Cfg_Default,
            -atof(vFullScale2()->to_str().c_str()) / 2.0,
            atof(vFullScale2()->to_str().c_str()) / 2.0,
            DAQmx_Val_Volts,
            NULL
            ), "Channel Creation");
    }
    CheckError(DAQmxRegisterDoneEvent(m_task, 0, &XNIDAQmxDSO::_acqCallBack, this),
        "Register Event");
    CheckError(DAQmxStartTask(m_task), "Start Task");
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
    CheckError(DAQmxSendSoftwareTrigger(m_task, DAQmx_Val_AdvanceTrigger),
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
    CheckError(status, "Event");
    int len = *recordLength();
    int32 cnt;
    std::vector<float64> buf(len * 2);
    CheckError(DAQmxReadAnalogF64(m_task, DAQmx_Val_Auto,
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
        CheckError(DAQmxDisableStartTrigger(m_task), "Disable Trigger");       
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
	XScopedLock<XMutex> lock(m_tasklock);
    float64 rate;
    DAQmxGetSampClkRate(m_task, &rate);
    return 1.0 / rate;
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
