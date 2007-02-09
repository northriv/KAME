#ifndef nidaqdsoH
#define nidaqdsoH

#include "dso.h"

#include "nidaqmxdriver.h"

#ifdef HAVE_NI_DAQMX

//! Software DSO w/ NI DAQmx
class XNIDAQmxDSO : public XNIDAQmxDriver<XDSO>
{
 XNODE_OBJECT
 protected:
  XNIDAQmxDSO(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
  virtual ~XNIDAQmxDSO();
  //! convert raw to record
  virtual void convertRaw() throw (XRecordError&);
 protected:
  //! Be called just after opening interface. Call start() inside this routine appropriately.
  virtual void open() throw (XInterface::XInterfaceError &);
  //! Be called during stopping driver. Call interface()->stop() inside this routine.
  virtual void close() throw (XInterface::XInterfaceError &);

  virtual void onTrace1Changed(const shared_ptr<XValueNodeBase> &);
  virtual void onTrace2Changed(const shared_ptr<XValueNodeBase> &);
  virtual void onAverageChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onSingleChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onTrigSourceChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onTrigPosChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onTrigLevelChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onTrigFallingChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onTimeWidthChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onVFullScale1Changed(const shared_ptr<XValueNodeBase> &);
  virtual void onVFullScale2Changed(const shared_ptr<XValueNodeBase> &);
  virtual void onVOffset1Changed(const shared_ptr<XValueNodeBase> &);
  virtual void onVOffset2Changed(const shared_ptr<XValueNodeBase> &);
  virtual void onRecordLengthChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onForceTriggerTouched(const shared_ptr<XNode> &);

  virtual double getTimeInterval();
  //! clear count or start sequence measurement
  virtual void startSequence();
  virtual int acqCount(bool *seq_busy);

  //! load waveform and settings from instrument
  virtual void getWave(std::deque<std::string> &channels);
 private:
 typedef int16 tRawAI;
  scoped_ptr<XNIDAQmxInterface::XNIDAQmxRoute> m_trigRoute;
  std::vector<tRawAI> m_record_buf;
  std::vector<int32_t> m_record;
#define CAL_POLY_ORDER 4
	float64 m_coeffAI[2][CAL_POLY_ORDER];
//	float64 m_upperLimAI[2];
//	float64 m_lowerLimAI[2];
	inline float64 aiRawToVolt(const float64 *pcoeff, float64 raw);
  int m_accumCount;
  //! for moving av.
  std::deque<std::vector<tRawAI> > m_record_av; 
  unsigned int m_record_length;
  std::deque<std::string> m_analogTrigSrc, m_digitalTrigSrc;
  TaskHandle m_task;
  bool m_bPollMode;
  double m_interval;
  int m_acqCount;
  void setupAcquision();
  void setupTrigger();
  void setupTiming();
  void createChannels();
  static int32 _acqCallBack(TaskHandle, int32, void*);
  int32 acqCallBack(TaskHandle task, int32 status);
  void acquire(TaskHandle task);
};

#endif //HAVE_NI_DAQMX

#endif
