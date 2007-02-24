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
  shared_ptr<XNIDAQmxInterface::VirtualTrigger> m_virtualTrigger;
  atomic_shared_ptr<XNIDAQmxInterface::VirtualTrigger::VirtualTriggerList> m_virtualTriggerList; 
  shared_ptr<XListener> m_lsnOnVirtualTrigStart;
  void onVirtualTrigStart(const shared_ptr<XNIDAQmxInterface::VirtualTrigger> &);
  shared_ptr<XThread<XNIDAQmxDSO> > m_threadReadAI;
  void *executeReadAI(const atomic<bool> &);
  atomic<bool> m_suspendRead;
  atomic<bool> m_running;
  std::vector<tRawAI> m_recordBuf;
  std::vector<int32_t> m_record;
enum {CAL_POLY_ORDER = 4};
	float64 m_coeffAI[2][CAL_POLY_ORDER];
	inline float64 aiRawToVolt(const float64 *pcoeff, float64 raw);
  unsigned int m_accumCount;
  //! for moving av.
  std::deque<std::vector<tRawAI> > m_record_av; 
  unsigned int m_recordLength;
  std::deque<std::string> m_analogTrigSrc;
  TaskHandle m_task;
  double m_interval;
  unsigned int m_preTriggerPos;
  int m_acqCount;
  void clearAcquision();
  void setupAcquision();
  void disableTrigger();
  void setupTrigger();
  void setupTiming();
  void createChannels();
  void acquire(const atomic<bool> &terminated);

  XRecursiveMutex m_readMutex;

  inline bool tryReadAISuspend(const atomic<bool> &terminated);
};

#endif //HAVE_NI_DAQMX

#endif
