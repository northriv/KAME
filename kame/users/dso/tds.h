#ifndef tdsH
#define tdsH

#include "dso.h"
#include "chardevicedriver.h"
//---------------------------------------------------------------------------

//! Tektronix DSO
class XTDS : public XCharDeviceDriver<XDSO>
{
 XNODE_OBJECT
 protected:
  XTDS(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
  ~XTDS() {}
  //! convert raw to record
  virtual void convertRaw() throw (XRecordError&);
 protected:
  virtual void onTrace1Changed(const shared_ptr<XValueNodeBase> &) {}
  virtual void onTrace2Changed(const shared_ptr<XValueNodeBase> &) {}
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

  //! Be called just after opening interface. Call start() inside this routine appropriately.
  virtual void open() throw (XInterface::XInterfaceError &);
  
  virtual double getTimeInterval();
  //! clear count or start sequence measurement
  virtual void startSequence();
  virtual int acqCount(bool *seq_busy);

  //! load waveform and settings from instrument
  virtual void getWave(std::deque<std::string> &channels);
 private:
};

#endif
