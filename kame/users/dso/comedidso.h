#ifndef COMEDIDSO_H_
#define COMEDIDSO_H_

#include "comediinterface.h"

#ifdef HAVE_COMEDI

#include "dso.h"

//! DSO wrapper for COMEDI.
class XComediDSO : public XPrimaryDriver
{
 XNODE_OBJECT
 protected:
  XComediDSO(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
  ~XComediDSO() {}
  //! convert raw to record
  virtual void convertRaw() throw (XRecordError&);
 protected:
  virtual void afterStop();
  const shared_ptr<XComediInterface> &analogInput() const {return m_analogInput;} 
  const shared_ptr<XComediInterface> &counter() const {return m_counter;} 
 
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
  shared_ptr<XListener> m_lsnOnOpen, m_lsnOnClose;
  void onOpen(const shared_ptr<XInterface> &);
  void onClose(const shared_ptr<XInterface> &);
  shared_ptr<XComediInterface> m_analogInput, m_counter;

  void setupCommand();
  struct Command {
  	bool pretrig;
  	comedi_cmd ai;
  	int chanlist[2];
  	std::vector<comedi_insn> insnlist;
  	std::vector<lsampl_t> configlist;
  };
  atomic_shared_ptr<Command> m_command;
};

#endif //HAVE_COMEDI

#endif /*COMEDIDSO_H_*/
