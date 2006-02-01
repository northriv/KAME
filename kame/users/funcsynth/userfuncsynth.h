#include "funcsynth.h"

class XWAVEFACTORY : public XFuncSynth
{
 XNODE_OBJECT
 protected:
  XWAVEFACTORY(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers);
protected:
  virtual void onOutputChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onTrigTouched(const shared_ptr<XNode> &);
  virtual void onModeChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onFunctionChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onFreqChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onAmpChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onPhaseChanged(const shared_ptr<XValueNodeBase> &);
  virtual void onOffsetChanged(const shared_ptr<XValueNodeBase> &);
};
