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
#include "funcsynth.h"
#include "chardevicedriver.h"

class XWAVEFACTORY : public XCharDeviceDriver<XFuncSynth>
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
