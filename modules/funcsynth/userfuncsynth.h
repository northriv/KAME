/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
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

class XWAVEFACTORY : public XCharDeviceDriver<XFuncSynth> {
public:
	XWAVEFACTORY(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
protected:
	virtual void onOutputChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onTrigTouched(const Snapshot &shot, XTouchableNode *);
	virtual void onModeChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onFunctionChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onFreqChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onAmpChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onPhaseChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onOffsetChanged(const Snapshot &shot, XValueNodeBase *);
};
