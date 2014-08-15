/***************************************************************************
		Copyright (C) 2002-2014 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "chardevicedriver.h"
#include <vector>

#include "dso.h"

#include "ezusbthamway.h"

//! Thamway DV14U25 A/D conversion board
class XThamwayDVUSBDSO : public XCharDeviceDriver<XDSO, XWinCUSBInterface> {
public:
    XThamwayDVUSBDSO(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XThamwayDVUSBDSO();
    //! Converts raw to record
    virtual void convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);
protected:
    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() throw (XKameError &);
    //! Be called during stopping driver. Call interface()->stop() inside this routine.
    virtual void close() throw (XKameError &);

    virtual void onTrace1Changed(const Snapshot &shot, XValueNodeBase *);
    virtual void onTrace2Changed(const Snapshot &shot, XValueNodeBase *);
    virtual void onTrace3Changed(const Snapshot &shot, XValueNodeBase *);
    virtual void onTrace4Changed(const Snapshot &shot, XValueNodeBase *);
    virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onSingleChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onTrigSourceChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onTrigPosChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onTrigLevelChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onTrigFallingChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onTimeWidthChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onVFullScale1Changed(const Snapshot &shot, XValueNodeBase *);
    virtual void onVFullScale2Changed(const Snapshot &shot, XValueNodeBase *);
    virtual void onVFullScale3Changed(const Snapshot &shot, XValueNodeBase *);
    virtual void onVFullScale4Changed(const Snapshot &shot, XValueNodeBase *);
    virtual void onVOffset1Changed(const Snapshot &shot, XValueNodeBase *);
    virtual void onVOffset2Changed(const Snapshot &shot, XValueNodeBase *);
    virtual void onVOffset3Changed(const Snapshot &shot, XValueNodeBase *);
    virtual void onVOffset4Changed(const Snapshot &shot, XValueNodeBase *);
    virtual void onRecordLengthChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onForceTriggerTouched(const Snapshot &shot, XTouchableNode *);

    virtual double getTimeInterval();
    //! Clears count or start sequence measurement
    virtual void startSequence();
    virtual int acqCount(bool *seq_busy);

    //! Loads waveform and settings from instrument
    virtual void getWave(shared_ptr<RawData> &writer, std::deque<XString> &channels);

    virtual bool isDRFCoherentSGSupported() const {return false;}
private:

    void acquire(const atomic<bool> &terminated);

};
