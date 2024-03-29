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
#include "chardevicedriver.h"
#include <vector>

#include "dso.h"

#include "thamwayusbinterface.h"

#define ADDR_OFFSET_DV 0x20

class XThamwayDVCUSBInterface : public XThamwayFX2USBInterface {
public:
    XThamwayDVCUSBInterface(const char *name, bool runtime, const shared_ptr<XDriver> &driver)
        : XThamwayFX2USBInterface(name, runtime, driver, ADDR_OFFSET_DV, "DV14") {}
    virtual ~XThamwayDVCUSBInterface() {}
};

//! Thamway DV14U25 A/D conversion board
class XThamwayDVUSBDSO : public XCharDeviceDriver<XDSO, XThamwayDVCUSBInterface> {
public:
    XThamwayDVUSBDSO(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XThamwayDVUSBDSO();
    //! Converts raw to record
    virtual void convertRaw(RawDataReader &reader, Transaction &tr) override;
protected:
    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;
    //! Be called during stopping driver. Call interface()->stop() inside this routine.
    virtual void close() override;

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

    virtual double getTimeInterval() override;
    //! Clears count or start sequence measurement
    virtual void startSequence() override;
    virtual int acqCount(bool *seq_busy) override;

    //! Loads waveform and settings from instrument
    virtual void getWave(shared_ptr<RawData> &writer, std::deque<XString> &channels) override;

    virtual bool isDRFCoherentSGSupported() const override {return false;}
private:
    void acquire(const atomic<bool> &terminated);
    bool m_pending;
    unsigned int m_adConvBits;
};
