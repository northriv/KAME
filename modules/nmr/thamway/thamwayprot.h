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
#ifndef thamwayprotH
#define thamwayprotH

#include "networkanalyzer.h"
#include "signalgenerator.h"
#include "chardevicedriver.h"

//! Thamway Impedance Analyzer T300-1049A
class XThamwayT300ImpedanceAnalyzer : public XCharDeviceDriver<XNetworkAnalyzer> {
public:
    XThamwayT300ImpedanceAnalyzer(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XThamwayT300ImpedanceAnalyzer() {}
protected:
    virtual void onStartFreqChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onStopFreqChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) {}
    virtual void onPointsChanged(const Snapshot &shot, XValueNodeBase *);

    virtual void onCalOpenTouched(const Snapshot &shot, XTouchableNode *);
    virtual void onCalShortTouched(const Snapshot &shot, XTouchableNode *);
    virtual void onCalTermTouched(const Snapshot &shot, XTouchableNode *);
    virtual void onCalThruTouched(const Snapshot &shot, XTouchableNode *) {}

    virtual void getMarkerPos(unsigned int num, double &x, double &y);
    virtual void oneSweep();
    virtual void startContSweep();
    virtual void acquireTrace(shared_ptr<RawData> &, unsigned int ch);
    //! Converts raw to dispaly-able
    virtual void convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);

    //! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() throw (XKameError &);
};


//! Thamway NMR PROT series
class XThamwayPROTSG : public XCharDeviceDriver<XSG> {
public:
    XThamwayPROTSG(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XThamwayPROTSG() {}
protected:
    virtual void changeFreq(double mhz);
    virtual void onRFONChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onOLevelChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onFMONChanged(const Snapshot &shot, XValueNodeBase *);
    virtual void onAMONChanged(const Snapshot &shot, XValueNodeBase *);
private:
};
#endif
