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
#ifndef usernetworkanalyerH
#define usernetworkanalyerH

#include "networkanalyzer.h"
#include "chardevicedriver.h"
//---------------------------------------------------------------------------

//! Base class for HP/Agilent Network Analyzer.
class XAgilentNetworkAnalyzer : public XCharDeviceDriver<XNetworkAnalyzer> {
public:
	XAgilentNetworkAnalyzer(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XAgilentNetworkAnalyzer() {}
protected:
	virtual void onStartFreqChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onStopFreqChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onPointsChanged(const Snapshot &shot, XValueNodeBase *);

	virtual void onCalOpenTouched(const Snapshot &shot, XTouchableNode *) {}
	virtual void onCalShortTouched(const Snapshot &shot, XTouchableNode *) {}
	virtual void onCalTermTouched(const Snapshot &shot, XTouchableNode *) {}
	virtual void onCalThruTouched(const Snapshot &shot, XTouchableNode *) {}

	virtual void getMarkerPos(unsigned int num, double &x, double &y);
	virtual void oneSweep();
	virtual void startContSweep();
	virtual void acquireTrace(shared_ptr<RawData> &, unsigned int ch);
	//! Converts raw to dispaly-able
	virtual void convertRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);

	//! Be called just after opening interface. Call start() inside this routine appropriately.
	virtual void open() throw (XKameError &);

	virtual void acquireTraceData(unsigned int ch, unsigned int len) = 0;
	virtual void convertRawBlock(RawDataReader &reader, Transaction &tr,
		unsigned int len) throw (XRecordError&) = 0;
private:
};

//! HP/Agilent 8711C/8712C/8713C/8714C Network Analyzer.
class XHP8711 : public XAgilentNetworkAnalyzer {
public:
	XHP8711(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
			 XAgilentNetworkAnalyzer(name, runtime, ref(tr_meas), meas) {}
	virtual ~XHP8711() {}

	virtual void acquireTraceData(unsigned int ch, unsigned int len);
	virtual void convertRawBlock(RawDataReader &reader, Transaction &tr,
		unsigned int len) throw (XRecordError&);
private:
};

//! Agilent E5061A/5062A Network Analyzer.
class XAgilentE5061 : public XHP8711 {
public:
	XAgilentE5061(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
			 XHP8711(name, runtime, ref(tr_meas), meas) {}
	virtual ~XAgilentE5061() {}

	virtual void acquireTraceData(unsigned int ch, unsigned int len);
	virtual void convertRawBlock(RawDataReader &reader, Transaction &tr,
		unsigned int len) throw (XRecordError&);
private:
};

//! DG8SAQ VNWA3E via a custom DLL.
class XVNWA3ENetworkAnalyzer : public XCharDeviceDriver<XNetworkAnalyzer> {
public:
	XVNWA3ENetworkAnalyzer(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XVNWA3ENetworkAnalyzer() {}
protected:
	virtual void onStartFreqChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onStopFreqChanged(const Snapshot &shot, XValueNodeBase *);
	virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) {}
	virtual void onPointsChanged(const Snapshot &shot, XValueNodeBase *) {}

	virtual void onCalOpenTouched(const Snapshot &shot, XTouchableNode *) {}
	virtual void onCalShortTouched(const Snapshot &shot, XTouchableNode *) {}
	virtual void onCalTermTouched(const Snapshot &shot, XTouchableNode *) {}
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
#endif
