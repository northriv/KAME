/***************************************************************************
        Copyright (C) 2002-2018 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
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
    virtual void onStartFreqChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onStopFreqChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onPointsChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onPowerChanged(const Snapshot &shot, XValueNodeBase *) override;

    virtual void onCalOpenTouched(const Snapshot &shot, XTouchableNode *) override {}
    virtual void onCalShortTouched(const Snapshot &shot, XTouchableNode *) override {}
    virtual void onCalTermTouched(const Snapshot &shot, XTouchableNode *) override {}
    virtual void onCalThruTouched(const Snapshot &shot, XTouchableNode *) override {}

    virtual void getMarkerPos(unsigned int num, double &x, double &y) override;
    virtual void oneSweep() override;
    virtual void startContSweep() override;
    virtual void acquireTrace(shared_ptr<RawData> &, unsigned int ch) override;
	//! Converts raw to dispaly-able
    virtual void convertRaw(RawDataReader &reader, Transaction &tr) override;

	//! Be called just after opening interface. Call start() inside this routine appropriately.
    virtual void open() override;

    virtual unsigned int acquireTraceData(unsigned int ch, unsigned int len) = 0;
    //! may throw XRecordError if mal-formatted.
    virtual void convertRawBlock(RawDataReader &reader, Transaction &tr,
        unsigned int len) = 0;
private:
};

//! HP/Agilent 8711C/8712C/8713C/8714C Network Analyzer.
class XHP8711 : public XAgilentNetworkAnalyzer {
public:
	XHP8711(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
			 XAgilentNetworkAnalyzer(name, runtime, ref(tr_meas), meas) {}
	virtual ~XHP8711() {}

    virtual unsigned int acquireTraceData(unsigned int ch, unsigned int len);
	virtual void convertRawBlock(RawDataReader &reader, Transaction &tr,
		unsigned int len);
private:
};

//! Agilent E5061A/5062A Network Analyzer.
class XAgilentE5061 : public XHP8711 {
public:
	XAgilentE5061(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas) :
			 XHP8711(name, runtime, ref(tr_meas), meas) {}
	virtual ~XAgilentE5061() {}

    virtual unsigned int acquireTraceData(unsigned int ch, unsigned int len);
	virtual void convertRawBlock(RawDataReader &reader, Transaction &tr,
		unsigned int len);
private:
};

//! Copper Mountain Planar TR1300/1, TR5048, TR4530 Vector Network Analyzer.
class XCopperMtTRVNA : public XAgilentE5061 {
public:
    XCopperMtTRVNA(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XCopperMtTRVNA() {}
private:
};

//! DG8SAQ VNWA3E via a custom DLL.
class XVNWA3ENetworkAnalyzer : public XCharDeviceDriver<XNetworkAnalyzer> {
public:
	XVNWA3ENetworkAnalyzer(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XVNWA3ENetworkAnalyzer() {}
protected:
    virtual void onStartFreqChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onStopFreqChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onPointsChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onPowerChanged(const Snapshot &shot, XValueNodeBase *) override {}

    virtual void onCalOpenTouched(const Snapshot &shot, XTouchableNode *) override {}
    virtual void onCalShortTouched(const Snapshot &shot, XTouchableNode *) override {}
    virtual void onCalTermTouched(const Snapshot &shot, XTouchableNode *) override {}
    virtual void onCalThruTouched(const Snapshot &shot, XTouchableNode *) override {}

	virtual void getMarkerPos(unsigned int num, double &x, double &y);
	virtual void oneSweep();
	virtual void startContSweep();
	virtual void acquireTrace(shared_ptr<RawData> &, unsigned int ch);
	//! Converts raw to dispaly-able
	virtual void convertRaw(RawDataReader &reader, Transaction &tr);

	//! Be called just after opening interface. Call start() inside this routine appropriately.
	virtual void open();
};

//! DG8SAQ VNWA3E via TCP/IP interface.
class XVNWA3ENetworkAnalyzerTCPIP : public XCharDeviceDriver<XNetworkAnalyzer> {
public:
    XVNWA3ENetworkAnalyzerTCPIP(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XVNWA3ENetworkAnalyzerTCPIP() {}
protected:
    virtual void onStartFreqChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onStopFreqChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) override {}
    virtual void onPointsChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onPowerChanged(const Snapshot &shot, XValueNodeBase *) override {}

    virtual void onCalOpenTouched(const Snapshot &shot, XTouchableNode *) override {}
    virtual void onCalShortTouched(const Snapshot &shot, XTouchableNode *) override {}
    virtual void onCalTermTouched(const Snapshot &shot, XTouchableNode *) override {}
    virtual void onCalThruTouched(const Snapshot &shot, XTouchableNode *) override {}

    virtual void getMarkerPos(unsigned int num, double &x, double &y) override;
    virtual void oneSweep() override;
    virtual void startContSweep() override;
    virtual void acquireTrace(shared_ptr<RawData> &, unsigned int ch) override;
    //! Converts raw to dispaly-able
    virtual void convertRaw(RawDataReader &reader, Transaction &tr) override;

    virtual void open() override;
    virtual void close() override;

    virtual shared_ptr<XCharInterface> interface2() const {return m_interface2;}
private:
    const shared_ptr<XCharInterface> m_interface2;
};

//! LibreVNA via TCP/IP interface.
class XLibreVNASCPI : public XCharDeviceDriver<XNetworkAnalyzer> {
public:
    XLibreVNASCPI(const char *name, bool runtime,
        Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
    virtual ~XLibreVNASCPI() {}
protected:
    virtual void onStartFreqChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onStopFreqChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onAverageChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onPointsChanged(const Snapshot &shot, XValueNodeBase *) override;
    virtual void onPowerChanged(const Snapshot &shot, XValueNodeBase *) override;

    virtual void onCalOpenTouched(const Snapshot &shot, XTouchableNode *) override {}
    virtual void onCalShortTouched(const Snapshot &shot, XTouchableNode *) override {}
    virtual void onCalTermTouched(const Snapshot &shot, XTouchableNode *) override {}
    virtual void onCalThruTouched(const Snapshot &shot, XTouchableNode *) override {}

    virtual void getMarkerPos(unsigned int num, double &x, double &y) override;
    virtual void oneSweep() override;
    virtual void startContSweep() override;
    virtual void acquireTrace(shared_ptr<RawData> &, unsigned int ch) override;
    //! Converts raw to dispaly-able
    virtual void convertRaw(RawDataReader &reader, Transaction &tr) override;
private:
};
#endif
