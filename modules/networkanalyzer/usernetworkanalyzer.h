/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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
class XAgilentNetworkAnalyzer : public XCharDeviceDriver<XNetworkAnalyzer>
{
	XNODE_OBJECT
protected:
	XAgilentNetworkAnalyzer(const char *name, bool runtime,
		 const shared_ptr<XScalarEntryList> &scalarentries,
		 const shared_ptr<XInterfaceList> &interfaces,
		 const shared_ptr<XThermometerList> &thermometers,
		 const shared_ptr<XDriverList> &drivers);
	virtual ~XAgilentNetworkAnalyzer() {}

	virtual void onStartFreqChanged(const shared_ptr<XValueNodeBase> &);
	virtual void onStopFreqChanged(const shared_ptr<XValueNodeBase> &);
	virtual void onAverageChanged(const shared_ptr<XValueNodeBase> &);
	virtual void onPointsChanged(const shared_ptr<XValueNodeBase> &);
	virtual void getMarkerPos(unsigned int num, double &x, double &y);
	virtual void oneSweep();
	virtual void startContSweep();
	virtual void acquireTrace(unsigned int ch);
	//! convert raw to dispaly-able
	virtual void convertRaw() throw (XRecordError&);

	//! Be called just after opening interface. Call start() inside this routine appropriately.
	virtual void open() throw (XInterface::XInterfaceError &);

	virtual void acquireTraceData(unsigned int ch, unsigned int len) = 0;
	virtual void convertRawBlock(unsigned int len) throw (XRecordError&) = 0;
private:
};

//! HP/Agilent 8711C/8712C/8713C/8714C Network Analyzer.
class XHP8711 : public XAgilentNetworkAnalyzer
{
	XNODE_OBJECT
protected:
	XHP8711(const char *name, bool runtime,
		 const shared_ptr<XScalarEntryList> &scalarentries,
		 const shared_ptr<XInterfaceList> &interfaces,
		 const shared_ptr<XThermometerList> &thermometers,
		 const shared_ptr<XDriverList> &drivers) :
			 XHP8711(name, runtime, scalarentries, interfaces, thermometers, drivers) {}
	virtual ~XHP8711() {}

	virtual void acquireTraceData(unsigned int ch, unsigned int len);
	virtual void convertRawBlock(unsigned int len) throw (XRecordError&);
private:
};

//! Agilent E5061A/5062A Network Analyzer.
class XAgilentE5061 : public XHP8711
{
	XNODE_OBJECT
protected:
	XAgilentE5061(const char *name, bool runtime,
		 const shared_ptr<XScalarEntryList> &scalarentries,
		 const shared_ptr<XInterfaceList> &interfaces,
		 const shared_ptr<XThermometerList> &thermometers,
		 const shared_ptr<XDriverList> &drivers) :
			 XHP8711(name, runtime, scalarentries, interfaces, thermometers, drivers) {}
	virtual ~XAgilentE5061() {}

	virtual void acquireTraceData(unsigned int ch, unsigned int len);
	virtual void convertRawBlock(unsigned int len) throw (XRecordError&);
private:
};

#endif
