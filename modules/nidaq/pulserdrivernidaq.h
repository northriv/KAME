/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef PULSERDRIVERNIDAQ_H_
#define PULSERDRIVERNIDAQ_H_

#include "pulserdrivernidaqmx.h"

class XNIDAQAODOPulser : public XNIDAQmxPulser
{
	XNODE_OBJECT
protected:
	XNIDAQAODOPulser(const char *name, bool runtime,
					 const shared_ptr<XScalarEntryList> &scalarentries,
					 const shared_ptr<XInterfaceList> &interfaces,
					 const shared_ptr<XThermometerList> &thermometers,
					 const shared_ptr<XDriverList> &drivers)
		: XNIDAQmxPulser(name, runtime, scalarentries, interfaces, thermometers, drivers) {
	}
public:
	virtual ~XNIDAQAODOPulser() {}

protected:
	virtual void open() throw (XInterface::XInterfaceError &);
    //! existense of AO ports.
    virtual bool haveQAMPorts() const {return true;}
};

class XNIDAQDOPulser : public XNIDAQmxPulser
{
	XNODE_OBJECT
protected:
	XNIDAQDOPulser(const char *name, bool runtime,
				   const shared_ptr<XScalarEntryList> &scalarentries,
				   const shared_ptr<XInterfaceList> &interfaces,
				   const shared_ptr<XThermometerList> &thermometers,
				   const shared_ptr<XDriverList> &drivers)
		: XNIDAQmxPulser(name, runtime, scalarentries, interfaces, thermometers, drivers) {
    }
public:
	virtual ~XNIDAQDOPulser() {}

protected:
	virtual void open() throw (XInterface::XInterfaceError &);
    //! existense of AO ports.
    virtual bool haveQAMPorts() const {return false;}
};

class XNIDAQMSeriesWithSSeriesPulser : public XNIDAQmxPulser
{
	XNODE_OBJECT
protected:
	XNIDAQMSeriesWithSSeriesPulser(const char *name, bool runtime,
								   const shared_ptr<XScalarEntryList> &scalarentries,
								   const shared_ptr<XInterfaceList> &interfaces,
								   const shared_ptr<XThermometerList> &thermometers,
								   const shared_ptr<XDriverList> &drivers);
public:
	virtual ~XNIDAQMSeriesWithSSeriesPulser() {}

protected:
	virtual void open() throw (XInterface::XInterfaceError &);
    //! existense of AO ports.
    virtual bool haveQAMPorts() const {return true;}
    
	virtual const shared_ptr<XNIDAQmxInterface> &intfAO() const {return m_ao_interface;} 
	virtual const shared_ptr<XNIDAQmxInterface> &intfCtr() const {return interface();} 
private:
 
	const shared_ptr<XNIDAQmxInterface> m_ao_interface;
};

#endif /*PULSERDRIVERNIDAQ_H_*/
