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
#ifndef PULSERDRIVERNIDAQ_H_
#define PULSERDRIVERNIDAQ_H_

#include "pulserdrivernidaqmx.h"

#ifdef HAVE_NI_DAQMX

class XNIDAQSSeriesPulser : public XNIDAQmxPulser
{
 XNODE_OBJECT
 protected:
  XNIDAQSSeriesPulser(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers)
   : XNIDAQmxPulser(name, runtime, scalarentries, interfaces, thermometers, drivers) {}
 public:
  virtual ~XNIDAQSSeriesPulser() {}

 protected:
	virtual void open() throw (XInterface::XInterfaceError &);
    //! existense of AO ports.
    virtual bool haveQAMPorts() const {return true;}
};

class XNIDAQMSeriesPulser : public XNIDAQmxPulser
{
 XNODE_OBJECT
 protected:
  XNIDAQMSeriesPulser(const char *name, bool runtime,
   const shared_ptr<XScalarEntryList> &scalarentries,
   const shared_ptr<XInterfaceList> &interfaces,
   const shared_ptr<XThermometerList> &thermometers,
   const shared_ptr<XDriverList> &drivers)
    : XNIDAQmxPulser(name, runtime, scalarentries, interfaces, thermometers, drivers) {}
 public:
  virtual ~XNIDAQMSeriesPulser() {}

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
	virtual const shared_ptr<XNIDAQmxInterface> &intfCtr() const {return m_ctr_interface;} 
 private:
 
	const shared_ptr<XNIDAQmxInterface> m_ao_interface;
	shared_ptr<XNIDAQmxInterface> m_ctr_interface;
	shared_ptr<XListener> m_lsnOnOpenAO, m_lsnOnCloseAO;
	void onOpenAO(const shared_ptr<XInterface> &);
	void onCloseAO(const shared_ptr<XInterface> &);
};

#endif //HAVE_NI_DAQMX

#endif /*PULSERDRIVERNIDAQ_H_*/
