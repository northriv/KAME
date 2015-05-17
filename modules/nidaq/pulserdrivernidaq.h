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
#ifndef PULSERDRIVERNIDAQ_H_
#define PULSERDRIVERNIDAQ_H_

#include "pulserdrivernidaqmx.h"

class XNIDAQAODOPulser : public XNIDAQmxPulser {
public:
	XNIDAQAODOPulser(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
		: XNIDAQmxPulser(name, runtime, ref(tr_meas), meas) {
	}
	virtual ~XNIDAQAODOPulser() {}

protected:
	virtual void open() throw (XKameError &);
    //! existense of AO ports.
    virtual bool hasQAMPorts() const {return true;}
};

class XNIDAQDOPulser : public XNIDAQmxPulser {
public:
	XNIDAQDOPulser(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas)
		: XNIDAQmxPulser(name, runtime, ref(tr_meas), meas) {
    }
	virtual ~XNIDAQDOPulser() {}

protected:
	virtual void open() throw (XKameError &);
    //! existense of AO ports.
    virtual bool hasQAMPorts() const {return false;}
};

class XNIDAQMSeriesWithSSeriesPulser : public XNIDAQmxPulser {
public:
	XNIDAQMSeriesWithSSeriesPulser(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	virtual ~XNIDAQMSeriesWithSSeriesPulser() {}

protected:
	virtual void open() throw (XKameError &);
    //! existense of AO ports.
    virtual bool hasQAMPorts() const {return true;}
    
	virtual const shared_ptr<XNIDAQmxInterface> &intfAO() const {return m_ao_interface;} 
	virtual const shared_ptr<XNIDAQmxInterface> &intfCtr() const {return interface();}
private:
 
	const shared_ptr<XNIDAQmxInterface> m_ao_interface;
};

#endif /*PULSERDRIVERNIDAQ_H_*/
