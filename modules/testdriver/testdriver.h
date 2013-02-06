/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef testdriverH
#define testdriverH
//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "dummydriver.h"

class XScalarEntry;

class XTestDriver : public XDummyDriver<XPrimaryDriverWithThread> {
public:
	XTestDriver(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
	virtual ~XTestDriver() {}
	//! show all forms belonging to driver
	virtual void showForms();

	struct Payload : public XPrimaryDriver::Payload {
		double x() const {return m_x;}
		double y() const {return m_y;}
	private:
		friend class XTestDriver;
		double m_x,m_y;
	};
protected:
	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
	virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);
private:
	const shared_ptr<XScalarEntry> m_entryX, m_entryY;
	void *execute(const atomic<bool> &);
  
};

//---------------------------------------------------------------------------
#endif
