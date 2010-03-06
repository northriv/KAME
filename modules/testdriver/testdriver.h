/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
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
#include "primarydriver.h"
#include "dummydriver.h"

class XScalarEntry;

class XTestDriver : public XDummyDriver<XPrimaryDriver> {
public:
	XTestDriver(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
	virtual ~XTestDriver() {}
	//! show all forms belonging to driver
	virtual void showForms();
protected:
	//! Start up your threads, connect GUI, and activate signals
	virtual void start();
	//! Shut down your threads, unconnect GUI, and deactivate signals
	//! this may be called even if driver has already stopped.
	virtual void stop();
  
	//! this is called when raw is written 
	//! unless dependency is broken
	//! convert raw to record
	virtual void analyzeRaw() throw (XRecordError&);
	//! this is called after analyze() or analyzeRaw()
	//! record is readLocked
	virtual void visualize();
private:
	shared_ptr<XThread<XTestDriver> > m_thread;
	double m_x,m_y;
	const shared_ptr<XScalarEntry> m_entryX, m_entryY;
	void *execute(const atomic<bool> &);
  
};

//---------------------------------------------------------------------------
#endif
