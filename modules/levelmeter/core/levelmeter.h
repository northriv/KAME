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
#ifndef levelmeterH
#define levelmeterH
//---------------------------------------------------------------------------
#include "primarydriver.h"
#include "xnodeconnector.h"

class XScalarEntry;

class XLevelMeter : public XPrimaryDriver
{
	XNODE_OBJECT
protected:
	XLevelMeter(const char *name, bool runtime,
			  const shared_ptr<XScalarEntryList> &scalarentries,
			  const shared_ptr<XInterfaceList> &interfaces,
			  const shared_ptr<XThermometerList> &thermometers,
			  const shared_ptr<XDriverList> &drivers);
public:
	//! usually nothing to do
	virtual ~XLevelMeter() {}
	//! show all forms belonging to driver
	virtual void showForms();
 
	//! Records
	unsigned int channelNumRecorded() const {return m_levelRecorded.size();}
	double levelRecorded(unsigned int ch) const {return m_levelRecorded[ch];}
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
  
	//! driver specific part below
protected:
	//! register channel names in your constructor
	//! \param channel_names array of pointers to channel name. ends with null pointer.
	void createChannels(const shared_ptr<XScalarEntryList> &scalarentries,
						const char **channel_names);

	virtual double getLevel(unsigned int ch) = 0;
private:
 
	shared_ptr<XThread<XLevelMeter> > m_thread;
  
	//! Records
	std::vector<double> m_levelRecorded;

	std::deque<shared_ptr<XScalarEntry> > m_entries;
    
	void *execute(const atomic<bool> &);  
};

#endif
