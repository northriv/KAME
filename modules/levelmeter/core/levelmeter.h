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
#ifndef levelmeterH
#define levelmeterH
//---------------------------------------------------------------------------
#include "primarydriverwiththread.h"
#include "xnodeconnector.h"

class XScalarEntry;

class XLevelMeter : public XPrimaryDriverWithThread {
public:
	XLevelMeter(const char *name, bool runtime,
		Transaction &tr_meas, const shared_ptr<XMeasure> &meas);
	//! usually nothing to do
	virtual ~XLevelMeter() {}
	//! Shows all forms belonging to driver
	virtual void showForms();
 
	struct Payload : public XPrimaryDriver::Payload {
		unsigned int channelNum() const {return m_levels.size();}
		double level(unsigned int ch) const {return m_levels[ch];}
	private:
		friend class XLevelMeter;
		std::vector<double> m_levels;
	};
protected:
	//! This function will be called when raw data are written.
	//! Implement this function to convert the raw data to the record (Payload).
	//! \sa analyze()
	virtual void analyzeRaw(RawDataReader &reader, Transaction &tr) throw (XRecordError&);
	//! This function is called after committing XPrimaryDriver::analyzeRaw() or XSecondaryDriver::analyze().
	//! This might be called even if the record is invalid (time() == false).
	virtual void visualize(const Snapshot &shot);
  
	//! driver specific part below
protected:
	//! register channel names in your constructor
	//! \param channel_names array of pointers to channel name. ends with null pointer.
	void createChannels(Transaction &tr_meas, const shared_ptr<XMeasure> &meas,
						const char **channel_names);

	virtual double getLevel(unsigned int ch) = 0;
private:
	std::deque<shared_ptr<XScalarEntry> > m_entries;
    
	void *execute(const atomic<bool> &);  
};

#endif
