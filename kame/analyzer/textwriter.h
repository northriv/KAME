/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef textwriterH
#define textwriterH
//---------------------------------------------------------------------------
#include "xnode.h"
#include "xnodeconnector.h"
#include "driver.h"

#include <fstream>

class XScalarEntryList;
class XCalibratedEntryList;

class XTextWriter : public XNode {
public:
	XTextWriter(const char *name, bool runtime,
				const shared_ptr<XDriverList> &driverlist, const shared_ptr<XScalarEntryList> &entrylist);
    //! Also monitor calibrated entries as driver sources for onVisualization.
    void addCalibratedEntrySource(const shared_ptr<XCalibratedEntryList> &calibEntries);

	const shared_ptr<XStringNode> &filename() const {return m_filename;}
	const shared_ptr<XBoolNode> &recording() const {return m_recording;}
	const shared_ptr<XStringNode> &lastLine() const {return m_lastLine;}
	const shared_ptr<XStringNode> &logFilename() const {return m_logFilename;}
	const shared_ptr<XBoolNode> &logRecording() const {return m_logRecording;}
	const shared_ptr<XUIntNode> &logEvery() const {return m_logEvery;}
protected:
	virtual void onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e);
	virtual void onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e);
private:
	const shared_ptr<XDriverList> m_drivers;
	const shared_ptr<XScalarEntryList> m_entries;
	const shared_ptr<XStringNode> m_filename;
	const shared_ptr<XStringNode> m_lastLine;
	const shared_ptr<XBoolNode> m_recording;
	const shared_ptr<XStringNode> m_logFilename;
	const shared_ptr<XBoolNode> m_logRecording;
	const shared_ptr<XUIntNode> m_logEvery;
	shared_ptr<Listener> m_lsnOnVisualization;
	shared_ptr<Listener> m_lsnOnFlush;
	shared_ptr<Listener> m_lsnOnCatch;
	shared_ptr<Listener> m_lsnOnRelease;
	shared_ptr<Listener> m_lsnOnCatchCalib;
	shared_ptr<Listener> m_lsnOnReleaseCalib;
	shared_ptr<Listener> m_lsnOnLastLineChanged;
	shared_ptr<Listener> m_lsnOnFilenameChanged;
	shared_ptr<Listener> m_lsnOnLogFilenameChanged;
	shared_ptr<Listener> m_lsnOnLogRecord;
	void onVisualization(const Snapshot &shot, bool afterRecorded, XDriver *);
	void onFlush(const Snapshot &shot, XValueNodeBase *);
	void onLastLineChanged(const Snapshot &shot, XValueNodeBase *);
	void onFilenameChanged(const Snapshot &shot, XValueNodeBase *);
	void onLogFilenameChanged(const Snapshot &shot, XValueNodeBase *);
    
	std::fstream m_stream;
	std::fstream m_logStream;
	XRecursiveMutex m_filemutex;
	XRecursiveMutex m_logFilemutex;
	XTime m_loggedTime;
};


#endif
