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

#ifndef recorderH
#define recorderH
//---------------------------------------------------------------------------
#include "xnode.h"
#include "xnodeconnector.h"
#include "driver.h"

#include <fstream>

#define MAX_RAW_RECORD_SIZE 1000000

class XRawStream : public XNode {
public:
	XRawStream(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist);
	virtual ~XRawStream();
	const shared_ptr<XStringNode> &filename() const {return m_filename;}  
protected:
	shared_ptr<XDriverList> m_drivers;
	//! file descriptor of GZip
	void *m_pGFD;
	XMutex m_filemutex;
private:
	shared_ptr<XStringNode> m_filename;

};
class XRawStreamRecorder : public XRawStream {
public:
	XRawStreamRecorder(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist);
	virtual ~XRawStreamRecorder() {}
	const shared_ptr<XBoolNode> &recording() const {return m_recording;}
protected:
	virtual void onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e);
	virtual void onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e);
private:
	void onOpen(const shared_ptr<XValueNodeBase> &);
  
	shared_ptr<XListener> m_lsnOnRecord;
	shared_ptr<XListener> m_lsnOnCatch;
	shared_ptr<XListener> m_lsnOnRelease;
	shared_ptr<XListener> m_lsnOnFlush;
	shared_ptr<XListener> m_lsnOnOpen;
  
	void onRecord(const Snapshot &shot, XDriver *driver);
	void onFlush(const shared_ptr<XValueNodeBase> &);
	const shared_ptr<XBoolNode> m_recording;
};


class XScalarEntryList;

class XTextWriter : public XNode {
public:
	XTextWriter(const char *name, bool runtime,
				const shared_ptr<XDriverList> &driverlist, const shared_ptr<XScalarEntryList> &entrylist);
	virtual ~XTextWriter() {}

	const shared_ptr<XStringNode> &filename() const {return m_filename;}
	const shared_ptr<XBoolNode> &recording() const {return m_recording;}
	const shared_ptr<XStringNode> &lastLine() const {return m_lastLine;}

protected:
	virtual void onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e);
	virtual void onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e);
private:
	const shared_ptr<XDriverList> m_drivers;
	const shared_ptr<XScalarEntryList> m_entries;
	const shared_ptr<XStringNode> m_filename;
	const shared_ptr<XStringNode> m_lastLine;
	const shared_ptr<XBoolNode> m_recording;
	shared_ptr<XListener> m_lsnOnRecord;
	shared_ptr<XListener> m_lsnOnFlush;
	shared_ptr<XListener> m_lsnOnCatch;
	shared_ptr<XListener> m_lsnOnRelease; 
	shared_ptr<XListener> m_lsnOnLastLineChanged; 
	shared_ptr<XListener> m_lsnOnFilenameChanged;
	void onRecord(const Snapshot &shot, XDriver *driver);
	void onFlush(const shared_ptr<XValueNodeBase> &);
	void onLastLineChanged(const shared_ptr<XValueNodeBase> &);
	void onFilenameChanged(const shared_ptr<XValueNodeBase> &);
    
	std::fstream m_stream;
	XRecursiveMutex m_filemutex;
};


#endif
