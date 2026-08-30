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

#ifndef rawstreamH
#define rawstreamH
//---------------------------------------------------------------------------
#include "xnode.h"
#include "xnodeconnector.h"
#include "driver.h"

#define MAX_RAW_RECORD_SIZE 100000000uL

//! A raw record starts with this.
//!
//! 'K','A','M','B' little-endian is 0x424d414b, which is far larger than
//! MAX_RAW_RECORD_SIZE -- so four bytes that are a plausible *length* instead
//! are unmistakably a record written before this existed, and both kinds can
//! be read without being told which is which.  \sa doc/design/PROVENANCE.md
#define KAMB_RECORD_MAGIC 0x424d414buL

//! Bytes before the driver name: magic, check, allsize, sec, usec.
#define KAMB_HEADER_SIZE (5 * sizeof(uint32_t))
//! The same, for a record written before the magic existed.
#define KAMB_HEADER_SIZE_LEGACY (3 * sizeof(uint32_t))

//! FNV-1a over the twelve bytes the header commits to.
//!
//! Not a checksum of the data -- gzip already carries one of those -- but a
//! test that these bytes really are a record header rather than something
//! that happens to look like one, which is what searching a file from the
//! middle needs.  Over the *values*, so it does not depend on the host's
//! byte order; over allsize as well as the time, because the length is what
//! a scan actually relies on afterwards.
inline uint32_t kamb_record_check(uint32_t allsize, int32_t sec, int32_t usec) {
    uint32_t w[3] = {allsize, (uint32_t)sec, (uint32_t)usec};
    uint32_t h = 2166136261u;
    for(int i = 0; i < 3; ++i)
        for(int b = 0; b < 4; ++b) {
            h ^= (w[i] >> (8 * b)) & 0xffu;
            h *= 16777619u;
        }
    return h;
}

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
	const shared_ptr<XBoolNode> &recording() const {return m_recording;}
	//! Bytes handed to zlib since this process started -- what the run is
	//! costing, for the Journal group's readout.  Uncompressed: it is the
	//! rate the instrument is producing, not what the disk ends up with.
	uintptr_t bytesWritten() const {return m_bytesWritten;}
protected:
	virtual void onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e);
	virtual void onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e);
private:
	void onOpen(const Snapshot &shot, XValueNodeBase *);
  
	shared_ptr<Listener> m_lsnOnRecord;
	shared_ptr<Listener> m_lsnOnCatch;
	shared_ptr<Listener> m_lsnOnRelease;
	shared_ptr<Listener> m_lsnOnFlush;
	shared_ptr<Listener> m_lsnOnOpen;
  
	void onRecord(const Snapshot &shot, XDriver *driver);
	void onFlush(const Snapshot &shot, XValueNodeBase *);
	const shared_ptr<XBoolNode> m_recording;
	atomic<uintptr_t> m_bytesWritten {0};
	//! Last Z_FULL_FLUSH.  \sa onRecord()
	XTime m_lastFlushed;
};



#endif
