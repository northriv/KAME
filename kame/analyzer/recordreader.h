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
#ifndef RECORDREADER_H_
#define RECORDREADER_H_

#include "recorder.h"
#include "xjournalreplay.h"

class XRawStreamRecordReader : public XRawStream {
public:
	XRawStreamRecordReader(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist);
  
	void terminate();
    void join();

	const shared_ptr<XComboNode> &speed() const {return m_speed;}
	const shared_ptr<XBoolNode> &fastForward() const {return m_fastForward;}
	const shared_ptr<XBoolNode> &rewind() const {return m_rewind;}
	const shared_ptr<XTouchableNode> &stop() const {return m_stop;}
	const shared_ptr<XTouchableNode> &first() const {return m_first;}
	const shared_ptr<XTouchableNode> &next() const {return m_next;}
	const shared_ptr<XTouchableNode> &back() const {return m_back;}
	const shared_ptr<XStringNode> &posString() const {return m_posString;}
	//! Where the next record is, in thousandths of the file.  By size, not by
	//! time: the length of a gzip stream is not known until it is read, and
	//! records are not evenly spaced in time anyway.
	const shared_ptr<XUIntNode> &position() const {return m_position;}
	//! Written to go somewhere else, in the same thousandths.  Separate from
	//! position() so that reporting where we are cannot be mistaken for being
	//! asked to move.
	const shared_ptr<XUIntNode> &seek() const {return m_seek;}
	//! The journal opened beside the raw stream, when there is one.  What the
	//! records were taken with, as opposed to what the tree happens to hold
	//! now.  \sa doc/design/PROVENANCE.md
	const XJournalFile &journal() const {return m_journal;}
private:
	struct XRecordError : public XKameError {
        XRecordError(const XString &msg, const char *file, int line)
			: XKameError(msg, file, line) {}
	};
	struct XIOError : public XRecordError {
        XIOError(const char *file, int line);
        XIOError(const XString &msg, const char *file, int line);
	};
	struct XBufferOverflowError : public XIOError {
        XBufferOverflowError(const char *file, int line);
	};
	struct XBrokenRecordError : public XRecordError {
        XBrokenRecordError(const char *file, int line);
	};
	struct XNoDriverError : public XRecordError {
		XNoDriverError(const XString &driver_name, const char *file, int line);
		XString name;
	};
 
	const shared_ptr<XComboNode> m_speed;
	const shared_ptr<XBoolNode> m_fastForward;
	const shared_ptr<XBoolNode> m_rewind;
	const shared_ptr<XTouchableNode> m_stop;
	const shared_ptr<XTouchableNode> m_first, m_next, m_back;
	const shared_ptr<XStringNode> m_posString;
	const shared_ptr<XUIntNode> m_position, m_seek;
	void onPlayCondChanged(const Snapshot &shot, XValueNodeBase *);
	void onStop(const Snapshot &shot, XTouchableNode *);
	void onFirst(const Snapshot &shot, XTouchableNode *);
	void onNext(const Snapshot &shot, XTouchableNode *);
	void onBack(const Snapshot &shot, XTouchableNode *);
  
	void onOpen(const Snapshot &shot, XValueNodeBase *); 
	void onSeek(const Snapshot &shot, XValueNodeBase *);
	shared_ptr<Listener> m_lsnOnOpen, m_lsnOnSeek;
  
	uint32_t m_allsize;
	XTime m_time;
	XJournalFile m_journal;
	//! Of the compressed file, from the filesystem.  The uncompressed length
	//! is not knowable without reading the whole thing, which is the point.
	uint64_t m_fileSize = 0;
	//! Thousandths, or -1.  Set from the UI, acted on by the playback thread:
	//! a seek reads everything before its target, and the main thread must
	//! not be the one waiting for that.
	atomic<int> m_seekRequest {-1};

    //! changes position without parsing
    void first_(void *); // throw (XIOError &)
    //! Back one record.  \return false when there is nothing before this one --
    //! which is an answer, not an error.
    bool stepBack_(void *); // throw (XRecordError &)
    //! To the first whole record at or after \a permille of the file.
    void seek_(void *, int permille); // throw (XRecordError &)
    void reportPosition_(void *);
    int permilleOf_(void *) const;
    void previous_(void *); // throw (XRecordError &)
    void next_(void *); // throw (XRecordError &)
    void goToHeader(void *); // throw (XRecordError &)

    void readHeader(void *); // throw (XRecordError &)
	//! Parse current pos and go next
    void parseOne(void *, XMutex &mutex); //  throw (XRecordError &)

    void gzgetline(void*fd, unsigned char*buf, unsigned int len, int del); // throw (XIOError &)
  
    std::vector<unique_ptr<XThread>> m_threads;
	void *execute(const atomic<bool> &);      
	XCondition m_condition;
	double m_periodicTerm;
	XMutex m_drivermutex;
  
	shared_ptr<Listener> m_lsnStop, m_lsnFirst, m_lsnNext, m_lsnBack;
	shared_ptr<Listener> m_lsnPlayCond;
};

#endif /*RECORDREADER_H_*/
