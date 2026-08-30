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
#ifndef JOURNALREADER_H_
#define JOURNALREADER_H_

#include "rawstream.h"
#include "xjournalreplay.h"

#include <deque>

class XJournalReader : public XRawStream {
public:
	//! \a root is where a journal's paths are resolved from -- the whole tree,
	//! not just the drivers, since a run's settings are spread across it.
	//! Weak, because it is this node's own ancestor.
	XJournalReader(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist,
		const weak_ptr<XNode> &root);
  
	void terminate();
    void join();

	const shared_ptr<XComboNode> &speed() const {return m_speed;}
	const shared_ptr<XBoolNode> &fastForward() const {return m_fastForward;}
	const shared_ptr<XBoolNode> &rewind() const {return m_rewind;}
	const shared_ptr<XTouchableNode> &stop() const {return m_stop;}
	const shared_ptr<XTouchableNode> &first() const {return m_first;}
	const shared_ptr<XTouchableNode> &next() const {return m_next;}
	const shared_ptr<XTouchableNode> &back() const {return m_back;}
	const shared_ptr<XStringNode> &recordTime() const {return m_recordTime;}

	//! Whether the raw records are fed back through the drivers at all.
	//!
	//! The settings need no switch: opening the `.kamj` rather than the
	//! `.kamb` is the intent, and saying it twice is ceremony.  What is left
	//! to choose is the expensive half -- untick this and the same transport
	//! walks the settings history alone, which is also what a journal with no
	//! raw stream does.
	const shared_ptr<XBoolNode> &followRaw() const {return m_followRaw;}
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
	const shared_ptr<XStringNode> m_recordTime;
	const shared_ptr<XBoolNode> m_followRaw;
	const shared_ptr<XUIntNode> m_position, m_seek;
	void onPlayCondChanged(const Snapshot &shot, XValueNodeBase *);
	void onStop(const Snapshot &shot, XTouchableNode *);
	void onFirst(const Snapshot &shot, XTouchableNode *);
	void onNext(const Snapshot &shot, XTouchableNode *);
	void onBack(const Snapshot &shot, XTouchableNode *);
  
	void onOpen(const Snapshot &shot, XValueNodeBase *); 
	void onSeek(const Snapshot &shot, XValueNodeBase *);
	//! One value on its way back into the tree, resolved by path.
	struct RestoreItem {
		XString path;
		XString value;
		double exact = 0.0;
		bool hasExact = false;
	};
	//! True when there is no raw stream to play and the journal itself is what
	//! is walked: a session journal, or a run recorded as settings only.  There
	//! is nothing to re-analyse then, but the settings history is all there.
	bool journalOnly() const;
	void journalFirst_();
	//! One step: everything stamped with the next instant in the journal.
	//! \return false at the end of it.
	bool journalStep_();
	//! Back one step, by starting again and replaying to the instant before
	//! this one -- a journal is a stream and only goes forwards.
	void journalBack_();
	void journalSeek_(int permille);
	void reportJournalPosition_(const XTime &when);
	//! The rule for what a replay puts back, shared by both playback paths.
	void takeIfRequest_(const XJournalFile::Event &e, std::vector<RestoreItem> &out) const;

	//! \return how many landed; \a missing counts paths this tree does not have.
	unsigned int applyValues(const std::vector<RestoreItem> &items, unsigned int *missing);
	//! Puts the held dump into the live tree.  \return how many values landed.
	//! \a quiet for the playback paths, which reach it once per lap.
	unsigned int restoreDump(bool quiet = false);
	//! Whether a replay may write to the tree at all: the user asked it to,
	//! and no interface is open to carry a setting to an instrument.
	bool mayApply_() const;
	//! How many interfaces are open, which is what decides whether restoring
	//! a setting reaches an instrument: a driver's I/O listeners exist only
	//! between its start() and stop(), and those follow the interface.
	unsigned int openInterfaces() const;
	shared_ptr<Listener> m_lsnOnOpen, m_lsnOnSeek;

	//! One value out of the dump, kept by the id the journal gave its node.
	struct DumpValue {
		uint32_t id = 0;
		XString value;
		double exact = 0.0;
		bool hasExact = false;
	};
	//! The state the run started in, read at open and held until another file
	//! is opened.  One entry per value node, so it is bounded by the tree.
	std::vector<DumpValue> m_dump;
	const weak_ptr<XNode> m_root;
  
	uint32_t m_allsize;
	//! Of the record last read: records written before the magic existed have
	//! a shorter one, and both kinds are read without being told which.
	uint32_t m_headerBytes = KAMB_HEADER_SIZE_LEGACY;
	XTime m_time;
	XJournalFile m_journal;
	//! Of the compressed file, from the filesystem.  The uncompressed length
	//! is not knowable without reading the whole thing, which is the point.
	uint64_t m_fileSize = 0;
	//! Of the journal, for the position readout when it is what is being
	//! walked.  Compressed, as m_fileSize is, and for the same reason.
	uint64_t m_journalSize = 0;
	//! Set when the file that was opened was the journal, which is how the
	//! user says the settings are wanted.  Opening the raw stream instead
	//! means today's settings -- re-running a recording with one parameter
	//! changed, which is the normal scientific move.
	bool m_restoreWanted = false;
	//! Instants already stepped through, so that going back one has something
	//! to aim at.  Bounded: past its cap, back goes to the beginning instead.
	std::deque<XTime> m_journalVisited;
	//! The cursor is behind the record about to be played -- set whenever the
	//! raw stream is moved backwards, since a journal can only be walked
	//! forwards.  Acted on where the settings are applied, not where the seek
	//! happens, because rewinding means re-applying the dump.
	atomic<bool> m_journalRewind {false};
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

#endif /*JOURNALREADER_H_*/
