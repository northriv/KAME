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
#include "recordreader.h"
#include "analyzer.h"
#include "primarydriver.h"
#include "xtime.h"
#include "measure.h"

#include <zlib.h>
#include <vector>

#include <QFileInfo>
#include <QString>

#define IFSMODE std::ios::in
#define SPEED_FASTEST "Fastest"
#define SPEED_FAST "Fast"
#define SPEED_NORMAL "Normal"
#define SPEED_SLOW "Slow"

#define RECORDREADER_DELAY 20
#define RECORD_READER_NUM_THREADS 1

XRawStreamRecordReader::XIOError::XIOError(const char *file, int line)
	: XRecordError(i18n("IO Error"), file, line) {}
XRawStreamRecordReader::XIOError::XIOError(const XString &msg, const char *file, int line)
	: XRecordError(msg, file, line) {}
XRawStreamRecordReader::XBufferOverflowError::XBufferOverflowError(const char *file, int line)
	: XIOError(i18n("Buffer Overflow Error"), file, line) {}
XRawStreamRecordReader::XBrokenRecordError::XBrokenRecordError(const char *file, int line)
	: XRecordError(i18n("Broken Record Error"), file, line) {}
XRawStreamRecordReader::XNoDriverError::
XNoDriverError(const XString &driver_name, const char *file, int line)
	: XRecordError(i18n("No Driver Error: ") + driver_name, file, line),
	  name(driver_name) {}
         
XRawStreamRecordReader::XRawStreamRecordReader(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist)
	: XRawStream(name, runtime, driverlist),
	  m_speed(create<XComboNode>("Speed", true, true)),
	  m_fastForward(create<XBoolNode>("FastForward", true)),
	  m_rewind(create<XBoolNode>("Rewind", true)),
	  m_stop(create<XTouchableNode>("Stop", true)),
	  m_first(create<XTouchableNode>("First", true)),
	  m_next(create<XTouchableNode>("Next", true)),
	  m_back(create<XTouchableNode>("Back", true)),
	  m_posString(create<XStringNode>("PosString", true)),
	  m_position(create<XUIntNode>("Position", true)),
	  m_seek(create<XUIntNode>("Seek", true)),
	  m_periodicTerm(0) {

    iterate_commit([=](Transaction &tr){
        tr[ *m_speed].add(SPEED_FASTEST);
        tr[ *m_speed].add(SPEED_FAST);
        tr[ *m_speed].add(SPEED_NORMAL);
        tr[ *m_speed].add(SPEED_SLOW);
        tr[ *m_speed] = SPEED_FAST;

        m_lsnOnOpen = tr[ *filename()].onValueChanged().connectWeakly(
            shared_from_this(), &XRawStreamRecordReader::onOpen);
        m_lsnOnSeek = tr[ *m_seek].onValueChanged().connectWeakly(
            shared_from_this(), &XRawStreamRecordReader::onSeek);
		m_lsnFirst = tr[ *m_first].onTouch().connectWeakly(
			shared_from_this(), &XRawStreamRecordReader::onFirst,
			Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
		m_lsnBack = tr[ *m_back].onTouch().connectWeakly(
			shared_from_this(), &XRawStreamRecordReader::onBack,
			Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
		m_lsnNext = tr[ *m_next].onTouch().connectWeakly(
			shared_from_this(), &XRawStreamRecordReader::onNext,
			Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
		m_lsnStop = tr[ *m_stop].onTouch().connectWeakly(
			shared_from_this(), &XRawStreamRecordReader::onStop,
			Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
	    m_lsnPlayCond = tr[ *m_fastForward].onValueChanged().connectWeakly(
			shared_from_this(),
			&XRawStreamRecordReader::onPlayCondChanged,
			Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
	    tr[ *m_rewind].onValueChanged().connect(m_lsnPlayCond);
	    tr[ *m_speed].onValueChanged().connect(m_lsnPlayCond);
    });
    
    for(int i = 0; i < RECORD_READER_NUM_THREADS; ++i) {
        m_threads.emplace_back(new XThread(shared_from_this(), &XRawStreamRecordReader::execute));
    }
}
//! A run is two files, and either name opens the pair.
//!
//! The `.kamj` is the better one to be given: its header names its raw stream
//! outright, so a pair that has been renamed still finds itself.  From a
//! `.kamb` the sibling is guessed by stem, which is what the writer produced
//! and all that can be done from that end.
//!
//! Neither file is required to have the other.  A raw stream from before the
//! journal existed replays as it always did -- with today's settings, which
//! is policy `off` and a legitimate use, not a degraded one.
void
XRawStreamRecordReader::onOpen(const Snapshot &shot, XValueNodeBase *) {
	XString given = ( **filename())->to_str();
	XString rawpath = given, journalpath;
	if(QString::fromStdString(given).endsWith(".kamj", Qt::CaseInsensitive)) {
		journalpath = given;
		rawpath.clear();
	}
	else {
		journalpath = XJournalFile::journalBeside(given);
	}

	XJournalFile journal;
	if(journalpath.length()) {
		XString err;
		//Nothing is restored yet, so the dump goes nowhere; opening still
		//reads it, which is what settles whether this is a journal at all.
		if( !journal.open(journalpath, [](const XJournalFile::Event &){}, err))
			gWarnPrint(i18n("Journal: ") + journalpath + " " + err);
	}
	if(journal.isOpen()) {
		if(rawpath.empty())
			rawpath = journal.rawPath();
		if( !journal.timesKnown())
			gWarnPrint(i18n("This journal predates the machine-readable timestamp; "
				"its entries cannot be placed against the records: ") + journalpath);
		if(rawpath.empty())
			gWarnPrint(i18n("This journal recorded no raw stream: ") + journalpath);
		else
			gMessagePrint(i18n("Journal: ") + journalpath
				+ formatString(" (%s, %u nodes)", journal.kind().c_str(),
					(unsigned)journal.nodes().size()));
	}

	m_filemutex.lock();
	m_journal = std::move(journal);
	if(m_pGFD) gzclose(static_cast<gzFile>(m_pGFD));
	m_pGFD = rawpath.length() ?
		gzopen(QString::fromStdString(rawpath).toLocal8Bit().data(), "rb") : nullptr;
	//By size of the compressed file, which the filesystem knows.  How long a
	//gzip stream is when unpacked is not knowable without unpacking it, and
	//that is exactly what a position readout must not do.
	m_fileSize = m_pGFD ?
		(uint64_t)QFileInfo(QString::fromStdString(rawpath)).size() : 0;
	if(m_pGFD)
		reportPosition_(m_pGFD);
	m_filemutex.unlock();

	//A file that will not open used to be silent, and every button then did
	//nothing for no stated reason.
	if(rawpath.length() && !m_pGFD)
		gErrPrint(i18n("Cannot open ") + rawpath);
}
void
XRawStreamRecordReader::readHeader(void *_fd) {
	gzFile fd = static_cast<gzFile>(_fd);

	if(gzeof(fd))
		throw XIOError(__FILE__, __LINE__);
	//Four bytes first, because they decide which layout this is: the magic,
	//or -- in a file written before it existed -- the length itself.
	std::vector<char> head(sizeof(uint32_t));
	if(gzread(fd, &head[0], (unsigned)head.size()) == -1) throw XIOError(__FILE__, __LINE__);
	uint32_t first;
	{
		XPrimaryDriver::RawDataReader reader(head);
		first = reader.pop<uint32_t>();
	}
	bool magic = (first == (uint32_t)KAMB_RECORD_MAGIC);
	m_headerBytes = magic ? KAMB_HEADER_SIZE : KAMB_HEADER_SIZE_LEGACY;
	std::vector<char> buf(m_headerBytes - sizeof(uint32_t));
	if(gzread(fd, &buf[0], (unsigned)buf.size()) == -1) throw XIOError(__FILE__, __LINE__);
	XPrimaryDriver::RawDataReader reader(buf);
	uint32_t check = magic ? reader.pop<uint32_t>() : 0;
	m_allsize = magic ? reader.pop<uint32_t>() : first;
	long sec = reader.pop<int32_t>();
	long usec = reader.pop<int32_t>();
	if(magic && (check != kamb_record_check(m_allsize, (int32_t)sec, (int32_t)usec)))
		throw XBrokenRecordError(__FILE__, __LINE__);
    m_time = XTime(sec, usec);
}
void
XRawStreamRecordReader::parseOne(void *_fd, XMutex &mutex) {
	gzFile fd = static_cast<gzFile>(_fd);

	readHeader(fd);
	char name[256], sup[256];
	gzgetline(fd, (unsigned char*)name, 256, '\0');
	gzgetline(fd, (unsigned char*)sup, 256, '\0');
	if(strlen(name) == 0) {
		throw XBrokenRecordError(__FILE__, __LINE__);
	}
	shared_ptr<XNode> driver_precast = m_drivers->getChild(name);
	auto driver = dynamic_pointer_cast<XPrimaryDriver>(driver_precast);
	uint32_t size = 
		m_allsize - (
			m_headerBytes //magic, check, allsize, sec, usec -- or fewer, in an old file
			+ strlen(name) //name of driver
			+ strlen(sup) //reserved
			+ 2 //two null chars
			+ sizeof(uint32_t)  //allsize
			);
    // m_time must be copied before unlocking
    XTime time(m_time);
    //Both read before the transaction: an iterate_commit closure re-runs on
    //every retry, and nothing inside it may touch the file.
    XString timestr = time.getTimeStr();
    unsigned int permille = (unsigned int)permilleOf_(fd);
    iterate_commit([&](Transaction &tr){
        tr[ *m_posString] = timestr;
        tr[ *m_position] = permille;
    });
    if( !driver || (size > MAX_RAW_RECORD_SIZE)) {
        if(gzseek(fd, size + sizeof(uint32_t), SEEK_CUR) == -1)
			throw XIOError(__FILE__, __LINE__);
		if(driver)
			throw XBrokenRecordError(__FILE__, __LINE__);
		if(driver_precast)
	        throw XNoDriverError(formatString_tr(I18N_NOOP("Typemismatch: %s"), name),
	         __FILE__, __LINE__);
		else
	        throw XNoDriverError(name, __FILE__, __LINE__);
    }
    auto rawdata = std::make_shared<XPrimaryDriver::RawData>();
	try {
		rawdata->resize(size);
		if(gzread(fd, &rawdata->at(0), size) == -1)
			throw XIOError(__FILE__, __LINE__);
		std::vector<char> buf(sizeof(uint32_t));
		if(gzread(fd, &buf[0], sizeof(uint32_t)) == -1)
			throw XIOError(__FILE__, __LINE__);
		XPrimaryDriver::RawDataReader reader(buf);
		uint32_t footer_allsize = reader.pop<uint32_t>();
		if(footer_allsize != m_allsize)
			throw XBrokenRecordError(__FILE__, __LINE__);
	}
	catch (XRecordError &e) {
		driver->finishWritingRaw(rawdata, XTime(), XTime());
		throw e;
	}
	mutex.unlock();
	{ XScopedLock<XMutex> lock(m_drivermutex);
	driver->finishWritingRaw(rawdata, XTime::now(), time);
	}
}
void
XRawStreamRecordReader::gzgetline(void* _fd, unsigned char*buf, unsigned int len, int del) {
	gzFile fd = static_cast<gzFile>(_fd);

	int c;
	for(unsigned int i = 0; i < len; i++) {
		c = gzgetc(fd);
		if(c == -1) throw XIOError(__FILE__, __LINE__);
		*(buf++) = (unsigned char)c;
		if(c == del) return;
	}
	throw XBufferOverflowError(__FILE__, __LINE__);
}
void
XRawStreamRecordReader::first_(void *fd) {
	gzrewind(static_cast<gzFile>(fd));
}
//! Where the next record is, in thousandths of the compressed file.
int
XRawStreamRecordReader::permilleOf_(void *fd) const {
	if( !m_fileSize)
		return 0;
	auto off = gzoffset(static_cast<gzFile>(fd));
	if(off < 0)
		return 0;
	uint64_t p = (uint64_t)off * 1000u / m_fileSize;
	return (int)std::min(p, (uint64_t)1000u);
}
void
XRawStreamRecordReader::reportPosition_(void *fd) {
	int permille = permilleOf_(fd);
	trans( *m_position) = (unsigned int)permille;
}
//! Back one record.  At the head of the file there is nothing before the
//! first record, and saying so is not the same as failing: seeking to the
//! start and then stepping back used to raise an IO error, because the seek
//! for the length word ran off the front of the file.
bool
XRawStreamRecordReader::stepBack_(void *fd) {
	if(gztell(static_cast<gzFile>(fd)) == 0)
		return false;   //!< nothing has been read: the first record is next anyway
	previous_(fd);      //!< to the start of the record just parsed
	if(gztell(static_cast<gzFile>(fd)) == 0)
		return false;   //!< that was the first one; it is what comes next
	previous_(fd);
	return true;
}
//! To the first whole record at or after a point in the file.
//!
//! A gzip stream has no index, so reaching a point costs decompressing
//! everything before it.  That is why this runs on the playback thread and
//! not on the one drawing the window.
//!
//! Landing is by structure, not by luck: a record carries its own length as
//! the first and the last four bytes, so a boundary is where those two agree.
//! Four bytes matching a length that lands exactly on itself is not something
//! arbitrary data does often.
void
XRawStreamRecordReader::seek_(void *_fd, int permille) {
	gzFile fd = static_cast<gzFile>(_fd);
	if((permille <= 0) || !m_fileSize) {
		first_(fd);
		return;
	}
	uint64_t target = m_fileSize * (uint64_t)permille / 1000u;
	if(gzrewind(fd) != 0)
		throw XIOError(__FILE__, __LINE__);
	std::vector<char> skip(65536);
	while( !gzeof(fd) && ((uint64_t)std::max<int64_t>(gzoffset(fd), 0) < target)) {
		int n = gzread(fd, &skip[0], (unsigned)skip.size());
		if(n < 0) throw XIOError(__FILE__, __LINE__);
		if(n == 0) break;
	}
	//The length words are little-endian on disk whatever the host is
	//(RawDataReader::pop() reads them that way), so read them as bytes.
	std::vector<unsigned char> win(1024 * 1024);
	for(int chunk = 0; chunk < 8; ++chunk) {
		int64_t base = gztell(fd);
		int n = gzread(fd, &win[0], (unsigned)win.size());
		if(n < 0) throw XIOError(__FILE__, __LINE__);
		if(n < 24) break;
		for(int i = 0; i + (int)KAMB_HEADER_SIZE <= n; ++i) {
			auto le32 = [&](int at) -> uint32_t {
				const unsigned char *p = &win[at];
				return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
					| ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
			};
			uint32_t allsize;
			if(le32(i) == (uint32_t)KAMB_RECORD_MAGIC) {
				//The record says what it is.  Magic and check together are
				//2^-64 against arbitrary bytes, so this is the answer, not a
				//candidate -- and it holds without reading anything else.
				allsize = le32(i + 8);
				if((allsize < KAMB_HEADER_SIZE + 6) || (allsize > MAX_RAW_RECORD_SIZE))
					continue;
				if(le32(i + 4) != kamb_record_check(allsize,
						(int32_t)le32(i + 12), (int32_t)le32(i + 16)))
					continue;
			}
			else {
				//Written before the magic existed.  All such a record offers
				//is its length at both ends, so a boundary is where the two
				//agree -- weaker, and the reason the magic was added.
				allsize = le32(i);
				if((allsize < KAMB_HEADER_SIZE_LEGACY + 6) || (allsize > MAX_RAW_RECORD_SIZE))
					continue;
				if((int64_t)i + allsize > n)
					continue;   //!< cannot be checked from here; a later window will
				if(le32(i + allsize - 4) != allsize)
					continue;
			}
			//Found.  Going back to it rewinds and re-reads, since zlib cannot
			//seek backwards either -- twice the work, and still the only way.
			if(gzseek(fd, base + i, SEEK_SET) == -1)
				throw XIOError(__FILE__, __LINE__);
			return;
		}
		if(n < (int)win.size())
			break;  //!< end of file, and no boundary after the target
	}
	//Past the last record, or a stream that says nothing we recognise.
	first_(fd);
}
void
XRawStreamRecordReader::onSeek(const Snapshot &shot, XValueNodeBase *) {
	//Handed to the playback thread rather than done here: this call arrives
	//on whichever thread moved the slider, and a seek reads everything before
	//its target.
	m_seekRequest = (int)(unsigned int)shot[ *m_seek];
	XScopedLock<XCondition> lock(m_condition);
	m_condition.broadcast();
}
void
XRawStreamRecordReader::previous_(void *fd) {
	if(gzseek(static_cast<gzFile>(fd), -sizeof(uint32_t), SEEK_CUR) == -1) throw XIOError(__FILE__, __LINE__);
	goToHeader(fd);
}
void
XRawStreamRecordReader::next_(void *fd) {
	readHeader(fd);
	if(gzseek(static_cast<gzFile>(fd), m_allsize - m_headerBytes, SEEK_CUR) == -1)
		throw XIOError(__FILE__, __LINE__);
}
void
XRawStreamRecordReader::goToHeader(void *_fd) {
	gzFile fd = static_cast<gzFile>(_fd);

	if(gzeof(fd)) throw XIOError(__FILE__, __LINE__);
	std::vector<char> buf(sizeof(uint32_t));
	XPrimaryDriver::RawDataReader reader(buf);
	if(gzread(fd, &buf[0], sizeof(uint32_t)) == Z_NULL) throw XIOError(__FILE__, __LINE__);
	int allsize = reader.pop<uint32_t>();
	if(gzseek(fd, -allsize, SEEK_CUR) == -1) throw XIOError(__FILE__, __LINE__);
}
void
XRawStreamRecordReader::terminate() {
    m_periodicTerm = 0;
    for(auto &&x: m_threads) {
        x->terminate();
    }
    XScopedLock<XCondition> lock(m_condition);
    m_condition.broadcast();
}
void
XRawStreamRecordReader::join() {
    for(auto &&x: m_threads) {
        x->join();
    }
}

void
XRawStreamRecordReader::onPlayCondChanged(const Snapshot &shot, XValueNodeBase *) {
	Snapshot shot_this( *this);
    double ms = 1.0;
    if(shot_this[ *m_speed].to_str() == SPEED_FASTEST) ms = 0.1;
    if(shot_this[ *m_speed].to_str() == SPEED_FAST) ms = 10.0;
    if(shot_this[ *m_speed].to_str() == SPEED_NORMAL) ms = 30.0;
    if(shot_this[ *m_speed].to_str() == SPEED_SLOW) ms = 100.0;
    if( !shot_this[ *m_fastForward] && !shot_this[ *m_rewind]) ms = 0;
    if(shot_this[ *m_rewind]) ms = -ms;
    m_periodicTerm = ms;
    XScopedLock<XCondition> lock(m_condition);
    m_condition.broadcast();
}
void
XRawStreamRecordReader::onStop(const Snapshot &shot, XTouchableNode *) {
    m_periodicTerm = 0;
    g_statusPrinter->printMessage(i18n("Stopped"));
	iterate_commit([=](Transaction &tr){
		tr[ *m_fastForward] = false;
		tr[ *m_rewind] = false;
		tr.unmark(m_lsnPlayCond);
    });
}
void
XRawStreamRecordReader::onFirst(const Snapshot &shot, XTouchableNode *) {
	if(m_pGFD) {
		try {
			m_filemutex.lock();
			first_(m_pGFD);
			parseOne(m_pGFD, m_filemutex);
			g_statusPrinter->printMessage(i18n("First"));
		}
		catch (XRecordError &e) {
			m_filemutex.unlock();
			e.print(i18n("No Record, because "));
		}
	}
}
void
XRawStreamRecordReader::onNext(const Snapshot &shot, XTouchableNode *) {
	if(m_pGFD) {
		try {
			m_filemutex.lock(); 
			parseOne(m_pGFD, m_filemutex);
			g_statusPrinter->printMessage(i18n("Next"));
		}
		catch (XRecordError &e) {
			m_filemutex.unlock();
			e.print(i18n("No Record, because "));
		}
	}
}
void
XRawStreamRecordReader::onBack(const Snapshot &shot, XTouchableNode *) {
	if(m_pGFD) {
		try {
			m_filemutex.lock(); 
			bool moved = stepBack_(m_pGFD);
			parseOne(m_pGFD, m_filemutex);
			g_statusPrinter->printMessage(moved ? i18n("Previous") : i18n("First"));
		}
		catch (XRecordError &e) {
			m_filemutex.unlock();
			e.print(i18n("No Record, because "));
		}
	}
}

void *XRawStreamRecordReader::execute(const atomic<bool> &terminated) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);
    while( !terminated) {
		double ms = 0.0;
		{
			XScopedLock<XCondition> lock(m_condition);
			while((fabs((ms = m_periodicTerm)) < 1e-4) && (m_seekRequest < 0) && !terminated)
				m_condition.wait();
		}
    
		if(terminated) break;

		if( !m_pGFD) {
			m_seekRequest = -1;     //!< nothing to seek in
			msecsleep(100);
			continue;
		}

		//Claimed here, so a slider dragged while playing moves the same
		//cursor the playback is walking, rather than racing it.
		int req = m_seekRequest.exchange(-1);

		try {
			m_filemutex.lock();
			if(req >= 0) {
				seek_(m_pGFD, req);
			}
			else if(ms > 0.0 && gzeof(static_cast<gzFile>(m_pGFD))) {
				first_(m_pGFD);
			}
			else if(ms < 0.0) {
				if( !stepBack_(m_pGFD)) {
					//At the head.  Stopping is the honest end of a rewind;
					//the alternative is replaying the first record for ever.
					m_periodicTerm = 0.0;
					m_filemutex.unlock();
					iterate_commit([=](Transaction &tr){
						tr[ *m_rewind] = false;
						tr.unmark(m_lsnPlayCond);
					});
					g_statusPrinter->printMessage(i18n("First"));
					continue;
				}
			}
			parseOne(m_pGFD, m_filemutex);
		}
		catch (XNoDriverError &e) {
			m_filemutex.unlock();
			e.print(i18n("No such driver :") + e.name);
		}
		catch (XRecordError &e) {
			m_periodicTerm = 0.0;
			iterate_commit([=](Transaction &tr){
				tr[ *m_fastForward] = false;
				tr[ *m_rewind] = false;
				tr.unmark(m_lsnPlayCond);
            });
			m_filemutex.unlock();
			e.print(i18n("No Record, because "));
		}
     
		msecsleep(lrint(fabs(ms)));
	}
    return NULL;
}
