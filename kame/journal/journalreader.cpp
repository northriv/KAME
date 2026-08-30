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
#include "journalreader.h"
#include "analyzer.h"
#include "primarydriver.h"
#include "xtime.h"
#include "measure.h"
#include "interface.h"
#include "xlistnode.h"

#include <zlib.h>
#include <algorithm>
#include <vector>

#include <QFileInfo>
#include <QString>
#include <QStringList>

#define IFSMODE std::ios::in
#define SPEED_FASTEST "Fastest"
#define SPEED_FAST "Fast"
#define SPEED_NORMAL "Normal"
#define SPEED_SLOW "Slow"

#define RECORDREADER_DELAY 20
#define RECORD_READER_NUM_THREADS 1

XJournalReader::XIOError::XIOError(const char *file, int line)
	: XRecordError(i18n("IO Error"), file, line) {}
XJournalReader::XIOError::XIOError(const XString &msg, const char *file, int line)
	: XRecordError(msg, file, line) {}
XJournalReader::XBufferOverflowError::XBufferOverflowError(const char *file, int line)
	: XIOError(i18n("Buffer Overflow Error"), file, line) {}
XJournalReader::XBrokenRecordError::XBrokenRecordError(const char *file, int line)
	: XRecordError(i18n("Broken Record Error"), file, line) {}
XJournalReader::XNoDriverError::
XNoDriverError(const XString &driver_name, const char *file, int line)
	: XRecordError(i18n("No Driver Error: ") + driver_name, file, line),
	  name(driver_name) {}
         
XJournalReader::XJournalReader(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist,
	const weak_ptr<XNode> &root)
	: XRawStream(name, runtime, driverlist),
	  m_root(root),
	  m_speed(create<XComboNode>("Speed", true, true)),
	  m_fastForward(create<XBoolNode>("FastForward", true)),
	  m_rewind(create<XBoolNode>("Rewind", true)),
	  m_stop(create<XTouchableNode>("Stop", true)),
	  m_first(create<XTouchableNode>("First", true)),
	  m_next(create<XTouchableNode>("Next", true)),
	  m_back(create<XTouchableNode>("Back", true)),
	  m_recordTime(create<XStringNode>("RecordTime", true)),
	  m_position(create<XUIntNode>("Position", true)),
	  m_seek(create<XUIntNode>("Seek", true)),
	  m_followRaw(create<XBoolNode>("FollowRawRecords", true)),
	  m_periodicTerm(0) {

    iterate_commit([=](Transaction &tr){
        tr[ *m_speed].add(SPEED_FASTEST);
        tr[ *m_speed].add(SPEED_FAST);
        tr[ *m_speed].add(SPEED_NORMAL);
        tr[ *m_speed].add(SPEED_SLOW);
        tr[ *m_speed] = SPEED_FAST;
        tr[ *m_followRaw] = true;

        m_lsnOnOpen = tr[ *filename()].onValueChanged().connectWeakly(
            shared_from_this(), &XJournalReader::onOpen);
        m_lsnOnSeek = tr[ *m_seek].onValueChanged().connectWeakly(
            shared_from_this(), &XJournalReader::onSeek);
		m_lsnFirst = tr[ *m_first].onTouch().connectWeakly(
			shared_from_this(), &XJournalReader::onFirst,
			Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
		m_lsnBack = tr[ *m_back].onTouch().connectWeakly(
			shared_from_this(), &XJournalReader::onBack,
			Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
		m_lsnNext = tr[ *m_next].onTouch().connectWeakly(
			shared_from_this(), &XJournalReader::onNext,
			Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
		m_lsnStop = tr[ *m_stop].onTouch().connectWeakly(
			shared_from_this(), &XJournalReader::onStop,
			Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
	    m_lsnPlayCond = tr[ *m_fastForward].onValueChanged().connectWeakly(
			shared_from_this(),
			&XJournalReader::onPlayCondChanged,
			Listener::FLAG_MAIN_THREAD_CALL | Listener::FLAG_AVOID_DUP | Listener::FLAG_DELAY_ADAPTIVE);
	    tr[ *m_rewind].onValueChanged().connect(m_lsnPlayCond);
	    tr[ *m_speed].onValueChanged().connect(m_lsnPlayCond);
    });
    
    for(int i = 0; i < RECORD_READER_NUM_THREADS; ++i) {
        m_threads.emplace_back(new XThread(shared_from_this(), &XJournalReader::execute));
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
XJournalReader::onOpen(const Snapshot &shot, XValueNodeBase *) {
	XString given = ( **filename())->to_str();
	XString rawpath = given, journalpath;
	bool wantsSettings = QString::fromStdString(given).endsWith(".kamj", Qt::CaseInsensitive);
	if(wantsSettings) {
		journalpath = given;
		rawpath.clear();
	}
	else {
		journalpath = XJournalFile::journalBeside(given);
	}

	XJournalFile journal;
	std::vector<DumpValue> dump;
	if(journalpath.length()) {
		XString err;
		//The dump is the state the run began in.  Whether it is put back is
		//decided by which file was opened, not by a switch pressed afterwards.
		dump.reserve(1024);
		if( !journal.open(journalpath, [&dump](const XJournalFile::Event &e) {
				if((e.kind == XJournalFile::Event::Kind::VALUE) && e.fromDump)
					dump.push_back({e.id, e.value, e.exact, e.hasExact});
			}, err))
			gWarnPrint(i18n("Journal: ") + journalpath + " " + err);
	}
	if(journal.isOpen()) {
		if(rawpath.empty())
			rawpath = journal.rawPath();
		if( !journal.timesKnown())
			gWarnPrint(i18n("This journal predates the machine-readable timestamp; "
				"its entries cannot be placed against the records: ") + journalpath);
		if(rawpath.empty())
			gMessagePrint(i18n("No raw stream in this journal, so there is nothing to "
				"re-analyse: playback steps through its settings instead."));
		else
			gMessagePrint(i18n("Journal: ") + journalpath
				+ formatString(" (%s, %u settings of %u nodes)", journal.kind().c_str(),
					(unsigned)dump.size(), (unsigned)journal.nodes().size()));
	}

	unsigned int settings = (unsigned int)dump.size();
	uint64_t journalsize = journal.isOpen() ?
		(uint64_t)QFileInfo(QString::fromStdString(journalpath)).size() : 0;
	//The head names only what existed when recording began.  Everything else
	//-- a driver added during the run, and nearly all of a session journal --
	//is named in the body, so the body is read once for those names alone.
	std::map<uint32_t, XJournalFile::NodeInfo> allnodes;
	if(journal.isOpen())
		XJournalFile::scanNodes(journalpath, allnodes);
	m_filemutex.lock();
	m_journal = std::move(journal);
	m_dump = std::move(dump);
	m_journalRewind = false;   //!< a freshly opened cursor is already at the first entry
	m_journalSize = journalsize;
	m_journalVisited.clear();
	m_allNodes = std::move(allnodes);
	//m_journal, NOT journal: it was moved from two lines ago, so asking the
	//husk whether it is open answers no, always -- and every value a replay
	//would have applied was silently dropped.
	m_restoreWanted = wantsSettings && m_journal.isOpen();
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

	//Opening the journal IS the request for its settings; asking again with a
	//switch would be ceremony.  Opening the raw stream instead is the other
	//answer -- re-analyse with today's settings -- and is how a recording is
	//run through again with one parameter changed.  What a running driver
	//owns is left alone, and restoreDump() says how much that was.
	if(settings && wantsSettings)
		restoreDump();
	else if(settings)
		gMessagePrint(formatString_tr(I18N_NOOP(
			"Its %u settings are not applied: the raw stream was opened, so the "
			"records are re-analysed with today's settings. Open the .kamj instead "
			"to use the recorded ones."), settings));
}
void
XJournalReader::readHeader(void *_fd) {
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
XJournalReader::parseOne(void *_fd, XMutex &mutex) {
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
        tr[ *m_recordTime] = timestr;
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
	//The settings this record was taken with, gathered while the file is still
	//ours and applied once it is not: applying is a transaction per node, and
	//the mutex here guards the streams, not the tree.
	std::vector<RestoreItem> pending;
	bool rewound = false;
	if(m_journal.isOpen() && mayApply_()) {
		if(m_journalRewind.exchange(false)) {
			//A journal is a stream and only goes forwards, so reaching an
			//earlier record means starting again from its head -- and the head
			//is the dump, which has to go back too.
			rewound = m_journal.rewind([](const XJournalFile::Event &){});
		}
		m_journal.advanceTo(time, [&](const XJournalFile::Event &e) {
			takeIfRequest_(e, pending);
		});
	}

	mutex.unlock();

	if(rewound)
		restoreDump(true);
	if( !pending.empty())
		applyValues(pending, busyDrivers(), nullptr, nullptr);

	{ XScopedLock<XMutex> lock(m_drivermutex);
	driver->finishWritingRaw(rawdata, XTime::now(), time);
	}
}
void
XJournalReader::gzgetline(void* _fd, unsigned char*buf, unsigned int len, int del) {
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
XJournalReader::first_(void *fd) {
	gzrewind(static_cast<gzFile>(fd));
	m_journalRewind = true;
}
//! Where the next record is, in thousandths of the compressed file.
int
XJournalReader::permilleOf_(void *fd) const {
	if( !m_fileSize)
		return 0;
	auto off = gzoffset(static_cast<gzFile>(fd));
	if(off < 0)
		return 0;
	uint64_t p = (uint64_t)off * 1000u / m_fileSize;
	return (int)std::min(p, (uint64_t)1000u);
}
void
XJournalReader::reportPosition_(void *fd) {
	int permille = permilleOf_(fd);
	trans( *m_position) = (unsigned int)permille;
}
//! Back one record.  At the head of the file there is nothing before the
//! first record, and saying so is not the same as failing: seeking to the
//! start and then stepping back used to raise an IO error, because the seek
//! for the length word ran off the front of the file.
bool
XJournalReader::stepBack_(void *fd) {
	m_journalRewind = true;
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
XJournalReader::seek_(void *_fd, int permille) {
	gzFile fd = static_cast<gzFile>(_fd);
	if((permille <= 0) || !m_fileSize) {
		first_(fd);
		return;
	}
	m_journalRewind = true;
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
//! Walks a dumped path ("/Drivers/ODMR2D/Average") down the live tree.
//! Null when any step is missing, which is the normal answer for a journal
//! recorded on a rig that is not this one.
static shared_ptr<XNode> nodeAt(const shared_ptr<XNode> &root, const XString &path) {
	shared_ptr<XNode> node = root;
	for(auto &&part: QString::fromStdString(path).split('/', Qt::SkipEmptyParts)) {
		if( !node)
			return {};
		node = node->getChild(part.toStdString());
	}
	return node;
}

//! What decides whether restoring is a private act or a public one.
//!
//! Skipping runtime nodes does NOT keep a restore off the wire, which is worth
//! stating because it looks as though it should: a driver's settings are
//! precisely the non-runtime ones -- XDCSource's `Value` is
//! `create<XDoubleNode>("Value", false)` and its listener writes to the
//! instrument.  What makes a restore harmless is the interface being closed,
//! since those listeners are connected in start() and dropped in stop().
std::set<XString>
XJournalReader::busyDrivers() const {
	std::set<XString> names;
	auto meas = dynamic_pointer_cast<XMeasure>(m_root.lock());
	if( !meas)
		return names;
	Snapshot shot( *meas->interfaces());
	if(shot.size())
		for(auto &&x: *shot.list())
			if(auto intf = dynamic_pointer_cast<XInterface>(x))
				if(intf->isOpened())
					if(auto drv = intf->driver())
						names.insert(drv->getName());
	return names;
}

unsigned int
XJournalReader::openInterfaces() const {
	auto meas = dynamic_pointer_cast<XMeasure>(m_root.lock());
	if( !meas)
		return 0;
	unsigned int n = 0;
	Snapshot shot( *meas->interfaces());
	if(shot.size())
		for(auto &&x: *shot.list())
			if(auto intf = dynamic_pointer_cast<XInterface>(x))
				if(intf->isOpened())
					++n;
	return n;
}

//! Runtime nodes are skipped: what they hold is a reading, not a setting, and
//! the driver that owns one would contradict it on its next record anyway.
//! The settings are wanted when the journal is what was opened.  They are
//! safe to put back when no interface is open, since a driver's I/O listeners
//! exist only between its start() and stop(); with one open, File > Open
//! Measurement is the deliberate way to push a recorded setup at live
//! instruments, and this pane is not.
//! A journal names its nodes by path, and a path means nothing on a tree that
//! does not have it: opening a run on a fresh KAME used to restore precisely
//! nothing, because every driver it spoke of was absent.  The dump says what
//! each node was created as -- the registry key, when it has one -- so what is
//! missing can be made.
//!
//! Main thread only.  XDriverList says it is not safe to create in from
//! anywhere else (isThreadSafeDuringCreationByTypename), which is why the
//! .kam loaders hop to the main thread for the same call.
unsigned int
XJournalReader::restoreStructure_(bool quiet) {
	auto root = m_root.lock();
	if( !root)
		return 0;
	if( !isMainThread()) {
		if( !quiet)
			gWarnPrint(i18n("Drivers can only be created from the main thread; "
				"open the journal from the pane rather than from a script."));
		return 0;
	}
	//By depth, so a parent is made before its children.  NOT by id: an id is
	//handed out when a node is first seen, and three of them in a real run
	//came out ahead of their parent's.
	std::vector<const XJournalFile::NodeInfo *> order;
	for(auto &&kv: m_allNodes)
		order.push_back( &kv.second);
	std::stable_sort(order.begin(), order.end(),
		[](const XJournalFile::NodeInfo *a, const XJournalFile::NodeInfo *b) {
			return std::count(a->path.begin(), a->path.end(), '/')
				< std::count(b->path.begin(), b->path.end(), '/');
		});
	unsigned int made = 0, failed = 0;
	for(auto &&np: order) {
		const XJournalFile::NodeInfo &n = *np;
		//No registry key means it was not created by name in the first place:
		//a fixed child, which exists as soon as its parent does.
		if(n.type.empty() || n.runtime || n.path.empty() || n.name.empty())
			continue;
		auto slash = n.path.rfind('/');
		if(slash == XString::npos)
			continue;
		auto parent = nodeAt(root, n.path.substr(0, slash));
		auto list = dynamic_pointer_cast<XListNodeBase>(parent);
		if( !list || list->isAliasList())
			continue;   //!< referenced, not owned: navigated by name, never created
		if(list->getChild(n.name))
			continue;
		if(list->createByTypename(n.type, n.name))
			++made;
		else
			++failed;   //!< a type this build does not have: a module not loaded
	}
	if(failed && !quiet)
		gWarnPrint(formatString_tr(I18N_NOOP(
			"%u nodes could not be created: a driver type this build does not have, "
			"or a name already taken by something else."), failed));
	return made;
}

bool
XJournalReader::mayApply_() const {
	return m_restoreWanted;
}

//! No raw stream, or not being asked for: either way the journal itself is
//! what the transport walks.
bool
XJournalReader::journalOnly() const {
	if( !m_journal.isOpen())
		return false;
	return !m_pGFD || !Snapshot( *m_followRaw)[ *m_followRaw];
}

unsigned int
XJournalReader::restoreDump(bool quiet) {
	auto root = m_root.lock();
	if( !root)
		return 0;
	unsigned int made = restoreStructure_(quiet);
	if(made && !quiet)
		gMessagePrint(formatString_tr(I18N_NOOP("Created %u nodes this tree did not have."), made));
	//Copied under the lock and applied outside it: applying means a
	//transaction per node, and the playback thread must not wait on that.
	std::vector<RestoreItem> items;
	unsigned int skipped = 0;
	{
		m_filemutex.lock();
		items.reserve(m_dump.size());
		for(auto &&v: m_dump) {
			auto it = m_journal.nodes().find(v.id);
			if((it == m_journal.nodes().end()) || it->second.runtime) {
				++skipped;
				continue;
			}
			items.push_back({it->second.path, v.value, v.exact, v.hasExact});
		}
		m_filemutex.unlock();
	}
	if(items.empty()) {
		gWarnPrint(i18n("No settings in this journal to restore."));
		return 0;
	}
	unsigned int missing = 0, held = 0;
	unsigned int done = applyValues(items, busyDrivers(), &missing, &held);
	if(quiet)
		return done;
	gMessagePrint(formatString_tr(
		I18N_NOOP("Restored %u settings; %u not found, %u skipped."),
		done, missing, skipped + (unsigned int)items.size() - done - missing - held));
	if(held)
		gWarnPrint(formatString_tr(I18N_NOOP(
			"%u of them belong to drivers that are running and were left alone."), held));
	return done;
}

//! \sa restoreDump(), which is this over the head of a journal, and parseOne(),
//! which is this over the entries that reach the record being played.
//! A value is withheld only from a driver that is running.  Blocking every
//! write while any interface was open blocked far too much: an analysis
//! parameter of a secondary driver -- NMRPulseAnalyzer's NumEcho, say -- talks
//! to nothing at all, and refusing it made replay look broken on a live rig.
unsigned int
XJournalReader::applyValues(const std::vector<RestoreItem> &items,
	const std::set<XString> &busy, unsigned int *missing, unsigned int *held) {
	auto root = m_root.lock();
	if( !root)
		return 0;
	unsigned int done = 0;
	for(auto &&item: items) {
		if( !busy.empty()) {
			//"/Drivers/<name>/..." -- the only paths that can reach hardware.
			static const XString prefix = "/Drivers/";
			if(item.path.compare(0, prefix.size(), prefix) == 0) {
				auto end = item.path.find('/', prefix.size());
				if((end != XString::npos) &&
					busy.count(item.path.substr(prefix.size(), end - prefix.size()))) {
					if(held)
						++( *held);
					continue;
				}
			}
		}
		auto node = nodeAt(root, item.path);
		auto vnode = dynamic_pointer_cast<XValueNodeBase>(node);
		if( !vnode) {
			if(missing)
				++( *missing);   //!< absent here, or not a value: a tree that has moved on
			continue;
		}
		try {
			auto dnode = dynamic_pointer_cast<XDoubleNode>(vnode);
			if(item.hasExact && dnode) {
				//The eight bytes, not the four digits to_str() would have
				//printed: a restored number should be the number recorded.
				dnode->iterate_commit([&](Transaction &tr){
					tr[ *dnode] = item.exact;
				});
			}
			else {
				vnode->iterate_commit([&](Transaction &tr){
					tr[ *vnode].str(item.value);
				});
			}
			++done;
		}
		catch (XKameError &) {
			//A value this node will not take any more.
		}
	}
	return done;
}
//! Requests only.  A report is a driver talking about itself -- the 37 it was
//! passing through on its way to the 100 that was asked for, written to the
//! node that holds the request -- and putting one back would contradict the
//! driver that owns it.  Runtime nodes are not settings at all.
void
XJournalReader::takeIfRequest_(const XJournalFile::Event &e, std::vector<RestoreItem> &out) const {
	if((e.kind != XJournalFile::Event::Kind::VALUE) || !e.request)
		return;
	//An unstamped value IS taken here, unlike at the head: a node that appears
	//in the middle of a journal brings its initial state with it, and that
	//state is what it had at the moment it appeared.  A driver created while
	//the run was running would otherwise start from whatever this tree
	//happened to hold.
	//The full map, not the cursor's: the cursor knows a node only once it has
	//walked past the line that named it, and an entry can precede that in a
	//journal whose head did not have it.
	const XJournalFile::NodeInfo *info = nullptr;
	auto it = m_allNodes.find(e.id);
	if(it != m_allNodes.end())
		info = &it->second;
	else {
		auto jt = m_journal.nodes().find(e.id);
		if(jt != m_journal.nodes().end())
			info = &jt->second;
	}
	if( !info || info->runtime)
		return;
	out.push_back({info->path, e.value, e.exact, e.hasExact});
}

void
XJournalReader::reportJournalPosition_(const XTime &when) {
	XString timestr = when.getTimeStr();
	unsigned int permille = 0;
	if(m_journalSize) {
		int64_t off = m_journal.offset();
		if(off > 0)
			permille = (unsigned int)std::min((uint64_t)off * 1000u / m_journalSize,
				(uint64_t)1000u);
	}
	iterate_commit([&](Transaction &tr){
		tr[ *m_recordTime] = timestr;
		tr[ *m_position] = permille;
	});
}

//! Back to the state the journal opens with.
void
XJournalReader::journalFirst_() {
	m_filemutex.lock();
	m_journal.rewind([](const XJournalFile::Event &){});
	m_filemutex.unlock();
	m_journalVisited.clear();
	if(mayApply_())
		restoreDump(true);
	reportJournalPosition_(XTime());
}

bool
XJournalReader::journalStep_() {
	std::vector<RestoreItem> pending;
	XTime when;
	m_filemutex.lock();
	bool more = m_journal.peekTime( &when);
	if(more)
		m_journal.advanceTo(when, [&](const XJournalFile::Event &e) {
			takeIfRequest_(e, pending);
		});
	m_filemutex.unlock();
	if( !more)
		return false;
	//A step is an INSTANT, not a line: several settings changed together are
	//one act, and stopping between them would show a state that never was.
	//
	//With following off, or an interface open, the cursor still moves and the
	//readout still says where it is: walking the history to see what changed
	//when is worth having on its own, and it touches nothing.
	m_lastApplied = mayApply_() ?
		applyValues(pending, busyDrivers(), nullptr, nullptr) : 0;
	m_journalVisited.push_back(when);
	if(m_journalVisited.size() > 4096)
		m_journalVisited.pop_front();
	reportJournalPosition_(when);
	return true;
}

void
XJournalReader::journalBack_() {
	if(m_journalVisited.size() < 2) {
		journalFirst_();
		return;
	}
	m_journalVisited.pop_back();             //!< where we are now
	XTime target = m_journalVisited.back();
	std::deque<XTime> keep;
	keep.swap(m_journalVisited);
	journalFirst_();                          //!< clears the history it just used
	std::vector<RestoreItem> pending;
	m_filemutex.lock();
	m_journal.advanceTo(target, [&](const XJournalFile::Event &e) {
		takeIfRequest_(e, pending);
	});
	m_filemutex.unlock();
	if(mayApply_())
		applyValues(pending, busyDrivers(), nullptr, nullptr);
	m_journalVisited.swap(keep);
	reportJournalPosition_(target);
}

//! To a point in the journal by size of the file, as the raw stream's seek is.
//! Cheaper here: lines are self-delimiting, so there is no boundary to find --
//! but the journal is a state, so everything before the target has to be
//! applied on the way rather than skipped.
void
XJournalReader::journalSeek_(int permille) {
	journalFirst_();
	if(permille <= 0)
		return;
	uint64_t target = m_journalSize * (uint64_t)permille / 1000u;
	//Once, not once per step: a seek can walk thousands of instants.
	std::set<XString> busy = busyDrivers();
	XTime when, last;
	for(;;) {
		std::vector<RestoreItem> pending;
		m_filemutex.lock();
		bool more = m_journal.peekTime( &when);
		int64_t off = m_journal.offset();
		if(more && ((off < 0) || ((uint64_t)off < target)))
			m_journal.advanceTo(when, [&](const XJournalFile::Event &e) {
				takeIfRequest_(e, pending);
			});
		else
			more = false;
		m_filemutex.unlock();
		if( !more)
			break;
		if(mayApply_())
			applyValues(pending, busy, nullptr, nullptr);
		last = when;
		m_journalVisited.push_back(when);
		if(m_journalVisited.size() > 4096)
			m_journalVisited.pop_front();
	}
	reportJournalPosition_(last);
}

void
XJournalReader::stopPlayback_(const XString &msg) {
	m_periodicTerm = 0.0;
	iterate_commit([=](Transaction &tr){
		tr[ *m_fastForward] = false;
		tr[ *m_rewind] = false;
		tr.unmark(m_lsnPlayCond);
	});
	g_statusPrinter->printMessage(msg);
}

void
XJournalReader::onSeek(const Snapshot &shot, XValueNodeBase *) {
	//Handed to the playback thread rather than done here: this call arrives
	//on whichever thread moved the slider, and a seek reads everything before
	//its target.
	m_seekRequest = (int)(unsigned int)shot[ *m_seek];
	XScopedLock<XCondition> lock(m_condition);
	m_condition.broadcast();
}
void
XJournalReader::previous_(void *fd) {
	if(gzseek(static_cast<gzFile>(fd), -sizeof(uint32_t), SEEK_CUR) == -1) throw XIOError(__FILE__, __LINE__);
	goToHeader(fd);
}
void
XJournalReader::next_(void *fd) {
	readHeader(fd);
	if(gzseek(static_cast<gzFile>(fd), m_allsize - m_headerBytes, SEEK_CUR) == -1)
		throw XIOError(__FILE__, __LINE__);
}
void
XJournalReader::goToHeader(void *_fd) {
	gzFile fd = static_cast<gzFile>(_fd);

	if(gzeof(fd)) throw XIOError(__FILE__, __LINE__);
	std::vector<char> buf(sizeof(uint32_t));
	XPrimaryDriver::RawDataReader reader(buf);
	if(gzread(fd, &buf[0], sizeof(uint32_t)) == Z_NULL) throw XIOError(__FILE__, __LINE__);
	int allsize = reader.pop<uint32_t>();
	if(gzseek(fd, -allsize, SEEK_CUR) == -1) throw XIOError(__FILE__, __LINE__);
}
void
XJournalReader::terminate() {
    m_periodicTerm = 0;
    for(auto &&x: m_threads) {
        x->terminate();
    }
    XScopedLock<XCondition> lock(m_condition);
    m_condition.broadcast();
}
void
XJournalReader::join() {
    for(auto &&x: m_threads) {
        x->join();
    }
}

void
XJournalReader::onPlayCondChanged(const Snapshot &shot, XValueNodeBase *) {
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
XJournalReader::onStop(const Snapshot &shot, XTouchableNode *) {
    m_periodicTerm = 0;
    g_statusPrinter->printMessage(i18n("Stopped"));
	iterate_commit([=](Transaction &tr){
		tr[ *m_fastForward] = false;
		tr[ *m_rewind] = false;
		tr.unmark(m_lsnPlayCond);
    });
}
void
XJournalReader::onFirst(const Snapshot &shot, XTouchableNode *) {
	if(journalOnly()) {
		journalFirst_();
		g_statusPrinter->printMessage(i18n("First"));
		return;
	}
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
XJournalReader::onNext(const Snapshot &shot, XTouchableNode *) {
	if(journalOnly()) {
		if( !journalStep_())
			g_statusPrinter->printMessage(i18n("End of journal"));
		else
			g_statusPrinter->printMessage(formatString_tr(
				I18N_NOOP("Next: %u settings"), m_lastApplied));
		return;
	}
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
XJournalReader::onBack(const Snapshot &shot, XTouchableNode *) {
	if(journalOnly()) {
		journalBack_();
		g_statusPrinter->printMessage(i18n("Previous"));
		return;
	}
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

void *XJournalReader::execute(const atomic<bool> &terminated) {
    Transactional::setCurrentPriorityMode(Transactional::Priority::NORMAL);
    while( !terminated) {
		double ms = 0.0;
		{
			XScopedLock<XCondition> lock(m_condition);
			while((fabs((ms = m_periodicTerm)) < 1e-4) && (m_seekRequest < 0) && !terminated)
				m_condition.wait();
		}
    
		if(terminated) break;

		if( !m_pGFD && !journalOnly()) {
			m_seekRequest = -1;     //!< nothing to seek in
			msecsleep(100);
			continue;
		}

		//Claimed here, so a slider dragged while playing moves the same
		//cursor the playback is walking, rather than racing it.
		int req = m_seekRequest.exchange(-1);

		//No raw stream: the journal itself is the recording, and a step is one
		//instant of it.  Nothing is re-analysed -- there are no records -- so
		//what plays back is the settings history.
		if(journalOnly()) {
			if(req >= 0)
				journalSeek_(req);
			else if(ms > 0.0) {
				if( !journalStep_()) {
					stopPlayback_(i18n("End of journal"));
					continue;
				}
			}
			else if(ms < 0.0) {
				if(m_journalVisited.size() < 2) {
					//At the head: stopping is the honest end of a rewind.
					m_periodicTerm = 0.0;
					iterate_commit([=](Transaction &tr){
						tr[ *m_rewind] = false;
						tr.unmark(m_lsnPlayCond);
					});
					g_statusPrinter->printMessage(i18n("First"));
					continue;
				}
				journalBack_();
			}
			msecsleep(lrint(fabs(ms)));
			continue;
		}

		try {
			m_filemutex.lock();
			if(req >= 0) {
				seek_(m_pGFD, req);
			}
			else if(ms > 0.0 && gzeof(static_cast<gzFile>(m_pGFD))) {
				//It used to start over here.  With a journal attached that put
				//the whole tree back to the head of the recording without
				//saying so; and a replay that has reached the end has reached
				//the end.  FIRST is one button away.
				m_filemutex.unlock();
				stopPlayback_(i18n("End of records"));
				continue;
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
