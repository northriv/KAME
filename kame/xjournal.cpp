/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "xjournal.h"
#include "xlistnode.h"
#include "xitemnode.h"
#include "analyzer/recorder.h"
#include "support.h"
#include <zlib.h>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <QByteArray>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>
#include <QUuid>


const char *
XJournalRecorder::modeLabel(Mode m) {
    switch(m) {
    case Mode::SETUP: return "Setup";
    case Mode::LOGBOOK: return "Logbook";
    default: return "Logbook + raw";
    }
}

XJournalRecorder::XJournalRecorder(const char *name, bool runtime,
    const shared_ptr<XDriverList> &drivers) :
    XNode(name, runtime),
    //Transient like the raw stream's own path: a run's file is chosen for
    //that run.  The mode is a preference and is saved.
    m_filename(create<XStringNode>("Filename", true)),
    //auto_set_any: a journal and a .kam store a combo by its LABEL, and the
    //value can arrive before the items do (loading sets it, the constructor
    //fills the list).  Without it the label is dropped on the floor.
    m_mode(create<XComboNode>("Mode", false, true)),
    m_recording(create<XBoolNode>("Recording", true)),
    //Saved, unlike the run's switch: whether this rig journals its sessions
    //at all is a property of the rig, not of today.
    m_sessionJournal(create<XBoolNode>("SessionJournal", false)),
    m_statistics(create<XStringNode>("Statistics", true)),
    //Runtime: a run's stream is chosen for that run, and nothing about it
    //belongs in a settings file.
    m_rawstream(create<XRawStreamRecorder>("RawStream", true, drivers)) {
    //Own transaction, not the caller's: these children are inserted outside
    //any transaction the constructor was handed.
    iterate_commit([=](Transaction &tr){
        tr[ *m_mode].add({modeLabel(Mode::SETUP), modeLabel(Mode::LOGBOOK),
            modeLabel(Mode::LOGBOOK_RAW)});
        //Setup, not the top tier: a fresh KAME must not sit there implying
        //that pressing Write means 10 GB/hr.  The mode is saved (it is
        //non-runtime), so a rig that records raw says so once and remembers.
        tr[ *m_mode] = (int)Mode::SETUP;
        tr[ *m_recording] = false;
        tr[ *m_sessionJournal] = true;
        m_lsnOnRecordingChanged = tr[ *m_recording].onValueChanged().connectWeakly(
            shared_from_this(), &XJournalRecorder::onRecordingChanged);
        m_lsnOnFilenameChanged = tr[ *m_filename].onValueChanged().connectWeakly(
            shared_from_this(), &XJournalRecorder::onFilenameChanged);
        //Nothing is named yet, so there is no run to configure.
        tr[ *m_mode].setUIEnabled(false);
        tr[ *m_recording].setUIEnabled(false);
    });
}

void
XJournalRecorder::setSessionPath(const XString &path) {
    XString was = m_sessionPath;
    m_sessionPath = path;
    XString shown = Snapshot( *this)[ *m_filename].to_str();
    //Fill the field only when it is not naming a run: switching the session
    //journal back on must not wipe a name the user typed.
    bool showsSession = !shown.length()
        || (was.length() && (journalPathOf(shown) == journalPathOf(was)));
    if(showsSession)
        trans( *m_filename) = path;
    else
        updateRunControls();
}

//! A run needs a file of its own; while the field still names the session
//! journal there is nothing to start and nothing to choose, so both controls
//! are disabled rather than hidden -- a collapsing form looks broken.
void
XJournalRecorder::onFilenameChanged(const Snapshot &shot, XValueNodeBase *) {
    updateRunControls();
}

void
XJournalRecorder::updateRunControls() {
    Snapshot shot_this( *this);
    XString path = shot_this[ *m_filename].to_str();
    //A run's tier is fixed for its duration: its header says what it is.
    bool running = shot_this[ *m_recording];
    //Compared through journalPathOf on both sides: the field holds whatever
    //the user typed, the session path its full name, and "the same file" has
    //to mean the same file.
    bool own = path.length()
        && ( !m_sessionPath.length() || (journalPathOf(path) != journalPathOf(m_sessionPath)));
    iterate_commit([=](Transaction &tr){
        tr[ *m_mode].setUIEnabled(own && !running);
        tr[ *m_recording].setUIEnabled(own);
        //Only when it DIFFERS.  An XNode assignment marks its talker whether
        //or not the value changed, and onRecordingChanged is what calls this
        //-- so an unconditional write recurses through commit and talk until
        //the stack is gone.  It did, at 22:55 on 2026-08-29, on every launch:
        //the field starts empty, so !own is the startup case.
        if( !own && (bool)tr[ *m_recording])
            tr[ *m_recording] = false;
    });
}

XJournalRecorder::Mode
XJournalRecorder::modeOf(const Snapshot &shot) const {
    int idx = shot[ *m_mode];
    if((idx < 0) || (idx > (int)Mode::LOGBOOK_RAW))
        return Mode::LOGBOOK_RAW;
    return (Mode)idx;
}

static XString withExtension(const XString &given, const char *ext) {
    QString s = QString::fromStdString(given).trimmed();
    if(s.isEmpty())
        return {};
    //Strip whichever of the family the user happened to type, and nothing
    //else: a name with a dot in it is a name, not an extension.  Repeated,
    //so a stray pairing ("run042.kamj.gz", typed by someone who knows the
    //file is gzip) comes apart without enumerating the pairs.
    for(bool again = true; again; ) {
        again = false;
        for(auto &&known: {".kamj", ".kamb", ".kam", ".bin", ".gz"})
            if(s.endsWith(known, Qt::CaseInsensitive)) {
                s.chop(QString(known).length());
                again = true;
                break;
            }
    }
    return (s + ext).toStdString();
}
uintptr_t
XJournalRecorder::rawBytesWritten() const {
    return m_rawstream->bytesWritten();
}

XString XJournalRecorder::journalPathOf(const XString &given) {return withExtension(given, ".kamj");}
XString XJournalRecorder::rawPathOf(const XString &given) {return withExtension(given, ".kamb");}

//! The raw stream follows: its path is set only as recording starts, since
//! XRawStreamRecorder opens (and truncates) its file the moment its filename
//! changes -- a Setup run must not leave an empty .kamb behind.
void
XJournalRecorder::onRecordingChanged(const Snapshot &shot, XValueNodeBase *) {
    auto &raws = m_rawstream;
    Snapshot shot_this( *this);
    bool rec = shot_this[ *m_recording];
    bool wantsRaw = rec && (modeOf(shot_this) == Mode::LOGBOOK_RAW);
    if(wantsRaw) {
        XString path = rawPathOf(shot_this[ *m_filename].to_str());
        if(path.length())
            trans( *raws->filename()) = path;
    }
    trans( *raws->recording()) = wantsRaw;
    updateRunControls(); //!< the tier freezes while a run is open
}

//! The capture path: no lookup, no allocation, no lock.  The id is baked into
//! the Sink the talker holds, and the serial carries both the ordering and the
//! committing thread, so this is a ring push and nothing else.
void
XJournal::capture(uint32_t id, uint32_t kind, const Snapshot &shot,
    const XNode &node) noexcept {
    //The emitting node is in the snapshot by construction -- this is the
    //transaction that just committed the write.  Its payload's serial is the
    //serial of THAT write, which is more precise than the snapshot's own.
    int64_t serial = shot[node].serial();
    Record r;
    r.id = id;
    r.kind = kind;
    r.when = XTime::now();
    r.value = nullptr;
    r.exact = 0.0;
    r.flags = 0;
    m_ring.capture(serial, r);
}

void
XJournal::Sink::onValueChanged(const Snapshot &shot, XValueNodeBase *node) {
    journal->captureValue( *this, shot, *node);
}

//! The value travels with the record, as a pool-allocated string the ring
//! owns: freed by the drain that writes it, or by this producer when the ring
//! is full.  No truncation -- the allocator is lock-free, so allocating here
//! is acceptable, and a provenance record that silently shortens a string is
//! not one.
//!
//! Nothing is formatted at all unless a Logbook is actually being written:
//! to_str() on a double goes through formatDouble(), which is not free at
//! acquisition rate, and a run that keeps no values must not pay for them.
void
XJournal::captureValue(Sink &sink, const Snapshot &shot,
    XValueNodeBase &node) noexcept {
    XTime now = XTime::now();
    int64_t nowUs = (int64_t)now.sec() * 1000000 + now.usec();
    int64_t serial = shot[node].serial();
    //Attribution decides the cap, not the runtime flag: what needs thinning
    //is what a DRIVER writes, whatever node it writes to.  A request is the
    //user or a script operating the instrument and is never dropped.
    bool isReport = (threadClassOf((unsigned int)((uint64_t)serial & 0xffffu))
        == ThreadClass::UNKNOWN);
    uint32_t where = FLAG_SESSION | FLAG_RUN;
    if(isReport) {
        //A run keeps every report; only the always-on session journal thins
        //them, and by silence rather than by rate.  Measured from the last
        //WRITE for that reason: a node written at 4 Hz has never been
        //silent, and what is kept is the first report after a quiet stretch
        //-- the state a device announces at open or when something changed.
        int64_t lastWrite = sink.lastReportUs.load(std::memory_order_relaxed);
        sink.lastReportUs.store(nowUs, std::memory_order_relaxed);
        if(nowUs - lastWrite <= (int64_t)SESSION_QUIET_US) {
            where &= ~(uint32_t)FLAG_SESSION;
            ++m_sessionSkipped;
        }
    }
    if( !m_runKeepsValues)
        where &= ~(uint32_t)FLAG_RUN;
    if( !where)
        return; //!< nobody would keep it, so it is not even formatted
    Record r;
    r.id = sink.id;
    r.kind = KIND_VALUE;
    r.when = now;
    r.exact = 0.0;
    r.flags = where;
    r.value = nullptr;
    try {
        XString str = shot[node].to_str();
        char *blob = new char[str.size() + 1];
        memcpy(blob, str.c_str(), str.size() + 1);
        r.value = blob;
        if(sink.isDouble) {
            r.exact = (double)static_cast<const XDoubleNode::Payload &>(shot[node]);
            r.flags |= FLAG_EXACT;
        }
    }
    catch(...) {
        //A node whose to_str() throws is still worth an entry saying it
        //changed; it is not worth taking the committing thread down.
    }
    if( !m_ring.capture(serial, r))
        delete[] r.value; //!< refused: nobody else will free it
}
void
XJournal::Sink::onTouch(const Snapshot &shot, XTouchableNode *node) {
    journal->capture(id, KIND_TOUCH, shot, *node);
}
//! Whether a node is in the tree is decided by one thing only: whether a
//! list holds it (user, 2026-08-29).  Every other child is created by its
//! parent's constructor and lives exactly as long as the parent, so it never
//! has to be asked.  That makes structure an EVENT rather than something to
//! be rediscovered by walking: onCatch and onRelease name the node, the list
//! and the index, at the moment it happens and with the committing serial.
void
XJournal::Sink::onCatch(const Snapshot &shot,
    const XListNodeBase::Payload::CatchEvent &e) {
    journal->capture(id, KIND_CATCH, shot, *e.emitter);
    journal->pushPending(e.caught, id, true, e.index);
}
void
XJournal::Sink::onRelease(const Snapshot &shot,
    const XListNodeBase::Payload::ReleaseEvent &e) {
    journal->capture(id, KIND_RELEASE, shot, *e.emitter);
    journal->pushPending(e.released, id, false, e.index);
}
void
XJournal::Sink::onMove(const Snapshot &shot,
    const XListNodeBase::Payload::MoveEvent &e) {
    //Order is the meaning in the lists that have one, so a reordering is a
    //change like any other.
    journal->capture(id, KIND_MOVE, shot, *e.emitter);
}

//! Hands a structural event to the journal's own thread.  Subscribing (or
//! marking off) touches the node table and takes transactions, neither of
//! which may happen here: this runs inside the committing thread's commit.
void
XJournal::pushPending(const shared_ptr<XNode> &node, uint32_t listId, bool caught,
    int index) {
    if( !node)
        return;
    XScopedLock<XCondition> lock(m_wake);
    m_pending.push_back(Pending{node, listId, caught, index});
    m_wake.signal();
}

XJournal::XJournal() {}

XJournal::~XJournal() {
    stop();
}

bool
XJournal::engineWanted() {
    const char *v = getenv("KAME_JOURNAL");
    return !(v && (XString(v) == "0"));
}

bool
XJournal::enabledByEnvironment() {
    const char *v = getenv("KAME_JOURNAL");
    return v && *v && (XString(v) != "0");
}

shared_ptr<XJournal>
XJournal::start(const shared_ptr<XNode> &root,
    const shared_ptr<XJournalRecorder> &recorder) {
    auto journal = std::make_shared<XJournal>();
    journal->m_root = root;
    journal->m_recorder = recorder;
    journal->m_started = XTime::now();
    if(const char *v = getenv("KAME_JOURNAL_REPORT_SEC")) {
        int sec = atoi(v);
        if(sec > 0)
            journal->m_reportInterval = sec;
    }
    //The survey report is a developer artifact -- what a listener on every
    //node costs, and who writes what -- so it stays behind KAME_JOURNAL.
    //Capture itself is unconditional now that a run can be written.
    if(enabledByEnvironment()) {
        XString dir = managedDirectory("journal");
        if(dir.empty())
            gWarnPrint(i18n_noncontext("Journal: no writable directory for the survey."));
        else {
            journal->m_reportPath = dir + "/capture-"
                + QDateTime::currentDateTime().toString("yyyyMMdd-hhmmss").toStdString()
                + ".txt";
            gMessagePrint(XString(i18n_noncontext("Journal survey: "))
                + journal->m_reportPath);
        }
    }
    journal->m_thread.reset(new XThread(journal, &XJournal::execute));
    return journal;
}

void
XJournal::stop() {
    if( !m_thread)
        return;
    m_thread->terminate();
    {
        //It may be asleep on the condition; do not make quitting wait out a
        //drain interval.
        XScopedLock<XCondition> lock(m_wake);
        m_wake.broadcast();
    }
    m_thread->join();
    m_thread.reset();
    {
        //Nothing of the tree outlives the thread: a Pending entry holds a
        //shared_ptr to a node, and holding one across a teardown would keep
        //a node alive past the destruction its parent is performing.
        XScopedLock<XCondition> lock(m_wake);
        m_pending.clear();
    }
    //Release the listeners while the Sinks they point at are still alive.
    //A talker keeps only a weak reference, so dropping ours is what
    //unsubscribes.
    for(auto &&rec: m_nodes) {
        rec.lsnValue.reset();
        rec.lsnTouch.reset();
        rec.lsnCatch.reset();
        rec.lsnRelease.reset();
        rec.lsnMove.reset();
    }
}

//! Written once per thread at its start, read only when a report is
//! written, so a plain mutex is the right instrument -- and it keeps this
//! usable before any journal exists.
static XMutex s_threadClassMutex;
static std::map<unsigned int, XJournal::ThreadClass> s_threadClasses;

void
XJournal::declareThisThread(ThreadClass cls) {
    XScopedLock<XMutex> lock(s_threadClassMutex);
    s_threadClasses[(unsigned int)Transactional::ProcessCounter::id()] = cls;
}

XJournal::ThreadClass
XJournal::threadClassOf(unsigned int id) {
    XScopedLock<XMutex> lock(s_threadClassMutex);
    auto it = s_threadClasses.find(id);
    return (it == s_threadClasses.end()) ? ThreadClass::UNKNOWN : it->second;
}

const char *
XJournal::threadClassName(ThreadClass cls) {
    switch(cls) {
    case ThreadClass::UI: return "UI";
    case ThreadClass::SCRIPT: return "script";
    default: return "driver?";
    }
}

XString
XJournal::managedDirectory(const char *sub) {
    QString base = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    if(base.isEmpty())
        return {};
    QString path = base + "/" + sub;
    if( !QDir().mkpath(path))
        return {};
    return path.toStdString();
}

//! How long the thread sleeps with nothing to do.  Not a latency for
//! structural events (a condition signal wakes it for those) and not a
//! resolution for timestamps (they are stamped at the write): only how long
//! records may sit in the ring, which has room for 8192 of them.
static const int DRAIN_INTERVAL_US = 20000;


bool
XJournal::Out::open(const XString &path) {
    close();
    m_gz = gzopen(path.c_str(), "wb");
    m_bytes = 0;
    return !!m_gz;
}
void
XJournal::Out::line(const XString &s) {
    if( !m_gz)
        return;
    gzwrite((gzFile)m_gz, s.c_str(), (unsigned)s.size());
    m_bytes += s.size();
    m_dirty = true;
}
void
XJournal::Out::flush(bool force) {
    if( !m_gz || !m_dirty)
        return;
    XTime now = XTime::now();
    if( !force && m_flushedAt.isSet() && (now.diff_msec(m_flushedAt) < 1000))
        return;
    //Z_FULL_FLUSH, not Z_SYNC_FLUSH: it ends the deflate block AND resets the
    //dictionary, so everything up to here stays readable on its own -- the
    //property a half-copied or half-killed file needs.  It costs compression,
    //which is why it is once a second and not once a drain.
    gzflush((gzFile)m_gz, Z_FULL_FLUSH);
    m_flushedAt = now;
    m_dirty = false;
}
void
XJournal::Out::close() {
    if(m_gz)
        gzclose((gzFile)m_gz);
    m_gz = nullptr;
}

//! Text, so `zdiff run1.kamj run2.kamj` and `zgrep` work with no tool at all --
//! which is most of the value of a provenance file ten years from now.
static XString jsonEscape(const XString &str) {
    XString out;
    out.reserve(str.size() + 8);
    for(unsigned char c: str) {
        switch(c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if(c < 0x20)
                out += formatString("\\u%04x", (int)c);
            else
                out += (char)c;
        }
    }
    return out;
}

//! The exact number, as the eight bytes it is: base64 of binary64,
//! little-endian.  to_str() goes through formatDouble() and is %.12g at best
//! -- fine for a settings file, not for provenance, where a rounded value is
//! both a spurious diff and a reproduction that used a different number.
static XString exactOf(double v) {
    unsigned char b[8];
    uint64_t bits;
    static_assert(sizeof(bits) == sizeof(v), "binary64 expected");
    memcpy( &bits, &v, sizeof(bits));
    for(int i = 0; i < 8; ++i)
        b[i] = (unsigned char)((bits >> (8 * i)) & 0xffu); //little-endian, stated
    return QByteArray((const char *)b, 8).toBase64().toStdString();
}

void
XJournal::writeHeader(Out &out, const char *kind) {
    auto rec = m_recorder.lock();
    Snapshot shot( *rec);
    XString s = XString("{\"format\":\"kame-journal\",\"version\":1,\"kind\":\"")
        + kind + "\",\"session\":\"" + m_session
        + "\",\"started\":\"" + jsonEscape(XTime::now().getTimeStr())
        + "\",\"kame\":\"" VERSION "\"";
    if(XString("run") == kind) {
        s += XString(",\"mode\":\"") + XJournalRecorder::modeLabel(rec->modeOf(shot)) + "\"";
        if(rec->modeOf(shot) == XJournalRecorder::Mode::LOGBOOK_RAW) {
            XString raw = XJournalRecorder::rawPathOf(shot[ *rec->filename()].to_str());
            s += ",\"raw\":\"" + jsonEscape(QFileInfo(QString::fromStdString(raw))
                .fileName().toStdString()) + "\"";
        }
        s += ",\"sessionfile\":\"" + jsonEscape(QFileInfo(
            QString::fromStdString(m_sessionPath)).fileName().toStdString()) + "\"";
    }
    out.line(s + "}\n");
}

//! The dump: what the tree was, at one instant.
//!
//! ONE root snapshot, deliberately.  Reading node by node has no consistency
//! cut at all -- each value would carry its own serial and the collection as
//! a whole never existed.  A root Snapshot is an ordinary operation here (the
//! node browser takes one whenever the pointed node changes); what is
//! forbidden is a root TRANSACTION.
void
XJournal::writeDump(Out &out) {
    auto root = m_root.lock();
    if( !root)
        return;
    Snapshot shot( *root);
    dumpSubtree(out, shot, root, "", 0, -1);
}

void
XJournal::dumpSubtree(Out &out, const Snapshot &shot,
    const shared_ptr<XNode> &node, const XString &path,
    uint32_t parentId, int index) {
    uint32_t id = subscribe(node, path);
    NodeRec &rec = m_nodes[id];
    rec.reachable = true;
    rec.inTree = true;

    XString s = formatString("{\"t\":\"n\",\"id\":%u", (unsigned)id);
    if(index >= 0)
        s += formatString(",\"p\":%u,\"i\":%d", (unsigned)parentId, index);
    s += ",\"name\":\"" + jsonEscape(node->getName())
        + "\",\"path\":\"" + jsonEscape(path) + "\"";
    //The registry key, and only when there IS one: a mangled type name is
    //never written as an instruction.  \sa doc/design/PROVENANCE.md
    XString key = node->storedTypename();
    if(key.length())
        s += ",\"type\":\"" + jsonEscape(key) + "\"";
    s += ",\"class\":\"" + jsonEscape(node->getTypename()) + "\"";
    if(shot[ *node].isRuntime())
        s += ",\"runtime\":true";
    auto lnode = dynamic_pointer_cast<XListNodeBase>(node);
    if(lnode)
        s += XString(",\"list\":\"") + (lnode->isAliasList() ? "alias" : "own") + "\"";
    out.line(s + "}\n");

    if(auto vnode = dynamic_pointer_cast<XValueNodeBase>(node)) {
        XString v = formatString("{\"t\":\"v\",\"id\":%u,\"s\":%lld", (unsigned)id,
            (long long)shot[ *vnode].serial())
            + ",\"v\":\"" + jsonEscape(shot[ *vnode].to_str()) + "\"";
        if(auto dnode = dynamic_pointer_cast<XDoubleNode>(node))
            v += ",\"x\":\"" + exactOf((double)shot[ *dnode]) + "\"";
        out.line(v + "}\n");
    }

    if( !shot.size(node))
        return;
    if(lnode && lnode->isAliasList())
        return; //!< referenced, not owned -- navigated by name, never created
    int i = 0;
    for(auto &&child: *shot.list(node)) {
        XString name = child->getName();
        dumpSubtree(out, shot, child,
            path + "/" + (name.length() ? name : formatString("[%d]", i)), id, i);
        ++i;
    }
}

void
XJournal::writeEntry(Out &out, const JournalT::Entry &e) {
    if(e.record.kind != KIND_VALUE)
        return; //structure is written where it is subscribed, with its names
    XString s = formatString("{\"t\":\"v\",\"id\":%u", (unsigned)e.record.id)
        + ",\"ts\":\"" + jsonEscape(e.record.when.getTimeStr()) + "\""
        + formatString(",\"s\":%lld", (long long)e.serial)
        + ",\"c\":\"" + ((threadClassOf((unsigned int)((uint64_t)e.serial & 0xffffu))
            == ThreadClass::UNKNOWN) ? "report" : "request") + "\"";
    if(e.record.value)
        s += ",\"v\":\"" + jsonEscape(e.record.value) + "\"";
    if(e.record.flags & FLAG_EXACT)
        s += ",\"x\":\"" + exactOf(e.record.exact) + "\"";
    out.line(s + "}\n");
}

//! Provenance must exist for a session in which nobody started a recording --
//! that is the whole point of it being always on, and the reason a `.kam`
//! stops being something you must remember to save.  Opened once, where KAME
//! keeps its own files, and headed by the same dump a run gets.
void
XJournal::openSession() {
    XString dir = managedDirectory("journal");
    if(dir.empty()) {
        gWarnPrint(i18n_noncontext("Journal: no writable directory; this session is not journaled."));
        return;
    }
    m_session = QUuid::createUuid().toString(QUuid::WithoutBraces).toStdString();
    m_sessionPath = dir + "/session-"
        + QDateTime::currentDateTime().toString("yyyyMMdd-hhmmss").toStdString() + ".kamj";
    if( !m_sessionOut.open(m_sessionPath)) {
        gWarnPrint(i18n_noncontext("Journal: cannot write ") + m_sessionPath);
        return;
    }
    writeHeader(m_sessionOut, "session");
    writeDump(m_sessionOut);
    m_sessionOut.flush(true);
    //Show what is being written, so "always on" is visible rather than
    //merely true.
    if(auto rec = m_recorder.lock())
        rec->setSessionPath(m_sessionPath);
}

void
XJournal::syncRun() {
    auto rec = m_recorder.lock();
    if( !rec)
        return;
    Snapshot shot( *rec);
    //The always-on journal is refusable.  A dump is not free -- an ODMR tree
    //is 3000 nodes, 600 KB before compression -- and a background writer
    //nobody can switch off is impolite whatever its size.
    bool wantSession = shot[ *rec->sessionJournal()];
    if(wantSession && !m_sessionOut.isOpen())
        openSession();
    else if( !wantSession && m_sessionOut.isOpen()) {
        m_sessionOut.line(XString("{\"t\":\"session\",\"state\":\"end\",\"ts\":\"")
            + jsonEscape(XTime::now().getTimeStr()) + "\"}\n");
        m_sessionOut.close();
        m_sessionPath.clear();
        //Do not leave the field naming a file nothing is writing -- but do
        //not wipe a run name either; setSessionPath knows the difference.
        rec->setSessionPath(XString());
    }
    bool on = shot[ *rec->recording()];
    auto mode = rec->modeOf(shot);
    if(on && !m_runOpen) {
        XString path = XJournalRecorder::journalPathOf(shot[ *rec->filename()].to_str());
        if( !path.length()
            || (m_sessionPath.length()
                && (path == XJournalRecorder::journalPathOf(m_sessionPath)))) {
            gWarnPrint(i18n_noncontext("Journal: name the run first."));
            trans( *rec->recording()) = false;
            return;
        }
        if( !m_runOut.open(path)) {
            gErrPrint(i18n_noncontext("Journal: cannot write ") + path);
            trans( *rec->recording()) = false;
            return;
        }
        m_openPath = path;
        m_bytesJournal = 0;
        m_bytesLast = 0;
        m_rawBytesAtStart = rec->rawBytesWritten();
        m_runOpen = true;
        m_runKeepsValues = (mode != XJournalRecorder::Mode::SETUP);
        writeHeader(m_runOut, "run");
        writeDump(m_runOut);
        m_runOut.flush(true);
        //The session journal says where its runs went, so either half leads
        //to the other.
        m_sessionOut.line(XString("{\"t\":\"run\",\"state\":\"start\",\"file\":\"")
            + jsonEscape(QFileInfo(QString::fromStdString(m_openPath)).fileName().toStdString())
            + "\",\"ts\":\"" + jsonEscape(XTime::now().getTimeStr()) + "\"}\n");
        m_sessionOut.flush();
        gMessagePrint(i18n_noncontext("Journal: ") + m_openPath);
    }
    //Deliberately NOT re-read while a run is open: the header states the
    //tier, and a tier that changed underneath it would make the file lie
    //about its own contents.  A run keeps the tier it was opened with; the
    //combo is greyed meanwhile, and this is what makes that true even for a
    //script that writes the node directly.
    if( !on && m_runOpen) {
        drainOnce(); //!< whatever is still in the ring belongs in the file
        m_runOut.close();
        m_runOpen = false;
        m_runKeepsValues = false;
        m_sessionOut.line(XString("{\"t\":\"run\",\"state\":\"end\",\"file\":\"")
            + jsonEscape(QFileInfo(QString::fromStdString(m_openPath)).fileName().toStdString())
            + "\",\"ts\":\"" + jsonEscape(XTime::now().getTimeStr()) + "\"}\n");
        m_sessionOut.flush();
    }
}

void
XJournal::updateStatistics() {
    auto rec = m_recorder.lock();
    if( !rec)
        return;
    XTime now = XTime::now();
    if( !m_statsAt.isSet()) {
        m_statsAt = now;
        return;
    }
    double dt = now.diff_msec(m_statsAt) / 1000.0;
    if(dt < 1.0)
        return;
    uintptr_t total = m_bytesJournal
        + (m_runOpen ? (rec->rawBytesWritten() - m_rawBytesAtStart) : 0);
    double rate = (total >= m_bytesLast) ? (total - m_bytesLast) / dt : 0.0;
    m_statsAt = now;
    m_bytesLast = total;
    XString s = m_runOpen
        ? formatString("%.1f MB/s  %.2f GB", rate / 1e6, total / 1e9)
        : XString();
    //Only when it actually changed: this node is journaled like any other,
    //and an idle run should not write a line a second saying so.
    if(Snapshot( *rec)[ *rec->statistics()].to_str() != s)
        trans( *rec->statistics()) = s;
}

void
XJournal::execute(const atomic<bool> &terminated) {
    XTime lastReport = XTime::now();
    //The first report goes out as soon as the tree is subscribed, so that
    //what a listener on every node costs is visible without waiting out an
    //interval -- that being the first thing this stage is here to measure.
    bool first = true;
    XTime lastSweep = XTime::now();
    sweep(); //!< the one full walk that has to happen: the opening state.
    while( !terminated) {
        syncRun();
        processPending();
        drainOnce();
        updateStatistics();
        //The sweep is a safety net and a measurement, not the mechanism --
        //hence periodic and slow, where the events are prompt.
        if(XTime::now().diff_msec(lastSweep) > (long)m_sweepInterval * 1000) {
            sweep();
            lastSweep = XTime::now();
        }
        if(first || XTime::now().diff_msec(lastReport) > (long)m_reportInterval * 1000) {
            writeReport();
            lastReport = XTime::now();
            first = false;
        }
        //Sleeps until a structural event arrives, or the drain interval is
        //up -- whichever first.  A caught node is subscribed within the
        //latency of a condition signal; the timeout is only there to keep
        //the ring from filling, which needs no urgency at 8192 entries.
        XScopedLock<XCondition> lock(m_wake);
        if(m_pending.empty())
            m_wake.wait(DRAIN_INTERVAL_US);
    }
    processPending();
    if(auto rec = m_recorder.lock())
        trans( *rec->recording()) = false; //!< closes the run file through syncRun()
    syncRun();
    drainOnce();
    m_sessionOut.line(XString("{\"t\":\"session\",\"state\":\"end\",\"ts\":\"")
        + jsonEscape(XTime::now().getTimeStr()) + "\"}\n");
    m_sessionOut.close();
    writeReport();
}

//! Alias lists reference nodes another parent owns, so walking them would
//! record a hard-linked node under a path nothing addresses it by -- the
//! interface listed as /Interfaces/Interface for every driver is reached at
//! /Drivers/<name>/Interface, where the name is unique.
void
XJournal::sweep() {
    XTime t0 = XTime::now();
    bool opening = (m_walks == 0);
    size_t known = m_nodes.size();
    for(auto &&rec: m_nodes)
        rec.reachable = false;
    if(auto root = m_root.lock())
        walk(root, "");
    ++m_walks;

    for(size_t i = 0; i < m_nodes.size(); ++i) {
        NodeRec &rec = m_nodes[i];
        //What the events should have told us, and did not.  Zero here says
        //the sweep is redundant and can go; anything else is a hole in the
        //rule that membership is a list's business alone.
        if( !opening) {
            if((i >= known) && !rec.catchAnnounced)
                ++m_sweepFoundNew;
            if( !rec.reachable && rec.inTree && !rec.detachAnnounced)
                ++m_sweepFoundDetached;
        }
        if( !rec.reachable)
            rec.inTree = false;
        else
            rec.inTree = true;
        if(rec.destroyed || !rec.node.expired())
            continue;
        rec.destroyed = true;
        //Its talker went with its payload; the listener is dead weight now.
        rec.lsnValue.reset();
        rec.lsnTouch.reset();
        rec.lsnCatch.reset();
        rec.lsnRelease.reset();
        rec.lsnMove.reset();
    }
    //A raw address is only a key while its node lives, and a later node
    //allocated at the same address would otherwise inherit this record --
    //exactly the case provenance has to keep apart.  The record itself stays,
    //with its statistics and the path it had.
    for(auto it = m_index.begin(); it != m_index.end();) {
        if(m_nodes[it->second].destroyed)
            it = m_index.erase(it);
        else
            ++it;
    }
    m_lastWalkMS = (unsigned int)XTime::now().diff_msec(t0);
    m_totalWalkMS += m_lastWalkMS;
}

//! Subscribes what onCatch announced, and marks off what onRelease did.
void
XJournal::processPending() {
    std::deque<Pending> pending;
    {
        XScopedLock<XCondition> lock(m_wake);
        pending.swap(m_pending);
    }
    for(auto &&p: pending) {
        if(p.caught) {
            ++m_catches;
            XString path = (p.listId < m_nodes.size() ? m_nodes[p.listId].path : XString())
                + "/" + p.node->getName();
            //A caught node arrives with the children its constructor made,
            //so the subtree is walked -- bounded by what was added, never
            //the whole tree.
            //Structure is written where it is learned, with the names and
            //the type keys: a driver added at 14:22 cannot be replayed from
            //a line that only says something changed.
            walk(p.node, path);
            if(m_sessionOut.isOpen() || m_runOut.isOpen()) {
                Snapshot shot( *p.node);
                if(m_sessionOut.isOpen()) {
                    dumpSubtree(m_sessionOut, shot, p.node, path, p.listId, p.index);
                    m_sessionOut.flush(true);
                }
                if(m_runOut.isOpen()) {
                    dumpSubtree(m_runOut, shot, p.node, path, p.listId, p.index);
                    m_runOut.flush(true);
                }
            }
            auto it = m_index.find(p.node.get());
            if(it != m_index.end()) {
                m_nodes[it->second].catchAnnounced = true;
                m_nodes[it->second].inTree = true;
            }
        }
        else {
            ++m_releases;
            auto it = m_index.find(p.node.get());
            if(it != m_index.end()) {
                XString line = formatString("{\"t\":\"released\",\"id\":%u,",
                    (unsigned)it->second)
                    + "\"ts\":\"" + jsonEscape(XTime::now().getTimeStr()) + "\"}\n";
                m_sessionOut.line(line);
                m_runOut.line(line);
                detachSubtree(it->second);
            }
        }
    }
}

void
XJournal::detachSubtree(uint32_t id) {
    NodeRec &rec = m_nodes[id];
    rec.inTree = false;
    rec.reachable = false;
    rec.detachAnnounced = true;
    auto node = rec.node.lock();
    if( !node)
        return;
    Snapshot shot( *node, false);
    if( !shot.size())
        return;
    for(auto &&child: *shot.list()) {
        auto it = m_index.find(child.get());
        if(it != m_index.end())
            detachSubtree(it->second);
    }
}

void
XJournal::walk(const shared_ptr<XNode> &node, const XString &path) {
    uint32_t id = subscribe(node, path);
    m_nodes[id].reachable = true;
    //Read AFTER subscribing, never before -- see the ordering rule in
    //subscribe().
    //
    //multi_nodal = false.  Not because a root snapshot would be forbidden --
    //XNodeBrowser takes one whenever the pointed node changes, and
    //XRubyWriter for every .kam save -- but because THIS walk re-runs on
    //every structural change and has nothing to gain from a consistent view:
    //it enumerates children and reads one flag.  A full Snapshot bundles the
    //subtree, so doing it from the root would bundle the whole tree at every
    //driver creation and all the way through a .kam load.  A single-node
    //snapshot still carries this node's own payload and its child list, which
    //is all the walk reads.  The dump, which runs once and does need a
    //consistency cut, is the opposite case -- see doc/design/PROVENANCE.md.
    Snapshot shot( *node, false);
    m_nodes[id].runtime = shot[ *node].isRuntime();
    if( !shot.size())
        return;
    if(auto lnode = dynamic_pointer_cast<XListNodeBase>(node)) {
        if(lnode->isAliasList())
            return;
    }
    int idx = 0;
    for(auto &&child: *shot.list()) {
        XString name = child->getName();
        //Children with empty names are the case where ORDER carries the
        //meaning (a calibration table's rows); index them so the report can
        //tell them apart, even though position is not identity.
        walk(child, path + "/" + (name.length() ? name : formatString("[%d]", idx)));
        ++idx;
    }
}

uint32_t
XJournal::subscribe(const shared_ptr<XNode> &node, const XString &path) {
    auto it = m_index.find(node.get());
    if(it != m_index.end())
        return it->second; //!< already subscribed, or reached again by a hard link.

    auto vnode = dynamic_pointer_cast<XValueNodeBase>(node);
    auto tnode = dynamic_pointer_cast<XTouchableNode>(node);
    auto lnode = dynamic_pointer_cast<XListNodeBase>(node);
    //The throughput readout is the journal talking about itself, once a
    //second, and it is derived from the files rather than from the
    //instrument.  It goes in the dump like any other node and is not
    //subscribed: a journal whose entries are mostly its own bookkeeping is
    //not a record of the measurement.
    if(auto rec = m_recorder.lock())
        if(node == rec->statistics())
            vnode.reset();

    uint32_t id = (uint32_t)m_nodes.size();
    m_nodes.emplace_back();
    NodeRec &rec = m_nodes.back();
    rec.node = node;
    rec.path = path;
    rec.type = node->getTypename();
    rec.isValue = !!vnode;
    rec.isTouchable = !!tnode;
    rec.isList = !!lnode;
    m_sinks.emplace_back(this, id, !!dynamic_pointer_cast<XDoubleNode>(node));
    Sink &sink = m_sinks.back();

    if(vnode || tnode || lnode) {
        //Subscribing BEFORE anything is read is the whole ordering rule: a
        //change in the gap would otherwise appear nowhere, and a loss cannot
        //be repaired where a duplicate can.
        //
        //Single-node (multi_nodal = false, as the trans() macro uses): only
        //this node's payload is written, and bundling a driver's whole
        //subtree to attach one listener is exactly the disturbance the walk
        //above avoids.
        //
        //audit-ok: connect() is re-run on a CAS retry, but every retry clones
        //the payload afresh from committed state, so the listener list is
        //rebuilt rather than doubled; the discarded clones take the
        //superseded Listener objects with them.
        for(Transaction tr( *node, false);;) {
            if(vnode)
                rec.lsnValue = tr[ *vnode].onValueChanged().connect(
                    sink, &Sink::onValueChanged);
            if(tnode)
                rec.lsnTouch = tr[ *tnode].onTouch().connect(
                    sink, &Sink::onTouch);
            if(lnode) {
                //onCatch/onRelease rather than onListChanged: the latter is
                //coalesced to one per transaction and says only THAT
                //something changed, where these name what did.
                rec.lsnCatch = tr[ *lnode].onCatch().connect(
                    sink, &Sink::onCatch);
                rec.lsnRelease = tr[ *lnode].onRelease().connect(
                    sink, &Sink::onRelease);
                rec.lsnMove = tr[ *lnode].onMove().connect(
                    sink, &Sink::onMove);
            }
            if(tr.commitOrNext())
                break;
        }
    }
    m_index.emplace(node.get(), id);
    return id;
}

void
XJournal::drainOnce() {
    m_dropped += m_ring.takeDropped();
    m_ring.drain([this](const JournalT::Entry &e){
        if(e.record.id >= m_nodes.size())
            return;
        NodeRec &rec = m_nodes[e.record.id];
        ++rec.writes;
        ++rec.byThread[(unsigned int)((uint64_t)e.serial & 0xFFFFu)];
        if( !rec.first.isSet())
            rec.first = e.record.when;
        rec.last = e.record.when;
        if( !rec.bucketStart.isSet())
            rec.bucketStart = e.record.when;
        if(e.record.when.diff_msec(rec.bucketStart) >= 1000) {
            if(rec.bucketCount > rec.peakPerSec)
                rec.peakPerSec = rec.bucketCount;
            rec.bucketStart = e.record.when;
            rec.bucketCount = 0;
        }
        ++rec.bucketCount;
        ++m_captured;
        if(e.record.kind < NUM_KINDS)
            ++m_byKind[e.record.kind];
        if(e.record.flags & FLAG_SESSION)
            writeEntry(m_sessionOut, e);
        //A Setup run is the dump and nothing after it; entries belong to a
        //Logbook.
        if(m_runKeepsValues && (e.record.flags & FLAG_RUN))
            writeEntry(m_runOut, e);
        delete[] e.record.value;
    });
    //Flushed at a line boundary, so a copy taken now is both current and
    //parsable -- the same property that makes a killed session readable.
    m_sessionOut.flush();
    m_runOut.flush();
    m_bytesJournal = m_sessionOut.bytes() + m_runOut.bytes();
}

XString
XJournal::writeReport() {
    if(m_reportPath.empty())
        return {};
    std::ofstream ofs(m_reportPath.c_str(), std::ios::trunc);
    if( !ofs.good())
        return {};

    double elapsed = XTime::now().diff_msec(m_started) / 1000.0;
    if(elapsed < 1e-3)
        elapsed = 1e-3;

    size_t values = 0, touchables = 0, lists = 0, runtimes = 0, detached = 0,
        destroyed = 0, listeners = 0, written = 0;
    std::map<unsigned int, uintptr_t> byThread;
    for(auto &&rec: m_nodes) {
        if(rec.isValue) ++values;
        if(rec.isTouchable) ++touchables;
        if(rec.isList) ++lists;
        if(rec.runtime) ++runtimes;
        if( !rec.inTree) ++detached;
        if(rec.destroyed) ++destroyed;
        listeners += !!rec.lsnValue + !!rec.lsnTouch + !!rec.lsnCatch
            + !!rec.lsnRelease + !!rec.lsnMove;
        if(rec.writes) ++written;
        for(auto &&t: rec.byThread)
            byThread[t.first] += t.second;
    }

    ofs << "KAME provenance journal -- capture survey (stage 1: no file format, no replay)\n"
        << "see doc/design/PROVENANCE.md\n\n"
        << "session started : " << m_started.getTimeStr() << "\n"
        << "report written  : " << XTime::now().getTimeStr() << "\n"
        << formatString("elapsed         : %.0f s\n\n", elapsed);

    ofs << "SUBSCRIPTION\n"
        << formatString("  nodes known           : %d\n", (int)m_nodes.size())
        << formatString("    value nodes         : %d\n", (int)values)
        << formatString("    touchable nodes     : %d\n", (int)touchables)
        << formatString("    lists               : %d\n", (int)lists)
        << formatString("    runtime == true     : %d\n", (int)runtimes)
        << formatString("    left the tree       : %d (of which destroyed: %d)\n",
            (int)detached, (int)destroyed)
        << formatString("  structural events     : %u caught, %u released\n",
            m_catches, m_releases)
        << formatString("  the sweep found first : %llu arrivals, %llu departures\n",
            (unsigned long long)m_sweepFoundNew,
            (unsigned long long)m_sweepFoundDetached)
        << "                          (both should stay 0: membership is a list's\n"
        << "                           business alone, so onCatch/onRelease should\n"
        << "                           be the whole story and the sweep redundant)\n"
        << formatString("  listeners held        : %d\n", (int)listeners)
        << formatString("  tree walks            : %u (last %u ms, total %u ms)\n\n",
            m_walks, m_lastWalkMS, m_totalWalkMS);

    ofs << "CAPTURE\n"
        << formatString("  entries recorded      : %llu (%.2f /s)\n",
            (unsigned long long)m_captured, m_captured / elapsed)
        << formatString("  dropped (ring full)   : %llu\n", (unsigned long long)m_dropped)
        << formatString("  by kind               : value %llu, touch %llu, "
            "catch %llu, release %llu, move %llu\n",
            (unsigned long long)m_byKind[KIND_VALUE],
            (unsigned long long)m_byKind[KIND_TOUCH],
            (unsigned long long)m_byKind[KIND_CATCH],
            (unsigned long long)m_byKind[KIND_RELEASE],
            (unsigned long long)m_byKind[KIND_MOVE])
        << formatString("  nodes ever written    : %d of %d\n",
            (int)written, (int)m_nodes.size())
        << "  by committing thread  : (a thread says what it is at its start; one\n"
        << "                           that never did is a driver thread -- or a\n"
        << "                           thread nobody has taught to speak yet)\n";
    for(auto &&t: byThread) {
        ofs << formatString("      thread %-3u %-8s: %10llu\n", t.first,
            threadClassName(threadClassOf(t.first)),
            (unsigned long long)t.second);
        //What a thread writes says which thread it is, without any registry.
        auto countOf = [this, &t](size_t i)->uintptr_t {
            auto p = m_nodes[i].byThread.find(t.first);
            return (p == m_nodes[i].byThread.end()) ? 0 : p->second;
        };
        std::deque<size_t> top;
        for(size_t i = 0; i < m_nodes.size(); ++i)
            if(countOf(i))
                top.push_back(i);
        std::sort(top.begin(), top.end(),
            [&countOf](size_t a, size_t b){return countOf(a) > countOf(b);});
        for(size_t k = 0; k < std::min<size_t>(3, top.size()); ++k)
            ofs << formatString("          %8llu  %s\n",
                (unsigned long long)countOf(top[k]), m_nodes[top[k]].path.c_str());
    }

    ofs << "\nWRITES PER NODE\n"
        << "  'request' counts writes committed by the UI or a scripting thread --\n"
        << "  somebody asked for a value.  'report' counts every other thread: a\n"
        << "  driver writing back what the instrument says.  A node with BOTH is a\n"
        << "  SETTING THAT A DRIVER ALSO WRITES -- the category no flag can decide,\n"
        << "  and the reason attribution is per write rather than per node.\n"
        << "  Caveat: a Python driver commits from the scripting thread, so its\n"
        << "  reports are counted here as requests.\n\n"
        << "  Two rates, because the design asks two different questions of them.\n"
        << "  'session/s' x elapsed is what the node COSTS -- bytes per hour of\n"
        << "  session.  'peak/s' is the most writes in any one second, which is\n"
        << "  what a rate cap has to survive and what separates a node running at\n"
        << "  acquisition rate from one written now and then.  Times come from the\n"
        << "  write itself, so a burst is resolved as a burst; the buckets tumble,\n"
        << "  so a burst split across a boundary reads low -- peak/s is a lower\n"
        << "  bound.\n\n"
        << "  session/s    peak/s      writes   request    report  rt  path\n"
        << "  ('d' = left the tree but still alive, 'x' = destroyed)\n";
    std::deque<size_t> order;
    for(size_t i = 0; i < m_nodes.size(); ++i)
        if(m_nodes[i].writes)
            order.push_back(i);
    std::sort(order.begin(), order.end(), [this](size_t a, size_t b){
        return m_nodes[a].writes > m_nodes[b].writes;});
    const size_t LIMIT = 400;
    for(size_t k = 0; k < order.size(); ++k) {
        if(k >= LIMIT) {
            ofs << formatString("  ... and %d more nodes, not listed\n",
                (int)(order.size() - LIMIT));
            break;
        }
        NodeRec &rec = m_nodes[order[k]];
        uintptr_t request = 0, report = 0;
        for(auto &&t: rec.byThread)
            ((threadClassOf(t.first) == ThreadClass::UNKNOWN) ? report : request)
                += t.second;
        //The bucket still open can hold the maximum.
        uintptr_t peak = std::max(rec.peakPerSec, rec.bucketCount);
        ofs << formatString("  %9.3f  %8llu  %10llu  %8llu  %8llu  %s%s  %s\n",
            rec.writes / elapsed, (unsigned long long)peak,
            (unsigned long long)rec.writes,
            (unsigned long long)request, (unsigned long long)report,
            rec.runtime ? "R" : "-",
            rec.destroyed ? "x" : (rec.inTree ? " " : "d"),
            rec.path.c_str());
    }
    ofs.flush();
    return m_reportPath;
}
