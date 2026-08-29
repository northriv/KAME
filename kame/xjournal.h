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
//! Provenance journal, first stage: capture only.
//! \sa doc/design/PROVENANCE.md
#ifndef xjournalH
#define xjournalH

#include "xnode.h"
#include "xlistnode.h" //!< for the catch/release events, which are its Payload's
#include "xitemnode.h" //!< XComboNode, for the run mode
#include "transaction_journal.h"
#include "xthread.h"
#include <deque>
#include <map>
#include <unordered_map>

class XValueNodeBase;
class XTouchableNode;
class XRawStreamRecorder;
class XDriverList;

//! What a run records, and what it is costing: the nodes behind the Journal
//! group in the driver pane.
//!
//! The user names the RUN here, not the binary -- the journal always exists
//! where the raw stream is optional, and of the two only the journal is
//! interpretable alone, since it names its own data file.  The raw path is
//! derived from this one (`run042.kamj` / `run042.kamb`).
class DECLSPEC_KAME XJournalRecorder : public XNode {
public:
    XJournalRecorder(const char *name, bool runtime,
        const shared_ptr<XDriverList> &drivers);

    //! The raw stream, which the journal OWNS rather than sits beside.  It
    //! kept its own Filename and Recording nodes while it was a sibling of
    //! the journal, which meant two ways to say what one run is doing -- and
    //! a script that set them got a .kamb with no .kamj.  It is still its own
    //! class: its capture is a file mutex and I/O from inside a listener,
    //! where the journal's is lock-free inside every commit, and those two
    //! disciplines are safer apart.  \sa doc/design/PROVENANCE.md
    const shared_ptr<XRawStreamRecorder> &rawStream() const {return m_rawstream;}

    //! The run's name.  Extensions are derived, not typed.
    const shared_ptr<XStringNode> &filename() const {return m_filename;}
    //! How much this run keeps.  \sa Mode
    const shared_ptr<XComboNode> &mode() const {return m_mode;}
    const shared_ptr<XBoolNode> &recording() const {return m_recording;}
    //! Whether the always-on session journal is written at all.  It has to
    //! be refusable: a dump is not free (an ODMR tree is 3000 nodes), and a
    //! background writer nobody can switch off is impolite whatever its size.
    const shared_ptr<XBoolNode> &sessionJournal() const {return m_sessionJournal;}
    //! "12.4 MB/s   3.2 GB" -- what it is costing, beside the controls.
    const shared_ptr<XStringNode> &statistics() const {return m_statistics;}

    //! Cumulative tiers, in order of magnitude: a few hundred KB, ~36 MB/hr,
    //! ~10 GB/hr.  **The labels are what a file carries** and are not to be
    //! respelled -- see doc/design/PROVENANCE.md.
    enum class Mode {SETUP = 0, LOGBOOK = 1, LOGBOOK_RAW = 2};
    static const char *modeLabel(Mode);
    Mode modeOf(const Snapshot &shot) const;

    //! Uncompressed bytes the raw stream has taken, 0 when there is none.
    uintptr_t rawBytesWritten() const;

    //! Told once, as the session journal opens: the file that is being
    //! written when the user has named nothing.  Showing it is what makes
    //! "always on" visible instead of merely true, and Mode and Recording
    //! mean nothing while it is the target -- a run cannot be started INTO
    //! the session journal -- so they are disabled until a run is named.
    void setSessionPath(const XString &);
    const XString &sessionPath() const {return m_sessionPath;}

    //! `<base>.kamj` / `<base>.kamb`, whatever extension the user typed.
    static XString journalPathOf(const XString &given);
    static XString rawPathOf(const XString &given);
private:
    void onRecordingChanged(const Snapshot &shot, XValueNodeBase *);
    void onFilenameChanged(const Snapshot &shot, XValueNodeBase *);
    //! Mode and Recording mean something only once the field names a run.
    void updateRunControls();
    XString m_sessionPath;
    const shared_ptr<XStringNode> m_filename;
    const shared_ptr<XComboNode> m_mode;
    const shared_ptr<XBoolNode> m_recording;
    const shared_ptr<XBoolNode> m_sessionJournal;
    const shared_ptr<XStringNode> m_statistics;
    const shared_ptr<XRawStreamRecorder> m_rawstream;
    shared_ptr<Listener> m_lsnOnRecordingChanged, m_lsnOnFilenameChanged;
};

//! Subscribes to every node in the tree and records what changes, who changed
//! it and when — the capture half of the provenance journal, with no file
//! format and no replay yet.
//!
//! This stage exists to be measured rather than shipped: what a listener on
//! every node costs, how fast settings actually change, and which nodes are
//! written by the UI, by a script, or by a driver.  That last question is the
//! one the design turns on — the class of a write, not a flag on the node —
//! and nothing but running it on a real instrument can answer it.  The report
//! it writes is therefore also the audit of `runtime` flags that the design
//! says the journal produces rather than requires.
//!
//! Off unless \a KAME_JOURNAL is set in the environment.  Subscription is the
//! switch: with no listener attached `Talker::createMessage()` returns nullptr
//! before allocating anything, so not starting this costs exactly nothing —
//! there is no global flag and no branch in the commit path to say so.
class XJournal {
public:
    XJournal();
    //! Virtual only because XThread's constructor dynamic_pointer_casts the
    //! owner it is handed.
    virtual ~XJournal();

    //! \a KAME_JOURNAL=1 — additionally write the developer survey.
    static bool enabledByEnvironment();
    //! Whether the capture engine should run at all.  On unless
    //! \a KAME_JOURNAL=0, which exists so that a crash can be attributed to
    //! this or cleared of it: a subsystem with no off switch cannot be
    //! bisected, and one that subscribes to every node in the tree is the
    //! first thing anyone will suspect.
    static bool engineWanted();
    //! Subscribes to everything under \a root and starts the drain thread.
    //! \a recorder carries what the user chose for the run; the capture
    //! itself runs whether or not anything is being written.
    static shared_ptr<XJournal> start(const shared_ptr<XNode> &root,
        const shared_ptr<XJournalRecorder> &recorder);
    //! Joins the drain thread after a last drain and a final report.
    //! **Must be called by the owner**: the running thread holds a
    //! shared_ptr back to this object, so the destructor cannot be what
    //! stops it.
    void stop();

    //! What a thread IS, which is what decides whether a write is a request
    //! (the user or a script asked for a value) or a report (a driver wrote
    //! back what the instrument says).  The serial carries the committing
    //! thread for free, but a thread id alone cannot tell a scripting thread
    //! from a driver thread -- measured on a real session, where the driver
    //! was thread 6 and the IPython kernel thread 4, indistinguishable
    //! without this.  So the threads say who they are, once each.
    enum class ThreadClass : int {UNKNOWN = 0, UI = 1, SCRIPT = 2};
    //! Call once from the thread itself.  Cheap, and independent of whether
    //! a journal is running at all.
    static void declareThisThread(ThreadClass);
    //! \param id the serial's low 16 bits.
    static ThreadClass threadClassOf(unsigned int id);
    static const char *threadClassName(ThreadClass);

    //! Where KAME keeps files of its own: \a AppLocalDataLocation, with
    //! \a sub under it, created if absent.  Local rather than roaming: on
    //! Windows these are machine-local and not small.
    //! \return empty when the directory could not be created.
    static XString managedDirectory(const char *sub);

    //! Writes the accumulated survey.  Drain thread only.
    //! \return the path written, or empty on failure.
    XString writeReport();

private:
    //! Which files an entry belongs in, decided where it is captured so that
    //! a value nobody will keep is never even formatted.
    enum : uint32_t {FLAG_EXACT = 0x1, FLAG_SESSION = 0x2, FLAG_RUN = 0x4};
    enum : uint32_t {KIND_VALUE = 0, KIND_TOUCH = 1, KIND_CATCH = 2,
        KIND_RELEASE = 3, KIND_MOVE = 4, NUM_KINDS = 5};

    //! What the capture path knows: which node, and what kind of change.
    //! The serial travels beside it in the ring and carries both the ordering
    //! and the committing thread.
    struct Record {
        uint32_t id;
        uint32_t kind;
        //! The value as text, pool-allocated and owned by the ring: freed by
        //! whoever ends up with it, the drain or the producer that could not
        //! place it.  Null unless a Logbook is actually being written -- the
        //! capture path does not format numbers nobody asked for.
        const char *value;
        //! The same number without the rounding to_str() applies, for the
        //! exact field.  Only meaningful with FLAG_EXACT.
        double exact;
        uint32_t flags;
        //! Stamped where the write happened, not where it is drained.  A
        //! journal that answers "what was it at 3:14" cannot take its times
        //! from whenever the reader got around to looking: that granularity
        //! would be the drain period, not the instrument's.
        XTime when;
    };
    using JournalT = Transactional::Journal<Record, 8192>;

    //! One per subscribed node.  A talker holds a reference to this object,
    //! so its address must never move — hence a deque, never a vector.  The
    //! id it carries is what removes any lookup from the capture path.
    struct Sink {
        Sink(XJournal *j, uint32_t i, bool d) noexcept
            : journal(j), id(i), isDouble(d) {}
        XJournal *journal;
        uint32_t id;
        //! Decided once, at subscribe time, so the capture path never asks.
        bool isDouble;
        //! When a DRIVER last wrote this node.  Read and written from
        //! whatever thread is committing; a lost update costs one extra
        //! entry, which is why it needs no lock.  Silence detection, so it
        //! counts every write: a node written at 4 Hz has never been silent.
        atomic<int64_t> lastReportUs {0};

        void onValueChanged(const Snapshot &shot, XValueNodeBase *node);
        void onTouch(const Snapshot &shot, XTouchableNode *node);
        //! Membership is the ONLY way a node joins or leaves the tree, and it
        //! happens exclusively through a list -- so these two events are the
        //! whole structural story, and they name the node rather than merely
        //! saying that something changed.
        void onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e);
        void onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e);
        void onMove(const Snapshot &shot, const XListNodeBase::Payload::MoveEvent &e);
    };

    //! The node table: written by the drain thread alone, which is also the
    //! thread that walks and subscribes.
    struct NodeRec {
        weak_ptr<XNode> node;
        XString path;
        XString type;
        bool runtime = false;
        bool isValue = false, isTouchable = false, isList = false;
        //! Still reachable from the root at the last walk.  A node can leave
        //! the tree and go on living -- a released driver whose object a
        //! script still holds -- so being detached and being destroyed are
        //! two different things, and the survey has to say which.
        bool reachable = true;
        //! Still a member of the tree as far as we have been told.
        bool inTree = true;
        //! onRelease said so, rather than a sweep noticing afterwards.
        bool detachAnnounced = false;
        //! onCatch announced its arrival, rather than a sweep finding it.
        bool catchAnnounced = false;
        bool destroyed = false;
        shared_ptr<Listener> lsnValue, lsnTouch, lsnCatch, lsnRelease, lsnMove;
        uintptr_t writes = 0;
        std::map<unsigned int, uintptr_t> byThread; //!< serial's low 16 bits
        XTime first, last;
        //! Most writes seen in any one second -- the number a rate cap has to
        //! survive, and the one that says whether this node runs at
        //! acquisition rate or is written now and then.  An average over the
        //! node's whole active window cannot answer either question: it is
        //! diluted by every idle stretch inside that window.  Tumbling
        //! buckets, so a burst split across a boundary reads low: this is a
        //! LOWER BOUND on the peak.
        XTime bucketStart;
        uintptr_t bucketCount = 0, peakPerSec = 0;
    };

    //! One journal file: JSON Lines, gzipped, named `.kamj`.  The dump
    //! dominates and compresses ten to one (608 KB of an ODMR tree becomes
    //! 62 KB, measured), and the compression is part of the FORMAT rather
    //! than something done to the file afterwards -- which is why the name
    //! does not carry `.gz`, the same choice `.docx`, `.jar` and `.epub`
    //! make.  Flushed at a line boundary
    //! with Z_FULL_FLUSH, so a copy taken mid-write is both current and
    //! parsable, which is also what makes a killed session readable.
    struct Out {
        ~Out() {close();}
        bool open(const XString &path);
        void line(const XString &s);
        //! Ends a deflate block so everything so far reads on its own.
        //! Throttled: Z_FULL_FLUSH resets the dictionary, so flushing on
        //! every drain would both cost compression and bloat the file.
        //! \param force at a boundary that matters -- a run opening or
        //!        closing, a structural change, the end of the session.
        void flush(bool force = false);
        void close();
        bool isOpen() const {return !!m_gz;}
        uintptr_t bytes() const {return m_bytes;} //!< uncompressed
    private:
        void *m_gz = nullptr; //!< gzFile
        uintptr_t m_bytes = 0;
        bool m_dirty = false;
        XTime m_flushedAt;
    };
    void capture(uint32_t id, uint32_t kind, const Snapshot &shot, const XNode &node) noexcept;
    void captureValue(Sink &sink, const Snapshot &shot, XValueNodeBase &node) noexcept;
    //! Opens / closes the run file as the user's switch says, writes the
    //! dump when it opens, and keeps the statistics node current.
    void syncRun();
    void writeHeader(Out &out, const char *kind);
    void writeDump(Out &out);
    void dumpSubtree(Out &out, const Snapshot &shot,
        const shared_ptr<XNode> &node, const XString &path,
        uint32_t parentId, int index);
    void writeEntry(Out &out, const JournalT::Entry &e);
    //! Opens the always-on session journal and writes its dump.
    void openSession();
    void updateStatistics();
    void pushPending(const shared_ptr<XNode> &node, uint32_t listId, bool caught,
        int index);
    void execute(const atomic<bool> &terminated);
    //! Subscribes what onCatch announced and marks off what onRelease did,
    //! on this object's own thread rather than inside somebody's commit.
    void processPending();
    //! Marks \a id and everything under it as no longer in the tree.
    void detachSubtree(uint32_t id);
    //! The safety net, and a measurement in itself: a full walk that counts
    //! what the structural events failed to announce.  If those counts stay
    //! at zero on a real instrument, the sweep is provably redundant.
    void sweep();
    void walk(const shared_ptr<XNode> &node, const XString &path);
    //! \return the node's id, new or already known.  Known also covers
    //! "reached again through a hard link", which is how a node hard-linked
    //! under two parents is counted once and keeps its owner's path.
    uint32_t subscribe(const shared_ptr<XNode> &node, const XString &path);
    void drainOnce();

    JournalT m_ring;
    //! What onCatch / onRelease handed over, waiting for the journal's own
    //! thread.  A structural event is rare enough that a short mutex costs
    //! nothing; it is taken for a push and a swap, never across a snapshot.
    //!
    //! The condition IS the queue's mutex, so a caught node wakes the thread
    //! at once instead of waiting out a poll: the gap between a node joining
    //! the tree and being subscribed is a gap in the record.
    struct Pending {
        shared_ptr<XNode> node;
        uint32_t listId;
        bool caught;
        int index; //!< position, which is the meaning in the lists that have one
    };
    XCondition m_wake;
    std::deque<Pending> m_pending;
    weak_ptr<XNode> m_root;
    //! Declared BEFORE the node table on purpose: a talker holds a reference
    //! to a Sink, and the shared_ptr that keeps that listener alive lives in
    //! the node table.  Destroying the table first expires the listeners, so
    //! no talker can reach a Sink that has already gone.
    std::deque<Sink> m_sinks;
    std::deque<NodeRec> m_nodes;
    std::unordered_map<const XNode *, uint32_t> m_index;
    unique_ptr<XThread> m_thread;

    weak_ptr<XJournalRecorder> m_recorder;
    //! Always open unless the user says otherwise: provenance must exist for
    //! a session in which no recording was started.  Requests and structure
    //! go here in full; an acquisition stream does not, or it would not stay
    //! small.
    Out m_sessionOut;
    //! Open between the Write switch going on and off: the run.
    Out m_runOut;
    XString m_sessionPath, m_openPath, m_session;
    //! Whether the RUN wants values -- false for a Setup run, and while no
    //! run is open.  The session journal keeps its own (much sparser) share
    //! regardless, so this gates one stream, not capture itself.
    atomic<bool> m_runKeepsValues {false};
    bool m_runOpen = false;
    uintptr_t m_bytesJournal = 0, m_bytesLast = 0, m_rawBytesAtStart = 0;
    uintptr_t m_sessionSkipped = 0;
    //! A report on a node a driver has written within this long is the
    //! acquisition stream, and is not what an ALWAYS-ON journal is for.  A
    //! report after a longer silence is the state a device announces -- at
    //! `open`, or when something changed -- and is exactly what it is for.
    //!
    //! This governs the session journal alone.  A run keeps everything: the
    //! Logbook tier is what the user asked for, and thinning it silently is
    //! what a provenance record must not do.  \sa doc/design/PROVENANCE.md
    enum : int64_t {SESSION_QUIET_US = 10 * 1000000};
    XTime m_statsAt;

    XTime m_started;
    uintptr_t m_captured = 0, m_dropped = 0;
    uintptr_t m_byKind[NUM_KINDS] = {};
    unsigned int m_walks = 0, m_lastWalkMS = 0, m_totalWalkMS = 0;
    unsigned int m_catches = 0, m_releases = 0;
    //! Nodes a sweep had to discover, and detachments a sweep had to notice:
    //! both are event-path misses, and both should be zero.
    uintptr_t m_sweepFoundNew = 0, m_sweepFoundDetached = 0;
    unsigned int m_reportInterval = 60; //!< s
    unsigned int m_sweepInterval = 30; //!< s
    XString m_reportPath;
};

#endif /*xjournalH*/
