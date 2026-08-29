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
#include "transaction_journal.h"
#include "xthread.h"
#include <deque>
#include <map>
#include <unordered_map>

class XValueNodeBase;
class XTouchableNode;
class XListNodeBase;

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

    //! \a KAME_JOURNAL — unset means the journal never subscribes.
    static bool enabledByEnvironment();
    //! Subscribes to everything under \a root and starts the drain thread.
    static shared_ptr<XJournal> start(const shared_ptr<XNode> &root);
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
    enum : uint32_t {KIND_VALUE = 0, KIND_TOUCH = 1, KIND_LIST = 2, NUM_KINDS = 3};
    enum : uint32_t {NO_NODE = ~(uint32_t)0u};

    //! What the capture path knows: which node, and what kind of change.
    //! The serial travels beside it in the ring and carries both the ordering
    //! and the committing thread.
    struct Record {
        uint32_t id;
        uint32_t kind;
    };
    using JournalT = Transactional::Journal<Record, 8192>;

    //! One per subscribed node.  A talker holds a reference to this object,
    //! so its address must never move — hence a deque, never a vector.  The
    //! id it carries is what removes any lookup from the capture path.
    struct Sink {
        XJournal *journal;
        uint32_t id;
        void onValueChanged(const Snapshot &shot, XValueNodeBase *node);
        void onTouch(const Snapshot &shot, XTouchableNode *node);
        void onListChanged(const Snapshot &shot, XListNodeBase *node);
    };

    //! The node table: written by the drain thread alone, which is also the
    //! thread that walks and subscribes.
    struct NodeRec {
        weak_ptr<XNode> node;
        XString path;
        XString type;
        bool runtime = false;
        bool isValue = false, isTouchable = false, isList = false;
        bool released = false;
        shared_ptr<Listener> lsnValue, lsnTouch, lsnList;
        uintptr_t writes = 0;
        std::map<unsigned int, uintptr_t> byThread; //!< serial's low 16 bits
        XTime first, last;
    };

    void capture(uint32_t id, uint32_t kind, const Snapshot &shot, const XNode &node) noexcept;
    void execute(const atomic<bool> &terminated);
    void walkAll();
    void walk(const shared_ptr<XNode> &node, const XString &path);
    //! \return the new node's id, or NO_NODE when it was already subscribed.
    uint32_t subscribe(const shared_ptr<XNode> &node, const XString &path);
    void drainOnce();

    JournalT m_ring;
    atomic<bool> m_structureDirty {true};
    weak_ptr<XNode> m_root;
    //! Declared BEFORE the node table on purpose: a talker holds a reference
    //! to a Sink, and the shared_ptr that keeps that listener alive lives in
    //! the node table.  Destroying the table first expires the listeners, so
    //! no talker can reach a Sink that has already gone.
    std::deque<Sink> m_sinks;
    std::deque<NodeRec> m_nodes;
    std::unordered_map<const XNode *, uint32_t> m_index;
    unique_ptr<XThread> m_thread;

    XTime m_started;
    uintptr_t m_captured = 0, m_dropped = 0;
    uintptr_t m_byKind[NUM_KINDS] = {};
    unsigned int m_walks = 0, m_lastWalkMS = 0, m_totalWalkMS = 0;
    unsigned int m_reportInterval = 60; //!< s
    XString m_reportPath;
};

#endif /*xjournalH*/
