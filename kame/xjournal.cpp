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
#include "support.h"
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <QDateTime>
#include <QDir>
#include <QStandardPaths>

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
    m_ring.capture(serial, r);
}

void
XJournal::Sink::onValueChanged(const Snapshot &shot, XValueNodeBase *node) {
    journal->capture(id, KIND_VALUE, shot, *node);
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
    journal->pushPending(e.caught, id, true);
}
void
XJournal::Sink::onRelease(const Snapshot &shot,
    const XListNodeBase::Payload::ReleaseEvent &e) {
    journal->capture(id, KIND_RELEASE, shot, *e.emitter);
    journal->pushPending(e.released, id, false);
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
XJournal::pushPending(const shared_ptr<XNode> &node, uint32_t listId, bool caught) {
    if( !node)
        return;
    XScopedLock<XCondition> lock(m_wake);
    m_pending.push_back(Pending{node, listId, caught});
    m_wake.signal();
}

XJournal::XJournal() {}

XJournal::~XJournal() {
    stop();
}

bool
XJournal::enabledByEnvironment() {
    const char *v = getenv("KAME_JOURNAL");
    return v && *v && (XString(v) != "0");
}

shared_ptr<XJournal>
XJournal::start(const shared_ptr<XNode> &root) {
    auto journal = std::make_shared<XJournal>();
    journal->m_root = root;
    journal->m_started = XTime::now();
    if(const char *v = getenv("KAME_JOURNAL_REPORT_SEC")) {
        int sec = atoi(v);
        if(sec > 0)
            journal->m_reportInterval = sec;
    }
    XString dir = managedDirectory("journal");
    if(dir.empty())
        gWarnPrint(i18n_noncontext("Journal: no writable directory; capturing without a report."));
    else
        journal->m_reportPath = dir + "/capture-"
            + QDateTime::currentDateTime().toString("yyyyMMdd-hhmmss").toStdString()
            + ".txt";
    gMessagePrint(XString(i18n_noncontext("Journaling this session: "))
        + (journal->m_reportPath.empty() ? XString("(no report)") : journal->m_reportPath));
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
        processPending();
        drainOnce();
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
    drainOnce();
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
            walk(p.node, path);
            auto it = m_index.find(p.node.get());
            if(it != m_index.end()) {
                m_nodes[it->second].catchAnnounced = true;
                m_nodes[it->second].inTree = true;
            }
        }
        else {
            ++m_releases;
            auto it = m_index.find(p.node.get());
            if(it != m_index.end())
                detachSubtree(it->second);
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

    uint32_t id = (uint32_t)m_nodes.size();
    m_nodes.emplace_back();
    NodeRec &rec = m_nodes.back();
    rec.node = node;
    rec.path = path;
    rec.type = node->getTypename();
    rec.isValue = !!vnode;
    rec.isTouchable = !!tnode;
    rec.isList = !!lnode;
    m_sinks.push_back(Sink{this, id});
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
    });
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
