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
#include "xjournalreplay.h"

#include "support.h"

#include <zlib.h>
#include <cstring>

#include <QByteArray>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>

//! One line, however long.  gzgets stops at the buffer end as well as at the
//! newline and says nothing about which happened, so the only reliable test
//! is whether what came back ends in one.  Values are arbitrary strings and
//! a dump line can be long; nothing here may assume a maximum.
static bool gzline(gzFile fd, QByteArray &line) {
    line.clear();
    char buf[4096];
    for(;;) {
        if( !gzgets(fd, buf, sizeof(buf)))
            return !line.isEmpty(); //!< EOF: a last line without its newline still counts
        line += QByteArray(buf);
        if(line.endsWith('\n'))
            return true;
    }
}

//! The eight bytes back into the number they were.  \sa exactOf() in xjournal.cpp
static bool exactFrom(const QString &b64, double *out) {
    QByteArray b = QByteArray::fromBase64(b64.toLatin1());
    if(b.size() != 8)
        return false;
    uint64_t bits = 0;
    for(int i = 0; i < 8; ++i)
        bits |= (uint64_t)(unsigned char)b[i] << (8 * i); //little-endian, as written
    double v;
    static_assert(sizeof(v) == sizeof(bits), "binary64 expected");
    memcpy( &v, &bits, sizeof(v));
    *out = v;
    return true;
}

static XString withExtension(const XString &given, const char *ext) {
    QString s = QString::fromStdString(given).trimmed();
    if(s.isEmpty())
        return {};
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

XJournalFile::~XJournalFile() {
    close();
}
XJournalFile::XJournalFile(XJournalFile &&x) {
    *this = std::move(x);
}
XJournalFile &
XJournalFile::operator=(XJournalFile &&x) {
    if(this == &x)
        return *this;
    close();
    m_gz = x.m_gz; x.m_gz = nullptr;
    m_path = std::move(x.m_path);
    m_kind = std::move(x.m_kind);
    m_session = std::move(x.m_session);
    m_mode = std::move(x.m_mode);
    m_rawFile = std::move(x.m_rawFile);
    m_nodes = std::move(x.m_nodes);
    m_at = x.m_at;
    m_unknown = x.m_unknown;
    m_timesKnown = x.m_timesKnown;
    m_held = x.m_held;
    m_holding = std::move(x.m_holding);
    return *this;
}

void
XJournalFile::close() {
    if(m_gz)
        gzclose((gzFile)m_gz);
    m_gz = nullptr;
    m_held = false;
}

bool
XJournalFile::scanNodes(const XString &path, std::map<uint32_t, NodeInfo> &out) {
    gzFile fd = gzopen(QString::fromStdString(path).toLocal8Bit().data(), "rb");
    if( !fd)
        return false;
    QByteArray line;
    while(gzline(fd, line)) {
        //Cheap first: most of a journal is values, and parsing them here would
        //make this pass cost several times what decompressing it does.
        if( !line.contains("\"t\":\"n\""))
            continue;
        QJsonDocument doc = QJsonDocument::fromJson(line);
        if( !doc.isObject())
            continue;
        QJsonObject o = doc.object();
        if(o.value("t").toString() != "n")
            continue;
        NodeInfo n;
        n.id = (uint32_t)o.value("id").toInteger(0);
        n.parent = (uint32_t)o.value("p").toInteger(0);
        n.index = (int)o.value("i").toInteger(-1);
        n.name = o.value("name").toString().toStdString();
        n.path = o.value("path").toString().toStdString();
        n.type = o.value("type").toString().toStdString();
        n.cls = o.value("class").toString().toStdString();
        n.runtime = o.value("runtime").toBool(false);
        XString list = o.value("list").toString().toStdString();
        n.isList = !list.empty();
        n.isAliasList = (list == "alias");
        out[n.id] = n;
    }
    gzclose(fd);
    return true;
}

XString
XJournalFile::journalBeside(const XString &rawpath) {
    XString j = withExtension(rawpath, ".kamj");
    if(j.empty() || !QFileInfo(QString::fromStdString(j)).isFile())
        return {};
    return j;
}

XString
XJournalFile::rawPath() const {
    if(m_rawFile.empty() || m_path.empty())
        return {};
    //The header records a base name, never a path: a pair that is moved or
    //copied together must still find itself, and an absolute path from
    //another machine would be a lie the moment the file was mailed.
    return QFileInfo(QString::fromStdString(m_path)).dir()
        .filePath(QString::fromStdString(m_rawFile)).toStdString();
}

//! \return false at end of file.  Header lines are consumed here rather than
//! reported: they are about the file, not about the run.
bool
XJournalFile::nextLine(Event &e) {
    QByteArray line;
    while(gzline((gzFile)m_gz, line)) {
        QJsonDocument doc = QJsonDocument::fromJson(line);
        if( !doc.isObject()) {
            //A journal is flushed once a minute and a killed KAME leaves a
            //partial last line.  That is expected, not corruption.
            ++m_unknown;
            continue;
        }
        QJsonObject o = doc.object();

        if(o.contains("tu")) {
            //toInteger, not toDouble: a serial is 64 bits by construction
            //(48-bit counter over a 16-bit thread id) and microseconds since
            //the epoch are 51 today.  Qt 6 keeps whole numbers whole.
            long long tu = (long long)o.value("tu").toInteger(0);
            m_at = XTime(tu / 1000000, tu % 1000000);
        }
        if(o.contains("format")) {
            if( !m_kind.length()) {
                m_kind = o.value("kind").toString().toStdString();
                m_session = o.value("session").toString().toStdString();
                m_mode = o.value("mode").toString().toStdString();
                m_rawFile = o.value("raw").toString().toStdString();
            }
            continue;
        }

        XString t = o.value("t").toString().toStdString();
        //A line that says when it happened, in either spelling.  Only the
        //dump omits both -- it is a state, not an act -- so this is also the
        //test for that.
        bool stamped = o.contains("tu") || o.contains("ts");
        if(stamped && !o.contains("tu"))
            m_timesKnown = false;
        e = Event();
        e.when = m_at;
        e.stamped = stamped;
        e.id = (uint32_t)o.value("id").toInteger(0);

        if(t == "n") {
            NodeInfo n;
            n.id = e.id;
            n.parent = (uint32_t)o.value("p").toInteger(0);
            n.index = (int)o.value("i").toInteger(-1);
            n.name = o.value("name").toString().toStdString();
            n.path = o.value("path").toString().toStdString();
            n.type = o.value("type").toString().toStdString();
            n.cls = o.value("class").toString().toStdString();
            n.runtime = o.value("runtime").toBool(false);
            XString list = o.value("list").toString().toStdString();
            n.isList = !list.empty();
            n.isAliasList = (list == "alias");
            m_nodes[n.id] = n;
            e.kind = Event::Kind::NODE;
        }
        else if(t == "v") {
            e.kind = Event::Kind::VALUE;
            e.fromDump = !stamped;
            e.serial = (int64_t)o.value("s").toInteger(0);
            //A dump line carries no attribution -- it is a state, not an act.
            //Reading that as a request is the useful default: what a replay
            //restores from a baseline is exactly what a user would have set.
            e.request = (o.value("c").toString() != "report");
            e.value = o.value("v").toString().toStdString();
            if(o.contains("x"))
                e.hasExact = exactFrom(o.value("x").toString(), &e.exact);
        }
        else if(t == "released") {
            e.kind = Event::Kind::RELEASED;
        }
        else if((t == "run") || (t == "session")) {
            e.kind = Event::Kind::MARKER;
            e.marker = t + " " + o.value("state").toString().toStdString();
            e.file = o.value("file").toString().toStdString();
        }
        else {
            ++m_unknown;
            continue;
        }
        return true;
    }
    return false;
}

//! The head is everything before the first timestamped line: the header, and
//! the dump that is the baseline for everything after it.  Bounded by the tree.
bool
XJournalFile::readHead(const Apply &apply, XString &errmsg) {
    m_nodes.clear();
    m_at = XTime();
    m_held = false;
    for(;;) {
        Event e;
        if( !nextLine(e))
            break;      //!< a journal that recorded nothing is a legal journal
        if(e.stamped) {
            m_holding = std::move(e);
            m_held = true;
            break;
        }
        apply(e);
    }
    if( !m_kind.length()) {
        errmsg = i18n_noncontext("has no journal header");
        return false;
    }
    return true;
}

bool
XJournalFile::open(const XString &path, const Apply &apply, XString &errmsg) {
    close();
    m_kind.clear(); m_session.clear(); m_mode.clear(); m_rawFile.clear();
    m_unknown = 0;
    m_timesKnown = true;
    m_gz = gzopen(QString::fromStdString(path).toLocal8Bit().data(), "rb");
    if( !m_gz) {
        errmsg = i18n_noncontext("cannot be opened");
        return false;
    }
    m_path = path;
    if( !readHead(apply, errmsg)) {
        close();
        return false;
    }
    return true;
}

bool
XJournalFile::rewind(const Apply &apply) {
    if( !m_gz)
        return false;
    //gzrewind rather than a reopen: it restarts the same open stream, so a
    //journal that is still being appended to -- the session's own, most
    //likely -- is not swapped for a different file halfway through a replay.
    if(gzrewind((gzFile)m_gz) != 0)
        return false;
    XString errmsg;
    m_kind.clear(); m_session.clear(); m_mode.clear(); m_rawFile.clear();
    return readHead(apply, errmsg);
}

bool
XJournalFile::peekTime(XTime *when) {
    if( !m_gz)
        return false;
    if( !m_held) {
        if( !nextLine(m_holding))
            return false;
        m_held = true;
    }
    if(when)
        *when = m_holding.when;
    return true;
}

int64_t
XJournalFile::offset() const {
    if( !m_gz)
        return -1;
    return (int64_t)gzoffset((gzFile)m_gz);
}

bool
XJournalFile::advanceTo(const XTime &until, const Apply &apply) {
    if( !m_gz)
        return false;
    for(;;) {
        if( !m_held) {
            if( !nextLine(m_holding))
                return false;
            m_held = true;
        }
        //A line with no stamp of its own belongs where it appears -- a node
        //that came into existence between two entries -- so it travels with
        //the entry before it and is never what stops the walk.
        if(m_holding.stamped && (m_holding.when > until))
            return true;
        m_held = false;
        apply(m_holding);
    }
}
