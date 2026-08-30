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

bool
XJournalFile::open(const XString &path, XString &errmsg) {
    gzFile fd = gzopen(QString::fromStdString(path).toLocal8Bit().data(), "rb");
    if( !fd) {
        errmsg = i18n_noncontext("cannot be opened");
        return false;
    }
    XJournalFile got;
    got.m_path = path;

    //Written by whoever wrote the line before: a line that carries no stamp
    //of its own -- the initial state of a node that appeared mid-run -- did
    //happen there, and dropping it to zero would sort it before the run.
    XTime now;
    bool haveHeader = false;
    QByteArray line;
    while(gzline(fd, line)) {
        QJsonDocument doc = QJsonDocument::fromJson(line);
        if( !doc.isObject()) {
            //A journal is flushed once a second and a killed KAME leaves a
            //partial last line.  That is expected, not corruption.
            ++got.m_unknown;
            continue;
        }
        QJsonObject o = doc.object();

        if(o.contains("tu")) {
            //toInteger, not toDouble: a serial is 64 bits by construction
            //(48-bit counter over a 16-bit thread id) and microseconds since
            //the epoch are 51 today.  Qt 6 keeps whole numbers whole.
            long long tu = (long long)o.value("tu").toInteger(0);
            now = XTime(tu / 1000000, tu % 1000000);
        }

        if(o.contains("format")) {
            if(o.value("format").toString() != "kame-journal") {
                gzclose(fd);
                errmsg = i18n_noncontext("is not a KAME journal");
                return false;
            }
            if( !haveHeader) {
                got.m_kind = o.value("kind").toString().toStdString();
                got.m_session = o.value("session").toString().toStdString();
                got.m_mode = o.value("mode").toString().toStdString();
                got.m_rawFile = o.value("raw").toString().toStdString();
                haveHeader = true;
            }
            continue;
        }

        XString t = o.value("t").toString().toStdString();
        //A line that says when it happened, in either spelling.  Only the
        //dump omits both -- it is a state, not an act -- so this is also the
        //test for that.
        bool stamped = o.contains("tu") || o.contains("ts");
        Event e;
        e.when = now;
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
            got.m_nodes[n.id] = n;
            e.kind = Event::Kind::NODE;
        }
        else if(t == "v") {
            e.kind = Event::Kind::VALUE;
            //No stamp of its own means it is a node's initial state, written
            //where the node was first seen -- the dump at the head, or a
            //subtree that appeared later.  It is a baseline, not a change.
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
            ++got.m_unknown;
            continue;
        }
        got.m_events.push_back(std::move(e));
    }
    gzclose(fd);

    if( !haveHeader) {
        errmsg = i18n_noncontext("has no journal header");
        return false;
    }
    got.m_isOpen = true;
    *this = std::move(got);
    return true;
}
