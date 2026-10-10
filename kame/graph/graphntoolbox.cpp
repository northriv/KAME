/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "graphntoolbox.h"

#include "ui_graphnurlform.h"
#include "graphwidget.h"
#include "graph.h"
#include <iomanip>

#include <QPushButton>
#include <QStatusBar>
#include <QStyle>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>

#define OFSMODE (std::ios::out | std::ios::app | std::ios::ate)

//---------------------------------------------------------------------------

XGraphNToolBox::XGraphNToolBox(const char *name, bool runtime, FrmGraphNURL *item) :
    XGraphNToolBox(name, runtime, item->m_graphwidget, item->m_edUrl,
        item->m_btnUrl, item->m_btnDump) {

}
XGraphNToolBox::XGraphNToolBox(const char *name, bool runtime, XQGraph *graphwidget,
    QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump, const char *selfilter) :
    XNode(name, runtime), m_btnDump(btndump),
    m_graph(create<XGraph> (name, false)),
    m_dump(create<XTouchableNode> ("Dump", true)),
    m_filename(create<XStringNode> ("FileName", true)) {
    graphwidget->setGraph(m_graph);
    if(ed && btn)
        m_conFilename = xqcon_create<XFilePathConnector> (m_filename, ed, btn,
            selfilter, true);
    if(btndump)
        m_conDump = xqcon_create<XQButtonConnector> (m_dump, btndump);

    iterate_commit([=](Transaction &tr){
        m_lsnOnFilenameChanged = tr[ *filename()].onValueChanged().connectWeakly(
            shared_from_this(), &XGraphNToolBox::onFilenameChanged);
        m_lsnOnIconChanged = tr[ *this].onIconChanged().connectWeakly(
            shared_from_this(),
            &XGraphNToolBox::onIconChanged, Listener::FLAG_MAIN_THREAD_CALL
                | Listener::FLAG_AVOID_DUP);
        tr.mark(tr[ *this].onIconChanged(), false);

        tr[ *dump()].setUIEnabled(false);
//        tr[ *m_graph->persistence()] = 0.4;
    });
}

XGraphNToolBox::~XGraphNToolBox() {
    m_stream.close();
}

void
XGraphNToolBox::onIconChanged(const Snapshot &shot, bool v) {
    if( !m_conDump)
        return;
    if( !m_conDump->isAlive()) return;
    if( !v)
        m_btnDump->setIcon(QApplication::style()->
            standardIcon(QStyle::SP_DialogSaveButton));
    else
        m_btnDump->setIcon(QApplication::style()->
            standardIcon(QStyle::SP_BrowserReload));
}
void
XGraphNToolBox::onFilenameChanged(const Snapshot &shot, XValueNodeBase *) {
    {
        XScopedLock<XMutex> lock(m_filemutex);

        if(m_stream.is_open())
            m_stream.close();
        m_stream.clear();
        const XString fname = shot[ *filename()].to_str();
        auto idx = fname.find_last_of('.');
        m_ext = (idx != std::string::npos) ? fname.substr(idx + 1) : "";
        m_oneFilePerShot = fname.length() && dumpsOneFilePerShot(m_ext);
        m_shotSeq = 0;
        bool ok;
        if(m_oneFilePerShot) {
            //Nothing to open yet: each Dump creates its own file in this folder.
            QFileInfo dir(QFileInfo(QString(fname.c_str())).absolutePath());
            ok = dir.isDir() && dir.isWritable();
        }
        else {
            m_stream.open(
                (const char*)QString(fname.c_str()).toLocal8Bit().data(),
                OFSMODE);
            ok = m_stream.good();
        }

        iterate_commit([=](Transaction &tr){
            if(ok) {
                m_lsnOnDumpTouched = tr[ *dump()].onTouch().connectWeakly(
                    shared_from_this(), &XGraphNToolBox::onDumpTouched);
                tr[ *dump()].setUIEnabled(true);
            }
            else {
                m_lsnOnDumpTouched.reset();
                tr[ *dump()].setUIEnabled(false);
            }
            tr.mark(tr[ *this].onIconChanged(), false);
        });
        if( !ok)
            gErrPrint(i18n("Failed to open file.")); //outside the closure, which may run more than once.
    }
}

void
XGraphNToolBox::dumpOneShot(const Snapshot &shot) {
    const QFileInfo templ(QString(shot[ *filename()].to_str().c_str()));
    const QString dir = templ.absolutePath(), stem = templ.completeBaseName(), suffix = templ.suffix();
    if( !m_shotSeq) {
        //Continues after the highest number already there.
        const QRegularExpression re("^" + QRegularExpression::escape(stem) + "_(\\d+)_");
        for(auto &&name: QDir(dir).entryList({stem + "_*." + suffix}, QDir::Files)) {
            auto m = re.match(name);
            if(m.hasMatch())
                m_shotSeq = std::max(m_shotSeq, m.captured(1).toUInt());
        }
    }
    QString path;
    do {
        ++m_shotSeq;
        path = QString("%1/%2_%3_%4.%5").arg(dir, stem)
            .arg(m_shotSeq, 4, 10, QChar('0'))
            .arg(QString(XTime::now().getTimeFmtStr("%Y%m%d-%H%M%S", false).c_str()), suffix); //no " +0.123"
    } while(QFileInfo::exists(path));
    //Hidden, in the same folder so the rename is atomic.
    const QString tmp = dir + "/." + QFileInfo(path).fileName() + ".part";
    {
        std::fstream stream((const char*)tmp.toLocal8Bit().data(),
            std::ios::out | std::ios::trunc | std::ios::binary);
        if( !stream.good()) {
            gErrPrint(i18n("Failed to open file.") + " " + tmp);
            return;
        }
        try {
            dumpToFileThreaded(stream, shot, suffix.toStdString());
        }
        catch(...) {
            stream.close();
            QFile::remove(tmp);
            throw;
        }
        stream.close();
        if(stream.fail()) {
            gErrPrint(i18n("Failed to write file.") + " " + tmp);
            QFile::remove(tmp);
            return;
        }
    }
    if( !QFile::rename(tmp, path)) {
        gErrPrint(i18n("Failed to rename file.") + " " + path);
        QFile::remove(tmp);
        return;
    }
    gMessagePrint(formatString_tr(I18N_NOOP("Succesfully written into %s."), path.toUtf8().constData()));
}

void
XGraphNToolBox::onDumpTouched(const Snapshot &, XTouchableNode *) {
    if(m_filemutex.trylock()) {
        m_filemutex.unlock();
    }
    else {
        gWarnPrint(i18n("Previous dump is still on going. It is deferred."));
    }
    m_threadDump.reset(new XThread{shared_from_this(),
        [this](const atomic<bool>&, Snapshot &&shot){
        XScopedLock<XMutex> filelock(m_filemutex);
        if( !m_oneFilePerShot && !m_stream.good()) {
            gErrPrint(i18n("File cannot open."));
            return;
        }
        Transactional::setCurrentPriorityMode(Priority::UI_DEFERRABLE);

        try {
            if(m_oneFilePerShot)
                dumpOneShot(shot);
            else
                dumpToFileThreaded(m_stream, shot, m_ext);
        }
        catch (XKameError &e) {
            // UI_DEFERRABLE can hit the STM starvation timeout under heavy
            // measurement contention; an uncaught exception in this XThread
            // would terminate the process.  The dump is simply lost.
            e.print(i18n("Dump failed: "));
        }

        if( !m_oneFilePerShot)
            m_stream.flush();
    }, Snapshot( *this)});

    iterate_commit([=](Transaction &tr){
        tr.mark(tr[ *this].onIconChanged(), true);
    });
}
