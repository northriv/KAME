/***************************************************************************
        Copyright (C) 2002-2023 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
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

#define OFSMODE (std::ios::out | std::ios::app | std::ios::ate)

//---------------------------------------------------------------------------

XGraphNToolBox::XGraphNToolBox(const char *name, bool runtime, FrmGraphNURL *item) :
    XGraphNToolBox(name, runtime, item->m_graphwidget, item->m_edUrl,
        item->m_btnUrl, item->m_btnDump) {

}
XGraphNToolBox::XGraphNToolBox(const char *name, bool runtime, XQGraph *graphwidget,
    QLineEdit *ed, QAbstractButton *btn, QPushButton *btndump) :
    XNode(name, runtime), m_btnDump(btndump),
    m_graph(create<XGraph> (name, false)),
    m_dump(create<XTouchableNode> ("Dump", true)),
    m_filename(create<XStringNode> ("FileName", true)) {
    graphwidget->setGraph(m_graph);
    if(ed && btn)
        m_conFilename = xqcon_create<XFilePathConnector> (m_filename, ed, btn,
            "Data files (*.dat);;All files (*.*)", true);
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
        m_stream.open(
            (const char*)QString(shot[ *filename()].to_str().c_str()).toLocal8Bit().data(),
            OFSMODE);

        iterate_commit([=](Transaction &tr){
            if(m_stream.good()) {
                m_lsnOnDumpTouched = tr[ *dump()].onTouch().connectWeakly(
                    shared_from_this(), &XGraphNToolBox::onDumpTouched);
                tr[ *dump()].setUIEnabled(true);
            }
            else {
                m_lsnOnDumpTouched.reset();
                tr[ *dump()].setUIEnabled(false);
                gErrPrint(i18n("Failed to open file."));
            }
            tr.mark(tr[ *this].onIconChanged(), false);
        });
    }
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
        if( !m_stream.good()) {
            gErrPrint(i18n("File cannot open."));
            return;
        }
        Transactional::setCurrentPriorityMode(Priority::UI_DEFERRABLE);

        dumpToFileThreaded(m_stream);

        m_stream.flush();
    }, Snapshot( *this)});

    iterate_commit([=](Transaction &tr){
        tr.mark(tr[ *this].onIconChanged(), true);
    });
}
