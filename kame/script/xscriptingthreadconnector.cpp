/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "xrubysupport.h"
#include "xscriptingthreadconnector.h"
#include "xscriptingthread.h"
#include <QPushButton>
#include <QLabel>
#include <QTextBrowser>
#include <QLineEdit>
#include "ui_scriptingthreadtool.h"
#include "icons/icon.h"
#include <QCloseEvent>
#include <QStyle>

FrmScriptingThread::FrmScriptingThread(QWidget *w) :
    QForm<QWidget, Ui_FrmScriptingThread>(w), m_closable(false) {

}

void
FrmScriptingThread::closeEvent(QCloseEvent* ce) {
    if( !m_closable) {
        ce->ignore();
        gWarnPrint(i18n("Ruby thread is still running."));
    }
}

XScriptingThreadConnector::XScriptingThreadConnector(
    const shared_ptr<XScriptingThread> &rbthread, FrmScriptingThread *form,
    const shared_ptr<XRuby> &rbsupport) :
    XQConnector(rbthread, form),
    m_resume(XNode::createOrphan<XTouchableNode>("Resume", true)),
    m_kill(XNode::createOrphan<XTouchableNode>("Kill", true)),
    m_pForm(form),
    m_scriptThread(rbthread),
    m_rubySupport(rbsupport),
    m_conFilename(xqcon_create<XQLabelConnector>(
    	rbthread->filename(), form->m_plblFilename)),
    m_conStatus(xqcon_create<XQLabelConnector>(
    	rbthread->status(), form->m_plblStatus)),
    m_conResume(xqcon_create<XQButtonConnector>(
    	m_resume, (form->m_pbtnResume))),
    m_conKill(xqcon_create<XQButtonConnector>(
    	m_kill, form->m_pbtnKill)),
	m_conLineinput(xqcon_create<XQLineEditConnector>(
		rbthread->lineinput(), form->m_edLineinput)) {

    form->m_pbtnResume->setIcon(
        QApplication::style()->standardIcon(QStyle::SP_MediaPlay));
    form->m_pbtnKill->setIcon(
        QApplication::style()->standardIcon(QStyle::SP_BrowserStop));
            
    m_pForm->m_ptxtDefout->setReadOnly(true);
    m_pForm->m_ptxtDefout->setOpenLinks(false);
    m_pForm->m_ptxtDefout->setOpenExternalLinks(false);

    m_resume->iterate_commit([=](Transaction &tr){
		m_lsnOnResumeTouched = tr[ *m_resume].onTouch().connectWeakly(
			shared_from_this(), &XScriptingThreadConnector::onResumeTouched);
    });
    m_kill->iterate_commit([=](Transaction &tr){
		m_lsnOnKillTouched = tr[ *m_kill].onTouch().connectWeakly(
			shared_from_this(), &XScriptingThreadConnector::onKillTouched);
    });
    Snapshot shot = rbthread->iterate_commit([=](Transaction &tr){
	    m_lsnOnDefout = tr[ *rbthread].onMessageOut().connectWeakly(
            shared_from_this(), &XScriptingThreadConnector::onDefout, Listener::FLAG_MAIN_THREAD_CALL);
	    m_lsnOnStatusChanged = tr[ *rbthread->status()].onValueChanged().connectWeakly(
	        shared_from_this(), &XScriptingThreadConnector::onStatusChanged);
    });
    onStatusChanged(shot, rbthread->status().get());

    form->setWindowIcon( *g_pIconScript);
    form->setWindowTitle(rbthread->getLabel());
}
XScriptingThreadConnector::~XScriptingThreadConnector() {
    if(isItemAlive()) {
        m_pForm->m_ptxtDefout->clear();
    }
//    m_scriptThread->kill();
    m_rubySupport->release(m_scriptThread);
}
void
XScriptingThreadConnector::onStatusChanged(const Snapshot &shot, XValueNodeBase *) {
    bool alive = m_scriptThread->isAlive();
    if(isItemAlive()) {
        m_pForm->m_closable = !alive;
    }
    m_kill->setUIEnabled(alive);
    bool running = m_scriptThread->isRunning();
    m_resume->setUIEnabled(alive && !running);
}
void
XScriptingThreadConnector::onResumeTouched(const Snapshot &shot, XTouchableNode *node) {
    m_scriptThread->resume();
}
void
XScriptingThreadConnector::onKillTouched(const Snapshot &shot, XTouchableNode *node) {
    m_scriptThread->kill();
}
void
XScriptingThreadConnector::onDefout(const Snapshot &shot, const shared_ptr<XString> &str) {
    m_pForm->m_ptxtDefout->append( *str);
}