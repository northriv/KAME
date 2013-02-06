/***************************************************************************
		Copyright (C) 2002-2013 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "xrubythreadconnector.h"
#include "xrubysupport.h"
#include "xrubythread.h"
#include <qpushbutton.h>
#include <qlabel.h>
#include <qtextbrowser.h>
#include <qlineedit.h>
#include "ui_rubythreadtool.h"
#include "icons/icon.h"
#include <kapplication.h>
#include <kiconloader.h>

XRubyThreadConnector::XRubyThreadConnector(
    const shared_ptr<XRubyThread> &rbthread, FrmRubyThread *form,
    const shared_ptr<XRuby> &rbsupport) :
    XQConnector(rbthread, form),
    m_resume(XNode::createOrphan<XTouchableNode>("Resume", true)),
    m_kill(XNode::createOrphan<XTouchableNode>("Kill", true)),
    m_pForm(form),
    m_rubyThread(rbthread),
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

    KIconLoader *loader = KIconLoader::global();
	form->m_pbtnResume->setIcon(loader->loadIcon("system-reboot",
		KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
	form->m_pbtnKill->setIcon(loader->loadIcon("process-stop",
		KIconLoader::Toolbar, KIconLoader::SizeSmall, true ) );
            
    m_pForm->m_ptxtDefout->setMaxLogLines(10000);
    m_pForm->m_ptxtDefout->setTextFormat(Qt::LogText);    
    
    for(Transaction tr( *m_resume);; ++tr) {
		m_lsnOnResumeTouched = tr[ *m_resume].onTouch().connectWeakly(
			shared_from_this(), &XRubyThreadConnector::onResumeTouched);
		if(tr.commit())
			break;
    }
    for(Transaction tr( *m_kill);; ++tr) {
		m_lsnOnKillTouched = tr[ *m_kill].onTouch().connectWeakly(
			shared_from_this(), &XRubyThreadConnector::onKillTouched);
		if(tr.commit())
			break;
    }
	for(Transaction tr( *rbthread);; ++tr) {
	    m_lsnOnDefout = tr[ *rbthread].onMessageOut().connectWeakly(
	        shared_from_this(), &XRubyThreadConnector::onDefout, XListener::FLAG_MAIN_THREAD_CALL);
	    m_lsnOnStatusChanged = tr[ *rbthread->status()].onValueChanged().connectWeakly(
	        shared_from_this(), &XRubyThreadConnector::onStatusChanged);
		if(tr.commit()) {
		    onStatusChanged(tr, rbthread->status().get());
			break;
		}
	}
        
    form->setWindowIcon( *g_pIconScript);
    form->setWindowTitle(rbthread->getLabel());
}
XRubyThreadConnector::~XRubyThreadConnector() {
    if(isItemAlive()) {
        m_pForm->m_ptxtDefout->clear();
    }
//    m_rubyThread->kill();
    m_rubySupport->release(m_rubyThread);
}
void
XRubyThreadConnector::onStatusChanged(const Snapshot &shot, XValueNodeBase *) {
    bool alive = m_rubyThread->isAlive();
    m_kill->setUIEnabled(alive);
    bool running = m_rubyThread->isRunning();
    m_resume->setUIEnabled(alive && !running);
}
void
XRubyThreadConnector::onResumeTouched(const Snapshot &shot, XTouchableNode *node) {
    m_rubyThread->resume();
}
void
XRubyThreadConnector::onKillTouched(const Snapshot &shot, XTouchableNode *node) {
    m_rubyThread->kill();
}
void
XRubyThreadConnector::onDefout(const Snapshot &shot, const shared_ptr<XString> &str) {
    m_pForm->m_ptxtDefout->append( *str);
}
