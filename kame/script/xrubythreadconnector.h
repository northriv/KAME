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
#ifndef XRUBYTHREADCONNECTOR_H_
#define XRUBYTHREADCONNECTOR_H_

#include "xnodeconnector.h"
#include "ui_rubythreadtool.h"

class XRubyThread;
class XRuby;
class FrmRubyThread;

class XRubyThreadConnector : public XQConnector {
	Q_OBJECT
public:
    XRubyThreadConnector(const shared_ptr<XRubyThread> &rbthread, FrmRubyThread *form,
						 const shared_ptr<XRuby> &rbsupport);
    virtual ~XRubyThreadConnector();
    
	const shared_ptr<XTouchableNode> &resume() const {return m_resume;}
	const shared_ptr<XTouchableNode> &kill() const {return m_kill;}
private:
    const shared_ptr<XTouchableNode> m_resume;
    const shared_ptr<XTouchableNode> m_kill;
    shared_ptr<Listener> m_lsnOnResumeTouched;
    shared_ptr<Listener> m_lsnOnKillTouched;
    shared_ptr<Listener> m_lsnOnDefout;
    shared_ptr<Listener> m_lsnOnStatusChanged;
    void onResumeTouched(const Snapshot &shot, XTouchableNode *node);
    void onKillTouched(const Snapshot &shot, XTouchableNode *node);
    void onDefout(const Snapshot &shot, const shared_ptr<XString> &str);
    void onStatusChanged(const Snapshot &shot, XValueNodeBase *node);
    FrmRubyThread *const m_pForm;
    const shared_ptr<XRubyThread> m_rubyThread;
    const shared_ptr<XRuby> m_rubySupport;
    xqcon_ptr m_conFilename, m_conStatus, m_conResume, m_conKill, m_conLineinput;
};

class QCloseEvent;
class FrmRubyThread : public QForm<QWidget, Ui_FrmRubyThread> {
public:
    FrmRubyThread(QWidget *w);
protected:
    void closeEvent(QCloseEvent* ce);
private:
    friend class XRubyThreadConnector;
    bool m_closable;
};

#endif /*XRUBYTHREADCONNECTOR_H_*/
