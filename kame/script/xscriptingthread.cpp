/***************************************************************************
        Copyright (C) 2002-2024 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "xscriptingthread.h"

XScriptingThread::XScriptingThread(const char *name, bool runtime, const XString &filename)
	: XNode(name, runtime),
	  m_filename(create<XStringNode>("Filename", true)),
	  m_status(create<XStringNode>("Status", true)),
	  m_action(create<XStringNode>("Action", true)),
	  m_lineinput(create<XStringNode>("LineInput", true)),
      m_threadID(create<XStringNode>("ThreadID", true)) {

	iterate_commit([=](Transaction &tr){
        tr[ *m_threadID] = "-1";
	    tr[ *m_filename] = filename;
        tr[ *m_action] = SCRIPTING_THREAD_ACTION_STARTING;
        tr[ *m_status] = SCRIPTING_THREAD_STATUS_STARTING;
	    tr[ *lineinput()].setUIEnabled(false);

	    m_lsnOnLineChanged = tr[ *lineinput()].onValueChanged().connectWeakly(shared_from_this(),
            &XScriptingThread::onLineChanged);
    });
}
 
bool
XScriptingThread::isRunning() const {
    return (( **m_status)->to_str() == SCRIPTING_THREAD_STATUS_RUN);
}
bool
XScriptingThread::isAlive() const {
    return (( **m_status)->to_str() != SCRIPTING_THREAD_STATUS_N_A);
}
void
XScriptingThread::kill() {
    trans( *m_action) = SCRIPTING_THREAD_ACTION_KILL;
	trans( *lineinput()).setUIEnabled(false);
}
void
XScriptingThread::resume() {
    trans( *m_action) = SCRIPTING_THREAD_ACTION_WAKEUP;
}
void
XScriptingThread::onLineChanged(const Snapshot &shot, XValueNodeBase *) {
	XString line = shot[ *lineinput()];
	XScopedLock<XMutex> lock(m_lineBufferMutex);
	m_lineBuffer.push_back(line);
	iterate_commit([=](Transaction &tr){
		tr[ *lineinput()] = "";
		tr.unmark(m_lsnOnLineChanged);
    });
}

XString
XScriptingThread::gets() {
	XScopedLock<XMutex> lock(m_lineBufferMutex);
	if( !m_lineBuffer.size()) {
		lineinput()->setUIEnabled(true);
        return {};
	}
	XString line = m_lineBuffer.front();
	m_lineBuffer.pop_front();
//	lineinput()->setUIEnabled(false);
	return line + "\n";
}

XScriptingThreadList::XScriptingThreadList(const char *name, bool runtime, const shared_ptr<XMeasure> &measure)
: XAliasListNode<XScriptingThread>(name, runtime),
m_measure(measure) {
    iterate_commit([=](Transaction &tr){
        m_lsnChildCreated = tr[ *this].onChildCreated().connectWeakly(shared_from_this(),
            &XScriptingThreadList::onChildCreated, Listener::FLAG_MAIN_THREAD_CALL);
    });
    m_thread.reset(new XThread(shared_from_this(), &XScriptingThreadList::execute));
}
void
XScriptingThreadList::onChildCreated(const Snapshot &, const shared_ptr<Payload::tCreateChild> &x)  {
    x->child = x->lnode->createByTypename(x->type, x->name);
    x->lnode.reset();
    XScopedLock<XCondition> lock(x->cond);
    x->cond.signal();
}
