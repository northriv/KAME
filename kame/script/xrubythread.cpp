/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "xrubythread.h"

XRubyThread::XRubyThread(const char *name, bool runtime, const XString &filename)
	: XNode(name, runtime),
	  m_filename(create<XStringNode>("Filename", true)),
	  m_status(create<XStringNode>("Status", true)),
	  m_action(create<XStringNode>("Action", true)),
	  m_threadID(create<XLongNode>("ThreadID", true)),
	  m_lineinput(create<XStringNode>("LineInput", true)) {

	for(Transaction tr( *this);; ++tr) {
	    tr[ *m_threadID] = -1;
	    tr[ *m_filename] = filename;
	    tr[ *m_action] = RUBY_THREAD_ACTION_STARTING;
	    tr[ *m_status] = RUBY_THREAD_STATUS_STARTING;
	    tr[ *lineinput()].setUIEnabled(false);

	    m_lsnOnLineChanged = tr[ *lineinput()].onValueChanged().connectWeakly(shared_from_this(),
	        &XRubyThread::onLineChanged);
		if(tr.commit())
			break;
	}
}
 
bool
XRubyThread::isRunning() const {
    return (( **m_status)->to_str() == RUBY_THREAD_STATUS_RUN);
}
bool
XRubyThread::isAlive() const {
    return (( **m_status)->to_str() != RUBY_THREAD_STATUS_N_A);
}
void
XRubyThread::kill() {
    trans( *m_action) = RUBY_THREAD_ACTION_KILL;
	trans( *lineinput()).setUIEnabled(false);
}
void
XRubyThread::resume() {
    trans( *m_action) = RUBY_THREAD_ACTION_WAKEUP;
}
void
XRubyThread::onLineChanged(const Snapshot &shot, XValueNodeBase *) {
	XString line = shot[ *lineinput()];
	XScopedLock<XMutex> lock(m_lineBufferMutex);
	m_lineBuffer.push_back(line);
	for(Transaction tr( *this);; ++tr) {
		tr[ *lineinput()] = "";
		tr.unmark(m_lsnOnLineChanged);
		if(tr.commit())
			break;
	}
}

XString
XRubyThread::gets() {	
	XScopedLock<XMutex> lock(m_lineBufferMutex);
	if( !m_lineBuffer.size()) {
		lineinput()->setUIEnabled(true);
		return XString();
	}
	XString line = m_lineBuffer.front();
	m_lineBuffer.pop_front();
//	lineinput()->setUIEnabled(false);
	return line + "\n";
}
