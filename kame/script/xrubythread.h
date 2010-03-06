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
#ifndef xrubythreadH
#define xrubythreadH
//---------------------------------------------------------------------------
#include "xnode.h"

#include <ruby.h>

#define RUBY_THREAD_ACTION_KILL "kill"
#define RUBY_THREAD_ACTION_WAKEUP "wakeup"
#define RUBY_THREAD_ACTION_FAILURE "failure"
#define RUBY_THREAD_ACTION_STARTING "starting"

#define RUBY_THREAD_STATUS_RUN "run"
#define RUBY_THREAD_STATUS_SLEEP "sleep"
#define RUBY_THREAD_STATUS_ABORTING "aborting"
#define RUBY_THREAD_STATUS_STARTING "starting"
#define RUBY_THREAD_STATUS_N_A ""
//---------------------------------------------------------------------------
//! XRubyThread object is a communicator for Ruby thread.
//! \sa XRubySupport
class XRubyThread : public XNode {
public:
	XRubyThread(const char *name, bool runtime, const XString &filename);
	virtual ~XRubyThread() {}

	bool isRunning() const;
	bool isAlive() const;
	void kill();
	void resume();
  
	XTalker<shared_ptr<XString> > &onMessageOut() {return m_tlkOnMessageOut;}
	//! def. input gets(). Return "" if the buffer is empty.
	XString gets();
	const shared_ptr<XStringNode> &lineinput() const {return m_lineinput;}

	const shared_ptr<XStringNode> &status() const {return m_status;}
	const shared_ptr<XStringNode> &filename() const {return m_filename;}
//  shared_ptr<XStringNode> &action() const {return m_action;}
	const shared_ptr<XLongNode> &threadID() const {return m_threadID;}
private:
	XTalker<shared_ptr<XString> > m_tlkOnMessageOut;
	const shared_ptr<XStringNode> m_filename;
	shared_ptr<XStringNode> m_status;
	shared_ptr<XStringNode> m_action;
	shared_ptr<XStringNode> m_lineinput;
	shared_ptr<XLongNode> m_threadID;
	shared_ptr<XListener> m_lsnOnLineChanged;
	void onLineChanged(const shared_ptr<XValueNodeBase> &);
	std::deque<XString> m_lineBuffer;
	XMutex m_lineBufferMutex;
};

//---------------------------------------------------------------------------
#endif
