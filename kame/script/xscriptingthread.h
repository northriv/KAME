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
#ifndef xscriptingthreadH
#define xscriptingthreadH
//---------------------------------------------------------------------------
#include "xnode.h"
#include "xlistnode.h"

#define SCRIPTING_THREAD_ACTION_KILL "kill"
#define SCRIPTING_THREAD_ACTION_WAKEUP "wakeup"
#define SCRIPTING_THREAD_ACTION_FAILURE "failure"
#define SCRIPTING_THREAD_ACTION_STARTING "starting"

#define SCRIPTING_THREAD_STATUS_RUN "run"
#define SCRIPTING_THREAD_STATUS_SLEEP "sleep"
#define SCRIPTING_THREAD_STATUS_ABORTING "aborting"
#define SCRIPTING_THREAD_STATUS_STARTING "starting"
#define SCRIPTING_THREAD_STATUS_N_A ""
//---------------------------------------------------------------------------
//! XScriptingThread object is a communicator for Ruby thread.
//! \sa XRubySupport
class XScriptingThread : public XNode {
public:
    XScriptingThread(const char *name, bool runtime, const XString &filename);
    virtual ~XScriptingThread() {}

	bool isRunning() const;
	bool isAlive() const;
	void kill();
	void resume();
  
	//! def. input gets(). Return "" if the buffer is empty.
	XString gets();
	const shared_ptr<XStringNode> &lineinput() const {return m_lineinput;}

	const shared_ptr<XStringNode> &status() const {return m_status;}
	const shared_ptr<XStringNode> &filename() const {return m_filename;}
//  shared_ptr<XStringNode> &action() const {return m_action;}
    const shared_ptr<XStringNode> &threadID() const {return m_threadID;}

	struct Payload : public XNode::Payload {
        using Talker = Talker<shared_ptr<XString>>;
        Talker &onMessageOut() {return m_tlkOnMessageOut;}
        const Talker &onMessageOut() const {return m_tlkOnMessageOut;}
	private:
        Talker m_tlkOnMessageOut;
	};
private:
	const shared_ptr<XStringNode> m_filename;
	shared_ptr<XStringNode> m_status;
	shared_ptr<XStringNode> m_action;
	shared_ptr<XStringNode> m_lineinput;
    shared_ptr<XStringNode> m_threadID;
    shared_ptr<Listener> m_lsnOnLineChanged;
	void onLineChanged(const Snapshot &shot, XValueNodeBase *);
	std::deque<XString> m_lineBuffer;
	XMutex m_lineBufferMutex;
};

class XMeasure;

//! Base class for cripting support.
//! \sa XScriptingThread
class XScriptingThreadList : public XAliasListNode<XScriptingThread> {
public:
    XScriptingThreadList(const char *name, bool runtime, const shared_ptr<XMeasure> &measure);
    virtual ~XScriptingThreadList() {}

    void terminate() {m_thread->terminate();}
    void join() {m_thread->join();}

    struct Payload : public XAliasListNode<XScriptingThread>::Payload {
        struct tCreateChild {
            XString type;
            XString name;
            shared_ptr<XListNodeBase> lnode;
            XCondition cond;
            shared_ptr<XNode> child;
        };
        using Talker = Talker<shared_ptr<tCreateChild>>;
        Talker &onChildCreated() {return m_tlkOnChildCreated;}
        const Talker &onChildCreated() const {return m_tlkOnChildCreated;}
    private:
        Talker m_tlkOnChildCreated;
    };
protected:
    virtual void *execute(const atomic<bool> &) = 0;

    shared_ptr<Listener> m_lsnChildCreated;
    void onChildCreated(const Snapshot &shot, const shared_ptr<Payload::tCreateChild> &x);

    const weak_ptr<XMeasure> m_measure;
    unique_ptr<XThread> m_thread;
};
//---------------------------------------------------------------------------
#endif //scriptingthreadH
