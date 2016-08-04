/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
//---------------------------------------------------------------------------

#ifndef xrubysupportH
#define xrubysupportH

#include "rubywrapper.h"
#include "xrubythread.h"
#include "xnode.h"
#include "xlistnode.h"

class XMeasure;

//! Ruby scripting support, containing a thread running Ruby monitor program.
//! The monitor program synchronize Ruby threads and XRubyThread objects.
//! \sa XRubyThread
class XRuby : public XAliasListNode<XRubyThread> {
public:
	XRuby(const char *name, bool runtime, const shared_ptr<XMeasure> &measure);
	virtual ~XRuby();
  
	void resume() {m_thread.resume();}
	void terminate() {m_thread.terminate();}
    void join() {m_thread.waitFor();}

	struct Payload : public XAliasListNode<XRubyThread>::Payload {
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
	virtual void *execute(const atomic<bool> &);
private:
	//! Ruby Objects
    shared_ptr<XRubyThread> findRubyThread(const shared_ptr<XNode> &self, Ruby::Value threadid);
	//! def. output write()
    Ruby::Value my_rbdefout(const shared_ptr<XNode> &, Ruby::Value str, Ruby::Value threadid);
	//! def. input gets(). Return nil if the buffer is empty.
    Ruby::Value my_rbdefin(const shared_ptr<XNode> &, Ruby::Value threadid);
	//! 
    Ruby::Value is_main_terminated(const shared_ptr<XNode> &node);
	//! XNode wrappers
    static void strOnNode(const shared_ptr<XValueNodeBase> &node, Ruby::Value value);
    static Ruby::Value getValueOfNode(const shared_ptr<XValueNodeBase> &node);
    Ruby::Value rnode_create(const shared_ptr<XNode> &);
    Ruby::Value rnode_name(const shared_ptr<XNode> &);
    Ruby::Value rnode_touch(const shared_ptr<XNode> &);
    Ruby::Value rnode_count(const shared_ptr<XNode> &);
    Ruby::Value rnode_child(const shared_ptr<XNode> &, Ruby::Value);
    Ruby::Value rvaluenode_set(const shared_ptr<XNode> &node, Ruby::Value);
    Ruby::Value rvaluenode_load(const shared_ptr<XNode> &node, Ruby::Value);
    Ruby::Value rvaluenode_get(const shared_ptr<XNode> &node);
    Ruby::Value rvaluenode_to_str(const shared_ptr<XNode> &node);
    Ruby::Value rlistnode_create_child(const shared_ptr<XNode> &, Ruby::Value, Ruby::Value);
    Ruby::Value rlistnode_release_child(const shared_ptr<XNode> &, Ruby::Value);

	shared_ptr<Listener> m_lsnChildCreated;
	void onChildCreated(const Snapshot &shot, const shared_ptr<Payload::tCreateChild> &x);
  
	const weak_ptr<XMeasure> m_measure;
	XThread<XRuby> m_thread;

    unique_ptr<Ruby> m_ruby;
    unique_ptr<Ruby::Class<XRuby,XNode>> m_rubyClassNode;
    unique_ptr<Ruby::Class<XRuby,XNode>> m_rubyClassValueNode;
    unique_ptr<Ruby::Class<XRuby,XNode>> m_rubyClassListNode;
};

//---------------------------------------------------------------------------
#endif
