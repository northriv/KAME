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
//---------------------------------------------------------------------------
#include "xrubysupport.h"
#include "measure.h"
#include <QFile>
#include <QDataStream>
#include <math.h>


#define XRUBYSUPPORT_RB ":/script/xrubysupport.rb" //in the qrc.

XRuby::XRuby(const char *name, bool runtime, const shared_ptr<XMeasure> &measure)
    : XScriptingThreadList(name, runtime, measure) {
}
XRuby::~XRuby() {
}

Ruby::Value
XRuby::rnode_create(const shared_ptr<XNode> &node) {
    if(auto vnode = dynamic_pointer_cast<XValueNodeBase>(node)) {
        return m_rubyClassValueNode->rubyObject(vnode);
    }
    else if(auto lnode = dynamic_pointer_cast<XListNodeBase>(node)) {
        return m_rubyClassListNode->rubyObject(lnode);
    }
    else {
        return m_rubyClassNode->rubyObject(node);
    }
}

Ruby::Value
XRuby::rnode_child(const shared_ptr<XNode> &node, Ruby::Value var) {
    shared_ptr<XNode> child;

    if(Ruby::isConvertible<long>(var)) {
        long idx = Ruby::convert<long>(var);
        Snapshot shot( *node);
        if(shot.size()) {
            if ((idx >= 0) && (idx < (int)shot.size()))
                child = shot.list()->at(idx);
        }
        if(! child ) {
            throw (std::string)formatString("No such node idx:%ld on %s\n",
                idx, node->getName().c_str());
        }
    }
    else if(Ruby::isConvertible<const char*>(var)) {
        const char *name = Ruby::convert<const char*>(var);
        child = node->getChild(name);
        if( !child ) {
            throw (std::string)formatString("No such node name:%s on %s\n",
                name, node->getName().c_str());
        }
    }
    else
        throw (std::string)formatString("Ill format to find node on %s\n", node->getName().c_str());
    return rnode_create(child);
}
Ruby::Value
XRuby::rlistnode_create_child(const shared_ptr<XNode> &node, Ruby::Value rbtype, Ruby::Value rbname) {
    const char *type = Ruby::convert<const char*>(rbtype);
    const char *name = Ruby::convert<const char*>(rbname);

    shared_ptr<XNode> child;
    Snapshot shot( *node);
    if( shot[ *node].isRuntime() ) {
        throw (std::string)formatString("Node %s is run-time node!\n", node->getName().c_str());
    }
    auto lnode = dynamic_pointer_cast<XListNodeBase>(node);
    if( !lnode) {
        throw (std::string)formatString("Error on %s : Not ListNode. Could not make child"
            " name = %s, type = %s\n", node->getName().c_str(), name, type);
    }
    if(strlen(name))
        child = node->getChild(name);

    if( !child) {
        if(lnode->isThreadSafeDuringCreationByTypename()) {
            child = lnode->createByTypename(type, name);
        }
        else {
            shared_ptr<Payload::tCreateChild> x(new Payload::tCreateChild);
            x->lnode = lnode;
            x->type = type;
            x->name = name;
            Snapshot shot( *this);
            shot.talk(shot[ *this].onChildCreated(), x);
            XScopedLock<XCondition> lock(x->cond);
            while(x->lnode) {
                x->cond.wait();
            }
            child = x->child;
            x->child.reset();
        }
    }
    if( !child) {
        throw (std::string)formatString(
            "Error on %s : Could not make child"
            " name = %s, type = %s\n", node->getName().c_str(), name, type);
    }
    return rnode_create(child);
}
Ruby::Value
XRuby::rlistnode_release_child(const shared_ptr<XNode> &node, Ruby::Value rbchild) {
    auto lnode = dynamic_pointer_cast<XListNodeBase>(node);
    if( !lnode) {
        throw (std::string)formatString("Error on %s : Not ListNode.", node->getName().c_str());
    }
    Snapshot shot( *lnode);
    if( !shot[ *lnode].isUIEnabled()) {
        throw (std::string)formatString("Node %s is read-only!\n", node->getName().c_str());
    }
    shared_ptr<XNode> child(m_rubyClassNode->unwrap(rbchild));
    lnode->release(child);
    return Ruby::Nil;
}

Ruby::Value
XRuby::rnode_name(const shared_ptr<XNode> &node) {
    return Ruby::convertToRuby((std::string)node->getName());
}
Ruby::Value
XRuby::rnode_count(const shared_ptr<XNode> &node) {
    Snapshot shot( *node);
    return Ruby::convertToRuby(shot.size());
}
Ruby::Value
XRuby::rnode_touch(const shared_ptr<XNode> &node) {
    auto tnode = dynamic_pointer_cast<XTouchableNode>(node);
    if( !tnode)
        throw (std::string)formatString("Type mismatch on node %s\n", node->getName().c_str());
    dbgPrint(QString("Ruby, Node %1, touching."));
    tnode->iterate_commit([=](Transaction &tr){
        if(tr[ *tnode].isUIEnabled() )
            tr[ *tnode].touch();
        else
            throw (std::string)formatString("Node %s is read-only!\n", node->getName().c_str());
    });
    return Ruby::Nil;
}
Ruby::Value
XRuby::rvaluenode_set(const shared_ptr<XNode> &node, Ruby::Value var) {
    Snapshot shot( *node);
    auto vnode = dynamic_pointer_cast<XValueNodeBase>(node);
    assert(vnode);
    dbgPrint(QString("Ruby, Node %1, setting new value.").arg(node->getName()) );
    if( !shot[ *node].isUIEnabled()) {
        throw (std::string)formatString("Node %s is read-only!\n", node->getName().c_str());
    }
    XRuby::strOnNode(vnode, var);
    dbgPrint(QString("Ruby, Node %1, new value: %2.").arg(node->getName()).arg(shot[ *vnode].to_str()) );
    return XRuby::getValueOfNode(vnode);
}
Ruby::Value
XRuby::rvaluenode_load(const shared_ptr<XNode> &node, Ruby::Value var) {
    Snapshot shot( *node);
    auto vnode = dynamic_pointer_cast<XValueNodeBase>(node);
    assert(vnode);
    dbgPrint(QString("Ruby, Node %1, loading new value.").arg(node->getName()) );
    if( shot[ *node].isRuntime()) {
        throw (std::string)formatString("Node %s is run-time node!\n", node->getName().c_str());
    }
    XRuby::strOnNode(vnode, var);
    dbgPrint(QString("Ruby, Node %1, new value: %2.").arg(node->getName()).arg(shot[ *vnode].to_str()) );
    return XRuby::getValueOfNode(vnode);
}
Ruby::Value
XRuby::rvaluenode_get(const shared_ptr<XNode> &node) {
    auto vnode = dynamic_pointer_cast<XValueNodeBase>(node);
    assert(vnode);
    return XRuby::getValueOfNode(vnode);
}
Ruby::Value
XRuby::rvaluenode_to_str(const shared_ptr<XNode> &node) {
    auto vnode = dynamic_pointer_cast<XValueNodeBase>(node);
    assert(vnode);
    return Ruby::convertToRuby((std::string)( **vnode)->to_str());
}
void
XRuby::strOnNode(const shared_ptr<XValueNodeBase> &node, Ruby::Value value) {
	auto dnode = dynamic_pointer_cast<XDoubleNode>(node);
	auto inode = dynamic_pointer_cast<XIntNode>(node);
	auto uinode = dynamic_pointer_cast<XUIntNode>(node);
	auto lnode = dynamic_pointer_cast<XLongNode>(node);
	auto ulnode = dynamic_pointer_cast<XULongNode>(node);
	auto bnode = dynamic_pointer_cast<XBoolNode>(node);

    if(Ruby::isConvertible<long>(value)) {
        long integer = Ruby::convert<long>(value);
		if(uinode || ulnode) {
			if(integer < 0) {
                throw (std::string)formatString("Negative FIXNUM on %s\n", node->getName().c_str());
			}
		}
        if(inode) { trans( *inode) = integer; return; }
        if(lnode) { trans( *lnode) = integer; return; }
        if(uinode) { trans( *uinode) = integer; return; }
        if(ulnode) { trans( *ulnode) = integer; return; }
        double dbl = integer;
        if(dnode) { trans( *dnode) = dbl; return;}
        throw (std::string)formatString("FIXNUM is not appropreate on %s\n", node->getName().c_str());
    }
    else if(Ruby::isConvertible<double>(value)) {
        double dbl = Ruby::convert<double>(value);
        if(dnode) { trans( *dnode) = dbl; return;}
        throw (std::string)formatString("FLOAT is not appropreate on %s\n", node->getName().c_str());
    }
    else if(Ruby::isConvertible<const char*>(value)) {
		try {
            trans( *node).str(Ruby::convert<const char*>(value));
		}
		catch (XKameError &e) {
            throw (std::string)formatString("Validation error %s on %s\n"
				, (const char*)e.msg().c_str(), node->getName().c_str());
		}
    }
    else if(Ruby::isConvertible<bool>(value)) {
        bool v = Ruby::convert<bool>(value);
        if(bnode) { trans( *bnode) = v; return;}
        throw (std::string)formatString("TRUE is not appropreate on %s\n", node->getName().c_str());
    }
    else
        throw (std::string)formatString("UNKNOWN TYPE is not appropreate on %s\n", node->getName().c_str());
}
Ruby::Value
XRuby::getValueOfNode(const shared_ptr<XValueNodeBase> &node) {
	auto dnode = dynamic_pointer_cast<XDoubleNode>(node);
	auto inode = dynamic_pointer_cast<XIntNode>(node);
	auto uinode = dynamic_pointer_cast<XUIntNode>(node);
	auto lnode = dynamic_pointer_cast<XLongNode>(node);
	auto ulnode = dynamic_pointer_cast<XULongNode>(node);
	auto bnode = dynamic_pointer_cast<XBoolNode>(node);
	auto snode = dynamic_pointer_cast<XStringNode>(node);
	Snapshot shot( *node);
    if(dnode) {return Ruby::convertToRuby((double)shot[ *dnode]);}
    if(inode) {return Ruby::convertToRuby((int)shot[ *inode]);}
    if(uinode) {return Ruby::convertToRuby((unsigned int)shot[ *uinode]);}
    if(lnode) {return Ruby::convertToRuby((long)shot[ *lnode]);}
    if(ulnode) {return Ruby::convertToRuby((unsigned long)shot[ *ulnode]);}
    if(bnode) {return Ruby::convertToRuby((bool)shot[ *bnode]);}
    if(snode) {return Ruby::convertToRuby((const std::string&)shot[ *snode]);}
    return Ruby::Nil;
}

shared_ptr<XScriptingThread>
XRuby::findRubyThread(const shared_ptr<XNode> &, Ruby::Value threadid)
{
    XString id = Ruby::convert<const char*>(threadid);
	shared_ptr<XScriptingThread> rubythread;
    Snapshot shot(*this);
	if(shot.size()) {
		for(XNode::const_iterator it = shot.list()->begin(); it != shot.list()->end(); ++it) {
			auto th = dynamic_pointer_cast<XScriptingThread>( *it);
			assert(th);
            if(id == shot[ *th->threadID()].to_str())
				rubythread = th;
		}
	}
	return rubythread;
}
Ruby::Value
XRuby::my_rbdefout(const shared_ptr<XNode> &node, Ruby::Value str, Ruby::Value threadid) {
    shared_ptr<XString> sstr(new XString(Ruby::convert<const char*>(str)));
    shared_ptr<XScriptingThread> rubythread(findRubyThread(node, threadid));
	if(rubythread) {
		Snapshot shot( *rubythread);
		shot.talk(shot[ *rubythread].onMessageOut(), sstr);
		dbgPrint(QString("Ruby [%1]; %2").arg(shot[ *rubythread->filename()].to_str()).arg( *sstr));
	}
	else {
		dbgPrint(QString("Ruby [global]; %1").arg(*sstr));
	}
    return Ruby::Nil;
}
Ruby::Value
XRuby::my_rbdefin(const shared_ptr<XNode> &node, Ruby::Value threadid) {
    shared_ptr<XScriptingThread> rubythread(findRubyThread(node, threadid));
    if(rubythread) {
        XString line = rubythread->gets();
        if(line.length())
            return Ruby::convertToRuby((std::string)line);
        return Ruby::Nil;
    }
    else {
        throw XString("UNKNOWN Ruby thread\n");
    }
}
Ruby::Value
XRuby::is_main_terminated(const shared_ptr<XNode> &) {
    return Ruby::convertToRuby(m_thread->isTerminated());
}

void *
XRuby::execute(const atomic<bool> &terminated) {
    int dummy;
    Transactional::setCurrentPriorityMode(Transactional::Priority::UI_DEFERRABLE);
    m_ruby.reset(new Ruby("KAME", &dummy));
    shared_ptr<XRuby> xruby = dynamic_pointer_cast<XRuby>(shared_from_this());
    m_rubyClassNode.reset(new Ruby::Class<XRuby, XNode>(xruby, "XNode"));
    m_rubyClassNode->defineMethod<&XRuby::rnode_name>("name");
    m_rubyClassNode->defineMethod<&XRuby::rnode_touch>("touch");
    m_rubyClassNode->defineMethod<&XRuby::rnode_child>("child");
    m_rubyClassNode->defineMethod<&XRuby::rnode_count>("count");

    m_rubyClassValueNode.reset(new Ruby::Class<XRuby, XNode>(xruby, "XValueNode",
        m_rubyClassNode->rubyClassObject()));
    m_rubyClassValueNode->defineMethod<&XRuby::rvaluenode_set>("internal_set");
    m_rubyClassValueNode->defineMethod<&XRuby::rvaluenode_load>("internal_load");
    m_rubyClassValueNode->defineMethod<&XRuby::rvaluenode_get>("internal_get");

    m_rubyClassListNode.reset(new Ruby::Class<XRuby, XNode>(xruby, "XListNode",
        m_rubyClassNode->rubyClassObject()));
    m_rubyClassListNode->defineMethod<&XRuby::rlistnode_create_child>("internal_create");
    m_rubyClassListNode->defineMethod<&XRuby::rlistnode_release_child>("release");

    shared_ptr<XMeasure> measure = m_measure.lock();
    assert(measure);
    XString name = measure->getName();
    name[0] = toupper(name[0]);
    volatile Ruby::Value rbRootNode = rnode_create(measure); //volatile, not to be garbage collected.
    m_ruby->defineGlobalConst(name.c_str(), rbRootNode);
    m_ruby->defineGlobalConst("RootNode", rbRootNode);


    volatile Ruby::Value rbRubyThreads = rnode_create(shared_from_this()); //volatile, not to be garbage collected.
    m_ruby->defineGlobalConst("XScriptingThreads", rbRubyThreads);
    m_rubyClassNode->defineSingletonMethod<&XRuby::my_rbdefout>(
        rbRubyThreads, "my_rbdefout");
    m_rubyClassNode->defineSingletonMethod<&XRuby::my_rbdefin>(
        rbRubyThreads, "my_rbdefin");
    m_rubyClassNode->defineSingletonMethod<&XRuby::is_main_terminated>(
        rbRubyThreads, "is_main_terminated");

    {
        QFile scriptfile(XRUBYSUPPORT_RB);
        if( !scriptfile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            gErrPrint("No KAME ruby support file installed.");
            return NULL;
        }
        fprintf(stderr, "Loading ruby scripting monitor.\n");
        char data[65536];
        QDataStream( &scriptfile).readRawData(data, sizeof(data));

        while( !terminated) {
            int state = m_ruby->evalProtect(data);
            if(state) {
                fprintf(stderr, "Ruby, exception(s) occurred.\n");
                m_ruby->printErrorInfo();
                break;
            }
        }
    }

    fprintf(stderr, "ruby fin");
    m_ruby.reset();
    fprintf(stderr, "ished\n");
	return NULL;
}
