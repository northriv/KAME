/***************************************************************************
		Copyright (C) 2002-2012 Kentaro Kitagawa
		                   kitag@kochi-u.ac.jp

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
#include <ruby.h>

#include <math.h>
#include <kstandarddirs.h>

#define XRUBYSUPPORT_RB "xrubysupport.rb"

static inline VALUE string2RSTRING(const XString &str) {
	if(str.empty()) return rb_str_new2("");
	return rb_str_new2(str.c_str());
}

XRuby::XRuby(const char *name, bool runtime, const shared_ptr<XMeasure> &measure)
: XAliasListNode<XRubyThread>(name, runtime),
m_measure(measure), 
m_thread(shared_from_this(), &XRuby::execute) {
	for(Transaction tr( *this);; ++tr) {
		m_lsnChildCreated = tr[ *this].onChildCreated().connectWeakly(shared_from_this(),
			&XRuby::onChildCreated, XListener::FLAG_MAIN_THREAD_CALL);
		if(tr.commit())
			break;
	}
}
XRuby::~XRuby() {
}
void
XRuby::rnode_free(void *rnode) {
	struct rnode_ptr* st = static_cast<struct rnode_ptr*>(rnode);
	delete st;
}

VALUE
XRuby::rnode_create(const shared_ptr<XNode> &node, XRuby *xruby) {

	struct rnode_ptr *st
	= new rnode_ptr;

	st->ptr = node;
	st->xruby = xruby;
	VALUE rnew;
	auto vnode = dynamic_pointer_cast<XValueNodeBase>(node);
	if(vnode) {
		rnew = Data_Wrap_Struct(xruby->rbClassValueNode, 0, rnode_free, st);
	}
	else {
		auto lnode = dynamic_pointer_cast<XListNodeBase>(node);
		if(lnode) {
			rnew = Data_Wrap_Struct(xruby->rbClassListNode, 0, rnode_free, st);
		}
		else {
			rnew = Data_Wrap_Struct(xruby->rbClassNode, 0, rnode_free, st);
		}
	}
	return rnew;
}
VALUE
XRuby::rnode_child(VALUE self, VALUE var) {
	char errmsg[256];
	try {
		shared_ptr<XNode> child;
		struct rnode_ptr *st;
		Data_Get_Struct(self, struct rnode_ptr, st);

		if(shared_ptr<XNode> node = st->ptr.lock()) {
			switch (TYPE(var)) {
			long idx;
			case T_FIXNUM:
				idx = NUM2LONG(var);
				{ Snapshot shot( *node);
				if(shot.size()) {
					if ((idx >= 0) && (idx < (int)shot.size()))
						child = shot.list()->at(idx);
				}
				}
				if(! child ) {
					throw formatString("No such node idx:%ld on %s\n",
						idx, node->getName().c_str());
				}
				break;
			case T_STRING:
			{
				const char *name = RSTRING(var)->ptr;
				child = node->getChild(name);
				if( !child ) {
					throw formatString("No such node name:%s on %s\n",
						name, node->getName().c_str());
				}
			}
			break;
			default:
				throw formatString("Ill format to find node on %s\n", node->getName().c_str());
			}
		}
		else {
			throw XString("Node no longer exists\n");
		}
		return rnode_create(child, st->xruby );
	}
	catch (XString &s) {
		snprintf(errmsg, 256, "%s", s.c_str());
	}
	rb_raise(rb_eRuntimeError, "%s", errmsg);
	return Qnil;  
}
VALUE
XRuby::rlistnode_create_child(VALUE self, VALUE rbtype, VALUE rbname) {
	Check_Type(rbtype, T_STRING);
	if(TYPE(rbtype) != T_STRING) return Qnil;
	Check_Type(rbname, T_STRING);
	if(TYPE(rbname) != T_STRING) return Qnil;
	char *type = RSTRING(rbtype)->ptr;
	char *name = RSTRING(rbname)->ptr;

	char errmsg[256];
	try {
		shared_ptr<XNode> child;
		struct rnode_ptr *st;
		Data_Get_Struct(self, struct rnode_ptr, st);
		if(shared_ptr<XNode> node = st->ptr.lock()) {
			Snapshot shot( *node);
			if( shot[ *node].isRuntime() ) {
				throw formatString("Node %s is run-time node!\n", node->getName().c_str());
			}     
			auto lnode = dynamic_pointer_cast<XListNodeBase>(node);
			if( !lnode) {
				throw formatString("Error on %s : Not ListNode. Could not make child"
					" name = %s, type = %s\n", node->getName().c_str(), name, type);
			}
			if(strlen(name))
				child = node->getChild(name);
			/*
	      if(type != child->getTypename()) {
	          rb_raise(rb_eRuntimeError, "Different type of child exists on %s\n",
	             (const char*)node->getName().utf8());
	          return Qnil;
	      }
			 */
			if( !child) {
				shared_ptr<Payload::tCreateChild> x(new Payload::tCreateChild);
				x->lnode = lnode;
				x->type = type;
				x->name = name;
				Snapshot shot( *st->xruby);
				shot.talk(shot[ *st->xruby].onChildCreated(), x);
				XScopedLock<XCondition> lock(x->cond);
				while(x->lnode) {
					x->cond.wait();
				}
				child = x->child;
				x->child.reset();
			}
			if( !child) {
				throw formatString(
					"Error on %s : Could not make child"
					" name = %s, type = %s\n", node->getName().c_str(), name, type);
			}
		}
		else {
			throw XString("Node no longer exists\n");
		}
		return rnode_create(child, st->xruby );
	}
	catch (XString &s) {
		snprintf(errmsg, 256, "%s", s.c_str());
	}
	rb_raise(rb_eRuntimeError, "%s", errmsg);
	return Qnil;  
}
void
XRuby::onChildCreated(const Snapshot &shot, const shared_ptr<Payload::tCreateChild> &x)  {
	x->child = x->lnode->createByTypename(x->type, x->name);
	x->lnode.reset();
	XScopedLock<XCondition> lock(x->cond);
	x->cond.signal();
}
VALUE
XRuby::rlistnode_release_child(VALUE self, VALUE rbchild) {
	struct rnode_ptr *st;
	Data_Get_Struct(self, struct rnode_ptr, st);
	char errmsg[256];
	try {
		if(shared_ptr<XNode> node = st->ptr.lock()) {
			Snapshot shot( *node);
			if( !shot[ *node].isUIEnabled()) {
				throw formatString("Node %s is read-only!\n", node->getName().c_str());
			}     
			auto lnode = dynamic_pointer_cast<XListNodeBase>(node);
			if( !lnode) {
				throw formatString("Error on %s : Not ListNode. Could not release child\n"
					, node->getName().c_str());
			}
			struct rnode_ptr *st_child;
			Data_Get_Struct(rbchild, struct rnode_ptr, st_child);
			if(shared_ptr<XNode> child = st_child->ptr.lock()) {
				lnode->release(child);
				return Qnil;  
			}
			else {
				throw XString("Node no longer exists\n");
			}
		}
		else {
			throw XString("Node no longer exists\n");
		}
		return self;
	}
	catch (XString &s) {
		snprintf(errmsg, 256, "%s", s.c_str());
	}
	rb_raise(rb_eRuntimeError, "%s", errmsg);
	return Qnil;  
}

VALUE
XRuby::rnode_name(VALUE self) {
	struct rnode_ptr *st;
	Data_Get_Struct(self, struct rnode_ptr, st);
	char errmsg[256];
	try {
		if(shared_ptr<XNode> node = st->ptr.lock()) {
			return string2RSTRING(node->getName());
		}
		else {
			throw XString("Node no longer exists\n");
		}
	}
	catch (XString &s) {
		snprintf(errmsg, 256, "%s", s.c_str());
	}
	rb_raise(rb_eRuntimeError, "%s", errmsg);
	return Qnil;  
}
VALUE
XRuby::rnode_count(VALUE self) {
	struct rnode_ptr *st;
	Data_Get_Struct(self, struct rnode_ptr, st);
	char errmsg[256];
	try {
		if(shared_ptr<XNode> node = st->ptr.lock()) {
			Snapshot shot( *node);
			VALUE count = INT2NUM(shot.size());
			return count;
		}
		else {
			throw XString("Node no longer exists\n");
		}
	}
	catch (XString &s) {
		snprintf(errmsg, 256, "%s", s.c_str());
	}
	rb_raise(rb_eRuntimeError, "%s", errmsg);
	return Qnil;  
}
VALUE
XRuby::rnode_touch(VALUE self) {
	struct rnode_ptr *st;
	Data_Get_Struct(self, struct rnode_ptr, st);
	char errmsg[256];
	try {
		if(shared_ptr<XNode> node = st->ptr.lock()) {
			auto tnode = dynamic_pointer_cast<XTouchableNode>(node);
			if( !node)
				throw formatString("Type mismatch on node %s\n", node->getName().c_str());
			dbgPrint(QString("Ruby, Node %1, touching."));
			for(Transaction tr( *tnode);; ++tr) {
				if(tr[ *tnode].isUIEnabled() )
					tr[ *tnode].touch();
				else
					throw formatString("Node %s is read-only!\n", node->getName().c_str());
				if(tr.commit())
					break;
			}
			return Qnil;  
		}
		else {
			throw XString("Node no longer exists\n");
		}
	}
	catch (XString &s) {
		snprintf(errmsg, 256, "%s", s.c_str());
	}
	rb_raise(rb_eRuntimeError, "%s", errmsg);
	return Qnil;  
}
VALUE
XRuby::rvaluenode_set(VALUE self, VALUE var) {
	struct rnode_ptr *st;
	Data_Get_Struct(self, struct rnode_ptr, st);
	char errmsg[256];
	try {
		if(shared_ptr<XNode> node = st->ptr.lock()) {
			Snapshot shot( *node);
			auto vnode = dynamic_pointer_cast<XValueNodeBase>(node);
			assert(vnode);
			dbgPrint(QString("Ruby, Node %1, setting new value.").arg(node->getName()) );
			if( !shot[ *node].isUIEnabled()) {
				throw formatString("Node %s is read-only!\n", node->getName().c_str());
			}
			if(XRuby::strOnNode(vnode, var)) {
				return Qnil;
			}
			else {
				dbgPrint(QString("Ruby, Node %1, new value: %2.").arg(node->getName()).arg(shot[ *vnode].to_str()) );
				return XRuby::getValueOfNode(vnode);
			}
		}
		else {
			throw XString("Node no longer exists\n");
		}
	}
	catch (XString &s) {
		snprintf(errmsg, 256, "%s", s.c_str());
	}
	rb_raise(rb_eRuntimeError, "%s", errmsg);
	return Qnil;  
}
VALUE
XRuby::rvaluenode_load(VALUE self, VALUE var) {
	char errmsg[256];
	struct rnode_ptr *st;
	Data_Get_Struct(self, struct rnode_ptr, st);
	try {
		if(shared_ptr<XNode> node = st->ptr.lock()) {
			Snapshot shot( *node);
			auto vnode = dynamic_pointer_cast<XValueNodeBase>(node);
			assert(vnode);
			dbgPrint(QString("Ruby, Node %1, loading new value.").arg(node->getName()) );
			if( shot[ *node].isRuntime()) {
				throw formatString("Node %s is run-time node!\n", node->getName().c_str());
			}
			if(XRuby::strOnNode(vnode, var)) {
				return Qnil;
			}
			else {
				dbgPrint(QString("Ruby, Node %1, new value: %2.").arg(node->getName()).arg(shot[ *vnode].to_str()) );
				return XRuby::getValueOfNode(vnode);
			}
		}
		else {
			throw XString("Node no longer exists\n");
		}
	}
	catch (XString &s) {
		snprintf(errmsg, 256, "%s", s.c_str());
	}
	rb_raise(rb_eRuntimeError, "%s", errmsg);
	return Qnil;  

}
VALUE
XRuby::rvaluenode_get(VALUE self) {
	struct rnode_ptr *st;
	Data_Get_Struct(self, struct rnode_ptr, st);
	char errmsg[256];
	try {
		if(shared_ptr<XNode> node = st->ptr.lock()) {
			auto vnode = dynamic_pointer_cast<XValueNodeBase>(node);
			assert(vnode);
			return XRuby::getValueOfNode(vnode);
		}
		else {
			throw XString("Node no longer exists\n");
		}
	}
	catch (XString &s) {
		snprintf(errmsg, 256, "%s", s.c_str());
	}
	rb_raise(rb_eRuntimeError, "%s", errmsg);
	return Qnil;  
}
VALUE
XRuby::rvaluenode_to_str(VALUE self) {
	struct rnode_ptr *st;
	Data_Get_Struct(self, struct rnode_ptr, st);
	char errmsg[256];
	try {
		if(shared_ptr<XNode> node = st->ptr.lock()) {
			auto vnode = dynamic_pointer_cast<XValueNodeBase>(node);
			assert(vnode);
			return string2RSTRING(( **vnode)->to_str());
		}
		else {
			throw XString("Node no longer exists\n");
		}
	}
	catch (XString &s) {
		snprintf(errmsg, 256, "%s", s.c_str());
	}
	rb_raise(rb_eRuntimeError, "%s", errmsg);
	return Qnil;  
}

int
XRuby::strOnNode(const shared_ptr<XValueNodeBase> &node, VALUE value) {
	double dbl = 0;
	long integer = 0;

	auto dnode = dynamic_pointer_cast<XDoubleNode>(node);
	auto inode = dynamic_pointer_cast<XIntNode>(node);
	auto uinode = dynamic_pointer_cast<XUIntNode>(node);
	auto lnode = dynamic_pointer_cast<XLongNode>(node);
	auto ulnode = dynamic_pointer_cast<XULongNode>(node);
	auto bnode = dynamic_pointer_cast<XBoolNode>(node);

	switch (TYPE(value)) {
	case T_FIXNUM:
		integer = FIX2LONG(value);
		if(uinode || ulnode) {
			if(integer < 0) {
				throw formatString("Negative FIXNUM on %s\n", node->getName().c_str());
			}
		}
		if(inode) { trans( *inode) = integer; return 0; }
		if(lnode) { trans( *lnode) = integer; return 0; }
		if(uinode) { trans( *uinode) = integer; return 0; }
		if(ulnode) { trans( *ulnode) = integer; return 0; }
		dbl = integer;
		if(dnode) { trans( *dnode) = dbl; return 0;}
		throw formatString("FIXNUM is not appropreate on %s\n", node->getName().c_str());
	case T_FLOAT:
		dbl = RFLOAT(value)->value;
		if(dnode) { trans( *dnode) = dbl; return 0;}
		throw formatString("FLOAT is not appropreate on %s\n", node->getName().c_str());
	case T_BIGNUM:
		dbl = NUM2DBL(value);
		if(dnode) { trans( *dnode) = dbl; return 0;}
		throw formatString("BIGNUM is not appropreate on %s\n", node->getName().c_str());
		//    integer = lrint(dbl);
		//    if(inode && (dbl <= INT_MAX)) {inode->value(integer); return 0;}
	case T_STRING:
		try {
			trans( *node).str(XString(RSTRING(value)->ptr));
		}
		catch (XKameError &e) {
			throw formatString("Validation error %s on %s\n"
				, (const char*)e.msg().c_str(), node->getName().c_str());
		}
		return 0;
	case T_TRUE:
		if(bnode) { trans( *bnode) = true; return 0;}
		throw formatString("TRUE is not appropreate on %s\n", node->getName().c_str());
	case T_FALSE:
		if(bnode) { trans( *bnode) = false; return 0;}
		throw formatString("FALSE is not appropreate on %s\n", node->getName().c_str());
	default:
		throw formatString("UNKNOWN TYPE is not appropreate on %s\n", node->getName().c_str());
	}
	return -1;
}
VALUE
XRuby::getValueOfNode(const shared_ptr<XValueNodeBase> &node) {
	auto dnode = dynamic_pointer_cast<XDoubleNode>(node);
	auto inode = dynamic_pointer_cast<XIntNode>(node);
	auto uinode = dynamic_pointer_cast<XUIntNode>(node);
	auto lnode = dynamic_pointer_cast<XLongNode>(node);
	auto ulnode = dynamic_pointer_cast<XULongNode>(node);
	auto bnode = dynamic_pointer_cast<XBoolNode>(node);
	auto snode = dynamic_pointer_cast<XStringNode>(node);
	Snapshot shot( *node);
	if(dnode) {return rb_float_new((double)shot[ *dnode]);}
	if(inode) {return INT2NUM(shot[ *inode]);}
	if(uinode) {return UINT2NUM(shot[ *uinode]);}
	if(lnode) {return LONG2NUM(shot[ *lnode]);}
	if(ulnode) {return ULONG2NUM(shot[ *ulnode]);}
	if(bnode) {return (shot[ *bnode]) ? Qtrue : Qfalse;}
	if(snode) {return string2RSTRING(shot[ *snode]);}
	return Qnil;
}

shared_ptr<XRubyThread>
XRuby::findRubyThread(VALUE self, VALUE threadid)
{
	long id = NUM2LONG(threadid);
	struct rnode_ptr *st;
	Data_Get_Struct(self, struct rnode_ptr, st);
	shared_ptr<XRubyThread> rubythread;
	Snapshot shot(*st->xruby);
	if(shot.size()) {
		for(XNode::const_iterator it = shot.list()->begin(); it != shot.list()->end(); ++it) {
			auto th = dynamic_pointer_cast<XRubyThread>( *it);
			assert(th);
			if(id == shot[ *th->threadID()])
				rubythread = th;
		}
	}
	return rubythread;
}
VALUE
XRuby::my_rbdefout(VALUE self, VALUE str, VALUE threadid) {
	shared_ptr<XString> sstr(new XString(RSTRING(str)->ptr));
	shared_ptr<XRubyThread> rubythread(findRubyThread(self, threadid));
	if(rubythread) {
		Snapshot shot( *rubythread);
		shot.talk(shot[ *rubythread].onMessageOut(), sstr);
		dbgPrint(QString("Ruby [%1]; %2").arg(shot[ *rubythread->filename()].to_str()).arg( *sstr));
	}
	else {
		dbgPrint(QString("Ruby [global]; %1").arg(*sstr));
	}
	return Qnil;
}
VALUE
XRuby::my_rbdefin(VALUE self, VALUE threadid) {
	shared_ptr<XRubyThread> rubythread(findRubyThread(self, threadid));
	char errmsg[256];
	try {
		if(rubythread) {
			XString line = rubythread->gets();
			if(line.length())
				return string2RSTRING(line);
			return Qnil;  
		}
		else {
			throw XString("UNKNOWN Ruby thread\n");
		}
	}
	catch (XString &s) {
		snprintf(errmsg, 256, "%s", s.c_str());
	}
	rb_raise(rb_eRuntimeError, "%s", errmsg);
	return Qnil;  
}
VALUE
XRuby::is_main_terminated(VALUE self) {
	struct rnode_ptr *st;
	Data_Get_Struct(self, struct rnode_ptr, st);
	return (st->xruby->m_thread.isTerminated()) ? Qtrue : Qfalse;
}

void *
XRuby::execute(const atomic<bool> &terminated) {
	while ( !terminated) {
		ruby_init();
		ruby_script("KAME");

		ruby_init_loadpath();

		rbClassNode = rb_define_class("XNode", rb_cObject);
		rb_global_variable(&rbClassNode);
		typedef VALUE(*fp)(...);
		rb_define_method(rbClassNode, "name", (fp)rnode_name, 0);
		rb_define_method(rbClassNode, "touch", (fp)rnode_touch, 0);
		rb_define_method(rbClassNode, "child", (fp)rnode_child, 1);
		//      rb_define_method(rbClassNode, "[]", (fp)rnode_child, 1);
		rb_define_method(rbClassNode, "count", (fp)rnode_count, 0);
		rbClassValueNode = rb_define_class("XValueNode", rbClassNode);
		rb_global_variable(&rbClassValueNode);
		rb_define_method(rbClassValueNode, "internal_set", (fp)rvaluenode_set, 1);
		rb_define_method(rbClassValueNode, "internal_load", (fp)rvaluenode_load, 1);
		rb_define_method(rbClassValueNode, "internal_get", (fp)rvaluenode_get, 0);
		rb_define_method(rbClassValueNode, "to_str", (fp)rvaluenode_to_str, 0);
		rbClassListNode = rb_define_class("XListNode", rbClassNode);
		rb_global_variable(&rbClassListNode);
		rb_define_method(rbClassListNode, "internal_create", (fp)rlistnode_create_child, 2);
		rb_define_method(rbClassListNode, "release", (fp)rlistnode_release_child, 1);

		{
			shared_ptr<XMeasure> measure = m_measure.lock();
			assert(measure);
			XString name = measure->getName();
			name[0] = toupper(name[0]);
			VALUE rbRootNode = rnode_create(measure, this);
			rb_define_global_const(name.c_str(), rbRootNode);
			rb_define_global_const("RootNode", rbRootNode);
		}
		{
			VALUE rbRubyThreads = rnode_create(shared_from_this(), this);
			rb_define_singleton_method(rbRubyThreads, "my_rbdefout", (fp)my_rbdefout, 2);
			rb_define_singleton_method(rbRubyThreads, "my_rbdefin", (fp)my_rbdefin, 1);
			rb_define_singleton_method(rbRubyThreads, "is_main_terminated", (fp)is_main_terminated, 0);
			rb_define_global_const("XRubyThreads", rbRubyThreads);
		}

		{
			int state = 0;
			QString filename = KStandardDirs::locate("appdata", XRUBYSUPPORT_RB);
			if(filename.isEmpty()) {
				g_statusPrinter->printError("No KAME ruby support file installed.");
			}
			else {
				fprintf(stderr, "Loading ruby scripting monitor:%s\n", filename.toLatin1().data());
				rb_load_protect (string2RSTRING(filename), 0, &state);
				if(state) {
					fprintf(stderr, "Ruby, exception(s) occurred\n");
				}
			}
		}
		ruby_finalize();

		fprintf(stderr, "ruby finished\n");

	}
	return NULL;
}
