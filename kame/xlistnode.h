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
#ifndef XLISTNODE_H_
#define XLISTNODE_H_

#include "xnode.h"
#include <boost/function.hpp>
using boost::function;

class XListNodeBase : public XNode {
public:
	explicit XListNodeBase(const char *name, bool runtime = false);
	virtual ~XListNodeBase() {}

	//! Create a object, whose class is determined from \a type.
	//! Scripting only. Use XNode::create for coding instead.
	virtual shared_ptr<XNode> createByTypename(
        const XString &type, const XString &name) = 0;

	struct Payload : public XNode::Payload {
		Payload() : XNode::Payload() {}
		Talker<XListNodeBase*, XListNodeBase*> &onListChanged() {return m_tlkOnListChanged;}
		struct MoveEvent {
			unsigned int src_idx, dst_idx;
			XListNodeBase *emitter;
		};
		Talker<MoveEvent> &onMove() {return m_tlkOnMove;}
		const Talker<MoveEvent> &onMove() const {return m_tlkOnMove;}
		struct CatchEvent {
			XListNodeBase *emitter;
			shared_ptr<XNode> caught;
			int index;
		};
		Talker<CatchEvent> &onCatch() {return m_tlkOnCatch;}
		const Talker<CatchEvent> &onCatch() const {return m_tlkOnCatch;}
		struct ReleaseEvent {
			XListNodeBase *emitter;
			shared_ptr<XNode> released;
			int index;
		};
		Talker<ReleaseEvent> &onRelease() {return m_tlkOnRelease;}
		const Talker<ReleaseEvent> &onRelease() const {return m_tlkOnRelease;}
	private:
		TalkerSingleton<XListNodeBase*, XListNodeBase*> m_tlkOnListChanged;
		Talker<MoveEvent> m_tlkOnMove;
		Talker<CatchEvent> m_tlkOnCatch;
		Talker<ReleaseEvent> m_tlkOnRelease;
		virtual void catchEvent(const shared_ptr<XNode>&, int);
		virtual void releaseEvent(const shared_ptr<XNode>&, int);
		virtual void moveEvent(unsigned int src_idx, unsigned int dst_idx);
		virtual void listChangeEvent();
	};
protected:    
};

template <class NT>
class XListNode : public  XListNodeBase {
public:
	explicit XListNode(const char *name, bool runtime = false)
		:  XListNodeBase(name, runtime) {}
	virtual ~XListNode() {}

	virtual shared_ptr<XNode> createByTypename(
        const XString &, const XString &name) {
		return this->create<NT>(name.c_str(), false);
	}
};

//! creation by UI is not allowed.
template <class NT>
class XAliasListNode : public  XListNodeBase {
public:
	explicit XAliasListNode(const char *name, bool runtime = false)
		:  XListNodeBase(name, runtime) {}
	virtual ~XAliasListNode() {}

	virtual shared_ptr<XNode> createByTypename(
        const XString &, const XString &) {
		return shared_ptr<XNode>();
	}
};

template <class NT>
class XCustomTypeListNode : public  XListNodeBase {
public:
	explicit XCustomTypeListNode(const char *name, bool runtime = false)
		:  XListNodeBase(name, runtime) {}
	virtual ~XCustomTypeListNode() {}

	virtual shared_ptr<XNode> createByTypename(
        const XString &type, const XString &name) = 0;
};  

shared_ptr<XNode> empty_creator_(const char *, bool = false);
template <typename X>
shared_ptr<XNode> empty_creator_(const char *, bool, X) {
    return shared_ptr<XNode>();
}
template <typename X, typename Y>
shared_ptr<XNode> empty_creator_(const char *, bool, X, Y) {
    return shared_ptr<XNode>();
}
template <typename X, typename Y, typename Z>
shared_ptr<XNode> empty_creator_(const char *, bool, X, Y, Z) {
    return shared_ptr<XNode>();
}
template <typename X, typename Y, typename Z, typename ZZ>
shared_ptr<XNode> empty_creator_(const char *, bool, X, Y, Z, ZZ) {
    return shared_ptr<XNode>();
}
template <typename T>
shared_ptr<XNode> creator_(const char *name, bool runtime) {
    return XNode::createOrphan<T>(name, runtime);
}
template <typename T, typename X>
shared_ptr<XNode> creator_(const char *name, bool runtime, X x) {
    return XNode::createOrphan<T>(name, runtime, x);
}
template <typename T, typename X, typename Y>
shared_ptr<XNode> creator_(const char *name, bool runtime, X x, Y y) {
    return XNode::createOrphan<T>(name, runtime, x, y);
}
template <typename T, typename X, typename Y, typename Z>
shared_ptr<XNode> creator_(const char *name, bool runtime, X x, Y y, Z z) {
    return XNode::createOrphan<T>(name, runtime, x, y, z);
}
template <typename T, typename X, typename Y, typename Z, typename ZZ>
shared_ptr<XNode> creator_(const char *name, bool runtime, X x, Y y, Z z, ZZ zz) {
    return XNode::createOrphan<T>(name, runtime, x, y, z, zz);
}

//! Register typename and constructor.
//! make static member of TypeHolder<> in your class
//! After def. of static TypeHolder<>, define Creator to register.
//! call creator(type)(type, name, ...) to create children.
template <class tFunc>
struct XTypeHolder {
	XTypeHolder() {
            fprintf(stderr, "New typeholder\n");
	}
		
	template <class tChild>
	struct Creator {
		Creator(XTypeHolder &holder, const char *name, const char *label = 0L) {
			tFunc create_typed = creator_<tChild>;
			if( !label)
				label = name;
			if(std::find(holder.names.begin(), holder.names.end(), XString(name)) != holder.names.end()) {
				fprintf(stderr, "Duplicated name!\n");
				return;
			}
			holder.creators.push_back(create_typed);
			holder.names.push_back(XString(name));
			holder.labels.push_back(XString(label));
			fprintf(stderr, "%s %s\n", name, label);
		}
	};
	tFunc creator(const XString &tp) {
		for(unsigned int i = 0; i < names.size(); i++) {
            if(names[i] == tp) return creators[i];
		}
		return empty_creator_;
	}
	std::deque<tFunc> creators;
	std::deque<XString> names, labels;
};

#define DEFINE_TYPE_HOLDER_W_FUNC(tFunc__) \
  typedef XTypeHolder<tFunc__> TypeHolder; \
  static TypeHolder s_types; \
  static tFunc__ creator(const XString &tp) {return s_types.creator(tp);} \
  static std::deque<XString> &typenames() {return s_types.names;} \
  static std::deque<XString> &typelabels() {return s_types.labels;}

#define DEFINE_TYPE_HOLDER \
  typedef shared_ptr<XNode>(*fpCreateFunc)(const char *, bool \
	); \
  DEFINE_TYPE_HOLDER_W_FUNC(fpCreateFunc);

#define DEFINE_TYPE_HOLDER_EXTRA_PARAM(extra_param) \
  typedef shared_ptr<XNode>(*fpCreateFunc)(const char *, bool \
	, extra_param); \
  DEFINE_TYPE_HOLDER_W_FUNC(fpCreateFunc);

#define DEFINE_TYPE_HOLDER_EXTRA_PARAMS_2(par1, par2) \
  typedef shared_ptr<XNode>(*fpCreateFunc)(const char *, bool \
	, par1, par2); \
  DEFINE_TYPE_HOLDER_W_FUNC(fpCreateFunc);

#define DEFINE_TYPE_HOLDER_EXTRA_PARAMS_3(par1, par2, par3) \
  typedef shared_ptr<XNode>(*fpCreateFunc)(const char *, bool \
	, par1, par2, par3); \
  DEFINE_TYPE_HOLDER_W_FUNC(fpCreateFunc);
  
#define DEFINE_TYPE_HOLDER_EXTRA_PARAMS_4(par1, par2, par3, par4) \
  typedef shared_ptr<XNode>(*fpCreateFunc)(const char *, bool \
	, par1, par2, par3, par4); \
  DEFINE_TYPE_HOLDER_W_FUNC(fpCreateFunc);

#define DECLARE_TYPE_HOLDER(list) \
    list::TypeHolder list::s_types;

#define REGISTER_TYPE_2__(list, type, name, label) list::TypeHolder::Creator<type> \
    g_driver_type_ ## name(list::s_types, # name, label);
    
#define REGISTER_TYPE(list, type, label) REGISTER_TYPE_2__(list, X ## type, type, label)

class XStringList : public  XListNode<XStringNode> {
public:
	explicit XStringList(const char *name, bool runtime = false)
		:  XListNode<XStringNode>(name, runtime) {}
	virtual ~XStringList() {}
};

#endif /*XLISTNODE_H_*/
