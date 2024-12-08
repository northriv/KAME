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
#ifndef XLISTNODE_H_
#define XLISTNODE_H_

#include "xnode.h"
#include <functional>

class DECLSPEC_KAME XListNodeBase : public XNode {
public:
	explicit XListNodeBase(const char *name, bool runtime = false);
    virtual ~XListNodeBase() = default;

	//! Create a object, whose class is determined from \a type.
	//! Scripting only. Use XNode::create for coding instead.
	virtual shared_ptr<XNode> createByTypename(
        const XString &type, const XString &name) = 0;

	virtual bool isThreadSafeDuringCreationByTypename() const = 0;

    struct DECLSPEC_KAME Payload : public XNode::Payload {
        Talker<XListNodeBase*> &onListChanged() {return m_tlkOnListChanged;}
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
        TalkerOnce<XListNodeBase*> m_tlkOnListChanged;
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

//! List node for simples nodes, such like XIntNode.
template <class NT>
class XListNode : public  XListNodeBase {
public:
	explicit XListNode(const char *name, bool runtime = false)
		:  XListNodeBase(name, runtime) {}
    virtual ~XListNode() = default;

    virtual bool isThreadSafeDuringCreationByTypename() const override {return true;}

	virtual shared_ptr<XNode> createByTypename(
        const XString &, const XString &name) override {
		return this->create<NT>(name.c_str(), false);
	}
};

//! creation by UI is not allowed.
template <class NT>
class XAliasListNode : public  XListNodeBase {
public:
	explicit XAliasListNode(const char *name, bool runtime = false)
		:  XListNodeBase(name, runtime) {}
    virtual ~XAliasListNode() = default;

    virtual bool isThreadSafeDuringCreationByTypename() const override {return true;}

	virtual shared_ptr<XNode> createByTypename(
        const XString &, const XString &) override {
        return {};
	}
};

template <class NT>
class XCustomTypeListNode : public  XListNodeBase {
public:
	explicit XCustomTypeListNode(const char *name, bool runtime = false)
		:  XListNodeBase(name, runtime) {}
    virtual ~XCustomTypeListNode() = default;

    virtual bool isThreadSafeDuringCreationByTypename() const override {return false;} //! default behavior for safety.
};  

#include <functional>

//! Register typename and constructor.
//! make static member of TypeHolder<> in your class
//! After def. of static TypeHolder<>, define Creator to register.
//! call creator(type)(type, name, ...) to create children.
template <class... ArgTypes>
struct XTypeHolder {
    using creator_t = std::function<shared_ptr<XNode>(const char*, bool, ArgTypes&&...)>;

    XTypeHolder() {
            fprintf(stderr, "New typeholder\n");
	}
		
    creator_t creator(const std::string &tp) {
        XScopedLock<XMutex> lock(m_mutex);
        try {
            return m_map.at(tp).creator;
        }
        catch(std::out_of_range &e) {
            return [](const char*, bool, ArgTypes&&...){return shared_ptr<XNode>();}; //empty
        }
	}
    std::deque<XString> keys() const {
        std::deque<XString> list;
        for(auto &&x: m_map)
            list.push_back(x.first);
        return list;
    }
    std::deque<XString> labels() const {
        std::deque<XString> list;
        for(auto &&x: m_map)
            list.push_back(x.second.label);
        return list;
    }
    void insertCreator(const std::string &key, creator_t &&creator, const std::string &label) {
        XScopedLock<XMutex> lock(m_mutex);
        auto [it,ret] = m_map.insert(std::make_pair(key, XTypeHolder::CreatorInfo{creator, label}));
        if( !ret) {
            fprintf(stderr, "Duplicated key to typeholder.!\n");
            return;
        }
        fprintf(stderr, "%s %s\n", key.c_str(), label.c_str());
    }
    void eraseCreator(const std::string &key) {
        XScopedLock<XMutex> lock(m_mutex);
        m_map.erase(key);
    }
    //Default creator wrapping create<>.
    template <class tChild>
    struct Creator {
        Creator(XTypeHolder &holder, const char *name, const char *label = 0L) {
            creator_t create_typed =
                    [](const char *name, bool runtime, ArgTypes&&... args)->shared_ptr<XNode>
                    {return XNode::createOrphan<tChild>(name, runtime, std::forward<ArgTypes>(args)...);};
            if( !label)
                label = name;
            holder.insertCreator(name, std::move(create_typed), label);
        }
    };
    struct CreatorInfo {
        creator_t creator;
        XString label;
    };
    std::map<std::string, CreatorInfo> m_map;
    XMutex m_mutex;
};

#define DEFINE_TYPE_HOLDER(...) \
    typedef XTypeHolder<__VA_ARGS__> TypeHolder; \
    static TypeHolder s_types; \
    static TypeHolder::creator_t creator__(const XString &tp) {return s_types.creator(tp);} \
    virtual TypeHolder::creator_t creator(const XString &tp) {return creator__(tp);} \
    static std::deque<XString> typenames__() {return s_types.keys();} \
    static std::deque<XString> typelabels__() {return s_types.labels();} \
    virtual std::deque<XString> typenames() {return typenames__();} \
    virtual std::deque<XString> typelabels() {return typelabels__();}

#define DECLARE_TYPE_HOLDER(list) \
    list::TypeHolder list::s_types;

#define REGISTER_TYPE_2__(list, type, name, label) list::TypeHolder::Creator<type> \
    g_driver_type_ ## name(list::s_types, # name, label);
    
#define REGISTER_TYPE(list, type, label) REGISTER_TYPE_2__(list, X ## type, type, label)

class DECLSPEC_KAME XStringList : public  XListNode<XStringNode> {
public:
	explicit XStringList(const char *name, bool runtime = false)
		:  XListNode<XStringNode>(name, runtime) {}
    virtual ~XStringList() = default;
};

#endif /*XLISTNODE_H_*/
