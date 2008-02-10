/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef XITEMNODE_H_
#define XITEMNODE_H_

#include <xnode.h>
#include <xlistnode.h>

//! Posses a pointer to a member of a list
class XItemNodeBase : public XValueNodeBase
{
	XNODE_OBJECT
protected:
	explicit XItemNodeBase(const char *name, bool runtime = false, bool auto_set_any = false);
public:
	virtual ~XItemNodeBase() {}
  
	struct Item {
		std::string name, label;
	};
	virtual shared_ptr<const std::deque<Item> > itemStrings() const = 0;
	XTalker<shared_ptr<XItemNodeBase> >  &onListChanged() {return m_tlkOnListChanged;}
	bool autoSetAny() const {return m_lsnTryAutoSet;}
private:
	XTalker<shared_ptr<XItemNodeBase> > m_tlkOnListChanged;
	shared_ptr<XListener> m_lsnTryAutoSet;
	void onTryAutoSet(const shared_ptr<XItemNodeBase>& node);
};

void
_xpointeritemnode_throwConversionError();

template <class TL>
class XPointerItemNode : public XItemNodeBase
{
	XNODE_OBJECT
protected:
	XPointerItemNode(const char *name, bool runtime, const shared_ptr<TL> &list, bool auto_set_any = false)
		:  XItemNodeBase(name, runtime, auto_set_any)
		, m_var(new weak_ptr<XNode>()), m_list(list) {
		m_lsnOnItemReleased = list->onRelease().connectWeak(
			shared_from_this(), 
			&XPointerItemNode<TL>::onItemReleased);
		m_lsnOnListChanged = list->onListChanged().connectWeak(
			shared_from_this(), 
			&XPointerItemNode<TL>::lsnOnListChanged);
    }
public:
	virtual ~XPointerItemNode() {}

	virtual std::string to_str() const {
		shared_ptr<XNode> node(*this);
		if(node)
			return node->getName();
		else
			return std::string();
	}
	operator shared_ptr<XNode>() const {return m_var->lock();}
	virtual void value(const shared_ptr<XNode> &t) = 0;
protected:
	virtual void _str(const std::string &var) throw (XKameError &)
	{
		if(var.empty()) {
			value(shared_ptr<XNode>());
			return;
		}
		atomic_shared_ptr<const XNode::NodeList> children(m_list->children());
		if(children) { 
			for(NodeList::const_iterator it = children->begin(); it != children->end(); it++) {
				if((*it)->getName() == var) {
					value(*it);
					return;
				}
			}
		}
		_xpointeritemnode_throwConversionError();
	}
	atomic_shared_ptr<weak_ptr<XNode> > m_var;
	shared_ptr<TL> m_list;
	XRecursiveMutex m_write_mutex;
private:  
	void onItemReleased(const shared_ptr<XNode>& node)
	{
		XScopedLock<XRecursiveMutex> lock(m_write_mutex);
		if(node == m_var->lock())
			value(shared_ptr<XNode>());
	}
	void lsnOnListChanged(const shared_ptr<XListNodeBase>&)
	{
		onListChanged().talk(dynamic_pointer_cast<XItemNodeBase>(shared_from_this()));
	}
	shared_ptr<XListener> m_lsnOnItemReleased, m_lsnOnListChanged;
};
//! A pointer to a XListNode TL, T is value type
template <class TL, class T1>
class _XItemNode : public XPointerItemNode<TL>
{
	XNODE_OBJECT
protected:
	_XItemNode(const char *name, bool runtime, const shared_ptr<TL> &list, bool auto_set_any = false)
		:  XPointerItemNode<TL>(name, runtime, list, auto_set_any) {
	}
public:
	virtual ~_XItemNode() {}
	operator shared_ptr<T1>() const {
        return dynamic_pointer_cast<T1>(this->m_var->lock());
	}
	virtual void value(const shared_ptr<XNode> &t) {
		shared_ptr<XValueNodeBase> ptr = 
			dynamic_pointer_cast<XValueNodeBase>(this->shared_from_this());
		XScopedLock<XRecursiveMutex> lock(this->m_write_mutex);
		this->m_tlkBeforeValueChanged.talk(ptr);
		this->m_var.reset(new weak_ptr<XNode>(t));
		this->m_tlkOnValueChanged.talk(ptr); //, 1, &statusmutex);
	}
};
//! A pointer to a XListNode TL, T is value type
template <class TL, class T1, class T2 = T1>
class XItemNode : public _XItemNode<TL, T1>
{
	XNODE_OBJECT
protected:
	XItemNode(const char *name, bool runtime, const shared_ptr<TL> &list, bool auto_set_any = false)
		:  _XItemNode<TL, T1>(name, runtime, list, auto_set_any) {
	}
public:
	virtual ~XItemNode() {}
	operator shared_ptr<T2>() const {
        return dynamic_pointer_cast<T2>(this->m_var->lock());
	}
	virtual shared_ptr<const std::deque<XItemNodeBase::Item> > itemStrings() const
	{
		shared_ptr<std::deque<XItemNodeBase::Item> > items(new std::deque<XItemNodeBase::Item>());
		atomic_shared_ptr<const XNode::NodeList> children(this->m_list->children());
		if(children) {
			for(XNode::NodeList::const_iterator it = children->begin(); it != children->end(); it++) {
				if(dynamic_pointer_cast<T1>(*it) || dynamic_pointer_cast<T2>(*it)) {
					XItemNodeBase::Item item;
					item.name = (*it)->getName();
					item.label = (*it)->getLabel();
					items->push_back(item);
				}
			}
		}
		return items;
	}
};

//! Contain strings, value is one of strings
class XComboNode : public XItemNodeBase
{
	XNODE_OBJECT
protected:
	explicit XComboNode(const char *name, bool runtime = false, bool auto_set_any = false);
public:
	virtual ~XComboNode() {}
  
	virtual std::string to_str() const;
	virtual void add(const std::string &str);
	virtual void clear();
	virtual operator int() const;
	virtual void value(int t);
	virtual void value(const std::string &);
	virtual shared_ptr<const std::deque<XItemNodeBase::Item> > itemStrings() const;
protected:
	virtual void _str(const std::string &value) throw (XKameError &);
private:
	atomic_shared_ptr<std::deque<std::string> > m_strings;
	atomic_shared_ptr<std::pair<std::string, int> > m_var;
	XRecursiveMutex m_write_mutex;
};

#endif /*XITEMNODE_H_*/
