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
#ifndef XITEMNODE_H_
#define XITEMNODE_H_

#include "xnode.h"
#include "xlistnode.h"

//! Posses a pointer to a member of a list
class XItemNodeBase : public XValueNodeBase {
public:
	explicit XItemNodeBase(const char *name, bool runtime = false, bool auto_set_any = false);
	virtual ~XItemNodeBase() {}
  
	struct Item { XString name, label; };
	virtual shared_ptr<const std::deque<Item> > itemStrings(const Snapshot &shot_of_list) const = 0;

	bool autoSetAny() const {return m_lsnTryAutoSet;}

	struct Payload : public XValueNodeBase::Payload {
		Payload() : XValueNodeBase::Payload() {}
		Talker<XItemNodeBase*, XItemNodeBase*> &onListChanged() {return m_tlkOnListChanged;}
		const Talker<XItemNodeBase*, XItemNodeBase*> &onListChanged() const {return m_tlkOnListChanged;}
	private:
		TalkerSingleton<XItemNodeBase*, XItemNodeBase*> m_tlkOnListChanged;
	};
private:
	shared_ptr<XListener> m_lsnTryAutoSet;
	void onTryAutoSet(const Snapshot &shot, XItemNodeBase *);
};

void
_xpointeritemnode_throwConversionError();

template <class TL>
class XPointerItemNode : public XItemNodeBase {
public:
	XPointerItemNode(const char *name, bool runtime, Transaction &tr_list,
		const shared_ptr<TL> &list, bool auto_set_any = false)
		:  XItemNodeBase(name, runtime, auto_set_any), m_list(list) {
		m_lsnOnItemReleased = tr_list[ *list].onRelease().connect(*this, &XPointerItemNode<TL>::onItemReleased);
		m_lsnOnListChanged = tr_list[ *list].onListChanged().connect(*this, &XPointerItemNode<TL>::lsnOnListChanged);
    }
	virtual ~XPointerItemNode() {}

	operator shared_ptr<XNode>() const { return **this;}
	void value(const shared_ptr<XNode> &x);

	struct Payload : public XItemNodeBase::Payload {
		Payload() : XItemNodeBase::Payload() {}
		operator shared_ptr<XNode>() const { return m_var.lock();}
		virtual XString to_str() const {
			shared_ptr<XNode> node(*this);
			if(node)
				return node->getName();
			else
				return XString();
		}
		Payload &operator=(const shared_ptr<XNode> &t) {
			m_var = t;
		    tr().mark(onValueChanged(), static_cast<XValueNodeBase*>(&node()));
		    return *this;
		}
	protected:
		virtual void _str(const XString &var) {
			if(var.empty()) {
				*this = shared_ptr<XNode>();
				return;
			}
			if(shared_ptr<TL> list = static_cast<const XPointerItemNode&>(node()).m_list.lock()) {
				Snapshot shot(*list);
				if(shot.size()) {
					for(NodeList::const_iterator it = shot.list()->begin(); it != shot.list()->end(); ++it) {
						if((*it)->getName() == var) {
							*this = *it;
							return;
						}
					}
				}
			}
			_xpointeritemnode_throwConversionError();
		}
		weak_ptr<XNode> m_var;
	};
private:  
	void onItemReleased(const Snapshot& shot, const XListNodeBase::Payload::ReleaseEvent &e) {
		for(Snapshot shot(*this);;) {
			if(e.released != (shared_ptr<XNode>)shot[*this])
				break;
			Transaction tr(shot);
			tr[ *this] = shared_ptr<XNode>();
			if(tr.commit()) break;
		}
	}
	void lsnOnListChanged(const Snapshot& shot, XListNodeBase* node) {
		if(shared_ptr<TL> list = m_list.lock()) {
			ASSERT(node == list.get());
			shot.talk(( **this)->onListChanged(), this);
		}
	}
	shared_ptr<XListener> m_lsnOnItemReleased, m_lsnOnListChanged;
protected:
	weak_ptr<TL> m_list;
};
//! A pointer to a XListNode TL, T1 is value type
template <class TL, class T1>
class _XItemNode : public XPointerItemNode<TL> {
public:
	_XItemNode(const char *name, bool runtime, Transaction &tr_list,
		const shared_ptr<TL> &list, bool auto_set_any = false)
		:  XPointerItemNode<TL>(name, runtime, tr_list, list, auto_set_any) {
	}
	virtual ~_XItemNode() {}
	operator shared_ptr<T1>() const { return **this;}
	struct Payload : public XPointerItemNode<TL>::Payload {
		Payload() : XPointerItemNode<TL>::Payload() {}
		operator shared_ptr<T1>() const {
	        return dynamic_pointer_cast<T1>(shared_ptr<XNode>(*this));
		}
		Payload &operator=(const shared_ptr<XNode> &t) {
			XPointerItemNode<TL>::Payload::operator=(t);
			return *this;
		}
	};
};
//! A pointer to a XListNode TL, T is value type
template <class TL, class T1, class T2 = T1>
class XItemNode : public _XItemNode<TL, T1> {
public:
	XItemNode(const char *name, bool runtime, Transaction &tr_list,
		const shared_ptr<TL> &list, bool auto_set_any = false)
		:  _XItemNode<TL, T1>(name, runtime, tr_list, list, auto_set_any) {
	}
	virtual ~XItemNode() {}
	operator shared_ptr<T2>() const { return **this;}
	struct Payload : public _XItemNode<TL, T1>::Payload {
		Payload() : _XItemNode<TL, T1>::Payload() {}
		operator shared_ptr<T2>() const { return dynamic_pointer_cast<T2>((shared_ptr<XNode>)*this);}
		Payload &operator=(const shared_ptr<XNode> &t) {
			_XItemNode<TL, T1>::Payload::operator=(t);
			return *this;
		}
	};
	virtual shared_ptr<const std::deque<XItemNodeBase::Item> > itemStrings(const Snapshot &shot) const {
		shared_ptr<std::deque<XItemNodeBase::Item> > items(new std::deque<XItemNodeBase::Item>());
		if(shared_ptr<TL> list = this->m_list.lock()) {
			if(shot.size(list)) {
				for(typename XNode::const_iterator it = shot.list(list)->begin(); it != shot.list(list)->end(); ++it) {
					if(dynamic_pointer_cast<T1>(*it) || dynamic_pointer_cast<T2>(*it)) {
						XItemNodeBase::Item item;
						item.name = (*it)->getName();
						item.label = (*it)->getLabel();
						items->push_back(item);
					}
				}
			}
		}
		return items;
	}
};

//! Contains strings, value is one of strings
class XComboNode : public XItemNodeBase {
public:
	explicit XComboNode(const char *name, bool runtime = false, bool auto_set_any = false);
	virtual ~XComboNode() {}
  
	void add(const XString &str);
	void clear();
	operator int() const  { return **this;}
	void value(int x);
	void value(const XString &x);

	virtual shared_ptr<const std::deque<XItemNodeBase::Item> > itemStrings(const Snapshot &shot) const {
		return shot[ *this].itemStrings();
	}

	struct Payload : public XItemNodeBase::Payload {
		Payload() : XItemNodeBase::Payload(), m_strings(new std::deque<XString>),
			m_var(std::pair<XString, int>("", -1)) {}
		void add(const XString &str);
		void clear();
		operator int() const { return m_var.second;}
		virtual XString to_str() const { return m_var.first;}
		Payload &operator=(int t);
		Payload &operator=(const XString &);
		virtual shared_ptr<const std::deque<XItemNodeBase::Item> > itemStrings() const;
	protected:
		virtual void _str(const XString &);
	private:
		shared_ptr<std::deque<XString> > m_strings;
		std::pair<XString, int> m_var;
	};
};

template <class TL>
void
XPointerItemNode<TL>::value(const shared_ptr<XNode> &t) {
    if(this->beforeValueChanged().empty() && this->onValueChanged().empty()) {
        trans(*this) = t;
    }
    else {
		shared_ptr<XValueNodeBase> ptr =
			dynamic_pointer_cast<XValueNodeBase>(this->shared_from_this());
        XScopedLock<XRecursiveMutex> lock(this->m_talker_mutex);
        this->beforeValueChanged().talk(ptr);
        trans(*this) = t;
        this->onValueChanged().talk(ptr);
    }
}

#endif /*XITEMNODE_H_*/
