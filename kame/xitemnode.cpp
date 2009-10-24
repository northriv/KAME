/***************************************************************************
		Copyright (C) 2002-2009 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "xitemnode.h"


XItemNodeBase::XItemNodeBase(const char *name, bool runtime, bool auto_set_any) : 
    XValueNodeBase(name, runtime)
{
	if(auto_set_any) {
	    m_lsnTryAutoSet = onListChanged().connectWeak(
	        shared_from_this(), 
	        &XItemNodeBase::onTryAutoSet);
	}
}
void
XItemNodeBase::onTryAutoSet(const shared_ptr<XItemNodeBase>&) {
	if(!autoSetAny()) return;
	XString var = to_str();
	if(var.length()) return;
	shared_ptr<const std::deque<Item> > items = itemStrings();
	if(items->size()) {
		str(items->front().name);
	}
}

void
_xpointeritemnode_throwConversionError() {
	throw XKameError(i18n("No item."), __FILE__, __LINE__);
}

XComboNode::XComboNode(const char *name, bool runtime, bool auto_set_any)
	: XItemNodeBase(name, runtime, auto_set_any),
	  m_var(new std::pair<XString, int>("", -1)) {
}

void
XComboNode::_str(const XString &var) throw (XKameError &)
{
    shared_ptr<XValueNodeBase> ptr = 
        dynamic_pointer_cast<XValueNodeBase>(shared_from_this());
    XScopedLock<XRecursiveMutex> lock(m_write_mutex);
    m_tlkBeforeValueChanged.talk(ptr);
	if(var.length()) {
		atomic_list<XString>::reader strings(m_strings);
		if(strings) {
			unsigned int i = 0;
			for(atomic_list<XString>::const_iterator it = strings->begin(); it != strings->end(); it++) {
				if(*it == var) {
					m_var.reset(new std::pair<XString, int>(var, i));
					m_tlkOnValueChanged.talk(ptr);
					return;
				}
				i++;
			}
		}
	}
	m_var.reset(new std::pair<XString, int>(var, -1));
	m_tlkOnValueChanged.talk(ptr);
}

void
XComboNode::value(const XString &s)
{
    try {
        str(s);
    }
    catch (XKameError &e) {
        e.print();
    }
}

XComboNode::operator int() const {
	atomic_shared_ptr<std::pair<XString, int> > var(m_var);
    return var->second;
}
XString
XComboNode::to_str() const {
	atomic_shared_ptr<std::pair<XString, int> > var(m_var);
	return var->first;
}

void
XComboNode::add(const XString &str)
{
	m_strings.push_back(str);
    onListChanged().talk(dynamic_pointer_cast<XItemNodeBase>(shared_from_this()));
    XString var = to_str();
    if(var == str) {
		value(str);
		onListChanged().talk(dynamic_pointer_cast<XItemNodeBase>(shared_from_this()));
    }
}

void
XComboNode::clear()
{
    m_strings.reset();
    onListChanged().talk(dynamic_pointer_cast<XItemNodeBase>(shared_from_this()));
	value(to_str());
}

shared_ptr<const std::deque<XItemNodeBase::Item> >
XComboNode::itemStrings() const
{
    shared_ptr<std::deque<XItemNodeBase::Item> > items(new std::deque<XItemNodeBase::Item>());
	atomic_list<XString>::reader strings(m_strings);
	if(strings) {
		for(atomic_list<XString>::const_iterator it = strings->begin(); it != strings->end(); it++) {
			XItemNodeBase::Item item;
			item.name = *it;
			item.label = *it;
			items->push_back(item);
		}
    }
    if(*this < 0) {
	    XItemNodeBase::Item item;
        item.name = to_str();
        if(item.name.length()) {
	        item.label = formatString("(%s)", item.name.c_str());
	    	items->push_back(item);
        }
    }
    return items;
}

void
XComboNode::value(int t) {
    shared_ptr<XValueNodeBase> ptr = 
        dynamic_pointer_cast<XValueNodeBase>(shared_from_this());
    XScopedLock<XRecursiveMutex> lock(m_write_mutex);
    m_tlkBeforeValueChanged.talk(ptr);
    atomic_list<XString>::reader strings(m_strings);
    if(strings && (t >= 0) && (t < (int)strings->size()))
	    m_var.reset(new std::pair<XString, int>(strings->at(t), t));
	else
	    m_var.reset(new std::pair<XString, int>("", -1));
    m_tlkOnValueChanged.talk(ptr);
}
