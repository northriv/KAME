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
#include "xitemnode.h"

XItemNodeBase::XItemNodeBase(const char *name, bool runtime, bool auto_set_any) : 
    XValueNodeBase(name, runtime) {
	if(auto_set_any) {
		for(Transaction tr( *this);; ++tr) {
			m_lsnTryAutoSet = tr[ *this].onListChanged().connect( *this,
				&XItemNodeBase::onTryAutoSet);
			if(tr.commit())
				break;
		}
	}
}
void
XItemNodeBase::onTryAutoSet(const Snapshot &shot, const Payload::ListChangeEvent &e) {
	if( !autoSetAny()) return;
	XString var = shot[ *this].to_str();
	if(var.length()) return;
	shared_ptr<const std::deque<Item> > items = itemStrings(e.shot_of_list);
	if(items->size()) {
		trans( *this).str(items->front().name);
	}
}

void
xpointeritemnode_throwConversionError_() {
	throw XKameError(i18n("No item."), __FILE__, __LINE__);
}

XComboNode::XComboNode(const char *name, bool runtime, bool auto_set_any)
	: XItemNodeBase(name, runtime, auto_set_any) {
}

void
XComboNode::Payload::str_(const XString &var) {
	*this = var;
}

XComboNode::Payload&
XComboNode::Payload::operator=(const XString &var) {
	int i = -1;
	if(var.length()) {
		for(i = 0; i < m_strings->size(); ++i) {
			if(m_strings->at(i) == var) {
				break;
			}
		}
	}
	if(i == m_strings->size())
		i = -1;
	m_var = std::pair<XString, int>(var, i);
    tr().mark(onValueChanged(), static_cast<XValueNodeBase*>( &node()));
	return *this;
}

XComboNode::Payload&
XComboNode::Payload::operator=(int t) {
    if((t >= 0) && (t < (int)m_strings->size()))
	    m_var = std::pair<XString, int>(m_strings->at(t), t);
	else
	    m_var = std::pair<XString, int>("", -1);
    tr().mark(onValueChanged(), static_cast<XValueNodeBase*>( &node()));
	return *this;
}

void
XComboNode::Payload::add(const XString &str) {
	m_strings.reset(new std::deque<XString>( *m_strings));
	m_strings->push_back(str);
	ListChangeEvent e(tr(), static_cast<XItemNodeBase*>( &node()));
	tr().mark(onListChanged(), e);
	if(str == m_var.first) {
		m_var.second = m_strings->size() - 1;
	    tr().mark(onValueChanged(), static_cast<XValueNodeBase*>( &node()));
	}
}

void
XComboNode::Payload::clear() {
	m_strings.reset(new std::deque<XString>( *m_strings));
    m_strings->clear();
	ListChangeEvent e(tr(), static_cast<XItemNodeBase*>( &node()));
	tr().mark(onListChanged(), e);
	if(m_var.second >= 0) {
	    m_var.second = -1;
	    tr().mark(onValueChanged(), static_cast<XValueNodeBase*>( &node()));
	}
}

shared_ptr<const std::deque<XItemNodeBase::Item> >
XComboNode::Payload::itemStrings() const {
    shared_ptr<std::deque<XItemNodeBase::Item> > items(new std::deque<XItemNodeBase::Item>());
	for(std::deque<XString>::const_iterator it = m_strings->begin(); it != m_strings->end(); it++) {
		XItemNodeBase::Item item;
		item.name = *it;
		item.label = *it;
		items->push_back(item);
	}
	ASSERT(m_strings->size() || (m_var.second < 0));
    if(m_var.second < 0) {
	    XItemNodeBase::Item item;
        item.name = m_var.first;
        if(item.name.length()) {
	        item.label = formatString("(%s)", item.name.c_str());
	    	items->push_back(item);
        }
    }
    return items;
}
