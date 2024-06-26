/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
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
        iterate_commit([=](Transaction &tr){
			m_lsnTryAutoSet = tr[ *this].onListChanged().connect( *this,
				&XItemNodeBase::onTryAutoSet);
        });
	}
}
void
XItemNodeBase::onTryAutoSet(const Snapshot &shot, const Payload::ListChangeEvent &e) {
	if( !autoSetAny()) return;
	XString var = shot[ *this].to_str();
	if(var.length()) return;
    auto items = itemStrings(e.shot_of_list);
    if(items.size()) {
        trans( *this).str(items.front().label);
	}
}

void
xpointeritemnode_throwConversionError_() {
    throw XKameError(i18n_noncontext("No item."), __FILE__, __LINE__);
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
    m_var = {var, i};
    tr().mark(onValueChanged(), static_cast<XValueNodeBase*>( &node()));
	return *this;
}

XComboNode::Payload&
XComboNode::Payload::operator=(int t) {
    if((t >= 0) && (t < (int)m_strings->size()))
        m_var = {m_strings->at(t), t};
	else
        m_var = {"", -1};
    tr().mark(onValueChanged(), static_cast<XValueNodeBase*>( &node()));
	return *this;
}

void
XComboNode::Payload::add(const XString &str) {
    m_strings = std::make_shared<std::deque<XString>>( *m_strings);
	m_strings->push_back(str);
    tr().mark(onListChanged(), ListChangeEvent({tr(), static_cast<XItemNodeBase*>( &node())}));
	if(str == m_var.first) {
		m_var.second = m_strings->size() - 1;
	    tr().mark(onValueChanged(), static_cast<XValueNodeBase*>( &node()));
	}
}
void
XComboNode::Payload::add(const std::vector<Item> &items) {
    for(auto &&x: items)
        add(x.label);
}

void
XComboNode::Payload::clear() {
    m_strings = std::make_shared<std::deque<XString>>();
    tr().mark(onListChanged(), ListChangeEvent({tr(), static_cast<XItemNodeBase*>( &node())}));
	if(m_var.second >= 0) {
	    m_var.second = -1;
	    tr().mark(onValueChanged(), static_cast<XValueNodeBase*>( &node()));
	}
}

std::vector<XItemNodeBase::Item> XComboNode::Payload::itemStrings() const {
    std::vector<XItemNodeBase::Item> items;
	for(auto it = m_strings->begin(); it != m_strings->end(); it++) {
        items.push_back({*it, *it});
	}
	assert(m_strings->size() || (m_var.second < 0));
    if(m_var.second < 0) {
        if(m_var.first.length()) {
            items.push_back({m_var.first, "(" + m_var.first + ")"});
        }
    }
    return items;
}
