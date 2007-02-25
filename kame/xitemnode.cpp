/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#include "xitemnode.h"
#include <klocale.h>

XItemNodeBase::XItemNodeBase(const char *name, bool runtime, bool auto_set_any) : 
    XValueNodeBase(name, runtime)
{
	if(auto_set_any) {
	    m_lsnTryAutoSet = onListChanged().connectWeak(
	        false, shared_from_this(), 
	        &XItemNodeBase::onTryAutoSet);
	}
}
void
XItemNodeBase::onTryAutoSet(const shared_ptr<XItemNodeBase>& node) {
	if(!autoSetAny()) return;
	std::string var = to_str();
	if(var.length()) return;
	shared_ptr<const std::deque<Item> > items = itemStrings();
	if(items->size()) {
		str(items->front().name);
	}
}

void
_xpointeritemnode_throwConversionError() {
   throw XKameError(KAME::i18n("No item."), __FILE__, __LINE__);
}

XComboNode::XComboNode(const char *name, bool runtime, bool auto_set_any)
   : XItemNodeBase(name, runtime, auto_set_any),
    m_strings(new std::deque<std::string>()),
    m_var(-1) {
}

void
XComboNode::_str(const std::string &var) throw (XKameError &)
{
  if(var.empty()) {
        value(-1);
        return;
  }
  atomic_shared_ptr<const std::deque<std::string> > strings(m_strings);
  unsigned int i = 0;
  for(std::deque<std::string>::const_iterator it = strings->begin(); it != strings->end(); it++) {
        if(*it == var) {
            value(i);
            return;
        }
        i++;
   }
   throw XKameError(KAME::i18n("No item."), __FILE__, __LINE__);
}

void
XComboNode::value(const std::string &s)
{
    try {
        str(s);
    }
    catch (XKameError &e) {
        e.print();
    }
}

XComboNode::operator int() const {
    return m_var;
}
std::string
XComboNode::to_str() const
{
    int i = m_var;
    atomic_shared_ptr<const std::deque<std::string> > strings(m_strings);
    if((i >= 0) && (i < (int)strings->size()))
        return strings->at(i);
    else
        return std::string();
}

void
XComboNode::add(const std::string &str)
{
    for(;;) {
      atomic_shared_ptr<std::deque<std::string> > old_strings(m_strings);
      atomic_shared_ptr<std::deque<std::string> > new_strings(new std::deque<std::string>(*old_strings));
      new_strings->push_back(str);
      if(new_strings.compareAndSwap(old_strings, m_strings))
        break;
    }
    onListChanged().talk(dynamic_pointer_cast<XItemNodeBase>(shared_from_this()));
}

void
XComboNode::clear()
{
	value(-1);
    m_strings.reset(new std::deque<std::string>());
    onListChanged().talk(dynamic_pointer_cast<XItemNodeBase>(shared_from_this()));
}

shared_ptr<const std::deque<XItemNodeBase::Item> >
XComboNode::itemStrings() const
{
    shared_ptr<std::deque<XItemNodeBase::Item> > items(new std::deque<XItemNodeBase::Item>());
    atomic_shared_ptr<const std::deque<std::string> > strings(m_strings);
    for(std::deque<std::string>::const_iterator it = strings->begin(); it != strings->end(); it++) {
    XItemNodeBase::Item item;
        item.name = *it;
        item.label = *it;
        items->push_back(item);
    }
    return items;
}

void
XComboNode::value(int t) {
    shared_ptr<XValueNodeBase> ptr = 
        dynamic_pointer_cast<XValueNodeBase>(shared_from_this());
    XScopedLock<XRecursiveMutex> lock(m_write_mutex);
    m_tlkBeforeValueChanged.talk(ptr);
    m_var = t;
    m_tlkOnValueChanged.talk(ptr);
}
