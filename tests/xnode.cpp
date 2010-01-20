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
#include "xnode.h"
#include <typeinfo>

XThreadLocal<std::deque<shared_ptr<XNode> > > XNode::stl_thisCreating;

XNode::XNode(const char *name, bool runtime)
	: m_name(name ? name : "") {
	// temporaly shared_ptr to be able to use shared_from_this() in constructors
	XNode::stl_thisCreating->push_back(shared_ptr<XNode>(this));
	ASSERT(shared_from_this());
	trans(*this).setRuntime(runtime);
	dbgPrint(QString("xnode %1 is created., addr=0x%2, size=0x%3")
			 .arg(getName())
			 .arg((uintptr_t)this, 0, 16)
			 .arg((uintptr_t)sizeof(XNode), 0, 16));
}
XNode::~XNode() {
	dbgPrint(QString("xnode %1 is being deleted., addr=0x%2").arg(getName()).arg((uintptr_t)this, 0, 16));
}
XString
XNode::getName() const {
    return m_name;
}
XString
XNode::getTypename() const {
    XString name = typeid(*this).name();
    int i = name.find('X');
    ASSERT(i != std::string::npos);
    ASSERT(i + 1 < name.length());
    return name.substr(i + 1);
}
void
XNode::insert(const shared_ptr<XNode> &ptr) {
    ASSERT(ptr);
    if( ! ptr->m_parent.lock())
    	ptr->m_parent = shared_from_this();
    Node::insert(ptr);
}
void
XNode::disable() {
	trans(*this).disable();
    onUIEnabled().talk(shared_from_this());
}
void
XNode::setUIEnabled(bool v) {
	trans(*this).setUIEnabled(v);
    onUIEnabled().talk(shared_from_this());
}
void
XNode::touch() {
    onTouch().talk(shared_from_this());
}

void
XNode::clearChildren() {
    releaseAll();
}
int
XNode::releaseChild(const shared_ptr<XNode> &node) {
	for(;;) {
		Snapshot shot(*this);
		shared_ptr<const NodeList> list(shot.list());
        NodeList::const_iterator it = find(list->begin(), list->end(), node);
        if(it == list->end())
        	return -1;
		if(release(shot, node))
			return 0;
	}
}

shared_ptr<XNode>
XNode::getChild(const XString &var) const {
	Snapshot shot(*this);
	shared_ptr<XNode> node;
	shared_ptr<const NodeList> list(shot.list());
	if(list) {
		for(NodeList::const_iterator it = list->begin(); it != list->end(); it++) {
			if(dynamic_pointer_cast<XNode>(*it)->getName() == var) {
                node = dynamic_pointer_cast<XNode>(*it);
                break;
			}
		}
	}
	return node;
}
shared_ptr<XNode>
XNode::getParent() const {
	return m_parent.lock();
}
