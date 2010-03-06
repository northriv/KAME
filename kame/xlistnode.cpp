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
#include "xlistnode.h"

XListNodeBase::XListNodeBase(const char *name, bool runtime) :
    XNode(name, runtime) {
}
void
XListNodeBase::clearChildren() {
	Transaction tr(*this);
	if( ! tr.size())
		return;
	for(;; ++tr) {
		if( !tr.size()) {
			onListChanged().talk(dynamic_pointer_cast<XListNodeBase>(shared_from_this()));
			return;
		}
		shared_ptr<XNode> node(tr.list()->back());
		if( !release(tr, node))
			continue;
		if(tr.commit()) {
			onRelease().talk(node);
			continue;
		}
	}
}
int
XListNodeBase::releaseChild(const shared_ptr<XNode> &node) {
	for(Transaction tr( *this);; ++tr) {
		NodeList::const_iterator it = std::find(tr.list()->begin(), tr.list()->end(), node);
		if(it == tr.list()->end())
			return -1;
		if( !release(tr, node))
			continue;
		if(tr.commit())
			break;
	}
    onRelease().talk(node);
    onListChanged().talk(dynamic_pointer_cast<XListNodeBase>(shared_from_this()));
    return 0;
}
void
XListNodeBase::insert(const shared_ptr<XNode> &ptr) {
	XNode::insert(ptr);
	onCatch().talk(ptr);
	onListChanged().talk(dynamic_pointer_cast<XListNodeBase>(shared_from_this()));
}
bool
XListNodeBase::insert(Transaction &tr, const shared_ptr<XNode> &ptr, bool online_after_insertion) {
	return XNode::insert(tr, ptr, online_after_insertion);
}
void
XListNodeBase::move(unsigned int src_idx, unsigned int dst_idx) {
	for(Transaction tr(*this);; ++tr) {
        if(src_idx >= tr.size())
        	return;
        if(dst_idx >= tr.size())
        	return;
        shared_ptr<XNode> snode = tr.list()->at(src_idx);
        shared_ptr<XNode> dnode = tr.list()->at(dst_idx);
        swap(tr, snode, dnode);
        if(tr.commit())
        	break;
    }
    MoveEvent e;
    e.src_idx = src_idx;
    e.dst_idx = dst_idx;
    e.emitter = dynamic_pointer_cast<XListNodeBase>(shared_from_this());
    onMove().talk(e);
    onListChanged().talk(dynamic_pointer_cast<XListNodeBase>(shared_from_this()));    
}

void
XListNodeBase::Payload::catchEvent(const shared_ptr<XNode>& var, int idx) {
	CatchEvent e;
	e.emitter = static_cast<XListNodeBase*>(&node());
	e.caught = var;
	e.index = idx;
	tr().mark(onCatch(), e);
}
void
XListNodeBase::Payload::releaseEvent(const shared_ptr<XNode>& var, int idx) {
	ReleaseEvent e;
	e.emitter = static_cast<XListNodeBase*>(&node());
	e.released = var;
	e.index = idx;
	tr().mark(onRelease(), e);
}
void
XListNodeBase::Payload::moveEvent(unsigned int src_idx, unsigned int dst_idx) {
	MoveEvent e;
	e.emitter = static_cast<XListNodeBase*>(&node());
	e.src_idx = src_idx;
	e.dst_idx = dst_idx;
	tr().mark(onMove(), e);
}
void
XListNodeBase::Payload::listChangeEvent() {
	tr().mark(onListChanged(), static_cast<XListNodeBase*>(&node()));
}

shared_ptr<XNode> _empty_creator(const char *, bool ) {
    return shared_ptr<XNode>();
}
