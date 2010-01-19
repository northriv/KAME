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
#include "xlistnode.h"

XListNodeBase::XListNodeBase(const char *name, bool runtime) :
    XNode(name, runtime)
{
}
void
XListNodeBase::clearChildren()
{
	NodeList::reader rd(m_children);
	if(!rd || rd->empty())
		return;
	for(;;) {
		NodeList::writer tr(m_children);
		if(!tr || tr->empty()) {
			onListChanged().talk(dynamic_pointer_cast<XListNodeBase>(shared_from_this()));
			return;
		}
		shared_ptr<XNode> node(tr->back());
		tr->pop_back();
		if(tr->empty())
			tr.reset();
		if(tr.commit()) {
			onRelease().talk(node);
			continue;
		}
	}
}
int
XListNodeBase::releaseChild(const shared_ptr<XNode> &node)
{
    if(XNode::releaseChild(node)) return -1;
    onRelease().talk(node);

    onListChanged().talk(dynamic_pointer_cast<XListNodeBase>(shared_from_this()));
    return 0;
}
void
XListNodeBase::insert(const shared_ptr<XNode> &ptr)
{
	XNode::insert(ptr);
	onCatch().talk(ptr);
	onListChanged().talk(dynamic_pointer_cast<XListNodeBase>(shared_from_this()));
}
void
XListNodeBase::move(unsigned int src_idx, unsigned int dst_idx)
{
    for(;;) {
        NodeList::writer tr(m_children);
        if(src_idx >= tr->size())
        	return;
        shared_ptr<XNode> snode = tr->at(src_idx);
        tr->at(src_idx).reset();
        if(dst_idx > tr->size()) return;
        XNode::NodeList::iterator dit = tr->begin();
        dit += dst_idx;
        tr->insert(dit, snode);
        tr->erase(std::find(tr->begin(), tr->end(), shared_ptr<XNode>()));
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

shared_ptr<XNode> empty_creator(const char *, bool ) {
    return shared_ptr<XNode>();
}
