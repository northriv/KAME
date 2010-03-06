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
