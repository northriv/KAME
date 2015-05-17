/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef XDOTWRITER_H_
#define XDOTWRITER_H_

#include "xnode.h"

class XDotWriter {
public:
    XDotWriter(const shared_ptr<XNode> &root, std::ofstream &ofs);
    ~XDotWriter();
    void write();
private:
    void write(const shared_ptr<XNode> &node, const Snapshot &shot);
    shared_ptr<XNode> m_root;
    std::ofstream &m_ofs;    
    std::deque<shared_ptr<XNode> > m_nodes;
    int m_unnamedcnt;
};


#endif /*XDOTWRITER_H_*/
