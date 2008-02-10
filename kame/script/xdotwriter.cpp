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
#include "xdotwriter.h"
#include <fstream>
#include "config.h"
#include "xitemnode.h"
#include "xlistnode.h"

XDotWriter::XDotWriter(const shared_ptr<XNode> &root, std::ofstream &ofs)
	: m_root(root), m_ofs(ofs), m_unnamedcnt(0)
{
    ASSERT(ofs.good());
    ofs << "/* KAME2 measurement configuration file" << std::endl
        << "* Automatically created. KAME version. " VERSION << std::endl
        << "* date: " << XTime::now().getTimeStr() << std::endl
        << "*/" << std::endl;
}
XDotWriter::~XDotWriter()
{
    m_ofs.flush();
}
void 
XDotWriter::write()
{
    m_ofs << "digraph "
          << "G" //(const char *)m_root->getName().c_str()
          << " {"
          << std::endl;
    m_ofs << "node [shape=box,style=filled,color=green];" << std::endl;
          
    write(m_root);
    m_ofs << "}"
          << std::endl;
}
void 
XDotWriter::write(const shared_ptr<XNode> &node)
{
    if(std::find(m_nodes.begin(), m_nodes.end(), node) == m_nodes.end()) {
        m_ofs << "obj_" << (int)node.get()
              << " [label=\"" << node->getName()
              << "\"]" << std::endl;
        m_nodes.push_back(node);
    }
//    shared_ptr<XValueNodeBase> vnode = dynamic_pointer_cast<XValueNodeBase>(node);
        
//    shared_ptr<XListNodeBase> lnode = dynamic_pointer_cast<XListNodeBase>(node);
    int unnamed = 0;
    atomic_shared_ptr<const XNode::NodeList> list(m_root->children());
    if(list) { 
		for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++) {
			shared_ptr<XNode> child = *it;
           
			if(child->getName().empty()) {
				unnamed++;
			}
			else {
				m_ofs << "obj_" << (int)child.get()
					  << " -> "
					  << "obj_" << (int)node.get()
					  << std::endl;
				write(child);
			}
		}
    }
    if(unnamed) {
        m_unnamedcnt++;
        m_ofs << "unnamedobj_" << (int)m_unnamedcnt
              << " [label=\"" << (const char*)QString("%1 obj.").arg(unnamed).utf8()
              << "\"]" << std::endl;
        m_ofs << "unnamedobj_" << (int)m_unnamedcnt
              << " -> "
              << "obj_" << (int)node.get()
              << std::endl;
    }
}
