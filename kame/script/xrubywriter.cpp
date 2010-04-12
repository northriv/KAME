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
#include <fstream>
#include "xrubywriter.h"
#include "xitemnode.h"
#include "xlistnode.h"

XRubyWriter::XRubyWriter(const shared_ptr<XNode> &root, std::ofstream &ofs)
	: m_root(root), m_ofs(ofs)
{
    ASSERT(ofs.good());
    ofs << "# KAME2 measurement configuration file" << std::endl
        << "# Automatically created. KAME version. " VERSION << std::endl
        << "# date: " << XTime::now().getTimeStr() << std::endl;
    ofs << "x = Array.new" << std::endl;
}
XRubyWriter::~XRubyWriter()
{
    m_ofs.flush();
}
void 
XRubyWriter::write()
{
    XString name = m_root->getName();
    name[0] = toupper(name[0]);
    m_ofs << "x << " 
		  << name
		  << std::endl;
	Snapshot shot( *m_root);
    write(m_root, shot, false, 0);
}
void 
XRubyWriter::write(
    const shared_ptr<XNode> &node, const Snapshot &shot,
    bool ghost, int level)
{
	int size = shot.size(node);
    ghost = ghost || shot[ *node].isRuntime();
    shared_ptr<XValueNodeBase> vnode = dynamic_pointer_cast<XValueNodeBase>(node);
    if(vnode) {
        if(size) {
            for(int j = 0; j < level; j++) m_ofs << "\t";
            if(ghost)
                m_ofs << "# ";
            m_ofs << "x.last";
        }
        else {
            if(ghost)
                m_ofs << "\t# ";
        }
        QString s(shot[ *vnode].to_str());
        s.replace( QChar('\\'), "\\\\");
        s.replace( QChar('\n'), "\\n");
        s.replace( QChar('\r'), "\\r");
        s.replace( QChar('\t'), "\\t");
        m_ofs << ".load(\"" 
			  << (const char *)s.toUtf8().data()
			  << "\")" 
			  << std::endl;
    }
    else
        if( ! size) {m_ofs << std::endl;}
        
    shared_ptr<XListNodeBase> lnode = dynamic_pointer_cast<XListNodeBase>(node);
    bool write_typename = false;
    if(lnode) {
        // XListNode doesn't want typename
        write_typename = (lnode->getTypename().find("XListNode") != 0);
        // XAliasListNode doesn't want creation of child
        if(lnode->getTypename().find( "XAliasListNode") == 0) lnode.reset();
    }
    unsigned idx = 0;
    if(size) {
    	const XNode::NodeList &list( *shot.list(node));
        for(XNode::const_iterator it = list.begin(); it != list.end(); it++) {
            shared_ptr<XNode> child = *it;
            for(int j = 0; j < level; j++) m_ofs << "\t";
            if(ghost)
                m_ofs << "# ";
            int child_size = shot.size(child);
            if(child_size) {
                m_ofs << "x << ";
            }
            if(lnode) {
                m_ofs << "x.last.create(";
                if(write_typename || child->getName().length()) {
                    m_ofs << "\""
						  << (write_typename ? child->getTypename().c_str() : "")
						  << "\"";
                }
                if(child->getName().length()) {
                    m_ofs << ",\""
						  << child->getName()
						  << "\"";
                }
                m_ofs <<  ")";
            }
            else {
                if(child->getName().length()) {
                    m_ofs << "x.last[\""
						  << child->getName()
						  <<  "\"]";
                }
                else {
                    m_ofs << "x.last["
						  << idx
						  <<  "]";
                }
            }
            if(child_size) {
                m_ofs << std::endl;
            }
            write(child, shot, ghost, level + 1);
            if(child_size) {
                for(int j = 0; j < level; j++) m_ofs << "\t";
                if(ghost)
                    m_ofs << "# ";
                m_ofs << "x.pop"
                      << std::endl;
            }
            
            idx++;
        }
    }
}
