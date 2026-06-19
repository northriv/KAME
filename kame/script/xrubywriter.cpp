/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include <fstream>
#include "xrubywriter.h"
#include "xitemnode.h"
#include "xlistnode.h"

// Opt-in STM commit logging: when KAME_STM_LOG_COMMITS=1 is set in the
// environment, the .kam save path sums per-node m_tx_commit_count over
// the whole tree, emits it as a footer comment, and surfaces it in the
// message log.  Default off — bit-for-bit identical to the
// pre-instrumentation .kam output (no footer, no gMessagePrint call,
// no tree-walk overhead of the extra load-per-node).  Lambda-static
// caches the env lookup at first save.
static bool _kame_log_commits_enabled() noexcept {
    static const bool v = []() {
        const char *e = std::getenv("KAME_STM_LOG_COMMITS");
        return e && e[0] == '1';
    }();
    return v;
}

XRubyWriter::XRubyWriter(const shared_ptr<XNode> &root, std::ofstream &ofs)
	: m_root(root), m_ofs(ofs)
{
    assert(ofs.good());
    ofs << "# KAME2 measurement configuration file" << std::endl
        << "# Automatically created. KAME version. " << VERSION << std::endl
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
    // Aggregate STM commit count over the whole tree, paired with the
    // "# date:" header stamp so that two saved .kam files give commits/s
    // = Delta(stm_total_tx_commits) / Delta(date).
    // Gated on KAME_STM_LOG_COMMITS=1 so production saves stay bit-for-bit
    // identical (no extra footer, no log message).
    if(_kame_log_commits_enabled()) {
        m_ofs << "# stm_total_tx_commits: " << m_totalCommits << std::endl;
        gMessagePrint(formatString("STM total committed transactions: %llu",
                                   (unsigned long long)m_totalCommits));
    }
}
void 
XRubyWriter::write(
    const shared_ptr<XNode> &node, const Snapshot &shot,
    bool ghost, int level)
{
	int size = shot.size(node);
    // Skip the per-node m_tx_commit_count load entirely in the
    // common (logging-disabled) path so non-instrumented saves pay
    // zero extra cost on the tree walk.
    if(_kame_log_commits_enabled())
        m_totalCommits += node->numTransactionsCommitted();
    ghost = ghost || shot[ *node].isRuntime();
    auto vnode = dynamic_pointer_cast<XValueNodeBase>(node);
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
        
    auto lnode = dynamic_pointer_cast<XListNodeBase>(node);
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
        for(auto it = list.begin(); it != list.end(); it++) {
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
