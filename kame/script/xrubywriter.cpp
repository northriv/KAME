#include "xrubywriter.h"
#include <fstream>
#include "config.h"
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
    QString name = m_root->getName();
    name[0] = name[0].upper();
    m_ofs << "x << " 
        << name.utf8()
        << std::endl;
    { XScopedReadLock<XRecursiveRWLock> lock(m_root->childMutex());
        write(m_root, false, 0);
    }
}
void 
XRubyWriter::write(const shared_ptr<XNode> &node, bool ghost, int level)
{
    ghost = ghost || node->isRunTime();
    shared_ptr<XValueNodeBase> vnode = dynamic_pointer_cast<XValueNodeBase>(node);
    if(vnode) {
        if(vnode->count()) {
            for(int j = 0; j < level; j++) m_ofs << "\t";
            if(ghost)
                m_ofs << "# ";
            m_ofs << "x.last";
        }
        else {
            if(ghost)
                m_ofs << "\t# ";
        }
        QString s(vnode->to_str());
        s.replace( QChar('\\'), "\\\\");
        s.replace( QChar('\n'), "\\n");
        s.replace( QChar('\r'), "\\r");
        s.replace( QChar('\t'), "\\t");
        m_ofs << ".load(\"" 
            << (const char *)s.utf8()
            << "\")" 
            << std::endl;
    }
    else
        if(node->count() == 0) {m_ofs << std::endl;}
        
    shared_ptr<XListNodeBase> lnode = dynamic_pointer_cast<XListNodeBase>(node);
    bool write_typename = false;
    if(lnode) {
        // XListNode doesn't want typename
        write_typename = (lnode->getTypename().find("XListNode") != 0);
        // XAliasListNode doesn't want creation of child
        if(lnode->getTypename().find("XAliasListNode") == 0) lnode.reset();
    }
    for(unsigned int i = 0; i < node->count(); i++) {
        shared_ptr<XNode> child = node->getChild<XNode>(i);
        XScopedReadLock<XRecursiveRWLock> lock(child->childMutex());
        
        for(int j = 0; j < level; j++) m_ofs << "\t";
        if(ghost)
            m_ofs << "# ";
        if(child->count()) {
            m_ofs << "x << ";
        }
        if(lnode) {
            m_ofs << "x.last.create(\""
                << (write_typename ? child->getTypename().c_str() : "")
                << "\",\""
                << (const char *)child->getName().utf8()
                <<  "\")";
        }
        else {
            if(child->getName().length()) {
                m_ofs << "x.last[\""
                    << (const char *)child->getName().utf8()
                    <<  "\"]";
            }
            else {
                m_ofs << "x.last["
                    << i
                    <<  "]";
            }
        }
        if(child->count()) {
            m_ofs << std::endl;
        }
        write(child, ghost, level + 1);
        if(child->count()) {
            for(int j = 0; j < level; j++) m_ofs << "\t";
            if(ghost)
                m_ofs << "# ";
            m_ofs << "x.pop"
                  << std::endl;
        }
    }
}
