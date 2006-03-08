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
    atomic_shared_ptr<const XNode::NodeList> list = m_root->children();
    write(m_root, list, false, 0);
}
void 
XRubyWriter::write(
    const shared_ptr<XNode> &node, const atomic_shared_ptr<const XNode::NodeList> &list,
    bool ghost, int level)
{
    ghost = ghost || node->isRunTime();
    shared_ptr<XValueNodeBase> vnode = dynamic_pointer_cast<XValueNodeBase>(node);
    if(vnode) {
        if(list) {
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
        if(!list) {m_ofs << std::endl;}
        
    shared_ptr<XListNodeBase> lnode = dynamic_pointer_cast<XListNodeBase>(node);
    bool write_typename = false;
    if(lnode) {
        // XListNode doesn't want typename
        write_typename = (lnode->getTypename().find("XListNode") != 0);
        // XAliasListNode doesn't want creation of child
        if(lnode->getTypename().find("XAliasListNode") == 0) lnode.reset();
    }
    unsigned idx = 0;
    if(list) {
        for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++) {
            shared_ptr<XNode> child = *it;
            for(int j = 0; j < level; j++) m_ofs << "\t";
            if(ghost)
                m_ofs << "# ";
            atomic_shared_ptr<const XNode::NodeList> child_list = child->children();
            if(child_list) {
                m_ofs << "x << ";
            }
            if(lnode) {
                m_ofs << "x.last.create(\""
                    << (write_typename ? child->getTypename().c_str() : "")
                    << "\",\""
                    << child->getName()
                    <<  "\")";
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
            if(child_list) {
                m_ofs << std::endl;
            }
            write(child, child_list, ghost, level + 1);
            if(child_list) {
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
