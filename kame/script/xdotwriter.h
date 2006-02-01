#ifndef XDOTWRITER_H_
#define XDOTWRITER_H_

#include "xnode.h"

class XDotWriter 
{
public:
    XDotWriter(const shared_ptr<XNode> &root, std::ofstream &ofs);
    ~XDotWriter();
    void write();
private:
    void write(const shared_ptr<XNode> &node);
    shared_ptr<XNode> m_root;
    std::ofstream &m_ofs;    
    std::deque<shared_ptr<XNode> > m_nodes;
    int m_unnamedcnt;
};


#endif /*XDOTWRITER_H_*/
