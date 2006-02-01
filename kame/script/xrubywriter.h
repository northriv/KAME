#ifndef XRUBYWRITER_H_
#define XRUBYWRITER_H_

#include "xnode.h"

class XRubyWriter 
{
public:
    XRubyWriter(const shared_ptr<XNode> &root, std::ofstream &ofs);
    ~XRubyWriter();
    void write();
private:
    void write(const shared_ptr<XNode> &node, bool ghost, int level);
    shared_ptr<XNode> m_root;
    std::ofstream &m_ofs;    
};


#endif /*XRUBYWRITER_H_*/
