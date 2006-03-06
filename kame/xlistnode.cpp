#include "xlistnode.h"

XListNodeBase::XListNodeBase(const char *name, bool runtime) :
    XNode(name, runtime)
{
}
void
XListNodeBase::clearChildren()
{
    atomic_shared_ptr<NodeList> old_list;
    for(;;) {
        old_list = m_children;
        atomic_shared_ptr<NodeList> new_list(new NodeList);
        if(new_list.compareAndSwap(old_list, m_children)) break;
    }

  bool deleted = false;
  for(;;)
    {
      if(old_list->empty())
            break;
      onRelease().talk(old_list->back());
        
      old_list->pop_back();
        
      deleted = true;
    }
  if(deleted)
    onListChanged().talk(dynamic_pointer_cast<XListNodeBase>(shared_from_this()));
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
        atomic_shared_ptr<NodeList> old_list(m_children);
        atomic_shared_ptr<NodeList> new_list(new NodeList(*old_list));        
        NodeList::iterator dit = new_list->begin();
        for(unsigned int i = 0; i < dst_idx; i++) {
            if(dit == new_list->end()) break;
            dit++;
        }
        NodeList::iterator sit = new_list->begin();
        for(unsigned int i = 0; i < src_idx; i++) {
            if(sit == new_list->end()) return;
            sit++;
        }
        shared_ptr<XNode> node(*sit);
        new_list->insert(dit, node);
        new_list->erase(sit);
        if(new_list.compareAndSwap(old_list, m_children)) break;
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