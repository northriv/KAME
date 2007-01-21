#include "xlistnode.h"

XListNodeBase::XListNodeBase(const char *name, bool runtime) :
    XNode(name, runtime)
{
}
void
XListNodeBase::clearChildren()
{
  atomic_shared_ptr<NodeList> old_list;
  old_list.swap(m_children);

  if(!old_list) return;
  for(;;)
    {
      if(old_list->empty())
            break;
      onRelease().talk(old_list->back());
        
      old_list->pop_back();
    }
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
        if(!old_list) return;
        atomic_shared_ptr<NodeList> new_list(new NodeList(*old_list));
        if(src_idx >= new_list->size()) return;
        shared_ptr<XNode> snode = new_list->at(src_idx);
        new_list->at(src_idx).reset();
        if(dst_idx > new_list->size()) return;
        XNode::NodeList::iterator dit = new_list->begin();
        dit += dst_idx;
        new_list->insert(dit, snode);
        new_list->erase(std::find(new_list->begin(), new_list->end(), shared_ptr<XNode>()));
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