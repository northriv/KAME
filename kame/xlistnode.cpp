#include "xlistnode.h"

XListNodeBase::XListNodeBase(const char *name, bool runtime) :
    XNode(name, runtime)
{
}
void
XListNodeBase::clearChildren()
{
  bool deleted = false;
  m_childmutex.readLock();
  for(;;)
    {
      if(m_children.empty()) {
            break;
      }
      onRelease().talk(m_children.back());
        
      m_childmutex.writeLock();
      m_children.pop_back();
      m_childmutex.writeUnlock();
        
      deleted = true;
    }
  if(deleted)
    onListChanged().talk(dynamic_pointer_cast<XListNodeBase>(shared_from_this()));
  m_childmutex.readUnlock();
}
int
XListNodeBase::releaseChild(unsigned int index)
{
  std::deque<shared_ptr<XNode> >::iterator it;

  m_childmutex.readLock();
  it = m_children.begin();
  for(unsigned int i = 0; i < index; i++) it++;
  ASSERT(it != m_children.end());
  onRelease().talk(*it);

  m_childmutex.writeLock();
  m_children.erase(it);
  m_childmutex.writeUnlock();

  onListChanged().talk(dynamic_pointer_cast<XListNodeBase>(shared_from_this()));
  m_childmutex.readUnlock();
  return 0;
}
int
XListNodeBase::releaseChild(const shared_ptr<XNode> &node)
{
  std::deque<shared_ptr<XNode> >::iterator it;

  m_childmutex.readLock();
  it = find(m_children.begin(), m_children.end(), node);
  ASSERT(it != m_children.end());
  onRelease().talk(*it);

  m_childmutex.writeLock();
  m_children.erase(it);
  m_childmutex.writeUnlock();

  onListChanged().talk(dynamic_pointer_cast<XListNodeBase>(shared_from_this()));
  m_childmutex.readUnlock();
  return 0;
}
void
XListNodeBase::insert(const shared_ptr<XNode> &ptr)
{
  m_childmutex.writeLock();
  XNode::insert(ptr);
  m_childmutex.writeUnlockNReadLock();   
  onCatch().talk(ptr);
  onListChanged().talk(dynamic_pointer_cast<XListNodeBase>(shared_from_this()));
  m_childmutex.readUnlock();
}
void
XListNodeBase::move(unsigned int src_idx, unsigned int dst_idx)
{
    ASSERT(m_childmutex.isLocked());
    XScopedWriteLock<XRecursiveRWLock> lock(m_childmutex);
    tchildren_it dit = m_children.begin();
    for(int i = 0; i < dst_idx; i++) {
        dit++;
    }
    tchildren_it sit = m_children.begin();
    for(int i = 0; i < src_idx; i++) {
        ASSERT(sit != m_children.end());
        sit++;
    }
    shared_ptr<XNode> node(*sit);
    m_children.insert(dit, node);
    m_children.erase(sit);
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