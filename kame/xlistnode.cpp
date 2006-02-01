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

shared_ptr<XNode> empty_creator(const char *, bool ) {
    return shared_ptr<XNode>();
}