#include "xitemnode.h"
#include <klocale.h>
#include <qdeepcopy.h>

XItemNodeBase::XItemNodeBase(const char *name, bool runtime) : 
    XValueNodeBase(name, runtime)
{
}

void
_xpointeritemnode_throwConversionError() {
   throw XKameError(i18n("No item."), __FILE__, __LINE__);
}

XComboNode::XComboNode(const char *name, bool runtime)
   : XItemNodeBase(name, runtime), m_var(-1) {
}

void
XComboNode::_str(const QString &var) throw (XKameError &)
{
  if(var.isEmpty()) {
        value(-1);
        return;
  }
  XScopedReadLock<XRecursiveRWLock> lock(listMutex());
  for(unsigned int i = 0; i < m_strings.size(); i++)
    {
        if(m_strings[i] == var)
        {
            value(i);
            return;
        }
    }
   throw XKameError(i18n("No item."), __FILE__, __LINE__);
}

void
XComboNode::value(const QString &s)
{
    try {
        str(s);
    }
    catch (XKameError &e) {
        e.print();
    }
}

XComboNode::operator int() const {
    return m_var;
}
QString
XComboNode::to_str() const
{
    int i = m_var;
    XScopedReadLock<XRecursiveRWLock> lock(listMutex());    
    if((i >= 0) && (i < (int)m_strings.size()))
        return QString(QDeepCopy<QString>(m_strings[i]));
    else
        return QString();
}

void
XComboNode::add(const QString &str)
{
   XScopedReadLock<XRecursiveRWLock> lock(listMutex());    
   { XScopedWriteLock<XRecursiveRWLock> writelock(m_listmutex);    
     m_strings.push_back(QString(QDeepCopy<QString>(str)));
   }
    onListChanged().talk(dynamic_pointer_cast<XItemNodeBase>(shared_from_this()));
}

void
XComboNode::clear()
{
   XScopedReadLock<XRecursiveRWLock> lock(listMutex());    
   { XScopedWriteLock<XRecursiveRWLock> writelock(m_listmutex);    
        m_strings.clear();
   }
    onListChanged().talk(dynamic_pointer_cast<XItemNodeBase>(shared_from_this()));
}

QString
XComboNode::operator[](unsigned int index) const
{
    ASSERT(listMutex().isLocked());
    if(index < m_strings.size())
      return QString(QDeepCopy<QString>(m_strings[index]));
    else
      return QString();
}
unsigned int
XComboNode::itemCount() const
{
    ASSERT(listMutex().isLocked());
    return m_strings.size();
}
void
XComboNode::value(int t) {
    shared_ptr<XValueNodeBase> ptr = 
        dynamic_pointer_cast<XValueNodeBase>(shared_from_this());
    XScopedLock<XRecursiveMutex> lock(m_write_mutex);
    m_tlkBeforeValueChanged.talk(ptr);
    m_var = t;
    m_tlkOnValueChanged.talk(ptr);
}
