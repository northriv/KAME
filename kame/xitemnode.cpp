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
   : XItemNodeBase(name, runtime),
    m_strings(new std::deque<QString>()),
    m_var(-1) {
}

void
XComboNode::_str(const QString &var) throw (XKameError &)
{
  if(var.isEmpty()) {
        value(-1);
        return;
  }
  atomic_shared_ptr<const std::deque<QString> > strings(m_strings);
  unsigned int i = 0;
  for(std::deque<QString>::const_iterator it = strings->begin(); it != strings->end(); it++) {
        if(*it == var) {
            value(i);
            return;
        }
        i++;
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
    atomic_shared_ptr<const std::deque<QString> > strings(m_strings);
    if((i >= 0) && (i < (int)strings->size()))
        return QString(QDeepCopy<QString>(strings->at(i)));
    else
        return QString();
}

void
XComboNode::add(const QString &str)
{
    for(;;) {
      atomic_shared_ptr<std::deque<QString> > old_strings(m_strings);
      atomic_shared_ptr<std::deque<QString> > new_strings(new std::deque<QString>(*old_strings));
      new_strings->push_back(QString(QDeepCopy<QString>(str)));
      if(new_strings.compareAndSwap(old_strings, m_strings))
        break;
    }
    onListChanged().talk(dynamic_pointer_cast<XItemNodeBase>(shared_from_this()));
}

void
XComboNode::clear()
{
    atomic_shared_ptr<std::deque<QString> > new_strings(new std::deque<QString>());
    m_strings = new_strings;
    onListChanged().talk(dynamic_pointer_cast<XItemNodeBase>(shared_from_this()));
}

shared_ptr<const std::deque<QString> >
XComboNode::itemStrings() const
{
    shared_ptr<std::deque<QString> > strings_copy(new std::deque<QString>);
    atomic_shared_ptr<const std::deque<QString> > strings(m_strings);
    for(std::deque<QString>::const_iterator it = strings->begin(); it != strings->end(); it++) {
        strings_copy->push_back(QString(QDeepCopy<QString>(*it)));
    }
    return strings_copy;
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
