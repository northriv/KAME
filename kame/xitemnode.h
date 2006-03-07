#ifndef XITEMNODE_H_
#define XITEMNODE_H_

#include "xnode.h"
#include "xlistnode.h"

//! Posses a pointer to a member of a list
class XItemNodeBase : public XValueNodeBase
{
 XNODE_OBJECT
 protected:
  explicit XItemNodeBase(const char *name, bool runtime = false);
 public:
  virtual ~XItemNodeBase() {}
  
  struct Item {
    QString name, label;
  };
  virtual shared_ptr<const std::deque<Item> > itemStrings() const = 0;
  XTalker<shared_ptr<XItemNodeBase> >  &onListChanged() {return m_tlkOnListChanged;}
 private:
  XTalker<shared_ptr<XItemNodeBase> > m_tlkOnListChanged;
};

void
_xpointeritemnode_throwConversionError();

template <class TL>
class XPointerItemNode : public XItemNodeBase
{
 XNODE_OBJECT
 protected:
  XPointerItemNode(const char *name, bool runtime, const shared_ptr<TL> &list)
   :  XItemNodeBase(name, runtime)
    , m_var(new shared_ptr<XNode>()), m_list(list) {
    m_lsnOnItemReleased = list->onRelease().connectWeak(
        false, shared_from_this(), 
        &XPointerItemNode<TL>::onItemReleased);
    m_lsnOnListChanged = list->onListChanged().connectWeak(
        false, shared_from_this(), 
        &XPointerItemNode<TL>::lsnOnListChanged);
    }
 public:
  virtual ~XPointerItemNode() {}

  virtual QString to_str() const {
    shared_ptr<XNode> node(*this);
    if(node)
        return node->getName();
    else
        return QString();
  }
  virtual operator shared_ptr<XNode>() const {return *m_var;}
  virtual void value(const shared_ptr<XNode> &t) = 0;
 protected:
  virtual void _str(const QString &var) throw (XKameError &)
  {
    if(var.isEmpty()) {
        value(shared_ptr<XNode>());
        return;
    }
    atomic_shared_ptr<const XNode::NodeList> children(m_list->children());
    if(children) { 
        for(NodeList::const_iterator it = children->begin(); it != children->end(); it++) {
          if((*it)->getName() == var) {
               value(*it);
               return;
          }
        }
    }
    _xpointeritemnode_throwConversionError();
  }
  atomic_shared_ptr<shared_ptr<XNode> > m_var;
  shared_ptr<TL> m_list;
 protected:
 private:  
  void onItemReleased(const shared_ptr<XNode>& node)
  {
      if(node == *m_var)
            value(shared_ptr<XNode>());
  }
  void lsnOnListChanged(const shared_ptr<XListNodeBase>&)
  {
        onListChanged().talk(dynamic_pointer_cast<XItemNodeBase>(shared_from_this()));
  }
  shared_ptr<XListener> m_lsnOnItemReleased, m_lsnOnListChanged;
};
//! A pointer to a XListNode TL, T is value type
template <class TL, class T>
class XItemNode : public XPointerItemNode<TL>
{
 XNODE_OBJECT
 protected:
  XItemNode(const char *name, bool runtime, const shared_ptr<TL> &list)
   :  XPointerItemNode<TL>(name, runtime, list) {
   }
 public:
  virtual ~XItemNode() {}
  virtual operator shared_ptr<T>() const {
        return dynamic_pointer_cast<T>(*XPointerItemNode<TL>::m_var);
  }
  virtual void value(const shared_ptr<XNode> &t) {
    shared_ptr<XValueNodeBase> ptr = 
        dynamic_pointer_cast<XValueNodeBase>(XPointerItemNode<TL>::shared_from_this());
    XScopedLock<XRecursiveMutex> lock(m_write_mutex);
    XPointerItemNode<TL>::m_tlkBeforeValueChanged.talk(ptr);
    XPointerItemNode<TL>::m_var.reset(new shared_ptr<XNode>(t));
    XPointerItemNode<TL>::m_tlkOnValueChanged.talk(ptr); //, 1, &statusmutex);
  }
  virtual shared_ptr<const std::deque<XItemNodeBase::Item> > itemStrings() const
  {
        shared_ptr<std::deque<XItemNodeBase::Item> > items(new std::deque<XItemNodeBase::Item>());
        atomic_shared_ptr<const XNode::NodeList> children(m_list->children());
        if(children) {
            for(NodeList::const_iterator it = children->begin(); it != children->end(); it++) {
                if(dynamic_pointer_cast<T>(*it)) {
                XItemNodeBase::Item item;
                    item.name = (*it)->getName();
                    item.label = (*it)->getLabel();
                    items->push_back(item);
                }
            }
        }
        return items;
  }
 protected:
 private:
  XRecursiveMutex m_write_mutex;
};

//! Contain strings, value is one of strings
class XComboNode : public XItemNodeBase
{
 XNODE_OBJECT
 protected:
  explicit XComboNode(const char *name, bool runtime = false);
 public:
  virtual ~XComboNode() {}
  
  virtual QString to_str() const;
  virtual void add(const QString &str);
  virtual void clear();
  virtual operator int() const;
  virtual void value(int t);
  virtual void value(const QString &);
  virtual shared_ptr<const std::deque<XItemNodeBase::Item> > itemStrings() const;
 protected:
  virtual void _str(const QString &value) throw (XKameError &);
 private:
  atomic_shared_ptr<std::deque<QString> > m_strings;
  atomic<int> m_var;
  XRecursiveMutex m_write_mutex;
};

#endif /*XITEMNODE_H_*/
