/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitagawa@scphys.kyoto-u.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef XLISTNODE_H_
#define XLISTNODE_H_

#include "xnode.h"

class XListNodeBase : public XNode
{
 XNODE_OBJECT
 protected:
  explicit XListNodeBase(const char *name, bool runtime = false);
 public:
  virtual ~XListNodeBase() {}

  virtual void clearChildren();
  virtual int releaseChild(const shared_ptr<XNode> &node);
        
  //! called after creation, moving, deleting.
   XTalker<shared_ptr<XListNodeBase> > &onListChanged() {return m_tlkOnListChanged;}

  //! called after moving.
   struct MoveEvent {
     unsigned int src_idx, dst_idx;
     shared_ptr<XListNodeBase> emitter;
   };
   XTalker<MoveEvent> &onMove() {return m_tlkOnMove;}

  //! called after creating a child
   XTalker<shared_ptr<XNode> > &onCatch() {return m_tlkOnCatch;}
  //! called before deleting a child
   XTalker<shared_ptr<XNode> > &onRelease() {return m_tlkOnRelease;}
  
  //! append new item.
  virtual void insert(const shared_ptr<XNode> &ptr);

  void move(unsigned int src_idx, unsigned int dest_idx);
  
  //! Create a object, whose class is determined from \a type.
  //! Scripting only. Use XNode::create for coding instead.
  virtual shared_ptr<XNode> createByTypename(
        const std::string &type, const std::string &name) = 0;        
 protected:    
  XTalker<shared_ptr<XListNodeBase> > m_tlkOnListChanged;
  XTalker<MoveEvent> m_tlkOnMove;
  XTalker<shared_ptr<XNode> > m_tlkOnCatch, m_tlkOnRelease;
};

template <class NT>
class XListNode : public  XListNodeBase
{
 XNODE_OBJECT
 protected:
  explicit XListNode(const char *name, bool runtime = false)
   :  XListNodeBase(name, runtime) {}
 public:
  virtual ~XListNode() {}

  virtual shared_ptr<XNode> createByTypename(
        const std::string &, const std::string &name) {
    return this->create<NT>(name.c_str(), false);
  }
};

//! creation by UI is not allowed.
template <class NT>
class XAliasListNode : public  XListNodeBase
{
 XNODE_OBJECT
 protected:
  explicit XAliasListNode(const char *name, bool runtime = false)
   :  XListNodeBase(name, runtime) {}
 public:
  virtual ~XAliasListNode() {}

  virtual shared_ptr<XNode> createByTypename(
        const std::string &, const std::string &) {
    return shared_ptr<XNode>();
  }
};

template <class NT>
class XCustomTypeListNode : public  XListNodeBase
{
 XNODE_OBJECT
 protected:
  explicit XCustomTypeListNode(const char *name, bool runtime = false)
   :  XListNodeBase(name, runtime) {}
 public:
  virtual ~XCustomTypeListNode() {}

  virtual shared_ptr<XNode> createByTypename(
        const std::string &type, const std::string &name) = 0;
};  

  shared_ptr<XNode> empty_creator(const char *, bool = false);
  template <typename X>
  shared_ptr<XNode> empty_creator(const char *, bool, X) {
    return shared_ptr<XNode>();
  }
  template <typename X, typename Y>
  shared_ptr<XNode> empty_creator(const char *, bool, X, Y) {
    return shared_ptr<XNode>();
  }
  template <typename X, typename Y, typename Z>
  shared_ptr<XNode> empty_creator(const char *, bool, X, Y, Z) {
    return shared_ptr<XNode>();
  }
  template <typename X, typename Y, typename Z, typename ZZ>
  shared_ptr<XNode> empty_creator(const char *, bool, X, Y, Z, ZZ) {
    return shared_ptr<XNode>();
  }
  template <typename T>
  shared_ptr<XNode> _creator(const char *name, bool runtime) {
    return createOrphan<T>(name, runtime);
  }
  template <typename T, typename X>
  shared_ptr<XNode> _creator(const char *name, bool runtime, X x) {
    return createOrphan<T>(name, runtime, x);
  }
  template <typename T, typename X, typename Y>
  shared_ptr<XNode> _creator(const char *name, bool runtime, X x, Y y) {
    return createOrphan<T>(name, runtime, x, y);
  }
  template <typename T, typename X, typename Y, typename Z>
  shared_ptr<XNode> _creator(const char *name, bool runtime, X x, Y y, Z z) {
    return createOrphan<T>(name, runtime, x, y, z);
  }
  template <typename T, typename X, typename Y, typename Z, typename ZZ>
  shared_ptr<XNode> _creator(const char *name, bool runtime, X x, Y y, Z z, ZZ zz) {
    return createOrphan<T>(name, runtime, x, y, z, zz);
  }

//! Register typename and constructor.
//! make static member of TypeHolder<> in your class
//! After def. of static TypeHolder<>, define Creator to register.
//! call creator(type)(type, name, ...) to create children.
template <class tFunc = 
    shared_ptr<XNode>(*)(const char *, bool)>
struct XTypeHolder
{
     template <class tChild>
     struct Creator {
         Creator(XTypeHolder &holder, const char *name, const char *label = 0L) {
            tFunc create_typed = _creator<tChild>;
            holder.creators.push_back(create_typed);
            holder.names.push_back(std::string(name));
            holder.labels.push_back(std::string(label ? label : name));
         }
     };
     tFunc creator(const std::string &tp) {
         for(unsigned int i = 0; i < names.size(); i++) {
            if(names[i] == tp) return creators[i];
         }
         return empty_creator;
     }
     std::deque<tFunc> creators;
     std::deque<std::string> names, labels;
};

#define DEFINE_TYPE_HOLDER_W_FUNC(func) \
  typedef XTypeHolder<func> TypeHolder; \
  static TypeHolder s_types; \
  static tCreateFunc creator(const std::string &tp) {return s_types.creator(tp);} \
  static std::deque<std::string> &typenames() {return s_types.names;} \
  static std::deque<std::string> &typelabels() {return s_types.labels;}

#define DEFINE_TYPE_HOLDER \
  typedef shared_ptr<XNode>(*tCreateFunc)(const char *, bool); \
  DEFINE_TYPE_HOLDER_W_FUNC(tCreateFunc);

#define DEFINE_TYPE_HOLDER_EXTRA_PARAM(extra_param) \
  typedef shared_ptr<XNode>(*tCreateFunc)(const char *, bool \
  ,extra_params); \
  DEFINE_TYPE_HOLDER_W_FUNC(tCreateFunc);

#define DEFINE_TYPE_HOLDER_EXTRA_PARAMS_2(par1, par2) \
  typedef shared_ptr<XNode>(*tCreateFunc)(const char *, bool \
  ,par1, par2); \
  DEFINE_TYPE_HOLDER_W_FUNC(tCreateFunc);

#define DEFINE_TYPE_HOLDER_EXTRA_PARAMS_3(par1, par2, par3) \
  typedef shared_ptr<XNode>(*tCreateFunc)(const char *, bool \
  ,par1, par2, par3); \
  DEFINE_TYPE_HOLDER_W_FUNC(tCreateFunc);
  
#define DEFINE_TYPE_HOLDER_EXTRA_PARAMS_4(par1, par2, par3, par4) \
  typedef shared_ptr<XNode>(*tCreateFunc)(const char *, bool \
  ,par1, par2, par3, par4); \
  DEFINE_TYPE_HOLDER_W_FUNC(tCreateFunc);

#define DECLARE_TYPE_HOLDER \
    LIST::TypeHolder LIST::s_types;

#define _REGISTER_TYPE_2(type, name, label) LIST::TypeHolder::Creator<type> \
    g_driver_type_ ## name(LIST::s_types, # name, label);
    
#define REGISTER_TYPE(type, label) _REGISTER_TYPE_2(X ## type, type, label)

class XStringList : public  XListNode<XStringNode>
{
 XNODE_OBJECT
 protected:
  explicit XStringList(const char *name, bool runtime = false)
   :  XListNode<XStringNode>(name, runtime) {}
 public:
  virtual ~XStringList() {}
};

#endif /*XLISTNODE_H_*/
