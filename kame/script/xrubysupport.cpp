//---------------------------------------------------------------------------

#include "xrubysupport.h"
#include "measure.h"
extern "C" {
#include <ruby.h>
}
#include <qdeepcopy.h>
#include <klocale.h>
//---------------------------------------------------------------------------

#include <math.h>
#include <kstandarddirs.h>

#define XRUBYSUPPORT_RB "xrubysupport.rb"

static inline VALUE QString2RSTRING(const QString &qstr) {
    if(qstr.isEmpty()) return rb_str_new2("");
    return rb_str_new2(qstr.utf8());
}

XRuby::XRuby(const char *name, bool runtime, const shared_ptr<XMeasure> &measure)
  : XAliasListNode<XRubyThread>(name, runtime),
  m_measure(measure), 
  m_thread(shared_from_this(), &XRuby::execute)
{
    m_lsnCreateChild = m_tlkCreateChild.connectWeak(true, shared_from_this(),
        &XRuby::onCreateChild);
}
XRuby::~XRuby()
{
}
void
XRuby::rnode_free(void *rnode)
{
    struct rnode_ptr* st = reinterpret_cast<struct rnode_ptr*>(rnode);
    delete st;
}

VALUE
XRuby::rnode_create(const shared_ptr<XNode> &node, XRuby *xruby)
{

  struct rnode_ptr *st
     = new rnode_ptr;

  st->ptr = node;
  st->xruby = xruby;
  VALUE rnew;
  shared_ptr<XValueNodeBase> vnode = dynamic_pointer_cast<XValueNodeBase>(node);
  if(vnode)
  {
      rnew = Data_Wrap_Struct(xruby->rbClassValueNode, 0, rnode_free, st);
  }
  else
  {
      shared_ptr<XListNodeBase> lnode = dynamic_pointer_cast<XListNodeBase>(node);
      if(lnode)
      {
          rnew = Data_Wrap_Struct(xruby->rbClassListNode, 0, rnode_free, st);
      }
      else
      {
          rnew = Data_Wrap_Struct(xruby->rbClassNode, 0, rnode_free, st);
      }
  }
  return rnew;
}
VALUE
XRuby::rnode_child(VALUE self, VALUE var)
{
  shared_ptr<XNode> child;
  struct rnode_ptr *st;
  Data_Get_Struct(self, struct rnode_ptr, st);
  if(shared_ptr<XNode> node = st->ptr.lock()) {
      switch (TYPE(var)) {
       int idx;
      case T_FIXNUM:
        idx = NUM2INT(var);
        { atomic_shared_ptr<const NodeList> list = node->children();
            if ((idx >= 0) && (idx < (int)list->size()))
                child = list->at(idx);
        }
        if(! child ) {
          rb_raise(rb_eRuntimeError, "No such node idx:%d on %s\n",
             idx, (const char*)node->getName().utf8());
          return Qnil;
        }
        break;
      case T_STRING:
        {
        const char *name = RSTRING(var)->ptr;
            child = node->getChild(name);
            if(! child ) {
              rb_raise(rb_eRuntimeError, "No such node name:%s on %s\n",
                 name, (const char*)node->getName().utf8());
              return Qnil;
            }
        }
        break;
      default:
        rb_raise(rb_eRuntimeError, "Ill format to find node on %s\n",
          (const char*)node->getName().utf8());
        return Qnil;
        break;
      }
  }
  else {
      rb_raise(rb_eRuntimeError, "Node no longer exists\n");
      return Qnil;
  }
  return rnode_create(child, st->xruby );
}
VALUE
XRuby::rlistnode_create_child(VALUE self, VALUE rbtype, VALUE rbname)
{
  Check_Type(rbtype, T_STRING);
  if(TYPE(rbtype) != T_STRING) return Qnil;
  Check_Type(rbname, T_STRING);
  if(TYPE(rbname) != T_STRING) return Qnil;
  char *type = RSTRING(rbtype)->ptr;
  char *name = RSTRING(rbname)->ptr;
  
  shared_ptr<XNode> child;
  struct rnode_ptr *st;
  Data_Get_Struct(self, struct rnode_ptr, st);
  if(shared_ptr<XNode> node = st->ptr.lock()) {
      if( node->isRunTime() )
      {
          rb_raise(rb_eRuntimeError, "Node %s is run-time node!\n",
             (const char*)node->getName().utf8());
          return Qnil;
      }     
      shared_ptr<XListNodeBase> lnode =
        dynamic_pointer_cast<XListNodeBase>(node);
      if(!lnode)
      {
           rb_raise(rb_eRuntimeError, "Error on %s : Not ListNode. Could not make child"
                " name = %s, type = %s\n", (const char*)node->getName().utf8(), name, type);
            return Qnil;
      }
      if(strlen(name))
        child = node->getChild(name);
      /*
      if(type != child->getTypename()) {
          rb_raise(rb_eRuntimeError, "Different type of child exists on %s\n",
             (const char*)node->getName().utf8());
          return Qnil;
      }
      */
      if(!child) {
            shared_ptr<tCreateChild> x(new tCreateChild);
            x->lnode = lnode;
            x->type = type;
            x->name = name;
            st->xruby->m_tlkCreateChild.talk(x);
            while(x->lnode) {
                x->cond.wait(50000);
            }
            child = x->child;
      }
      if(!child) {
           rb_raise(rb_eRuntimeError, "Error on %s : Could not make child"
                " name = %s, type = %s\n", (const char*)node->getName().utf8(), name, type);
            return Qnil;
      }
  }
  else {
      rb_raise(rb_eRuntimeError, "Node no longer exists\n");
      return Qnil;
  }
  return rnode_create(child, st->xruby );
}
void
XRuby::onCreateChild(const shared_ptr<tCreateChild> &x) 
{
    x->child = x->lnode->createByTypename(x->type, x->name);
    x->lnode.reset();
    x->cond.signal();
}
VALUE
XRuby::rlistnode_release_child(VALUE self, VALUE rbchild)
{
  struct rnode_ptr *st;
  Data_Get_Struct(self, struct rnode_ptr, st);
  if(shared_ptr<XNode> node = st->ptr.lock()) {
      if(!node->isUIEnabled() )
      {
          rb_raise(rb_eRuntimeError, "Node %s is read-only!\n",
             (const char*)node->getName().utf8());
          return Qnil;
      }     
      shared_ptr<XListNodeBase> lnode =
        dynamic_pointer_cast<XListNodeBase>(node);
      if(!lnode)
        {
           rb_raise(rb_eRuntimeError, "Error on %s : Not ListNode. Could not release child\n"
               , (const char*)node->getName().utf8());
            return Qnil;
        }
      struct rnode_ptr *st_child;
      Data_Get_Struct(rbchild, struct rnode_ptr, st_child);
      if(shared_ptr<XNode> child = st_child->ptr.lock()) {
            lnode->releaseChild(child);
      }
      else {
          rb_raise(rb_eRuntimeError, "Node no longer exists\n");
          return Qnil;
      }
  }
  else {
      rb_raise(rb_eRuntimeError, "Node no longer exists\n");
      return Qnil;
  }
  return self;
}

VALUE
XRuby::rnode_name(VALUE self)
{
  struct rnode_ptr *st;
  Data_Get_Struct(self, struct rnode_ptr, st);
  if(shared_ptr<XNode> node = st->ptr.lock()) {
      return QString2RSTRING(node->getName());
  }
  else {
      rb_raise(rb_eRuntimeError, "Node no longer exists\n");
      return Qnil;
  }
}
VALUE
XRuby::rnode_count(VALUE self)
{
  struct rnode_ptr *st;
  Data_Get_Struct(self, struct rnode_ptr, st);
  if(shared_ptr<XNode> node = st->ptr.lock()) {
      atomic_shared_ptr<const NodeList> list = node->children();
      VALUE count = INT2NUM(list->size());
      return count;
  }
  else {
      rb_raise(rb_eRuntimeError, "Node no longer exists\n");
      return Qnil;
  }
}
VALUE
XRuby::rnode_touch(VALUE self)
{
  struct rnode_ptr *st;
  Data_Get_Struct(self, struct rnode_ptr, st);
  if(shared_ptr<XNode> node = st->ptr.lock()) {
      dbgPrint(QString("Ruby, Node %1, touching."));
      if(node->isUIEnabled() )
          node->touch();
      else
          rb_raise(rb_eRuntimeError, "Node %s is read-only!\n",
             (const char*)node->getName().utf8());
      return Qnil;
  }
  else {
      rb_raise(rb_eRuntimeError, "Node no longer exists\n");
      return Qnil;
  }
}
VALUE
XRuby::rvaluenode_set(VALUE self, VALUE var)
{
  struct rnode_ptr *st;
  Data_Get_Struct(self, struct rnode_ptr, st);
  if(shared_ptr<XNode> node = st->ptr.lock()) {
      shared_ptr<XValueNodeBase> vnode = dynamic_pointer_cast<XValueNodeBase>(node);
      ASSERT(vnode);
      dbgPrint(QString("Ruby, Node %1, setting new value.").arg(node->getName()) );
      if(!node->isUIEnabled() )
      {
//          rb_raise(rb_eRuntimeError, "Node %s is read-only!\n",
          rb_warning("Node %s is read-only!\n",
             (const char*)node->getName().utf8());
          return Qnil;
      }
      if(XRuby::strOnNode(vnode, var))
        {
          return Qnil;
        }
      else
        {
          dbgPrint(QString("Ruby, Node %1, new value: %2.").arg(node->getName()).arg(vnode->to_str()) );
          return XRuby::getValueOfNode(vnode);
        }
  }
  else {
      rb_raise(rb_eRuntimeError, "Node no longer exists\n");
      return Qnil;
  }
}
VALUE
XRuby::rvaluenode_load(VALUE self, VALUE var)
{
  struct rnode_ptr *st;
  Data_Get_Struct(self, struct rnode_ptr, st);
  if(shared_ptr<XNode> node = st->ptr.lock()) {
      shared_ptr<XValueNodeBase> vnode = dynamic_pointer_cast<XValueNodeBase>(node);
      ASSERT(vnode);
      dbgPrint(QString("Ruby, Node %1, loading new value.").arg(node->getName()) );
      if( node->isRunTime() )
      {
          rb_raise(rb_eRuntimeError, "Node %s is run-time node!\n",
             (const char*)node->getName().utf8());
          return Qnil;
      }
      if(XRuby::strOnNode(vnode, var))
        {
          return Qnil;
        }
      else
        {
          dbgPrint(QString("Ruby, Node %1, new value: %2.").arg(node->getName()).arg(vnode->to_str()) );
          return XRuby::getValueOfNode(vnode);
        }
  }
  else {
      rb_raise(rb_eRuntimeError, "Node no longer exists\n");
      return Qnil;
  }
}
VALUE
XRuby::rvaluenode_get(VALUE self)
{
  struct rnode_ptr *st;
  Data_Get_Struct(self, struct rnode_ptr, st);
  if(shared_ptr<XNode> node = st->ptr.lock()) {
      shared_ptr<XValueNodeBase> vnode = dynamic_pointer_cast<XValueNodeBase>(node);
      ASSERT(vnode);
      return XRuby::getValueOfNode(vnode);
  }
  else {
      rb_raise(rb_eRuntimeError, "Node no longer exists\n");
      return Qnil;
  }
}
VALUE
XRuby::rvaluenode_to_str(VALUE self)
{
  struct rnode_ptr *st;
  Data_Get_Struct(self, struct rnode_ptr, st);
  if(shared_ptr<XNode> node = st->ptr.lock()) {
      shared_ptr<XValueNodeBase> vnode = dynamic_pointer_cast<XValueNodeBase>(node);
      ASSERT(vnode);
      return QString2RSTRING(vnode->to_str());
  }
  else {
      rb_raise(rb_eRuntimeError, "Node no longer exists\n");
      return Qnil;
  }
}

int
XRuby::strOnNode(const shared_ptr<XValueNodeBase> &node, VALUE value)
{
  double dbl = 0;
  int integer = 0;

  shared_ptr<XDoubleNode> dnode = dynamic_pointer_cast<XDoubleNode>(node);
  shared_ptr<XIntNode> inode = dynamic_pointer_cast<XIntNode>(node);
  shared_ptr<XUIntNode> uinode = dynamic_pointer_cast<XUIntNode>(node);
  shared_ptr<XBoolNode> bnode = dynamic_pointer_cast<XBoolNode>(node);
  
  switch (TYPE(value)) {
  case T_FIXNUM:
    integer = NUM2INT(value);
    if(uinode) {
        if(integer >= 0) {
              uinode->value(integer); return 0;
        }
        else {
            rb_raise(rb_eRuntimeError, "Negative FIXNUM on %s\n"
                , (const char*)node->getName().utf8());
            break;
        }
    }
    if(inode) {
          inode->value(integer); return 0;
    }
    dbl = integer;
    if(dnode) {dnode->value(dbl); return 0;}
    rb_raise(rb_eRuntimeError, "FIXNUM is not appropreate on %s\n"
        , (const char*)node->getName().utf8());
    break;
  case T_FLOAT:
    dbl = RFLOAT(value)->value;
    if(dnode) {dnode->value(dbl); return 0;}
    rb_raise(rb_eRuntimeError, "FLOAT is not appropreate on %s\n"
        , (const char*)node->getName().utf8());
    break;
  case T_BIGNUM:
    dbl = NUM2DBL(value);
    if(dnode) {dnode->value(dbl); return 0;}
    rb_raise(rb_eRuntimeError, "BIGNUM is not appropreate on %s\n"
        , (const char*)node->getName().utf8());
//    integer = lrint(dbl);
//    if(inode && (dbl <= INT_MAX)) {inode->value(integer); return 0;}
    break;
  case T_STRING:
    try {
        QString qstr = QString::fromUtf8(RSTRING(value)->ptr);
        node->str(qstr);
    }
    catch (XKameError &e) {
        rb_raise(rb_eRuntimeError, "Validation error %s on %s\n"
        , (const char*)e.msg().utf8(), (const char*)node->getName().utf8());
        return -1;
    }
    return 0;
  case T_TRUE:
    if(bnode) {bnode->value(true); return 0;}
    rb_raise(rb_eRuntimeError, "TRUE is not appropreate on %s\n"
        , (const char*)node->getName().utf8());
    break;
  case T_FALSE:
    if(bnode) {bnode->value(false); return 0;}
    rb_raise(rb_eRuntimeError, "FALSE is not appropreate on %s\n"
        , (const char*)node->getName().utf8());
    break;
  default:
    rb_raise(rb_eRuntimeError, "UNKNOWN TYPE is not appropreate on %s\n"
        , (const char*)node->getName().utf8());
    break;
  }
  return -1;
}
VALUE
XRuby::getValueOfNode(const shared_ptr<XValueNodeBase> &node)
{
  shared_ptr<XDoubleNode> dnode = dynamic_pointer_cast<XDoubleNode>(node);
  shared_ptr<XIntNode> inode = dynamic_pointer_cast<XIntNode>(node);
  shared_ptr<XUIntNode> uinode = dynamic_pointer_cast<XUIntNode>(node);
  shared_ptr<XBoolNode> bnode = dynamic_pointer_cast<XBoolNode>(node);
  shared_ptr<XStringNode> snode = dynamic_pointer_cast<XStringNode>(node);
  if(dnode) {return rb_float_new((double)*dnode);}
  if(inode) {return INT2NUM(*inode);}
  if(uinode) {return UINT2NUM(*uinode);}
  if(bnode) {return (*bnode) ? Qtrue : Qfalse;}
  if(snode) {return QString2RSTRING(*snode);}
  return Qnil;
}

VALUE
XRuby::my_rbdefout(VALUE self, VALUE str, VALUE threadid)
{
  int id = NUM2INT(threadid);
  QString qstr = QString::fromUtf8(RSTRING(str)->ptr);
  struct rnode_ptr *st;
  Data_Get_Struct(self, struct rnode_ptr, st);
  shared_ptr<XRubyThread> rubythread;
  atomic_shared_ptr<const NodeList> list = st->xruby->children();
  for(unsigned int i = 0; i < list->size(); i++) {
    if(id == *dynamic_pointer_cast<XRubyThread>(list->at(i))->threadID())
        rubythread = dynamic_pointer_cast<XRubyThread>(list->at(i));
  }
  if(rubythread) {
      shared_ptr<QString> buf(new QString(QDeepCopy<QString>(qstr)));
      rubythread->onMessageOut().talk(buf);
      dbgPrint(QString("Ruby [%1]; %2").arg(rubythread->filename()->to_str()).arg(qstr));
  }
  else {
      dbgPrint(QString("Ruby [global]; %1").arg(qstr));
  }
  return Qnil;
}
VALUE
XRuby::is_main_terminated(VALUE self)
{
  struct rnode_ptr *st;
  Data_Get_Struct(self, struct rnode_ptr, st);
  return (st->xruby->m_thread.isTerminated()) ? Qtrue : Qfalse;
}

void *
XRuby::execute(const atomic<bool> &terminated)
{
    
  while (!terminated) {
      ruby_init();
      ruby_script("KAME");
      
      ruby_init_loadpath();
      
      rbClassNode = rb_define_class("XNode", rb_cObject);
      rb_global_variable(&rbClassNode);
      typedef VALUE(*fp)(...);
      rb_define_method(rbClassNode, "name", (fp)rnode_name, 0);
      rb_define_method(rbClassNode, "touch", (fp)rnode_touch, 0);
      rb_define_method(rbClassNode, "child", (fp)rnode_child, 1);
      rb_define_method(rbClassNode, "[]", (fp)rnode_child, 1);
      rb_define_method(rbClassNode, "count", (fp)rnode_count, 0);
      rbClassValueNode = rb_define_class("XValueNode", rbClassNode);
      rb_global_variable(&rbClassValueNode);
      rb_define_method(rbClassValueNode, "set", (fp)rvaluenode_set, 1);
      rb_define_method(rbClassValueNode, "internal_load", (fp)rvaluenode_load, 1);
      rb_define_method(rbClassValueNode, "get", (fp)rvaluenode_get, 0);
      rb_define_method(rbClassValueNode, "to_str", (fp)rvaluenode_to_str, 0);
      rbClassListNode = rb_define_class("XListNode", rbClassNode);
      rb_global_variable(&rbClassListNode);
      rb_define_method(rbClassListNode, "create", (fp)rlistnode_create_child, 2);
      rb_define_method(rbClassListNode, "release", (fp)rlistnode_release_child, 1);
      
      {
          shared_ptr<XMeasure> measure = m_measure.lock();
          ASSERT(measure);
          QString name = measure->getName();
          name[0] = name[0].upper();
          VALUE rbRootNode = rnode_create(measure, this);
          rb_define_global_const(name.utf8(), rbRootNode);
          rb_define_global_const("RootNode", rbRootNode);
      }
      {
          VALUE rbRubyThreads = rnode_create(shared_from_this(), this);
          rb_define_singleton_method(rbRubyThreads, "my_rbdefout", (fp)my_rbdefout, 2);
          rb_define_singleton_method(rbRubyThreads, "is_main_terminated", (fp)is_main_terminated, 0);
          rb_define_global_const("XRubyThreads", rbRubyThreads);
      }
      
      {
          int state;
          QString filename = ::locate("appdata", XRUBYSUPPORT_RB);
          if(filename.isEmpty()) {
              g_statusPrinter->printError("No KAME ruby support file installed.");
          }
          else {
              rb_load_protect (QString2RSTRING(filename), 0, &state);
          }
      }
      ruby_finalize();
    
      fprintf(stderr, "ruby finished\n");

  }
  return NULL;
}
