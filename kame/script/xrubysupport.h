//---------------------------------------------------------------------------

#ifndef xrubysupportH
#define xrubysupportH

#include "xnode.h"
#include "xrubythread.h"
#include "xlistnode.h"

class XMeasure;

//ruby running thread
class XRuby : public XAliasListNode<XRubyThread>
{
 XNODE_OBJECT
 protected:
  XRuby(const char *name, bool runtime, const shared_ptr<XMeasure> &measure);
 public:
  virtual ~XRuby();
  
  void resume() {m_thread.resume();}
  void terminate() {m_thread.terminate();}

 protected:
  virtual void *execute(const atomic<bool> &);
 private:
   
  struct rnode_ptr
  {
    weak_ptr<XNode> ptr;
    XRuby *xruby;
  };
  //! Ruby Objects
  VALUE rbClassNode, rbClassValueNode, rbClassListNode;
  //! delete Wrapper struct
  static void rnode_free(void *);
  //! def. output
  static VALUE my_rbdefout(VALUE self, VALUE str, VALUE threadid);
  //! 
  static VALUE is_main_terminated(VALUE self);
  //! XNode wrappers
  static int strOnNode(const shared_ptr<XValueNodeBase> &node, VALUE value);
  static VALUE getValueOfNode(const shared_ptr<XValueNodeBase> &node);
  static VALUE rnode_create(const shared_ptr<XNode> &, XRuby *);
  static VALUE rnode_name(VALUE);
  static VALUE rnode_touch(VALUE);
  static VALUE rnode_count(VALUE);
  static VALUE rnode_child(VALUE, VALUE);
  static VALUE rvaluenode_set(VALUE, VALUE);
  static VALUE rvaluenode_load(VALUE, VALUE);
  static VALUE rvaluenode_get(VALUE);
  static VALUE rvaluenode_to_str(VALUE);
  static VALUE rlistnode_create_child(VALUE, VALUE, VALUE);
  static VALUE rlistnode_release_child(VALUE, VALUE);

  struct tCreateChild
  {
    std::string type;
    std::string name;
    shared_ptr<XListNodeBase> lnode;
    XCondition cond;
    shared_ptr<XNode> child;
  };
  XTalker<shared_ptr<tCreateChild> > m_tlkCreateChild;
  shared_ptr<XListener> m_lsnCreateChild;
  void onCreateChild(const shared_ptr<tCreateChild> &x);
  
  const weak_ptr<XMeasure> m_measure;
  XThread<XRuby> m_thread;
  
};

//---------------------------------------------------------------------------
#endif
