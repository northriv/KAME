#ifndef XSIGNAL_PRV_H_
#define XSIGNAL_PRV_H_

template <class Arg>
class _XListenerImpl : public XListener
{
 protected:
  _XListenerImpl(bool mainthreadcall, bool avoid_dup, unsigned int _delay_ms)
    : XListener(mainthreadcall, avoid_dup, _delay_ms), arg() {}
 public:
  virtual ~_XListenerImpl() {}
  virtual void operator() (const Arg &) const = 0;
  //! this is used when m_bAvoidDup is on.
  atomic_scoped_ptr<Arg> arg;
};
template <class Arg>
class _XListenerStatic : public _XListenerImpl<Arg>
{
  friend class XTalker<Arg>;
 protected:
  _XListenerStatic(void (*func)(const Arg &),
    bool mainthreadcall, bool avoid_dup, unsigned int _delay_ms) :
    _XListenerImpl<Arg>(mainthreadcall, avoid_dup, _delay_ms), m_func(func) {
    }
 public:
  virtual void operator() (const Arg &x) const {
     (*m_func)(x);
  }
 private:
  void (*const m_func)(const Arg &);
};
template <class tClass, class Arg>
class _XListenerWeak : public _XListenerImpl<Arg>
{
  friend class XTalker<Arg>;
 protected:
  _XListenerWeak(const shared_ptr<tClass> &obj, void (tClass::*func)(const Arg &),
    bool mainthreadcall, bool avoid_dup, unsigned int _delay_ms) :
    _XListenerImpl<Arg>(mainthreadcall, avoid_dup, _delay_ms), m_func(func), m_obj(obj) {
        ASSERT(obj);
    }
 public:
  virtual void operator() (const Arg &x) const {
     if(shared_ptr<tClass> p = m_obj.lock() ) ((p.get())->*m_func)(x);
  }
 private:
  void (tClass::*const m_func)(const Arg &);
  const weak_ptr<tClass> m_obj;
};
template <class tClass, class Arg>
class _XListenerShared : public _XListenerImpl<Arg>
{
  friend class XTalker<Arg>;
 protected:
  _XListenerShared(const shared_ptr<tClass> &obj, void (tClass::*func)(const Arg &),
    bool mainthreadcall, bool avoid_dup, unsigned int delay_ms) :
    _XListenerImpl<Arg>(mainthreadcall, avoid_dup, delay_ms), m_obj(obj), m_func(func)   {
        ASSERT(obj);
   }
 public:
  virtual void operator() (const Arg &x) const {((m_obj.get())->*m_func)(x);}
 private:
  void (tClass::*m_func)(const Arg &);
  const shared_ptr<tClass> m_obj;
};

#endif /*XSIGNAL_PRV_H_*/
