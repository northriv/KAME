#ifndef XNODECONNECTOR_PRV_H_
#define XNODECONNECTOR_PRV_H_

#include "support.h"
#include "xsignal.h"

#include <fstream>
#include <qobject.h>

class QWidget;

void _sharedPtrQDeleter(QObject *);

template <class T>
class qshared_ptr : public shared_ptr<T>
{
 public:
    qshared_ptr() : shared_ptr<T>() {}
    template <class Y>
    qshared_ptr(const qshared_ptr<Y> &p) 
        : shared_ptr<T>(static_cast<const shared_ptr<Y> &>(p) ) {}
    template <class Y>
    explicit qshared_ptr(Y * p) 
        : shared_ptr<T>(p, _sharedPtrQDeleter) {
            ASSERT(isMainThread());
         }
    template <class Y>
    qshared_ptr<T> &operator=(const qshared_ptr<Y> &p) {
        shared_ptr<T>::operator=(p);
        return *this;
    }
};

class XQConnector;

class _XQConnectorHolder : public QObject
{
  Q_OBJECT
 public:
  _XQConnectorHolder(XQConnector *con);
  ~_XQConnectorHolder();
  bool isAlive() const;
 private slots:
 protected slots:
  void destroyed ();
 protected:
  shared_ptr<XQConnector> m_connector;
};

typedef qshared_ptr<_XQConnectorHolder> xqcon_ptr;

//! function for creating XQConnector instances
template <class T, class A, class B>
xqcon_ptr xqcon_create(const shared_ptr<A> &a, B *b) {
    xqcon_ptr pHolder(new _XQConnectorHolder( new
          T(a, b)));
    return pHolder;
}
//! function for creating XQConnector instances
template <class T, class A, class B, typename C>
xqcon_ptr xqcon_create(const shared_ptr<A> &a, B *b, C c) {
    xqcon_ptr pHolder(new _XQConnectorHolder( new
          T(a, b, c)));        
    return pHolder;
}
//! function for creating XQConnector instances
template <class T, class A, class B, typename C, typename D>
xqcon_ptr xqcon_create(const shared_ptr<A> &a, B *b, C c, D d) {
    xqcon_ptr pHolder(new _XQConnectorHolder( new
          T(a, b, c, d)));        
    return pHolder;
}

#define XQCON_OBJECT template <class T, class A, class B> \
friend xqcon_ptr xqcon_create(const shared_ptr<A> &a, B *b); \
template <class T, class A, class B, typename C> \
friend xqcon_ptr xqcon_create(const shared_ptr<A> &a, B *b, C c); \
template <class T, class A, class B, typename C, typename D> \
friend xqcon_ptr xqcon_create(const shared_ptr<A> &a, B *b, C c, D d); \
friend class _XQConnectorHolder;


#endif /*XNODECONNECTOR_PRV_H_*/
