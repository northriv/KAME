#ifndef FAST_SMART_PTR_H_
#define FAST_SMART_PTR_H_

#include "atomic.h"

//! Faster implementation of \a boost::smart_ptr.
//! Code itself is not copied from \a boost, however, ideas are copied.
namespace KAME {

template <typename T>
class scoped_ptr
{
 public:
    scoped_ptr() : m_ptr(0) {}
    
    explicit scoped_ptr(T *t) : m_ptr(t) {}
    
    ~scoped_ptr() { delete m_ptr;}

    void reset(T *t = 0) {
        delete m_ptr;
        m_ptr = t;
    }
    void swap(scoped_ptr &x) {
        T *tmp = x.m_ptr;
        x.m_ptr = m_ptr;
        m_ptr = tmp;
    }
    
    T &operator*() const { ASSERT(m_ptr); return (T&)*m_ptr;}

    T *operator->() const { ASSERT(m_ptr); return (T*)m_ptr;}
    
    T *get() const {
        return (T*)m_ptr;
    }
 private:
    T *m_ptr;
};

struct _DeleterBase {
    virtual ~_DeleterBase() {}
};
template <typename T>
struct _StandardDeleter : public _DeleterBase {
    _StandardDeleter(T *p) : m_ptr(p) {}
    virtual ~_StandardDeleter() {
        delete m_ptr;
    }
    T *m_ptr;
};
template <typename T, typename D>
struct _CustomDeleter : public _DeleterBase {
    _CustomDeleter(T *p, D d) : m_deleter(d), m_ptr(p) {}
    virtual ~_CustomDeleter() {
        m_deleter(m_ptr);
    }
    D m_deleter;
    T *m_ptr;
};

//! Instance of _ReferenceCounter is only one per one object.
struct _ReferenceCounter {
    template <typename T>
    _ReferenceCounter(T *p) : refcnt(1), weakrefcnt(1), deleter(new _StandardDeleter<T>(p)) {}
    template <typename T, typename D>
    _ReferenceCounter(T *p, D d)
         : refcnt(1), weakrefcnt(1), deleter(new _CustomDeleter<T, D>(p, d)) {}
    unsigned int refcnt;
    //! weakrefcnt + (refcnt) ? 1 : 0.
    unsigned int weakrefcnt;
    _DeleterBase *deleter;
};
struct _DynamicCastFlag {};

template <typename T> class enable_shared_from_this;
template<typename T> class shared_ptr;
template<typename T> class weak_ptr;

template <typename Y, typename T>
inline void _enable_weak_from_this(shared_ptr<Y> *s, enable_shared_from_this<T>*p) {
    p->m_weak_to_this = *s;
}
inline void _enable_weak_from_this(void *, void *) {}

template <class _T, class Y>
shared_ptr<_T> dynamic_pointer_cast(const shared_ptr<Y>&);
  
template <typename T>
class shared_ptr
{
    typedef _ReferenceCounter Ref;
 public:
    shared_ptr() : m_pref(0), m_ptr(0) {}
    
    template<typename Y> explicit shared_ptr(Y *y) : m_pref(new Ref(y)), m_ptr(y) {
        _enable_weak_from_this(this, y);
    }
    template<typename Y, typename D> explicit shared_ptr(Y *y, D d) : m_pref(new Ref(y, d)), m_ptr(y) {
        _enable_weak_from_this(this, y);
    }
    
    shared_ptr(const shared_ptr &t) : m_pref(t._scan_()), m_ptr(t.m_ptr) {}
    template<typename Y> shared_ptr(const shared_ptr<Y> &y) : m_pref(y._scan_()), m_ptr(y.get()) {}
    explicit shared_ptr(const weak_ptr<T> &t) : m_pref(t._scan_()), m_ptr(t.m_ptr) {}

    ~shared_ptr();

    shared_ptr &operator=(const shared_ptr &t) {
        shared_ptr(t).swap(*this);
        return *this;
    }
    template<typename Y> shared_ptr &operator=(const shared_ptr<Y> &y) {
        shared_ptr(y).swap(*this);
        return *this;
    }
    void reset() {
        shared_ptr().swap(*this);
    }
    template<typename Y> void reset(Y *y) {
        shared_ptr(y).swap(*this);
    }
    
    //! \param x \p x is atomically swapped.
    //! Nevertheless, this object is not atomically replaced. 
    void swap(shared_ptr &);
    
    //! These functions must be called while writing is blocked.
    T &operator*() const { ASSERT(m_ptr); return *m_ptr;}

    T *operator->() const { ASSERT(m_ptr); return m_ptr;}
    
    T *get() const { return m_ptr; }

    bool operator!() const {return !m_ptr;}
    typedef T *(shared_ptr<T>::*fake_bool)() const;
    operator fake_bool() const {return m_ptr ? (&shared_ptr<T>::get) : 0;}    

    //! atomically increase local reference count.
    Ref *_scan_() const {
        if(m_pref) atomicInc(&m_pref->refcnt);
        return m_pref;
    }
 private:
    friend class weak_ptr<T>;
    template <class _T, class Y>
    friend shared_ptr<_T> dynamic_pointer_cast(const shared_ptr<Y>&);

    template<typename Y> shared_ptr(const shared_ptr<Y> &y, _DynamicCastFlag)
     : m_ptr(dynamic_cast<T*>(y.get())) {
        if(m_ptr) m_pref = y._scan_();
     }

    Ref *m_pref;
    T *m_ptr;
};

template<class X, class Y>
bool operator<(const shared_ptr<X> &x, const shared_ptr<Y>&y) {
    return x.get() < y.get();
}
template<class X, class Y>
bool operator==(const shared_ptr<X> &x, const shared_ptr<Y>&y) {
    return x.get() == y.get();
}
template<class X, class Y>
bool operator!=(const shared_ptr<X> &x, const shared_ptr<Y>&y) {
    return x.get() != y.get();
}

template<typename T>
class weak_ptr
{
    typedef _ReferenceCounter Ref;
    friend class shared_ptr<T>;
 public:
    weak_ptr() : m_pref(0), m_ptr(0) {}
    
    weak_ptr(const weak_ptr &r) : m_pref(r._scan_()), m_ptr(r._get_()) {}
    template<typename Y> weak_ptr(const weak_ptr<Y> &r) : m_pref(r._scan_()), m_ptr(r._get_()) {}
    
    template<typename Y> weak_ptr(const shared_ptr<Y> &r) : m_pref(r._scan_()), m_ptr(r.get()) {}
        
    ~weak_ptr();
    
    weak_ptr& operator=(const weak_ptr &r) {
        weak_ptr(r).swap(*this);
        return *this;
    }
    template<typename Y> weak_ptr& operator=(const weak_ptr<Y> &r) {
        weak_ptr(r).swap(*this);
        return *this;
    }
    template<typename Y> weak_ptr& operator=(const shared_ptr<Y> &r) {
        weak_ptr(r).swap(*this);
        return *this;
    }
    shared_ptr<T> lock() const {
        return expired() ? shared_ptr<T>() : shared_ptr<T>(*this);
    }
    void reset() {
        weak_ptr().swap(*this);
    }
    bool expired() const {return m_pref && (m_pref->weakrefcnt > 1);}
    void swap(weak_ptr<T> &b);

    //! atomically increase local reference count.
    Ref *_scan_() const {
        if(m_pref) atomicInc(&m_pref->weakrefcnt);
        return m_pref;
    }
    T *_get_() const {return m_ptr;}
 private:
    Ref *m_pref;
    T *m_ptr;
};

template <typename T>
shared_ptr<T>::~shared_ptr() {
    Ref *pref = m_pref;
    if(!pref) return;
    if(atomicDecAndTest(&pref->refcnt)) {
        if(pref->deleter)
            delete pref->deleter;
        if(atomicDecAndTest(&pref->weakrefcnt)) {
            delete pref;
        }
    }
}

template <typename T>
void
shared_ptr<T>::swap(shared_ptr<T> &r) {
    Ref *pref = r.m_pref;
    r.m_pref = m_pref;
    m_pref = pref;
    T *ptr = r.m_ptr;
    r.m_ptr = m_ptr;
    m_ptr = ptr;
}

template <typename T>
weak_ptr<T>::~weak_ptr() {
    Ref *pref = m_pref;
    if(!pref) return;
    if(atomicDecAndTest(&pref->weakrefcnt)) {
        delete pref;
    }
}

template <typename T>
void
weak_ptr<T>::swap(weak_ptr<T> &r) {
    Ref *pref;
    pref = r.m_pref;
    r.m_pref = m_pref;
    m_pref = pref;
}

template <class T>
class enable_shared_from_this
{
protected:
    enable_shared_from_this() {}
    enable_shared_from_this(const enable_shared_from_this &) {}
    enable_shared_from_this & operator=(const enable_shared_from_this &) {return *this;}
public:
     shared_ptr<T> shared_from_this() {
        return shared_ptr<T>(m_weak_to_this);
     }
     shared_ptr<const T> shared_from_this() const {
        return shared_ptr<const T>(m_weak_to_this);
     }
private:
    template <typename Y, typename A>
    friend void _enable_weak_from_this(shared_ptr<Y> *s, enable_shared_from_this<A>*p);
    weak_ptr<T> m_weak_to_this;
};


template <class T, class Y>
inline shared_ptr<T> dynamic_pointer_cast(const shared_ptr<Y>& r) {
    return shared_ptr<T>(r, _DynamicCastFlag());
}


} //namesepace KAME
#endif /*FAST_SMART_PTR_H_*/
