#ifndef ATOMIC_SMART_PTR_H_
#define ATOMIC_SMART_PTR_H_

#include "atomic.h"

#ifdef HAVE_CAS_2
    #define ATOMIC_SMART_PTR_USE_CAS2
#else
    #define ATOMIC_SMART_PTR_USE_LOCKFREE_READ
    #include "thread.h"
#endif

//! This is improved version of boost::scoped_ptr<>.
//! This object itself is not atomically changed.
//! However, targeted object ( argument of swap() ) can be atomically swapped.
//! Therefore, atomic_scoped_ptr<> can be shared among threads by the use of swap() as a argument.
//! That is, destructive read. Use atomic_shared_ptr<> for non-destructive reading.
template <typename T>
class atomic_scoped_ptr
{
 public:
    atomic_scoped_ptr() : m_ptr(0) {
    }
    
    explicit atomic_scoped_ptr(T *t) : m_ptr(t) {
    }
    
    ~atomic_scoped_ptr() { readBarrier(); delete m_ptr;}

    void reset(T *t = 0) {
        T *old = atomicSwap(t, &m_ptr);
        readBarrier();
        delete old;
    }
    //! \param x \p x is atomically swapped.
    //! Nevertheless, this object is not atomically replaced. 
    void swap(atomic_scoped_ptr &x) {
        m_ptr = atomicSwap(m_ptr, &x.m_ptr);
        readBarrier();
    }
    
    //! These functions must be called while writing is blocked.
    T &operator*() const { ASSERT(m_ptr); return (T&)*m_ptr;}

    T *operator->() const { ASSERT(m_ptr); return (T*)m_ptr;}
    
    T *get() const {
        return (T*)m_ptr;
    }
 private:
    atomic_scoped_ptr(const atomic_scoped_ptr &) {}
    atomic_scoped_ptr& operator=(const atomic_scoped_ptr &) {return *this;}
    T *m_ptr;
};

template <typename T>
class atomic_shared_ptr
{
    struct Ref;
 public:
    atomic_shared_ptr() : m_ptr_instant(0) {
    }
    
    template<typename Y> explicit atomic_shared_ptr(Y *y) : m_ref(new Ref(y)) {
        m_ptr_instant = y;
    }
    
    atomic_shared_ptr(const atomic_shared_ptr &t) : m_ref(t._scan_()) {
        m_ptr_instant = !m_ref.pref ? 0 : m_ref.pref->ptr;
    }
    template<typename Y> atomic_shared_ptr(const atomic_shared_ptr<Y> &y)
     : m_ref((typename atomic_shared_ptr::Ref*)y._scan_()) {
        m_ptr_instant = !m_ref.pref ? 0 : ((typename atomic_shared_ptr<Y>::Ref *)m_ref.pref)->ptr;
    }

    ~atomic_shared_ptr();

    atomic_shared_ptr &operator=(const atomic_shared_ptr &t) {
        atomic_shared_ptr(t).swap(*this);
        return *this;
    }
    template<typename Y> atomic_shared_ptr &operator=(const atomic_shared_ptr<Y> &y) {
        atomic_shared_ptr(y).swap(*this);
        return *this;
    }
    void reset() {
        atomic_shared_ptr().swap(*this);
    }
    template<typename Y> void reset(Y *y) {
        atomic_shared_ptr(y).swap(*this);
    }
    
    //! \param x \p x is atomically swapped.
    //! Nevertheless, this object is not atomically replaced. 
    void swap(atomic_shared_ptr &);

    //! \return true if set.
    bool compareAndSwap(const atomic_shared_ptr &oldr, atomic_shared_ptr &r);
    
    //! These functions must be called while writing is blocked.
    T &operator*() const { ASSERT(m_ptr_instant); return *m_ptr_instant;}

    T *operator->() const { ASSERT(m_ptr_instant); return m_ptr_instant;}
    
    T *get() const {
        return m_ptr_instant;
    }

    bool operator!() const {return !m_ptr_instant;}
    operator bool() const {return m_ptr_instant;}    

 private:
    //! for instant (not atomic) access.
    T *m_ptr_instant;
    //! Instance of Ref is only one per one object.
    struct Ref {
        template <class Y>
        Ref(Y *p) : ptr(p), refcnt(1) {}
        ~Ref() { ASSERT(refcnt == 0); delete ptr; }
        T *ptr;
        unsigned int refcnt;
    };

    typedef union {
        struct {
        #if SIZEOF_INT == 4
            uint16_t refcnt, serial; 
        #elif SIZEOF_INT == 8
            uint32_t refcnt, serial; 
        #endif
        } split;
        unsigned int both;
     } RefcntNSerial;
 public:
    //! internal functions below.
    //! atomically scan \a Ref and increase global reference counter.
    Ref *_scan_() const;
    //! atomically scan \a Ref and increase local reference counter.
    Ref *_reserve_scan_(RefcntNSerial *) const;    
    //! try to decrease local reference counter.
    void _leave_scan_(Ref *, unsigned int serial) const;    
 private:
    struct _RefLocal {
        _RefLocal(Ref *r) : pref(r)
        #if defined ATOMIC_SMART_PTR_USE_LOCKFREE_READ
            , pref_old(r), serial_update(0)
        #endif
            { refcnt_n_serial.both = 0; }
        _RefLocal() : pref(0)
        #if defined ATOMIC_SMART_PTR_USE_LOCKFREE_READ
            , pref_old(0), serial_update(0)
        #endif
            { refcnt_n_serial.both = 0; }
        Ref *pref;
        RefcntNSerial refcnt_n_serial;
        #ifdef ATOMIC_SMART_PTR_USE_LOCKFREE_READ
            Ref *pref_old;
            unsigned int serial_update;
        #endif
    };
    mutable _RefLocal m_ref;
    #ifdef ATOMIC_SMART_PTR_USE_LOCKFREE_READ
        XMutex m_updatemutex;
    #endif
};

template <typename T>
atomic_shared_ptr<T>::~atomic_shared_ptr() {
    readBarrier();
    ASSERT(m_ref.refcnt_n_serial.split.refcnt == 0);
    Ref *pref = m_ref.pref;
    if(!pref) return;
    if(atomicDecAndTest(&pref->refcnt)) {
        delete pref;
    }
}

template <typename T>
typename atomic_shared_ptr<T>::Ref *atomic_shared_ptr<T>::_scan_() const {
    RefcntNSerial rs;
    Ref *pref = _reserve_scan_(&rs);
    if(!pref) return 0;
    atomicInc(&pref->refcnt);
    _leave_scan_(pref, rs.split.serial);
    readBarrier();
    return pref;
}

#ifdef ATOMIC_SMART_PTR_USE_CAS2
template <typename T>
typename atomic_shared_ptr<T>::Ref *
atomic_shared_ptr<T>::_reserve_scan_(RefcntNSerial *rs) const {
    Ref *pref;
    RefcntNSerial rs_new;
    for(;;) {
        pref = m_ref.pref;
        RefcntNSerial rs_old = m_ref.refcnt_n_serial;
        if(!pref) {
            *rs = rs_old;
            return 0;
        }
        rs_new = rs_old;
        rs_new.split.refcnt++;
        C_ASSERT(sizeof(m_ref) == sizeof(unsigned int) * 2);
        if(atomicCompareAndSet2(
            (unsigned int)pref, rs_old.both,
            (unsigned int)pref, rs_new.both,
                 (unsigned int*)&m_ref))
                    break;
    }
    *rs = rs_new;
    return pref;
}
template <typename T>
void
atomic_shared_ptr<T>::_leave_scan_(Ref *pref, unsigned int serial) const {
    for(;;) {
        RefcntNSerial rs_old = m_ref.refcnt_n_serial;
        rs_old.split.serial = serial;
        RefcntNSerial rs_new = rs_old;
        rs_new.split.refcnt--;
        if(atomicCompareAndSet2(
            (unsigned int)pref, rs_old.both,
            (unsigned int)pref, rs_new.both,
                 (unsigned int*)&m_ref))
                    break;
        if((pref != m_ref.pref) || (serial != m_ref.refcnt_n_serial.split.serial)) {
            if(atomicDecAndTest((int*)&pref->refcnt))
                delete pref;
            break;
        }
    }
}

#elif defined ATOMIC_SMART_PTR_USE_LOCKFREE_READ

template <typename T>
typename atomic_shared_ptr<T>::Ref *
atomic_shared_ptr<T>::_reserve_scan_(RefcntNSerial *rs) const {
    Ref *pref;
    RefcntNSerial rs_new;
    for(;;) {
        readBarrier();
        RefcntNSerial rs1 = m_ref.refcnt_n_serial;
        readBarrier();
        Ref *pref_new = m_ref.pref;
        readBarrier();
        unsigned int serial1 = m_ref.serial_update;
        readBarrier();
        Ref *pref_old = m_ref.pref_old;
        readBarrier();
        RefcntNSerial rs2 = m_ref.refcnt_n_serial;
        if((rs1.split.serial != rs2.split.serial)) continue;
        rs_new = rs2;
        if(rs2.split.serial == serial1) {
            pref = pref_new;
        }
        else {
            pref = pref_old;
        }
        if(!pref) {
            break;
        }
        rs_new.split.refcnt++;
        if(atomicCompareAndSet(rs2.both, rs_new.both, &m_ref.refcnt_n_serial.both))
            break;
    }
    *rs = rs_new;
    return pref;
}
template <typename T>
void
atomic_shared_ptr<T>::_leave_scan_(Ref *pref, unsigned int serial) const {
    for(;;) {
        RefcntNSerial rs_new;

        readBarrier();
        RefcntNSerial rs1 = m_ref.refcnt_n_serial;
        readBarrier();
        Ref *pref_new = m_ref.pref;
        readBarrier();
        unsigned int serial1 = m_ref.serial_update;
        readBarrier();
        Ref *pref_old = m_ref.pref_old;
        readBarrier();
        RefcntNSerial rs2 = m_ref.refcnt_n_serial;
        if((rs1.split.serial != rs2.split.serial)) continue;
        rs_new = rs2;
        Ref *pref_now;
        if(rs2.split.serial == serial1) {
            pref_now = pref_new;
        }
        else {
            pref_now = pref_old;
        }
        if((pref != pref_now) || (rs2.split.serial != serial)) {
            if(atomicDecAndTest((int*)&pref->refcnt))
                delete pref;
            break;
        }
        rs_new.split.refcnt--;
        if(atomicCompareAndSet(rs2.both, rs_new.both, &m_ref.refcnt_n_serial.both))
            break;
    }
}

#endif
    
template <typename T>
void
atomic_shared_ptr<T>::swap(atomic_shared_ptr<T> &r) {
    Ref *pref;
    if(m_ref.pref) {
        if(m_ref.refcnt_n_serial.split.refcnt)
            atomicAdd(&m_ref.pref->refcnt, (unsigned int)m_ref.refcnt_n_serial.split.refcnt);
        m_ref.refcnt_n_serial.split.refcnt = 0;
        m_ref.refcnt_n_serial.split.serial++;
    }
#ifdef ATOMIC_SMART_PTR_USE_CAS2
    for(;;) {
        RefcntNSerial rs_old, rs_new;
        pref = r._reserve_scan_(&rs_old);
        if(pref) {
            if(rs_old.split.refcnt)
                atomicAdd(&pref->refcnt, (unsigned int)rs_old.split.refcnt);
        }
        rs_new.split.refcnt = 0;
        rs_new.split.serial = rs_old.split.serial + 1;
        if(atomicCompareAndSet2(
                (unsigned int)pref, rs_old.both,
                (unsigned int)m_ref.pref, rs_new.both,
                     (unsigned int*)&r.m_ref))
                    break;
        if(pref) {
            if(rs_old.split.refcnt)
                atomicAdd((int*)&pref->refcnt, - (int) rs_old.split.refcnt);
            r._leave_scan_(pref, rs_old.split.serial);
        }
    }
    readBarrier();
#elif defined ATOMIC_SMART_PTR_USE_LOCKFREE_READ
    m_ref.serial_update = m_ref.refcnt_n_serial.split.serial;
    { XScopedLock<XMutex> lock(r.m_updatemutex);
        unsigned int serial = ++r.m_ref.serial_update;
        readBarrier();
        pref = r.m_ref.pref;
        r.m_ref.pref = m_ref.pref;
        readBarrier();
        RefcntNSerial rs_new;
        rs_new.split.serial = serial;
        rs_new.split.refcnt = 0;
        for(;;) {
            RefcntNSerial rs_old = r.m_ref.refcnt_n_serial;
            if(pref)
                if(rs_old.split.refcnt)
                    atomicAdd((int*)&pref->refcnt, (int)rs_old.split.refcnt);
            if(atomicCompareAndSet(rs_old.both, rs_new.both, 
                    &r.m_ref.refcnt_n_serial.both))
                    break;
            if(pref)
                if(rs_old.split.refcnt)
                    atomicAdd((int*)&pref->refcnt, -(int)rs_old.split.refcnt);
         }
        readBarrier();
        r.m_ref.pref_old = m_ref.pref;
    }
    m_ref.pref_old = pref;
#endif 
    m_ref.pref = pref;
    m_ptr_instant = !m_ref.pref ? 0 : m_ref.pref->ptr;
    r.m_ptr_instant = !r.m_ref.pref ? 0 : r.m_ref.pref->ptr;
}

template <typename T>
bool
atomic_shared_ptr<T>::compareAndSwap(const atomic_shared_ptr<T> &oldr, atomic_shared_ptr<T> &r) {
    Ref *pref;
    if(m_ref.pref) {
        if(m_ref.refcnt_n_serial.split.refcnt)
            atomicAdd(&m_ref.pref->refcnt, (unsigned int)m_ref.refcnt_n_serial.split.refcnt);
        m_ref.refcnt_n_serial.split.refcnt = 0;
        m_ref.refcnt_n_serial.split.serial++;
    }
#ifdef ATOMIC_SMART_PTR_USE_CAS2
    for(;;) {
        RefcntNSerial rs_old, rs_new;
        pref = r._reserve_scan_(&rs_old);
        if(pref != oldr.m_ref.pref) {
            r._leave_scan_(pref, rs_old.split.serial);
            return false;
        }
        if(pref) {
            if(rs_old.split.refcnt)
                atomicAdd(&pref->refcnt, (unsigned int)rs_old.split.refcnt);
        }
        rs_new.split.refcnt = 0;
        rs_new.split.serial = rs_old.split.serial + 1;
        if(atomicCompareAndSet2(
                (unsigned int)pref, rs_old.both,
                (unsigned int)m_ref.pref, rs_new.both,
                     (unsigned int*)&r.m_ref))
                    break;
        if(pref) {
            if(rs_old.split.refcnt)
                atomicAdd((int*)&pref->refcnt, - (int) rs_old.split.refcnt);
            r._leave_scan_(pref, rs_old.split.serial);
        }
    }
    readBarrier();
#elif defined ATOMIC_SMART_PTR_USE_LOCKFREE_READ
    m_ref.serial_update = m_ref.refcnt_n_serial.split.serial;
    { XScopedLock<XMutex> lock(r.m_updatemutex);
        pref = r.m_ref.pref;
        if(pref != oldr.m_ref.pref) return false;
        unsigned int serial = ++r.m_ref.serial_update;
        readBarrier();
        r.m_ref.pref = m_ref.pref;
        readBarrier();
        RefcntNSerial rs_new;
        rs_new.split.serial = serial;
        rs_new.split.refcnt = 0;
        for(;;) {
            RefcntNSerial rs_old = r.m_ref.refcnt_n_serial;
            if(pref)
                if(rs_old.split.refcnt)
                    atomicAdd((int*)&pref->refcnt, (int)rs_old.split.refcnt);
            if(atomicCompareAndSet(rs_old.both, rs_new.both, 
                    &r.m_ref.refcnt_n_serial.both))
                    break;
            if(pref)
                if(rs_old.split.refcnt)
                    atomicAdd((int*)&pref->refcnt, -(int)rs_old.split.refcnt);
         }
        readBarrier();
        r.m_ref.pref_old = m_ref.pref;
    }
    m_ref.pref_old = pref;
#endif 
    m_ref.pref = pref;
    m_ptr_instant = !m_ref.pref ? 0 : m_ref.pref->ptr;
    r.m_ptr_instant = !r.m_ref.pref ? 0 : r.m_ref.pref->ptr;
    return true;
}

#endif /*ATOMIC_SMART_PTR_H_*/
