#ifndef ATOMIC_SMART_PTR_H_
#define ATOMIC_SMART_PTR_H_

#include "atomic.h"

#ifdef HAVE_CAS_2
    #define ATOMIC_SMART_PTR_USE_CAS2
#else
    #define ATOMIC_SMART_PTR_USE_LOCKFREE_READ
    #include "thread.h"
#endif

//! This is an improved version of boost::scoped_ptr<>.
//! atomic_scoped_ptr<> can be shared among threads by the use of swap() as the argument.
//! That is, a destructive read. Use atomic_shared_ptr<> for non-destructive reading.
template <typename T>
class atomic_scoped_ptr
{
    typedef T* t_ptr;
 public:
    atomic_scoped_ptr() : m_ptr(0) {
    }
    
    explicit atomic_scoped_ptr(t_ptr t) : m_ptr(t) {
    }
    
    ~atomic_scoped_ptr() { readBarrier(); delete m_ptr;}

    void reset(t_ptr t = 0) {
        t_ptr old = atomicSwap(t, &m_ptr);
        readBarrier();
        delete old;
    }
    //! \param x \p x is atomically swapped.
    //! Nevertheless, this object is not atomically replaced. 
    void swap(atomic_scoped_ptr &x) {
        m_ptr = atomicSwap(m_ptr, &x.m_ptr);
        memoryBarrier();
    }
    
    //! These functions must be called while writing is blocked.
    T &operator*() const { ASSERT(m_ptr); return (T&)*m_ptr;}

    t_ptr operator->() const { ASSERT(m_ptr); return (t_ptr)m_ptr;}
    
    t_ptr get() const {
        return (t_ptr )m_ptr;
    }
 private:
    atomic_scoped_ptr(const atomic_scoped_ptr &) {}
    atomic_scoped_ptr& operator=(const atomic_scoped_ptr &) {return *this;}
    
    t_ptr m_ptr;
};

//! Instance of Ref is only one per one object.
template <typename T>
struct atomic_shared_ptr_ref {
    template <class Y>
    atomic_shared_ptr_ref(Y *p) : ptr(p), refcnt(1) {}
    ~atomic_shared_ptr_ref() { ASSERT(refcnt == 0); delete ptr; }
    T *ptr;
    //! Global reference counter.
    unsigned int refcnt;
};


//! This is an improved version of boost::shared_ptr<>.
//! atomic_shared_ptr<> can be shared among threads by the use of operator=(), swap() as the argument.
template <typename T>
class atomic_shared_ptr
{
 public:
    typedef atomic_shared_ptr_ref<T> Ref;
    
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

    //! \param t This object is atomically replaced with \a t.
    atomic_shared_ptr &operator=(const atomic_shared_ptr &t) {
        atomic_shared_ptr(t).swap(*this);
        return *this;
    }
    //! \param y This object is atomically replaced with \a t.
    template<typename Y> atomic_shared_ptr &operator=(const atomic_shared_ptr<Y> &y) {
        atomic_shared_ptr(y).swap(*this);
        return *this;
    }
    //! This object is atomically reset.
    void reset() {
        atomic_shared_ptr().swap(*this);
    }
    //! \param y This object is atomically reset with a pointer \a y.
    template<typename Y> void reset(Y *y) {
        atomic_shared_ptr(y).swap(*this);
    }
    
    //! \param x \p x is atomically swapped.
    //! Nevertheless, this object is not atomically replaced. 
    void swap(atomic_shared_ptr &);

    //! \return true if succeeded.
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
        typedef Ref* Ref_ptr;
        //! A pointer to global reference struct.
        Ref_ptr pref;
	 	//! Local reference counter w/ serial number for ABA problem.
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
	    readBarrier();
        delete pref;
    }
}

template <typename T>
typename atomic_shared_ptr<T>::Ref *atomic_shared_ptr<T>::_scan_() const {
    RefcntNSerial rs;
    Ref *pref = _reserve_scan_(&rs);
    if(!pref) return 0;
    memoryBarrier();
    atomicInc(&pref->refcnt);
    _leave_scan_(pref, rs.split.serial);
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
        RefcntNSerial rs_old;
        rs_old.both = m_ref.refcnt_n_serial.both;
        if(!pref) {
        	// target is null.
        	rs->both = rs_old.both;
            return 0;
        }
        rs_new = rs_old;
        rs_new.split.refcnt++;
        C_ASSERT(sizeof(m_ref) == sizeof(unsigned int) * 2);
        // try to increase local reference counter w/ same serial.
        if(atomicCompareAndSet2(
            (unsigned int)pref, rs_old.both,
            (unsigned int)pref, rs_new.both,
                 (unsigned int*)&m_ref))
                    break;
    }
    rs->both = rs_new.both;
    return pref;
}
template <typename T>
void
atomic_shared_ptr<T>::_leave_scan_(Ref *pref, unsigned int serial) const {
    for(;;) {
        RefcntNSerial rs_old;
        rs_old.both = m_ref.refcnt_n_serial.both;
        rs_old.split.serial = serial;
        RefcntNSerial rs_new = rs_old;
        rs_new.split.refcnt--;
    	// try to dec. reference counter and change serial if stored pointer is unchanged.
        if(atomicCompareAndSet2(
            (unsigned int)pref, rs_old.both,
            (unsigned int)pref, rs_new.both,
                 (unsigned int*)&m_ref))
                    break;
        if((pref != m_ref.pref) || (serial != m_ref.refcnt_n_serial.split.serial)) {
        	// local reference of this context has released by other processes.
            if(atomicDecAndTest((int*)&pref->refcnt)) {
			    readBarrier();
                delete pref;
            }
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
    T *oldptr = 0;
    if(m_ref.pref) {
    // Release local reference.
        if(m_ref.refcnt_n_serial.split.refcnt)
            atomicAdd(&m_ref.pref->refcnt, (unsigned int)m_ref.refcnt_n_serial.split.refcnt);
        m_ref.refcnt_n_serial.split.serial++;
        oldptr = m_ref.pref->ptr;
    }
    m_ref.refcnt_n_serial.split.refcnt = 0;
    memoryBarrier();
#ifdef ATOMIC_SMART_PTR_USE_CAS2
    for(;;) {
        RefcntNSerial rs_old, rs_new;
        pref = r._reserve_scan_(&rs_old);
        if(pref) {
            ASSERT(rs_old.split.refcnt);
            memoryBarrier();
            atomicAdd(&pref->refcnt, (unsigned int)(rs_old.split.refcnt - 1));
        }
        rs_new.split.refcnt = 0;
        rs_new.split.serial = rs_old.split.serial + 1;
        if(atomicCompareAndSet2(
                (unsigned int)pref, rs_old.both,
                (unsigned int)m_ref.pref, rs_new.both,
                     (unsigned int*)&r.m_ref))
                    break;
        if(pref) {
            ASSERT(rs_old.split.refcnt);
            atomicAdd((int*)&pref->refcnt, - (int)(rs_old.split.refcnt - 1));
            r._leave_scan_(pref, rs_old.split.serial);
        }
    }
#elif defined ATOMIC_SMART_PTR_USE_LOCKFREE_READ
    m_ref.serial_update = m_ref.refcnt_n_serial.split.serial;
    { XScopedLock<XMutex> lock(r.m_updatemutex);
        unsigned int serial = ++r.m_ref.serial_update;
        writeBarrier();
        pref = r.m_ref.pref;
        r.m_ref.pref = m_ref.pref;
        writeBarrier();
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
        writeBarrier();
        r.m_ref.pref_old = m_ref.pref;
    }
    m_ref.pref_old = pref;
#endif 
    m_ref.pref = pref;
    m_ptr_instant = !m_ref.pref ? 0 : m_ref.pref->ptr;
    r.m_ptr_instant = oldptr;
}

template <typename T>
bool
atomic_shared_ptr<T>::compareAndSwap(const atomic_shared_ptr<T> &oldr, atomic_shared_ptr<T> &r) {
    Ref *pref;
    T *oldptr = 0;
    if(m_ref.pref) {
    // Release local reference.
        if(m_ref.refcnt_n_serial.split.refcnt)
            atomicAdd(&m_ref.pref->refcnt, (unsigned int)m_ref.refcnt_n_serial.split.refcnt);
        m_ref.refcnt_n_serial.split.serial++;
        oldptr = m_ref.pref->ptr;
    }
    m_ref.refcnt_n_serial.split.refcnt = 0;
    memoryBarrier();
#ifdef ATOMIC_SMART_PTR_USE_CAS2
    for(;;) {
        RefcntNSerial rs_old, rs_new;
        pref = r._reserve_scan_(&rs_old);
        if(pref != oldr.m_ref.pref) {
        	if(pref)
	            r._leave_scan_(pref, rs_old.split.serial);
            return false;
        }
        if(pref) {
            ASSERT(rs_old.split.refcnt);
            memoryBarrier();
            atomicAdd(&pref->refcnt, (unsigned int)(rs_old.split.refcnt - 1));
        }
        rs_new.split.refcnt = 0;
        rs_new.split.serial = rs_old.split.serial + 1;
        if(atomicCompareAndSet2(
                (unsigned int)pref, rs_old.both,
                (unsigned int)m_ref.pref, rs_new.both,
                     (unsigned int*)&r.m_ref))
                    break;
        if(pref) {
            ASSERT(rs_old.split.refcnt);
            atomicAdd((int*)&pref->refcnt, - (int)(rs_old.split.refcnt - 1));
            r._leave_scan_(pref, rs_old.split.serial);
        }
    }
#elif defined ATOMIC_SMART_PTR_USE_LOCKFREE_READ
    m_ref.serial_update = m_ref.refcnt_n_serial.split.serial;
    { XScopedLock<XMutex> lock(r.m_updatemutex);
        pref = r.m_ref.pref;
        if(pref != oldr.m_ref.pref) return false;
        unsigned int serial = ++r.m_ref.serial_update;
        writeBarrier();
        r.m_ref.pref = m_ref.pref;
        writeBarrier();
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
        writeBarrier();
        r.m_ref.pref_old = m_ref.pref;
    }
    m_ref.pref_old = pref;
#endif 
    m_ref.pref = pref;
    m_ptr_instant = !m_ref.pref ? 0 : m_ref.pref->ptr;
    r.m_ptr_instant = oldptr;
    return true;
}

#endif /*ATOMIC_SMART_PTR_H_*/
