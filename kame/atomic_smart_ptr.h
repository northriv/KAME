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
#ifndef ATOMIC_SMART_PTR_H_
#define ATOMIC_SMART_PTR_H_

#include <atomic.h>

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
    	writeBarrier();
    }
    
    ~atomic_scoped_ptr() { readBarrier(); delete m_ptr;}

    void reset(t_ptr t = 0) {
        writeBarrier();
        t_ptr old = atomicSwap(t, &m_ptr);
        readBarrier();
        delete old;
    }
    //! \param x \p x is atomically swapped.
    //! Nevertheless, this object is not atomically replaced. 
    void swap(atomic_scoped_ptr &x) {
        writeBarrier();
        m_ptr = atomicSwap(m_ptr, &x.m_ptr);
        readBarrier();
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
    	m_ref.pref = 0;
    	m_ref.refcnt_n_serial = 0;
    }
    
    template<typename Y> explicit atomic_shared_ptr(Y *y) {
    	m_ref.pref = new Ref(y);
    	m_ref.refcnt_n_serial = 0;
        m_ptr_instant = y;
        writeBarrier();
    }
    
    atomic_shared_ptr(const atomic_shared_ptr &t) {
    	m_ref.pref = t._scan_();
    	m_ref.refcnt_n_serial = 0;
        m_ptr_instant = !m_ref.pref ? 0 : m_ref.pref->ptr;
    }
    template<typename Y> atomic_shared_ptr(const atomic_shared_ptr<Y> &y) {
    	m_ref.pref = (typename atomic_shared_ptr::Ref*)y._scan_();
    	m_ref.refcnt_n_serial = 0;
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

	typedef uint16_t Serial;
	typedef uint_cas2_each RefcntNSerial;
	static inline uint_cas2_each _refcnt(RefcntNSerial x) {return x / (1uL << 8 * sizeof(uint16_t));}
	static inline Serial _serial(RefcntNSerial x) {return x % (1uL << 8 * sizeof(uint16_t));}
	static inline RefcntNSerial _combine(uint_cas2_each ref, Serial serial) {
		return ref * (1uL << 8 * sizeof(uint16_t)) + (serial % (1uL << 8 * sizeof(uint16_t)));
	}
public:
    //! internal functions below.
    //! atomically scan \a Ref and increase global reference counter.
    Ref *_scan_() const;
    //! atomically scan \a Ref and increase local reference counter.
    Ref *_reserve_scan_(RefcntNSerial *) const;    
    //! try to decrease local reference counter.
    void _leave_scan_(Ref *, Serial serial) const;    
private:
    struct _RefLocal {
        //! A pointer to global reference struct.
        Ref* pref;
	 	//! Local reference counter w/ serial number for ABA problem.
        RefcntNSerial refcnt_n_serial;
    };
    mutable _RefLocal m_ref;
};

template <typename T>
atomic_shared_ptr<T>::~atomic_shared_ptr() {
	readBarrier();
	ASSERT(_refcnt(m_ref.refcnt_n_serial) == 0);
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
    atomicInc(&pref->refcnt);
    _leave_scan_(pref, _serial(rs));
    return pref;
}

template <typename T>
typename atomic_shared_ptr<T>::Ref *
atomic_shared_ptr<T>::_reserve_scan_(RefcntNSerial *rs) const {
	Ref *pref;
	RefcntNSerial rs_new;
	for(;;) {
		pref = m_ref.pref;
		RefcntNSerial rs_old;
		rs_old = m_ref.refcnt_n_serial;
		if(!pref) {
			// target is null.
			*rs = rs_old;
			return 0;
		}
		rs_new = _combine(_refcnt(rs_old) + 1, _serial(rs_old));
		// try to increase local reference counter w/ same serial.
		if(atomicCompareAndSet2(
			   (unsigned int)pref, rs_old,
			   (unsigned int)pref, rs_new,
			   (unsigned int*)&m_ref))
			break;
	}
	*rs = rs_new;
	return pref;
}
template <typename T>
void
atomic_shared_ptr<T>::_leave_scan_(Ref *pref, Serial serial) const {
	for(;;) {
		RefcntNSerial rs_old;
		rs_old = _combine(_refcnt(m_ref.refcnt_n_serial), serial);
		RefcntNSerial rs_new = _combine(_refcnt(rs_old) - 1, _serial(rs_old));
		// try to dec. reference counter and change serial if stored pointer is unchanged.
		if(atomicCompareAndSet2(
			   (unsigned int)pref, rs_old,
			   (unsigned int)pref, rs_new,
			   (unsigned int*)&m_ref))
			break;
		if((pref != m_ref.pref) || (serial != _serial(m_ref.refcnt_n_serial))) {
			// local reference of this context has released by other processes.
			if(atomicDecAndTest((int*)&pref->refcnt)) {
				readBarrier();
				delete pref;
			}
			break;
		}
	}
	writeBarrier();
}

template <typename T>
void
atomic_shared_ptr<T>::swap(atomic_shared_ptr<T> &r) {
	Ref *pref;
	T *oldptr = 0;
	if(m_ref.pref) {
		// Release local reference.
		unsigned int refcnt = _refcnt(m_ref.refcnt_n_serial);
		if(refcnt)
		    atomicAdd(&m_ref.pref->refcnt, refcnt);
		m_ref.refcnt_n_serial = _combine(0, _serial(m_ref.refcnt_n_serial) + 1);
		oldptr = m_ref.pref->ptr;
	}
	else
		m_ref.refcnt_n_serial = _combine(0, _serial(m_ref.refcnt_n_serial));
	writeBarrier();
	for(;;) {
		RefcntNSerial rs_old, rs_new;
		pref = r._reserve_scan_(&rs_old);
		if(pref) {
			ASSERT(_refcnt(rs_old));
			atomicAdd(&pref->refcnt, (unsigned int)(_refcnt(rs_old) - 1));
		}
		rs_new = _combine(0, _refcnt(rs_old) + 1);
		if(atomicCompareAndSet2(
			   (unsigned int)pref, rs_old,
			   (unsigned int)m_ref.pref, rs_new,
			   (unsigned int*)&r.m_ref))
			break;
		if(pref) {
			ASSERT(_refcnt(rs_old));
			atomicAdd((int*)&pref->refcnt, - (int)(_refcnt(rs_old) - 1));
			r._leave_scan_(pref, _serial(rs_old));
		}
	}
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
    	unsigned int refcnt = _refcnt(m_ref.refcnt_n_serial);
		// Release local reference.
        if(refcnt)
            atomicAdd(&m_ref.pref->refcnt, refcnt);
		m_ref.refcnt_n_serial = _combine(0, _serial(m_ref.refcnt_n_serial) + 1);
        oldptr = m_ref.pref->ptr;
    }
	else
		m_ref.refcnt_n_serial = _combine(0, _serial(m_ref.refcnt_n_serial));
	writeBarrier();
	for(;;) {
		RefcntNSerial rs_old, rs_new;
		pref = r._reserve_scan_(&rs_old);
		if(pref != oldr.m_ref.pref) {
			if(pref)
				r._leave_scan_(pref, _serial(rs_old));
			return false;
		}
		if(pref) {
			ASSERT(_refcnt(rs_old));
			atomicAdd(&pref->refcnt, (unsigned int)(_refcnt(rs_old)- 1));
		}
		rs_new = _combine(0, _refcnt(rs_old) + 1);
		if(atomicCompareAndSet2(
			   (unsigned int)pref, rs_old,
			   (unsigned int)m_ref.pref, rs_new,
			   (unsigned int*)&r.m_ref))
			break;
		if(pref) {
			ASSERT(_refcnt(rs_old));
			atomicAdd((int*)&pref->refcnt, - (int)(_refcnt(rs_old)- 1));
			r._leave_scan_(pref, _serial(rs_old));
		}
	}
	m_ref.pref = pref;
	m_ptr_instant = !m_ref.pref ? 0 : m_ref.pref->ptr;
	r.m_ptr_instant = oldptr;
	return true;
}

template <typename T, class Enable>
class atomic
{
public:
	atomic() : m_var(new T) {}
	atomic(T t) : m_var(new T(t)) {}
	atomic(const atomic &t) : m_var(t) {}
	~atomic() {}
	operator T() const {
	    atomic_shared_ptr<T> x = m_var;
		return *x;
	}
	atomic &operator=(T t) {
        m_var.reset(new T(t));
		return *this;
	}
	T swap(T newv) {
	    atomic_shared_ptr<T> x(newv);
		x.swap(m_var);
		return *x;
	}
protected:
	atomic_shared_ptr<T> m_var;
};

#endif /*ATOMIC_SMART_PTR_H_*/
