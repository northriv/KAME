/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp

		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.

		You should have received a copy of the GNU Library General
		Public License and a list of authors along with this program;
		see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef ATOMIC_H_
#define ATOMIC_H_

#include "atomic_prv_basic.h"
#include "atomic_smart_ptr.h"

#include <type_traits>

//! Atomic access for a copy-able class which does not require transactional writing.
template <typename T, class Enable>
class atomic {
public:
    atomic() : m_var(new T) {}
    atomic(T t) : m_var(new T(t)) {}
    atomic(const atomic &t) : m_var(t.m_var) {}
    ~atomic() {}
    operator T() const {
        local_shared_ptr<T> x = m_var;
        return *x;
    }
    atomic &operator=(T t) {
        m_var.reset(new T(t));
        return *this;
    }
    atomic &operator=(const atomic &x) {
        m_var = x.m_var;
        return *this;
    }
    bool compare_set_strong(const T &oldv, const T &newv) {
        local_shared_ptr<T> oldx(m_var);
        if( *oldx != oldv)
            return false;
        local_shared_ptr<T> newx(new T(newv));
        bool ret = m_var.compareAndSet(oldx, newx);
        return ret;
    }
protected:
    atomic_shared_ptr<T> m_var;
};

#endif /*ATOMIC_H_*/
