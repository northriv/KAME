/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This file is dual-licensed under your choice of EITHER:

          * Apache License, Version 2.0
            (http://www.apache.org/licenses/LICENSE-2.0, or see
            LICENSE-APACHE-2.0 in this directory)

        -- OR --

          * GNU General Public License, version 2 of the License,
            or (at your option) any later version
            (http://www.gnu.org/licenses/old-licenses/gpl-2.0.html,
            or see LICENSE-GPL-2.0 in this directory).

        Pick whichever license suits your project.  Unless required
        by applicable law or agreed to in writing, this file is
        distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
        CONDITIONS OF ANY KIND, either express or implied
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
    atomic(const atomic &t) noexcept : m_var(t.m_var) {}
    operator T() const noexcept {
        local_shared_ptr<T> x = m_var;
        return *x;
    }
    atomic &operator=(T t) {
        m_var.reset(new T(t));
        return *this;
    }
    atomic &operator=(const atomic &x) noexcept {
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
