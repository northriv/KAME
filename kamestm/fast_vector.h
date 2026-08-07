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
#ifndef KAMESTM_FAST_VECTOR_H_
#define KAMESTM_FAST_VECTOR_H_

// Transactional::fast_vector<T, N> — small-vector with N elements stored
// inline (T m_array[N], union'd with a std::vector<T> fallback used once
// size > N).  Pure C++17 stdlib; uses std::vector's default allocator —
// NO dependence on any kame pool allocator runtime or symbol.
//
// Lives in kamestm/ because every instantiation is inside kamestm
// (transaction_signal.h, transaction.h, transaction_impl.h); kamepoolalloc
// never instantiated it.  Keeps kamestm headers self-contained for the
// standalone subtree mirror — building kamestm needs nothing under
// kamepoolalloc/.

#include <array>
#include <vector>
#include <limits>
// `fast_vector` below uses `assert()` — explicit `<cassert>` so we don't
// depend on `<vector>` / `<array>` transitively pulling it.  MinGW64 / MSVC
// standard-library implementations do not guarantee the transitive
// include and trip "error: 'assert' was not declared in this scope".
#include <cassert>
#include <stdexcept>
#include <cstddef>

namespace Transactional {

template <typename T, size_t SIZE_HINT = 1>
class fast_vector {
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using pointer = T*;
    using const_pointer = const T*;
    static constexpr size_t max_fixed_size = (8 * sizeof(pointer) <= sizeof(T) * SIZE_HINT) ? SIZE_HINT : (8 * sizeof(pointer) / sizeof(T));
public:
    using iterator = pointer;
    using const_iterator = const_pointer;
    fast_vector() : m_size(0) {}
    ~fast_vector() { destroy(); }
    fast_vector(size_type size) {
        if(size > max_fixed_size) {
            new (&m_vector) std::vector<T>(size);
            m_size = HAS_STD_VECTOR;
        }
        else {
            for(size_type i = 0; i < size; ++i)
                new (m_array + i) T();
            m_size = size;
        }
    }
    fast_vector(const fast_vector &r) : m_size(0) {
        this->operator=(r);
    }
    fast_vector(fast_vector &&r) : m_size(0) {
        this->operator=(std::move(r));
    }
    fast_vector& operator=(const fast_vector &r) {
        destroy();
        if(r.is_fixed()) {
            m_size = r.m_size;
            for(size_type i = 0; i < m_size; ++i) {
                new (m_array + i) T(r.m_array[i]);
            }
        }
        else if(r.m_vector.size() <= max_fixed_size) {
            m_size = r.m_vector.size();
            for(size_type i = 0; i < m_size; ++i) {
                new (m_array + i) T(r.m_vector[i]);
            }
        }
        else {
            m_size = HAS_STD_VECTOR;
            new (&m_vector) std::vector<T>(r.m_vector);
        }
        return *this;
    }
    fast_vector& operator=(fast_vector &&r) {
        destroy();
        if(r.is_fixed()) {
            m_size = r.m_size;
            for(size_type i = 0; i < m_size; ++i) {
                new (m_array + i) T(std::move(r.m_array[i]));
            }
            r.clear_fixed();
        }
        else {
            m_size = HAS_STD_VECTOR;
            new (&m_vector) std::vector<T>(std::move(r.m_vector));
        }
        return *this;
    }
    // Heap mode uses data() rather than &m_vector[n]: operator[] is only
    // valid for n < size(), so `&m_vector[0]` on an empty vector and
    // `&m_vector[m_vector.size()]` for end() are both out-of-range
    // subscripts (UB, and a hard abort under _GLIBCXX_ASSERTIONS).
    // data() / data()+size() are the well-defined spellings.
    iterator begin() noexcept {return is_fixed() ? &m_array[0] : m_vector.data();}
    const_iterator begin() const noexcept {return is_fixed() ? &m_array[0] : m_vector.data();}
    iterator end() noexcept {return is_fixed() ? &m_array[m_size] : m_vector.data() + m_vector.size();}
    const_iterator end() const noexcept {return is_fixed() ? &m_array[m_size] : m_vector.data() + m_vector.size();}
    size_type size() const noexcept {return is_fixed() ? m_size : m_vector.size();}
    bool empty() const noexcept {return !size();}
    reference operator[](size_type n) {return is_fixed() ? m_array[n] : m_vector[n];}
    const_reference operator[](size_type n) const {return is_fixed() ? m_array[n] : m_vector[n];}
    const_reference at(size_type n) const {if(n >= size()) throw std::out_of_range(""); return (*this)[n];}
    reference at(size_type n) {if(n >= size()) throw std::out_of_range(""); return (*this)[n];}
    reference front() {return (*this)[0];}
    const_reference front() const {return (*this)[0];}
    reference back() {return (*this)[this->size() - 1];}
    const_reference back() const {return (*this)[this->size() - 1];}
    void push_back(const T& x) {
        emplace_back(x);
    }
    void push_back(T&& x) {
        emplace_back(std::move(x));
    }
    template <class... Args>
    void emplace_back(Args&&... args) {
        if(m_size < max_fixed_size) {
            new (m_array + m_size) T(std::forward<Args>(args)...);
            ++m_size;
        }
        else {
            if(m_size == max_fixed_size) {
                move_fixed_to_var(m_size);
            }
            m_vector.emplace_back(std::forward<Args>(args)...);
        }
    }
    iterator erase(const_iterator position) {
        if(is_fixed()) {
            for(auto it = const_cast<iterator>(position);;) {
                 auto nit = it + 1;
                 if(nit == end()) {
                     it->~T();
                     break;
                 }
                 else
                     *it = std::move(*nit);
                 it = nit;
            }
            --m_size;
            return const_cast<iterator>(position);
        }
        else {
            auto idx = position - begin();
            m_vector.erase(m_vector.begin() + idx);
            // Not `&*it`: erasing the last element returns end(), and
            // dereferencing it is UB.  data() + idx is the same address,
            // well-defined even when idx == size() (one-past-the-end).
            return m_vector.data() + idx;
        }
    }
//    iterator erase(const_iterator first, const_iterator last);
    void clear() {
        if(is_fixed()) {
            clear_fixed();
        }
        else {
            m_vector.clear();
        }
    }
    void resize(size_type sz) {
        if(is_fixed()) {
            if(sz > max_fixed_size) {
                move_fixed_to_var(sz);
                m_vector.resize(sz);
            }
            else {
                for(size_type i = m_size; i < sz; ++i)
                    new (m_array + i) T();
                for(size_type i = sz; i < m_size; ++i)
                    m_array[i].~T();
                m_size = sz;
            }
        }
        else {
            m_vector.resize(sz);
            // Deliberately NOT shrink_to_fit() here.  It is safe to call now
            // that the is_fixed() test below is right, but it would be a
            // pessimisation: both call sites grow by one
            // (transaction_impl.h, `resize(size() + 1)`), and after a
            // geometric grow the slack exceeds max_fixed_size for any list
            // past a handful of elements -- so every insertion would
            // reallocate, making child insertion O(n) instead of amortized
            // O(1).  Callers that really want the memory back can call
            // shrink_to_fit() explicitly.
        }
    }
    void shrink_to_fit() {
        // Inline (fixed) storage has no spare capacity to release, and
        // m_vector is NOT the active union member here -- touching it at all
        // (even just capacity()/size()) reads m_array's bytes reinterpreted
        // as a std::vector's three pointers.  The test used to be inverted,
        // which returned early exactly when a real vector existed and ran the
        // shrink on garbage otherwise; whether that corrupted memory depended
        // on the stale bytes past the live elements, so it crashed only
        // sporadically (Linux shutdown SIGSEGV via Talker::disconnect).
        if(is_fixed()) return;
        if(m_vector.capacity() - m_vector.size() > max_fixed_size) {
            m_vector.shrink_to_fit();
        }
    }
private:
    void destroy() {
        clear();
        if(!is_fixed())
            m_vector.~vector();
    }
    bool is_fixed() const noexcept {return m_size != (size_type)HAS_STD_VECTOR;}
    void clear_fixed() noexcept {
        assert(is_fixed());
        for(size_type i = 0; i < m_size; ++i)
            m_array[i].~T();
        m_size = 0;
    }
    void move_fixed_to_var(size_type reserve_size) {
        std::vector<T> tmp;
        tmp.reserve(m_size);
        for(size_type i = 0; i < m_size; ++i) {
            tmp.emplace_back(std::move(m_array[i]));
            m_array[i].~T();
        }
        new (&m_vector) std::vector<T>();
        m_vector.reserve(std::max(reserve_size, (size_type)(max_fixed_size * 2)));
        assert(reserve_size >= m_size);
        for(auto &&x: tmp)
            m_vector.emplace_back(std::move(x));
        m_size = HAS_STD_VECTOR;
    }
    size_type m_size;
    static constexpr size_type HAS_STD_VECTOR = (size_type)-1;
    union {
        T m_array[max_fixed_size];
        std::vector<T> m_vector;
    };
};

}

#endif /*KAMESTM_FAST_VECTOR_H_*/
