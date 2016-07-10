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

#ifndef ALLOCATOR_H_
#define ALLOCATOR_H_

#include <array>
#include <vector>
#include "atomic.h"

namespace Transactional {

enum {mempool_max_size = 4};
struct MemoryPool : public std::array<atomic<void*>, mempool_max_size> {
    MemoryPool() {
        std::fill(this->begin(), this->end(), nullptr);
    }
    ~MemoryPool() {
        memoryBarrier();
        for(auto &&x: *this) {
            operator delete((void*)x);
            x = (void*)((uintptr_t)(void*)x | 1u); //invalid address.
        }
    }
};

template<typename T>
class allocator {
public:
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T& reference;
	typedef const T& const_reference;
	typedef T value_type;

	template<class Y>
	struct rebind {
		typedef allocator<Y> other;
	};

    allocator(MemoryPool *pool) noexcept : m_pool(pool) {}
    allocator(const allocator& x) noexcept : m_pool(x.m_pool) {}

    template<typename Y> allocator(const allocator<Y> &x) noexcept : m_pool(x.m_pool) {}

    ~allocator() {
    }

    unsigned int pool_size() const {
        return std::min((int)mempool_max_size, std::max(2, 256 / (int)sizeof(T)));
    }
    pointer allocate(size_type num, const void * /*hint*/ = 0) {
        for(unsigned int i = 0; i < pool_size(); ++i) {
            auto &x = (*m_pool)[i];
            void *ptr = x.exchange(nullptr);
            if(ptr) {
//                if((uintptr_t)ptr & 1u)
//                    throw ptr; //invalid address.
                return static_cast<pointer>(ptr);
            }
        }
        return (pointer) (operator new(num * sizeof(T)));
	}
    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        new((void*) p) U(std::forward<Args>(args)...);
	}

    void deallocate(pointer ptr, size_type /*num*/) {
        void *p = ptr;
        for(unsigned int i = 0; i < pool_size(); ++i) {
            auto &x = (*m_pool)[i];
            p = x.exchange(p);
            if((uintptr_t)p & 1u)
                throw ptr; //invalid address.
            if( !p) {
                return; //left in the pool.
            }
        }
        operator delete(p);
    }
    template <class U>
    void destroy(U* p) {
        p->~U();
	}

    pointer address(reference value) const noexcept {
		return &value;
	}
    const_pointer address(const_reference value) const noexcept {
		return &value;
	}

    size_type max_size() const noexcept {
		return std::numeric_limits<size_t>::max() / sizeof(T);
	}
private:
    template <typename Y> friend class allocator;
    MemoryPool *m_pool;
};

template <class T1, class T2>
bool operator==(const allocator<T1>&, const allocator<T2>&) noexcept {
    return true;
}

template <class T1, class T2>
bool operator!=(const allocator<T1>&, const allocator<T2>&) noexcept {
    return false;
}

template <typename T, int max_fixed_size = 4>
class fast_vector {
public:
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;
    fast_vector() : m_size(0) {m_vector.shrink_to_fit();}
//    ~fast_vector() {clear();}
    fast_vector(const fast_vector &r) : m_size(r.m_size), m_array(), m_vector() {
        if(r.is_fixed()) {
            std::copy(r.m_array.begin(), r.m_array.end(), m_array.begin());
        }
        else if(r.m_vector.size() <= max_fixed_size) {
            std::copy(r.m_vector.begin(), r.m_vector.end(), m_array.begin());
        }
        else {
            m_vector = std::move(std::vector<T>(r.m_vector));
        }
    }
    iterator begin() noexcept {return is_fixed() ? &m_array[0] : &m_vector[0];}
    const_iterator begin() const noexcept {return is_fixed() ? &m_array[0] : &m_vector[0];}
    iterator end() noexcept {return is_fixed() ? &m_array[m_size] : &m_vector[m_vector.size()];}
    const_iterator end() const noexcept {return is_fixed() ? &m_array[m_size] : &m_vector[m_vector.size()];}
    size_type size() const noexcept {return m_size;}
    bool empty() const noexcept {return !size();}
    reference operator[](size_type n) {return is_fixed() ? m_array[n] : m_vector[n];}
    const_reference operator[](size_type n) const {return is_fixed() ? m_array[n] : m_vector[n];}
    const_reference at(size_type n) const {if(n >= size()) throw std::out_of_range(""); return (*this)[n];}
    reference at(size_type n) {if(n >= size()) throw std::out_of_range(""); return (*this)[n];}
    reference front() {return is_fixed() ? m_array.front() : m_vector.front();}
    const_reference front() const {return is_fixed() ? m_array.front() : m_vector.front();}
    reference back() {return (*this)[this->size() - 1];}
    const_reference back() const {return (*this)[this->size() - 1];}
    void push_back(T&& x) {
        if(m_size < max_fixed_size) {
            m_array[m_size] = std::move(x);
        }
        else {
            if(m_size == max_fixed_size) {
                m_vector = std::move(std::vector<T>(m_array.begin(), m_array.end()));
                for(auto &&x: m_array)
                    x = std::move(T());
            }
            m_vector.push_back(x);
        }
        ++m_size;
    }
    template <class... Args>
    void emplace_back(Args&&... args) {
        if(m_size < max_fixed_size) {
            m_array[m_size] = T(std::forward<Args>(args)...);
        }
        else {
            if(m_size == max_fixed_size) {
                m_vector = std::move(std::vector<T>(m_array.begin(), m_array.end()));
                for(auto &&x: m_array)
                    x = std::move(T());
            }
            m_vector.emplace_back(std::forward<Args>(args)...);
        }
        ++m_size;
    }
    iterator erase(const_iterator position) {
        if(is_fixed()) {
            for(auto it = const_cast<iterator>(position);;) {
                 auto nit = it + 1;
                 if(nit == end()) {
                     *it = std::move(T());
                     break;
                 }
                 else
                     *it = *nit;
                 it = nit;
            }
            --m_size;
            return const_cast<iterator>(position);
        }
        else {
            auto it = m_vector.erase(m_vector.begin() + (position - begin()));
            --m_size;
            return &*it;
        }
    }
//    iterator erase(const_iterator first, const_iterator last);
    void clear() {
        if(is_fixed()) {
            clear_fixed();
        }
        else {
            m_vector.clear();
//            shrink_to_fit();
        }
        m_size = 0;
    }
    void resize(size_type sz) {
        if(is_fixed()) {
            if(sz > max_fixed_size) {
                auto eit = m_array.begin() + m_size;
                m_vector = std::move(std::vector<T>(m_array.begin(), eit));
                clear_fixed();
                m_vector.resize(sz);
            }
            else {
                for(size_type i = sz; i < m_size; ++i)
                    m_array[i] = std::move(T());
            }
        }
        else {
            m_vector.resize(sz);
//            shrink_to_fit();
        }
        m_size = sz;
    }
private:
    bool is_fixed() const noexcept {return m_vector.empty();}
    void shrink_to_fit() {
        if( !is_fixed()) return;
        if(m_vector.capacity() - m_vector.size() > max_fixed_size) {
            m_vector.shrink_to_fit();
        }
    }
    void clear_fixed() {
        auto eit = m_array.begin() + m_size;
        for(auto it = m_array.begin(); it != eit; ++it)
            *it = std::move(T());
    }
    size_type m_size;
    std::array<T, max_fixed_size> m_array;
    std::vector<T> m_vector;
};

}
#endif /* ALLOCATOR_H_ */
