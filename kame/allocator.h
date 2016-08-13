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
//            x = (void*)((uintptr_t)(void*)x | 1u); //invalid address.
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
        return std::min((int)mempool_max_size, std::max(4, 256 / (int)sizeof(T)));
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
//            if((uintptr_t)p & 1u)
//                throw ptr; //invalid address.
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
    iterator begin() noexcept {return is_fixed() ? &m_array[0] : &m_vector[0];}
    const_iterator begin() const noexcept {return is_fixed() ? &m_array[0] : &m_vector[0];}
    iterator end() noexcept {return is_fixed() ? &m_array[m_size] : &m_vector[m_vector.size()];}
    const_iterator end() const noexcept {return is_fixed() ? &m_array[m_size] : &m_vector[m_vector.size()];}
    size_type size() const noexcept {return is_fixed() ? m_size : m_vector.size();}
    bool empty() const noexcept {return !size();}
    reference operator[](size_type n) {return is_fixed() ? m_array[n] : m_vector[n];}
    const_reference operator[](size_type n) const {return is_fixed() ? m_array[n] : m_vector[n];}
    const_reference at(size_type n) const {if(n >= size()) throw std::out_of_range(""); return (*this)[n];}
    reference at(size_type n) {if(n >= size()) throw std::out_of_range(""); return (*this)[n];}
    reference front() {return is_fixed() ? m_array.front() : m_vector.front();}
    const_reference front() const {return is_fixed() ? m_array.front() : m_vector.front();}
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
            auto it = m_vector.erase(m_vector.begin() + (position - begin()));
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
//            shrink_to_fit();
        }
    }
    void shrink_to_fit() {
        if( !is_fixed()) return;
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
#endif /* ALLOCATOR_H_ */
