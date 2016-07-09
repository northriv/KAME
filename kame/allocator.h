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
#include "atomic.h"

namespace Transactional {

struct MemoryPool : public std::array<atomic<void*>, 6> {
    MemoryPool() {
        std::fill(this->begin(), this->end(), nullptr);
    }
    ~MemoryPool() {
        for(auto &&x: *this)
            operator delete((void*)x);
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

    pointer allocate(size_type num, const void * /*hint*/ = 0) {
        const int unsigned pool_size = std::max(2, std::min(6, 256 / (int)sizeof(T)));
        for(unsigned int i = 0; i < pool_size; ++i) {
            auto &x = (*m_pool)[i];
            void *ptr = x.exchange(nullptr);
            if(ptr) {
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
        const int unsigned pool_size = std::max(2, std::min(6, 256 / (int)sizeof(T)));
        for(unsigned int i = 0; i < pool_size; ++i) {
            auto &x = (*m_pool)[i];
            p = x.exchange(p);
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

}
#endif /* ALLOCATOR_H_ */
