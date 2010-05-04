/***************************************************************************
		Copyright (C) 2002-2010 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp

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

#include "allocator_prv.h"

extern inline void* operator new(std::size_t size) throw(std::bad_alloc) {
	return __new_redirected(size);
}
extern inline void* operator new(std::size_t size, const std::nothrow_t&) throw() {
	return operator new(size);
}
extern inline void* operator new[](std::size_t size) throw(std::bad_alloc) {
	return operator new(size);
}
extern inline void* operator new[](std::size_t size, const std::nothrow_t&) throw() {
	return operator new(size);
}

extern void __deallocate_pooled_or_free(void* p) throw();

extern inline void operator delete(void* p) throw() {
	return __deallocate_pooled_or_free(p);
}
extern inline void operator delete(void* p, const std::nothrow_t&) throw() {
	operator delete(p);
}
extern inline void operator delete[](void* p) throw() {
	operator delete(p);
}
extern inline void operator delete[](void* p, const std::nothrow_t&) throw() {
	operator delete(p);
}

extern void release_pools();

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

	allocator() throw () { }
	allocator(const allocator&) throw () { }
	template<typename Y> allocator(const allocator<Y> &) throw () {}
	~allocator() throw () {}

	pointer allocate(size_type num, const void *hint = 0) {
		return (pointer) (operator new(num * sizeof(T)));
	}
	void construct(pointer p, const T& value) {
		new((void*) p) T(value);
	}

	void deallocate(pointer p, size_type num) {
		operator delete((void *) p);
	}
	void destroy(pointer p) {
		p->~T();
	}

	pointer address(reference value) const {
		return &value;
	}
	const_pointer address(const_reference value) const {
		return &value;
	}

	size_type max_size() const throw () {
		return std::numeric_limits<size_t>::max() / sizeof(T);
	}
};

template <class T1, class T2>
bool operator==(const allocator<T1>&, const allocator<T2>&) throw() { return true; }

template <class T1, class T2>
bool operator!=(const allocator<T1>&, const allocator<T2>&) throw() { return false; }

#endif /* ALLOCATOR_H_ */
