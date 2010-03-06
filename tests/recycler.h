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

#ifndef RECYCLER_H_
#define RECYCLER_H_

#include "atomic.h"
#include "threadlocal.h"

template <typename T, int size>
struct recycler {
	void *operator new(size_t s) {
		ASSERT(sizeof(T) == s);
		for(unsigned int i = 0; i < size; ++i) {
			if( tls_garbage->ptrs[i]) {
				void *p = tls_garbage->ptrs[i];
				tls_garbage->ptrs[i] = 0;
				return p;
			}
		}
		return ::operator new(s);
	}
	void operator delete(void *p) {
		for(unsigned int i = 0; i < size; ++i) {
			if( !tls_garbage->ptrs[i]) {
				tls_garbage->ptrs[i] = static_cast<T*>(p);
				return;
			}
		}
		::operator delete(p);
	}
	struct Garbage {
		Garbage() {
			for(unsigned int i = 0; i < size; ++i)
				ptrs[i] = 0;
		}
		~Garbage() {
			for(unsigned int i = 0; i < size; ++i)
				::operator delete(ptrs[i]);
		}
		T *ptrs[size];
	};
	static XThreadLocal<Garbage> tls_garbage;
};

#endif /* RECYCLER_H_ */
