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

#include "atomic.h"

#define BANKS 8192
#define SIZE_OF_BANK 1024
#define MAX_SIZE (128)
char rmem[BANKS][SIZE_OF_BANK];
atomic<int> lasts[BANKS];
atomic<int> counts[BANKS];
void* operator new(size_t size) throw() {
	if(size < MAX_SIZE) {
		for(;;) {
			for(int bank = 0; bank < BANKS; bank++) {
				int last = lasts[bank];
				if(last >= SIZE_OF_BANK - MAX_SIZE) {
					int count = counts[bank];
					if(count == 0) {
						if(counts[bank].compareAndSet(0, 1)) {
							lasts[bank] = size;
							return &rmem[bank][0];
						}
					}
					continue;
				}
				++counts[bank];
				if(lasts[bank].compareAndSet(last, last + size)) {
					return &rmem[bank][last];
				}
				--counts[bank];
			}
		}
	}
	else
		return malloc(size);
}
void operator delete(void* p) {
	if((p >= &rmem[0][0]) && (p < &rmem[BANKS][SIZE_OF_BANK])) {
		for(int bank = 0; bank < BANKS; bank++) {
			if(p < &rmem[bank][SIZE_OF_BANK]) {
				--counts[bank];
				return;
			}
		}
		abort();
	}
	free(p);
}


#include "atomic_queue.h"

template <typename T>
int
atomic_shared_ptr_base<T,
typename boost::enable_if<boost::is_base_of<atomic_countable, T> >::type >::
deleter(T *p) {
#define DELAYED_DELETE_QUEUE_SIZE ((200000/sizeof(T)+32))
	struct Queue : public atomic_pointer_queue<T, DELAYED_DELETE_QUEUE_SIZE> {
		~Queue() {
			while( !this->empty()) {
				delete this->front();
				this->pop();
			}
		}
	};
	static Queue queue;
	if(p) {
		if(queue.atomicPush(p))
			return 0;
//		printf("!");
		delete p;
		return 1;
	}
	for(int i = 0;; ++i) {
		if(queue.empty())
			return i;
		p = queue.front();
		queue.pop();
		delete p;
	}
}

#endif /* ALLOCATOR_H_ */
