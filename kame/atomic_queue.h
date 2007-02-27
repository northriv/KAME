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
#ifndef ATOMIC_QUEUE_H_
#define ATOMIC_QUEUE_H_

#include "atomic.h"
#include <memory>

template <typename T, unsigned int SIZE>
class atomic_pointer_queue
{
public:
    struct nospace_error {};
    
    atomic_pointer_queue() : m_pFirst(m_ptrs), m_pLast(m_ptrs), m_count(0) {
        std::uninitialized_fill_n(m_ptrs, SIZE, (T*)0); 
    }

    void push(T* t) {
        ASSERT(t);
        for(;;) {
            T **last = m_pLast;
            T **first = m_pFirst;
            while(*last != 0) {
                last++;
                if(last == m_ptrs + SIZE) last = m_ptrs;
                if(last == first) {
                    throw nospace_error();
                }
            }
            if(atomicCompareAndSet((T*)0, t, last)) {
		        memoryBarrier();
                m_pLast = last;
                break;
            }
        }
        atomicInc(&m_count);
		writeBarrier();
    }
    //! This is not reentrant.
    void pop() {
        atomicDec(&m_count);
        writeBarrier();
        *m_pFirst = 0;
    }
    //! This is not reentrant.
    T *front() {
        readBarrier();
        T **first = m_pFirst;        
        while(*first == 0) {
            first++;
            if(first == m_ptrs + SIZE) first = m_ptrs;
        }
        m_pFirst = first;
        return *first;
    }
    //! This is not reentrant.
    bool empty() const {
        readBarrier();
        return m_count == 0;
    }
    unsigned int size() const {
        readBarrier();
        return m_count;
    }
private:
    T *m_ptrs[SIZE];
    T **m_pFirst;
    T **m_pLast;
    unsigned int m_count;
};

//! Atomic FIFO
template <typename T, unsigned int SIZE>
class atomic_queue
{
public:
    typedef typename atomic_pointer_queue<T, SIZE>::nospace_error nospace_error;
    
    atomic_queue() {}
    ~atomic_queue() {
        while(!empty()) pop();
    }
    
    void push(const T&t) {
        T *obj = new T(t);
        try {
            m_queue.push(obj);
        }
        catch (nospace_error &e) {
            delete obj;
            throw e;
        }
    }
    //! This is not reentrant.
    void pop() {
        delete m_queue.front();
        m_queue.pop();
    }
    //! This is not reentrant.
    T &front() {
        return *m_queue.front();
    }
    //! This is not reentrant.
    bool empty() const {
        return m_queue.empty();
    }
    unsigned int size() const {
        return m_queue.size();
    }
private:
    atomic_pointer_queue<T, SIZE> m_queue;
};

//! Atomic FIFO
template <typename T, unsigned int SIZE>
class atomic_queue_reserved
{
public:
    typedef typename atomic_pointer_queue<T, SIZE>::nospace_error nospace_error;
    
    atomic_queue_reserved() {
    	for(unsigned int i = 0; i < SIZE; i++)
    		m_obj[i] = new T;
    }
    ~atomic_queue_reserved() {
        while(!empty()) pop();
    	for(unsigned int i = 0; i < SIZE; i++) {
    		ASSERT(m_obj[i]);
    		delete m_obj[i];
    	}
    }
    
    void push(const T&t) {
    	int i = m_queue.size();
    	for(;;) {
	    	for(; i < SIZE; i++) {
	    		if(!m_obj[i]) continue;
    			T *obj = atomicSwap((T*)0L, &m_obj[i]);
    			if(obj) {
    				*obj = t;
					m_queue.push(obj);
					return;
    			}
	    	}
	    	i = 0;
	    	if(m_queue.size() == SIZE)
	    		throw nospace_error();
	    }
    }
    //! This is not reentrant.
    void pop() {
        T *obj = m_queue.front();
    	int i = m_queue.size();
    	for(;;) {
	    	for(; i >= 0; i--) {
	    		if(m_obj[i]) continue;
    			obj = atomicSwap(obj, &m_obj[i]);
    			if(!obj) {
    				m_queue.pop();
    				return;
    			}
	    	}
	    	i = SIZE - 1;
	    }
    }
    //! This is not reentrant.
    T &front() {
        return *m_queue.front();
    }
    //! This is not reentrant.
    bool empty() const {
        return m_queue.empty();
    }
    unsigned int size() const {
        return m_queue.size();
    }
private:
    atomic_pointer_queue<T, SIZE> m_queue;
    T *m_obj[SIZE];
};

#endif /*ATOMIC_QUEUE_H_*/
