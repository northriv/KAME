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
    	for(unsigned int i = 0; i < SIZE; i++)
    		m_ptrs[i] = 0;
    }

    void push(T* t) {
        ASSERT(t);
        readBarrier();
        for(;;) {
        	if(size() == SIZE)
                throw nospace_error();
            T **last = m_pLast;
            T **first = m_pFirst;
            while(*last != 0) {
                last++;
                if(last == &m_ptrs[SIZE]) last = m_ptrs;
                if(last == first) {
                	memoryBarrier();
                	break;
                }
            }
            if(atomicCompareAndSet((T*)0, t, last)) {
		        memoryBarrier();
                m_pLast = last;
                break;
            }
        }
		writeBarrier();
        atomicInc(&m_count);
    }
    //! This is not reentrant.
    void pop() {
        ASSERT(*m_pFirst);
        *m_pFirst = 0;
        writeBarrier();
        atomicDec(&m_count);
    }
    //! This is not reentrant.
    T *front() {
        readBarrier();
        T **first = m_pFirst;        
        while(*first == 0) {
            first++;
            if(first == &m_ptrs[SIZE])
            	first = m_ptrs;
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

	//! Try to pop the front item.
	//! \arg item to be released.
	//! \return true if succeeded.
    bool atomicPop(const T *item) {
        if(atomicCompareAndSet((T*)item, (T*)0, m_pFirst)) {
			writeBarrier();
	        atomicDec(&m_count);
        	return true;
        }
        return false;
    }
    //! Try to obtain the front item.
    const T *atomicFront() {
        if(empty())
        	return 0L;
        T **first = m_pFirst;        
        while(*first == 0) {
            first++;
            if(first == &m_ptrs[SIZE]) {
            	first = m_ptrs;
	            if(empty())
	            	return 0L;
            }
        }
        m_pFirst = first;
        return *first;
    }
    T *atomicPopAny() {
        if(empty())
        	return 0L;
    	T **first = m_pFirst;
    	for(;;) {
    		if(*first) {
				T *obj = atomicSwap((T*)0L, first);
				if(obj) {
		            m_pFirst = first;
					writeBarrier();
			        atomicDec(&m_count);
					return obj;
				}
    		}
            first++;
            if(first == &m_ptrs[SIZE]) {
            	first = m_ptrs;
	            if(empty())
	            	return 0L;
            }
    	}
    }
private:
    T *m_ptrs[SIZE];
    T **m_pFirst;
    T **m_pLast;
    unsigned int m_count;
};

//! Atomic FIFO for copy-constructable class.
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

//! Atomic FIFO for copieable class.
template <typename T, unsigned int SIZE>
class atomic_queue_reserved
{
public:
    typedef typename atomic_pointer_queue<T, SIZE>::nospace_error nospace_error;
    
    atomic_queue_reserved() {
    	for(unsigned int i = 0; i < SIZE; i++) {
			m_reservoir.push(&m_array[i]);
    	}
    }
    ~atomic_queue_reserved() {
        while(!empty()) pop();
        ASSERT(m_reservoir.size() == SIZE);
    }
    
    void push(const T&t) {
    	T *obj = m_reservoir.atomicPopAny(); 
    	if(!obj)
    		throw nospace_error();
    	*obj = t;
    	m_queue.push(obj);
    }
    //! This is not reentrant.
    void pop() {
        T *t = m_queue.front();
		m_queue.pop();
		m_reservoir.push(t);
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
    
	//! Try to pop the front item.
	//! \arg item to be released.
	//! \return true if succeeded.
    bool atomicPop(const T *item) {
        if(m_queue.atomicPop(item)) {
			m_reservoir.push((T*)item);
	    	return true;
        }
        return false;
    }
    //! Try to obtain the front item.
    const T *atomicFront() {
        return m_queue.atomicFront();
    }
private:
    atomic_pointer_queue<T, SIZE> m_queue, m_reservoir;
    T m_array[SIZE];
};

#endif /*ATOMIC_QUEUE_H_*/
