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

template <typename T, unsigned int SIZE, typename const_ref = T>
class atomic_nonzero_value_queue
{
public:
    struct nospace_error {};
    
    atomic_nonzero_value_queue() : m_pFirst(m_ptrs), m_pLast(m_ptrs), m_count(0) {
    	memset(m_ptrs, 0, SIZE * sizeof(T));
    }

    void push(T t) {
        ASSERT(t);
        writeBarrier();
        for(;;) {
        	if(m_count == SIZE) {
        		readBarrier();
	        	if(m_count == SIZE)
    	            throw nospace_error();
        	}
            T *last = m_pLast;
            T *first = m_pFirst;
            readBarrier();
            while(*last != 0) {
                last++;
                if(last == &m_ptrs[SIZE]) {
                	readBarrier();
                	last = m_ptrs;
                }
                if(last == first) {
                	break;
                }
            }
            if(atomicCompareAndSet((T)0, t, last)) {
				m_pLast = last;
				break;
            }
        }
		atomicInc(&m_count);
		writeBarrier();
    }
    //! This is not reentrant.
    void pop() {
        ASSERT(*m_pFirst);
        *m_pFirst = 0;
        atomicDec(&m_count);
        writeBarrier();
    }
    //! This is not reentrant.
    T front() {
        T *first = m_pFirst;
        readBarrier();
        while(*first == 0) {
            first++;
            if(first == &m_ptrs[SIZE]) {
            	first = m_ptrs;
		        readBarrier();
            }
        }
        m_pFirst = first;
        writeBarrier();
		readBarrier();
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
    bool atomicPop(const_ref item) {
    	ASSERT(item);
        if(atomicCompareAndSet((T)item, (T)0, m_pFirst)) {
			atomicDec(&m_count);
			writeBarrier();
        	return true;
        }
        return false;
    }
    //! Try to obtain the front item.
    const_ref atomicFront() {
        if(empty())
        	return 0L;
        T *first = m_pFirst;
        readBarrier();
        while(*first == 0) {
            first++;
            if(first == &m_ptrs[SIZE]) {
            	first = m_ptrs;
	            if(empty())
	            	return 0L;
            }
        }
        m_pFirst = first;
		readBarrier();
        return *first;
    }
    T atomicPopAny() {
        if(empty())
        	return 0L;
    	T *first = m_pFirst;
    	readBarrier();
    	for(;;) {
    		if(*first) {
				T obj = atomicSwap((T)0, first);
				if(obj) {
		            m_pFirst = first;
			        atomicDec(&m_count);
					writeBarrier();
					readBarrier();
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
    T m_ptrs[SIZE];
    T *m_pFirst;
    T *m_pLast;
    unsigned int m_count;
};

template <typename T, unsigned int SIZE>
class atomic_pointer_queue : public atomic_nonzero_value_queue<T*, SIZE, const T*> {};

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
    typedef ssize_t key;
    
    atomic_queue_reserved() {
    	C_ASSERT(SIZE < (1 << (sizeof(key) * 8 - 8)));
    	for(unsigned int i = 0; i < SIZE; i++) {
			m_reservoir.push(key_index_serial(i, 0));
    	}
    }
    ~atomic_queue_reserved() {
        while(!empty()) pop();
        ASSERT(m_reservoir.size() == SIZE);
    }
    
    void push(const T&t) {
    	key pack = m_reservoir.atomicPopAny(); 
    	if(!pack)
    		throw nospace_error();
    	int idx = key2index(pack);
    	m_array[idx] = t;
    	int serial = key2serial(pack) + 1;
    	pack = key_index_serial(idx, serial);
    	try {
	    	m_queue.push(pack);
    	}
    	catch (nospace_error&e) {
	    	try {
	    		m_reservoir.push(pack);
	    	}
	    	catch (nospace_error&) {
	    		abort();
	    	}
    		throw e;
    	}
    }
    //! This is not reentrant.
    void pop() {
        key pack = m_queue.front();
		m_queue.pop();
    	try {
    		m_reservoir.push(pack);
    	}
    	catch (nospace_error&) {
    		abort();
    	}
    }
    //! This is not reentrant.
    T &front() {
        return m_array[key2index(m_queue.front())];
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
    bool atomicPop(key item) {
        if(m_queue.atomicPop(item)) {
	    	try {
				m_reservoir.push(item);
	    	}
	    	catch (nospace_error&) {
	    		abort();
	    	}
	    	return true;
        }
        return false;
    }
    //! Try to obtain the front item.
    key atomicFront(T *val) {
        key pack = m_queue.atomicFront();
        if(pack)
        	*val = m_array[key2index(pack)];
        return pack;
    }
private:
    int key2index(key i) {return (unsigned int)i / 0x100;}
    int key2serial(key i) {return ((unsigned int)i % 0x100) - 1;}
    key key_index_serial(int index, int serial) {return index * 0x100 + (serial % 0xff) + 1;}
    atomic_nonzero_value_queue<key, SIZE> m_queue, m_reservoir;
    T m_array[SIZE];
};

#endif /*ATOMIC_QUEUE_H_*/
