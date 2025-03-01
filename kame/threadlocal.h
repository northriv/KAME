/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef THREADLOCAL_H_
#define THREADLOCAL_H_

#include "support.h"

#if defined __GNUC__ && __GNUC__ >= 4
// #define USE_STD_TLS
#endif

#ifdef USE_QTHREAD
    #include <QThreadStorage>
#endif

//! Thread Local Storage template.
//! \p T must have constructor T()
//! object \p T will be deleted only when the thread is finished.
#ifdef USE_STD_TLS
template <typename T>
class XThreadLocal {
public:
    template <typename ...Arg>
    XThreadLocal(Arg&& ...arg) : m_var(std::forward<Arg>(arg)...) {}
    //! \return return thread local object. Create an object if not allocated.
    T &operator*() const {return m_var;}
    //! \sa operator T&()
    T *operator->() const {return &m_var;}
private:
    static T thread_local m_var;
};
#else
template <typename T>
class XThreadLocal {
public:
    XThreadLocal();
    ~XThreadLocal();
    //! \return return thread local object. Create an object if not allocated.
    inline T &operator*() const;
    //! \sa operator T&()
    inline T *operator->() const;
private:
#ifdef USE_QTHREAD
    mutable QThreadStorage<T*> m_tls;
#endif
#ifdef USE_PTHREAD
    mutable pthread_key_t m_key;
    static void delete_tls(void *var);
#endif
};

#ifdef USE_QTHREAD
    template <typename T>
    XThreadLocal<T>::XThreadLocal() {}
    template <typename T>
    XThreadLocal<T>::~XThreadLocal() {}
    template <typename T>
    inline T &XThreadLocal<T>::operator*() const {
        if( !m_tls.hasLocalData())
            m_tls.setLocalData(new T);
        return *m_tls.localData();
    }

#else
    #ifdef USE_PTHREAD
    template <typename T>
    XThreadLocal<T>::XThreadLocal() {
        int ret = pthread_key_create( &m_key, &XThreadLocal<T>::delete_tls);
        assert( !ret);
    }
    template <typename T>
    XThreadLocal<T>::~XThreadLocal() {
        delete static_cast<T *>(pthread_getspecific(m_key));
        int ret = pthread_key_delete(m_key);
        assert( !ret);
    }
    template <typename T>
    void
    XThreadLocal<T>::delete_tls(void *var) {
        delete static_cast<T *>(var);
    }
    template <typename T>
    inline T &XThreadLocal<T>::operator*() const {
        void *p = pthread_getspecific(m_key);
        if(p == NULL) {
            int ret = pthread_setspecific(m_key, p = new T);
            assert( !ret);
        }
        return *static_cast<T*>(p);
    }
    #endif //USE_PTHREAD
#endif //USE_QTHREAD

    template <typename T>
    inline T *XThreadLocal<T>::operator->() const {
        return &( **this);
    }

#endif /*USE_STD_TLS*/


#endif /*THREADLOCAL_H_*/
