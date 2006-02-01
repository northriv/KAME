#ifndef THREADLOCAL_H_
#define THREADLOCAL_H_

#include "support.h"

#undef USE__THREAD_TLS

//! Thread Local Storage template.
//! \p T must have constructor T()
//! object \p T will be deleted only when the thread is finished.
template <typename T>
class XThreadLocal
{
public:
    XThreadLocal();
    ~XThreadLocal();
    //! \return return thread local object. Create an object if not allocated.
    T &operator*() const;
    //! \sa operator T&()
    T *operator->() const;
private:
#ifdef USE__THREAD_TLS
    __thread T m_var;
#else
    mutable pthread_key_t m_key;
    static void delete_tls(void *var);
#endif /*USE__THREAD_TLS*/
};

#ifdef USE__THREAD_TLS
    template <typename T>
    XThreadLocal<T>::XThreadLocal() {
    }
    template <typename T>
    XThreadLocal<T>::~XThreadLocal() {
    }
    template <typename T>
    T& XThreadLocal<T>::operator*() const {
        return m_var;
    }
    template <typename T>
    T *
    XThreadLocal<T>::operator->() const {
        return &m_var;
    }
#else
    template <typename T>
    XThreadLocal<T>::XThreadLocal() {
        int ret = pthread_key_create(&m_key, &XThreadLocal<T>::delete_tls);
        ASSERT(!ret);
    }
    template <typename T>
    XThreadLocal<T>::~XThreadLocal() {
       // int ret = pthread_key_delete(&m_key);
       // ASSERT(!ret);
    }
    template <typename T>
    void
    XThreadLocal<T>::delete_tls(void *var) {
            delete reinterpret_cast<T*>(var);
    }
    template <typename T>
    T &XThreadLocal<T>::operator*() const {
        void *p = pthread_getspecific(m_key);
        if(p == NULL) {
            dbgPrint(QString("New TLS."));
            int ret = pthread_setspecific(m_key, p = 
        #ifdef HAVE_LIBGCCPP
                new (NoGC) T);
        #else
                new T);
        #endif
            ASSERT(!ret);
        }
        return *reinterpret_cast<T*>(p);
    }
    template <typename T>
    T *
    XThreadLocal<T>::operator->() const {
        return &(**this);
    }
        
#endif /*USE__THREAD_TLS*/

#endif /*THREADLOCAL_H_*/
