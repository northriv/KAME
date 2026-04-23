/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#ifndef THREADLOCAL_H_
#define THREADLOCAL_H_

#include "support.h"
#include <type_traits>

#ifdef USE_QTHREAD
    #include <QThreadStorage>
#endif

//! Thread Local Storage template.
//! \p T must have constructor T()
//! object \p T will be deleted only when the thread is finished.
//!
//! Two specializations picked via SFINAE on is_trivially_destructible<T>:
//!   * trivially-destructible T  →  native C++ `thread_local` (fast). Safe on
//!                                  libstdc++/libc++ because no per-thread
//!                                  destructor is ever required to run — the
//!                                  historical TLS-destructor portability
//!                                  gaps do not apply to trivial T.
//!   * non-trivially-destructible →  pthread_key_t / QThreadStorage path
//!                                  (slower but runs the destructor at
//!                                  thread exit). Preserved for types that
//!                                  actually need cleanup.
template <typename T, typename = void>
class XThreadLocal;

// Fast path: trivially-destructible T uses compiler-native thread_local.
// Non-trivial constructors are fine — thread-local dynamic init is standard.
// NOTE: the thread_local storage is per-T (class-static), so having two
// XThreadLocal<T> instances of the *same* T would make them alias. KAME
// uses distinct T per instance (ProcessCounter, SerialGenerator::cnt_t,
// Priority__, FuncPayloadCreator, ...), so the aliasing is benign.
template <typename T>
class XThreadLocal<T, typename std::enable_if<std::is_trivially_destructible<T>::value>::type> {
public:
    template <typename ...Arg>
    XThreadLocal(Arg&& ...) noexcept {}
    T &operator*() const {return m_var;}
    T *operator->() const {return &m_var;}
private:
    static thread_local T m_var;
};
template <typename T>
thread_local T XThreadLocal<T, typename std::enable_if<std::is_trivially_destructible<T>::value>::type>::m_var{};

// Slow path: non-trivially-destructible T keeps the pthread/QThread layout
// so the destructor runs at thread exit. Impl defined inline to avoid the
// verbose out-of-line syntax on a SFINAE specialization.
template <typename T>
class XThreadLocal<T, typename std::enable_if<!std::is_trivially_destructible<T>::value>::type> {
public:
#ifdef USE_QTHREAD
    XThreadLocal() {}
    ~XThreadLocal() {}
    T &operator*() const {
        if( !m_tls.hasLocalData())
            m_tls.setLocalData(new T);
        return *m_tls.localData();
    }
#elif defined(USE_PTHREAD)
    XThreadLocal() {
        int ret = pthread_key_create( &m_key, &XThreadLocal::delete_tls);
        assert( !ret);
    }
    ~XThreadLocal() {
        delete static_cast<T *>(pthread_getspecific(m_key));
        int ret = pthread_key_delete(m_key);
        assert( !ret);
    }
    T &operator*() const {
        void *p = pthread_getspecific(m_key);
        if(p == NULL) {
            int ret = pthread_setspecific(m_key, p = new T);
            assert( !ret);
        }
        return *static_cast<T*>(p);
    }
#else
    // Portable fallback: native C++11 thread_local. The spec guarantees
    // T's destructor runs at thread exit. MSVC before VS 2017 15.5 had
    // issues with non-trivial thread_local across DLL boundaries — that
    // matters only for old-MSVC + DLL builds, for which the Qt build
    // should define USE_QTHREAD. Linux/macOS pthread is the primary
    // platform; this branch exists so a bare `#include "threadlocal.h"`
    // in a toolchain-minimal TU (e.g. a unit test with neither Qt nor
    // explicit USE_PTHREAD) still links.
    //
    // NOTE: thread_local storage is class-static and keyed by T, same
    // constraint as the trivial specialization — two XThreadLocal<same T>
    // instances would alias. KAME's convention of unique T per instance
    // makes this benign.
    XThreadLocal() = default;
    T &operator*() const {return m_var;}
#endif
    T *operator->() const {return &( **this);}
private:
#ifdef USE_QTHREAD
    mutable QThreadStorage<T*> m_tls;
#elif defined(USE_PTHREAD)
    mutable pthread_key_t m_key;
    static void delete_tls(void *var) { delete static_cast<T *>(var); }
#else
    static thread_local T m_var;
#endif
};

#if !defined(USE_QTHREAD) && !defined(USE_PTHREAD)
template <typename T>
thread_local T XThreadLocal<T,
    typename std::enable_if<!std::is_trivially_destructible<T>::value>::type>::m_var{};
#endif


#endif /*THREADLOCAL_H_*/
