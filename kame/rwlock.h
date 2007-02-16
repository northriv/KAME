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
#ifndef RWLOCK_H_
#define RWLOCK_H_

#include "threadlocal.h"

//! read lock mutex during its life time.
template <class RWLock>
struct XScopedReadLock
{
    explicit XScopedReadLock(const RWLock &lock) : m_lock(lock) {
        m_lock.readLock();
    }
    ~XScopedReadLock() {
        m_lock.readUnlock();
    }
private:
    const RWLock &m_lock;
};

//! write lock mutex during its life time.
template <class RWLock>
struct XScopedWriteLock
{
    explicit XScopedWriteLock(RWLock &lock) : m_lock(lock) {
        m_lock.writeLock();
    }
    ~XScopedWriteLock() {
        m_lock.writeUnlock();
    }
private:
    RWLock &m_lock;
};


//! recursive (readLock() only) read-write lock
class XRWLock
{
 public:
  XRWLock();
  ~XRWLock();
  void readLock() const;
  void readUnlock() const;
  void writeLock();
  void writeUnlock();
 private:
  mutable pthread_rwlock_t m_lock;
};

#include <deque>
//! fully recursive read-write lock
//! writeLock/Unlock inside readLock/Unlock routine is allowed
class XRecursiveRWLock
{
 public:
  XRecursiveRWLock();
  ~XRecursiveRWLock();
  void readLock() const;
  //! must unlock with the locking thread
  void readUnlock() const;
  void writeLock();
  //! \ret true if unlocked
  //! must unlock with the locking thread
  bool writeUnlock();
  bool writeUnlockNReadLock();
  //! ReadLocked or WriteLocked
  bool isLocked() const;
 private:
  mutable pthread_cond_t m_cond;
  //! mutex for condition and m_rdlockingcnt
  mutable pthread_mutex_t m_mutex_write;
  mutable int m_wrlockwaitingcnt;
  mutable int m_rdlockingcnt, m_wrlockingcnt;
  mutable threadid_t m_wrlockingthread;
  typedef std::deque<const XRecursiveRWLock*> tLockedList;
  typedef tLockedList::iterator tLockedList_it;
  static XThreadLocal<tLockedList> s_tlRdLockedList;
};

#endif /*RWLOCK_H_*/
