/***************************************************************************
		Copyright (C) 2002-2007 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
#include "thread.h"
#include "rwlock.h"
#include "xtime.h"

#include <sys/time.h>

XThreadLocal<XRecursiveRWLock::tLockedList> XRecursiveRWLock::s_tlRdLockedList;

XRWLock::XRWLock() {
    int ret = pthread_rwlock_init(&m_lock, NULL);
    if(DEBUG_XTHREAD) ASSERT(!ret);
}
XRWLock::~XRWLock() {
    int ret = pthread_rwlock_destroy(&m_lock);
    if(DEBUG_XTHREAD) ASSERT(!ret);
}
void
XRWLock::readLock() const {
    int ret = pthread_rwlock_rdlock(&m_lock);
    if(DEBUG_XTHREAD) ASSERT(!ret);
}
void
XRWLock::readUnlock() const {
    int ret = pthread_rwlock_unlock(&m_lock);
    if(DEBUG_XTHREAD) ASSERT(!ret);
}
void
XRWLock::writeLock() {
    int ret = pthread_rwlock_wrlock(&m_lock);
    if(DEBUG_XTHREAD) ASSERT(!ret);
}
void
XRWLock::writeUnlock() {
    int ret = pthread_rwlock_unlock(&m_lock);
    if(DEBUG_XTHREAD) ASSERT(!ret);
}

XRecursiveRWLock::XRecursiveRWLock() :
	m_wrlockwaitingcnt(0),
	m_rdlockingcnt(0),
	m_wrlockingcnt(0),
	m_wrlockingthread((threadid_t)-1)
{
	int ret = pthread_mutex_init(&m_mutex_write, NULL);
	if(DEBUG_XTHREAD) ASSERT(!ret);
	ret = pthread_cond_init(&m_cond, NULL);
	if(DEBUG_XTHREAD) ASSERT(!ret);
}
XRecursiveRWLock::~XRecursiveRWLock()
{
	if(DEBUG_XTHREAD) ASSERT(m_wrlockwaitingcnt == 0);
	if(DEBUG_XTHREAD) ASSERT((int)m_wrlockingthread == -1);
	if(DEBUG_XTHREAD) ASSERT(m_rdlockingcnt == 0);
	int ret = pthread_cond_destroy(&m_cond);
	if(DEBUG_XTHREAD) ASSERT(!ret);
	ret = pthread_mutex_destroy(&m_mutex_write);
	if(DEBUG_XTHREAD) ASSERT(!ret);
}
void 
XRecursiveRWLock::readLock() const
{
	int ret;
	threadid_t threadid = threadID();
	if(!pthread_equal(m_wrlockingthread, threadid)) {
		tLockedList_it it = std::find(s_tlRdLockedList->begin(),
									  s_tlRdLockedList->end(), this);
		if(it == s_tlRdLockedList->end()) {
      
			ret = pthread_mutex_lock(&m_mutex_write);
			if(DEBUG_XTHREAD) ASSERT(!ret);
          
			if(DEBUG_XTHREAD) ASSERT(m_wrlockingcnt == 0);
          
			int max_retry = m_wrlockwaitingcnt;
			while (m_wrlockwaitingcnt && (max_retry--)) {
				ret = pthread_cond_broadcast(&m_cond);
				if(DEBUG_XTHREAD) ASSERT(!ret);
			}
          
			m_rdlockingcnt++;

			ret = pthread_mutex_unlock(&m_mutex_write);
			if(DEBUG_XTHREAD) ASSERT(!ret);      
		}
		else {
			ret = pthread_mutex_lock(&m_mutex_write);
			if(DEBUG_XTHREAD) ASSERT(!ret);
          
			if(DEBUG_XTHREAD) ASSERT(m_wrlockingcnt == 0);
      
			m_rdlockingcnt++;
          
			ret = pthread_mutex_unlock(&m_mutex_write);
			if(DEBUG_XTHREAD) ASSERT(!ret);      
		}
		s_tlRdLockedList->push_back(this);
    }
	else {
		m_rdlockingcnt++;
	}
}
void 
XRecursiveRWLock::readUnlock() const
{
	int ret;
	if(!pthread_equal(m_wrlockingthread, threadID())) {
		tLockedList_it it = std::find(s_tlRdLockedList->begin(),
									  s_tlRdLockedList->end(), this);
		if(DEBUG_XTHREAD) ASSERT(it != s_tlRdLockedList->end());
		s_tlRdLockedList->erase(it);
                
		ret = pthread_mutex_lock(&m_mutex_write);
		if(DEBUG_XTHREAD) ASSERT(!ret);

		if(DEBUG_XTHREAD) ASSERT(m_wrlockingcnt == 0);
  
		m_rdlockingcnt--;
		if(DEBUG_XTHREAD) ASSERT(m_rdlockingcnt >= 0);
      
		ret = pthread_cond_broadcast(&m_cond);
		if(DEBUG_XTHREAD) ASSERT(!ret);
      
		ret = pthread_mutex_unlock(&m_mutex_write);
		if(DEBUG_XTHREAD) ASSERT(!ret);
    }
    else {
		m_rdlockingcnt--;
		if(DEBUG_XTHREAD) ASSERT(m_rdlockingcnt >= 0);
    }
}
inline bool
XRecursiveRWLock::_writeLock(bool trylock)
{
	int ret;
	if(!pthread_equal(m_wrlockingthread, threadID()))
	{
		ret = pthread_mutex_lock(&m_mutex_write);
		if(DEBUG_XTHREAD) ASSERT(!ret);

		int tlRdLockedCnt = (int)std::count(s_tlRdLockedList->begin(),
											s_tlRdLockedList->end(), this);
      
		while(m_rdlockingcnt > tlRdLockedCnt) {
			if(trylock) {
				ret = pthread_mutex_unlock(&m_mutex_write);
				if(DEBUG_XTHREAD) ASSERT(!ret);
				return false;
			}
			m_wrlockwaitingcnt++;
			ret = pthread_cond_wait(&m_cond, &m_mutex_write);
			if(DEBUG_XTHREAD) ASSERT(!ret);
			m_wrlockwaitingcnt--;
		}

		if(DEBUG_XTHREAD) ASSERT(m_rdlockingcnt <= tlRdLockedCnt);
		if(DEBUG_XTHREAD) ASSERT(m_wrlockingcnt == 0);
		m_wrlockingthread = threadID();
	}
	m_wrlockingcnt++;
	return true;
}
bool
XRecursiveRWLock::tryWriteLock()
{
	return _writeLock(true);
}
void 
XRecursiveRWLock::writeLock()
{
	_writeLock(false);
}
bool
XRecursiveRWLock::writeUnlock()
{
	int ret;
	if(DEBUG_XTHREAD) ASSERT(pthread_equal(m_wrlockingthread, threadID()));
	m_wrlockingcnt--;
	if(m_wrlockingcnt == 0)
	{
		m_wrlockingthread = (threadid_t)-1;
        
		ret = pthread_mutex_unlock(&m_mutex_write);
		if(DEBUG_XTHREAD) ASSERT(!ret);

		return true;
	}
	return  false;
}
bool
XRecursiveRWLock::writeUnlockNReadLock()
{
	int ret;
	if(DEBUG_XTHREAD) ASSERT(pthread_equal(m_wrlockingthread, threadID()));
	m_wrlockingcnt--;
	m_rdlockingcnt++;
	if(m_wrlockingcnt == 0)
	{
		s_tlRdLockedList->push_back(this);

		m_wrlockingthread = (threadid_t)-1;
        
		ret = pthread_mutex_unlock(&m_mutex_write);
		if(DEBUG_XTHREAD) ASSERT(!ret);

		return true;
	}
	return  false;
}
bool
XRecursiveRWLock::isLocked() const
{
    readBarrier();
    return (m_rdlockingcnt > 0) || (m_wrlockingcnt > 0);
}
