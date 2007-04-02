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
//---------------------------------------------------------------------------

#include "thread.h"
#include <assert.h>
#include <errno.h>
#include <algorithm>
#include <sys/time.h>

//---------------------------------------------------------------------------

XPthreadMutex::XPthreadMutex()
{
	pthread_mutexattr_t attr;
	int ret;
	ret = pthread_mutexattr_init(&attr);
	if(DEBUG_XTHREAD) ASSERT(!ret);

	ret = pthread_mutex_init(&m_mutex, &attr);
	if(DEBUG_XTHREAD) ASSERT(!ret);

	ret = pthread_mutexattr_destroy(&attr);
	if(DEBUG_XTHREAD) ASSERT(!ret);
}

XPthreadMutex::~XPthreadMutex()
{
	int ret = pthread_mutex_destroy(&m_mutex);
	if(DEBUG_XTHREAD) ASSERT(!ret);
}
void
XPthreadMutex::lock() {
	int ret = pthread_mutex_lock(&m_mutex);
	if(DEBUG_XTHREAD) ASSERT(!ret);
}
bool
XPthreadMutex::trylock() {
	int ret = pthread_mutex_trylock(&m_mutex);
	if(DEBUG_XTHREAD) ASSERT(ret != EINVAL);
	return (ret == 0);
}
void
XPthreadMutex::unlock() {
	int ret = pthread_mutex_unlock(&m_mutex);
	if(DEBUG_XTHREAD) ASSERT(!ret);
}

XPthreadCondition::XPthreadCondition() : XPthreadMutex()
{
	int ret = pthread_cond_init(&m_cond, NULL);
	if(DEBUG_XTHREAD) ASSERT(!ret);
}
XPthreadCondition::~XPthreadCondition()
{
	int ret = pthread_cond_destroy(&m_cond);
	if(DEBUG_XTHREAD) ASSERT(!ret);
}
int
XPthreadCondition::wait(int usec)
{
	int ret;
	if(usec > 0)
	{
		struct timespec abstime;
		timeval tv;
		long nsec;
		gettimeofday(&tv, NULL);
		abstime.tv_sec = tv.tv_sec;
		nsec = (tv.tv_usec + usec) * 1000;
		if(nsec >= 1000000000) {
			nsec -= 1000000000; abstime.tv_sec++;
		}
		abstime.tv_nsec = nsec;
		ret = pthread_cond_timedwait(&m_cond, &m_mutex, &abstime);
	}
	else {
		ret = pthread_cond_wait(&m_cond, &m_mutex);
	}
	return ret;
}
void 
XPthreadCondition::signal()
{
	int ret = pthread_cond_signal(&m_cond);
	if(DEBUG_XTHREAD) ASSERT(!ret);
}
void 
XPthreadCondition::broadcast()
{
	int ret = pthread_cond_broadcast(&m_cond);
	if(DEBUG_XTHREAD) ASSERT(!ret);
}
