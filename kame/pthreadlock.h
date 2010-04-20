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
#ifndef PTHREADLOCK_H_
#define PTHREADLOCK_H_


/*! non-recursive mutex.
 * double lock is inhibited.
 * \sa XRecursiveMutex.
 */
class XPthreadMutex
{
public:
	XPthreadMutex();
	~XPthreadMutex();

	void lock();
	void unlock();
	//! \return true if locked.
	bool trylock();
protected:
	pthread_mutex_t m_mutex;
};

//! condition class.
class XPthreadCondition : public XPthreadMutex
{
public:
	XPthreadCondition();
	~XPthreadCondition();
	//! Lock me before calling me.
	//! go asleep until signal is emitted.
	//! \param usec if non-zero, timeout occurs after \a usec.
	//! \return zero if locked thread is waked up.
	int wait(int usec = 0);
	//! wake-up at most one thread.
	//! \sa broadcast()
	void signal();
	//! wake-up all waiting threads.
	//! \sa signal()
	void broadcast();
private:
	pthread_cond_t m_cond;
};

#endif /*PTHREADLOCK_H_*/
