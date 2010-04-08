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
#ifndef XSCHEDULER_H_
#define XSCHEDULER_H_
#include <xsignal.h>
#include <atomic_queue.h>

//! Synchronize requests in talkers with main-thread
//! \sa Talker, XListener
class XSignalBuffer {
public:
	XSignalBuffer();
	~XSignalBuffer();
	//! Called by Talker
	void registerTransactionList(_XTransaction *);
	//! be called by thread pool
	bool synchronize(); //!< \return true if not busy
private:
	typedef atomic_pointer_queue<_XTransaction, 1000> Queue;
	typedef std::deque<std::pair<_XTransaction*, unsigned long> > SkippedQueue;
	_XTransaction *popOldest();
	Queue m_queue;
	SkippedQueue m_skippedQueue;
	atomic<unsigned long> m_oldest_timestamp;
};

extern unsigned int g_adaptiveDelay; //!< ms.
extern shared_ptr<XSignalBuffer> g_signalBuffer;

#endif /*XSCHEDULER_H_*/
