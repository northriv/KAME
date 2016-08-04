/***************************************************************************
		Copyright (C) 2002-2015 Kentaro Kitagawa
		                   kitagawa@phys.s.u-tokyo.ac.jp
		
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
#include "transaction_signal.h"
#include "atomic_queue.h"
#include <deque>

namespace Transactional {

//! Synchronize requests in talkers with main-thread
//! \sa Transactional::Talker, Transactional::Listener
class SignalBuffer {
public:
	//! Called by Talker
    static void registerEvent(BufferedEvent *event);
	//! be called by thread pool
    static bool synchronize(); //!< \return true if not busy

    static void initialize();
    static void cleanup();
    static unsigned int adaptiveDelay(); //!< ms
private:
    SignalBuffer();

    typedef atomic_pointer_queue<BufferedEvent, 1000> Queue;
    typedef std::deque<std::pair<BufferedEvent*, XTime> > SkippedQueue;
    BufferedEvent *popOldest();
	Queue m_queue;
	SkippedQueue m_skippedQueue;
    atomic<XTime> m_oldest_timestamp;
    atomic<unsigned int> m_adaptiveDelay; //!< ms.

    enum : int {ADAPTIVE_DELAY_MIN=5, ADAPTIVE_DELAY_MAX=50};

    void register_event(BufferedEvent *event);
    bool synchronize__(); //!< \return true if not busy

    static shared_ptr<SignalBuffer> s_signalBuffer;
};

}

#endif /*XSCHEDULER_H_*/
