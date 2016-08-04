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
#include "xscheduler.h"
 
using namespace Transactional;

Listener::Listener(FLAGS flags) :
    m_flags(flags) {
    if(flags & FLAG_AVOID_DUP) {
        assert(flags & FLAG_MAIN_THREAD_CALL);
    }
    if((flags & FLAG_DELAY_SHORT) || (flags & FLAG_DELAY_ADAPTIVE)) {
        assert(flags & FLAG_AVOID_DUP);
    }
}

unsigned int
Listener::delay_ms() const {
    unsigned int delay = std::min(20u, SignalBuffer::adaptiveDelay());
    if(m_flags & FLAG_DELAY_ADAPTIVE)
        delay = SignalBuffer::adaptiveDelay();
    if(m_flags & FLAG_DELAY_SHORT)
        delay /= 4;
    return delay;
}

void
BufferedEvent::registerEvent(BufferedEvent *e) {
    SignalBuffer::registerEvent(e);
}

shared_ptr<SignalBuffer>
SignalBuffer::s_signalBuffer;

void
SignalBuffer::initialize() {
    s_signalBuffer.reset(new SignalBuffer());
}
void
SignalBuffer::cleanup() {
    s_signalBuffer.reset();
}
unsigned int
SignalBuffer::adaptiveDelay() {
    return s_signalBuffer->m_adaptiveDelay;
}

SignalBuffer::SignalBuffer()
    : m_oldest_timestamp(XTime::now()), m_adaptiveDelay(ADAPTIVE_DELAY_MIN) {
}
BufferedEvent *
SignalBuffer::popOldest() {
    BufferedEvent *item = 0L, *skipped_item = 0L;
	if(m_queue.size()) {
		item = m_queue.front();
		if(m_skippedQueue.size()) {
			if((long)(m_queue.front()->registered_time - m_skippedQueue.front().second) > 0) {
				skipped_item = m_skippedQueue.front().first;
				item = 0L;
			}
		}
	}
	else {
		if(m_skippedQueue.size())
			skipped_item = m_skippedQueue.front().first;
	}
	if(item)
		m_queue.pop();
	if(skipped_item)	
		m_skippedQueue.pop_front();
	if(m_queue.size()) {
		if(m_skippedQueue.size()) {
			if((long)(m_queue.front()->registered_time - m_skippedQueue.front().second) > 0)
				m_oldest_timestamp = m_skippedQueue.front().second;
			else
				m_oldest_timestamp = m_queue.front()->registered_time;
		}
		else
			m_oldest_timestamp = m_queue.front()->registered_time;
	}
	else {
		if(m_skippedQueue.size())
			m_oldest_timestamp = m_skippedQueue.front().second;
	}
	if(item) {
		assert( !skipped_item);
		return item;
	}
	return skipped_item;
}
void
SignalBuffer::registerEvent(Transactional::BufferedEvent *event) {
    s_signalBuffer->register_event(event);
}

void 
SignalBuffer::register_event(Transactional::BufferedEvent *event) {
    XTime time(event->registered_time);
    for(;;) {
    	for(unsigned int i = 0; i < 20; i++) {
        	if(isMainThread())
        		break;
			unsigned long cost = 0;
			if( !m_queue.empty()) {
                cost += XTime::now().diff_usec(m_oldest_timestamp);
			}
            if(cost < (m_adaptiveDelay + 10) * 1000uL) {
				for(;;) {
                    unsigned int delay = m_adaptiveDelay;
					if(delay <= ADAPTIVE_DELAY_MIN) break;
                    if(m_adaptiveDelay.compare_set_strong(delay, delay - 1)) {
						break;
					}
				}
				break;
			}
            if(cost > 100uL) {
                if(m_adaptiveDelay < ADAPTIVE_DELAY_MAX) {
                    ++m_adaptiveDelay;
				}
			}
            msecsleep(std::min(cost / 1000uL, 10uL));
            time = XTime::now();
    	}
        try {
            bool empty = m_queue.empty();
            if(empty) m_oldest_timestamp = event->registered_time;
            m_queue.push(event);
            break;
        }
        catch (Queue::nospace_error &) {
        	if(isMainThread())
        		synchronize();
        	else
        		msecsleep(10);
        }
    }
}
bool
SignalBuffer::synchronize() {
    return s_signalBuffer->synchronize__();
}
bool
SignalBuffer::synchronize__() {
	bool dotalk = true;
    XTime time_stamp_start(XTime::now());
	unsigned int skipped_cnt = 0;
  
	for(;;) {
		if(m_queue.empty() && (m_skippedQueue.size() <= skipped_cnt)) {
			dotalk = !m_skippedQueue.empty();
			break;
		}
        BufferedEvent *event = popOldest();
        if( !event) {
			dotalk = false;
			break;
		}
		bool skip = false;
		try {
            skip = event->talkBuffered();
		}
		catch (XKameError &e) {
			e.print();
		}
		if(skip) {
            m_skippedQueue.emplace_back(event, XTime::now());
			skipped_cnt++;
		}
		else {
            delete event;
		}
        if(XTime::now().diff_msec(time_stamp_start) > 30uL) break;
	}
	return !dotalk;
}

