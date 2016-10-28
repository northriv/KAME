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
BufferedEvent::registerEvent(std::unique_ptr<BufferedEvent> e) {
    SignalBuffer::registerEvent(std::move(e));
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

#include "xthread.h"
XMutex mutex;

SignalBuffer::SignalBuffer()
    : m_oldest_timestamp(XTime::now()), m_adaptiveDelay(ADAPTIVE_DELAY_MIN) {
}
std::unique_ptr<BufferedEvent> SignalBuffer::popOldest() {
    std::unique_ptr<BufferedEvent> item;
    if(m_queue.size()) {
        if(m_skippedQueue.size() &&
            (m_queue.front()->registered_time > m_skippedQueue.front().second)) {
            //an event is skipped before the ordinary event is registered.
            item = std::move(m_skippedQueue.front().first);
            m_skippedQueue.pop();
        }
        else {
            item.reset(m_queue.front());
            XScopedLock<XMutex> lock(mutex);
            m_queue.pop();
        }
	}
    else if(m_skippedQueue.size()) {
        item = std::move(m_skippedQueue.front().first);
        m_skippedQueue.pop();
	}

    if(m_queue.size())
        m_oldest_timestamp = m_queue.front()->registered_time;

    return item;
}
void
SignalBuffer::registerEvent(std::unique_ptr<Transactional::BufferedEvent> event) {
    s_signalBuffer->register_event(std::move(event));
}

void 
SignalBuffer::register_event(std::unique_ptr<Transactional::BufferedEvent> event) {
    for(;;) {
        unsigned long cost = 0;
        if( !m_queue.empty()) {
            cost = XTime::now().diff_msec(m_oldest_timestamp);
        }
        unsigned long new_delay = cost / 2;
        new_delay = std::min((unsigned long)ADAPTIVE_DELAY_MAX, new_delay);
        new_delay = std::max((unsigned long)ADAPTIVE_DELAY_MIN, new_delay);
        m_adaptiveDelay = new_delay;
        try {
            if(m_queue.empty()) m_oldest_timestamp = event->registered_time;
            auto *e = event.get();
            event.release();
            {
            XScopedLock<XMutex> lock(mutex);
            m_queue.push(e);
            }
//        if(m_queue.size() > 4000) {
        }
        catch (Queue::nospace_error &) {
            fprintf(stderr, "Queue size exceeded to %d.\n", (int)m_queue.size());
            if(isMainThread())
                synchronize();
            else
                msecsleep(10);
        }
        break;
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
        auto event = popOldest();
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
            m_skippedQueue.emplace(std::move(event), XTime::now());
            skipped_cnt++;
            assert( !event);
        }
        if(XTime::now().diff_msec(time_stamp_start) > 30) break;
	}
	return !dotalk;
}

