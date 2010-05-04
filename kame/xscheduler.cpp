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
#include "xscheduler.h"
 
shared_ptr<XSignalBuffer> g_signalBuffer;

#define ADAPTIVE_DELAY_MIN 10
#define ADAPTIVE_DELAY_MAX 100

unsigned int g_adaptiveDelay = ADAPTIVE_DELAY_MIN;

void 
registerTransactionList(XTransaction__ *transaction) {
	g_signalBuffer->registerTransactionList(transaction);
}

XSignalBuffer::XSignalBuffer()
	: m_oldest_timestamp(timeStamp()) {
}
XSignalBuffer::~XSignalBuffer() {
}
XTransaction__ *
XSignalBuffer::popOldest() {
	XTransaction__ *item = 0L, *skipped_item = 0L;
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
		ASSERT( !skipped_item);
		return item;
	}
	return skipped_item;
}
void 
XSignalBuffer::registerTransactionList(XTransaction__ *transaction) {
    unsigned long time(transaction->registered_time);
    for(;;) {
    	for(unsigned int i = 0; i < 20; i++) {
        	if(isMainThread())
        		break;
			unsigned long cost = 0;
			if( !m_queue.empty()) {
				cost += time - m_oldest_timestamp;
			}
			if(cost < (g_adaptiveDelay + 10) * 1000uL) {
				for(;;) {
					unsigned int delay = g_adaptiveDelay;
					if(delay <= ADAPTIVE_DELAY_MIN) break;
					if(atomicCompareAndSet(delay, delay - 1, &g_adaptiveDelay)) {
						break;
					}
				}
				break;
			}
			if(cost > 100000uL) {
				if(g_adaptiveDelay < ADAPTIVE_DELAY_MAX) {
					atomicInc( &g_adaptiveDelay);
				}
			}
			msecsleep(std::min(cost / 1000000uL, 10uL));
			time = timeStamp();
    	}
        try {
            bool empty = m_queue.empty();
            if(empty) m_oldest_timestamp = transaction->registered_time;
            m_queue.push(transaction);
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
XSignalBuffer::synchronize() {
	bool dotalk = true;
	XTime time_start(XTime::now());
	unsigned long time_stamp_start(timeStamp());
	unsigned long clock_in_ms = 1000;
	unsigned int skipped_cnt = 0;
  
	for(;;) {
		if(m_queue.empty() && (m_skippedQueue.size() <= skipped_cnt)) {
			dotalk = !m_skippedQueue.empty();
			break;
		}
		XTransaction__ *transaction = popOldest();
		if( !transaction) {
			dotalk = false;
			break;
		}
		bool skip = false;
		try {
			skip = transaction->talkBuffered();  
		}
		catch (XKameError &e) {
			e.print();
		}
		if(skip) {
			m_skippedQueue.push_back(std::pair<XTransaction__*, unsigned long>(transaction, timeStamp()));
			skipped_cnt++;
		}
		else {
			delete transaction;
		}
		if((timeStamp() - time_stamp_start) / clock_in_ms > 30uL) break;
	}
	unsigned int msec = XTime::now().diff_msec(time_start);
	if(msec > 10)
		clock_in_ms = (timeStamp() - time_stamp_start) / msec;
	return !dotalk;
}

