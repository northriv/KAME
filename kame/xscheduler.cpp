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
 #include "xscheduler.h"
 
shared_ptr<XSignalBuffer> g_signalBuffer;

void 
registerTransactionList(_XTransaction *transaction) {
	g_signalBuffer->registerTransactionList(transaction);
}

XSignalBuffer::XSignalBuffer()
 : m_queue(new Queue()), m_queue_oldest_timestamp(timeStamp())
{
}
XSignalBuffer::~XSignalBuffer()
{
}
void 
XSignalBuffer::registerTransactionList(_XTransaction *transaction)
{
    unsigned long time(transaction->registered_time);
    ASSERT(!isMainThread());
    for(;;) {
    	for(unsigned int i = 0; i < 20; i++) {
    	  unsigned long cost = 0;
    	  if(!m_queue->empty()) {
    		  cost += time - m_queue_oldest_timestamp;
    	  }
    	  if(cost < 50000uL) {
            break;
          }
    	  msecsleep(std::min(cost * i / 1000000uL, 10uL));
	      time = timeStamp();
    	}
        try {
            bool empty = m_queue->empty();
            m_queue->push(transaction);
            if(empty) m_queue_oldest_timestamp = transaction->registered_time;
            break;
        }
        catch (Queue::nospace_error &) {
            msecsleep(10);
        }
    }
}
bool
XSignalBuffer::synchronize()
{
  bool dotalk = true;
  XTime time_start(XTime::now());
  unsigned long time_stamp_start(timeStamp());
  unsigned long clock_in_ms = 1000;
  unsigned int skipped_cnt = 0;
  
  for(;;) {
	if(m_queue->size() <= skipped_cnt) {
		dotalk = (m_queue->size() > 0);
		break;
	}
    _XTransaction *transaction;
    transaction = m_queue->front();
    
    bool skip = false;
    try {
        skip = transaction->talkBuffered();  
    }
    catch (XKameError &e) {
        e.print();
    }
    catch (std::bad_alloc &) {
        gErrPrint("Memory Allocation Failed!");
    }
    catch (...) {
        gErrPrint("Unhandled Exception Occurs!");
    }    
    
	if(skip) {
        try {
        		m_queue->push(transaction);
        }
        catch (Queue::nospace_error &) {
            continue;
        }
        skipped_cnt++;
	}
    else {
        delete transaction;
    }
    m_queue->pop();
    if(!m_queue->empty()) m_queue_oldest_timestamp = m_queue->front()->registered_time;
    if((timeStamp() - time_stamp_start) / clock_in_ms > 30uL) break;
  }
  unsigned int msec = XTime::now().diff_msec(time_start);
  if(msec > 10)
      clock_in_ms = (timeStamp() - time_stamp_start) / msec;
  return !dotalk;
}

