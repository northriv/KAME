#include "xsignal.h"
#include "atomic_queue.h"

shared_ptr<XSignalBuffer> g_signalBuffer;

threadid_t g_main_thread_id = threadID();
bool isMainThread() {
    return pthread_equal(threadID(), g_main_thread_id);
}

XListener::XListener(bool mainthreadcall, bool avoid_dup, unsigned int delay_ms) :
     m_bMainThreadCall(mainthreadcall),
     m_bAvoidDup(avoid_dup), 
     m_delay_ms(delay_ms), 
     m_bMasked(false) {
    if(avoid_dup) {
        ASSERT(mainthreadcall);
    }
    if(delay_ms) {
        ASSERT(avoid_dup);
    }
}
XListener::~XListener() {}
void
XListener::mask() {
    ASSERT(!m_bMasked);
    m_bMasked = true;
}
void
XListener::unmask() {
    ASSERT(m_bMasked);
    m_bMasked = false;
}

_XTalkerBase::_XTalkerBase() : m_bMasked(false) {}
_XTalkerBase::~_XTalkerBase() {}
void
_XTalkerBase::mask() {
    ASSERT(!m_bMasked);
    m_bMasked = true;
}
void
_XTalkerBase::unmask() {
    ASSERT(m_bMasked);
    m_bMasked = false;
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

