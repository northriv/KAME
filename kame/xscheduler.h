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
#ifndef XSCHEDULER_H_
#define XSCHEDULER_H_
#include "xsignal.h"
#include "atomic_queue.h"

//! Synchronize requests in talkers with main-thread
//! \sa XTalker, XListener
class XSignalBuffer
{
 public:
  XSignalBuffer();
  ~XSignalBuffer();
  //! Called by XTalker
  void registerTransactionList(_XTransaction *);
  //! be called by thread pool
  bool synchronize(); //return true if not busy
  
 private:
  typedef atomic_pointer_queue<_XTransaction, 10000> Queue;
  const scoped_ptr<Queue> m_queue;
  atomic<unsigned long> m_queue_oldest_timestamp;
};

extern shared_ptr<XSignalBuffer> g_signalBuffer;

#endif /*XSCHEDULER_H_*/
