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
#ifndef RECORDREADER_H_
#define RECORDREADER_H_

#include "recorder.h"

class XRawStreamRecordReader : public XRawStream
{
 XNODE_OBJECT
 protected:
  XRawStreamRecordReader(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist);
 public:
  virtual ~XRawStreamRecordReader() {}
  
  void terminate();
  
  const shared_ptr<XComboNode> &speed() const {return m_speed;}
  const shared_ptr<XBoolNode> &fastForward() const {return m_fastForward;}
  const shared_ptr<XBoolNode> &rewind() const {return m_rewind;}
  const shared_ptr<XNode> &stop() const {return m_stop;}
  const shared_ptr<XNode> &first() const {return m_first;}
  const shared_ptr<XNode> &next() const {return m_next;}
  const shared_ptr<XNode> &back() const {return m_back;}
  const shared_ptr<XStringNode> &posString() const {return m_posString;}
 private:
  struct XRecordError : public XKameError {
        XRecordError(const QString &msg, const char *file, int line)
             : XKameError(msg, file, line) {}
  };
  struct XIOError : public XRecordError {
        XIOError(const char *file, int line);
        XIOError(const QString &msg, const char *file, int line);
  };
  struct XBufferOverflowError : public XIOError {
        XBufferOverflowError(const char *file, int line);
  };
  struct XBrokenRecordError : public XRecordError {
        XBrokenRecordError(const char *file, int line);
  };
  struct XNoDriverError : public XRecordError {
    XNoDriverError(const char *driver_name, const char *file, int line);
    QString name;
  };
 
  const shared_ptr<XComboNode> m_speed;
  const shared_ptr<XBoolNode> m_fastForward;
  const shared_ptr<XBoolNode> m_rewind;
  const shared_ptr<XNode> m_stop;
  const shared_ptr<XNode> m_first, m_next, m_back;
  const shared_ptr<XStringNode> m_posString;
  void onPlayCondChanged(const shared_ptr<XValueNodeBase> &);
  void onStop(const shared_ptr<XNode> &);
  void onFirst(const shared_ptr<XNode> &);
  void onNext(const shared_ptr<XNode> &);
  void onBack(const shared_ptr<XNode> &);
  
  void onOpen(const shared_ptr<XValueNodeBase> &); 
  shared_ptr<XListener> m_lsnOnOpen;
  
  uint32_t m_allsize;
  XTime m_time;

  //! change position without parsing
  void _first(void *) throw (XIOError &);
  void _previous(void *) throw (XRecordError &);
  void _next(void *) throw (XRecordError &);
  void goToHeader(void *) throw (XRecordError &);

  void readHeader(void *) throw (XRecordError &);
  //! Parse current pos and go next
  void parseOne(void *, XMutex &mutex)  throw (XRecordError &);

  void gzgetline(void*fd, unsigned char*buf, unsigned int len, int del) throw (XIOError &);
  
  typedef shared_ptr<XThread<XRawStreamRecordReader> > tThread;
  typedef std::deque<tThread> tThreadList;
  typedef tThreadList::iterator tThreadIt;
  tThreadList m_threads;  
  void *execute(const atomic<bool> &);      
  XCondition m_condition;
  double m_periodicTerm;
  XMutex m_drivermutex;
  
  shared_ptr<XListener> m_lsnStop, m_lsnFirst, m_lsnNext, m_lsnBack;
  shared_ptr<XListener> m_lsnPlayCond;
};

#endif /*RECORDREADER_H_*/
