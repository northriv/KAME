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
#include "recordreader.h"
#include "analyzer.h"
#include "primarydriver.h"
#include "xtime.h"
#include "measure.h"

#include <zlib.h>
#include <vector>
#include <klocale.h>

#define IFSMODE std::ios::in
#define SPEED_FASTEST "Fastest"
#define SPEED_FAST "Fast"
#define SPEED_NORMAL "Normal"
#define SPEED_SLOW "Slow"

#define RECORDREADER_DELAY 20
#define RECORD_READER_NUM_THREADS 2

XRawStreamRecordReader::XIOError::XIOError(const char *file, int line)
 : XRecordError(KAME::i18n("IO Error"), file, line) {}
XRawStreamRecordReader::XIOError::XIOError(const QString &msg, const char *file, int line)
 : XRecordError(msg, file, line) {}
XRawStreamRecordReader::XBufferOverflowError::XBufferOverflowError(const char *file, int line)
 : XIOError(KAME::i18n("Buffer Overflow Error"), file, line) {}
XRawStreamRecordReader::XBrokenRecordError::XBrokenRecordError(const char *file, int line)
 : XRecordError(KAME::i18n("Broken Record Error"), file, line) {}
XRawStreamRecordReader::XNoDriverError::
 XNoDriverError(const char *driver_name, const char *file, int line)
   : XRecordError(KAME::i18n("No Driver Error: ") + driver_name, file, line),
    name(driver_name) {}
         
XRawStreamRecordReader::XRawStreamRecordReader(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist)
  : XRawStream(name, runtime, driverlist),
  m_speed(create<XComboNode>("Speed", true)),
  m_fastForward(create<XBoolNode>("FastForward", true)),
  m_rewind(create<XBoolNode>("Rewind", true)),
  m_stop(create<XNode>("Stop", true)),
  m_first(create<XNode>("First", true)),
  m_next(create<XNode>("Next", true)),
  m_back(create<XNode>("Back", true)),
  m_posString(create<XStringNode>("PosString", true)),
  m_periodicTerm(0)
{
    m_lsnOnOpen = filename()->onValueChanged().connectWeak(
        shared_from_this(), &XRawStreamRecordReader::onOpen);
    m_speed->add(SPEED_FASTEST);
    m_speed->add(SPEED_FAST);
    m_speed->add(SPEED_NORMAL);
    m_speed->add(SPEED_SLOW);
    m_speed->value(SPEED_FAST);
    
    m_lsnFirst = m_first->onTouch().connectWeak(
           shared_from_this(), &XRawStreamRecordReader::onFirst,
           XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
    m_lsnBack = m_back->onTouch().connectWeak(
           shared_from_this(), &XRawStreamRecordReader::onBack,
           XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
    m_lsnNext = m_next->onTouch().connectWeak(
           shared_from_this(), &XRawStreamRecordReader::onNext,
           XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
    m_lsnStop = m_stop->onTouch().connectWeak(
           shared_from_this(), &XRawStreamRecordReader::onStop,
           XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
    m_lsnPlayCond = m_fastForward->onValueChanged().connectWeak(
           shared_from_this(), 
           &XRawStreamRecordReader::onPlayCondChanged,
           XListener::FLAG_MAIN_THREAD_CALL | XListener::FLAG_AVOID_DUP | XListener::FLAG_DELAY_ADAPTIVE);
    m_rewind->onValueChanged().connect(m_lsnPlayCond);
    m_speed->onValueChanged().connect(m_lsnPlayCond);
    
    m_threads.resize(RECORD_READER_NUM_THREADS);
    for(tThreadIt it = m_threads.begin(); it != m_threads.end(); it++) {
        it->reset(new XThread<XRawStreamRecordReader>(shared_from_this(),
            &XRawStreamRecordReader::execute));
        (*it)->resume();
    }
}
void
XRawStreamRecordReader::onOpen(const shared_ptr<XValueNodeBase> &)
{
  if(m_pGFD) gzclose(m_pGFD);
  m_pGFD = gzopen(QString(filename()->to_str()).local8Bit(), "rb");
}
void
XRawStreamRecordReader::readHeader(void *fd)
 throw (XRawStreamRecordReader::XRecordError &)
{
  if(gzeof(fd))
         throw XIOError(__FILE__, __LINE__);
  uint32_t size =
                sizeof(uint32_t) //allsize
                + sizeof(int32_t) //time().sec()
                + sizeof(int32_t); //time().usec()
  std::vector<char> buf(size);
  std::vector<char>::iterator it = buf.begin();
  if(gzread(fd, &buf[0], size) == -1) throw XIOError(__FILE__, __LINE__);
  m_allsize = XPrimaryDriver::pop<uint32_t>(it);
  long sec = XPrimaryDriver::pop<int32_t>(it);
  long usec = XPrimaryDriver::pop<int32_t>(it);
  m_time = XTime(sec, usec);
}
void
XRawStreamRecordReader::parseOne(void *fd, XMutex &mutex)
 throw (XRawStreamRecordReader::XRecordError &)
{
  readHeader(fd);
  char name[256], sup[256];
  gzgetline(fd, (unsigned char*)name, 256, '\0');
  gzgetline(fd, (unsigned char*)sup, 256, '\0');
  if(strlen(name) == 0) {
    throw XBrokenRecordError(__FILE__, __LINE__);
  }
  shared_ptr<XPrimaryDriver> driver = dynamic_pointer_cast<XPrimaryDriver>(m_drivers->getChild(name));
  uint32_t size = 
    m_allsize - (
                sizeof(uint32_t) //allsize
                + sizeof(int32_t) //time().sec()
                + sizeof(int32_t) //time().usec()
                + strlen(name) //name of driver
                + strlen(sup) //reserved
                + 2 //two null chars
                + sizeof(uint32_t)  //allsize
                );
    if(driver) {
      if(size > MAX_RAW_RECORD_SIZE)
         driver.reset();  //too big
    }
    // m_time must be copied before unlocking
    XTime time(m_time);
    m_posString->value(time.getTimeStr());
    if(!driver) {
        if(gzseek(fd, size + sizeof(uint32_t), SEEK_CUR) == -1)
                 throw XIOError(__FILE__, __LINE__);
        throw XNoDriverError(name, __FILE__, __LINE__);
    }
    try {
        driver->clearRaw();
        driver->rawData().resize(size);
        if(gzread(fd, &driver->rawData()[0], size) == -1)
            throw XIOError(__FILE__, __LINE__);
        std::vector<char> buf(sizeof(uint32_t));
        if(gzread(fd, &buf[0], sizeof(uint32_t)) == -1)
            throw XIOError(__FILE__, __LINE__);
        std::vector<char>::iterator it = buf.begin();
        uint32_t footer_allsize = XPrimaryDriver::pop<uint32_t>(it);
        if(footer_allsize != m_allsize)
            throw XBrokenRecordError(__FILE__, __LINE__);
    }
    catch (XRecordError &e) {
        driver->finishWritingRaw(XTime(), XTime());
        throw e;
    }
    mutex.unlock();
    { XScopedLock<XMutex> lock(m_drivermutex);
        driver->finishWritingRaw(XTime::now(), time);
    }
}
void
XRawStreamRecordReader::gzgetline(void*fd, unsigned char*buf, unsigned int len, int del)
 throw (XIOError &)
{
int c;
   for(unsigned int i = 0; i < len; i++) {
    c = gzgetc(fd);
    if(c == -1) throw XIOError(__FILE__, __LINE__);
    *(buf++) = (unsigned char)c;
    if(c == del) return;
   }
   throw XBufferOverflowError(__FILE__, __LINE__);
}
void
XRawStreamRecordReader::_first(void *fd)
 throw (XRawStreamRecordReader::XIOError &)
{
  gzrewind(fd);
}
void
XRawStreamRecordReader::_previous(void *fd)
 throw (XRawStreamRecordReader::XRecordError &)
{
  if(gzseek(fd, -sizeof(uint32_t), SEEK_CUR) == -1) throw XIOError(__FILE__, __LINE__);
  goToHeader(fd);
}
void
XRawStreamRecordReader::_next(void *fd)
 throw (XRawStreamRecordReader::XRecordError &)
{
  readHeader(fd);
  uint32_t headersize = sizeof(uint32_t) //allsize
                + sizeof(int32_t) //time().sec()
                + sizeof(int32_t); //time().usec()
  if(gzseek(fd, m_allsize - headersize, SEEK_CUR) == -1) throw XIOError(__FILE__, __LINE__);
}
void
XRawStreamRecordReader::goToHeader(void *fd)
 throw (XRawStreamRecordReader::XRecordError &)
{
  if(gzeof(fd)) throw XIOError(__FILE__, __LINE__);
  std::vector<char> buf(sizeof(uint32_t));
  std::vector<char>::iterator it = buf.begin();
  if(gzread(fd, &buf[0], sizeof(uint32_t)) == Z_NULL) throw XIOError(__FILE__, __LINE__);
  int allsize = XPrimaryDriver::pop<uint32_t>(it);
  if(gzseek(fd, -allsize, SEEK_CUR) == -1) throw XIOError(__FILE__, __LINE__);
}
void
XRawStreamRecordReader::terminate()
{
    m_periodicTerm = 0;
    for(tThreadIt it = m_threads.begin(); it != m_threads.end(); it++) {
        (*it)->terminate();
    }
    XScopedLock<XCondition> lock(m_condition);
    m_condition.broadcast();
}

void
XRawStreamRecordReader::onPlayCondChanged(const shared_ptr<XValueNodeBase> &)
{
    double ms = 1.0;
    if(m_speed->to_str() == SPEED_FASTEST) ms = 0.1;
    if(m_speed->to_str() == SPEED_FAST) ms = 10.0;
    if(m_speed->to_str() == SPEED_NORMAL) ms = 30.0;
    if(m_speed->to_str() == SPEED_SLOW) ms = 100.0;
    if(!*m_fastForward && !*m_rewind) ms = 0;
    if(*m_rewind) ms = -ms;
    m_periodicTerm = ms;
    XScopedLock<XCondition> lock(m_condition);
    m_condition.broadcast();
}
void
XRawStreamRecordReader::onStop(const shared_ptr<XNode> &)
{
    m_periodicTerm = 0;
    g_statusPrinter->printMessage(KAME::i18n("Stopped"));
    m_fastForward->onValueChanged().mask();
    m_fastForward->value(false);
    m_fastForward->onValueChanged().unmask();
    m_rewind->onValueChanged().mask();
    m_rewind->value(false);
    m_rewind->onValueChanged().unmask();
}
void
XRawStreamRecordReader::onFirst(const shared_ptr<XNode> &)
{
  if(m_pGFD)
    {
      try {
            m_filemutex.lock();
            _first(m_pGFD);
            parseOne(m_pGFD, m_filemutex);
            g_statusPrinter->printMessage(KAME::i18n("First"));
      }
      catch (XRecordError &e) {
            m_filemutex.unlock();
            e.print(KAME::i18n("No Record, because "));
      }
    }
}
void
XRawStreamRecordReader::onNext(const shared_ptr<XNode> &)
{
  if(m_pGFD)
    {
      try {
           m_filemutex.lock(); 
           parseOne(m_pGFD, m_filemutex);
           g_statusPrinter->printMessage(KAME::i18n("Next"));
      }
      catch (XRecordError &e) {
            m_filemutex.unlock();
            e.print(KAME::i18n("No Record, because "));
      }
    }
}
void
XRawStreamRecordReader::onBack(const shared_ptr<XNode> &)
{
  if(m_pGFD)
    {
      try {
           m_filemutex.lock(); 
           _previous(m_pGFD);
           _previous(m_pGFD);
           parseOne(m_pGFD, m_filemutex);
           g_statusPrinter->printMessage(KAME::i18n("Previous"));
      }
      catch (XRecordError &e) {
            m_filemutex.unlock();
            e.print(KAME::i18n("No Record, because "));
      }
    }
}

void *XRawStreamRecordReader::execute(const atomic<bool> &terminated)
{
  while(!terminated)
    {
    double ms = 0.0;
        {
        XScopedLock<XCondition> lock(m_condition);
        while((fabs((ms = m_periodicTerm)) < 1e-4) && !terminated)
                m_condition.wait();
        }
    
      if(terminated) break;
      
      try {
          m_filemutex.lock(); 
          if(ms < 0.0) {
                _previous(m_pGFD);
                _previous(m_pGFD);
          }
          parseOne(m_pGFD, m_filemutex);
      }
      catch (XNoDriverError &e) {
          m_filemutex.unlock();
          e.print(KAME::i18n("No such driver :") + e.name);
      }
      catch (XRecordError &e) {
          m_periodicTerm = 0.0;
          m_fastForward->onValueChanged().mask();
          m_fastForward->value(false);
          m_fastForward->onValueChanged().unmask();
          m_rewind->onValueChanged().mask();
          m_rewind->value(false);
          m_rewind->onValueChanged().unmask();
          m_filemutex.unlock();
          e.print(KAME::i18n("No Record, because "));
      }
     
      msecsleep(lrint(fabs(ms)));
    }
    return NULL;
}
