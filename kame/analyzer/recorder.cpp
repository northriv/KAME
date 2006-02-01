//---------------------------------------------------------------------------

#include "recorder.h"
#include "analyzer.h"
#include "primarydriver.h"
#include "xtime.h"

#include <zlib.h>
#include <vector>

//---------------------------------------------------------------------------
#define OFSMODE std::ios::out | std::ios::app | std::ios::ate

XRawStream::XRawStream(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist)
 : XNode(name, runtime),
    m_drivers(driverlist),
    m_pGFD(0),
    m_filename(create<XStringNode>("Filename", true))
{
}
XRawStream::~XRawStream()
{
    if(m_pGFD) gzclose(m_pGFD);
}    

XRawStreamRecorder::XRawStreamRecorder(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist)
 : XRawStream(name, runtime, driverlist),
    m_recording(create<XBoolNode>("Recording", true))
{
    recording()->value(false);
    
    m_lsnOnOpen = filename()->onValueChanged().connectWeak(
        false, shared_from_this(), &XRawStreamRecorder::onOpen);
    m_lsnOnFlush = recording()->onValueChanged().connectWeak(
        false, shared_from_this(), &XRawStreamRecorder::onFlush);
    m_lsnOnCatch = m_drivers->onCatch().connectWeak(
        false, shared_from_this(), &XRawStreamRecorder::onCatch);
    m_lsnOnRelease = m_drivers->onRelease().connectWeak(
        false, shared_from_this(), &XRawStreamRecorder::onRelease);
}
void
XRawStreamRecorder::onCatch(const shared_ptr<XNode> &node)
{
    shared_ptr<XDriver> driver = dynamic_pointer_cast<XDriver>(node);
    if(m_lsnOnRecord)
        driver->onRecord().connect(m_lsnOnRecord);
    else
        m_lsnOnRecord = driver->onRecord().connectWeak(
            false, shared_from_this(), &XRawStreamRecorder::onRecord);
}
void
XRawStreamRecorder::onRelease(const shared_ptr<XNode> &node)
{
    shared_ptr<XDriver> driver = dynamic_pointer_cast<XDriver>(node);
    driver->onRecord().disconnect(m_lsnOnRecord);
}
void
XRawStreamRecorder::onOpen(const shared_ptr<XValueNodeBase> &)
{
  if(m_pGFD) gzclose(m_pGFD);
  m_pGFD = gzopen(filename()->to_str().local8Bit(), "wb");
}
void
XRawStreamRecorder::onFlush(const shared_ptr<XValueNodeBase> &)
{
  if(!*recording())
     if(m_pGFD) {
        m_filemutex.lock();    
        gzflush(m_pGFD, Z_FULL_FLUSH);
        m_filemutex.unlock();    
     }
}
void
XRawStreamRecorder::onRecord(const shared_ptr<XDriver> &d)
{
    if(*recording() && m_pGFD) {
        shared_ptr<XPrimaryDriver> driver = dynamic_pointer_cast<XPrimaryDriver>(d);
        if(driver) {
            driver->readLockRaw();
            uint32_t size = driver->rawData().size();
            if(size) {
                uint32_t headersize =
                    sizeof(uint32_t) //allsize
                    + sizeof(int32_t) //time().sec()
                    + sizeof(int32_t); //time().usec()            
                std::vector<char> buf;
                // size of raw record wrapped by header and footer
                uint32_t allsize =
                    headersize
                    + strlen(driver->getName().utf8()) //name of driver
                    + 2 //two null chars
                    + size //rawData
                    + sizeof(uint32_t); //allsize
                XPrimaryDriver::push((uint32_t)allsize, buf);
                XPrimaryDriver::push((int32_t)driver->time().sec(), buf);
                XPrimaryDriver::push((int32_t)driver->time().usec(), buf);
                ASSERT(buf.size() == headersize);
    
                m_filemutex.lock();
                gzwrite(m_pGFD, &buf[0], buf.size());
                gzprintf(m_pGFD, "%s", (const char*)driver->getName().utf8());
                gzputc(m_pGFD, '\0');
                gzputc(m_pGFD, '\0'); //Reserved
                gzwrite(m_pGFD, &driver->rawData()[0], size);
                buf.clear();
                XPrimaryDriver::push((uint32_t)allsize, buf);
                gzwrite(m_pGFD, &buf[0], buf.size());
                m_filemutex.unlock();
            }
            driver->readUnlockRaw();
        }
    } 
}


XTextWriter::XTextWriter(const char *name, bool runtime,
     const shared_ptr<XDriverList> &driverlist, const shared_ptr<XScalarEntryList> &entrylist)
  : XNode(name, runtime),
    m_drivers(driverlist),
    m_entries(entrylist),
    m_filename(create<XStringNode>("Filename", true)),
    m_lastLine(create<XStringNode>("LastLine", true)),
    m_recording(create<XBoolNode>("Recording", true))
{
    recording()->value(false);
    lastLine()->setUIEnabled(false);
  
    m_lsnOnFilenameChanged = filename()->onValueChanged().connectWeak(
        false, shared_from_this(), &XTextWriter::onFilenameChanged);
    m_lsnOnCatch = m_drivers->onCatch().connectWeak(
        false, shared_from_this(), &XTextWriter::onCatch);
    m_lsnOnRelease = m_drivers->onRelease().connectWeak(
        false, shared_from_this(), &XTextWriter::onRelease);
}
void
XTextWriter::onCatch(const shared_ptr<XNode> &node)
{
    shared_ptr<XDriver> driver = dynamic_pointer_cast<XDriver>(node);
    if(m_lsnOnRecord)
        driver->onRecord().connect(m_lsnOnRecord);
    else
        m_lsnOnRecord = driver->onRecord().connectWeak(
            false, shared_from_this(), &XTextWriter::onRecord);
}
void
XTextWriter::onRelease(const shared_ptr<XNode> &node)
{
    shared_ptr<XDriver> driver = dynamic_pointer_cast<XDriver>(node);
    driver->onRecord().disconnect(m_lsnOnRecord);
}
void
XTextWriter::onLastLineChanged(const shared_ptr<XValueNodeBase> &) {
  m_filemutex.lock();
  if(m_stream.good())
  {
        m_stream << (const char*)lastLine()->to_str().utf8()
                << std::endl;
  }
  m_filemutex.unlock();
}
void
XTextWriter::onRecord(const shared_ptr<XDriver> &driver)
{
     if(*recording() == true)
	{
	  if(driver->time())
	    {
            XTime triggered_time;
            std::deque<shared_ptr<XScalarEntry> > locked_entries;
            m_entries->childLock();
	        for(unsigned int i = 0; i < m_entries->count(); i++)
        		{
                  shared_ptr<XScalarEntry> entry = (*m_entries)[i];
                  if(!*entry->store()) continue;
                  shared_ptr<XDriver> d(entry->driver());
                  if(!d) continue;
                  locked_entries.push_back(entry);
                  d->readLockRecord();
                  if(entry->isTriggered()) triggered_time = entry->driver()->time();
        		}
            m_entries->childUnlock();
            if(triggered_time) {
                    XRecordDependency dep;
                    for(std::deque<shared_ptr<XScalarEntry> >::iterator it = locked_entries.begin();
                        it != locked_entries.end(); it++) {
                          shared_ptr<XDriver> d((*it)->driver());
                          if(!d) continue;
                          dep.merge(d);
                          if(dep.isConflict()) break;
                    }
                    if(!dep.isConflict()) {
                        QString buf;
                        for(std::deque<shared_ptr<XScalarEntry> >::iterator it = locked_entries.begin();
                            it != locked_entries.end(); it++) {
                              if(!*(*it)->store()) continue;
                              (*it)->storeValue();
                              buf += (*it)->storedValue()->to_str() + " ";
                        }
                    	    buf += driver->time().getTimeFmtStr("%Y/%m/%d %H:%M:%S");
            	            lastLine()->value(buf);
                    }
            }
            for(std::deque<shared_ptr<XScalarEntry> >::iterator it = locked_entries.begin();
                it != locked_entries.end(); it++) {
                shared_ptr<XDriver> d((*it)->driver());
                if(!d) continue;
                d->readUnlockRecord();
            }
    	    }
	}
}

void
XTextWriter::onFilenameChanged(const shared_ptr<XValueNodeBase> &)
{
  m_filemutex.lock();
  
  if(m_stream.is_open()) m_stream.close();
  m_stream.clear();
  m_stream.open((const char*)filename()->to_str().local8Bit(), OFSMODE);

  if(m_stream.good()) {
    m_lsnOnFlush = recording()->onValueChanged().connectWeak(
        false, shared_from_this(), &XTextWriter::onFlush);
    m_lsnOnLastLineChanged = lastLine()->onValueChanged().connectWeak(
        false, shared_from_this(), &XTextWriter::onLastLineChanged);
    lastLine()->setUIEnabled(true);

     QString buf;
     buf += "#";
      m_entries->childLock();
      for(unsigned int i = 0; i < m_entries->count(); i++)
            {
              shared_ptr<XScalarEntry> entry = (*m_entries)[i];
              if(!*entry->store()) continue;
              buf += entry->getEntryTitle();
          buf += " ";
            }
      buf += "Time";
      m_entries->childUnlock();
     
     lastLine()->value(buf);
  }
  else {
    m_lsnOnFlush.reset();
    m_lsnOnLastLineChanged.reset();
    lastLine()->setUIEnabled(false);
  }
  m_filemutex.unlock();        
}
void
XTextWriter::onFlush(const shared_ptr<XValueNodeBase> &)
{
    lastLine()->setUIEnabled(*recording());
	if(!*recording()) {
      m_filemutex.lock();
  	  if(m_stream.good())
             m_stream.flush();
      m_filemutex.unlock();
	}
}
