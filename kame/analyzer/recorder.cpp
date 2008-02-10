/***************************************************************************
		Copyright (C) 2002-2008 Kentaro Kitagawa
		                   kitag@issp.u-tokyo.ac.jp
		
		This program is free software; you can redistribute it and/or
		modify it under the terms of the GNU Library General Public
		License as published by the Free Software Foundation; either
		version 2 of the License, or (at your option) any later version.
		
		You should have received a copy of the GNU Library General 
		Public License and a list of authors along with this program; 
		see the files COPYING and AUTHORS.
***************************************************************************/
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
        shared_from_this(), &XRawStreamRecorder::onOpen);
    m_lsnOnFlush = recording()->onValueChanged().connectWeak(
        shared_from_this(), &XRawStreamRecorder::onFlush);
    m_lsnOnCatch = m_drivers->onCatch().connectWeak(
        shared_from_this(), &XRawStreamRecorder::onCatch);
    m_lsnOnRelease = m_drivers->onRelease().connectWeak(
        shared_from_this(), &XRawStreamRecorder::onRelease);
}
void
XRawStreamRecorder::onCatch(const shared_ptr<XNode> &node)
{
    shared_ptr<XDriver> driver = dynamic_pointer_cast<XDriver>(node);
    if(m_lsnOnRecord)
        driver->onRecord().connect(m_lsnOnRecord);
    else
        m_lsnOnRecord = driver->onRecord().connectWeak(
            shared_from_this(), &XRawStreamRecorder::onRecord);
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
	m_pGFD = gzopen(QString(filename()->to_str()).local8Bit(), "wb");
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
            std::vector<char> &rawdata(*XPrimaryDriver::s_tlRawData);
            uint32_t size = rawdata.size();
            if(size) {
                uint32_t headersize =
                    sizeof(uint32_t) //allsize
                    + sizeof(int32_t) //time().sec()
                    + sizeof(int32_t); //time().usec()            
                std::vector<char> buf;
                // size of raw record wrapped by header and footer
                uint32_t allsize =
                    headersize
                    + driver->getName().size() //name of driver
                    + 2 //two null chars
                    + size //rawData
                    + sizeof(uint32_t); //allsize
                XPrimaryDriver::push((uint32_t)allsize, buf);
                XPrimaryDriver::push((int32_t)driver->time().sec(), buf);
                XPrimaryDriver::push((int32_t)driver->time().usec(), buf);
                ASSERT(buf.size() == headersize);
    
                m_filemutex.lock();
                gzwrite(m_pGFD, &buf[0], buf.size());
                gzprintf(m_pGFD, "%s", (const char*)driver->getName().c_str());
                gzputc(m_pGFD, '\0');
                gzputc(m_pGFD, '\0'); //Reserved
                gzwrite(m_pGFD, &rawdata[0], size);
                buf.clear();
                XPrimaryDriver::push((uint32_t)allsize, buf);
                gzwrite(m_pGFD, &buf[0], buf.size());
                m_filemutex.unlock();
            }
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
        shared_from_this(), &XTextWriter::onFilenameChanged);
    m_lsnOnCatch = m_drivers->onCatch().connectWeak(
        shared_from_this(), &XTextWriter::onCatch);
    m_lsnOnRelease = m_drivers->onRelease().connectWeak(
        shared_from_this(), &XTextWriter::onRelease);
}
void
XTextWriter::onCatch(const shared_ptr<XNode> &node)
{
    shared_ptr<XDriver> driver = dynamic_pointer_cast<XDriver>(node);
    if(m_lsnOnRecord)
        driver->onRecord().connect(m_lsnOnRecord);
    else
        m_lsnOnRecord = driver->onRecord().connectWeak(
            shared_from_this(), &XTextWriter::onRecord);
}
void
XTextWriter::onRelease(const shared_ptr<XNode> &node)
{
    shared_ptr<XDriver> driver = dynamic_pointer_cast<XDriver>(node);
    driver->onRecord().disconnect(m_lsnOnRecord);
}
void
XTextWriter::onLastLineChanged(const shared_ptr<XValueNodeBase> &) {
	XScopedLock<XRecursiveMutex> lock(m_filemutex);  
	if(m_stream.good())
	{
		m_stream << lastLine()->to_str()
				 << std::endl;
	}
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

			atomic_shared_ptr<const XNode::NodeList> list(m_entries->children());
			if(list) { 
				for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++) {
					shared_ptr<XScalarEntry> entry = dynamic_pointer_cast<XScalarEntry>(*it);
					if(!*entry->store()) continue;
					shared_ptr<XDriver> d(entry->driver());
					if(!d) continue;
					locked_entries.push_back(entry);
					d->readLockRecord();
					if(entry->isTriggered()) triggered_time = entry->driver()->time();
				}
			}
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
					std::string buf;
					for(std::deque<shared_ptr<XScalarEntry> >::iterator it = locked_entries.begin();
						it != locked_entries.end(); it++) {
						if(!*(*it)->store()) continue;
						(*it)->storeValue();
						buf.append((*it)->storedValue()->to_str() + " ");
					}
					buf.append(driver->time().getTimeFmtStr("%Y/%m/%d %H:%M:%S"));
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
	XScopedLock<XRecursiveMutex> lock(m_filemutex);  
	if(m_stream.is_open()) m_stream.close();
	m_stream.clear();
	m_stream.open((const char*)QString(filename()->to_str()).local8Bit(), OFSMODE);

	if(m_stream.good()) {
		m_lsnOnFlush = recording()->onValueChanged().connectWeak(
			shared_from_this(), &XTextWriter::onFlush);
		m_lsnOnLastLineChanged = lastLine()->onValueChanged().connectWeak(
			shared_from_this(), &XTextWriter::onLastLineChanged);
		lastLine()->setUIEnabled(true);

		std::string buf;
		buf = "#";
		atomic_shared_ptr<const XNode::NodeList> list(m_entries->children());
		if(list) { 
			for(XNode::NodeList::const_iterator it = list->begin(); it != list->end(); it++) {
				shared_ptr<XScalarEntry> entry = dynamic_pointer_cast<XScalarEntry>(*it);
				if(!*entry->store()) continue;
				buf.append(entry->getLabel());
				buf.append(" ");
			}
		}
		buf.append("Time");
		lastLine()->value(buf);
	}
	else {
		m_lsnOnFlush.reset();
		m_lsnOnLastLineChanged.reset();
		lastLine()->setUIEnabled(false);
	}
}
void
XTextWriter::onFlush(const shared_ptr<XValueNodeBase> &)
{
    lastLine()->setUIEnabled(*recording());
	if(!*recording()) {
		XScopedLock<XRecursiveMutex> lock(m_filemutex);  
		if(m_stream.good())
			m_stream.flush();
	}
}
