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
	  m_filename(create<XStringNode>("Filename", true)) {
}
XRawStream::~XRawStream() {
    if(m_pGFD) gzclose(m_pGFD);
}    

XRawStreamRecorder::XRawStreamRecorder(const char *name, bool runtime, const shared_ptr<XDriverList> &driverlist)
	: XRawStream(name, runtime, driverlist),
	  m_recording(create<XBoolNode>("Recording", true)) {
    recording()->value(false);
    
	for(Transaction tr( *this);; ++tr) {
	    m_lsnOnOpen = tr[ *filename()].onValueChanged().connectWeakly(
	        shared_from_this(), &XRawStreamRecorder::onOpen);
	    m_lsnOnFlush = tr[ *recording()].onValueChanged().connectWeakly(
	        shared_from_this(), &XRawStreamRecorder::onFlush);
		if(tr.commit())
			break;
	}
    for(Transaction tr( *m_drivers);; ++tr) {
        m_lsnOnCatch = tr[ *m_drivers].onCatch().connect( *this, &XRawStreamRecorder::onCatch);
        m_lsnOnRelease = tr[ *m_drivers].onRelease().connect( *this, &XRawStreamRecorder::onRelease);
    	if(tr.commit())
    		break;
    }
}
void
XRawStreamRecorder::onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) {
    shared_ptr<XDriver> driver = static_pointer_cast<XDriver>(e.caught);
    for(Transaction tr( *driver);; ++tr) {
		if(m_lsnOnRecord)
			tr[ *driver].onRecord().connect(m_lsnOnRecord);
		else
			m_lsnOnRecord = tr[ *driver].onRecord().connectWeakly(
				shared_from_this(), &XRawStreamRecorder::onRecord);
		if(tr.commit())
			break;
    }
}
void
XRawStreamRecorder::onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
    shared_ptr<XDriver> driver = static_pointer_cast<XDriver>(e.released);
    for(Transaction tr( *driver);; ++tr) {
		tr[ *driver].onRecord().disconnect(m_lsnOnRecord);
		if(tr.commit())
			break;
    }
}
void
XRawStreamRecorder::onOpen(const Snapshot &shot, XValueNodeBase *) {
	if(m_pGFD) gzclose(m_pGFD);
	m_pGFD = gzopen(QString(filename()->to_str()).toLocal8Bit().data(), "wb");
}
void
XRawStreamRecorder::onFlush(const Snapshot &shot, XValueNodeBase *) {
	if( !*recording())
		if(m_pGFD) {
			m_filemutex.lock();    
			gzflush(m_pGFD, Z_FULL_FLUSH);
			m_filemutex.unlock();    
		}
}
void
XRawStreamRecorder::onRecord(const Snapshot &shot, XDriver *d) {
    if( *recording() && m_pGFD) {
        XPrimaryDriver *driver = dynamic_cast<XPrimaryDriver*>(d);
        if(driver) {
        	const XPrimaryDriver::RawData &rawdata(shot[ *driver].rawData());
            uint32_t size = rawdata.size();
            if(size) {
                uint32_t headersize =
                    sizeof(uint32_t) //allsize
                    + sizeof(int32_t) //time().sec()
                    + sizeof(int32_t); //time().usec()            
                // size of raw record wrapped by header and footer
                uint32_t allsize =
                    headersize
                    + driver->getName().size() //name of driver
                    + 2 //two null chars
                    + size //rawData
                    + sizeof(uint32_t); //allsize
                XPrimaryDriver::RawData header;
                header.push((uint32_t)allsize);
                header.push((int32_t)shot[ *driver].time().sec());
                header.push((int32_t)shot[ *driver].time().usec());
                ASSERT(header.size() == headersize);
    
                m_filemutex.lock();
                gzwrite(m_pGFD, &header[0], header.size());
                gzprintf(m_pGFD, "%s", (const char*)driver->getName().c_str());
                gzputc(m_pGFD, '\0');
                gzputc(m_pGFD, '\0'); //Reserved
                gzwrite(m_pGFD, &rawdata[0], size);
                header.clear(); //using as a footer.
                header.push((uint32_t)allsize);
                gzwrite(m_pGFD, &header[0], header.size());
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
	  m_recording(create<XBoolNode>("Recording", true)) {
    recording()->value(false);
    lastLine()->setUIEnabled(false);
  
	for(Transaction tr( *this);; ++tr) {
	    m_lsnOnFilenameChanged = tr[ *filename()].onValueChanged().connectWeakly(
	        shared_from_this(), &XTextWriter::onFilenameChanged);
		if(tr.commit())
			break;
	}
    for(Transaction tr( *m_drivers);; ++tr) {
        m_lsnOnCatch = tr[ *m_drivers].onCatch().connect( *this, &XTextWriter::onCatch);
        m_lsnOnRelease = tr[ *m_drivers].onRelease().connect( *this, &XTextWriter::onRelease);
    	if(tr.commit())
    		break;
    }
}
void
XTextWriter::onCatch(const Snapshot &shot, const XListNodeBase::Payload::CatchEvent &e) {
    shared_ptr<XDriver> driver = static_pointer_cast<XDriver>(e.caught);
    for(Transaction tr( *driver);; ++tr) {
		if(m_lsnOnRecord)
			tr[ *driver].onRecord().connect(m_lsnOnRecord);
		else
			m_lsnOnRecord = tr[ *driver].onRecord().connectWeakly(
				shared_from_this(), &XTextWriter::onRecord);
		if(tr.commit())
			break;
    }
}
void
XTextWriter::onRelease(const Snapshot &shot, const XListNodeBase::Payload::ReleaseEvent &e) {
    shared_ptr<XDriver> driver = static_pointer_cast<XDriver>(e.released);
    for(Transaction tr( *driver);; ++tr) {
		tr[ *driver].onRecord().disconnect(m_lsnOnRecord);
		if(tr.commit())
			break;
    }
}
void
XTextWriter::onLastLineChanged(const Snapshot &shot, XValueNodeBase *) {
	XScopedLock<XRecursiveMutex> lock(m_filemutex);  
	if(m_stream.good())
	{
		m_stream << lastLine()->to_str()
				 << std::endl;
	}
}
void
XTextWriter::onRecord(const Snapshot &shot, XDriver *driver) {
	if( *recording()) {
		if(shot[ *driver].time()) {
			for(;;) {
				Snapshot shot_entries( *m_entries);
				if( !shot_entries.size())
					break;
				const XNode::NodeList &entries_list( *shot_entries.list());
				bool triggered = false;
				for(XNode::const_iterator it = entries_list.begin(); it != entries_list.end(); it++) {
					shared_ptr<XScalarEntry> entry = static_pointer_cast<XScalarEntry>( *it);
					if( !*entry->store()) continue;
					shared_ptr<XDriver> d(entry->driver());
					if( !d) continue;
					if((d.get() == driver) && shot[ *entry].isTriggered()) {
						triggered = true;
						break;
					}
				}
				if( !triggered)
					break;
				Transaction tr_entries(shot_entries);
				XString buf;
				for(XNode::const_iterator it = entries_list.begin(); it != entries_list.end(); it++) {
					shared_ptr<XScalarEntry> entry = static_pointer_cast<XScalarEntry>( *it);
					if( !shot_entries[ *entry->store()]) continue;
					entry->storeValue(tr_entries);
					buf.append(shot_entries[ *entry->storedValue()].to_str() + " ");
				}
				buf.append(shot[ *driver].time().getTimeFmtStr("%Y/%m/%d %H:%M:%S"));
				if(tr_entries.commit()) {
					lastLine()->value(buf);
					return;
				}
			}
		}
	}
}

void
XTextWriter::onFilenameChanged(const Snapshot &shot, XValueNodeBase *) {
	XScopedLock<XRecursiveMutex> lock(m_filemutex);  
	if(m_stream.is_open()) m_stream.close();
	m_stream.clear();
	m_stream.open((const char*)QString(filename()->to_str()).toLocal8Bit().data(), OFSMODE);

	if(m_stream.good()) {
		for(Transaction tr( *this);; ++tr) {
			m_lsnOnFlush = tr[ *recording()].onValueChanged().connectWeakly(
				shared_from_this(), &XTextWriter::onFlush);
			m_lsnOnLastLineChanged = tr[ *lastLine()].onValueChanged().connectWeakly(
				shared_from_this(), &XTextWriter::onLastLineChanged);
			if(tr.commit())
				break;
		}
		lastLine()->setUIEnabled(true);

		XString buf;
		buf = "#";
		Snapshot shot_entries( *m_entries);
		if(shot_entries.size()) {
			const XNode::NodeList &entries_list( *shot_entries.list());
			for(XNode::const_iterator it = entries_list.begin(); it != entries_list.end(); it++) {
				shared_ptr<XScalarEntry> entry = static_pointer_cast<XScalarEntry>( *it);
				if( !*entry->store()) continue;
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
XTextWriter::onFlush(const Snapshot &shot, XValueNodeBase *) {
    lastLine()->setUIEnabled(*recording());
	if( !*recording()) {
		XScopedLock<XRecursiveMutex> lock(m_filemutex);  
		if(m_stream.good())
			m_stream.flush();
	}
}
